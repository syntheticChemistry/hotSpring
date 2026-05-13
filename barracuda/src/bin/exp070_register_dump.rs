// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 070: BAR0 register dump for sovereign reverse engineering.
//!
//! Reads named GPU registers and outputs structured JSON for automated diffing
//! across backends.
//!
//! ## Access modes
//!
//! - **Direct mmap** (default, requires `low-level` feature + sudo):
//!   Opens `/sys/bus/pci/devices/<bdf>/resource0` via `mmap`, wrapped in a
//!   RAII [`Bar0View`] that bounds-checks every read and drops the mapping on
//!   exit. Unsafe is confined to the two volatile-read lines inside `Bar0View`.
//!
//! - **Ember-routed** (`--via-ember`):
//!   Routes all reads through `ember.mmio.read` IPC, using the fork-isolated,
//!   circuit-breaker-protected MMIO path in toadstool-ember. No sudo required if
//!   ember holds the VFIO fd. Validates that the ember MMIO pipeline produces
//!   identical results to direct access.
//!
//! Usage:
//!   sudo cargo run --release --features low-level --bin exp070_register_dump -- <bdf> [output.json]
//!   cargo run --release --bin exp070_register_dump -- --via-ember <bdf> [output.json]

use std::fs::File;
use std::io::Write as _;

use hotspring_barracuda::register_maps::{RegisterDump, RegisterEntry, detect_register_map};

#[cfg(feature = "low-level")]
#[allow(unsafe_code)]
#[path = "../low_level/bar0.rs"]
mod bar0_mmio;

#[cfg(feature = "low-level")]
use bar0_mmio::Bar0View;

// ── Access mode enum ─────────────────────────────────────────────────────────

enum AccessMode {
    #[cfg(feature = "low-level")]
    DirectMmap(Bar0View),
    EmberIpc {
        client: hotspring_barracuda::fleet_client::EmberClient,
        bdf: String,
    },
}

impl AccessMode {
    fn read_register(&self, offset: u32) -> Result<u32, String> {
        match self {
            #[cfg(feature = "low-level")]
            AccessMode::DirectMmap(view) => view.read_u32(offset),
            AccessMode::EmberIpc { client, bdf } => {
                let result = client.mmio_read(bdf, offset).map_err(|e| e.to_string())?;
                Ok(result.value)
            }
        }
    }

    fn mode_label(&self) -> &str {
        match self {
            #[cfg(feature = "low-level")]
            AccessMode::DirectMmap(_) => "direct-mmap",
            AccessMode::EmberIpc { .. } => "ember-ipc",
        }
    }
}

fn read_vendor_id(bdf: &str) -> u16 {
    std::fs::read_to_string(format!("/sys/bus/pci/devices/{bdf}/vendor"))
        .ok()
        .and_then(|s| u16::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .unwrap_or(0)
}

fn read_device_id(bdf: &str) -> u16 {
    std::fs::read_to_string(format!("/sys/bus/pci/devices/{bdf}/device"))
        .ok()
        .and_then(|s| u16::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .unwrap_or(0)
}

fn resolve_ember_socket(bdf: &str) -> std::path::PathBuf {
    use hotspring_barracuda::fleet_client::{FleetDiscovery, discover_diesel_ember_socket};

    if let Some(sock) = discover_diesel_ember_socket(bdf) {
        return sock;
    }
    if let Ok(disc) = FleetDiscovery::load_default() {
        if let Some(sock) = disc.file().routes.get(bdf) {
            return std::path::PathBuf::from(sock);
        }
    }
    let slug = bdf.replace(':', "-");
    let fleet_sock = std::path::PathBuf::from(format!("/run/toadstool/fleet/ember-{slug}.sock"));
    if fleet_sock.exists() {
        return fleet_sock;
    }
    let per_device = std::path::PathBuf::from(format!("/run/toadstool/ember-{slug}.sock"));
    if per_device.exists() {
        return per_device;
    }
    per_device
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let via_ember = args.iter().any(|a| a == "--via-ember");
    let positional: Vec<&str> = args[1..]
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .collect();

    let bdf = positional.first().map_or_else(
        || std::env::var("HOTSPRING_BDF").unwrap_or_else(|_| "0000:03:00.0".to_string()),
        std::string::ToString::to_string,
    );
    let output_path = positional.get(1).copied();

    let vendor_id = read_vendor_id(&bdf);
    let device_id = read_device_id(&bdf);

    let Some(reg_map) = detect_register_map(vendor_id) else {
        eprintln!("ERROR: unknown vendor {vendor_id:#06x} for {bdf} — no register map available");
        eprintln!("  Supported: NVIDIA (0x10de), AMD (0x1002)");
        std::process::exit(1);
    };

    let access = if via_ember {
        let socket = resolve_ember_socket(&bdf);
        eprintln!("Mode: ember-ipc via {}", socket.display());
        AccessMode::EmberIpc {
            client: hotspring_barracuda::fleet_client::EmberClient::connect(&socket),
            bdf: bdf.clone(),
        }
    } else {
        #[cfg(feature = "low-level")]
        {
            match Bar0View::open(&bdf) {
                Ok(view) => {
                    eprintln!("Mode: direct-mmap via /sys/bus/pci/devices/{bdf}/resource0");
                    AccessMode::DirectMmap(view)
                }
                Err(e) => {
                    eprintln!("ERROR: {e}");
                    eprintln!(
                        "  Hint: run with sudo/pkexec, or use --via-ember for ember-routed access"
                    );
                    std::process::exit(1);
                }
            }
        }
        #[cfg(not(feature = "low-level"))]
        {
            eprintln!("ERROR: direct mmap requires --features low-level; use --via-ember instead");
            std::process::exit(1);
        }
    };

    eprintln!(
        "Detected: {vendor} {arch} (PCI {vendor_id:#06x}:{device_id:#06x})",
        vendor = reg_map.vendor(),
        arch = reg_map.arch(),
    );

    let regs = reg_map.registers();
    let boot0 = access
        .read_register(regs.first().map_or(0, |r| r.offset))
        .unwrap_or(0);

    let temp_str = if let Some(offset) = reg_map.thermal_offset() {
        let raw = access.read_register(offset).unwrap_or(0);
        match reg_map.decode_temp_c(raw) {
            Some(c) => format!("{c}°C"),
            None => format!("raw={raw:#010x}"),
        }
    } else {
        "N/A".into()
    };

    let boot_id = reg_map.decode_boot_id(boot0);

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!(
        "║  Exp 070: {vendor} {arch} Register Dump — {bdf} ({mode})",
        vendor = reg_map.vendor(),
        arch = reg_map.arch(),
        mode = access.mode_label(),
    );
    println!("║  ID: {boot_id}  TEMP: {temp_str}");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let mut entries = Vec::new();

    for reg in regs {
        let val = access.read_register(reg.offset);
        let (val_u32, val_str) = match val {
            Ok(v) => (v, format!("{v:#010x}")),
            Err(_) => (0, "ERROR".to_string()),
        };

        println!(
            "║ [{:#08x}] {:<40} = {val_str}  ({group})",
            reg.offset,
            reg.name,
            group = reg.group
        );

        entries.push(RegisterEntry {
            offset: format!("{:#08x}", reg.offset),
            name: reg.name.to_string(),
            group: reg.group.to_string(),
            value: val_str,
            raw_offset: reg.offset,
            raw_value: val_u32,
        });
    }

    println!("╚══════════════════════════════════════════════════════════════════╝");

    // `access` is dropped here — Bar0View::drop() calls munmap automatically.

    let dump = RegisterDump {
        vendor: reg_map.vendor().to_string(),
        arch: reg_map.arch().to_string(),
        pci_id: format!("{vendor_id:#06x}:{device_id:#06x}"),
        bdf,
        timestamp: chrono_iso8601(),
        registers: entries,
    };

    let json = serde_json::to_string_pretty(&dump).expect("JSON serialization");

    if let Some(path) = output_path {
        let mut f = File::create(path).expect("create output file");
        f.write_all(json.as_bytes()).expect("write JSON");
        println!("\nWrote {path}");
    } else {
        println!("\n{json}");
    }
}

fn chrono_iso8601() -> String {
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let s = secs % 60;
    let days = secs / 86400;
    let (y, m, d) = civil_from_days(days as i64);
    format!("{y:04}-{m:02}-{d:02}T{hours:02}:{mins:02}:{s:02}Z")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
