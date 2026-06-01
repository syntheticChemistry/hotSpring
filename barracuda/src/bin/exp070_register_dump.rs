// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 070: BAR0 register dump for sovereign reverse engineering.
//!
//! Reads named GPU registers and outputs structured JSON for automated diffing
//! across backends. All MMIO access routes through toadStool ember
//! (`mmio.read32`) — fork-isolated, circuit-breaker-protected. No direct
//! BAR0 mmap or sudo required when ember holds the VFIO fd.
//!
//! Usage:
//!   cargo run --release --bin exp070_register_dump -- [--bdf <bdf>] [output.json]
//!   cargo run --release --bin exp070_register_dump -- <bdf> [output.json]
//!   cargo run --release --bin exp070_register_dump -- --bdf 0000:03:00.0 dump.json

use std::fs::File;
use std::io::Write as _;

use hotspring_barracuda::register_maps::{RegisterDump, RegisterEntry, detect_register_map};

#[path = "../bin_helpers/sovereignty/mod.rs"]
mod sovereignty;
use sovereignty::connect::{connect_ember, extract_arg, resolve_target_bdf};

fn read_register(
    client: &hotspring_barracuda::fleet_client::EmberClient,
    bdf: &str,
    offset: u32,
) -> Result<u32, String> {
    client
        .mmio_read(bdf, offset)
        .map(|r| r.value)
        .map_err(|e| e.to_string())
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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let positional: Vec<&str> = args[1..]
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .collect();

    let bdf = if let Some(p) = positional.first() {
        (*p).to_string()
    } else {
        resolve_target_bdf(&args, 0)
    };
    if bdf.is_empty() {
        eprintln!("ERROR: no target BDF — pass --bdf, positional <bdf>, or set HOTSPRING_BDF");
        std::process::exit(1);
    }

    let output_flag = extract_arg(&args, "--output");
    let output_path = positional
        .get(1)
        .copied()
        .or(output_flag.as_deref());
    let ember_socket = extract_arg(&args, "--ember-socket");

    let ember = connect_ember(&bdf, ember_socket.as_deref());
    eprintln!("Mode: ember-ipc (mmio.read32)");

    let vendor_id = read_vendor_id(&bdf);
    let device_id = read_device_id(&bdf);

    let Some(reg_map) = detect_register_map(vendor_id) else {
        eprintln!("ERROR: unknown vendor {vendor_id:#06x} for {bdf} — no register map available");
        eprintln!("  Supported: NVIDIA (0x10de), AMD (0x1002)");
        std::process::exit(1);
    };

    eprintln!(
        "Detected: {vendor} {arch} (PCI {vendor_id:#06x}:{device_id:#06x})",
        vendor = reg_map.vendor(),
        arch = reg_map.arch(),
    );

    let regs = reg_map.registers();
    let boot0 = read_register(&ember, &bdf, regs.first().map_or(0, |r| r.offset)).unwrap_or(0);

    let temp_str = if let Some(offset) = reg_map.thermal_offset() {
        let raw = read_register(&ember, &bdf, offset).unwrap_or(0);
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
        "║  Exp 070: {vendor} {arch} Register Dump — {bdf} (ember-ipc)",
        vendor = reg_map.vendor(),
        arch = reg_map.arch(),
    );
    println!("║  ID: {boot_id}  TEMP: {temp_str}");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let mut entries = Vec::new();

    for reg in regs {
        let val = read_register(&ember, &bdf, reg.offset);
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
