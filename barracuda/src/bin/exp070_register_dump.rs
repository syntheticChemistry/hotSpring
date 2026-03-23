// SPDX-License-Identifier: AGPL-3.0-only
//
// Hardware-touching binary: reads GPU BAR0 registers via sysfs mmap.
#![allow(unsafe_code)]

//! Experiment 070: BAR0 register dump for sovereign reverse engineering.
//!
//! Reads named GPU registers from `/sys/bus/pci/devices/<bdf>/resource0` and
//! outputs structured JSON for automated diffing across backends.
//!
//! **Vendor-agnostic**: auto-detects NVIDIA vs AMD via PCI vendor ID and
//! uses the appropriate register map from [`register_maps`](hotspring_barracuda::register_maps).
//!
//! Usage:
//!   sudo cargo run --release --bin exp070_register_dump -- `<bdf>` \[output.json\]
//!
//! Examples:
//!   sudo cargo run --release --bin exp070_register_dump -- 0000:03:00.0 data/070/titan_v_warm.json
//!   sudo cargo run --release --bin exp070_register_dump -- 0000:4c:00.0 data/070/radeon_vii_warm.json

use std::fs::File;
use std::io::Write;
use std::os::unix::io::AsRawFd;

use hotspring_barracuda::register_maps::{detect_register_map, RegisterDump, RegisterEntry};

const BAR0_MAP_SIZE: usize = 16 * 1024 * 1024; // 16 MiB

fn read_bar0_register(mm: *const u8, offset: u32) -> Result<u32, String> {
    if (offset as usize) + 4 > BAR0_MAP_SIZE {
        return Err(format!("offset {offset:#x} out of BAR0 range"));
    }
    // SAFETY: offset is bounds-checked above; mm points to a valid mmap of BAR0.
    let ptr = unsafe { mm.add(offset as usize) };
    let mut buf = [0u8; 4];
    for (i, byte) in buf.iter_mut().enumerate() {
        *byte = unsafe { std::ptr::read_volatile(ptr.add(i)) };
    }
    Ok(u32::from_le_bytes(buf))
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
    let bdf = args.get(1).map_or("0000:03:00.0", String::as_str);
    let output_path = args.get(2).map(String::as_str);

    let vendor_id = read_vendor_id(bdf);
    let device_id = read_device_id(bdf);

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

    let resource_path = format!("/sys/bus/pci/devices/{bdf}/resource0");

    let file = match File::options().read(true).open(&resource_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ERROR: cannot open {resource_path}: {e}");
            eprintln!("  Hint: run with sudo/pkexec, or ensure {bdf} is bound to a driver that exposes resource0");
            std::process::exit(1);
        }
    };

    // SAFETY: resource0 is a PCI BAR mmap exposed by the kernel. Read-only, 16 MiB.
    let mm = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            BAR0_MAP_SIZE,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };

    if mm == libc::MAP_FAILED {
        eprintln!("ERROR: mmap of {resource_path} failed");
        std::process::exit(1);
    }

    let base = mm.cast::<u8>();
    let regs = reg_map.registers();

    let boot0 = read_bar0_register(base, regs.first().map_or(0, |r| r.offset)).unwrap_or(0);

    let temp_str = if let Some(offset) = reg_map.thermal_offset() {
        let raw = read_bar0_register(base, offset).unwrap_or(0);
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
        "║  Exp 070: {vendor} {arch} Register Dump — {bdf}",
        vendor = reg_map.vendor(),
        arch = reg_map.arch(),
    );
    println!("║  ID: {boot_id}  TEMP: {temp_str}");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let mut entries = Vec::new();

    for reg in regs {
        let val = read_bar0_register(base, reg.offset);
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

    // SAFETY: clean up the mmap.
    unsafe { libc::munmap(mm, BAR0_MAP_SIZE) };

    let dump = RegisterDump {
        vendor: reg_map.vendor().to_string(),
        arch: reg_map.arch().to_string(),
        pci_id: format!("{vendor_id:#06x}:{device_id:#06x}"),
        bdf: bdf.to_string(),
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
