// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Hardware-touching binary: reads GPU BAR0 registers via sysfs mmap.
#![allow(unsafe_code)]

//! Experiment 070: BAR0 register dump for sovereign reverse engineering.
//!
//! Reads named GPU registers from `/sys/bus/pci/devices/{bdf}/resource0` and
//! outputs structured JSON for automated diffing across backends.
//!
//! Usage:
//!   sudo cargo run --release --bin exp070_register_dump -- <bdf> [output.json]
//!
//! Examples:
//!   sudo cargo run --release --bin exp070_register_dump -- 0000:03:00.0 data/070/cold_oracle.json
//!   sudo cargo run --release --bin exp070_register_dump -- 0000:4a:00.0 data/070/cold_target.json

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::os::unix::io::AsRawFd;

const BAR0_MAP_SIZE: usize = 16 * 1024 * 1024; // 16 MiB

struct RegDef {
    offset: u32,
    name: &'static str,
    group: &'static str,
}

const REGISTER_MAP: &[RegDef] = &[
    // PMC — Power Management Controller
    RegDef {
        offset: 0x000000,
        name: "BOOT0",
        group: "PMC",
    },
    RegDef {
        offset: 0x000004,
        name: "BOOT1",
        group: "PMC",
    },
    RegDef {
        offset: 0x000200,
        name: "PMC_ENABLE",
        group: "PMC",
    },
    RegDef {
        offset: 0x000204,
        name: "PMC_DEVICE_ENABLE",
        group: "PMC",
    },
    // PBUS
    RegDef {
        offset: 0x001C00,
        name: "PBUS_EXT_CG",
        group: "PBUS",
    },
    RegDef {
        offset: 0x001C04,
        name: "PBUS_EXT_CG1",
        group: "PBUS",
    },
    // PFIFO — Scheduler / Engine
    RegDef {
        offset: 0x002004,
        name: "PFIFO_PBDMA_MAP",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002100,
        name: "PFIFO_INTR",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002140,
        name: "PFIFO_INTR_EN",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002200,
        name: "PFIFO_ENABLE",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002254,
        name: "PFIFO_FB_TIMEOUT",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002270,
        name: "RUNLIST_BASE",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002274,
        name: "RUNLIST_SUBMIT",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002280,
        name: "RUNLIST0_BASE",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002284,
        name: "RUNLIST0_INFO",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x00228C,
        name: "RUNLIST1_INFO",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002390,
        name: "PBDMA_RUNL_MAP_0",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002394,
        name: "PBDMA_RUNL_MAP_1",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002398,
        name: "PBDMA_RUNL_MAP_2",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x00239C,
        name: "PBDMA_RUNL_MAP_3",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002504,
        name: "SCHED_EN",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002508,
        name: "SCHED_STATUS",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x00252C,
        name: "BIND_ERROR",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002630,
        name: "SCHED_DISABLE",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002634,
        name: "PREEMPT",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002638,
        name: "PREEMPT_PENDING",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002640,
        name: "ENGN0_STATUS",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002A00,
        name: "RUNLIST_ACK",
        group: "PFIFO",
    },
    RegDef {
        offset: 0x002A04,
        name: "PBDMA_INTR_EN",
        group: "PFIFO",
    },
    // PBDMA idle
    RegDef {
        offset: 0x003080,
        name: "PBDMA0_IDLE",
        group: "PBDMA_IDLE",
    },
    RegDef {
        offset: 0x003084,
        name: "PBDMA1_IDLE",
        group: "PBDMA_IDLE",
    },
    RegDef {
        offset: 0x003088,
        name: "PBDMA2_IDLE",
        group: "PBDMA_IDLE",
    },
    RegDef {
        offset: 0x00308C,
        name: "PBDMA3_IDLE",
        group: "PBDMA_IDLE",
    },
    // PBDMA0 (base 0x040000)
    RegDef {
        offset: 0x040040,
        name: "PBDMA0_GP_BASE_LO",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040044,
        name: "PBDMA0_GP_BASE_HI",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040048,
        name: "PBDMA0_GP_FETCH",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x04004C,
        name: "PBDMA0_GP_STATE",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040050,
        name: "PBDMA0_GP_PUT_HI",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040054,
        name: "PBDMA0_GP_PUT_LO",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400A8,
        name: "PBDMA0_TARGET",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400AC,
        name: "PBDMA0_SET_CHANNEL_INFO",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400B0,
        name: "PBDMA0_CHANNEL_STATE",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400C0,
        name: "PBDMA0_SIGNATURE",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400D0,
        name: "PBDMA0_USERD_LO",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x0400D4,
        name: "PBDMA0_USERD_HI",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040108,
        name: "PBDMA0_INTR",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x04010C,
        name: "PBDMA0_INTR_EN",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x040148,
        name: "PBDMA0_HCE",
        group: "PBDMA0",
    },
    RegDef {
        offset: 0x04014C,
        name: "PBDMA0_HCE_EN",
        group: "PBDMA0",
    },
    // PBDMA2 (base 0x044000 — GR engine PBDMA on Volta)
    RegDef {
        offset: 0x044040,
        name: "PBDMA2_GP_BASE_LO",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x044044,
        name: "PBDMA2_GP_BASE_HI",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x044048,
        name: "PBDMA2_GP_FETCH",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x04404C,
        name: "PBDMA2_GP_STATE",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x044054,
        name: "PBDMA2_GP_PUT_LO",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440A8,
        name: "PBDMA2_TARGET",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440AC,
        name: "PBDMA2_SET_CHANNEL_INFO",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440B0,
        name: "PBDMA2_CHANNEL_STATE",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440C0,
        name: "PBDMA2_SIGNATURE",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440D0,
        name: "PBDMA2_USERD_LO",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x0440D4,
        name: "PBDMA2_USERD_HI",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x044108,
        name: "PBDMA2_INTR",
        group: "PBDMA2",
    },
    RegDef {
        offset: 0x04410C,
        name: "PBDMA2_INTR_EN",
        group: "PBDMA2",
    },
    // PRIV ring
    RegDef {
        offset: 0x012070,
        name: "PRIV_RING_INTR",
        group: "PRIV",
    },
    // Thermal
    RegDef {
        offset: 0x020460,
        name: "GPU_TEMP",
        group: "THERM",
    },
    // PTOP — Topology
    RegDef {
        offset: 0x022700,
        name: "PTOP_INFO_0",
        group: "PTOP",
    },
    RegDef {
        offset: 0x022704,
        name: "PTOP_INFO_1",
        group: "PTOP",
    },
    RegDef {
        offset: 0x022708,
        name: "PTOP_INFO_2",
        group: "PTOP",
    },
    RegDef {
        offset: 0x02270C,
        name: "PTOP_INFO_3",
        group: "PTOP",
    },
    // SEC2 falcon (0x087000)
    RegDef {
        offset: 0x087100,
        name: "SEC2_CPUCTL",
        group: "SEC2",
    },
    RegDef {
        offset: 0x087104,
        name: "SEC2_BOOTVEC",
        group: "SEC2",
    },
    RegDef {
        offset: 0x087240,
        name: "SEC2_SCTL",
        group: "SEC2",
    },
    // FECS falcon (0x409000)
    RegDef {
        offset: 0x409100,
        name: "FECS_CPUCTL",
        group: "FECS",
    },
    RegDef {
        offset: 0x409104,
        name: "FECS_BOOTVEC",
        group: "FECS",
    },
    RegDef {
        offset: 0x409110,
        name: "FECS_PC",
        group: "FECS",
    },
    RegDef {
        offset: 0x409240,
        name: "FECS_SCTL",
        group: "FECS",
    },
    // PMU falcon (0x10A000)
    RegDef {
        offset: 0x10A100,
        name: "PMU_CPUCTL",
        group: "PMU",
    },
    RegDef {
        offset: 0x10A104,
        name: "PMU_BOOTVEC",
        group: "PMU",
    },
    RegDef {
        offset: 0x10A240,
        name: "PMU_SCTL",
        group: "PMU",
    },
    // MMU fault buffers
    RegDef {
        offset: 0x100A2C,
        name: "MMU_FAULT_STATUS",
        group: "MMU",
    },
    RegDef {
        offset: 0x100A30,
        name: "MMU_FAULT_ADDR_LO",
        group: "MMU",
    },
    RegDef {
        offset: 0x100A34,
        name: "MMU_FAULT_ADDR_HI",
        group: "MMU",
    },
    RegDef {
        offset: 0x100A38,
        name: "MMU_FAULT_INST_LO",
        group: "MMU",
    },
    RegDef {
        offset: 0x100A3C,
        name: "MMU_FAULT_INST_HI",
        group: "MMU",
    },
    RegDef {
        offset: 0x100A40,
        name: "MMU_FAULT_INFO",
        group: "MMU",
    },
    RegDef {
        offset: 0x100C80,
        name: "MMU_PRI_CTRL",
        group: "MMU",
    },
    RegDef {
        offset: 0x100CBC,
        name: "MMU_TLB_FLUSH",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E24,
        name: "MMU_FAULT_BUF0_LO",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E28,
        name: "MMU_FAULT_BUF0_HI",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E2C,
        name: "MMU_FAULT_BUF0_SIZE",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E30,
        name: "MMU_FAULT_BUF0_GET",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E34,
        name: "MMU_FAULT_BUF0_PUT",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E44,
        name: "MMU_FAULT_BUF1_LO",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E48,
        name: "MMU_FAULT_BUF1_HI",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E4C,
        name: "MMU_FAULT_BUF1_SIZE",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E50,
        name: "MMU_FAULT_BUF1_GET",
        group: "MMU",
    },
    RegDef {
        offset: 0x100E54,
        name: "MMU_FAULT_BUF1_PUT",
        group: "MMU",
    },
    // BAR1/BAR2 block
    RegDef {
        offset: 0x001704,
        name: "BAR1_BLOCK",
        group: "BAR",
    },
    RegDef {
        offset: 0x001710,
        name: "BIND_STATUS",
        group: "BAR",
    },
    RegDef {
        offset: 0x001714,
        name: "BAR2_BLOCK",
        group: "BAR",
    },
    // PCCSR channels
    RegDef {
        offset: 0x800000,
        name: "PCCSR_INST_0",
        group: "PCCSR",
    },
    RegDef {
        offset: 0x800004,
        name: "PCCSR_CHAN_0",
        group: "PCCSR",
    },
    RegDef {
        offset: 0x800008,
        name: "PCCSR_INST_1",
        group: "PCCSR",
    },
    RegDef {
        offset: 0x80000C,
        name: "PCCSR_CHAN_1",
        group: "PCCSR",
    },
    // USERMODE
    RegDef {
        offset: 0x810000,
        name: "USERMODE_CFG",
        group: "USERMODE",
    },
    RegDef {
        offset: 0x810004,
        name: "USERMODE_4",
        group: "USERMODE",
    },
    RegDef {
        offset: 0x810010,
        name: "USERMODE_TIME_LO",
        group: "USERMODE",
    },
    RegDef {
        offset: 0x810014,
        name: "USERMODE_TIME_HI",
        group: "USERMODE",
    },
    RegDef {
        offset: 0x810090,
        name: "USERMODE_NOTIFY_CHAN_PENDING",
        group: "USERMODE",
    },
    // GPCCS scan range — sample points across 0x400000–0x520000
    RegDef {
        offset: 0x400000,
        name: "GPC0_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x410000,
        name: "GPC0_GR_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x418000,
        name: "GPC0_GPCCS_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x418100,
        name: "GPC0_GPCCS_CPUCTL",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x418110,
        name: "GPC0_GPCCS_PC",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x418240,
        name: "GPC0_GPCCS_SCTL",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x420000,
        name: "GPC1_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x428000,
        name: "GPC1_GPCCS_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x428100,
        name: "GPC1_GPCCS_CPUCTL",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x428110,
        name: "GPC1_GPCCS_PC",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x428240,
        name: "GPC1_GPCCS_SCTL",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x430000,
        name: "GPC2_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x438000,
        name: "GPC2_GPCCS_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x438100,
        name: "GPC2_GPCCS_CPUCTL",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x440000,
        name: "GPC3_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x448000,
        name: "GPC3_GPCCS_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x448100,
        name: "GPC3_GPCCS_CPUCTL",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x450000,
        name: "GPC4_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x458000,
        name: "GPC4_GPCCS_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x460000,
        name: "GPC5_BOOT0",
        group: "GPCCS_SCAN",
    },
    RegDef {
        offset: 0x468000,
        name: "GPC5_GPCCS_BOOT0",
        group: "GPCCS_SCAN",
    },
];

fn read_bar0_register(mm: *const u8, offset: u32) -> Result<u32, String> {
    if (offset as usize) + 4 > BAR0_MAP_SIZE {
        return Err(format!("offset {offset:#x} out of BAR0 range"));
    }
    // SAFETY: offset is bounds-checked above; mm points to a valid mmap of BAR0.
    // Read as bytes and assemble LE u32 to avoid alignment cast.
    let ptr = unsafe { mm.add(offset as usize) };
    let mut buf = [0u8; 4];
    for (i, byte) in buf.iter_mut().enumerate() {
        *byte = unsafe { std::ptr::read_volatile(ptr.add(i)) };
    }
    Ok(u32::from_le_bytes(buf))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bdf = args.get(1).map_or("0000:03:00.0", String::as_str);
    let output_path = args.get(2).map(String::as_str);

    let resource_path = format!("/sys/bus/pci/devices/{bdf}/resource0");

    let file = match File::options().read(true).open(&resource_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ERROR: cannot open {resource_path}: {e}");
            eprintln!("  Hint: run with sudo, or ensure {bdf} is bound to a driver that exposes resource0");
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

    let mut results: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut registers = Vec::new();

    let boot0 = read_bar0_register(base, 0).unwrap_or(0);
    let temp_raw = read_bar0_register(base, 0x020460).unwrap_or(0);
    let temp_c = (temp_raw >> 8) & 0xFF;

    results.insert("bdf".into(), serde_json::Value::String(bdf.to_string()));
    results.insert(
        "boot0".into(),
        serde_json::Value::String(format!("{boot0:#010x}")),
    );
    results.insert(
        "gpu_temp_c".into(),
        serde_json::Value::Number(temp_c.into()),
    );
    results.insert(
        "timestamp".into(),
        serde_json::Value::String(chrono_iso8601()),
    );

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Exp 070: BAR0 Register Dump — {bdf}");
    println!("║  BOOT0={boot0:#010x}  GPU_TEMP={temp_c}°C");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    for reg in REGISTER_MAP {
        let val = read_bar0_register(base, reg.offset);
        let (val_u32, val_str, status) = match val {
            Ok(v) => (v, format!("{v:#010x}"), "ok"),
            Err(_) => (0, "ERROR".to_string(), "error"),
        };

        println!(
            "║ [{:#08x}] {:<36} = {val_str}  ({group})",
            reg.offset,
            reg.name,
            group = reg.group
        );

        let mut entry = serde_json::Map::new();
        entry.insert(
            "offset".into(),
            serde_json::Value::String(format!("{:#08x}", reg.offset)),
        );
        entry.insert(
            "name".into(),
            serde_json::Value::String(reg.name.to_string()),
        );
        entry.insert(
            "group".into(),
            serde_json::Value::String(reg.group.to_string()),
        );
        entry.insert("value".into(), serde_json::Value::String(val_str));
        entry.insert("raw".into(), serde_json::Value::Number(val_u32.into()));
        entry.insert(
            "status".into(),
            serde_json::Value::String(status.to_string()),
        );
        registers.push(serde_json::Value::Object(entry));
    }

    println!("╚══════════════════════════════════════════════════════════════════╝");

    results.insert("registers".into(), serde_json::Value::Array(registers));

    // SAFETY: clean up the mmap.
    unsafe { libc::munmap(mm, BAR0_MAP_SIZE) };

    let json = serde_json::to_string_pretty(&results).expect("JSON serialization");

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
