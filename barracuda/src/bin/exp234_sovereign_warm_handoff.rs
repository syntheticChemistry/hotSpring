// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 234: Safe Sovereign Warm Handoff Caller
//!
//! Wraps `sovereign.warm_handoff` with lockup defenses learned from Runs 1-6:
//!
//! - File lock prevents concurrent calls (Run #6: double RPC race)
//! - PCI config space health pre-check (Run #6: I/O error before insmod)
//! - Single RPC with explicit timeout monitoring
//! - NMI watchdog verification (Run #6: kernel deadlock invisible)
//! - MMIO liveness probe before and after
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin exp234_sovereign_warm_handoff -- \
//!   --bdf 0000:02:00.0 \
//!   --strategy nvidia_catalyst_minimal_nop_titanv \
//!   [--dry-run] [--settle-secs 60]
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hotspring_barracuda::fleet_client::EmberClient;
use hotspring_barracuda::validation::ValidationHarness;

#[path = "../bin_helpers/sovereignty/mod.rs"]
mod sovereignty;
use sovereignty::connect::{extract_arg, is_dry_run, resolve_target_bdf, try_connect_ember_probed};

fn lock_file_path() -> String {
    let dir = std::env::var("BIOMEOS_SOCKET_DIR")
        .or_else(|_| std::env::var("XDG_RUNTIME_DIR").map(|d| format!("{d}/biomeos")))
        .unwrap_or_else(|_| "/run/user/1000/biomeos".into());
    format!("{dir}/hotspring-warm-handoff.lock")
}
const DEFAULT_STRATEGY: &str = "nvidia_catalyst_minimal_nop_titanv";
const DEFAULT_SETTLE_SECS: u64 = 60;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Exp 234: Safe Sovereign Warm Handoff                      ║");
    println!("║  Defenses: file lock, PCI pre-check, NMI verify, MMIO     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut harness = ValidationHarness::new("exp234_sovereign_warm_handoff");

    let args: Vec<String> = std::env::args().collect();
    let bdf = resolve_target_bdf(&args, 0);
    if bdf.is_empty() {
        eprintln!("FATAL: no target BDF — pass --bdf or set HOTSPRING_BARRACUDA_TARGET_BDF");
        std::process::exit(1);
    }
    let strategy = extract_arg(&args, "--strategy").unwrap_or_else(|| DEFAULT_STRATEGY.to_string());
    let settle_secs: u64 = extract_arg(&args, "--settle-secs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SETTLE_SECS);
    let dry_run = is_dry_run(&args);

    println!("  Target BDF:    {bdf}");
    println!("  Strategy:      {strategy}");
    println!("  Settle:        {settle_secs}s");
    println!("  Dry run:       {dry_run}\n");

    // ── Defense 1: File lock (prevent double-RPC race) ──
    println!("━━━ Defense 1: Acquire Exclusive Lock ━━━\n");
    let lock_ok = acquire_lock(&bdf);
    harness.check_bool("exclusive lock acquired (no concurrent caller)", lock_ok);
    if !lock_ok {
        eprintln!("  ABORT: another warm handoff is in progress");
        harness.finish();
    }

    // ── Defense 2: NMI watchdog verification ──
    println!("\n━━━ Defense 2: NMI Watchdog Status ━━━\n");
    let nmi_ok = check_nmi_watchdog(&mut harness);
    if !nmi_ok {
        eprintln!("  WARNING: NMI watchdog not active — kernel lockups will be silent");
        eprintln!("  Fix: add 'nmi_watchdog=1 softlockup_panic=1' to kernel cmdline");
        eprintln!("  (continuing anyway — this is advisory)\n");
    }

    // ── Defense 3: Pre-handoff MMIO liveness ──
    println!("\n━━━ Defense 3: Pre-Handoff MMIO Liveness ━━━\n");
    let ember = try_connect_ember_probed(&bdf);
    if let Some(ref e) = ember {
        let boot0 = check_mmio_liveness(e, &bdf, &mut harness);
        if boot0.is_none() {
            eprintln!("  WARNING: MMIO probe failed — GPU may be unhealthy");
        }
    } else {
        println!("  ember not reachable for MMIO probe (cold start expected)");
        harness.check_bool("pre-handoff MMIO alive", false);
    }

    // ── Defense 4: PCI config space health ──
    println!("\n━━━ Defense 4: PCI Config Space Health ━━━\n");
    let pci_ok = check_pci_config_health(&bdf, &mut harness);
    if !pci_ok {
        eprintln!("  ABORT: PCI config space unhealthy — handoff would likely lockup");
        release_lock();
        harness.finish();
    }

    // ── Defense 5: Verify no stale nvsov module ──
    println!("\n━━━ Defense 5: Module State Clean ━━━\n");
    let module_clean = check_no_stale_module(&mut harness);
    if !module_clean {
        eprintln!("  ABORT: stale nvsov/nvidia-470 module detected — reboot required");
        release_lock();
        harness.finish();
    }

    if dry_run {
        println!("\n  DRY RUN — all pre-flight checks passed. Would call:");
        println!(
            "    sovereign.warm_handoff(bdf={bdf}, strategy={strategy}, settle={settle_secs}s)"
        );
        release_lock();
        harness.check_bool("dry run pre-flight PASS", true);
        println!();
        harness.finish();
    }

    // ── Execute: Single sovereign.warm_handoff RPC ──
    println!("\n━━━ Execute: sovereign.warm_handoff ━━━\n");
    println!("  Sending SINGLE RPC to toadStool compute socket...");
    println!("  (catalyst watchdog: 450s, NMI watchdog: kernel-level)");

    let start = Instant::now();
    let result = call_warm_handoff(&bdf, &strategy, settle_secs);
    let elapsed = start.elapsed();

    match result {
        Ok(resp) => {
            println!("  Response received in {elapsed:.1?}");
            let success = resp
                .get("result")
                .and_then(|r| r.get("success"))
                .and_then(|s| s.as_bool())
                .unwrap_or(false);
            let tier = resp
                .get("result")
                .and_then(|r| r.get("tier"))
                .and_then(|t| t.as_str())
                .unwrap_or("None");
            let total_ms = resp
                .get("result")
                .and_then(|r| r.get("total_ms"))
                .and_then(|t| t.as_u64())
                .unwrap_or(0);

            println!("  success:  {success}");
            println!("  tier:     {tier}");
            println!("  total_ms: {total_ms}");

            harness.check_bool("warm handoff RPC completed", true);
            harness.check_bool("warm handoff succeeded", success);

            if success {
                println!("\n  Post-handoff MMIO liveness check...");
                if let Some(e) = try_connect_ember_probed(&bdf) {
                    check_mmio_liveness(&e, &bdf, &mut harness);
                }
            }
        }
        Err(e) => {
            println!("  RPC failed after {elapsed:.1?}: {e}");
            harness.check_bool("warm handoff RPC completed", false);
        }
    }

    release_lock();
    println!();
    harness.finish();
}

fn acquire_lock(bdf: &str) -> bool {
    let lock_file = lock_file_path();
    let lock_path = Path::new(&lock_file);
    if lock_path.exists() {
        if let Ok(contents) = fs::read_to_string(lock_path) {
            eprintln!("  Lock file exists: {contents}");
            if let Some(pid_str) = contents.lines().find(|l| l.starts_with("pid=")) {
                let pid = pid_str.trim_start_matches("pid=");
                let proc_path = format!("/proc/{pid}");
                if Path::new(&proc_path).exists() {
                    eprintln!("  Process {pid} still alive — refusing to proceed");
                    return false;
                }
                eprintln!("  Stale lock (pid {pid} dead) — overwriting");
            }
        }
    }
    let mut f = match fs::File::create(lock_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  Cannot create lock file: {e}");
            return false;
        }
    };
    let _ = writeln!(f, "pid={}", std::process::id());
    let _ = writeln!(f, "bdf={bdf}");
    let _ = writeln!(
        f,
        "time={}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    println!("  Lock acquired (pid={})", std::process::id());
    true
}

fn release_lock() {
    let _ = fs::remove_file(lock_file_path());
}

fn check_nmi_watchdog(harness: &mut ValidationHarness) -> bool {
    let nmi = fs::read_to_string("/proc/sys/kernel/nmi_watchdog")
        .unwrap_or_default()
        .trim()
        .to_string();
    let softlockup = fs::read_to_string("/proc/sys/kernel/softlockup_panic")
        .unwrap_or_default()
        .trim()
        .to_string();
    let hardlockup = fs::read_to_string("/proc/sys/kernel/hardlockup_panic")
        .unwrap_or_default()
        .trim()
        .to_string();

    println!("  nmi_watchdog:      {nmi}");
    println!("  softlockup_panic:  {softlockup}");
    println!("  hardlockup_panic:  {hardlockup}");

    let active = nmi == "1";
    harness.check_bool("NMI watchdog active", active);
    harness.check_bool("softlockup_panic enabled", softlockup == "1");

    active
}

fn check_mmio_liveness(
    ember: &EmberClient,
    bdf: &str,
    harness: &mut ValidationHarness,
) -> Option<u32> {
    match ember.mmio_read(bdf, 0) {
        Ok(result) => {
            let val = result.value;
            println!("  BOOT0 = 0x{val:08x}");
            let alive = val != 0 && val != 0xFFFF_FFFF;
            harness.check_bool("pre-handoff MMIO alive", alive);
            if alive { Some(val) } else { None }
        }
        Err(e) => {
            println!("  MMIO read failed: {e}");
            harness.check_bool("pre-handoff MMIO alive", false);
            None
        }
    }
}

fn check_pci_config_health(bdf: &str, harness: &mut ValidationHarness) -> bool {
    let config_path = format!("/sys/bus/pci/devices/{bdf}/config");
    match fs::read(&config_path) {
        Ok(data) if data.len() >= 4 => {
            let vendor = u16::from_le_bytes([data[0], data[1]]);
            let device = u16::from_le_bytes([data[2], data[3]]);
            println!("  PCI config: vendor=0x{vendor:04x} device=0x{device:04x}");

            let valid = vendor == 0x10de && device != 0xFFFF;
            harness.check_bool("PCI config space readable", true);
            harness.check_bool("PCI vendor/device valid (NVIDIA)", valid);

            if data.len() >= 8 {
                let status = u16::from_le_bytes([data[6], data[7]]);
                let master_abort = (status & 0x2000) != 0;
                let parity_err = (status & 0x8000) != 0;
                println!(
                    "  PCI status: 0x{status:04x} (master_abort={master_abort}, parity={parity_err})"
                );
                if master_abort || parity_err {
                    eprintln!("  PCI ERROR: status bits indicate bus fault");
                    harness.check_bool("PCI status clean (no bus faults)", false);
                    return false;
                }
                harness.check_bool("PCI status clean (no bus faults)", true);
            }
            valid
        }
        Ok(data) => {
            eprintln!("  PCI config too short: {} bytes", data.len());
            harness.check_bool("PCI config space readable", false);
            false
        }
        Err(e) => {
            eprintln!("  PCI config read failed: {e}");
            harness.check_bool("PCI config space readable", false);
            false
        }
    }
}

fn check_no_stale_module(harness: &mut ValidationHarness) -> bool {
    let modules = fs::read_to_string("/proc/modules").unwrap_or_default();
    let nvsov_loaded = modules.lines().any(|l| l.starts_with("nvsov "));
    let nvidia470_loaded = modules
        .lines()
        .any(|l| l.starts_with("nvidia ") && l.contains("470"));

    if nvsov_loaded {
        eprintln!("  STALE: nvsov module still loaded");
    }
    if nvidia470_loaded {
        eprintln!("  STALE: nvidia-470 module still loaded");
    }

    let clean = !nvsov_loaded && !nvidia470_loaded;
    harness.check_bool("no stale nvsov/nvidia-470 module", clean);
    clean
}

fn call_warm_handoff(
    bdf: &str,
    strategy: &str,
    settle_secs: u64,
) -> Result<serde_json::Value, String> {
    let port_file = PathBuf::from("/run/toadstool/toadstool-jsonrpc-port");
    let port_content =
        fs::read_to_string(&port_file).map_err(|e| format!("cannot read port file: {e}"))?;

    let addr = port_content.trim().trim_start_matches("tcp:");
    let addr = if addr.is_empty() {
        return Err("empty port file".into());
    } else {
        addr.to_string()
    };

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "sovereign.warm_handoff",
        "params": {
            "bdf": bdf,
            "strategy": strategy,
            "settle_secs": settle_secs,
        },
        "id": 1
    });

    println!("  Connecting to toadStool at {addr}...");
    let stream =
        std::net::TcpStream::connect(&addr).map_err(|e| format!("TCP connect to {addr}: {e}"))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(600)))
        .ok();
    stream
        .set_write_timeout(Some(std::time::Duration::from_secs(10)))
        .ok();

    let mut writer = std::io::BufWriter::new(&stream);
    let mut reader = std::io::BufReader::new(&stream);

    serde_json::to_writer(&mut writer, &request).map_err(|e| format!("write request: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write newline: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    println!("  RPC sent, waiting for response (timeout: 600s)...");

    let mut line = String::new();
    std::io::BufRead::read_line(&mut reader, &mut line)
        .map_err(|e| format!("read response: {e}"))?;

    serde_json::from_str(&line).map_err(|e| format!("parse response: {e}"))
}
