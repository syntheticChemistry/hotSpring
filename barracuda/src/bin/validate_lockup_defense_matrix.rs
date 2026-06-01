// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lockup Defense Matrix Validation
//!
//! Verifies all 5 diesel engine defense mechanisms are active and functional
//! for the given GPU fleet. References the crash vector catalog from Exp 229/232.
//!
//! ## Defenses validated
//!
//! 1. Interrupt quench (InterruptProfile generation-aware INTR_EN disable)
//! 2. Post-exit quench (pmc::quench_interrupts + intx_disable after nvidia_close)
//! 3. Exclusion guard (HandoffExclusionGuard mutual exclusion with keepalive)
//! 4. Fire-and-poll unbind (330s deadline sysfs driver rotation)
//! 5. Kernel sentinel (/dev/kmsg crash signature capture)
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin validate_lockup_defense_matrix
//! ```

use std::time::Instant;

use hotspring_barracuda::validation::ValidationHarness;

#[path = "../bin_helpers/sovereignty/mod.rs"]
mod sovereignty;
use sovereignty::connect::{extract_arg, try_connect_ember, try_connect_glowplug};
use sovereignty::lockup_vectors::{ALL_VECTORS, DEFENSE_MECHANISMS, CrashCategory};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Lockup Defense Matrix — Exp 229/232 Validation            ║");
    println!("║  9 crash vectors, 5 defense mechanisms                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut harness = ValidationHarness::new("validate_lockup_defense_matrix");
    let args: Vec<String> = std::env::args().collect();
    let bdf = extract_arg(&args, "--bdf")
        .unwrap_or_else(|| std::env::var("HOTSPRING_BDF").unwrap_or_else(|_| "0000:02:00.0".to_string()));

    let t0 = Instant::now();

    // Phase 1: Fleet connectivity
    println!("── Phase 1: Fleet connectivity ──────────────────────────────\n");

    let glowplug = try_connect_glowplug();
    harness.check_bool("glowplug reachable", glowplug.is_some());

    let ember = try_connect_ember(&bdf);
    harness.check_bool(&format!("ember reachable for {bdf}"), ember.is_some());

    let Some(glowplug) = glowplug else {
        eprintln!("  FATAL: glowplug not available — cannot validate defense matrix");
        harness.finish();
    };

    let Some(ember) = ember else {
        eprintln!("  FATAL: ember not available for {bdf} — cannot validate defenses");
        harness.finish();
    };

    // Phase 2: Device health baseline
    println!("\n── Phase 2: Device health baseline ─────────────────────────\n");

    match glowplug.list_devices() {
        Ok(devices) => {
            println!("  Fleet: {} device(s) registered", devices.len());
            harness.check_bool("fleet has devices", !devices.is_empty());
            for dev in &devices {
                let protected = if dev.protected { " [PROTECTED]" } else { "" };
                let name = dev.name.as_deref().unwrap_or("unknown");
                println!(
                    "    {} — {} / {}{}",
                    dev.bdf, name, dev.personality, protected
                );
            }
        }
        Err(e) => {
            eprintln!("  device.list failed: {e}");
            harness.check_bool("fleet has devices", false);
        }
    }

    match ember.mmio_read(&bdf, 0x000000) {
        Ok(r) => {
            let boot0 = r.value;
            let alive = boot0 != 0xFFFF_FFFF && boot0 != 0x0000_0000;
            println!(
                "  BOOT0 = 0x{:08x} — {}",
                boot0,
                if alive { "GPU alive" } else { "DEAD or D3cold" }
            );
            harness.check_bool("GPU BOOT0 alive", alive);
        }
        Err(e) => {
            eprintln!("  BOOT0 read failed: {e}");
            harness.check_bool("GPU BOOT0 alive", false);
        }
    }

    // Phase 3: Defense mechanism probes
    println!("\n── Phase 3: Defense mechanism probes ───────────────────────\n");

    for mechanism in DEFENSE_MECHANISMS {
        match glowplug.rpc_call(
            "sovereign.defense_status",
            &serde_json::json!({
                "bdf": bdf,
                "mechanism": mechanism,
            }),
        ) {
            Ok(resp) => {
                let active = resp.get("active").and_then(|v| v.as_bool()).unwrap_or(false);
                println!(
                    "  {}: {}",
                    mechanism,
                    if active { "ACTIVE" } else { "INACTIVE" }
                );
                harness.check_bool(&format!("defense.{mechanism}"), active);
            }
            Err(e) => {
                println!("  {}: probe failed ({e}) — marking as unknown", mechanism);
                harness.check_bool(&format!("defense.{mechanism}"), false);
            }
        }
    }

    // Phase 4: Watchdog heartbeat
    println!("\n── Phase 4: Watchdog heartbeat ─────────────────────────────\n");

    match glowplug.rpc_call("sovereign.watchdog_status", &serde_json::json!({ "bdf": bdf })) {
        Ok(resp) => {
            let running = resp.get("running").and_then(|v| v.as_bool()).unwrap_or(false);
            let timeout_s = resp.get("timeout_s").and_then(|v| v.as_u64()).unwrap_or(0);
            println!(
                "  Watchdog: {} (timeout: {}s)",
                if running { "RUNNING" } else { "STOPPED" },
                timeout_s
            );
            harness.check_bool("catalyst_watchdog running", running);
        }
        Err(e) => {
            println!("  Watchdog probe failed: {e}");
            harness.check_bool("catalyst_watchdog running", false);
        }
    }

    // Phase 5: Crash vector catalog
    println!("\n── Phase 5: Crash vector catalog ───────────────────────────\n");

    let kills = ALL_VECTORS
        .iter()
        .filter(|v| v.category == CrashCategory::ConfirmedKill)
        .count();
    let hangs = ALL_VECTORS
        .iter()
        .filter(|v| v.category == CrashCategory::ConfirmedHang)
        .count();
    println!(
        "  Catalog: {} confirmed kills, {} confirmed hangs",
        kills, hangs
    );
    println!("  All vectors have documented defenses in diesel engine\n");

    for vector in ALL_VECTORS {
        println!(
            "  [{}] {} — {}",
            vector.id, vector.description, vector.defense
        );
    }

    harness.check_bool("crash vector catalog complete", ALL_VECTORS.len() >= 9);

    // Summary
    let elapsed = t0.elapsed();
    println!("\n═══════════════════════════════════════════════════════════════");
    println!(
        "  Lockup defense matrix validation complete ({:.1}s)",
        elapsed.as_secs_f64()
    );
    println!("═══════════════════════════════════════════════════════════════");

    harness.finish();
}
