// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 155: K80 Warm-Cycle FECS Dispatch
//!
//! Validates the highest-ROI sovereign compute path for Tesla K80 (GK210):
//! no ACR barrier — once VRAM is alive via nouveau warm cycle, FECS can be
//! uploaded and started directly.
//!
//! ## Pipeline (all through ember IPC)
//!
//! 1. Swap K80 to nouveau (warm cycle trains VRAM)
//! 2. Swap back to vfio-pci (ember regains MMIO)
//! 3. Verify VRAM alive via PRAMIN probe
//! 4. FECS PIO upload via falcon.upload_imem
//! 5. FECS start via falcon.start_cpu
//! 6. Poll FECS via falcon.poll + fecs.state
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin exp155_k80_warm_fecs -- [--bdf 0000:41:00.0]
//! ```
//!
//! Requires running coral-glowplug + coral-ember fleet.

use std::path::Path;
use std::time::Duration;

use hotspring_barracuda::ember_types::MmioBatchOp;
use hotspring_barracuda::fleet_client::{EmberClient, FleetDiscovery};
use hotspring_barracuda::glowplug_client::GlowplugClient;
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::validation::ValidationHarness;

const PRAMIN_WINDOW: u32 = 0x700000;
const FECS_BASE: u32 = 0x409000;
const PGRAPH_STATUS: u32 = 0x400700;

const VRAM_DEAD_PATTERNS: [u32; 5] = [
    0xBAD0_AC00,
    0xBAD0_AC01,
    0xBAD0_AC02,
    0xBADF_3000,
    0xFFFF_FFFF,
];

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  K80 Warm-Cycle FECS Dispatch — Experiment 155             ║");
    println!("║  Pipeline: nouveau warm → VFIO → FECS upload → dispatch    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut harness = ValidationHarness::new("exp155_k80_warm_fecs");

    let args: Vec<String> = std::env::args().collect();
    let bdf = extract_arg(&args, "--bdf").unwrap_or_else(|| {
        std::env::var("HOTSPRING_BDF").unwrap_or_else(|_| "0000:41:00.0".to_string())
    });
    println!("  Target BDF: {bdf}\n");

    // ── Connect to glowplug + ember ──
    let glowplug = connect_glowplug();
    let ember = connect_ember(&bdf);

    let Some(glowplug) = glowplug else {
        harness.check_bool("glowplug reachable", false);
        harness.finish();
    };
    harness.check_bool("glowplug reachable", true);

    let Some(ember) = ember else {
        harness.check_bool("ember reachable for K80", false);
        harness.finish();
    };
    harness.check_bool("ember reachable for K80", true);

    // ── Phase 1: Pre-warm MMIO baseline ──
    println!("━━━ Phase 1: Pre-Warm Baseline ━━━\n");
    phase1_baseline(&mut harness, &ember, &bdf);

    // ── Phase 2: Nouveau warm cycle (skip for cold VFIO K80) ──
    // Cold K80 in VFIO goes D-state on ANY driver swap attempt (unbind/remove).
    // Detect cold state from Phase 1 PGRAPH reading and skip the warm cycle.
    let pgraph_cold = ember
        .mmio_read(&bdf, PGRAPH_STATUS)
        .map_or(true, |r| r.value & 0xFFFF_0000 == 0xBADF_0000);

    println!("\n━━━ Phase 2: Nouveau Warm Cycle ━━━\n");
    let warm_ok = if pgraph_cold {
        println!(
            "  Cold VFIO K80 detected (PGRAPH={:#010x}) — skipping warm cycle",
            ember.mmio_read(&bdf, PGRAPH_STATUS).map_or(0, |r| r.value)
        );
        println!("  (Cold Kepler goes D-state on driver swap; use agentReagents VM to POST)");
        harness.check_bool("warm cycle skipped (cold VFIO)", true);
        false
    } else {
        phase2_warm_cycle(&mut harness, &glowplug, &ember, &bdf)
    };

    // ── Phase 3: VRAM liveness check ──
    println!("\n━━━ Phase 3: VRAM Liveness ━━━\n");
    let vram_alive = phase3_vram_check(&mut harness, &ember, &bdf);

    // ── Phase 3b: PGRAPH enable (cold K80 needs PMC_ENABLE bit 12) ──
    if pgraph_cold {
        println!("\n━━━ Phase 3b: PMC PGRAPH Enable (cold K80 path) ━━━\n");
        phase3b_pmc_pgraph_enable(&mut harness, &ember, &bdf);
    }

    // ── Phase 4: FECS upload ──
    // IMEM upload is PIO — proceed even with cold VRAM (like exp154 IMEM path).
    println!("\n━━━ Phase 4: FECS Upload ━━━\n");
    if !vram_alive && !warm_ok {
        println!("  NOTE: VRAM dead, warm cycle skipped — proceeding with IMEM-only path");
    }
    phase4_fecs_upload(&mut harness, &ember, &bdf);

    // ── Phase 5: FECS start + poll ──
    println!("\n━━━ Phase 5: FECS Start + Poll ━━━\n");
    if !vram_alive && !warm_ok {
        println!("  NOTE: Cold K80 — FECS results are best-effort");
    }
    phase5_fecs_start(&mut harness, &ember, &bdf);

    // ── Phase 6: FECS state query ──
    println!("\n━━━ Phase 6: FECS State ━━━\n");
    phase6_fecs_state(&mut harness, &ember, &bdf);

    // ── Cleanup ──
    println!("\n━━━ Cleanup ━━━\n");
    if let Ok(result) = ember.cleanup_dma(&bdf) {
        println!(
            "  DMA cleanup: ok={}, decontaminated={:?}",
            result.ok, result.decontaminated
        );
    }
    if glowplug.experiment_end(&bdf).is_ok() {
        println!("  Experiment ended for {bdf}");
    }

    println!();
    harness.finish();
}

fn phase1_baseline(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) {
    let ops = vec![
        MmioBatchOp::read(0x000000), // BOOT0
        MmioBatchOp::read(0x000200), // PMC_ENABLE
        MmioBatchOp::read(PGRAPH_STATUS),
        MmioBatchOp::read(PRAMIN_WINDOW),
    ];

    match ember.mmio_batch(bdf, &ops) {
        Ok(result) => {
            println!("  MMIO batch (pre-warm):");
            for (i, op) in ops.iter().enumerate() {
                let val = result.read_value(i).unwrap_or(0xDEAD_DEAD);
                println!("    [{:#08x}] = {val:#010x}", op.offset);
            }
            harness.check_bool("pre-warm MMIO batch readable", true);

            let pramin_val = result.read_value(ops.len() - 1).unwrap_or(0);
            let dead = VRAM_DEAD_PATTERNS
                .iter()
                .any(|&p| pramin_val & 0xFFFF_FF00 == p & 0xFFFF_FF00);
            println!("  PRAMIN[{PRAMIN_WINDOW:#x}] = {pramin_val:#010x} (dead={dead})");
        }
        Err(e) => {
            println!("  MMIO batch failed: {e}");
            harness.check_bool("pre-warm MMIO batch readable", false);
        }
    }
}

fn phase2_warm_cycle(
    harness: &mut ValidationHarness,
    glowplug: &GlowplugClient,
    ember: &EmberClient,
    bdf: &str,
) -> bool {
    // Mark experiment start
    if let Err(e) = glowplug.experiment_start(bdf) {
        println!("  experiment_start warning: {e}");
    }

    // Swap to nouveau (warm cycle)
    println!("  Swapping {bdf} to nouveau (with trace)...");
    match glowplug.device_swap(bdf, "nouveau", true) {
        Ok(_) => {
            println!("  Swap to nouveau: OK");
            harness.check_bool("swap to nouveau", true);
        }
        Err(e) => {
            println!("  Swap to nouveau failed: {e}");
            println!("  (Cold VFIO K80 — D-state expected, proceeding without warm cycle)");
            harness.check_bool("swap to nouveau", false);
            return false;
        }
    }

    // Wait for nouveau to train VRAM
    println!("  Waiting 5s for nouveau to POST and train VRAM...");
    std::thread::sleep(Duration::from_secs(5));

    // Swap back to vfio-pci
    println!("  Swapping {bdf} back to vfio-pci...");
    match glowplug.device_swap(bdf, "vfio-pci", true) {
        Ok(_) => {
            println!("  Swap to vfio-pci: OK");
            harness.check_bool("swap back to vfio-pci", true);
        }
        Err(e) => {
            println!("  Swap to vfio-pci failed: {e}");
            harness.check_bool("swap back to vfio-pci", false);
            return false;
        }
    }

    // Wait for ember to re-acquire the device
    println!("  Waiting for ember to re-acquire device...");
    std::thread::sleep(Duration::from_secs(3));

    // Prepare DMA (AER mask, optional bus master)
    match ember.prepare_dma(bdf, false) {
        Ok(r) => {
            println!(
                "  DMA prepare: pmc_before={:?}, pmc_after={:?}",
                r.pmc_before, r.pmc_after
            );
            harness.check_bool("DMA prepare after warm cycle", r.ok);
        }
        Err(e) => {
            println!("  DMA prepare failed: {e}");
            harness.check_bool("DMA prepare after warm cycle", false);
        }
    }
    true
}

fn phase3_vram_check(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) -> bool {
    match ember.mmio_read(bdf, PRAMIN_WINDOW) {
        Ok(result) => {
            let val = result.value;
            let dead = VRAM_DEAD_PATTERNS
                .iter()
                .any(|&p| val & 0xFFFF_FF00 == p & 0xFFFF_FF00);
            println!("  PRAMIN[{PRAMIN_WINDOW:#x}] = {val:#010x}");
            println!("  VRAM alive: {}", !dead);
            harness.check_bool("VRAM alive after warm cycle", !dead);

            if !dead {
                // Read a range to further verify
                match ember.pramin_read(bdf, PRAMIN_WINDOW as u64, 16) {
                    Ok(data) => {
                        println!(
                            "  PRAMIN read (16 bytes): {:02x?}",
                            &data[..data.len().min(16)]
                        );
                        harness.check_bool("PRAMIN bulk read succeeds", true);
                    }
                    Err(e) => {
                        println!("  PRAMIN read failed: {e}");
                        harness.check_bool("PRAMIN bulk read succeeds", false);
                    }
                }
            }

            !dead
        }
        Err(e) => {
            println!("  PRAMIN read failed: {e}");
            harness.check_bool("VRAM alive after warm cycle", false);
            false
        }
    }
}

/// Enable PGRAPH in PMC_ENABLE (bit 12) for cold K80.
/// On cold VFIO Kepler, PGRAPH is disabled so FECS registers read as 0xbadf1200.
fn phase3b_pmc_pgraph_enable(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) {
    const PMC_ENABLE: u32 = 0x200;
    const PGRAPH_BIT: u32 = 1 << 12;

    match ember.mmio_read(bdf, PMC_ENABLE) {
        Ok(r) => {
            let before = r.value;
            println!("  PMC_ENABLE before: {before:#010x}");
            let has_pgraph = before & PGRAPH_BIT != 0;
            if has_pgraph {
                println!("  PGRAPH already enabled (bit 12 set)");
                harness.check_bool("PMC PGRAPH enable", true);
                return;
            }
            let target = before | PGRAPH_BIT;
            println!("  Enabling PGRAPH: {before:#010x} → {target:#010x}");
            match ember.mmio_write(bdf, PMC_ENABLE, target) {
                Ok(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    let after = ember.mmio_read(bdf, PMC_ENABLE).map_or(0, |r| r.value);
                    println!("  PMC_ENABLE after: {after:#010x}");
                    let pgraph_live = after & PGRAPH_BIT != 0;
                    println!("  PGRAPH enabled: {pgraph_live}");

                    let fecs_val = ember
                        .mmio_read(bdf, PGRAPH_STATUS)
                        .map_or(0xDEAD, |r| r.value);
                    println!("  FECS PGRAPH_STATUS after enable: {fecs_val:#010x}");
                    let fecs_responsive = fecs_val != 0xBADF_1201 && fecs_val != 0xBADF_1200;
                    println!("  FECS responsive: {fecs_responsive}");
                    harness.check_bool("PMC PGRAPH enable", pgraph_live);
                }
                Err(e) => {
                    println!("  PMC_ENABLE write failed: {e}");
                    harness.check_bool("PMC PGRAPH enable", false);
                }
            }
        }
        Err(e) => {
            println!("  PMC_ENABLE read failed: {e}");
            harness.check_bool("PMC PGRAPH enable", false);
        }
    }
}

fn phase4_fecs_upload(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) {
    // Minimal FECS micro-firmware: NOP sled + halt
    // For GK210, FECS IMEM uses PIO upload at base 0x409000
    let nop_halt_fw: Vec<u8> = {
        let mut fw = Vec::new();
        // 16 NOPs (falcon NOP = 0x00000000 in 32-bit encoding)
        for _ in 0..16 {
            fw.extend_from_slice(&0x0000_0000u32.to_le_bytes());
        }
        // HALT: falcon uses 0xF8000000 (illegal instruction) as halting pattern
        fw.extend_from_slice(&0xF800_0000u32.to_le_bytes());
        fw
    };

    println!(
        "  Uploading FECS micro-firmware ({} bytes)...",
        nop_halt_fw.len()
    );

    match ember.falcon_upload_imem(bdf, FECS_BASE, 0, &nop_halt_fw, 0, false) {
        Ok(result) => {
            println!(
                "  FECS IMEM upload: ok={}, bytes={:?}",
                result.ok, result.bytes
            );
            harness.check_bool("FECS IMEM upload", result.ok);
        }
        Err(e) => {
            println!("  FECS IMEM upload failed: {e}");
            harness.check_bool("FECS IMEM upload", false);
        }
    }
}

fn phase5_fecs_start(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) {
    match ember.falcon_start_cpu(bdf, FECS_BASE) {
        Ok(result) => {
            println!(
                "  FECS start: ok={}, pc={:?}, exci={:?}, cpuctl={:?}",
                result.ok,
                result.pc.map(|v| format!("{v:#010x}")),
                result.exci.map(|v| format!("{v:#010x}")),
                result.cpuctl.map(|v| format!("{v:#010x}")),
            );
            harness.check_bool("FECS falcon start_cpu", result.ok);
        }
        Err(e) => {
            println!("  FECS start failed: {e}");
            harness.check_bool("FECS falcon start_cpu", false);
            return;
        }
    }

    // Poll for halt or mailbox activity
    println!("  Polling FECS (5s timeout, sentinel 0xDEADA5A5)...");
    match ember.falcon_poll(bdf, FECS_BASE, 5000, 0xDEAD_A5A5) {
        Ok(result) => {
            println!(
                "  FECS poll: {} snapshots, pc_trace={:?}",
                result.snapshots.len(),
                result.pc_trace
            );
            if let Some(final_state) = &result.final_state {
                println!(
                    "  Final: pc={:?}, cpuctl={:?}, mb0={:?}",
                    final_state.pc.map(|v| format!("{v:#010x}")),
                    final_state.cpuctl.map(|v| format!("{v:#010x}")),
                    final_state.mailbox0.map(|v| format!("{v:#010x}")),
                );
            }
            harness.check_bool("FECS falcon poll completed", true);
        }
        Err(e) => {
            println!("  FECS poll failed: {e}");
            harness.check_bool("FECS falcon poll completed", false);
        }
    }
}

fn phase6_fecs_state(harness: &mut ValidationHarness, ember: &EmberClient, bdf: &str) {
    match ember.fecs_state(bdf) {
        Ok(state) => {
            println!(
                "  FECS state: {}",
                serde_json::to_string_pretty(&state).unwrap_or_default()
            );
            harness.check_bool("ember.fecs.state readable", true);
        }
        Err(e) => {
            println!("  FECS state failed: {e}");
            harness.check_bool("ember.fecs.state readable", false);
        }
    }
}

fn connect_glowplug() -> Option<GlowplugClient> {
    let nucleus = NucleusContext::detect();
    if let Ok(g) = GlowplugClient::from_nucleus(&nucleus) {
        Some(g)
    } else {
        let sock = Path::new("/run/coralreef/glowplug.sock");
        if sock.exists() {
            Some(GlowplugClient::from_socket(sock))
        } else {
            None
        }
    }
}

fn connect_ember(bdf: &str) -> Option<EmberClient> {
    if let Ok(disc) = FleetDiscovery::load_default() {
        if let Some(sock) = disc.file().routes.get(bdf) {
            return Some(EmberClient::connect(sock));
        }
    }
    let slug = bdf.replace(':', "-");
    let fleet_sock = format!("/run/coralreef/fleet/ember-{slug}.sock");
    if Path::new(&fleet_sock).exists() {
        return Some(EmberClient::connect(&fleet_sock));
    }
    let per_device = format!("/run/coralreef/ember-{slug}.sock");
    if Path::new(&per_device).exists() {
        return Some(EmberClient::connect(&per_device));
    }
    let legacy = Path::new("/run/coralreef/ember.sock");
    if legacy.exists() {
        return Some(EmberClient::connect(legacy));
    }
    None
}

fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}
