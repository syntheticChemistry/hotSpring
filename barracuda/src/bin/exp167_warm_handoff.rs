// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 167: Warm Handoff — Sovereign Compute via Nouveau HBM2 Training
//!
//! Validates the Phase 1 warm handoff path for Titan V (GV100):
//!
//! 1. Swap from vfio-pci to nouveau (PCI rescan, skip_sysfs_unbind)
//! 2. Nouveau trains HBM2 during probe
//! 3. Swap back to vfio-pci (PCI rescan preserves warm state)
//! 4. Ember re-acquires VFIO fds
//! 5. Verify VRAM alive (HBM2 training survived the swap)
//!
//! The critical invariant: `AdaptiveLifecycle` must forward
//! `skip_sysfs_unbind()` from the inner `NvidiaLifecycle`, so the swap
//! uses PCI remove+rescan instead of sysfs `driver/unbind` (which D-states
//! on Volta+ GPUs).
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin exp167_warm_handoff -- [--bdf 0000:03:00.0]
//! ```
//!
//! Requires running coral-glowplug + coral-ember.

use std::path::Path;
use std::time::{Duration, Instant};

use hotspring_barracuda::fleet_client::{EmberClient, FleetDiscovery};
use hotspring_barracuda::glowplug_client::GlowplugClient;
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Warm Handoff — Experiment 167                             ║");
    println!("║  Pipeline: vfio → nouveau (HBM2) → vfio → ember hold      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut harness = ValidationHarness::new("exp167_warm_handoff");

    let args: Vec<String> = std::env::args().collect();
    let bdf = extract_arg(&args, "--bdf").unwrap_or_else(|| {
        std::env::var("HOTSPRING_BDF").unwrap_or_else(|_| "0000:03:00.0".to_string())
    });
    println!("  Target BDF: {bdf}\n");

    let glowplug = connect_glowplug();
    let ember = connect_ember(&bdf);

    let Some(glowplug) = glowplug else {
        eprintln!("  FATAL: glowplug not reachable");
        harness.check_bool("glowplug reachable", false);
        harness.finish();
    };
    harness.check_bool("glowplug reachable", true);

    let Some(ember) = ember else {
        eprintln!("  FATAL: ember not reachable");
        harness.check_bool("ember reachable", false);
        harness.finish();
    };
    harness.check_bool("ember reachable", true);

    // ── Phase 1: Pre-swap baseline ──
    println!("━━━ Phase 1: Pre-Swap Baseline ━━━\n");
    let pre = phase1_baseline(&mut harness, &glowplug, &ember, &bdf);

    // ── Phase 2: Swap vfio → nouveau ──
    println!("\n━━━ Phase 2: Swap to Nouveau (HBM2 Training) ━━━\n");
    let swap_to_nouveau_ok = phase2_swap_to_nouveau(&mut harness, &glowplug, &bdf);
    if !swap_to_nouveau_ok {
        eprintln!("  FATAL: swap to nouveau failed — aborting");
        harness.finish();
    }

    // ── Phase 3: Verify nouveau probed ──
    println!("\n━━━ Phase 3: Verify Nouveau Probe ━━━\n");
    phase3_verify_nouveau(&mut harness, &glowplug, &bdf);

    // ── Phase 4: Swap nouveau → vfio (warm state preserved) ──
    println!("\n━━━ Phase 4: Swap Back to VFIO (Warm Handoff) ━━━\n");
    let swap_to_vfio_ok = phase4_swap_to_vfio(&mut harness, &glowplug, &bdf);
    if !swap_to_vfio_ok {
        eprintln!("  FATAL: swap to vfio failed — aborting");
        harness.finish();
    }

    // ── Phase 5: Verify ember re-acquisition + warm state ──
    println!("\n━━━ Phase 5: Verify Ember Re-Acquisition ━━━\n");
    phase5_verify_ember(&mut harness, &glowplug, &ember, &bdf, &pre);

    // ── Cleanup ──
    println!("\n━━━ Cleanup ━━━\n");
    if let Err(e) = glowplug.experiment_end(&bdf) {
        println!("  experiment_end: {e}");
    }
    println!("  experiment ended for {bdf}");

    println!();
    harness.finish();
}

struct PreSwapState {
    personality: String,
    vram_alive: bool,
    domains_alive: usize,
}

fn phase1_baseline(
    harness: &mut ValidationHarness,
    glowplug: &GlowplugClient,
    ember: &EmberClient,
    bdf: &str,
) -> PreSwapState {
    // Mark experiment start
    if let Err(e) = glowplug.experiment_start(bdf) {
        println!("  experiment_start warning: {e}");
    }

    let detail = match glowplug.device_status(bdf) {
        Ok(d) => {
            println!("  chip:        {}", d.chip);
            println!("  personality: {}", d.personality);
            println!("  power:       {}", d.power);
            println!("  vram_alive:  {}", d.vram_alive);
            println!("  has_vfio_fd: {}", d.has_vfio_fd);
            println!(
                "  domains:     alive={}, faulted={}",
                d.domains_alive, d.domains_faulted
            );
            harness.check_bool("device visible in glowplug", true);
            PreSwapState {
                personality: d.personality.clone(),
                vram_alive: d.vram_alive,
                domains_alive: d.domains_alive,
            }
        }
        Err(e) => {
            println!("  device.get failed: {e}");
            harness.check_bool("device visible in glowplug", false);
            PreSwapState {
                personality: "unknown".into(),
                vram_alive: false,
                domains_alive: 0,
            }
        }
    };

    match ember.status() {
        Ok(v) => {
            let devices = v
                .get("devices")
                .and_then(|d| d.as_array())
                .map_or(0, std::vec::Vec::len);
            let uptime = v
                .get("uptime_secs")
                .and_then(serde_json::value::Value::as_u64)
                .unwrap_or(0);
            println!("  ember: {devices} device(s) held, uptime {uptime}s");
            harness.check_bool("ember healthy", true);
        }
        Err(e) => {
            println!("  ember status: {e}");
            harness.check_bool("ember healthy", false);
        }
    }

    detail
}

fn phase2_swap_to_nouveau(
    harness: &mut ValidationHarness,
    glowplug: &GlowplugClient,
    bdf: &str,
) -> bool {
    println!("  Swapping {bdf} from vfio → nouveau (with trace)...");
    println!("  (This tests skip_sysfs_unbind + PCI rescan path)");
    let start = Instant::now();

    match glowplug.device_swap(bdf, "nouveau", true) {
        Ok(v) => {
            let elapsed = start.elapsed();
            println!("  Swap to nouveau: OK ({elapsed:.1?})");
            if let Some(journal) = v.get("journal_entry") {
                println!("  Journal: {journal}");
            }
            harness.check_bool("swap vfio → nouveau (no D-state)", true);

            println!("  Waiting 15s for nouveau to probe and train HBM2...");
            std::thread::sleep(Duration::from_secs(15));
            true
        }
        Err(e) => {
            let elapsed = start.elapsed();
            eprintln!("  Swap to nouveau FAILED ({elapsed:.1?}): {e}");
            eprintln!("  If this timed out, the skip_sysfs_unbind fix may not be active.");
            harness.check_bool("swap vfio → nouveau (no D-state)", false);
            false
        }
    }
}

fn phase3_verify_nouveau(harness: &mut ValidationHarness, glowplug: &GlowplugClient, bdf: &str) {
    match glowplug.device_status(bdf) {
        Ok(d) => {
            println!("  personality: {}", d.personality);
            println!("  vram_alive:  {}", d.vram_alive);
            println!("  chip:        {}", d.chip);

            let is_nouveau = d.personality.contains("nouveau");
            harness.check_bool("device on nouveau after swap", is_nouveau);

            if d.vram_alive {
                println!("  HBM2 training confirmed (vram_alive=true)");
            } else {
                println!("  WARNING: vram_alive=false — nouveau may not have trained HBM2");
                println!("  (GV100 needs the PMU falcon to fully train, nouveau may stub)");
            }
            harness.check_bool("HBM2 trained by nouveau", d.vram_alive);
        }
        Err(e) => {
            println!("  device.get failed: {e}");
            harness.check_bool("device on nouveau after swap", false);
            harness.check_bool("HBM2 trained by nouveau", false);
        }
    }
}

fn phase4_swap_to_vfio(
    harness: &mut ValidationHarness,
    glowplug: &GlowplugClient,
    bdf: &str,
) -> bool {
    println!("  Swapping {bdf} from nouveau → vfio-pci (PCI rescan preserves warm)...");
    let start = Instant::now();

    match glowplug.device_swap(bdf, "vfio-pci", true) {
        Ok(v) => {
            let elapsed = start.elapsed();
            println!("  Swap to vfio-pci: OK ({elapsed:.1?})");
            if let Some(journal) = v.get("journal_entry") {
                println!("  Journal: {journal}");
            }
            harness.check_bool("swap nouveau → vfio-pci (PCI rescan)", true);

            println!("  Waiting 5s for ember to re-acquire...");
            std::thread::sleep(Duration::from_secs(5));
            true
        }
        Err(e) => {
            let elapsed = start.elapsed();
            eprintln!("  Swap to vfio-pci FAILED ({elapsed:.1?}): {e}");
            harness.check_bool("swap nouveau → vfio-pci (PCI rescan)", false);
            false
        }
    }
}

fn phase5_verify_ember(
    harness: &mut ValidationHarness,
    glowplug: &GlowplugClient,
    ember: &EmberClient,
    bdf: &str,
    pre: &PreSwapState,
) {
    // Verify ember re-acquired
    match ember.status() {
        Ok(v) => {
            let devices: Vec<String> = v
                .get("devices")
                .and_then(|d| d.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let held = devices.contains(&bdf.to_string());
            println!("  ember devices: {devices:?}");
            println!("  {bdf} held by ember: {held}");
            harness.check_bool("ember re-acquired device after swap", held);
        }
        Err(e) => {
            println!("  ember status: {e}");
            harness.check_bool("ember re-acquired device after swap", false);
        }
    }

    // Verify device state via glowplug
    match glowplug.device_status(bdf) {
        Ok(d) => {
            let is_vfio = d.personality.contains("vfio");
            println!("  personality: {} (want vfio)", d.personality);
            println!("  has_vfio_fd: {}", d.has_vfio_fd);
            println!("  vram_alive:  {}", d.vram_alive);
            println!(
                "  domains:     alive={}, faulted={}",
                d.domains_alive, d.domains_faulted
            );
            println!("  power:       {}", d.power);

            harness.check_bool("device back on vfio after round-trip", is_vfio);
            harness.check_bool("ember holds VFIO fd post-swap", d.has_vfio_fd);

            // The key result: did HBM2 training survive the round-trip?
            if d.vram_alive {
                println!(
                    "\n  ✓ WARM HANDOFF SUCCESS: HBM2 state preserved through nouveau→vfio swap"
                );
                println!("    Device is ready for open_warm() → GPFIFO channel → compute dispatch");
            } else {
                println!("\n  ✗ WARM HANDOFF: VRAM not alive after round-trip");
                println!("    Possible causes:");
                println!("    - PCI rescan tore down too much state");
                println!("    - Nouveau did not fully train HBM2 (PMU stub)");
                println!("    - Domain health check is too aggressive for post-swap state");
            }
            harness.check_bool("VRAM alive after warm handoff round-trip", d.vram_alive);

            // Compare with pre-swap state
            println!("\n  Pre/post comparison:");
            println!("    personality: {} → {}", pre.personality, d.personality);
            println!("    vram_alive:  {} → {}", pre.vram_alive, d.vram_alive);
            println!(
                "    domains:     {} → {}",
                pre.domains_alive, d.domains_alive
            );
        }
        Err(e) => {
            println!("  device.get failed: {e}");
            harness.check_bool("device back on vfio after round-trip", false);
            harness.check_bool("ember holds VFIO fd post-swap", false);
            harness.check_bool("VRAM alive after warm handoff round-trip", false);
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
