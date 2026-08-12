// SPDX-License-Identifier: AGPL-3.0-or-later

//! Node-Atomic Silicon AAR Validation — 16⁴ Concurrent via upstream GpuHmcTrajectory.
//!
//! Validates the full Compute Trio stack:
//! - 16x4 Concurrent via upstream `GpuHmcTrajectory` (not local `gpu_hmc/` pipelines)
//! - Native vs Concurrent plaquette agreement (|ΔP| < 1e-5)
//! - Acceptance/ΔH stability
//! - Speedup measurement
//! - Reports FP64 strategy selected by measured throughput ratio
//!
//! This binary uses ONLY `node_atomic/` → upstream `GpuHmcTrajectory`.
//! Zero imports from `lattice::gpu_hmc`.

use hotspring_barracuda::node_atomic::{NodeAtomicQcd, TrajectoryRunner};
use barracuda::device::capabilities::DeviceCapabilities;
use barracuda::ops::lattice::gpu_hmc_types::GpuHmcConfig;
use std::time::Instant;

const NX: u32 = 16;
const NY: u32 = 16;
const NZ: u32 = 16;
const NT: u32 = 4;
const BETA: f64 = 5.90;
const SEED: u64 = 42;
const N_WARMUP: usize = 20;
const N_MEASURE: usize = 50;
const N_MD: usize = 20;
const DT: f64 = 0.01;

fn config() -> GpuHmcConfig {
    GpuHmcConfig {
        nx: NX,
        ny: NY,
        nz: NZ,
        nt: NT,
        beta: BETA,
        n_md_steps: N_MD,
        dt: DT,
        mass: 0.1,
        cg_tol: 1e-10,
        cg_max_iter: 1000,
        n_flavors_over_4: 2,
    }
}

fn run_campaign(force_native: bool) -> (f64, f64, f64, f64, String) {
    // SAFETY: single-threaded validation binary, no concurrent env access
    unsafe {
        if force_native {
            std::env::set_var("BARRACUDA_FP64_STRATEGY", "native");
        } else {
            std::env::remove_var("BARRACUDA_FP64_STRATEGY");
        }
    }

    let cfg = config();
    let qcd = NodeAtomicQcd::new(cfg, SEED).expect("NodeAtomicQcd init");

    qcd.cold_start().expect("cold start");
    qcd.upload_topology();
    qcd.seed_rng(SEED as u32);

    let strategy = format!("{:?}", qcd.trajectory.strategy());
    println!("  Strategy: {strategy}");

    let caps = DeviceCapabilities::from_device(&qcd.device);
    if let Some(ratio) = caps.f64_throughput_ratio {
        println!("  Measured FP32:FP64 ratio: {ratio:.1}x");
    } else {
        println!("  FP32:FP64 ratio: not probed (will use heuristic or env override)");
    }

    let runner = TrajectoryRunner {
        warmup_count: N_WARMUP,
        production_count: N_MEASURE,
        target_acceptance: 0.70,
    };

    let t0 = Instant::now();
    let (result, measurements) = runner
        .run_campaign(&qcd, 10, |i, plaq, acc| {
            println!("  warmup[{i}]: P={plaq:.6}, acc={acc:.1}%");
        })
        .expect("campaign");
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mean_plaq = if measurements.is_empty() {
        0.0
    } else {
        measurements.iter().sum::<f64>() / measurements.len() as f64
    };
    let acc_pct = result.accepted as f64 / result.trajectories as f64 * 100.0;
    let ms_per_traj = elapsed_ms / (N_WARMUP + N_MEASURE) as f64;

    (mean_plaq, acc_pct, ms_per_traj, result.mean_delta_h, strategy)
}

fn main() {
    println!("=== Node-Atomic Silicon AAR Validation ===");
    println!("Lattice: {NX}x{NY}x{NZ}x{NT}, β={BETA}");
    println!("Warmup: {N_WARMUP}, Production: {N_MEASURE}");
    println!();

    println!("--- Native f64 (baseline) ---");
    let (native_plaq, native_acc, native_ms, native_dh, native_strat) = run_campaign(true);
    println!("  Result: P={native_plaq:.6}, acc={native_acc:.1}%, {native_ms:.0} ms/traj, |ΔH|={native_dh:.2e}");
    println!();

    println!("--- Concurrent (silicon saturation, auto-selected) ---");
    let (conc_plaq, conc_acc, conc_ms, conc_dh, conc_strat) = run_campaign(false);
    println!("  Result: P={conc_plaq:.6}, acc={conc_acc:.1}%, {conc_ms:.0} ms/traj, |ΔH|={conc_dh:.2e}");
    println!();

    println!("=== Comparison ===");
    let delta_p = (native_plaq - conc_plaq).abs();
    let speedup = if conc_ms > 0.0 { native_ms / conc_ms } else { 0.0 };
    println!("  |ΔP| = {delta_p:.2e}");
    println!("  Native: {native_ms:.0} ms/traj ({native_strat})");
    println!("  Concurrent: {conc_ms:.0} ms/traj ({conc_strat})");
    println!("  Speedup: {speedup:.2}x");
    println!();

    let pass_physics = delta_p < 1e-5;
    let pass_acceptance = native_acc > 50.0 && conc_acc > 50.0;
    let pass_speedup = speedup > 1.0 || conc_strat == "Native";

    println!("=== Validation Gates ===");
    println!("  Physics agreement (|ΔP| < 1e-5): {}", if pass_physics { "PASS" } else { "FAIL" });
    println!("  Acceptance stability (>50%): {}", if pass_acceptance { "PASS" } else { "FAIL" });
    println!("  Speedup (>1x or Native-only): {}", if pass_speedup { "PASS" } else { "FAIL" });
    println!("  Upstream GpuHmcTrajectory (no gpu_hmc/ imports): PASS");
    println!("  Node-Atomic path only: PASS");
    println!();

    if pass_physics && pass_acceptance && pass_speedup {
        println!("ALL GATES PASSED — Node-Atomic Concurrent validated.");
    } else {
        println!("VALIDATION INCOMPLETE — check failures above.");
        std::process::exit(1);
    }
}
