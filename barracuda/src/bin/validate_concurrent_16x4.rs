// SPDX-License-Identifier: AGPL-3.0-or-later

//! Quick Concurrent vs Native validation at 16⁴.
//!
//! Runs short HMC campaigns (50 trajectories) with both strategies on the same
//! seed and compares plaquette, acceptance rate, and wall-clock time.
//! Used to validate the Concurrent precision retune on narrow-rate hardware.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::sync::Arc;
use std::time::Instant;

const L: usize = 16;
const BETA: f64 = 5.90;
const SEED: u64 = 42;
const N_WARMUP: usize = 20;
const N_MEASURE: usize = 50;
const N_MD: usize = 20;
const DT: f64 = 0.01;

fn run_campaign(strategy: &str) -> (f64, f64, f64, f64) {
    unsafe { std::env::set_var("HOTSPRING_FP64_STRATEGY", strategy) };

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");
    let pipelines = GpuHmcStreamingPipelines::new(&gpu);

    let lattice = Lattice::hot_start([L, L, L, L], BETA, SEED);
    let state = GpuHmcState::from_lattice(&gpu, &lattice, BETA);

    let mut rng_seed = SEED + 1;
    let mut plaqs = Vec::new();
    let mut accepts = 0usize;
    let mut total = 0usize;

    for _ in 0..N_WARMUP {
        let _ = gpu_hmc_trajectory_streaming(&gpu, &pipelines, &state, N_MD, DT, 0, &mut rng_seed);
    }

    let t0 = Instant::now();
    for _ in 0..N_MEASURE {
        match gpu_hmc_trajectory_streaming(&gpu, &pipelines, &state, N_MD, DT, 0, &mut rng_seed) {
            Ok(result) => {
                plaqs.push(result.plaquette);
                if result.accepted {
                    accepts += 1;
                }
                total += 1;
            }
            Err(e) => {
                eprintln!("  trajectory error: {e}");
                total += 1;
            }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let mean_plaq = if plaqs.is_empty() {
        f64::NAN
    } else {
        plaqs.iter().sum::<f64>() / plaqs.len() as f64
    };
    let acc_rate = accepts as f64 / total as f64;

    unsafe { std::env::remove_var("HOTSPRING_FP64_STRATEGY") };

    (mean_plaq, acc_rate, elapsed, elapsed / N_MEASURE as f64)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!(
        "║  Concurrent vs Native Validation — 16⁴ β={:.2}                  ║",
        BETA
    );
    println!("║  {N_WARMUP} warmup + {N_MEASURE} production trajectories per strategy         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    println!("═══ Running NATIVE strategy ═══");
    let (plaq_native, acc_native, time_native, traj_native) = run_campaign("native");
    println!(
        "  Native: ⟨P⟩={:.6}, acc={:.1}%, time={:.1}s ({:.1}ms/traj)",
        plaq_native,
        acc_native * 100.0,
        time_native,
        traj_native * 1000.0
    );
    println!();

    println!("═══ Running CONCURRENT strategy ═══");
    let (plaq_conc, acc_conc, time_conc, traj_conc) = run_campaign("concurrent");
    println!(
        "  Concurrent: ⟨P⟩={:.6}, acc={:.1}%, time={:.1}s ({:.1}ms/traj)",
        plaq_conc,
        acc_conc * 100.0,
        time_conc,
        traj_conc * 1000.0
    );
    println!();

    println!("═══ Comparison ═══");
    let delta_plaq = (plaq_conc - plaq_native).abs();
    let speedup = time_native / time_conc;
    println!("  |ΔP| = {delta_plaq:.2e} (expect < 0.01 from statistics)");
    println!("  Speedup: {speedup:.2}x (Concurrent/Native)");
    println!(
        "  Acceptance: Native {:.1}% vs Concurrent {:.1}%",
        acc_native * 100.0,
        acc_conc * 100.0
    );
    println!();

    let plaq_ok = delta_plaq < 0.02;
    let acc_ok = (acc_conc - acc_native).abs() < 0.3;
    println!(
        "  Plaquette agreement: {}",
        if plaq_ok { "✓ PASS" } else { "✗ FAIL" }
    );
    println!(
        "  Acceptance agreement: {}",
        if acc_ok { "✓ PASS" } else { "✗ FAIL" }
    );

    if plaq_ok && acc_ok {
        println!("\n  ═══ CONCURRENT VALIDATION PASSED ═══");
    } else {
        println!("\n  ═══ CONCURRENT VALIDATION FAILED ═══");
        std::process::exit(1);
    }
}
