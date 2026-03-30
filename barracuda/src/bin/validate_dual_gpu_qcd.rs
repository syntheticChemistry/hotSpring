// SPDX-License-Identifier: AGPL-3.0-only

//! Dual-GPU parallel QCD validation: run the same physics on two GPUs simultaneously.
//!
//! Uses `discover_primary_and_secondary_adapters()` or env vars
//! `HOTSPRING_GPU_PRIMARY` / `HOTSPRING_GPU_SECONDARY` to select two GPUs.
//! Spawns a thread per GPU, runs GPU HMC thermalization + plaquette measurement
//! on each, then compares results for cross-vendor parity.
//!
//! # Usage
//!
//! ```bash
//! # Auto-discover (picks two highest-memory f64-capable GPUs)
//! cargo run --release --bin validate_dual_gpu_qcd
//!
//! # Explicit selection (NVIDIA RTX 3090 + AMD RX 6950 XT)
//! HOTSPRING_GPU_PRIMARY=3090 HOTSPRING_GPU_SECONDARY=6950 \
//!     cargo run --release --bin validate_dual_gpu_qcd
//! ```

use hotspring_barracuda::gpu::{GpuF64, discover_primary_and_secondary_adapters};
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

struct GpuResult {
    adapter_name: String,
    plaquettes: Vec<f64>,
    wall_secs: f64,
}

fn run_on_gpu(name_hint: &str, dims: [usize; 4], beta: f64, seed: u64) -> GpuResult {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt
        .block_on(GpuF64::from_adapter_name(name_hint))
        .unwrap_or_else(|e| panic!("Failed to open GPU '{name_hint}': {e}"));

    let adapter_name = gpu.adapter_name.clone();
    println!("  [{adapter_name}] GPU opened");

    let pipelines = GpuHmcStreamingPipelines::new(&gpu);
    println!("  [{adapter_name}] HMC pipelines compiled");

    let lat = Lattice::hot_start(dims, beta, seed);
    let n_md = 10;
    let dt = 0.1;

    let hmc_state = GpuHmcState::from_lattice(&gpu, &lat, beta);

    let start = Instant::now();

    let n_therm = 30;
    let n_meas = 10;
    let mut rng_seed = seed;

    for i in 0..n_therm {
        let result = gpu_hmc_trajectory_streaming(
            &gpu,
            &pipelines,
            &hmc_state,
            n_md,
            dt,
            i as u32,
            &mut rng_seed,
        );
        if (i + 1) % 10 == 0 {
            println!(
                "  [{adapter_name}] therm {}/{n_therm}: P={:.6}",
                i + 1,
                result.plaquette
            );
        }
    }

    let mut plaquettes = Vec::with_capacity(n_meas);
    for j in 0..n_meas {
        let result = gpu_hmc_trajectory_streaming(
            &gpu,
            &pipelines,
            &hmc_state,
            n_md,
            dt,
            (n_therm + j) as u32,
            &mut rng_seed,
        );
        plaquettes.push(result.plaquette);
    }

    let wall_secs = start.elapsed().as_secs_f64();
    println!(
        "  [{adapter_name}] Done: {n_meas} measurements in {wall_secs:.1}s, mean P={:.6}",
        plaquettes.iter().sum::<f64>() / plaquettes.len() as f64
    );

    GpuResult {
        adapter_name,
        plaquettes,
        wall_secs,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Dual-GPU Parallel QCD Validation                          ║");
    println!("║  Same physics on two GPUs — cross-vendor parity check      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("dual_gpu_qcd");

    let (primary_opt, secondary_opt) = discover_primary_and_secondary_adapters();
    let Some(primary) = primary_opt else {
        println!("  No primary GPU found (need SHADER_F64)");
        harness.check_bool("Primary GPU available", false);
        harness.finish();
    };
    let Some(secondary) = secondary_opt else {
        println!("  No secondary GPU found (need 2 GPUs with SHADER_F64)");
        println!("  Set HOTSPRING_GPU_PRIMARY and HOTSPRING_GPU_SECONDARY");
        harness.check_bool("Secondary GPU available", false);
        harness.finish();
    };

    println!("  Primary:   {primary}");
    println!("  Secondary: {secondary}");
    println!();

    harness.check_bool("Two GPUs discovered", true);

    let dims = [4, 4, 4, 4];
    let beta = 6.0;
    let seed = 42u64;

    println!("  Config: {}⁴, β={beta}, seed={seed}", dims[0]);
    println!("  Launching parallel GPU threads...");
    println!();

    let p_name = primary;
    let s_name = secondary;

    let handle_primary = std::thread::spawn(move || run_on_gpu(&p_name, dims, beta, seed));
    let handle_secondary = std::thread::spawn(move || run_on_gpu(&s_name, dims, beta, seed));

    let r1 = handle_primary.join().expect("primary GPU thread panicked");
    let r2 = handle_secondary
        .join()
        .expect("secondary GPU thread panicked");

    println!();
    println!("═══ Results ═══");
    println!();
    println!(
        "  {} : mean P = {:.6}, {:.1}s",
        r1.adapter_name,
        r1.plaquettes.iter().sum::<f64>() / r1.plaquettes.len() as f64,
        r1.wall_secs
    );
    println!(
        "  {} : mean P = {:.6}, {:.1}s",
        r2.adapter_name,
        r2.plaquettes.iter().sum::<f64>() / r2.plaquettes.len() as f64,
        r2.wall_secs
    );

    let mean1 = r1.plaquettes.iter().sum::<f64>() / r1.plaquettes.len() as f64;
    let mean2 = r2.plaquettes.iter().sum::<f64>() / r2.plaquettes.len() as f64;
    let diff = (mean1 - mean2).abs();

    println!();
    println!("  |ΔP| = {diff:.2e}");

    // With same seed and same beta, both GPUs should thermalize to similar
    // plaquette values. The exact trajectories may differ due to floating-point
    // non-associativity across different hardware, but the ensemble average
    // should agree within statistical noise (~0.01 for 10 measurements at 4^4).
    let stat_tol = 0.02;
    harness.check_bool(
        &format!(
            "Cross-GPU plaquette agreement |ΔP| < {stat_tol} ({} vs {})",
            r1.adapter_name, r2.adapter_name
        ),
        diff < stat_tol,
    );

    // Both must be in physically reasonable range for β=6.0 quenched SU(3)
    let physical_range = 0.50..0.65;
    harness.check_bool(
        &format!("{} plaquette in physical range", r1.adapter_name),
        physical_range.contains(&mean1),
    );
    harness.check_bool(
        &format!("{} plaquette in physical range", r2.adapter_name),
        physical_range.contains(&mean2),
    );

    // Per-measurement comparison
    for (i, (p1, p2)) in r1.plaquettes.iter().zip(r2.plaquettes.iter()).enumerate() {
        let d = (p1 - p2).abs();
        println!(
            "  meas {}: {} P={p1:.6}, {} P={p2:.6}, |Δ|={d:.2e}",
            i, r1.adapter_name, r2.adapter_name
        );
    }

    println!();
    harness.finish();
}
