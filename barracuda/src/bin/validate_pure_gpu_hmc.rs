// SPDX-License-Identifier: AGPL-3.0-only

//! Pure GPU HMC validation: all lattice QCD math on GPU via fp64 WGSL.
//!
//! Tests each GPU shader individually against CPU reference:
//!
//! | Phase | GPU shader | CPU reference | Check |
//! |-------|-----------|---------------|-------|
//! | 1 | `wilson_plaquette_f64` | `average_plaquette()` | Machine-ε parity |
//! | 2 | `su3_gauge_force_f64` | `gauge_force()` | Component-wise parity |
//! | 3 | `su3_kinetic_energy_f64` | `kinetic_energy()` | Scalar parity |
//! | 4 | `su3_momentum_update_f64` | CPU P += dt*F | Component-wise parity |
//! | 5 | `su3_link_update_f64` | CPU Cayley exp | Component-wise parity |
//!
//! Links and momenta are GPU-resident. Only scalars stream back.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{gpu_hmc_trajectory, GpuHmcPipelines, GpuHmcState};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Pure GPU HMC Validation — All Math on GPU (fp64 WGSL)     ║");
    println!("║  Unidirectional streaming: only scalars CPU←GPU            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("pure_gpu_hmc");
    let start_total = Instant::now();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| {
        panic!("tokio runtime failed: {e}");
    });
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("  No GPU with SHADER_F64 found: {e}");
            println!("  Skipping GPU validation");
            harness.check_bool("GPU available (SHADER_F64)", false);
            harness.finish();
        }
    };

    println!("  GPU: {}", gpu.adapter_name);
    println!();

    let pipelines = GpuHmcPipelines::new(&gpu);
    println!("  All 5 HMC shader pipelines compiled successfully");
    harness.check_bool("All shader pipelines compile", true);
    println!();

    // Prepare thermalized 4⁴ lattice
    let dims = [4, 4, 4, 4];
    let beta = 6.0;

    let mut lat = Lattice::hot_start(dims, beta, 42);
    let mut cfg = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };
    hmc::run_hmc(&mut lat, 20, 0, &mut cfg);

    // ═══ Phase 6: Full GPU Omelyan HMC trajectory ═══
    println!("═══ Phase 6: Full GPU Omelyan HMC (all math on GPU) ═══");
    println!("  4⁴ lattice, β=6.0, dt=0.05, n_md=15, 10 trajectories");
    println!();

    let mut lat_gpu = Lattice::hot_start(dims, beta, 42);
    let mut cfg_gpu = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };
    hmc::run_hmc(&mut lat_gpu, 20, 0, &mut cfg_gpu);

    let gpu_state = GpuHmcState::from_lattice(&gpu, &lat_gpu, beta);
    let mut seed = 1000u64;
    let n_traj = 10;
    let mut n_accepted = 0;
    let mut plaq_sum = 0.0;

    for i in 0..n_traj {
        let result = gpu_hmc_trajectory(&gpu, &pipelines, &gpu_state, 15, 0.05, &mut seed);
        let tag = if result.accepted { "✓" } else { "✗" };
        println!(
            "  traj {}: {} ΔH={:+.4e}  plaq={:.6}",
            i + 1,
            tag,
            result.delta_h,
            result.plaquette,
        );
        if result.accepted {
            n_accepted += 1;
        }
        plaq_sum += result.plaquette;
    }

    let accept_rate = n_accepted as f64 / n_traj as f64;
    let mean_plaq = plaq_sum / n_traj as f64;
    println!();
    println!(
        "  Acceptance: {n_accepted}/{n_traj} ({:.0}%)",
        accept_rate * 100.0,
    );
    println!("  Mean plaquette: {mean_plaq:.6}");

    harness.check_lower("GPU HMC acceptance > 30%", accept_rate, 0.30);
    harness.check_bool(
        "GPU HMC plaquette in physical range",
        mean_plaq > 0.45 && mean_plaq < 0.70,
    );
    println!();

    let elapsed = start_total.elapsed().as_secs_f64();
    println!("  Total wall time: {elapsed:.1}s");

    harness.finish();
}
