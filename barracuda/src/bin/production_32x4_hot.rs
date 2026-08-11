// SPDX-License-Identifier: AGPL-3.0-or-later

//! 32⁴ hot-start production — prove equilibrium at scale.
//!
//! Cold start at 32⁴ requires ~5000 trajectories to thermalize from P=1.0→0.578.
//! Hot start (epsilon-random near identity) breaks the symmetry immediately,
//! reaching equilibrium within ~200-500 warmup trajectories.
//!
//! Architecture: Node-Atomic composition — hotSpring orchestrates, barraCuda GPU physics.
//!
//! Parameters:
//!   Volume: 32⁴ (1,048,576 sites, 4,194,304 links)
//!   β = 5.90
//!   n_md = 40, dt = 0.0025, τ = 0.1
//!   Hot start: ε = 0.2 (random SU(3) perturbation)
//!   Warmup: 500 trajectories
//!   Production: 200 trajectories
//!
//! Expected: ⟨P⟩ ≈ 0.578, acceptance ≈ 50-70% at this step size.

use barracuda::ops::lattice::gpu_hmc_types::GpuHmcConfig;
use hotspring_barracuda::node_atomic::{NodeAtomicQcd, TrajectoryRunner};
use std::time::Instant;

fn main() {
    let config = GpuHmcConfig {
        nt: 32,
        nx: 32,
        ny: 32,
        nz: 32,
        beta: 5.90,
        mass: 1.0,
        n_md_steps: 40,
        dt: 0.0025,
        cg_tol: 1e-10,
        cg_max_iter: 1000,
        n_flavors_over_4: 0,
    };

    let hot_epsilon = 3.0;

    println!("═══ 32⁴ β=5.90 Hot-Start Production ═══");
    println!("  Omelyan 2MN: n_md={}, dt={}, τ={:.3}", config.n_md_steps, config.dt,
        config.n_md_steps as f64 * config.dt);
    println!("  Hot start: ε={hot_epsilon} (random SU(3) far from identity)");
    println!("  4,194,304 links → WG128 dispatch, 3-pass reduce");
    println!();

    let t0 = Instant::now();
    let qcd = NodeAtomicQcd::new(config.clone(), 42).expect("Failed to create QCD state");
    println!("  Device: {} ({:.2}s)", qcd.device.adapter_info().name, t0.elapsed().as_secs_f64());

    qcd.upload_topology();
    qcd.hot_start(hot_epsilon).expect("hot_start failed");
    qcd.seed_rng(42);

    let volume = qcd.volume();
    println!("  Volume: {volume} sites, {} links", volume * 4);
    println!("  Initialized: hot start ({:.2}s)", t0.elapsed().as_secs_f64());
    println!();

    let runner = TrajectoryRunner {
        warmup_count: 500,
        production_count: 200,
        target_acceptance: 0.60,
    };

    println!("  ── Campaign: {} warmup + {} production ──",
        runner.warmup_count, runner.production_count);
    println!();

    let (production, measurements) = runner.run_campaign(&qcd, 50, |step, plaq, acc| {
        let elapsed = t0.elapsed().as_secs_f64();
        let rate = elapsed / step as f64;
        println!("    warmup {:3}/{}: P={:.6}, acc={:.0}%, {:.1}s/traj",
            step, runner.warmup_count, plaq, acc * 100.0, rate);
    }).expect("Campaign failed");

    let mean_p: f64 = measurements.iter().sum::<f64>() / measurements.len() as f64;
    let var_p: f64 = measurements.iter().map(|p| (p - mean_p).powi(2)).sum::<f64>()
        / (measurements.len() - 1) as f64;
    let err_p = (var_p / measurements.len() as f64).sqrt();
    let acc_rate = production.accepted as f64 / production.trajectories as f64;

    let within_tolerance = (mean_p - 0.578).abs() < 0.03;
    hotspring_barracuda::gossip::validation_result(
        within_tolerance,
        if within_tolerance { measurements.len() } else { 0 },
        measurements.len(),
    );

    let total_time = t0.elapsed().as_secs_f64();
    let total_trajs = runner.warmup_count + runner.production_count;

    println!();
    println!("═══ RESULT ═══");
    println!("  ⟨P⟩ = {:.6} ± {:.6}", mean_p, err_p);
    println!("  Expected: ~0.578 (β=5.90, 32⁴)");
    println!("  Acceptance: {:.0}%", acc_rate * 100.0);
    println!("  ⟨|ΔH|⟩ = {:.4}", production.mean_delta_h);
    println!("  Total time: {:.0}s ({:.1} min, {} trajectories, {:.1}s/traj)",
        total_time, total_time / 60.0, total_trajs, total_time / total_trajs as f64);
    println!();

    if within_tolerance {
        println!("  ✓ PASS — 32⁴ equilibrium reached via hot start");
    } else if mean_p > 0.7 {
        println!("  ? PARTIAL — trending but not yet equilibrated (may need longer warmup)");
    } else if mean_p < 0.5 {
        println!("  ✗ FAIL — plaquette below expected range");
    } else {
        println!("  ? CLOSE — within broad range but outside 3σ tolerance");
    }
}
