// SPDX-License-Identifier: AGPL-3.0-or-later

//! 16⁴ Node-Atomic validation via barraCuda composition.
//!
//! Demonstrates the complete Node-Atomic architecture:
//!   hotSpring orchestration → barraCuda GPU HMC → scalar readback
//!
//! Pure gauge SU(3) at β=5.90. Expected equilibrium: ⟨P⟩ ≈ 0.578.
//! Uses TrajectoryRunner for campaign orchestration.

use barracuda::ops::lattice::gpu_hmc_types::GpuHmcConfig;
use hotspring_barracuda::node_atomic::{NodeAtomicQcd, TrajectoryRunner};

fn main() {
    let config = GpuHmcConfig {
        nt: 16,
        nx: 16,
        ny: 16,
        nz: 16,
        beta: 5.90,
        mass: 1.0,
        n_md_steps: 20,
        dt: 0.005,
        cg_tol: 1e-10,
        cg_max_iter: 1000,
        n_flavors_over_4: 0,
    };

    println!("═══ 16⁴ β=5.90 Node-Atomic Validation ═══");
    println!("  Omelyan 2MN: n_md={}, dt={}, τ={:.3}", config.n_md_steps, config.dt,
        config.n_md_steps as f64 * config.dt);
    println!("  Pure gauge (n_flavors_over_4=0)");
    println!();

    let t_init = std::time::Instant::now();
    let qcd = NodeAtomicQcd::new(config.clone(), 42).expect("Failed to create QCD state");
    println!("  Device: {} ({:.2}s)", qcd.device.adapter_info().name, t_init.elapsed().as_secs_f64());

    qcd.upload_topology();
    qcd.cold_start().expect("cold_start failed");
    qcd.seed_rng(42);
    println!("  Initialized: volume={}, cold start + topology ({:.2}s)",
        qcd.volume(), t_init.elapsed().as_secs_f64());
    println!();

    // Single trajectory diagnostic
    let t_traj = std::time::Instant::now();
    let r1 = qcd.run_trajectory().expect("First trajectory failed");
    let plaq = 1.0 - r1.gauge_action / (6.0 * qcd.volume() as f64 * config.beta);
    println!("  First trajectory: {:.2}s, P={:.8}, ΔH={:+.4}, acc={}",
        t_traj.elapsed().as_secs_f64(), plaq, r1.delta_h, r1.accepted);

    // TrajectoryRunner warmup campaign
    let runner = TrajectoryRunner {
        warmup_count: 100,
        production_count: 50,
        target_acceptance: 0.70,
    };

    println!("\n  ── Warmup ({} trajectories) ──", runner.warmup_count);
    let warmup = runner.run_warmup(&qcd, 20, |step, plaq, acc| {
        println!("    step {:3}: P={:.6}, acc={:.0}%", step, plaq, acc * 100.0);
    }).expect("Warmup failed");

    println!("  Warmup done: P={:.6}, acc={:.0}%, ⟨|ΔH|⟩={:.4}",
        warmup.final_plaquette, warmup.accepted as f64 / warmup.trajectories as f64 * 100.0,
        warmup.mean_delta_h);

    // Production measurements
    println!("\n  ── Production ({} trajectories) ──", runner.production_count);
    let mut measurements = Vec::with_capacity(runner.production_count);
    let production = runner.run_production(&qcd, &mut measurements)
        .expect("Production failed");

    let mean_p: f64 = measurements.iter().sum::<f64>() / measurements.len() as f64;
    let var_p: f64 = measurements.iter().map(|p| (p - mean_p).powi(2)).sum::<f64>()
        / (measurements.len() - 1) as f64;
    let err_p = (var_p / measurements.len() as f64).sqrt();

    println!("\n═══ RESULT ═══");
    println!("  ⟨P⟩ = {:.6} ± {:.6}", mean_p, err_p);
    println!("  Expected: ~0.578 (β=5.90, 16⁴)");
    println!("  Acceptance: {:.0}%", production.accepted as f64 / production.trajectories as f64 * 100.0);
    println!("  ⟨|ΔH|⟩ = {:.4}", production.mean_delta_h);
    println!("  Total time: {:.1}s ({} trajectories, {:.2}s/traj)",
        t_init.elapsed().as_secs_f64(),
        1 + runner.warmup_count + runner.production_count,
        t_init.elapsed().as_secs_f64() / (1 + runner.warmup_count + runner.production_count) as f64);
}
