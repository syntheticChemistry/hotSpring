// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/GPU Parity Validation
//!
//! Proves that barracuda enables identical physics on pure CPU and pure GPU.
//! Same algorithm (Velocity-Verlet + Yukawa + Berendsen), same initial
//! conditions (FCC lattice, seed 42), different hardware → same science.
//!
//! **What this proves**: the math is determined by the algorithm, not the
//! chip specialization. CPU f64 scalar and GPU f64 WGSL produce the same
//! energy conservation, temperature equilibration, VACF, and D*.
//!
//! The only expected differences are FP ordering (GPU uses `j != i` full-loop
//! vs CPU `j > i` Newton-3rd), which causes ~1e-10 relative force differences
//! that grow slowly over many steps but do NOT affect statistical observables.
//!
//! Exit code 0 = parity confirmed, exit code 1 = divergence detected.

use hotspring_barracuda::md::config;
use hotspring_barracuda::md::cpu_reference;
use hotspring_barracuda::md::observables::{compute_vacf, validate_energy};
use hotspring_barracuda::md::simulation;
use hotspring_barracuda::md::transport::d_star_daligault;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CPU / GPU Parity Validation                               ║");
    println!("║  Same algorithm, same initial conditions, different chip    ║");
    println!("║  Proving: hardware-agnostic physics via barracuda           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("cpu_gpu_parity");

    // ── System parameters ──
    // Use a manageable system: N=108 (3³ FCC), enough steps for VACF
    let kappa = 2.0;
    let gamma = 50.0;
    let n_particles = 108;

    let mut cfg = config::MdConfig {
        label: format!("parity_k{kappa}_G{gamma}"),
        n_particles,
        kappa,
        gamma,
        dt: 0.01,
        rc: 6.0,
        equil_steps: 2000,
        prod_steps: 5000,
        dump_step: 10,
        berendsen_tau: 2.0,
        rdf_bins: 200,
        vel_snapshot_interval: 1,
    };

    let temperature = cfg.temperature();
    let box_side = cfg.box_side();
    let d_star_analytical = d_star_daligault(gamma, kappa);

    println!("  System: N={n_particles}, κ={kappa}, Γ={gamma}, T*={temperature:.6}");
    println!("  Box: L={box_side:.4} a_ws, rc={}", cfg.rc);
    println!(
        "  Steps: {equil}+{prod} (equil+prod)",
        equil = cfg.equil_steps,
        prod = cfg.prod_steps
    );
    println!("  D*(Daligault) = {d_star_analytical:.4e}");
    println!();

    // ══════════════════════════════════════════════════════════════
    //  Phase 1: CPU MD
    // ══════════════════════════════════════════════════════════════
    println!("═══ Phase 1: Pure CPU MD (Rust f64 scalar) ═══════════════════");
    let cpu_sim = cpu_reference::run_simulation_cpu(&cfg);

    let cpu_energy_val = validate_energy(&cpu_sim.energy_history, &cfg);
    println!(
        "  Energy drift: {:.4}% (threshold: {}%)",
        cpu_energy_val.drift_pct,
        tolerances::ENERGY_DRIFT_PCT
    );
    harness.check_upper(
        "CPU energy drift %",
        cpu_energy_val.drift_pct,
        tolerances::ENERGY_DRIFT_PCT,
    );

    let cpu_d_star = if cpu_sim.velocity_snapshots.len() >= 10 {
        let dt_snap = cfg.dt * cfg.dump_step as f64 * cfg.vel_snapshot_interval as f64;
        let max_lag = (cpu_sim.velocity_snapshots.len() / 2).max(10);
        let vacf = compute_vacf(&cpu_sim.velocity_snapshots, n_particles, dt_snap, max_lag);
        println!("  D*(CPU) = {:.4e}", vacf.diffusion_coeff);
        Some(vacf.diffusion_coeff)
    } else {
        println!(
            "  WARNING: Not enough velocity snapshots for VACF ({})",
            cpu_sim.velocity_snapshots.len()
        );
        None
    };

    let cpu_final_t = cpu_sim.energy_history.last().map_or(0.0, |e| e.temperature);
    println!("  Final T* = {cpu_final_t:.6} (target {temperature:.6})");
    println!(
        "  Wall time: {:.2}s ({:.0} steps/s)",
        cpu_sim.wall_time_s, cpu_sim.steps_per_sec
    );
    println!();

    // ══════════════════════════════════════════════════════════════
    //  Phase 2: GPU MD
    // ══════════════════════════════════════════════════════════════
    println!("═══ Phase 2: Pure GPU MD (WGSL f64 shader) ═══════════════════");
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    // Ensure identical config for GPU run
    cfg.label = format!("parity_gpu_k{kappa}_G{gamma}");
    let gpu_sim = match rt.block_on(simulation::run_simulation(&cfg)) {
        Ok(sim) => sim,
        Err(e) => {
            println!("  GPU simulation failed: {e}");
            println!("  (GPU may not be available — CPU path still validated)");
            harness.check_bool("GPU available", false);
            harness.finish();
        }
    };

    let gpu_energy_val = validate_energy(&gpu_sim.energy_history, &cfg);
    println!(
        "  Energy drift: {:.4}% (threshold: {}%)",
        gpu_energy_val.drift_pct,
        tolerances::ENERGY_DRIFT_PCT
    );
    harness.check_upper(
        "GPU energy drift %",
        gpu_energy_val.drift_pct,
        tolerances::ENERGY_DRIFT_PCT,
    );

    let gpu_d_star = if gpu_sim.velocity_snapshots.len() >= 10 {
        let dt_snap = cfg.dt * cfg.dump_step as f64 * cfg.vel_snapshot_interval as f64;
        let max_lag = (gpu_sim.velocity_snapshots.len() / 2).max(10);
        let vacf = compute_vacf(&gpu_sim.velocity_snapshots, n_particles, dt_snap, max_lag);
        println!("  D*(GPU) = {:.4e}", vacf.diffusion_coeff);
        Some(vacf.diffusion_coeff)
    } else {
        println!(
            "  WARNING: Not enough velocity snapshots for VACF ({})",
            gpu_sim.velocity_snapshots.len()
        );
        None
    };

    let gpu_final_t = gpu_sim.energy_history.last().map_or(0.0, |e| e.temperature);
    println!("  Final T* = {gpu_final_t:.6} (target {temperature:.6})");
    println!(
        "  Wall time: {:.2}s ({:.0} steps/s)",
        gpu_sim.wall_time_s, gpu_sim.steps_per_sec
    );
    let speedup = cpu_sim.wall_time_s / gpu_sim.wall_time_s;
    println!("  GPU speedup: {speedup:.1}×");
    println!();

    // ══════════════════════════════════════════════════════════════
    //  Phase 3: Parity Comparison
    // ══════════════════════════════════════════════════════════════
    println!("═══ Phase 3: CPU vs GPU Parity ════════════════════════════════");

    // 3a. Energy conservation: both should conserve energy similarly
    let drift_diff = (cpu_energy_val.drift_pct - gpu_energy_val.drift_pct).abs();
    println!(
        "  Energy drift difference: {drift_diff:.4}% (CPU={:.4}%, GPU={:.4}%)",
        cpu_energy_val.drift_pct, gpu_energy_val.drift_pct
    );
    harness.check_bool(
        "Both paths conserve energy",
        cpu_energy_val.passed && gpu_energy_val.passed,
    );

    // 3b. Temperature: both should equilibrate to similar final T*
    let t_rel_diff = if temperature > 1e-30 {
        ((cpu_final_t - gpu_final_t) / temperature).abs()
    } else {
        0.0
    };
    println!(
        "  Temperature difference: {t_rel_diff:.2e} (CPU={cpu_final_t:.6}, GPU={gpu_final_t:.6})"
    );
    harness.check_upper("T* CPU vs GPU relative diff", t_rel_diff, 0.25);

    // 3c. D* comparison: CPU, GPU, and analytical should agree
    if let (Some(d_cpu), Some(d_gpu)) = (cpu_d_star, gpu_d_star) {
        let cpu_vs_fit = if d_star_analytical.abs() > f64::EPSILON {
            ((d_cpu - d_star_analytical) / d_star_analytical).abs()
        } else {
            0.0
        };
        let gpu_vs_fit = if d_star_analytical.abs() > f64::EPSILON {
            ((d_gpu - d_star_analytical) / d_star_analytical).abs()
        } else {
            0.0
        };
        let cpu_vs_gpu = if d_cpu.abs() > f64::EPSILON {
            ((d_cpu - d_gpu) / d_cpu).abs()
        } else {
            0.0
        };

        println!();
        println!("  ┌─────────────────────────────────────────────────────┐");
        println!("  │  Transport Coefficient D* Comparison                │");
        println!("  ├─────────────┬────────────┬──────────────────────────┤");
        println!("  │ Source      │ D*         │ vs Daligault fit         │");
        println!("  ├─────────────┼────────────┼──────────────────────────┤");
        println!(
            "  │ CPU (Rust)  │ {d_cpu:10.4e} │ {:.1}%{:>21} │",
            cpu_vs_fit * 100.0,
            ""
        );
        println!(
            "  │ GPU (WGSL)  │ {d_gpu:10.4e} │ {:.1}%{:>21} │",
            gpu_vs_fit * 100.0,
            ""
        );
        println!(
            "  │ Analytical  │ {:10.4e} │ (reference){:>14} │",
            d_star_analytical, ""
        );
        println!("  ├─────────────┼────────────┼──────────────────────────┤");
        println!(
            "  │ CPU vs GPU  │            │ {:.1}% relative diff{:>6} │",
            cpu_vs_gpu * 100.0,
            ""
        );
        println!("  └─────────────┴────────────┴──────────────────────────┘");

        // D* vs Daligault: INFORMATIONAL — not a parity check.
        // Absolute D* accuracy requires larger systems (N ≥ 2000) and longer
        // runs (≥ 20k steps). At N=108 / 5k steps the VACF is dominated by
        // ballistic motion, not long-time diffusion. Both CPU and GPU show
        // the SAME inaccuracy, proving the chip doesn't affect the physics.
        println!();
        if cpu_vs_fit > tolerances::TRANSPORT_D_STAR_VS_FIT {
            println!(
                "  NOTE: D* vs Daligault {:.0}% — expected at N={n_particles}",
                cpu_vs_fit * 100.0
            );
            println!("        Accurate D* requires N≥2000, prod≥20k steps.");
            println!("        This is a system-size limitation, not a parity failure.");
        }

        // CORE PARITY CHECK: CPU vs GPU D* must agree.
        // Both run the same algorithm from the same initial conditions.
        // Trajectories diverge due to FP ordering differences (GPU uses
        // j!=i full-loop vs CPU j>i Newton-3rd), but the VACF average
        // should be statistically equivalent.
        harness.check_upper(
            "D* CPU vs GPU relative (parity)",
            cpu_vs_gpu,
            0.15, // 15% — generous for N=108 statistical noise
        );
    } else {
        println!("  D* comparison skipped — insufficient velocity snapshots");
    }

    // 3d. Energy history shape comparison
    let n_common = cpu_sim
        .energy_history
        .len()
        .min(gpu_sim.energy_history.len());
    if n_common >= 2 {
        let cpu_mean_e: f64 = cpu_sim.energy_history.iter().map(|e| e.total).sum::<f64>()
            / cpu_sim.energy_history.len() as f64;
        let gpu_mean_e: f64 = gpu_sim.energy_history.iter().map(|e| e.total).sum::<f64>()
            / gpu_sim.energy_history.len() as f64;
        let mean_e_diff = if cpu_mean_e.abs() > f64::EPSILON {
            ((cpu_mean_e - gpu_mean_e) / cpu_mean_e).abs()
        } else {
            0.0
        };
        println!();
        println!(
            "  Mean total energy: CPU={cpu_mean_e:.4}, GPU={gpu_mean_e:.4}, diff={:.2e}",
            mean_e_diff
        );
        harness.check_upper("Mean energy CPU vs GPU", mean_e_diff, 0.05);
    }

    println!();
    println!("═══ Verdict ═════════════════════════════════════════════════");
    println!("  Same algorithm + same initial conditions + different hardware");
    println!("  → Same physics within documented f64 tolerances");
    println!("  The math is determined by the algorithm, not the chip.");
    println!();
    harness.finish();
}
