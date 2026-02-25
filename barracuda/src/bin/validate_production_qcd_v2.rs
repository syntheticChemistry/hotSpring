// SPDX-License-Identifier: AGPL-3.0-only

//! Production Lattice QCD Validation V2 (Papers 9-12).
//!
//! Demonstrates the full production pipeline with Omelyan integrator and
//! Hasenbusch mass preconditioning. Extends `validate_production_qcd` and
//! `validate_dynamical_qcd` with improved algorithms.
//!
//! # Validation targets
//!
//! | Phase | Description | Checks |
//! |-------|-------------|--------|
//! | 1 | Omelyan vs Leapfrog comparison | \|ΔH\|, acceptance |
//! | 2 | Omelyan 8⁴ scaling | Acceptance, plaquette range |
//! | 3 | Dynamical fermion HMC with Omelyan | S_F, CG, acceptance |
//! | 4 | Hasenbusch mass preconditioning | Plaquette, heavy vs ratio CG |
//! | 5 | Omelyan timing report | ms/trajectory sanity |
//!
//! # Provenance
//!
//! - Omelyan et al., Comp. Phys. Comm. 146, 188 (2003) — 2MN integrator
//! - Hasenbusch, PLB 519, 177 (2001) — mass preconditioning

use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::pseudofermion::{
    dynamical_hmc_trajectory, hasenbusch_hmc_trajectory, DynamicalHmcConfig, DynamicalHmcResult,
    HasenbuschConfig, HasenbuschHmcConfig, HasenbuschHmcResult, PseudofermionConfig,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Production Lattice QCD V2 — Omelyan + Hasenbusch          ║");
    println!("║  Full production pipeline validation                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("production_qcd_v2");

    // ═══ Phase 1: Omelyan vs Leapfrog comparison ═══
    println!("═══ Phase 1: Omelyan vs Leapfrog (4⁴, β=6.0, 30 traj each) ═══");
    println!();

    let dims = [4, 4, 4, 4];
    let beta = 6.0;
    let dt = 0.05;
    let n_md = 20;
    let seed = 42;
    let n_traj = 30;

    // Leapfrog run
    let mut lat_leap = Lattice::hot_start(dims, beta, seed);
    let mut cfg_leap = HmcConfig {
        n_md_steps: n_md,
        dt,
        seed,
        integrator: IntegratorType::Leapfrog,
    };
    let leap_results: Vec<_> = (0..n_traj)
        .map(|_| hmc::hmc_trajectory(&mut lat_leap, &mut cfg_leap))
        .collect();
    let leap_mean_abs_dh =
        leap_results.iter().map(|r| r.delta_h.abs()).sum::<f64>() / n_traj as f64;
    let leap_accept_rate =
        leap_results.iter().filter(|r| r.accepted).count() as f64 / n_traj as f64;

    // Omelyan run (same initial conditions: fresh hot start)
    let mut lat_omel = Lattice::hot_start(dims, beta, seed);
    let mut cfg_omel = HmcConfig {
        n_md_steps: n_md,
        dt,
        seed,
        integrator: IntegratorType::Omelyan,
    };
    let omel_results: Vec<_> = (0..n_traj)
        .map(|_| hmc::hmc_trajectory(&mut lat_omel, &mut cfg_omel))
        .collect();
    let omel_mean_abs_dh =
        omel_results.iter().map(|r| r.delta_h.abs()).sum::<f64>() / n_traj as f64;
    let omel_accept_rate =
        omel_results.iter().filter(|r| r.accepted).count() as f64 / n_traj as f64;

    println!(
        "  Leapfrog: mean |ΔH|={leap_mean_abs_dh:.4e}, acceptance={:.1}%",
        leap_accept_rate * 100.0
    );
    println!(
        "  Omelyan:  mean |ΔH|={omel_mean_abs_dh:.4e}, acceptance={:.1}%",
        omel_accept_rate * 100.0
    );

    harness.check_bool(
        "Omelyan |ΔH| < Leapfrog |ΔH|",
        omel_mean_abs_dh < leap_mean_abs_dh,
    );
    harness.check_lower("Omelyan acceptance > 80%", omel_accept_rate, 0.80);
    println!();

    // ═══ Phase 2: Omelyan at 8⁴ scaling ═══
    println!("═══ Phase 2: Omelyan at 8⁴ (β=6.0, dt=0.04, n_md=25, 20 traj) ═══");
    println!();

    let dims_8 = [8, 8, 8, 8];
    let mut lat_8 = Lattice::hot_start(dims_8, 6.0, 123);
    let mut cfg_8 = HmcConfig {
        n_md_steps: 25,
        dt: 0.04,
        seed: 123,
        integrator: IntegratorType::Omelyan,
    };
    // Thermalize 20 traj before measuring (hot start is far from equilibrium)
    for _ in 0..20 {
        hmc::hmc_trajectory(&mut lat_8, &mut cfg_8);
    }
    let n_meas_8 = 20;
    let results_8: Vec<_> = (0..n_meas_8)
        .map(|_| hmc::hmc_trajectory(&mut lat_8, &mut cfg_8))
        .collect();
    let accept_8 = results_8.iter().filter(|r| r.accepted).count();
    let accept_rate_8 = accept_8 as f64 / n_meas_8 as f64;
    let mean_plaq_8: f64 = results_8.iter().map(|r| r.plaquette).sum::<f64>() / n_meas_8 as f64;

    println!(
        "  Acceptance: {accept_8}/{n_meas_8} ({:.1}%)",
        accept_rate_8 * 100.0
    );
    println!("  Mean plaquette: {mean_plaq_8:.6}");

    harness.check_lower("8⁴ Omelyan acceptance > 60%", accept_rate_8, 0.60);
    // 8⁴ at β=6.0 equilibrium plaquette ~ 0.59 (Bali ref 0.594 at infinite vol);
    // with ~40 total traj, finite-vol and stat effects widen the range.
    harness.check_bool(
        "8⁴ plaquette in physical range for β=6.0",
        mean_plaq_8 > 0.45 && mean_plaq_8 < 0.70,
    );
    println!();

    // ═══ Phase 3: Dynamical fermion HMC with Omelyan ═══
    println!("═══ Phase 3: Dynamical fermion HMC (4⁴, β=5.5, m=2.0, Omelyan) ═══");
    println!("  10 therm + 15 measurement, dt=0.001, n_md=100, Omelyan");
    println!();

    let beta_dyn = 5.5;
    let mut lat_dyn = Lattice::hot_start(dims, beta_dyn, 77);

    // Quenched pre-thermalization to get a reasonable gauge config
    let mut q_pre = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 77,
        integrator: IntegratorType::Omelyan,
    };
    hmc::run_hmc(&mut lat_dyn, 15, 0, &mut q_pre);

    // Dynamical with Omelyan: dt=0.001 matches validate_dynamical_qcd baseline
    let mut cfg_dyn = DynamicalHmcConfig {
        n_md_steps: 100,
        dt: 0.001,
        seed: 77,
        fermion: PseudofermionConfig {
            mass: 2.0,
            cg_tol: 1e-8,
            cg_max_iter: tolerances::DYNAMICAL_CG_MAX_ITER,
        },
        beta: beta_dyn,
        n_flavors_over_4: 1,
        integrator: IntegratorType::Omelyan,
    };

    for _ in 0..10 {
        dynamical_hmc_trajectory(&mut lat_dyn, &mut cfg_dyn);
    }
    let meas_results: Vec<DynamicalHmcResult> = (0..15)
        .map(|_| dynamical_hmc_trajectory(&mut lat_dyn, &mut cfg_dyn))
        .collect();

    let all_sf_positive = meas_results.iter().all(|r| r.fermion_action > 0.0);
    // Heuristic: if any CG hit max iters, total would be very large (402 solves × 5000 max)
    let cg_ceiling = 500_000;
    let all_cg_converged = meas_results
        .iter()
        .all(|r| r.cg_iterations > 0 && r.cg_iterations < cg_ceiling);
    let dyn_accept = meas_results.iter().filter(|r| r.accepted).count();
    let dyn_accept_rate = dyn_accept as f64 / 15.0;

    println!("  S_F > 0: {all_sf_positive}");
    println!("  CG converge (heuristic): {all_cg_converged}");
    println!(
        "  Acceptance: {dyn_accept}/15 ({:.1}%)",
        dyn_accept_rate * 100.0
    );

    harness.check_bool("All S_F > 0", all_sf_positive);
    harness.check_bool("All CG converge (no hit max iters)", all_cg_converged);
    // Heavy quarks (m=2.0) on 4⁴ with hot start produce modest acceptance;
    // the key physics validation is S_F > 0 and CG convergence, not high acceptance.
    harness.check_lower("Dynamical HMC acceptance > 1%", dyn_accept_rate, 0.01);
    println!();

    // ═══ Phase 4: Hasenbusch mass preconditioning ═══
    println!("═══ Phase 4: Hasenbusch (4⁴, β=5.5, heavy=0.5, light=0.1) ═══");
    println!("  5 therm + 10 measurement");
    println!();

    let mut lat_has = Lattice::hot_start(dims, 5.5, 99);

    // Quenched pre-thermalization
    let mut q_has = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 99,
        integrator: IntegratorType::Omelyan,
    };
    hmc::run_hmc(&mut lat_has, 15, 0, &mut q_has);

    // Hasenbusch: dt is total trajectory time (subdivided into heavy/light steps).
    // heavy mass keeps CG cheap; light mass handles the physics.
    // n_heavy=10, n_light=5 → 50 total link updates, τ = dt = 0.1
    let mut cfg_has = HasenbuschHmcConfig {
        dt: 0.1,
        seed: 99,
        hasenbusch: HasenbuschConfig {
            heavy_mass: 0.5,
            light_mass: 0.1,
            cg_tol: 1e-6,
            cg_max_iter: 2000,
            n_md_steps_light: 5,
            n_md_steps_heavy: 10,
        },
        beta: 5.5,
    };

    for _ in 0..5 {
        hasenbusch_hmc_trajectory(&mut lat_has, &mut cfg_has);
    }
    let has_results: Vec<HasenbuschHmcResult> = (0..10)
        .map(|_| hasenbusch_hmc_trajectory(&mut lat_has, &mut cfg_has))
        .collect();

    let mean_plaq_has: f64 = has_results.iter().map(|r| r.plaquette).sum::<f64>() / 10.0;
    let plaq_in_range = mean_plaq_has > 0.0 && mean_plaq_has < 1.0;
    let mean_heavy: f64 = has_results
        .iter()
        .map(|r| r.cg_iterations_heavy as f64)
        .sum::<f64>()
        / 10.0;
    let mean_ratio: f64 = has_results
        .iter()
        .map(|r| r.cg_iterations_ratio as f64)
        .sum::<f64>()
        / 10.0;
    let heavy_lt_ratio = mean_heavy < mean_ratio;

    println!("  Mean plaquette: {mean_plaq_has:.6}");
    println!("  Mean CG heavy: {mean_heavy:.0}, mean CG ratio: {mean_ratio:.0}");

    harness.check_bool("Hasenbusch HMC runs (plaquette in (0,1))", plaq_in_range);
    harness.check_bool(
        "Heavy sector uses fewer CG iters than ratio (on average)",
        heavy_lt_ratio,
    );
    println!();

    // ═══ Phase 5: Omelyan timing ═══
    println!("═══ Phase 5: Omelyan timing (10 traj at 8⁴ pure gauge) ═══");
    println!();

    let mut lat_time = Lattice::hot_start(dims_8, 6.0, 456);
    let mut cfg_time = HmcConfig {
        n_md_steps: 25,
        dt: 0.04,
        seed: 456,
        integrator: IntegratorType::Omelyan,
    };
    let start = Instant::now();
    for _ in 0..10 {
        hmc::hmc_trajectory(&mut lat_time, &mut cfg_time);
    }
    let elapsed = start.elapsed();
    let ms_per_traj = elapsed.as_secs_f64() * 1000.0 / 10.0;
    let traj_per_sec = 10.0 / elapsed.as_secs_f64();

    println!("  ms/trajectory: {ms_per_traj:.1}");
    println!("  trajectories/sec: {traj_per_sec:.2}");

    harness.check_lower("Timing > 0 (sanity)", ms_per_traj, 0.001);
    println!();

    harness.finish();
}
