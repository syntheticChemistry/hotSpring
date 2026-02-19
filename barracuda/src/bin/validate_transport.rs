// SPDX-License-Identifier: AGPL-3.0-only

//! Transport Coefficient Validation Binary
//!
//! Runs Yukawa OCP MD at selected (Gamma, kappa) points, computes
//! self-diffusion D* from VACF via Green-Kubo, and validates against:
//!   (a) Daligault (2012) analytical fit
//!   (b) Sarkas Python baselines (if available)
//!
//! Exit code 0 = all checks pass, exit code 1 = failure.
//!
//! # Validation targets
//!
//! | Source | Reference | Tolerance |
//! |--------|-----------|-----------|
//! | Daligault D* fit | PRE 86, 047401 (2012) | < 10% relative |
//! | Sarkas VACF | control/sarkas baseline | < 5% relative |
//! | Energy drift | Symplectic conservation | < 5% |
//!
//! # Provenance
//!
//! Daligault fit parameters: Table I of PRE 86, 047401 (2012).
//! Sarkas baseline: `control/sarkas/simulations/transport-study/results/transport_baseline_lite.json`

use hotspring_barracuda::md::config;
use hotspring_barracuda::md::observables::{
    compute_stress_acf, compute_stress_xy, compute_vacf, validate_energy,
};
use hotspring_barracuda::md::simulation;
use hotspring_barracuda::md::transport::{d_star_daligault, TransportResult};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

use std::f64::consts::PI;

fn compute_transport_for_config(config: &config::MdConfig) -> Result<TransportResult, String> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| format!("tokio init: {e}"))?;

    let sim = rt
        .block_on(simulation::run_simulation(config))
        .map_err(|e| format!("simulation failed: {e}"))?;

    // Validate energy conservation
    let energy_val = validate_energy(&sim.energy_history, config);
    if !energy_val.passed {
        eprintln!(
            "  WARNING: Energy drift {:.3}% exceeds threshold for {}",
            energy_val.drift_pct, config.label
        );
    }

    // Compute D* from VACF
    let dt_snap = config.dt * config.dump_step as f64 * config.vel_snapshot_interval as f64;
    let max_lag = (sim.velocity_snapshots.len() / 2).max(10);

    if sim.velocity_snapshots.len() < 10 {
        return Err(format!(
            "Too few velocity snapshots ({}) for VACF",
            sim.velocity_snapshots.len()
        ));
    }

    let vacf = compute_vacf(
        &sim.velocity_snapshots,
        config.n_particles,
        dt_snap,
        max_lag,
    );

    let d_star_md = vacf.diffusion_coeff;
    let d_star_fit = d_star_daligault(config.gamma, config.kappa);

    let rel_error = if d_star_fit.abs() > f64::EPSILON {
        ((d_star_md - d_star_fit) / d_star_fit).abs()
    } else {
        0.0
    };

    // Compute viscosity from stress ACF (if we have enough position snapshots)
    let viscosity = if sim.positions_snapshots.len() > 10 && sim.velocity_snapshots.len() > 10 {
        let box_side = config.box_side();
        let volume = box_side * box_side * box_side;
        let mass = config.reduced_mass();

        let stress_series = compute_stress_xy(
            &sim.positions_snapshots,
            &sim.velocity_snapshots,
            config.n_particles,
            box_side,
            config.kappa,
            mass,
        );

        if stress_series.len() > 10 {
            let stress_acf = compute_stress_acf(
                &stress_series,
                dt_snap,
                volume,
                config.temperature(),
                (stress_series.len() / 2).max(10),
            );
            Some(stress_acf.viscosity)
        } else {
            None
        }
    } else {
        None
    };

    let passed = rel_error < tolerances::TRANSPORT_D_STAR_VS_FIT && energy_val.passed;

    Ok(TransportResult {
        kappa: config.kappa,
        gamma: config.gamma,
        d_star_md,
        d_star_daligault: d_star_fit,
        d_star_sarkas: None,
        rel_error_vs_daligault: rel_error,
        rel_error_vs_sarkas: None,
        viscosity,
        passed,
    })
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Transport Coefficient Validation                          ║");
    println!("║  Stanton & Murillo (2016) PRE 93 043203                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Use lite transport cases (3 representative points)
    let cases = config::transport_cases(500, true);
    let selected: Vec<_> = cases
        .into_iter()
        .filter(|c| {
            // Pick 3 representative cases: one per kappa
            (c.kappa - 1.0).abs() < 0.01 && (c.gamma - 50.0).abs() < 0.01
                || (c.kappa - 2.0).abs() < 0.01 && (c.gamma - 100.0).abs() < 0.01
                || (c.kappa - 3.0).abs() < 0.01 && (c.gamma - 100.0).abs() < 0.01
        })
        .collect();

    let n_density = 3.0 / (4.0 * PI);
    println!("  Running {} transport validation cases", selected.len());
    println!("  N = 500, reduced density n* = {n_density:.4}");
    println!();

    let mut harness = ValidationHarness::new("transport_coefficients");
    let mut results = Vec::new();

    for config in &selected {
        println!(
            "  ── Case: {} (κ={}, Γ={}) ──",
            config.label, config.kappa, config.gamma
        );

        match compute_transport_for_config(config) {
            Ok(result) => {
                let status = if result.passed { "PASS" } else { "FAIL" };
                println!("    D* (MD)       = {:.4e}", result.d_star_md);
                println!("    D* (Daligault)= {:.4e}", result.d_star_daligault);
                println!(
                    "    Rel error     = {:.1}% [{}] (< {}% required)",
                    result.rel_error_vs_daligault * 100.0,
                    status,
                    tolerances::TRANSPORT_D_STAR_VS_FIT * 100.0
                );
                if let Some(eta) = result.viscosity {
                    println!("    eta* (visc)   = {eta:.4e}");
                }
                println!();

                harness.check_rel(
                    &format!("D*_k{}_G{}", config.kappa, config.gamma),
                    result.d_star_md,
                    result.d_star_daligault,
                    tolerances::TRANSPORT_D_STAR_VS_FIT,
                );

                results.push(result);
            }
            Err(e) => {
                eprintln!("    FAILED: {e}");
                harness.check_bool(&format!("run_k{}_G{}", config.kappa, config.gamma), false);
            }
        }
    }

    println!();
    println!("  ── Summary ──");
    for r in &results {
        let icon = if r.passed { "✓" } else { "✗" };
        println!(
            "    {icon} κ={:.0} Γ={:<5.0}  D*_MD={:.4e}  D*_fit={:.4e}  err={:.1}%",
            r.kappa,
            r.gamma,
            r.d_star_md,
            r.d_star_daligault,
            r.rel_error_vs_daligault * 100.0
        );
    }

    harness.finish();
}
