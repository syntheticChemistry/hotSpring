// SPDX-License-Identifier: AGPL-3.0-only

//! Stanton-Murillo Transport Coefficient Validation (Paper 5).
//!
//! Validates Green-Kubo transport pipeline against Sarkas-calibrated
//! analytical fits (Daligault 2012, Stanton & Murillo 2016) and checks
//! internal consistency (MSD vs VACF), energy conservation, temperature
//! stability, and correct sign/ordering of D*, η*, λ*.
//!
//! Fit coefficients were recalibrated Feb 2026 using 12 Sarkas DSF study
//! D* values at N=2000 (Green-Kubo in physical units → D*=D/(a²ωp)).
//!
//! # Validation targets
//!
//! | Check                  | Tolerance | Basis                           |
//! |------------------------|-----------|---------------------------------|
//! | Energy conservation    | < 5%      | Symplectic VV + NVE             |
//! | T stability (NVE)      | < 30%     | Properly equilibrated liquid    |
//! | D* MSD vs fit          | < 80%     | Small-N noise vs calibrated fit |
//! | MSD ≈ VACF D*          | < 50%     | Internal consistency            |
//! | D* positive, finite    | —         | Physical                        |
//! | η* positive, finite    | —         | Physical                        |
//! | D*(κ=2) < D*(κ=1)     | —         | Stronger coupling → slower diff |

use hotspring_barracuda::md::config;
use hotspring_barracuda::md::cpu_reference;
use hotspring_barracuda::md::observables::{
    compute_d_star_msd, compute_heat_acf, compute_heat_current, compute_stress_acf,
    compute_stress_xy, compute_vacf, validate_energy,
};
use hotspring_barracuda::md::transport::{
    d_star_daligault, eta_star_stanton_murillo, lambda_star_stanton_murillo, sarkas_d_star_lookup,
};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn rel_error(a: f64, b: f64) -> f64 {
    if b.abs() > f64::EPSILON {
        ((a - b) / b).abs()
    } else {
        0.0
    }
}

fn main() {
    struct CaseResult {
        kappa: f64,
        gamma: f64,
        d_star_msd: f64,
        d_star_vacf: f64,
        eta_star_md: f64,
        lambda_star_md: f64,
        t_mean: f64,
        t_target: f64,
    }
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Stanton-Murillo Transport Validation (Paper 5)            ║");
    println!("║  D*, η*, λ* across (Γ, κ) phase diagram                   ║");
    println!("║  Stanton & Murillo (2016) PRE 93, 043203                   ║");
    println!("║  Extended: 6 transport points + Sarkas D* reference        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let cases = config::transport_cases(500, true);
    let selected: Vec<_> = cases
        .into_iter()
        .filter(|c| {
            // Original 2 points
            ((c.kappa - 1.0).abs() < 0.01 && (c.gamma - 50.0).abs() < 0.01)
                || ((c.kappa - 2.0).abs() < 0.01 && (c.gamma - 100.0).abs() < 0.01)
                // +4 Sarkas-matched points spanning weak → strong coupling
                || ((c.kappa - 1.0).abs() < 0.01 && (c.gamma - 72.0).abs() < 0.01)
                || ((c.kappa - 2.0).abs() < 0.01 && (c.gamma - 31.0).abs() < 0.01)
                || ((c.kappa - 3.0).abs() < 0.01 && (c.gamma - 100.0).abs() < 0.01)
                || ((c.kappa - 2.0).abs() < 0.01 && (c.gamma - 158.0).abs() < 0.01)
        })
        .collect();

    println!(
        "  Cases: {} transport points (2 original + 4 Sarkas-matched)",
        selected.len()
    );
    println!("  N = 500, lite mode (20k production steps)");
    println!();

    let mut harness = ValidationHarness::new("stanton_murillo_transport");
    let mut results = Vec::new();

    for cfg in &selected {
        let d_fit = d_star_daligault(cfg.gamma, cfg.kappa);
        let eta_fit = eta_star_stanton_murillo(cfg.gamma, cfg.kappa);
        let lambda_fit = lambda_star_stanton_murillo(cfg.gamma, cfg.kappa);

        println!(
            "═══ κ={}, Γ={} ═══════════════════════════════════════════",
            cfg.kappa, cfg.gamma
        );
        println!("  Analytical fits (informational): D*={d_fit:.4e}, η*={eta_fit:.4e}, λ*={lambda_fit:.4e}");

        let sim = cpu_reference::run_simulation_cpu(cfg);
        let ev = validate_energy(&sim.energy_history, cfg);

        harness.check_bool(
            &format!("energy_ok k{} G{}", cfg.kappa, cfg.gamma),
            ev.passed,
        );

        let dt_snap = cfg.dt * cfg.dump_step as f64 * cfg.vel_snapshot_interval as f64;
        let box_side = cfg.box_side();
        let volume = box_side.powi(3);
        let temperature = cfg.temperature();
        let mass = cfg.reduced_mass();

        // Mean production temperature
        let t_mean = if sim.energy_history.is_empty() {
            0.0
        } else {
            sim.energy_history
                .iter()
                .map(|e| e.temperature)
                .sum::<f64>()
                / sim.energy_history.len() as f64
        };
        let t_drift = rel_error(t_mean, temperature);
        println!(
            "  T*(mean) = {t_mean:.6}  (target: {temperature:.6}, drift: {:.1}%)",
            t_drift * 100.0
        );

        harness.check_upper(
            &format!("T_stability k{} G{}", cfg.kappa, cfg.gamma),
            t_drift,
            tolerances::TRANSPORT_T_STABILITY,
        );

        // D* from MSD (primary)
        let d_star_msd =
            compute_d_star_msd(&sim.positions_snapshots, cfg.n_particles, box_side, dt_snap);
        println!("  D*(MSD)  = {d_star_msd:.4e}");

        // D* from VACF (cross-check)
        let max_lag = (sim.velocity_snapshots.len() / 2).max(10);
        let vacf = compute_vacf(&sim.velocity_snapshots, cfg.n_particles, dt_snap, max_lag);
        let d_star_vacf = vacf.diffusion_coeff;
        println!("  D*(VACF) = {d_star_vacf:.4e}");
        println!("  D*(fit)  = {d_fit:.4e}  (Sarkas-calibrated)");

        // D* MSD vs calibrated fit: 80% tolerance (N=500 small-system noise)
        if d_star_msd > 0.0 && d_fit > 0.0 {
            let d_fit_err = rel_error(d_star_msd, d_fit);
            println!("  D* vs fit: {:.1}% relative", d_fit_err * 100.0);
            harness.check_upper(
                &format!("D* vs fit k{} G{}", cfg.kappa, cfg.gamma),
                d_fit_err,
                tolerances::TRANSPORT_D_STAR_VS_FIT_LITE,
            );
        }

        // MSD and VACF should agree within 50% (internal consistency)
        if d_star_msd > 0.0 && d_star_vacf > 0.0 {
            let msd_vacf_diff = rel_error(d_star_msd, d_star_vacf);
            println!("  MSD≈VACF: {:.1}% relative", msd_vacf_diff * 100.0);
            harness.check_upper(
                &format!("D* MSD≈VACF k{} G{}", cfg.kappa, cfg.gamma),
                msd_vacf_diff,
                tolerances::TRANSPORT_MSD_VACF_AGREEMENT,
            );
        }

        // D* must be positive and finite
        harness.check_bool(
            &format!("D* positive k{} G{}", cfg.kappa, cfg.gamma),
            d_star_msd.is_finite() && d_star_msd > 0.0,
        );

        // D* vs Sarkas Green-Kubo reference (informational at N=500)
        if let Some(d_sarkas) = sarkas_d_star_lookup(cfg.kappa, cfg.gamma) {
            let sarkas_err = rel_error(d_star_msd, d_sarkas);
            println!(
                "  D*(Sarkas) = {d_sarkas:.4e}  (N=2000 Green-Kubo reference, err: {:.1}%)",
                sarkas_err * 100.0,
            );
        }

        // η* from stress ACF
        let stress_series = compute_stress_xy(
            &sim.positions_snapshots,
            &sim.velocity_snapshots,
            cfg.n_particles,
            box_side,
            cfg.kappa,
            mass,
        );
        let stress_acf = compute_stress_acf(&stress_series, dt_snap, volume, temperature, max_lag);
        let eta_star_md = stress_acf.viscosity;
        println!("  η*(MD)   = {eta_star_md:.4e}  (fit: {eta_fit:.4e})");

        // η* must be positive and finite
        harness.check_bool(
            &format!("η* positive k{} G{}", cfg.kappa, cfg.gamma),
            eta_star_md.is_finite() && eta_star_md > 0.0,
        );

        // λ* from heat current ACF
        let jq_series = compute_heat_current(
            &sim.positions_snapshots,
            &sim.velocity_snapshots,
            cfg.n_particles,
            box_side,
            cfg.kappa,
            mass,
        );
        let heat_acf = compute_heat_acf(&jq_series, dt_snap, volume, temperature, max_lag);
        let lambda_star_md = heat_acf.thermal_conductivity;
        println!("  λ*(MD)   = {lambda_star_md:.4e}  (fit: {lambda_fit:.4e})");

        results.push(CaseResult {
            kappa: cfg.kappa,
            gamma: cfg.gamma,
            d_star_msd,
            d_star_vacf,
            eta_star_md,
            lambda_star_md,
            t_mean,
            t_target: temperature,
        });
        println!();
    }

    // Cross-case physics checks: D* ordering
    // Sort by coupling strength (Gamma) within each kappa
    for kappa_int in [1, 2, 3] {
        let kappa = f64::from(kappa_int);
        let mut kappa_results: Vec<&CaseResult> = results
            .iter()
            .filter(|r| (r.kappa - kappa).abs() < 0.01)
            .collect();
        kappa_results.sort_by(|a, b| {
            a.gamma
                .partial_cmp(&b.gamma)
                .expect("finite gamma comparison")
        });
        for w in kappa_results.windows(2) {
            harness.check_bool(
                &format!(
                    "D* ordering: D*(k{},G{}) > D*(k{},G{})",
                    kappa_int, w[0].gamma as u32, kappa_int, w[1].gamma as u32,
                ),
                w[0].d_star_msd > w[1].d_star_msd,
            );
        }
    }

    // Cross-kappa check: at similar Gamma, higher kappa → higher D* (weaker eff. coupling)
    let find = |k: f64, g: f64| -> Option<f64> {
        results
            .iter()
            .find(|r| (r.kappa - k).abs() < 0.01 && (r.gamma - g).abs() < 0.5)
            .map(|r| r.d_star_msd)
    };
    if let (Some(d_k2_g100), Some(d_k3_g100)) = (find(2.0, 100.0), find(3.0, 100.0)) {
        harness.check_bool(
            "D* screening: D*(k3,G100) > D*(k2,G100)",
            d_k3_g100 > d_k2_g100,
        );
    }

    println!("═══ Phase Diagram Summary ══════════════════════════════════");
    println!(
        "  {:>3} {:>5} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "κ", "Γ", "D*(MSD)", "D*(VACF)", "D*(Sarkas)", "η*(MD)", "λ*(MD)", "T*/T*_tgt"
    );
    for r in &results {
        let d_sarkas_str = match sarkas_d_star_lookup(r.kappa, r.gamma) {
            Some(d) => format!("{d:>11.4e}"),
            None => format!("{:>11}", "—"),
        };
        println!(
            "  {:>3.0} {:>5.0} {:>11.4e} {:>11.4e} {} {:>11.4e} {:>11.4e} {:>11.3}",
            r.kappa,
            r.gamma,
            r.d_star_msd,
            r.d_star_vacf,
            d_sarkas_str,
            r.eta_star_md,
            r.lambda_star_md,
            r.t_mean / r.t_target,
        );
    }
    println!();
    println!(
        "  Green-Kubo transport pipeline: {} cases validated.",
        results.len()
    );
    println!("  D* fit coefficients Sarkas-calibrated (12 points, N=2000).");
    println!("  η*/λ* fit coefficients proportionally rescaled (not independently calibrated).");
    println!("  Sarkas D* references shown for points with N=2000 ground truth.");
    println!();

    harness.finish();
}
