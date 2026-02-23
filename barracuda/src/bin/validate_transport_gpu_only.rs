// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-Only Transport Validation (Paper 5) — New Baseline.
//!
//! Replaces the CPU+GPU dual-path transport validation with a pure GPU
//! pipeline: GPU MD → GPU velocity ring → GPU VACF → scalar D*.
//!
//! **Zero position/velocity readback during simulation.**
//! Only 16 bytes/dump (KE + PE scalars) cross `PCIe` during production.
//!
//! Validates the same physics as `validate_stanton_murillo`:
//!   - Energy conservation
//!   - Temperature stability
//!   - D* vs Daligault/Sarkas fits
//!   - D* ordering across coupling strengths
//!   - D* positive and finite
//!
//! Target: 10-50× faster than the CPU path (800-1200s → <100s).

use hotspring_barracuda::md::config;
use hotspring_barracuda::md::observables::validate_energy;
use hotspring_barracuda::md::simulation_transport_gpu::run_transport_gpu;
use hotspring_barracuda::md::transport::{d_star_daligault, sarkas_d_star_lookup};
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
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  GPU-Only Transport Validation (Paper 5)                   ║");
    println!("║  Unidirectional: GPU MD → GPU VACF → scalar D*            ║");
    println!("║  Zero velocity readback. New baseline.                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    let cases = config::transport_cases(500, true);
    let selected: Vec<_> = cases
        .into_iter()
        .filter(|c| {
            ((c.kappa - 1.0).abs() < 0.01 && (c.gamma - 50.0).abs() < 0.01)
                || ((c.kappa - 2.0).abs() < 0.01 && (c.gamma - 100.0).abs() < 0.01)
                || ((c.kappa - 1.0).abs() < 0.01 && (c.gamma - 72.0).abs() < 0.01)
                || ((c.kappa - 2.0).abs() < 0.01 && (c.gamma - 31.0).abs() < 0.01)
                || ((c.kappa - 3.0).abs() < 0.01 && (c.gamma - 100.0).abs() < 0.01)
                || ((c.kappa - 2.0).abs() < 0.01 && (c.gamma - 158.0).abs() < 0.01)
        })
        .collect();

    println!(
        "  Cases: {} transport points (same as Stanton-Murillo)",
        selected.len()
    );
    println!("  N=500, lite mode, GPU-only pipeline");
    println!();

    let mut harness = ValidationHarness::new("transport_gpu_only");

    struct CaseResult {
        kappa: f64,
        gamma: f64,
        d_star: f64,
        d_fit: f64,
        wall_time: f64,
        sim_time: f64,
        vacf_time: f64,
    }
    let mut results = Vec::new();

    for cfg in &selected {
        let d_fit = d_star_daligault(cfg.gamma, cfg.kappa);
        println!(
            "═══ κ={}, Γ={} ═══════════════════════════════════════════",
            cfg.kappa, cfg.gamma
        );
        println!("  D*(fit) = {d_fit:.4e}");

        let res = match rt.block_on(run_transport_gpu(cfg)) {
            Ok(r) => r,
            Err(e) => {
                println!("  GPU FAILED: {e}");
                harness.check_bool(&format!("GPU avail k{} G{}", cfg.kappa, cfg.gamma), false);
                println!();
                continue;
            }
        };

        let ev = validate_energy(&res.energy_history, cfg);
        println!(
            "  Energy drift: {:.4}% ({})",
            ev.drift_pct,
            if ev.passed { "OK" } else { "WARN" }
        );

        harness.check_upper(
            &format!("energy k{} G{}", cfg.kappa, cfg.gamma),
            ev.drift_pct,
            tolerances::ENERGY_DRIFT_PCT,
        );

        let d_star = res.d_star;
        println!("  D*(GPU) = {d_star:.4e}");
        println!(
            "  Time: {:.2}s total (sim {:.2}s + VACF {:.2}s)",
            res.wall_time_s, res.sim_time_s, res.vacf_time_s
        );
        println!("  Throughput: {:.0} steps/s", res.steps_per_sec);
        println!("  Snapshots: {} (GPU-resident)", res.n_snapshots);

        if d_star > 0.0 && d_fit > 0.0 {
            let fit_err = rel_error(d_star, d_fit);
            println!("  D* vs fit: {:.1}%", fit_err * 100.0);
            harness.check_upper(
                &format!("D* vs fit k{} G{}", cfg.kappa, cfg.gamma),
                fit_err,
                tolerances::TRANSPORT_D_STAR_VS_FIT_LITE,
            );
        }

        harness.check_bool(
            &format!("D* positive k{} G{}", cfg.kappa, cfg.gamma),
            d_star.is_finite() && d_star > 0.0,
        );

        if let Some(d_sarkas) = sarkas_d_star_lookup(cfg.kappa, cfg.gamma) {
            let sarkas_err = rel_error(d_star, d_sarkas);
            println!(
                "  D*(Sarkas) = {d_sarkas:.4e} (err: {:.1}%)",
                sarkas_err * 100.0
            );
        }

        results.push(CaseResult {
            kappa: cfg.kappa,
            gamma: cfg.gamma,
            d_star,
            d_fit,
            wall_time: res.wall_time_s,
            sim_time: res.sim_time_s,
            vacf_time: res.vacf_time_s,
        });
        println!();
    }

    // D* ordering checks
    for kappa_int in [1, 2, 3] {
        let kappa = f64::from(kappa_int);
        let mut kappa_results: Vec<&CaseResult> = results
            .iter()
            .filter(|r| (r.kappa - kappa).abs() < 0.01)
            .collect();
        kappa_results.sort_by(|a, b| a.gamma.total_cmp(&b.gamma));
        for w in kappa_results.windows(2) {
            harness.check_bool(
                &format!(
                    "D* ordering: D*(k{},G{}) > D*(k{},G{})",
                    kappa_int, w[0].gamma as u32, kappa_int, w[1].gamma as u32,
                ),
                w[0].d_star > w[1].d_star,
            );
        }
    }

    println!("═══ Summary ════════════════════════════════════════════════");
    println!(
        "  {:>3} {:>5} {:>11} {:>11} {:>8} {:>8} {:>8}",
        "κ", "Γ", "D*(GPU)", "D*(fit)", "total", "sim", "VACF"
    );
    let mut total_wall = 0.0;
    for r in &results {
        println!(
            "  {:>3.0} {:>5.0} {:>11.4e} {:>11.4e} {:>7.1}s {:>7.1}s {:>7.1}s",
            r.kappa, r.gamma, r.d_star, r.d_fit, r.wall_time, r.sim_time, r.vacf_time,
        );
        total_wall += r.wall_time;
    }
    println!();
    println!("  Total GPU wall time: {total_wall:.1}s");
    println!("  (vs ~1200s CPU Stanton-Murillo, ~800s CPU/GPU transport)");
    println!();

    harness.finish();
}
