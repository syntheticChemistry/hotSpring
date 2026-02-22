// SPDX-License-Identifier: AGPL-3.0-only

//! Nuclear EOS Level 3 — Deformed HFB Pipeline
//!
//! Physics: Axially-deformed HFB + BCS + Coulomb + Skyrme
//! Tests whether deformation improves accuracy over spherical L2 for
//! nuclei in known deformed regions (rare earths, actinides).
//!
//! Strategy:
//!   1. Use best L2 parameters as starting point
//!   2. Run deformed HFB on full nucleus set
//!   3. Compare: for each nucleus, use min(spherical, deformed) energy
//!   4. Report per-region accuracy breakdown
//!
//! Uses 100% `BarraCuda` native math.
//!
//! Run: cargo run --release --bin `nuclear_eos_l3_ref` [--seed=42]

use hotspring_barracuda::data;
use hotspring_barracuda::physics::hfb::binding_energy_l2;
use hotspring_barracuda::physics::hfb_deformed::binding_energy_l3;
use hotspring_barracuda::physics::nuclear_matter_properties;
use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;

use rayon::prelude::*;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
// Known good parameter sets
// ═══════════════════════════════════════════════════════════════════

/// Best L2 parameters from seed=42 (`chi2_BE=16.11`)
const BEST_L2_SEED42: [f64; 10] = [
    -1680.1857, 221.4840, -41.2736, 12119.2440, -0.2493, 0.3049, -0.4487, -0.2838, 0.4098, 199.9537,
];

/// Best L2 parameters from seed=123, lambda=1.0 (`chi2_BE=19.29`, all NMP within 2σ)
const BEST_L2_SEED123: [f64; 10] = [
    -1704.6959, 217.6615, -171.4343, 11613.0661, 0.1599, -1.5892, -0.7146, 0.3646, 0.3518, 93.8914,
];

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let param_set = args
        .iter()
        .find(|a| a.starts_with("--params="))
        .map_or("best_l2_42", |a| &a[9..]);

    let params: &[f64] = match param_set {
        "best_l2_123" => &BEST_L2_SEED123,
        "sly4" => &provenance::SLY4_PARAMS,
        _ => &BEST_L2_SEED42, // "best_l2_42" and all other inputs
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L3 — Deformed HFB Validation                  ║");
    println!("║  Axially-deformed HFB + BCS + Skyrme                       ║");
    println!("║  Math: 100% BarraCuda native                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Parameter set: {param_set}");
    println!("  Rayon threads: {}", rayon::current_num_threads());
    println!();

    // ── Load data ──────────────────────────────────────────────────
    let ctx = data::load_eos_context().expect("Failed to load EOS context");
    let base = &ctx.base;
    let exp_data = ctx.exp_data.clone();

    let nuclei: Vec<(usize, usize, f64)> = exp_data
        .iter()
        .map(|(&(z, n), &(b_exp, _))| (z, n, b_exp))
        .collect();

    println!("  Experimental nuclei: {}", nuclei.len());
    println!();

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: Spherical L2 baseline
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PHASE 1: Spherical L2 baseline");
    println!("═══════════════════════════════════════════════════════════════");

    let t1 = Instant::now();
    let l2_results: Vec<(usize, usize, f64, f64, bool)> = nuclei
        .par_iter()
        .map(|&(z, n, b_exp)| {
            let (b_calc, conv) = binding_energy_l2(z, n, params).expect("HFB solve");
            (z, n, b_exp, b_calc, conv)
        })
        .collect();
    let l2_time = t1.elapsed().as_secs_f64();
    println!(
        "  L2 spherical: {:.1}s ({} nuclei)",
        l2_time,
        l2_results.len()
    );

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: Deformed L3 evaluation
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PHASE 2: Deformed L3 evaluation");
    println!("═══════════════════════════════════════════════════════════════");

    let t2 = Instant::now();

    // Run deformed HFB for each nucleus
    let l3_results: Vec<(usize, usize, f64, f64, bool, f64)> = nuclei
        .par_iter()
        .map(|&(z, n, b_exp)| {
            let (b_calc, conv, beta2) = binding_energy_l3(z, n, params).expect("L3 solve");
            (z, n, b_exp, b_calc, conv, beta2)
        })
        .collect();
    let l3_time = t2.elapsed().as_secs_f64();
    println!(
        "  L3 deformed:  {:.1}s ({} nuclei)",
        l3_time,
        l3_results.len()
    );

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: Comparison — per-nucleus
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PER-NUCLEUS COMPARISON (L2 spherical vs L3 deformed)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  {:>4} {:>4} {:>5} {:>10} {:>10} {:>10} {:>8} {:>6} {:>6}",
        "Z", "N", "A", "B_exp", "B_L2", "B_L3", "beta2", "|dB_L2|", "|dB_L3|"
    );
    println!(
        "  {} {} {} {} {} {} {} {} {}",
        "-".repeat(4),
        "-".repeat(4),
        "-".repeat(5),
        "-".repeat(10),
        "-".repeat(10),
        "-".repeat(10),
        "-".repeat(8),
        "-".repeat(6),
        "-".repeat(6)
    );

    let mut chi2_l2 = 0.0;
    let mut chi2_l3 = 0.0;
    let mut chi2_best = 0.0;
    let mut n_valid = 0;
    let mut n_l3_better = 0;
    let mut n_l2_better = 0;

    let mut observed_l2 = Vec::new();
    let mut observed_l3 = Vec::new();
    let mut observed_best = Vec::new();
    let mut expected_vals = Vec::new();

    for i in 0..nuclei.len() {
        let (z, n, b_exp) = nuclei[i];
        let a = z + n;
        let b_l2 = l2_results[i].3;
        let b_l3 = l3_results[i].3;
        let beta2 = l3_results[i].5;
        let db_l2 = (b_l2 - b_exp).abs();
        let db_l3 = (b_l3 - b_exp).abs();
        let b_best = if b_l3 > 0.0 && db_l3 < db_l2 {
            b_l3
        } else {
            b_l2
        };
        let _db_best = (b_best - b_exp).abs();

        let sigma_theo = tolerances::sigma_theo(b_exp);

        if b_l2 > 0.0 && b_l3 > 0.0 {
            let c_l2 = ((b_l2 - b_exp) / sigma_theo).powi(2);
            let c_l3 = ((b_l3 - b_exp) / sigma_theo).powi(2);
            let c_best = ((b_best - b_exp) / sigma_theo).powi(2);
            chi2_l2 += c_l2;
            chi2_l3 += c_l3;
            chi2_best += c_best;
            n_valid += 1;

            if db_l3 < db_l2 {
                n_l3_better += 1;
            } else {
                n_l2_better += 1;
            }

            observed_l2.push(b_l2);
            observed_l3.push(b_l3);
            observed_best.push(b_best);
            expected_vals.push(b_exp);
        }

        let better = if b_l3 > 0.0 && db_l3 < db_l2 {
            "L3*"
        } else {
            "L2"
        };
        println!(
            "  {z:>4} {n:>4} {a:>5} {b_exp:>10.2} {b_l2:>10.2} {b_l3:>10.2} {beta2:>8.4} {db_l2:>6.1} {db_l3:>6.1} {better}"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    if n_valid > 0 {
        println!("  {:30} {:>12} {:>12}", "Method", "chi2/datum", "RMS(MeV)");
        println!(
            "  {:30} {:>12} {:>12}",
            "-".repeat(30),
            "-".repeat(12),
            "-".repeat(12)
        );

        let rms_l2 = (observed_l2
            .iter()
            .zip(&expected_vals)
            .map(|(o, e)| (o - e).powi(2))
            .sum::<f64>()
            / f64::from(n_valid))
        .sqrt();
        let rms_l3 = (observed_l3
            .iter()
            .zip(&expected_vals)
            .map(|(o, e)| (o - e).powi(2))
            .sum::<f64>()
            / f64::from(n_valid))
        .sqrt();
        let rms_best = (observed_best
            .iter()
            .zip(&expected_vals)
            .map(|(o, e)| (o - e).powi(2))
            .sum::<f64>()
            / f64::from(n_valid))
        .sqrt();

        println!(
            "  {:30} {:>12.4} {:>12.3}",
            "L2 (spherical)",
            chi2_l2 / f64::from(n_valid),
            rms_l2
        );
        println!(
            "  {:30} {:>12.4} {:>12.3}",
            "L3 (deformed)",
            chi2_l3 / f64::from(n_valid),
            rms_l3
        );
        println!(
            "  {:30} {:>12.4} {:>12.3}",
            "Best(L2,L3)",
            chi2_best / f64::from(n_valid),
            rms_best
        );
        println!(
            "  {:30} {:>12.2} {:>12}",
            "Python/scipy L2",
            provenance::L2_PYTHON_CHI2.value,
            "~45"
        );
        println!();
        println!("  L3 better for: {n_l3_better} / {n_valid} nuclei");
        println!("  L2 better for: {n_l2_better} / {n_valid} nuclei");

        if chi2_l3 / f64::from(n_valid) > chi2_l2 / f64::from(n_valid) * 0.95 {
            println!();
            println!("  NOTE: L3 deformed is not yet improving on L2 spherical.");
            println!("  This is expected — the deformed solver needs:");
            println!("    1. More SCF iterations (convergence tuning)");
            println!("    2. Better Coulomb in cylindrical coordinates");
            println!("    3. Constrained deformation (Q20 constraint)");
            println!("    4. Energy surface scanning (multiple beta2 values)");
            println!("    5. Spin-orbit and effective mass in deformed basis");
        }
    }

    // Per-region analysis
    println!();
    println!("  Accuracy by mass region:");
    println!(
        "  {:>15} {:>6} {:>10} {:>10} {:>10} {:>12}",
        "Region", "Count", "RMS_L2", "RMS_L3", "RMS_best", "L3_wins"
    );
    for (label, lo, hi) in &[
        ("Light A<56", 0, 56),
        ("Medium 56-100", 56, 100),
        ("Heavy 100-200", 100, 200),
        ("V.Heavy 200+", 200, 999),
    ] {
        let region: Vec<usize> = nuclei
            .iter()
            .enumerate()
            .filter(|(_, (z, n, _))| {
                let a = z + n;
                a >= *lo && a < *hi
            })
            .map(|(i, _)| i)
            .filter(|&i| i < observed_l2.len())
            .collect();
        if region.is_empty() {
            continue;
        }

        let mut sq_l2 = 0.0;
        let mut sq_l3 = 0.0;
        let mut sq_best = 0.0;
        let mut l3wins = 0;
        for &i in &region {
            sq_l2 += (observed_l2[i] - expected_vals[i]).powi(2);
            sq_l3 += (observed_l3[i] - expected_vals[i]).powi(2);
            sq_best += (observed_best[i] - expected_vals[i]).powi(2);
            if (observed_l3[i] - expected_vals[i]).abs() < (observed_l2[i] - expected_vals[i]).abs()
            {
                l3wins += 1;
            }
        }
        let cnt = region.len() as f64;
        println!(
            "  {:>15} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>9}/{:<3}",
            label,
            region.len(),
            (sq_l2 / cnt).sqrt(),
            (sq_l3 / cnt).sqrt(),
            (sq_best / cnt).sqrt(),
            l3wins,
            region.len()
        );
    }

    // NMP analysis
    if let Some(nmp) = nuclear_matter_properties(params) {
        println!();
        let vals = [
            nmp.rho0_fm3,
            nmp.e_a_mev,
            nmp.k_inf_mev,
            nmp.m_eff_ratio,
            nmp.j_mev,
        ];
        let targets = provenance::NMP_TARGETS.values();
        let sigmas = provenance::NMP_TARGETS.sigmas();
        let names = ["rho0", "E/A", "K_inf", "m*/m", "J"];
        println!("  NMP:");
        for i in 0..5 {
            let dev = (vals[i] - targets[i]) / sigmas[i];
            let ok = if dev.abs() <= 2.0 { "OK" } else { "!!" };
            println!(
                "    {} {:6} = {:>10.4} ({:>+6.2} sigma)",
                ok, names[i], vals[i], dev
            );
        }
    }

    // Timing
    println!();
    println!(
        "  Timing: L2={:.1}s, L3={:.1}s, total={:.1}s",
        l2_time,
        l3_time,
        l2_time + l3_time
    );
    println!("  L3/L2 cost ratio: {:.1}x", l3_time / l2_time.max(0.1));

    // Save results
    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();
    let result_json = serde_json::json!({
        "level": 3,
        "engine": "barracuda::l3_deformed_hfb",
        "params_set": param_set,
        "n_nuclei": nuclei.len(),
        "chi2_l2_per_datum": if n_valid > 0 { chi2_l2 / f64::from(n_valid) } else { 0.0 },
        "chi2_l3_per_datum": if n_valid > 0 { chi2_l3 / f64::from(n_valid) } else { 0.0 },
        "chi2_best_per_datum": if n_valid > 0 { chi2_best / f64::from(n_valid) } else { 0.0 },
        "n_l3_better": n_l3_better,
        "n_l2_better": n_l2_better,
        "l2_time_s": l2_time,
        "l3_time_s": l3_time,
    });
    let path = results_dir.join("barracuda_l3_deformed.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&result_json).expect("JSON serialize"),
    )
    .ok();
    println!("\n  Results saved to: {}", path.display());
}
