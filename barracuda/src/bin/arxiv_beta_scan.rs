// SPDX-License-Identifier: AGPL-3.0-or-later

//! arXiv β-scan — validate SU(3) plaquette against published data.
//!
//! Runs CPU HMC at multiple β values on 8⁴ lattice and compares
//! mean plaquette against published SU(3) Monte Carlo data
//! (Gattringer & Lang 2010, Creutz 1983, Bali et al. 2000).
//!
//! This validates that our Wilson action and HMC implementation
//! reproduce the known SU(3) phase structure.

use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

struct BetaScanPoint {
    beta: f64,
    mean_plaq: f64,
    std_err: f64,
    acceptance_rate: f64,
    published_plaq: Option<f64>,
    published_source: &'static str,
    n_prod: usize,
}

fn cpu_beta_scan(
    dims: [usize; 4],
    beta: f64,
    n_therm: usize,
    n_prod: usize,
    published: Option<f64>,
    source: &'static str,
) -> BetaScanPoint {
    let mut lat = Lattice::hot_start(dims, beta, 42);
    let cfg = &mut HmcConfig {
        n_md_steps: 20,
        dt: 0.02,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };

    for _ in 0..n_therm {
        hmc::hmc_trajectory(&mut lat, cfg);
    }

    let mut plaqs = Vec::with_capacity(n_prod);
    let mut accepted = 0usize;
    for _ in 0..n_prod {
        let r = hmc::hmc_trajectory(&mut lat, cfg);
        plaqs.push(r.plaquette);
        if r.accepted {
            accepted += 1;
        }
    }

    let mean: f64 = plaqs.iter().sum::<f64>() / plaqs.len() as f64;
    let var: f64 = plaqs.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (plaqs.len() - 1) as f64;
    let std_err = (var / plaqs.len() as f64).sqrt();

    BetaScanPoint {
        beta,
        mean_plaq: mean,
        std_err,
        acceptance_rate: accepted as f64 / n_prod as f64,
        published_plaq: published,
        published_source: source,
        n_prod,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  arXiv β-Scan — SU(3) Pure Gauge Plaquette Validation      ║");
    println!("║  CPU HMC (Omelyan, 20 steps, dt=0.02) on 8⁴ lattice       ║");
    println!("║  Comparing against published SU(3) Monte Carlo data        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let dims = [8, 8, 8, 8];
    let n_therm = 50;
    let n_prod = 100;

    // Published SU(3) plaquette values for reference.
    //
    // Strong coupling (β < 4): strong-coupling expansion β/18 + O(β⁴)
    // Crossover region (β ≈ 5.5-5.7): rapid change
    // Weak coupling (β > 5.7): well-measured Monte Carlo values
    //
    // Sources:
    //   [GL10] Gattringer & Lang, QCD on the Lattice (2010), Table 3.1
    //   [C83]  Creutz, Phys. Rev. D 21, 2308 (1980); updated in Quarks, Gluons and Lattices (1983)
    //   [B00]  Bali et al., Phys. Rev. D 62, 054503 (2000)
    //   [SC]   Strong-coupling expansion: ⟨P⟩ ≈ β/18 at leading order
    let scan_points: Vec<(f64, Option<f64>, &str)> = vec![
        (2.0, Some(0.111), "SC: β/18"),
        (2.3, Some(0.150), "SC+corrections"),
        (3.0, Some(0.220), "SC+corrections"),
        (5.5, Some(0.505), "GL10/C83"),
        (5.7, Some(0.546), "GL10/B00"),
        (6.0, Some(0.593), "GL10/B00"),
        (6.5, None, "extrapolation"),
    ];

    println!("  Configuration: {0}⁴ lattice, {n_therm} therm + {n_prod} production trajectories", dims[0]);
    println!("  Integrator: Omelyan 2MN, 20 steps, dt=0.02");
    println!();

    let start = Instant::now();

    let mut results = Vec::new();
    for (beta, published, source) in &scan_points {
        let t0 = Instant::now();
        let point = cpu_beta_scan(dims, *beta, n_therm, n_prod, *published, source);
        let elapsed = t0.elapsed().as_secs_f64();
        print!("  β={:.1}: ⟨P⟩ = {:.6} ± {:.6}, accept={:.0}%",
            point.beta, point.mean_plaq, point.std_err, point.acceptance_rate * 100.0);
        if let Some(pub_val) = point.published_plaq {
            let delta = (point.mean_plaq - pub_val).abs();
            let sigma = if point.std_err > 0.0 { delta / point.std_err } else { 0.0 };
            print!("  | pub={:.3} |Δ|={:.4} ({:.1}σ) [{source}]", pub_val, delta, sigma);
        }
        println!("  [{:.1}s]", elapsed);
        results.push(point);
    }
    println!();

    let total_s = start.elapsed().as_secs_f64();
    println!("  Total wall time: {total_s:.1}s");
    println!();

    // Validation checks
    println!("═══ Validation Results ═══");
    println!();

    // 1. Monotonicity
    let monotonic = results.windows(2).all(|w| w[1].mean_plaq > w[0].mean_plaq - 0.001);
    println!("  [{}] Plaquette monotonically increasing with β",
        if monotonic { "PASS" } else { "FAIL" });

    // 2. Strong coupling: β=2.3 should give ⟨P⟩ ≈ 0.15
    let p_23 = results.iter().find(|r| (r.beta - 2.3).abs() < 0.01);
    if let Some(p) = p_23 {
        let ok = (p.mean_plaq - 0.150).abs() < 0.015;
        println!("  [{}] β=2.3: ⟨P⟩={:.6} (expect ~0.150 ± 0.015)",
            if ok { "PASS" } else { "FAIL" }, p.mean_plaq);
    }

    // 3. Weak coupling: β=5.7 should give ⟨P⟩ ≈ 0.546
    let p_57 = results.iter().find(|r| (r.beta - 5.7).abs() < 0.01);
    if let Some(p) = p_57 {
        let ok = (p.mean_plaq - 0.546).abs() < 0.020;
        println!("  [{}] β=5.7: ⟨P⟩={:.6} (published 0.546 ± 0.020)",
            if ok { "PASS" } else { "FAIL" }, p.mean_plaq);
    }

    // 4. Weak coupling: β=6.0 should give ⟨P⟩ ≈ 0.593
    let p_60 = results.iter().find(|r| (r.beta - 6.0).abs() < 0.01);
    if let Some(p) = p_60 {
        let ok = (p.mean_plaq - 0.593).abs() < 0.015;
        println!("  [{}] β=6.0: ⟨P⟩={:.6} (published 0.593 ± 0.015)",
            if ok { "PASS" } else { "FAIL" }, p.mean_plaq);
    }

    // 5. Agreement with published data (all points within 3σ)
    let mut all_within_3sigma = true;
    for r in &results {
        if let Some(pub_val) = r.published_plaq {
            let delta = (r.mean_plaq - pub_val).abs();
            let sigma = if r.std_err > 0.0 { delta / r.std_err } else { 0.0 };
            if sigma > 3.0 {
                println!("  [WARN] β={:.1}: {:.1}σ deviation from published value", r.beta, sigma);
                all_within_3sigma = false;
            }
        }
    }
    println!("  [{}] All points within 3σ of published SU(3) data",
        if all_within_3sigma { "PASS" } else { "FAIL" });

    // 6. Mean acceptance rate
    let mean_accept: f64 = results.iter().map(|r| r.acceptance_rate).sum::<f64>() / results.len() as f64;
    let accept_ok = mean_accept > 0.50;
    println!("  [{}] Mean acceptance rate: {:.0}% (threshold: >50%)",
        if accept_ok { "PASS" } else { "FAIL" }, mean_accept * 100.0);

    println!();

    // Summary table for paper
    println!("═══ Table for arXiv Section 4.4 ═══");
    println!();
    println!("| β   | ⟨P⟩ (this work) | σ_stat  | Published  | Source      | |Δ|/σ |");
    println!("|-----|-----------------|---------|------------|-------------|-------|");
    for r in &results {
        let pub_str = r.published_plaq.map_or("—".to_string(), |v| format!("{v:.3}"));
        let delta_sigma = if let Some(pub_val) = r.published_plaq {
            let delta = (r.mean_plaq - pub_val).abs();
            if r.std_err > 0.0 {
                format!("{:.1}", delta / r.std_err)
            } else {
                "—".to_string()
            }
        } else {
            "—".to_string()
        };
        println!("| {:.1} | {:.6}        | {:.5} | {:<10} | {:<11} | {:<5} |",
            r.beta, r.mean_plaq, r.std_err, pub_str, r.published_source, delta_sigma);
    }
    println!();
    println!("  SU(3) pure gauge HMC, 8⁴ lattice, Omelyan 2MN, dt=0.02, N_md=20");
    println!("  {n_therm} thermalization + {n_prod} production trajectories per β");
}
