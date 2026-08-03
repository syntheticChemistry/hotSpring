// SPDX-License-Identifier: AGPL-3.0-or-later

//! arXiv β-scan — validate SU(3) plaquette against published data.
//!
//! Runs CPU HMC at multiple β values on 8⁴ lattice and compares
//! mean plaquette against published SU(3) Monte Carlo data
//! (Gattringer & Lang 2010, Creutz 1983, Bali et al. 2000).
//!
//! Produces a Novel Fermentation Transcript (NFT) via the provenance
//! trio: rhizoCrypt DAG events per β point, sweetGrass braid for the
//! full scan, loamSpine ledger entry for permanent record.
//! Falls back to local JSON receipt when NUCLEUS is unavailable.

use hotspring_barracuda::dag_provenance::{self, DagEvent, DagSession};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::measurement::RunManifest;
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::primal_bridge::NucleusContext;
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
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

    // ═══ Novel Fermentation Transcript (NFT) ═══
    // Braid = input parameters, Fermentation Transcript = computation results
    println!();
    println!("═══ Novel Fermentation Transcript (NFT) ═══");

    let manifest = RunManifest::capture("arxiv_beta_scan");

    #[derive(Serialize)]
    struct BetaScanReceipt {
        run: RunManifest,
        experiment_id: String,
        gauge_group: String,
        lattice_dims: [usize; 4],
        n_therm: usize,
        n_prod: usize,
        integrator: String,
        dt: f64,
        n_md_steps: usize,
        results: Vec<BetaScanResult>,
        wall_seconds: f64,
    }

    #[derive(Serialize)]
    struct BetaScanResult {
        beta: f64,
        plaquette_mean: f64,
        plaquette_stderr: f64,
        acceptance_rate: f64,
        published_reference: Option<f64>,
        reference_source: String,
    }

    let receipt_results: Vec<BetaScanResult> = results
        .iter()
        .map(|r| BetaScanResult {
            beta: r.beta,
            plaquette_mean: r.mean_plaq,
            plaquette_stderr: r.std_err,
            acceptance_rate: r.acceptance_rate,
            published_reference: r.published_plaq,
            reference_source: r.published_source.to_string(),
        })
        .collect();

    let receipt = BetaScanReceipt {
        run: manifest,
        experiment_id: "arxiv-su3-beta-scan-8x4".to_string(),
        gauge_group: "SU(3)".to_string(),
        lattice_dims: dims,
        n_therm,
        n_prod,
        integrator: "Omelyan 2MN".to_string(),
        dt: 0.02,
        n_md_steps: 20,
        results: receipt_results,
        wall_seconds: total_s,
    };

    let receipt_json = serde_json::to_string_pretty(&receipt).unwrap_or_default();
    let receipt_hash = dag_provenance::blake3_hex(receipt_json.as_bytes());

    // Write local receipt
    let receipt_path = "arxiv_beta_scan_receipt.json";
    if let Err(e) = std::fs::write(receipt_path, &receipt_json) {
        println!("  [WARN] Could not write receipt: {e}");
    } else {
        println!("  Receipt written to {receipt_path}");
        println!("  BLAKE3: {receipt_hash}");
    }

    // Attempt trio provenance commit if NUCLEUS is available
    let nucleus = NucleusContext::detect();
    if nucleus.any_alive() {
        println!("  NUCLEUS detected ({} primals alive)", nucleus.alive_names().len());

        if let Some(mut dag) = DagSession::begin(&nucleus, "arxiv-su3-beta-scan") {
            for r in &results {
                dag.append(
                    &nucleus,
                    DagEvent {
                        phase: format!("beta_{:.1}", r.beta),
                        input_hash: None,
                        output_hash: Some(dag_provenance::blake3_hex(
                            format!("{:.12}", r.mean_plaq).as_bytes(),
                        )),
                        wall_seconds: total_s / results.len() as f64,
                        summary: serde_json::json!({
                            "beta": r.beta,
                            "plaquette": r.mean_plaq,
                            "stderr": r.std_err,
                            "acceptance": r.acceptance_rate,
                        }),
                    },
                );
            }

            let provenance = dag.dehydrate(&nucleus);

            if let Some(commit_result) = dag_provenance::commit_provenance(
                &nucleus,
                &provenance,
                "arxiv-su3-beta-scan-8x4",
                Some("arXiv:hep-lat/ecoPrimals-SU3-pure-gauge"),
            ) {
                println!(
                    "  Trio commit: {}",
                    serde_json::to_string_pretty(&commit_result).unwrap_or_default()
                );
            }
        }
    } else {
        println!("  NUCLEUS not available — local receipt only");
        println!("  (Run inside NUCLEUS composition for full trio provenance)");
    }
}
