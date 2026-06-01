// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nuclear EOS Level 2 — NMP-Constrained HFB Pipeline (Evolved)
//!
//! Run: cargo run --release --bin `nuclear_eos_l2_ref` [--lambda=0.1] [--seed=42]
//!      [--rounds=5] [--nm-starts=10] [--evals=100] [--patience=3] [--multi=3]

use hotspring_barracuda::bin_helpers::nuclear_eos_gpu::{
    compute_l2_binding_energies, run_l2_seed_pipeline, L2PipelineCli, SeedResult,
};
use hotspring_barracuda::data;
use hotspring_barracuda::physics::nuclear_matter_properties;
use hotspring_barracuda::provenance;

use barracuda::device::Auto;

use std::sync::Arc;

fn parse_args() -> L2PipelineCli {
    let args: Vec<String> = std::env::args().collect();
    let get = |prefix: &str| -> Option<String> {
        args.iter()
            .find(|a| a.starts_with(prefix))
            .map(|a| a[prefix.len()..].to_string())
    };
    let has = |flag: &str| -> bool { args.iter().any(|a| a == flag) };

    let lambda = get("--lambda=").and_then(|s| s.parse().ok()).unwrap_or(0.1);
    L2PipelineCli {
        seed: get("--seed=").and_then(|s| s.parse().ok()).unwrap_or(42),
        lambda,
        lambda_l1: get("--lambda-l1=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10.0),
        n_l1: get("--l1-samples=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(5000),
        nm_starts: get("--nm-starts=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10),
        evals_per_start: get("--evals=").and_then(|s| s.parse().ok()).unwrap_or(100),
        n_rounds: get("--rounds=").and_then(|s| s.parse().ok()).unwrap_or(5),
        patience: get("--patience=").and_then(|s| s.parse().ok()).unwrap_or(3),
        multi: get("--multi=").and_then(|s| s.parse().ok()).unwrap_or(1),
        sparsity: has("--sparsity"),
    }
}

fn main() {
    let cli = parse_args();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L2 — NMP-Constrained HFB Pipeline (Evolved)   ║");
    println!("║  Physics: p/n HFB + BCS + Coulomb + T_eff + CM             ║");
    println!("║  Math: 100% BarraCuda (gradient_1d, eigh_f64, brent)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  lambda(L2):      {}", cli.lambda);
    println!("  lambda(L1 seed): {}", cli.lambda_l1);
    println!("  seed:            {}", cli.seed);
    println!("  L1 screening:    {} samples", cli.n_l1);
    println!("  NM starts:       {} (from top-K L1)", cli.nm_starts);
    println!("  Evals per start: {}", cli.evals_per_start);
    println!("  Rounds:          {}", cli.n_rounds);
    println!("  Patience:        {}", cli.patience);
    println!("  Multi-seed:      {}", cli.multi);
    println!(
        "  SparsitySampler: {}",
        if cli.sparsity { "YES" } else { "no" }
    );
    println!("  Rayon threads:   {}", rayon::current_num_threads());
    println!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let discovered = rt
        .block_on(Auto::new())
        .expect("GPU device required for RBF surrogate (barracuda::Auto)");
    let device: Arc<_> = discovered
        .wgpu_device()
        .expect("RBF surrogate requires local wgpu device, not sovereign IPC")
        .clone();

    let ctx = data::load_eos_context().expect("Failed to load EOS context");
    let base = &ctx.base;
    let exp_data = ctx.exp_data.clone();
    let bounds = &ctx.bounds;

    let nuclei: Vec<(usize, usize, f64)> = exp_data
        .iter()
        .map(|(&(z, n), &(b_exp, _))| (z, n, b_exp))
        .collect();

    println!("  Experimental nuclei: {}", nuclei.len());
    println!("  Parameters:          {} dimensions", bounds.len());
    println!();

    let mut all_results: Vec<SeedResult> = Vec::new();

    for seed_idx in 0..cli.multi {
        let current_seed = cli.seed + seed_idx as u64 * 1000;
        all_results.push(run_l2_seed_pipeline(
            device.clone(),
            &cli,
            current_seed,
            bounds,
            &exp_data,
            &nuclei,
            seed_idx,
        ));
    }

    if cli.multi > 1 {
        print_multi_seed_summary(&cli, &all_results);
    }

    let best = all_results
        .iter()
        .min_by(|a, b| a.chi2_be.total_cmp(&b.chi2_be))
        .expect("at least one multi-seed result");

    print_comparison_summary(best);
    print_paper_parity(best, &nuclei);
    save_results(&cli, best, &all_results, base);
}

fn print_multi_seed_summary(cli: &L2PipelineCli, all_results: &[SeedResult]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  MULTI-SEED SUMMARY ({} seeds)                              ║",
        cli.multi
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  {:>8} {:>12} {:>12} {:>12} {:>8} {:>8}",
        "Seed", "chi2_BE", "chi2_NMP", "chi2_total", "Evals", "Time(s)"
    );
    for r in all_results {
        println!(
            "  {:>8} {:>12.4} {:>12.4} {:>12.4} {:>8} {:>8.1}",
            r.seed, r.chi2_be, r.chi2_nmp, r.chi2_total, r.n_evals, r.time
        );
    }

    let best = all_results
        .iter()
        .min_by(|a, b| a.chi2_be.total_cmp(&b.chi2_be))
        .expect("at least one multi-seed result");
    let mean_be: f64 = all_results.iter().map(|r| r.chi2_be).sum::<f64>() / cli.multi as f64;
    let std_be = if cli.multi > 1 {
        let var: f64 = all_results
            .iter()
            .map(|r| (r.chi2_be - mean_be).powi(2))
            .sum::<f64>()
            / (cli.multi - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    println!();
    println!(
        "  Best chi2_BE/datum:  {:.4} (seed={})",
        best.chi2_be, best.seed
    );
    println!("  Mean chi2_BE/datum:  {mean_be:.4} +/- {std_be:.4}");
}

fn print_comparison_summary(best: &SeedResult) {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  {:42} {:>10} {:>8} {:>8}",
        "Method", "chi2/dat", "Evals", "Time"
    );
    println!(
        "  {:42} {:>10} {:>8} {:>8}",
        "-".repeat(42),
        "-".repeat(10),
        "-".repeat(8),
        "-".repeat(8)
    );
    println!(
        "  {:42} {:>10.2} {:>8} {:>6.0}s",
        "DirectSampler (this run)", best.chi2_be, best.n_evals, best.time
    );

    if let Some((sp_be, _, _, sp_evals, sp_time)) = best.sparsity {
        println!(
            "  {:42} {:>10.2} {:>8} {:>6.0}s",
            "SparsitySampler (this run)", sp_be, sp_evals, sp_time
        );
    }

    println!(
        "  {:42} {:>10.2} {:>8} {:>8}",
        "Prev BarraCuda L2 (pre-fix)",
        provenance::L2_PYTHON_TOTAL_CHI2.value,
        4022,
        "743s"
    );
    println!(
        "  {:42} {:>10.2} {:>8} {:>8}",
        "Prev BarraCuda L2 (post-fix, small budget)", 16.11, 40, "1559s"
    );
    println!(
        "  {:42} {:>10.2} {:>8} {:>8}",
        "Python/scipy control",
        provenance::L2_PYTHON_CHI2.value,
        provenance::L2_PYTHON_CANDIDATES.value as i32,
        "~600s"
    );

    if best.chi2_be < provenance::L2_PYTHON_CHI2.value {
        let improvement = provenance::L2_PYTHON_CHI2.value / best.chi2_be;
        println!(
            "\n  PYTHON PARITY: EXCEEDED by {:.1}x (BarraCuda {:.2} vs Python {})",
            improvement,
            best.chi2_be,
            provenance::L2_PYTHON_CHI2.value
        );
    } else {
        let gap = best.chi2_be / provenance::L2_PYTHON_CHI2.value;
        println!("\n  PYTHON PARITY: {gap:.1}x gap remaining");
    }
}

fn print_paper_parity(best: &SeedResult, nuclei: &[(usize, usize, f64)]) {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PAPER PARITY ANALYSIS (target ~10^-6 relative accuracy)");
    println!("═══════════════════════════════════════════════════════════════");

    let (observed, expected, _sigma) = compute_l2_binding_energies(&best.params, nuclei);
    if observed.is_empty() {
        return;
    }

    let mut sum_rel_sq = 0.0;
    let mut sum_abs_sq = 0.0;
    for i in 0..observed.len() {
        let rel = ((observed[i] - expected[i]) / expected[i]).abs();
        sum_rel_sq += rel * rel;
        sum_abs_sq += (observed[i] - expected[i]).powi(2);
    }
    let rms_rel = (sum_rel_sq / observed.len() as f64).sqrt();
    let rms_abs = (sum_abs_sq / observed.len() as f64).sqrt();

    println!();
    println!("  RMS |dB/B|:      {rms_rel:.6e}");
    println!("  RMS |dB| (MeV):  {rms_abs:.3}");
    println!("  Paper target:    ~1.0e-06 (relative)");
    println!(
        "  Gap to paper:    {:.1} orders of magnitude",
        (rms_rel / 1e-6).log10()
    );
    println!();
    println!("  Physics model capabilities:");
    println!("    L1 SEMF floor:     ~5e-3 relative (~2-3 MeV RMS)");
    println!("    L2 HFB floor:      ~3e-4 relative (~0.5 MeV RMS)");
    println!("    L3 (deformed HFB): ~1e-5 relative (~0.1 MeV RMS)");
    println!("    Paper (beyond-MF): ~1e-6 relative (~keV RMS)");
    println!();
    if rms_rel < 1e-3 {
        println!("  L2 HFB is performing well. Next step: L3 deformation.");
    } else if rms_rel < 1e-2 {
        println!("  L2 HFB needs more optimization budget or constraint tuning.");
    } else {
        println!("  Significant gap remains. Need more L2 budget and lambda tuning.");
    }
}

fn save_results(cli: &L2PipelineCli, best: &SeedResult, all_results: &[SeedResult], base: &std::path::Path) {
    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();
    let result_json = serde_json::json!({
        "level": 2,
        "engine": "barracuda::l2_evolved_hfb",
        "physics": "p/n_HFB + BCS + Coulomb_Poisson_Slater + T_eff + CM",
        "math": "100pct_barracuda (gradient_1d_2ndorder, eigh_f64, brent)",
        "lambda": cli.lambda,
        "lambda_l1": cli.lambda_l1,
        "seed": cli.seed,
        "multi": cli.multi,
        "chi2_be_per_datum": best.chi2_be,
        "chi2_nmp_per_datum": best.chi2_nmp,
        "chi2_total_per_datum": best.chi2_total,
        "n_evals": best.n_evals,
        "time_seconds": best.time,
        "best_params": best.params,
        "nmp": nuclear_matter_properties(&best.params).map(|n| serde_json::json!({
            "rho0": n.rho0_fm3,
            "E_A": n.e_a_mev,
            "K_inf": n.k_inf_mev,
            "m_eff": n.m_eff_ratio,
            "J": n.j_mev,
        })),
        "all_seeds": all_results.iter().map(|r| serde_json::json!({
            "seed": r.seed,
            "chi2_be": r.chi2_be,
            "chi2_nmp": r.chi2_nmp,
            "n_evals": r.n_evals,
        })).collect::<Vec<_>>(),
        "references": {
            "prev_barracuda_l2_prefix": { "chi2_per_datum": provenance::L2_PYTHON_TOTAL_CHI2.value },
            "prev_barracuda_l2_postfix": { "chi2_per_datum": 16.11 },
            "python_scipy": { "chi2_per_datum": provenance::L2_PYTHON_CHI2.value },
        },
    });
    let path = results_dir.join("barracuda_l2_evolved.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&result_json).expect("JSON serialize"),
    )
    .ok();
    println!("\n  Results saved to: {}", path.display());
}
