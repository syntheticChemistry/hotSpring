// SPDX-License-Identifier: AGPL-3.0-only

//! Summary and JSON output for the quenched mixed pipeline (Exp 022).
//!
//! Extracted from `production_mixed_pipeline` to reduce binary size.

use crate::production::BetaResult;
use crate::production::beta_scan::QuenchedNpuStats;
use crate::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;

/// Print the mixed pipeline summary table, NPU stats, comparison, and physics quality.
pub fn print_mixed_summary(
    results: &[BetaResult],
    npu_stats: &QuenchedNpuStats,
    lattice: usize,
    total_trajectories: usize,
    total_wall: f64,
    _adaptive_count: usize,
    _final_beta_c: f64,
    gpu_name: &str,
    gpu_titan_name: Option<&str>,
    trajectory_log_path: Option<&str>,
) {
    let total_therm_budget: usize = results.iter().map(|r| r.therm_budget).sum();
    let total_therm_used: usize = results.iter().map(|r| r.therm_used).sum();
    let therm_savings_pct = if total_therm_budget > 0 {
        (1.0 - total_therm_used as f64 / total_therm_budget as f64) * 100.0
    } else {
        0.0
    };

    let total_meas: usize = results.iter().map(|r| r.n_traj).sum();
    let exp013_traj = 12 * (200 + 1000);
    let exp013_wall = 48988.3;
    let exp018_wall = 25560.0;
    let exp013_energy_kwh = exp013_wall * 300.0 / 3_600_000.0;
    let exp018_energy_kwh = exp018_wall * 300.0 / 3_600_000.0;
    let mixed_energy_kwh = total_wall * 300.0 / 3_600_000.0;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Mixed Pipeline β-Scan Summary: {lattice}⁴ Quenched SU(3)");
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>6} {:>10} {:>8}",
        "β", "⟨P⟩", "σ(P)", "|L|", "χ", "acc%", "traj", "therm", "time"
    );
    for r in results {
        let therm_col = if r.npu_therm_early_exit {
            format!("{}/{}", r.therm_used, r.therm_budget)
        } else {
            format!("{}", r.therm_used)
        };
        println!(
            "  {:>6.4} {:>10.6} {:>10.6} {:>10.4} {:>10.4} {:>7.1}% {:>6} {:>10} {:>7.1}s",
            r.beta,
            r.mean_plaq,
            r.std_plaq,
            r.polyakov,
            r.susceptibility,
            r.acceptance * 100.0,
            r.n_traj,
            therm_col,
            r.wall_s
        );
    }
    println!();

    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │  NPU Offload Statistics                                │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!(
        "  │  Therm early-exits:     {:<6} / {:<6} β points       │",
        npu_stats.therm_early_exits,
        results.len()
    );
    println!(
        "  │  Therm traj saved:      {:<6} / {:<6} ({:.1}%)       │",
        npu_stats.therm_total_saved, total_therm_budget, therm_savings_pct
    );
    println!(
        "  │  Reject predictions:    {:<6} (correct: {:<6})       │",
        npu_stats.reject_predictions, npu_stats.reject_correct
    );
    println!(
        "  │  Phase classifications: {:<6}                        │",
        npu_stats.phase_classifications
    );
    println!(
        "  │  Steering queries:      {:<6}                        │",
        npu_stats.steer_queries
    );
    println!(
        "  │  Total NPU calls:       {:<6}                        │",
        npu_stats.total_npu_calls
    );
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();

    println!("  ┌───────────────────────────────────────────────────────────────────┐");
    println!("  │  Comparison: Exp 022 (NPU offload) vs Exp 013/018               │");
    println!("  ├───────────────────────────────────────────────────────────────────┤");
    println!("  │  Metric             Exp 022       Exp 018 (DF64)  Exp 013 (f64)  │");
    println!(
        "  │  β points           {:<14}{:<16}{}              │",
        results.len(),
        12,
        12
    );
    println!(
        "  │  Total trajectories {:<14}{:<16}{}              │",
        total_trajectories, "~14400", exp013_traj
    );
    println!(
        "  │  Measurement traj   {:<14}{:<16}{}              │",
        total_meas, 12000, 12000
    );
    println!(
        "  │  Wall time          {total_wall:<13.1}s {exp018_wall:<15.1}s {exp013_wall:.1}s          │"
    );
    println!(
        "  │  Wall time (hrs)    {:<13.2}h {:<15.2}h {:.2}h          │",
        total_wall / 3600.0,
        exp018_wall / 3600.0,
        exp013_wall / 3600.0
    );
    println!(
        "  │  Energy (est.)      {mixed_energy_kwh:<12.2} kWh {exp018_energy_kwh:<14.2} kWh {exp013_energy_kwh:.2} kWh       │"
    );
    println!(
        "  │  Therm savings      {therm_savings_pct:<13.0}%                                   │"
    );
    println!(
        "  │  Speedup vs 013     {:<13.1}×                                   │",
        exp013_wall / total_wall
    );
    println!(
        "  │  Energy save vs 013 {:<12.0}%                                   │",
        (1.0 - mixed_energy_kwh / exp013_energy_kwh) * 100.0
    );
    println!("  └───────────────────────────────────────────────────────────────────┘");
    println!();

    // Physics quality check
    println!("  Physics Quality:");
    let near_bc: Vec<&BetaResult> = results
        .iter()
        .filter(|r| (r.beta - KNOWN_BETA_C).abs() < 0.15)
        .collect();
    let far_confined: Vec<&BetaResult> = results.iter().filter(|r| r.beta < 5.5).collect();
    let far_deconfined: Vec<&BetaResult> = results.iter().filter(|r| r.beta > 6.0).collect();

    if !near_bc.is_empty() {
        let max_chi = near_bc
            .iter()
            .map(|r| r.susceptibility)
            .fold(0.0_f64, f64::max);
        println!("    Susceptibility peak near β_c: χ_max = {max_chi:.2}");
    }
    if !far_confined.is_empty() {
        let mean_poly: f64 =
            far_confined.iter().map(|r| r.polyakov).sum::<f64>() / far_confined.len() as f64;
        println!("    Confined |L| (β<5.5):  {mean_poly:.4}");
    }
    if !far_deconfined.is_empty() {
        let mean_poly: f64 =
            far_deconfined.iter().map(|r| r.polyakov).sum::<f64>() / far_deconfined.len() as f64;
        println!("    Deconfined |L| (β>6.0): {mean_poly:.4}");
    }
    let monotonic = results
        .windows(2)
        .all(|w| w[1].mean_plaq >= w[0].mean_plaq - 0.01);
    println!(
        "    Plaquette monotonicity: {}",
        if monotonic { "PASS" } else { "MARGINAL" }
    );
    println!();

    println!("  GPU: {gpu_name}");
    if let Some(t) = gpu_titan_name {
        println!("  Titan V: {t}");
    }
    println!(
        "  Total wall time: {:.1}s ({:.2} hours)",
        total_wall,
        total_wall / 3600.0
    );
    if let Some(path) = trajectory_log_path {
        println!("  Trajectory log: {path}");
    }
    println!();
}

/// Context for writing mixed pipeline JSON results.
pub struct MixedJsonContext<'a> {
    /// Output file path.
    pub path: &'a str,
    /// Per-beta results.
    pub results: &'a [BetaResult],
    /// Lattice side length.
    pub lattice: usize,
    /// Lattice dimensions.
    pub dims: [usize; 4],
    /// Lattice volume.
    pub vol: usize,
    /// Primary GPU name.
    pub gpu_name: &'a str,
    /// Optional Titan V name.
    pub gpu_titan_name: Option<&'a str>,
    /// NPU name.
    pub npu_name: &'a str,
    /// Maximum thermalization trajectories.
    pub n_therm_max: usize,
    /// RNG seed.
    pub seed: u64,
    /// Total wall-clock time (seconds).
    pub total_wall: f64,
    /// Total trajectories across all betas.
    pub total_trajectories: usize,
    /// Total measurements.
    pub total_meas: usize,
    /// Number of adaptive rounds.
    pub adaptive_count: usize,
    /// Final ESN β_c estimate.
    pub final_beta_c: f64,
    /// NPU statistics.
    pub npu_stats: &'a QuenchedNpuStats,
}

/// Write mixed pipeline results to JSON.
pub fn write_mixed_json(ctx: &MixedJsonContext<'_>) {
    let MixedJsonContext {
        path,
        results,
        lattice,
        dims,
        vol,
        gpu_name,
        gpu_titan_name,
        npu_name,
        n_therm_max,
        seed,
        total_wall,
        total_trajectories,
        total_meas,
        adaptive_count,
        final_beta_c,
        npu_stats,
    } = ctx;
    let total_therm_budget: usize = results.iter().map(|r| r.therm_budget).sum();
    let total_therm_used: usize = results.iter().map(|r| r.therm_used).sum();
    let therm_savings_pct = if total_therm_budget > 0 {
        (1.0 - total_therm_used as f64 / total_therm_budget as f64) * 100.0
    } else {
        0.0
    };

    let exp013_traj = 12 * (200 + 1000);
    let exp013_wall = 48988.3;
    let exp018_wall = 25560.0;
    let exp013_energy_kwh = exp013_wall * 300.0 / 3_600_000.0;
    let mixed_energy_kwh = *total_wall * 300.0 / 3_600_000.0;

    let json = serde_json::json!({
        "experiment": "022_NPU_OFFLOAD_MIXED_PIPELINE",
        "lattice": lattice,
        "dims": dims,
        "volume": vol,
        "gpu": gpu_name,
        "titan_v": gpu_titan_name,
        "npu": npu_name,
        "n_therm_max": n_therm_max,
        "seed": seed,
        "total_wall_s": total_wall,
        "total_trajectories": total_trajectories,
        "total_measurements": total_meas,
        "adaptive_rounds": adaptive_count,
        "esn_beta_c": final_beta_c,
        "npu_stats": {
            "therm_early_exits": npu_stats.therm_early_exits,
            "therm_traj_saved": npu_stats.therm_total_saved,
            "therm_savings_pct": therm_savings_pct,
            "reject_predictions": npu_stats.reject_predictions,
            "reject_correct": npu_stats.reject_correct,
            "phase_classifications": npu_stats.phase_classifications,
            "steer_queries": npu_stats.steer_queries,
            "total_npu_calls": npu_stats.total_npu_calls,
        },
        "comparison": {
            "exp013_wall_s": exp013_wall,
            "exp013_trajectories": exp013_traj,
            "exp018_wall_s": exp018_wall,
            "speedup_vs_013": exp013_wall / *total_wall,
            "speedup_vs_018": exp018_wall / *total_wall,
            "trajectory_reduction_pct": (1.0 - *total_trajectories as f64 / f64::from(exp013_traj)) * 100.0,
            "energy_savings_vs_013_pct": (1.0 - mixed_energy_kwh / exp013_energy_kwh) * 100.0,
        },
        "points": results.iter().map(|r| serde_json::json!({
            "beta": r.beta,
            "mean_plaquette": r.mean_plaq,
            "std_plaquette": r.std_plaq,
            "polyakov": r.polyakov,
            "susceptibility": r.susceptibility,
            "action_density": r.action_density,
            "acceptance": r.acceptance,
            "n_trajectories": r.n_traj,
            "therm_used": r.therm_used,
            "therm_budget": r.therm_budget,
            "npu_therm_early_exit": r.npu_therm_early_exit,
            "npu_reject_predictions": r.npu_reject_predictions,
            "npu_reject_correct": r.npu_reject_correct,
            "wall_s": r.wall_s,
            "phase": r.phase,
        })).collect::<Vec<_>>(),
    });
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let json_str = serde_json::to_string_pretty(&json).unwrap_or_default();
    std::fs::write(path, json_str).unwrap_or_else(|e| eprintln!("  Failed to write {path}: {e}"));
    println!("  Results saved to: {path}");
}
