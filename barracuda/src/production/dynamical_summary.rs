// SPDX-License-Identifier: AGPL-3.0-only

//! Summary, NPU stats, and JSON output for the dynamical mixed pipeline (Exp 023).
//!
//! Extracted from production_dynamical_mixed to reduce binary size.

use crate::production::BetaResult;

/// NPU statistics for the 11-head dynamical mixed pipeline.
#[derive(Clone, Debug)]
pub struct DynamicalNpuStats {
    /// Pre-computation β priority screens.
    pub pre_screen_calls: usize,
    /// Parameter suggestions.
    pub param_suggests: usize,
    /// CG iteration estimates.
    pub cg_estimates: usize,
    /// Quenched-length predictions.
    pub quenched_length_predictions: usize,
    /// Quenched early-exits.
    pub quenched_early_exits: usize,
    /// Quenched steps saved.
    pub quenched_steps_saved: usize,
    /// Dynamical therm early-exits.
    pub therm_early_exits: usize,
    /// Dynamical therm total saved.
    pub therm_total_saved: usize,
    /// Reject predictions.
    pub reject_predictions: usize,
    /// Reject predictions correct.
    pub reject_correct: usize,
    /// Phase classifications.
    pub phase_classifications: usize,
    /// Quality scores.
    pub quality_scores: usize,
    /// Anomaly checks.
    pub anomaly_checks: usize,
    /// Anomalies found.
    pub anomalies_found: usize,
    /// Adaptive steer queries.
    pub adaptive_steered: usize,
    /// Adaptive points inserted.
    pub adaptive_inserted: usize,
    /// Next-run recommendations.
    pub next_run_recommendations: usize,
    /// Total NPU calls.
    pub total_npu_calls: usize,
}

impl DynamicalNpuStats {
    /// Create a new stats tracker with all counters at zero.
    pub fn new() -> Self {
        Self {
            pre_screen_calls: 0,
            param_suggests: 0,
            cg_estimates: 0,
            quenched_length_predictions: 0,
            quenched_early_exits: 0,
            quenched_steps_saved: 0,
            therm_early_exits: 0,
            therm_total_saved: 0,
            reject_predictions: 0,
            reject_correct: 0,
            phase_classifications: 0,
            quality_scores: 0,
            anomaly_checks: 0,
            anomalies_found: 0,
            adaptive_steered: 0,
            adaptive_inserted: 0,
            next_run_recommendations: 0,
            total_npu_calls: 0,
        }
    }
}

impl Default for DynamicalNpuStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the dynamical startup banner.
#[derive(Clone, Debug)]
pub struct DynamicalBannerConfig<'a> {
    /// Lattice extent per dimension.
    pub lattice: usize,
    /// Total lattice volume (L^4).
    pub vol: usize,
    /// Bare quark mass.
    pub mass: f64,
    /// β values to scan.
    pub betas: &'a [f64],
    /// CG convergence tolerance.
    pub cg_tol: f64,
    /// Maximum CG iterations per solve.
    pub cg_max_iter: usize,
    /// NPU check interval (trajectories between checks).
    pub check_interval: usize,
    /// Maximum thermalization trajectories per β.
    pub n_therm: usize,
    /// Quenched pre-thermalization trajectories.
    pub n_quenched_pretherm: usize,
    /// Measurement trajectories per β.
    pub n_meas: usize,
    /// Random seed.
    pub seed: u64,
}

/// HMC auto step size and MD steps from volume.
pub fn hmc_auto_params(vol: usize) -> (f64, usize) {
    let vol_f = vol as f64;
    let scale = (4096.0_f64 / vol_f).powf(0.25);
    let auto_dt = (0.01 * scale).max(0.002);
    let auto_n_md = ((1.0 / auto_dt).round() as usize).max(20);
    (auto_dt, auto_n_md)
}

/// Create trajectory log writer if path is given.
pub fn create_trajectory_log_writer(
    path: Option<&str>,
) -> Option<std::io::BufWriter<std::fs::File>> {
    let path = path?;
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let f = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("Cannot create trajectory log {path}: {e}"));
    Some(std::io::BufWriter::new(f))
}

/// Print the dynamical mixed pipeline startup banner.
pub fn print_dynamical_startup_banner(c: &DynamicalBannerConfig<'_>) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Dynamical Mixed Pipeline: Resident CG + 11-Head NPU Offload  ║");
    println!("║  Experiment 023: Dynamical Fermion + NPU GPU-Prep Assist      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Lattice:  {}⁴ ({} sites)", c.lattice, c.vol);
    println!("  Mass:     {}", c.mass);
    println!("  β values: {:?}", c.betas);
    println!(
        "  CG:       tol={:.0e}, max_iter={}, check_interval={}",
        c.cg_tol, c.cg_max_iter, c.check_interval
    );
    println!(
        "  Therm:    {} dyn + {} quenched pre-therm",
        c.n_therm, c.n_quenched_pretherm
    );
    println!("  Meas:     {}", c.n_meas);
    println!("  Seed:     {}", c.seed);
    println!();
}

/// Print the dynamical mixed pipeline summary table and NPU stats box.
pub fn print_dynamical_summary(
    results: &[BetaResult],
    npu_stats: &DynamicalNpuStats,
    lattice: usize,
    mass: f64,
    total_wall: f64,
    quenched_savings_pct: f64,
    therm_savings_pct: f64,
    gpu_name: &str,
    trajectory_log_path: Option<&str>,
) {
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Dynamical Mixed Pipeline Summary: {lattice}⁴ SU(3), m={mass}");
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>6} {:>10}",
        "β", "⟨P⟩", "σ(P)", "|L|", "χ", "acc%", "⟨CG⟩", "phase", "time"
    );
    for r in results {
        println!(
            "  {:>6.4} {:>10.6} {:>10.6} {:>10.4} {:>10.4} {:>7.1}% {:>8.0} {:>6} {:>9.1}s",
            r.beta,
            r.mean_plaq,
            r.std_plaq,
            r.polyakov,
            r.susceptibility,
            r.acceptance * 100.0,
            r.mean_cg_iters,
            &r.phase[..r.phase.len().min(6)],
            r.wall_s,
        );
    }
    println!();

    println!("  ┌──────────────────────────────────────────────────────────┐");
    println!("  │  11-Head NPU Offload Statistics                         │");
    println!("  ├──────────────────────────────────────────────────────────┤");
    println!("  │  PRE-COMPUTATION (GPU prep)                             │");
    println!(
        "  │    β priority screens:   {:<6}                         │",
        npu_stats.pre_screen_calls
    );
    println!(
        "  │    Parameter suggestions:{:<6}                         │",
        npu_stats.param_suggests
    );
    println!(
        "  │    CG estimates:         {:<6}                         │",
        npu_stats.cg_estimates
    );
    println!(
        "  │    Quenched-length preds:{:<6}                         │",
        npu_stats.quenched_length_predictions
    );
    println!("  │  DURING QUENCHED (NPU-monitored)                        │");
    println!(
        "  │    Quenched early-exits: {:<6} ({:.1}% steps saved)    │",
        npu_stats.quenched_early_exits, quenched_savings_pct
    );
    println!("  │  DURING DYNAMICAL                                       │");
    println!(
        "  │    Therm early-exits:    {:<6} / {:<6} ({:.1}% saved) │",
        npu_stats.therm_early_exits,
        results.len(),
        therm_savings_pct
    );
    println!(
        "  │    Reject predictions:   {:<6} (correct: {:<6})       │",
        npu_stats.reject_predictions, npu_stats.reject_correct
    );
    println!(
        "  │    Phase classifications:{:<6}                         │",
        npu_stats.phase_classifications
    );
    println!("  │  ADAPTIVE STEERING                                       │");
    println!(
        "  │    Steer queries:        {:<6} (inserted: {:<6})      │",
        npu_stats.adaptive_steered, npu_stats.adaptive_inserted
    );
    println!("  │  POST-COMPUTATION                                       │");
    println!(
        "  │    Quality scores:       {:<6}                         │",
        npu_stats.quality_scores
    );
    println!(
        "  │    Anomaly checks:       {:<6} (found: {:<6})         │",
        npu_stats.anomaly_checks, npu_stats.anomalies_found
    );
    println!(
        "  │    Next-run recommends:  {:<6}                         │",
        npu_stats.next_run_recommendations
    );
    println!(
        "  │  TOTAL NPU calls:        {:<6}                         │",
        npu_stats.total_npu_calls
    );
    println!("  └──────────────────────────────────────────────────────────┘");
    println!();

    println!("  GPU: {gpu_name}");
    println!(
        "  Total wall time: {:.1}s ({:.1} min)",
        total_wall,
        total_wall / 60.0,
    );
    if let Some(path) = trajectory_log_path {
        println!("  Trajectory log: {path}");
    }
    println!();
}

/// Write dynamical mixed pipeline results to JSON.
#[allow(clippy::too_many_arguments)]
pub fn write_dynamical_json(
    path: &str,
    results: &[BetaResult],
    lattice: usize,
    dims: [usize; 4],
    vol: usize,
    mass: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    check_interval: usize,
    gpu_name: &str,
    npu_name: &str,
    n_quenched_pretherm: usize,
    n_therm_max: usize,
    n_meas: usize,
    seed: u64,
    total_wall: f64,
    total_meas: usize,
    npu_stats: &DynamicalNpuStats,
    quenched_savings_pct: f64,
    therm_savings_pct: f64,
) {
    let json = serde_json::json!({
        "experiment": "023_DYNAMICAL_NPU_MIXED",
        "lattice": lattice,
        "dims": dims,
        "volume": vol,
        "mass": mass,
        "cg_tol": cg_tol,
        "cg_max_iter": cg_max_iter,
        "check_interval": check_interval,
        "gpu": gpu_name,
        "npu": npu_name,
        "n_quenched_pretherm": n_quenched_pretherm,
        "n_therm_max": n_therm_max,
        "n_meas": n_meas,
        "seed": seed,
        "total_wall_s": total_wall,
        "total_measurements": total_meas,
        "npu_stats": {
            "heads": 11,
            "pre_screen_calls": npu_stats.pre_screen_calls,
            "param_suggests": npu_stats.param_suggests,
            "cg_estimates": npu_stats.cg_estimates,
            "quenched_length_predictions": npu_stats.quenched_length_predictions,
            "quenched_early_exits": npu_stats.quenched_early_exits,
            "quenched_steps_saved": npu_stats.quenched_steps_saved,
            "quenched_savings_pct": quenched_savings_pct,
            "therm_early_exits": npu_stats.therm_early_exits,
            "therm_savings_pct": therm_savings_pct,
            "reject_predictions": npu_stats.reject_predictions,
            "reject_correct": npu_stats.reject_correct,
            "phase_classifications": npu_stats.phase_classifications,
            "quality_scores": npu_stats.quality_scores,
            "anomaly_checks": npu_stats.anomaly_checks,
            "anomalies_found": npu_stats.anomalies_found,
            "adaptive_steered": npu_stats.adaptive_steered,
            "adaptive_inserted": npu_stats.adaptive_inserted,
            "next_run_recommendations": npu_stats.next_run_recommendations,
            "total_npu_calls": npu_stats.total_npu_calls,
        },
        "points": build_point_array(results),
    });
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let json_str = serde_json::to_string_pretty(&json).unwrap_or_default();
    std::fs::write(path, json_str).unwrap_or_else(|e| eprintln!("  Failed to write {path}: {e}"));
    println!("  Results saved to: {path}");
}

fn build_point_array(results: &[BetaResult]) -> Vec<serde_json::Value> {
    results
        .iter()
        .map(|r| {
            serde_json::json!({
                "beta": r.beta,
                "mass": r.mass,
                "mean_plaquette": r.mean_plaq,
                "std_plaquette": r.std_plaq,
                "polyakov": r.polyakov,
                "susceptibility": r.susceptibility,
                "action_density": r.action_density,
                "acceptance": r.acceptance,
                "mean_cg_iterations": r.mean_cg_iters,
                "n_trajectories": r.n_traj,
                "wall_s": r.wall_s,
                "phase": r.phase,
                "therm_used": r.therm_used,
                "therm_budget": r.therm_budget,
                "dt_used": r.dt_used,
                "n_md_used": r.n_md_used,
                "npu_therm_early_exit": r.npu_therm_early_exit,
                "npu_quenched_budget": r.npu_quenched_budget,
                "npu_quenched_used": r.npu_quenched_used,
                "npu_quenched_early_exit": r.npu_quenched_early_exit,
                "npu_reject_predictions": r.npu_reject_predictions,
                "npu_reject_correct": r.npu_reject_correct,
                "npu_anomalies": r.npu_anomalies,
                "npu_cg_check_interval": r.npu_cg_check_interval
            })
        })
        .collect()
}
