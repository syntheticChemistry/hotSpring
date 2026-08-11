// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rung 1 Analysis Pipeline — reads production_v2 JSON, computes jackknife
//! statistics, generates markdown tables for direct paper injection.
//!
//! No Python. No shell. Pure Rust deterministic analysis.
//!
//! Output modes:
//!   --markdown    → paper-ready tables (default)
//!   --json       → structured JSON for pseudoSpore manifest
//!   --validate   → convergence checks + literature comparison

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Deserialize)]
struct TrajectoryMeasurement {
    traj: usize,
    plaquette: f64,
    accepted: bool,
    delta_h: f64,
    polyakov_re: f64,
    polyakov_im: f64,
    wilson_loops: Option<Vec<[f64; 3]>>,
}

#[derive(Deserialize)]
struct RunResult {
    dims: [usize; 4],
    beta: f64,
    seed: u64,
    n_warmup: usize,
    n_production: usize,
    dt: f64,
    n_md: usize,
    integrator: String,
    start_type: String,
    gpu_name: String,
    wall_time_s: f64,
    warmup_accept_rate: f64,
    production_accept_rate: f64,
    measurements: Vec<TrajectoryMeasurement>,
    final_plaquette: f64,
    mean_plaquette: f64,
    plaquette_std: f64,
    config_blake3: String,
}

#[derive(Serialize, Clone)]
struct GridPoint {
    volume: usize,
    beta: f64,
    n_seeds: usize,
    plaquette_mean: f64,
    plaquette_jk_error: f64,
    polyakov_mean: f64,
    polyakov_jk_error: f64,
    acceptance_rate: f64,
    wall_time_total_s: f64,
    ms_per_trajectory: f64,
    wilson_loops: BTreeMap<String, f64>,
    creutz_ratios: BTreeMap<String, f64>,
}

#[derive(Serialize)]
struct AnalysisSummary {
    n_configs: usize,
    n_grid_points: usize,
    protocol: ProtocolMeta,
    grid: Vec<GridPoint>,
    literature_comparison: Vec<LitComparison>,
    convergence_checks: Vec<ConvergenceCheck>,
}

#[derive(Serialize)]
struct ProtocolMeta {
    integrator: String,
    dt: f64,
    n_md: usize,
    tau: f64,
    n_warmup: usize,
    n_production: usize,
    start_type: String,
    seeds_per_point: usize,
}

#[derive(Serialize)]
struct LitComparison {
    beta: f64,
    volume: usize,
    our_value: f64,
    our_error: f64,
    lit_value: f64,
    lit_error: f64,
    source: String,
    sigma_deviation: f64,
}

#[derive(Serialize)]
struct ConvergenceCheck {
    beta: f64,
    volumes: Vec<usize>,
    plaquettes: Vec<f64>,
    monotonic: bool,
    direction: String,
}

fn production_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/production_v2")
}

fn load_all_runs() -> Vec<RunResult> {
    let dir = production_dir();
    let mut runs = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(contents) = std::fs::read_to_string(&path) {
                    match serde_json::from_str::<RunResult>(&contents) {
                        Ok(run) => {
                            if run.mean_plaquette < 0.99 {
                                runs.push(run);
                            }
                        }
                        Err(e) => eprintln!("  WARN: failed to parse {:?}: {}", path, e),
                    }
                }
            }
        }
    }
    runs.sort_by(|a, b| {
        a.dims[0].cmp(&b.dims[0])
            .then(a.beta.partial_cmp(&b.beta).unwrap())
            .then(a.seed.cmp(&b.seed))
    });
    runs
}

fn jackknife_mean_error(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (values[0], 0.0);
    }
    let total: f64 = values.iter().sum();
    let mean = total / n as f64;

    let jk_means: Vec<f64> = (0..n)
        .map(|i| (total - values[i]) / (n - 1) as f64)
        .collect();

    let jk_var: f64 = jk_means.iter().map(|jk| (*jk - mean).powi(2)).sum::<f64>()
        * (n - 1) as f64 / n as f64;

    (mean, jk_var.sqrt())
}

fn compute_grid(runs: &[RunResult]) -> Vec<GridPoint> {
    let mut groups: BTreeMap<(usize, u64), Vec<&RunResult>> = BTreeMap::new();

    for run in runs {
        let beta_key = (run.beta * 100.0) as u64;
        let key = (run.dims[0], beta_key);
        groups.entry(key).or_default().push(run);
    }

    let mut grid = Vec::new();

    for ((vol, _beta_key), seeds) in &groups {
        let beta = seeds[0].beta;
        let n_seeds = seeds.len();

        // Plaquette jackknife across seeds
        let plaq_means: Vec<f64> = seeds.iter().map(|r| r.mean_plaquette).collect();
        let (plaq_mean, plaq_jk_err) = jackknife_mean_error(&plaq_means);

        // Polyakov loop jackknife
        let poly_means: Vec<f64> = seeds.iter().map(|r| {
            let polys: Vec<f64> = r.measurements.iter()
                .filter(|m| m.polyakov_re.abs() > 1e-15 || m.polyakov_im.abs() > 1e-15)
                .map(|m| (m.polyakov_re.powi(2) + m.polyakov_im.powi(2)).sqrt())
                .collect();
            if polys.is_empty() { 0.0 } else { polys.iter().sum::<f64>() / polys.len() as f64 }
        }).collect();
        let (poly_mean, poly_jk_err) = jackknife_mean_error(&poly_means);

        // Acceptance rate
        let acc = seeds.iter().map(|r| r.production_accept_rate).sum::<f64>() / n_seeds as f64;

        // Timing
        let total_wall: f64 = seeds.iter().map(|r| r.wall_time_s).sum();
        let total_trajs: f64 = seeds.iter()
            .map(|r| (r.n_warmup + r.n_production) as f64)
            .sum();
        let ms_per_traj = total_wall / total_trajs * 1000.0;

        // Wilson loops (from trajectories that have them)
        let mut wl_accum: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for seed in seeds.iter() {
            for m in &seed.measurements {
                if let Some(ref wls) = m.wilson_loops {
                    for wl in wls {
                        let key = format!("W({},{})", wl[0] as usize, wl[1] as usize);
                        wl_accum.entry(key).or_default().push(wl[2]);
                    }
                }
            }
        }
        let wilson_loops: BTreeMap<String, f64> = wl_accum.iter()
            .map(|(k, vs)| (k.clone(), vs.iter().sum::<f64>() / vs.len() as f64))
            .collect();

        // Creutz ratios from Wilson loops
        let mut creutz_ratios = BTreeMap::new();
        for r in 2..=4 {
            for t in 2..=4 {
                let w_rt = wilson_loops.get(&format!("W({},{})", r, t));
                let w_r1t1 = wilson_loops.get(&format!("W({},{})", r - 1, t - 1));
                let w_rt1 = wilson_loops.get(&format!("W({},{})", r, t - 1));
                let w_r1t = wilson_loops.get(&format!("W({},{})", r - 1, t));
                if let (Some(&wrt), Some(&wr1t1), Some(&wrt1), Some(&wr1t)) =
                    (w_rt, w_r1t1, w_rt1, w_r1t)
                {
                    if wrt > 0.0 && wr1t1 > 0.0 && wrt1 > 0.0 && wr1t > 0.0 {
                        let chi = -((wrt * wr1t1) / (wrt1 * wr1t)).ln();
                        creutz_ratios.insert(format!("χ({},{})", r, t), chi);
                    }
                }
            }
        }

        grid.push(GridPoint {
            volume: *vol,
            beta,
            n_seeds,
            plaquette_mean: plaq_mean,
            plaquette_jk_error: plaq_jk_err,
            polyakov_mean: poly_mean,
            polyakov_jk_error: poly_jk_err,
            acceptance_rate: acc,
            wall_time_total_s: total_wall,
            ms_per_trajectory: ms_per_traj,
            wilson_loops,
            creutz_ratios,
        });
    }
    grid
}

fn literature_comparisons(grid: &[GridPoint]) -> Vec<LitComparison> {
    // Known literature values for SU(3) pure gauge
    let lit = vec![
        (5.9, 16, 0.5850, 0.0002, "Bali et al. (2000)"),
        (6.0, 16, 0.5935, 0.0002, "Bali et al. (2000)"),
        (6.0, 24, 0.5941, 0.0001, "Necco-Sommer (2002)"),
        (6.0, 32, 0.5942, 0.0001, "Necco-Sommer (2002)"),
        (6.2, 16, 0.6136, 0.0002, "Bali et al. (2000)"),
        (6.2, 32, 0.6139, 0.0001, "Necco-Sommer (2002)"),
    ];

    let mut comparisons = Vec::new();
    for (beta, vol, lit_val, lit_err, source) in lit {
        if let Some(gp) = grid.iter().find(|g| g.volume == vol && (g.beta - beta).abs() < 0.01) {
            let combined_err = (gp.plaquette_jk_error * gp.plaquette_jk_error + lit_err * lit_err).sqrt();
            let sigma = if combined_err > 0.0 {
                (gp.plaquette_mean - lit_val).abs() / combined_err
            } else {
                0.0
            };
            comparisons.push(LitComparison {
                beta,
                volume: vol,
                our_value: gp.plaquette_mean,
                our_error: gp.plaquette_jk_error,
                lit_value: lit_val,
                lit_error: lit_err,
                source: source.to_string(),
                sigma_deviation: sigma,
            });
        }
    }
    comparisons
}

fn convergence_checks(grid: &[GridPoint]) -> Vec<ConvergenceCheck> {
    let mut betas: Vec<f64> = Vec::new();
    for g in grid {
        if !betas.iter().any(|&b| (b - g.beta).abs() < 0.001) {
            betas.push(g.beta);
        }
    }
    betas.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut checks = Vec::new();
    for beta in betas {
        let mut points: Vec<&GridPoint> = grid.iter()
            .filter(|g| (g.beta - beta).abs() < 0.01)
            .collect();
        points.sort_by_key(|g| g.volume);

        if points.len() >= 2 {
            let volumes: Vec<usize> = points.iter().map(|p| p.volume).collect();
            let plaquettes: Vec<f64> = points.iter().map(|p| p.plaquette_mean).collect();

            let monotonic = plaquettes.windows(2).all(|w| w[1] >= w[0]);
            let direction = if monotonic { "increasing (correct)" } else { "non-monotonic" };

            checks.push(ConvergenceCheck {
                beta,
                volumes,
                plaquettes,
                monotonic,
                direction: direction.to_string(),
            });
        }
    }
    checks
}

fn emit_markdown(summary: &AnalysisSummary) {
    println!("<!-- GENERATED by arxiv_analysis — do not hand-edit -->");
    println!("<!-- Production v2: {} configs, {} grid points -->", summary.n_configs, summary.n_grid_points);
    println!();

    // Protocol summary
    println!("### Production Protocol");
    println!();
    println!("| Parameter | Value |");
    println!("|-----------|-------|");
    println!("| Integrator | {} |", summary.protocol.integrator);
    println!("| dt | {} |", summary.protocol.dt);
    println!("| n_md | {} |", summary.protocol.n_md);
    println!("| τ = dt × n_md | {} |", summary.protocol.tau);
    println!("| Warmup | {} trajectories (cold start) |", summary.protocol.n_warmup);
    println!("| Production | {} trajectories × {} seeds |", summary.protocol.n_production, summary.protocol.seeds_per_point);
    println!();

    // Main plaquette table
    println!("### Plaquette Results");
    println!();
    println!("| Volume | β | ⟨P⟩ | σ_JK | Accept | ms/traj |");
    println!("|--------|-----|-----|------|--------|---------|");
    for gp in &summary.grid {
        println!("| {}⁴ | {:.1} | {:.8} | {:.2e} | {:.0}% | {:.1} |",
                 gp.volume, gp.beta, gp.plaquette_mean, gp.plaquette_jk_error,
                 gp.acceptance_rate * 100.0, gp.ms_per_trajectory);
    }
    println!();

    // Literature comparison
    if !summary.literature_comparison.is_empty() {
        println!("### Literature Comparison");
        println!();
        println!("| β | V | This work | Literature | Source | Deviation |");
        println!("|---|---|-----------|-----------|--------|-----------|");
        for lc in &summary.literature_comparison {
            println!("| {:.1} | {}⁴ | {:.6}({:.0}) | {:.4}({:.0}) | {} | {:.1}σ |",
                     lc.beta, lc.volume,
                     lc.our_value, lc.our_error * 1e6,
                     lc.lit_value, lc.lit_error * 1e4,
                     lc.source, lc.sigma_deviation);
        }
        println!();
    }

    // Volume convergence
    if !summary.convergence_checks.is_empty() {
        println!("### Volume Convergence");
        println!();
        for vc in &summary.convergence_checks {
            let pts: Vec<String> = vc.volumes.iter().zip(&vc.plaquettes)
                .map(|(v, p)| format!("{}⁴→{:.6}", v, p))
                .collect();
            let status = if vc.monotonic { "✓" } else { "⚠" };
            println!("- β={:.1}: {} {} ({})", vc.beta, pts.join(", "), status, vc.direction);
        }
        println!();
    }

    // Wilson loops
    let has_wilson = summary.grid.iter().any(|g| !g.wilson_loops.is_empty());
    if has_wilson {
        println!("### Wilson Loops W(R,T)");
        println!();
        println!("| Volume | β | W(1,2) | W(1,3) | W(2,2) | W(2,3) | W(3,3) |");
        println!("|--------|---|--------|--------|--------|--------|--------|");
        for gp in &summary.grid {
            if gp.wilson_loops.is_empty() { continue; }
            println!("| {}⁴ | {:.1} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} |",
                     gp.volume, gp.beta,
                     gp.wilson_loops.get("W(1,2)").unwrap_or(&0.0),
                     gp.wilson_loops.get("W(1,3)").unwrap_or(&0.0),
                     gp.wilson_loops.get("W(2,2)").unwrap_or(&0.0),
                     gp.wilson_loops.get("W(2,3)").unwrap_or(&0.0),
                     gp.wilson_loops.get("W(3,3)").unwrap_or(&0.0));
        }
        println!();
    }

    // Creutz ratios
    let has_creutz = summary.grid.iter().any(|g| !g.creutz_ratios.is_empty());
    if has_creutz {
        println!("### Creutz Ratios χ(R,T)");
        println!();
        println!("| Volume | β | χ(2,2) | χ(2,3) | χ(3,3) | χ(3,4) | χ(4,4) |");
        println!("|--------|---|--------|--------|--------|--------|--------|");
        for gp in &summary.grid {
            if gp.creutz_ratios.is_empty() { continue; }
            println!("| {}⁴ | {:.1} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
                     gp.volume, gp.beta,
                     gp.creutz_ratios.get("χ(2,2)").unwrap_or(&0.0),
                     gp.creutz_ratios.get("χ(2,3)").unwrap_or(&0.0),
                     gp.creutz_ratios.get("χ(3,3)").unwrap_or(&0.0),
                     gp.creutz_ratios.get("χ(3,4)").unwrap_or(&0.0),
                     gp.creutz_ratios.get("χ(4,4)").unwrap_or(&0.0));
        }
        println!();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("--markdown");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Rung 1 Analysis Pipeline — Deterministic Rust             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let runs = load_all_runs();
    println!("  Loaded: {} valid production configs", runs.len());
    println!("  Source: {:?}", production_dir());
    println!();

    if runs.is_empty() {
        eprintln!("  ERROR: No valid production data found.");
        std::process::exit(1);
    }

    let grid = compute_grid(&runs);
    let lit_comp = literature_comparisons(&grid);
    let conv_checks = convergence_checks(&grid);

    let protocol = ProtocolMeta {
        integrator: runs[0].integrator.clone(),
        dt: runs[0].dt,
        n_md: runs[0].n_md,
        tau: runs[0].dt * runs[0].n_md as f64,
        n_warmup: runs[0].n_warmup,
        n_production: runs[0].n_production,
        start_type: runs[0].start_type.clone(),
        seeds_per_point: 5,
    };

    let summary = AnalysisSummary {
        n_configs: runs.len(),
        n_grid_points: grid.len(),
        protocol,
        grid,
        literature_comparison: lit_comp,
        convergence_checks: conv_checks,
    };

    match mode {
        "--json" => {
            let json = serde_json::to_string_pretty(&summary).unwrap();
            println!("{}", json);
        }
        "--validate" => {
            println!("═══ VALIDATION REPORT ═══");
            println!();
            for vc in &summary.convergence_checks {
                let status = if vc.monotonic { "PASS" } else { "FAIL" };
                println!("  [{}] β={:.1} volume convergence: {}", status, vc.beta, vc.direction);
            }
            println!();
            for lc in &summary.literature_comparison {
                let status = if lc.sigma_deviation < 3.0 { "PASS" } else { "WARN" };
                println!("  [{}] β={:.1} {}⁴: {:.1}σ from {} ({})",
                         status, lc.beta, lc.volume, lc.sigma_deviation,
                         lc.source, if lc.sigma_deviation < 3.0 { "consistent" } else { "tension" });
            }
        }
        _ => {
            emit_markdown(&summary);
        }
    }
}
