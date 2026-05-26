// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-target analysis dispatch and GROMACS process management.

use std::path::Path;
use std::process::{Command, Stdio};

use crate::colvar;
use crate::config;
use crate::fes;
use crate::hills;
use crate::report::{BasinMatch, TargetReport};
use crate::stats;

/// Reference basin definitions for alanine dipeptide (AMBER99SB vacuum).
struct AlanineRef;
impl AlanineRef {
    const C7EQ: (f64, f64) = (-1.4, 1.0);
    const C7AX: (f64, f64) = (1.0, -0.7);
    const TOLERANCE_RAD: f64 = 0.5;
    const BARRIER_RANGE: (f64, f64) = (20.0, 45.0);
}

/// Analyze a single target directory, dispatching to the appropriate method.
pub fn analyze_target(target_dir: &Path) -> TargetReport {
    let name = target_dir.file_name().unwrap().to_string_lossy().to_string();

    if name.contains("alanine_dipeptide") {
        analyze_alanine(target_dir)
    } else if name.contains("chignolin") {
        analyze_chignolin(target_dir)
    } else {
        TargetReport::skipped(&name)
    }
}

fn analyze_alanine(target_dir: &Path) -> TargetReport {
    let mut report = TargetReport {
        target_id: "01_alanine_dipeptide".to_string(),
        method: "well-tempered_metadynamics".to_string(),
        passes: Vec::new(),
        fails: Vec::new(),
        pass_rate: 0.0,
        industry_standard: false,
        skipped: false,
        metrics: serde_json::Value::Null,
    };

    let hills_path = target_dir.join("output/HILLS");
    if !hills_path.exists() {
        report.skipped = true;
        return report;
    }

    let hills_data = match hills::parse_hills_2d(&hills_path) {
        Ok(h) => h,
        Err(e) => {
            report.fails.push(format!("Failed to parse HILLS: {e}"));
            report.compute_rates();
            return report;
        }
    };

    let pi = std::f64::consts::PI;
    let nbins = 150;
    let fes_2d = fes::reconstruct_2d(
        &hills_data, -pi, pi, -pi, pi, nbins, nbins, (true, true),
    );

    // Find minima
    let minima = fes::find_minima_2d(&fes_2d, 20.0, 3);

    // Basin classification
    let basins_to_check = [
        ("C7eq", AlanineRef::C7EQ.0, AlanineRef::C7EQ.1),
        ("C7ax", AlanineRef::C7AX.0, AlanineRef::C7AX.1),
    ];

    let mut matched_basins = Vec::new();
    for (name, ref_x, ref_y) in &basins_to_check {
        let mut best_dist = f64::INFINITY;
        let mut best_min = None;
        for m in &minima {
            let dy = m.y.unwrap_or(0.0) - ref_y;
            let dist = ((m.x - ref_x).powi(2) + dy.powi(2)).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_min = Some(m);
            }
        }

        if let Some(m) = best_min {
            if best_dist < AlanineRef::TOLERANCE_RAD {
                report.passes.push(format!("{name} minimum within {:.1} rad", AlanineRef::TOLERANCE_RAD));
                matched_basins.push(BasinMatch {
                    name: name.to_string(),
                    ref_x: *ref_x,
                    ref_y: Some(*ref_y),
                    found_x: m.x,
                    found_y: m.y,
                    found_energy: m.energy,
                    distance: best_dist,
                    within_tolerance: true,
                });
            } else {
                report.fails.push(format!("{name} minimum too far: {best_dist:.2} rad"));
            }
        } else {
            report.fails.push(format!("{name} minimum not found"));
        }
    }

    // 1D phi projection and barrier analysis
    let kt = 2.494; // 300K
    let fes_phi = project_phi(&fes_2d, kt);

    // Find barrier from 1D projection
    let c7eq_idx = fes_phi.grid.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| ((*a - (-1.4)).abs()).partial_cmp(&((*b - (-1.4)).abs())).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let barrier_region: Vec<f64> = fes_phi.free_energy.iter()
        .zip(fes_phi.grid.iter())
        .filter(|(_, &x)| x > -0.5 && x < 0.5)
        .map(|(&e, _)| e)
        .collect();

    if !barrier_region.is_empty() {
        let barrier = barrier_region.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - fes_phi.free_energy[c7eq_idx];
        if barrier >= AlanineRef::BARRIER_RANGE.0 && barrier <= AlanineRef::BARRIER_RANGE.1 {
            report.passes.push(format!("Barrier {barrier:.1} kJ/mol within reference range"));
        } else {
            report.fails.push(format!("Barrier {barrier:.1} kJ/mol outside range ({:.0}, {:.0})",
                AlanineRef::BARRIER_RANGE.0, AlanineRef::BARRIER_RANGE.1));
        }
    }

    // Convergence via stride reconstruction (2D → project to phi)
    let n_windows = 10;
    let n_gauss = hills_data.n_gaussians();
    let window_size = n_gauss / n_windows;

    let mut stride_phi_fes: Vec<fes::Fes1D> = Vec::new();
    for w in 1..=n_windows {
        let end = (w * window_size).min(n_gauss);
        let partial = hills::Hills2D {
            time: hills_data.time[..end].to_vec(),
            centers_x: hills_data.centers_x[..end].to_vec(),
            centers_y: hills_data.centers_y[..end].to_vec(),
            sigmas_x: hills_data.sigmas_x[..end].to_vec(),
            sigmas_y: hills_data.sigmas_y[..end].to_vec(),
            heights: hills_data.heights[..end].to_vec(),
            biasfactor: hills_data.biasfactor[..end].to_vec(),
        };
        let fes_2d_w = fes::reconstruct_2d(&partial, -pi, pi, -pi, pi, nbins, nbins, (true, true));
        stride_phi_fes.push(project_phi(&fes_2d_w, kt));
    }

    let block_avg = fes::block_average(&stride_phi_fes, 5);
    if let Some(ref ba) = block_avg {
        if ba.converged {
            report.passes.push(format!("Block averaging converged (max stderr {:.2} kJ/mol)", ba.max_stderr));
        } else {
            report.fails.push(format!("Block averaging NOT converged (max stderr {:.2} kJ/mol)", ba.max_stderr));
        }
    }

    let convergence = fes::convergence_barriers(&stride_phi_fes, -1.4);
    if convergence.converged {
        report.passes.push("Barrier convergence < 3 kJ/mol over last 3 windows".to_string());
    } else {
        report.fails.push(format!("Barrier NOT converged (std = {:.1} kJ/mol)", convergence.std_last_3));
    }

    // Statistical analysis on HILLS heights (convergence indicator)
    let height_stats = stats::full_analysis(&hills_data.heights);

    // Metrics
    report.metrics = serde_json::json!({
        "n_gaussians": hills_data.n_gaussians(),
        "minima_found": minima.len(),
        "fes_resolution": format!("{nbins}x{nbins}"),
        "block_averaging": block_avg,
        "convergence": convergence,
        "basin_matches": matched_basins,
        "hills_height_decay": hills_data.heights.last().unwrap_or(&0.0)
            / hills_data.heights.first().unwrap_or(&1.0),
        "height_stats": {
            "autocorrelation_time": height_stats.autocorrelation_time,
            "statistical_inefficiency": height_stats.statistical_inefficiency,
            "effective_n": height_stats.effective_n,
            "equilibration_t0_fraction": height_stats.equilibration.t0_fraction,
            "production_fraction": height_stats.equilibration.production_fraction,
            "optimal_block_size": height_stats.optimal_block.optimal_block_size,
        },
    });

    report.compute_rates();
    report
}

fn analyze_chignolin(target_dir: &Path) -> TargetReport {
    let mut report = TargetReport {
        target_id: "02_chignolin_opes".to_string(),
        method: "OPES_METAD+OPES_METAD_EXPLORE".to_string(),
        passes: Vec::new(),
        fails: Vec::new(),
        pass_rate: 0.0,
        industry_standard: false,
        skipped: false,
        metrics: serde_json::Value::Null,
    };

    // Try to find COLVAR
    let colvar_path = ["COLVARb", "COLVAR", "output/COLVARb", "output/COLVAR"]
        .iter()
        .map(|p| target_dir.join(p))
        .find(|p| p.exists());

    let colvar_path = match colvar_path {
        Some(p) => p,
        None => {
            report.skipped = true;
            return report;
        }
    };

    let cv = match colvar::parse_colvar(&colvar_path) {
        Ok(c) => c,
        Err(e) => {
            report.fails.push(format!("Failed to parse COLVAR: {e}"));
            report.compute_rates();
            return report;
        }
    };

    let hlda = cv.column("hlda").unwrap_or_default();
    let total_bias: Vec<f64> = {
        let opes_bias = cv.column("opes.bias").unwrap_or_else(|| vec![0.0; cv.n_frames]);
        let opese_bias = cv.column("opesE.bias").unwrap_or_else(|| vec![0.0; cv.n_frames]);
        opes_bias.iter().zip(opese_bias.iter()).map(|(a, b)| a + b).collect()
    };

    let kt = 2.83; // 340K
    let sim_time_ns = cv.total_time_ns();

    // Folding free energy via reweighting
    let (fold_events, unfold_events) = colvar::count_transitions(&hlda, 1.2, 0.3);
    let total_transitions = fold_events + unfold_events;

    if total_transitions > 0 {
        report.passes.push(format!("Observed {total_transitions} folding/unfolding transitions"));
    } else {
        report.fails.push("No folding/unfolding transitions observed".to_string());
    }

    // Reweighted FES
    let (_centers, _fes_hlda) = colvar::reweighted_fes_1d(&hlda, &total_bias, 80, kt);

    // Compute folding FE
    let weights: Vec<f64> = total_bias.iter().map(|b| (b / kt).exp()).collect();
    let w_sum: f64 = weights.iter().sum();
    let norm_weights: Vec<f64> = weights.iter().map(|w| w / w_sum).collect();

    let p_folded: f64 = hlda.iter().zip(norm_weights.iter())
        .filter(|(&h, _)| h > 1.0 && h < 2.5)
        .map(|(_, &w)| w)
        .sum();
    let p_unfolded: f64 = hlda.iter().zip(norm_weights.iter())
        .filter(|(&h, _)| h < 0.5)
        .map(|(_, &w)| w)
        .sum();

    let delta_g = if p_folded > 0.0 && p_unfolded > 0.0 {
        -kt * (p_folded / p_unfolded).ln()
    } else {
        f64::NAN
    };

    if delta_g.is_finite() {
        let ref_range = config::load_config(target_dir)
            .and_then(|c| c.reference.folding_fe_range_kjmol)
            .map(|r| (r[0], r[1]))
            .unwrap_or((15.0, 24.0));
        if delta_g >= ref_range.0 && delta_g <= ref_range.1 {
            report.passes.push(format!("ΔG_fold = {delta_g:.1} kJ/mol within reference range"));
        } else {
            report.fails.push(format!("ΔG_fold = {delta_g:.1} kJ/mol outside range ({:.0}, {:.0})",
                ref_range.0, ref_range.1));
        }
    }

    // Convergence
    let n_windows = 10.min(cv.n_frames / 100);
    if n_windows >= 3 {
        let window_size = cv.n_frames / n_windows;
        let mut delta_gs = Vec::new();
        for w in 1..=n_windows {
            let end = (w * window_size).min(cv.n_frames);
            let w_slice: Vec<f64> = total_bias[..end].iter().map(|b| (b / kt).exp()).collect();
            let ws: f64 = w_slice.iter().sum();
            let nw: Vec<f64> = w_slice.iter().map(|x| x / ws).collect();

            let pf: f64 = hlda[..end].iter().zip(nw.iter())
                .filter(|(&h, _)| h > 1.0 && h < 2.5)
                .map(|(_, &w)| w)
                .sum();
            let pu: f64 = hlda[..end].iter().zip(nw.iter())
                .filter(|(&h, _)| h < 0.5)
                .map(|(_, &w)| w)
                .sum();

            if pf > 0.0 && pu > 0.0 {
                delta_gs.push(-kt * (pf / pu).ln());
            }
        }

        if delta_gs.len() >= 3 {
            let last3 = &delta_gs[delta_gs.len() - 3..];
            let mean = last3.iter().sum::<f64>() / 3.0;
            let std = (last3.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 3.0).sqrt();

            if std < 4.0 {
                report.passes.push("FE convergence < 4 kJ/mol over last 3 windows".to_string());
            } else {
                report.fails.push(format!("FE NOT converged (std = {std:.1} kJ/mol)"));
            }
        }
    }

    // Statistical analysis on HLDA CV
    let hlda_stats = stats::full_analysis(&hlda);

    report.metrics = serde_json::json!({
        "simulation_time_ns": sim_time_ns,
        "hlda_mean": hlda.iter().sum::<f64>() / hlda.len() as f64,
        "fold_events": fold_events,
        "unfold_events": unfold_events,
        "delta_g_fold_kj": delta_g,
        "n_frames": cv.n_frames,
        "hlda_statistics": {
            "autocorrelation_time": hlda_stats.autocorrelation_time,
            "statistical_inefficiency": hlda_stats.statistical_inefficiency,
            "effective_n": hlda_stats.effective_n,
            "sem_correlated": hlda_stats.sem_correlated,
            "equilibration_t0_fraction": hlda_stats.equilibration.t0_fraction,
            "production_fraction": hlda_stats.equilibration.production_fraction,
            "optimal_block_size": hlda_stats.optimal_block.optimal_block_size,
        },
    });

    report.compute_rates();
    report
}

/// Project 2D FES onto phi axis via Boltzmann integration over psi.
fn project_phi(fes_2d: &fes::Fes2D, kt: f64) -> fes::Fes1D {
    let nx = fes_2d.nbins_x;
    let ny = fes_2d.nbins_y;
    let mut fes_phi = vec![0.0_f64; nx];

    for i in 0..nx {
        let prob: f64 = (0..ny).map(|j| (-fes_2d.free_energy[i][j] / kt).exp()).sum();
        fes_phi[i] = -kt * prob.ln();
    }

    let min_val = fes_phi.iter().cloned().fold(f64::INFINITY, f64::min);
    for f in &mut fes_phi {
        *f -= min_val;
    }

    fes::Fes1D {
        grid: fes_2d.grid_x.clone(),
        free_energy: fes_phi,
        nbins: nx,
    }
}

/// Convert 2D HILLS to 1D (phi only) for stride analysis.
fn to_1d_hills_phi(hills_2d: &hills::Hills2D) -> hills::Hills1D {
    hills::Hills1D {
        time: hills_2d.time.clone(),
        centers: hills_2d.centers_x.clone(),
        sigmas: hills_2d.sigmas_x.clone(),
        heights: hills_2d.heights.clone(),
        biasfactor: hills_2d.biasfactor.clone(),
    }
}

/// Placeholder for ingest — will replace ingest.sh.
pub fn ingest_targets(root: &Path, _filter: Option<&str>) {
    eprintln!("TODO: Implement Rust-native ingestion (download + validate)");
    eprintln!("Root: {}", root.display());
}

/// Launch GROMACS+PLUMED simulation with proper process management.
pub fn run_target_simulation(target_dir: &Path) {
    let tpr = target_dir.join("inputs/topol.tpr");
    if !tpr.exists() {
        eprintln!("ERROR: No topol.tpr in {}/inputs/", target_dir.display());
        std::process::exit(1);
    }

    let plumed_dat = find_plumed_dat(target_dir);
    let plumed_dat = match plumed_dat {
        Some(p) => p,
        None => {
            eprintln!("ERROR: No PLUMED .dat file found in {}/plumed/", target_dir.display());
            std::process::exit(1);
        }
    };

    println!("Launching GROMACS+PLUMED:");
    println!("  TPR:    {}", tpr.display());
    println!("  PLUMED: {}", plumed_dat.display());

    let output_prefix = target_dir.join("output/production");

    let mut child = Command::new("gmx")
        .arg("mdrun")
        .args(["-s", &tpr.to_string_lossy()])
        .args(["-plumed", &plumed_dat.to_string_lossy()])
        .args(["-deffnm", &output_prefix.to_string_lossy()])
        .args(["-ntmpi", "1"])
        .args(["-ntomp", "8"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn gmx mdrun");

    println!("  PID:    {}", child.id());

    let status = child.wait().expect("Failed to wait for gmx mdrun");

    if status.success() {
        println!("\x1b[32m  COMPLETED\x1b[0m");
    } else {
        eprintln!("\x1b[31m  FAILED (exit code: {:?})\x1b[0m", status.code());
        std::process::exit(1);
    }
}

fn find_plumed_dat(target_dir: &Path) -> Option<std::path::PathBuf> {
    let plumed_dir = target_dir.join("plumed");
    if !plumed_dir.is_dir() {
        return None;
    }

    // Prefer plumed_gromacs.dat, then plumed.dat
    let preferred = ["plumed_gromacs.dat", "plumed.dat"];
    for name in &preferred {
        let p = plumed_dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }

    // Fall back to first .dat file
    std::fs::read_dir(&plumed_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().map(|e| e == "dat").unwrap_or(false))
}
