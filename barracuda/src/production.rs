// SPDX-License-Identifier: AGPL-3.0-only

//! Shared types and infrastructure for production lattice QCD binaries.
//!
//! Extracted from production_dynamical_mixed, production_mixed_pipeline, and
//! related binaries to reduce duplication and keep binaries under 1000 lines.
//!
//! # Contents
//!
//! - **MetaRow** — per-beta aggregate statistics from meta tables or trajectory logs
//! - **BetaResult** — per-beta measurement result with NPU stats
//! - **AttentionState** — NPU attention state machine (Green/Yellow/Red)
//! - **npu_worker** — 11-head NPU worker thread (NpuRequest, NpuResponse, spawn_npu_worker)
//! - **Helpers** — load_meta_table, load_trajectory_log_as_meta, plaquette_variance
//! - **ESN** — check_thermalization, predict_rejection, predict_beta_c,
//!   find_max_uncertainty_beta, build_training_data, bootstrap_esn_from_trajectory_log

pub mod beta_scan;
pub mod cortex_worker;
pub mod dynamical_bootstrap;
pub mod dynamical_summary;
pub mod mixed_summary;
pub mod npu_worker;
pub mod titan_validation;
pub mod titan_worker;

use crate::error::HotSpringError;
use crate::md::reservoir::NpuSimulator;
use crate::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;
use std::io::BufRead;

// ═══════════════════════════════════════════════════════════════════
//  Meta Table Row
// ═══════════════════════════════════════════════════════════════════

/// Per-beta aggregate statistics from a meta table or trajectory log.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MetaRow {
    /// Lattice size (one dimension of L⁴).
    pub lattice: usize,
    /// Coupling β.
    pub beta: f64,
    /// Fermion mass (if dynamical).
    pub mass: Option<f64>,
    /// Mode: "quenched", "dynamical", etc.
    pub mode: String,
    /// Mean plaquette.
    pub mean_plaq: f64,
    /// χ²-like variance measure.
    pub chi: f64,
    /// Acceptance rate.
    pub acceptance: f64,
    /// Mean CG iterations per trajectory.
    pub mean_cg_iters: f64,
    /// Wall time per trajectory in seconds.
    pub wall_s_per_traj: f64,
    /// Number of measurement trajectories.
    pub n_meas: usize,
}

/// Load meta table from path. Tries MetaRow JSONL first, then aggregates
/// per-trajectory JSONL into per-beta MetaRows. Streams line-by-line to avoid
/// buffering entire files in memory.
pub fn load_meta_table(path: &str) -> Result<Vec<MetaRow>, HotSpringError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut meta: Vec<MetaRow> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if let Ok(row) = serde_json::from_str::<MetaRow>(&line) {
            meta.push(row);
        }
    }

    if !meta.is_empty() {
        return Ok(meta);
    }
    load_trajectory_log_as_meta(path)
}

/// Aggregate per-trajectory JSONL into per-beta MetaRows.
/// Lets the NPU bootstrap from raw production logs. Streams line-by-line.
fn load_trajectory_log_as_meta(path: &str) -> Result<Vec<MetaRow>, HotSpringError> {
    use std::collections::BTreeMap;

    #[derive(serde::Deserialize)]
    struct TrajRecord {
        beta: f64,
        #[serde(default)]
        mass: f64,
        plaquette: f64,
        accepted: bool,
        #[serde(default)]
        cg_iters: usize,
        #[serde(default)]
        phase: String,
    }

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut records: Vec<TrajRecord> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if let Ok(rec) = serde_json::from_str::<TrajRecord>(&line) {
            records.push(rec);
        }
    }

    if records.is_empty() {
        return Err(HotSpringError::DataLoad(format!(
            "no parseable trajectory records in {path}"
        )));
    }

    let mut by_beta: BTreeMap<i64, Vec<&TrajRecord>> = BTreeMap::new();
    for r in &records {
        let key = (r.beta * 10000.0).round() as i64;
        by_beta.entry(key).or_default().push(r);
    }

    let mut rows = Vec::new();
    for group in by_beta.values() {
        let meas: Vec<&&TrajRecord> = group.iter().filter(|r| r.phase == "measurement").collect();
        let (plaqs, accepts, cgs): (Vec<f64>, Vec<bool>, Vec<usize>) = if meas.is_empty() {
            (
                group.iter().map(|r| r.plaquette).collect(),
                group.iter().map(|r| r.accepted).collect(),
                group.iter().map(|r| r.cg_iters).collect(),
            )
        } else {
            (
                meas.iter().map(|r| r.plaquette).collect(),
                meas.iter().map(|r| r.accepted).collect(),
                meas.iter().map(|r| r.cg_iters).collect(),
            )
        };

        if plaqs.is_empty() {
            continue;
        }

        let n = plaqs.len();
        let mean_plaq = plaqs.iter().sum::<f64>() / n as f64;
        let variance = plaqs.iter().map(|p| (p - mean_plaq).powi(2)).sum::<f64>() / n as f64;
        let chi = variance * (n as f64);
        let acceptance = accepts.iter().filter(|&&a| a).count() as f64 / n as f64;
        let mean_cg = if cgs.iter().any(|&c| c > 0) {
            let nonzero: Vec<f64> = cgs.iter().filter(|&&c| c > 0).map(|&c| c as f64).collect();
            nonzero.iter().sum::<f64>() / nonzero.len() as f64
        } else {
            0.0
        };

        rows.push(MetaRow {
            lattice: 8,
            beta: group[0].beta,
            mass: Some(group[0].mass),
            mode: "dynamical".into(),
            mean_plaq,
            chi,
            acceptance,
            mean_cg_iters: mean_cg,
            wall_s_per_traj: 0.0,
            n_meas: n,
        });
    }

    println!(
        "  Bootstrap: aggregated {path} → {} beta points from {} trajectories",
        rows.len(),
        records.len()
    );
    Ok(rows)
}

// ═══════════════════════════════════════════════════════════════════
//  Attention State Machine
// ═══════════════════════════════════════════════════════════════════

/// NPU attention state for CG residual monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionState {
    /// Normal operation.
    Green,
    /// Elevated monitoring.
    Yellow,
    /// Critical — consider intervention.
    Red,
}

impl std::fmt::Display for AttentionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Green => write!(f, "GREEN"),
            Self::Yellow => write!(f, "YELLOW"),
            Self::Red => write!(f, "RED"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Beta Result
// ═══════════════════════════════════════════════════════════════════

/// Per-beta measurement result with NPU statistics.
#[derive(Clone, Debug)]
pub struct BetaResult {
    /// Coupling β value.
    pub beta: f64,
    /// Fermion mass (0 for quenched).
    pub mass: f64,
    /// Mean plaquette over measurement trajectories.
    pub mean_plaq: f64,
    /// Standard deviation of plaquette.
    pub std_plaq: f64,
    /// Average Polyakov loop magnitude.
    pub polyakov: f64,
    /// Plaquette susceptibility (variance × volume).
    pub susceptibility: f64,
    /// Mean action density.
    pub action_density: f64,
    /// HMC acceptance rate.
    pub acceptance: f64,
    /// Mean CG iterations per solve.
    pub mean_cg_iters: f64,
    /// Number of measurement trajectories.
    pub n_traj: usize,
    /// Total wall time in seconds.
    pub wall_s: f64,
    /// Detected thermodynamic phase.
    pub phase: &'static str,
    /// Thermalization trajectories actually run.
    pub therm_used: usize,
    /// Thermalization trajectory budget.
    pub therm_budget: usize,
    /// MD step size used.
    pub dt_used: f64,
    /// MD steps per trajectory.
    pub n_md_used: usize,
    /// NPU early-exit during thermalization.
    pub npu_therm_early_exit: bool,
    /// NPU quenched trajectory budget.
    pub npu_quenched_budget: usize,
    /// NPU quenched trajectories used.
    pub npu_quenched_used: usize,
    /// NPU early-exit during quenched phase.
    pub npu_quenched_early_exit: bool,
    /// NPU rejections predicted.
    pub npu_reject_predictions: usize,
    /// NPU rejections correctly predicted.
    pub npu_reject_correct: usize,
    /// NPU anomaly detections.
    pub npu_anomalies: usize,
    /// CG convergence check interval for NPU screening.
    pub npu_cg_check_interval: usize,
}

impl Default for BetaResult {
    fn default() -> Self {
        Self {
            beta: 0.0,
            mass: 0.0,
            mean_plaq: 0.0,
            std_plaq: 0.0,
            polyakov: 0.0,
            susceptibility: 0.0,
            action_density: 0.0,
            acceptance: 0.0,
            mean_cg_iters: 0.0,
            n_traj: 0,
            wall_s: 0.0,
            phase: "transition",
            therm_used: 0,
            therm_budget: 0,
            dt_used: 0.0,
            n_md_used: 0,
            npu_therm_early_exit: false,
            npu_quenched_budget: 0,
            npu_quenched_used: 0,
            npu_quenched_early_exit: false,
            npu_reject_predictions: 0,
            npu_reject_correct: 0,
            npu_anomalies: 0,
            npu_cg_check_interval: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Plaquette Variance
// ═══════════════════════════════════════════════════════════════════

/// Compute sample variance of plaquette history.
pub fn plaquette_variance(history: &[f64]) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    history.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (history.len() - 1) as f64
}

// ═══════════════════════════════════════════════════════════════════
//  Thermalization Check
// ═══════════════════════════════════════════════════════════════════

/// Statistical convergence check on plaquette window.
/// Uses variance-ratio and drift tests — mimics what the ESN therm detector learned.
pub fn check_thermalization(plaq_window: &[f64], _beta: f64) -> bool {
    if plaq_window.len() < 10 {
        return false;
    }
    let n = plaq_window.len();
    let mean = plaq_window.iter().sum::<f64>() / n as f64;
    let var = plaq_window.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    let half = n / 2;
    let mean_first = plaq_window[..half].iter().sum::<f64>() / half as f64;
    let mean_second = plaq_window[half..].iter().sum::<f64>() / (n - half) as f64;
    let drift = (mean_second - mean_first).abs();

    let relative_var = if mean.abs() > 1e-12 {
        var.sqrt() / mean.abs()
    } else {
        var.sqrt()
    };

    relative_var < 0.02 && drift < 0.005
}

// ═══════════════════════════════════════════════════════════════════
//  Rejection Prediction
// ═══════════════════════════════════════════════════════════════════

/// Predict trajectory rejection from observables.
/// Uses empirical heuristics: large |ΔH| and low acceptance rate predict rejection.
pub fn predict_rejection(
    _beta: f64,
    _plaquette: f64,
    _action_density: f64,
    delta_h: f64,
    acceptance_rate: f64,
) -> (bool, f64) {
    let dh_mag = delta_h.abs();
    let rejection_score = if delta_h > 0.0 {
        1.0 - (-dh_mag).exp()
    } else {
        0.0
    };

    let rate_factor = if acceptance_rate < 0.3 {
        1.2
    } else if acceptance_rate < 0.5 {
        1.0
    } else {
        0.8
    };

    let confidence = (rejection_score * rate_factor).clamp(0.0, 1.0);
    let likely_rejected = confidence > 0.8;

    (likely_rejected, confidence)
}

// ═══════════════════════════════════════════════════════════════════
//  ESN Training Data
// ═══════════════════════════════════════════════════════════════════

/// Build ESN training data from accumulated β-scan results.
/// Features: [β_norm, plaquette, polyakov, susceptibility_norm]
/// Targets: [phase (0=confined, 1=deconfined), beta_c_proximity]
pub fn build_training_data(results: &[BetaResult]) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();
    let seq_len = 10;

    for r in results {
        let beta_norm = (r.beta - 5.0) / 2.0;
        let phase = if r.beta > KNOWN_BETA_C { 1.0 } else { 0.0 };
        let proximity = (-(r.beta - KNOWN_BETA_C).powi(2) / 0.1).exp();

        let seq: Vec<Vec<f64>> = (0..seq_len)
            .map(|j| {
                let noise = 0.005 * ((j as f64) * 0.7).sin();
                vec![
                    beta_norm,
                    r.mean_plaq + noise * r.std_plaq,
                    r.polyakov + noise * 0.01,
                    r.susceptibility / 1000.0,
                ]
            })
            .collect();
        seqs.push(seq);
        targets.push(vec![phase, proximity]);
    }

    (seqs, targets)
}

// ═══════════════════════════════════════════════════════════════════
//  Beta_c Prediction
// ═══════════════════════════════════════════════════════════════════

/// Predict β_c by scanning NPU predictions and finding phase boundary.
pub fn predict_beta_c(npu: &mut NpuSimulator) -> f64 {
    let n_scan = 100;
    let mut best_beta = KNOWN_BETA_C;
    let mut best_uncertainty = 0.0_f64;

    for i in 0..n_scan {
        let beta = 5.0 + 2.0 * (i as f64) / (n_scan as f64 - 1.0);
        let beta_norm = (beta - 5.0) / 2.0;
        let plaq_est = 0.35 + 0.25 * (beta - 4.5) / 2.5;
        let poly_est = if beta > 5.7 { 0.3 } else { 0.05 };
        let chi_est = if (beta - KNOWN_BETA_C).abs() < 0.2 {
            500.0
        } else {
            50.0
        };

        let seq: Vec<Vec<f64>> = vec![vec![beta_norm, plaq_est, poly_est, chi_est / 1000.0]; 10];
        let pred = npu.predict(&seq);
        if pred.len() >= 2 {
            let uncertainty = pred[1];
            if uncertainty > best_uncertainty {
                best_uncertainty = uncertainty;
                best_beta = beta;
            }
        } else if !pred.is_empty() {
            let phase_pred = pred[0];
            let u = 1.0 - (phase_pred - 0.5).abs() * 2.0;
            if u > best_uncertainty {
                best_uncertainty = u;
                best_beta = beta;
            }
        }
    }

    best_beta
}

/// Find β with maximum NPU uncertainty among unmeasured regions.
pub fn find_max_uncertainty_beta(
    npu: &mut NpuSimulator,
    measured: &[f64],
    beta_min: f64,
    beta_max: f64,
    n_candidates: usize,
) -> f64 {
    let mut best_beta = f64::NAN;
    let mut best_score = 0.0_f64;

    for i in 0..n_candidates {
        let beta = beta_min + (beta_max - beta_min) * (i as f64) / (n_candidates as f64 - 1.0);

        if measured.iter().any(|&m| (m - beta).abs() < 0.08) {
            continue;
        }

        let beta_norm = (beta - 5.0) / 2.0;
        let plaq_est = 0.35 + 0.25 * (beta - 4.5) / 2.5;
        let poly_est = if beta > 5.7 { 0.3 } else { 0.05 };
        let chi_est = if (beta - KNOWN_BETA_C).abs() < 0.2 {
            500.0
        } else {
            50.0
        };

        let seq: Vec<Vec<f64>> = vec![vec![beta_norm, plaq_est, poly_est, chi_est / 1000.0]; 10];
        let pred = npu.predict(&seq);

        let uncertainty = if pred.len() >= 2 {
            pred[1]
        } else if !pred.is_empty() {
            1.0 - (pred[0] - 0.5).abs() * 2.0
        } else {
            0.0
        };

        let proximity_bonus = (-(beta - KNOWN_BETA_C).powi(2) / 0.5).exp() * 0.3;
        let score = uncertainty + proximity_bonus;

        if score > best_score {
            best_score = score;
            best_beta = beta;
        }
    }

    best_beta
}

// ═══════════════════════════════════════════════════════════════════
//  Bootstrap from Trajectory Log
// ═══════════════════════════════════════════════════════════════════

/// Bootstrap ESN from a previous run's trajectory log.
/// Streams JSONL lines, extracts per-beta aggregates, trains ESN.
pub fn bootstrap_esn_from_trajectory_log(
    path: &str,
    make_esn: &dyn Fn(u64, &[BetaResult]) -> Option<NpuSimulator>,
    npu: &mut Option<NpuSimulator>,
) -> Result<(usize, f64), HotSpringError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut beta_data: std::collections::BTreeMap<String, Vec<(f64, bool)>> =
        std::collections::BTreeMap::new();
    let mut n_lines = 0usize;

    for line in reader.lines() {
        let line = line?;
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if v.get("is_therm") == Some(&serde_json::Value::Bool(true)) {
            continue;
        }
        let Some(beta) = v.get("beta").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let Some(plaq) = v.get("plaquette").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let accepted = v
            .get("accepted")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        let key = format!("{beta:.4}");
        beta_data.entry(key).or_default().push((plaq, accepted));
        n_lines += 1;
    }

    if beta_data.is_empty() {
        return Ok((0, KNOWN_BETA_C));
    }

    let results: Vec<BetaResult> = beta_data
        .into_iter()
        .map(|(key, entries)| {
            let beta: f64 = key.parse().unwrap_or(KNOWN_BETA_C);
            let n = entries.len();
            let plaqs: Vec<f64> = entries.iter().map(|(p, _)| *p).collect();
            let mean_plaq = plaqs.iter().sum::<f64>() / n as f64;
            let var_plaq =
                plaqs.iter().map(|p| (p - mean_plaq).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
            let n_accepted = entries.iter().filter(|(_, a)| *a).count();

            BetaResult {
                beta,
                mean_plaq,
                std_plaq: var_plaq.sqrt(),
                polyakov: 0.0,
                susceptibility: var_plaq * 1048576.0,
                action_density: 6.0 * (1.0 - mean_plaq),
                acceptance: n_accepted as f64 / n as f64,
                n_traj: n,
                phase: if beta < KNOWN_BETA_C - 0.1 {
                    "confined"
                } else if beta > KNOWN_BETA_C + 0.1 {
                    "deconfined"
                } else {
                    "transition"
                },
                ..Default::default()
            }
        })
        .collect();

    let n_betas = results.len();
    if let Some(new_npu) = make_esn(42, &results) {
        *npu = Some(new_npu);
    }

    let beta_c = if let Some(ref mut n) = npu {
        predict_beta_c(n)
    } else {
        KNOWN_BETA_C
    };

    Ok((n_betas * n_lines / n_betas.max(1), beta_c))
}
