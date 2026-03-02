// SPDX-License-Identifier: AGPL-3.0-only

//! NPU experiment campaign: trajectory generation, dataset builders, evaluators,
//! placement strategies, and characterization.
//!
//! Extracted from `npu_experiment_campaign` binary for reuse and to keep the
//! binary under 1000 lines. Supports 6 pipeline placements (A–F), thermalization
//! detection, rejection prediction, and 6-output multi-task NPU evaluation.

mod placements;

use crate::error::HotSpringError;
use crate::lattice::hmc::{self, HmcConfig};
use crate::lattice::wilson::Lattice;
use crate::md::reservoir::{EchoStateNetwork, EsnConfig, ExportedWeights, NpuSimulator};
use crate::production::plaquette_variance;
use crate::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;
use std::io::Write;
use std::time::Instant;

/// Window size for thermalization detector input.
pub const WINDOW_SIZE: usize = 10;
/// Number of thermalization trajectories per β-point.
pub const N_THERM: usize = 50;
/// Number of measurement trajectories per β-point.
pub const N_MEAS: usize = 100;
/// Total trajectories per β-point (thermalization + measurement).
pub const N_TOTAL: usize = N_THERM + N_MEAS;

/// Per-trajectory record from HMC simulation.
#[derive(Clone, Debug)]
pub struct TrajectoryRecord {
    /// Coupling β.
    pub beta: f64,
    /// Trajectory index within the run.
    pub traj_idx: usize,
    /// Whether this trajectory is in the thermalization phase.
    pub is_therm: bool,
    /// Whether the HMC step was accepted.
    pub accepted: bool,
    /// Plaquette value.
    pub plaquette: f64,
    /// Polyakov loop magnitude (real part).
    pub polyakov_re: f64,
    /// Hamiltonian change ΔH.
    pub delta_h: f64,
    /// CG iterations (0 if not tracked).
    pub cg_iters: usize,
    /// Plaquette variance over recent history.
    pub plaquette_var: f64,
    /// Polyakov loop phase.
    pub polyakov_phase: f64,
    /// Action density (6 × (1 − plaquette)).
    pub action_density: f64,
    /// Wall time in microseconds.
    pub wall_us: u64,
}

/// Result of a single placement experiment.
#[derive(Clone, Debug)]
pub struct PlacementResult {
    /// Placement name (e.g. "A: Pre-thermalization").
    pub name: String,
    /// Wall time in milliseconds.
    pub wall_ms: f64,
    /// Accuracy (0–1).
    pub accuracy: f64,
    /// Number of trajectories saved by early exit.
    pub traj_saved: usize,
}

/// Metrics from 6-output multi-task model evaluation.
#[derive(Clone, Debug)]
pub struct MultiOutputMetrics {
    /// Phase classification accuracy.
    pub phase_acc: f64,
    /// Mean absolute error on β_c prediction.
    pub beta_c_err: f64,
    /// Thermalization classification accuracy.
    pub therm_acc: f64,
    /// Mean absolute error on acceptance prediction.
    pub accept_err: f64,
    /// Anomaly detection AUC.
    pub anomaly_auc: f64,
    /// Mean absolute error on CG iteration prediction.
    pub cg_err: f64,
}

/// NPU behavior characterization results.
#[derive(Clone, Debug)]
pub struct CharacterizationResults {
    /// Latency p50 in microseconds.
    pub latency_p50_us: f64,
    /// Latency p95 in microseconds.
    pub latency_p95_us: f64,
    /// Latency p99 in microseconds.
    pub latency_p99_us: f64,
    /// Mean latency in microseconds.
    pub latency_mean_us: f64,
    /// Throughput (inferences/ms) per batch size.
    pub batch_throughput: Vec<(usize, f64)>,
    /// Weight mutation time in milliseconds.
    pub mutation_time_ms: f64,
    /// Whether predictions changed after weight mutation.
    pub mutation_changed_predictions: bool,
    /// Drift measurements over sequential batches.
    pub drift_measurements: Vec<f64>,
    /// Maximum observed drift.
    pub max_drift: f64,
    /// Accuracy vs training set size (n_train, accuracy %).
    pub accuracy_vs_training_size: Vec<(usize, f64)>,
}

/// Generate per-trajectory training data via CPU HMC on a 4⁴ lattice.
pub fn generate_trajectory_data(betas: &[f64]) -> Vec<TrajectoryRecord> {
    let mut records = Vec::with_capacity(betas.len() * N_TOTAL);

    for (bi, &beta) in betas.iter().enumerate() {
        let mut lat = Lattice::hot_start([4, 4, 4, 4], beta, 42 + bi as u64);
        let mut config = HmcConfig {
            n_md_steps: 20,
            dt: 0.02,
            seed: 1000 + bi as u64 * 100,
            ..Default::default()
        };

        let mut plaq_history: Vec<f64> = Vec::with_capacity(32);

        for traj in 0..N_TOTAL {
            let traj_start = Instant::now();
            let result = hmc::hmc_trajectory(&mut lat, &mut config);
            let wall_us = traj_start.elapsed().as_micros() as u64;

            let plaq = result.plaquette;
            plaq_history.push(plaq);
            if plaq_history.len() > 32 {
                plaq_history.remove(0);
            }

            let plaq_var = plaquette_variance(&plaq_history);
            let poly = lat.polyakov_loop([0, 0, 0]);
            let poly_mag = (poly.re * poly.re + poly.im * poly.im).sqrt();
            let poly_phase = poly.im.atan2(poly.re);

            records.push(TrajectoryRecord {
                beta,
                traj_idx: traj,
                is_therm: traj < N_THERM,
                accepted: result.accepted,
                plaquette: plaq,
                polyakov_re: poly_mag,
                delta_h: result.delta_h,
                cg_iters: 0,
                plaquette_var: plaq_var,
                polyakov_phase: poly_phase,
                action_density: 6.0 * (1.0 - plaq),
                wall_us,
            });
        }
    }

    records
}

/// Build thermalization detector dataset from trajectory records.
pub fn build_thermalization_dataset(
    records: &[TrajectoryRecord],
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();

    let betas: Vec<f64> = {
        let mut v: Vec<f64> = records.iter().map(|r| r.beta).collect();
        v.dedup();
        v
    };

    for &beta in &betas {
        let beta_records: Vec<&TrajectoryRecord> = records
            .iter()
            .filter(|r| (r.beta - beta).abs() < 1e-10)
            .collect();

        let plaq_series: Vec<f64> = beta_records.iter().map(|r| r.plaquette).collect();

        let meas_plaqs: Vec<f64> = beta_records
            .iter()
            .filter(|r| !r.is_therm)
            .map(|r| r.plaquette)
            .collect();
        let equilibrium_mean = if meas_plaqs.is_empty() {
            plaq_series.last().copied().unwrap_or(0.5)
        } else {
            meas_plaqs.iter().sum::<f64>() / meas_plaqs.len() as f64
        };
        let equilibrium_var = if meas_plaqs.len() > 1 {
            meas_plaqs
                .iter()
                .map(|p| (p - equilibrium_mean).powi(2))
                .sum::<f64>()
                / (meas_plaqs.len() - 1) as f64
        } else {
            1e-6
        };
        let therm_threshold = 3.0 * equilibrium_var.sqrt();

        for start in 0..plaq_series.len().saturating_sub(WINDOW_SIZE) {
            let window: Vec<Vec<f64>> = plaq_series[start..start + WINDOW_SIZE]
                .iter()
                .map(|&p| vec![p])
                .collect();

            let window_mean =
                plaq_series[start..start + WINDOW_SIZE].iter().sum::<f64>() / WINDOW_SIZE as f64;
            let is_thermalized = (window_mean - equilibrium_mean).abs() < therm_threshold;

            seqs.push(window);
            targets.push(vec![if is_thermalized { 1.0 } else { 0.0 }]);
        }
    }

    (seqs, targets)
}

/// Build rejection predictor dataset from trajectory records.
pub fn build_rejection_dataset(
    records: &[TrajectoryRecord],
    betas: &[f64],
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();

    let dt = 0.02;
    let n_md = 20;

    for record in records.iter().filter(|r| !r.is_therm) {
        let beta_norm = (record.beta - betas[0]) / (betas[betas.len() - 1] - betas[0]);
        let features = vec![
            beta_norm,
            record.plaquette,
            record.action_density,
            n_md as f64 / 50.0,
            dt * 10.0,
        ];
        seqs.push(vec![features]);
        targets.push(vec![if record.accepted { 1.0 } else { 0.0 }]);
    }

    (seqs, targets)
}

/// Build 6-output multi-task dataset from trajectory records.
pub fn build_multi_output_dataset(
    records: &[TrajectoryRecord],
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();

    let betas: Vec<f64> = {
        let mut v: Vec<f64> = records.iter().map(|r| r.beta).collect();
        v.dedup();
        v
    };

    for &beta in &betas {
        let beta_records: Vec<&TrajectoryRecord> = records
            .iter()
            .filter(|r| (r.beta - beta).abs() < 1e-10 && !r.is_therm)
            .collect();

        if beta_records.is_empty() {
            continue;
        }

        let mut running_acc = 0.0;
        let mut acc_count = 0;
        let mut feature_seq = Vec::new();

        for r in &beta_records {
            acc_count += 1;
            if r.accepted {
                running_acc += 1.0;
            }
            let acc_rate = running_acc / acc_count as f64;

            feature_seq.push(vec![
                r.plaquette,
                r.plaquette_var,
                r.polyakov_re,
                r.polyakov_phase,
                r.action_density,
                acc_rate,
                r.delta_h.abs(),
                r.cg_iters as f64,
            ]);
        }

        let phase = if beta > KNOWN_BETA_C { 1.0 } else { 0.0 };
        let is_near_transition = if (beta - KNOWN_BETA_C).abs() < 0.3 {
            1.0
        } else {
            0.0
        };
        let mean_acceptance = running_acc / acc_count.max(1) as f64;
        let anomaly_score = if (beta - KNOWN_BETA_C).abs() < 0.2 {
            0.5
        } else {
            0.0
        };
        let mean_cg =
            beta_records.iter().map(|r| r.cg_iters as f64).sum::<f64>() / beta_records.len() as f64;

        targets.push(vec![
            phase,
            (KNOWN_BETA_C - 5.0) / 1.5,
            is_near_transition,
            mean_acceptance,
            anomaly_score,
            mean_cg,
        ]);
        seqs.push(feature_seq);
    }

    (seqs, targets)
}

/// Evaluate thermalization detector ESN on test data.
/// Returns (accuracy, savings_fraction).
pub fn evaluate_thermalization_detector(
    esn: &mut EchoStateNetwork,
    test_seqs: &[Vec<Vec<f64>>],
    test_targets: &[Vec<f64>],
) -> (f64, f64) {
    let mut correct = 0;
    let mut total_therm_windows = 0;
    let mut early_therm_count = 0;

    for (seq, target) in test_seqs.iter().zip(test_targets.iter()) {
        let pred = esn.predict(seq).unwrap_or_else(|_| vec![0.0]);
        let pred_class = if pred[0] > 0.5 { 1.0 } else { 0.0 };
        if (pred_class - target[0]).abs() < 0.01 {
            correct += 1;
        }
        if target[0] < 0.5 {
            total_therm_windows += 1;
            if pred[0] > 0.5 {
                early_therm_count += 1;
            }
        }
    }

    let accuracy = correct as f64 / test_seqs.len().max(1) as f64;
    let savings = if total_therm_windows > 0 {
        early_therm_count as f64 / total_therm_windows as f64
    } else {
        0.0
    };

    (accuracy, savings)
}

/// Evaluate rejection predictor ESN on test data.
/// Returns (accuracy, early_abort_rate).
pub fn evaluate_rejection_predictor(
    esn: &mut EchoStateNetwork,
    test_seqs: &[Vec<Vec<f64>>],
    test_targets: &[Vec<f64>],
) -> (f64, f64) {
    let mut correct = 0;
    let mut rejected_total = 0;
    let mut rejected_predicted = 0;

    for (seq, target) in test_seqs.iter().zip(test_targets.iter()) {
        let pred = esn.predict(seq).unwrap_or_else(|_| vec![0.5]);
        let pred_class = if pred[0] > 0.5 { 1.0 } else { 0.0 };
        if (pred_class - target[0]).abs() < 0.01 {
            correct += 1;
        }
        if target[0] < 0.5 {
            rejected_total += 1;
            if pred[0] < 0.5 {
                rejected_predicted += 1;
            }
        }
    }

    let accuracy = correct as f64 / test_seqs.len().max(1) as f64;
    let early_abort = if rejected_total > 0 {
        rejected_predicted as f64 / rejected_total as f64
    } else {
        0.0
    };

    (accuracy, early_abort)
}

/// Evaluate 6-output multi-task ESN on test data.
pub fn evaluate_multi_output(
    esn: &mut EchoStateNetwork,
    test_seqs: &[Vec<Vec<f64>>],
    test_targets: &[Vec<f64>],
) -> MultiOutputMetrics {
    let mut phase_correct = 0;
    let mut beta_c_errs = Vec::new();
    let mut therm_correct = 0;
    let mut accept_errs = Vec::new();
    let mut anomaly_scores = Vec::new();
    let mut cg_errs = Vec::new();
    let total = test_seqs.len();

    for (seq, target) in test_seqs.iter().zip(test_targets.iter()) {
        let pred = match esn.predict(seq) {
            Ok(p) => p,
            Err(_) => vec![0.0; 6],
        };

        if (if pred[0] > 0.5 { 1.0 } else { 0.0 } - target[0]).abs() < 0.01 {
            phase_correct += 1;
        }
        beta_c_errs.push((pred[1] - target[1]).abs());
        if (if pred[2] > 0.5 { 1.0 } else { 0.0 } - target[2]).abs() < 0.01 {
            therm_correct += 1;
        }
        accept_errs.push((pred[3] - target[3]).abs());
        anomaly_scores.push((pred[4], target[4]));
        cg_errs.push((pred[5] - target[5]).abs());
    }

    let anomaly_auc = if anomaly_scores.is_empty() {
        0.5
    } else {
        let mut sorted = anomaly_scores.clone();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let n_pos = sorted.iter().filter(|(_, t)| *t > 0.1).count();
        let n_neg = sorted.len() - n_pos;
        if n_pos == 0 || n_neg == 0 {
            0.5
        } else {
            let mut auc = 0.0;
            let mut tp = 0.0;
            for (_, t) in &sorted {
                if *t > 0.1 {
                    tp += 1.0;
                } else {
                    auc += tp / n_pos as f64;
                }
            }
            auc / n_neg as f64
        }
    };

    MultiOutputMetrics {
        phase_acc: phase_correct as f64 / total.max(1) as f64,
        beta_c_err: beta_c_errs.iter().sum::<f64>() / beta_c_errs.len().max(1) as f64,
        therm_acc: therm_correct as f64 / total.max(1) as f64,
        accept_err: accept_errs.iter().sum::<f64>() / accept_errs.len().max(1) as f64,
        anomaly_auc,
        cg_err: cg_errs.iter().sum::<f64>() / cg_errs.len().max(1) as f64,
    }
}

/// Count unique β values in trajectory records.
pub fn count_unique_betas(records: &[TrajectoryRecord]) -> usize {
    let mut betas: Vec<u64> = records.iter().map(|r| r.beta.to_bits()).collect();
    betas.sort_unstable();
    betas.dedup();
    betas.len()
}

/// Split sequences and targets into train/test by fraction.
pub fn split_dataset(
    seqs: &[Vec<Vec<f64>>],
    targets: &[Vec<f64>],
    train_frac: f64,
) -> (
    (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
    (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
) {
    let n = seqs.len();
    let n_train = (n as f64 * train_frac) as usize;

    let train_seqs = seqs[..n_train].to_vec();
    let train_targets = targets[..n_train].to_vec();
    let test_seqs = seqs[n_train..].to_vec();
    let test_targets = targets[n_train..].to_vec();

    ((train_seqs, train_targets), (test_seqs, test_targets))
}

/// Run all 6 placement experiments (A–F).
pub fn run_placement_experiments(
    records: &[TrajectoryRecord],
    betas: &[f64],
    therm_esn: &mut EchoStateNetwork,
    reject_esn: &mut EchoStateNetwork,
    multi_esn: &mut EchoStateNetwork,
) -> Vec<PlacementResult> {
    vec![
        placements::placement_a_pre_thermalization(records, therm_esn),
        placements::placement_b_mid_trajectory(records, betas, reject_esn),
        placements::placement_c_post_trajectory(records, multi_esn),
        placements::placement_d_inter_beta(records, multi_esn),
        placements::placement_e_pre_run(records, multi_esn),
        placements::placement_f_combined(records, betas, therm_esn, reject_esn, multi_esn),
    ]
}

/// Characterize NPU behavior: latency, batch throughput, weight mutation, drift.
pub fn characterize_npu_behavior(
    test_seqs: &[Vec<Vec<f64>>],
    npu_sim: &mut NpuSimulator,
    _weights: &ExportedWeights,
    multi_esn: &EchoStateNetwork,
) -> Result<CharacterizationResults, HotSpringError> {
    let n_inferences = 1000;
    let mut latencies_us = Vec::with_capacity(n_inferences);

    let seq = if test_seqs.is_empty() {
        vec![vec![0.5; 8]; 5]
    } else {
        test_seqs[0].clone()
    };

    for _ in 0..n_inferences {
        let t = Instant::now();
        let _ = npu_sim.predict(&seq);
        latencies_us.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50 = latencies_us[n_inferences / 2];
    let p95 = latencies_us[n_inferences * 95 / 100];
    let p99 = latencies_us[n_inferences * 99 / 100];
    let mean = latencies_us.iter().sum::<f64>() / n_inferences as f64;

    let batch_sizes = [1, 2, 4, 8];
    let mut batch_throughput = Vec::new();

    for &batch_size in &batch_sizes {
        let batch_seqs: Vec<&Vec<Vec<f64>>> = test_seqs.iter().take(batch_size).collect();
        let n_reps = 200;
        let t = Instant::now();
        for _ in 0..n_reps {
            for s in &batch_seqs {
                let _ = npu_sim.predict(s);
            }
        }
        let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
        let total_inferences = n_reps * batch_seqs.len();
        let throughput = total_inferences as f64 / elapsed_ms;
        batch_throughput.push((batch_size, throughput));
    }

    let pred_before = npu_sim.predict(&seq);
    let mutation_start = Instant::now();
    let new_weights = multi_esn.export_weights().ok_or_else(|| {
        HotSpringError::InvalidOperation("ESN export_weights failed (untrained?)".into())
    })?;
    let mut npu_after = NpuSimulator::from_exported(&new_weights);
    let mutation_time_ms = mutation_start.elapsed().as_secs_f64() * 1000.0;
    let pred_after = npu_after.predict(&seq);
    let mutation_changed = pred_before
        .iter()
        .zip(pred_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);

    let mut drift_measurements = Vec::new();
    let reference = npu_sim.predict(&seq);
    for _ in 0..50 {
        let p = npu_sim.predict(&seq);
        let drift: f64 = p
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        drift_measurements.push(drift);
    }
    let max_drift = drift_measurements.iter().copied().fold(0.0, f64::max);

    let mut accuracy_vs_size = Vec::new();
    let (full_seqs, full_targets) = {
        let mut all_records = Vec::new();
        let beta_values: Vec<f64> = (0..12).map(|i| 5.0 + 1.5 * (i as f64) / 11.0).collect();
        for (bi, &beta) in beta_values.iter().enumerate() {
            let mut lat = Lattice::hot_start([4, 4, 4, 4], beta, 42 + bi as u64);
            let mut config = HmcConfig {
                n_md_steps: 20,
                dt: 0.02,
                seed: 1000 + bi as u64 * 100,
                ..Default::default()
            };
            for _ in 0..10 {
                hmc::hmc_trajectory(&mut lat, &mut config);
            }
            let mut acc_count = 0.0;
            let mut meas_seq = Vec::new();
            for t in 0..20 {
                let r = hmc::hmc_trajectory(&mut lat, &mut config);
                if r.accepted {
                    acc_count += 1.0;
                }
                let poly = lat.polyakov_loop([0, 0, 0]);
                meas_seq.push(vec![
                    r.plaquette,
                    0.001,
                    (poly.re * poly.re + poly.im * poly.im).sqrt(),
                    poly.im.atan2(poly.re),
                    6.0 * (1.0 - r.plaquette),
                    acc_count / (t + 1) as f64,
                    r.delta_h.abs(),
                    0.0,
                ]);
            }
            let phase = if beta > KNOWN_BETA_C { 1.0 } else { 0.0 };
            all_records.push((meas_seq, phase));
        }
        let seqs: Vec<Vec<Vec<f64>>> = all_records.iter().map(|(s, _)| s.clone()).collect();
        let tgts: Vec<f64> = all_records.iter().map(|(_, p)| *p).collect();
        (seqs, tgts)
    };

    for n_train in [2, 4, 6, 8, 10] {
        if n_train > full_seqs.len() {
            break;
        }
        let train_s = &full_seqs[..n_train];
        let train_t: Vec<Vec<f64>> = full_targets[..n_train].iter().map(|&p| vec![p]).collect();
        let test_s = &full_seqs[n_train..];
        let test_t = &full_targets[n_train..];

        let cfg = EsnConfig {
            input_size: 8,
            reservoir_size: 50,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-2,
            seed: 42,
        };
        let mut esn = EchoStateNetwork::new(cfg);
        esn.train(train_s, &train_t);

        let w = esn.export_weights().ok_or_else(|| {
            HotSpringError::InvalidOperation("ESN export_weights failed (untrained?)".into())
        })?;
        let mut sim = NpuSimulator::from_exported(&w);
        let mut correct = 0;
        for (s, &t) in test_s.iter().zip(test_t.iter()) {
            let pred = sim.predict(s);
            let pred_class = if pred[0] > 0.5 { 1.0 } else { 0.0 };
            if (pred_class - t).abs() < 0.01 {
                correct += 1;
            }
        }
        let acc = correct as f64 / test_s.len().max(1) as f64 * 100.0;
        accuracy_vs_size.push((n_train, acc));
    }

    Ok(CharacterizationResults {
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        latency_mean_us: mean,
        batch_throughput,
        mutation_time_ms,
        mutation_changed_predictions: mutation_changed,
        drift_measurements,
        max_drift,
        accuracy_vs_training_size: accuracy_vs_size,
    })
}

/// Write campaign summary and placement results to JSONL.
pub fn write_jsonl_summary(
    records: &[TrajectoryRecord],
    placements: &[PlacementResult],
    char_results: &CharacterizationResults,
    therm_accuracy: f64,
    therm_savings: f64,
    reject_accuracy: f64,
    multi_metrics: &MultiOutputMetrics,
) {
    let path = std::env::temp_dir()
        .join("hotspring-runs")
        .join("v0614")
        .join("npu_campaign_results.jsonl");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let Ok(f) = std::fs::File::create(&path) else {
        eprintln!("  Cannot write results to {}", path.display());
        return;
    };
    let mut w = std::io::BufWriter::new(f);

    let summary = serde_json::json!({
        "type": "campaign_summary",
        "n_trajectories": records.len(),
        "n_betas": count_unique_betas(records),
        "therm_detector": {
            "accuracy": therm_accuracy,
            "savings_fraction": therm_savings,
            "projected_hours_saved": therm_savings * 5.1,
        },
        "reject_predictor": {
            "accuracy": reject_accuracy,
        },
        "multi_output": {
            "phase_accuracy": multi_metrics.phase_acc,
            "beta_c_error": multi_metrics.beta_c_err,
            "therm_accuracy": multi_metrics.therm_acc,
            "accept_error": multi_metrics.accept_err,
            "anomaly_auc": multi_metrics.anomaly_auc,
            "cg_error": multi_metrics.cg_err,
        },
        "characterization": {
            "latency_p50_us": char_results.latency_p50_us,
            "latency_p95_us": char_results.latency_p95_us,
            "latency_p99_us": char_results.latency_p99_us,
            "latency_mean_us": char_results.latency_mean_us,
            "mutation_time_ms": char_results.mutation_time_ms,
            "max_drift": char_results.max_drift,
            "batch_throughput": char_results.batch_throughput.iter()
                .map(|(b, t)| serde_json::json!({"batch": b, "inferences_per_ms": t}))
                .collect::<Vec<_>>(),
        },
    });
    writeln!(w, "{summary}").ok();

    for p in placements {
        let line = serde_json::json!({
            "type": "placement_result",
            "name": p.name,
            "wall_ms": p.wall_ms,
            "accuracy": p.accuracy,
            "traj_saved": p.traj_saved,
        });
        writeln!(w, "{line}").ok();
    }

    w.flush().ok();
    println!("  Results written to: {}", path.display());
}
