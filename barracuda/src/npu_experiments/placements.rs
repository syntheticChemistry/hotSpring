// SPDX-License-Identifier: AGPL-3.0-only

//! Placement strategies A–F for NPU pipeline experiments.

use crate::md::reservoir::EchoStateNetwork;
use crate::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;
use std::time::Instant;

use super::{build_multi_output_dataset, PlacementResult, TrajectoryRecord, N_THERM, WINDOW_SIZE};

/// Placement A: Pre-thermalization early exit.
pub(super) fn placement_a_pre_thermalization(
    records: &[TrajectoryRecord],
    therm_esn: &mut EchoStateNetwork,
) -> PlacementResult {
    let start = Instant::now();
    let mut saved = 0;
    let mut correct_phases = 0;
    let mut total_tested = 0;

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

        let plaqs: Vec<f64> = beta_records.iter().map(|r| r.plaquette).collect();

        for start_idx in 0..plaqs.len().saturating_sub(WINDOW_SIZE) {
            if start_idx >= N_THERM {
                break;
            }
            let window: Vec<Vec<f64>> = plaqs[start_idx..start_idx + WINDOW_SIZE]
                .iter()
                .map(|&p| vec![p])
                .collect();

            let pred = therm_esn.predict(&window).unwrap_or_else(|_| vec![0.0]);
            if pred[0] > 0.5 && start_idx + WINDOW_SIZE < N_THERM {
                saved += N_THERM - (start_idx + WINDOW_SIZE);
                break;
            }
        }

        let meas_records: Vec<&TrajectoryRecord> = beta_records
            .iter()
            .filter(|r| !r.is_therm)
            .copied()
            .collect();
        if !meas_records.is_empty() {
            total_tested += 1;
            let is_deconfined = beta > KNOWN_BETA_C;
            let last_plaq = meas_records[meas_records.len() - 1].plaquette;
            let pred_deconfined = last_plaq > 0.5;
            if is_deconfined == pred_deconfined {
                correct_phases += 1;
            }
        }
    }

    PlacementResult {
        name: "A: Pre-thermalization".into(),
        wall_ms: start.elapsed().as_secs_f64() * 1000.0,
        accuracy: correct_phases as f64 / total_tested.max(1) as f64,
        traj_saved: saved,
    }
}

/// Placement B: Mid-trajectory early exit on rejection prediction.
pub(super) fn placement_b_mid_trajectory(
    records: &[TrajectoryRecord],
    betas: &[f64],
    reject_esn: &mut EchoStateNetwork,
) -> PlacementResult {
    let start = Instant::now();
    let mut saved = 0;
    let mut correct = 0;
    let meas_records: Vec<&TrajectoryRecord> = records.iter().filter(|r| !r.is_therm).collect();
    let total = meas_records.len();

    let dt = 0.02;
    let n_md = 20;

    for r in &meas_records {
        let beta_norm = (r.beta - betas[0]) / (betas[betas.len() - 1] - betas[0]);
        let features = vec![
            beta_norm,
            r.plaquette,
            r.action_density,
            n_md as f64 / 50.0,
            dt * 10.0,
        ];
        let pred = reject_esn
            .predict(&[features])
            .unwrap_or_else(|_| vec![0.5]);
        let pred_accepted = pred[0] > 0.5;
        let actual_accepted = r.accepted;

        if pred_accepted == actual_accepted {
            correct += 1;
        }
        if !pred_accepted && !actual_accepted {
            saved += 1;
        }
    }

    PlacementResult {
        name: "B: Mid-trajectory exit".into(),
        wall_ms: start.elapsed().as_secs_f64() * 1000.0,
        accuracy: correct as f64 / total.max(1) as f64,
        traj_saved: saved,
    }
}

/// Placement C: Post-trajectory baseline (no early exit).
pub(super) fn placement_c_post_trajectory(
    records: &[TrajectoryRecord],
    multi_esn: &mut EchoStateNetwork,
) -> PlacementResult {
    let start = Instant::now();
    let (seqs, targets) = build_multi_output_dataset(records);
    let mut correct = 0;

    for (seq, target) in seqs.iter().zip(targets.iter()) {
        let pred = multi_esn.predict(seq).unwrap_or_else(|_| vec![0.0; 6]);
        let pred_phase = if pred[0] > 0.5 { 1.0 } else { 0.0 };
        if (pred_phase - target[0]).abs() < 0.01 {
            correct += 1;
        }
    }

    PlacementResult {
        name: "C: Post-trajectory (baseline)".into(),
        wall_ms: start.elapsed().as_secs_f64() * 1000.0,
        accuracy: correct as f64 / seqs.len().max(1) as f64,
        traj_saved: 0,
    }
}

/// Placement D: Inter-β steering.
pub(super) fn placement_d_inter_beta(
    records: &[TrajectoryRecord],
    multi_esn: &mut EchoStateNetwork,
) -> PlacementResult {
    let start = Instant::now();

    let betas: Vec<f64> = {
        let mut v: Vec<f64> = records.iter().map(|r| r.beta).collect();
        v.dedup();
        v
    };

    let mut steered_correctly = 0;
    let mut total_steered = 0;

    for window in betas.windows(2) {
        let beta_lo = window[0];
        let beta_hi = window[1];

        let lo_records: Vec<&TrajectoryRecord> = records
            .iter()
            .filter(|r| (r.beta - beta_lo).abs() < 1e-10 && !r.is_therm)
            .collect();

        if lo_records.len() < 5 {
            continue;
        }

        let mut acc_count = 0.0;
        let mut acc_n = 0;
        let feature_seq: Vec<Vec<f64>> = lo_records
            .iter()
            .take(10)
            .map(|r| {
                acc_n += 1;
                if r.accepted {
                    acc_count += 1.0;
                }
                let acc_rate = acc_count / acc_n as f64;
                vec![
                    r.plaquette,
                    r.plaquette_var,
                    r.polyakov_re,
                    r.polyakov_phase,
                    r.action_density,
                    acc_rate,
                    r.delta_h.abs(),
                    r.cg_iters as f64,
                ]
            })
            .collect();

        let pred = multi_esn
            .predict(&feature_seq)
            .unwrap_or_else(|_| vec![0.0; 6]);
        let pred_phase = pred[0];

        total_steered += 1;
        let should_go_higher = pred_phase < 0.5;
        let actually_near_transition =
            (beta_hi - KNOWN_BETA_C).abs() < (beta_lo - KNOWN_BETA_C).abs();
        if should_go_higher == actually_near_transition {
            steered_correctly += 1;
        }
    }

    PlacementResult {
        name: "D: Inter-beta steering".into(),
        wall_ms: start.elapsed().as_secs_f64() * 1000.0,
        accuracy: steered_correctly as f64 / total_steered.max(1) as f64,
        traj_saved: 0,
    }
}

/// Placement E: Pre-run bootstrap with synthetic features.
pub(super) fn placement_e_pre_run(
    records: &[TrajectoryRecord],
    multi_esn: &mut EchoStateNetwork,
) -> PlacementResult {
    let start = Instant::now();

    let betas: Vec<f64> = {
        let mut v: Vec<f64> = records.iter().map(|r| r.beta).collect();
        v.dedup();
        v
    };

    let mut correct = 0;
    let mut total = 0;

    for &beta in &betas {
        let plaq = 0.35 + 0.25 * (beta - 5.0) / 1.5;
        let poly = if beta > KNOWN_BETA_C {
            0.3 + 0.4 * (beta - KNOWN_BETA_C) / 1.0
        } else {
            0.05
        };
        let synthetic_seq: Vec<Vec<f64>> = (0..5)
            .map(|_| vec![plaq, 0.001, poly, 0.0, 6.0 * (1.0 - plaq), 0.7, 0.1, 0.0])
            .collect();

        let pred = multi_esn
            .predict(&synthetic_seq)
            .unwrap_or_else(|_| vec![0.0; 6]);
        let pred_phase: f64 = if pred[0] > 0.5 { 1.0 } else { 0.0 };
        let actual_phase: f64 = if beta > KNOWN_BETA_C { 1.0 } else { 0.0 };

        total += 1;
        if (pred_phase - actual_phase).abs() < 0.01 {
            correct += 1;
        }
    }

    PlacementResult {
        name: "E: Pre-run bootstrap".into(),
        wall_ms: start.elapsed().as_secs_f64() * 1000.0,
        accuracy: correct as f64 / total.max(1) as f64,
        traj_saved: 0,
    }
}

/// Placement F: Combined A + B + C.
pub(super) fn placement_f_combined(
    records: &[TrajectoryRecord],
    betas: &[f64],
    therm_esn: &mut EchoStateNetwork,
    reject_esn: &mut EchoStateNetwork,
    multi_esn: &mut EchoStateNetwork,
) -> PlacementResult {
    let start = Instant::now();

    let a = placement_a_pre_thermalization(records, therm_esn);
    let b = placement_b_mid_trajectory(records, betas, reject_esn);
    let c = placement_c_post_trajectory(records, multi_esn);

    let total_saved = a.traj_saved + b.traj_saved;
    let combined_accuracy = (a.accuracy + b.accuracy + c.accuracy) / 3.0;

    PlacementResult {
        name: "F: All combined".into(),
        wall_ms: start.elapsed().as_secs_f64() * 1000.0,
        accuracy: combined_accuracy,
        traj_saved: total_saved,
    }
}
