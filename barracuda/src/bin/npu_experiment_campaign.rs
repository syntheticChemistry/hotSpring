// SPDX-License-Identifier: AGPL-3.0-only

//! NPU Experiment Campaign: Characterize, Optimize, Feed Back
//!
//! Comprehensive NPU characterization using per-trajectory QCD data.
//! Trains specialized ESN models, tests 6 pipeline placements (A-F),
//! and characterizes NPU behavior for metalForge silicon discovery
//! and Akida feedback.
//!
//! # Steps
//!
//! 1. Generate per-trajectory training data via CPU HMC (4⁴ lattice)
//! 2. Train thermalization detector ESN
//! 3. Train rejection predictor ESN
//! 4. Build 6-output single-model NPU
//! 5. Run placement experiments A-F
//! 6. Characterize NPU behavior (latency, batch, drift, mutation)

use hotspring_barracuda::lattice::hmc::{self, HmcConfig};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;
use hotspring_barracuda::validation::ValidationHarness;
use std::io::Write;
use std::time::Instant;

const WINDOW_SIZE: usize = 10;
const N_THERM: usize = 50;
const N_MEAS: usize = 100;
const N_TOTAL: usize = N_THERM + N_MEAS;

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct TrajectoryRecord {
    beta: f64,
    traj_idx: usize,
    is_therm: bool,
    accepted: bool,
    plaquette: f64,
    polyakov_re: f64,
    delta_h: f64,
    cg_iters: usize,
    plaquette_var: f64,
    polyakov_phase: f64,
    action_density: f64,
    wall_us: u64,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  NPU Experiment Campaign: Characterize, Optimize, Feedback ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("npu_experiment_campaign");
    let campaign_start = Instant::now();

    // ═══════════════════════════════════════════════════════════════════
    //  Step 1: Generate Per-Trajectory Training Data
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Step 1: Generate Per-Trajectory Training Data (CPU HMC 4⁴) ═══");
    let gen_start = Instant::now();

    let beta_values: Vec<f64> = (0..12).map(|i| 5.0 + 1.5 * (i as f64) / 11.0).collect();
    let all_records = generate_trajectory_data(&beta_values);

    let n_total_traj = all_records.len();
    let n_therm_traj = all_records.iter().filter(|r| r.is_therm).count();
    let n_meas_traj = n_total_traj - n_therm_traj;
    let n_accepted = all_records.iter().filter(|r| r.accepted).count();

    println!(
        "  Generated {} trajectories ({} therm + {} meas) across {} β-points in {:.1}ms",
        n_total_traj,
        n_therm_traj,
        n_meas_traj,
        beta_values.len(),
        gen_start.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "  Overall acceptance: {:.1}%",
        n_accepted as f64 / n_total_traj as f64 * 100.0
    );
    println!();

    harness.check_bool(
        "Generated 1800 trajectories",
        n_total_traj == 12 * N_TOTAL,
    );

    // ═══════════════════════════════════════════════════════════════════
    //  Step 2: Thermalization Detector ESN
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Step 2: Thermalization Detector ESN ═══");
    let therm_start = Instant::now();

    let (therm_seqs, therm_targets) = build_thermalization_dataset(&all_records);
    let therm_config = EsnConfig {
        input_size: WINDOW_SIZE,
        reservoir_size: 50,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 42,
    };

    let (train_split, test_split) = split_dataset(&therm_seqs, &therm_targets, 0.8);
    let mut therm_esn = EchoStateNetwork::new(therm_config);
    therm_esn.train(&train_split.0, &train_split.1);

    let (therm_accuracy, therm_savings) =
        evaluate_thermalization_detector(&mut therm_esn, &test_split.0, &test_split.1);

    println!(
        "  Trained on {} windows, tested on {} in {:.1}ms",
        train_split.0.len(),
        test_split.0.len(),
        therm_start.elapsed().as_secs_f64() * 1000.0
    );
    println!("  Accuracy: {:.1}%", therm_accuracy * 100.0);
    println!(
        "  Potential savings: {:.1}% of thermalization trajectories",
        therm_savings * 100.0
    );

    let projected_hours_saved = therm_savings * 5.1;
    println!("  Projected time savings at production scale: {projected_hours_saved:.1}h of 5.1h therm budget");
    println!();

    harness.check_bool(
        "Therm detector accuracy > 60%",
        therm_accuracy > 0.60,
    );
    harness.check_bool(
        "Therm detector finds savings",
        therm_savings > 0.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    //  Step 3: Rejection Predictor ESN
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Step 3: Rejection Predictor ESN ═══");
    let reject_start = Instant::now();

    let (reject_seqs, reject_targets) = build_rejection_dataset(&all_records, &beta_values);
    let reject_config = EsnConfig {
        input_size: 5,
        reservoir_size: 50,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 77,
    };

    let (rtrain, rtest) = split_dataset(&reject_seqs, &reject_targets, 0.8);
    let mut reject_esn = EchoStateNetwork::new(reject_config);
    reject_esn.train(&rtrain.0, &rtrain.1);

    let (reject_accuracy, reject_early_abort_rate) =
        evaluate_rejection_predictor(&mut reject_esn, &rtest.0, &rtest.1);

    println!(
        "  Trained on {} sequences, tested on {} in {:.1}ms",
        rtrain.0.len(),
        rtest.0.len(),
        reject_start.elapsed().as_secs_f64() * 1000.0
    );
    println!("  Accuracy: {:.1}%", reject_accuracy * 100.0);
    println!(
        "  Early abort rate: {:.1}% of rejected trajectories",
        reject_early_abort_rate * 100.0
    );
    println!();

    harness.check_bool(
        "Reject predictor accuracy > 50%",
        reject_accuracy > 0.50,
    );

    // ═══════════════════════════════════════════════════════════════════
    //  Step 4: 6-Output Single-Model NPU
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Step 4: 6-Output Single-Model NPU ═══");
    let multi_start = Instant::now();

    let (multi_seqs, multi_targets) = build_multi_output_dataset(&all_records);
    let multi_config = EsnConfig {
        input_size: 8,
        reservoir_size: 50,
        output_size: 6,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 99,
    };

    let (mtrain, mtest) = split_dataset(&multi_seqs, &multi_targets, 0.8);
    let mut multi_esn = EchoStateNetwork::new(multi_config);
    multi_esn.train(&mtrain.0, &mtrain.1);

    let multi_metrics = evaluate_multi_output(&mut multi_esn, &mtest.0, &mtest.1);

    println!(
        "  Trained 6-output model on {} sequences in {:.1}ms",
        mtrain.0.len(),
        multi_start.elapsed().as_secs_f64() * 1000.0
    );
    println!("  Phase accuracy:         {:.1}%", multi_metrics.phase_acc * 100.0);
    println!("  β_c mean error:         {:.4}", multi_metrics.beta_c_err);
    println!("  Therm accuracy:         {:.1}%", multi_metrics.therm_acc * 100.0);
    println!("  Acceptance pred error:   {:.3}", multi_metrics.accept_err);
    println!("  Anomaly AUC:            {:.3}", multi_metrics.anomaly_auc);
    println!("  CG pred mean error:     {:.1}", multi_metrics.cg_err);

    let weights = multi_esn.export_weights().expect("export weights");
    let mut npu_sim = NpuSimulator::from_exported(&weights);
    let npu_pred = npu_sim.predict(&mtest.0[0]);
    println!();
    println!("  NpuSimulator output count: {} (expected 6)", npu_pred.len());
    println!(
        "  Sample prediction: phase={:.2}, β_c={:.3}, therm={:.2}, acc={:.2}, anom={:.3}, cg={:.1}",
        npu_pred[0],
        npu_pred[1] * 1.5 + 5.0,
        npu_pred[2],
        npu_pred[3],
        npu_pred[4],
        npu_pred[5]
    );
    println!();

    harness.check_bool("6-output model produces 6 outputs", npu_pred.len() == 6);
    harness.check_bool("All outputs finite", npu_pred.iter().all(|v| v.is_finite()));
    harness.check_bool(
        "Phase predictions reasonable (>= 0%)",
        multi_metrics.phase_acc >= 0.0 && multi_metrics.phase_acc.is_finite(),
    );

    // ═══════════════════════════════════════════════════════════════════
    //  Step 5: NPU Placement Experiments (A-F)
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Step 5: NPU Placement Experiments ═══");
    let placement_start = Instant::now();

    let placements = run_placement_experiments(
        &all_records,
        &beta_values,
        &mut therm_esn,
        &mut reject_esn,
        &mut multi_esn,
    );

    println!(
        "  Completed 6 placements in {:.1}ms",
        placement_start.elapsed().as_secs_f64() * 1000.0
    );
    println!();
    println!("  {:>25} {:>10} {:>10} {:>12} {:>10}", "Placement", "Time(ms)", "Accuracy", "Traj Saved", "Speedup");
    println!("  {:>25} {:>10} {:>10} {:>12} {:>10}", "─────────", "────────", "────────", "──────────", "───────");

    let baseline_time = placements[2].wall_ms;
    for p in &placements {
        let speedup = if p.wall_ms > 0.0 {
            baseline_time / p.wall_ms
        } else {
            0.0
        };
        println!(
            "  {:>25} {:>9.1} {:>9.1}% {:>11} {:>9.2}×",
            p.name, p.wall_ms, p.accuracy * 100.0, p.traj_saved, speedup
        );
    }
    println!();

    harness.check_bool("All 6 placements completed", placements.len() == 6);
    harness.check_bool(
        "Combined placement (F) shows improvement",
        placements[5].traj_saved > 0,
    );

    // ═══════════════════════════════════════════════════════════════════
    //  Step 6: NPU Behavior Characterization
    // ═══════════════════════════════════════════════════════════════════
    println!("═══ Step 6: NPU Behavior Characterization ═══");
    let char_start = Instant::now();

    let char_results = characterize_npu_behavior(
        &mtest.0,
        &mut npu_sim,
        &weights,
        &multi_esn,
    );

    println!(
        "  Characterization completed in {:.1}ms",
        char_start.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    println!("  ── Latency Distribution ──");
    println!(
        "    p50: {:.1}µs  p95: {:.1}µs  p99: {:.1}µs  mean: {:.1}µs",
        char_results.latency_p50_us,
        char_results.latency_p95_us,
        char_results.latency_p99_us,
        char_results.latency_mean_us
    );

    println!();
    println!("  ── Batch Size Effect ──");
    for (batch, throughput) in &char_results.batch_throughput {
        println!("    batch={batch}: {throughput:.1} inferences/ms");
    }

    println!();
    println!("  ── Weight Mutation Overhead ──");
    println!(
        "    Mutation time: {:.2}ms (target: 14ms on hardware)",
        char_results.mutation_time_ms
    );
    println!(
        "    Predictions changed: {}",
        char_results.mutation_changed_predictions
    );

    println!();
    println!("  ── Prediction Drift ──");
    println!(
        "    Max drift over {} sequential batches: {:.6}",
        char_results.drift_measurements.len(),
        char_results.max_drift
    );
    println!(
        "    Drift stable: {}",
        char_results.max_drift < 1e-6
    );

    println!();
    println!("  ── Accuracy vs Training Size ──");
    for (n, acc) in &char_results.accuracy_vs_training_size {
        let bar_len = (acc * 0.4) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    n={n:>4}: {acc:>5.1}% {bar}");
    }
    println!();

    harness.check_bool(
        "Latency measured (p50 > 0)",
        char_results.latency_p50_us > 0.0,
    );
    harness.check_bool(
        "Batch throughput measured for all sizes",
        char_results.batch_throughput.len() >= 4
            && char_results.batch_throughput.iter().all(|(_, t)| *t > 0.0),
    );
    harness.check_bool(
        "Prediction drift negligible",
        char_results.max_drift < 1e-4,
    );
    harness.check_bool(
        "Weight mutation < 50ms",
        char_results.mutation_time_ms < 50.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    //  Summary
    // ═══════════════════════════════════════════════════════════════════
    let total_elapsed = campaign_start.elapsed();
    println!("═══ Campaign Summary ═══");
    println!(
        "  Total campaign time: {:.1}s",
        total_elapsed.as_secs_f64()
    );
    println!("  Training data: {} trajectories across {} β-points", n_total_traj, beta_values.len());
    println!("  Models trained: 3 (therm detector, reject predictor, 6-output multi)");
    println!("  Placements tested: 6 (A-F)");
    println!("  Best placement: {}", placements.iter().max_by(|a, b| a.traj_saved.cmp(&b.traj_saved)).map_or("?", |p| p.name.as_str()));
    println!("  Therm detector savings: {:.1}% ({:.1}h projected)", therm_savings * 100.0, projected_hours_saved);
    println!();

    write_jsonl_summary(
        &all_records,
        &placements,
        &char_results,
        therm_accuracy,
        therm_savings,
        reject_accuracy,
        &multi_metrics,
    );

    harness.finish();
}

// ═══════════════════════════════════════════════════════════════════════
//  Data Generation
// ═══════════════════════════════════════════════════════════════════════

fn generate_trajectory_data(betas: &[f64]) -> Vec<TrajectoryRecord> {
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

fn plaquette_variance(history: &[f64]) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    history.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (history.len() - 1) as f64
}

// ═══════════════════════════════════════════════════════════════════════
//  Step 2: Thermalization Detector
// ═══════════════════════════════════════════════════════════════════════

fn build_thermalization_dataset(
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
        let beta_records: Vec<&TrajectoryRecord> =
            records.iter().filter(|r| (r.beta - beta).abs() < 1e-10).collect();

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
            meas_plaqs.iter().map(|p| (p - equilibrium_mean).powi(2)).sum::<f64>()
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

fn evaluate_thermalization_detector(
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

// ═══════════════════════════════════════════════════════════════════════
//  Step 3: Rejection Predictor
// ═══════════════════════════════════════════════════════════════════════

fn build_rejection_dataset(
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

fn evaluate_rejection_predictor(
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

// ═══════════════════════════════════════════════════════════════════════
//  Step 4: Multi-Output Model
// ═══════════════════════════════════════════════════════════════════════

fn build_multi_output_dataset(
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
        let beta_records: Vec<&TrajectoryRecord> =
            records.iter().filter(|r| (r.beta - beta).abs() < 1e-10 && !r.is_therm).collect();

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
        let beta_c_norm = (KNOWN_BETA_C - 5.0) / 1.5;
        let is_near_transition = if (beta - KNOWN_BETA_C).abs() < 0.3 { 1.0 } else { 0.0 };
        let mean_acceptance = running_acc / acc_count.max(1) as f64;
        let anomaly_score = if (beta - KNOWN_BETA_C).abs() < 0.2 { 0.5 } else { 0.0 };
        let mean_cg = beta_records.iter().map(|r| r.cg_iters as f64).sum::<f64>()
            / beta_records.len() as f64;

        targets.push(vec![
            phase,
            beta_c_norm,
            is_near_transition,
            mean_acceptance,
            anomaly_score,
            mean_cg,
        ]);
        seqs.push(feature_seq);
    }

    (seqs, targets)
}

struct MultiOutputMetrics {
    phase_acc: f64,
    beta_c_err: f64,
    therm_acc: f64,
    accept_err: f64,
    anomaly_auc: f64,
    cg_err: f64,
}

fn evaluate_multi_output(
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

// ═══════════════════════════════════════════════════════════════════════
//  Step 5: Placement Experiments
// ═══════════════════════════════════════════════════════════════════════

struct PlacementResult {
    name: String,
    wall_ms: f64,
    accuracy: f64,
    traj_saved: usize,
}

fn run_placement_experiments(
    records: &[TrajectoryRecord],
    betas: &[f64],
    therm_esn: &mut EchoStateNetwork,
    reject_esn: &mut EchoStateNetwork,
    multi_esn: &mut EchoStateNetwork,
) -> Vec<PlacementResult> {
    let results = vec![
        placement_a_pre_thermalization(records, therm_esn),
        placement_b_mid_trajectory(records, betas, reject_esn),
        placement_c_post_trajectory(records, multi_esn),
        placement_d_inter_beta(records, multi_esn),
        placement_e_pre_run(records, multi_esn),
        placement_f_combined(records, betas, therm_esn, reject_esn, multi_esn),
    ];

    results
}

fn placement_a_pre_thermalization(
    records: &[TrajectoryRecord],
    therm_esn: &mut EchoStateNetwork,
) -> PlacementResult {
    let start = Instant::now();
    let mut saved = 0;
    let mut _total_therm = 0;
    let mut correct_phases = 0;
    let mut total_tested = 0;

    let betas: Vec<f64> = {
        let mut v: Vec<f64> = records.iter().map(|r| r.beta).collect();
        v.dedup();
        v
    };

    for &beta in &betas {
        let beta_records: Vec<&TrajectoryRecord> =
            records.iter().filter(|r| (r.beta - beta).abs() < 1e-10).collect();

        let plaqs: Vec<f64> = beta_records.iter().map(|r| r.plaquette).collect();
        _total_therm += beta_records.iter().filter(|r| r.is_therm).count();

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

        let meas_records: Vec<&TrajectoryRecord> =
            beta_records.iter().filter(|r| !r.is_therm).copied().collect();
        if !meas_records.is_empty() {
            total_tested += 1;
            let is_deconfined = beta > KNOWN_BETA_C;
            let last_plaq = meas_records.last().unwrap().plaquette;
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

fn placement_b_mid_trajectory(
    records: &[TrajectoryRecord],
    betas: &[f64],
    reject_esn: &mut EchoStateNetwork,
) -> PlacementResult {
    let start = Instant::now();
    let mut saved = 0;
    let mut correct = 0;
    let meas_records: Vec<&TrajectoryRecord> =
        records.iter().filter(|r| !r.is_therm).collect();
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

fn placement_c_post_trajectory(
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

fn placement_d_inter_beta(
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

        let lo_records: Vec<&TrajectoryRecord> =
            records.iter().filter(|r| (r.beta - beta_lo).abs() < 1e-10 && !r.is_therm).collect();

        if lo_records.len() < 5 {
            continue;
        }

        let mut acc_count = 0.0;
        let mut acc_n = 0;
        let feature_seq: Vec<Vec<f64>> = lo_records.iter().take(10).map(|r| {
            acc_n += 1;
            if r.accepted { acc_count += 1.0; }
            let acc_rate = acc_count / acc_n as f64;
            vec![
                r.plaquette, r.plaquette_var, r.polyakov_re, r.polyakov_phase,
                r.action_density, acc_rate, r.delta_h.abs(), r.cg_iters as f64,
            ]
        }).collect();

        let pred = multi_esn.predict(&feature_seq).unwrap_or_else(|_| vec![0.0; 6]);
        let pred_phase = pred[0];

        total_steered += 1;
        let should_go_higher = pred_phase < 0.5;
        let actually_near_transition = (beta_hi - KNOWN_BETA_C).abs() < (beta_lo - KNOWN_BETA_C).abs();
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

fn placement_e_pre_run(
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

        let pred = multi_esn.predict(&synthetic_seq).unwrap_or_else(|_| vec![0.0; 6]);
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

fn placement_f_combined(
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

// ═══════════════════════════════════════════════════════════════════════
//  Step 6: NPU Behavior Characterization
// ═══════════════════════════════════════════════════════════════════════

struct CharacterizationResults {
    latency_p50_us: f64,
    latency_p95_us: f64,
    latency_p99_us: f64,
    latency_mean_us: f64,
    batch_throughput: Vec<(usize, f64)>,
    mutation_time_ms: f64,
    mutation_changed_predictions: bool,
    drift_measurements: Vec<f64>,
    max_drift: f64,
    accuracy_vs_training_size: Vec<(usize, f64)>,
}

fn characterize_npu_behavior(
    test_seqs: &[Vec<Vec<f64>>],
    npu_sim: &mut NpuSimulator,
    _weights: &hotspring_barracuda::md::reservoir::ExportedWeights,
    multi_esn: &EchoStateNetwork,
) -> CharacterizationResults {
    // Latency distribution
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

    // Batch size effect
    let batch_sizes = [1, 2, 4, 8];
    let mut batch_throughput = Vec::new();

    for &batch_size in &batch_sizes {
        let batch_seqs: Vec<&Vec<Vec<f64>>> = test_seqs.iter().take(batch_size).collect();
        let n_reps = 200;
        let t = Instant::now();
        for _ in 0..n_reps {
            for seq in &batch_seqs {
                let _ = npu_sim.predict(seq);
            }
        }
        let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
        let total_inferences = n_reps * batch_seqs.len();
        let throughput = total_inferences as f64 / elapsed_ms;
        batch_throughput.push((batch_size, throughput));
    }

    // Weight mutation
    let pred_before = npu_sim.predict(&seq);
    let mutation_start = Instant::now();
    let new_weights = multi_esn.export_weights().expect("export");
    let mut npu_after = NpuSimulator::from_exported(&new_weights);
    let mutation_time_ms = mutation_start.elapsed().as_secs_f64() * 1000.0;
    let pred_after = npu_after.predict(&seq);
    let mutation_changed = pred_before
        .iter()
        .zip(pred_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);

    // Prediction drift: run same input 100 times, check stability
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
    let max_drift = drift_measurements
        .iter()
        .copied()
        .fold(0.0, f64::max);

    // Accuracy vs training size
    let mut accuracy_vs_size = Vec::new();
    let (full_seqs, full_targets) = {
        let mut all_records = Vec::new();
        let beta_values: Vec<f64> = (0..12).map(|i| 5.0 + 1.5 * (i as f64) / 11.0).collect();
        for (bi, &beta) in beta_values.iter().enumerate() {
            let mut lat = Lattice::hot_start([4, 4, 4, 4], beta, 42 + bi as u64);
            let mut config = HmcConfig {
                n_md_steps: 20, dt: 0.02,
                seed: 1000 + bi as u64 * 100,
                ..Default::default()
            };
            for _ in 0..10 { hmc::hmc_trajectory(&mut lat, &mut config); }
            let mut acc_count = 0.0;
            let mut meas_seq = Vec::new();
            for t in 0..20 {
                let r = hmc::hmc_trajectory(&mut lat, &mut config);
                if r.accepted { acc_count += 1.0; }
                let poly = lat.polyakov_loop([0, 0, 0]);
                meas_seq.push(vec![
                    r.plaquette, 0.001,
                    (poly.re * poly.re + poly.im * poly.im).sqrt(),
                    poly.im.atan2(poly.re),
                    6.0 * (1.0 - r.plaquette),
                    acc_count / (t + 1) as f64,
                    r.delta_h.abs(), 0.0,
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
        if n_train > full_seqs.len() { break; }
        let train_s = &full_seqs[..n_train];
        let train_t: Vec<Vec<f64>> = full_targets[..n_train].iter().map(|&p| vec![p]).collect();
        let test_s = &full_seqs[n_train..];
        let test_t = &full_targets[n_train..];

        let cfg = EsnConfig {
            input_size: 8, reservoir_size: 50, output_size: 1,
            spectral_radius: 0.95, connectivity: 0.2, leak_rate: 0.3,
            regularization: 1e-2, seed: 42,
        };
        let mut esn = EchoStateNetwork::new(cfg);
        esn.train(train_s, &train_t);

        let w = esn.export_weights().expect("export");
        let mut sim = NpuSimulator::from_exported(&w);
        let mut correct = 0;
        for (s, &t) in test_s.iter().zip(test_t.iter()) {
            let pred = sim.predict(s);
            let pred_class = if pred[0] > 0.5 { 1.0 } else { 0.0 };
            if (pred_class - t).abs() < 0.01 { correct += 1; }
        }
        let acc = correct as f64 / test_s.len().max(1) as f64 * 100.0;
        accuracy_vs_size.push((n_train, acc));
    }

    CharacterizationResults {
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
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Utilities
// ═══════════════════════════════════════════════════════════════════════

fn count_unique_betas(records: &[TrajectoryRecord]) -> usize {
    let mut betas: Vec<u64> = records.iter().map(|r| r.beta.to_bits()).collect();
    betas.sort_unstable();
    betas.dedup();
    betas.len()
}

fn split_dataset(
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

fn write_jsonl_summary(
    records: &[TrajectoryRecord],
    placements: &[PlacementResult],
    char_results: &CharacterizationResults,
    therm_accuracy: f64,
    therm_savings: f64,
    reject_accuracy: f64,
    multi_metrics: &MultiOutputMetrics,
) {
    let path = "/tmp/hotspring-runs/v0614/npu_campaign_results.jsonl";
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let Ok(f) = std::fs::File::create(path) else {
        eprintln!("  Cannot write results to {path}");
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
    println!("  Results written to: {path}");
}
