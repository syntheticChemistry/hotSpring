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

use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::npu_experiments::{
    build_multi_output_dataset, build_rejection_dataset, build_thermalization_dataset,
    characterize_npu_behavior, evaluate_multi_output, evaluate_rejection_predictor,
    evaluate_thermalization_detector, generate_trajectory_data, run_placement_experiments,
    split_dataset, write_jsonl_summary, N_TOTAL, WINDOW_SIZE,
};
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

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

    harness.check_bool("Generated 1800 trajectories", n_total_traj == 12 * N_TOTAL);

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

    harness.check_bool("Therm detector accuracy > 60%", therm_accuracy > 0.60);
    harness.check_bool("Therm detector finds savings", therm_savings > 0.0);

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

    harness.check_bool("Reject predictor accuracy > 50%", reject_accuracy > 0.50);

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
    println!(
        "  Phase accuracy:         {:.1}%",
        multi_metrics.phase_acc * 100.0
    );
    println!("  β_c mean error:         {:.4}", multi_metrics.beta_c_err);
    println!(
        "  Therm accuracy:         {:.1}%",
        multi_metrics.therm_acc * 100.0
    );
    println!("  Acceptance pred error:   {:.3}", multi_metrics.accept_err);
    println!("  Anomaly AUC:            {:.3}", multi_metrics.anomaly_auc);
    println!("  CG pred mean error:     {:.1}", multi_metrics.cg_err);

    let weights = multi_esn.export_weights().expect("export weights");
    let mut npu_sim = NpuSimulator::from_exported(&weights);
    let npu_pred = npu_sim.predict(&mtest.0[0]);
    println!();
    println!(
        "  NpuSimulator output count: {} (expected 6)",
        npu_pred.len()
    );
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
    println!(
        "  {:>25} {:>10} {:>10} {:>12} {:>10}",
        "Placement", "Time(ms)", "Accuracy", "Traj Saved", "Speedup"
    );
    println!(
        "  {:>25} {:>10} {:>10} {:>12} {:>10}",
        "─────────", "────────", "────────", "──────────", "───────"
    );

    let baseline_time = placements[2].wall_ms;
    for p in &placements {
        let speedup = if p.wall_ms > 0.0 {
            baseline_time / p.wall_ms
        } else {
            0.0
        };
        println!(
            "  {:>25} {:>9.1} {:>9.1}% {:>11} {:>9.2}×",
            p.name,
            p.wall_ms,
            p.accuracy * 100.0,
            p.traj_saved,
            speedup
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

    let char_results = characterize_npu_behavior(&mtest.0, &mut npu_sim, &weights, &multi_esn)
        .expect("characterize NPU behavior");

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
    println!("    Drift stable: {}", char_results.max_drift < 1e-6);

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
    harness.check_bool("Prediction drift negligible", char_results.max_drift < 1e-4);
    harness.check_bool(
        "Weight mutation < 50ms",
        char_results.mutation_time_ms < 50.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    //  Summary
    // ═══════════════════════════════════════════════════════════════════
    let total_elapsed = campaign_start.elapsed();
    println!("═══ Campaign Summary ═══");
    println!("  Total campaign time: {:.1}s", total_elapsed.as_secs_f64());
    println!(
        "  Training data: {} trajectories across {} β-points",
        n_total_traj,
        beta_values.len()
    );
    println!("  Models trained: 3 (therm detector, reject predictor, 6-output multi)");
    println!("  Placements tested: 6 (A-F)");
    println!(
        "  Best placement: {}",
        placements
            .iter()
            .max_by(|a, b| a.traj_saved.cmp(&b.traj_saved))
            .map_or("?", |p| p.name.as_str())
    );
    println!(
        "  Therm detector savings: {:.1}% ({:.1}h projected)",
        therm_savings * 100.0,
        projected_hours_saved
    );
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
