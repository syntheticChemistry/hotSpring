// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 6: QCD-specific workload comparison (Tasks A & B).

use hotspring_barracuda::bench::GpuEsn;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::validation::ValidationHarness;

use super::INPUT_SIZE;

pub fn run(gpu: &GpuF64, harness: &mut ValidationHarness, jsonl_records: &mut Vec<String>) {
    println!();
    println!("═══ Experiment 6: QCD-Specific Workload Comparison ═══");

    let n_traj = 200;
    let mut qcd_features: Vec<Vec<f64>> = Vec::with_capacity(n_traj);
    for i in 0..n_traj {
        let t = i as f64 / n_traj as f64;
        let is_therm = i < 50;
        let plaq = if is_therm {
            0.2f64.mul_add((-(i as f64) * 0.1).exp(), 0.3)
        } else {
            0.01f64.mul_add(((i as f64) * 0.3).sin(), 0.55)
        };
        let poly = if is_therm {
            0.01 * t
        } else {
            0.05f64.mul_add(((i as f64) * 0.2).cos(), 0.3)
        };
        qcd_features.push(vec![
            plaq,
            0.001 * (i as f64),
            poly,
            0.0,
            6.0 * (1.0 - plaq),
            if is_therm { 0.5 } else { 0.75 },
            if is_therm { 0.5 } else { 0.05 },
            0.0,
        ]);
    }

    // Task A: Thermalization detection across substrates
    {
        let config = EsnConfig {
            input_size: INPUT_SIZE,
            reservoir_size: 50,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
            ..Default::default()
        };
        let window = 10;
        let mut train_seqs = Vec::new();
        let mut train_targets = Vec::new();
        for start in (0..n_traj - window).step_by(5) {
            let seq: Vec<Vec<f64>> = qcd_features[start..start + window].to_vec();
            let is_therm = start + window <= 50;
            train_seqs.push(seq);
            train_targets.push(vec![if is_therm { 1.0 } else { 0.0 }]);
        }

        let mut esn = EchoStateNetwork::new(config);
        esn.train(&train_seqs, &train_targets);
        let exported = esn.export_weights().expect("export");
        let mut npu = NpuSimulator::from_exported(&exported);
        let gpu_esn = GpuEsn::new(gpu, &exported);

        let mut npu_correct = 0;
        let mut gpu_correct = 0;
        let n_test = train_seqs.len();

        for (i, seq) in train_seqs.iter().enumerate() {
            let expected = train_targets[i][0];
            let npu_pred = npu.predict(seq)[0];
            let gpu_pred = gpu_esn.predict(gpu, seq)[0];

            if ((if npu_pred > 0.5 { 1.0 } else { 0.0 }) - expected).abs() < 0.01 {
                npu_correct += 1;
            }
            if ((if gpu_pred > 0.5 { 1.0 } else { 0.0 }) - expected).abs() < 0.01 {
                gpu_correct += 1;
            }
        }

        let npu_acc = f64::from(npu_correct) / n_test as f64;
        let gpu_acc = f64::from(gpu_correct) / n_test as f64;
        println!("  Thermalization detection:");
        println!(
            "    NPU-sim: {npu_correct}/{n_test} = {:.1}%",
            npu_acc * 100.0
        );
        println!(
            "    GPU-ESN: {gpu_correct}/{n_test} = {:.1}%",
            gpu_acc * 100.0
        );
        harness.check_bool("qcd_therm_npu", npu_acc >= 0.7);
        harness.check_bool("qcd_therm_gpu", gpu_acc >= 0.7);

        jsonl_records.push(
            serde_json::json!({
                "experiment": "qcd_thermalization",
                "npu_accuracy": npu_acc, "gpu_accuracy": gpu_acc, "n_test": n_test,
            })
            .to_string(),
        );
    }

    // Task B: Multi-observable anomaly scoring
    {
        let config = EsnConfig {
            input_size: INPUT_SIZE,
            reservoir_size: 50,
            output_size: 3,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
            ..Default::default()
        };
        let window = 5;
        let mut train_seqs = Vec::new();
        let mut train_targets = Vec::new();
        for start in (50..n_traj - window).step_by(3) {
            let seq: Vec<Vec<f64>> = qcd_features[start..start + window].to_vec();
            let last = &qcd_features[start + window - 1];
            train_seqs.push(seq);
            train_targets.push(vec![last[0], last[2], last[4]]);
        }

        let mut esn = EchoStateNetwork::new(config);
        esn.train(&train_seqs, &train_targets);
        let exported = esn.export_weights().expect("export");
        let mut npu = NpuSimulator::from_exported(&exported);
        let gpu_esn = GpuEsn::new(gpu, &exported);

        let mut npu_errors = Vec::new();
        let mut gpu_errors = Vec::new();
        for (i, seq) in train_seqs.iter().enumerate() {
            let target = &train_targets[i];
            let npu_pred = npu.predict(seq);
            let gpu_pred = gpu_esn.predict(gpu, seq);

            let npu_err: f64 = npu_pred
                .iter()
                .zip(target.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                .sqrt();
            let gpu_err: f64 = gpu_pred
                .iter()
                .zip(target.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                .sqrt();

            npu_errors.push(npu_err);
            gpu_errors.push(gpu_err);
        }

        let npu_rmse =
            (npu_errors.iter().map(|e| e.powi(2)).sum::<f64>() / npu_errors.len() as f64).sqrt();
        let gpu_rmse =
            (gpu_errors.iter().map(|e| e.powi(2)).sum::<f64>() / gpu_errors.len() as f64).sqrt();

        println!("  Multi-observable anomaly scoring (3 outputs):");
        println!("    NPU-sim RMSE: {npu_rmse:.6}");
        println!("    GPU-ESN RMSE: {gpu_rmse:.6}");
        harness.check_upper("qcd_anomaly_npu_rmse", npu_rmse, 1.0);
        harness.check_upper("qcd_anomaly_gpu_rmse", gpu_rmse, 1.0);

        jsonl_records.push(
            serde_json::json!({
                "experiment": "qcd_anomaly_scoring",
                "npu_rmse": npu_rmse, "gpu_rmse": gpu_rmse,
                "n_samples": train_seqs.len(),
            })
            .to_string(),
        );
    }
}
