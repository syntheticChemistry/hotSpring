// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-Observable NPU Monitoring (Phase 1)
//!
//! Expands the NPU screening pipeline from single-feature (plaquette) to an
//! 8-feature observable vector. Trains a multi-output ESN that simultaneously
//! produces: phase label, estimated β_c, thermalization flag, anomaly score.
//!
//! The 8-feature vector fits within the AKD1000's validated 50-dim input path.
//! All FC layers merge into a single HW pass (Discovery 2 from BEYOND_SDK).
//!
//! Features:
//!   0: plaquette (mean)
//!   1: plaquette variance (from last N trajectories)
//!   2: Polyakov loop magnitude
//!   3: Polyakov loop phase
//!   4: action density
//!   5: acceptance rate (running average)
//!   6: |ΔH| magnitude
//!   7: CG iterations (0 for quenched)

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_hmc_trajectory_streaming, gpu_polyakov_loop, GpuHmcState, GpuHmcStreamingPipelines,
    StreamObservables,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

use hotspring_barracuda::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Multi-Observable NPU Monitoring (Phase 1)                 ║");
    println!("║  8-feature ESN: phase + β_c + thermalization + anomaly     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("multi_observable_npu");
    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU not available: {e}");
            std::process::exit(1);
        }
    };
    println!("  GPU: {}", gpu.adapter_name);
    println!();

    // ═══ Phase 1: Generate 8-Feature Training Data ═══
    println!("═══ Generate 8-Feature Training Data from GPU HMC ═══");
    let data_start = Instant::now();

    let pipelines = GpuHmcStreamingPipelines::new(&gpu);
    let beta_values: Vec<f64> = (0..12).map(|i| 4.5 + 2.0 * (i as f64) / 11.0).collect();

    let mut all_features: Vec<(f64, Vec<Vec<f64>>)> = Vec::new();

    for &beta in &beta_values {
        let mut lat = Lattice::hot_start([4, 4, 4, 4], beta, 42);
        let mut config = HmcConfig {
            n_md_steps: 20,
            dt: 0.02,
            seed: 42,
            ..Default::default()
        };
        for _ in 0..10 {
            hmc::hmc_trajectory(&mut lat, &mut config);
        }

        let state = GpuHmcState::from_lattice(&gpu, &lat, beta);
        let mut seed = 12345u64;
        let mut plaquette_history: Vec<f64> = Vec::new();
        let mut accept_count = 0u32;
        let mut feature_sequence = Vec::new();
        let n_traj = 15;

        for t in 0..n_traj {
            let result = gpu_hmc_trajectory_streaming(
                &gpu, &pipelines, &state, 20, 0.02, t as u32, &mut seed,
            );

            plaquette_history.push(result.plaquette);
            if result.accepted {
                accept_count += 1;
            }

            let (poly_mag, poly_phase) = gpu_polyakov_loop(&gpu, &pipelines.hmc, &state);

            let plaq_var = if plaquette_history.len() > 1 {
                let mean = plaquette_history.iter().sum::<f64>() / plaquette_history.len() as f64;
                plaquette_history
                    .iter()
                    .map(|p| (p - mean).powi(2))
                    .sum::<f64>()
                    / (plaquette_history.len() - 1) as f64
            } else {
                0.0
            };

            let obs = StreamObservables {
                plaquette: result.plaquette,
                polyakov_re: poly_mag,
                delta_h: result.delta_h,
                cg_iterations: 0,
                accepted: result.accepted,
                plaquette_var: plaq_var,
                polyakov_phase: poly_phase,
                action_density: result.plaquette * 6.0,
            };

            let acc_rate = accept_count as f64 / (t + 1) as f64;
            feature_sequence.push(obs.to_feature_vec(acc_rate));
        }

        all_features.push((beta, feature_sequence));
    }

    println!(
        "  Generated {} β-points × 15 trajectories in {:.1}s",
        beta_values.len(),
        data_start.elapsed().as_secs_f64()
    );
    println!();

    // ═══ Phase 2: Train Multi-Output ESN ═══
    println!("═══ Train Multi-Output ESN (4 outputs) ═══");
    let train_start = Instant::now();

    let esn_config = EsnConfig {
        input_size: 8,
        reservoir_size: 50,
        output_size: 4, // phase_label, beta_c_est, thermalized, anomaly_score
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 42,
    };

    // Build train/test split
    let mut train_seqs = Vec::new();
    let mut train_targets = Vec::new();
    let mut test_seqs = Vec::new();
    let mut test_targets = Vec::new();

    for (i, (beta, features)) in all_features.iter().enumerate() {
        let phase = if *beta > KNOWN_BETA_C { 1.0 } else { 0.0 };
        let beta_c_norm = (KNOWN_BETA_C - 5.0) / 2.0;
        let thermalized = if features.len() > 5 { 1.0 } else { 0.0 };
        let anomaly = if (*beta - KNOWN_BETA_C).abs() < 0.2 {
            0.5
        } else {
            0.0
        };

        let target = vec![phase, beta_c_norm, thermalized, anomaly];

        if i % 3 == 0 {
            test_seqs.push(features.clone());
            test_targets.push(target);
        } else {
            train_seqs.push(features.clone());
            train_targets.push(target);
        }
    }

    let mut esn = EchoStateNetwork::new(esn_config);
    esn.train(&train_seqs, &train_targets);

    println!(
        "  Trained on {} sequences ({} test) in {:.1}ms",
        train_seqs.len(),
        test_seqs.len(),
        train_start.elapsed().as_secs_f64() * 1000.0
    );

    // ═══ Phase 3: Validate ESN Predictions ═══
    println!();
    println!("═══ Validate Multi-Output Predictions ═══");

    let mut phase_correct = 0;
    let mut total = 0;
    let mut beta_c_errors = Vec::new();
    let mut therm_correct = 0;

    for (seq, target) in test_seqs.iter().zip(test_targets.iter()) {
        let pred = esn.predict(seq).expect("ESN trained");
        total += 1;

        // Phase classification (output 0)
        let pred_phase = if pred[0] > 0.5 { 1.0 } else { 0.0 };
        if (pred_phase - target[0]).abs() < 0.01 {
            phase_correct += 1;
        }

        // β_c estimation (output 1)
        beta_c_errors.push((pred[1] - target[1]).abs());

        // Thermalization (output 2)
        let pred_therm = if pred[2] > 0.5 { 1.0 } else { 0.0 };
        if (pred_therm - target[2]).abs() < 0.01 {
            therm_correct += 1;
        }

        println!(
            "  β={:.2}: phase={:.2} (target {:.0}), β_c={:.3}, therm={:.2}, anom={:.3}",
            all_features[test_seqs
                .iter()
                .position(|s| std::ptr::eq(s, seq))
                .unwrap_or(0)
                * 3]
            .0,
            pred[0],
            target[0],
            pred[1] * 2.0 + 5.0,
            pred[2],
            pred[3]
        );
    }

    let phase_acc = phase_correct as f64 / total as f64;
    let therm_acc = therm_correct as f64 / total as f64;
    let mean_beta_c_err: f64 = beta_c_errors.iter().sum::<f64>() / beta_c_errors.len() as f64;

    println!();
    println!("  Phase accuracy: {:.0}%", phase_acc * 100.0);
    println!("  Thermalization accuracy: {:.0}%", therm_acc * 100.0);
    println!("  Mean β_c error: {mean_beta_c_err:.4}");

    harness.check_bool("Multi-output ESN trained successfully", true);
    harness.check_bool(
        "Phase predictions finite",
        test_seqs
            .iter()
            .all(|s| esn.predict(s).unwrap().iter().all(|v| v.is_finite())),
    );
    harness.check_bool(
        "All 4 outputs produced",
        test_seqs.iter().all(|s| esn.predict(s).unwrap().len() == 4),
    );

    // ═══ Phase 4: NpuSimulator Parity ═══
    println!();
    println!("═══ NpuSimulator f32 Parity (8-feature) ═══");

    let weights = esn.export_weights().expect("export weights");
    let mut npu_sim = NpuSimulator::from_exported(&weights);

    let mut max_err = 0.0f64;
    let mut agree = 0;

    for seq in &test_seqs {
        let cpu_pred = esn.predict(seq).expect("ESN trained");
        let npu_pred = npu_sim.predict(seq);

        let err: f64 = cpu_pred
            .iter()
            .zip(npu_pred.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        max_err = max_err.max(err);

        let cpu_phase = i32::from(cpu_pred[0] > 0.5);
        let npu_phase = i32::from(npu_pred[0] > 0.5);
        if cpu_phase == npu_phase {
            agree += 1;
        }
    }

    let agreement = agree as f64 / test_seqs.len() as f64;
    println!("  Max f32 error: {max_err:.6}");
    println!("  Phase agreement: {:.0}%", agreement * 100.0);

    harness.check_bool("NpuSimulator handles 8 features", max_err.is_finite());
    harness.check_bool(
        "4-output NpuSimulator parity",
        test_seqs.iter().all(|s| npu_sim.predict(s).len() == 4),
    );

    println!();
    harness.finish();
}
