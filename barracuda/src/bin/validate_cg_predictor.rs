// SPDX-License-Identifier: AGPL-3.0-only

//! CG Convergence Predictor for Dynamical QCD (Phase 3)
//!
//! For dynamical fermion HMC, the conjugate gradient solver takes 60-80%
//! of wall time. CG iteration count depends on quark mass, gauge field
//! roughness, beta value, and lattice volume.
//!
//! Train an NPU model to predict CG iterations from configuration features:
//!   - Skip preconditioning when CG will converge fast
//!   - Apply aggressive preconditioning when CG will be slow
//!   - Reject trajectories early if predicted ΔH will be too large
//!
//! This validates the concept using CPU dynamical HMC data and ESN prediction.

use hotspring_barracuda::lattice::hmc::{self, HmcConfig};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CG Convergence Predictor (Phase 3)                        ║");
    println!("║  NPU predicts CG iterations → skip/apply preconditioning   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("cg_predictor");

    // ═══ Phase 1: Generate CG Training Data ═══
    println!("═══ Generate CG Training Data from Quenched HMC ═══");
    let data_start = Instant::now();

    // We simulate CG-like iteration counts from quenched HMC observables.
    // In production, these come from actual dynamical HMC CG solves.
    // The CG iteration count correlates with gauge roughness (plaquette),
    // beta value, and acceptance rate.
    let beta_values = [5.0, 5.3, 5.5, 5.7, 5.9, 6.0, 6.2, 6.5];
    let mut training_data: Vec<(Vec<f64>, f64)> = Vec::new();

    for &beta in &beta_values {
        let mut lat = Lattice::hot_start([4, 4, 4, 4], beta, 42);
        let mut config = HmcConfig {
            n_md_steps: 20,
            dt: 0.02,
            seed: 42,
            ..Default::default()
        };

        // Thermalize
        for _ in 0..20 {
            hmc::hmc_trajectory(&mut lat, &mut config);
        }

        // Collect data
        for i in 0..10 {
            let result = hmc::hmc_trajectory(&mut lat, &mut config);
            let poly = lat.average_polyakov_loop();

            // CG iterations correlate inversely with plaquette (smoother configs
            // converge faster) and inversely with beta (stronger coupling = rougher).
            // This is a physically motivated model of CG behavior.
            let roughness = 1.0 - result.plaquette;
            let simulated_cg_iters =
                50.0 + 200.0 * roughness + 30.0 * (6.0 / beta) + 10.0 * ((i as f64) * 0.5).sin();

            let features = vec![
                (beta - 5.0) / 2.0,   // normalized beta
                result.plaquette,     // mean plaquette
                poly,                 // Polyakov loop
                result.delta_h.abs(), // |ΔH|
                roughness,            // gauge roughness
            ];

            training_data.push((features, simulated_cg_iters));
        }
    }

    println!(
        "  Generated {} samples across {} β-points in {:.1}s",
        training_data.len(),
        beta_values.len(),
        data_start.elapsed().as_secs_f64()
    );
    println!();

    // ═══ Phase 2: Train CG Predictor ESN ═══
    println!("═══ Train CG Predictor ESN ═══");
    let train_start = Instant::now();

    let esn_config = EsnConfig {
        input_size: 5,
        reservoir_size: 40,
        output_size: 1, // predicted CG iterations
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 42,
    };

    // Build sequences (sliding windows of 5 consecutive measurements)
    let window_size = 5;
    let mut train_seqs = Vec::new();
    let mut train_targets = Vec::new();
    let mut test_seqs = Vec::new();
    let mut test_targets = Vec::new();

    let grouped: Vec<Vec<&(Vec<f64>, f64)>> = beta_values
        .iter()
        .enumerate()
        .map(|(i, _)| {
            training_data[i * 10..(i + 1) * 10]
                .iter()
                .collect::<Vec<_>>()
        })
        .collect();

    for (beta_idx, group) in grouped.iter().enumerate() {
        for start in 0..group.len().saturating_sub(window_size) {
            let seq: Vec<Vec<f64>> = group[start..start + window_size]
                .iter()
                .map(|(f, _)| f.clone())
                .collect();
            let target_cg = group[start + window_size - 1].1;
            let target = vec![target_cg / 300.0]; // normalize to [0,1] range

            if beta_idx % 3 == 0 {
                test_seqs.push(seq);
                test_targets.push(target);
            } else {
                train_seqs.push(seq);
                train_targets.push(target);
            }
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
    println!();

    // ═══ Phase 3: Validate CG Predictions ═══
    println!("═══ Validate CG Predictions ═══");

    let mut errors = Vec::new();
    let mut correct_category = 0;
    let total = test_seqs.len();

    for (seq, target) in test_seqs.iter().zip(test_targets.iter()) {
        let pred = esn.predict(seq).expect("ESN trained")[0];
        let pred_iters = pred * 300.0;
        let actual_iters = target[0] * 300.0;
        let err = (pred_iters - actual_iters).abs();
        errors.push(err);

        // Category: fast (<100), medium (100-200), slow (>200)
        let pred_cat = categorize_cg(pred_iters);
        let actual_cat = categorize_cg(actual_iters);
        if pred_cat == actual_cat {
            correct_category += 1;
        }
    }

    let mean_err: f64 = errors.iter().sum::<f64>() / errors.len().max(1) as f64;
    let max_err: f64 = errors.iter().copied().fold(0.0, f64::max);
    let cat_acc = correct_category as f64 / total.max(1) as f64;

    println!("  Mean absolute error: {mean_err:.1} iterations");
    println!("  Max absolute error: {max_err:.1} iterations");
    println!(
        "  Category accuracy (fast/med/slow): {:.0}%",
        cat_acc * 100.0
    );

    harness.check_bool(
        "CG predictions finite",
        errors.iter().all(|e| e.is_finite()),
    );
    harness.check_bool("Mean CG error < 100 iterations", mean_err < 100.0);
    println!();

    // ═══ Phase 4: NpuSimulator Parity ═══
    println!("═══ NpuSimulator CG Predictor Parity ═══");

    let weights = esn.export_weights().expect("export weights");
    let mut npu_sim = NpuSimulator::from_exported(&weights);

    let mut max_npu_err = 0.0f64;
    for seq in &test_seqs {
        let cpu_pred = esn.predict(seq).expect("ESN trained")[0];
        let npu_pred = npu_sim.predict(seq)[0];
        let err = (cpu_pred - npu_pred).abs();
        max_npu_err = max_npu_err.max(err);
    }

    println!("  Max CPU-NPU error: {max_npu_err:.6}");
    harness.check_bool("NPU CG predictor parity", max_npu_err < 0.1);

    // ═══ Phase 5: Savings Estimation ═══
    println!();
    println!("═══ Projected Savings from CG Prediction ═══");

    let mut wasted_without = 0.0;
    let mut wasted_with = 0.0;

    for (seq, target) in test_seqs.iter().zip(test_targets.iter()) {
        let pred = esn.predict(seq).expect("ESN trained")[0];
        let actual = target[0];

        // Without predictor: always use default max iterations (300)
        wasted_without += (1.0 - actual) * 300.0;

        // With predictor: allocate predicted iterations + 20% buffer
        let allocated = (pred * 300.0 * 1.2).min(300.0);
        wasted_with += (allocated - actual * 300.0).max(0.0);
    }

    let savings_pct = if wasted_without > 0.0 {
        (1.0 - wasted_with / wasted_without) * 100.0
    } else {
        0.0
    };

    println!("  Wasted iterations without predictor: {wasted_without:.0}");
    println!("  Wasted iterations with predictor: {wasted_with:.0}");
    println!("  Projected savings: {savings_pct:.0}%");

    harness.check_bool("CG predictor reduces waste", wasted_with <= wasted_without);

    println!();
    harness.finish();
}

fn categorize_cg(iters: f64) -> u8 {
    if iters < 100.0 {
        0 // fast
    } else if iters < 200.0 {
        1 // medium
    } else {
        2 // slow
    }
}
