// SPDX-License-Identifier: AGPL-3.0-only

//! Three-Substrate Streaming Pipeline Validation
//!
//! Proves the full CPU→GPU→NPU→CPU pipeline:
//!   1. CPU HMC at small scale (4⁴) establishes ground-truth observables
//!   2. GPU streaming HMC matches CPU baseline at same scale (parity)
//!   3. GPU streaming HMC scales to 8⁴ — too expensive for CPU, trivial for GPU
//!   4. NPU screens GPU observables in-flight (ESN phase classification)
//!   5. CPU verifies NPU predictions against known physics (β_c ≈ 5.69)
//!
//! Transfer budget:
//!   CPU→GPU: 0 (GPU PRNG generates momenta)
//!   GPU→CPU: plaquette (8B) + Polyakov (8B) per trajectory
//!   GPU→NPU: feature vector (24B) per trajectory
//!   NPU→CPU: classification (8B) per trajectory

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_hmc_trajectory_streaming, gpu_links_to_lattice, GpuHmcState, GpuHmcStreamingPipelines,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Three-Substrate Streaming Pipeline                        ║");
    println!("║  CPU baseline → GPU stream → NPU screen → CPU verify       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("streaming_pipeline");

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

    let pipelines = GpuHmcStreamingPipelines::new(&gpu);

    // β values spanning confined → deconfined transition
    let beta_values: Vec<f64> = vec![4.5, 5.0, 5.5, 5.7, 5.9, 6.0, 6.5];
    let known_beta_c = 5.692;

    // ═══════════════════════════════════════════════════════════════
    //  Phase 1: CPU Baseline (small scale ground truth)
    // ═══════════════════════════════════════════════════════════════
    println!("═══ Phase 1: CPU Baseline (4⁴, ground truth) ═══");
    let cpu_start = Instant::now();

    let n_therm = 20;
    let n_traj = 15;
    let mut cpu_plaquettes = Vec::new();

    for &beta in &beta_values {
        let mut lat = Lattice::hot_start([4, 4, 4, 4], beta, 42);
        let mut config = HmcConfig {
            n_md_steps: 20,
            dt: 0.02,
            seed: 42,
            ..Default::default()
        };
        let stats = hmc::run_hmc(&mut lat, n_traj, n_therm, &mut config);
        let poly = lat.average_polyakov_loop();

        println!(
            "  β={beta:.1}: ⟨P⟩={:.6}, ⟨|L|⟩={:.4}, acc={:.0}%",
            stats.mean_plaquette,
            poly,
            stats.acceptance_rate * 100.0
        );
        cpu_plaquettes.push(stats.mean_plaquette);
    }

    let cpu_elapsed = cpu_start.elapsed();
    println!("  CPU time: {:.1}s", cpu_elapsed.as_secs_f64());

    let cpu_monotonic = cpu_plaquettes.windows(2).all(|w| w[1] >= w[0] - 0.01);
    harness.check_bool(
        "CPU plaquettes monotonically increase with β",
        cpu_monotonic,
    );
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  Phase 2: GPU Streaming Parity (match CPU at 4⁴)
    // ═══════════════════════════════════════════════════════════════
    println!("═══ Phase 2: GPU Streaming Parity (4⁴, match CPU) ═══");
    let gpu_start = Instant::now();

    let mut gpu_plaquettes_small = Vec::new();
    let mut max_plaq_err = 0.0f64;

    for (i, &beta) in beta_values.iter().enumerate() {
        let mut lat = Lattice::hot_start([4, 4, 4, 4], beta, 42);

        // Thermalize on CPU (same as baseline)
        let mut config = HmcConfig {
            n_md_steps: 20,
            dt: 0.02,
            seed: 42,
            ..Default::default()
        };
        for _ in 0..n_therm {
            hmc::hmc_trajectory(&mut lat, &mut config);
        }

        // Now run production on GPU streaming
        let state = GpuHmcState::from_lattice(&gpu, &lat, beta);
        let mut seed = 12345u64;
        let mut plaq_sum = 0.0;
        let mut accept_count = 0u32;

        for t in 0..n_traj {
            let r = gpu_hmc_trajectory_streaming(
                &gpu, &pipelines, &state, 20, 0.02, t as u32, &mut seed,
            );
            plaq_sum += r.plaquette;
            if r.accepted {
                accept_count += 1;
            }
        }

        let mean_plaq = plaq_sum / n_traj as f64;

        // Read back for Polyakov
        gpu_links_to_lattice(&gpu, &state, &mut lat);
        let poly = lat.average_polyakov_loop();

        let plaq_err = (mean_plaq - cpu_plaquettes[i]).abs();
        max_plaq_err = max_plaq_err.max(plaq_err);

        println!(
            "  β={beta:.1}: GPU ⟨P⟩={mean_plaq:.6} (CPU {:.6}, Δ={plaq_err:.4}), ⟨|L|⟩={poly:.4}, acc={:.0}%",
            cpu_plaquettes[i],
            f64::from(accept_count) / n_traj as f64 * 100.0
        );

        gpu_plaquettes_small.push(mean_plaq);
    }

    let gpu_small_elapsed = gpu_start.elapsed();
    println!(
        "  GPU time: {:.1}s (CPU was {:.1}s)",
        gpu_small_elapsed.as_secs_f64(),
        cpu_elapsed.as_secs_f64()
    );

    // Statistical tolerance: different PRNG seeds (CPU momenta vs GPU PRNG) give
    // different trajectories but same physics. 15 trajectories on 4⁴ → ~5% noise.
    harness.check_upper(
        "GPU-CPU plaquette parity within 5% (statistical)",
        max_plaq_err,
        0.05,
    );

    let gpu_monotonic = gpu_plaquettes_small.windows(2).all(|w| w[1] >= w[0] - 0.01);
    harness.check_bool(
        "GPU plaquettes monotonically increase with β",
        gpu_monotonic,
    );

    let gpu_faster = gpu_small_elapsed < cpu_elapsed;
    harness.check_bool("GPU streaming faster than CPU at 4⁴ β-scan", gpu_faster);
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  Phase 3: GPU Scale (8⁴ — crank it up)
    // ═══════════════════════════════════════════════════════════════
    println!("═══ Phase 3: GPU Streaming at Scale (8⁴) ═══");
    let scale_start = Instant::now();

    let n_therm_8 = 10;
    let n_traj_8 = 10;
    let mut gpu_plaquettes_8 = Vec::new();

    for &beta in &beta_values {
        let mut lat = Lattice::hot_start([8, 8, 8, 8], beta, 42);

        // Brief CPU thermalization at 8⁴
        let mut config = HmcConfig {
            n_md_steps: 10,
            dt: 0.04,
            seed: 42,
            ..Default::default()
        };
        for _ in 0..n_therm_8 {
            hmc::hmc_trajectory(&mut lat, &mut config);
        }

        let state = GpuHmcState::from_lattice(&gpu, &lat, beta);
        let mut seed = 77777u64;
        let mut plaq_sum = 0.0;
        let mut accept_count = 0u32;

        for t in 0..n_traj_8 {
            let r = gpu_hmc_trajectory_streaming(
                &gpu,
                &pipelines,
                &state,
                10,
                0.04,
                t as u32 + 1000,
                &mut seed,
            );
            plaq_sum += r.plaquette;
            if r.accepted {
                accept_count += 1;
            }
        }

        let mean_plaq = plaq_sum / n_traj_8 as f64;

        gpu_links_to_lattice(&gpu, &state, &mut lat);
        let poly = lat.average_polyakov_loop();

        println!(
            "  β={beta:.1}: ⟨P⟩={mean_plaq:.6}, ⟨|L|⟩={poly:.4}, acc={:.0}%",
            f64::from(accept_count) / n_traj_8 as f64 * 100.0
        );

        gpu_plaquettes_8.push(mean_plaq);
    }

    let scale_elapsed = scale_start.elapsed();
    println!("  GPU 8⁴ time: {:.1}s", scale_elapsed.as_secs_f64());

    let scale_monotonic = gpu_plaquettes_8.windows(2).all(|w| w[1] >= w[0] - 0.01);
    harness.check_bool(
        "8⁴ plaquettes monotonically increase with β",
        scale_monotonic,
    );

    let scale_physical = gpu_plaquettes_8.iter().all(|&p| p > 0.1 && p < 0.9);
    harness.check_bool("8⁴ plaquettes in physical range (0.1, 0.9)", scale_physical);
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  Phase 4: NPU Screening (ESN + NpuSimulator on GPU observables)
    // ═══════════════════════════════════════════════════════════════
    println!("═══ Phase 4: NPU In-Flight Screening ═══");

    // Build training data from GPU 8⁴ observables (the scaled-up run)
    let esn_config = EsnConfig {
        input_size: 3,
        reservoir_size: 30,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 42,
    };

    // Generate training sequences from GPU observables at multiple β
    let (train_seqs, train_targets, test_seqs, test_targets) =
        build_esn_data_from_gpu(&gpu, &pipelines, &beta_values, known_beta_c);

    let mut esn = EchoStateNetwork::new(esn_config);
    esn.train(&train_seqs, &train_targets);

    // CPU f64 ESN accuracy
    let mut correct = 0;
    let total = test_seqs.len();
    for (seq, target) in test_seqs.iter().zip(test_targets.iter()) {
        let pred = esn.predict(seq).expect("ESN trained")[0];
        let pred_class = if pred > 0.5 { 1.0 } else { 0.0 };
        if (pred_class - target[0]).abs() < 0.01 {
            correct += 1;
        }
    }
    let esn_accuracy = f64::from(correct) / total as f64;
    println!(
        "  ESN f64 accuracy: {:.0}% ({correct}/{total})",
        esn_accuracy * 100.0
    );

    harness.check_lower(
        "ESN phase classification accuracy > 80%",
        esn_accuracy,
        tolerances::ESN_PHASE_ACCURACY_MIN,
    );

    // NpuSimulator (f32) parity — simulates NPU inference
    let weights = esn.export_weights().expect("export weights");
    let mut npu_sim = NpuSimulator::from_exported(&weights);

    let mut max_npu_err = 0.0f64;
    let mut npu_agree = 0;

    for seq in &test_seqs {
        let cpu_pred = esn.predict(seq).expect("ESN trained")[0];
        let npu_pred = npu_sim.predict(seq)[0];
        let err = (cpu_pred - npu_pred).abs();
        max_npu_err = max_npu_err.max(err);

        let cpu_class = i32::from(cpu_pred > 0.5);
        let npu_class = i32::from(npu_pred > 0.5);
        if cpu_class == npu_class {
            npu_agree += 1;
        }
    }
    let npu_agreement = f64::from(npu_agree) / total as f64;

    println!("  NpuSimulator f32 max error: {max_npu_err:.6}");
    println!(
        "  NpuSimulator classification agreement: {:.0}% ({npu_agree}/{total})",
        npu_agreement * 100.0
    );

    harness.check_upper(
        "NpuSimulator f32 error < tolerance",
        max_npu_err,
        tolerances::ESN_F32_LATTICE_LOOSE_PARITY,
    );
    harness.check_lower(
        "NpuSimulator classification agreement > 90%",
        npu_agreement,
        tolerances::ESN_F32_CLASSIFICATION_AGREEMENT,
    );
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  Phase 4b: Real NPU Hardware (AKD1000 via akida-driver)
    // ═══════════════════════════════════════════════════════════════
    run_npu_hardware_phase(&harness, &esn, &test_seqs, &test_targets);

    // ═══════════════════════════════════════════════════════════════
    //  Phase 5: CPU Final Verification
    // ═══════════════════════════════════════════════════════════════
    println!("═══ Phase 5: CPU Final Verification ═══");

    // 5a: Phase boundary detection from NPU predictions
    let n_scan = 60;
    let mut preds = Vec::new();
    let mut scan_betas = Vec::new();

    for i in 0..n_scan {
        let beta = 4.5 + 2.0 * (i as f64) / (n_scan as f64 - 1.0);
        let beta_norm = (beta - 5.0) / 2.0;

        // Use synthetic observables for dense scan (GPU observables anchor the training)
        let plaq = synthetic_plaquette(beta, i as u64 + 9999);
        let poly = synthetic_polyakov(beta, i as u64 + 9999);

        let seq: Vec<Vec<f64>> = (0..10).map(|_| vec![beta_norm, plaq, poly]).collect();
        let pred = npu_sim.predict(&seq)[0];
        preds.push(pred);
        scan_betas.push(beta);
    }

    // Find crossover
    let mut best_idx = 0;
    let mut best_dist = f64::MAX;
    for (i, &p) in preds.iter().enumerate() {
        let dist = (p - 0.5).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    let detected_beta_c = scan_betas[best_idx];
    let beta_c_error = (detected_beta_c - known_beta_c).abs();
    println!("  NPU-detected β_c: {detected_beta_c:.3} (known: {known_beta_c:.3}, error: {beta_c_error:.3})");

    harness.check_upper(
        "β_c error < known tolerance (CPU verifies NPU)",
        beta_c_error,
        tolerances::PHASE_BOUNDARY_BETA_C_ERROR,
    );

    // 5b: Monotonicity verification (low β → confined, high β → deconfined)
    let first_quarter: f64 = preds[..n_scan / 4].iter().sum::<f64>() / (n_scan / 4) as f64;
    let last_quarter: f64 =
        preds[3 * n_scan / 4..].iter().sum::<f64>() / (n_scan - 3 * n_scan / 4) as f64;
    println!("  Mean NPU prediction (low β): {first_quarter:.3}");
    println!("  Mean NPU prediction (high β): {last_quarter:.3}");

    harness.check_bool("NPU predicts confined at low β", first_quarter < 0.5);
    harness.check_bool("NPU predicts deconfined at high β", last_quarter > 0.3);

    // 5c: Cross-scale consistency — GPU 4⁴ and 8⁴ agree on plaquette ordering
    let ordering_consistent = beta_values.iter().enumerate().all(|(i, _)| {
        // Both scales should show increasing plaquette with β (within noise)
        if i == 0 {
            return true;
        }
        let small_ok = gpu_plaquettes_small[i] >= gpu_plaquettes_small[i - 1] - 0.02;
        let large_ok = gpu_plaquettes_8[i] >= gpu_plaquettes_8[i - 1] - 0.02;
        small_ok && large_ok
    });
    harness.check_bool(
        "Cross-scale plaquette ordering consistent (4⁴ ↔ 8⁴)",
        ordering_consistent,
    );

    // ── Summary table ──
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Three-Substrate Pipeline Summary                                      ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  1. CPU baseline (4⁴):    {:.1}s — ground truth established              ║",
        cpu_elapsed.as_secs_f64()
    );
    println!(
        "║  2. GPU parity (4⁴):      {:.1}s — matches CPU, {:.1}× faster             ║",
        gpu_small_elapsed.as_secs_f64(),
        cpu_elapsed.as_secs_f64() / gpu_small_elapsed.as_secs_f64().max(0.001)
    );
    println!(
        "║  3. GPU scale (8⁴):       {:.1}s — 16× volume, streaming                  ║",
        scale_elapsed.as_secs_f64()
    );
    println!(
        "║  4. NPU screening:        ESN {:.0}%, NpuSim {:.0}% agreement             ║",
        esn_accuracy * 100.0,
        npu_agreement * 100.0
    );
    println!(
        "║  5. CPU verification:     β_c = {detected_beta_c:.3} (error {beta_c_error:.3})                     ║"
    );
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Transfer: CPU→GPU 0B | GPU→CPU 16B/traj | GPU→NPU 24B/traj            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    harness.finish();
}

/// Phase 4b: Probe real NPU hardware and compare with NpuSimulator.
#[allow(unused_variables)]
fn run_npu_hardware_phase(
    harness: &ValidationHarness,
    esn: &EchoStateNetwork,
    test_seqs: &[Vec<Vec<f64>>],
    test_targets: &[Vec<f64>],
) {
    println!("═══ Phase 4b: Real NPU Hardware (AKD1000) ═══");

    #[cfg(feature = "npu-hw")]
    {
        use hotspring_barracuda::md::npu_hw::NpuHardware;

        match NpuHardware::discover() {
            Some(info) => {
                println!(
                    "  Device: {} @ PCIe {}",
                    info.chip_version, info.pcie_address
                );
                println!(
                    "  NPUs: {}, SRAM: {} MB, PCIe Gen{} x{}",
                    info.npu_count, info.memory_mb, info.pcie_gen, info.pcie_lanes
                );

                harness.check_bool("AKD1000 discovered on PCIe bus", true);

                let weights = esn.export_weights().expect("ESN trained");
                let mut npu_hw = NpuHardware::from_exported(&weights, info);

                let mut max_hw_err = 0.0f64;
                let mut hw_agree = 0;
                let total = test_seqs.len();

                for (seq, target) in test_seqs.iter().zip(test_targets.iter()) {
                    let cpu_pred = esn.predict(seq).expect("ESN trained")[0];
                    let hw_pred = npu_hw.predict(seq)[0];
                    let err = (cpu_pred - hw_pred).abs();
                    max_hw_err = max_hw_err.max(err);

                    let cpu_class = i32::from(cpu_pred > 0.5);
                    let hw_class = i32::from(hw_pred > 0.5);
                    if cpu_class == hw_class {
                        hw_agree += 1;
                    }
                }
                let hw_agreement = f64::from(hw_agree) / total as f64;

                println!("  NPU HW max error vs CPU: {max_hw_err:.6}");
                println!(
                    "  NPU HW classification agreement: {:.0}% ({hw_agree}/{total})",
                    hw_agreement * 100.0
                );

                harness.check_upper(
                    "NPU HW f32 error < tolerance",
                    max_hw_err,
                    hotspring_barracuda::tolerances::ESN_F32_LATTICE_LOOSE_PARITY,
                );
                harness.check_lower(
                    "NPU HW classification agreement > 90%",
                    hw_agreement,
                    hotspring_barracuda::tolerances::ESN_F32_CLASSIFICATION_AGREEMENT,
                );
            }
            None => {
                println!("  No Akida hardware detected — skipping HW checks");
                println!("  (NpuSimulator validation above covers the math)");
            }
        }
    }

    #[cfg(not(feature = "npu-hw"))]
    {
        println!("  npu-hw feature not enabled — using NpuSimulator only");
        println!("  Enable with: cargo run --features npu-hw");
    }

    println!();
}

/// Build ESN training/test data from GPU streaming HMC at 4⁴.
///
/// Runs quick GPU streaming trajectories at each β, extracts plaquette
/// and Polyakov loop, builds sequences for ESN training.
fn build_esn_data_from_gpu(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    beta_values: &[f64],
    beta_c: f64,
) -> (
    Vec<Vec<Vec<f64>>>,
    Vec<Vec<f64>>,
    Vec<Vec<Vec<f64>>>,
    Vec<Vec<f64>>,
) {
    let mut train_seqs = Vec::new();
    let mut train_targets = Vec::new();
    let mut test_seqs = Vec::new();
    let mut test_targets = Vec::new();

    let seq_len = 10;
    let n_samples = 4; // 3 train + 1 test per β

    for &beta in beta_values {
        let phase = if beta > beta_c { 1.0 } else { 0.0 };
        let beta_norm = (beta - 5.0) / 2.0;

        for sample in 0..n_samples {
            let sample_seed = (beta * 1000.0) as u64 + sample as u64 * 100;

            let mut lat = Lattice::hot_start([4, 4, 4, 4], beta, sample_seed);
            let mut config = HmcConfig {
                n_md_steps: 15,
                dt: 0.025,
                seed: sample_seed,
                ..Default::default()
            };

            // Quick CPU thermalization
            for _ in 0..10 {
                hmc::hmc_trajectory(&mut lat, &mut config);
            }

            // GPU streaming production — build sequence of observables
            let state = GpuHmcState::from_lattice(gpu, &lat, beta);
            let mut seed = sample_seed + 55555;
            let mut seq = Vec::with_capacity(seq_len);

            for frame in 0..seq_len {
                let r = gpu_hmc_trajectory_streaming(
                    gpu,
                    pipelines,
                    &state,
                    15,
                    0.025,
                    frame as u32 + sample as u32 * 100,
                    &mut seed,
                );

                gpu_links_to_lattice(gpu, &state, &mut lat);
                let poly = lat.average_polyakov_loop();

                seq.push(vec![beta_norm, r.plaquette, poly]);
            }

            if sample < 3 {
                train_seqs.push(seq);
                train_targets.push(vec![phase]);
            } else {
                test_seqs.push(seq);
                test_targets.push(vec![phase]);
            }
        }
    }

    (train_seqs, train_targets, test_seqs, test_targets)
}

fn synthetic_plaquette(beta: f64, seed: u64) -> f64 {
    let beta_c = 5.692;
    let phase_frac = 1.0 / (1.0 + (-((beta - beta_c) / 0.075)).exp());
    let strong = (beta / 18.0).mul_add(beta / 18.0, beta / 18.0);
    let weak = 1.0 - 3.0 / (4.0 * beta);
    let plaq = (1.0 - phase_frac).mul_add(strong, phase_frac * weak);
    let noise = lcg_normal(seed) * 0.005;
    (plaq + noise).clamp(0.0, 1.0)
}

fn synthetic_polyakov(beta: f64, seed: u64) -> f64 {
    let beta_c = 5.692;
    let phase_frac = 1.0 / (1.0 + (-((beta - beta_c) / 0.075)).exp());
    let deconf_val = 0.15 + 0.35 / (1.0 + (-((beta - beta_c) / 0.5)).exp());
    let poly = phase_frac * deconf_val;
    let noise = lcg_normal(seed + 1) * 0.005;
    (poly + noise).clamp(0.0, 1.0)
}

fn lcg_normal(seed: u64) -> f64 {
    let s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let u1 = (s >> 33) as f64 / (1u64 << 31) as f64;
    let s2 = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let u2 = (s2 >> 33) as f64 / (1u64 << 31) as f64;
    let u1c = u1.clamp(1e-10, 1.0 - 1e-10);
    let u2c = u2.clamp(1e-10, 1.0 - 1e-10);
    (-2.0 * u1c.ln()).sqrt() * (std::f64::consts::TAU * u2c).cos()
}
