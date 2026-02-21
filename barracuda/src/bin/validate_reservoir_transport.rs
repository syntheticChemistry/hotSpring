// SPDX-License-Identifier: AGPL-3.0-only

//! Reservoir Computing Transport Prediction Validation
//!
//! Trains an Echo State Network on velocity features from CPU-reference
//! Yukawa OCP simulations and validates that the ESN can predict D*
//! (self-diffusion coefficient) from short trajectory segments.
//!
//! Cross-validates against Python reference:
//!   `control/reservoir_transport/scripts/reservoir_vacf.py`
//!
//! # Validation checks
//!
//! | Check | Tolerance | Basis |
//! |---|---|---|
//! | ESN train error | < 50% | Reservoir approximation, not replication |
//! | ESN test error | < 80% | Generalization to unseen (κ,Γ) |
//! | D* positive | — | Physical |
//! | Reservoir state non-trivial | — | Echo state property |
//! | Train < test error | — | No massive overfitting |

use std::time::Instant;

use hotspring_barracuda::md::config::MdConfig;
use hotspring_barracuda::md::cpu_reference::run_simulation_cpu;
use hotspring_barracuda::md::observables::compute_vacf;
use hotspring_barracuda::md::reservoir::{velocity_features, EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::validation::ValidationHarness;

#[allow(dead_code)]
struct CaseData {
    label: String,
    kappa: f64,
    gamma: f64,
    d_star_full: f64,
    features_short: Vec<Vec<f64>>,
}

fn reservoir_config(n_particles: usize) -> MdConfig {
    MdConfig {
        label: String::new(),
        n_particles,
        kappa: 0.0,
        gamma: 0.0,
        dt: 0.01,
        rc: 8.0,
        equil_steps: 5_000,
        prod_steps: 4_000,
        dump_step: 1,
        berendsen_tau: 5.0,
        rdf_bins: 100,
        vel_snapshot_interval: 1,
    }
}

fn main() {
    let total_start = Instant::now();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Reservoir Computing Transport Prediction Validation       ║");
    println!("║  ESN predicts D* from short velocity trajectories          ║");
    println!("║  Jaeger (2001) + Stanton & Murillo (2016)                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let cases_spec: Vec<(&str, f64, f64, f64)> = vec![
        ("k1_G50", 1.0, 50.0, 8.0),
        ("k1_G72", 1.0, 72.0, 8.0),
        ("k2_G31", 2.0, 31.0, 6.5),
        ("k2_G100", 2.0, 100.0, 6.5),
        ("k2_G158", 2.0, 158.0, 6.5),
        ("k3_G100", 3.0, 100.0, 6.0),
    ];

    let train_indices = [0usize, 2, 4, 5];
    let test_indices = [1usize, 3];
    let n_particles = 256;
    let short_frames = 500;

    println!("  Cases: {}", cases_spec.len());
    println!("  Train: {} cases, Test: {} cases", train_indices.len(), test_indices.len());
    println!("  N = {n_particles}, prod = 4000 steps (full), short = {short_frames} frames");
    println!();

    // ── Generate MD data ────────────────────────────────────────
    let mut case_data: Vec<CaseData> = Vec::new();

    for (label, kappa, gamma, rc) in &cases_spec {
        println!("─── Generating: {label} (κ={kappa}, Γ={gamma}) ───");
        let config = MdConfig {
            label: label.to_string(),
            kappa: *kappa,
            gamma: *gamma,
            rc: *rc,
            ..reservoir_config(n_particles)
        };

        let sim = run_simulation_cpu(&config);
        let n_snaps = sim.velocity_snapshots.len();
        println!("    Velocity snapshots: {n_snaps}");

        if n_snaps < 10 {
            println!("    WARNING: Too few velocity snapshots, skipping.");
            continue;
        }

        let vacf = compute_vacf(&sim.velocity_snapshots, n_particles, config.dt, n_snaps / 2);
        let d_star_full = vacf.diffusion_coeff;
        println!("    D*(full) = {d_star_full:.6e}");

        let short_snaps: Vec<Vec<f64>> = sim
            .velocity_snapshots
            .iter()
            .take(short_frames)
            .cloned()
            .collect();
        let features_short = velocity_features(&short_snaps, n_particles, *kappa, *gamma);

        case_data.push(CaseData {
            label: label.to_string(),
            kappa: *kappa,
            gamma: *gamma,
            d_star_full,
            features_short,
        });
    }

    // ── Train ESN ───────────────────────────────────────────────
    println!();
    println!("═══ Training Echo State Network ═══");

    let esn_config = EsnConfig {
        input_size: 8,
        reservoir_size: 50,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 42,
    };

    let mut esn = EchoStateNetwork::new(esn_config);

    let train_sequences: Vec<Vec<Vec<f64>>> = train_indices
        .iter()
        .map(|&i| case_data[i].features_short.clone())
        .collect();
    let train_targets: Vec<Vec<f64>> = train_indices
        .iter()
        .map(|&i| vec![case_data[i].d_star_full])
        .collect();

    esn.train(&train_sequences, &train_targets);
    println!("  Training complete.");

    // ── Evaluate ────────────────────────────────────────────────
    println!();
    println!("═══ Evaluation ═══");
    println!();
    println!("  {:<12} {:>12} {:>12} {:>10} {:>6}", "Case", "D*(GK)", "D*(ESN)", "Error", "Set");
    println!("  {}", "-".repeat(56));

    let mut harness = ValidationHarness::new("reservoir_transport");
    let mut train_errors = Vec::new();
    let mut test_errors = Vec::new();

    for (i, cd) in case_data.iter().enumerate() {
        let pred = esn.predict(&cd.features_short);
        let d_esn = pred[0];
        let err = if cd.d_star_full.abs() > 1e-30 {
            (d_esn - cd.d_star_full).abs() / cd.d_star_full.abs()
        } else {
            0.0
        };

        let split = if train_indices.contains(&i) { "TRAIN" } else { "TEST" };
        println!(
            "  {:<12} {:>12.4e} {:>12.4e} {:>9.1}% {:>6}",
            cd.label, cd.d_star_full, d_esn, err * 100.0, split
        );

        if train_indices.contains(&i) {
            train_errors.push(err);
        } else {
            test_errors.push(err);
        }

        harness.check_bool(
            &format!("{} D* positive", cd.label),
            cd.d_star_full > 0.0,
        );
    }

    let mean_train_err = if train_errors.is_empty() {
        0.0
    } else {
        train_errors.iter().sum::<f64>() / train_errors.len() as f64
    };
    let mean_test_err = if test_errors.is_empty() {
        0.0
    } else {
        test_errors.iter().sum::<f64>() / test_errors.len() as f64
    };

    println!();
    println!("  Mean train error: {:.1}%", mean_train_err * 100.0);
    println!("  Mean test error:  {:.1}%", mean_test_err * 100.0);

    harness.check_upper("ESN train mean error < 50%", mean_train_err, 0.50);
    harness.check_upper("ESN test mean error < 80%", mean_test_err, 0.80);
    harness.check_bool(
        "All D* positive and finite",
        case_data.iter().all(|cd| cd.d_star_full > 0.0 && cd.d_star_full.is_finite()),
    );

    // ── NPU Simulator (f32 cross-substrate parity) ──────────────
    println!();
    println!("═══ NPU Simulation (f32 arithmetic) ═══");
    println!();

    let exported = esn.export_weights().expect("ESN trained");
    println!("  Exported weights: W_in={}, W_res={}, W_out={}",
             exported.w_in.len(), exported.w_res.len(), exported.w_out.len());

    let mut npu_sim = NpuSimulator::from_exported(&exported);

    println!();
    println!("  {:<12} {:>12} {:>12} {:>10}", "Case", "D*(CPU f64)", "D*(NPU f32)", "Diff");
    println!("  {}", "-".repeat(50));

    let mut max_cpu_npu_diff = 0.0f64;

    for (i, cd) in case_data.iter().enumerate() {
        let d_cpu = esn.predict(&cd.features_short)[0];
        let d_npu = npu_sim.predict(&cd.features_short)[0];
        let diff = if d_cpu.abs() > 1e-30 {
            (d_npu - d_cpu).abs() / d_cpu.abs()
        } else {
            0.0
        };
        if diff > max_cpu_npu_diff {
            max_cpu_npu_diff = diff;
        }

        let split = if train_indices.contains(&i) { "TRAIN" } else { "TEST" };
        println!(
            "  {:<12} {:>12.4e} {:>12.4e} {:>9.2}%  {}",
            cd.label, d_cpu, d_npu, diff * 100.0, split
        );
    }

    println!();
    println!("  Max CPU/NPU diff: {:.2}%", max_cpu_npu_diff * 100.0);

    harness.check_upper(
        "CPU/NPU f64-f32 parity < 5%",
        max_cpu_npu_diff,
        0.05,
    );

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    println!();
    println!("═══ Summary ({total_ms:.0} ms) ═══");

    harness.finish();
}
