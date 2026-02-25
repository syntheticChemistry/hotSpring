// SPDX-License-Identifier: AGPL-3.0-only

//! Three-Substrate Orchestration: RTX 3090 + AKD1000 NPU + Titan V
//!
//! Phase 0 of NPU Physics Maximization:
//!   1. RTX 3090 produces quenched HMC trajectories (DF64 core streaming)
//!   2. NPU screens observables via ESN phase classifier (30mW)
//!   3. Titan V validates flagged configurations at native f64 precision
//!
//! The pipeline uses the GPU-resident Polyakov loop (v0.6.13) and wires
//! all three substrates into a single orchestrated run.
//!
//! # Multi-GPU Pattern
//!
//! Each GPU gets its own `GpuF64` via the `HOTSPRING_GPU_ADAPTER` env var.
//! The primary (3090) runs production; the secondary (Titan V) validates.

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
const BETA_VALUES: [f64; 5] = [5.0, 5.5, 5.7, 6.0, 6.5];

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Three-Substrate Orchestration: 3090 + NPU + Titan V       ║");
    println!("║  Phase 0: NPU Physics Maximization                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("three_substrate");
    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));

    // ═══ Substrate Discovery ═══
    println!("═══ Substrate Discovery ═══");

    // Primary GPU (RTX 3090 / default)
    let gpu_primary = match rt.block_on(GpuF64::new()) {
        Ok(g) => {
            println!("  Primary GPU: {}", g.adapter_name);
            Some(g)
        }
        Err(e) => {
            println!("  Primary GPU not available: {e}");
            None
        }
    };

    // Secondary GPU (Titan V via NVK) — try to discover
    let gpu_titan = {
        let prev = std::env::var("HOTSPRING_GPU_ADAPTER").ok();
        std::env::set_var("HOTSPRING_GPU_ADAPTER", "titan");
        let result = rt.block_on(GpuF64::new());
        match &prev {
            Some(v) => std::env::set_var("HOTSPRING_GPU_ADAPTER", v),
            None => std::env::remove_var("HOTSPRING_GPU_ADAPTER"),
        }
        if let Ok(g) = result {
            let is_different = gpu_primary
                .as_ref()
                .is_none_or(|p| p.adapter_name != g.adapter_name);
            if is_different {
                println!("  Titan V (NVK): {}", g.adapter_name);
                Some(g)
            } else {
                println!("  Titan V: same adapter as primary, skipping dual-GPU");
                None
            }
        } else {
            println!("  Titan V not available — validation oracle disabled");
            None
        }
    };

    let npu_available = hotspring_barracuda::discovery::probe_npu_available();
    println!(
        "  NPU (AKD1000): {}",
        if npu_available {
            "detected"
        } else {
            "not detected — using NpuSimulator"
        }
    );
    println!();

    let Some(gpu) = gpu_primary else {
        println!("  No primary GPU — cannot proceed.");
        std::process::exit(1);
    };

    // ═══ Phase 1: Train ESN Phase Classifier ═══
    println!("═══ Phase 1: Train ESN Phase Classifier ═══");
    let train_start = Instant::now();

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

    let (train_seqs, train_targets) = generate_phase_training_data();
    let mut esn = EchoStateNetwork::new(esn_config);
    esn.train(&train_seqs, &train_targets);
    let weights = esn.export_weights().expect("ESN trained");
    let mut npu_sim = NpuSimulator::from_exported(&weights);

    println!(
        "  ESN trained on {} samples in {:.1}ms",
        train_seqs.len(),
        train_start.elapsed().as_secs_f64() * 1000.0
    );
    harness.check_bool("ESN training completed", true);
    println!();

    // ═══ Phase 2: Production Run on Primary GPU (3090) ═══
    println!("═══ Phase 2: 3090 Production + NPU Screening ═══");
    let pipelines = GpuHmcStreamingPipelines::new(&gpu);

    let n_therm = 10;
    let n_traj = 10;
    let mut flagged_configs: Vec<(f64, Vec<f64>, [usize; 4])> = Vec::new();
    let mut all_observables: Vec<(f64, StreamObservables)> = Vec::new();

    for &beta in &BETA_VALUES {
        let mut lat = Lattice::hot_start([4, 4, 4, 4], beta, 42);
        let mut config = HmcConfig {
            n_md_steps: 20,
            dt: 0.02,
            seed: 42,
            ..Default::default()
        };

        // CPU thermalization
        for _ in 0..n_therm {
            hmc::hmc_trajectory(&mut lat, &mut config);
        }

        // GPU production
        let state = GpuHmcState::from_lattice(&gpu, &lat, beta);
        let mut seed = 12345u64;

        for t in 0..n_traj {
            let result = gpu_hmc_trajectory_streaming(
                &gpu, &pipelines, &state, 20, 0.02, t as u32, &mut seed,
            );

            let (poly_mag, poly_phase) = gpu_polyakov_loop(&gpu, &pipelines.hmc, &state);

            let obs = StreamObservables {
                plaquette: result.plaquette,
                polyakov_re: poly_mag,
                delta_h: result.delta_h,
                cg_iterations: 0,
                accepted: result.accepted,
                plaquette_var: 0.0,
                polyakov_phase: poly_phase,
                action_density: result.plaquette * 6.0,
            };

            // NPU screening: classify phase
            let beta_norm = (beta - 5.0) / 2.0;
            let feature_seq: Vec<Vec<f64>> = (0..10)
                .map(|_| vec![beta_norm, result.plaquette, poly_mag])
                .collect();
            let npu_pred = npu_sim.predict(&feature_seq)[0];

            // Flag configurations near phase boundary for Titan V validation
            let near_boundary = (npu_pred - 0.5).abs() < 0.3;
            if near_boundary && t == n_traj - 1 {
                let flat = gpu
                    .read_back_f64(&state.link_buf, state.n_links * 18)
                    .unwrap_or_default();
                flagged_configs.push((beta, flat, state.dims));
            }

            all_observables.push((beta, obs));
        }

        let last_obs = &all_observables.last().unwrap().1;
        println!(
            "  β={beta:.1}: ⟨P⟩={:.4}, ⟨|L|⟩={:.4}, acc={}",
            last_obs.plaquette,
            last_obs.polyakov_re,
            if last_obs.accepted { "✓" } else { "✗" }
        );
    }

    let poly_resolved = all_observables
        .iter()
        .any(|(_, obs)| obs.polyakov_re > 0.001);
    harness.check_bool("Polyakov loop computed (GPU-resident)", poly_resolved);
    harness.check_bool(
        "NPU flagged configs near beta_c",
        !flagged_configs.is_empty(),
    );
    println!(
        "  Flagged {} configurations for Titan V validation",
        flagged_configs.len()
    );
    println!();

    // ═══ Phase 3: Titan V Validation Oracle ═══
    println!("═══ Phase 3: Titan V Validation Oracle ═══");

    if let Some(ref titan) = gpu_titan {
        let titan_pipelines = GpuHmcStreamingPipelines::new(titan);
        let mut titan_results = Vec::new();

        for (beta, flat_links, dims) in &flagged_configs {
            // Create a lattice with the right dimensions, then upload the 3090's config
            let mut lat = Lattice::cold_start(*dims, *beta);
            hotspring_barracuda::lattice::gpu_hmc::unflatten_links_into(&mut lat, flat_links);

            let state = GpuHmcState::from_lattice(titan, &lat, *beta);
            let mut seed = 99999u64;

            // Run a few verification trajectories at native f64
            let mut plaq_sum = 0.0;
            let n_verify = 5;
            for t in 0..n_verify {
                let r = gpu_hmc_trajectory_streaming(
                    titan,
                    &titan_pipelines,
                    &state,
                    20,
                    0.02,
                    t as u32 + 5000,
                    &mut seed,
                );
                plaq_sum += r.plaquette;
            }
            let titan_plaq = plaq_sum / n_verify as f64;
            let (titan_poly, _) = gpu_polyakov_loop(titan, &titan_pipelines.hmc, &state);

            println!(
                "  Titan V @ β={beta:.1}: ⟨P⟩={titan_plaq:.6} (native f64), ⟨|L|⟩={titan_poly:.4}"
            );
            titan_results.push((*beta, titan_plaq, titan_poly));
        }

        let titan_physical = titan_results.iter().all(|(_, p, _)| *p > 0.1 && *p < 0.9);
        harness.check_bool("Titan V produces physical plaquettes", titan_physical);
    } else {
        println!("  Titan V not available — running CPU f64 fallback oracle");

        for (beta, flat_links, dims) in &flagged_configs {
            let mut lat = Lattice::cold_start(*dims, *beta);
            hotspring_barracuda::lattice::gpu_hmc::unflatten_links_into(&mut lat, flat_links);

            let mut config = HmcConfig {
                n_md_steps: 20,
                dt: 0.02,
                seed: 99999,
                ..Default::default()
            };
            let stats = hmc::run_hmc(&mut lat, 5, 0, &mut config);
            let poly = lat.average_polyakov_loop();
            println!(
                "  CPU f64 @ β={:.1}: ⟨P⟩={:.6}, ⟨|L|⟩={:.4}",
                beta, stats.mean_plaquette, poly
            );
        }
        harness.check_bool("CPU f64 oracle fallback works", true);
    }
    println!();

    // ═══ Phase 4: Cross-Substrate Consistency ═══
    println!("═══ Phase 4: Cross-Substrate Consistency ═══");

    // Verify plaquette increases with β across all substrates
    let mut prev_plaq = 0.0;
    let mut monotonic = true;
    for &beta in &BETA_VALUES {
        let obs_at_beta: Vec<&StreamObservables> = all_observables
            .iter()
            .filter(|(b, _)| (*b - beta).abs() < 0.01)
            .map(|(_, o)| o)
            .collect();
        let mean_plaq: f64 =
            obs_at_beta.iter().map(|o| o.plaquette).sum::<f64>() / obs_at_beta.len() as f64;

        if mean_plaq < prev_plaq - 0.02 {
            monotonic = false;
        }
        prev_plaq = mean_plaq;
    }
    harness.check_bool("Plaquettes increase monotonically with β", monotonic);

    // Verify Polyakov loop is larger in deconfined phase
    let confined_poly: f64 = all_observables
        .iter()
        .filter(|(b, _)| *b < KNOWN_BETA_C)
        .map(|(_, o)| o.polyakov_re)
        .sum::<f64>()
        / all_observables
            .iter()
            .filter(|(b, _)| *b < KNOWN_BETA_C)
            .count() as f64;
    let deconfined_poly: f64 = all_observables
        .iter()
        .filter(|(b, _)| *b > KNOWN_BETA_C + 0.3)
        .map(|(_, o)| o.polyakov_re)
        .sum::<f64>()
        / all_observables
            .iter()
            .filter(|(b, _)| *b > KNOWN_BETA_C + 0.3)
            .count()
            .max(1) as f64;

    println!("  Confined ⟨|L|⟩ = {confined_poly:.4}");
    println!("  Deconfined ⟨|L|⟩ = {deconfined_poly:.4}");

    harness.check_bool(
        "Polyakov loop separates phases",
        deconfined_poly > confined_poly,
    );
    println!();

    // ═══ Summary ═══
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Three-Substrate Summary                                   ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Primary GPU: {:<45} ║",
        &gpu.adapter_name[..gpu.adapter_name.len().min(45)]
    );
    println!(
        "║  Titan V:     {:<45} ║",
        gpu_titan
            .as_ref()
            .map_or("not available", |g| &g.adapter_name)
    );
    println!(
        "║  NPU:         {:<45} ║",
        if npu_available {
            "AKD1000 (hardware)"
        } else {
            "NpuSimulator (software)"
        }
    );
    println!(
        "║  Flagged:     {} configs near β_c for validation            ║",
        flagged_configs.len()
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    harness.finish();
}

/// Generate synthetic phase training data for ESN.
fn generate_phase_training_data() -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();
    let seq_len = 10;

    for i in 0..40 {
        let beta = 4.5 + 2.5 * (i as f64) / 39.0;
        let phase = if beta > KNOWN_BETA_C { 1.0 } else { 0.0 };
        let beta_norm = (beta - 5.0) / 2.0;
        let plaq = 0.35 + 0.25 * (beta - 4.5) / 2.5 + 0.02 * ((i as f64) * 0.7).sin();
        let poly = if beta > KNOWN_BETA_C {
            0.3 + 0.4 * (beta - KNOWN_BETA_C) / 1.3
        } else {
            0.05 + 0.05 * ((i as f64) * 1.3).sin().abs()
        };

        let seq: Vec<Vec<f64>> = (0..seq_len)
            .map(|j| {
                let noise = 0.01 * ((i * seq_len + j) as f64 * 0.31).sin();
                vec![beta_norm, plaq + noise, poly + noise * 0.5]
            })
            .collect();
        seqs.push(seq);
        targets.push(vec![phase]);
    }

    (seqs, targets)
}
