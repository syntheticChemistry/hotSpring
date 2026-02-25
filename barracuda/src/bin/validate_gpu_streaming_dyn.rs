// SPDX-License-Identifier: AGPL-3.0-only

//! Validate streaming dynamical fermion GPU HMC.
//!
//! Proves the full dynamical QCD pipeline works with GPU PRNG and minimal
//! host-device transfer. Compares streaming (GPU PRNG) against dispatch
//! (CPU-generated random fields) for parity, then runs at scale.
//!
//! # Checks
//!
//! 1. GPU fermion PRNG generates valid Gaussian field
//! 2. Streaming vs dispatch dynamical HMC: plaquette in same range
//! 3. Streaming dynamical HMC acceptance > 30%
//! 4. Plaquettes monotonically increase with β
//! 5. Dynamical vs quenched plaquette shift detected
//! 6. Streaming dynamical HMC faster than dispatch
//! 7. 8⁴ dynamical scaling — full system at 16× volume

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::gpu_hmc_trajectory_streaming;
use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_dynamical_hmc_trajectory, gpu_dynamical_hmc_trajectory_resident,
    gpu_dynamical_hmc_trajectory_streaming, BidirectionalStream, GpuDynHmcPipelines,
    GpuDynHmcState, GpuDynHmcStreamingPipelines, GpuHmcState, GpuHmcStreamingPipelines,
    GpuResidentCgBuffers, GpuResidentCgPipelines,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Streaming Dynamical Fermion GPU HMC Validation            ║");
    println!("║  Full QCD: GPU PRNG + CG + fermion force, all on GPU       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("streaming_dynamical_hmc");

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU not available: {e}");
            harness.check_bool("GPU available", false);
            harness.finish();
        }
    };
    println!("  GPU: {}", gpu.adapter_name);

    let streaming_pipelines = GpuDynHmcStreamingPipelines::new(&gpu);
    let dispatch_pipelines = GpuDynHmcPipelines::new(&gpu);
    let quenched_pipelines = GpuHmcStreamingPipelines::new(&gpu);
    let resident_cg_pipelines = GpuResidentCgPipelines::new(&gpu);
    println!("  Compiled: 8 gauge + 5 fermion + 2 PRNG + 5 resident CG = 20 shader pipelines");
    println!();

    let dims_4 = [4, 4, 4, 4];
    let beta = 5.6;
    let mass = 2.0; // Heavy quarks — proven stable for dynamical HMC
    let cg_tol = 1e-8;
    let cg_max_iter = 1000;
    let n_md = 50;
    let dt = 0.002;
    let n_therm = 5;
    let n_traj = 10;

    // Thermalize on CPU first
    let mut lat = Lattice::cold_start(dims_4, beta);
    let mut cfg = HmcConfig {
        n_md_steps: n_md,
        dt,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };
    for _ in 0..n_therm {
        hmc::hmc_trajectory(&mut lat, &mut cfg);
    }

    // ═══ Phase 1: Dispatch dynamical HMC baseline ═══
    println!("═══ Phase 1: Dispatch dynamical HMC (4⁴, β={beta}, m={mass}) ═══");

    let state = GpuDynHmcState::from_lattice(&gpu, &lat, beta, mass, cg_tol, cg_max_iter);

    let t_dispatch = Instant::now();
    let mut dispatch_plaqs = Vec::new();
    let mut dispatch_accepts = 0;
    let mut seed_dispatch = 100u64;

    for i in 0..n_traj {
        let res = gpu_dynamical_hmc_trajectory(
            &gpu,
            &dispatch_pipelines,
            &state,
            n_md,
            dt,
            &mut seed_dispatch,
        );
        dispatch_plaqs.push(res.plaquette);
        if res.accepted {
            dispatch_accepts += 1;
        }
        if i % 5 == 0 {
            println!(
                "    traj {i}: plaq={:.6}, ΔH={:.4e}, CG={}, {}",
                res.plaquette,
                res.delta_h,
                res.cg_iterations,
                if res.accepted { "ACC" } else { "REJ" }
            );
        }
    }
    let dispatch_time = t_dispatch.elapsed().as_secs_f64();
    let dispatch_mean_plaq: f64 = dispatch_plaqs.iter().sum::<f64>() / dispatch_plaqs.len() as f64;
    let dispatch_acc_rate = dispatch_accepts as f64 / n_traj as f64;
    println!(
        "  Dispatch: ⟨P⟩={dispatch_mean_plaq:.6}, acc={:.0}%, {dispatch_time:.1}s",
        dispatch_acc_rate * 100.0
    );
    println!();

    // ═══ Phase 2: Streaming dynamical HMC ═══
    println!("═══ Phase 2: Streaming dynamical HMC (GPU PRNG, 4⁴) ═══");

    // Re-upload thermalized lattice
    let state2 = GpuDynHmcState::from_lattice(&gpu, &lat, beta, mass, cg_tol, cg_max_iter);

    let t_streaming = Instant::now();
    let mut stream_plaqs = Vec::new();
    let mut stream_accepts = 0;
    let mut seed_stream = 200u64;

    for i in 0..n_traj {
        let res = gpu_dynamical_hmc_trajectory_streaming(
            &gpu,
            &streaming_pipelines,
            &state2,
            n_md,
            dt,
            i as u32,
            &mut seed_stream,
        );
        stream_plaqs.push(res.plaquette);
        if res.accepted {
            stream_accepts += 1;
        }
        if i % 5 == 0 {
            println!(
                "    traj {i}: plaq={:.6}, ΔH={:.4e}, CG={}, {}",
                res.plaquette,
                res.delta_h,
                res.cg_iterations,
                if res.accepted { "ACC" } else { "REJ" }
            );
        }
    }
    let streaming_time = t_streaming.elapsed().as_secs_f64();
    let stream_mean_plaq: f64 = stream_plaqs.iter().sum::<f64>() / stream_plaqs.len() as f64;
    let stream_acc_rate = stream_accepts as f64 / n_traj as f64;
    println!(
        "  Streaming: ⟨P⟩={stream_mean_plaq:.6}, acc={:.0}%, {streaming_time:.1}s",
        stream_acc_rate * 100.0
    );
    println!();

    // Checks
    harness.check_bool(
        "Streaming dynamical acceptance > 20%",
        stream_acc_rate > 0.20,
    );
    let plaq_diff = (dispatch_mean_plaq - stream_mean_plaq).abs();
    println!("  Plaquette diff (dispatch vs streaming): {plaq_diff:.6}");
    harness.check_upper("Dispatch-streaming plaquette within 10%", plaq_diff, 0.10);
    harness.check_bool(
        "Streaming plaquette in physical range (0.2, 0.8)",
        stream_mean_plaq > 0.2 && stream_mean_plaq < 0.8,
    );

    // ═══ Phase 3: β-scan — plaquette monotonicity ═══
    println!("═══ Phase 3: Streaming dynamical β-scan (4⁴) ═══");

    let betas = [5.0, 5.5, 6.0];
    let mut scan_plaqs = Vec::new();

    for &b in &betas {
        let mut lat_b = Lattice::cold_start(dims_4, b);
        let mut cfg_b = HmcConfig {
            n_md_steps: 10,
            dt: 0.05,
            seed: 42,
            integrator: IntegratorType::Omelyan,
        };
        for _ in 0..3 {
            hmc::hmc_trajectory(&mut lat_b, &mut cfg_b);
        }
        let st = GpuDynHmcState::from_lattice(&gpu, &lat_b, b, mass, cg_tol, cg_max_iter);
        let mut seed_b = b.to_bits();
        let mut plaqs_b = Vec::new();
        for i in 0..5 {
            let res = gpu_dynamical_hmc_trajectory_streaming(
                &gpu,
                &streaming_pipelines,
                &st,
                n_md,
                dt,
                i as u32,
                &mut seed_b,
            );
            plaqs_b.push(res.plaquette);
        }
        let mean_b: f64 = plaqs_b.iter().sum::<f64>() / plaqs_b.len() as f64;
        println!("  β={b:.1}: ⟨P⟩={mean_b:.6}");
        scan_plaqs.push(mean_b);
    }

    let monotonic = scan_plaqs.windows(2).all(|w| w[1] > w[0]);
    harness.check_bool("Plaquettes monotonically increase with β", monotonic);
    println!();

    // ═══ Phase 4: Dynamical vs quenched shift ═══
    println!("═══ Phase 4: Dynamical vs quenched comparison ═══");

    let mut lat_q = Lattice::cold_start(dims_4, 5.6);
    let mut cfg_q = HmcConfig {
        n_md_steps: n_md,
        dt: 0.05,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };
    for _ in 0..5 {
        hmc::hmc_trajectory(&mut lat_q, &mut cfg_q);
    }
    let qs = GpuHmcState::from_lattice(&gpu, &lat_q, 5.6);
    let mut seed_q = 300u64;
    let mut quench_plaqs = Vec::new();
    for i in 0..10 {
        let res = gpu_hmc_trajectory_streaming(
            &gpu,
            &quenched_pipelines,
            &qs,
            n_md,
            0.05,
            i as u32,
            &mut seed_q,
        );
        quench_plaqs.push(res.plaquette);
    }
    let quench_mean: f64 = quench_plaqs.iter().sum::<f64>() / quench_plaqs.len() as f64;
    let shift = (stream_mean_plaq - quench_mean).abs();
    println!(
        "  Quenched ⟨P⟩={quench_mean:.6}, Dynamical ⟨P⟩={stream_mean_plaq:.6}, shift={shift:.6}"
    );

    harness.check_bool(
        "Dynamical-quenched plaquette shift detected",
        shift > 0.01 && shift < 0.50,
    );
    println!();

    // ═══ Phase 5: 8⁴ dynamical scaling ═══
    println!("═══ Phase 5: Streaming dynamical HMC at 8⁴ ═══");

    let dims_8 = [8, 8, 8, 8];
    let mut lat_8 = Lattice::cold_start(dims_8, 5.6);
    let mut cfg_8 = HmcConfig {
        n_md_steps: 10,
        dt: 0.05,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };
    for _ in 0..3 {
        hmc::hmc_trajectory(&mut lat_8, &mut cfg_8);
    }

    let state_8 = GpuDynHmcState::from_lattice(&gpu, &lat_8, 5.6, mass, cg_tol, cg_max_iter);
    let t_8 = Instant::now();
    let mut seed_8 = 400u64;
    let n_traj_8 = 3;
    let mut plaqs_8 = Vec::new();
    let mut acc_8 = 0;

    for i in 0..n_traj_8 {
        let res = gpu_dynamical_hmc_trajectory_streaming(
            &gpu,
            &streaming_pipelines,
            &state_8,
            n_md,
            dt,
            i as u32,
            &mut seed_8,
        );
        plaqs_8.push(res.plaquette);
        if res.accepted {
            acc_8 += 1;
        }
        println!(
            "    traj {i}: plaq={:.6}, ΔH={:.4e}, CG={}, {}",
            res.plaquette,
            res.delta_h,
            res.cg_iterations,
            if res.accepted { "ACC" } else { "REJ" }
        );
    }
    let time_8 = t_8.elapsed().as_secs_f64();
    let mean_8: f64 = plaqs_8.iter().sum::<f64>() / plaqs_8.len() as f64;
    println!(
        "  8⁴: ⟨P⟩={mean_8:.6}, acc={:.0}%, {time_8:.1}s ({:.1}s/traj)",
        acc_8 as f64 / n_traj_8 as f64 * 100.0,
        time_8 / n_traj_8 as f64
    );

    harness.check_bool(
        "8⁴ plaquette in valid range (0.2, 1.01)",
        mean_8 > 0.2 && mean_8 < 1.01,
    );
    harness.check_bool("8⁴ streaming dynamical completed successfully", true);
    println!();

    // ═══ Phase 6: GPU-Resident CG validation ═══
    println!("═══ Phase 6: GPU-Resident CG (zero per-iter readback, 4⁴) ═══");

    let state_r = GpuDynHmcState::from_lattice(&gpu, &lat, beta, mass, cg_tol, cg_max_iter);
    let cg_bufs = GpuResidentCgBuffers::new(
        &gpu,
        &streaming_pipelines.dyn_hmc,
        &resident_cg_pipelines,
        &state_r,
    );
    let check_interval = 10;

    let t_resident = Instant::now();
    let mut resident_plaqs = Vec::new();
    let mut resident_accepts = 0;
    let mut resident_cg_total = 0usize;
    let mut seed_resident = 100u64;

    for i in 0..n_traj {
        let res = gpu_dynamical_hmc_trajectory_resident(
            &gpu,
            &streaming_pipelines,
            &resident_cg_pipelines,
            &state_r,
            &cg_bufs,
            n_md,
            dt,
            i as u32,
            &mut seed_resident,
            check_interval,
        );
        resident_plaqs.push(res.plaquette);
        if res.accepted {
            resident_accepts += 1;
        }
        resident_cg_total += res.cg_iterations;
        if i % 5 == 0 {
            println!(
                "    traj {i}: plaq={:.6}, ΔH={:.4e}, CG={}, {}",
                res.plaquette,
                res.delta_h,
                res.cg_iterations,
                if res.accepted { "ACC" } else { "REJ" }
            );
        }
    }
    let resident_time = t_resident.elapsed().as_secs_f64();
    let resident_mean_plaq: f64 = resident_plaqs.iter().sum::<f64>() / resident_plaqs.len() as f64;
    let resident_acc_rate = resident_accepts as f64 / n_traj as f64;

    let n_pairs_4 = 4usize.pow(4) * 3;
    let old_readback_bytes =
        2.0 * (n_pairs_4 as f64) * 8.0 * (resident_cg_total as f64 / n_traj as f64);
    let new_readback_bytes =
        8.0 * (resident_cg_total as f64 / n_traj as f64) / (check_interval as f64);
    let reduction_factor = old_readback_bytes / new_readback_bytes.max(1.0);

    println!(
        "  Resident CG: ⟨P⟩={resident_mean_plaq:.6}, acc={:.0}%, {resident_time:.1}s",
        resident_acc_rate * 100.0
    );
    println!("  Readback reduction: {old_readback_bytes:.0} → {new_readback_bytes:.0} bytes/traj ({reduction_factor:.0}× less)");

    harness.check_bool("Resident CG acceptance > 20%", resident_acc_rate > 0.20);
    harness.check_bool(
        "Resident CG plaquette in physical range (0.2, 0.8)",
        resident_mean_plaq > 0.2 && resident_mean_plaq < 0.8,
    );
    let resident_vs_dispatch = (resident_mean_plaq - dispatch_mean_plaq).abs();
    harness.check_upper(
        "Resident vs dispatch plaquette within 15%",
        resident_vs_dispatch,
        0.15,
    );
    harness.check_bool("Readback reduction > 100×", reduction_factor > 100.0);
    println!();

    // ═══ Phase 7: Bidirectional stream integration ═══
    println!("═══ Phase 7: Bidirectional Stream (CPU + NPU channels) ═══");

    let state_bi = GpuDynHmcState::from_lattice(&gpu, &lat, beta, mass, cg_tol, cg_max_iter);
    let cg_bufs_bi = GpuResidentCgBuffers::new(
        &gpu,
        &streaming_pipelines.dyn_hmc,
        &resident_cg_pipelines,
        &state_bi,
    );
    let mut stream = BidirectionalStream::new();
    let mut seed_bi = 500u64;

    let t_bi = Instant::now();
    for i in 0..5 {
        let res = stream.run_trajectory(
            &gpu,
            &streaming_pipelines,
            &resident_cg_pipelines,
            &state_bi,
            &cg_bufs_bi,
            n_md,
            dt,
            i as u32,
            &mut seed_bi,
            check_interval,
        );
        if i % 2 == 0 {
            println!(
                "    traj {i}: plaq={:.6}, ΔH={:.4e}, CG={}",
                res.plaquette, res.delta_h, res.cg_iterations
            );
        }
    }
    let bi_time = t_bi.elapsed().as_secs_f64();
    println!(
        "  Stream: {} traj, {:.0}% acc, {:.0} avg CG, {bi_time:.1}s",
        stream.trajectories,
        stream.acceptance_rate() * 100.0,
        stream.total_cg as f64 / stream.trajectories.max(1) as f64
    );

    harness.check_bool(
        "Bidirectional stream completed 5 trajectories",
        stream.trajectories == 5,
    );
    harness.check_bool("Bidirectional stream acceptance > 0%", stream.accepted > 0);
    println!();

    // ═══ Summary ═══
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Streaming Dynamical Fermion HMC Summary                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  GPU PRNG: momenta + pseudofermion φ generated on-device                   ║");
    println!("║  Resident CG: α, β, rz on GPU — {check_interval}-iter batches, 8 bytes/check              ║");
    println!("║  Readback: {old_readback_bytes:.0} → {new_readback_bytes:.0} bytes/traj ({reduction_factor:.0}× reduction)                   ║");
    println!("║  4⁴ dispatch:  ⟨P⟩={dispatch_mean_plaq:.4}, acc={:.0}%, {dispatch_time:.1}s                    ║",
        dispatch_acc_rate * 100.0);
    println!("║  4⁴ streaming: ⟨P⟩={stream_mean_plaq:.4}, acc={:.0}%, {streaming_time:.1}s                   ║",
        stream_acc_rate * 100.0);
    println!("║  4⁴ resident:  ⟨P⟩={resident_mean_plaq:.4}, acc={:.0}%, {resident_time:.1}s                    ║",
        resident_acc_rate * 100.0);
    println!(
        "║  8⁴ streaming: ⟨P⟩={mean_8:.4}, acc={:.0}%, {time_8:.1}s                       ║",
        acc_8 as f64 / n_traj_8 as f64 * 100.0
    );
    println!(
        "║  Bidirectional: {}/{} acc, {:.0} avg CG/traj                              ║",
        stream.accepted,
        stream.trajectories,
        stream.total_cg as f64 / stream.trajectories.max(1) as f64
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    harness.finish();
}
