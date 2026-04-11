// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU trajectory execution (RHMC / streaming HMC) for production silicon QCD.

use super::support::{
    AmdGpuPower, BetaSummary, CliArgs, InstrumentedResult, SiliconBudget, estimate_traj_bytes,
    estimate_traj_flops, is_dynamical, mode_label,
};
use hotspring_barracuda::bench::{GpuTelemetry, PowerMonitor};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::resident_shifted_cg::GpuResidentShiftedCgBuffers;
use hotspring_barracuda::lattice::gpu_hmc::true_multishift_cg::TrueMultiShiftBuffers;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuDynHmcPipelines, GpuDynHmcState, GpuHmcState, GpuHmcStreamingPipelines, GpuRhmcPipelines,
    GpuRhmcState, TrajectoryResult, UniHamiltonianBuffers, UniPipelines,
    gpu_hmc_trajectory_streaming, gpu_rhmc_trajectory_unidirectional,
};
use hotspring_barracuda::lattice::rhmc::RhmcConfig;
use hotspring_barracuda::lattice::wilson::Lattice;

use std::io::Write;
use std::time::Instant;

use super::flow::{build_summary, run_gradient_flow_uni, run_quenched_gradient_flow};

#[expect(
    clippy::expect_used,
    reason = "GPU trajectory failure is unrecoverable in this pipeline"
)]
pub fn run_beta_point(
    gpu: &GpuF64,
    budget: &SiliconBudget,
    amd_power: &AmdGpuPower,
    telemetry: &GpuTelemetry,
    args: &CliArgs,
    beta: f64,
    dims: [usize; 4],
    vol: usize,
    seed: &mut u64,
    out_file: &mut Option<std::io::BufWriter<std::fs::File>>,
) -> BetaSummary {
    let l = args.lattice;
    let dynamical = is_dynamical(&args.mode);

    let hw = telemetry.snapshot();
    eprintln!(
        "━━━ β = {beta:.4} ({}) ━━━  [{}]",
        mode_label(args),
        hw.status_line()
    );

    // Hot start
    let lattice = Lattice::hot_start(dims, beta, *seed);

    // Quenched pre-thermalization (streaming path — already zero-sync)
    let quenched_state = GpuHmcState::from_lattice(gpu, &lattice, beta);
    if args.n_quenched_pretherm > 0 {
        let quenched_pipelines = GpuHmcStreamingPipelines::new_with_tmu(gpu);
        for i in 0..args.n_quenched_pretherm {
            let r = gpu_hmc_trajectory_streaming(
                gpu,
                &quenched_pipelines,
                &quenched_state,
                20,
                0.1,
                i as u32,
                seed,
            )
            .expect("streaming HMC trajectory");
            if (i + 1) % 10 == 0 {
                let hw = telemetry.snapshot();
                eprintln!(
                    "  pretherm {}/{}: P={:.6} ΔH={:.4} {}  [{}]",
                    i + 1,
                    args.n_quenched_pretherm,
                    r.plaquette,
                    r.delta_h,
                    if r.accepted { "Y" } else { "N" },
                    hw.status_line()
                );
            }
        }
    }

    if !dynamical {
        return run_quenched_beta(
            gpu,
            budget,
            amd_power,
            telemetry,
            args,
            beta,
            dims,
            vol,
            seed,
            out_file,
            &quenched_state,
        );
    }

    // ── Dynamical: UNIDIRECTIONAL RHMC (GPU-resident CG, minimal readback) ──
    let mut rhmc_config = match args.mode.as_str() {
        "nf1" | "1" => RhmcConfig::nf1(args.mass, beta),
        "nf2+1" | "2+1" => RhmcConfig::nf2p1(args.mass, args.strange_mass, beta),
        "nf3" | "3" => RhmcConfig::nf3(args.mass, beta),
        "nf4" | "4" => RhmcConfig::nf4(args.mass, beta),
        "nf2+1+1" => RhmcConfig::nf2p1p1(args.mass, args.strange_mass, args.charm_mass, beta),
        _ => RhmcConfig::nf2(args.mass, beta),
    };
    rhmc_config.dt = args.dt;
    rhmc_config.n_md_steps = args.n_md_steps;
    rhmc_config.cg_tol = args.cg_tol;
    rhmc_config.cg_max_iter = args.cg_max_iter;

    let dyn_state = GpuDynHmcState::from_lattice(
        gpu,
        &lattice,
        beta,
        rhmc_config.sectors[0].mass,
        args.cg_tol,
        args.cg_max_iter,
    );
    let rhmc_state = GpuRhmcState::new(gpu, &rhmc_config, dyn_state);

    if args.n_quenched_pretherm > 0 {
        let n_bytes = (quenched_state.n_links * 18 * 8) as u64;
        let mut enc = gpu.begin_encoder("copy_therm_links");
        enc.copy_buffer_to_buffer(
            &quenched_state.link_buf,
            0,
            &rhmc_state.gauge.gauge.link_buf,
            0,
            n_bytes,
        );
        gpu.submit_encoder(enc);
    }

    let dyn_pipelines = GpuDynHmcPipelines::new(gpu);
    let rhmc_pipelines = GpuRhmcPipelines::new(gpu);

    // Build unidirectional buffers for GPU-resident CG (~50x fewer sync points)
    let uni_pipelines = UniPipelines::new_saturated(gpu, vol);
    let scg_bufs = GpuResidentShiftedCgBuffers::new(
        gpu,
        &dyn_pipelines,
        &uni_pipelines.shifted_cg,
        &rhmc_state.gauge,
    );
    let ham_bufs = UniHamiltonianBuffers::new(
        gpu,
        &uni_pipelines.shifted_cg.base.reduce_pipeline,
        &rhmc_state.gauge.gauge,
        &rhmc_state.gauge,
    );

    // True multi-shift CG: shared Krylov, N_shifts fewer D†D ops per iteration
    let max_shifts = rhmc_config
        .sectors
        .iter()
        .map(|s| s.action_approx.sigma.len().max(s.force_approx.sigma.len()))
        .max()
        .unwrap_or(0);
    let ms_bufs = if max_shifts > 0 {
        Some(TrueMultiShiftBuffers::new(
            gpu,
            &dyn_pipelines,
            &uni_pipelines.true_ms_cg,
            &rhmc_state.gauge,
            max_shifts,
        ))
    } else {
        None
    };
    eprintln!("  True multi-shift CG: {max_shifts} shifts, shared Krylov basis");

    eprintln!(
        "  Thermalizing: {} RHMC trajectories (unidirectional)...",
        args.n_therm
    );
    let mut therm_accepted = 0;
    for i in 0..args.n_therm {
        let r = run_uni_traj(
            gpu,
            &dyn_pipelines,
            &rhmc_pipelines,
            &uni_pipelines,
            &rhmc_state,
            &scg_bufs,
            ms_bufs.as_ref(),
            &ham_bufs,
            &rhmc_config,
            seed,
        );
        if r.accepted {
            therm_accepted += 1;
        }
        if (i + 1) % 20 == 0 || i == 0 {
            let hw = telemetry.snapshot();
            eprintln!(
                "    therm {}/{}: P={:.6} ΔH={:.4e} CG={} acc={:.0}%  [{}]",
                i + 1,
                args.n_therm,
                r.plaquette,
                r.delta_h,
                r.total_cg_iterations,
                therm_accepted as f64 / (i + 1) as f64 * 100.0,
                hw.status_line()
            );
        }
    }

    // Measurement phase with energy + telemetry
    let monitor = PowerMonitor::start();
    let meas_start = Instant::now();
    let mut amd_power_samples: Vec<f64> = Vec::new();

    let mut results: Vec<InstrumentedResult> = Vec::new();
    let mut meas_accepted = 0;
    let mut plaq_sum = 0.0;

    for i in 0..args.n_meas {
        if let Some(w) = amd_power.read_watts() {
            amd_power_samples.push(w);
        }

        let t0 = Instant::now();
        let r = run_uni_traj(
            gpu,
            &dyn_pipelines,
            &rhmc_pipelines,
            &uni_pipelines,
            &rhmc_state,
            &scg_bufs,
            ms_bufs.as_ref(),
            &ham_bufs,
            &rhmc_config,
            seed,
        );
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let est_flops = estimate_traj_flops(vol, args.n_md_steps, r.total_cg_iterations, false);
        let est_gflops = est_flops / 1e9;
        let est_gflop_per_s = est_gflops / (wall_ms / 1000.0);
        let est_bytes = estimate_traj_bytes(vol, args.n_md_steps, r.total_cg_iterations, false);
        let est_gb_s = (est_bytes / 1e9) / (wall_ms / 1000.0);

        if r.accepted {
            meas_accepted += 1;
        }
        plaq_sum += r.plaquette;

        let ir = InstrumentedResult {
            traj_idx: i,
            accepted: r.accepted,
            delta_h: r.delta_h,
            plaquette: r.plaquette,
            cg_iters: r.total_cg_iterations,
            wall_ms,
            est_gflops,
            est_gflop_per_s,
            est_gb_s,
        };

        // CSV with telemetry columns
        let hw = telemetry.snapshot();
        let line = format!(
            "{},{beta:.4},{l},{},{},{},{:.6e},{:.8},{},{:.1},{:.3},{:.1},{:.1},meas",
            args.mode,
            budget.name,
            i,
            i32::from(ir.accepted),
            ir.delta_h,
            ir.plaquette,
            ir.cg_iters,
            ir.wall_ms,
            ir.est_gflops,
            ir.est_gflop_per_s,
            ir.est_gb_s
        );
        println!("{line}");
        if let Some(f) = out_file.as_mut() {
            writeln!(f, "{line}").ok();
            f.flush().ok();
        }

        results.push(ir);

        if (i + 1) % 10 == 0 || i + 1 == args.n_meas {
            let n = (i + 1) as f64;
            let mean_p = plaq_sum / n;
            let rate = meas_accepted as f64 / n * 100.0;
            eprintln!(
                "    meas {}/{}: ⟨P⟩={:.6} acc={:.0}% {:.0}ms/traj {:.0}GFLOP/s  [{}]",
                i + 1,
                args.n_meas,
                mean_p,
                rate,
                wall_ms,
                est_gflop_per_s,
                hw.status_line()
            );
        }
    }

    let meas_wall_s = meas_start.elapsed().as_secs_f64();
    let mut energy = monitor.stop();

    if energy.gpu_joules < 0.01 && !amd_power_samples.is_empty() {
        let avg_w = amd_power_samples.iter().sum::<f64>() / amd_power_samples.len() as f64;
        energy.gpu_joules = avg_w * meas_wall_s;
        energy.gpu_watts_avg = avg_w;
        energy.gpu_watts_peak = amd_power_samples.iter().copied().fold(0.0f64, f64::max);
        energy.gpu_samples = amd_power_samples.len();
    }

    let flow_results = if args.flow {
        Some(run_gradient_flow_uni(
            (),
            gpu,
            args,
            &rhmc_state,
            &rhmc_config,
            dims,
            seed,
        ))
    } else {
        None
    };

    build_summary(&results, args, beta, budget, energy, flow_results)
}

#[expect(
    clippy::expect_used,
    reason = "GPU trajectory failure is unrecoverable in this pipeline"
)]
pub fn run_quenched_beta(
    gpu: &GpuF64,
    budget: &SiliconBudget,
    amd_power: &AmdGpuPower,
    telemetry: &GpuTelemetry,
    args: &CliArgs,
    beta: f64,
    dims: [usize; 4],
    vol: usize,
    seed: &mut u64,
    out_file: &mut Option<std::io::BufWriter<std::fs::File>>,
    quenched_state: &GpuHmcState,
) -> BetaSummary {
    let l = args.lattice;
    let quenched_pipelines = GpuHmcStreamingPipelines::new_with_tmu(gpu);

    eprintln!(
        "  Thermalizing: {} quenched HMC trajectories (streaming)...",
        args.n_therm
    );
    let mut therm_accepted = 0;
    for i in 0..args.n_therm {
        let r = gpu_hmc_trajectory_streaming(
            gpu,
            &quenched_pipelines,
            quenched_state,
            args.n_md_steps,
            args.dt,
            (args.n_quenched_pretherm + i) as u32,
            seed,
        )
        .expect("streaming HMC trajectory");
        if r.accepted {
            therm_accepted += 1;
        }
        if (i + 1) % 20 == 0 || i == 0 {
            let hw = telemetry.snapshot();
            eprintln!(
                "    therm {}/{}: P={:.6} ΔH={:.4} acc={:.0}%  [{}]",
                i + 1,
                args.n_therm,
                r.plaquette,
                r.delta_h,
                therm_accepted as f64 / (i + 1) as f64 * 100.0,
                hw.status_line()
            );
        }
    }

    let monitor = PowerMonitor::start();
    let meas_start = Instant::now();
    let mut amd_power_samples: Vec<f64> = Vec::new();

    let mut results: Vec<InstrumentedResult> = Vec::new();
    let mut meas_accepted = 0;
    let mut plaq_sum = 0.0;

    for i in 0..args.n_meas {
        if let Some(w) = amd_power.read_watts() {
            amd_power_samples.push(w);
        }

        let traj_id = (args.n_quenched_pretherm + args.n_therm + i) as u32;
        let t0 = Instant::now();
        let r = gpu_hmc_trajectory_streaming(
            gpu,
            &quenched_pipelines,
            quenched_state,
            args.n_md_steps,
            args.dt,
            traj_id,
            seed,
        )
        .expect("streaming HMC trajectory");
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let est_flops = estimate_traj_flops(vol, args.n_md_steps, 0, true);
        let est_gflops = est_flops / 1e9;
        let est_gflop_per_s = est_gflops / (wall_ms / 1000.0);
        let est_bytes = estimate_traj_bytes(vol, args.n_md_steps, 0, true);
        let est_gb_s = (est_bytes / 1e9) / (wall_ms / 1000.0);

        if r.accepted {
            meas_accepted += 1;
        }
        plaq_sum += r.plaquette;

        let ir = InstrumentedResult {
            traj_idx: i,
            accepted: r.accepted,
            delta_h: r.delta_h,
            plaquette: r.plaquette,
            cg_iters: 0,
            wall_ms,
            est_gflops,
            est_gflop_per_s,
            est_gb_s,
        };

        let line = format!(
            "{},{beta:.4},{l},{},{},{},{:.6e},{:.8},{},{:.1},{:.3},{:.1},{:.1},meas",
            args.mode,
            budget.name,
            i,
            i32::from(ir.accepted),
            ir.delta_h,
            ir.plaquette,
            ir.cg_iters,
            ir.wall_ms,
            ir.est_gflops,
            ir.est_gflop_per_s,
            ir.est_gb_s
        );
        println!("{line}");
        if let Some(f) = out_file.as_mut() {
            writeln!(f, "{line}").ok();
            f.flush().ok();
        }

        results.push(ir);

        if (i + 1) % 25 == 0 || i + 1 == args.n_meas {
            let n = (i + 1) as f64;
            let mean_p = plaq_sum / n;
            let rate = meas_accepted as f64 / n * 100.0;
            let hw = telemetry.snapshot();
            eprintln!(
                "    meas {}/{}: ⟨P⟩={:.6} acc={:.0}% {:.0}ms/traj {:.0}GFLOP/s  [{}]",
                i + 1,
                args.n_meas,
                mean_p,
                rate,
                wall_ms,
                est_gflop_per_s,
                hw.status_line()
            );
        }
    }

    let meas_wall_s = meas_start.elapsed().as_secs_f64();
    let mut energy = monitor.stop();

    if energy.gpu_joules < 0.01 && !amd_power_samples.is_empty() {
        let avg_w = amd_power_samples.iter().sum::<f64>() / amd_power_samples.len() as f64;
        energy.gpu_joules = avg_w * meas_wall_s;
        energy.gpu_watts_avg = avg_w;
        energy.gpu_watts_peak = amd_power_samples.iter().copied().fold(0.0f64, f64::max);
        energy.gpu_samples = amd_power_samples.len();
    }

    let flow_results = if args.flow {
        Some(run_quenched_gradient_flow(
            gpu,
            args,
            quenched_state,
            dims,
            seed,
        ))
    } else {
        None
    };

    build_summary(&results, args, beta, budget, energy, flow_results)
}

/// Wrapper: run one unidirectional RHMC trajectory with timing.
#[expect(
    clippy::expect_used,
    reason = "GPU trajectory failure is unrecoverable in this pipeline"
)]
pub fn run_uni_traj(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    rhmc_pipelines: &GpuRhmcPipelines,
    uni_pipelines: &UniPipelines,
    rhmc_state: &GpuRhmcState,
    scg_bufs: &GpuResidentShiftedCgBuffers,
    ms_bufs: Option<&TrueMultiShiftBuffers>,
    ham_bufs: &UniHamiltonianBuffers,
    config: &RhmcConfig,
    seed: &mut u64,
) -> TrajectoryResult {
    let t0 = Instant::now();
    let r = gpu_rhmc_trajectory_unidirectional(
        gpu,
        dyn_pipelines,
        rhmc_pipelines,
        uni_pipelines,
        rhmc_state,
        scg_bufs,
        ms_bufs,
        ham_bufs,
        config,
        seed,
    )
    .expect("unidirectional RHMC trajectory failed");
    TrajectoryResult {
        accepted: r.accepted,
        delta_h: r.delta_h,
        plaquette: r.plaquette,
        total_cg_iterations: r.total_cg_iterations,
        elapsed_secs: t0.elapsed().as_secs_f64(),
    }
}
