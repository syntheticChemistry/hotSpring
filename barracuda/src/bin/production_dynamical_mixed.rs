// SPDX-License-Identifier: AGPL-3.0-only
#![recursion_limit = "256"]

//! Production Dynamical + Multi-Head NPU Mixed Pipeline
//!
//! Exp 023: Full dynamical fermion HMC with 9-head NPU offloading for
//! pre/during/post computation screening. Builds on:
//! - Exp 022 (quenched NPU offload pipeline)
//! - `production_dynamical_scan` (resident CG dynamical HMC)
//! - Multi-head ESN (9 readout heads, zero-cost multi-output on Akida)
//!
//! # NPU Offload Architecture (11-head)
//!
//! Pre: β priority, param suggest, CG estimate, quenched length. During: therm, reject, phase.
//! Post: quality, anomaly, next-run.
//!
//! # Usage
//!
//! `HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_dynamical_mixed -- \
//!   --lattice=8 --betas=5.0,5.5,5.69,6.0 --mass=0.1 --therm=200 --meas=500 --seed=42`

use hotspring_barracuda::lattice::gpu_hmc::{
    GpuDynHmcStreamingPipelines, GpuHmcStreamingPipelines, GpuResidentCgPipelines,
};
use hotspring_barracuda::production::{
    dynamical_bootstrap::{
        acquire_dynamical_workers, run_bootstrap, run_post_computation, run_pre_computation,
        spawn_brain_residual_forwarder, DynamicalWorkers,
    },
    dynamical_mixed_pipeline::{
        run_dynamical_mixed_scan, DynamicalMixedConfig, DynamicalMixedScanContext,
    },
    dynamical_summary::{
        create_trajectory_log_writer, hmc_auto_params, print_dynamical_startup_banner,
        print_dynamical_summary, write_dynamical_json, DynamicalBannerConfig, DynamicalNpuStats,
    },
    BetaResult, MetaRow,
};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
//  CLI
// ═══════════════════════════════════════════════════════════════════

struct CliArgs {
    lattice: usize,
    betas: Vec<f64>,
    mass: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    check_interval: usize,
    n_therm: usize,
    n_quenched_pretherm: usize,
    n_meas: usize,
    seed: u64,
    dt_override: Option<f64>,
    n_md_override: Option<usize>,
    output: Option<String>,
    trajectory_log: Option<String>,
    bootstrap_from: Option<String>,
    save_weights: Option<String>,
    no_titan: bool,
    no_npu_control: bool,
    max_adaptive: usize,
    n_fields: usize,
}

fn parse_args() -> CliArgs {
    let mut lattice = 8;
    let mut betas = vec![5.0, 5.5, 5.69, 6.0];
    let mut mass = 0.1;
    let mut cg_tol = 1e-8;
    let mut cg_max_iter = 5000;
    let mut check_interval = 10;
    let mut n_therm = 200;
    let mut n_quenched_pretherm = 50;
    let mut n_meas = 500;
    let mut seed = 42u64;
    let mut dt_override: Option<f64> = None;
    let mut n_md_override: Option<usize> = None;
    let mut output = None;
    let mut trajectory_log = None;
    let mut bootstrap_from = None;
    let mut save_weights = None;
    let mut no_titan = false;
    let mut no_npu_control = false;
    let mut max_adaptive: usize = 12;
    let mut n_fields: usize = 1;

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--lattice=") {
            lattice = val.parse().expect("--lattice=N");
        } else if let Some(val) = arg.strip_prefix("--betas=") {
            betas = val
                .split(',')
                .map(|s| s.parse().expect("beta float"))
                .collect();
        } else if let Some(val) = arg.strip_prefix("--mass=") {
            mass = val.parse().expect("--mass=F");
        } else if let Some(val) = arg.strip_prefix("--cg-tol=") {
            cg_tol = val.parse().expect("--cg-tol=F");
        } else if let Some(val) = arg.strip_prefix("--cg-max-iter=") {
            cg_max_iter = val.parse().expect("--cg-max-iter=N");
        } else if let Some(val) = arg.strip_prefix("--check-interval=") {
            check_interval = val.parse().expect("--check-interval=N");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            n_therm = val.parse().expect("--therm=N");
        } else if let Some(val) = arg.strip_prefix("--quenched-pretherm=") {
            n_quenched_pretherm = val.parse().expect("--quenched-pretherm=N");
        } else if let Some(val) = arg.strip_prefix("--meas=") {
            n_meas = val.parse().expect("--meas=N");
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            seed = val.parse().expect("--seed=N");
        } else if let Some(val) = arg.strip_prefix("--output=") {
            output = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--trajectory-log=") {
            trajectory_log = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--bootstrap-from=") {
            bootstrap_from = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--save-weights=") {
            save_weights = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--dt=") {
            dt_override = Some(val.parse().expect("--dt=F"));
        } else if let Some(val) = arg.strip_prefix("--n-md=") {
            n_md_override = Some(val.parse().expect("--n-md=N"));
        } else if arg == "--no-titan" {
            no_titan = true;
        } else if arg == "--no-npu-control" {
            no_npu_control = true;
        } else if let Some(val) = arg.strip_prefix("--max-adaptive=") {
            max_adaptive = val.parse().expect("--max-adaptive=N");
        } else if let Some(val) = arg.strip_prefix("--n-fields=") {
            n_fields = val.parse().expect("--n-fields=N (Nf = 4*N)");
        }
    }

    CliArgs {
        lattice,
        betas,
        mass,
        cg_tol,
        cg_max_iter,
        check_interval,
        n_therm,
        n_quenched_pretherm,
        n_meas,
        seed,
        dt_override,
        n_md_override,
        output,
        trajectory_log,
        bootstrap_from,
        save_weights,
        no_titan,
        no_npu_control,
        max_adaptive,
        n_fields,
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args = parse_args();
    let dims = [args.lattice, args.lattice, args.lattice, args.lattice];
    let vol: usize = dims.iter().product();

    print_dynamical_startup_banner(&DynamicalBannerConfig {
        lattice: args.lattice,
        vol,
        mass: args.mass,
        betas: &args.betas,
        cg_tol: args.cg_tol,
        cg_max_iter: args.cg_max_iter,
        check_interval: args.check_interval,
        n_therm: args.n_therm,
        n_quenched_pretherm: args.n_quenched_pretherm,
        n_meas: args.n_meas,
        seed: args.seed,
    });

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let Ok(workers) = acquire_dynamical_workers(&rt, args.no_titan, args.lattice) else {
        std::process::exit(1);
    };
    let DynamicalWorkers {
        gpu,
        npu_tx,
        npu_rx,
        brain_interrupt_rx,
        titan_handles,
        cortex_handles,
        npu_available,
    } = workers;

    let meta_context: Vec<MetaRow> =
        run_bootstrap(args.bootstrap_from.as_deref(), &npu_tx, &npu_rx);
    println!();

    let (auto_dt, auto_n_md) = hmc_auto_params(vol);
    let mut dt = args.dt_override.unwrap_or(auto_dt);
    let mut n_md = args.n_md_override.unwrap_or(auto_n_md);

    let nf = args.n_fields * 4;
    println!(
        "  HMC:      dt={dt:.4}, n_md={n_md}, traj_length={:.3}, npu_control={}, Nf={nf} ({} field{})",
        dt * n_md as f64,
        !args.no_npu_control,
        args.n_fields,
        if args.n_fields > 1 { "s" } else { "" },
    );
    println!();

    let quenched_pipelines = GpuHmcStreamingPipelines::new(&gpu);
    let dyn_streaming_pipelines = GpuDynHmcStreamingPipelines::new(&gpu);
    let resident_cg_pipelines = GpuResidentCgPipelines::new(&gpu);

    let brain_residual_tx = spawn_brain_residual_forwarder(npu_tx.clone()).unwrap_or_else(|e| {
        eprintln!("  Brain residual forwarder spawn failed: {e}");
        std::process::exit(1);
    });

    let total_start = Instant::now();
    let mut results: Vec<BetaResult> = Vec::new();
    let mut npu_stats = DynamicalNpuStats::new();

    let mut traj_writer = create_trajectory_log_writer(args.trajectory_log.as_deref())
        .expect("failed to create trajectory log file");

    let mut beta_order = run_pre_computation(
        &args.betas,
        &meta_context,
        args.mass,
        args.lattice,
        &npu_tx,
        &npu_rx,
        &mut npu_stats,
    );

    let config = DynamicalMixedConfig {
        lattice: args.lattice,
        mass: args.mass,
        cg_tol: args.cg_tol,
        cg_max_iter: args.cg_max_iter,
        check_interval: args.check_interval,
        n_therm: args.n_therm,
        n_quenched_pretherm: args.n_quenched_pretherm,
        n_meas: args.n_meas,
        seed: args.seed,
        no_npu_control: args.no_npu_control,
        max_adaptive: args.max_adaptive,
        n_fields: args.n_fields,
    };

    let ctx = DynamicalMixedScanContext {
        gpu: &gpu,
        quenched_pipelines: &quenched_pipelines,
        dyn_streaming_pipelines: &dyn_streaming_pipelines,
        resident_cg_pipelines: &resident_cg_pipelines,
        brain_residual_tx: &brain_residual_tx,
        brain_interrupt_rx: &brain_interrupt_rx,
        npu_tx: &npu_tx,
        npu_rx: &npu_rx,
        titan_handles: titan_handles.as_ref(),
        cortex_handles: &cortex_handles,
    };

    run_dynamical_mixed_scan(
        &config,
        &ctx,
        &meta_context,
        &mut dt,
        &mut n_md,
        &mut beta_order,
        &mut results,
        &mut npu_stats,
        &mut traj_writer,
    );

    run_post_computation(
        &results,
        meta_context,
        args.save_weights.as_deref(),
        titan_handles.as_ref(),
        &mut traj_writer,
        &npu_tx,
        &npu_rx,
        &mut npu_stats,
    );
    drop(cortex_handles.cortex_tx);

    // ═══ Summary ═══
    let total_wall = total_start.elapsed().as_secs_f64();
    results.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap());

    let total_therm_budget: usize = results.iter().map(|r| r.therm_budget).sum();
    let total_therm_used: usize = results.iter().map(|r| r.therm_used).sum();
    let therm_savings_pct = if total_therm_budget > 0 {
        (1.0 - total_therm_used as f64 / total_therm_budget as f64) * 100.0
    } else {
        0.0
    };
    let quenched_savings_pct = if npu_stats.quenched_length_predictions > 0 {
        npu_stats.quenched_steps_saved as f64
            / (npu_stats.quenched_length_predictions as f64 * args.n_quenched_pretherm as f64)
                .max(1.0)
            * 100.0
    } else {
        0.0
    };

    print_dynamical_summary(
        &results,
        &npu_stats,
        args.lattice,
        args.mass,
        total_wall,
        quenched_savings_pct,
        therm_savings_pct,
        &gpu.adapter_name,
        args.trajectory_log.as_deref(),
    );

    if let Some(ref path) = args.output {
        let total_meas: usize = results.iter().map(|r| r.n_traj).sum();
        let npu_name = if npu_available {
            "AKD1000"
        } else {
            "MultiHeadNpu"
        };
        write_dynamical_json(
            path,
            &results,
            args.lattice,
            dims,
            vol,
            args.mass,
            args.cg_tol,
            args.cg_max_iter,
            args.check_interval,
            &gpu.adapter_name,
            npu_name,
            args.n_quenched_pretherm,
            args.n_therm,
            args.n_meas,
            args.seed,
            total_wall,
            total_meas,
            &npu_stats,
            quenched_savings_pct,
            therm_savings_pct,
        );
    }
}
