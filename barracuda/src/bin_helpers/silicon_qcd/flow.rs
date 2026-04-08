// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-resident gradient flow sweeps and silicon `BetaSummary` assembly.

use super::support::{BetaSummary, CliArgs, FlowSummary, InstrumentedResult, SiliconBudget};
use hotspring_barracuda::bench::EnergyReport;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_flow::{
    FlowReduceBuffers, GpuFlowPipelines, GpuFlowState, gpu_gradient_flow_resident,
};
use hotspring_barracuda::lattice::gpu_hmc::resident_shifted_cg::GpuResidentShiftedCgBuffers;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuDynHmcPipelines, GpuHmcState, GpuHmcStreamingPipelines, GpuRhmcPipelines, GpuRhmcState,
    UniHamiltonianBuffers, UniPipelines, gpu_hmc_trajectory_streaming,
    gpu_rhmc_trajectory_unidirectional,
};
use hotspring_barracuda::lattice::gradient_flow::{find_t0, find_w0, FlowIntegrator};
use hotspring_barracuda::lattice::rhmc::RhmcConfig;
use hotspring_barracuda::production_support::{mean, std_dev};

pub fn run_gradient_flow_uni(
    _uni: (), // placeholder — we use free function directly
    gpu: &GpuF64,
    args: &CliArgs,
    rhmc_state: &GpuRhmcState,
    rhmc_config: &RhmcConfig,
    dims: [usize; 4],
    seed: &mut u64,
) -> FlowSummary {
    eprintln!(
        "  Running gradient flow on {} configs (skip={}, unidirectional)...",
        args.flow_configs, args.flow_skip
    );

    let vol: usize = dims.iter().product();
    let dyn_pipelines = GpuDynHmcPipelines::new(gpu);
    let rhmc_pipelines = GpuRhmcPipelines::new(gpu);
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

    // GPU-resident flow: eliminates B4 (gpu_links_to_lattice) transfer
    let flow_pipelines = GpuFlowPipelines::new(gpu);

    let mut all_t0 = Vec::new();
    let mut all_w0 = Vec::new();

    for cfg_idx in 0..args.flow_configs {
        for _ in 0..args.flow_skip {
            gpu_rhmc_trajectory_unidirectional(
                gpu,
                &dyn_pipelines,
                &rhmc_pipelines,
                &uni_pipelines,
                rhmc_state,
                &scg_bufs,
                None,
                &ham_bufs,
                rhmc_config,
                seed,
            )
            .expect("unidirectional RHMC trajectory failed");
        }

        // GPU-GPU link copy (no PCI-e round-trip) → GPU gradient flow
        let flow_state = GpuFlowState::from_gpu_gauge(gpu, &rhmc_state.gauge.gauge);
        let flow_reduce = FlowReduceBuffers::new(gpu, &flow_pipelines.reduce_pipeline, &flow_state);

        let flow_result = gpu_gradient_flow_resident(
            gpu,
            &flow_pipelines,
            &flow_state,
            &flow_reduce,
            FlowIntegrator::Lscfrk3w7,
            args.flow_epsilon,
            args.flow_t_max,
            1,
        );

        let t0_val = find_t0(&flow_result.measurements);
        let w0_val = find_w0(&flow_result.measurements);
        if let Some(t) = t0_val {
            all_t0.push(t);
        }
        if let Some(w) = w0_val {
            all_w0.push(w);
        }

        eprintln!(
            "    flow cfg {}/{}: t0={} w0={} ({:.1}s GPU flow)",
            cfg_idx + 1,
            args.flow_configs,
            t0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            w0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            flow_result.wall_seconds
        );
    }

    FlowSummary {
        n_configs: args.flow_configs,
        mean_t0: if all_t0.is_empty() {
            None
        } else {
            Some(mean(&all_t0))
        },
        std_t0: if all_t0.len() < 2 {
            None
        } else {
            Some(std_dev(&all_t0))
        },
        mean_w0: if all_w0.is_empty() {
            None
        } else {
            Some(mean(&all_w0))
        },
        std_w0: if all_w0.len() < 2 {
            None
        } else {
            Some(std_dev(&all_w0))
        },
    }
}

pub fn run_quenched_gradient_flow(
    gpu: &GpuF64,
    args: &CliArgs,
    state: &GpuHmcState,
    _dims: [usize; 4],
    seed: &mut u64,
) -> FlowSummary {
    eprintln!(
        "  Running gradient flow on {} quenched configs (skip={})...",
        args.flow_configs, args.flow_skip
    );

    let pipelines = GpuHmcStreamingPipelines::new_with_tmu(gpu);
    let mut all_t0 = Vec::new();
    let mut all_w0 = Vec::new();

    let flow_pipelines = GpuFlowPipelines::new(gpu);

    let beta = args.betas[0]; // current beta from caller context
    let _ = beta; // used implicitly via state.beta
    for cfg_idx in 0..args.flow_configs {
        for s in 0..args.flow_skip {
            let traj_id = (1000 + cfg_idx * args.flow_skip + s) as u32;
            gpu_hmc_trajectory_streaming(
                gpu,
                &pipelines,
                state,
                args.n_md_steps,
                args.dt,
                traj_id,
                seed,
            )
            .expect("streaming HMC trajectory");
        }

        // GPU-GPU link copy → GPU gradient flow (no B4 transfer)
        let flow_state = GpuFlowState::from_gpu_gauge(gpu, state);
        let flow_reduce = FlowReduceBuffers::new(gpu, &flow_pipelines.reduce_pipeline, &flow_state);

        let flow_result = gpu_gradient_flow_resident(
            gpu,
            &flow_pipelines,
            &flow_state,
            &flow_reduce,
            FlowIntegrator::Lscfrk3w7,
            args.flow_epsilon,
            args.flow_t_max,
            1,
        );

        let t0_val = find_t0(&flow_result.measurements);
        let w0_val = find_w0(&flow_result.measurements);
        if let Some(t) = t0_val {
            all_t0.push(t);
        }
        if let Some(w) = w0_val {
            all_w0.push(w);
        }

        eprintln!(
            "    flow cfg {}/{}: t0={} w0={} ({:.1}s GPU flow)",
            cfg_idx + 1,
            args.flow_configs,
            t0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            w0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            flow_result.wall_seconds
        );
    }

    FlowSummary {
        n_configs: args.flow_configs,
        mean_t0: if all_t0.is_empty() {
            None
        } else {
            Some(mean(&all_t0))
        },
        std_t0: if all_t0.len() < 2 {
            None
        } else {
            Some(std_dev(&all_t0))
        },
        mean_w0: if all_w0.is_empty() {
            None
        } else {
            Some(mean(&all_w0))
        },
        std_w0: if all_w0.len() < 2 {
            None
        } else {
            Some(std_dev(&all_w0))
        },
    }
}

pub fn build_summary(
    results: &[InstrumentedResult],
    args: &CliArgs,
    beta: f64,
    budget: &SiliconBudget,
    energy: EnergyReport,
    flow_results: Option<FlowSummary>,
) -> BetaSummary {
    let n = results.len() as f64;
    let plaq_sum: f64 = results.iter().map(|r| r.plaquette).sum();
    let plaq_sq_sum: f64 = results.iter().map(|r| r.plaquette * r.plaquette).sum();
    let mean_plaq = plaq_sum / n;
    let var_plaq = ((plaq_sq_sum / n) - mean_plaq.powi(2)).max(0.0);
    let std_plaq = var_plaq.sqrt();

    let accepted = results.iter().filter(|r| r.accepted).count();
    let acceptance_pct = accepted as f64 / n * 100.0;
    let mean_ms = results.iter().map(|r| r.wall_ms).sum::<f64>() / n;
    let mean_gflop_s = results.iter().map(|r| r.est_gflop_per_s).sum::<f64>() / n;
    let total_gflops: f64 = results.iter().map(|r| r.est_gflops).sum();
    let total_wall_s: f64 = results.iter().map(|r| r.wall_ms).sum::<f64>() / 1000.0;

    let eco_tflops = mean_gflop_s / 1000.0; // GFLOP/s → TFLOP/s
    let silicon_util = eco_tflops / budget.total_tflops * 100.0;
    let quda_util = budget.quda_style_tflops / budget.total_tflops * 100.0;

    let joules_per_traj = if results.is_empty() {
        0.0
    } else {
        (energy.gpu_joules + energy.cpu_joules) / n
    };

    let summary = BetaSummary {
        beta,
        mode: args.mode.clone(),
        lattice: args.lattice,
        gpu_name: budget.name.clone(),
        n_meas: results.len(),
        acceptance_pct,
        mean_plaq,
        std_plaq,
        mean_ms_per_traj: mean_ms,
        mean_gflop_per_s: eco_tflops,
        total_wall_s,
        total_gflops,
        energy,
        joules_per_traj,
        quda_tflops: budget.quda_style_tflops,
        eco_tflops,
        silicon_util_pct: silicon_util,
        quda_util_pct: quda_util,
        flow_results,
    };

    eprintln!();
    eprintln!(
        "  β={beta:.4} summary: ⟨P⟩={mean_plaq:.6}±{std_plaq:.2e} acc={acceptance_pct:.0}% {mean_ms:.0}ms/traj {eco_tflops:.2}TFLOP/s ({silicon_util:.2}% silicon)"
    );
    if summary.energy.gpu_joules > 0.0 {
        eprintln!(
            "  Energy: {:.1}J GPU + {:.1}J CPU = {:.2}J/traj (avg {:.0}W GPU)",
            summary.energy.gpu_joules,
            summary.energy.cpu_joules,
            joules_per_traj,
            summary.energy.gpu_watts_avg
        );
    }
    eprintln!();

    summary
}
