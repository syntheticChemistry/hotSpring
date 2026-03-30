// SPDX-License-Identifier: AGPL-3.0-only

//! Dynamical mixed β-scan pipeline orchestration.
//!
//! Extracted from `production_dynamical_mixed` binary to reduce binary size.

mod single_beta;

use crate::lattice::gpu_hmc::{GpuDynHmcStreamingPipelines, GpuHmcStreamingPipelines};
use crate::production::{
    BetaResult, MetaRow,
    cortex_worker::CortexWorkerHandles,
    dynamical_summary::DynamicalNpuStats,
    npu_worker::{NpuRequest, NpuResponse},
    titan_worker::{TitanRequest, TitanWorkerHandles},
};

/// Configuration for a dynamical mixed β-scan (quenched+dynamical+NPU pipeline).
#[derive(Clone, Debug)]
pub struct DynamicalMixedConfig {
    /// Lattice extent L (L⁴ hypercubic).
    pub lattice: usize,
    /// Fermion mass parameter.
    pub mass: f64,
    /// CG solver tolerance.
    pub cg_tol: f64,
    /// CG solver maximum iterations.
    pub cg_max_iter: usize,
    /// Cortex check interval (HMC trajectories between checks).
    pub check_interval: usize,
    /// Number of dynamical thermalization trajectories.
    pub n_therm: usize,
    /// Number of quenched pre-thermalization trajectories.
    pub n_quenched_pretherm: usize,
    /// Number of measurement trajectories per β.
    pub n_meas: usize,
    /// PRNG seed for reproducibility.
    pub seed: u64,
    /// Disable NPU steering (control experiment).
    pub no_npu_control: bool,
    /// Maximum adaptive β-point insertions.
    pub max_adaptive: usize,
    /// Number of staggered fermion fields.
    pub n_fields: usize,
}

/// Runtime context holding GPU pipelines and worker handles for the scan.
pub struct DynamicalMixedScanContext<'a> {
    /// GPU FP64 compute device.
    pub gpu: &'a crate::gpu::GpuF64,
    /// Pre-built quenched HMC streaming pipelines.
    pub quenched_pipelines: &'a GpuHmcStreamingPipelines,
    /// Pre-built dynamical HMC streaming pipelines.
    pub dyn_streaming_pipelines: &'a GpuDynHmcStreamingPipelines,
    /// Pre-built GPU-resident CG pipelines.
    pub resident_cg_pipelines: &'a crate::lattice::gpu_hmc::GpuResidentCgPipelines,
    /// Channel to forward CG residuals to the brain worker.
    pub brain_residual_tx: &'a std::sync::mpsc::Sender<crate::lattice::gpu_hmc::CgResidualUpdate>,
    /// Channel to receive brain interrupt signals.
    pub brain_interrupt_rx: &'a std::sync::mpsc::Receiver<crate::lattice::gpu_hmc::BrainInterrupt>,
    /// Channel to send NPU inference requests.
    pub npu_tx: &'a std::sync::mpsc::Sender<NpuRequest>,
    /// Channel to receive NPU inference responses.
    pub npu_rx: &'a std::sync::mpsc::Receiver<NpuResponse>,
    /// Optional Titan V worker for precision ground truth.
    pub titan_handles: Option<&'a TitanWorkerHandles>,
    /// Cortex worker for proxy pipeline steering.
    pub cortex_handles: &'a CortexWorkerHandles,
}

/// Execute the full dynamical mixed β-scan: iterate over β values with
/// quenched pre-thermalization, dynamical HMC, NPU steering, and adaptive
/// β-point insertion.
pub fn run_dynamical_mixed_scan(
    config: &DynamicalMixedConfig,
    ctx: &DynamicalMixedScanContext<'_>,
    meta_context: &[MetaRow],
    dt: &mut f64,
    n_md: &mut usize,
    beta_order: &mut Vec<f64>,
    results: &mut Vec<BetaResult>,
    npu_stats: &mut DynamicalNpuStats,
    traj_writer: &mut Option<std::io::BufWriter<std::fs::File>>,
) {
    if let Some(handles) = ctx.titan_handles
        && beta_order.len() > 1
    {
        let first_next = beta_order[1];
        handles
            .titan_tx
            .send(TitanRequest::PreThermalize {
                beta: first_next,
                mass: config.mass,
                lattice: config.lattice,
                n_quenched: config.n_quenched_pretherm,
                seed: config.seed + 1500,
                dt: *dt,
                n_md: *n_md,
            })
            .ok();
        println!("  [Brain L2] Titan V pre-thermalizing β={first_next:.4} for scan start");
    }

    println!(
        "═══ Dynamical β-Scan ({} points × {} meas) ═══",
        beta_order.len(),
        config.n_meas,
    );

    let mut bi = 0;
    while bi < beta_order.len() {
        single_beta::run_single_beta(
            config,
            ctx,
            meta_context,
            dt,
            n_md,
            beta_order,
            bi,
            results,
            npu_stats,
            traj_writer,
        );
        bi += 1;
    }
}
