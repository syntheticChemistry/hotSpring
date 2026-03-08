// SPDX-License-Identifier: AGPL-3.0-only

//! GPU gradient flow — pure-GPU Wilson flow via HMC shader reuse.
//!
//! Promotes the CPU `gradient_flow` module to GPU by reusing the existing
//! SU(3) shader infrastructure:
//!
//! | Operation | Shader | Origin |
//! |-----------|--------|--------|
//! | Force Z = -∂S/∂U | `su3_gauge_force_f64` | HMC |
//! | K = Aᵢ K + Z | `su3_flow_accumulate_f64` | **NEW** |
//! | U = exp(ε Bᵢ K) U | `su3_link_update_f64` | HMC |
//! | Plaquette | `wilson_plaquette_f64` | HMC |
//!
//! Only one new WGSL shader (`su3_flow_accumulate_f64`) is introduced.
//! The rest is HMC infrastructure doing different physics.
//!
//! # LSCFRK 2N-Storage Algorithm (Bazavov & Chuna 2021)
//!
//! For each stage i = 1,...,s:
//!   1. K ← Aᵢ K + F(Yᵢ₋₁)
//!   2. Yᵢ ← exp(ε Bᵢ K) Yᵢ₋₁
//!
//! The force shader computes F (gauge force), the accumulate shader updates K,
//! and the link update shader applies the Cayley exponential.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use super::gpu_hmc::{
    gpu_force_dispatch, make_link_mom_params, make_u32x4_params, GpuHmcPipelines, GpuHmcState,
};
use barracuda::numerical::lscfrk::{self as lscfrk_lib, LscfrkCoefficients};

use super::gradient_flow::{FlowIntegrator, FlowMeasurement};
use super::wilson::Lattice;
use crate::gpu::GpuF64;

const WGSL_FLOW_ACCUMULATE: &str = include_str!("shaders/su3_flow_accumulate_f64.wgsl");

/// GPU gradient flow pipeline set.
pub struct GpuFlowPipelines {
    /// Reused HMC pipelines (force, link update, plaquette).
    pub hmc: GpuHmcPipelines,
    /// Flow accumulation shader: K = α K + Z.
    pub accumulate_pipeline: wgpu::ComputePipeline,
}

impl GpuFlowPipelines {
    /// Compile all flow pipelines.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            hmc: GpuHmcPipelines::new(gpu),
            accumulate_pipeline: gpu.create_pipeline_f64(WGSL_FLOW_ACCUMULATE, "flow_accum"),
        }
    }
}

/// GPU-resident flow state: link buffer + K accumulator.
pub struct GpuFlowState {
    /// Underlying HMC state (links, force, neighbors, plaquette buffers).
    pub hmc: GpuHmcState,
    /// K-buffer for 2N-storage accumulation (same layout as force buffer).
    pub k_buf: wgpu::Buffer,
}

impl GpuFlowState {
    /// Upload a lattice and create GPU-resident flow state.
    #[must_use]
    pub fn from_lattice(gpu: &GpuF64, lattice: &Lattice, beta: f64) -> Self {
        let hmc = GpuHmcState::from_lattice(gpu, lattice, beta);
        let k_buf = gpu.create_f64_output_buffer(hmc.n_links * 18, "flow_k");
        Self { hmc, k_buf }
    }
}

/// Map integrator to 2N-storage (A, B) coefficients for GPU dispatch.
///
/// Caller must validate that `integrator` is not `Rk2` before calling.
fn coeffs_for(integrator: FlowIntegrator) -> LscfrkCoefficients {
    match integrator {
        FlowIntegrator::Euler => LscfrkCoefficients {
            a: &[0.0],
            b: &[1.0],
        },
        FlowIntegrator::Rk2 => unreachable!("RK2 rejected at gpu_gradient_flow entry"),
        FlowIntegrator::Rk3Luscher => lscfrk_lib::LSCFRK3_W6.clone(),
        FlowIntegrator::Lscfrk3w7 => lscfrk_lib::LSCFRK3_W7.clone(),
        FlowIntegrator::Lscfrk4ck => lscfrk_lib::LSCFRK4_CK.clone(),
    }
}

fn gpu_accumulate_dispatch(
    gpu: &GpuF64,
    pipelines: &GpuFlowPipelines,
    state: &GpuFlowState,
    alpha: f64,
) {
    let params = make_link_mom_params(state.hmc.n_links, alpha, gpu.full_df64_mode);
    let param_buf = gpu.create_uniform_buffer(&params, "flow_accum_p");
    let bg = gpu.create_bind_group(
        &pipelines.accumulate_pipeline,
        &[&param_buf, &state.hmc.force_buf, &state.k_buf],
    );
    gpu.dispatch(&pipelines.accumulate_pipeline, &bg, state.hmc.wg_links);
}

fn gpu_flow_link_update_dispatch(
    gpu: &GpuF64,
    pipelines: &GpuFlowPipelines,
    state: &GpuFlowState,
    dt: f64,
) {
    let params = make_link_mom_params(state.hmc.n_links, dt, gpu.full_df64_mode);
    let param_buf = gpu.create_uniform_buffer(&params, "flow_link_p");
    let bg = gpu.create_bind_group(
        &pipelines.hmc.link_pipeline,
        &[&param_buf, &state.k_buf, &state.hmc.link_buf],
    );
    gpu.dispatch(&pipelines.hmc.link_pipeline, &bg, state.hmc.wg_links);
}

fn gpu_flow_plaquette(gpu: &GpuF64, pipelines: &GpuFlowPipelines, state: &GpuFlowState) -> f64 {
    let params = make_u32x4_params(state.hmc.volume as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "flow_plaq_p");
    let bg = gpu.create_bind_group(
        &pipelines.hmc.plaquette_pipeline,
        &[
            &param_buf,
            &state.hmc.link_buf,
            &state.hmc.nbr_buf,
            &state.hmc.plaq_out_buf,
        ],
    );
    gpu.dispatch(&pipelines.hmc.plaquette_pipeline, &bg, state.hmc.wg_vol);
    let Ok(per_site) = gpu.read_back_f64(&state.hmc.plaq_out_buf, state.hmc.volume) else {
        return f64::NAN;
    };
    per_site.iter().sum::<f64>() / (6.0 * state.hmc.volume as f64)
}

fn zero_k_buffer(gpu: &GpuF64, state: &GpuFlowState) {
    let zeros = vec![0.0_f64; state.hmc.n_links * 18];
    gpu.upload_f64(&state.k_buf, &zeros);
}

/// Result of a GPU gradient flow run.
pub struct GpuFlowResult {
    /// Flow measurements at each sampled flow time.
    pub measurements: Vec<FlowMeasurement>,
    /// Total wall time (seconds).
    pub wall_seconds: f64,
}

/// Run Wilson gradient flow on GPU.
///
/// The lattice in `state` is modified in-place on GPU. Measurements are
/// returned at each `measure_interval` steps (and at the final step).
///
/// # Panics
///
/// Panics if `epsilon` is not positive, `t_max` is not positive, or
/// `integrator` is `Rk2` (the midpoint method is not a 2N-storage scheme
/// and cannot run on GPU — use `Rk3Luscher`, `Lscfrk3w7`, or `Lscfrk4ck`).
#[must_use]
pub fn gpu_gradient_flow(
    gpu: &GpuF64,
    pipelines: &GpuFlowPipelines,
    state: &GpuFlowState,
    integrator: FlowIntegrator,
    epsilon: f64,
    t_max: f64,
    measure_interval: usize,
) -> GpuFlowResult {
    assert!(
        !matches!(integrator, FlowIntegrator::Rk2),
        "RK2 midpoint method is not a 2N-storage scheme and cannot run on GPU. \
         Use Rk3Luscher, Lscfrk3w7, or Lscfrk4ck instead."
    );
    assert!(epsilon > 0.0, "epsilon must be positive");
    assert!(t_max > 0.0, "t_max must be positive");

    let start = std::time::Instant::now();
    let n_steps = (t_max / epsilon).round() as usize;
    let coeffs = coeffs_for(integrator);
    let mut measurements = Vec::new();

    let p0 = gpu_flow_plaquette(gpu, pipelines, state);
    let e0 = (1.0 - p0) * 6.0;
    measurements.push(FlowMeasurement {
        t: 0.0,
        energy_density: e0,
        t2_e: 0.0,
        plaquette: p0,
    });

    for step in 1..=n_steps {
        zero_k_buffer(gpu, state);

        for stage in 0..coeffs.a.len() {
            gpu_force_dispatch(gpu, &pipelines.hmc, &state.hmc);
            gpu_accumulate_dispatch(gpu, pipelines, state, coeffs.a[stage]);
            gpu_flow_link_update_dispatch(gpu, pipelines, state, epsilon * coeffs.b[stage]);
        }

        if step % measure_interval == 0 || step == n_steps {
            let t = step as f64 * epsilon;
            let plaq = gpu_flow_plaquette(gpu, pipelines, state);
            let e = (1.0 - plaq) * 6.0;
            measurements.push(FlowMeasurement {
                t,
                energy_density: e,
                t2_e: t * t * e,
                plaquette: plaq,
            });
        }
    }

    GpuFlowResult {
        measurements,
        wall_seconds: start.elapsed().as_secs_f64(),
    }
}

/// Read GPU links back into a CPU lattice (for validation).
pub fn gpu_flow_links_to_lattice(gpu: &GpuF64, state: &GpuFlowState, lattice: &mut Lattice) {
    super::gpu_hmc::gpu_links_to_lattice(gpu, &state.hmc, lattice);
}

// GPU tests live in validation binaries (validate_gpu_gradient_flow)
// rather than #[test] because they require async GPU initialization
// and hardware availability.
