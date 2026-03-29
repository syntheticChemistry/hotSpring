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

use super::cg::WGSL_SUM_REDUCE_F64;
use super::gpu_hmc::resident_cg_buffers::{
    build_reduce_chain_pub, encode_reduce_chain, ReduceChain,
};
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
    /// Sum-reduce for O(1) plaquette readback.
    pub reduce_pipeline: wgpu::ComputePipeline,
}

impl GpuFlowPipelines {
    /// Compile all flow pipelines.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            hmc: GpuHmcPipelines::new(gpu),
            accumulate_pipeline: gpu.create_pipeline_f64(WGSL_FLOW_ACCUMULATE, "flow_accum"),
            reduce_pipeline: gpu.create_pipeline_f64(WGSL_SUM_REDUCE_F64, "flow_reduce"),
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

    /// Create a flow state from an existing GPU gauge field (B4 elimination).
    ///
    /// Copies link and neighbor buffers on GPU (no CPU round-trip). Only
    /// allocates new buffers for force, K-accumulator, and plaquette output.
    /// The GPU-GPU link copy replaces the 37+ MB `gpu_links_to_lattice`
    /// readback that was the B4 bottleneck.
    #[must_use]
    pub fn from_gpu_gauge(gpu: &GpuF64, source: &GpuHmcState) -> Self {
        let n_links = source.n_links;
        let vol = source.volume;
        let link_bytes = (n_links * 18 * 8) as u64;
        let nbr_bytes = (vol * 8 * 4) as u64; // 8 neighbors per site, u32

        let link_buf = gpu.create_f64_output_buffer(n_links * 18, "flow_links");
        let force_buf = gpu.create_f64_output_buffer(n_links * 18, "flow_force");
        let plaq_out_buf = gpu.create_f64_output_buffer(vol, "flow_plaq");
        let ke_out_buf = gpu.create_f64_output_buffer(n_links, "flow_ke");
        let k_buf = gpu.create_f64_output_buffer(n_links * 18, "flow_k");

        // Neighbor table copy (read-only, same topology)
        let nbr_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow_nbr"),
            size: nbr_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // GPU-GPU copy: links + neighbors (no PCI-e round-trip)
        let mut enc = gpu.begin_encoder("flow_gpu_copy");
        enc.copy_buffer_to_buffer(&source.link_buf, 0, &link_buf, 0, link_bytes);
        enc.copy_buffer_to_buffer(&source.nbr_buf, 0, &nbr_buf, 0, nbr_bytes);
        gpu.submit_encoder(enc);

        let hmc = GpuHmcState {
            link_buf,
            link_backup: gpu.create_f64_output_buffer(1, "flow_bk_dummy"),
            mom_buf: gpu.create_f64_output_buffer(1, "flow_mom_dummy"),
            force_buf,
            ke_out_buf,
            plaq_out_buf,
            poly_out_buf: gpu.create_f64_output_buffer(1, "flow_poly_dummy"),
            poly_params_buf: gpu.create_f64_output_buffer(1, "flow_pp_dummy"),
            nbr_buf,
            dims: source.dims,
            volume: source.volume,
            n_links: source.n_links,
            beta: source.beta,
            spatial_vol: source.spatial_vol,
            wg_links: source.wg_links,
            wg_vol: source.wg_vol,
        };
        Self { hmc, k_buf }
    }
}

/// GPU-resident reduce chain for O(1) plaquette readback during flow.
pub struct FlowReduceBuffers {
    /// Reduce chain: plaq_out (vol entries) → plaq_sum_buf (1 scalar).
    pub reduce_plaq: ReduceChain,
    /// Final scalar output.
    pub plaq_sum_buf: wgpu::Buffer,
    /// Scratch A.
    pub scratch_a: wgpu::Buffer,
    /// Scratch B.
    pub scratch_b: wgpu::Buffer,
    /// Staging for readback.
    pub staging: wgpu::Buffer,
}

impl FlowReduceBuffers {
    /// Allocate reduce chain for flow plaquette measurements.
    #[must_use]
    pub fn new(gpu: &GpuF64, reduce_pl: &wgpu::ComputePipeline, state: &GpuFlowState) -> Self {
        let vol = state.hmc.volume;
        let max_wg = vol.div_ceil(256);
        let scratch_a = gpu.create_f64_output_buffer(max_wg, "flow_scratch_a");
        let scratch_b = gpu.create_f64_output_buffer(max_wg, "flow_scratch_b");
        let plaq_sum_buf = gpu.create_f64_output_buffer(1, "flow_plaq_sum");
        let staging = gpu.create_staging_buffer(8, "flow_staging");

        let reduce_plaq = build_reduce_chain_pub(
            gpu,
            reduce_pl,
            &state.hmc.plaq_out_buf,
            &scratch_a,
            &scratch_b,
            &plaq_sum_buf,
            vol,
        );
        Self {
            reduce_plaq,
            plaq_sum_buf,
            scratch_a,
            scratch_b,
            staging,
        }
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
    gpu.zero_buffer(&state.k_buf, (state.hmc.n_links * 18 * 8) as u64);
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

/// O(1) plaquette via GPU reduce chain — single 8-byte readback.
fn gpu_flow_plaquette_reduced(
    gpu: &GpuF64,
    pipelines: &GpuFlowPipelines,
    state: &GpuFlowState,
    reduce: &FlowReduceBuffers,
) -> f64 {
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

    let mut enc = gpu.begin_encoder("flow_plaq_reduce");
    GpuF64::encode_pass(
        &mut enc,
        &pipelines.hmc.plaquette_pipeline,
        &bg,
        state.hmc.wg_vol,
    );
    encode_reduce_chain(&mut enc, &pipelines.reduce_pipeline, &reduce.reduce_plaq);
    enc.copy_buffer_to_buffer(&reduce.plaq_sum_buf, 0, &reduce.staging, 0, 8);
    gpu.submit_encoder(enc);

    gpu.read_staging_f64(&reduce.staging)
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or(f64::NAN)
        / (6.0 * state.hmc.volume as f64)
}

/// Run Wilson gradient flow on GPU with O(1) plaquette readback.
///
/// Equivalent to `gpu_gradient_flow` but uses reduce-chain plaquette
/// measurements (8-byte readback per measurement instead of O(V)).
/// Use `GpuFlowState::from_gpu_gauge` to avoid the B4 link transfer.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn gpu_gradient_flow_resident(
    gpu: &GpuF64,
    pipelines: &GpuFlowPipelines,
    state: &GpuFlowState,
    reduce: &FlowReduceBuffers,
    integrator: FlowIntegrator,
    epsilon: f64,
    t_max: f64,
    measure_interval: usize,
) -> GpuFlowResult {
    assert!(
        !matches!(integrator, FlowIntegrator::Rk2),
        "RK2 midpoint method is not a 2N-storage scheme — use W7 or CK"
    );
    assert!(epsilon > 0.0, "epsilon must be positive");
    assert!(t_max > 0.0, "t_max must be positive");

    let start = std::time::Instant::now();
    let n_steps = (t_max / epsilon).round() as usize;
    let coeffs = coeffs_for(integrator);
    let mut measurements = Vec::new();

    let p0 = gpu_flow_plaquette_reduced(gpu, pipelines, state, reduce);
    let e0 = (1.0 - p0) * 6.0;
    measurements.push(FlowMeasurement {
        t: 0.0,
        energy_density: e0,
        t2_e: 0.0,
        plaquette: p0,
    });

    for step in 1..=n_steps {
        gpu.zero_buffer(&state.k_buf, (state.hmc.n_links * 18 * 8) as u64);

        for stage in 0..coeffs.a.len() {
            gpu_force_dispatch(gpu, &pipelines.hmc, &state.hmc);
            gpu_accumulate_dispatch(gpu, pipelines, state, coeffs.a[stage]);
            gpu_flow_link_update_dispatch(gpu, pipelines, state, epsilon * coeffs.b[stage]);
        }

        if step % measure_interval == 0 || step == n_steps {
            let t = step as f64 * epsilon;
            let plaq = gpu_flow_plaquette_reduced(gpu, pipelines, state, reduce);
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
