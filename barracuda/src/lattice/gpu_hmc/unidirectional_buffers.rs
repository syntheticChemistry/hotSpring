// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hamiltonian buffer allocation and GPU encode passes for unidirectional RHMC (B2).

use super::dynamical::{GpuDynHmcPipelines, GpuDynHmcState};
use super::gpu_rhmc::GpuRhmcSectorBuffers;
use super::resident_cg_buffers::{ReduceChain, build_reduce_chain_pub, encode_reduce_chain};
use super::unidirectional_pipelines::UniPipelines;
use super::{GpuF64, GpuHmcState, make_u32x4_params};

use crate::lattice::rhmc::RationalApproximation;

/// Maximum rational approximation poles per sector (dots buffer sizing).
pub(crate) const MAX_RATIONAL_POLES: usize = 32;

/// Persistent buffers for GPU-side Hamiltonian reduction.
///
/// Replaces the O(V)-readback pattern (read per-site/per-link arrays + CPU sum)
/// with GPU reduce chains producing single f64 scalars.
///
/// Phase 2 (B2+B3): adds GPU-resident H assembly and Metropolis buffers
/// so the entire Hamiltonian + accept/reject lives on GPU with a single
/// readback at the end of the trajectory.
pub struct UniHamiltonianBuffers {
    /// Reduced plaquette sum → 1 f64.
    pub plaq_sum_buf: wgpu::Buffer,
    /// Reduced kinetic energy → 1 f64.
    pub t_buf: wgpu::Buffer,
    /// Temp scalar for fermion dot products → 1 f64.
    pub temp_dot_buf: wgpu::Buffer,
    /// Reduction scratch (ping).
    pub scratch_a: wgpu::Buffer,
    /// Reduction scratch (pong).
    pub scratch_b: wgpu::Buffer,
    /// Staging for scalar readbacks (MAP_READ).
    pub staging_buf: wgpu::Buffer,
    /// plaq_out_buf (vol entries) → plaq_sum_buf.
    pub reduce_plaq: ReduceChain,
    /// ke_out_buf (n_links entries) → t_buf.
    pub reduce_ke: ReduceChain,
    /// dot_buf (n_pairs entries) → temp_dot_buf.
    pub reduce_dot: ReduceChain,

    // ── B2+B3: GPU-resident Hamiltonian + Metropolis ────────────
    /// Accumulated fermion action S_f (zeroed before each H computation).
    pub s_ferm_buf: wgpu::Buffer,
    /// Dot product results for one sector (up to MAX_RATIONAL_POLES+1 f64s).
    pub dots_buf: wgpu::Buffer,
    /// GPU-resident H_old scalar.
    pub h_old_buf: wgpu::Buffer,
    /// GPU-resident H_new scalar.
    pub h_new_buf: wgpu::Buffer,
    /// Diagnostics for H_old: {s_gauge, t, s_ferm}.
    pub diag_old_buf: wgpu::Buffer,
    /// Diagnostics for H_new: {s_gauge, t, s_ferm}.
    pub diag_new_buf: wgpu::Buffer,
    /// Metropolis result buffer (9 f64s, MAP_READ staging).
    pub metropolis_staging: wgpu::Buffer,
}

impl UniHamiltonianBuffers {
    /// Allocate Hamiltonian reduction buffers.
    #[must_use]
    pub fn new(
        gpu: &GpuF64,
        reduce_pl: &wgpu::ComputePipeline,
        gauge: &GpuHmcState,
        dyn_state: &GpuDynHmcState,
    ) -> Self {
        let vol = gauge.volume;
        let n_links = gauge.n_links;
        let n_pairs = vol * 3;

        let max_wg = [vol, n_links, n_pairs]
            .iter()
            .map(|&n| n.div_ceil(256))
            .max()
            .unwrap_or(1);
        let scratch_a = gpu.create_f64_output_buffer(max_wg, "uni_ham_scratch_a");
        let scratch_b = gpu.create_f64_output_buffer(max_wg, "uni_ham_scratch_b");

        let plaq_sum_buf = gpu.create_f64_output_buffer(1, "uni_plaq_sum");
        let t_buf = gpu.create_f64_output_buffer(1, "uni_t");
        let temp_dot_buf = gpu.create_f64_output_buffer(1, "uni_temp_dot");
        let staging_buf = gpu.create_staging_buffer(512, "uni_staging");

        let reduce_plaq = build_reduce_chain_pub(
            gpu,
            reduce_pl,
            &gauge.plaq_out_buf,
            &scratch_a,
            &scratch_b,
            &plaq_sum_buf,
            vol,
        );
        let reduce_ke = build_reduce_chain_pub(
            gpu,
            reduce_pl,
            &gauge.ke_out_buf,
            &scratch_a,
            &scratch_b,
            &t_buf,
            n_links,
        );
        let reduce_dot = build_reduce_chain_pub(
            gpu,
            reduce_pl,
            &dyn_state.dot_buf,
            &scratch_a,
            &scratch_b,
            &temp_dot_buf,
            n_pairs,
        );

        // B2+B3 buffers
        let s_ferm_buf = gpu.create_f64_output_buffer(1, "uni_s_ferm");
        let dots_buf = gpu.create_f64_output_buffer(MAX_RATIONAL_POLES + 1, "uni_dots");
        let h_old_buf = gpu.create_f64_output_buffer(1, "uni_h_old");
        let h_new_buf = gpu.create_f64_output_buffer(1, "uni_h_new");
        let diag_old_buf = gpu.create_f64_output_buffer(3, "uni_diag_old");
        let diag_new_buf = gpu.create_f64_output_buffer(3, "uni_diag_new");
        let metropolis_staging = gpu.create_staging_buffer(9 * 8, "uni_metro_staging");

        Self {
            plaq_sum_buf,
            t_buf,
            temp_dot_buf,
            scratch_a,
            scratch_b,
            staging_buf,
            reduce_plaq,
            reduce_ke,
            reduce_dot,
            s_ferm_buf,
            dots_buf,
            h_old_buf,
            h_new_buf,
            diag_old_buf,
            diag_new_buf,
            metropolis_staging,
        }
    }
}

/// Encode plaquette dispatch + reduce → plaq_sum_buf (GPU-resident scalar).
pub(crate) fn encode_wilson_plaquette_reduce(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    gauge: &GpuHmcState,
    reduce_pl: &wgpu::ComputePipeline,
    ham: &UniHamiltonianBuffers,
) {
    let params = make_u32x4_params(gauge.volume as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "uni_plaq_p");
    let bg = gpu.create_bind_group(
        &dyn_pipelines.gauge.plaquette_pipeline,
        &[&pbuf, &gauge.link_buf, &gauge.nbr_buf, &gauge.plaq_out_buf],
    );
    GpuF64::encode_pass(
        enc,
        &dyn_pipelines.gauge.plaquette_pipeline,
        &bg,
        gauge.wg_vol,
    );
    encode_reduce_chain(enc, reduce_pl, &ham.reduce_plaq);
}

/// Encode KE dispatch + reduce → t_buf (GPU-resident scalar).
pub(crate) fn encode_kinetic_energy_reduce(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    gauge: &GpuHmcState,
    reduce_pl: &wgpu::ComputePipeline,
    ham: &UniHamiltonianBuffers,
) {
    let params = make_u32x4_params(gauge.n_links as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "uni_ke_p");
    let bg = gpu.create_bind_group(
        &dyn_pipelines.gauge.kinetic_pipeline,
        &[&pbuf, &gauge.mom_buf, &gauge.ke_out_buf],
    );
    GpuF64::encode_pass(
        enc,
        &dyn_pipelines.gauge.kinetic_pipeline,
        &bg,
        gauge.wg_links,
    );
    encode_reduce_chain(enc, reduce_pl, &ham.reduce_ke);
}

/// Encode a dot product ⟨a|b⟩ + reduce → temp_dot_buf (GPU-resident scalar).
pub(crate) fn encode_dot_reduce(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    dyn_state: &GpuDynHmcState,
    reduce_pl: &wgpu::ComputePipeline,
    ham: &UniHamiltonianBuffers,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
) {
    let n_pairs = dyn_state.gauge.volume * 3;
    let wg = (n_pairs as u32).div_ceil(64);
    let params = make_u32x4_params(n_pairs as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "uni_dot_p");
    let bg = gpu.create_bind_group(
        &dyn_pipelines.dot_pipeline,
        &[&pbuf, a, b, &dyn_state.dot_buf],
    );
    GpuF64::encode_pass(enc, &dyn_pipelines.dot_pipeline, &bg, wg);
    encode_reduce_chain(enc, reduce_pl, &ham.reduce_dot);
}

/// Encode fermion dot products for one sector → dots_buf (GPU-only, no readback).
///
/// After this encoder runs, `ham.dots_buf[0..n_dots]` contains the dot products
/// needed for the fermion action weighted sum.
pub(crate) fn encode_fermion_dots_to_gpu(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    dyn_state: &GpuDynHmcState,
    sector: &GpuRhmcSectorBuffers,
    approx: &RationalApproximation,
    reduce_pl: &wgpu::ComputePipeline,
    ham: &UniHamiltonianBuffers,
) {
    // ⟨φ|φ⟩ → reduce → temp_dot → dots_buf[0]
    encode_dot_reduce(
        enc,
        gpu,
        dyn_pipelines,
        dyn_state,
        reduce_pl,
        ham,
        &sector.phi_buf,
        &sector.phi_buf,
    );
    enc.copy_buffer_to_buffer(&ham.temp_dot_buf, 0, &ham.dots_buf, 0, 8);

    // ⟨φ|x_s⟩ → reduce → temp_dot → dots_buf[s+1]
    for (s, _) in approx.alpha.iter().enumerate() {
        encode_dot_reduce(
            enc,
            gpu,
            dyn_pipelines,
            dyn_state,
            reduce_pl,
            ham,
            &sector.phi_buf,
            &sector.x_bufs[s],
        );
        enc.copy_buffer_to_buffer(&ham.temp_dot_buf, 0, &ham.dots_buf, (8 * (s + 1)) as u64, 8);
    }
}

/// Build params for the fermion action sum kernel: [n_dots_as_f64, alpha_0] as raw f64.
pub fn make_fermion_action_sum_params(n_dots: u32, alpha_0: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&(n_dots as f64).to_le_bytes());
    v.extend_from_slice(&alpha_0.to_le_bytes());
    v
}

/// Encode the fermion action weighted sum: S_f += α₀·dots[0] + Σ αₛ·dots[s+1].
///
/// Accumulates into `ham.s_ferm_buf` (caller must zero it before first sector).
pub(crate) fn encode_fermion_action_sum(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    uni_pipelines: &UniPipelines,
    ham: &UniHamiltonianBuffers,
    approx: &RationalApproximation,
) {
    let n_dots = (1 + approx.alpha.len()) as u32;
    let params = make_fermion_action_sum_params(n_dots, approx.alpha_0);
    let pbuf = gpu.create_storage_buffer_init(&params, "uni_sf_sum_p");

    let alpha_data: Vec<u8> = approx.alpha.iter().flat_map(|a| a.to_le_bytes()).collect();
    let alpha_buf = gpu.create_storage_buffer_init(&alpha_data, "uni_sf_alphas");

    let bg = gpu.create_bind_group(
        &uni_pipelines.fermion_action_sum_pipeline,
        &[&pbuf, &ham.dots_buf, &alpha_buf, &ham.s_ferm_buf],
    );
    GpuF64::encode_pass(enc, &uni_pipelines.fermion_action_sum_pipeline, &bg, 1);
}

/// Build params buffer for the Hamiltonian assembly kernel: [beta, 6V] as f64.
pub fn make_h_assembly_params(beta: f64, six_v: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&beta.to_le_bytes());
    v.extend_from_slice(&six_v.to_le_bytes());
    v
}

/// Encode H = beta*(6V - plaq_sum) + T + S_f → h_buf (GPU-only, no readback).
pub(crate) fn encode_hamiltonian_assembly(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    uni_pipelines: &UniPipelines,
    ham: &UniHamiltonianBuffers,
    h_buf: &wgpu::Buffer,
    diag_buf: &wgpu::Buffer,
    beta: f64,
    volume: usize,
) {
    let params = make_h_assembly_params(beta, 6.0 * volume as f64);
    let pbuf = gpu.create_storage_buffer_init(&params, "uni_h_asm_p");
    let bg = gpu.create_bind_group(
        &uni_pipelines.hamiltonian_assembly_pipeline,
        &[
            &pbuf,
            &ham.plaq_sum_buf,
            &ham.t_buf,
            &ham.s_ferm_buf,
            h_buf,
            diag_buf,
        ],
    );
    GpuF64::encode_pass(enc, &uni_pipelines.hamiltonian_assembly_pipeline, &bg, 1);
}
