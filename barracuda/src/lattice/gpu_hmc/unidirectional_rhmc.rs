// SPDX-License-Identifier: AGPL-3.0-only

//! Unidirectional GPU RHMC pipeline — zero per-iteration readback.
//!
//! Transforms the RHMC trajectory from ~19,000 CPU-GPU syncs (sync-bound) to
//! ~100 syncs (compute-bound) by:
//!
//! 1. **Resident shifted CG** — batches of ~50 CG iterations per submission,
//!    one 8-byte convergence readback per batch.
//! 2. **GPU-resident Hamiltonian (B2)** — Wilson action, KE, and fermion
//!    action are all reduced and assembled into H on GPU. Zero readback for
//!    either H_old or H_new. Three WGSL kernels: `hamiltonian_assembly_f64`,
//!    `fermion_action_sum_f64`, plus existing reduce chains.
//! 3. **GPU-resident Metropolis (B3)** — accept/reject decision runs as a
//!    single-thread WGSL kernel (`metropolis_f64`) reading H_old/H_new from
//!    GPU buffers. Single 56-byte readback for all trajectory diagnostics.
//! 4. **Async cortex** — `UnidirectionalRhmc` struct with fire/poll API
//!    for non-blocking dual-GPU dispatch.
//!
//! Readback budget: ~100 × 8 bytes (CG convergence) + 56 bytes (Metropolis).
//! This eliminates B2 (CPU Hamiltonian assembly) and B3 (CPU Metropolis)
//! bottlenecks from `SILICON_TIER_ROUTING.md`.

use super::dynamical::{gpu_axpy, GpuDynHmcPipelines, GpuDynHmcState, WGSL_RANDOM_MOMENTA};
use super::gpu_rhmc::{
    dirac_dispatch, fermion_force_dispatch, GpuRhmcPipelines, GpuRhmcResult,
    GpuRhmcSectorBuffers, GpuRhmcState,
};
use super::resident_cg_buffers::{build_reduce_chain_pub, encode_reduce_chain, ReduceChain};
use super::resident_shifted_cg::{
    gpu_multi_shift_cg_solve_resident, GpuResidentShiftedCgBuffers,
    GpuResidentShiftedCgPipelines,
};
use super::true_multishift_cg::{
    gpu_true_multi_shift_cg_solve, TrueMultiShiftBuffers, TrueMultiShiftPipelines,
};
use super::streaming::{make_ferm_prng_params, WGSL_GAUSSIAN_FERMION};
use super::{
    gpu_force_dispatch, gpu_link_update_dispatch, gpu_mom_update_dispatch, make_link_mom_params,
    make_prng_params, make_u32x4_params, GpuF64, GpuHmcState,
};

use crate::lattice::rhmc::{RhmcConfig, RhmcFermionConfig};

// ── WGSL shaders for GPU-resident Hamiltonian + Metropolis (B2+B3) ──
const WGSL_HAMILTONIAN_ASSEMBLY: &str =
    include_str!("../shaders/hamiltonian_assembly_f64.wgsl");
const WGSL_FERMION_ACTION_SUM: &str =
    include_str!("../shaders/fermion_action_sum_f64.wgsl");
const WGSL_METROPOLIS: &str =
    include_str!("../shaders/metropolis_f64.wgsl");

/// Maximum rational approximation poles per sector (dots buffer sizing).
const MAX_RATIONAL_POLES: usize = 32;

// ═══════════════════════════════════════════════════════════════════
//  Pipelines (compiled once per GPU)
// ═══════════════════════════════════════════════════════════════════

/// All pipelines for unidirectional RHMC.
pub struct UniPipelines {
    /// GPU-resident shifted CG pipelines (legacy sequential path).
    pub shifted_cg: GpuResidentShiftedCgPipelines,
    /// True multi-shift CG pipelines (shared Krylov, N_shifts fewer D†D ops).
    pub true_ms_cg: TrueMultiShiftPipelines,
    /// GPU PRNG for SU(3) algebra momenta.
    pub momenta_prng_pipeline: wgpu::ComputePipeline,
    /// GPU Gaussian sampler for pseudofermion heat-bath η.
    pub fermion_prng_pipeline: wgpu::ComputePipeline,
    /// H = beta*(6V - plaq) + T + S_f  (single-thread GPU kernel).
    pub hamiltonian_assembly_pipeline: wgpu::ComputePipeline,
    /// S_f weighted sum from dot products + alpha coefficients.
    pub fermion_action_sum_pipeline: wgpu::ComputePipeline,
    /// Metropolis accept/reject + diagnostics (single-thread GPU kernel).
    pub metropolis_pipeline: wgpu::ComputePipeline,
}

impl UniPipelines {
    /// Compile all unidirectional RHMC pipelines.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            shifted_cg: GpuResidentShiftedCgPipelines::new(gpu),
            true_ms_cg: TrueMultiShiftPipelines::new(gpu),
            momenta_prng_pipeline: gpu.create_pipeline_f64(&WGSL_RANDOM_MOMENTA, "uni_mom_prng"),
            fermion_prng_pipeline: gpu
                .create_pipeline_f64(&WGSL_GAUSSIAN_FERMION, "uni_ferm_prng"),
            hamiltonian_assembly_pipeline: gpu
                .create_pipeline_f64(WGSL_HAMILTONIAN_ASSEMBLY, "uni_h_asm"),
            fermion_action_sum_pipeline: gpu
                .create_pipeline_f64(WGSL_FERMION_ACTION_SUM, "uni_sf_sum"),
            metropolis_pipeline: gpu
                .create_pipeline_f64(WGSL_METROPOLIS, "uni_metropolis"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  GPU-resident Hamiltonian buffers + reduce chains (Phase 2)
// ═══════════════════════════════════════════════════════════════════

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
fn encode_wilson_plaquette_reduce(
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
    GpuF64::encode_pass(enc, &dyn_pipelines.gauge.plaquette_pipeline, &bg, gauge.wg_vol);
    encode_reduce_chain(enc, reduce_pl, &ham.reduce_plaq);
}

/// Encode KE dispatch + reduce → t_buf (GPU-resident scalar).
fn encode_kinetic_energy_reduce(
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
    GpuF64::encode_pass(enc, &dyn_pipelines.gauge.kinetic_pipeline, &bg, gauge.wg_links);
    encode_reduce_chain(enc, reduce_pl, &ham.reduce_ke);
}

/// Encode a dot product ⟨a|b⟩ + reduce → temp_dot_buf (GPU-resident scalar).
fn encode_dot_reduce(
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

// ═══════════════════════════════════════════════════════════════════
//  B2+B3: GPU-resident Hamiltonian assembly + Metropolis
// ═══════════════════════════════════════════════════════════════════

/// Encode fermion dot products for one sector → dots_buf (GPU-only, no readback).
///
/// After this encoder runs, `ham.dots_buf[0..n_dots]` contains the dot products
/// needed for the fermion action weighted sum.
fn encode_fermion_dots_to_gpu(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    dyn_state: &GpuDynHmcState,
    sector: &GpuRhmcSectorBuffers,
    approx: &crate::lattice::rhmc::RationalApproximation,
    reduce_pl: &wgpu::ComputePipeline,
    ham: &UniHamiltonianBuffers,
) {
    // ⟨φ|φ⟩ → reduce → temp_dot → dots_buf[0]
    encode_dot_reduce(enc, gpu, dyn_pipelines, dyn_state, reduce_pl, ham,
        &sector.phi_buf, &sector.phi_buf);
    enc.copy_buffer_to_buffer(&ham.temp_dot_buf, 0, &ham.dots_buf, 0, 8);

    // ⟨φ|x_s⟩ → reduce → temp_dot → dots_buf[s+1]
    for (s, _) in approx.alpha.iter().enumerate() {
        encode_dot_reduce(enc, gpu, dyn_pipelines, dyn_state, reduce_pl, ham,
            &sector.phi_buf, &sector.x_bufs[s]);
        enc.copy_buffer_to_buffer(&ham.temp_dot_buf, 0, &ham.dots_buf, (8 * (s + 1)) as u64, 8);
    }
}

/// Build params for the fermion action sum kernel: [n_dots_as_f64, alpha_0] as raw f64.
fn make_fermion_action_sum_params(n_dots: u32, alpha_0: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&(n_dots as f64).to_le_bytes());
    v.extend_from_slice(&alpha_0.to_le_bytes());
    v
}

/// Encode the fermion action weighted sum: S_f += α₀·dots[0] + Σ αₛ·dots[s+1].
///
/// Accumulates into `ham.s_ferm_buf` (caller must zero it before first sector).
fn encode_fermion_action_sum(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    uni_pipelines: &UniPipelines,
    ham: &UniHamiltonianBuffers,
    approx: &crate::lattice::rhmc::RationalApproximation,
) {
    let n_dots = (1 + approx.alpha.len()) as u32;
    let params = make_fermion_action_sum_params(n_dots, approx.alpha_0);
    let pbuf = gpu.create_storage_buffer_init(&params, "uni_sf_sum_p");

    let alpha_data: Vec<u8> = approx.alpha.iter()
        .flat_map(|a| a.to_le_bytes())
        .collect();
    let alpha_buf = gpu.create_storage_buffer_init(&alpha_data, "uni_sf_alphas");

    let bg = gpu.create_bind_group(
        &uni_pipelines.fermion_action_sum_pipeline,
        &[&pbuf, &ham.dots_buf, &alpha_buf, &ham.s_ferm_buf],
    );
    GpuF64::encode_pass(enc, &uni_pipelines.fermion_action_sum_pipeline, &bg, 1);
}

/// Build params buffer for the Hamiltonian assembly kernel: [beta, 6V] as f64.
fn make_h_assembly_params(beta: f64, six_v: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&beta.to_le_bytes());
    v.extend_from_slice(&six_v.to_le_bytes());
    v
}

/// Encode H = beta*(6V - plaq_sum) + T + S_f → h_buf (GPU-only, no readback).
fn encode_hamiltonian_assembly(
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
        &[&pbuf, &ham.plaq_sum_buf, &ham.t_buf, &ham.s_ferm_buf, h_buf, diag_buf],
    );
    GpuF64::encode_pass(enc, &uni_pipelines.hamiltonian_assembly_pipeline, &bg, 1);
}

/// Compute full Hamiltonian on GPU: gauge+KE reduces, fermion CG+dots, assembly.
///
/// Returns the number of CG iterations used. The result lives in `h_buf` and
/// `diag_buf` on GPU — no readback.
#[allow(clippy::too_many_arguments)]
fn compute_h_gpu(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    uni_pipelines: &UniPipelines,
    state: &GpuRhmcState,
    scg_bufs: &GpuResidentShiftedCgBuffers,
    ms_bufs: Option<&TrueMultiShiftBuffers>,
    ham: &UniHamiltonianBuffers,
    config: &RhmcConfig,
    h_buf: &wgpu::Buffer,
    diag_buf: &wgpu::Buffer,
) -> usize {
    let gauge = &state.gauge.gauge;
    let reduce_pl = &uni_pipelines.shifted_cg.base.reduce_pipeline;

    // Zero the fermion action accumulator
    gpu.zero_buffer(&ham.s_ferm_buf, 8);

    // Gauge + KE reduces (GPU-only, no readback)
    {
        let mut enc = gpu.begin_encoder("b2_gauge_ke");
        encode_wilson_plaquette_reduce(&mut enc, gpu, dyn_pipelines, gauge, reduce_pl, ham);
        encode_kinetic_energy_reduce(&mut enc, gpu, dyn_pipelines, gauge, reduce_pl, ham);
        gpu.submit_encoder(enc);
    }

    // Per-sector: CG solve + fermion dots + weighted sum (GPU-only)
    let mut total_cg = 0usize;
    for (sector, fconfig) in state.sectors.iter().zip(config.sectors.iter()) {
        let cg = if let Some(ms) = ms_bufs {
            if fconfig.action_approx.sigma.len() <= ms.n_shifts {
                gpu_true_multi_shift_cg_solve(
                    gpu, dyn_pipelines, &uni_pipelines.true_ms_cg,
                    &state.gauge, ms, &sector.x_bufs, &sector.phi_buf,
                    &fconfig.action_approx.sigma, config.cg_tol,
                    config.cg_max_iter, CG_CHECK_INTERVAL,
                )
            } else {
                gpu_multi_shift_cg_solve_resident(
                    gpu, dyn_pipelines, &uni_pipelines.shifted_cg,
                    &state.gauge, scg_bufs, &sector.x_bufs, &sector.phi_buf,
                    &fconfig.action_approx.sigma, config.cg_tol,
                    config.cg_max_iter, CG_CHECK_INTERVAL,
                )
            }
        } else {
            gpu_multi_shift_cg_solve_resident(
                gpu, dyn_pipelines, &uni_pipelines.shifted_cg,
                &state.gauge, scg_bufs, &sector.x_bufs, &sector.phi_buf,
                &fconfig.action_approx.sigma, config.cg_tol,
                config.cg_max_iter, CG_CHECK_INTERVAL,
            )
        };
        total_cg += cg;

        let mut enc = gpu.begin_encoder("b2_sf_dots");
        encode_fermion_dots_to_gpu(
            &mut enc, gpu, dyn_pipelines, &state.gauge, sector,
            &fconfig.action_approx, reduce_pl, ham,
        );
        encode_fermion_action_sum(
            &mut enc, gpu, uni_pipelines, ham, &fconfig.action_approx,
        );
        gpu.submit_encoder(enc);
    }

    // H = beta*(6V - plaq) + T + S_f  (GPU-only, no readback)
    {
        let mut enc = gpu.begin_encoder("b2_h_asm");
        encode_hamiltonian_assembly(
            &mut enc, gpu, uni_pipelines, ham,
            h_buf, diag_buf, gauge.beta, gauge.volume,
        );
        gpu.submit_encoder(enc);
    }

    total_cg
}

/// Build params for the Metropolis kernel: [rand, six_v] as raw f64.
fn make_metropolis_params(seed: &mut u64, six_v: f64) -> Vec<u8> {
    let r: f64 = crate::lattice::constants::lcg_uniform_f64(seed);
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&r.to_le_bytes());
    v.extend_from_slice(&six_v.to_le_bytes());
    v
}

/// GPU Metropolis: compute delta_H, accept/reject, write diagnostics.
///
/// Returns `GpuRhmcResult` from a single 56-byte readback.
fn gpu_metropolis(
    gpu: &GpuF64,
    uni_pipelines: &UniPipelines,
    ham: &UniHamiltonianBuffers,
    gauge: &GpuHmcState,
    total_cg: usize,
    seed: &mut u64,
) -> GpuRhmcResult {
    let six_v = 6.0 * gauge.volume as f64;

    let mut enc = gpu.begin_encoder("b3_metropolis");
    let params = make_metropolis_params(seed, six_v);
    let pbuf = gpu.create_storage_buffer_init(&params, "uni_metro_p");

    let result_buf = gpu.create_f64_output_buffer(9, "uni_metro_result");

    let bg = gpu.create_bind_group(
        &uni_pipelines.metropolis_pipeline,
        &[&pbuf, &ham.h_old_buf, &ham.h_new_buf, &ham.plaq_sum_buf,
          &ham.diag_old_buf, &ham.diag_new_buf, &result_buf],
    );
    GpuF64::encode_pass(&mut enc, &uni_pipelines.metropolis_pipeline, &bg, 1);
    enc.copy_buffer_to_buffer(&result_buf, 0, &ham.metropolis_staging, 0, 9 * 8);
    gpu.submit_encoder(enc);

    // Single readback: 72 bytes (9 f64s)
    let data = gpu
        .read_staging_f64_n(&ham.metropolis_staging, 9)
        .unwrap_or_else(|_| vec![0.0; 9]);

    let accepted = data[0] > 0.5;

    if !accepted {
        let n_links = gauge.n_links;
        let mut enc = gpu.begin_encoder("uni_restore");
        enc.copy_buffer_to_buffer(
            &gauge.link_backup, 0,
            &gauge.link_buf, 0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    GpuRhmcResult {
        accepted,
        delta_h: data[1],
        plaquette: data[2],
        total_cg_iterations: total_cg,
        s_gauge_old: data[3],
        s_gauge_new: data[4],
        t_old: data[5],
        t_new: data[6],
        s_ferm_old: data[7],
        s_ferm_new: data[8],
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Legacy CPU-side Hamiltonian helpers (kept for validation tooling)
// ═══════════════════════════════════════════════════════════════════

/// Compute gauge action + KE via GPU reduce, read back 16 bytes.
///
/// Returns `(plaq_sum, kinetic_energy)`. Caller computes
/// `S_gauge = beta * (6*V - plaq_sum)`.
///
/// Superseded by `compute_h_gpu` which avoids these scalar readbacks.
#[allow(dead_code)]
fn compute_gauge_ke_resident(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    gauge: &GpuHmcState,
    reduce_pl: &wgpu::ComputePipeline,
    ham: &UniHamiltonianBuffers,
) -> (f64, f64) {
    let mut enc = gpu.begin_encoder("uni_gauge_ke");
    encode_wilson_plaquette_reduce(&mut enc, gpu, dyn_pipelines, gauge, reduce_pl, ham);
    encode_kinetic_energy_reduce(&mut enc, gpu, dyn_pipelines, gauge, reduce_pl, ham);
    enc.copy_buffer_to_buffer(&ham.plaq_sum_buf, 0, &ham.staging_buf, 0, 8);
    enc.copy_buffer_to_buffer(&ham.t_buf, 0, &ham.staging_buf, 8, 8);
    gpu.submit_encoder(enc);

    let data = gpu.read_staging_f64_n(&ham.staging_buf, 2).unwrap_or_else(|_| vec![0.0; 2]);
    (data[0], data[1])
}

/// Compute fermion action for one sector: `S_f = α₀·⟨φ|φ⟩ + Σ αₛ·⟨φ|x_s⟩`.
///
/// All dot products encode in one encoder. Single readback of (1+n_shifts) f64s.
///
/// Superseded by `encode_fermion_dots_to_gpu` + `encode_fermion_action_sum` which
/// keep the result on GPU without readback.
#[allow(dead_code)]
fn compute_fermion_action_resident(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    dyn_state: &GpuDynHmcState,
    sector: &GpuRhmcSectorBuffers,
    approx: &crate::lattice::rhmc::RationalApproximation,
    reduce_pl: &wgpu::ComputePipeline,
    ham: &UniHamiltonianBuffers,
) -> f64 {
    let n_dots = 1 + approx.alpha.len();
    let mut enc = gpu.begin_encoder("uni_sf_dots");

    // ⟨φ|φ⟩ → reduce → temp_dot → staging[0]
    encode_dot_reduce(
        &mut enc,
        gpu,
        dyn_pipelines,
        dyn_state,
        reduce_pl,
        ham,
        &sector.phi_buf,
        &sector.phi_buf,
    );
    enc.copy_buffer_to_buffer(&ham.temp_dot_buf, 0, &ham.staging_buf, 0, 8);

    // ⟨φ|x_s⟩ → reduce → temp_dot → staging[8*(s+1)]
    for (s, _) in approx.alpha.iter().enumerate() {
        encode_dot_reduce(
            &mut enc,
            gpu,
            dyn_pipelines,
            dyn_state,
            reduce_pl,
            ham,
            &sector.phi_buf,
            &sector.x_bufs[s],
        );
        enc.copy_buffer_to_buffer(
            &ham.temp_dot_buf,
            0,
            &ham.staging_buf,
            (8 * (s + 1)) as u64,
            8,
        );
    }

    gpu.submit_encoder(enc);

    let data = gpu
        .read_staging_f64_n(&ham.staging_buf, n_dots)
        .unwrap_or_else(|_| vec![0.0; n_dots]);

    let phi_phi = data[0];
    let mut action = approx.alpha_0 * phi_phi;
    for (s, a_s) in approx.alpha.iter().enumerate() {
        action += a_s * data[s + 1];
    }

    action
}

// ═══════════════════════════════════════════════════════════════════
//  Full unidirectional RHMC trajectory (Phase 4)
// ═══════════════════════════════════════════════════════════════════

/// CG check interval — iterations per batch before convergence readback.
const CG_CHECK_INTERVAL: usize = 50;

/// Run one RHMC trajectory using the fully GPU-resident pipeline (B2+B3).
///
/// All CG solves use GPU-resident shifted CG (zero per-iteration readback).
/// Hamiltonian assembly (H_old, H_new) is computed entirely on GPU — no
/// scalar readbacks for plaquette, KE, or fermion action components.
/// Metropolis accept/reject runs as a GPU kernel.
///
/// Total readback budget: ~100 × 8 bytes (CG convergence) + 56 bytes (Metropolis).
/// This eliminates 4+ sync points vs the prior CPU-assembly path.
#[allow(clippy::too_many_arguments)]
pub fn gpu_rhmc_trajectory_unidirectional(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    _rhmc_pipelines: &GpuRhmcPipelines,
    uni_pipelines: &UniPipelines,
    state: &GpuRhmcState,
    scg_bufs: &GpuResidentShiftedCgBuffers,
    ms_bufs: Option<&TrueMultiShiftBuffers>,
    ham_bufs: &UniHamiltonianBuffers,
    config: &RhmcConfig,
    seed: &mut u64,
) -> GpuRhmcResult {
    let gauge = &state.gauge.gauge;
    let n_links = gauge.n_links;
    let n_md_steps = config.n_md_steps;
    let dt = config.dt;

    // ── 1. GPU PRNG momenta + pseudofermion η + backup links ────
    {
        let mut enc = gpu.begin_encoder("uni_init");

        let mom_prng_params = make_prng_params(n_links as u32, 0, seed);
        let mom_prng_pbuf = gpu.create_uniform_buffer(&mom_prng_params, "uni_mom_p");
        let mom_prng_bg = gpu.create_bind_group(
            &uni_pipelines.momenta_prng_pipeline,
            &[&mom_prng_pbuf, &gauge.mom_buf],
        );
        GpuF64::encode_pass(
            &mut enc,
            &uni_pipelines.momenta_prng_pipeline,
            &mom_prng_bg,
            gauge.wg_links,
        );

        enc.copy_buffer_to_buffer(
            &gauge.link_buf, 0,
            &gauge.link_backup, 0,
            (n_links * 18 * 8) as u64,
        );

        gpu.submit_encoder(enc);
    }

    // ── 2. RHMC heatbath: generate φ for each sector ─────────────
    let mut total_cg: usize = 0;
    for (si, (sector, fconfig)) in state.sectors.iter().zip(config.sectors.iter()).enumerate() {
        let cg = uni_heatbath_sector(
            gpu, dyn_pipelines, uni_pipelines, &state.gauge,
            sector, fconfig, scg_bufs, ms_bufs,
            config.cg_tol, config.cg_max_iter, seed, si,
        );
        total_cg += cg;
    }

    // ── 3. H_old (GPU-resident, zero readback) ──────────────────
    total_cg += compute_h_gpu(
        gpu, dyn_pipelines, uni_pipelines, state, scg_bufs, ms_bufs,
        ham_bufs, config, &ham_bufs.h_old_buf, &ham_bufs.diag_old_buf,
    );

    // ── 4. Omelyan MD integration ────────────────────────────────
    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for _step in 0..n_md_steps {
        let cg = uni_total_force_dispatch(
            gpu, dyn_pipelines, uni_pipelines,
            &state.gauge, &state.sectors, &config.sectors,
            scg_bufs, ms_bufs, lam * dt, config.cg_tol, config.cg_max_iter,
        );
        total_cg += cg;
        gpu_link_update_dispatch(gpu, &dyn_pipelines.gauge, gauge, 0.5 * dt);

        let cg = uni_total_force_dispatch(
            gpu, dyn_pipelines, uni_pipelines,
            &state.gauge, &state.sectors, &config.sectors,
            scg_bufs, ms_bufs, 2.0f64.mul_add(-lam, 1.0) * dt,
            config.cg_tol, config.cg_max_iter,
        );
        total_cg += cg;
        gpu_link_update_dispatch(gpu, &dyn_pipelines.gauge, gauge, 0.5 * dt);

        let cg = uni_total_force_dispatch(
            gpu, dyn_pipelines, uni_pipelines,
            &state.gauge, &state.sectors, &config.sectors,
            scg_bufs, ms_bufs, lam * dt, config.cg_tol, config.cg_max_iter,
        );
        total_cg += cg;
    }

    // ── 5. H_new (GPU-resident, zero readback) ──────────────────
    total_cg += compute_h_gpu(
        gpu, dyn_pipelines, uni_pipelines, state, scg_bufs, ms_bufs,
        ham_bufs, config, &ham_bufs.h_new_buf, &ham_bufs.diag_new_buf,
    );

    // ── 6. GPU Metropolis + single 56-byte readback ─────────────
    gpu_metropolis(gpu, uni_pipelines, ham_bufs, gauge, total_cg, seed)
}

// ═══════════════════════════════════════════════════════════════════
//  Heatbath + force helpers (internal)
// ═══════════════════════════════════════════════════════════════════

/// RHMC heatbath for one sector using GPU PRNG + multi-shift CG.
#[allow(clippy::too_many_arguments)]
fn uni_heatbath_sector(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    uni_pipelines: &UniPipelines,
    state: &GpuDynHmcState,
    sector: &GpuRhmcSectorBuffers,
    config: &RhmcFermionConfig,
    scg_bufs: &GpuResidentShiftedCgBuffers,
    ms_bufs: Option<&TrueMultiShiftBuffers>,
    cg_tol: f64,
    cg_max_iter: usize,
    seed: &mut u64,
    sector_idx: usize,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;
    let approx = &config.heatbath_approx;

    // GPU PRNG: Gaussian noise η → phi_buf
    {
        let wg_vol = (vol as u32).div_ceil(64);
        let ferm_params = make_ferm_prng_params(vol as u32, sector_idx as u32 * 1000, seed);
        let ferm_pbuf = gpu.create_uniform_buffer(&ferm_params, "uni_ferm_p");
        let ferm_bg = gpu.create_bind_group(
            &uni_pipelines.fermion_prng_pipeline,
            &[&ferm_pbuf, &sector.phi_buf],
        );
        gpu.dispatch(&uni_pipelines.fermion_prng_pipeline, &ferm_bg, wg_vol);
    }

    // Multi-shift CG: (D†D + σ_s) x_s = η
    let cg_iters = if let Some(ms) = ms_bufs {
        if approx.sigma.len() <= ms.n_shifts {
            gpu_true_multi_shift_cg_solve(
                gpu, dyn_pipelines, &uni_pipelines.true_ms_cg,
                state, ms, &sector.x_bufs, &sector.phi_buf,
                &approx.sigma, cg_tol, cg_max_iter, CG_CHECK_INTERVAL,
            )
        } else {
            gpu_multi_shift_cg_solve_resident(
                gpu, dyn_pipelines, &uni_pipelines.shifted_cg,
                state, scg_bufs, &sector.x_bufs, &sector.phi_buf,
                &approx.sigma, cg_tol, cg_max_iter, CG_CHECK_INTERVAL,
            )
        }
    } else {
        gpu_multi_shift_cg_solve_resident(
            gpu, dyn_pipelines, &uni_pipelines.shifted_cg,
            state, scg_bufs, &sector.x_bufs, &sector.phi_buf,
            &approx.sigma, cg_tol, cg_max_iter, CG_CHECK_INTERVAL,
        )
    };

    // Accumulate: temp = α₀·η + Σ αₛ·x_s → phi_buf
    gpu.zero_buffer(&state.x_buf, (n_flat * 8) as u64);
    gpu_axpy(
        gpu,
        &dyn_pipelines.axpy_pipeline,
        approx.alpha_0,
        &sector.phi_buf,
        &state.x_buf,
        n_flat,
    );
    for (s, a_s) in approx.alpha.iter().enumerate() {
        gpu_axpy(
            gpu,
            &dyn_pipelines.axpy_pipeline,
            *a_s,
            &sector.x_bufs[s],
            &state.x_buf,
            n_flat,
        );
    }
    {
        let mut enc = gpu.begin_encoder("uni_phi_copy");
        enc.copy_buffer_to_buffer(&state.x_buf, 0, &sector.phi_buf, 0, (n_flat * 8) as u64);
        gpu.submit_encoder(enc);
    }

    cg_iters
}

/// RHMC total force: gauge + Σ_sectors Σ_poles fermion force.
///
/// Uses true multi-shift CG (shared Krylov) when `ms_bufs` is available,
/// otherwise falls back to sequential shifted CG.
#[allow(clippy::too_many_arguments)]
fn uni_total_force_dispatch(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    uni_pipelines: &UniPipelines,
    state: &GpuDynHmcState,
    sectors: &[GpuRhmcSectorBuffers],
    configs: &[RhmcFermionConfig],
    scg_bufs: &GpuResidentShiftedCgBuffers,
    ms_bufs: Option<&TrueMultiShiftBuffers>,
    dt: f64,
    cg_tol: f64,
    cg_max_iter: usize,
) -> usize {
    let gauge = &state.gauge;
    let n_links = gauge.n_links;

    // Gauge force → P += dt · F_gauge
    gpu_force_dispatch(gpu, &dyn_pipelines.gauge, gauge);
    gpu_mom_update_dispatch(gpu, &dyn_pipelines.gauge, gauge, dt);

    let mut total_cg = 0;

    for (sector, fconfig) in sectors.iter().zip(configs.iter()) {
        let approx = &fconfig.force_approx;

        let cg = if let Some(ms) = ms_bufs {
            if approx.sigma.len() <= ms.n_shifts {
                gpu_true_multi_shift_cg_solve(
                    gpu, dyn_pipelines, &uni_pipelines.true_ms_cg,
                    state, ms, &sector.x_bufs, &sector.phi_buf,
                    &approx.sigma, cg_tol, cg_max_iter, CG_CHECK_INTERVAL,
                )
            } else {
                gpu_multi_shift_cg_solve_resident(
                    gpu, dyn_pipelines, &uni_pipelines.shifted_cg,
                    state, scg_bufs, &sector.x_bufs, &sector.phi_buf,
                    &approx.sigma, cg_tol, cg_max_iter, CG_CHECK_INTERVAL,
                )
            }
        } else {
            gpu_multi_shift_cg_solve_resident(
                gpu, dyn_pipelines, &uni_pipelines.shifted_cg,
                state, scg_bufs, &sector.x_bufs, &sector.phi_buf,
                &approx.sigma, cg_tol, cg_max_iter, CG_CHECK_INTERVAL,
            )
        };
        total_cg += cg;

        // Per-pole: fermion force from x_s
        for (s, a_s) in approx.alpha.iter().enumerate() {
            {
                let n_flat = gauge.volume * 6;
                let mut enc = gpu.begin_encoder("uni_xcopy");
                enc.copy_buffer_to_buffer(
                    &sector.x_bufs[s],
                    0,
                    &state.x_buf,
                    0,
                    (n_flat * 8) as u64,
                );
                gpu.submit_encoder(enc);
            }

            dirac_dispatch(
                gpu,
                dyn_pipelines,
                gauge,
                &state.phases_buf,
                &state.x_buf,
                &state.y_buf,
                fconfig.mass,
                1.0,
            );
            fermion_force_dispatch(
                gpu,
                dyn_pipelines,
                gauge,
                &state.phases_buf,
                &state.x_buf,
                &state.y_buf,
                &state.ferm_force_buf,
            );

            let ferm_mom_params =
                make_link_mom_params(n_links, *a_s * dt, gpu.full_df64_mode);
            let ferm_mom_pbuf = gpu.create_uniform_buffer(&ferm_mom_params, "uni_fmom_p");
            let ferm_mom_bg = gpu.create_bind_group(
                &dyn_pipelines.gauge.momentum_pipeline,
                &[&ferm_mom_pbuf, &state.ferm_force_buf, &gauge.mom_buf],
            );
            gpu.dispatch(
                &dyn_pipelines.gauge.momentum_pipeline,
                &ferm_mom_bg,
                gauge.wg_links,
            );
        }
    }

    total_cg
}

// ═══════════════════════════════════════════════════════════════════
//  Async CPU cortex (Phase 5)
// ═══════════════════════════════════════════════════════════════════

/// Result from one trajectory — the only data CPU needs from the GPU.
#[derive(Debug, Clone)]
pub struct TrajectoryResult {
    /// Metropolis accept/reject.
    pub accepted: bool,
    /// ΔH = H_new - H_old.
    pub delta_h: f64,
    /// Mean plaquette (post-trajectory).
    pub plaquette: f64,
    /// Total CG iterations across all CG solves in the trajectory.
    pub total_cg_iterations: usize,
    /// Wall-clock time for this trajectory.
    pub elapsed_secs: f64,
}

/// Unidirectional RHMC handle for one GPU.
///
/// Owns the GPU, pipelines, state, and buffers. Provides `fire`/`poll`
/// for async trajectory dispatch, enabling dual-GPU cortex patterns.
pub struct UnidirectionalRhmc {
    gpu: GpuF64,
    dyn_pipelines: GpuDynHmcPipelines,
    rhmc_pipelines: GpuRhmcPipelines,
    uni_pipelines: UniPipelines,
    state: GpuRhmcState,
    scg_bufs: GpuResidentShiftedCgBuffers,
    ham_bufs: UniHamiltonianBuffers,
    /// True multi-shift CG buffers (allocated on first trajectory).
    ms_bufs: Option<TrueMultiShiftBuffers>,
}

impl UnidirectionalRhmc {
    /// Initialize a unidirectional RHMC instance on a GPU.
    #[must_use]
    pub fn new(
        gpu: GpuF64,
        dyn_pipelines: GpuDynHmcPipelines,
        rhmc_pipelines: GpuRhmcPipelines,
        state: GpuRhmcState,
    ) -> Self {
        let uni_pipelines = UniPipelines::new(&gpu);
        let scg_bufs = GpuResidentShiftedCgBuffers::new(
            &gpu,
            &dyn_pipelines,
            &uni_pipelines.shifted_cg,
            &state.gauge,
        );
        let ham_bufs = UniHamiltonianBuffers::new(
            &gpu,
            &uni_pipelines.shifted_cg.base.reduce_pipeline,
            &state.gauge.gauge,
            &state.gauge,
        );
        Self {
            gpu,
            dyn_pipelines,
            rhmc_pipelines,
            uni_pipelines,
            state,
            scg_bufs,
            ham_bufs,
            ms_bufs: None,
        }
    }

    /// Run one trajectory synchronously (blocking). Fast (~1-2s for 8^4).
    pub fn run_trajectory(&mut self, config: &RhmcConfig, seed: &mut u64) -> TrajectoryResult {
        // Lazy-allocate true multi-shift CG buffers on first call
        if self.ms_bufs.is_none() {
            let max_shifts = config.sectors.iter().map(|s| {
                s.action_approx.sigma.len()
                    .max(s.force_approx.sigma.len())
            }).max().unwrap_or(0);
            if max_shifts > 0 {
                self.ms_bufs = Some(TrueMultiShiftBuffers::new(
                    &self.gpu,
                    &self.dyn_pipelines,
                    &self.uni_pipelines.true_ms_cg,
                    &self.state.gauge,
                    max_shifts,
                ));
            }
        }
        let t0 = std::time::Instant::now();
        let result = gpu_rhmc_trajectory_unidirectional(
            &self.gpu,
            &self.dyn_pipelines,
            &self.rhmc_pipelines,
            &self.uni_pipelines,
            &self.state,
            &self.scg_bufs,
            self.ms_bufs.as_ref(),
            &self.ham_bufs,
            config,
            seed,
        );
        TrajectoryResult {
            accepted: result.accepted,
            delta_h: result.delta_h,
            plaquette: result.plaquette,
            total_cg_iterations: result.total_cg_iterations,
            elapsed_secs: t0.elapsed().as_secs_f64(),
        }
    }

    /// Name of the GPU adapter backing this instance.
    pub fn adapter_name(&self) -> &str {
        &self.gpu.adapter_name
    }

    /// Access the underlying GPU state for diagnostics.
    pub fn state(&self) -> &GpuRhmcState {
        &self.state
    }

    /// Access the GPU handle.
    pub fn gpu(&self) -> &GpuF64 {
        &self.gpu
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Dual-GPU dispatch (Phase 6)
// ═══════════════════════════════════════════════════════════════════

/// Result from a dual-GPU trajectory pair.
#[derive(Debug)]
pub struct DualGpuResult {
    /// Result from GPU A.
    pub a: TrajectoryResult,
    /// Result from GPU B.
    pub b: TrajectoryResult,
}

/// Run one trajectory on each GPU in parallel (thread-based).
///
/// Each GPU gets an independent `UnidirectionalRhmc`. CPU dispatches
/// both, polls both, returns whichever finishes first (or both).
/// Cross-GPU parity checking happens at the caller level.
pub fn dual_gpu_trajectories(
    gpu_a: &mut UnidirectionalRhmc,
    gpu_b: &mut UnidirectionalRhmc,
    config: &RhmcConfig,
    seed_a: &mut u64,
    seed_b: &mut u64,
) -> DualGpuResult {
    let config_a = config.clone();
    let config_b = config.clone();
    let mut sa = *seed_a;
    let mut sb = *seed_b;

    // Both GPUs run trajectories via scoped threads sharing the borrows
    let (result_a, result_b) = std::thread::scope(|scope| {
        let handle_a = scope.spawn(|| {
            let r = gpu_a.run_trajectory(&config_a, &mut sa);
            (r, sa)
        });
        let handle_b = scope.spawn(|| {
            let r = gpu_b.run_trajectory(&config_b, &mut sb);
            (r, sb)
        });
        let (ra, new_sa) = handle_a.join().expect("GPU A thread");
        let (rb, new_sb) = handle_b.join().expect("GPU B thread");
        *seed_a = new_sa;
        *seed_b = new_sb;
        (ra, rb)
    });

    DualGpuResult {
        a: result_a,
        b: result_b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gpu() -> GpuF64 {
        let rt = tokio::runtime::Runtime::new().expect("tokio");
        rt.block_on(GpuF64::new()).expect("GPU required for test")
    }

    /// Minimal shader: writes f64 constant to verify pipeline plumbing.
    const WGSL_CONSTANT_WRITE: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main() {
    out[0] = f64(42.5);
}
"#;

    #[test]
    fn f64_pipeline_basic_sanity() {
        let gpu = make_gpu();
        eprintln!("  full_df64_mode={}", gpu.full_df64_mode);

        let out = gpu.create_f64_output_buffer(1, "sanity_out");
        let pl = gpu.create_pipeline_f64(WGSL_CONSTANT_WRITE, "sanity");
        let bg = gpu.create_bind_group(&pl, &[&out]);

        let mut enc = gpu.begin_encoder("sanity");
        GpuF64::encode_pass(&mut enc, &pl, &bg, 1);
        gpu.submit_encoder(enc);

        let val = gpu.read_back_f64(&out, 1).unwrap();
        eprintln!("  constant-write: out[0]={}", val[0]);
        assert!((val[0] - 42.5).abs() < 1e-6, "got {}", val[0]);
    }

    /// Reads from input buffer → writes to output.
    const WGSL_PASSTHROUGH: &str = r#"
@group(0) @binding(0) var<storage, read> inp: array<f64>;
@group(0) @binding(1) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main() {
    out[0] = inp[0];
    out[1] = inp[0] + f64(1.0);
}
"#;

    #[test]
    fn f64_pipeline_passthrough() {
        let gpu = make_gpu();
        let inp = gpu.create_f64_output_buffer(1, "pt_in");
        let out = gpu.create_f64_output_buffer(2, "pt_out");
        gpu.upload_f64(&inp, &[7.0]);

        let pl = gpu.create_pipeline_f64(WGSL_PASSTHROUGH, "pt");
        let bg = gpu.create_bind_group(&pl, &[&inp, &out]);

        let mut enc = gpu.begin_encoder("pt");
        GpuF64::encode_pass(&mut enc, &pl, &bg, 1);
        gpu.submit_encoder(enc);

        let val = gpu.read_back_f64(&out, 2).unwrap();
        eprintln!("  passthrough: out[0]={} out[1]={}", val[0], val[1]);
        assert!((val[0] - 7.0).abs() < 1e-6, "pass got {}", val[0]);
        assert!((val[1] - 8.0).abs() < 1e-6, "add got {}", val[1]);
    }

    /// Full H assembly kernel test with known scalar inputs.
    #[test]
    fn hamiltonian_assembly_kernel_roundtrip() {
        let gpu = make_gpu();

        let plaq_buf = gpu.create_f64_output_buffer(1, "test_plaq");
        let t_buf = gpu.create_f64_output_buffer(1, "test_t");
        let sf_buf = gpu.create_f64_output_buffer(1, "test_sf");
        let h_buf = gpu.create_f64_output_buffer(1, "test_h");
        let diag_buf = gpu.create_f64_output_buffer(3, "test_diag");

        gpu.upload_f64(&plaq_buf, &[100.0]);
        gpu.upload_f64(&t_buf, &[5.0]);
        gpu.upload_f64(&sf_buf, &[3.0]);

        let pl = gpu.create_pipeline_f64(WGSL_HAMILTONIAN_ASSEMBLY, "test_h_asm");

        let beta = 5.5;
        let volume = 4096usize;
        let six_v = 6.0 * volume as f64;
        let params = make_h_assembly_params(beta, six_v);
        let pbuf = gpu.create_storage_buffer_init(&params, "test_h_p");
        let bg = gpu.create_bind_group(&pl, &[&pbuf, &plaq_buf, &t_buf, &sf_buf, &h_buf, &diag_buf]);

        let mut enc = gpu.begin_encoder("test_h_asm");
        GpuF64::encode_pass(&mut enc, &pl, &bg, 1);
        gpu.submit_encoder(enc);

        let h_val = gpu.read_back_f64(&h_buf, 1).unwrap();
        let diag = gpu.read_back_f64(&diag_buf, 3).unwrap();

        let expected_sg = beta * (six_v - 100.0);
        let expected_h = expected_sg + 5.0 + 3.0;

        eprintln!("H assembly test: h={}, expected={}", h_val[0], expected_h);
        eprintln!("  diag: sg={} t={} sf={}", diag[0], diag[1], diag[2]);

        assert!((h_val[0] - expected_h).abs() < 1e-6,
            "H mismatch: got {} expected {}", h_val[0], expected_h);
        assert!((diag[0] - expected_sg).abs() < 1e-6);
        assert!((diag[1] - 5.0).abs() < 1e-6);
        assert!((diag[2] - 3.0).abs() < 1e-6);
    }

    /// Smoke-test: fermion action weighted sum kernel.
    #[test]
    fn fermion_action_sum_kernel_roundtrip() {
        let gpu = make_gpu();

        let dots_buf = gpu.create_f64_output_buffer(4, "test_dots");
        let sf_buf = gpu.create_f64_output_buffer(1, "test_sf");

        // dots = [10.0, 20.0, 30.0, 40.0]
        // alpha_0 = 0.5, alphas = [1.0, 2.0, 3.0]
        // expected: 0.5*10 + 1.0*20 + 2.0*30 + 3.0*40 = 5+20+60+120 = 205
        gpu.upload_f64(&dots_buf, &[10.0, 20.0, 30.0, 40.0]);
        gpu.zero_buffer(&sf_buf, 8);

        let pl = gpu.create_pipeline_f64(WGSL_FERMION_ACTION_SUM, "test_sf_sum");

        let n_dots = 4u32;
        let alpha_0 = 0.5f64;
        let params = make_fermion_action_sum_params(n_dots, alpha_0);
        let pbuf = gpu.create_storage_buffer_init(&params, "test_sf_p");

        let alphas: Vec<u8> = [1.0f64, 2.0, 3.0].iter()
            .flat_map(|a| a.to_le_bytes())
            .collect();
        let alpha_buf = gpu.create_storage_buffer_init(&alphas, "test_alphas");

        let bg = gpu.create_bind_group(&pl, &[&pbuf, &dots_buf, &alpha_buf, &sf_buf]);

        let mut enc = gpu.begin_encoder("test_sf_sum");
        GpuF64::encode_pass(&mut enc, &pl, &bg, 1);
        gpu.submit_encoder(enc);

        let sf = gpu.read_back_f64(&sf_buf, 1).unwrap();
        eprintln!("S_f sum test: sf={}, expected=205.0", sf[0]);
        assert!((sf[0] - 205.0).abs() < 1e-6,
            "S_f mismatch: got {} expected 205.0", sf[0]);
    }

    /// Smoke-test: Metropolis kernel with known H values.
    #[test]
    fn metropolis_kernel_roundtrip() {
        let gpu = make_gpu();

        let h_old = gpu.create_f64_output_buffer(1, "test_h_old");
        let h_new = gpu.create_f64_output_buffer(1, "test_h_new");
        let plaq = gpu.create_f64_output_buffer(1, "test_plaq");
        let diag_old = gpu.create_f64_output_buffer(3, "test_do");
        let diag_new = gpu.create_f64_output_buffer(3, "test_dn");
        let result = gpu.create_f64_output_buffer(9, "test_res");
        let staging = gpu.create_staging_buffer(9 * 8, "test_stg");

        // H_new < H_old → delta_h < 0 → always accept
        gpu.upload_f64(&h_old, &[100.0]);
        gpu.upload_f64(&h_new, &[99.0]);
        gpu.upload_f64(&plaq, &[12000.0]);
        gpu.upload_f64(&diag_old, &[90.0, 5.0, 5.0]);
        gpu.upload_f64(&diag_new, &[89.0, 5.0, 5.0]);

        let pl = gpu.create_pipeline_f64(WGSL_METROPOLIS, "test_metro");

        let r_val: f64 = 0.5;
        let six_v: f64 = 24576.0;
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&r_val.to_le_bytes());
        params.extend_from_slice(&six_v.to_le_bytes());
        let pbuf = gpu.create_storage_buffer_init(&params, "test_mp");

        let bg = gpu.create_bind_group(&pl, &[&pbuf, &h_old, &h_new, &plaq, &diag_old, &diag_new, &result]);

        let mut enc = gpu.begin_encoder("test_metro");
        GpuF64::encode_pass(&mut enc, &pl, &bg, 1);
        enc.copy_buffer_to_buffer(&result, 0, &staging, 0, 9 * 8);
        gpu.submit_encoder(enc);

        let data = gpu.read_staging_f64_n(&staging, 9).unwrap();
        eprintln!("Metropolis test: accepted={} dH={} plaq={}", data[0], data[1], data[2]);
        eprintln!("  sg_old={} sg_new={} t_old={} t_new={}", data[3], data[4], data[5], data[6]);
        eprintln!("  sf_old={} sf_new={}", data[7], data[8]);

        assert!(data[0] > 0.5, "Should accept: delta_h=-1.0");
        assert!((data[1] - (-1.0)).abs() < 1e-6, "delta_h wrong: {}", data[1]);
        assert!((data[2] - 12000.0 / 24576.0).abs() < 1e-6, "plaq wrong: {}", data[2]);
        assert!((data[7] - 5.0).abs() < 1e-6, "s_ferm_old wrong: {}", data[7]);
        assert!((data[8] - 5.0).abs() < 1e-6, "s_ferm_new wrong: {}", data[8]);
    }
}
