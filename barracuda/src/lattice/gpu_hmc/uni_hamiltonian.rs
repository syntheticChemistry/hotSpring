// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-resident Hamiltonian evaluation for unidirectional RHMC (B2).
//!
//! Wilson plaquette and kinetic energy reductions, fermion dot products,
//! weighted fermion action sum, and full `H` assembly run entirely on GPU
//! with no scalar readback. Shared by [`super::unidirectional_rhmc`].

use super::dynamical::{
    GpuDynHmcPipelines, GpuDynHmcState, WGSL_RANDOM_MOMENTA, WGSL_RANDOM_MOMENTA_TMU,
};
use super::gpu_rhmc::{GpuRhmcSectorBuffers, GpuRhmcState};
use super::resident_cg_buffers::{ReduceChain, build_reduce_chain_pub, encode_reduce_chain};
use super::resident_shifted_cg::{
    GpuResidentShiftedCgBuffers, GpuResidentShiftedCgPipelines, gpu_multi_shift_cg_solve_resident,
};
use super::rop_force_accum::RopForceAccumulator;
use super::streaming::WGSL_GAUSSIAN_FERMION;
use super::tmu_tables::TmuLookupTables;
use super::true_multishift_cg::{
    TrueMultiShiftBuffers, TrueMultiShiftPipelines, gpu_true_multi_shift_cg_solve,
};
use super::{GpuF64, GpuHmcState, make_u32x4_params};

use crate::lattice::rhmc::RhmcConfig;

// ── WGSL shaders for GPU-resident Hamiltonian (B2) ───────────────

/// Hamiltonian assembly kernel: `H = beta*(6V - plaq) + T + S_f`.
pub(crate) const WGSL_HAMILTONIAN_ASSEMBLY: &str =
    include_str!("../shaders/hamiltonian_assembly_f64.wgsl");
/// Weighted sum of fermion dot products into `S_f`.
pub(crate) const WGSL_FERMION_ACTION_SUM: &str =
    include_str!("../shaders/fermion_action_sum_f64.wgsl");
/// Metropolis accept/reject + diagnostics (single-thread GPU kernel).
pub(crate) const WGSL_METROPOLIS: &str = include_str!("../shaders/metropolis_f64.wgsl");

/// Maximum rational approximation poles per sector (dots buffer sizing).
const MAX_RATIONAL_POLES: usize = 32;

/// CG check interval — iterations per initial batch before first convergence readback.
///
/// Exponential back-off doubles this after each check (capped at 2000).
/// Shared with unidirectional heatbath and force paths.
pub(crate) const CG_CHECK_INTERVAL: usize = 200;

// ── Pipelines (compiled once per GPU) ────────────────────────────

/// All pipelines for unidirectional RHMC.
pub struct UniPipelines {
    /// GPU-resident shifted CG pipelines (legacy sequential path).
    pub shifted_cg: GpuResidentShiftedCgPipelines,
    /// True multi-shift CG pipelines (shared Krylov, N_shifts fewer D†D ops).
    pub true_ms_cg: TrueMultiShiftPipelines,
    /// GPU PRNG for SU(3) algebra momenta (ALU path).
    pub momenta_prng_pipeline: wgpu::ComputePipeline,
    /// TMU-accelerated PRNG pipeline (Tier 0). `None` if TMU unavailable.
    pub tmu_prng_pipeline: Option<wgpu::ComputePipeline>,
    /// TMU lookup tables for Box-Muller (log, trig). `None` if TMU path not compiled.
    pub tmu_tables: Option<TmuLookupTables>,
    /// GPU Gaussian sampler for pseudofermion heat-bath η.
    pub fermion_prng_pipeline: wgpu::ComputePipeline,
    /// H = beta*(6V - plaq) + T + S_f  (single-thread GPU kernel).
    pub hamiltonian_assembly_pipeline: wgpu::ComputePipeline,
    /// S_f weighted sum from dot products + alpha coefficients.
    pub fermion_action_sum_pipeline: wgpu::ComputePipeline,
    /// Metropolis accept/reject + diagnostics (single-thread GPU kernel).
    pub metropolis_pipeline: wgpu::ComputePipeline,
    /// ROP-accelerated fermion force accumulation (Tier 3).
    /// When `Some`, fuses force+momentum per pole via fixed-point `atomicAdd`.
    pub rop_accum: Option<RopForceAccumulator>,
}

impl UniPipelines {
    /// Compile all unidirectional RHMC pipelines (ALU PRNG, no ROP).
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            shifted_cg: GpuResidentShiftedCgPipelines::new(gpu),
            true_ms_cg: TrueMultiShiftPipelines::new(gpu),
            momenta_prng_pipeline: gpu.create_pipeline_f64(&WGSL_RANDOM_MOMENTA, "uni_mom_prng"),
            tmu_prng_pipeline: None,
            tmu_tables: None,
            fermion_prng_pipeline: gpu.create_pipeline_f64(&WGSL_GAUSSIAN_FERMION, "uni_ferm_prng"),
            hamiltonian_assembly_pipeline: gpu
                .create_pipeline_f64(WGSL_HAMILTONIAN_ASSEMBLY, "uni_h_asm"),
            fermion_action_sum_pipeline: gpu
                .create_pipeline_f64(WGSL_FERMION_ACTION_SUM, "uni_sf_sum"),
            metropolis_pipeline: gpu.create_pipeline_f64(WGSL_METROPOLIS, "uni_metropolis"),
            rop_accum: None,
        }
    }

    /// Compile all unidirectional RHMC pipelines with ROP force accumulation.
    ///
    /// `volume` is the lattice volume, used to size the atomic accumulation buffer.
    #[must_use]
    pub fn new_with_rop(gpu: &GpuF64, volume: usize) -> Self {
        let mut pipelines = Self::new(gpu);
        eprintln!("[ROP] Fermion force accumulation: ENABLED (atomicAdd i32, Tier 3)");
        pipelines.rop_accum = Some(RopForceAccumulator::new(gpu, volume));
        pipelines
    }

    /// Compile with full silicon saturation: TMU PRNG (Tier 0) + ROP atomics (Tier 3).
    ///
    /// `volume` is the lattice volume for ROP buffer sizing.
    #[must_use]
    pub fn new_saturated(gpu: &GpuF64, volume: usize) -> Self {
        let tables = TmuLookupTables::new(gpu);
        let tmu_pl = gpu.create_pipeline_f64(&WGSL_RANDOM_MOMENTA_TMU, "uni_mom_prng_tmu");
        eprintln!("[TMU] Momenta PRNG: ENABLED (Box-Muller via texture lookup, Tier 0)");
        let mut pipelines = Self {
            shifted_cg: GpuResidentShiftedCgPipelines::new(gpu),
            true_ms_cg: TrueMultiShiftPipelines::new(gpu),
            momenta_prng_pipeline: gpu.create_pipeline_f64(&WGSL_RANDOM_MOMENTA, "uni_mom_prng"),
            tmu_prng_pipeline: Some(tmu_pl),
            tmu_tables: Some(tables),
            fermion_prng_pipeline: gpu.create_pipeline_f64(&WGSL_GAUSSIAN_FERMION, "uni_ferm_prng"),
            hamiltonian_assembly_pipeline: gpu
                .create_pipeline_f64(WGSL_HAMILTONIAN_ASSEMBLY, "uni_h_asm"),
            fermion_action_sum_pipeline: gpu
                .create_pipeline_f64(WGSL_FERMION_ACTION_SUM, "uni_sf_sum"),
            metropolis_pipeline: gpu.create_pipeline_f64(WGSL_METROPOLIS, "uni_metropolis"),
            rop_accum: None,
        };
        eprintln!("[ROP] Fermion force accumulation: ENABLED (atomicAdd i32, Tier 3)");
        pipelines.rop_accum = Some(RopForceAccumulator::new(gpu, volume));
        pipelines
    }
}

// ── GPU-resident Hamiltonian buffers + reduce chains (Phase 2) ─

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
    GpuF64::encode_pass(
        enc,
        &dyn_pipelines.gauge.plaquette_pipeline,
        &bg,
        gauge.wg_vol,
    );
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
    GpuF64::encode_pass(
        enc,
        &dyn_pipelines.gauge.kinetic_pipeline,
        &bg,
        gauge.wg_links,
    );
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
pub(crate) fn make_fermion_action_sum_params(n_dots: u32, alpha_0: f64) -> Vec<u8> {
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

    let alpha_data: Vec<u8> = approx.alpha.iter().flat_map(|a| a.to_le_bytes()).collect();
    let alpha_buf = gpu.create_storage_buffer_init(&alpha_data, "uni_sf_alphas");

    let bg = gpu.create_bind_group(
        &uni_pipelines.fermion_action_sum_pipeline,
        &[&pbuf, &ham.dots_buf, &alpha_buf, &ham.s_ferm_buf],
    );
    GpuF64::encode_pass(enc, &uni_pipelines.fermion_action_sum_pipeline, &bg, 1);
}

/// Build params buffer for the Hamiltonian assembly kernel: [beta, 6V] as f64.
pub(crate) fn make_h_assembly_params(beta: f64, six_v: f64) -> Vec<u8> {
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

/// Compute full Hamiltonian on GPU: gauge+KE reduces, fermion CG+dots, assembly.
///
/// Returns the number of CG iterations used. The result lives in `h_buf` and
/// `diag_buf` on GPU — no readback.
pub(crate) fn compute_h_gpu(
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
                    gpu,
                    dyn_pipelines,
                    &uni_pipelines.true_ms_cg,
                    &state.gauge,
                    ms,
                    &sector.x_bufs,
                    &sector.phi_buf,
                    &fconfig.action_approx.sigma,
                    config.cg_tol,
                    config.cg_max_iter,
                    CG_CHECK_INTERVAL,
                )
            } else {
                gpu_multi_shift_cg_solve_resident(
                    gpu,
                    dyn_pipelines,
                    &uni_pipelines.shifted_cg,
                    &state.gauge,
                    scg_bufs,
                    &sector.x_bufs,
                    &sector.phi_buf,
                    &fconfig.action_approx.sigma,
                    config.cg_tol,
                    config.cg_max_iter,
                    CG_CHECK_INTERVAL,
                )
            }
        } else {
            gpu_multi_shift_cg_solve_resident(
                gpu,
                dyn_pipelines,
                &uni_pipelines.shifted_cg,
                &state.gauge,
                scg_bufs,
                &sector.x_bufs,
                &sector.phi_buf,
                &fconfig.action_approx.sigma,
                config.cg_tol,
                config.cg_max_iter,
                CG_CHECK_INTERVAL,
            )
        };
        total_cg += cg;

        let mut enc = gpu.begin_encoder("b2_sf_dots");
        encode_fermion_dots_to_gpu(
            &mut enc,
            gpu,
            dyn_pipelines,
            &state.gauge,
            sector,
            &fconfig.action_approx,
            reduce_pl,
            ham,
        );
        encode_fermion_action_sum(&mut enc, gpu, uni_pipelines, ham, &fconfig.action_approx);
        gpu.submit_encoder(enc);
    }

    // H = beta*(6V - plaq) + T + S_f  (GPU-only, no readback)
    {
        let mut enc = gpu.begin_encoder("b2_h_asm");
        encode_hamiltonian_assembly(
            &mut enc,
            gpu,
            uni_pipelines,
            ham,
            h_buf,
            diag_buf,
            gauge.beta,
            gauge.volume,
        );
        gpu.submit_encoder(enc);
    }

    total_cg
}
