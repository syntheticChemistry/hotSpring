// SPDX-License-Identifier: AGPL-3.0-or-later

//! Streaming GPU HMC — zero dispatch overhead via batched command encoders.
//!
//! Pure gauge streaming batches all Omelyan MD steps into a single encoder
//! submission. Dynamical streaming adds GPU PRNG for momenta and pseudofermion
//! heat bath, eliminating all CPU→GPU transfers.
//!
//! All H computation uses GPU-resident reduce chains (O(1) readback) via
//! `resident_observables` — no per-site/per-link readback to CPU.

use super::dynamical::{
    GpuDynHmcPipelines, GpuDynHmcResult, GpuDynHmcState, WGSL_RANDOM_MOMENTA,
    WGSL_RANDOM_MOMENTA_TMU, gpu_fermion_action_all, gpu_total_force_dispatch,
};
use super::resident_cg::WGSL_SUM_REDUCE;
use super::resident_observables::{
    ResidentObservableBuffers, gauge_ke_resident, plaquette_resident,
};
use super::tmu_tables::TmuLookupTables;
use super::{
    GpuF64, GpuHmcPipelines, GpuHmcResult, GpuHmcState, gpu_link_update_dispatch,
    make_force_params, make_link_mom_params, make_prng_params,
};
use crate::error::HotSpringError;

/// Streaming HMC pipelines: quenched HMC + GPU PRNG + GPU reduce for observables.
pub struct GpuHmcStreamingPipelines {
    /// Base quenched HMC pipeline set (force, link/momentum update, plaquette).
    pub hmc: GpuHmcPipelines,
    /// GPU PRNG shader for on-device SU(3) algebra momentum generation (ALU path).
    pub prng_pipeline: wgpu::ComputePipeline,
    /// TMU-accelerated PRNG pipeline (Tier 0 silicon routing). `None` if TMU unavailable.
    pub tmu_prng_pipeline: Option<wgpu::ComputePipeline>,
    /// TMU lookup tables for Box-Muller (log, trig). `None` if TMU path not compiled.
    pub tmu_tables: Option<TmuLookupTables>,
    /// Tree-reduce pipeline for GPU-resident scalar accumulation.
    pub reduce_pipeline: wgpu::ComputePipeline,
}

impl GpuHmcStreamingPipelines {
    /// Compile all streaming HMC pipelines including GPU PRNG and reduce.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            hmc: GpuHmcPipelines::new(gpu),
            prng_pipeline: gpu.create_pipeline_f64(&WGSL_RANDOM_MOMENTA, "hmc_prng"),
            tmu_prng_pipeline: None,
            tmu_tables: None,
            reduce_pipeline: gpu.create_pipeline_f64(WGSL_SUM_REDUCE, "hmc_reduce"),
        }
    }

    /// Compile streaming pipelines with TMU-accelerated PRNG (Tier 0).
    ///
    /// The TMU path offloads Box-Muller log/cos/sin to texture units, freeing
    /// ALU for concurrent physics computation (composition multiplier ~2-3x).
    #[must_use]
    pub fn new_with_tmu(gpu: &GpuF64) -> Self {
        let tables = TmuLookupTables::new(gpu);
        let tmu_pl = gpu.create_pipeline_f64(&WGSL_RANDOM_MOMENTA_TMU, "hmc_prng_tmu");
        Self {
            hmc: GpuHmcPipelines::new(gpu),
            prng_pipeline: gpu.create_pipeline_f64(&WGSL_RANDOM_MOMENTA, "hmc_prng"),
            tmu_prng_pipeline: Some(tmu_pl),
            tmu_tables: Some(tables),
            reduce_pipeline: gpu.create_pipeline_f64(WGSL_SUM_REDUCE, "hmc_reduce"),
        }
    }
}

/// Run one pure-GPU streaming Omelyan HMC trajectory.
///
/// **All** dispatches for the MD integration are batched into a single
/// encoder submission. Momenta are generated on GPU (no CPU→GPU upload).
/// H computation uses GPU reduce chains — 16-byte readback per H eval.
///
/// GPU submissions per trajectory:
///   1. PRNG momenta + backup links
///   2. Gauge+KE reduce for H_old (16 bytes readback)
///   3. All N_md×5 MD dispatches (one encoder)
///   4. Gauge+KE reduce for H_new (16 bytes readback)
///   5. Plaquette reduce for observable (8 bytes readback)
pub fn gpu_hmc_trajectory_streaming(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    state: &GpuHmcState,
    n_md_steps: usize,
    dt: f64,
    traj_id: u32,
    seed: &mut u64,
) -> Result<GpuHmcResult, HotSpringError> {
    let n_links = state.n_links;
    let p = &pipelines.hmc;
    let obs = ResidentObservableBuffers::new(gpu, &pipelines.reduce_pipeline, state);

    {
        let mut enc = gpu.begin_encoder("stream_init");
        let prng_params = make_prng_params(n_links as u32, traj_id, seed);
        let prng_pbuf = gpu.create_uniform_buffer(&prng_params, "prng_p");
        encode_prng_dispatch(
            &mut enc,
            gpu,
            pipelines,
            &prng_pbuf,
            &state.mom_buf,
            state.wg_links,
        );
        enc.copy_buffer_to_buffer(
            &state.link_buf,
            0,
            &state.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let (plaq_old, ke_old) =
        gauge_ke_resident(gpu, p, state, &pipelines.reduce_pipeline, &obs)?;
    let h_old = state.beta * 6.0f64.mul_add(state.volume as f64, -plaq_old) + ke_old;

    gpu_streaming_md_encoder(gpu, p, state, n_md_steps, dt);

    let (plaq_new, ke_new) =
        gauge_ke_resident(gpu, p, state, &pipelines.reduce_pipeline, &obs)?;
    let h_new = state.beta * 6.0f64.mul_add(state.volume as f64, -plaq_new) + ke_new;

    let delta_h = h_new - h_old;
    let r: f64 = super::super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("stream_restore");
        enc.copy_buffer_to_buffer(
            &state.link_backup,
            0,
            &state.link_buf,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let plaquette =
        plaquette_resident(gpu, p, state, &pipelines.reduce_pipeline, &obs)?;

    Ok(GpuHmcResult {
        accepted,
        delta_h,
        plaquette,
    })
}

/// Streaming HMC with CPU-generated momenta (for parity testing).
///
/// Uses encoder batching for all MD dispatches (streaming dispatch), but
/// momenta are generated on CPU and uploaded — proving the encoder
/// batching itself produces bit-identical physics to per-dispatch mode.
/// H computation uses GPU reduce chains (O(1) readback).
pub fn gpu_hmc_trajectory_streaming_cpu_mom(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    state: &GpuHmcState,
    n_md_steps: usize,
    dt: f64,
    seed: &mut u64,
) -> Result<GpuHmcResult, HotSpringError> {
    let n_links = state.n_links;
    let p = &pipelines.hmc;
    let obs = ResidentObservableBuffers::new(gpu, &pipelines.reduce_pipeline, state);

    let momenta: Vec<super::super::su3::Su3Matrix> = (0..n_links)
        .map(|_| super::super::su3::Su3Matrix::random_algebra(seed))
        .collect();
    let mom_flat = super::flatten_momenta(&momenta);
    gpu.upload_f64(&state.mom_buf, &mom_flat);

    {
        let mut enc = gpu.begin_encoder("scm_backup");
        enc.copy_buffer_to_buffer(
            &state.link_buf,
            0,
            &state.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let (plaq_old, ke_old) =
        gauge_ke_resident(gpu, p, state, &pipelines.reduce_pipeline, &obs)?;
    let h_old = state.beta * 6.0f64.mul_add(state.volume as f64, -plaq_old) + ke_old;

    gpu_streaming_md_encoder(gpu, p, state, n_md_steps, dt);

    let (plaq_new, ke_new) =
        gauge_ke_resident(gpu, p, state, &pipelines.reduce_pipeline, &obs)?;
    let h_new = state.beta * 6.0f64.mul_add(state.volume as f64, -plaq_new) + ke_new;

    let delta_h = h_new - h_old;
    let r: f64 = super::super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("scm_restore");
        enc.copy_buffer_to_buffer(
            &state.link_backup,
            0,
            &state.link_buf,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let plaquette =
        plaquette_resident(gpu, p, state, &pipelines.reduce_pipeline, &obs)?;

    Ok(GpuHmcResult {
        accepted,
        delta_h,
        plaquette,
    })
}

/// Dispatch PRNG momenta generation via TMU path if available, else ALU fallback.
fn encode_prng_dispatch(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    params_buf: &wgpu::Buffer,
    mom_buf: &wgpu::Buffer,
    wg_links: u32,
) {
    if let (Some(tmu_pl), Some(tables)) = (&pipelines.tmu_prng_pipeline, &pipelines.tmu_tables) {
        let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prng_tmu_bg"),
            layout: &tmu_pl.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mom_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&tables.log_table),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&tables.trig_table),
                },
            ],
        });
        GpuF64::encode_pass(enc, tmu_pl, &bg, wg_links);
    } else {
        let bg = gpu.create_bind_group(&pipelines.prng_pipeline, &[params_buf, mom_buf]);
        GpuF64::encode_pass(enc, &pipelines.prng_pipeline, &bg, wg_links);
    }
}

fn gpu_streaming_md_encoder(
    gpu: &GpuF64,
    p: &GpuHmcPipelines,
    state: &GpuHmcState,
    n_md_steps: usize,
    dt: f64,
) {
    let n_links = state.n_links;
    let lam = crate::tolerances::OMELYAN_LAMBDA;

    let force_params = make_force_params(state.volume, state.beta, gpu.full_df64_mode);
    let force_pbuf = gpu.create_uniform_buffer(&force_params, "s_force_p");
    let force_bg = gpu.create_bind_group(
        &p.force_pipeline,
        &[
            &force_pbuf,
            &state.link_buf,
            &state.nbr_buf,
            &state.force_buf,
        ],
    );

    let mom_lam_params = make_link_mom_params(n_links, lam * dt, gpu.full_df64_mode);
    let mom_lam_pbuf = gpu.create_uniform_buffer(&mom_lam_params, "s_mom_lam");
    let mom_lam_bg = gpu.create_bind_group(
        &p.momentum_pipeline,
        &[&mom_lam_pbuf, &state.force_buf, &state.mom_buf],
    );

    let mom_mid_params =
        make_link_mom_params(n_links, 2.0f64.mul_add(-lam, 1.0) * dt, gpu.full_df64_mode);
    let mom_mid_pbuf = gpu.create_uniform_buffer(&mom_mid_params, "s_mom_mid");
    let mom_mid_bg = gpu.create_bind_group(
        &p.momentum_pipeline,
        &[&mom_mid_pbuf, &state.force_buf, &state.mom_buf],
    );

    let link_params = make_link_mom_params(n_links, 0.5 * dt, gpu.full_df64_mode);
    let link_pbuf = gpu.create_uniform_buffer(&link_params, "s_link_p");
    let link_bg = gpu.create_bind_group(
        &p.link_pipeline,
        &[&link_pbuf, &state.mom_buf, &state.link_buf],
    );

    let mut enc = gpu.begin_encoder("stream_md");

    for _step in 0..n_md_steps {
        GpuF64::encode_pass(&mut enc, &p.force_pipeline, &force_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.momentum_pipeline, &mom_lam_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.link_pipeline, &link_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.force_pipeline, &force_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.momentum_pipeline, &mom_mid_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.link_pipeline, &link_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.force_pipeline, &force_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.momentum_pipeline, &mom_lam_bg, state.wg_links);
    }

    gpu.submit_encoder(enc);
}

/// WGSL shared PRNG core (PCG hash → uniform f64).
const WGSL_PRNG_CORE: &str = include_str!("../shaders/prng_pcg_f64.wgsl");
/// WGSL shader: GPU-resident PRNG for Gaussian fermion fields.
pub static WGSL_GAUSSIAN_FERMION: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let body = include_str!("../shaders/gaussian_fermion_f64.wgsl");
    format!("{WGSL_PRNG_CORE}\n{body}")
});

/// Streaming pipelines for dynamical fermion HMC.
///
/// Extends `GpuDynHmcPipelines` with GPU PRNG for momenta and pseudofermion
/// heat bath, plus GPU reduce for O(1)-readback H computation. The CG solver
/// still requires per-iteration convergence readbacks, but gauge action and
/// kinetic energy are fully GPU-resident.
pub struct GpuDynHmcStreamingPipelines {
    /// Base dynamical fermion HMC pipeline set.
    pub dyn_hmc: GpuDynHmcPipelines,
    /// GPU PRNG shader for SU(3) algebra momentum generation.
    pub momenta_prng_pipeline: wgpu::ComputePipeline,
    /// GPU Gaussian sampler for pseudofermion heat-bath η.
    pub fermion_prng_pipeline: wgpu::ComputePipeline,
    /// Tree-reduce pipeline for GPU-resident scalar accumulation.
    pub reduce_pipeline: wgpu::ComputePipeline,
}

impl GpuDynHmcStreamingPipelines {
    /// Compile all dynamical streaming HMC pipelines including GPU PRNG and reduce.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            dyn_hmc: GpuDynHmcPipelines::new(gpu),
            momenta_prng_pipeline: gpu.create_pipeline_f64(&WGSL_RANDOM_MOMENTA, "sdyn_mom_prng"),
            fermion_prng_pipeline: gpu
                .create_pipeline_f64(&WGSL_GAUSSIAN_FERMION, "sdyn_ferm_prng"),
            reduce_pipeline: gpu.create_pipeline_f64(WGSL_SUM_REDUCE, "sdyn_reduce"),
        }
    }
}

/// Full streaming dynamical fermion HMC trajectory.
///
/// GPU PRNG generates both momenta and pseudofermion field η on-device.
/// The pseudofermion heat bath φ = D†η is computed via GPU Dirac dispatch.
/// Gauge action and kinetic energy use GPU reduce chains (16-byte readback
/// per H eval). CG iterations still require per-batch convergence readbacks.
///
/// Transfer budget per trajectory:
///   CPU→GPU: 0 bytes (GPU PRNG for momenta + pseudofermion)
///   GPU→CPU: ~(8 × CG_iters/check_interval) convergence + 32 bytes gauge+KE + 8 bytes plaq
pub fn gpu_dynamical_hmc_trajectory_streaming(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcStreamingPipelines,
    state: &GpuDynHmcState,
    n_md_steps: usize,
    dt: f64,
    traj_id: u32,
    seed: &mut u64,
) -> Result<GpuDynHmcResult, HotSpringError> {
    let vol = state.gauge.volume;
    let n_links = state.gauge.n_links;
    let gs = &state.gauge;
    let dp = &pipelines.dyn_hmc;
    let obs = ResidentObservableBuffers::new(gpu, &pipelines.reduce_pipeline, gs);

    {
        let mut enc = gpu.begin_encoder("sdyn_prng");
        let mom_prng_params = make_prng_params(n_links as u32, traj_id, seed);
        let mom_prng_pbuf = gpu.create_uniform_buffer(&mom_prng_params, "sdyn_mom_p");
        let mom_prng_bg = gpu.create_bind_group(
            &pipelines.momenta_prng_pipeline,
            &[&mom_prng_pbuf, &gs.mom_buf],
        );
        GpuF64::encode_pass(
            &mut enc,
            &pipelines.momenta_prng_pipeline,
            &mom_prng_bg,
            gs.wg_links,
        );

        let wg_vol = (vol as u32).div_ceil(64);
        for (fi, phi_buf) in state.phi_bufs.iter().enumerate() {
            let ferm_prng_params =
                make_ferm_prng_params(vol as u32, traj_id + fi as u32 * 1000, seed);
            let ferm_prng_pbuf =
                gpu.create_uniform_buffer(&ferm_prng_params, &format!("sdyn_ferm_p_{fi}"));
            let ferm_prng_bg = gpu.create_bind_group(
                &pipelines.fermion_prng_pipeline,
                &[&ferm_prng_pbuf, phi_buf],
            );
            GpuF64::encode_pass(
                &mut enc,
                &pipelines.fermion_prng_pipeline,
                &ferm_prng_bg,
                wg_vol,
            );
        }

        enc.copy_buffer_to_buffer(
            &gs.link_buf,
            0,
            &gs.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let (plaq_old, ke_old) =
        gauge_ke_resident(gpu, &dp.gauge, gs, &pipelines.reduce_pipeline, &obs)?;
    let s_gauge_old = gs.beta * 6.0f64.mul_add(gs.volume as f64, -plaq_old);
    let (s_ferm_old, cg_iters_old) = gpu_fermion_action_all(gpu, dp, state);
    let h_old = s_gauge_old + ke_old + s_ferm_old;
    let mut total_cg = cg_iters_old;

    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for _step in 0..n_md_steps {
        let cg1 = gpu_total_force_dispatch(gpu, dp, state, lam * dt);
        gpu_link_update_dispatch(gpu, &dp.gauge, gs, 0.5 * dt);
        let cg2 = gpu_total_force_dispatch(gpu, dp, state, 2.0f64.mul_add(-lam, 1.0) * dt);
        gpu_link_update_dispatch(gpu, &dp.gauge, gs, 0.5 * dt);
        let cg3 = gpu_total_force_dispatch(gpu, dp, state, lam * dt);
        total_cg += cg1 + cg2 + cg3;
    }

    let (plaq_new, ke_new) =
        gauge_ke_resident(gpu, &dp.gauge, gs, &pipelines.reduce_pipeline, &obs)?;
    let s_gauge_new = gs.beta * 6.0f64.mul_add(gs.volume as f64, -plaq_new);
    let (s_ferm_new, cg_iters_new) = gpu_fermion_action_all(gpu, dp, state);
    let h_new = s_gauge_new + ke_new + s_ferm_new;
    total_cg += cg_iters_new;

    let delta_h = h_new - h_old;
    let r: f64 = super::super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("sdyn_restore");
        enc.copy_buffer_to_buffer(
            &gs.link_backup,
            0,
            &gs.link_buf,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let plaquette =
        plaquette_resident(gpu, &dp.gauge, gs, &pipelines.reduce_pipeline, &obs)?;

    Ok(GpuDynHmcResult {
        accepted,
        delta_h,
        plaquette,
        cg_iterations: total_cg,
    })
}

pub(super) fn make_ferm_prng_params(volume: u32, traj_id: u32, seed: &mut u64) -> Vec<u8> {
    super::super::constants::lcg_step(seed);
    let s = *seed;
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&volume.to_le_bytes());
    v.extend_from_slice(&traj_id.to_le_bytes());
    v.extend_from_slice(&(s as u32).to_le_bytes());
    v.extend_from_slice(&((s >> 32) as u32).to_le_bytes());
    v
}
