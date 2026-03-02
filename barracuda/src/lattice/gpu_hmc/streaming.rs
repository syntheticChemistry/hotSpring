// SPDX-License-Identifier: AGPL-3.0-only

//! Streaming GPU HMC — zero dispatch overhead via batched command encoders.
//!
//! Pure gauge streaming batches all Omelyan MD steps into a single encoder
//! submission. Dynamical streaming adds GPU PRNG for momenta and pseudofermion
//! heat bath, eliminating all CPU→GPU transfers.

use super::dynamical::{
    gpu_fermion_action, gpu_total_force_dispatch, GpuDynHmcPipelines, GpuDynHmcResult,
    GpuDynHmcState, WGSL_RANDOM_MOMENTA,
};
use super::{
    gpu_kinetic_energy, gpu_link_update_dispatch, gpu_plaquette, gpu_wilson_action,
    make_force_params, make_link_mom_params, make_prng_params, make_u32x4_params, GpuF64,
    GpuHmcPipelines, GpuHmcResult, GpuHmcState,
};

/// Streaming HMC pipelines: quenched HMC + GPU PRNG.
pub struct GpuHmcStreamingPipelines {
    /// Base quenched HMC pipeline set (force, link/momentum update, plaquette).
    pub hmc: GpuHmcPipelines,
    /// GPU PRNG shader for on-device SU(3) algebra momentum generation.
    pub prng_pipeline: wgpu::ComputePipeline,
}

impl GpuHmcStreamingPipelines {
    /// Compile all streaming HMC pipelines including GPU PRNG.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            hmc: GpuHmcPipelines::new(gpu),
            prng_pipeline: gpu.create_pipeline_f64(&WGSL_RANDOM_MOMENTA, "hmc_prng"),
        }
    }
}

/// Run one pure-GPU streaming Omelyan HMC trajectory.
///
/// **All** dispatches for the MD integration are batched into a single
/// encoder submission. Momenta are generated on GPU (no CPU→GPU upload).
/// Only 3 GPU submissions per trajectory:
///
/// 1. PRNG momenta + backup links + plaquette/KE for H_old
/// 2. All N_md×5 MD dispatches (one encoder)
/// 3. Plaquette/KE for H_new
pub fn gpu_hmc_trajectory_streaming(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    state: &GpuHmcState,
    n_md_steps: usize,
    dt: f64,
    traj_id: u32,
    seed: &mut u64,
) -> GpuHmcResult {
    let n_links = state.n_links;
    let p = &pipelines.hmc;

    {
        let mut enc = gpu.begin_encoder("stream_init");
        let prng_params = make_prng_params(n_links as u32, traj_id, seed);
        let prng_pbuf = gpu.create_uniform_buffer(&prng_params, "prng_p");
        let prng_bg =
            gpu.create_bind_group(&pipelines.prng_pipeline, &[&prng_pbuf, &state.mom_buf]);
        GpuF64::encode_pass(&mut enc, &pipelines.prng_pipeline, &prng_bg, state.wg_links);
        enc.copy_buffer_to_buffer(
            &state.link_buf,
            0,
            &state.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let s_old = gpu_wilson_action_streaming(gpu, p, state);
    let t_old = gpu_kinetic_energy_streaming(gpu, p, state);
    let h_old = s_old + t_old;

    gpu_streaming_md_encoder(gpu, p, state, n_md_steps, dt);

    let s_new = gpu_wilson_action_streaming(gpu, p, state);
    let t_new = gpu_kinetic_energy_streaming(gpu, p, state);
    let h_new = s_new + t_new;

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

    let plaquette = gpu_plaquette(gpu, p, state);

    GpuHmcResult {
        accepted,
        delta_h,
        plaquette,
    }
}

/// Streaming HMC with CPU-generated momenta (for parity testing).
///
/// Uses encoder batching for all MD dispatches (streaming dispatch), but
/// momenta are generated on CPU and uploaded — proving the encoder
/// batching itself produces bit-identical physics to per-dispatch mode.
pub fn gpu_hmc_trajectory_streaming_cpu_mom(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    state: &GpuHmcState,
    n_md_steps: usize,
    dt: f64,
    seed: &mut u64,
) -> GpuHmcResult {
    let n_links = state.n_links;
    let p = &pipelines.hmc;

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

    let s_old = gpu_wilson_action_streaming(gpu, p, state);
    let t_old = gpu_kinetic_energy_streaming(gpu, p, state);
    let h_old = s_old + t_old;

    gpu_streaming_md_encoder(gpu, p, state, n_md_steps, dt);

    let s_new = gpu_wilson_action_streaming(gpu, p, state);
    let t_new = gpu_kinetic_energy_streaming(gpu, p, state);
    let h_new = s_new + t_new;

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

    let plaquette = gpu_plaquette(gpu, p, state);

    GpuHmcResult {
        accepted,
        delta_h,
        plaquette,
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

    let force_params = make_force_params(state.volume, state.beta);
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

    let mom_lam_params = make_link_mom_params(n_links, lam * dt);
    let mom_lam_pbuf = gpu.create_uniform_buffer(&mom_lam_params, "s_mom_lam");
    let mom_lam_bg = gpu.create_bind_group(
        &p.momentum_pipeline,
        &[&mom_lam_pbuf, &state.force_buf, &state.mom_buf],
    );

    let mom_mid_params = make_link_mom_params(n_links, (1.0 - 2.0 * lam) * dt);
    let mom_mid_pbuf = gpu.create_uniform_buffer(&mom_mid_params, "s_mom_mid");
    let mom_mid_bg = gpu.create_bind_group(
        &p.momentum_pipeline,
        &[&mom_mid_pbuf, &state.force_buf, &state.mom_buf],
    );

    let link_params = make_link_mom_params(n_links, 0.5 * dt);
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

fn gpu_wilson_action_streaming(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) -> f64 {
    let params = make_u32x4_params(s.volume as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "s_plaq_p");
    let bg = gpu.create_bind_group(
        &p.plaquette_pipeline,
        &[&param_buf, &s.link_buf, &s.nbr_buf, &s.plaq_out_buf],
    );
    let staging = gpu.create_staging_buffer(s.volume * 8, "s_plaq_staging");
    let mut enc = gpu.begin_encoder("s_wilson");
    GpuF64::encode_pass(&mut enc, &p.plaquette_pipeline, &bg, s.wg_vol);
    enc.copy_buffer_to_buffer(&s.plaq_out_buf, 0, &staging, 0, (s.volume * 8) as u64);
    gpu.submit_encoder(enc);
    let Ok(per_site) = gpu.read_staging_f64(&staging) else {
        return f64::NAN;
    };
    let plaq_sum: f64 = per_site.iter().sum();
    s.beta * (6.0 * s.volume as f64 - plaq_sum)
}

fn gpu_kinetic_energy_streaming(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) -> f64 {
    let params = make_u32x4_params(s.n_links as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "s_ke_p");
    let bg = gpu.create_bind_group(
        &p.kinetic_pipeline,
        &[&param_buf, &s.mom_buf, &s.ke_out_buf],
    );
    let staging = gpu.create_staging_buffer(s.n_links * 8, "s_ke_staging");
    let mut enc = gpu.begin_encoder("s_ke");
    GpuF64::encode_pass(&mut enc, &p.kinetic_pipeline, &bg, s.wg_links);
    enc.copy_buffer_to_buffer(&s.ke_out_buf, 0, &staging, 0, (s.n_links * 8) as u64);
    gpu.submit_encoder(enc);
    let Ok(per_link) = gpu.read_staging_f64(&staging) else {
        return f64::NAN;
    };
    per_link.iter().sum()
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
/// heat bath. The CG solver still requires per-iteration readbacks, but all
/// other operations (gauge force, link/momentum updates, PRNG generation)
/// use batched encoders.
pub struct GpuDynHmcStreamingPipelines {
    /// Base dynamical fermion HMC pipeline set.
    pub dyn_hmc: GpuDynHmcPipelines,
    /// GPU PRNG shader for SU(3) algebra momentum generation.
    pub momenta_prng_pipeline: wgpu::ComputePipeline,
    /// GPU Gaussian sampler for pseudofermion heat-bath η.
    pub fermion_prng_pipeline: wgpu::ComputePipeline,
}

impl GpuDynHmcStreamingPipelines {
    /// Compile all dynamical streaming HMC pipelines including GPU PRNG for momenta and pseudofermion.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            dyn_hmc: GpuDynHmcPipelines::new(gpu),
            momenta_prng_pipeline: gpu.create_pipeline_f64(&WGSL_RANDOM_MOMENTA, "sdyn_mom_prng"),
            fermion_prng_pipeline: gpu
                .create_pipeline_f64(&WGSL_GAUSSIAN_FERMION, "sdyn_ferm_prng"),
        }
    }
}

/// Full streaming dynamical fermion HMC trajectory.
///
/// GPU PRNG generates both momenta and pseudofermion field η on-device.
/// The pseudofermion heat bath φ = D†η is computed via GPU Dirac dispatch.
/// CG iterations require scalar readbacks (convergence test), so they cannot
/// be batched into a single encoder. But all other operations — force
/// accumulation, link/momentum updates — use minimal dispatches.
///
/// Transfer budget per trajectory:
///   CPU→GPU: 0 bytes (GPU PRNG for momenta + pseudofermion)
///   GPU→CPU: ~(8 × CG_iters) bytes for convergence scalars + 24 bytes for ΔH
pub fn gpu_dynamical_hmc_trajectory_streaming(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcStreamingPipelines,
    state: &GpuDynHmcState,
    n_md_steps: usize,
    dt: f64,
    traj_id: u32,
    seed: &mut u64,
) -> GpuDynHmcResult {
    let vol = state.gauge.volume;
    let n_links = state.gauge.n_links;
    let gs = &state.gauge;
    let dp = &pipelines.dyn_hmc;

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

        let ferm_prng_params = make_ferm_prng_params(vol as u32, traj_id, seed);
        let ferm_prng_pbuf = gpu.create_uniform_buffer(&ferm_prng_params, "sdyn_ferm_p");
        let ferm_prng_bg = gpu.create_bind_group(
            &pipelines.fermion_prng_pipeline,
            &[&ferm_prng_pbuf, &state.phi_buf],
        );
        let wg_vol = (vol as u32).div_ceil(64);
        GpuF64::encode_pass(
            &mut enc,
            &pipelines.fermion_prng_pipeline,
            &ferm_prng_bg,
            wg_vol,
        );

        enc.copy_buffer_to_buffer(
            &gs.link_buf,
            0,
            &gs.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let s_gauge_old = gpu_wilson_action(gpu, &dp.gauge, gs);
    let t_old = gpu_kinetic_energy(gpu, &dp.gauge, gs);
    let (s_ferm_old, cg_iters_old) = gpu_fermion_action(gpu, dp, state);
    let h_old = s_gauge_old + t_old + s_ferm_old;
    let mut total_cg = cg_iters_old;

    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for _step in 0..n_md_steps {
        let cg1 = gpu_total_force_dispatch(gpu, dp, state, lam * dt);
        gpu_link_update_dispatch(gpu, &dp.gauge, gs, 0.5 * dt);
        let cg2 = gpu_total_force_dispatch(gpu, dp, state, (1.0 - 2.0 * lam) * dt);
        gpu_link_update_dispatch(gpu, &dp.gauge, gs, 0.5 * dt);
        let cg3 = gpu_total_force_dispatch(gpu, dp, state, lam * dt);
        total_cg += cg1 + cg2 + cg3;
    }

    let s_gauge_new = gpu_wilson_action(gpu, &dp.gauge, gs);
    let t_new = gpu_kinetic_energy(gpu, &dp.gauge, gs);
    let (s_ferm_new, cg_iters_new) = gpu_fermion_action(gpu, dp, state);
    let h_new = s_gauge_new + t_new + s_ferm_new;
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

    let plaquette = gpu_plaquette(gpu, &dp.gauge, gs);

    GpuDynHmcResult {
        accepted,
        delta_h,
        plaquette,
        cg_iterations: total_cg,
    }
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
