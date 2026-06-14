// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::gpu::GpuF64;

pub fn gpu_dirac_dispatch(
    gpu: &GpuF64,
    pipelines: &super::dynamical::GpuDynHmcPipelines,
    state: &super::dynamical::GpuDynHmcState,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    hop_sign: f64,
) {
    let vol = state.gauge.volume;
    let wg = (vol as u32).div_ceil(64);
    let mut params = Vec::with_capacity(24);
    params.extend_from_slice(&(vol as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&state.mass.to_le_bytes());
    params.extend_from_slice(&hop_sign.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "dirac_p");
    let bg = gpu.create_bind_group(
        &pipelines.dirac_pipeline,
        &[
            &pbuf,
            &state.gauge.link_buf,
            input,
            output,
            &state.gauge.nbr_buf,
            &state.phases_buf,
        ],
    );
    gpu.dispatch(&pipelines.dirac_pipeline, &bg, wg);
}

pub fn gpu_fermion_force_dispatch(
    gpu: &GpuF64,
    pipelines: &super::dynamical::GpuDynHmcPipelines,
    state: &super::dynamical::GpuDynHmcState,
) {
    let vol = state.gauge.volume;
    let wg = (vol as u32).div_ceil(64);
    let params = super::make_u32x4_params(vol as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "fforce_p");
    let bg = gpu.create_bind_group(
        &pipelines.fermion_force_pipeline,
        &[
            &pbuf,
            &state.gauge.link_buf,
            &state.x_buf,
            &state.y_buf,
            &state.gauge.nbr_buf,
            &state.phases_buf,
            &state.ferm_force_buf,
        ],
    );
    gpu.dispatch(&pipelines.fermion_force_pipeline, &bg, wg);
}

#[deprecated(note = "use GPU-resident CG scalar buffers (resident_cg.rs / resident_shifted_cg.rs)")]
pub fn gpu_dot_re(
    gpu: &GpuF64,
    dot_pl: &wgpu::ComputePipeline,
    dot_buf: &wgpu::Buffer,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    n_pairs: usize,
) -> f64 {
    let wg = (n_pairs as u32).div_ceil(64);
    let params = super::make_u32x4_params(n_pairs as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "dot_p");
    let bg = gpu.create_bind_group(dot_pl, &[&pbuf, a, b, dot_buf]);
    gpu.dispatch(dot_pl, &bg, wg);
    match gpu.read_back_f64(dot_buf, n_pairs) {
        Ok(v) => v.iter().sum(),
        Err(_) => f64::NAN,
    }
}
