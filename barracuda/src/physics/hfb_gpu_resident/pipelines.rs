// SPDX-License-Identifier: AGPL-3.0-only

//! GPU compute pipeline factories for the HFB solver.
//!
//! Encapsulates shader module → compute pipeline creation for each HFB
//! pipeline stage: potentials (Skyrme, Coulomb, `f_q`), Hamiltonian build,
//! BCS v², density, mixing, and (optionally) energy.

use super::super::hfb_gpu_types::make_pipeline;

/// Create the five potential-stage pipelines from a single shader module.
///
/// Returns `(skyrme, coulomb_forward, coulomb_backward, finalize_proton, compute_f_q)`.
pub(super) fn create_potential_pipelines(
    raw_device: &wgpu::Device,
    pot_module: &wgpu::ShaderModule,
    pot_layout: &wgpu::BindGroupLayout,
) -> (
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
) {
    let sky_pipe = make_pipeline(raw_device, pot_module, "compute_skyrme", &[pot_layout]);
    let cfwd = make_pipeline(
        raw_device,
        pot_module,
        "compute_coulomb_forward",
        &[pot_layout],
    );
    let cbwd = make_pipeline(
        raw_device,
        pot_module,
        "compute_coulomb_backward",
        &[pot_layout],
    );
    let fin_pipe = make_pipeline(
        raw_device,
        pot_module,
        "finalize_proton_potential",
        &[pot_layout],
    );
    let fq_pipe = make_pipeline(raw_device, pot_module, "compute_f_q", &[pot_layout]);
    (sky_pipe, cfwd, cbwd, fin_pipe, fq_pipe)
}

/// Create the proton and neutron Hamiltonian-build pipelines.
pub(super) fn create_hamiltonian_pipelines(
    raw_device: &wgpu::Device,
    ham_module: &wgpu::ShaderModule,
    ham_layout: &wgpu::BindGroupLayout,
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let hp_pipe = make_pipeline(raw_device, ham_module, "build_hamiltonian", &[ham_layout]);
    let hn_pipe = make_pipeline(raw_device, ham_module, "build_hamiltonian", &[ham_layout]);
    (hp_pipe, hn_pipe)
}

/// Create the BCS v², density, and mixing pipelines.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_density_pipelines(
    raw_device: &wgpu::Device,
    density_module: &wgpu::ShaderModule,
    dp_layout: &wgpu::BindGroupLayout,
    bcs_write_layout: &wgpu::BindGroupLayout,
    bcs_read_layout: &wgpu::BindGroupLayout,
    dens_write_layout: &wgpu::BindGroupLayout,
    dens_read_layout: &wgpu::BindGroupLayout,
    mix_layout: &wgpu::BindGroupLayout,
) -> (
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
) {
    let bcs_v2_pipe = make_pipeline(
        raw_device,
        density_module,
        "compute_bcs_v2",
        &[dp_layout, bcs_write_layout],
    );
    let density_pipe = make_pipeline(
        raw_device,
        density_module,
        "compute_density",
        &[dp_layout, bcs_read_layout, dens_write_layout],
    );
    let mix_pipe = make_pipeline(
        raw_device,
        density_module,
        "mix_density",
        &[dp_layout, bcs_read_layout, dens_read_layout, mix_layout],
    );
    (bcs_v2_pipe, density_pipe, mix_pipe)
}
