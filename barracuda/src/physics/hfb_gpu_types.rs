// SPDX-License-Identifier: AGPL-3.0-only

//! GPU buffer types and helpers for the HFB GPU-resident pipeline.
//!
//! Extracted from `hfb_gpu_resident.rs` for module focus and code size.
//! Contains uniform structs (matching WGSL shader layouts), bind-group
//! and pipeline construction helpers, and the per-group resource struct.

// ═══════════════════════════════════════════════════════════════════
// Uniform buffer layouts (must match WGSL shader structs)
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PotentialDimsUniform {
    pub nr: u32,
    pub batch_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct HamiltonianDimsUniform {
    pub n_states: u32,
    pub nr: u32,
    pub batch_size: u32,
    pub _pad: u32,
}

/// Uniform for density shader (group 0). Must match `DensityParams` in WGSL.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DensityParamsUniform {
    pub n_states: u32,
    pub nr: u32,
    pub batch_size: u32,
    pub _pad: u32,
}

/// Uniform for density mixing (group 3). Must match `MixParams` in WGSL.
///
/// Alpha is passed as (numerator, denominator) to avoid bitcast issues
/// in Naga's WGSL validator (`bitcast<f64>(vec2<u32>)` not yet supported).
/// Alpha = f64(alpha_num) / f64(alpha_den). For typical values (0.3, 0.8),
/// this gives exact representation with integer ratios.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MixParamsUniform {
    pub total_size: u32,
    pub _pad1: u32,
    pub alpha_num: u32,
    pub alpha_den: u32,
}

impl MixParamsUniform {
    const SCALE: f64 = 10_000_000.0;

    pub fn new(total_size: u32, alpha: f64) -> Self {
        let num = (alpha * Self::SCALE).round() as u32;
        let den = Self::SCALE as u32;
        Self {
            total_size,
            _pad1: 0,
            alpha_num: num,
            alpha_den: den,
        }
    }
}

/// Uniform for energy shader (group 0). Must match `EnergyParams` in WGSL.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct EnergyParamsUniform {
    pub n_states: u32,
    pub nr: u32,
    pub batch_size: u32,
    pub _pad: u32,
    pub t0_lo: u32,
    pub t0_hi: u32,
    pub t3_lo: u32,
    pub t3_hi: u32,
    pub x0_lo: u32,
    pub x0_hi: u32,
    pub x3_lo: u32,
    pub x3_hi: u32,
    pub alpha_lo: u32,
    pub alpha_hi: u32,
    pub dr_lo: u32,
    pub dr_hi: u32,
    pub hw_lo: u32,
    pub hw_hi: u32,
}

/// Uniform for spin-orbit pack shader. Must match `PackParams` in WGSL.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PackParams {
    pub ns: u32,
    pub gns: u32,
    pub n_active: u32,
    pub dst_start: u32,
    pub dst_stride: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

// ═══════════════════════════════════════════════════════════════════
// GPU resource helpers
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn make_bind_group(
    device: &wgpu::Device,
    label: &str,
    entries: &[(wgpu::BufferBindingType, &wgpu::Buffer)],
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = entries
        .iter()
        .enumerate()
        .map(|(i, (ty, _))| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: *ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        entries: &layout_entries,
    });
    let bg_entries: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .enumerate()
        .map(|(i, (_, buf))| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &layout,
        entries: &bg_entries,
    });
    (layout, bg)
}

pub(crate) fn make_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry_point),
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&pl),
        module,
        entry_point,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Per-group GPU resources (pre-allocated once, reused across SCF)
// ═══════════════════════════════════════════════════════════════════

// EVOLUTION(GPU): Phase 4 energy fields (energy_*, e_pair_*, v2_*, rho_*_new_*) are wired
// into bind groups but dispatches deferred until SumReduceF64 + kinetic energy kernel available.
#[allow(dead_code)] // EVOLUTION: struct used; some fields deferred until energy kernel wired
pub(crate) struct GroupResources {
    pub ns: usize,
    pub nr: usize,
    pub group_indices: Vec<usize>,
    pub n_max: usize,
    pub mat_size: usize,

    pub rho_p_buf: wgpu::Buffer,
    pub rho_n_buf: wgpu::Buffer,
    pub rho_alpha_buf: wgpu::Buffer,
    pub rho_alpha_m1_buf: wgpu::Buffer,
    pub pot_dims_buf: wgpu::Buffer,
    pub ham_dims_buf: wgpu::Buffer,
    pub h_p_buf: wgpu::Buffer,
    pub h_n_buf: wgpu::Buffer,

    pub so_diag_buf: wgpu::Buffer,

    pub pot_bg: wgpu::BindGroup,
    pub hbg_p: wgpu::BindGroup,
    pub hbg_n: wgpu::BindGroup,

    pub sky_pipe: wgpu::ComputePipeline,
    pub cfwd: wgpu::ComputePipeline,
    pub cbwd: wgpu::ComputePipeline,
    pub fin_pipe: wgpu::ComputePipeline,
    pub fq_pipe: wgpu::ComputePipeline,
    pub hp_pipe: wgpu::ComputePipeline,
    pub hn_pipe: wgpu::ComputePipeline,

    pub density_params_buf: wgpu::Buffer,
    pub density_params_bg: wgpu::BindGroup,
    pub bcs_p_bg: wgpu::BindGroup,
    pub bcs_n_bg: wgpu::BindGroup,
    pub bcs_p_read_bg: wgpu::BindGroup,
    pub bcs_n_read_bg: wgpu::BindGroup,
    pub density_p_bg: wgpu::BindGroup,
    pub density_n_bg: wgpu::BindGroup,
    pub density_p_read_bg: wgpu::BindGroup,
    pub density_n_read_bg: wgpu::BindGroup,
    pub mix_p_bg: wgpu::BindGroup,
    pub mix_n_bg: wgpu::BindGroup,
    pub evals_p_buf: wgpu::Buffer,
    pub evals_n_buf: wgpu::Buffer,
    pub evecs_p_buf: wgpu::Buffer,
    pub evecs_n_buf: wgpu::Buffer,
    pub lambda_p_buf: wgpu::Buffer,
    pub lambda_n_buf: wgpu::Buffer,
    pub delta_buf: wgpu::Buffer,
    pub v2_p_buf: wgpu::Buffer,
    pub v2_n_buf: wgpu::Buffer,
    pub rho_p_new_buf: wgpu::Buffer,
    pub rho_n_new_buf: wgpu::Buffer,
    pub mix_params_buf: wgpu::Buffer,
    pub bcs_v2_pipe: wgpu::ComputePipeline,
    pub density_pipe: wgpu::ComputePipeline,
    pub mix_pipe: wgpu::ComputePipeline,
    pub rho_p_staging: wgpu::Buffer,
    pub rho_n_staging: wgpu::Buffer,

    pub energy_params_bg: wgpu::BindGroup,
    pub energy_pair_bg: wgpu::BindGroup,
    pub energy_integrands_buf: wgpu::Buffer,
    pub e_pair_buf: wgpu::Buffer,
    pub energy_staging: wgpu::Buffer,
    pub e_pair_staging: wgpu::Buffer,

    pub nr_wg: u32,
    pub ns_wg: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn potential_dims_uniform_layout() {
        let u = PotentialDimsUniform {
            nr: 100,
            batch_size: 18,
        };
        assert_eq!(u.nr, 100);
        assert_eq!(u.batch_size, 18);
        assert_eq!(std::mem::size_of::<PotentialDimsUniform>(), 8);
    }

    #[test]
    fn hamiltonian_dims_uniform_layout() {
        let u = HamiltonianDimsUniform {
            n_states: 30,
            nr: 100,
            batch_size: 18,
            _pad: 0,
        };
        assert_eq!(u.n_states, 30);
        assert_eq!(std::mem::size_of::<HamiltonianDimsUniform>(), 16);
    }

    #[test]
    fn density_params_uniform_layout() {
        let u = DensityParamsUniform {
            n_states: 30,
            nr: 100,
            batch_size: 18,
            _pad: 0,
        };
        assert_eq!(u.n_states, 30);
        assert_eq!(std::mem::size_of::<DensityParamsUniform>(), 16);
    }

    #[test]
    fn mix_params_uniform_alpha_encoding() {
        let u = MixParamsUniform::new(200, 0.3);
        assert_eq!(u.total_size, 200);
        let alpha_recovered = u.alpha_num as f64 / u.alpha_den as f64;
        assert!(
            (alpha_recovered - 0.3).abs() < 1e-6,
            "alpha round-trip: {alpha_recovered}"
        );
    }

    #[test]
    fn mix_params_uniform_extreme_alpha() {
        let u_zero = MixParamsUniform::new(100, 0.0);
        assert_eq!(u_zero.alpha_num, 0);

        let u_one = MixParamsUniform::new(100, 1.0);
        let alpha = u_one.alpha_num as f64 / u_one.alpha_den as f64;
        assert!((alpha - 1.0).abs() < 1e-6);
    }

    #[test]
    fn energy_params_uniform_layout() {
        assert_eq!(std::mem::size_of::<EnergyParamsUniform>(), 72);
    }

    #[test]
    fn pack_params_layout() {
        let p = PackParams {
            ns: 10,
            gns: 5,
            n_active: 8,
            dst_start: 0,
            dst_stride: 30,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        assert_eq!(p.ns, 10);
        assert_eq!(std::mem::size_of::<PackParams>(), 32);
    }
}
