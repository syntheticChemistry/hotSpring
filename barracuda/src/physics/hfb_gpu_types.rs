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
pub struct PotentialDimsUniform {
    pub nr: u32,
    pub batch_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HamiltonianDimsUniform {
    pub n_states: u32,
    pub nr: u32,
    pub batch_size: u32,
    pub _pad: u32,
}

/// Uniform for density shader (group 0). Must match `DensityParams` in WGSL.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DensityParamsUniform {
    pub n_states: u32,
    pub nr: u32,
    pub batch_size: u32,
    pub _pad: u32,
}

/// Uniform for density mixing (group 3). Must match `MixParams` in WGSL.
///
/// Alpha is passed as (numerator, denominator) to avoid bitcast issues
/// in Naga's WGSL validator (`bitcast<f64>(vec2<u32>)` not yet supported).
/// Alpha = `f64(alpha_num)` / `f64(alpha_den)`. For typical values (0.3, 0.8),
/// this gives exact representation with integer ratios.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MixParamsUniform {
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

/// Encode f64 as (lo, hi) u32 pair for WGSL `bitcast<f64>(vec2<u32>(lo, hi))`.
#[cfg(feature = "gpu_energy")]
pub(crate) fn f64_to_u32_pair(x: f64) -> (u32, u32) {
    let u: u64 = bytemuck::cast(x);
    ((u & 0xFFFF_FFFF) as u32, (u >> 32) as u32)
}

/// Uniform for energy shader (group 0). Must match `EnergyParams` in WGSL.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EnergyParamsUniform {
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
pub struct PackParams {
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

pub fn make_bind_group(
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
        .map(
            |(i, (_, buf)): (usize, &(wgpu::BufferBindingType, &wgpu::Buffer))| {
                wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buf.as_entire_binding(),
                }
            },
        )
        .collect();
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &layout,
        entries: &bg_entries,
    });
    (layout, bg)
}

pub fn make_pipeline(
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

// Energy pipeline (potential integrands + pairing) when gpu_energy feature is enabled.
pub struct GroupResources {
    pub ns: usize,
    pub nr: usize,
    pub group_indices: Vec<usize>,
    pub n_max: usize,

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
    #[allow(dead_code)] // GPU buffers kept alive for bind group validity
    pub v2_p_buf: wgpu::Buffer,
    #[allow(dead_code)]
    pub v2_n_buf: wgpu::Buffer,
    #[allow(dead_code)]
    pub rho_p_new_buf: wgpu::Buffer,
    #[allow(dead_code)]
    pub rho_n_new_buf: wgpu::Buffer,
    pub mix_params_buf: wgpu::Buffer,
    pub bcs_v2_pipe: wgpu::ComputePipeline,
    pub density_pipe: wgpu::ComputePipeline,
    pub mix_pipe: wgpu::ComputePipeline,
    pub rho_p_staging: wgpu::Buffer,
    pub rho_n_staging: wgpu::Buffer,

    #[cfg(feature = "gpu_energy")]
    pub energy_params_buf: wgpu::Buffer,
    #[cfg(feature = "gpu_energy")]
    pub energy_integrands_bg: wgpu::BindGroup,
    #[cfg(feature = "gpu_energy")]
    pub energy_pair_bg: wgpu::BindGroup,
    #[cfg(feature = "gpu_energy")]
    pub energy_integrands_pipe: wgpu::ComputePipeline,
    #[cfg(feature = "gpu_energy")]
    pub pairing_energy_pipe: wgpu::ComputePipeline,
    #[cfg(feature = "gpu_energy")]
    pub energy_integrands_buf: wgpu::Buffer,
    #[cfg(feature = "gpu_energy")]
    pub e_pair_buf: wgpu::Buffer,
    #[cfg(feature = "gpu_energy")]
    pub energy_staging: wgpu::Buffer,
    #[cfg(feature = "gpu_energy")]
    pub e_pair_staging: wgpu::Buffer,

    pub nr_wg: u32,
    pub ns_wg: u32,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
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
        let alpha_recovered = f64::from(u.alpha_num) / f64::from(u.alpha_den);
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
        let alpha = f64::from(u_one.alpha_num) / f64::from(u_one.alpha_den);
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

    #[test]
    fn potential_dims_uniform_zeroable() {
        let z: PotentialDimsUniform = bytemuck::Zeroable::zeroed();
        assert_eq!(z.nr, 0);
        assert_eq!(z.batch_size, 0);
    }

    #[test]
    fn hamiltonian_dims_uniform_zeroable() {
        let z: HamiltonianDimsUniform = bytemuck::Zeroable::zeroed();
        assert_eq!(z.n_states, 0);
        assert_eq!(z.nr, 0);
    }

    #[test]
    fn density_params_uniform_zeroable() {
        let z: DensityParamsUniform = bytemuck::Zeroable::zeroed();
        assert_eq!(z.n_states, 0);
    }

    #[test]
    fn mix_params_uniform_zeroable() {
        let z: MixParamsUniform = bytemuck::Zeroable::zeroed();
        assert_eq!(z.total_size, 0);
        assert_eq!(z.alpha_num, 0);
    }

    #[test]
    fn energy_params_uniform_zeroable() {
        let z: EnergyParamsUniform = bytemuck::Zeroable::zeroed();
        assert_eq!(z.n_states, 0);
    }

    #[test]
    fn pack_params_zeroable() {
        let z: PackParams = bytemuck::Zeroable::zeroed();
        assert_eq!(z.ns, 0);
        assert_eq!(z.gns, 0);
    }

    #[test]
    fn mix_params_uniform_various_alphas() {
        let u = MixParamsUniform::new(64, 0.5);
        let alpha = f64::from(u.alpha_num) / f64::from(u.alpha_den);
        assert!((alpha - 0.5).abs() < 1e-6);

        let u = MixParamsUniform::new(128, 0.8);
        let alpha = f64::from(u.alpha_num) / f64::from(u.alpha_den);
        assert!((alpha - 0.8).abs() < 1e-6);
    }

    // Non-zero construction with realistic HFB parameters
    #[test]
    fn potential_dims_uniform_realistic() {
        let u = PotentialDimsUniform {
            nr: 120,
            batch_size: 10,
        };
        assert_eq!(u.nr, 120);
        assert_eq!(u.batch_size, 10);
        assert_eq!(std::mem::size_of::<PotentialDimsUniform>(), 8);
    }

    #[test]
    fn hamiltonian_dims_uniform_realistic() {
        let u = HamiltonianDimsUniform {
            n_states: 30,
            nr: 100,
            batch_size: 18,
            _pad: 0,
        };
        assert_eq!(u.n_states, 30);
        assert_eq!(u.nr, 100);
        assert_eq!(std::mem::size_of::<HamiltonianDimsUniform>(), 16);
    }

    #[test]
    fn density_params_uniform_realistic() {
        let u = DensityParamsUniform {
            n_states: 30,
            nr: 100,
            batch_size: 18,
            _pad: 0,
        };
        assert_eq!(u.n_states, 30);
        assert_eq!(u.nr, 100);
        assert_eq!(std::mem::size_of::<DensityParamsUniform>(), 16);
    }

    #[test]
    fn energy_params_uniform_nonzero_layout() {
        let u = EnergyParamsUniform {
            n_states: 30,
            nr: 100,
            batch_size: 18,
            _pad: 0,
            t0_lo: 1,
            t0_hi: 2,
            t3_lo: 3,
            t3_hi: 4,
            x0_lo: 5,
            x0_hi: 6,
            x3_lo: 7,
            x3_hi: 8,
            alpha_lo: 9,
            alpha_hi: 10,
            dr_lo: 11,
            dr_hi: 12,
            hw_lo: 13,
            hw_hi: 14,
        };
        assert_eq!(u.n_states, 30);
        assert_eq!(u.t0_lo, 1);
        assert_eq!(std::mem::size_of::<EnergyParamsUniform>(), 72);
    }

    #[test]
    fn pack_params_realistic() {
        let p = PackParams {
            ns: 30,
            gns: 15,
            n_active: 25,
            dst_start: 0,
            dst_stride: 36,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        assert_eq!(p.ns, 30);
        assert_eq!(p.gns, 15);
        assert_eq!(std::mem::size_of::<PackParams>(), 32);
    }

    #[test]
    fn mix_params_uniform_realistic_scf() {
        let u = MixParamsUniform::new(120 * 2, 0.3);
        assert_eq!(u.total_size, 240);
        let alpha = f64::from(u.alpha_num) / f64::from(u.alpha_den);
        assert!((alpha - 0.3).abs() < 1e-6);
    }
}
