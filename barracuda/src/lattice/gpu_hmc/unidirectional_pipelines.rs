// SPDX-License-Identifier: AGPL-3.0-only

//! Pipeline compilation for unidirectional GPU RHMC (ALU/TMU PRNG, B2+B3 kernels, optional ROP).

use super::dynamical::{WGSL_RANDOM_MOMENTA, WGSL_RANDOM_MOMENTA_TMU};
use super::resident_shifted_cg::GpuResidentShiftedCgPipelines;
use super::rop_force_accum::RopForceAccumulator;
use super::streaming::WGSL_GAUSSIAN_FERMION;
use super::tmu_tables::TmuLookupTables;
use super::true_multishift_cg::TrueMultiShiftPipelines;
use crate::gpu::GpuF64;

/// WGSL: `H = beta*(6V - plaq) + T + S_f` (single-thread).
pub const WGSL_HAMILTONIAN_ASSEMBLY: &str =
    include_str!("../shaders/hamiltonian_assembly_f64.wgsl");
/// WGSL: weighted fermion action sum from dot products.
pub const WGSL_FERMION_ACTION_SUM: &str = include_str!("../shaders/fermion_action_sum_f64.wgsl");
/// WGSL: Metropolis accept/reject + diagnostics.
pub const WGSL_METROPOLIS: &str = include_str!("../shaders/metropolis_f64.wgsl");

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
