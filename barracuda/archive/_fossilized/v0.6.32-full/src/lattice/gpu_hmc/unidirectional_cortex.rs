// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unidirectional RHMC cortex — async dispatch and dual-GPU coordination.
//!
//! Provides `UnidirectionalRhmc` (single-GPU handle with fire/poll API) and
//! `dual_gpu_trajectories` (thread-parallel dispatch on two GPUs). The cortex
//! layer owns all GPU resources and presents a simple `run_trajectory` API
//! to the CPU scheduler.
//!
//! Split from `unidirectional_rhmc.rs` for the <1000 LOC rule.

use super::GpuF64;
use super::dynamical::GpuDynHmcPipelines;
use super::gpu_rhmc::{GpuRhmcPipelines, GpuRhmcState};
use super::resident_shifted_cg::GpuResidentShiftedCgBuffers;
use super::true_multishift_cg::TrueMultiShiftBuffers;
use super::uni_hamiltonian::{UniHamiltonianBuffers, UniPipelines};
use super::unidirectional_rhmc::gpu_rhmc_trajectory_unidirectional;
use crate::error::HotSpringError;
use crate::lattice::rhmc::RhmcConfig;

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
/// Owns the GPU, pipelines, state, and buffers. Provides `run_trajectory`
/// for sync dispatch, enabling dual-GPU cortex patterns where the CPU
/// async-schedules work to keep both GPUs saturated.
pub struct UnidirectionalRhmc {
    gpu: GpuF64,
    dyn_pipelines: GpuDynHmcPipelines,
    rhmc_pipelines: GpuRhmcPipelines,
    uni_pipelines: UniPipelines,
    state: GpuRhmcState,
    scg_bufs: GpuResidentShiftedCgBuffers,
    ham_bufs: UniHamiltonianBuffers,
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
        let vol = state.gauge.gauge.volume;
        let uni_pipelines = UniPipelines::new_saturated(&gpu, vol);
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
    pub fn run_trajectory(
        &mut self,
        config: &RhmcConfig,
        seed: &mut u64,
    ) -> Result<TrajectoryResult, HotSpringError> {
        if self.ms_bufs.is_none() {
            let max_shifts = config
                .sectors
                .iter()
                .map(|s| s.action_approx.sigma.len().max(s.force_approx.sigma.len()))
                .max()
                .unwrap_or(0);
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
        )?;
        Ok(TrajectoryResult {
            accepted: result.accepted,
            delta_h: result.delta_h,
            plaquette: result.plaquette,
            total_cg_iterations: result.total_cg_iterations,
            elapsed_secs: t0.elapsed().as_secs_f64(),
        })
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

    /// Query which silicon routing tiers are active on this instance.
    pub fn silicon_routing_tags(&self) -> super::brain_rhmc::SiliconRoutingTags {
        use barracuda::device::driver_profile::Fp64Strategy;
        super::brain_rhmc::SiliconRoutingTags {
            tmu_prng: false, // TMU PRNG is on streaming path (GpuHmcStreamingPipelines), not unidirectional
            subgroup_reduce: self.gpu.has_subgroups,
            rop_force_accum: self.uni_pipelines.rop_accum.is_some(),
            fp64_strategy_id: match self.gpu.fp64_strategy() {
                Fp64Strategy::Sovereign => 0,
                Fp64Strategy::Native => 1,
                Fp64Strategy::Hybrid => 2,
                Fp64Strategy::Concurrent => 3,
            },
            has_native_f64: self.gpu.has_f64,
        }
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
///
/// # Errors
///
/// Returns [`HotSpringError::ThreadPanicked`] if a worker thread panicked.
pub fn dual_gpu_trajectories(
    gpu_a: &mut UnidirectionalRhmc,
    gpu_b: &mut UnidirectionalRhmc,
    config: &RhmcConfig,
    seed_a: &mut u64,
    seed_b: &mut u64,
) -> Result<DualGpuResult, HotSpringError> {
    let config_a = config.clone();
    let config_b = config.clone();
    let mut sa = *seed_a;
    let mut sb = *seed_b;

    let (result_a, result_b) = std::thread::scope(|scope| {
        let handle_a = scope.spawn(|| -> Result<(TrajectoryResult, u64), HotSpringError> {
            let r = gpu_a.run_trajectory(&config_a, &mut sa)?;
            Ok((r, sa))
        });
        let handle_b = scope.spawn(|| -> Result<(TrajectoryResult, u64), HotSpringError> {
            let r = gpu_b.run_trajectory(&config_b, &mut sb)?;
            Ok((r, sb))
        });
        let (ra, new_sa) = handle_a
            .join()
            .map_err(|_| HotSpringError::ThreadPanicked("GPU A trajectory thread panicked"))??;
        let (rb, new_sb) = handle_b
            .join()
            .map_err(|_| HotSpringError::ThreadPanicked("GPU B trajectory thread panicked"))??;
        *seed_a = new_sa;
        *seed_b = new_sb;
        Ok::<(TrajectoryResult, TrajectoryResult), HotSpringError>((ra, rb))
    })?;

    Ok(DualGpuResult {
        a: result_a,
        b: result_b,
    })
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "GPU/tokio test setup uses unwrap and expect on known-good paths."
)]
mod tests {
    use super::super::uni_hamiltonian::{
        WGSL_FERMION_ACTION_SUM, WGSL_HAMILTONIAN_ASSEMBLY, WGSL_METROPOLIS,
        make_fermion_action_sum_params, make_h_assembly_params,
    };
    use super::*;

    fn make_gpu() -> GpuF64 {
        let rt = tokio::runtime::Runtime::new().expect("tokio");
        rt.block_on(GpuF64::new()).expect("GPU required for test")
    }

    const WGSL_CONSTANT_WRITE: &str = r"
@group(0) @binding(0) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main() {
    out[0] = f64(42.5);
}
";

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

    const WGSL_PASSTHROUGH: &str = r"
@group(0) @binding(0) var<storage, read> inp: array<f64>;
@group(0) @binding(1) var<storage, read_write> out: array<f64>;
@compute @workgroup_size(1)
fn main() {
    out[0] = inp[0];
    out[1] = inp[0] + f64(1.0);
}
";

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
        let bg =
            gpu.create_bind_group(&pl, &[&pbuf, &plaq_buf, &t_buf, &sf_buf, &h_buf, &diag_buf]);

        let mut enc = gpu.begin_encoder("test_h_asm");
        GpuF64::encode_pass(&mut enc, &pl, &bg, 1);
        gpu.submit_encoder(enc);

        let h_val = gpu.read_back_f64(&h_buf, 1).unwrap();
        let diag = gpu.read_back_f64(&diag_buf, 3).unwrap();

        let expected_sg = beta * (six_v - 100.0);
        let expected_h = expected_sg + 5.0 + 3.0;

        eprintln!("H assembly test: h={}, expected={}", h_val[0], expected_h);
        eprintln!("  diag: sg={} t={} sf={}", diag[0], diag[1], diag[2]);

        assert!(
            (h_val[0] - expected_h).abs() < 1e-6,
            "H mismatch: got {} expected {}",
            h_val[0],
            expected_h
        );
        assert!((diag[0] - expected_sg).abs() < 1e-6);
        assert!((diag[1] - 5.0).abs() < 1e-6);
        assert!((diag[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn fermion_action_sum_kernel_roundtrip() {
        let gpu = make_gpu();

        let dots_buf = gpu.create_f64_output_buffer(4, "test_dots");
        let sf_buf = gpu.create_f64_output_buffer(1, "test_sf");

        gpu.upload_f64(&dots_buf, &[10.0, 20.0, 30.0, 40.0]);
        gpu.zero_buffer(&sf_buf, 8);

        let pl = gpu.create_pipeline_f64(WGSL_FERMION_ACTION_SUM, "test_sf_sum");

        let n_dots = 4u32;
        let alpha_0 = 0.5f64;
        let params = make_fermion_action_sum_params(n_dots, alpha_0);
        let pbuf = gpu.create_storage_buffer_init(&params, "test_sf_p");

        let alphas: Vec<u8> = [1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|a| a.to_le_bytes())
            .collect();
        let alpha_buf = gpu.create_storage_buffer_init(&alphas, "test_alphas");

        let bg = gpu.create_bind_group(&pl, &[&pbuf, &dots_buf, &alpha_buf, &sf_buf]);

        let mut enc = gpu.begin_encoder("test_sf_sum");
        GpuF64::encode_pass(&mut enc, &pl, &bg, 1);
        gpu.submit_encoder(enc);

        let sf = gpu.read_back_f64(&sf_buf, 1).unwrap();
        eprintln!("S_f sum test: sf={}, expected=205.0", sf[0]);
        assert!(
            (sf[0] - 205.0).abs() < 1e-6,
            "S_f mismatch: got {} expected 205.0",
            sf[0]
        );
    }

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

        let bg = gpu.create_bind_group(
            &pl,
            &[&pbuf, &h_old, &h_new, &plaq, &diag_old, &diag_new, &result],
        );

        let mut enc = gpu.begin_encoder("test_metro");
        GpuF64::encode_pass(&mut enc, &pl, &bg, 1);
        enc.copy_buffer_to_buffer(&result, 0, &staging, 0, 9 * 8);
        gpu.submit_encoder(enc);

        let data = gpu.read_staging_f64_n(&staging, 9).unwrap();
        eprintln!(
            "Metropolis test: accepted={} dH={} plaq={}",
            data[0], data[1], data[2]
        );

        assert!(data[0] > 0.5, "Should accept: delta_h=-1.0");
        assert!(
            (data[1] - (-1.0)).abs() < 1e-6,
            "delta_h wrong: {}",
            data[1]
        );
        assert!(
            (data[2] - 12000.0 / 24576.0).abs() < 1e-6,
            "plaq wrong: {}",
            data[2]
        );
        assert!(
            (data[7] - 5.0).abs() < 1e-6,
            "s_ferm_old wrong: {}",
            data[7]
        );
        assert!(
            (data[8] - 5.0).abs() < 1e-6,
            "s_ferm_new wrong: {}",
            data[8]
        );
    }
}
