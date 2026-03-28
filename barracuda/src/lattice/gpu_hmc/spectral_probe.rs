// SPDX-License-Identifier: AGPL-3.0-only

//! GPU spectral probing for RHMC self-tuning.
//!
//! Discovers the eigenvalue range [λ_min, λ_max] of D†D on the current
//! gauge configuration, replacing hardcoded spectral ranges with measured
//! values. The rational approximation quality depends critically on
//! these bounds — too narrow risks catastrophic error at the extremes,
//! too wide wastes pole budget on unreachable spectral regions.
//!
//! # Algorithms
//!
//! - **λ_max**: GPU power iteration — repeatedly apply D†D and normalize.
//!   Converges geometrically to the largest eigenvalue. Cost: ~20 Dirac
//!   applications (cheap relative to a full CG solve).
//!
//! - **λ_min**: Analytical bound from the fermion mass. For staggered
//!   fermions, λ_min(D†D) ≥ m² (free-field bound, tight at weak coupling).
//!   Interaction effects can only increase the smallest eigenvalue for
//!   positive mass, so m² is a safe lower bound.
//!
//! # References
//!
//! - Golub & Van Loan, "Matrix Computations" 4th ed., §7.3 — power method
//! - Kalkreuter, hep-lat/9511009 — eigenvalue estimation for LQCD

use super::dynamical::GpuDynHmcPipelines;
#[allow(deprecated)]
use super::{gpu_dot_re, GpuDynHmcState, GpuF64};
use crate::tolerances::{
    RHMC_POWER_ITERATION_COUNT, RHMC_SPECTRAL_SAFETY_HIGH, RHMC_SPECTRAL_SAFETY_LOW,
};

/// Measured eigenvalue bounds of D†D on a gauge configuration.
#[derive(Clone, Debug)]
pub struct SpectralInfo {
    /// Estimated smallest eigenvalue of D†D.
    pub lambda_min: f64,
    /// Estimated largest eigenvalue of D†D.
    pub lambda_max: f64,
    /// Safe lower bound for rational approximation range.
    pub range_min: f64,
    /// Safe upper bound for rational approximation range.
    pub range_max: f64,
    /// Whether lambda_min was analytically bounded (vs measured).
    pub lambda_min_is_bound: bool,
}

impl SpectralInfo {
    /// Construct from measured/bounded eigenvalues with safety margins.
    #[must_use]
    pub fn new(lambda_min: f64, lambda_max: f64, lambda_min_is_bound: bool) -> Self {
        let range_min = (RHMC_SPECTRAL_SAFETY_LOW * lambda_min).max(1e-6);
        let range_max = RHMC_SPECTRAL_SAFETY_HIGH * lambda_max;
        Self {
            lambda_min,
            lambda_max,
            range_min,
            range_max,
            lambda_min_is_bound,
        }
    }

    /// Default conservative estimate (equivalent to old hardcoded [0.01, 64]).
    #[must_use]
    pub fn conservative_default(mass: f64) -> Self {
        Self::new(mass * mass, 64.0, true)
    }
}

/// Estimate λ_max(D†D) via GPU power iteration.
///
/// Starting from a random vector v, iterates w = D†D·v, then
/// λ ≈ ‖w‖/‖v‖. After `n_iter` steps, the Rayleigh quotient
/// converges geometrically with rate |λ₂/λ₁|.
///
/// Reuses the existing Dirac dispatch and dot product pipelines.
/// Uses `gpu_dot_re` (3 calls per iteration × ~5 iterations = ~15 total).
/// Low call count; not a hot path.
#[allow(deprecated)]
pub fn gpu_power_iteration_lambda_max(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    mass: f64,
    n_iter: usize,
) -> f64 {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;
    let n_pairs = vol * 3;
    let gauge = &state.gauge;
    let phases = &state.phases_buf;

    // Initialize with a deterministic pseudo-random vector.
    // Using a simple pattern avoids depending on RNG state.
    let mut v: Vec<f64> = (0..n_flat)
        .map(|i| {
            let t = i as f64 * 0.618_033_988_749_895;
            (t - t.floor()) * 2.0 - 1.0
        })
        .collect();

    // Normalize initial vector
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-30 {
        return 64.0; // fallback
    }
    for x in &mut v {
        *x /= norm;
    }

    // Upload to p_buf (workspace)
    gpu.upload_f64(&state.p_buf, &v);

    let mut lambda_est = 0.0;

    for _ in 0..n_iter {
        // w = D · v  (stored in temp_buf)
        super::gpu_rhmc::dirac_dispatch(
            gpu, pipelines, gauge, phases, &state.p_buf, &state.temp_buf, mass, 1.0,
        );
        // w = D† · (D · v)  (stored in ap_buf)
        super::gpu_rhmc::dirac_dispatch(
            gpu, pipelines, gauge, phases, &state.temp_buf, &state.ap_buf, mass, -1.0,
        );

        // Rayleigh quotient: λ ≈ ⟨v|D†D·v⟩ / ⟨v|v⟩
        let v_av = gpu_dot_re(
            gpu,
            &pipelines.dot_pipeline,
            &state.dot_buf,
            &state.p_buf,
            &state.ap_buf,
            n_pairs,
        );
        let v_v = gpu_dot_re(
            gpu,
            &pipelines.dot_pipeline,
            &state.dot_buf,
            &state.p_buf,
            &state.p_buf,
            n_pairs,
        );

        if v_v > 1e-30 {
            lambda_est = v_av / v_v;
        }

        // Normalize: v = D†D·v / ‖D†D·v‖
        let w_norm_sq = gpu_dot_re(
            gpu,
            &pipelines.dot_pipeline,
            &state.dot_buf,
            &state.ap_buf,
            &state.ap_buf,
            n_pairs,
        );
        let inv_norm = if w_norm_sq > 1e-30 {
            1.0 / w_norm_sq.sqrt()
        } else {
            break;
        };

        // p_buf = inv_norm * ap_buf  (via axpy: p = 0 + inv_norm * ap)
        let zeros = vec![0.0_f64; n_flat];
        gpu.upload_f64(&state.p_buf, &zeros);
        super::dynamical::gpu_axpy(
            gpu,
            &pipelines.axpy_pipeline,
            inv_norm,
            &state.ap_buf,
            &state.p_buf,
            n_flat,
        );
    }

    lambda_est.max(1.0) // never return less than 1.0
}

/// Probe the spectral range of D†D on the current gauge configuration.
///
/// - λ_max: measured via GPU power iteration
/// - λ_min: analytical bound m² (safe for positive mass staggered fermions)
///
/// Returns `SpectralInfo` with safety margins applied.
pub fn probe_spectral_range(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    mass: f64,
) -> SpectralInfo {
    let lambda_max = gpu_power_iteration_lambda_max(
        gpu,
        pipelines,
        state,
        mass,
        RHMC_POWER_ITERATION_COUNT,
    );

    // Analytical lower bound: λ_min(D†D) ≥ m² for staggered fermions
    let lambda_min = mass * mass;

    SpectralInfo::new(lambda_min, lambda_max, true)
}
