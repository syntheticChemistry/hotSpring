// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hadronic correlator and susceptibility measurements for lattice QCD.
//!
//! Provides time-slice propagator correlators, the HVP kernel for the
//! muon anomalous magnetic moment, and thermodynamic susceptibilities
//! for freeze-out curve extraction.
//!
//! # HVP correlator
//!
//! The hadronic vacuum polarization contribution to the muon g-2 is
//! computed from the current-current correlator:
//!
//!   C(t) = Σ_x,c |G_c(x,t)|²
//!
//! where G = (D†D)⁻¹ δ(0) is the point-to-all staggered propagator.
//! The HVP integral sums C(t) weighted by a known kernel.
//!
//! # Susceptibilities
//!
//! Plaquette and Polyakov loop susceptibilities locate the
//! deconfinement transition temperature T_c via their peaks.
//!
//! # References
//!
//! - Bernecker & Meyer, EPJA 47, 148 (2011) — HVP kernel
//! - Bazavov et al., PRD 111, 094508 (2025) — HVP g-2
//! - Bazavov et al., PRD 93, 014512 (2016) — freeze-out curvature

use super::cg::{CgResult, cg_solve, cg_solve_with_history};
use super::complex_f64::Complex64;
use super::dirac::FermionField;
use super::wilson::Lattice;

/// Time-slice propagator correlator result.
#[derive(Clone, Debug)]
pub struct CorrelatorResult {
    /// C(t) for t = `0..N_t`
    pub correlator: Vec<f64>,
    /// CG convergence info
    pub cg: CgResult,
}

/// Compute the point-to-all staggered propagator time-slice correlator.
///
/// Places a delta-function source at the origin (color 0), solves
/// (D†D)⁻¹ source, and sums |G|² over spatial slices.
///
/// Returns C(t) = Σ_{x spatial} `Σ_c` |`G_c(x,t)|²`
#[must_use]
pub fn point_propagator_correlator(
    lattice: &Lattice,
    mass: f64,
    cg_tol: f64,
    cg_max_iter: usize,
) -> CorrelatorResult {
    let vol = lattice.volume();
    let nt = lattice.dims[3];

    let mut source = FermionField::zeros(vol);
    source.data[0][0] = Complex64::ONE;

    let mut propagator = FermionField::zeros(vol);
    let cg = cg_solve(lattice, &mut propagator, &source, mass, cg_tol, cg_max_iter);

    let mut correlator = vec![0.0; nt];
    for idx in 0..vol {
        let coords = lattice.site_coords(idx);
        let t = coords[3];
        for c in 0..3 {
            correlator[t] += propagator.data[idx][c].abs_sq();
        }
    }

    CorrelatorResult { correlator, cg }
}

/// Same as `point_propagator_correlator` but records per-iteration CG residual history.
///
/// The `CgResult::residual_history` field will contain the relative residual
/// at each CG iteration, suitable for convergence analysis or comparison
/// with external solvers.
#[must_use]
pub fn point_propagator_correlator_with_history(
    lattice: &Lattice,
    mass: f64,
    cg_tol: f64,
    cg_max_iter: usize,
) -> CorrelatorResult {
    let vol = lattice.volume();
    let nt = lattice.dims[3];

    let mut source = FermionField::zeros(vol);
    source.data[0][0] = Complex64::ONE;

    let mut propagator = FermionField::zeros(vol);
    let cg = cg_solve_with_history(lattice, &mut propagator, &source, mass, cg_tol, cg_max_iter);

    let mut correlator = vec![0.0; nt];
    for idx in 0..vol {
        let coords = lattice.site_coords(idx);
        let t = coords[3];
        for c in 0..3 {
            correlator[t] += propagator.data[idx][c].abs_sq();
        }
    }

    CorrelatorResult { correlator, cg }
}

/// Multi-source averaged correlator for reduced variance.
///
/// Averages over `n_sources` random wall sources (different colors)
/// to reduce statistical noise.
#[must_use]
pub fn averaged_correlator(
    lattice: &Lattice,
    mass: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    n_sources: usize,
    seed: u64,
) -> CorrelatorResult {
    let vol = lattice.volume();
    let nt = lattice.dims[3];
    let mut rng = seed;
    let mut avg_corr = vec![0.0; nt];
    let mut last_cg = CgResult {
        converged: true,
        iterations: 0,
        final_residual: 0.0,
        initial_residual: 0.0,
        residual_history: Vec::new(),
    };

    for color in 0..n_sources.min(3) {
        let mut source = FermionField::zeros(vol);
        let src_site = (super::constants::lcg_uniform_f64(&mut rng) * vol as f64)
            .min(vol as f64 - 1.0) as usize;
        source.data[src_site][color] = Complex64::ONE;

        let mut propagator = FermionField::zeros(vol);
        last_cg = cg_solve(lattice, &mut propagator, &source, mass, cg_tol, cg_max_iter);

        let src_t = lattice.site_coords(src_site)[3];
        for idx in 0..vol {
            let coords = lattice.site_coords(idx);
            let dt = (coords[3] + nt - src_t) % nt;
            for c in 0..3 {
                avg_corr[dt] += propagator.data[idx][c].abs_sq();
            }
        }
    }

    let n_actual = n_sources.min(3) as f64;
    for c in &mut avg_corr {
        *c /= n_actual;
    }

    CorrelatorResult {
        correlator: avg_corr,
        cg: last_cg,
    }
}

/// Simplified HVP kernel K(t) for lattice units.
///
/// Based on the Bernecker-Meyer (2011) discretization.
/// K(t) ∝ t²(T/2 - t)² / T⁴ for 0 < t < T/2, else 0.
///
/// The proportionality constant absorbs `α²_em` and normalization
/// factors that are irrelevant for the relative shape validation.
#[must_use]
pub fn hvp_kernel(t: usize, nt: usize) -> f64 {
    let half = nt / 2;
    if t == 0 || t >= half {
        return 0.0;
    }
    let tf = t as f64;
    let ht = half as f64;
    let x = tf / ht;
    x * x * (1.0 - x) * (1.0 - x)
}

/// Compute the HVP integral from a time-slice correlator.
///
/// `a_μ^HVP` ∝ `Σ_t` K(t) C(t)
///
/// Returns the lattice-units HVP integral (positive if C(t) > 0).
#[must_use]
pub fn hvp_integral(correlator: &[f64]) -> f64 {
    let nt = correlator.len();
    let mut sum = 0.0;
    for (t, &c_t) in correlator.iter().enumerate().skip(1).take(nt / 2 - 1) {
        sum += hvp_kernel(t, nt) * c_t;
    }
    sum
}

/// Plaquette susceptibility `χ_P` = V × (⟨P²⟩ - ⟨P⟩²).
///
/// The susceptibility peaks at the deconfinement transition.
#[must_use]
pub fn plaquette_susceptibility(plaquettes: &[f64], volume: usize) -> f64 {
    let n = plaquettes.len() as f64;
    let mean = plaquettes.iter().sum::<f64>() / n;
    let mean_sq = plaquettes.iter().map(|p| p * p).sum::<f64>() / n;
    volume as f64 * mean.mul_add(-mean, mean_sq)
}

/// Polyakov loop susceptibility `χ_L` = `V_s` × (⟨|L|²⟩ - ⟨|L|⟩²).
///
/// Also peaks at the deconfinement transition.
#[must_use]
pub fn polyakov_susceptibility(poly_abs: &[f64], spatial_vol: usize) -> f64 {
    let n = poly_abs.len() as f64;
    let mean_abs = poly_abs.iter().sum::<f64>() / n;
    let mean_sq = poly_abs.iter().map(|p| p * p).sum::<f64>() / n;
    spatial_vol as f64 * mean_abs.mul_add(-mean_abs, mean_sq)
}

/// Stochastic estimate of the chiral condensate ⟨ψ̄ψ⟩.
#[derive(Clone, Debug)]
pub struct ChiralCondensateResult {
    /// Estimated condensate value
    pub condensate: f64,
    /// Stochastic error estimate (std. dev. of per-source estimates / sqrt(N))
    pub error: f64,
    /// Number of stochastic sources actually used
    pub n_sources: usize,
    /// Average CG iterations across all source inversions
    pub avg_cg_iters: f64,
}

/// Stochastic estimator for the chiral condensate ⟨ψ̄ψ⟩ = (1/V) Tr[D⁻¹].
///
/// Uses `n_sources` random Z₂ noise vectors and CG inversion to estimate
/// ⟨ψ̄ψ⟩ = (1/V) (1/N_src) Σ_i ξ_i† D⁻¹ ξ_i.
///
/// # Arguments
/// * `lattice` — gauge configuration
/// * `mass` — bare quark mass
/// * `cg_tol` — CG stopping tolerance
/// * `cg_max_iter` — maximum CG iterations
/// * `n_sources` — number of stochastic noise vectors
/// * `seed` — RNG seed
#[must_use]
pub fn chiral_condensate_stochastic(
    lattice: &Lattice,
    mass: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    n_sources: usize,
    seed: u64,
) -> ChiralCondensateResult {
    let vol = lattice.volume();
    let mut rng = seed;
    let mut estimates = Vec::with_capacity(n_sources);
    let mut total_cg_iters = 0u64;

    for _src in 0..n_sources {
        let mut source = FermionField::zeros(vol);
        for idx in 0..vol {
            for c in 0..3 {
                let r = super::constants::lcg_uniform_f64(&mut rng);
                let sign = if r < 0.5 { 1.0 } else { -1.0 };
                source.data[idx][c] = Complex64 { re: sign, im: 0.0 };
            }
        }

        let mut solution = FermionField::zeros(vol);
        let cg_result = cg_solve(lattice, &mut solution, &source, mass, cg_tol, cg_max_iter);
        total_cg_iters += cg_result.iterations as u64;

        let mut trace_est = 0.0;
        for idx in 0..vol {
            for c in 0..3 {
                trace_est += source.data[idx][c].re * solution.data[idx][c].re
                    + source.data[idx][c].im * solution.data[idx][c].im;
            }
        }
        estimates.push(trace_est / vol as f64);
    }

    let n = estimates.len() as f64;
    let mean = estimates.iter().sum::<f64>() / n;
    let variance = if estimates.len() > 1 {
        estimates.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let error = (variance / n).sqrt();

    ChiralCondensateResult {
        condensate: mean,
        error,
        n_sources: estimates.len(),
        avg_cg_iters: if estimates.is_empty() {
            0.0
        } else {
            total_cg_iters as f64 / estimates.len() as f64
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hvp_kernel_shape() {
        let nt = 16;
        let k: Vec<f64> = (0..nt).map(|t| hvp_kernel(t, nt)).collect();
        assert!((k[0]).abs() < 1e-15, "K(0) should be 0");
        assert!(k[1] > 0.0, "K(1) should be positive");
        assert!(k[nt / 2 - 1] > 0.0, "K(T/2-1) should be positive");
        for val in &k[nt / 2..nt] {
            assert!(val.abs() < 1e-15, "K(t >= T/2) should be 0");
        }
    }

    #[test]
    fn hvp_kernel_symmetric_peak() {
        let nt = 32;
        let half = nt / 2;
        let peak_t = half / 2;
        let k_peak = hvp_kernel(peak_t, nt);
        assert!(k_peak > hvp_kernel(1, nt), "kernel should peak near T/4");
        assert!(
            k_peak > hvp_kernel(half - 1, nt),
            "kernel should peak near T/4"
        );
    }

    #[test]
    fn hvp_kernel_different_lattice_sizes() {
        for nt in [8, 16, 32, 64] {
            let k: Vec<f64> = (0..nt).map(|t| hvp_kernel(t, nt)).collect();
            assert!(k[0].abs() < 1e-15);
            let sum: f64 = k.iter().sum();
            assert!(sum > 0.0, "kernel sum should be positive for nt={nt}");
        }
    }

    #[test]
    fn hvp_integral_positive_for_positive_correlator() {
        let nt = 16;
        let corr = vec![1.0; nt];
        let integral = hvp_integral(&corr);
        assert!(integral > 0.0, "HVP integral should be positive");
    }

    #[test]
    fn hvp_integral_zero_for_empty_correlator() {
        let corr = vec![0.0; 16];
        let integral = hvp_integral(&corr);
        assert!(integral.abs() < 1e-15, "zero correlator → zero integral");
    }

    #[test]
    fn hvp_integral_proportional_to_amplitude() {
        let nt = 16;
        let corr_1 = vec![1.0; nt];
        let corr_2: Vec<f64> = corr_1.iter().map(|&c| 2.0 * c).collect();
        let i1 = hvp_integral(&corr_1);
        let i2 = hvp_integral(&corr_2);
        assert!(
            (i2 - 2.0 * i1).abs() < 1e-14,
            "integral should scale linearly"
        );
    }

    #[test]
    fn susceptibility_zero_for_constant() {
        let plaq = vec![0.5; 100];
        let chi = plaquette_susceptibility(&plaq, 256);
        assert!(
            chi.abs() < 1e-10,
            "susceptibility of constant should be ~0, got {chi}"
        );
    }

    #[test]
    fn susceptibility_positive_for_fluctuations() {
        let plaq: Vec<f64> = (0..100).map(|i| 0.5 + 0.01 * (i as f64).sin()).collect();
        let chi = plaquette_susceptibility(&plaq, 256);
        assert!(
            chi > 0.0,
            "susceptibility should be positive for varying data"
        );
    }

    #[test]
    fn susceptibility_scales_with_volume() {
        let plaq: Vec<f64> = (0..100).map(|i| 0.5 + 0.01 * (i as f64).sin()).collect();
        let chi_small = plaquette_susceptibility(&plaq, 128);
        let chi_large = plaquette_susceptibility(&plaq, 256);
        assert!(
            (chi_large - 2.0 * chi_small).abs() / chi_large < 1e-10,
            "susceptibility should scale linearly with volume"
        );
    }

    #[test]
    fn polyakov_susceptibility_zero_for_constant() {
        let poly = vec![0.3; 50];
        let chi = polyakov_susceptibility(&poly, 64);
        assert!(chi.abs() < 1e-10, "constant Polyakov → zero susceptibility");
    }

    #[test]
    fn polyakov_susceptibility_positive_for_fluctuations() {
        let poly: Vec<f64> = (0..50)
            .map(|i| 0.3 + 0.05 * (i as f64 * 0.5).cos())
            .collect();
        let chi = polyakov_susceptibility(&poly, 64);
        assert!(chi > 0.0, "varying Polyakov → positive susceptibility");
    }

    #[test]
    fn averaged_correlator_on_cold_lattice() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let result = averaged_correlator(&lat, 0.5, 1e-6, 500, 3, 42);
        assert!(result.cg.converged, "CG should converge");
        let c0 = result.correlator[0];
        assert!(c0 > 0.0, "C(0) should be positive: {c0}");
    }

    #[test]
    fn correlator_on_cold_lattice() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let result = point_propagator_correlator(&lat, 0.5, 1e-6, 500);
        assert!(result.cg.converged, "CG should converge on cold lattice");
        assert!(
            result.correlator[0] > 0.0,
            "C(0) should be positive: {}",
            result.correlator[0]
        );
    }

    #[test]
    fn correlator_length_matches_nt() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let result = point_propagator_correlator(&lat, 0.5, 1e-6, 500);
        assert_eq!(
            result.correlator.len(),
            4,
            "correlator should have N_t entries"
        );
    }

    #[test]
    fn correlator_all_positive_on_cold() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let result = point_propagator_correlator(&lat, 0.5, 1e-6, 500);
        for (t, &c) in result.correlator.iter().enumerate() {
            assert!(c >= 0.0, "C({t}) = {c} should be non-negative");
        }
    }
}
