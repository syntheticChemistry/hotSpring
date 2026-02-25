// SPDX-License-Identifier: AGPL-3.0-only

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

use super::cg::{cg_solve, CgResult};
use super::complex_f64::Complex64;
use super::dirac::FermionField;
use super::wilson::Lattice;

/// Time-slice propagator correlator result.
#[derive(Clone, Debug)]
pub struct CorrelatorResult {
    /// C(t) for t = 0..N_t
    pub correlator: Vec<f64>,
    /// CG convergence info
    pub cg: CgResult,
}

/// Compute the point-to-all staggered propagator time-slice correlator.
///
/// Places a delta-function source at the origin (color 0), solves
/// (D†D)⁻¹ source, and sums |G|² over spatial slices.
///
/// Returns C(t) = Σ_{x spatial} Σ_c |G_c(x,t)|²
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

/// Multi-source averaged correlator for reduced variance.
///
/// Averages over `n_sources` random wall sources (different colors)
/// to reduce statistical noise.
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
/// The proportionality constant absorbs α²_em and normalization
/// factors that are irrelevant for the relative shape validation.
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
/// a_μ^HVP ∝ Σ_t K(t) C(t)
///
/// Returns the lattice-units HVP integral (positive if C(t) > 0).
pub fn hvp_integral(correlator: &[f64]) -> f64 {
    let nt = correlator.len();
    let mut sum = 0.0;
    for (t, &c_t) in correlator.iter().enumerate().skip(1).take(nt / 2 - 1) {
        sum += hvp_kernel(t, nt) * c_t;
    }
    sum
}

/// Plaquette susceptibility χ_P = V × (⟨P²⟩ - ⟨P⟩²).
///
/// The susceptibility peaks at the deconfinement transition.
pub fn plaquette_susceptibility(plaquettes: &[f64], volume: usize) -> f64 {
    let n = plaquettes.len() as f64;
    let mean = plaquettes.iter().sum::<f64>() / n;
    let mean_sq = plaquettes.iter().map(|p| p * p).sum::<f64>() / n;
    volume as f64 * (mean_sq - mean * mean)
}

/// Polyakov loop susceptibility χ_L = V_s × (⟨|L|²⟩ - ⟨|L|⟩²).
///
/// Also peaks at the deconfinement transition.
pub fn polyakov_susceptibility(poly_abs: &[f64], spatial_vol: usize) -> f64 {
    let n = poly_abs.len() as f64;
    let mean_abs = poly_abs.iter().sum::<f64>() / n;
    let mean_sq = poly_abs.iter().map(|p| p * p).sum::<f64>() / n;
    spatial_vol as f64 * (mean_sq - mean_abs * mean_abs)
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
        for t in nt / 2..nt {
            assert!((k[t]).abs() < 1e-15, "K(t >= T/2) should be 0");
        }
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
}
