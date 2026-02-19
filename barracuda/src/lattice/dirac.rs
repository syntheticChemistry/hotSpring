// SPDX-License-Identifier: AGPL-3.0-only

//! Staggered Dirac operator for lattice QCD.
//!
//! The staggered (Kogut-Susskind) Dirac operator acts on a single-component
//! complex color vector at each lattice site:
//!
//!   (D_st ψ)(x) = m ψ(x) + (1/2) Σ_μ η_μ(x) [U_μ(x) ψ(x+μ) - U_μ†(x-μ) ψ(x-μ)]
//!
//! where η_μ(x) = (-1)^{x_0 + ... + x_{μ-1}} are the staggered phases
//! that encode the Dirac structure.
//!
//! Each site carries a 3-component complex vector (color indices only).
//!
//! # References
//!
//! - Kogut & Susskind, PRD 11, 395 (1975)
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 5

use super::complex_f64::Complex64;
use super::su3::Su3Matrix;
use super::wilson::Lattice;

/// Color vector at a single lattice site: 3 complex components.
pub type ColorVector = [Complex64; 3];

/// Staggered fermion field: one `ColorVector` per lattice site.
pub struct FermionField {
    pub data: Vec<ColorVector>,
    pub volume: usize,
}

impl FermionField {
    /// Create a zero fermion field.
    pub fn zeros(volume: usize) -> Self {
        Self {
            data: vec![[Complex64::ZERO; 3]; volume],
            volume,
        }
    }

    /// Create a random fermion field for CG testing.
    pub fn random(volume: usize, seed: u64) -> Self {
        let mut rng = seed;
        let mut data = vec![[Complex64::ZERO; 3]; volume];
        for site in &mut data {
            for c in site.iter_mut() {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let re = (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let im = (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                *c = Complex64::new(re, im);
            }
        }
        Self { data, volume }
    }

    /// Dot product: <self | other> = Σ_x Σ_c self(x,c)* × other(x,c)
    pub fn dot(&self, other: &Self) -> Complex64 {
        let mut sum = Complex64::ZERO;
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            for c in 0..3 {
                sum += a[c].conj() * b[c];
            }
        }
        sum
    }

    /// Squared norm: ||self||² = <self | self>.re
    pub fn norm_sq(&self) -> f64 {
        self.dot(self).re
    }

    /// axpy: self = a × x + self
    pub fn axpy(&mut self, a: Complex64, x: &Self) {
        for (si, xi) in self.data.iter_mut().zip(x.data.iter()) {
            for c in 0..3 {
                si[c] += a * xi[c];
            }
        }
    }

    /// Scale in place: self *= a
    pub fn scale_inplace(&mut self, a: f64) {
        for site in &mut self.data {
            for c in site.iter_mut() {
                *c = c.scale(a);
            }
        }
    }

    /// Zero all entries.
    pub fn zero(&mut self) {
        for site in &mut self.data {
            *site = [Complex64::ZERO; 3];
        }
    }

    /// Copy from another field.
    pub fn copy_from(&mut self, other: &Self) {
        self.data.copy_from_slice(&other.data);
    }
}

/// Staggered phase η_μ(x) = (-1)^{x_0 + x_1 + ... + x_{μ-1}}
fn staggered_phase(x: [usize; 4], mu: usize) -> f64 {
    let mut sum = 0;
    for i in 0..mu {
        sum += x[i];
    }
    if sum % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Apply SU(3) matrix to a color vector: result_c = Σ_c' U_{c,c'} × v_{c'}
fn su3_times_vec(u: &Su3Matrix, v: &ColorVector) -> ColorVector {
    let mut result = [Complex64::ZERO; 3];
    for c in 0..3 {
        for cp in 0..3 {
            result[c] += u.m[c][cp] * v[cp];
        }
    }
    result
}

/// Apply U† to a color vector: result_c = Σ_c' U†_{c,c'} × v_{c'} = Σ_c' conj(U_{c',c}) × v_{c'}
fn su3_dagger_times_vec(u: &Su3Matrix, v: &ColorVector) -> ColorVector {
    let mut result = [Complex64::ZERO; 3];
    for c in 0..3 {
        for cp in 0..3 {
            result[c] += u.m[cp][c].conj() * v[cp];
        }
    }
    result
}

/// Apply staggered Dirac operator: out = D_st × psi
///
/// (D_st ψ)(x) = m ψ(x) + (1/2) Σ_μ η_μ(x) [U_μ(x) ψ(x+μ) - U_μ†(x-μ) ψ(x-μ)]
pub fn apply_dirac(lattice: &Lattice, psi: &FermionField, mass: f64) -> FermionField {
    let vol = lattice.volume();
    let mut result = FermionField::zeros(vol);

    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        let mut out = [Complex64::ZERO; 3];

        // Mass term
        for c in 0..3 {
            out[c] = psi.data[idx][c].scale(mass);
        }

        // Hopping terms
        for mu in 0..4 {
            let eta = staggered_phase(x, mu);
            let half_eta = 0.5 * eta;

            // Forward: +η/2 × U_μ(x) ψ(x+μ)
            let x_fwd = lattice.neighbor(x, mu, true);
            let idx_fwd = lattice.site_index(x_fwd);
            let u_fwd = lattice.link(x, mu);
            let fwd = su3_times_vec(&u_fwd, &psi.data[idx_fwd]);

            // Backward: -η/2 × U_μ†(x-μ) ψ(x-μ)
            let x_bwd = lattice.neighbor(x, mu, false);
            let idx_bwd = lattice.site_index(x_bwd);
            let u_bwd = lattice.link(x_bwd, mu);
            let bwd = su3_dagger_times_vec(&u_bwd, &psi.data[idx_bwd]);

            for c in 0..3 {
                out[c] += (fwd[c] - bwd[c]).scale(half_eta);
            }
        }

        result.data[idx] = out;
    }

    result
}

/// Apply D†D (the squared Dirac operator, positive definite) for CG.
///
/// D†D is Hermitian positive definite when mass > 0, suitable for CG.
pub fn apply_dirac_sq(lattice: &Lattice, psi: &FermionField, mass: f64) -> FermionField {
    let dpsi = apply_dirac(lattice, psi, mass);
    apply_dirac_adjoint(lattice, &dpsi, mass)
}

/// Apply D† (adjoint of staggered Dirac operator).
///
/// For staggered fermions: D† has the same structure but with η → -η
/// for the hopping terms (equivalently, reverse forward/backward).
fn apply_dirac_adjoint(lattice: &Lattice, psi: &FermionField, mass: f64) -> FermionField {
    let vol = lattice.volume();
    let mut result = FermionField::zeros(vol);

    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        let mut out = [Complex64::ZERO; 3];

        for c in 0..3 {
            out[c] = psi.data[idx][c].scale(mass);
        }

        for mu in 0..4 {
            let eta = staggered_phase(x, mu);
            let half_eta = 0.5 * eta;

            let x_fwd = lattice.neighbor(x, mu, true);
            let idx_fwd = lattice.site_index(x_fwd);
            let u_fwd = lattice.link(x, mu);
            let fwd = su3_times_vec(&u_fwd, &psi.data[idx_fwd]);

            let x_bwd = lattice.neighbor(x, mu, false);
            let idx_bwd = lattice.site_index(x_bwd);
            let u_bwd = lattice.link(x_bwd, mu);
            let bwd = su3_dagger_times_vec(&u_bwd, &psi.data[idx_bwd]);

            // D† flips the sign of the hopping term
            for c in 0..3 {
                out[c] -= (fwd[c] - bwd[c]).scale(half_eta);
            }
        }

        result.data[idx] = out;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirac_on_zero_field_is_zero() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let psi = FermionField::zeros(lat.volume());
        let result = apply_dirac(&lat, &psi, 0.1);
        let norm = result.norm_sq();
        assert!(norm < 1e-20, "D × 0 should be 0");
    }

    #[test]
    fn dirac_mass_term() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let psi = FermionField::random(vol, 42);
        let mass = 1.0;

        // On a cold (identity) lattice, the hopping terms cancel
        // for a uniform field — but random field won't cancel.
        // Just check the mass contribution is present.
        let result = apply_dirac(&lat, &psi, mass);
        assert!(result.norm_sq() > 0.0, "D × ψ should be nonzero");
    }

    #[test]
    fn dirac_sq_positive_definite() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let psi = FermionField::random(vol, 99);

        let ddpsi = apply_dirac_sq(&lat, &psi, 0.1);
        let inner = psi.dot(&ddpsi).re;
        assert!(inner > 0.0, "<ψ|D†D|ψ> should be positive: {inner}");
    }

    #[test]
    fn staggered_phases() {
        assert_eq!(staggered_phase([0, 0, 0, 0], 0), 1.0);
        assert_eq!(staggered_phase([1, 0, 0, 0], 1), -1.0);
        assert_eq!(staggered_phase([1, 1, 0, 0], 2), 1.0);
        assert_eq!(staggered_phase([1, 1, 1, 0], 3), -1.0);
    }

    #[test]
    fn fermion_field_dot_product() {
        let vol = 16;
        let a = FermionField::random(vol, 42);
        let b = FermionField::random(vol, 43);
        let dot_ab = a.dot(&b);
        let dot_ba = b.dot(&a);
        assert!(
            (dot_ab.re - dot_ba.re).abs() < 1e-12,
            "dot product should have Re<a|b> = Re<b|a>"
        );
    }
}
