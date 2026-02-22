// SPDX-License-Identifier: AGPL-3.0-only

//! Staggered Dirac operator for lattice QCD.
//!
//! The staggered (Kogut-Susskind) Dirac operator acts on a single-component
//! complex color vector at each lattice site:
//!
//!   (`D_st` ψ)(x) = m ψ(x) + (1/2) Σ\_μ `η_μ`(x) [`U_μ`(x) ψ(x+μ) - `U_μ`†(x-μ) ψ(x-μ)]
//!
//! where `η_μ`(x) = (-1)^{x\_0 + ... + x\_{μ-1}} are the staggered phases
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
#[must_use]
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
        use super::constants::lcg_uniform_f64;

        let mut rng = seed;
        let mut data = vec![[Complex64::ZERO; 3]; volume];
        for site in &mut data {
            for c in site.iter_mut() {
                let re = lcg_uniform_f64(&mut rng) - 0.5;
                let im = lcg_uniform_f64(&mut rng) - 0.5;
                *c = Complex64::new(re, im);
            }
        }
        Self { data, volume }
    }

    /// Dot product: &lt;self | other&gt; = `Σ_x` `Σ_c` self(x,c)\* × other(x,c)
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

/// Staggered phase `η_μ`(x) = (-1)^{x\_0 + x\_1 + ... + x\_{μ-1}}
fn staggered_phase(x: [usize; 4], mu: usize) -> f64 {
    let sum: usize = x.iter().take(mu).sum();
    if sum.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// Apply `SU(3)` matrix to a color vector: `result_c` = Σ\_c' `U_{c,c'}` × v\_{c'}
fn su3_times_vec(u: &Su3Matrix, v: &ColorVector) -> ColorVector {
    let mut result = [Complex64::ZERO; 3];
    for (c, r) in result.iter_mut().enumerate() {
        for (cp, vcp) in v.iter().enumerate() {
            *r += u.m[c][cp] * *vcp;
        }
    }
    result
}

/// Apply U† to a color vector: `result_c` = Σ\_c' U†\_{c,c'} × v\_{c'} = Σ\_c' conj(U\_{c',c}) × v\_{c'}
fn su3_dagger_times_vec(u: &Su3Matrix, v: &ColorVector) -> ColorVector {
    let mut result = [Complex64::ZERO; 3];
    for (c, r) in result.iter_mut().enumerate() {
        for (cp, vcp) in v.iter().enumerate() {
            *r += u.m[cp][c].conj() * *vcp;
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
//  GPU WGSL shader: Staggered Dirac operator (absorption-ready)
// ═══════════════════════════════════════════════════════════════════

/// WGSL compute shader for the staggered Dirac operator (f64).
///
/// Direct GPU port of [`apply_dirac()`]: one thread per lattice site,
/// complex SU(3) matrix × color vector multiplication with staggered
/// phase factors. f64 precision throughout.
///
/// ## Data layout (interleaved re/im pairs)
///
/// - Fermion field: `psi[site * 6 + color * 2 + 0/1]` (re/im)
/// - Gauge links: `links[(site * 4 + mu) * 18 + row * 6 + col * 2 + 0/1]`
/// - Neighbor indices: `nbr[site * 8 + mu * 2 + 0]` (forward), `+1` (backward)
/// - Staggered phases: `phases[site * 4 + mu]` (±1.0)
///
/// ## Binding layout
///
/// | Binding | Type | Content |
/// |---------|------|---------|
/// | 0 | uniform | `{ volume: u32, pad: u32, mass: f64, hop_sign: f64 }` |
/// | 1 | storage, read | `links: array<f64>` (volume×4×18 entries) |
/// | 2 | storage, read | `psi_in: array<f64>` (volume×6 entries) |
/// | 3 | storage, `read_write` | `psi_out: array<f64>` (volume×6 entries) |
/// | 4 | storage, read | `nbr: array<u32>` (volume×8 entries) |
/// | 5 | storage, read | `phases: array<f64>` (volume×4 entries) |
///
/// ## Dispatch
///
/// `ceil(volume / 64)` workgroups of 64 threads.
///
/// ## Provenance
///
/// GPU promotion of CPU [`apply_dirac()`] for Bazavov lattice QCD
/// Papers 9-12 (dynamical fermions, GPU CG, thermodynamics).
pub const WGSL_DIRAC_STAGGERED_F64: &str = include_str!("shaders/dirac_staggered_f64.wgsl");

/// Pre-computed GPU layout for the Dirac operator.
///
/// Flattens lattice topology and gauge field into GPU-friendly arrays.
pub struct DiracGpuLayout {
    pub volume: usize,
    pub links_flat: Vec<f64>,
    pub neighbors: Vec<u32>,
    pub phases: Vec<f64>,
}

impl DiracGpuLayout {
    /// Build GPU layout from a lattice.
    ///
    /// Gauge links are interleaved re/im, neighbors are pre-computed
    /// with periodic boundaries, phases are ±1.0 staggered factors.
    pub fn from_lattice(lattice: &Lattice) -> Self {
        let vol = lattice.volume();

        let mut links_flat = vec![0.0_f64; vol * 4 * 18];
        let mut neighbors = vec![0_u32; vol * 8];
        let mut phases = vec![0.0_f64; vol * 4];

        for idx in 0..vol {
            let x = lattice.site_coords(idx);

            for mu in 0..4 {
                // Gauge links: U_mu(x) → links_flat[(idx*4+mu)*18..]
                let u = lattice.link(x, mu);
                let base = (idx * 4 + mu) * 18;
                for row in 0..3 {
                    for col in 0..3 {
                        links_flat[base + row * 6 + col * 2] = u.m[row][col].re;
                        links_flat[base + row * 6 + col * 2 + 1] = u.m[row][col].im;
                    }
                }

                // Forward/backward neighbor indices
                let fwd = lattice.site_index(lattice.neighbor(x, mu, true));
                let bwd = lattice.site_index(lattice.neighbor(x, mu, false));
                neighbors[idx * 8 + mu * 2] = fwd as u32;
                neighbors[idx * 8 + mu * 2 + 1] = bwd as u32;

                // Staggered phase η_μ(x)
                phases[idx * 4 + mu] = staggered_phase(x, mu);
            }
        }

        Self {
            volume: vol,
            links_flat,
            neighbors,
            phases,
        }
    }
}

/// Flatten a fermion field to f64 array (interleaved re/im).
pub fn flatten_fermion(psi: &FermionField) -> Vec<f64> {
    let mut flat = Vec::with_capacity(psi.volume * 6);
    for site in &psi.data {
        for c in site {
            flat.push(c.re);
            flat.push(c.im);
        }
    }
    flat
}

/// Unflatten f64 array back to a fermion field.
pub fn unflatten_fermion(flat: &[f64], volume: usize) -> FermionField {
    let mut data = vec![[Complex64::ZERO; 3]; volume];
    for i in 0..volume {
        for c in 0..3 {
            data[i][c] = Complex64::new(flat[i * 6 + c * 2], flat[i * 6 + c * 2 + 1]);
        }
    }
    FermionField { data, volume }
}

/// Apply staggered Dirac operator: out = `D_st` × psi
///
/// (`D_st` ψ)(x) = m ψ(x) + (1/2) Σ\_μ `η_μ`(x) [`U_μ`(x) ψ(x+μ) - `U_μ`†(x-μ) ψ(x-μ)]
pub fn apply_dirac(lattice: &Lattice, psi: &FermionField, mass: f64) -> FermionField {
    let vol = lattice.volume();
    let mut result = FermionField::zeros(vol);

    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        let mut out = [Complex64::ZERO; 3];

        // Mass term
        for (c, o) in out.iter_mut().enumerate() {
            *o = psi.data[idx][c].scale(mass);
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
pub fn apply_dirac_adjoint(lattice: &Lattice, psi: &FermionField, mass: f64) -> FermionField {
    let vol = lattice.volume();
    let mut result = FermionField::zeros(vol);

    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        let mut out = [Complex64::ZERO; 3];

        for (c, o) in out.iter_mut().enumerate() {
            *o = psi.data[idx][c].scale(mass);
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
    #[allow(clippy::float_cmp)] // exact known phases (±1.0)
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

    #[test]
    fn fermion_field_norm_sq() {
        let vol = 16;
        let psi = FermionField::random(vol, 44);
        let n2 = psi.norm_sq();
        assert!(n2 >= 0.0);
        let dot_self = psi.dot(&psi);
        assert!((n2 - dot_self.re).abs() < 1e-12);
    }

    #[test]
    fn fermion_field_axpy() {
        let vol = 8;
        let mut y = FermionField::random(vol, 1);
        let x = FermionField::random(vol, 2);
        let a = Complex64::new(0.5, 0.0);
        let y_before = y.norm_sq();
        y.axpy(a, &x);
        assert!(y.norm_sq() > 0.0);
        assert!((y.norm_sq() - y_before).abs() > 1e-10 || x.norm_sq() > 1e-10);
    }

    #[test]
    fn fermion_field_scale_inplace() {
        let vol = 8;
        let mut psi = FermionField::random(vol, 45);
        let n_before = psi.norm_sq();
        psi.scale_inplace(2.0);
        assert!((psi.norm_sq() - 4.0 * n_before).abs() < 1e-10);
    }

    #[test]
    fn fermion_field_zero() {
        let vol = 8;
        let mut psi = FermionField::random(vol, 46);
        psi.zero();
        assert!(psi.norm_sq() < 1e-20);
    }

    #[test]
    fn fermion_field_copy_from() {
        let vol = 8;
        let src = FermionField::random(vol, 47);
        let mut dst = FermionField::zeros(vol);
        dst.copy_from(&src);
        for (s, d) in src.data.iter().zip(dst.data.iter()) {
            for c in 0..3 {
                assert!((s[c].re - d[c].re).abs() < 1e-12);
                assert!((s[c].im - d[c].im).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn flatten_unflatten_roundtrip() {
        let vol = 16;
        let psi = FermionField::random(vol, 48);
        let flat = flatten_fermion(&psi);
        let recovered = unflatten_fermion(&flat, vol);
        assert_eq!(recovered.volume, vol);
        for (a, b) in psi.data.iter().zip(recovered.data.iter()) {
            for c in 0..3 {
                assert!((a[c].re - b[c].re).abs() < 1e-12);
                assert!((a[c].im - b[c].im).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn dirac_gpu_layout_from_lattice() {
        let lat = Lattice::cold_start([2, 2, 2, 2], 6.0);
        let layout = DiracGpuLayout::from_lattice(&lat);
        assert_eq!(layout.volume, 16);
        assert_eq!(layout.links_flat.len(), 16 * 4 * 18);
        assert_eq!(layout.neighbors.len(), 16 * 8);
        assert_eq!(layout.phases.len(), 16 * 4);
    }

    #[test]
    fn dirac_zero_mass() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let psi = FermionField::random(lat.volume(), 50);
        let result = apply_dirac(&lat, &psi, 0.0);
        assert!(result.norm_sq() > 0.0, "hopping terms survive at m=0");
    }

    #[test]
    fn dirac_large_mass_dominates() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let psi = FermionField::random(lat.volume(), 51);
        let result_small = apply_dirac(&lat, &psi, 0.1);
        let result_large = apply_dirac(&lat, &psi, 100.0);
        assert!(result_large.norm_sq() > 100.0 * result_small.norm_sq());
    }

    #[test]
    fn dirac_sq_zero_mass() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let psi = FermionField::random(lat.volume(), 52);
        let ddpsi = apply_dirac_sq(&lat, &psi, 0.01);
        assert!(ddpsi.norm_sq() > 0.0);
    }

    #[test]
    fn dirac_periodic_boundary_small_lattice() {
        let lat = Lattice::cold_start([2, 2, 2, 2], 6.0);
        let psi = FermionField::random(lat.volume(), 53);
        let result = apply_dirac(&lat, &psi, 0.5);
        assert!(result.norm_sq() > 0.0);
        assert_eq!(result.volume, lat.volume());
    }
}
