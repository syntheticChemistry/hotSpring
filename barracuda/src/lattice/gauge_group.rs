// SPDX-License-Identifier: AGPL-3.0-or-later

//! Gauge group abstraction for SU(N) lattice gauge theory.
//!
//! Provides a trait that unifies SU(2), SU(3), and SU(N) operations,
//! enabling generic lattice, HMC, and observable code across gauge groups.
//!
//! The existing `Su3Matrix` + `Lattice` code remains the validated production
//! path. This trait enables the generalized thermalization grid and
//! measurement battery for the SU(N) memo table.
//!
//! # References
//!
//! - 't Hooft, Nucl. Phys. B 72, 461 (1974) — large-N expansion
//! - Cabibbo & Marinari, PLB 119, 387 (1982) — SU(N) via SU(2) subgroups

use super::complex_f64::Complex64;

/// Core operations for an SU(N) gauge group link variable.
///
/// Each implementor provides the matrix algebra needed for Wilson action,
/// HMC integration, and observable measurement. The trait is object-safe
/// only for `Copy` types (SU(2), SU(3)); SU(N>3) uses heap-allocated
/// matrices and a separate dispatch path.
pub trait GaugeGroup: Clone + Send + Sync + 'static {
    /// Number of colors (N in SU(N)): 2 for SU(2), 3 for SU(3), etc.
    const NC: usize;

    /// Number of real degrees of freedom per link: 2*N*N.
    const LINK_REALS: usize;

    /// Number of Lie algebra generators: N*N - 1.
    const N_GENERATORS: usize;

    /// String identifier for cache directory partitioning.
    fn gauge_group_tag() -> &'static str;

    /// β = 2N/g² convention factor (the N in β = 2N/g²).
    fn beta_prefactor(&self) -> f64 {
        2.0 * self.runtime_nc() as f64
    }

    /// Identity element (unit matrix).
    fn identity() -> Self;

    /// Zero element (all entries zero).
    fn zero() -> Self;

    /// Matrix multiplication: self * rhs.
    fn mul(&self, rhs: &Self) -> Self;

    /// Conjugate transpose (Hermitian adjoint).
    fn adjoint(&self) -> Self;

    /// Real part of trace.
    fn re_trace(&self) -> f64;

    /// Normalized trace: Re Tr / N_c.
    fn re_trace_normalized(&self) -> f64 {
        self.re_trace() / self.runtime_nc() as f64
    }

    /// Complex trace.
    fn trace(&self) -> Complex64;

    /// Determinant.
    fn det(&self) -> Complex64;

    /// Scale by a real number.
    fn scale(&self, s: f64) -> Self;

    /// Scale by a complex number.
    fn scale_complex(&self, s: Complex64) -> Self;

    /// Frobenius norm squared: Σ|m_ij|².
    fn norm_sq(&self) -> f64;

    /// Project back onto SU(N) manifold after numerical drift.
    fn reunitarize(&self) -> Self;

    /// Addition.
    fn add(&self, rhs: &Self) -> Self;

    /// Subtraction.
    fn sub(&self, rhs: &Self) -> Self;

    /// Random SU(N) matrix near identity (for hot starts).
    fn random_near_identity(seed: &mut u64, epsilon: f64) -> Self;

    /// Random su(N) Lie algebra element (traceless anti-Hermitian).
    /// Used for HMC momentum initialization.
    fn random_algebra(seed: &mut u64) -> Self;

    /// Random algebra element with explicit nc — for SU(N) with runtime-determined nc.
    fn random_algebra_for_nc(nc: usize, seed: &mut u64) -> Self {
        let _ = nc;
        Self::random_algebra(seed)
    }

    /// Zero matrix with explicit nc — for SU(N) with runtime-determined nc.
    fn zero_for_nc(nc: usize) -> Self {
        let _ = nc;
        Self::zero()
    }

    /// Traceless anti-Hermitian projection: X → (X - X†)/2 - Tr(...)/N.
    /// The standard projection used in the gauge force.
    fn proj_ta(&self) -> Self {
        let nc = self.runtime_nc() as f64;
        let xt = self.adjoint();
        let diff = self.sub(&xt).scale(0.5);
        let tr = diff.trace();
        let tr_over_n = tr.scale(1.0 / nc);
        let mut proj = diff;
        proj.sub_diagonal(tr_over_n);
        proj
    }

    /// Subtract a complex scalar from all diagonal elements.
    fn sub_diagonal(&mut self, val: Complex64);

    /// Matrix inverse (for Cayley exponential map).
    fn inverse(&self) -> Self;

    /// Cayley exponential: exp(dt*P) ≈ (I + dt*P/2)(I - dt*P/2)⁻¹, then reunitarize.
    fn exp_cayley(&self, dt: f64) -> Self {
        let half = self.scale(dt * 0.5);
        let plus = Self::identity().add(&half);
        let minus = Self::identity().sub(&half);
        let inv = minus.inverse();
        plus.mul(&inv).reunitarize()
    }

    /// Serialize link data to a flat f64 buffer (for save/load).
    fn write_to_buf(&self, buf: &mut Vec<u8>);

    /// Deserialize link data from a flat f64 buffer.
    fn read_from_buf(data: &[u8], offset: usize) -> Self;

    /// Number of bytes per link in serialized form.
    fn bytes_per_link() -> usize {
        Self::LINK_REALS * 8
    }

    /// Runtime NC — matches Self::NC for fixed-size types,
    /// returns the actual nc for heap-allocated SU(N).
    fn runtime_nc(&self) -> usize {
        Self::NC
    }

    /// Runtime bytes per link — for heap-allocated SU(N) with runtime nc.
    fn runtime_bytes_per_link(&self) -> usize {
        Self::bytes_per_link()
    }
}
