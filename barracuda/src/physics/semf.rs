//! Semi-Empirical Mass Formula (Bethe–Weizsäcker) with Skyrme-derived coefficients.
//!
//! The SEMF gives nuclear binding energy as a sum of volume, surface, Coulomb,
//! asymmetry, and pairing terms. Unlike textbook SEMF, coefficients are derived
//! from Skyrme nuclear matter properties — not fitted independently.
//!
//! Reference: von Weizsäcker, Z. Phys. 96, 431 (1935);
//!            Bethe & Bacher, Rev. Mod. Phys. 8, 82 (1936).
//! Port of: `control/surrogate/nuclear-eos/wrapper/skyrme_hf.py::semf_binding_energy`
//! See PHYSICS.md §4 for complete equation documentation.

use super::constants::*;
use super::nuclear_matter::nuclear_matter_properties;
use std::f64::consts::PI;

/// Binding energy via Bethe–Weizsäcker SEMF with Skyrme-derived coefficients.
///
/// B(Z,N) = a_v·A − a_s·A^(2/3) − a_c·Z(Z−1)/A^(1/3) − a_a·(N−Z)²/A + δ(Z,N)
///
/// Coefficients a_v, a_c, a_a are derived from infinite nuclear matter properties
/// (saturation density, symmetry energy) computed from the Skyrme EDF parameters.
pub fn semf_binding_energy(z: usize, n: usize, params: &[f64]) -> f64 {
    let a = z + n;
    if a == 0 {
        return 0.0;
    }
    let af = a as f64;
    let zf = z as f64;
    let nf = n as f64;

    // Derive SEMF coefficients from nuclear matter properties (PHYSICS.md §4.2)
    let nmp = match nuclear_matter_properties(params) {
        Some(nmp) => nmp,
        None => return 0.0,
    };

    // Volume: a_v = |E/A(ρ₀)| — binding energy per nucleon at saturation
    let a_v = nmp.e_a_mev.abs();
    // Nuclear radius parameter r₀ from saturation density: r₀ = (3/(4πρ₀))^(1/3)
    let r0 = (3.0 / (4.0 * PI * nmp.rho0_fm3)).powf(1.0 / 3.0);
    // Surface: a_s = 1.1 × a_v — empirical surface-to-volume ratio
    // (Thomas–Fermi calculations give 1.0–1.2 depending on parametrization)
    let a_s = a_v * 1.1;
    // Coulomb: a_c = 3e²/(5r₀) — uniform charge sphere
    let a_c = 3.0 * E2 / (5.0 * r0);
    // Asymmetry: a_a = J — nuclear symmetry energy (PHYSICS.md §3.5)
    let a_a = nmp.j_mev;
    // Pairing amplitude: Δ = 12/√A MeV — Ring & Schuck (2004), §6.2
    let a_p = 12.0 / af.max(1.0).sqrt();

    // Bethe–Weizsäcker mass formula
    let mut b = a_v * af;                              // Volume
    b -= a_s * af.powf(2.0 / 3.0);                    // Surface
    b -= a_c * zf * (zf - 1.0) / af.powf(1.0 / 3.0); // Coulomb
    b -= a_a * (nf - zf).powi(2) / af;                // Asymmetry

    // Pairing: δ = +a_p (even-even), 0 (odd-A), -a_p (odd-odd)
    if z % 2 == 0 && n % 2 == 0 {
        b += a_p;
    } else if z % 2 == 1 && n % 2 == 1 {
        b -= a_p;
    }

    b.max(0.0)
}

