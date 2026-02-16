// SPDX-License-Identifier: AGPL-3.0-only

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
    let mut b = a_v * af; // Volume
    b -= a_s * af.powf(2.0 / 3.0); // Surface
    b -= a_c * zf * (zf - 1.0) / af.powf(1.0 / 3.0); // Coulomb
    b -= a_a * (nf - zf).powi(2) / af; // Asymmetry

    // Pairing: δ = +a_p (even-even), 0 (odd-A), -a_p (odd-odd)
    if z.is_multiple_of(2) && n.is_multiple_of(2) {
        b += a_p;
    } else if z % 2 == 1 && n % 2 == 1 {
        b -= a_p;
    }

    b.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::SLY4_PARAMS;

    #[test]
    fn semf_zero_mass_returns_zero() {
        assert_eq!(semf_binding_energy(0, 0, &SLY4_PARAMS), 0.0);
    }

    #[test]
    fn semf_iron56_reasonable() {
        // Fe-56: most tightly bound nucleus per nucleon. B/A ≈ 8.8 MeV.
        let b = semf_binding_energy(26, 30, &SLY4_PARAMS);
        let b_per_a = b / 56.0;
        assert!(
            b_per_a > 5.0 && b_per_a < 12.0,
            "Fe-56 B/A should be ~8.8 MeV, got {b_per_a}"
        );
    }

    #[test]
    fn semf_lead208_order_of_magnitude() {
        // Pb-208: B_exp ≈ 1636 MeV
        let b = semf_binding_energy(82, 126, &SLY4_PARAMS);
        assert!(
            b > 1400.0 && b < 1800.0,
            "Pb-208 SEMF should be ~1636 MeV, got {b}"
        );
    }

    #[test]
    fn semf_pairing_term_sign() {
        // Even-even gets +δ, odd-odd gets -δ. Test with heavy nucleus
        // where pairing is small relative to other terms.
        // For Pb-208 (Z=82, N=126, both even): should be higher than
        // the same A with odd-odd (Z=81, N=127).
        let b_ee = semf_binding_energy(82, 126, &SLY4_PARAMS); // Pb-208 even-even
        let b_oo = semf_binding_energy(81, 127, &SLY4_PARAMS); // Tl-208 odd-odd
        assert!(
            b_ee > b_oo,
            "heavy even-even should have higher B than odd-odd: {b_ee} vs {b_oo}"
        );
    }

    #[test]
    fn semf_monotone_in_mass_number() {
        // B should generally increase with A (up to iron peak)
        let b_he4 = semf_binding_energy(2, 2, &SLY4_PARAMS);
        let b_o16 = semf_binding_energy(8, 8, &SLY4_PARAMS);
        let b_ca40 = semf_binding_energy(20, 20, &SLY4_PARAMS);
        assert!(b_o16 > b_he4, "O-16 > He-4");
        assert!(b_ca40 > b_o16, "Ca-40 > O-16");
    }

    #[test]
    fn semf_returns_positive() {
        // SEMF should return positive binding energy for stable nuclei
        let b = semf_binding_energy(50, 82, &SLY4_PARAMS); // Sn-132
        assert!(b > 0.0, "Sn-132 binding energy should be positive");
    }

    #[test]
    fn semf_determinism() {
        // Pure CPU computation must be bitwise reproducible across calls.
        let nuclei = [(26, 30), (82, 126), (50, 82), (8, 8), (2, 2), (92, 146)];
        let run = || -> Vec<f64> {
            nuclei
                .iter()
                .map(|&(z, n)| semf_binding_energy(z, n, &SLY4_PARAMS))
                .collect()
        };
        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len(), "determinism: vector length mismatch");
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                va.to_bits() == vb.to_bits(),
                "determinism: bitwise mismatch at nucleus {:?}: {va} vs {vb}",
                nuclei[i]
            );
        }
    }
}
