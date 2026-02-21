// SPDX-License-Identifier: AGPL-3.0-only

//! Nuclear matter properties from Skyrme parameters (analytic).
//!
//! Computes infinite symmetric nuclear matter (SNM) properties: saturation
//! density ρ₀, energy per nucleon E/A, incompressibility K∞, effective mass
//! m*/m, and symmetry energy J. These are used as SEMF coefficients (Level 1)
//! and physical constraint filters.
//!
//! Reference: Bender, Heenen, Reinhard, Rev. Mod. Phys. 75, 121 (2003), §III.A.
//! Port of: `control/surrogate/nuclear-eos/wrapper/skyrme_hf.py`
//! Uses: `barracuda::optimize::bisect` for saturation density.
//! See PHYSICS.md §3 for complete equation documentation.

use super::constants::{HBAR2_2M, HBAR_C, M_NUCLEON};
use barracuda::optimize::bisect;
use std::f64::consts::PI;

/// Nuclear matter property results.
///
/// All quantities evaluated at the saturation density ρ₀ where dE/dρ = 0.
#[derive(Debug, Clone)]
pub struct NuclearMatterProps {
    pub rho0_fm3: f64,    // Saturation density (fm⁻³). Empirical: ~0.16
    pub e_a_mev: f64,     // Energy per nucleon at saturation (MeV). Empirical: ~-16
    pub k_inf_mev: f64,   // Incompressibility (MeV). Empirical: ~230
    pub m_eff_ratio: f64, // Effective mass ratio m*/m. Empirical: ~0.7
    pub j_mev: f64,       // Symmetry energy (MeV). Empirical: ~32
}

/// Energy per nucleon in symmetric nuclear matter.
///
/// E/A(ρ) = E_kin + E_t0 + E_t3 + E_t1t2
///
/// See PHYSICS.md §3.1 for term-by-term derivation.
fn energy_per_nucleon_snm(rho: f64, p: &[f64; 10]) -> f64 {
    if rho <= 0.0 {
        return 0.0;
    }

    let (t0, t1, t2, t3) = (p[0], p[1], p[2], p[3]);
    let (_x0, _x1, x2, _x3) = (p[4], p[5], p[6], p[7]);
    let alpha = p[8];

    // Fermi momentum: k_F = (3π²ρ/2)^(1/3)
    let kf = (3.0 * PI * PI * rho / 2.0).powf(1.0 / 3.0);
    // Kinetic density: τ = (3/5) k_F² ρ  [fm⁻⁵]
    let tau = (3.0 / 5.0) * kf * kf * rho;

    // Free Fermi gas kinetic energy: (ℏ²/2m)(3/5)k_F²
    let e_kin = HBAR2_2M * (3.0 / 5.0) * kf * kf;
    // t₀ contact (s-wave): (3/8)t₀ρ
    let e_t0 = (3.0 / 8.0) * t0 * rho;
    // t₃ density-dependent: (1/16)t₃ρ^(α+1)
    let e_t3 = (1.0 / 16.0) * t3 * rho.powf(alpha + 1.0);

    // t₁t₂ momentum-dependent: (1/16)Θ·τ where Θ = 3t₁ + t₂(5+4x₂)
    let theta = 3.0 * t1 + t2 * (5.0 + 4.0 * x2);
    let e_t1t2 = (1.0 / 16.0) * theta * tau;

    e_kin + e_t0 + e_t3 + e_t1t2
}

/// Compute nuclear matter properties from Skyrme parameters.
///
/// Uses `barracuda::optimize::bisect` for saturation density (replacing scipy.optimize.brentq).
pub fn nuclear_matter_properties(params: &[f64]) -> Option<NuclearMatterProps> {
    if params.len() != 10 {
        return None;
    }

    let p: [f64; 10] = params.try_into().ok()?;
    let (t0, t1, t2, t3) = (p[0], p[1], p[2], p[3]);
    let (x0, x1, x2, x3) = (p[4], p[5], p[6], p[7]);
    let alpha = p[8];

    // Saturation density: solve d(E/A)/dρ = 0 via bisection (PHYSICS.md §3.2)
    // Search range [0.05, 0.30] fm⁻³ brackets all known Skyrme parametrizations.
    // Fallback 0.16 fm⁻³ is the empirical value.
    let de_drho = |rho: f64| {
        let dr = (rho * 1e-6_f64).max(1e-10);
        (energy_per_nucleon_snm(rho + dr, &p) - energy_per_nucleon_snm(rho - dr, &p)) / (2.0 * dr)
    };

    let rho0 = bisect(de_drho, 0.05, 0.30, 1e-10, 100).unwrap_or(0.16);

    let e_a = energy_per_nucleon_snm(rho0, &p);

    // Incompressibility: K∞ = 9ρ₀² d²(E/A)/dρ² (PHYSICS.md §3.3)
    // Empirical: ~230 ± 20 MeV — Blaizot, Phys. Rep. 64, 171 (1980)
    let dr = rho0 * 1e-4;
    let d2e = (energy_per_nucleon_snm(rho0 + dr, &p) - 2.0 * energy_per_nucleon_snm(rho0, &p)
        + energy_per_nucleon_snm(rho0 - dr, &p))
        / (dr * dr);
    let k_inf = 9.0 * rho0 * rho0 * d2e;

    // Effective mass: m*/m = 1/(1 + (m_N/4ℏ²c²)Θρ₀) (PHYSICS.md §3.4)
    let theta = 3.0 * t1 + t2 * (5.0 + 4.0 * x2);
    let m_eff = 1.0 / (1.0 + (M_NUCLEON / (4.0 * HBAR_C * HBAR_C)) * theta * rho0);

    // Symmetry energy: J = J_kin + J_t0 + J_t3 + J_t1t2 (PHYSICS.md §3.5)
    // Empirical: ~32 ± 2 MeV — Lattimer & Prakash, Phys. Rep. 621, 127 (2016)
    let kf0 = (3.0 * PI * PI * rho0 / 2.0).powf(1.0 / 3.0);
    let j_kin = HBAR2_2M * kf0 * kf0 / (3.0 * m_eff);
    let j_t0 = -(t0 / 4.0) * (2.0 * x0 + 1.0) * rho0;
    let j_t3 = -(t3 / 24.0) * (2.0 * x3 + 1.0) * rho0.powf(alpha + 1.0);
    let theta_s = t2 * (4.0 + 5.0 * x2) - 3.0 * t1 * x1;
    let tau0 = (3.0 / 5.0) * kf0 * kf0 * rho0;
    let j_t1t2 = -(1.0 / 24.0) * theta_s * tau0;
    let j = j_kin + j_t0 + j_t3 + j_t1t2;

    Some(NuclearMatterProps {
        rho0_fm3: rho0,
        e_a_mev: e_a,
        k_inf_mev: k_inf,
        m_eff_ratio: m_eff,
        j_mev: j,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::SLY4_PARAMS;

    #[test]
    fn sly4_saturation_density() {
        let nmp = nuclear_matter_properties(&SLY4_PARAMS).expect("SLY4 NMP");
        // SLy4: ρ₀ ≈ 0.1595 fm⁻³ (Chabanat 1998, Table II)
        assert!(
            (nmp.rho0_fm3 - 0.16).abs() < 0.01,
            "ρ₀ should be ~0.16, got {}",
            nmp.rho0_fm3
        );
    }

    #[test]
    fn sly4_binding_energy() {
        let nmp = nuclear_matter_properties(&SLY4_PARAMS).expect("SLY4 NMP");
        // SLy4: E/A ≈ -15.97 MeV
        assert!(
            (nmp.e_a_mev - (-15.97)).abs() < 1.0,
            "E/A should be ~-16, got {}",
            nmp.e_a_mev
        );
    }

    #[test]
    fn sly4_incompressibility() {
        let nmp = nuclear_matter_properties(&SLY4_PARAMS).expect("SLY4 NMP");
        // SLy4: K∞ ≈ 230 MeV
        assert!(
            (nmp.k_inf_mev - 230.0).abs() < 30.0,
            "K∞ should be ~230, got {}",
            nmp.k_inf_mev
        );
    }

    #[test]
    fn sly4_effective_mass() {
        let nmp = nuclear_matter_properties(&SLY4_PARAMS).expect("SLY4 NMP");
        // SLy4: m*/m typically 0.5–0.7 depending on parametrization details.
        // The exact value depends on the Θ combination of t₁, t₂, x₂.
        assert!(
            nmp.m_eff_ratio > 0.3 && nmp.m_eff_ratio < 1.0,
            "m*/m should be in (0.3, 1.0), got {}",
            nmp.m_eff_ratio
        );
    }

    #[test]
    fn sly4_symmetry_energy() {
        let nmp = nuclear_matter_properties(&SLY4_PARAMS).expect("SLY4 NMP");
        // SLy4: J ≈ 32 MeV
        assert!(
            (nmp.j_mev - 32.0).abs() < 3.0,
            "J should be ~32, got {}",
            nmp.j_mev
        );
    }

    #[test]
    fn wrong_param_count_returns_none() {
        assert!(nuclear_matter_properties(&[0.0; 5]).is_none());
        assert!(nuclear_matter_properties(&[]).is_none());
    }

    #[test]
    fn energy_at_saturation_is_minimum() {
        let nmp = nuclear_matter_properties(&SLY4_PARAMS).expect("SLY4 NMP");
        let p: [f64; 10] = SLY4_PARAMS;
        let e_at_rho0 = energy_per_nucleon_snm(nmp.rho0_fm3, &p);
        // E/A should be lower at ρ₀ than at neighboring densities
        let e_below = energy_per_nucleon_snm(nmp.rho0_fm3 * 0.9, &p);
        let e_above = energy_per_nucleon_snm(nmp.rho0_fm3 * 1.1, &p);
        assert!(
            e_at_rho0 < e_below && e_at_rho0 < e_above,
            "E/A at ρ₀ should be a minimum: {e_at_rho0} vs {e_below}, {e_above}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known value (0.0)
    fn zero_density_gives_zero() {
        let p: [f64; 10] = SLY4_PARAMS;
        assert_eq!(energy_per_nucleon_snm(0.0, &p), 0.0);
    }

    #[test]
    fn nuclear_matter_determinism() {
        // Saturation-density bisection and all derived NMP must be bitwise
        // identical across calls on identical input.
        let run = || nuclear_matter_properties(&SLY4_PARAMS).expect("SLY4 NMP");
        let a = run();
        let b = run();
        assert_eq!(
            a.rho0_fm3.to_bits(),
            b.rho0_fm3.to_bits(),
            "ρ₀ bitwise mismatch"
        );
        assert_eq!(
            a.e_a_mev.to_bits(),
            b.e_a_mev.to_bits(),
            "E/A bitwise mismatch"
        );
        assert_eq!(
            a.k_inf_mev.to_bits(),
            b.k_inf_mev.to_bits(),
            "K∞ bitwise mismatch"
        );
        assert_eq!(
            a.m_eff_ratio.to_bits(),
            b.m_eff_ratio.to_bits(),
            "m*/m bitwise mismatch"
        );
        assert_eq!(a.j_mev.to_bits(), b.j_mev.to_bits(), "J bitwise mismatch");
    }
}
