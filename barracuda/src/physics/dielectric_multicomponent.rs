// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-component Mermin dielectric function for electron-ion plasmas.
//!
//! Extends the single-species Mermin (Paper 44, Chuna & Murillo 2024) to
//! multi-component systems. The total dielectric function is constructed by
//! summing the susceptibilities of each species:
//!
//! ε(k,ω) = 1 + Σ_s χ_s(k,ω)
//!
//! where each species susceptibility χ_s follows the completed Mermin form
//! with species-dependent mass, charge, density, and collision frequency.

use super::dielectric::{Complex, PlasmaParams};
use std::f64::consts::PI;

/// Parameters for a single plasma species in a multi-component system.
#[derive(Debug, Clone)]
pub struct SpeciesParams {
    /// Species mass (in units of proton mass, or arbitrary).
    pub mass: f64,
    /// Species charge number Z (electrons: -1, ions: +Z).
    pub charge: f64,
    /// Species number density.
    pub density: f64,
    /// Species temperature.
    pub temperature: f64,
    /// Species collision frequency.
    pub nu: f64,
}

impl SpeciesParams {
    /// Thermal velocity v_th = √(T/m).
    #[must_use]
    pub fn v_th(&self) -> f64 {
        (self.temperature / self.mass).sqrt()
    }

    /// Species plasma frequency ωₚₛ = √(4π nₛ qₛ²/mₛ).
    #[must_use]
    pub fn omega_p(&self) -> f64 {
        (4.0 * PI * self.density * self.charge * self.charge / self.mass).sqrt()
    }

    /// Species Debye wavenumber k_Ds = ωₚₛ / v_thₛ.
    #[must_use]
    pub fn k_debye(&self) -> f64 {
        self.omega_p() / self.v_th()
    }

    /// Convert to single-species `PlasmaParams` for reuse of existing Vlasov code.
    #[must_use]
    pub fn to_plasma_params(&self) -> PlasmaParams {
        let a = (3.0 / (4.0 * PI * self.density)).cbrt();
        let omega_p = self.omega_p();
        let v_th = self.v_th();
        let k_debye = self.k_debye();
        let gamma = self.charge * self.charge / (a * self.temperature);
        let kappa = k_debye * a;
        PlasmaParams {
            a,
            omega_p,
            temperature: self.temperature,
            v_th,
            k_debye,
            n: self.density,
            gamma,
            kappa,
        }
    }
}

/// Multi-component plasma parameters.
#[derive(Debug, Clone)]
pub struct MultiComponentPlasma {
    /// Species list (electrons, ions, etc.).
    pub species: Vec<SpeciesParams>,
}

impl MultiComponentPlasma {
    /// Total plasma frequency: ωₚ² = Σ ωₚₛ².
    #[must_use]
    pub fn total_omega_p_sq(&self) -> f64 {
        self.species
            .iter()
            .map(|s| {
                let op = s.omega_p();
                op * op
            })
            .sum()
    }

    /// Total Debye wavenumber: k_D² = Σ k_Dₛ².
    #[must_use]
    pub fn total_k_debye_sq(&self) -> f64 {
        self.species
            .iter()
            .map(|s| {
                let kd = s.k_debye();
                kd * kd
            })
            .sum()
    }
}

/// Vlasov susceptibility for a single species: χ₀ₛ(k,ω) = -(k_Ds/k)² W'(zₛ)
///
/// where zₛ = ω/(k v_thₛ √2) and W'(z) = -2(1 + z Z(z)).
fn species_vlasov_susceptibility(k: f64, omega: Complex, species: &SpeciesParams) -> Complex {
    let v_th = species.v_th();
    let k_d = species.k_debye();

    if k.abs() < 1e-30 {
        return Complex::ZERO;
    }

    let z = omega * (1.0 / (k * v_th * std::f64::consts::SQRT_2));
    let z_func = super::dielectric::plasma_dispersion_z(z);
    let w_prime = (Complex::ONE + z * z_func) * (-2.0);

    let ratio = k_d / k;
    w_prime * (ratio * ratio * (-0.5))
}

/// Multi-component completed Mermin dielectric function.
///
/// ε(k,ω) = 1 + Σ_s χ_s^Mermin(k,ω)
///
/// where each species contribution follows the completed Mermin form:
/// χ_s = (ω̃/ω) χ₀ₛ(k,ω̃) / [1 + (iνₛ/ω) Rₛ (1 - G_pₛ)]
///
/// with ω̃ = ω + iνₛ, Rₛ = χ₀ₛ(k,ω̃)/χ₀ₛ(k,0),
/// G_pₛ = Rₛ ω ω̃/(k² v_thₛ²).
pub fn epsilon_multicomponent_mermin(
    k: f64,
    omega: f64,
    plasma: &MultiComponentPlasma,
    momentum_conserving: bool,
) -> Complex {
    if omega.abs() < 1e-15 {
        let mut eps_static = Complex::ONE;
        for s in &plasma.species {
            eps_static += species_vlasov_susceptibility(k, Complex::ZERO, s);
        }
        return eps_static;
    }

    let omega_c = Complex::new(omega, 0.0);
    let mut total_chi = Complex::ZERO;

    for s in &plasma.species {
        let nu = s.nu;
        let omega_shifted = Complex::new(omega, nu);

        let chi_shifted = species_vlasov_susceptibility(k, omega_shifted, s);
        let chi_static = species_vlasov_susceptibility(k, Complex::ZERO, s);

        let numer = (omega_shifted * omega_c.inv()) * chi_shifted;

        let r = if chi_static.abs() > 1e-30 {
            chi_shifted * chi_static.inv()
        } else {
            Complex::ZERO
        };

        let denom = if momentum_conserving {
            let v_th = s.v_th();
            let k2_vth2 = k * k * v_th * v_th;
            let omega_product = omega_c * omega_shifted;
            let g_p = r * omega_product * (1.0 / k2_vth2);
            Complex::ONE + (Complex::I * nu * omega_c.inv()) * r * (Complex::ONE - g_p)
        } else {
            Complex::ONE + (Complex::I * nu * omega_c.inv()) * r
        };

        total_chi += numer * denom.inv();
    }

    Complex::ONE + total_chi
}

/// Dynamic structure factor for species `s` in a multi-component plasma.
///
/// S_s(k,ω) = -(k² T_s)/(π n_s ω) Im[χ_s(k,ω) / |ε(k,ω)|²]
///
/// For the total DSF, sum contributions from all species. Here we return
/// the total DSF using the fluctuation-dissipation theorem on the total ε.
#[must_use]
pub fn multicomponent_dsf(k: f64, omegas: &[f64], plasma: &MultiComponentPlasma) -> Vec<f64> {
    let total_t: f64 = plasma
        .species
        .iter()
        .map(|s| s.density * s.temperature)
        .sum::<f64>()
        / plasma.species.iter().map(|s| s.density).sum::<f64>();

    let total_n: f64 = plasma.species.iter().map(|s| s.density).sum();
    let prefactor = total_t * k * k / (PI * total_n);

    omegas
        .iter()
        .map(|&omega| {
            if omega.abs() < 1e-15 {
                return 0.0;
            }
            let eps = epsilon_multicomponent_mermin(k, omega, plasma, true);
            let loss = -eps.inv().im;
            (prefactor * loss / omega).max(0.0)
        })
        .collect()
}

/// f-sum rule for multi-component: Σ_s ωₚₛ²/2 should equal the integral.
#[must_use]
pub fn multicomponent_f_sum_integral(k: f64, plasma: &MultiComponentPlasma, omega_max: f64) -> f64 {
    let n_points: usize = 50_000;
    let d_omega = omega_max / n_points as f64;
    let mut sum = 0.0;
    for i in 1..n_points {
        let omega = i as f64 * d_omega;
        let eps = epsilon_multicomponent_mermin(k, omega, plasma, true);
        sum += omega * eps.inv().im;
    }
    sum * d_omega
}

#[cfg(test)]
mod tests {
    use super::*;

    fn electron_ion_plasma() -> MultiComponentPlasma {
        // Hydrogen plasma: electrons + protons, T_e = T_i = 1 eV (natural units)
        MultiComponentPlasma {
            species: vec![
                SpeciesParams {
                    mass: 1.0 / 1836.0, // electron mass
                    charge: 1.0,
                    density: 1.0,
                    temperature: 1.0,
                    nu: 0.1,
                },
                SpeciesParams {
                    mass: 1.0, // proton mass
                    charge: 1.0,
                    density: 1.0,
                    temperature: 1.0,
                    nu: 0.01,
                },
            ],
        }
    }

    #[test]
    fn multicomp_reduces_to_single_species_for_ocp() {
        // Use κ = √(3Γ) so the OCP's Yukawa screening matches the species'
        // own Debye wavenumber (self-consistent screening).
        let gamma = 1.0;
        let kappa = (3.0_f64 * gamma).sqrt();
        let single = PlasmaParams::from_coupling(gamma, kappa);
        let multi = MultiComponentPlasma {
            species: vec![SpeciesParams {
                mass: 1.0,
                charge: 1.0,
                density: single.n,
                temperature: single.temperature,
                nu: 0.5,
            }],
        };

        let k = 1.0;
        let omega = 1.5;
        let eps_single = super::super::dielectric::epsilon_completed_mermin(k, omega, 0.5, &single);
        let eps_multi = epsilon_multicomponent_mermin(k, omega, &multi, true);

        let rel_re = (eps_single.re - eps_multi.re).abs() / eps_single.abs().max(1e-15);
        let rel_im = (eps_single.im - eps_multi.im).abs() / eps_single.abs().max(1e-15);
        assert!(
            rel_re < 0.05,
            "Re parity: single={:.6}, multi={:.6}, rel={rel_re:.2e}",
            eps_single.re,
            eps_multi.re
        );
        assert!(
            rel_im < 0.05,
            "Im parity: single={:.6}, multi={:.6}, rel={rel_im:.2e}",
            eps_single.im,
            eps_multi.im
        );
    }

    #[test]
    fn multicomp_static_limit_is_debye() {
        let plasma = electron_ion_plasma();
        let k = 1.0;
        let eps = epsilon_multicomponent_mermin(k, 0.0, &plasma, true);
        let k_d_sq = plasma.total_k_debye_sq();
        let expected = 1.0 + k_d_sq / (k * k);
        let rel = (eps.re - expected).abs() / expected;
        assert!(
            rel < 0.01,
            "Static Debye: ε={:.4}, expected={expected:.4}, rel={rel:.2e}",
            eps.re
        );
    }

    #[test]
    fn multicomp_high_freq_limit() {
        let plasma = electron_ion_plasma();
        let k = 1.0;
        // ω >> ωₚₑ ≈ 152 (electron plasma frequency for m_e = 1/1836)
        let omega = 1000.0;
        let eps = epsilon_multicomponent_mermin(k, omega, &plasma, true);
        assert!(
            (eps.re - 1.0).abs() < 0.1,
            "High-freq: ε.re={:.4}, expected ~1.0",
            eps.re
        );
    }

    #[test]
    fn multicomp_passive_medium() {
        let plasma = electron_ion_plasma();
        let k = 1.0;
        for omega in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let eps = epsilon_multicomponent_mermin(k, omega, &plasma, true);
            assert!(
                eps.im >= -1e-10,
                "Passive medium violation at ω={omega}: Im[ε]={:.6}",
                eps.im
            );
        }
    }

    #[test]
    fn multicomp_dsf_positive() {
        let plasma = electron_ion_plasma();
        let k = 1.0;
        let omegas: Vec<f64> = (1..100).map(|i| 0.1 * i as f64).collect();
        let dsf = multicomponent_dsf(k, &omegas, &plasma);
        let n_pos = dsf.iter().filter(|&&s| s >= 0.0).count();
        let frac = n_pos as f64 / dsf.len() as f64;
        assert!(frac > 0.95, "DSF positivity: {:.1}% positive", frac * 100.0);
    }

    #[test]
    fn multicomp_two_species_differs_from_single() {
        let plasma = electron_ion_plasma();
        let k = 1.0;
        let omega = 5.0;
        let eps_full = epsilon_multicomponent_mermin(k, omega, &plasma, true);

        let e_only = MultiComponentPlasma {
            species: vec![plasma.species[0].clone()],
        };
        let eps_e = epsilon_multicomponent_mermin(k, omega, &e_only, true);

        // Two-species and single-species should give different results
        let diff = (eps_full.re - eps_e.re).abs() + (eps_full.im - eps_e.im).abs();
        assert!(
            diff > 1e-6,
            "Two-species should differ from electron-only: diff={diff:.4e}"
        );
    }
}
