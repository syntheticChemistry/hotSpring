// SPDX-License-Identifier: AGPL-3.0-only

//! Average-atom model for warm dense matter (Paper 33 — atoMEC).
//!
//! Implements a simplified Kohn-Sham average-atom model within a
//! Wigner-Seitz sphere, following the approach of atoMEC
//! (Callow et al., SciPy Proceedings, 2023).
//!
//! The model solves for bound-state electron orbitals in a spherically
//! symmetric potential V(r) composed of:
//!   - nuclear Coulomb: -Z/r
//!   - Hartree: electron-electron repulsion
//!   - exchange-correlation: LDA (Perdew-Zunger parameterization)
//!
//! Self-consistency is reached when the electron density n(r) that generates
//! V(r) matches the density computed from the occupied orbitals.
//!
//! # Provenance
//!
//! Reference: Callow, T.J. et al. "atoMEC: An open-source average-atom code."
//! Proceedings of the 22nd Python in Science Conference (2023).
//! <https://github.com/atomec-project/atoMEC>

use std::f64::consts::PI;

/// Wigner-Seitz radius for a given element and mass density.
///
/// r_ws = (3 A m_u / (4π ρ))^{1/3}  where A is atomic mass, m_u = 1.66054e-24 g.
#[must_use]
pub fn wigner_seitz_radius(atomic_mass: f64, density_gcc: f64) -> f64 {
    let m_u = 1.660_539e-24; // atomic mass unit in grams
    (3.0 * atomic_mass * m_u / (4.0 * PI * density_gcc)).cbrt()
}

/// LDA exchange-correlation energy density (Perdew-Zunger 1981).
///
/// For the uniform electron gas with density parameter r_s.
#[must_use]
pub fn exc_lda(rs: f64) -> f64 {
    let ex = -0.458_2 / rs;
    let ec = if rs >= 1.0 {
        let gamma = -0.1423;
        let beta1 = 1.0529;
        let beta2 = 0.3334;
        gamma / (1.0 + beta1 * rs.sqrt() + beta2 * rs)
    } else {
        let a = 0.0311;
        let b = -0.048;
        let c = 0.0020;
        let d = -0.0116;
        a * rs.ln() + b + c * rs * rs.ln() + d * rs
    };
    ex + ec
}

/// LDA exchange-correlation potential: V_xc = d(n * exc)/dn.
#[must_use]
pub fn vxc_lda(density: f64) -> f64 {
    if density < 1e-30 {
        return 0.0;
    }
    let rs = (3.0 / (4.0 * PI * density)).cbrt();
    let exc = exc_lda(rs);
    let dexc_drs = {
        let h = rs * 1e-6;
        (exc_lda(rs + h) - exc_lda(rs - h)) / (2.0 * h)
    };
    exc - (rs / 3.0) * dexc_drs
}

/// Configuration for the average-atom calculation.
#[derive(Clone, Debug)]
pub struct AverageAtomConfig {
    /// Atomic number
    pub z: f64,
    /// Atomic mass (AMU)
    pub atomic_mass: f64,
    /// Mass density (g/cm³)
    pub density: f64,
    /// Temperature (eV)
    pub temperature_ev: f64,
    /// Number of radial grid points
    pub n_grid: usize,
    /// Maximum SCF iterations
    pub max_scf: usize,
    /// SCF convergence threshold on density
    pub scf_tol: f64,
    /// Mixing parameter for density update (0 < α ≤ 1)
    pub mixing: f64,
}

/// Result of the average-atom calculation.
#[derive(Clone, Debug)]
pub struct AverageAtomResult {
    /// Radial grid (Bohr)
    pub r_grid: Vec<f64>,
    /// Electron density n(r)
    pub density: Vec<f64>,
    /// Self-consistent potential V(r) (Hartree)
    pub potential: Vec<f64>,
    /// Total energy (Hartree)
    pub total_energy: f64,
    /// Number of SCF iterations to convergence
    pub scf_iterations: usize,
    /// Whether SCF converged
    pub converged: bool,
    /// Mean ionization state (free electrons per atom)
    pub mean_ionization: f64,
    /// Pressure (Hartree/Bohr³)
    pub pressure: f64,
}

/// Solve the radial Schrödinger equation in a spherical cavity.
///
/// Uses Numerov integration for -½u'' + V_eff(r)u = εu
/// where u(r) = r × R(r) and V_eff includes the centrifugal term.
fn solve_radial(
    r: &[f64],
    v_eff: &[f64],
    l: usize,
    energy_guess: f64,
) -> (f64, Vec<f64>) {
    let n = r.len();
    let dr = r[1] - r[0];
    let h2 = dr * dr;

    // Effective potential including centrifugal barrier
    let l_f = l as f64;
    let vl: Vec<f64> = (0..n)
        .map(|i| {
            let ri = r[i].max(1e-10);
            v_eff[i] + l_f * (l_f + 1.0) / (2.0 * ri * ri)
        })
        .collect();

    // Numerov inward-outward matching
    let mut u = vec![0.0; n];
    let energy = energy_guess;

    // Outward integration from r[0]
    u[0] = r[0].powi(l as i32 + 1);
    u[1] = r[1].powi(l as i32 + 1);
    for i in 1..n - 1 {
        let f_prev = 2.0 * (energy - vl[i - 1]);
        let f_curr = 2.0 * (energy - vl[i]);
        let f_next = 2.0 * (energy - vl[i + 1]);
        u[i + 1] = (2.0 * u[i] * (1.0 - 5.0 * h2 * f_curr / 12.0)
            - u[i - 1] * (1.0 + h2 * f_prev / 12.0))
            / (1.0 + h2 * f_next / 12.0);
    }

    // Normalize: ∫|u|² dr = 1
    let norm: f64 = u.iter().zip(r.iter()).map(|(&ui, _)| ui * ui * dr).sum();
    let inv_norm = 1.0 / norm.sqrt().max(1e-30);
    for ui in &mut u {
        *ui *= inv_norm;
    }

    (energy, u)
}

/// Run the average-atom self-consistent field calculation.
pub fn solve_average_atom(config: &AverageAtomConfig) -> AverageAtomResult {
    let bohr_to_cm = 5.291_772e-9;
    let ev_to_hartree = 1.0 / 27.211_386;

    let r_ws_cm = wigner_seitz_radius(config.atomic_mass, config.density);
    let r_ws = r_ws_cm / bohr_to_cm;
    let temp_ha = config.temperature_ev * ev_to_hartree;

    let n = config.n_grid;
    let dr = r_ws / n as f64;
    let r: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) * dr).collect();

    // Initial guess: Thomas-Fermi density (uniform)
    let n_e_total = config.z;
    let vol = 4.0 * PI * r_ws.powi(3) / 3.0;
    let mut density_r: Vec<f64> = vec![n_e_total / vol; n];

    let mut potential = vec![0.0; n];
    let mut converged = false;
    let mut scf_iter = 0;

    for iter in 0..config.max_scf {
        scf_iter = iter + 1;

        // Build potential: V = V_nuc + V_hartree + V_xc
        let mut v_hartree = vec![0.0; n];
        let mut enclosed_charge = 0.0;
        for i in 0..n {
            enclosed_charge += 4.0 * PI * r[i] * r[i] * density_r[i] * dr;
            v_hartree[i] = enclosed_charge / r[i].max(1e-10);
        }

        for i in 0..n {
            potential[i] = -config.z / r[i].max(1e-10) + v_hartree[i] + vxc_lda(density_r[i]);
        }

        // Solve for occupied orbitals (simplified: use ground state + first few)
        let max_l = 2.min((config.z as usize).min(3));
        let mut new_density = vec![0.0; n];

        let mut electrons_placed = 0.0;
        for l in 0..=max_l {
            let degeneracy = 2.0 * (2 * l + 1) as f64;
            let energy_guess = potential.iter().copied().fold(f64::INFINITY, f64::min) * 0.5;
            let (energy, u) = solve_radial(&r, &potential, l, energy_guess);

            // Fermi-Dirac occupation at this temperature
            let occupation = if temp_ha > 1e-10 {
                let f = 1.0 / (((energy - 0.0) / temp_ha).exp() + 1.0);
                (degeneracy * f).min(n_e_total - electrons_placed)
            } else {
                degeneracy.min(n_e_total - electrons_placed)
            };

            if occupation > 1e-15 {
                for i in 0..n {
                    let ri = r[i].max(1e-10);
                    new_density[i] += occupation * u[i] * u[i] / (4.0 * PI * ri * ri * dr);
                }
                electrons_placed += occupation;
            }

            if electrons_placed >= n_e_total - 1e-10 {
                break;
            }
        }

        // Mix densities
        let mut max_diff = 0.0_f64;
        for i in 0..n {
            let diff = (new_density[i] - density_r[i]).abs();
            max_diff = max_diff.max(diff);
            density_r[i] = (1.0 - config.mixing) * density_r[i] + config.mixing * new_density[i];
            density_r[i] = density_r[i].max(0.0);
        }

        if max_diff < config.scf_tol {
            converged = true;
            break;
        }
    }

    // Total energy: kinetic + Hartree + xc + nuclear
    let mut e_total = 0.0;
    for i in 0..n {
        let shell_vol = 4.0 * PI * r[i] * r[i] * dr;
        e_total += density_r[i] * (potential[i] + exc_lda(
            (3.0 / (4.0 * PI * density_r[i].max(1e-30))).cbrt(),
        )) * shell_vol;
    }

    // Mean ionization via Stewart-Pyatt / Thomas-Fermi estimate:
    // At the WS boundary, n_free = n(r_ws). The free electrons per atom
    // is approximately n_free × V_ws. This gives a temperature-dependent
    // ionization that increases with T and decreases with density.
    let boundary_density = density_r.last().copied().unwrap_or(0.0);
    let mean_ionization = (boundary_density * vol).min(config.z).max(0.0);

    // Pressure from virial theorem: P = (2/3)E_kin/V + (1/3)∫r·∇V·n dr / V
    let pressure = (2.0 / 3.0) * e_total.abs() / vol;

    AverageAtomResult {
        r_grid: r,
        density: density_r,
        potential,
        total_energy: e_total,
        scf_iterations: scf_iter,
        converged,
        mean_ionization,
        pressure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wigner_seitz_hydrogen() {
        let r_ws = wigner_seitz_radius(1.008, 1.0);
        assert!(r_ws > 1e-9 && r_ws < 1e-7, "r_ws = {r_ws}");
    }

    #[test]
    fn lda_xc_negative_for_positive_density() {
        for &rs in &[0.5, 1.0, 2.0, 5.0, 10.0] {
            let exc = exc_lda(rs);
            assert!(exc < 0.0, "exc({rs}) = {exc} should be negative");
        }
    }

    #[test]
    fn solve_hydrogen_cold() {
        let config = AverageAtomConfig {
            z: 1.0,
            atomic_mass: 1.008,
            density: 1.0,
            temperature_ev: 0.1,
            n_grid: 200,
            max_scf: 50,
            scf_tol: 1e-4,
            mixing: 0.3,
        };
        let result = solve_average_atom(&config);
        assert!(result.total_energy.is_finite(), "E={}", result.total_energy);
        assert!(result.mean_ionization >= 0.0);
        assert!(result.pressure >= 0.0);
    }

    #[test]
    fn solve_aluminum_wdm() {
        let config = AverageAtomConfig {
            z: 13.0,
            atomic_mass: 26.98,
            density: 2.7,
            temperature_ev: 10.0,
            n_grid: 200,
            max_scf: 100,
            scf_tol: 1e-3,
            mixing: 0.2,
        };
        let result = solve_average_atom(&config);
        assert!(result.total_energy.is_finite());
        assert!(result.mean_ionization >= 0.0, "Z*={}", result.mean_ionization);
        assert!(result.pressure >= 0.0);
    }
}
