// SPDX-License-Identifier: AGPL-3.0-only

//! Sarkas-compatible simulation configuration
//!
//! Defines plasma parameters in reduced units (`a_ws`, omega_p^-1)
//! matching the Sarkas DSF study matrix.

use std::f64::consts::PI;

/// Simulation configuration in reduced units
#[derive(Clone, Debug)]
#[must_use]
pub struct MdConfig {
    /// Label for this case
    pub label: String,
    /// Number of particles
    pub n_particles: usize,
    /// Screening parameter (κ = `a_ws` / `λ_D`)
    pub kappa: f64,
    /// Coupling parameter (Γ = q²/(4πε₀ `a_ws` `k_B` T))
    pub gamma: f64,
    /// Reduced timestep (`dt*` = dt × `ω_p`)
    pub dt: f64,
    /// Cutoff radius in `a_ws`
    pub rc: f64,
    /// Equilibration steps
    pub equil_steps: usize,
    /// Production steps
    pub prod_steps: usize,
    /// Dump interval (production)
    pub dump_step: usize,
    /// Berendsen thermostat relaxation (tau / dt)
    pub berendsen_tau: f64,
    /// RDF histogram bins
    pub rdf_bins: usize,
    /// Velocity snapshot interval in multiples of `dump_step`.
    /// Snapshot every `vel_snapshot_interval * dump_step` production steps.
    /// Default: 100 (i.e. every 1000 steps for `dump_step`=10).
    /// Set lower (e.g. 1) for transport coefficient studies needing fine VACF.
    pub vel_snapshot_interval: usize,
}

impl MdConfig {
    /// Box side length in reduced units: L = (4π N/3)^(1/3) `a_ws`
    #[must_use]
    pub fn box_side(&self) -> f64 {
        (4.0 * PI * self.n_particles as f64 / 3.0).cbrt()
    }

    /// Temperature in reduced units: T* = 1/Γ
    #[must_use]
    pub fn temperature(&self) -> f64 {
        1.0 / self.gamma
    }

    /// Yukawa force prefactor in reduced units: 1.0
    /// F* = exp(-κ r*) × (1 + κ r*) / r*²  (dimensionless in units of E₀/`a_ws`)
    /// The coupling Γ enters through the temperature: T* = 1/Γ
    #[must_use]
    pub const fn force_prefactor(&self) -> f64 {
        1.0
    }

    /// Effective mass in OCP reduced units (`a_ws`, `ω_p`⁻¹).
    ///
    /// Derivation: `ω_p²` = n q²/(ε₀ m), so m = n q²/(ε₀ `ω_p²`).
    /// In reduced units where `a_ws`=1, `ω_p`⁻¹=1, E₀=q²/(4πε₀ `a_ws`):
    ///   `m*` = m × `a_ws²` × `ω_p²` / E₀ = 4π n `a_ws³` = 4π × 3/(4π) = 3.0
    ///
    /// This is a *derived* constant from the OCP E₀-unit convention, not
    /// a magic number. The alternative Sarkas convention uses m*=1 with
    /// V* = Γ·exp(-κr)/r (energy in `k_B`T units). Both are standard;
    /// see Stanton & Murillo PRE 93, 043203 (2016) §II.A.
    ///
    /// Evolution note: consider migrating to the Sarkas m*=1 convention
    /// for ecosystem parity with Python scientific computing tools.
    #[must_use]
    pub const fn reduced_mass(&self) -> f64 {
        3.0
    }

    /// Number density in reduced units: n* = 3/(4π)
    #[must_use]
    pub fn number_density(&self) -> f64 {
        3.0 / (4.0 * PI)
    }
}

// ═══════════════════════════════════════════════════════════════════
// DSF Study Matrix — 9 PP Yukawa cases from hotSpring control
// ═══════════════════════════════════════════════════════════════════

/// Generate all 9 PP Yukawa configurations from the DSF study
#[must_use]
pub fn dsf_pp_cases(n_particles: usize, lite: bool) -> Vec<MdConfig> {
    let (equil, prod, dump) = if lite {
        (5_000, 30_000, 10)
    } else {
        (5_000, 80_000, 10)
    };

    let cases = vec![
        // κ=1: rc = 8.0 a_ws
        ("k1_G14", 1.0, 14.0, 8.0),
        ("k1_G72", 1.0, 72.0, 8.0),
        ("k1_G217", 1.0, 217.0, 8.0),
        // κ=2: rc = 6.5 a_ws
        ("k2_G31", 2.0, 31.0, 6.5),
        ("k2_G158", 2.0, 158.0, 6.5),
        ("k2_G476", 2.0, 476.0, 6.5),
        // κ=3: rc = 6.0 a_ws
        ("k3_G100", 3.0, 100.0, 6.0),
        ("k3_G503", 3.0, 503.0, 6.0),
        ("k3_G1510", 3.0, 1510.0, 6.0),
    ];

    cases
        .into_iter()
        .map(|(label, kappa, gamma, rc)| MdConfig {
            label: label.to_string(),
            n_particles,
            kappa,
            gamma,
            dt: 0.01,
            rc,
            equil_steps: equil,
            prod_steps: prod,
            dump_step: dump,
            berendsen_tau: 5.0,
            rdf_bins: 500,
            vel_snapshot_interval: 100,
        })
        .collect()
}

/// Single test case for quick validation
pub fn quick_test_case(n_particles: usize) -> MdConfig {
    MdConfig {
        label: String::from("k2_G158_test"),
        n_particles,
        kappa: 2.0,
        gamma: 158.0,
        dt: 0.01,
        rc: 6.5,
        equil_steps: 1_000,
        prod_steps: 5_000,
        dump_step: 10,
        berendsen_tau: 5.0,
        rdf_bins: 500,
        vel_snapshot_interval: 100,
    }
}

/// Paper-parity configuration: exact published parameters
///
/// Matches Choi, Dharuman, Murillo (Phys. Rev. E 100, 013206, 2019) and the
/// Dense Plasma Properties Database:
///   N=10,000, 5k equil + 80k production, dt=0.01 `ω_p⁻¹`
///
/// This is the headline validation: same physics, same parameters,
/// consumer GPU vs HPC cluster.
#[must_use]
pub fn paper_parity_cases() -> Vec<MdConfig> {
    dsf_pp_cases(10_000, false) // N=10,000, 5k equil + 80k production
}

/// Extended paper-parity: 100k production steps (upper range of published data).
///
/// Reference: Choi, Dharuman, Murillo (Phys. Rev. E 100, 013206, 2019) — see
/// Dense Plasma Properties Database for exact parameters.
#[must_use]
pub fn paper_parity_extended_cases() -> Vec<MdConfig> {
    let cases = vec![
        // κ=1: rc = 8.0 a_ws
        ("k1_G14", 1.0, 14.0, 8.0),
        ("k1_G72", 1.0, 72.0, 8.0),
        ("k1_G217", 1.0, 217.0, 8.0),
        // κ=2: rc = 6.5 a_ws
        ("k2_G31", 2.0, 31.0, 6.5),
        ("k2_G158", 2.0, 158.0, 6.5),
        ("k2_G476", 2.0, 476.0, 6.5),
        // κ=3: rc = 6.0 a_ws
        ("k3_G100", 3.0, 100.0, 6.0),
        ("k3_G503", 3.0, 503.0, 6.0),
        ("k3_G1510", 3.0, 1510.0, 6.0),
    ];

    cases
        .into_iter()
        .map(|(label, kappa, gamma, rc)| MdConfig {
            label: format!("{label}_paper"),
            n_particles: 10_000,
            kappa,
            gamma,
            dt: 0.01,
            rc,
            equil_steps: 5_000,
            prod_steps: 100_000,
            dump_step: 10,
            berendsen_tau: 5.0,
            rdf_bins: 500,
            vel_snapshot_interval: 100,
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Transport Study Matrix — Stanton & Murillo (2016) PRE 93 043203
// ═══════════════════════════════════════════════════════════════════

/// Transport coefficient study: Gamma-kappa grid for D*, eta*.
///
/// Longer production runs with fine velocity snapshot intervals
/// (every `dump_step`) for converged Green-Kubo VACF integration.
///
/// Reference: Stanton & Murillo (2016) PRE 93 043203
#[must_use]
pub fn transport_cases(n_particles: usize, lite: bool) -> Vec<MdConfig> {
    let (equil, prod, dump, snap_interval) = if lite {
        (50_000, 20_000, 5, 1)
    } else {
        (100_000, 100_000, 5, 1)
    };

    let cases = vec![
        // κ=1: rc = 8.0 a_ws
        ("t_k1_G10", 1.0, 10.0, 8.0),
        ("t_k1_G14", 1.0, 14.0, 8.0), // Sarkas DSF
        ("t_k1_G50", 1.0, 50.0, 8.0),
        ("t_k1_G72", 1.0, 72.0, 8.0), // Sarkas DSF
        ("t_k1_G100", 1.0, 100.0, 8.0),
        ("t_k1_G175", 1.0, 175.0, 8.0),
        ("t_k1_G217", 1.0, 217.0, 8.0), // Sarkas DSF
        // κ=2: rc = 6.5 a_ws
        ("t_k2_G10", 2.0, 10.0, 6.5),
        ("t_k2_G31", 2.0, 31.0, 6.5), // Sarkas DSF
        ("t_k2_G50", 2.0, 50.0, 6.5),
        ("t_k2_G100", 2.0, 100.0, 6.5),
        ("t_k2_G158", 2.0, 158.0, 6.5), // Sarkas DSF
        ("t_k2_G300", 2.0, 300.0, 6.5),
        ("t_k2_G476", 2.0, 476.0, 6.5), // Sarkas DSF
        // κ=3: rc = 6.0 a_ws
        ("t_k3_G10", 3.0, 10.0, 6.0),
        ("t_k3_G50", 3.0, 50.0, 6.0),
        ("t_k3_G100", 3.0, 100.0, 6.0), // Sarkas DSF
        ("t_k3_G300", 3.0, 300.0, 6.0),
        ("t_k3_G503", 3.0, 503.0, 6.0),   // Sarkas DSF
        ("t_k3_G1510", 3.0, 1510.0, 6.0), // Sarkas DSF
    ];

    cases
        .into_iter()
        .map(|(label, kappa, gamma, rc)| MdConfig {
            label: label.to_string(),
            n_particles,
            kappa,
            gamma,
            dt: 0.01,
            rc,
            equil_steps: equil,
            prod_steps: prod,
            dump_step: dump,
            berendsen_tau: 5.0,
            rdf_bins: 500,
            vel_snapshot_interval: snap_interval,
        })
        .collect()
}

/// Sarkas-matched transport cases only (κ>0 DSF study grid).
///
/// Returns the 9 (κ,Γ) points that have exact Sarkas `D_MKS` reference
/// values from the DSF study. κ=0 (Coulomb) cases are excluded because
/// they require PPPM, a different code path.
#[must_use]
pub fn sarkas_validated_cases(n_particles: usize, lite: bool) -> Vec<MdConfig> {
    transport_cases(n_particles, lite)
        .into_iter()
        .filter(|c| {
            matches!(
                (c.kappa as u32, c.gamma as u32),
                (1, 14 | 72 | 217) | (2, 31 | 158 | 476) | (3, 100 | 503 | 1510)
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_side_from_density() {
        // For N=500: L = (4π·500/3)^(1/3) ≈ 12.43 a_ws
        let config = quick_test_case(500);
        let l = config.box_side();
        assert!(
            l > 12.0 && l < 13.0,
            "box side for N=500 should be ~12.4, got {l}"
        );
    }

    #[test]
    fn temperature_from_gamma() {
        let config = quick_test_case(500);
        assert!((config.temperature() - 1.0 / 158.0).abs() < 1e-10);
    }

    #[test]
    fn dsf_pp_nine_cases() {
        let cases = dsf_pp_cases(500, true);
        assert_eq!(cases.len(), 9, "DSF study has 9 PP Yukawa cases");
    }

    #[test]
    fn dsf_kappa_values() {
        let cases = dsf_pp_cases(500, true);
        let kappas: Vec<f64> = cases.iter().map(|c| c.kappa).collect();
        assert_eq!(
            kappas.iter().filter(|&&k| (k - 1.0).abs() < 0.01).count(),
            3
        );
        assert_eq!(
            kappas.iter().filter(|&&k| (k - 2.0).abs() < 0.01).count(),
            3
        );
        assert_eq!(
            kappas.iter().filter(|&&k| (k - 3.0).abs() < 0.01).count(),
            3
        );
    }

    #[test]
    fn paper_parity_uses_n10k() {
        let cases = paper_parity_cases();
        assert_eq!(cases.len(), 9);
        for c in &cases {
            assert_eq!(c.n_particles, 10_000);
            assert_eq!(c.prod_steps, 80_000);
        }
    }

    #[test]
    fn force_prefactor_is_unity() {
        let config = quick_test_case(500);
        assert!((config.force_prefactor() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn reduced_mass_is_three() {
        let config = quick_test_case(500);
        assert!((config.reduced_mass() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn number_density_consistent() {
        let config = quick_test_case(500);
        let l = config.box_side();
        let n_density = config.n_particles as f64 / (l * l * l);
        assert!(
            (n_density - config.number_density()).abs() < 0.01,
            "number density should be consistent with box side"
        );
    }

    #[test]
    fn transport_cases_full_grid() {
        let cases = transport_cases(500, true);
        assert_eq!(
            cases.len(),
            20,
            "transport grid: 12 original + 8 new Sarkas DSF (k3_G100 was already in grid)"
        );
        for c in &cases {
            assert_eq!(c.vel_snapshot_interval, 1);
            assert_eq!(c.dump_step, 5);
        }
    }

    #[test]
    fn sarkas_validated_cases_nine() {
        let cases = sarkas_validated_cases(500, true);
        assert_eq!(cases.len(), 9, "9 Sarkas κ>0 DSF points");
        for c in &cases {
            assert!(c.kappa >= 1.0, "no κ=0 in Sarkas-validated set");
        }
    }

    #[test]
    fn transport_lite_vs_full_steps() {
        let lite = transport_cases(500, true);
        let full = transport_cases(500, false);
        assert_eq!(lite.len(), full.len());
        assert!(lite[0].prod_steps < full[0].prod_steps);
        assert!(lite[0].equil_steps < full[0].equil_steps);
    }

    #[test]
    fn paper_parity_extended_uses_100k_prod() {
        let cases = paper_parity_extended_cases();
        assert_eq!(cases.len(), 9);
        for c in &cases {
            assert_eq!(c.n_particles, 10_000);
            assert_eq!(c.prod_steps, 100_000);
            assert_eq!(c.equil_steps, 5_000);
            assert!(c.label.ends_with("_paper"));
        }
    }

    #[test]
    fn dsf_pp_lite_vs_full() {
        let lite = dsf_pp_cases(500, true);
        let full = dsf_pp_cases(500, false);
        assert_eq!(lite.len(), full.len());
        assert!(lite[0].equil_steps == full[0].equil_steps);
        assert!(
            lite[0].prod_steps < full[0].prod_steps,
            "lite has fewer prod steps"
        );
    }

    #[test]
    fn quick_test_case_fields() {
        let c = quick_test_case(1000);
        assert_eq!(c.label, "k2_G158_test");
        assert_eq!(c.n_particles, 1000);
        assert!((c.kappa - 2.0).abs() < 1e-10);
        assert!((c.gamma - 158.0).abs() < 1e-10);
        assert!((c.dt - 0.01).abs() < 1e-10);
        assert!((c.rc - 6.5).abs() < 1e-10);
        assert_eq!(c.equil_steps, 1_000);
        assert_eq!(c.prod_steps, 5_000);
        assert_eq!(c.dump_step, 10);
        assert!((c.berendsen_tau - 5.0).abs() < 1e-10);
        assert_eq!(c.rdf_bins, 500);
        assert_eq!(c.vel_snapshot_interval, 100);
    }

    #[test]
    fn sarkas_validated_subset_of_transport() {
        let transport = transport_cases(500, true);
        let sarkas = sarkas_validated_cases(500, true);
        let sarkas_labels: std::collections::HashSet<_> = sarkas.iter().map(|c| &c.label).collect();
        for c in &transport {
            let is_sarkas_point = matches!(
                (c.kappa as u32, c.gamma as u32),
                (1, 14 | 72 | 217) | (2, 31 | 158 | 476) | (3, 100 | 503 | 1510)
            );
            assert_eq!(
                sarkas_labels.contains(&c.label),
                is_sarkas_point,
                "sarkas_validated_cases should match filter"
            );
        }
    }
}
