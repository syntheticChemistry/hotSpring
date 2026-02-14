//! Sarkas-compatible simulation configuration
//!
//! Defines plasma parameters in reduced units (a_ws, omega_p^-1)
//! matching the Sarkas DSF study matrix.

use std::f64::consts::PI;

/// Simulation configuration in reduced units
#[derive(Clone, Debug)]
pub struct MdConfig {
    /// Label for this case
    pub label: String,
    /// Number of particles
    pub n_particles: usize,
    /// Screening parameter (κ = a_ws / λ_D)
    pub kappa: f64,
    /// Coupling parameter (Γ = q²/(4πε₀ a_ws k_B T))
    pub gamma: f64,
    /// Reduced timestep (dt* = dt × ω_p)
    pub dt: f64,
    /// Cutoff radius in a_ws
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
}

impl MdConfig {
    /// Box side length in reduced units: L = (4π N/3)^(1/3) a_ws
    pub fn box_side(&self) -> f64 {
        (4.0 * PI * self.n_particles as f64 / 3.0).cbrt()
    }

    /// Temperature in reduced units: T* = 1/Γ
    pub fn temperature(&self) -> f64 {
        1.0 / self.gamma
    }

    /// Yukawa force prefactor in reduced units: 1.0
    /// F* = exp(-κ r*) × (1 + κ r*) / r*²  (dimensionless in units of E₀/a_ws)
    /// The coupling Γ enters through the temperature: T* = 1/Γ
    pub fn force_prefactor(&self) -> f64 {
        1.0
    }

    /// Effective mass in OCP reduced units (a_ws, ω_p⁻¹).
    /// m* = m × a_ws × ω_p² / F₀ = 4π n a_ws³ = 3.0
    pub fn reduced_mass(&self) -> f64 {
        3.0
    }

    /// Number density in reduced units: n* = 3/(4π)
    pub fn number_density(&self) -> f64 {
        3.0 / (4.0 * PI)
    }
}

// ═══════════════════════════════════════════════════════════════════
// DSF Study Matrix — 9 PP Yukawa cases from hotSpring control
// ═══════════════════════════════════════════════════════════════════

/// Generate all 9 PP Yukawa configurations from the DSF study
pub fn dsf_pp_cases(n_particles: usize, lite: bool) -> Vec<MdConfig> {
    let (equil, prod, dump) = if lite {
        (5_000, 30_000, 10)
    } else {
        (5_000, 80_000, 10)
    };

    let cases = vec![
        // κ=1: rc = 8.0 a_ws
        ("k1_G14",   1.0,   14.0,  8.0),
        ("k1_G72",   1.0,   72.0,  8.0),
        ("k1_G217",  1.0,  217.0,  8.0),
        // κ=2: rc = 6.5 a_ws
        ("k2_G31",   2.0,   31.0,  6.5),
        ("k2_G158",  2.0,  158.0,  6.5),
        ("k2_G476",  2.0,  476.0,  6.5),
        // κ=3: rc = 6.0 a_ws
        ("k3_G100",  3.0,  100.0,  6.0),
        ("k3_G503",  3.0,  503.0,  6.0),
        ("k3_G1510", 3.0, 1510.0,  6.0),
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
        })
        .collect()
}

/// Single test case for quick validation
pub fn quick_test_case(n_particles: usize) -> MdConfig {
    MdConfig {
        label: "k2_G158_test".to_string(),
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
    }
}
