// SPDX-License-Identifier: AGPL-3.0-only

//! Two-Temperature Model (TTM) 0D ODE solver — laser-plasma electron-ion equilibration.
//!
//! Solves the 0D TTM equations:
//! ```text
//! dTe/dt = -ν_ei × (Te - Ti)    (electron temperature)
//! dTi/dt = +ν_ei × (Te - Ti) × me/mi    (ion temperature)
//! ```
//!
//! Where ν_ei is the Spitzer electron-ion collision frequency.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// Physical constants (SI units)
// ═══════════════════════════════════════════════════════════════════

/// Boltzmann constant (J/K)
const KB: f64 = 1.380649e-23;
/// Elementary charge (C)
const E_CHARGE: f64 = 1.602176634e-19;
/// Electron mass (kg)
const ME: f64 = 9.1093837015e-31;
/// Vacuum permittivity (F/m)
const EPS0: f64 = 8.8541878128e-12;
/// Atomic mass unit (kg)
const AMU: f64 = 1.66053906660e-27;

/// Species parameters for the 0D TTM.
#[derive(Debug, Clone)]
pub struct TtmSpecies {
    /// Species name (e.g. "Argon", "Xenon", "Helium")
    pub name: String,
    /// Ion mass in atomic mass units (amu)
    pub atomic_mass_amu: f64,
    /// Ionization state (effective Z)
    pub z_ion: f64,
    /// Number density (1/m³). For ions: n_i; electrons n_e = z_ion × n_i (quasineutrality).
    pub density_m3: f64,
    /// Initial electron temperature (K)
    pub te_initial_k: f64,
    /// Initial ion temperature (K)
    pub ti_initial_k: f64,
}

impl TtmSpecies {
    /// Ion mass in kg.
    #[must_use]
    pub fn ion_mass_kg(&self) -> f64 {
        self.atomic_mass_amu * AMU
    }

    /// Electron number density (quasineutrality: n_e = Z × n_i).
    #[must_use]
    pub fn electron_density_m3(&self) -> f64 {
        self.z_ion * self.density_m3
    }

    /// Ion number density.
    #[must_use]
    pub fn ion_density_m3(&self) -> f64 {
        self.density_m3
    }
}

/// Result of TTM integration.
#[derive(Debug, Clone)]
pub struct TtmResult {
    /// Time points (s)
    pub times: Vec<f64>,
    /// Electron temperature history (K)
    pub te_history: Vec<f64>,
    /// Ion temperature history (K)
    pub ti_history: Vec<f64>,
    /// Equilibrium temperature (K) — mass-weighted average at convergence
    pub equilibrium_temperature: f64,
    /// Time (s) when |Te - Ti| < threshold, if reached
    pub equilibration_time: Option<f64>,
}

/// Coulomb logarithm ln(Λ).
///
/// Standard plasma parameter: Λ = b_max/b_min ≈ 4π n_e λD³ where
/// λD = sqrt(ε₀ kB Te / (ne e²)) is the electron Debye length.
/// So ln(Λ) = ln(4π ne λD³) with λD³ = (ε₀ kB Te / (ne e²))^(3/2).
///
/// # Errors
/// Returns `Err` if arguments are invalid (Te ≤ 0, ne ≤ 0).
pub fn coulomb_log(te: f64, ne: f64) -> Result<f64, TtmError> {
    if te <= 0.0 || ne <= 0.0 {
        return Err(TtmError::InvalidArgument);
    }
    let lambda_d3 = (EPS0 * KB * te / (ne * E_CHARGE * E_CHARGE)).powf(1.5);
    let plasma_param = 4.0 * PI * ne * lambda_d3;
    let ln_lambda = plasma_param.ln().max(1.0);
    Ok(ln_lambda)
}

/// Spitzer electron-ion collision frequency ν_ei (Hz).
///
/// ν_ei = (4√(2π) × n_i Z² × e⁴ × ln(Λ)) / (3 × (4πε₀)² × me^0.5 × (kB×Te)^1.5)
/// In SI units. n_i = n_e/Z for quasineutrality.
///
/// # Errors
/// Returns `Err` if any argument is invalid.
pub fn collision_frequency(te: f64, ne: f64, z: f64, mi: f64) -> Result<f64, TtmError> {
    if te <= 0.0 || ne <= 0.0 || mi <= 0.0 || z <= 0.0 {
        return Err(TtmError::InvalidArgument);
    }
    let ln_lambda = coulomb_log(te, ne)?;
    let n_i = ne / z;
    let four_pi_eps0_sq = (4.0 * PI * EPS0).powi(2);
    let numer = 4.0 * (2.0 * PI).sqrt() * n_i * (z * z) * (E_CHARGE.powi(4)) * ln_lambda;
    let denom = 3.0 * four_pi_eps0_sq * ME.sqrt() * (KB * te).powf(1.5);
    let nu_ei = numer / denom;
    Ok(nu_ei)
}

/// RHS of the 0D TTM ODE: (dTe/dt, dTi/dt).
///
/// dTe/dt = -ν_ei × (Te - Ti)
/// dTi/dt = +ν_ei × (Te - Ti) × (ne/ni)
///
/// Energy conservation: Ce dTe/dt + Ci dTi/dt = 0 with Ce = (3/2)ne kB, Ci = (3/2)ni kB
/// requires dTi/dt = (ne/ni) × (-dTe/dt).
pub fn ttm_rhs(te: f64, ti: f64, species: &TtmSpecies) -> Result<(f64, f64), TtmError> {
    let ne = species.electron_density_m3();
    let ni = species.ion_density_m3();
    let mi = species.ion_mass_kg();
    let nu = collision_frequency(te, ne, species.z_ion, mi)?;
    let dtdt = te - ti;
    let dte_dt = -nu * dtdt;
    let dti_dt = nu * dtdt * (ne / ni);
    Ok((dte_dt, dti_dt))
}

/// Fourth-order Runge-Kutta integration of the 0D TTM.
///
/// Integrates from t=0 to t = dt × n_steps using fixed step size.
///
/// # Errors
/// Returns `Err` if integration fails (e.g. invalid temperatures).
pub fn integrate_ttm_rk4(
    species: &TtmSpecies,
    dt: f64,
    n_steps: usize,
) -> Result<TtmResult, TtmError> {
    if dt <= 0.0 || n_steps == 0 {
        return Err(TtmError::InvalidArgument);
    }

    let mut times = Vec::with_capacity(n_steps + 1);
    let mut te_history = Vec::with_capacity(n_steps + 1);
    let mut ti_history = Vec::with_capacity(n_steps + 1);

    let mut te = species.te_initial_k;
    let mut ti = species.ti_initial_k;
    let mut t = 0.0;

    times.push(t);
    te_history.push(te);
    ti_history.push(ti);

    for _ in 0..n_steps {
        let (k1e, k1i) = ttm_rhs(te, ti, species).map_err(|_| TtmError::IntegrationFailed)?;
        let te_k2 = te + 0.5 * dt * k1e;
        let ti_k2 = ti + 0.5 * dt * k1i;
        let (k2e, k2i) = ttm_rhs(te_k2, ti_k2, species).map_err(|_| TtmError::IntegrationFailed)?;
        let te_k3 = te + 0.5 * dt * k2e;
        let ti_k3 = ti + 0.5 * dt * k2i;
        let (k3e, k3i) = ttm_rhs(te_k3, ti_k3, species).map_err(|_| TtmError::IntegrationFailed)?;
        let te_k4 = te + dt * k3e;
        let ti_k4 = ti + dt * k3i;
        let (k4e, k4i) = ttm_rhs(te_k4, ti_k4, species).map_err(|_| TtmError::IntegrationFailed)?;

        te += (dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e);
        ti += (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i);
        t += dt;

        if !te.is_finite() || !ti.is_finite() || te < 0.0 || ti < 0.0 {
            return Err(TtmError::IntegrationFailed);
        }

        times.push(t);
        te_history.push(te);
        ti_history.push(ti);
    }

    let equilibrium_temperature = f64::midpoint(te, ti);
    let equilibration_time =
        find_equilibration_time_internal(&times, &te_history, &ti_history, 100.0);

    Ok(TtmResult {
        times,
        te_history,
        ti_history,
        equilibrium_temperature,
        equilibration_time,
    })
}

/// Find time when |Te - Ti| < threshold_k.
///
/// Returns the first time index where the condition is satisfied.
#[must_use]
pub fn find_equilibration_time(result: &TtmResult, threshold_k: f64) -> Option<f64> {
    find_equilibration_time_internal(
        &result.times,
        &result.te_history,
        &result.ti_history,
        threshold_k,
    )
}

fn find_equilibration_time_internal(
    times: &[f64],
    te: &[f64],
    ti: &[f64],
    threshold_k: f64,
) -> Option<f64> {
    for (i, (&t, (&te_val, &ti_val))) in times.iter().zip(te.iter().zip(ti.iter())).enumerate() {
        if i == 0 {
            continue;
        }
        if (te_val - ti_val).abs() < threshold_k {
            return Some(t);
        }
    }
    None
}

/// Theoretical equilibrium temperature from energy conservation.
///
/// T_eq = (Ce×Te0 + Ci×Ti0) / (Ce + Ci) = (ne×Te0 + ni×Ti0) / (ne + ni)
/// with Ce = (3/2) ne kB, Ci = (3/2) ni kB. For quasineutrality ne = Z×ni:
/// T_eq = (Z×Te0 + Ti0) / (Z + 1)
#[must_use]
pub fn equilibrium_temperature_theory(species: &TtmSpecies) -> f64 {
    let ne = species.electron_density_m3();
    let ni = species.ion_density_m3();
    (ne * species.te_initial_k + ni * species.ti_initial_k) / (ne + ni)
}

/// Error type for TTM operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtmError {
    /// Invalid argument (e.g. Te ≤ 0, ne ≤ 0)
    InvalidArgument,
    /// Integration produced invalid state
    IntegrationFailed,
}

impl std::fmt::Display for TtmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument => write!(f, "TTM: invalid argument"),
            Self::IntegrationFailed => write!(f, "TTM: integration failed"),
        }
    }
}

impl std::error::Error for TtmError {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_collision_frequency_invalid_negative_te() {
        let te = -100.0;
        let ne = 2.5e25;
        let z = 1.0;
        let mi = 40.0 * AMU;
        let result = collision_frequency(te, ne, z, mi);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TtmError::InvalidArgument);
    }

    #[test]
    fn test_collision_frequency_invalid_zero_ne() {
        let te = 15_000.0;
        let ne = 0.0;
        let z = 1.0;
        let mi = 40.0 * AMU;
        let result = collision_frequency(te, ne, z, mi);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TtmError::InvalidArgument);
    }

    #[test]
    fn test_collision_frequency_invalid_zero_z() {
        let te = 15_000.0;
        let ne = 2.5e25;
        let z = 0.0;
        let mi = 40.0 * AMU;
        let result = collision_frequency(te, ne, z, mi);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TtmError::InvalidArgument);
    }

    #[test]
    fn test_collision_frequency_invalid_zero_mi() {
        let te = 15_000.0;
        let ne = 2.5e25;
        let z = 1.0;
        let mi = 0.0;
        let result = collision_frequency(te, ne, z, mi);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TtmError::InvalidArgument);
    }

    #[test]
    fn test_integrate_ttm_rk4_one_step() {
        let species = TtmSpecies {
            name: "Argon".to_string(),
            atomic_mass_amu: 40.0,
            z_ion: 1.0,
            density_m3: 2.5e25,
            te_initial_k: 15_000.0,
            ti_initial_k: 300.0,
        };
        let dt = 1e-17;
        let n_steps = 1;
        let result = integrate_ttm_rk4(&species, dt, n_steps);
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.times.len(), 2, "1 step → 2 time points (t=0, t=dt)");
        assert_eq!(res.te_history.len(), 2);
        assert_eq!(res.ti_history.len(), 2);
        assert!((res.times[1] - dt).abs() < 1e-25);
    }

    #[test]
    fn test_integrate_ttm_rk4_invalid_dt_zero() {
        let species = TtmSpecies {
            name: "Argon".to_string(),
            atomic_mass_amu: 40.0,
            z_ion: 1.0,
            density_m3: 2.5e25,
            te_initial_k: 15_000.0,
            ti_initial_k: 300.0,
        };
        let result = integrate_ttm_rk4(&species, 0.0, 100);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TtmError::InvalidArgument);
    }

    #[test]
    fn test_integrate_ttm_rk4_invalid_n_steps_zero() {
        let species = TtmSpecies {
            name: "Argon".to_string(),
            atomic_mass_amu: 40.0,
            z_ion: 1.0,
            density_m3: 2.5e25,
            te_initial_k: 15_000.0,
            ti_initial_k: 300.0,
        };
        let result = integrate_ttm_rk4(&species, 1e-17, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TtmError::InvalidArgument);
    }

    #[test]
    fn test_equilibrium_temperature_theory_known_case() {
        // T_eq = (ne×Te0 + ni×Ti0) / (ne + ni). For Z=1: ne=ni, so T_eq = (Te0 + Ti0)/2.
        let species = TtmSpecies {
            name: "Test".to_string(),
            atomic_mass_amu: 40.0,
            z_ion: 1.0,
            density_m3: 2.5e25,
            te_initial_k: 10_000.0,
            ti_initial_k: 2_000.0,
        };
        let t_eq = equilibrium_temperature_theory(&species);
        let expected = f64::midpoint(10_000.0, 2_000.0);
        assert!(
            (t_eq - expected).abs() < 1e-10,
            "Z=1: T_eq = (Te+Ti)/2 = {expected}, got {t_eq}"
        );
    }

    #[test]
    fn test_equilibrium_temperature_theory_already_equilibrated() {
        // Te = Ti → T_eq = Te = Ti
        let species = TtmSpecies {
            name: "Test".to_string(),
            atomic_mass_amu: 40.0,
            z_ion: 1.0,
            density_m3: 2.5e25,
            te_initial_k: 5_000.0,
            ti_initial_k: 5_000.0,
        };
        let t_eq = equilibrium_temperature_theory(&species);
        assert!(
            (t_eq - 5_000.0).abs() < 1e-10,
            "Te=Ti: T_eq should equal both, got {t_eq}"
        );
    }

    #[test]
    fn test_ttm_determinism() {
        let species = TtmSpecies {
            name: "Argon".to_string(),
            atomic_mass_amu: 40.0,
            z_ion: 1.0,
            density_m3: 2.5e25,
            te_initial_k: 15_000.0,
            ti_initial_k: 300.0,
        };
        let dt = 1e-17;
        let n_steps = 100;
        let a = integrate_ttm_rk4(&species, dt, n_steps).unwrap();
        let b = integrate_ttm_rk4(&species, dt, n_steps).unwrap();
        assert_eq!(a.times.len(), b.times.len());
        for (ta, tb) in a.times.iter().zip(b.times.iter()) {
            assert_eq!(ta.to_bits(), tb.to_bits());
        }
        let te_final_a = a.te_history.last().copied().unwrap();
        let te_final_b = b.te_history.last().copied().unwrap();
        assert_eq!(te_final_a.to_bits(), te_final_b.to_bits());
    }

    #[test]
    fn test_collision_frequency_positive() {
        let te = 15_000.0;
        let ne = 2.5e25;
        let z = 1.0;
        let mi = 40.0 * AMU;
        let Ok(nu) = collision_frequency(te, ne, z, mi) else {
            panic!("collision_frequency returned Err for valid inputs");
        };
        assert!(nu > 0.0, "ν_ei should be positive for valid inputs");
    }

    #[test]
    fn test_coulomb_log_reasonable() {
        let te = 15_000.0;
        let ne = 2.5e25;
        let Ok(ln_lambda) = coulomb_log(te, ne) else {
            panic!("coulomb_log returned Err for valid inputs");
        };
        assert!(
            (1.0..=40.0).contains(&ln_lambda),
            "ln(Λ) should be between 1 and 40 for typical plasmas, got {ln_lambda}"
        );
    }

    #[test]
    fn test_rk4_energy_conservation() {
        let species = TtmSpecies {
            name: "Argon".to_string(),
            atomic_mass_amu: 40.0,
            z_ion: 1.0,
            density_m3: 2.5e25,
            te_initial_k: 15_000.0,
            ti_initial_k: 300.0,
        };
        let dt = 1e-17;
        let n_steps = 10_000;
        let Ok(result) = integrate_ttm_rk4(&species, dt, n_steps) else {
            panic!("integration returned Err");
        };
        let ne = species.electron_density_m3();
        let ni = species.ion_density_m3();
        let e0 = (3.0 / 2.0) * (ne * KB * species.te_initial_k + ni * KB * species.ti_initial_k);
        let te_final = result.te_history.last().copied().unwrap_or(0.0);
        let ti_final = result.ti_history.last().copied().unwrap_or(0.0);
        let e_final = (3.0 / 2.0) * (ne * KB * te_final + ni * KB * ti_final);

        let rel_err = ((e_final - e0) / e0).abs();
        assert!(
            rel_err < 0.01,
            "Energy conservation: rel_err = {rel_err:.6}"
        );
    }

    #[test]
    fn test_equilibrium_correct() {
        let species = TtmSpecies {
            name: "Test".to_string(),
            atomic_mass_amu: 40.0,
            z_ion: 1.0,
            density_m3: 2.5e25,
            te_initial_k: 15_000.0,
            ti_initial_k: 300.0,
        };
        let t_eq_theory = equilibrium_temperature_theory(&species);
        let dt = 1e-17;
        let n_steps = 100_000;
        let Ok(result) = integrate_ttm_rk4(&species, dt, n_steps) else {
            panic!("integration returned Err");
        };
        let te_final = result.te_history.last().copied().unwrap_or(0.0);
        let ti_final = result.ti_history.last().copied().unwrap_or(0.0);
        let t_eq_numerical = f64::midpoint(te_final, ti_final);

        let rel_err = ((t_eq_numerical - t_eq_theory) / t_eq_theory).abs();
        assert!(rel_err < 0.05,
            "Equilibrium T: theory={t_eq_theory:.1}, numerical={t_eq_numerical:.1}, rel_err={rel_err:.6}");
    }
}
