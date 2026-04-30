// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::physics::{NuclearMatterProps, nuclear_matter_properties};

/// Physical constraints on nuclear matter properties.
///
/// Parameter sets producing NMP outside these bounds are rejected
/// before any expensive calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct NMPConstraints {
    /// Minimum saturation density ρ₀ (fm⁻³).
    pub rho0_min: f64,
    /// Maximum saturation density ρ₀ (fm⁻³).
    pub rho0_max: f64,
    /// Minimum binding energy E/A (`MeV`).
    pub e_a_min: f64,
    /// Maximum binding energy E/A (`MeV`).
    pub e_a_max: f64,
    /// Minimum incompressibility K∞ (`MeV`).
    pub k_inf_min: f64,
    /// Maximum incompressibility K∞ (`MeV`).
    pub k_inf_max: f64,
    /// Minimum effective mass ratio m*/m.
    pub m_eff_min: f64,
    /// Maximum effective mass ratio m*/m.
    pub m_eff_max: f64,
    /// Minimum symmetry energy J (`MeV`).
    pub j_min: f64,
    /// Maximum symmetry energy J (`MeV`).
    pub j_max: f64,
}

impl Default for NMPConstraints {
    fn default() -> Self {
        Self {
            rho0_min: 0.10,
            rho0_max: 0.22,
            e_a_min: -22.0,
            e_a_max: -8.0,
            k_inf_min: 100.0,
            k_inf_max: 500.0,
            m_eff_min: 0.2,
            m_eff_max: 2.0,
            j_min: 20.0,
            j_max: 45.0,
        }
    }
}

/// Result of Tier 1 NMP pre-screening
#[derive(Debug, Clone)]
#[must_use]
pub enum NMPScreenResult {
    /// Passed — NMP within physical bounds
    Pass(NuclearMatterProps),
    /// Failed — specific reason for rejection
    Fail(String),
}

/// Tier 1: Check if Skyrme parameters produce physically reasonable NMP.
/// Cost: ~1μs (algebraic + bisection for ρ₀)
pub fn nmp_prescreen(params: &[f64], constraints: &NMPConstraints) -> NMPScreenResult {
    if params.len() != 10 || params[8] <= 0.01 || params[8] > 1.0 {
        return NMPScreenResult::Fail(String::from("alpha out of range"));
    }

    let Some(nmp) = nuclear_matter_properties(params) else {
        return NMPScreenResult::Fail(String::from("NMP calculation failed"));
    };

    if nmp.rho0_fm3 < constraints.rho0_min || nmp.rho0_fm3 > constraints.rho0_max {
        return NMPScreenResult::Fail(format!(
            "ρ₀={:.4} out of [{}, {}]",
            nmp.rho0_fm3, constraints.rho0_min, constraints.rho0_max
        ));
    }
    if nmp.e_a_mev < constraints.e_a_min || nmp.e_a_mev > constraints.e_a_max {
        return NMPScreenResult::Fail(format!(
            "E/A={:.2} out of [{}, {}]",
            nmp.e_a_mev, constraints.e_a_min, constraints.e_a_max
        ));
    }
    if nmp.k_inf_mev < constraints.k_inf_min || nmp.k_inf_mev > constraints.k_inf_max {
        return NMPScreenResult::Fail(format!(
            "K∞={:.1} out of [{}, {}]",
            nmp.k_inf_mev, constraints.k_inf_min, constraints.k_inf_max
        ));
    }
    if nmp.m_eff_ratio < constraints.m_eff_min || nmp.m_eff_ratio > constraints.m_eff_max {
        return NMPScreenResult::Fail(format!(
            "m*/m={:.3} out of [{}, {}]",
            nmp.m_eff_ratio, constraints.m_eff_min, constraints.m_eff_max
        ));
    }
    if nmp.j_mev < constraints.j_min || nmp.j_mev > constraints.j_max {
        return NMPScreenResult::Fail(format!(
            "J={:.1} out of [{}, {}]",
            nmp.j_mev, constraints.j_min, constraints.j_max
        ));
    }

    NMPScreenResult::Pass(nmp)
}
