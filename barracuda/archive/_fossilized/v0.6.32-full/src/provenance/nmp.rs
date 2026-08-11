// SPDX-License-Identifier: AGPL-3.0-or-later

/// Nuclear matter property targets and uncertainties.
///
/// Sources:
///   - ρ₀, K∞: Chabanat et al., Nucl. Phys. A 635, 231 (1998)
///   - E/A:     Bender, Heenen, Reinhard, Rev. Mod. Phys. 75, 121 (2003).
///     DOI: [10.1103/RevModPhys.75.121](https://doi.org/10.1103/RevModPhys.75.121)
///   - m*/m:    Chabanat 1998, Bender 2003
///   - J:       Lattimer & Prakash, Phys. Rep. 621, 127 (2016).
///     DOI: [10.1016/j.physrep.2015.12.005](https://doi.org/10.1016/j.physrep.2015.12.005)
pub const NMP_TARGETS: NmpTargets = NmpTargets {
    rho0: NmpTarget {
        value: 0.16,
        sigma: 0.005,
        unit: "fm^-3",
        source: "Chabanat 1998",
    },
    e_a: NmpTarget {
        value: -15.97,
        sigma: 0.5,
        unit: "MeV",
        source: "Bender 2003",
    },
    k_inf: NmpTarget {
        value: 230.0,
        sigma: 20.0,
        unit: "MeV",
        source: "Blaizot 1980, Chabanat 1998",
    },
    m_eff: NmpTarget {
        value: 0.69,
        sigma: 0.1,
        unit: "m*/m",
        source: "Chabanat 1998",
    },
    j: NmpTarget {
        value: 32.0,
        sigma: 2.0,
        unit: "MeV",
        source: "Lattimer & Prakash 2016",
    },
};

/// A single NMP target with uncertainty and source.
#[derive(Debug, Clone, Copy)]
pub struct NmpTarget {
    /// Target value.
    pub value: f64,
    /// Uncertainty (1σ).
    pub sigma: f64,
    /// Physical unit (e.g. "fm⁻³", "`MeV`").
    pub unit: &'static str,
    /// Literature source.
    pub source: &'static str,
}

/// All five NMP targets (ρ₀, E/A, K∞, m*/m, J).
#[derive(Debug, Clone, Copy)]
pub struct NmpTargets {
    /// Saturation density ρ₀ (fm⁻³).
    pub rho0: NmpTarget,
    /// Binding energy per nucleon E/A (`MeV`).
    pub e_a: NmpTarget,
    /// Incompressibility K∞ (`MeV`).
    pub k_inf: NmpTarget,
    /// Effective mass ratio m*/m.
    pub m_eff: NmpTarget,
    /// Symmetry energy J (`MeV`).
    pub j: NmpTarget,
}

impl NmpTargets {
    /// Values as array [ρ₀, E/A, K∞, m*/m, J] for compact comparison
    #[must_use]
    pub const fn values(&self) -> [f64; 5] {
        [
            self.rho0.value,
            self.e_a.value,
            self.k_inf.value,
            self.m_eff.value,
            self.j.value,
        ]
    }

    /// Sigmas as array
    #[must_use]
    pub const fn sigmas(&self) -> [f64; 5] {
        [
            self.rho0.sigma,
            self.e_a.sigma,
            self.k_inf.sigma,
            self.m_eff.sigma,
            self.j.sigma,
        ]
    }

    /// Check if NMP values are within `n_sigma` of targets
    #[must_use]
    pub fn within_sigma(&self, values: &[f64; 5], n_sigma: f64) -> [bool; 5] {
        let targets = self.values();
        let sigmas = self.sigmas();
        std::array::from_fn(|i| (values[i] - targets[i]).abs() < n_sigma * sigmas[i])
    }
}

/// `SLy4` Skyrme parameters (Chabanat et al., Nucl. Phys. A 635, 231-256, 1998, Table I).
///
/// DOI: [10.1016/S0375-9474(98)00180-8](https://doi.org/10.1016/S0375-9474(98)00180-8)
///
/// Order: \[t₀, t₁, t₂, t₃, x₀, x₁, x₂, x₃, α, W₀\] in MeV·fm units.
/// This is the canonical reference parametrization used throughout hotSpring.
pub const SLY4_PARAMS: [f64; 10] = [
    -2488.913, // t₀
    486.818,   // t₁
    -546.395,  // t₂
    13777.0,   // t₃
    0.834,     // x₀
    -0.344,    // x₁
    -1.0,      // x₂
    1.354,     // x₃
    1.0 / 6.0, // α
    123.0,     // W₀
];

/// UNEDF0 Skyrme parameters (Kortelainen et al., Phys. Rev. C 82, 024313, 2010).
///
/// DOI: [10.1103/PhysRevC.82.024313](https://doi.org/10.1103/PhysRevC.82.024313)
///
/// Order: same as [`SLY4_PARAMS`].
pub const UNEDF0_PARAMS: [f64; 10] = [
    -1883.68,  // t₀
    277.50,    // t₁
    -207.20,   // t₂
    14263.6,   // t₃
    0.0085,    // x₀
    -1.532,    // x₁
    -1.0,      // x₂
    0.397,     // x₃
    1.0 / 6.0, // α
    79.53,     // W₀
];

/// Standard 10 Skyrme parameter names, in canonical order.
pub const PARAM_NAMES: [&str; 10] = [
    "t0", "t1", "t2", "t3", "x0", "x1", "x2", "x3", "alpha", "W0",
];

/// Compute NMP χ² from nuclear matter property values.
///
/// `χ²_NMP` = Σ ((`value_i` - `target_i`) / `sigma_i`)²
///
/// Uses [`NMP_TARGETS`] as the reference. Returns the sum of squared pulls
/// for the five nuclear matter observables.
#[must_use]
pub fn nmp_chi2(values: &[f64; 5]) -> f64 {
    let targets = NMP_TARGETS.values();
    let sigmas = NMP_TARGETS.sigmas();
    targets
        .iter()
        .zip(sigmas.iter())
        .zip(values.iter())
        .map(|((&t, &s), &v)| ((v - t) / s).powi(2))
        .sum()
}

/// Compute NMP χ² from a [`NuclearMatterProps`](crate::physics::NuclearMatterProps).
///
/// Convenience wrapper that extracts the five NMP values and calls [`nmp_chi2`].
#[must_use]
pub fn nmp_chi2_from_props(nmp: &crate::physics::NuclearMatterProps) -> f64 {
    let values = [
        nmp.rho0_fm3,
        nmp.e_a_mev,
        nmp.k_inf_mev,
        nmp.m_eff_ratio,
        nmp.j_mev,
    ];
    nmp_chi2(&values)
}

/// Standard NMP observable names.
pub const NMP_NAMES: [&str; 5] = ["rho0", "E/A", "K_inf", "m*/m", "J"];

/// Standard NMP observable units.
pub const NMP_UNITS: [&str; 5] = ["fm^-3", "MeV", "MeV", "", "MeV"];

/// Print a formatted NMP analysis table for a Skyrme parametrization.
///
/// Shows each NMP observable, its computed value, the target, sigma,
/// pull (deviation in sigma units), and PASS/FAIL status.
pub fn print_nmp_analysis(nmp: &crate::physics::NuclearMatterProps) {
    let values = [
        nmp.rho0_fm3,
        nmp.e_a_mev,
        nmp.k_inf_mev,
        nmp.m_eff_ratio,
        nmp.j_mev,
    ];
    let targets = NMP_TARGETS.values();
    let sigmas = NMP_TARGETS.sigmas();

    println!("  NMP Analysis:");
    println!(
        "    {:>8}  {:>10}  {:>8}  {:>10}  {:>8}  {:>6}  Status",
        "Prop", "Value", "Unit", "Target", "Sigma", "Pull"
    );
    for i in 0..5 {
        let pull = (values[i] - targets[i]).abs() / sigmas[i];
        let status = if pull < crate::tolerances::NMP_SIGMA_THRESHOLD {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "    {:>8}  {:>10.4}  {:>8}  {:>10.4}  {:>8.4}  {:>6.2}σ  {status}",
            NMP_NAMES[i], values[i], NMP_UNITS[i], targets[i], sigmas[i], pull,
        );
    }
    println!("  NMP χ²/datum = {:.4}", nmp_chi2_from_props(nmp) / 5.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::NuclearMatterProps;

    #[test]
    fn nmp_targets_are_physical() {
        let t = NMP_TARGETS;
        assert!(t.rho0.value > 0.0, "saturation density must be positive");
        assert!(t.e_a.value < 0.0, "binding energy must be negative");
        assert!(t.k_inf.value > 0.0, "incompressibility must be positive");
        assert!(
            t.m_eff.value > 0.0 && t.m_eff.value < 1.0,
            "effective mass ratio must be in (0, 1)"
        );
        assert!(t.j.value > 0.0, "symmetry energy must be positive");
    }

    #[test]
    fn nmp_within_sigma_sly4() {
        let sly4 = [0.1595, -15.97, 230.0, 0.69, 32.0];
        let within = NMP_TARGETS.within_sigma(&sly4, 2.0);
        assert!(
            within.iter().all(|&b| b),
            "SLy4 should be within 2σ of all targets"
        );
    }

    #[test]
    fn nmp_targets_values_returns_correct_array() {
        let vals = NMP_TARGETS.values();
        assert_eq!(vals.len(), 5);
        assert!((vals[0] - 0.16).abs() < 1e-10, "ρ₀");
        assert!((vals[1] - (-15.97)).abs() < 1e-10, "E/A");
        assert!((vals[2] - 230.0).abs() < 1e-10, "K∞");
        assert!((vals[3] - 0.69).abs() < 1e-10, "m*/m");
        assert!((vals[4] - 32.0).abs() < 1e-10, "J");
    }

    #[test]
    fn nmp_targets_sigmas_returns_correct_array() {
        let sigmas = NMP_TARGETS.sigmas();
        assert_eq!(sigmas.len(), 5);
        assert!((sigmas[0] - 0.005).abs() < 1e-10);
        assert!((sigmas[1] - 0.5).abs() < 1e-10);
        assert!((sigmas[2] - 20.0).abs() < 1e-10);
        assert!((sigmas[3] - 0.1).abs() < 1e-10);
        assert!((sigmas[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn nmp_within_sigma_known_values() {
        let exact = NMP_TARGETS.values();
        let within = NMP_TARGETS.within_sigma(&exact, 0.1);
        assert!(within.iter().all(|&b| b));

        let far = [0.0, 0.0, 0.0, 0.0, 0.0];
        let within = NMP_TARGETS.within_sigma(&far, 0.1);
        assert!(!within.iter().any(|&b| b));

        let one_sigma_high = [
            NMP_TARGETS.rho0.value + NMP_TARGETS.rho0.sigma,
            NMP_TARGETS.e_a.value + NMP_TARGETS.e_a.sigma,
            NMP_TARGETS.k_inf.value + NMP_TARGETS.k_inf.sigma,
            NMP_TARGETS.m_eff.value + NMP_TARGETS.m_eff.sigma,
            NMP_TARGETS.j.value + NMP_TARGETS.j.sigma,
        ];
        let within_05 = NMP_TARGETS.within_sigma(&one_sigma_high, 0.5);
        let within_2 = NMP_TARGETS.within_sigma(&one_sigma_high, 2.0);
        assert!(!within_05.iter().all(|&b| b));
        assert!(within_2.iter().all(|&b| b));
    }

    #[test]
    fn print_nmp_analysis_no_panic() {
        let nmp = NuclearMatterProps {
            rho0_fm3: 0.1595,
            e_a_mev: -15.97,
            k_inf_mev: 230.0,
            m_eff_ratio: 0.69,
            j_mev: 32.0,
        };
        print_nmp_analysis(&nmp);
    }

    #[test]
    fn param_names_length_matches_sly4_params() {
        assert_eq!(PARAM_NAMES.len(), SLY4_PARAMS.len());
    }

    #[test]
    fn sly4_params_have_correct_length() {
        assert_eq!(SLY4_PARAMS.len(), 10);
        assert_eq!(UNEDF0_PARAMS.len(), 10);
        assert_eq!(PARAM_NAMES.len(), 10);
    }

    #[test]
    fn nmp_chi2_sly4_is_small() {
        let sly4_nmp = [0.1595, -15.97, 230.0, 0.69, 32.0];
        let chi2 = nmp_chi2(&sly4_nmp);
        assert!(chi2 < 25.0, "SLy4 NMP χ² should be small, got {chi2}");
    }

    #[test]
    fn nmp_chi2_exact_match_is_zero() {
        let exact = NMP_TARGETS.values();
        let chi2 = nmp_chi2(&exact);
        assert!(
            chi2.abs() < 1e-15,
            "exact match χ² should be ~0, got {chi2}"
        );
    }
}
