// SPDX-License-Identifier: AGPL-3.0-only

//! Daligault (2012) analytical model for Yukawa OCP self-diffusion.
//!
//! Reference: Daligault, PRE 86, 047401 (2012)
//!   "Practical model for the self-diffusion coefficient in Yukawa
//!    one-component plasmas"
//!
//! The model interpolates between weak-coupling (Landau-Spitzer) and
//! strong-coupling (Einstein) limits with a smooth crossover function.
//!
//! All quantities in reduced units: D* = D / (a_ws^2 * omega_p)

use std::f64::consts::PI;

/// Effective Coulomb logarithm for Yukawa potential.
///
/// For kappa=0 recovers OCP Coulomb logarithm.
/// Based on Daligault (2012) Eq. (3).
fn coulomb_log(gamma: f64, kappa: f64) -> f64 {
    let gamma_eff = gamma * (-kappa).exp();
    if gamma_eff < 0.1 {
        (1.0 / gamma_eff).ln().max(1.0)
    } else {
        (1.0 + 1.0 / gamma_eff).ln().max(0.1)
    }
}

/// Kappa-dependent weak-coupling correction factor.
///
/// C_w(κ) = exp(1.435 + 0.715κ + 0.401κ²)
///
/// At κ=0 this recovers C_w ≈ 4.20 (OCP reduced-unit normalization).
/// At higher κ, Yukawa screening suppresses the effective Coulomb
/// logarithm faster than the classical formula captures, requiring an
/// exponentially growing correction. Fitted from Sarkas VACF data at
/// 4 crossover-regime points (Γ ≈ Γ_x) where the weak-coupling term
/// dominates: (κ=0,Γ=10), (κ=1,Γ=14), (κ=2,Γ=31), (κ=3,Γ=100).
///
/// Calibration source: calibrate_daligault_fit.py → weak-coupling
/// correction analysis (Feb 2026).
fn c_weak(kappa: f64) -> f64 {
    (1.435 + 0.715 * kappa + 0.401 * kappa * kappa).exp()
}

/// Weak-coupling (Landau-Spitzer) self-diffusion in reduced units.
///
/// D*_w = C_w(κ) × (3 sqrt(π) / 4) / (Γ^(5/2) × ln(Λ))
///
/// The κ-dependent prefactor C_w(κ) accounts for reduced-unit
/// normalization and Yukawa screening effects on the Coulomb logarithm.
/// Calibrated against Sarkas DSF study Green-Kubo D at 12 (Γ,κ) points
/// (Feb 2026). Replaces constant C_w=5.3 (v0.5.13) which gave 44-63%
/// errors in the crossover regime.
fn d_star_weak(gamma: f64, kappa: f64) -> f64 {
    let cl = coulomb_log(gamma, kappa);
    c_weak(kappa) * 3.0 * PI.sqrt() / 4.0 / (gamma.powf(2.5) * cl)
}

/// Strong-coupling self-diffusion from Einstein frequency model.
///
/// D*_s = A(kappa) / Gamma^alpha(kappa)
///
/// Coefficients recalibrated against Sarkas DSF study (12 validated
/// Green-Kubo D* points, N=2000, Feb 2026). Original Daligault (2012)
/// Table I coefficients gave D* ~70× too small due to reduced-unit
/// normalization mismatch.
fn d_star_strong(gamma: f64, kappa: f64) -> f64 {
    let a = 0.808 + 0.423 * kappa - 0.152 * kappa * kappa;
    let alpha = 1.049 + 0.044 * kappa - 0.039 * kappa * kappa;
    a * gamma.powf(-alpha)
}

/// Crossover function between weak and strong coupling.
///
/// f(Gamma) = 1 / (1 + (Gamma / Gamma_x)^p)
///
/// where Gamma_x(kappa) is the crossover coupling parameter.
fn crossover(gamma: f64, kappa: f64) -> f64 {
    let gamma_x = 10.0 * (0.5 * kappa).exp();
    let p = 2.0;
    1.0 / (1.0 + (gamma / gamma_x).powf(p))
}

/// Reduced self-diffusion coefficient D*(Gamma, kappa).
///
/// Daligault (2012) practical model combining weak and strong coupling.
///
/// # Arguments
/// * `gamma` - Coupling parameter (Gamma = q^2 / (4pi eps0 a_ws kB T))
/// * `kappa` - Screening parameter (kappa = a_ws / lambda_D)
///
/// # Returns
/// D* = D / (a_ws^2 * omega_p) in reduced units.
pub fn d_star_daligault(gamma: f64, kappa: f64) -> f64 {
    let f = crossover(gamma, kappa);
    let dw = d_star_weak(gamma, kappa);
    let ds = d_star_strong(gamma, kappa);
    dw * f + ds * (1.0 - f)
}

// ═══════════════════════════════════════════════════════════════════
// Stanton & Murillo (2016) practical transport models
// Reference: PRE 93, 043203 (2016)
//   "Ionic transport in high-energy-density matter"
// ═══════════════════════════════════════════════════════════════════

/// Reduced shear viscosity η*(Γ, κ) in units of n m a_ws² ω_p.
///
/// Stanton & Murillo (2016) practical model combining:
///   - Weak coupling: Chapman-Enskog kinetic theory
///   - Strong coupling: Empirical fit to MD data
///
/// Fit parameters from PRE 93, 043203 Table I.
pub fn eta_star_stanton_murillo(gamma: f64, kappa: f64) -> f64 {
    let f = crossover(gamma, kappa);
    let ew = eta_star_weak(gamma, kappa);
    let es = eta_star_strong(gamma, kappa);
    ew * f + es * (1.0 - f)
}

/// Weak-coupling viscosity: Chapman-Enskog kinetic theory.
///
/// η*_w = C_w(κ) × (5 sqrt(π) / 16) × Γ^(-5/2) / Λ
///
/// Uses the same κ-dependent correction as D*_w — the Coulomb logarithm
/// enters identically in all Chapman-Enskog transport coefficients.
fn eta_star_weak(gamma: f64, kappa: f64) -> f64 {
    let cl = coulomb_log(gamma, kappa);
    c_weak(kappa) * 5.0 * PI.sqrt() / 16.0 / (gamma.powf(2.5) * cl)
}

/// Strong-coupling viscosity from MD fits.
///
/// η*_s = A_η(κ) × Γ^(-α_η(κ))
///
/// Coefficients rescaled proportionally to the D* recalibration
/// (same reduced-unit normalization fix). Not independently calibrated
/// against MD η* data — treat as estimated until stress ACF validation
/// at N≥2000 is available.
fn eta_star_strong(gamma: f64, kappa: f64) -> f64 {
    let a = 0.44 + 0.23 * kappa - 0.083 * kappa * kappa;
    let alpha = 0.76 + 0.040 * kappa - 0.036 * kappa * kappa;
    a * gamma.powf(-alpha)
}

/// Reduced thermal conductivity λ*(Γ, κ) in units of n k_B a_ws² ω_p.
///
/// Stanton & Murillo (2016) practical model combining:
///   - Weak coupling: Chapman-Enskog kinetic theory
///   - Strong coupling: Empirical fit to MD data
///
/// Fit parameters from PRE 93, 043203 Table I.
pub fn lambda_star_stanton_murillo(gamma: f64, kappa: f64) -> f64 {
    let f = crossover(gamma, kappa);
    let lw = lambda_star_weak(gamma, kappa);
    let ls = lambda_star_strong(gamma, kappa);
    lw * f + ls * (1.0 - f)
}

/// Weak-coupling thermal conductivity: Chapman-Enskog.
///
/// λ*_w = C_w(κ) × (75 sqrt(π) / 64) × Γ^(-5/2) / Λ
///
/// Uses the same κ-dependent correction as D*_w and η*_w.
fn lambda_star_weak(gamma: f64, kappa: f64) -> f64 {
    let cl = coulomb_log(gamma, kappa);
    c_weak(kappa) * 75.0 * PI.sqrt() / 64.0 / (gamma.powf(2.5) * cl)
}

/// Strong-coupling thermal conductivity from MD fits.
///
/// λ*_s = A_λ(κ) × Γ^(-α_λ(κ))
///
/// Coefficients rescaled proportionally to D* recalibration.
/// Not independently calibrated — estimated from D* normalization fix.
fn lambda_star_strong(gamma: f64, kappa: f64) -> f64 {
    let a = 1.04 + 0.54 * kappa - 0.19 * kappa * kappa;
    let alpha = 0.90 + 0.042 * kappa - 0.035 * kappa * kappa;
    a * gamma.powf(-alpha)
}

/// Transport validation result for a single (Gamma, kappa) point.
#[derive(Clone, Debug)]
pub struct TransportResult {
    pub kappa: f64,
    pub gamma: f64,
    pub d_star_md: f64,
    pub d_star_daligault: f64,
    pub d_star_sarkas: Option<f64>,
    pub rel_error_vs_daligault: f64,
    pub rel_error_vs_sarkas: Option<f64>,
    pub viscosity: Option<f64>,
    pub thermal_conductivity: Option<f64>,
    pub passed: bool,
}

// ═══════════════════════════════════════════════════════════════════
// Sarkas DSF Study Provenance Data (Feb 2026)
//
// 12 Green-Kubo self-diffusion coefficients D_MKS [m²/s] from Sarkas
// N=2000 VACF integration. Source: all_observables_validation.json
// Generated by: control/sarkas/simulations/dsf-study/scripts/
//               validate_all_observables.py
//
// Physical parameters (matching Sarkas DSF inputs):
//   Z=1, m=m_p, n=1.62e30 m⁻³
//   a_ws = (3/(4πn))^(1/3) = 5.2820e-11 m
//   ω_p  = sqrt(n e²/(ε₀ m)) = 1.6754e15 rad/s
//   a²ωp = 4.6738e-6 m²/s
//
// Conversion: D* = D_MKS / (a_ws² × ω_p)
// ═══════════════════════════════════════════════════════════════════

/// Sarkas-validated D_MKS reference: (kappa, gamma, D_MKS [m²/s]).
///
/// These are the 12 calibration points from which the Daligault fit
/// coefficients were derived. The fit MUST reproduce these to within
/// `DALIGAULT_FIT_VS_CALIBRATION` tolerance.
pub const SARKAS_D_MKS_REFERENCE: [(f64, f64, f64); 12] = [
    (0.0,   10.0, 5.856_310_235_929_256e-07),
    (0.0,   50.0, 6.053_635_489_140_258e-08),
    (0.0,  150.0, 2.016_617_284_253_193_2e-08),
    (1.0,   14.0, 4.903_065_815_923_215e-07),
    (1.0,   72.0, 5.118_468_699_472_465e-08),
    (1.0,  217.0, 1.697_168_369_389_337_4e-08),
    (2.0,   31.0, 3.000_897_981_857_406_6e-07),
    (2.0,  158.0, 3.384_385_046_277_217_6e-08),
    (2.0,  476.0, 1.203_615_045_379_587_6e-08),
    (3.0,  100.0, 1.356_076_604_608_652_4e-07),
    (3.0,  503.0, 1.787_218_453_909_503_8e-08),
    (3.0, 1510.0, 7.712_262_677_617_913e-09),
];

/// Sarkas-validated RDF first-peak reference: (kappa, gamma, peak_r_aws, peak_g).
///
/// First peak of g(r) from Sarkas N=2000 RDF computation.
/// peak_r_aws is the first-peak position in units of a_ws.
/// peak_g is the first-peak height g(r_peak).
///
/// Physics: peak position should increase slightly with Γ (more ordered),
/// peak height increases strongly with Γ (sharper structure).
pub const SARKAS_RDF_REFERENCE: [(f64, f64, f64, f64); 12] = [
    // (kappa, gamma, first_peak_r_aws, first_peak_g)
    (0.0,   10.0, 1.644, 1.140),
    (0.0,   50.0, 1.669, 1.663),
    (0.0,  150.0, 1.719, 2.358),
    (1.0,   14.0, 1.640, 1.160),
    (1.0,   72.0, 1.688, 1.735),
    (1.0,  217.0, 1.720, 2.480),
    (2.0,   31.0, 1.592, 1.211),
    (2.0,  158.0, 1.670, 1.817),
    (2.0,  476.0, 1.722, 2.567),
    (3.0,  100.0, 1.554, 1.306),
    (3.0,  503.0, 1.662, 1.954),
    (3.0, 1510.0, 1.698, 2.607),
];

/// Look up Sarkas RDF first-peak reference for a given (kappa, gamma).
pub fn sarkas_rdf_lookup(kappa: f64, gamma: f64) -> Option<(f64, f64)> {
    for &(k, g, peak_r, peak_g) in &SARKAS_RDF_REFERENCE {
        if (k - kappa).abs() < 0.01 && (g - gamma).abs() < 0.5 {
            return Some((peak_r, peak_g));
        }
    }
    None
}

/// D_MKS → D* conversion factor: a_ws² × ω_p [m²/s].
///
/// Derived from Sarkas DSF study physical parameters:
///   n = 1.62e30 m⁻³, Z = 1, m = m_p = 1.67262192369e-27 kg
///   a_ws = (3/(4πn))^(1/3) = 5.282005e-11 m
///   ω_p  = sqrt(n e²/(ε₀ m)) = 1.675694e15 rad/s
///   a²ωp = 4.675114e-6 m²/s
///
/// Source: calibrate_daligault_fit.py exact computation.
pub const A2_OMEGA_P: f64 = 4.675_114e-06;

/// Convert Sarkas reference to reduced D*.
pub fn sarkas_d_star_reference() -> [(f64, f64, f64); 12] {
    let mut out = [(0.0, 0.0, 0.0); 12];
    for (i, &(kappa, gamma, d_mks)) in SARKAS_D_MKS_REFERENCE.iter().enumerate() {
        out[i] = (kappa, gamma, d_mks / A2_OMEGA_P);
    }
    out
}

/// Look up Sarkas D* for a given (kappa, gamma), if available.
pub fn sarkas_d_star_lookup(kappa: f64, gamma: f64) -> Option<f64> {
    for &(k, g, d_mks) in &SARKAS_D_MKS_REFERENCE {
        if (k - kappa).abs() < 0.01 && (g - gamma).abs() < 0.5 {
            return Some(d_mks / A2_OMEGA_P);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d_star_weak_coupling_regime() {
        // At very weak coupling (small Gamma), D* should be large
        let d_weak = d_star_daligault(1.0, 1.0);
        let d_strong = d_star_daligault(100.0, 1.0);
        assert!(
            d_weak > d_strong,
            "D* should decrease with Gamma: D*(1)={d_weak} vs D*(100)={d_strong}"
        );
    }

    #[test]
    fn d_star_screening_effect() {
        // Screening (higher kappa) reduces effective coupling,
        // so D* should generally increase with kappa at fixed Gamma
        let d_k0 = d_star_daligault(50.0, 0.0);
        let d_k2 = d_star_daligault(50.0, 2.0);
        // This isn't always monotonic due to the fit, but for moderate coupling
        // higher screening should give higher D* (less collective trapping)
        assert!(d_k0 > 0.0 && d_k2 > 0.0);
    }

    #[test]
    fn d_star_positive() {
        for &kappa in &[0.0, 1.0, 2.0, 3.0] {
            for &gamma in &[1.0, 10.0, 50.0, 100.0, 175.0, 500.0] {
                let d = d_star_daligault(gamma, kappa);
                assert!(
                    d > 0.0,
                    "D*(Gamma={gamma}, kappa={kappa}) = {d} must be positive"
                );
            }
        }
    }

    #[test]
    fn d_star_crossover_is_smooth() {
        // Verify no discontinuity around Gamma_x ~ 10 * exp(0.5*kappa)
        let kappa = 1.0;
        let gamma_x = 10.0 * (0.5_f64).exp(); // ~16.5
        let d_before = d_star_daligault(gamma_x - 1.0, kappa);
        let d_at = d_star_daligault(gamma_x, kappa);
        let d_after = d_star_daligault(gamma_x + 1.0, kappa);

        let jump1 = ((d_before - d_at) / d_at).abs();
        let jump2 = ((d_at - d_after) / d_at).abs();
        assert!(
            jump1 < 0.3 && jump2 < 0.3,
            "Crossover should be smooth: jump1={jump1:.4}, jump2={jump2:.4}"
        );
    }

    #[test]
    fn d_star_matches_all_12_sarkas_calibration_points() {
        let ref_data = sarkas_d_star_reference();
        let mut max_err: f64 = 0.0;
        let mut sum_sq_err = 0.0;

        for &(kappa, gamma, d_star_sarkas) in &ref_data {
            let d_fit = d_star_daligault(gamma, kappa);
            let rel = (d_fit - d_star_sarkas).abs() / d_star_sarkas;
            max_err = max_err.max(rel);
            sum_sq_err += rel * rel;
            assert!(
                rel < 0.20,
                "κ={kappa} Γ={gamma}: D*_fit={d_fit:.4e} vs Sarkas {d_star_sarkas:.4e}, err={:.1}%",
                rel * 100.0
            );
        }

        let rmse = (sum_sq_err / 12.0).sqrt();
        assert!(
            rmse < 0.10,
            "RMSE across 12 Sarkas points: {rmse:.4} (must be < 10%)"
        );
        assert!(
            max_err < 0.20,
            "max error across 12 points: {:.1}% (must be < 20%)",
            max_err * 100.0
        );
    }

    #[test]
    fn d_star_physics_trends_across_sarkas_grid() {
        let ref_data = sarkas_d_star_reference();

        // Within each kappa, D* should decrease with increasing Gamma
        for kappa_int in 0..=3 {
            let kappa = kappa_int as f64;
            let mut points: Vec<(f64, f64)> = ref_data
                .iter()
                .filter(|(k, _, _)| (*k - kappa).abs() < 0.01)
                .map(|&(_, g, d)| (g, d))
                .collect();
            points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for w in points.windows(2) {
                let (g1, d1) = w[0];
                let (g2, d2) = w[1];
                assert!(
                    d1 > d2,
                    "κ={kappa}: D*(Γ={g1})={d1:.4e} should be > D*(Γ={g2})={d2:.4e}"
                );
            }
        }

        // At fixed Gamma=100 (available at kappa=2 and 3 as close matches),
        // stronger screening (higher kappa) should increase D* at strong coupling
        let d_k3_g100 = d_star_daligault(100.0, 3.0);
        let d_k2_g100 = d_star_daligault(100.0, 2.0);
        let d_k1_g100 = d_star_daligault(100.0, 1.0);
        assert!(
            d_k3_g100 > d_k2_g100,
            "D*(κ=3,Γ=100)={d_k3_g100:.4e} should > D*(κ=2,Γ=100)={d_k2_g100:.4e}"
        );
        assert!(
            d_k2_g100 > d_k1_g100,
            "D*(κ=2,Γ=100)={d_k2_g100:.4e} should > D*(κ=1,Γ=100)={d_k1_g100:.4e}"
        );
    }

    #[test]
    fn coulomb_log_positive() {
        for &g in &[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] {
            for &k in &[0.0, 1.0, 2.0, 3.0] {
                let cl = coulomb_log(g, k);
                assert!(
                    cl > 0.0,
                    "Coulomb log must be positive: Gamma={g}, kappa={k}"
                );
            }
        }
    }

    #[test]
    fn eta_star_positive() {
        for &kappa in &[0.0, 1.0, 2.0, 3.0] {
            for &gamma in &[1.0, 10.0, 50.0, 100.0, 175.0] {
                let eta = eta_star_stanton_murillo(gamma, kappa);
                assert!(
                    eta > 0.0,
                    "η*(Γ={gamma}, κ={kappa}) = {eta} must be positive"
                );
            }
        }
    }

    #[test]
    fn lambda_star_positive() {
        for &kappa in &[0.0, 1.0, 2.0, 3.0] {
            for &gamma in &[1.0, 10.0, 50.0, 100.0, 175.0] {
                let lam = lambda_star_stanton_murillo(gamma, kappa);
                assert!(
                    lam > 0.0,
                    "λ*(Γ={gamma}, κ={kappa}) = {lam} must be positive"
                );
            }
        }
    }

    #[test]
    fn viscosity_decreases_with_coupling() {
        let eta_weak = eta_star_stanton_murillo(1.0, 1.0);
        let eta_strong = eta_star_stanton_murillo(100.0, 1.0);
        assert!(
            eta_weak > eta_strong,
            "η* should decrease with Γ: η*(1)={eta_weak} vs η*(100)={eta_strong}"
        );
    }

    #[test]
    fn rdf_peak_height_increases_with_coupling() {
        for kappa_int in 0..=3 {
            let kappa = kappa_int as f64;
            let mut points: Vec<(f64, f64)> = SARKAS_RDF_REFERENCE
                .iter()
                .filter(|(k, _, _, _)| (*k - kappa).abs() < 0.01)
                .map(|&(_, g, _, peak_g)| (g, peak_g))
                .collect();
            points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for w in points.windows(2) {
                let (g1, h1) = w[0];
                let (g2, h2) = w[1];
                assert!(
                    h2 > h1,
                    "κ={kappa}: g(r_peak) at Γ={g2} ({h2:.3}) should be > at Γ={g1} ({h1:.3})"
                );
            }
        }
    }

    #[test]
    fn rdf_peak_position_in_physical_range() {
        for &(kappa, gamma, peak_r, peak_g) in &SARKAS_RDF_REFERENCE {
            assert!(
                (1.4..=1.8).contains(&peak_r),
                "κ={kappa} Γ={gamma}: first peak r={peak_r:.3} a_ws should be in [1.4, 1.8]"
            );
            assert!(
                peak_g > 1.0,
                "κ={kappa} Γ={gamma}: first peak g={peak_g:.3} should be > 1.0"
            );
        }
    }

    #[test]
    fn c_weak_matches_python_calibration() {
        let expected = [
            (0.0, 4.20),
            (1.0, 12.82),
            (2.0, 87.3),
            (3.0, 1325.0),
        ];
        for &(kappa, c_expected) in &expected {
            let c = c_weak(kappa);
            let err = (c - c_expected).abs() / c_expected;
            assert!(
                err < 0.02,
                "C_w(κ={kappa}) = {c:.1} vs expected {c_expected:.1}, err={:.1}%",
                err * 100.0
            );
        }
    }
}
