// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration tests: dielectric / plasma response (CPU, deterministic).

use hotspring_barracuda::physics::dielectric::{
    Complex, PlasmaParams, conductivity_dc, debye_screening, epsilon_completed_mermin,
    epsilon_mermin, epsilon_vlasov,
};
use std::f64::consts::PI;

#[test]
fn plasma_frequency_from_coupling_matches_free_electron_formula() {
    let p = PlasmaParams::from_coupling(1.2, 0.8);
    let n = 1.0_f64;
    let m = 1.0_f64;
    let q = 1.0_f64;
    let expected_wp = (4.0 * PI * n * q * q / m).sqrt();
    assert!(
        (p.omega_p - expected_wp).abs() < 1e-14,
        "ωₚ = √(4π n q²/m), got {} vs {expected_wp}",
        p.omega_p
    );
}

#[test]
fn mermin_static_limit_matches_vlasov_epsilon_at_zero_frequency() {
    let params = PlasmaParams::from_coupling(2.0, 1.5);
    let k = 0.7;
    let nu = 0.03;
    let eps_static = epsilon_vlasov(k, Complex::ZERO, &params);
    let eps_small_omega = epsilon_mermin(k, 1e-20, nu, &params);
    assert!(
        ((eps_small_omega.re - eps_static.re).abs() < 1e-10
            && (eps_small_omega.im - eps_static.im).abs() < 1e-10),
        "ω→0 Mermin should match ε(k,0): {:?} vs {:?}",
        eps_small_omega,
        eps_static
    );
}

#[test]
fn mermin_high_frequency_reaches_dielectric_vacuum() {
    let params = PlasmaParams::from_coupling(10.0, 1.0);
    let nu = 5.0;
    let k = 1.0;
    let omega = 100.0 * params.omega_p;
    let eps = epsilon_mermin(k, omega, nu, &params);
    assert!(
        (eps.re - 1.0).abs() < 0.02,
        "Re ε(ω→∞) → 1, got Re ε = {}",
        eps.re
    );
    assert!(
        eps.im.abs() < 0.05,
        "Im ε should be small at high ω, got {}",
        eps.im
    );

    let eps_cm = epsilon_completed_mermin(k, omega, nu, &params);
    assert!((eps_cm.re - 1.0).abs() < 0.02, "completed Mermin Re ε → 1");
    assert!(
        eps_cm.im.abs() < 0.05,
        "completed Mermin Im ε small at high ω"
    );
}

#[test]
fn drude_dc_conductivity_matches_exact_formula() {
    let params = PlasmaParams::from_coupling(5.0, 1.0);
    let nu = 0.4;
    let sigma = conductivity_dc(nu, &params);
    let expected = params.omega_p * params.omega_p / (4.0 * PI * nu);
    assert!(
        (sigma - expected).abs() < 1e-14,
        "σ(0)=ωₚ²/(4πν): {sigma} vs {expected}"
    );
    assert!(sigma > 0.0, "DC conductivity must be positive");
}

#[test]
fn static_dielectric_real_part_exceeds_one_for_debye_screening() {
    let params = PlasmaParams::from_coupling(1.0, 1.0);
    let k = 0.8;
    let (re_eps, expected_debye) = debye_screening(k, &params);
    assert!(re_eps > 1.0, "ε(k,0) should exceed 1 with Debye screening");
    let err = (re_eps - expected_debye).abs() / expected_debye;
    assert!(err < 1e-10, "Debye law mismatch: err={err}");
}
