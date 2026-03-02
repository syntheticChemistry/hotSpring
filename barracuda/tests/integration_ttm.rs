// SPDX-License-Identifier: AGPL-3.0-only
#![allow(clippy::unwrap_used)]

//! Integration tests: Two-Temperature Model (TTM) public API.
//!
//! Validates TTM module end-to-end: collision frequency, RK4 integration,
//! equilibrium temperature theory, and error handling.

use hotspring_barracuda::ttm::{
    collision_frequency, equilibrium_temperature_theory, integrate_ttm_rk4, TtmError, TtmSpecies,
};

#[test]
fn ttm_collision_frequency_valid_inputs() {
    let te = 15_000.0;
    let ne = 2.5e25;
    let z = 1.0;
    let mi = 40.0 * 1.66053906660e-27;
    let result = collision_frequency(te, ne, z, mi);
    assert!(result.is_ok());
    let nu = result.unwrap();
    assert!(nu > 0.0);
    assert!(nu.is_finite());
}

#[test]
fn ttm_collision_frequency_invalid_rejects() {
    assert!(collision_frequency(-1.0, 1e25, 1.0, 1e-26).is_err());
    assert!(collision_frequency(1e4, 0.0, 1.0, 1e-26).is_err());
    assert!(collision_frequency(1e4, 1e25, 0.0, 1e-26).is_err());
    assert!(collision_frequency(1e4, 1e25, 1.0, 0.0).is_err());
}

#[test]
fn ttm_integrate_produces_convergence() {
    let species = TtmSpecies {
        name: "Argon".to_string(),
        atomic_mass_amu: 40.0,
        z_ion: 1.0,
        density_m3: 2.5e25,
        te_initial_k: 15_000.0,
        ti_initial_k: 300.0,
    };
    let result = integrate_ttm_rk4(&species, 1e-17, 50_000);
    assert!(result.is_ok());
    let res = result.unwrap();
    let te_final = res.te_history.last().copied().unwrap();
    let ti_final = res.ti_history.last().copied().unwrap();
    let t_eq = equilibrium_temperature_theory(&species);
    let midpoint = f64::midpoint(te_final, ti_final);
    assert!(
        (midpoint - t_eq).abs() / t_eq < 0.1,
        "numerical equilibrium should approach theory"
    );
}

#[test]
fn ttm_integrate_invalid_args_err() {
    let species = TtmSpecies {
        name: "Test".to_string(),
        atomic_mass_amu: 40.0,
        z_ion: 1.0,
        density_m3: 2.5e25,
        te_initial_k: 15_000.0,
        ti_initial_k: 300.0,
    };
    assert_eq!(
        integrate_ttm_rk4(&species, 0.0, 100).unwrap_err(),
        TtmError::InvalidArgument
    );
    assert_eq!(
        integrate_ttm_rk4(&species, 1e-17, 0).unwrap_err(),
        TtmError::InvalidArgument
    );
}

#[test]
fn ttm_equilibrium_theory_known_value() {
    let species = TtmSpecies {
        name: "Test".to_string(),
        atomic_mass_amu: 40.0,
        z_ion: 1.0,
        density_m3: 2.5e25,
        te_initial_k: 10_000.0,
        ti_initial_k: 2_000.0,
    };
    let t_eq = equilibrium_temperature_theory(&species);
    assert!((t_eq - 6_000.0).abs() < 1e-6);
}

#[test]
fn ttm_determinism_integration() {
    let species = TtmSpecies {
        name: "Argon".to_string(),
        atomic_mass_amu: 40.0,
        z_ion: 1.0,
        density_m3: 2.5e25,
        te_initial_k: 15_000.0,
        ti_initial_k: 300.0,
    };
    let a = integrate_ttm_rk4(&species, 1e-17, 100).unwrap();
    let b = integrate_ttm_rk4(&species, 1e-17, 100).unwrap();
    assert_eq!(
        a.te_history.last().unwrap().to_bits(),
        b.te_history.last().unwrap().to_bits()
    );
}
