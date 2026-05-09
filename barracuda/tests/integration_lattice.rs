// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration tests: lattice QCD public API (Wilson action, plaquette, HMC).

use hotspring_barracuda::lattice::{hmc, wilson};

const DIM_4444: [usize; 4] = [4, 4, 4, 4];

#[test]
fn cold_start_unit_plaquette_and_zero_wilson_action() {
    let lat = wilson::Lattice::cold_start(DIM_4444, 6.0);
    let plaq = lat.average_plaquette();
    assert!(
        (plaq - 1.0).abs() < 1e-14,
        "cold start ⟨□⟩ should be 1, got {plaq}"
    );
    let s = lat.wilson_action();
    assert!(
        s.abs() < 1e-14,
        "cold start Wilson action S should be 0, got {s}"
    );
}

#[test]
fn hot_start_plaquette_below_unity() {
    let lat = wilson::Lattice::hot_start(DIM_4444, 6.0, 42);
    let plaq = lat.average_plaquette();
    assert!(
        plaq > 0.0 && plaq < 1.0,
        "hot start should disorder links: ⟨□⟩ in (0,1), got {plaq}"
    );
}

#[test]
fn hot_start_wilson_action_is_strictly_positive() {
    let beta = 5.5;
    let lat = wilson::Lattice::hot_start(DIM_4444, beta, 7);
    let s = lat.wilson_action();
    assert!(
        s.is_finite() && s > 0.0,
        "disordered links give S > 0, got {s}"
    );
}

#[test]
fn hmc_trajectory_keeps_plaquette_in_physical_range() {
    let mut lat_cold = wilson::Lattice::cold_start(DIM_4444, 6.0);
    let mut cfg = hmc::HmcConfig {
        n_md_steps: 24,
        dt: 0.04,
        seed: 12345,
        integrator: hmc::IntegratorType::Leapfrog,
    };
    let result = hmc::hmc_trajectory(&mut lat_cold, &mut cfg);
    let plaq = result.plaquette;
    assert!(
        (0.0..=1.0).contains(&plaq),
        "plaquette must stay in [0,1], got {plaq}"
    );

    let mut lat_hot = wilson::Lattice::hot_start(DIM_4444, 5.9, 99);
    let mut cfg_hot = hmc::HmcConfig {
        n_md_steps: 20,
        dt: 0.035,
        seed: 2026,
        integrator: hmc::IntegratorType::Omelyan,
    };
    let r2 = hmc::hmc_trajectory(&mut lat_hot, &mut cfg_hot);
    let p2 = r2.plaquette;
    assert!(
        (0.0..=1.0).contains(&p2),
        "after HMC plaquette must remain in [0,1], got {p2}"
    );
}
