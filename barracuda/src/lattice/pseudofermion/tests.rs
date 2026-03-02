// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::lattice::cg::cg_solve;
use crate::lattice::dirac::FermionField;
use crate::lattice::su3::Su3Matrix;
use crate::lattice::wilson::Lattice;

#[test]
fn heatbath_produces_nonzero_field() {
    let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
    let mut seed = 42u64;
    let phi = pseudofermion_heatbath(&lat, 0.1, &mut seed);
    let norm = phi.norm_sq();
    assert!(norm > 0.0, "pseudofermion should be nonzero: norm²={norm}");
}

#[test]
fn pseudofermion_action_positive() {
    let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
    let mut seed = 42u64;
    let phi = pseudofermion_heatbath(&lat, 0.5, &mut seed);
    let config = PseudofermionConfig {
        mass: 0.5,
        cg_tol: 1e-8,
        cg_max_iter: 1000,
    };
    let (action, cg_res, _) = pseudofermion_action(&lat, &phi, &config);
    assert!(cg_res.converged, "CG should converge");
    assert!(
        action > 0.0,
        "pseudofermion action should be positive: {action}"
    );
}

#[test]
fn fermion_force_is_traceless_antihermitian() {
    let lat = Lattice::hot_start([4, 4, 4, 4], 5.5, 42);
    let vol = lat.volume();
    let x_field = FermionField::random(vol, 99);
    let force = pseudofermion_force(&lat, &x_field, 0.1);

    for (i, f) in force.iter().enumerate().take(10) {
        let tr = f.m[0][0] + f.m[1][1] + f.m[2][2];
        assert!(
            tr.abs_sq() < 1e-20,
            "force[{i}] trace should be ~0: |tr|²={}",
            tr.abs_sq()
        );
        for a in 0..3 {
            for b in 0..3 {
                let diff = f.m[a][b] + f.m[b][a].conj();
                assert!(
                    diff.abs_sq() < 1e-20,
                    "force[{i}] ({a},{b}) should be anti-Hermitian: |f+f†|²={}",
                    diff.abs_sq()
                );
            }
        }
    }
}

#[test]
fn dynamical_hmc_cold_start_runs() {
    let mut lat = Lattice::cold_start([4, 4, 4, 4], 5.5);
    let mut config = DynamicalHmcConfig {
        n_md_steps: 5,
        dt: 0.01,
        seed: 42,
        fermion: PseudofermionConfig {
            mass: 0.5,
            cg_tol: 1e-6,
            cg_max_iter: 500,
        },
        beta: 5.5,
        n_flavors_over_4: 1,
        ..Default::default()
    };

    let result = dynamical_hmc_trajectory(&mut lat, &mut config);
    assert!(
        result.plaquette > 0.0 && result.plaquette <= 1.0,
        "plaquette should be in (0,1]: {}",
        result.plaquette
    );
    assert!(
        result.cg_iterations > 0,
        "should need at least 1 CG iteration"
    );
}

/// Compute squared Frobenius norm of a force field.
fn force_norm_sq(force: &[Su3Matrix]) -> f64 {
    force.iter().map(|f| f.norm_sq()).sum()
}

#[test]
fn hasenbusch_reduces_force_norm() {
    let lat = Lattice::hot_start([4, 4, 4, 4], 5.5, 42);
    let vol = lat.volume();
    let mut seed = 43u64;

    let phi = hasenbusch_ratio_heatbath(&lat, 0.4, &mut seed);

    let config = HasenbuschConfig {
        heavy_mass: 0.4,
        light_mass: 0.1,
        cg_tol: 1e-6,
        cg_max_iter: 500,
        n_md_steps_light: 4,
        n_md_steps_heavy: 2,
    };

    let mut x_heavy = FermionField::zeros(vol);
    let mut x_light = FermionField::zeros(vol);

    let _ = cg_solve(
        &lat,
        &mut x_heavy,
        &phi,
        config.heavy_mass,
        config.cg_tol,
        config.cg_max_iter,
    );
    let _ = cg_solve(
        &lat,
        &mut x_light,
        &phi,
        config.light_mass,
        config.cg_tol,
        config.cg_max_iter,
    );

    let force_heavy = pseudofermion_force(&lat, &x_heavy, config.heavy_mass);
    let force_light = pseudofermion_force(&lat, &x_light, config.light_mass);

    let norm_heavy = force_norm_sq(&force_heavy);
    let norm_light = force_norm_sq(&force_light);

    assert!(
        norm_heavy < norm_light,
        "heavy-mass force norm² ({norm_heavy}) should be smaller than light-mass ({norm_light})"
    );
}

#[test]
fn hasenbusch_cg_converges_faster() {
    let lat = Lattice::hot_start([4, 4, 4, 4], 5.5, 42);
    let vol = lat.volume();
    let mut seed = 44u64;

    let phi = hasenbusch_ratio_heatbath(&lat, 0.4, &mut seed);

    let cg_tol = 1e-6;
    let cg_max = 500;

    let mut x_heavy = FermionField::zeros(vol);
    let mut x_light = FermionField::zeros(vol);

    let res_heavy = cg_solve(&lat, &mut x_heavy, &phi, 0.4, cg_tol, cg_max);
    let res_light = cg_solve(&lat, &mut x_light, &phi, 0.1, cg_tol, cg_max);

    assert!(
        res_heavy.converged,
        "CG at heavy mass should converge: {} iters",
        res_heavy.iterations
    );
    assert!(
        res_light.converged,
        "CG at light mass should converge: {} iters",
        res_light.iterations
    );
    assert!(
        res_heavy.iterations < res_light.iterations,
        "CG at m_heavy ({}) should need fewer iterations than at m_light ({})",
        res_heavy.iterations,
        res_light.iterations
    );
}

#[test]
fn hasenbusch_hmc_runs() {
    let mut lat = Lattice::cold_start([4, 4, 4, 4], 5.5);
    let mut config = HasenbuschHmcConfig {
        dt: 0.02,
        seed: 42,
        hasenbusch: HasenbuschConfig {
            heavy_mass: 0.4,
            light_mass: 0.1,
            cg_tol: 1e-6,
            cg_max_iter: 500,
            n_md_steps_light: 4,
            n_md_steps_heavy: 2,
        },
        beta: 5.5,
    };

    let result = hasenbusch_hmc_trajectory(&mut lat, &mut config);

    assert!(
        result.plaquette > 0.0 && result.plaquette <= 1.0,
        "plaquette should be in (0,1]: {}",
        result.plaquette
    );
    assert!(
        result.cg_iterations_heavy > 0 || result.cg_iterations_ratio > 0,
        "should need CG iterations"
    );
    assert!(
        result.cg_iterations_heavy < result.cg_iterations_ratio,
        "heavy sector ({}) should use fewer CG iters than ratio ({})",
        result.cg_iterations_heavy,
        result.cg_iterations_ratio
    );
}
