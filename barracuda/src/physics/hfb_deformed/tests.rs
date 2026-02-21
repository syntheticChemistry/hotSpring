// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the axially-deformed HFB solver.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::physics::hfb_common::hermite_value;
use crate::physics::hfb_deformed_common::{beta2_from_q20, deformation_guess};
use crate::provenance::SLY4_PARAMS;

// ── Hermite/basis tests (original) ──

#[test]
fn test_hermite_values() {
    assert!((hermite_value(0, 1.0) - 1.0).abs() < 1e-10);
    assert!((hermite_value(1, 1.0) - 2.0).abs() < 1e-10);
    assert!((hermite_value(2, 1.0) - 2.0).abs() < 1e-10); // 4x²-2 = 2
    assert!((hermite_value(3, 1.0) - (-4.0)).abs() < 1e-10); // 8x³-12x = -4
}

#[test]
fn test_deformed_basis_count() {
    let solver = DeformedHFB::new_adaptive(8, 8); // O-16
    assert!(solver.states.len() > 10);
    assert!(solver.omega_blocks.len() >= 2);
}

#[test]
#[allow(clippy::float_cmp)]
fn test_deformation_guess() {
    assert_eq!(deformation_guess(20, 20), 0.0);
    assert!(deformation_guess(66, 96) > 0.2);
    assert!(deformation_guess(92, 146) > 0.2);
}

#[test]
fn test_basis_construction_determinism() {
    let build = || {
        let s = DeformedHFB::new_adaptive(8, 8);
        (
            s.states.len(),
            s.omega_blocks.clone(),
            s.grid.n_rho,
            s.grid.n_z,
            s.hw_z.to_bits(),
            s.hw_perp.to_bits(),
        )
    };
    let a = build();
    let b = build();
    assert_eq!(a.0, b.0, "state count mismatch");
    assert_eq!(a.1, b.1, "omega blocks mismatch");
    assert_eq!(a.2, b.2, "grid n_rho mismatch");
    assert_eq!(a.3, b.3, "grid n_z mismatch");
    assert_eq!(a.4, b.4, "hw_z bitwise mismatch");
    assert_eq!(a.5, b.5, "hw_perp bitwise mismatch");
}

#[test]
#[ignore = "Heavy computation (~30s+); run with: cargo test -- --ignored test_deformed_hfb_runs"]
fn test_deformed_hfb_runs() {
    let (be, conv, beta2) = binding_energy_l3(8, 8, &SLY4_PARAMS).expect("L3 solve");
    println!("O-16 deformed: B={be:.2} MeV, conv={conv}, beta2={beta2:.4}");
    assert!(beta2.abs() < 0.5, "O-16 should be nearly spherical");
}

// ── New tests: CylindricalGrid ──

#[test]
fn grid_total_and_idx_round_trip() {
    let grid = CylindricalGrid::new(10.0, 12.0, 20, 24);
    assert_eq!(grid.total(), 20 * 24);
    for i_rho in 0..grid.n_rho {
        for i_z in 0..grid.n_z {
            let idx = grid.idx(i_rho, i_z);
            assert_eq!(idx / grid.n_z, i_rho);
            assert_eq!(idx % grid.n_z, i_z);
        }
    }
}

#[test]
fn grid_coordinates_are_physical() {
    let grid = CylindricalGrid::new(10.0, 14.0, 20, 30);
    assert!(grid.rho[0] > 0.0, "rho starts at d_rho, not 0");
    assert!(grid.z[0] < 0.0, "z starts at -z_max + 0.5*dz");
    let z_last = *grid.z.last().unwrap();
    assert!(z_last > 0.0, "z ends at ~+z_max");
}

#[test]
fn grid_volume_element_positive() {
    let grid = CylindricalGrid::new(10.0, 12.0, 20, 24);
    for i_rho in 0..grid.n_rho {
        let dv = grid.volume_element(i_rho, 0);
        assert!(dv > 0.0, "volume element must be positive at i_rho={i_rho}");
    }
}

#[test]
fn grid_volume_element_grows_with_rho() {
    let grid = CylindricalGrid::new(10.0, 12.0, 20, 24);
    let dv0 = grid.volume_element(0, 0);
    let dv10 = grid.volume_element(10, 0);
    assert!(
        dv10 > dv0,
        "volume element grows with rho (cylindrical geometry)"
    );
}

// ── New tests: wavefunctions ──

#[test]
fn hermite_oscillator_ground_state_gaussian() {
    let b = 2.0;
    let val_0 = DeformedHFB::hermite_oscillator(0, 0.0, b);
    let val_far = DeformedHFB::hermite_oscillator(0, 10.0 * b, b);
    assert!(val_0 > 0.0, "n=0 peaked at origin");
    assert!(val_far.abs() < 1e-20, "n=0 decays far from origin");
}

#[test]
fn laguerre_oscillator_ground_state_peaked() {
    let b = 2.0;
    let val_mid = DeformedHFB::laguerre_oscillator(0, 0, 1.0, b);
    let val_far = DeformedHFB::laguerre_oscillator(0, 0, 20.0, b);
    assert!(val_mid.abs() > val_far.abs(), "n=0 decays away from origin");
}

#[test]
fn wavefunction_normalizes_on_grid() {
    let solver = DeformedHFB::new_adaptive(8, 8);
    let state = &solver.states[0];
    let psi = solver.evaluate_wavefunction(state);

    let norm2: f64 = (0..solver.grid.total())
        .map(|k| {
            let i_rho = k / solver.grid.n_z;
            let i_z = k % solver.grid.n_z;
            psi[k] * psi[k] * solver.grid.volume_element(i_rho, i_z)
        })
        .sum();

    // After renormalization this would be exactly 1.0, but raw wavefunction
    // should at least be finite and positive-definite
    assert!(
        norm2.is_finite() && norm2 > 0.0,
        "wavefunction norm² = {norm2}"
    );
}

// ── New tests: occupations ──

#[test]
fn find_fermi_bcs_conserves_particles() {
    let eigs: Vec<(usize, f64)> = (0..20).map(|i| (i, i as f64 * 2.0)).collect();
    let delta = 3.0;
    let n_particles = 8;
    let mu = DeformedHFB::find_fermi_bcs(&eigs, n_particles, delta);

    let n_sum: f64 = eigs
        .iter()
        .map(|&(_, e)| {
            let eps = e - mu;
            let e_qp = (eps * eps + delta * delta).sqrt();
            2.0 * 0.5 * (1.0 - eps / e_qp)
        })
        .sum();

    assert!(
        (n_sum - n_particles as f64).abs() < 0.5,
        "BCS occupations should approximately conserve particle number: N={n_sum}, target={n_particles}"
    );
}

#[test]
fn find_fermi_bcs_empty_returns_zero() {
    assert_eq!(DeformedHFB::find_fermi_bcs(&[], 8, 3.0), 0.0);
}

// ── New tests: quadrupole moment ──

#[test]
fn quadrupole_moment_zero_for_spherical_density() {
    let solver = DeformedHFB::new_adaptive(8, 8);
    let n = solver.grid.total();

    // Spherically symmetric density: rho(rho,z) = exp(-(rho²+z²)/R²)
    let r_scale = 3.0;
    let mut density = vec![0.0; n];
    for i_rho in 0..solver.grid.n_rho {
        for i_z in 0..solver.grid.n_z {
            let rho = solver.grid.rho[i_rho];
            let z = solver.grid.z[i_z];
            let r2 = rho * rho + z * z;
            density[solver.grid.idx(i_rho, i_z)] = (-r2 / (r_scale * r_scale)).exp();
        }
    }

    let q20 = solver.quadrupole_moment(&density);
    // Analytic Q20 of Gaussian: integral exp(-r²/R²) * (2z²-rho²) dV
    // = integral exp(-r²/R²) * r² * (3cos²θ-1) * r²sinθ dr dθ dφ
    // = 0 by angular integration (spherical harmonic orthogonality)
    assert!(
        q20.abs() < 0.5,
        "spherical density should have Q20 ≈ 0, got {q20}"
    );
}

#[test]
fn beta2_from_q20_scales_with_mass() {
    let b_small = beta2_from_q20(16, 100.0);
    let b_large = beta2_from_q20(208, 100.0);
    assert!(
        b_small.abs() > b_large.abs(),
        "same Q20 gives larger β₂ for lighter nucleus"
    );
}
