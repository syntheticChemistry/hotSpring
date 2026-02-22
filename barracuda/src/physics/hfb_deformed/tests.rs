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
#[allow(clippy::float_cmp)]
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

// ═══════════════════════════════════════════════════════════════════
// Basis.rs coverage: basis state enumeration, wavefunctions, quantum numbers
// ═══════════════════════════════════════════════════════════════════

#[test]
fn basis_small_shell_closure_counts() {
    let solver = DeformedHFB::new_test_minimal(2, 12, 12);
    assert!(
        solver.states.len() >= 4,
        "n_shells=2 should yield at least a few states"
    );
    assert!(!solver.omega_blocks.is_empty());
    let total_in_blocks: usize = solver.omega_blocks.values().map(Vec::len).sum();
    assert_eq!(
        total_in_blocks,
        solver.states.len(),
        "omega blocks should partition all states"
    );
}

#[test]
#[allow(clippy::used_underscore_binding)]
fn basis_quantum_number_constraints() {
    let solver = DeformedHFB::new_adaptive(8, 8);
    for s in &solver.states {
        assert!(
            s.omega_x2 > 0,
            "omega_x2 must be positive, got {}",
            s.omega_x2
        );
        assert_eq!(
            s.omega_x2,
            2 * s.lambda + s.sigma,
            "omega_x2 = 2*Lambda + sigma"
        );
        assert!(s.sigma == 1 || s.sigma == -1, "sigma must be ±1");
        let parity = if (s.n_z + 2 * s.n_perp + s.lambda.unsigned_abs() as usize).is_multiple_of(2)
        {
            1
        } else {
            -1
        };
        assert_eq!(
            s._parity, parity,
            "parity = (-1)^(n_z + 2*n_perp + |Lambda|)"
        );
    }
}

#[test]
fn basis_n0_state_at_origin_maximum() {
    let solver = DeformedHFB::new_test_minimal(3, 16, 16);
    let n0_state = solver.states.iter().find(|s| s.n_z == 0 && s.n_perp == 0);
    let n0 = n0_state.expect("n_z=0, n_perp=0 ground state must exist");
    let psi = solver.evaluate_wavefunction(n0);
    let origin_idx = solver.grid.idx(0, solver.grid.n_z / 2);
    let origin_val = psi[origin_idx];
    for &v in &psi {
        assert!(
            v.abs() <= origin_val.abs() + 1e-10,
            "n=0 state peaks at origin"
        );
    }
}

#[test]
fn wavefunction_large_radius_decays() {
    let solver = DeformedHFB::new_test_minimal(2, 20, 20);
    let state = &solver.states[0];
    let psi = solver.evaluate_wavefunction(state);
    let near_origin = psi[solver.grid.idx(1, solver.grid.n_z / 2)];
    let far_idx = solver.grid.idx(solver.grid.n_rho - 1, solver.grid.n_z - 1);
    let far_val = psi[far_idx];
    assert!(
        far_val.abs() < near_origin.abs() + 1e-10,
        "wavefunction should decay at large radius"
    );
}

#[test]
fn hermite_oscillator_n0_at_origin() {
    let b = 2.0;
    let val = DeformedHFB::hermite_oscillator(0, 0.0, b);
    let expected = 1.0 / (b * std::f64::consts::PI.sqrt()).sqrt();
    assert!(
        (val - expected).abs() < 1e-12,
        "H_0(0)*exp(0) with norm: got {val}, expected {expected}"
    );
}

#[test]
fn hermite_oscillator_n1_zero_at_origin() {
    let b = 2.0;
    let val = DeformedHFB::hermite_oscillator(1, 0.0, b);
    assert!(val.abs() < 1e-15, "H_1(0)=0 so wavefunction is 0 at origin");
}

#[test]
fn laguerre_oscillator_lambda_zero_at_origin() {
    let b = 2.0;
    let val = DeformedHFB::laguerre_oscillator(0, 0, 0.01, b);
    assert!(
        val.is_finite() && val > 0.0,
        "n=0, |Lambda|=0 nonzero at small rho"
    );
}

#[test]
fn laguerre_oscillator_lambda_nonzero_vanishes_at_origin() {
    let b = 2.0;
    let val = DeformedHFB::laguerre_oscillator(0, 1, 0.0, b);
    assert!(val.abs() < 1e-15, "rho^|Lambda| = 0 at rho=0 for Lambda≠0");
}

#[test]
fn renormalize_wavefunctions_unit_norm() {
    let solver = DeformedHFB::new_test_minimal(2, 12, 12);
    let mut wavefunctions: Vec<Vec<f64>> = solver
        .states
        .iter()
        .map(|s| solver.evaluate_wavefunction(s))
        .collect();
    solver.renormalize_wavefunctions(&mut wavefunctions);
    for (i, psi) in wavefunctions.iter().enumerate() {
        let norm2: f64 = (0..solver.grid.total())
            .map(|k| {
                let i_rho = k / solver.grid.n_z;
                let i_z = k % solver.grid.n_z;
                psi[k] * psi[k] * solver.grid.volume_element(i_rho, i_z)
            })
            .sum();
        assert!(
            (norm2 - 1.0).abs() < 1e-8,
            "state {i}: renormalized norm² = {norm2}, expected 1"
        );
    }
}

#[test]
fn basis_maximum_angular_momentum_in_shells() {
    let solver = DeformedHFB::new_test_minimal(4, 12, 12);
    let max_lambda = solver.states.iter().map(|s| s.lambda.unsigned_abs()).max();
    assert!(max_lambda.is_some());
    let max_l = max_lambda.unwrap();
    assert!(max_l <= 4, "max |Lambda| in n_shells=4 should be ≤ 4");
}

// ═══════════════════════════════════════════════════════════════════
// Mod.rs coverage: configuration, grid, helpers, density/occupation logic
// ═══════════════════════════════════════════════════════════════════

#[test]
fn new_adaptive_grid_physical_bounds() {
    let solver = DeformedHFB::new_adaptive(8, 8);
    assert!(solver.grid.n_rho >= 60);
    assert!(solver.grid.n_z >= 80);
    assert!(solver.grid.rho[0] > 0.0);
    assert!(solver.grid.z[0] < 0.0);
}

#[test]
fn new_adaptive_oscillator_lengths_positive() {
    let solver = DeformedHFB::new_adaptive(8, 8);
    assert!(solver.b_z > 0.0 && solver.b_perp > 0.0);
    assert!(solver.hw_z > 0.0 && solver.hw_perp > 0.0);
}

#[test]
fn deformed_state_omega() {
    let s = DeformedState {
        n_z: 0,
        n_perp: 0,
        lambda: 1,
        sigma: 1,
        omega_x2: 3,
        _parity: 1,
        _n_shell: 1,
    };
    assert!((s.omega() - 1.5).abs() < 1e-12);
}

#[test]
fn compute_densities_zero_occupation_zero_density() {
    let solver = DeformedHFB::new_test_minimal(2, 12, 12);
    let wavefunctions: Vec<Vec<f64>> = solver
        .states
        .iter()
        .map(|s| solver.evaluate_wavefunction(s))
        .collect();
    let occ: Vec<f64> = vec![0.0; solver.states.len()];
    let (rho_p, rho_n) = solver.compute_densities(&wavefunctions, &occ, &occ);
    assert!(rho_p.iter().all(|&x| x.abs() < 1e-20));
    assert!(rho_n.iter().all(|&x| x.abs() < 1e-20));
}

#[test]
fn compute_densities_single_state_positive() {
    let solver = DeformedHFB::new_test_minimal(2, 12, 12);
    let mut wavefunctions: Vec<Vec<f64>> = solver
        .states
        .iter()
        .map(|s| solver.evaluate_wavefunction(s))
        .collect();
    solver.renormalize_wavefunctions(&mut wavefunctions);
    let mut occ_p = vec![0.0; solver.states.len()];
    let mut occ_n = vec![0.0; solver.states.len()];
    occ_p[0] = 1.0;
    occ_n[0] = 0.5;
    let (rho_p, rho_n) = solver.compute_densities(&wavefunctions, &occ_p, &occ_n);
    assert!(rho_p.iter().all(|&x| x >= 0.0));
    assert!(rho_n.iter().all(|&x| x >= 0.0));
    let integral_p: f64 = rho_p
        .iter()
        .enumerate()
        .map(|(k, &r)| {
            r * solver
                .grid
                .volume_element(k / solver.grid.n_z, k % solver.grid.n_z)
        })
        .sum();
    assert!(
        integral_p > 0.5 && integral_p < 4.0,
        "occupation 1 → ~2 particles"
    );
}

#[test]
fn potential_matrix_element_orthogonality_self_overlap() {
    let solver = DeformedHFB::new_test_minimal(2, 12, 12);
    let mut wavefunctions: Vec<Vec<f64>> = solver
        .states
        .iter()
        .map(|s| solver.evaluate_wavefunction(s))
        .collect();
    solver.renormalize_wavefunctions(&mut wavefunctions);
    let v_zero = vec![0.0; solver.grid.total()];
    let me = solver.potential_matrix_element(&wavefunctions[0], &wavefunctions[0], &v_zero);
    assert!(me.abs() < 1e-12, "V=0 => <i|V|i> = 0");
}

#[test]
fn find_fermi_bcs_nofield_fills_lowest() {
    let eigs: Vec<(usize, f64)> = vec![(0, 0.0), (1, 1.0), (2, 2.0)];
    let mu = DeformedHFB::find_fermi_bcs(&eigs, 4, 0.001);
    assert!(mu.is_finite());
    let n: f64 = eigs
        .iter()
        .map(|&(_, e)| {
            let eps = e - mu;
            let eq = (eps * eps + 0.0001_f64).sqrt();
            2.0 * 0.5 * (1.0 - eps / eq)
        })
        .sum();
    assert!((n - 4.0).abs() < 0.1, "N=4 particles, got N={n}");
}

#[test]
fn cylindric_grid_d_rho_d_z_consistent() {
    let grid = CylindricalGrid::new(10.0, 8.0, 20, 16);
    assert!((grid.d_rho - 10.0 / 20.0).abs() < 1e-15);
    assert!((grid.d_z - 16.0 / 16.0).abs() < 1e-15);
}

#[test]
fn deformed_hfb_result_fields_accessible() {
    let _ = DeformedHFBResult {
        binding_energy_mev: -100.0,
        converged: true,
        iterations: 10,
        delta_e: 0.001,
        beta2: 0.1,
        q20_fm2: 50.0,
        rms_radius_fm: 3.0,
    };
}

// ═══════════════════════════════════════════════════════════════════
// diagonalize_blocks: eigenvalues, occupations, BCS vs sharp Fermi
// ═══════════════════════════════════════════════════════════════════

#[test]
fn diagonalize_blocks_v_zero_gives_ho_eigenvalues() {
    let solver = DeformedHFB::new_test_minimal(3, 12, 12);
    let mut wavefunctions: Vec<Vec<f64>> = solver
        .states
        .iter()
        .map(|s| solver.evaluate_wavefunction(s))
        .collect();
    solver.renormalize_wavefunctions(&mut wavefunctions);

    let v_zero = vec![0.0; solver.grid.total()];
    let (eigs, occ) = solver
        .diagonalize_blocks(&v_zero, &wavefunctions, 4, solver.delta_p)
        .expect("diagonalize_blocks with V=0");

    assert_eq!(eigs.len(), solver.states.len());
    assert_eq!(occ.len(), solver.states.len());
    for &e in &eigs {
        assert!(e.is_finite(), "eigenvalue must be finite");
        assert!(e > 0.0, "HO eigenvalue must be positive for V=0, got {e}");
    }
    let occ_sum: f64 = occ.iter().map(|&v| 2.0 * v).sum();
    assert!(
        (occ_sum - 4.0).abs() < 1.0,
        "occupation sum (degeneracy=2) ≈ 4 particles, got {occ_sum}"
    );
}

#[test]
fn diagonalize_blocks_sharp_fermi_with_zero_pairing() {
    let solver = DeformedHFB::new_test_minimal(3, 12, 12);
    let mut wavefunctions: Vec<Vec<f64>> = solver
        .states
        .iter()
        .map(|s| solver.evaluate_wavefunction(s))
        .collect();
    solver.renormalize_wavefunctions(&mut wavefunctions);

    let v_zero = vec![0.0; solver.grid.total()];
    let delta_zero = 1e-15; // below PAIRING_GAP_THRESHOLD
    let (eigs, occ) = solver
        .diagonalize_blocks(&v_zero, &wavefunctions, 4, delta_zero)
        .expect("diagonalize_blocks with zero pairing");

    assert!(eigs.iter().all(|e| e.is_finite()));
    let filled: usize = occ.iter().filter(|&&v| v > 0.5).count();
    assert!(filled >= 1, "at least one state filled for 4 particles");
    let n_total: f64 = occ.iter().map(|&v| 2.0 * v).sum();
    assert!(
        (n_total - 4.0).abs() < 0.1,
        "sharp Fermi fills exactly 4 particles, got {n_total}"
    );
}

#[test]
fn diagonalize_blocks_constant_potential_shifts_eigenvalues() {
    let solver = DeformedHFB::new_test_minimal(3, 12, 12);
    let mut wavefunctions: Vec<Vec<f64>> = solver
        .states
        .iter()
        .map(|s| solver.evaluate_wavefunction(s))
        .collect();
    solver.renormalize_wavefunctions(&mut wavefunctions);

    let v_zero = vec![0.0; solver.grid.total()];
    let v_const = vec![5.0; solver.grid.total()];
    let (eigs_0, _) = solver
        .diagonalize_blocks(&v_zero, &wavefunctions, 4, solver.delta_p)
        .expect("V=0 eigensolve");
    let (eigs_c, _) = solver
        .diagonalize_blocks(&v_const, &wavefunctions, 4, solver.delta_p)
        .expect("V=5 eigensolve");

    for (i, (&e0, &ec)) in eigs_0.iter().zip(&eigs_c).enumerate() {
        let shift = ec - e0;
        assert!(
            shift > 3.0 && shift < 7.0,
            "state {i}: V=5 shift = {shift}, expected ~5 (depends on basis overlap)"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// potential_matrix_element: constant V, symmetry
// ═══════════════════════════════════════════════════════════════════

#[test]
fn potential_matrix_element_constant_v_diagonal() {
    let solver = DeformedHFB::new_test_minimal(2, 12, 12);
    let mut wavefunctions: Vec<Vec<f64>> = solver
        .states
        .iter()
        .map(|s| solver.evaluate_wavefunction(s))
        .collect();
    solver.renormalize_wavefunctions(&mut wavefunctions);

    let v_const = vec![3.0; solver.grid.total()];
    let me = solver.potential_matrix_element(&wavefunctions[0], &wavefunctions[0], &v_const);
    assert!(
        (me - 3.0).abs() < 0.5,
        "<0|V=3|0> ≈ 3 for normalized state, got {me}"
    );
}

#[test]
fn potential_matrix_element_hermitian_symmetry() {
    let solver = DeformedHFB::new_test_minimal(3, 12, 12);
    let mut wavefunctions: Vec<Vec<f64>> = solver
        .states
        .iter()
        .map(|s| solver.evaluate_wavefunction(s))
        .collect();
    solver.renormalize_wavefunctions(&mut wavefunctions);

    let v: Vec<f64> = (0..solver.grid.total())
        .map(|k| {
            let i_rho = k / solver.grid.n_z;
            let r = solver.grid.rho[i_rho];
            r * r
        })
        .collect();

    if wavefunctions.len() >= 2 {
        let me_ij = solver.potential_matrix_element(&wavefunctions[0], &wavefunctions[1], &v);
        let me_ji = solver.potential_matrix_element(&wavefunctions[1], &wavefunctions[0], &v);
        assert!(
            (me_ij - me_ji).abs() < 1e-12,
            "<i|V|j> = <j|V|i> not satisfied: {me_ij} vs {me_ji}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// solve(): SCF loop smoke test (minimal grid, covers Broyden mixing)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn solve_smoke_test_minimal_grid() {
    let mut solver = DeformedHFB::new_test_minimal(2, 8, 8);
    let result = solver.solve(&SLY4_PARAMS).expect("solve should not panic");

    assert!(result.iterations > 0, "at least one iteration");
    assert!(
        result.binding_energy_mev.is_finite(),
        "energy must be finite"
    );
    assert!(result.beta2.is_finite(), "deformation must be finite");
    assert!(
        result.rms_radius_fm.is_finite(),
        "RMS radius must be finite"
    );
    assert!(result.delta_e.is_finite(), "delta_e must be finite");
}

#[test]
fn solve_returns_physical_result() {
    let mut solver = DeformedHFB::new_test_minimal(3, 10, 10);
    let result = solver.solve(&SLY4_PARAMS).expect("solve should not panic");

    assert!(
        result.rms_radius_fm > 0.0 && result.rms_radius_fm < 100.0,
        "RMS radius should be physical: {}",
        result.rms_radius_fm
    );
    assert!(
        result.beta2.abs() < 5.0,
        "deformation should be bounded: {}",
        result.beta2
    );
}

#[test]
fn solve_deterministic_on_same_input() {
    let run = || {
        let mut solver = DeformedHFB::new_test_minimal(2, 8, 8);
        solver.solve(&SLY4_PARAMS).expect("solve")
    };
    let a = run();
    let b = run();
    assert_eq!(
        a.binding_energy_mev.to_bits(),
        b.binding_energy_mev.to_bits(),
        "solve must be deterministic"
    );
    assert_eq!(a.iterations, b.iterations);
}

// ═══════════════════════════════════════════════════════════════════
// binding_energy_l3: public API smoke test
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "Heavy computation (~90s); run with: cargo test -- --ignored binding_energy_l3_returns_finite"]
fn binding_energy_l3_returns_finite() {
    let (be, _conv, beta2) = binding_energy_l3(2, 2, &SLY4_PARAMS).expect("He-4 L3");
    assert!(be.is_finite(), "He-4 binding energy must be finite: {be}");
    assert!(beta2.is_finite(), "He-4 beta2 must be finite: {beta2}");
}

// ═══════════════════════════════════════════════════════════════════
// Hermite/Laguerre norm integrals
// ═══════════════════════════════════════════════════════════════════

#[test]
fn hermite_oscillator_norm_integral() {
    let b = 2.0;
    for n in 0..3 {
        let n_pts = 500;
        let z_max = 15.0 * b;
        let dz = 2.0 * z_max / n_pts as f64;
        let norm2: f64 = (0..n_pts)
            .map(|i| {
                let z = -z_max + (i as f64 + 0.5) * dz;
                let phi = DeformedHFB::hermite_oscillator(n, z, b);
                phi * phi * dz
            })
            .sum();
        assert!(
            (norm2 - 1.0).abs() < 0.01,
            "H_{n} norm integral = {norm2}, expected 1"
        );
    }
}

#[test]
fn laguerre_oscillator_norm_integral_lambda0() {
    let b = 2.0;
    let n_pts = 500;
    let rho_max = 15.0 * b;
    let d_rho = rho_max / n_pts as f64;
    for n_perp in 0..2 {
        let norm2: f64 = (1..=n_pts)
            .map(|i| {
                let rho = i as f64 * d_rho;
                let phi = DeformedHFB::laguerre_oscillator(n_perp, 0, rho, b);
                phi * phi * 2.0 * std::f64::consts::PI * rho * d_rho
            })
            .sum();
        assert!(
            (norm2 - 1.0).abs() < 0.05,
            "L_{{n_perp,0}} norm = {norm2}, expected 1"
        );
    }
}
