// SPDX-License-Identifier: AGPL-3.0-only

//! Integration tests: physics pipeline end-to-end validation.
//!
//! These tests exercise the full pipeline from parameter input to binding
//! energy output, verifying that the public API composes correctly across
//! module boundaries.

use hotspring_barracuda::physics::{binding_energy_l2, semf_binding_energy, SphericalHFB};
use hotspring_barracuda::provenance::SLY4_PARAMS;
use hotspring_barracuda::tolerances;

#[test]
fn semf_binding_energy_oxygen16_positive() {
    let b = semf_binding_energy(8, 8, &SLY4_PARAMS);
    assert!(b > 100.0, "O-16 SEMF should be > 100 MeV, got {b}");
    assert!(b < 200.0, "O-16 SEMF should be < 200 MeV, got {b}");
}

#[test]
fn semf_binding_energy_lead208_scales() {
    let b_o = semf_binding_energy(8, 8, &SLY4_PARAMS);
    let b_pb = semf_binding_energy(82, 126, &SLY4_PARAMS);
    assert!(
        b_pb > b_o,
        "Pb-208 ({b_pb} MeV) should have more BE than O-16 ({b_o} MeV)"
    );
}

#[test]
fn binding_energy_l2_light_nucleus_uses_semf() {
    let (b, conv) = binding_energy_l2(8, 8, &SLY4_PARAMS).expect("L2 solve");
    assert!(conv, "light nucleus should converge via SEMF path");
    let b_semf = semf_binding_energy(8, 8, &SLY4_PARAMS);
    assert!(
        (b - b_semf).abs() < 1e-10,
        "A<56 should use SEMF: l2={b}, semf={b_semf}"
    );
}

#[test]
fn binding_energy_l2_medium_nucleus_uses_hfb() {
    let (b, _conv) = binding_energy_l2(28, 28, &SLY4_PARAMS).expect("L2 solve Ni-56");
    let b_semf = semf_binding_energy(28, 28, &SLY4_PARAMS);
    assert!(
        (b - b_semf).abs() > 1.0,
        "A=56 should use HFB, giving different result from SEMF"
    );
    assert!(b > 300.0, "Ni-56 should have BE > 300 MeV, got {b}");
}

#[test]
fn hfb_solver_respects_tolerance_ordering() {
    assert!(
        tolerances::EXACT_F64 < tolerances::ITERATIVE_F64,
        "exact < iterative"
    );
    assert!(
        tolerances::ITERATIVE_F64 < tolerances::GPU_VS_CPU_F64,
        "iterative < GPU vs CPU"
    );
}

#[test]
fn spherical_hfb_adaptive_round_trip() {
    let hfb = SphericalHFB::new_adaptive(20, 20);
    let ns = hfb.n_states();
    let nr = hfb.nr();

    assert!(ns > 0, "should have states");
    assert!(nr > 0, "should have grid points");

    let wf = hfb.wf_flat();
    assert_eq!(wf.len(), ns * nr, "wf_flat dimensions");

    let dwf = hfb.dwf_flat();
    assert_eq!(dwf.len(), ns * nr, "dwf_flat dimensions");

    let lj_same = hfb.lj_same_flat();
    assert_eq!(lj_same.len(), ns * ns, "lj_same dimensions");

    let ll1 = hfb.ll1_values();
    assert_eq!(ll1.len(), ns, "ll1 dimensions");
    assert!(ll1.iter().all(|&v| v >= 0.0), "l(l+1) must be non-negative");

    let degs = hfb.deg_values();
    assert_eq!(degs.len(), ns, "deg dimensions");
    assert!(degs.iter().all(|&d| d >= 1.0), "degeneracies must be >= 1");
}

#[test]
fn hamiltonian_build_round_trip() {
    let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
    let nr = hfb.nr();
    let ns = hfb.n_states();
    let rho = vec![0.01; nr];
    let h = hfb.build_hamiltonian(&rho, &rho, true, &SLY4_PARAMS, SLY4_PARAMS[9]);
    assert_eq!(h.len(), ns * ns, "Hamiltonian should be ns×ns");
    assert!(h.iter().all(|v| v.is_finite()), "all H elements finite");
}

#[test]
fn bcs_occupations_particle_conservation() {
    let hfb = SphericalHFB::new(20, 20, 6, 12.0, 60);
    let ns = hfb.n_states();
    let a = 40.0_f64;
    let delta = 12.0 / a.sqrt();
    let eigs: Vec<f64> = (0..ns).map(|i| -30.0 + 3.0 * i as f64).collect();

    let (v2, _lam) = hfb.bcs_occupations_from_eigs(&eigs, 20, delta);
    let degs = hfb.deg_values();
    let n_total: f64 = degs.iter().zip(v2.iter()).map(|(d, v)| d * v).sum();
    assert!(
        (n_total - 20.0).abs() < 2.0,
        "BCS particle number {n_total} should be near 20"
    );
}

#[test]
fn density_from_eigenstates_normalizes() {
    let hfb = SphericalHFB::new(8, 8, 6, 12.0, 60);
    let ns = hfb.n_states();

    let mut eigvecs = vec![0.0; ns * ns];
    for i in 0..ns {
        eigvecs[i * ns + i] = 1.0;
    }
    let v2 = vec![1.0; ns];
    let rho = hfb.density_from_eigenstates(&eigvecs, &v2, ns);

    assert_eq!(rho.len(), hfb.nr());
    assert!(rho.iter().all(|&x| x >= tolerances::DENSITY_FLOOR));
}

#[test]
fn energy_from_densities_finite() {
    let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
    let ns = hfb.n_states();
    let nr = hfb.nr();
    let rho = vec![0.01; nr];
    let mut evecs = vec![0.0; ns * ns];
    for i in 0..ns {
        evecs[i * ns + i] = 1.0;
    }
    let evals: Vec<f64> = (0..ns).map(|i| -20.0 + i as f64 * 2.0).collect();
    let e =
        hfb.compute_energy_from_densities(&rho, &rho, &evals, &evecs, &evals, &evecs, &SLY4_PARAMS);
    assert!(e.is_finite(), "energy must be finite, got {e}");
}

#[test]
fn binding_energy_l2_deterministic() {
    let run = || binding_energy_l2(28, 28, &SLY4_PARAMS).expect("L2 solve");
    let a = run();
    let b = run();
    assert_eq!(
        a.0.to_bits(),
        b.0.to_bits(),
        "L2 binding energy must be deterministic"
    );
}
