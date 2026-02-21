// SPDX-License-Identifier: AGPL-3.0-only

#![allow(clippy::expect_used)]

use super::*;
use barracuda::numerical::trapz;
use std::f64::consts::PI;

fn sly4_params() -> Vec<f64> {
    crate::provenance::SLY4_PARAMS.to_vec()
}

#[test]
fn radial_grid_generation() {
    let hfb = SphericalHFB::new(28, 28, 8, 15.0, 120);
    let r = hfb.r_grid();
    let dr = hfb.dr();

    assert_eq!(r.len(), 120);
    assert!((dr - 15.0 / 120.0).abs() < 1e-10);
    assert!((r[0] - dr).abs() < 1e-10);
    assert!((r[r.len() - 1] - 15.0).abs() < 1e-10);

    for i in 1..r.len() {
        let spacing = r[i] - r[i - 1];
        assert!(
            (spacing - dr).abs() < 1e-8,
            "grid should be uniform: r[{}] - r[{}] = {}",
            i,
            i - 1,
            spacing
        );
    }
}

#[test]
fn harmonic_oscillator_wavefunction_normalization() {
    let hfb = SphericalHFB::new(28, 28, 8, 15.0, 100);
    let r = hfb.r_grid();

    for (i, wf) in hfb.wf_flat().chunks(hfb.nr()).enumerate().take(3) {
        let integrand: Vec<f64> = r
            .iter()
            .zip(wf.iter())
            .map(|(&ri, &wi)| wi.powi(2) * ri.powi(2))
            .collect();
        let norm_sq = trapz(&integrand, r).unwrap_or(0.0);
        assert!(
            (norm_sq - 1.0).abs() < 0.05,
            "state {i}: ∫ R² r² dr = {norm_sq} (expect ~1)"
        );
    }
}

#[test]
fn density_from_eigenstates_single_state() {
    let hfb = SphericalHFB::new(8, 8, 6, 12.0, 60);
    let ns = hfb.n_states();
    let r = hfb.r_grid();

    let mut eigvecs = vec![0.0; ns * ns];
    for i in 0..ns {
        eigvecs[i * ns + i] = 1.0;
    }
    let mut v2 = vec![0.0; ns];
    v2[0] = 1.0;

    let rho = hfb.density_from_eigenstates(&eigvecs, &v2, ns);
    assert_eq!(rho.len(), 60);
    assert!(rho.iter().all(|&x| x >= 1e-15));
    let deg0 = hfb.deg_values()[0];
    let integrand: Vec<f64> = r
        .iter()
        .zip(rho.iter())
        .map(|(&ri, &rhi)| rhi * 4.0 * PI * ri.powi(2))
        .collect();
    let n_total: f64 = trapz(&integrand, r).unwrap_or(0.0);
    assert!(
        (n_total - deg0).abs() < 0.5,
        "single occupied state: integral ~{n_total} (deg={deg0})"
    );
}

#[test]
fn binding_energy_l2_semf_light_nucleus() {
    let params = sly4_params();
    let (b, conv) = binding_energy_l2(8, 8, &params).expect("HFB solve");
    assert!(conv, "SEMF path should always converge");
    assert!(b > 0.0 && b < 200.0, "O-16 binding ~{b} MeV");
}

#[test]
#[ignore = "HFB solve takes > 1s"]
fn hfb_full_solve_ni56() {
    let params = sly4_params();
    let hfb = SphericalHFB::new_adaptive(28, 28);
    let result = hfb.solve(&params, 200, 0.05, 0.3).expect("HFB solve");
    assert!(result.binding_energy_mev > 400.0);
    assert!(result.converged);
}

#[test]
fn adaptive_constructor_scales_with_mass() {
    let light = SphericalHFB::new_adaptive(8, 8);
    let medium = SphericalHFB::new_adaptive(28, 28);
    let heavy = SphericalHFB::new_adaptive(50, 82);

    assert!(
        light.n_states() <= medium.n_states(),
        "O-16 should have fewer states than Ni-56"
    );
    assert!(
        medium.n_states() <= heavy.n_states(),
        "Ni-56 should have fewer states than Sn-132"
    );
    assert!(
        light.nr() <= medium.nr(),
        "lighter nucleus needs fewer grid points"
    );
}

#[test]
fn build_hamiltonian_is_symmetric() {
    let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
    let ns = hfb.n_states();
    let nr = hfb.nr();
    let rho = vec![0.01; nr];
    let params = sly4_params();

    let w0 = params[9];
    let h = hfb.build_hamiltonian(&rho, &rho, true, &params, w0);
    assert_eq!(h.len(), ns * ns);

    for i in 0..ns {
        for j in i + 1..ns {
            let diff = (h[i * ns + j] - h[j * ns + i]).abs();
            assert!(
                diff < 1e-10,
                "H[{i},{j}]={} != H[{j},{i}]={}",
                h[i * ns + j],
                h[j * ns + i]
            );
        }
    }
}

#[test]
fn bcs_occupations_sum_constraint() {
    let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
    let ns = hfb.n_states();
    let eigs: Vec<f64> = (0..ns).map(|i| -20.0 + 5.0 * i as f64).collect();

    let (v2, _lambda) = hfb.bcs_occupations(&eigs, 8, 12.0 / (8.0_f64 + 8.0).sqrt());
    let degs = hfb.deg_values();
    let n_total: f64 = degs.iter().zip(v2.iter()).map(|(d, v)| d * v).sum();
    assert!(
        (n_total - 8.0).abs() < 1.0,
        "BCS particle number = {n_total}, expected ~8"
    );
}

#[test]
fn quantum_numbers_have_correct_degeneracy() {
    let hfb = SphericalHFB::new(20, 20, 6, 12.0, 60);
    let degs = hfb.deg_values();
    let lj = hfb.lj_quantum_numbers();

    assert_eq!(degs.len(), lj.len());
    for (i, &(l, j)) in lj.iter().enumerate() {
        let expected_deg = 2.0 * j + 1.0;
        assert!(
            (degs[i] - expected_deg).abs() < 1e-10,
            "state {i}: l={l}, j={j}, deg={}, expected {}",
            degs[i],
            expected_deg
        );
    }
}

#[test]
#[allow(clippy::float_cmp)]
fn wavefunction_accessor_consistency() {
    let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
    let ns = hfb.n_states();
    let nr = hfb.nr();
    let flat = hfb.wf_flat();

    assert_eq!(flat.len(), ns * nr);
    for i in 0..ns {
        let state = hfb.wf_state(i);
        assert_eq!(state.len(), nr);
        for k in 0..nr {
            assert_eq!(
                flat[i * nr + k],
                state[k],
                "flat[{i}*{nr}+{k}] != state[{i}][{k}]"
            );
        }
    }
}

#[test]
fn lj_same_flag_consistency() {
    let hfb = SphericalHFB::new(20, 20, 6, 12.0, 60);
    let ns = hfb.n_states();
    let lj = hfb.lj_quantum_numbers();
    let lj_same = hfb.lj_same_flat();

    assert_eq!(lj_same.len(), ns * ns);
    for i in 0..ns {
        for j in 0..ns {
            let expected = u32::from(lj[i] == lj[j]);
            assert_eq!(lj_same[i * ns + j], expected, "lj_same[{i},{j}] mismatch");
        }
    }
}

#[test]
fn binding_energy_l2_determinism() {
    let params = crate::provenance::SLY4_PARAMS;
    let run = || binding_energy_l2(28, 28, &params).expect("HFB solve");
    let a = run();
    let b = run();
    assert_eq!(
        a.0.to_bits(),
        b.0.to_bits(),
        "L2 binding energy not deterministic"
    );
}

#[test]
fn energy_from_densities_finite_for_nucleus() {
    let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
    let ns = hfb.n_states();
    let nr = hfb.nr();
    let rho = vec![0.01; nr];
    let mut evecs = vec![0.0; ns * ns];
    for i in 0..ns {
        evecs[i * ns + i] = 1.0;
    }
    let evals: Vec<f64> = (0..ns).map(|i| -20.0 + i as f64 * 2.0).collect();
    let params = sly4_params();
    let e = hfb.compute_energy_from_densities(&rho, &rho, &evals, &evecs, &evals, &evecs, &params);
    assert!(e.is_finite(), "energy must be finite, got {e}");
}

#[test]
fn energy_with_v2_finite() {
    let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
    let ns = hfb.n_states();
    let nr = hfb.nr();
    let rho = vec![0.01; nr];
    let mut evecs = vec![0.0; ns * ns];
    for i in 0..ns {
        evecs[i * ns + i] = 1.0;
    }
    let evals: Vec<f64> = (0..ns).map(|i| -20.0 + i as f64 * 2.0).collect();
    let v2 = vec![0.5; ns];
    let params = sly4_params();
    let e = hfb.compute_energy_with_v2(
        &rho, &rho, &evals, &evecs, &evals, &evecs, &v2, &v2, &params,
    );
    assert!(e.is_finite(), "energy_with_v2 must be finite, got {e}");
}

#[test]
fn pairing_gap_scales_with_mass() {
    let light = SphericalHFB::new_adaptive(8, 8);
    let heavy = SphericalHFB::new_adaptive(50, 82);
    assert!(
        light.pairing_gap() > heavy.pairing_gap(),
        "pairing gap should decrease with mass: light={}, heavy={}",
        light.pairing_gap(),
        heavy.pairing_gap()
    );
}

#[test]
fn hw_oscillator_frequency_positive() {
    let hfb = SphericalHFB::new(20, 20, 6, 12.0, 60);
    assert!(hfb.hw() > 0.0, "oscillator frequency must be positive");
    assert!(hfb.hw() < 50.0, "oscillator frequency unreasonably large");
}

#[test]
fn density_floor_enforced() {
    let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
    let ns = hfb.n_states();
    let evecs = vec![0.0; ns * ns];
    let v2 = vec![0.0; ns];
    let rho = hfb.density_from_eigenstates(&evecs, &v2, ns);
    assert!(
        rho.iter().all(|&x| x >= crate::tolerances::DENSITY_FLOOR),
        "density must be >= DENSITY_FLOOR everywhere"
    );
}

#[test]
fn wf_flat_and_dwf_flat_dimensions() {
    let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
    let ns = hfb.n_states();
    let nr = hfb.nr();
    assert_eq!(hfb.wf_flat().len(), ns * nr);
    assert_eq!(hfb.dwf_flat().len(), ns * nr);
}
