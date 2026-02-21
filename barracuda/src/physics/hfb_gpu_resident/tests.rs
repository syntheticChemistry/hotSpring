// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the GPU-resident spherical HFB solver.

use super::super::hfb_gpu_types::{HamiltonianDimsUniform, PotentialDimsUniform};
use super::types::GpuResidentL2Result;
use crate::tolerances::DENSITY_FLOOR;

#[test]
fn gpu_resident_result_fields() {
    let r = GpuResidentL2Result {
        results: vec![(28, 28, 483.0, true), (50, 82, 1095.0, true)],
        hfb_time_s: 2.5,
        gpu_dispatches: 10,
        total_gpu_dispatches: 70,
        n_hfb: 2,
        n_semf: 0,
    };
    assert_eq!(r.results.len(), 2);
    assert_eq!(r.n_hfb, 2);
    assert_eq!(r.total_gpu_dispatches, 70);
}

#[test]
fn potential_dims_uniform_layout() {
    let dims = PotentialDimsUniform {
        nr: 100,
        batch_size: 4,
    };
    assert_eq!(dims.nr, 100);
    assert_eq!(dims.batch_size, 4);
    let bytes = bytemuck::bytes_of(&dims);
    assert_eq!(bytes.len(), 8);
}

#[test]
fn hamiltonian_dims_uniform_layout() {
    let dims = HamiltonianDimsUniform {
        n_states: 14,
        nr: 100,
        batch_size: 4,
        _pad: 0,
    };
    assert_eq!(dims.n_states, 14);
    let bytes = bytemuck::bytes_of(&dims);
    assert_eq!(bytes.len(), 16);
}

#[test]
fn nucleus_grouping_by_ns_nr() {
    use crate::physics::hfb::SphericalHFB;
    let nuclei: Vec<(usize, usize)> = vec![(28, 28), (40, 50), (50, 82)];
    let hfb_nuclei: Vec<(usize, usize)> = nuclei
        .iter()
        .copied()
        .filter(|&(z, n)| (56..=132).contains(&(z + n)))
        .collect();

    let mut groups: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &(z, n)) in hfb_nuclei.iter().enumerate() {
        let hfb = SphericalHFB::new_adaptive(z, n);
        groups
            .entry((hfb.n_states(), hfb.nr()))
            .or_default()
            .push(i);
    }

    assert!(!groups.is_empty(), "should have at least one group");
    let total: usize = groups.values().map(std::vec::Vec::len).sum();
    assert_eq!(total, hfb_nuclei.len(), "all nuclei should be grouped");
}

#[test]
fn density_initialization_positive() {
    let z = 28;
    let n = 28;
    let a = z + n;
    let r_nuc = 1.2 * f64::from(a).powf(1.0 / 3.0);
    let rho0 = 3.0 * f64::from(a) / (4.0 * std::f64::consts::PI * r_nuc.powi(3));
    let nr = 100;
    let dr = 15.0 / nr as f64;

    let rho_p: Vec<f64> = (0..nr)
        .map(|k| {
            let r = (k + 1) as f64 * dr;
            if r < r_nuc {
                (rho0 * f64::from(z) / f64::from(a)).max(DENSITY_FLOOR)
            } else {
                DENSITY_FLOOR
            }
        })
        .collect();

    assert!(
        rho_p.iter().all(|&v| v > 0.0),
        "all densities must be positive"
    );
    assert!(rho_p[0] > 1e-3, "inner density should be substantial");
    assert!(rho_p[nr - 1] < 1e-10, "outer density should be near zero");
}

#[test]
#[allow(clippy::float_cmp)]
fn density_floor_applied() {
    let rho = -0.001_f64;
    let floored = rho.max(DENSITY_FLOOR);
    assert_eq!(floored, DENSITY_FLOOR);
    assert!(floored > 0.0);
}

#[test]
#[allow(clippy::float_cmp)]
fn spin_orbit_r_min_guard() {
    use crate::tolerances::SPIN_ORBIT_R_MIN;
    let r = 0.001_f64;
    let guarded = r.max(SPIN_ORBIT_R_MIN);
    assert_eq!(guarded, SPIN_ORBIT_R_MIN);
    assert!(guarded >= 0.1);
}
