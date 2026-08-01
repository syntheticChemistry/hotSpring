// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::*;

#[test]
fn test_single_gaussian_reconstruction() {
    let hills = Hills {
        centers: vec![1.5],
        sigmas: vec![0.1],
        heights: vec![1.0],
        biasfactor: 15.0,
        n_gaussians: 1,
    };

    let fes = reconstruct_fes(&hills, 0.0, std::f64::consts::PI, 100);
    assert_eq!(fes.nbins, 100);

    // Minimum should be near the center (1.5 rad)
    let min_idx = fes.free_energy.iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;
    let min_theta = fes.grid[min_idx];
    assert!((min_theta - 1.5).abs() < 0.05, "Min at {min_theta}, expected ~1.5");
}

#[test]
fn test_basin_detection() {
    // Synthetic FES with 3 basins
    let n = 100;
    let grid: Vec<f64> = (0..n).map(|i| i as f64 * std::f64::consts::PI / (n - 1) as f64).collect();
    let free_energy: Vec<f64> = grid.iter().map(|&x| {
        // Three basins at ~0.15, 1.57, 2.99 rad
        let v1 = 50.0 * (-(x - 0.15_f64).powi(2) / 0.02).exp();
        let v2 = 30.0 * (-(x - 1.57_f64).powi(2) / 0.02).exp();
        let v3 = 60.0 * (-(x - 2.99_f64).powi(2) / 0.02).exp();
        60.0 - v1 - v2 - v3
    }).collect();

    let fes = FesResult { grid, free_energy, nbins: n };
    let basins = find_basins(&fes);

    assert!(basins.len() >= 3, "Expected 3 basins, got {}", basins.len());
}

#[test]
fn test_parity_exact_match() {
    let fes = FesResult {
        grid: vec![0.0, 1.0, 2.0, 3.0],
        free_energy: vec![10.0, 0.0, 5.0, 8.0],
        nbins: 4,
    };
    let parity = check_parity(&fes, &fes, 1.0);
    assert_eq!(parity.status, "MATCH");
    assert!(parity.max_deviation_kjmol < 1e-10);
}

#[test]
fn test_2d_single_gaussian() {
    let hills = Hills2D {
        centers_x: vec![1.5],
        centers_y: vec![3.0],
        sigmas_x: vec![0.1],
        sigmas_y: vec![0.2],
        heights: vec![1.0],
        biasfactor: 15.0,
        n_gaussians: 1,
    };

    let fes = reconstruct_fes_2d(&hills, 0.0, std::f64::consts::PI, 0.0, 2.0 * std::f64::consts::PI, 50, 50, true);
    assert_eq!(fes.nbins_x, 50);
    assert_eq!(fes.nbins_y, 50);

    // Minimum should be near (1.5, 3.0)
    let mut min_val = f64::INFINITY;
    let mut min_ix = 0;
    let mut min_iy = 0;
    for i in 0..50 {
        for j in 0..50 {
            if fes.free_energy[i][j] < min_val {
                min_val = fes.free_energy[i][j];
                min_ix = i;
                min_iy = j;
            }
        }
    }
    assert!((fes.grid_x[min_ix] - 1.5).abs() < 0.15, "Min x at {}, expected ~1.5", fes.grid_x[min_ix]);
    assert!((fes.grid_y[min_iy] - 3.0).abs() < 0.3, "Min y at {}, expected ~3.0", fes.grid_y[min_iy]);
}

#[test]
fn test_2d_periodic_wrapping() {
    // Gaussian near y=0 should wrap around from 2*pi
    let hills = Hills2D {
        centers_x: vec![1.5],
        centers_y: vec![0.05],
        sigmas_x: vec![0.2],
        sigmas_y: vec![0.3],
        heights: vec![1.0],
        biasfactor: 15.0,
        n_gaussians: 1,
    };

    let two_pi = 2.0 * std::f64::consts::PI;
    let fes = reconstruct_fes_2d(&hills, 0.0, std::f64::consts::PI, 0.0, two_pi, 50, 50, true);

    // Check that energy near y = 2*pi - 0.05 is also affected (periodic wrap)
    let ix_center = 23; // ~1.5 rad
    let iy_end = 49; // near 2*pi
    let iy_start = 0; // near 0

    // Both should be low-energy (near the Gaussian center via wrapping)
    let e_start = fes.free_energy[ix_center][iy_start];
    let e_end = fes.free_energy[ix_center][iy_end];
    let e_far = fes.free_energy[ix_center][25]; // far from center in y
    assert!(e_start < e_far, "Periodic wrapping failed: start {} should be < far {}", e_start, e_far);
    assert!(e_end < e_far, "Periodic wrapping failed: end {} should be < far {}", e_end, e_far);
}

#[test]
fn test_2d_parity_self() {
    let hills = Hills2D {
        centers_x: vec![1.0, 2.0, 1.5],
        centers_y: vec![2.0, 4.0, 3.0],
        sigmas_x: vec![0.15, 0.15, 0.15],
        sigmas_y: vec![0.25, 0.25, 0.25],
        heights: vec![1.0, 0.8, 0.6],
        biasfactor: 15.0,
        n_gaussians: 3,
    };

    let fes = reconstruct_fes_2d(&hills, 0.0, std::f64::consts::PI, 0.0, 2.0 * std::f64::consts::PI, 40, 40, true);
    let parity = check_parity_2d(&fes, &fes, 1.0);
    assert_eq!(parity.status, "MATCH");
    assert!(parity.max_deviation_kjmol < 1e-10);
}

#[test]
fn test_compare_free_bound_identical() {
    let fes = FesResult {
        grid: (0..100).map(|i| i as f64 * std::f64::consts::PI / 99.0).collect(),
        free_energy: (0..100).map(|i| {
            let x = i as f64 * std::f64::consts::PI / 99.0;
            50.0 * (-(x - 0.2).powi(2) / 0.1).exp()
                + 30.0 * (-(x - 1.5).powi(2) / 0.1).exp()
                + 45.0 * (-(x - 2.9).powi(2) / 0.1).exp()
        }).collect(),
        nbins: 100,
    };

    let report = compare_free_bound(&fes, &fes, 5.0);
    assert!(report.rmsd_kjmol < 0.001);
    assert!(matches!(report.verdict, CrossLandscapeVerdict::IdenticalWithinNoise));
}

#[test]
fn test_compare_free_bound_distinct() {
    let pi = std::f64::consts::PI;
    let grid: Vec<f64> = (0..100).map(|i| i as f64 * pi / 99.0).collect();

    let free = FesResult {
        grid: grid.clone(),
        free_energy: grid.iter().map(|&x| {
            50.0 - 50.0 * (-(x - 0.2).powi(2) / 0.02).exp()
                 - 20.0 * (-(x - 1.5).powi(2) / 0.02).exp()
                 - 45.0 * (-(x - 2.9).powi(2) / 0.02).exp()
        }).collect(),
        nbins: 100,
    };

    // Enzyme-bound: barriers lowered significantly
    let bound = FesResult {
        grid: grid.clone(),
        free_energy: grid.iter().map(|&x| {
            30.0 - 50.0 * (-(x - 0.2).powi(2) / 0.02).exp()
                 - 35.0 * (-(x - 1.5).powi(2) / 0.02).exp()
                 - 45.0 * (-(x - 2.9).powi(2) / 0.02).exp()
        }).collect(),
        nbins: 100,
    };

    let report = compare_free_bound(&free, &bound, 3.0);
    assert!(report.rmsd_kjmol > 3.0, "RMSD {} should be > 3.0", report.rmsd_kjmol);
    assert!(matches!(report.verdict, CrossLandscapeVerdict::Distinct));
}

#[test]
fn test_ks_same_distribution() {
    let a: Vec<f64> = (0..1000).map(|i| (i as f64 / 1000.0) * std::f64::consts::PI).collect();
    let b = a.clone();
    let result = ks_two_sample(&a, &b);
    assert!(result.statistic < result.critical_value_05);
    assert!(result.distributions_same);
}

#[test]
fn test_ks_different_distribution() {
    let a: Vec<f64> = (0..1000).map(|i| (i as f64 / 1000.0) * 0.5).collect();
    let b: Vec<f64> = (0..1000).map(|i| 2.0 + (i as f64 / 1000.0) * 0.5).collect();
    let result = ks_two_sample(&a, &b);
    assert!(result.statistic > result.critical_value_05);
    assert!(!result.distributions_same);
}
