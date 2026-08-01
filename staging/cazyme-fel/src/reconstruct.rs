// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::{FesResult, FesResult2D, Hills, Hills2D};

/// Reconstruct 1D FES from HILLS via Gaussian kernel summation.
///
/// For well-tempered metadynamics, deposited heights encode the decay:
///   F(s) = -V(s, t→∞) + const
///   V(s) = Σᵢ hᵢ · exp(-(s - sᵢ)² / (2σᵢ²))
pub fn reconstruct_fes(hills: &Hills, grid_min: f64, grid_max: f64, nbins: usize) -> FesResult {
    let grid: Vec<f64> = (0..nbins)
        .map(|i| grid_min + (grid_max - grid_min) * i as f64 / (nbins - 1) as f64)
        .collect();

    let mut bias = vec![0.0_f64; nbins];

    for g in 0..hills.n_gaussians {
        let c = hills.centers[g];
        let s = hills.sigmas[g];
        let h = hills.heights[g];
        let inv_2s2 = 1.0 / (2.0 * s * s);

        for (i, b) in bias.iter_mut().enumerate() {
            let diff = grid[i] - c;
            *b += h * (-diff * diff * inv_2s2).exp();
        }
    }

    // F(s) = -V(s), then shift minimum to zero
    let mut fes: Vec<f64> = bias.iter().map(|v| -v).collect();
    let min_val = fes.iter().cloned().fold(f64::INFINITY, f64::min);
    for f in &mut fes {
        *f -= min_val;
    }

    FesResult { grid, free_energy: fes, nbins }
}

/// Reconstruct 2D FES from HILLS via Gaussian kernel summation.
///
/// `periodic_y`: if true, wraps the y-axis with image Gaussians (e.g. phi ∈ [0, 2π]).
pub fn reconstruct_fes_2d(
    hills: &Hills2D,
    grid_min_x: f64,
    grid_max_x: f64,
    grid_min_y: f64,
    grid_max_y: f64,
    nbins_x: usize,
    nbins_y: usize,
    periodic_y: bool,
) -> FesResult2D {
    let grid_x: Vec<f64> = (0..nbins_x)
        .map(|i| grid_min_x + (grid_max_x - grid_min_x) * i as f64 / (nbins_x - 1) as f64)
        .collect();
    let grid_y: Vec<f64> = (0..nbins_y)
        .map(|j| grid_min_y + (grid_max_y - grid_min_y) * j as f64 / (nbins_y - 1) as f64)
        .collect();

    let period_y = grid_max_y - grid_min_y;
    let mut bias = vec![vec![0.0_f64; nbins_y]; nbins_x];

    for g in 0..hills.n_gaussians {
        let cx = hills.centers_x[g];
        let cy = hills.centers_y[g];
        let sx = hills.sigmas_x[g];
        let sy = hills.sigmas_y[g];
        let h = hills.heights[g];
        let inv_2sx2 = 1.0 / (2.0 * sx * sx);
        let inv_2sy2 = 1.0 / (2.0 * sy * sy);

        for (i, brow) in bias.iter_mut().enumerate() {
            let dx = grid_x[i] - cx;
            let exp_x = (-dx * dx * inv_2sx2).exp();

            for (j, bval) in brow.iter_mut().enumerate() {
                let mut dy = grid_y[j] - cy;
                if periodic_y {
                    // minimum image convention
                    dy -= period_y * (dy / period_y).round();
                }
                *bval += h * exp_x * (-dy * dy * inv_2sy2).exp();
            }
        }
    }

    // F(s) = -V(s), shift min to zero
    let mut fes: Vec<Vec<f64>> = bias.iter().map(|row| row.iter().map(|v| -v).collect()).collect();
    let global_min = fes.iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    for row in &mut fes {
        for f in row.iter_mut() {
            *f -= global_min;
        }
    }

    FesResult2D { grid_x, grid_y, free_energy: fes, nbins_x, nbins_y }
}
