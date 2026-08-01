// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::constants::VERDICT_MIDPOINT;
use crate::interp::{interp, interp_2d};
use crate::types::{
    BasinDiff, CrossLandscapeReport, CrossLandscapeVerdict, FesResult, FesResult2D,
};

/// Compare a free-xylose FES against an enzyme-bound FES (1D).
///
/// If the enzyme-bound landscape is too similar to free xylose, the substrate
/// likely dissociated and is sampling bulk solvent (not the active site).
///
/// `min_divergence`: minimum RMSD (kJ/mol) below which landscapes are flagged.
/// Typical threshold: 5.0 kJ/mol (from tolerances.toml enzyme_barrier_reduction).
pub fn compare_free_bound(
    free: &FesResult,
    bound: &FesResult,
    min_divergence: f64,
) -> CrossLandscapeReport {
    let n = free.nbins.min(bound.nbins);

    let free_min = free.free_energy.iter().cloned().fold(f64::INFINITY, f64::min);
    let bound_min = bound.free_energy.iter().cloned().fold(f64::INFINITY, f64::min);

    let mut max_diff = 0.0_f64;
    let mut sum_diff = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for i in 0..n {
        let x = free.grid[i];
        let free_val = free.free_energy[i] - free_min;
        let bound_val = interp(x, &bound.grid, &bound.free_energy) - bound_min;
        let diff = (bound_val - free_val).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
        sum_sq += diff * diff;
    }

    let mean_diff = sum_diff / n as f64;
    let rmsd = (sum_sq / n as f64).sqrt();

    // Per-basin analysis: compare energies in the three canonical regions
    let basin_regions: &[(&str, f64, f64)] = &[
        ("4C1_chair", 0.0, 0.7),           // 0-40 deg
        ("boat_skewboat", 0.7, 2.44),      // 40-140 deg
        ("1C4_chair", 2.44, std::f64::consts::PI),  // 140-180 deg
    ];

    let mut basin_diffs = Vec::new();
    for &(label, theta_lo, theta_hi) in basin_regions {
        let mut free_sum = 0.0;
        let mut bound_sum = 0.0;
        let mut count = 0;

        for i in 0..n {
            let x = free.grid[i];
            if x >= theta_lo && x <= theta_hi {
                free_sum += free.free_energy[i] - free_min;
                bound_sum += interp(x, &bound.grid, &bound.free_energy) - bound_min;
                count += 1;
            }
        }

        let diff = if count > 0 {
            (bound_sum - free_sum) / count as f64
        } else {
            0.0
        };

        basin_diffs.push(BasinDiff {
            label: label.to_string(),
            theta_range: [theta_lo.to_degrees(), theta_hi.to_degrees()],
            free_energy_diff_kjmol: diff,
        });
    }

    let verdict = if rmsd < min_divergence * VERDICT_MIDPOINT {
        CrossLandscapeVerdict::IdenticalWithinNoise
    } else if rmsd < min_divergence {
        CrossLandscapeVerdict::SuspiciouslySimilar
    } else {
        CrossLandscapeVerdict::Distinct
    };

    CrossLandscapeReport {
        rmsd_kjmol: rmsd,
        max_diff_kjmol: max_diff,
        mean_diff_kjmol: mean_diff,
        basin_diffs,
        verdict,
    }
}

/// Compare free-xylose vs enzyme-bound 2D FES.
///
/// Same logic as 1D but operates on the full (qx, qy) or (theta, phi) surface.
pub fn compare_free_bound_2d(
    free: &FesResult2D,
    bound: &FesResult2D,
    min_divergence: f64,
) -> CrossLandscapeReport {
    let free_min = free.free_energy.iter()
        .flat_map(|r| r.iter()).cloned()
        .fold(f64::INFINITY, f64::min);
    let bound_min = bound.free_energy.iter()
        .flat_map(|r| r.iter()).cloned()
        .fold(f64::INFINITY, f64::min);

    let mut max_diff = 0.0_f64;
    let mut sum_diff = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;

    for i in 0..free.nbins_x {
        for j in 0..free.nbins_y {
            let x = free.grid_x[i];
            let y = free.grid_y[j];
            let free_val = free.free_energy[i][j] - free_min;
            let bound_val = interp_2d(x, y, &bound.grid_x, &bound.grid_y, &bound.free_energy) - bound_min;
            let diff = (bound_val - free_val).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
            sum_sq += diff * diff;
            count += 1;
        }
    }

    let mean_diff = sum_diff / count as f64;
    let rmsd = (sum_sq / count as f64).sqrt();

    let verdict = if rmsd < min_divergence * VERDICT_MIDPOINT {
        CrossLandscapeVerdict::IdenticalWithinNoise
    } else if rmsd < min_divergence {
        CrossLandscapeVerdict::SuspiciouslySimilar
    } else {
        CrossLandscapeVerdict::Distinct
    };

    CrossLandscapeReport {
        rmsd_kjmol: rmsd,
        max_diff_kjmol: max_diff,
        mean_diff_kjmol: mean_diff,
        basin_diffs: Vec::new(),
        verdict,
    }
}
