// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::interp::{interp, interp_2d};
use crate::types::{FesResult, FesResult2D, ParityCheck};

/// Compare reconstructed FES against a reference (parity check).
/// Interpolates the computed FES onto the reference grid for comparison.
pub fn check_parity(computed: &FesResult, reference: &FesResult, tolerance: f64) -> ParityCheck {
    let n = reference.nbins;
    let mut max_dev = 0.0_f64;
    let mut sum_dev = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    // Shift both to mintozero
    let comp_min = computed.free_energy.iter().cloned().fold(f64::INFINITY, f64::min);
    let ref_min = reference.free_energy.iter().cloned().fold(f64::INFINITY, f64::min);

    for i in 0..n {
        let ref_x = reference.grid[i];
        let comp_val = interp(ref_x, &computed.grid, &computed.free_energy) - comp_min;
        let ref_val = reference.free_energy[i] - ref_min;
        let diff = (comp_val - ref_val).abs();
        max_dev = max_dev.max(diff);
        sum_dev += diff;
        sum_sq += diff * diff;
    }

    let mean_dev = sum_dev / n as f64;
    let rmsd = (sum_sq / n as f64).sqrt();
    let status = if max_dev < tolerance { "MATCH" } else { "DIVERGENCE" };

    ParityCheck {
        max_deviation_kjmol: max_dev,
        mean_deviation_kjmol: mean_dev,
        rmsd_kjmol: rmsd,
        status: status.to_string(),
    }
}

/// Compare reconstructed 2D FES against a reference (parity check).
pub fn check_parity_2d(computed: &FesResult2D, reference: &FesResult2D, tolerance: f64) -> ParityCheck {
    let mut max_dev = 0.0_f64;
    let mut sum_dev = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;

    // Shift both to min-zero
    let comp_min = computed.free_energy.iter()
        .flat_map(|r| r.iter()).cloned()
        .fold(f64::INFINITY, f64::min);
    let ref_min = reference.free_energy.iter()
        .flat_map(|r| r.iter()).cloned()
        .fold(f64::INFINITY, f64::min);

    for i in 0..reference.nbins_x {
        for j in 0..reference.nbins_y {
            let rx = reference.grid_x[i];
            let ry = reference.grid_y[j];
            let ref_val = reference.free_energy[i][j] - ref_min;
            let comp_val = interp_2d(rx, ry, &computed.grid_x, &computed.grid_y, &computed.free_energy) - comp_min;
            let diff = (comp_val - ref_val).abs();
            max_dev = max_dev.max(diff);
            sum_dev += diff;
            sum_sq += diff * diff;
            count += 1;
        }
    }

    let mean_dev = sum_dev / count as f64;
    let rmsd = (sum_sq / count as f64).sqrt();
    let status = if max_dev < tolerance { "MATCH" } else { "DIVERGENCE" };

    ParityCheck {
        max_deviation_kjmol: max_dev,
        mean_deviation_kjmol: mean_dev,
        rmsd_kjmol: rmsd,
        status: status.to_string(),
    }
}
