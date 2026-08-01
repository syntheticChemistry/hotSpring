// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::Path;

use crate::constants::{GRID_MARGIN_SIGMA, PARITY_TOLERANCE_KJMOL};
use crate::parity::{check_parity, check_parity_2d};
use crate::parse::{parse_fes, parse_fes_2d, parse_hills, parse_hills_2d};
use crate::reconstruct::{reconstruct_fes, reconstruct_fes_2d};
use crate::topology::{find_barriers, find_basins};
use crate::types::{FesResult2D, ParityCheck, ValidationResult};

/// Full validation: parse HILLS, reconstruct FES, analyze topology.
pub fn run_validation(
    hills_path: &Path,
    reference_path: Option<&Path>,
    nbins: usize,
) -> Result<ValidationResult, String> {
    let hills = parse_hills(hills_path)?;

    let grid_min = hills.centers.iter().cloned().fold(f64::INFINITY, f64::min)
        - GRID_MARGIN_SIGMA * hills.sigmas.iter().cloned().fold(0.0_f64, f64::max);
    let grid_max = hills.centers.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        + GRID_MARGIN_SIGMA * hills.sigmas.iter().cloned().fold(0.0_f64, f64::max);

    let fes = reconstruct_fes(&hills, grid_min, grid_max, nbins);
    let basins = find_basins(&fes);
    let barriers = find_barriers(&fes, &basins);

    let chair_count = basins.iter().filter(|b| b.label.contains("chair")).count();
    let boat_found = basins.iter().any(|b| b.label.contains("boat"));
    let barrier_range = if barriers.is_empty() {
        [0.0, 0.0]
    } else {
        let min_b = barriers.iter().map(|b| b.height_kjmol).fold(f64::INFINITY, f64::min);
        let max_b = barriers.iter().map(|b| b.height_kjmol).fold(f64::NEG_INFINITY, f64::max);
        [min_b, max_b]
    };

    let parity = reference_path.map(|rp| {
        let ref_fes = parse_fes(rp).expect("Failed to parse reference FES");
        check_parity(&fes, &ref_fes, 1.0)
    });

    Ok(ValidationResult {
        basins,
        barriers,
        chair_basins_found: chair_count,
        boat_basin_found: boat_found,
        barrier_range_kjmol: barrier_range,
        parity,
    })
}

/// Full 2D validation with explicit grid bounds.
pub fn run_validation_2d_with_bounds(
    hills_path: &Path,
    reference_path: Option<&Path>,
    nbins_x: usize,
    nbins_y: usize,
    periodic_y: bool,
    grid_min_x: f64,
    grid_max_x: f64,
    grid_min_y: f64,
    grid_max_y: f64,
) -> Result<(FesResult2D, Option<ParityCheck>), String> {
    let hills = parse_hills_2d(hills_path)?;
    let fes = reconstruct_fes_2d(&hills, grid_min_x, grid_max_x, grid_min_y, grid_max_y, nbins_x, nbins_y, periodic_y);

    let parity = reference_path.map(|rp| {
        let ref_fes = parse_fes_2d(rp).expect("Failed to parse reference 2D FES");
        check_parity_2d(&fes, &ref_fes, PARITY_TOLERANCE_KJMOL)
    });

    Ok((fes, parity))
}

/// Full 2D validation: parse HILLS, reconstruct FES, compare with reference.
///
/// Grid bounds are inferred from data (with margin) for non-periodic CVs,
/// or from standard Cremer-Pople domains for periodic ones.
pub fn run_validation_2d(
    hills_path: &Path,
    reference_path: Option<&Path>,
    nbins_x: usize,
    nbins_y: usize,
    periodic_y: bool,
) -> Result<(FesResult2D, Option<ParityCheck>), String> {
    let hills = parse_hills_2d(hills_path)?;

    let margin_x = GRID_MARGIN_SIGMA * hills.sigmas_x.iter().cloned().fold(0.0_f64, f64::max);
    let margin_y = GRID_MARGIN_SIGMA * hills.sigmas_y.iter().cloned().fold(0.0_f64, f64::max);

    let (grid_min_x, grid_max_x) = {
        let min_x = hills.centers_x.iter().cloned().fold(f64::INFINITY, f64::min) - margin_x;
        let max_x = hills.centers_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + margin_x;
        (min_x, max_x)
    };

    let (grid_min_y, grid_max_y) = if periodic_y {
        (0.0, 2.0 * std::f64::consts::PI)
    } else {
        let min_y = hills.centers_y.iter().cloned().fold(f64::INFINITY, f64::min) - margin_y;
        let max_y = hills.centers_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + margin_y;
        (min_y, max_y)
    };

    let fes = reconstruct_fes_2d(&hills, grid_min_x, grid_max_x, grid_min_y, grid_max_y, nbins_x, nbins_y, periodic_y);

    let parity = reference_path.map(|rp| {
        let ref_fes = parse_fes_2d(rp).expect("Failed to parse reference 2D FES");
        check_parity_2d(&fes, &ref_fes, PARITY_TOLERANCE_KJMOL)
    });

    Ok((fes, parity))
}
