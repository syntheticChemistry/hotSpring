// SPDX-License-Identifier: AGPL-3.0-or-later

//! CAZyme conformational free energy landscape validation module.
//!
//! Reconstructs F(θ) from PLUMED HILLS files using Gaussian kernel summation
//! (well-tempered metadynamics). Validates topology: basin count, barrier
//! ranges, ground state identity.
//!
//! This is the Tier 2 (Rust) implementation paired with the Tier 1 Python
//! notebook at `notebooks/cazyme_fel/puckering_fel.py`. Both must produce
//! identical results (MATCH in ParityReport).

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hills {
    pub centers: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub heights: Vec<f64>,
    pub biasfactor: f64,
    pub n_gaussians: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FesResult {
    pub grid: Vec<f64>,
    pub free_energy: Vec<f64>,
    pub nbins: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Basin {
    pub theta_rad: f64,
    pub theta_deg: f64,
    pub energy_kjmol: f64,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Barrier {
    pub from_label: String,
    pub to_label: String,
    pub theta_deg: f64,
    pub height_kjmol: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub basins: Vec<Basin>,
    pub barriers: Vec<Barrier>,
    pub chair_basins_found: usize,
    pub boat_basin_found: bool,
    pub barrier_range_kjmol: [f64; 2],
    pub parity: Option<ParityCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityCheck {
    pub max_deviation_kjmol: f64,
    pub mean_deviation_kjmol: f64,
    pub rmsd_kjmol: f64,
    pub status: String,
}

/// Parse a 1D PLUMED HILLS file.
pub fn parse_hills(path: &Path) -> Result<Hills, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read HILLS: {e}"))?;

    let mut centers = Vec::new();
    let mut sigmas = Vec::new();
    let mut heights = Vec::new();
    let mut biasfactor = 0.0;

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }
        // 1D format: time center sigma height biasf
        let center: f64 = parts[1].parse().map_err(|e| format!("Parse error: {e}"))?;
        let sigma: f64 = parts[2].parse().map_err(|e| format!("Parse error: {e}"))?;
        let height: f64 = parts[3].parse().map_err(|e| format!("Parse error: {e}"))?;
        let bf: f64 = parts[4].parse().map_err(|e| format!("Parse error: {e}"))?;

        centers.push(center);
        sigmas.push(sigma);
        heights.push(height);
        if biasfactor == 0.0 {
            biasfactor = bf;
        }
    }

    let n = centers.len();
    Ok(Hills { centers, sigmas, heights, biasfactor, n_gaussians: n })
}

/// Parse a PLUMED FES output file (grid + free_energy columns).
pub fn parse_fes(path: &Path) -> Result<FesResult, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read FES: {e}"))?;

    let mut grid = Vec::new();
    let mut free_energy = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let theta: f64 = parts[0].parse().map_err(|e| format!("Parse error: {e}"))?;
        let energy: f64 = parts[1].parse().map_err(|e| format!("Parse error: {e}"))?;
        grid.push(theta);
        free_energy.push(energy);
    }

    let nbins = grid.len();
    Ok(FesResult { grid, free_energy, nbins })
}

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

/// Identify basins (local minima) in the 1D FEL.
pub fn find_basins(fes: &FesResult) -> Vec<Basin> {
    let mut basins = Vec::new();
    let n = fes.nbins;

    for i in 1..n - 1 {
        if fes.free_energy[i] < fes.free_energy[i - 1]
            && fes.free_energy[i] < fes.free_energy[i + 1]
        {
            let theta_rad = fes.grid[i];
            let theta_deg = theta_rad.to_degrees();
            let label = if theta_deg < 40.0 {
                "4C1 chair"
            } else if theta_deg > 140.0 {
                "1C4 chair"
            } else {
                "boat/skew-boat"
            };
            basins.push(Basin {
                theta_rad,
                theta_deg,
                energy_kjmol: fes.free_energy[i],
                label: label.to_string(),
            });
        }
    }

    // Check endpoints
    if n > 1 && fes.free_energy[0] < fes.free_energy[1] {
        let theta_deg = fes.grid[0].to_degrees();
        let label = if theta_deg < 40.0 { "4C1 chair" } else { "1C4 chair" };
        basins.insert(0, Basin {
            theta_rad: fes.grid[0],
            theta_deg,
            energy_kjmol: fes.free_energy[0],
            label: label.to_string(),
        });
    }
    if n > 1 && fes.free_energy[n - 1] < fes.free_energy[n - 2] {
        let theta_deg = fes.grid[n - 1].to_degrees();
        let label = if theta_deg > 140.0 { "1C4 chair" } else { "4C1 chair" };
        basins.push(Basin {
            theta_rad: fes.grid[n - 1],
            theta_deg,
            energy_kjmol: fes.free_energy[n - 1],
            label: label.to_string(),
        });
    }

    basins
}

/// Find barriers between adjacent basins.
pub fn find_barriers(fes: &FesResult, basins: &[Basin]) -> Vec<Barrier> {
    let mut barriers = Vec::new();
    let mut sorted: Vec<&Basin> = basins.iter().collect();
    sorted.sort_by(|a, b| a.theta_rad.partial_cmp(&b.theta_rad).unwrap());

    for pair in sorted.windows(2) {
        let b1 = pair[0];
        let b2 = pair[1];

        // Find grid indices for this pair
        let i1 = fes.grid.iter().position(|&x| (x - b1.theta_rad).abs() < 1e-6)
            .unwrap_or(0);
        let i2 = fes.grid.iter().position(|&x| (x - b2.theta_rad).abs() < 1e-6)
            .unwrap_or(fes.nbins - 1);

        if i1 >= i2 {
            continue;
        }

        let (max_idx, max_e) = fes.free_energy[i1..=i2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &e)| (i1 + i, e))
            .unwrap_or((i1, 0.0));

        let ref_energy = b1.energy_kjmol.min(b2.energy_kjmol);
        barriers.push(Barrier {
            from_label: b1.label.clone(),
            to_label: b2.label.clone(),
            theta_deg: fes.grid[max_idx].to_degrees(),
            height_kjmol: max_e - ref_energy,
        });
    }

    barriers
}

/// Linearly interpolate `computed` FES onto a target grid point.
fn interp(x: f64, grid: &[f64], values: &[f64]) -> f64 {
    if x <= grid[0] {
        return values[0];
    }
    if x >= *grid.last().unwrap() {
        return *values.last().unwrap();
    }
    // Binary search for bracketing interval
    let mut lo = 0;
    let mut hi = grid.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if grid[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - grid[lo]) / (grid[hi] - grid[lo]);
    values[lo] + t * (values[hi] - values[lo])
}

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

/// Full validation: parse HILLS, reconstruct FES, analyze topology.
pub fn run_validation(
    hills_path: &Path,
    reference_path: Option<&Path>,
    nbins: usize,
) -> Result<ValidationResult, String> {
    let hills = parse_hills(hills_path)?;

    let grid_min = hills.centers.iter().cloned().fold(f64::INFINITY, f64::min)
        - 3.0 * hills.sigmas.iter().cloned().fold(0.0_f64, f64::max);
    let grid_max = hills.centers.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        + 3.0 * hills.sigmas.iter().cloned().fold(0.0_f64, f64::max);

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

// ─── 2D support ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hills2D {
    pub centers_x: Vec<f64>,
    pub centers_y: Vec<f64>,
    pub sigmas_x: Vec<f64>,
    pub sigmas_y: Vec<f64>,
    pub heights: Vec<f64>,
    pub biasfactor: f64,
    pub n_gaussians: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FesResult2D {
    pub grid_x: Vec<f64>,
    pub grid_y: Vec<f64>,
    pub free_energy: Vec<Vec<f64>>,
    pub nbins_x: usize,
    pub nbins_y: usize,
}

/// Parse a 2D PLUMED HILLS file (7 fields: time cx cy sx sy height biasf).
pub fn parse_hills_2d(path: &Path) -> Result<Hills2D, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read HILLS: {e}"))?;

    let mut centers_x = Vec::new();
    let mut centers_y = Vec::new();
    let mut sigmas_x = Vec::new();
    let mut sigmas_y = Vec::new();
    let mut heights = Vec::new();
    let mut biasfactor = 0.0;

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 7 {
            continue;
        }
        // 2D format: time center_x center_y sigma_x sigma_y height biasf
        let cx: f64 = parts[1].parse().map_err(|e| format!("Parse cx: {e}"))?;
        let cy: f64 = parts[2].parse().map_err(|e| format!("Parse cy: {e}"))?;
        let sx: f64 = parts[3].parse().map_err(|e| format!("Parse sx: {e}"))?;
        let sy: f64 = parts[4].parse().map_err(|e| format!("Parse sy: {e}"))?;
        let h: f64 = parts[5].parse().map_err(|e| format!("Parse height: {e}"))?;
        let bf: f64 = parts[6].parse().map_err(|e| format!("Parse biasf: {e}"))?;

        centers_x.push(cx);
        centers_y.push(cy);
        sigmas_x.push(sx);
        sigmas_y.push(sy);
        heights.push(h);
        if biasfactor == 0.0 {
            biasfactor = bf;
        }
    }

    let n = centers_x.len();
    Ok(Hills2D { centers_x, centers_y, sigmas_x, sigmas_y, heights, biasfactor, n_gaussians: n })
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

/// Parse a PLUMED 2D FES output file (x y energy format, blank-line-separated blocks).
pub fn parse_fes_2d(path: &Path) -> Result<FesResult2D, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read 2D FES: {e}"))?;

    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut energies: Vec<f64> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }
        let x: f64 = parts[0].parse().map_err(|e| format!("Parse x: {e}"))?;
        let y: f64 = parts[1].parse().map_err(|e| format!("Parse y: {e}"))?;
        let e: f64 = parts[2].parse().map_err(|e| format!("Parse energy: {e}"))?;
        xs.push(x);
        ys.push(y);
        energies.push(e);
    }

    if xs.is_empty() {
        return Err("Empty 2D FES file".to_string());
    }

    // PLUMED 2D sum_hills format: x varies fast (inner), y varies slow (outer)
    // y0: x0 y0, x1 y0, ..., xN y0, [blank], y1: x0 y1, x1 y1, ...
    // Count x values in first y-block (entries with same first y value).
    let first_y_val = ys[0];
    let mut nbins_x = 0;
    for &y in &ys {
        if (y - first_y_val).abs() < 1e-10 {
            nbins_x += 1;
        } else {
            break;
        }
    }
    let nbins_y = xs.len() / nbins_x;

    if nbins_x * nbins_y != xs.len() {
        return Err(format!(
            "Grid dimensions don't match: {} points but {}x{} = {}",
            xs.len(), nbins_x, nbins_y, nbins_x * nbins_y
        ));
    }

    // grid_x: unique x values from first y-block
    let grid_x: Vec<f64> = (0..nbins_x).map(|i| xs[i]).collect();
    // grid_y: y value from start of each block
    let grid_y: Vec<f64> = (0..nbins_y).map(|j| ys[j * nbins_x]).collect();

    // Reshape: free_energy[ix][iy] from flat data[iy * nbins_x + ix]
    let mut free_energy = vec![vec![0.0; nbins_y]; nbins_x];
    for iy in 0..nbins_y {
        for ix in 0..nbins_x {
            free_energy[ix][iy] = energies[iy * nbins_x + ix];
        }
    }

    Ok(FesResult2D { grid_x, grid_y, free_energy, nbins_x, nbins_y })
}

/// Bilinear interpolation on a 2D grid.
fn interp_2d(x: f64, y: f64, grid_x: &[f64], grid_y: &[f64], values: &[Vec<f64>]) -> f64 {
    let nx = grid_x.len();
    let ny = grid_y.len();

    // Clamp x
    let x = x.max(grid_x[0]).min(*grid_x.last().unwrap());
    let y = y.max(grid_y[0]).min(*grid_y.last().unwrap());

    // Find bracketing x
    let mut ix = 0;
    for i in 0..nx - 1 {
        if grid_x[i + 1] >= x {
            ix = i;
            break;
        }
    }
    if x >= *grid_x.last().unwrap() {
        ix = nx - 2;
    }

    // Find bracketing y
    let mut iy = 0;
    for j in 0..ny - 1 {
        if grid_y[j + 1] >= y {
            iy = j;
            break;
        }
    }
    if y >= *grid_y.last().unwrap() {
        iy = ny - 2;
    }

    let tx = if (grid_x[ix + 1] - grid_x[ix]).abs() > 1e-15 {
        (x - grid_x[ix]) / (grid_x[ix + 1] - grid_x[ix])
    } else {
        0.0
    };
    let ty = if (grid_y[iy + 1] - grid_y[iy]).abs() > 1e-15 {
        (y - grid_y[iy]) / (grid_y[iy + 1] - grid_y[iy])
    } else {
        0.0
    };

    let v00 = values[ix][iy];
    let v10 = values[ix + 1][iy];
    let v01 = values[ix][iy + 1];
    let v11 = values[ix + 1][iy + 1];

    v00 * (1.0 - tx) * (1.0 - ty) + v10 * tx * (1.0 - ty) + v01 * (1.0 - tx) * ty + v11 * tx * ty
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
        check_parity_2d(&fes, &ref_fes, 2.0)
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

    let margin_x = 3.0 * hills.sigmas_x.iter().cloned().fold(0.0_f64, f64::max);
    let margin_y = 3.0 * hills.sigmas_y.iter().cloned().fold(0.0_f64, f64::max);

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
        check_parity_2d(&fes, &ref_fes, 2.0)
    });

    Ok((fes, parity))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
