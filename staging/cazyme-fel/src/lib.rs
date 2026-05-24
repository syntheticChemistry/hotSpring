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
}
