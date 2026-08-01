// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::Path;

use crate::types::{FesResult, FesResult2D, Hills, Hills2D};

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

/// Parse a COLVAR file and extract the binding distance column.
///
/// Returns (distances, wall_biases) — both as Vec<f64>.
/// The column index for d_bind and wall_bind.bias depends on the COLVAR header.
pub fn parse_binding_colvar(path: &Path) -> Result<(Vec<f64>, Vec<f64>), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read COLVAR: {e}"))?;

    let mut d_bind_col: Option<usize> = None;
    let mut wall_col: Option<usize> = None;
    let mut distances = Vec::new();
    let mut wall_biases = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("#! FIELDS") {
            let fields: Vec<&str> = line.split_whitespace().collect();
            for (i, f) in fields.iter().enumerate() {
                if *f == "d_bind" {
                    d_bind_col = Some(i - 2); // offset for "#! FIELDS"
                }
                if *f == "wall_bind.bias" {
                    wall_col = Some(i - 2);
                }
            }
            continue;
        }
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if let Some(col) = d_bind_col {
            if col < parts.len() {
                if let Ok(d) = parts[col].parse::<f64>() {
                    distances.push(d);
                }
            }
        }
        if let Some(col) = wall_col {
            if col < parts.len() {
                if let Ok(w) = parts[col].parse::<f64>() {
                    wall_biases.push(w);
                }
            }
        }
    }

    if distances.is_empty() {
        return Err("No d_bind column found in COLVAR".to_string());
    }
    if wall_biases.is_empty() {
        wall_biases = vec![0.0; distances.len()];
    }

    Ok((distances, wall_biases))
}

/// Extract theta values from a COLVAR file (column 2, after time).
pub fn parse_colvar_theta(path: &Path) -> Result<Vec<f64>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read COLVAR: {e}"))?;

    let mut theta_col: Option<usize> = None;
    let mut values = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("#! FIELDS") {
            let fields: Vec<&str> = line.split_whitespace().collect();
            for (i, f) in fields.iter().enumerate() {
                if *f == "puck.theta" {
                    theta_col = Some(i - 2);
                    break;
                }
            }
            continue;
        }
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        let col = theta_col.unwrap_or(1);
        if col < parts.len() {
            if let Ok(v) = parts[col].parse::<f64>() {
                values.push(v);
            }
        }
    }

    if values.is_empty() {
        return Err("No theta values found in COLVAR".to_string());
    }
    Ok(values)
}
