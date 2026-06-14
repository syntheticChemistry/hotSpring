// SPDX-License-Identifier: AGPL-3.0-or-later

//! Free energy surface reconstruction and analysis.
//!
//! Gaussian kernel summation for well-tempered metadynamics,
//! with minima detection, barrier analysis, convergence tracking,
//! and block averaging.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::hills::{Hills1D, Hills2D};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fes1D {
    pub grid: Vec<f64>,
    pub free_energy: Vec<f64>,
    pub nbins: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fes2D {
    pub grid_x: Vec<f64>,
    pub grid_y: Vec<f64>,
    pub free_energy: Vec<Vec<f64>>,
    pub nbins_x: usize,
    pub nbins_y: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Minimum {
    pub x: f64,
    pub y: Option<f64>,
    pub energy: f64,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierInfo {
    pub from: String,
    pub to: String,
    pub height_kjmol: f64,
    pub position_x: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockAveraging {
    pub n_blocks: usize,
    pub mean_fes: Vec<f64>,
    pub stderr: Vec<f64>,
    pub max_stderr: f64,
    pub mean_stderr: f64,
    pub converged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysis {
    pub barriers: Vec<f64>,
    pub final_barrier: f64,
    pub std_last_3: f64,
    pub converged: bool,
}

// ─── 1D FES reconstruction ─────────────────────────────────────────────────────

pub fn reconstruct_1d(hills: &Hills1D, grid_min: f64, grid_max: f64, nbins: usize) -> Fes1D {
    let grid: Vec<f64> = (0..nbins)
        .map(|i| grid_min + (grid_max - grid_min) * i as f64 / (nbins - 1) as f64)
        .collect();

    let mut bias = vec![0.0_f64; nbins];

    for g in 0..hills.n_gaussians() {
        let c = hills.centers[g];
        let s = hills.sigmas[g];
        let h = hills.heights[g];
        let inv_2s2 = 1.0 / (2.0 * s * s);

        for (i, b) in bias.iter_mut().enumerate() {
            let diff = grid[i] - c;
            *b += h * (-diff * diff * inv_2s2).exp();
        }
    }

    let mut fes: Vec<f64> = bias.iter().map(|v| -v).collect();
    let min_val = fes.iter().cloned().fold(f64::INFINITY, f64::min);
    for f in &mut fes {
        *f -= min_val;
    }

    Fes1D { grid, free_energy: fes, nbins }
}

/// Reconstruct 1D FES from HILLS at successive time windows (for convergence).
pub fn reconstruct_1d_stride(
    hills: &Hills1D,
    grid_min: f64,
    grid_max: f64,
    nbins: usize,
    n_windows: usize,
) -> Vec<Fes1D> {
    let n = hills.n_gaussians();
    let window_size = n / n_windows;
    let mut results = Vec::new();

    for w in 1..=n_windows {
        let end = (w * window_size).min(n);
        let partial = Hills1D {
            time: hills.time[..end].to_vec(),
            centers: hills.centers[..end].to_vec(),
            sigmas: hills.sigmas[..end].to_vec(),
            heights: hills.heights[..end].to_vec(),
            biasfactor: hills.biasfactor[..end].to_vec(),
        };
        results.push(reconstruct_1d(&partial, grid_min, grid_max, nbins));
    }

    results
}

// ─── 2D FES reconstruction ─────────────────────────────────────────────────────

pub fn reconstruct_2d(
    hills: &Hills2D,
    grid_min_x: f64,
    grid_max_x: f64,
    grid_min_y: f64,
    grid_max_y: f64,
    nbins_x: usize,
    nbins_y: usize,
    periodic: (bool, bool),
) -> Fes2D {
    let grid_x: Vec<f64> = (0..nbins_x)
        .map(|i| grid_min_x + (grid_max_x - grid_min_x) * i as f64 / (nbins_x - 1) as f64)
        .collect();
    let grid_y: Vec<f64> = (0..nbins_y)
        .map(|j| grid_min_y + (grid_max_y - grid_min_y) * j as f64 / (nbins_y - 1) as f64)
        .collect();

    let period_x = grid_max_x - grid_min_x;
    let period_y = grid_max_y - grid_min_y;
    let mut bias = vec![vec![0.0_f64; nbins_y]; nbins_x];

    for g in 0..hills.n_gaussians() {
        let cx = hills.centers_x[g];
        let cy = hills.centers_y[g];
        let sx = hills.sigmas_x[g];
        let sy = hills.sigmas_y[g];
        let h = hills.heights[g];
        let inv_2sx2 = 1.0 / (2.0 * sx * sx);
        let inv_2sy2 = 1.0 / (2.0 * sy * sy);

        for (i, row) in bias.iter_mut().enumerate() {
            let mut dx = grid_x[i] - cx;
            if periodic.0 {
                dx -= period_x * (dx / period_x).round();
            }
            let exp_x = (-dx * dx * inv_2sx2).exp();

            for (j, val) in row.iter_mut().enumerate() {
                let mut dy = grid_y[j] - cy;
                if periodic.1 {
                    dy -= period_y * (dy / period_y).round();
                }
                *val += h * exp_x * (-dy * dy * inv_2sy2).exp();
            }
        }
    }

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

    Fes2D { grid_x, grid_y, free_energy: fes, nbins_x, nbins_y }
}

// ─── Minima detection ───────────────────────────────────────────────────────────

pub fn find_minima_1d(fes: &Fes1D, energy_cutoff: f64) -> Vec<Minimum> {
    let mut minima = Vec::new();
    let n = fes.nbins;

    // Check interior points
    for i in 1..n - 1 {
        if fes.free_energy[i] < fes.free_energy[i - 1]
            && fes.free_energy[i] < fes.free_energy[i + 1]
            && fes.free_energy[i] < energy_cutoff
        {
            minima.push(Minimum {
                x: fes.grid[i],
                y: None,
                energy: fes.free_energy[i],
                label: String::new(),
            });
        }
    }

    // Endpoints
    if n > 1 && fes.free_energy[0] < fes.free_energy[1] && fes.free_energy[0] < energy_cutoff {
        minima.insert(0, Minimum {
            x: fes.grid[0],
            y: None,
            energy: fes.free_energy[0],
            label: String::new(),
        });
    }
    if n > 1 && fes.free_energy[n - 1] < fes.free_energy[n - 2] && fes.free_energy[n - 1] < energy_cutoff {
        minima.push(Minimum {
            x: fes.grid[n - 1],
            y: None,
            energy: fes.free_energy[n - 1],
            label: String::new(),
        });
    }

    minima.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
    minima
}

pub fn find_minima_2d(fes: &Fes2D, energy_cutoff: f64, radius: usize) -> Vec<Minimum> {
    let mut minima = Vec::new();
    let nx = fes.nbins_x;
    let ny = fes.nbins_y;

    for i in radius..nx - radius {
        for j in radius..ny - radius {
            let val = fes.free_energy[i][j];
            if val >= energy_cutoff {
                continue;
            }

            let mut is_min = true;
            'outer: for di in 0..=(2 * radius) {
                for dj in 0..=(2 * radius) {
                    let ni = i + di - radius;
                    let nj = j + dj - radius;
                    if ni == i && nj == j {
                        continue;
                    }
                    if fes.free_energy[ni][nj] < val {
                        is_min = false;
                        break 'outer;
                    }
                }
            }

            if is_min {
                minima.push(Minimum {
                    x: fes.grid_x[i],
                    y: Some(fes.grid_y[j]),
                    energy: val,
                    label: String::new(),
                });
            }
        }
    }

    minima.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
    minima
}

// ─── Barrier analysis ───────────────────────────────────────────────────────────

pub fn find_barriers_1d(fes: &Fes1D, minima: &[Minimum]) -> Vec<BarrierInfo> {
    let mut barriers = Vec::new();
    let mut sorted: Vec<&Minimum> = minima.iter().collect();
    sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

    for pair in sorted.windows(2) {
        let m1 = pair[0];
        let m2 = pair[1];

        let i1 = fes.grid.iter().position(|&x| (x - m1.x).abs() < 1e-8).unwrap_or(0);
        let i2 = fes.grid.iter().position(|&x| (x - m2.x).abs() < 1e-8).unwrap_or(fes.nbins - 1);

        if i1 >= i2 {
            continue;
        }

        let (max_idx, max_e) = fes.free_energy[i1..=i2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &e)| (i1 + i, e))
            .unwrap_or((i1, 0.0));

        let ref_energy = m1.energy.min(m2.energy);
        barriers.push(BarrierInfo {
            from: m1.label.clone(),
            to: m2.label.clone(),
            height_kjmol: max_e - ref_energy,
            position_x: fes.grid[max_idx],
        });
    }

    barriers
}

// ─── Block averaging ────────────────────────────────────────────────────────────

pub fn block_average(stride_fes: &[Fes1D], n_blocks: usize) -> Option<BlockAveraging> {
    if stride_fes.is_empty() || n_blocks == 0 {
        return None;
    }

    let n_files = stride_fes.len();
    let block_size = n_files / n_blocks;
    if block_size == 0 {
        return None;
    }

    let nbins = stride_fes[0].nbins;

    let mut block_means = vec![vec![0.0_f64; nbins]; n_blocks];
    for b in 0..n_blocks {
        let start = b * block_size;
        let end = ((b + 1) * block_size).min(n_files);
        let count = (end - start) as f64;

        for fes_idx in start..end {
            for (i, &val) in stride_fes[fes_idx].free_energy.iter().enumerate() {
                block_means[b][i] += val / count;
            }
        }
    }

    let mut mean_fes = vec![0.0_f64; nbins];
    for block in &block_means {
        for (i, &val) in block.iter().enumerate() {
            mean_fes[i] += val / n_blocks as f64;
        }
    }

    let mut stderr = vec![0.0_f64; nbins];
    for i in 0..nbins {
        let variance: f64 = block_means.iter()
            .map(|block| (block[i] - mean_fes[i]).powi(2))
            .sum::<f64>() / (n_blocks - 1).max(1) as f64;
        stderr[i] = (variance / n_blocks as f64).sqrt();
    }

    let max_stderr = stderr.iter().cloned().fold(0.0_f64, f64::max);
    let mean_stderr = stderr.iter().sum::<f64>() / nbins as f64;
    let converged = max_stderr < 5.0;

    Some(BlockAveraging {
        n_blocks,
        mean_fes,
        stderr,
        max_stderr,
        mean_stderr,
        converged,
    })
}

// ─── Convergence ────────────────────────────────────────────────────────────────

pub fn convergence_barriers(stride_fes: &[Fes1D], reference_x: f64) -> ConvergenceAnalysis {
    let mut barriers = Vec::new();

    for fes in stride_fes {
        let ref_idx = fes.grid.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - reference_x).abs()).partial_cmp(&((*b - reference_x).abs())).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let ref_val = fes.free_energy[ref_idx];

        // Find max in a region around center
        let n = fes.nbins;
        let center_start = n / 4;
        let center_end = 3 * n / 4;
        let max_val = fes.free_energy[center_start..center_end]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        barriers.push(max_val - ref_val);
    }

    let final_barrier = barriers.last().copied().unwrap_or(0.0);
    let std_last_3 = if barriers.len() >= 3 {
        let last3 = &barriers[barriers.len() - 3..];
        let mean = last3.iter().sum::<f64>() / 3.0;
        (last3.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 3.0).sqrt()
    } else {
        f64::INFINITY
    };

    ConvergenceAnalysis {
        barriers,
        final_barrier,
        std_last_3,
        converged: std_last_3 < 3.0,
    }
}

// ─── File I/O ───────────────────────────────────────────────────────────────────

pub fn parse_fes_file(path: &Path) -> Result<Fes1D, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read FES: {e}"))?;

    let mut grid = Vec::new();
    let mut free_energy = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with('@') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        grid.push(parts[0].parse::<f64>().map_err(|e| format!("Parse: {e}"))?);
        free_energy.push(parts[1].parse::<f64>().map_err(|e| format!("Parse: {e}"))?);
    }

    let nbins = grid.len();
    Ok(Fes1D { grid, free_energy, nbins })
}

pub fn write_fes_file(fes: &Fes1D, path: &Path) -> Result<(), String> {
    let mut output = String::from("#! FIELDS cv fes\n");
    for i in 0..fes.nbins {
        output.push_str(&format!("{:.8} {:.8}\n", fes.grid[i], fes.free_energy[i]));
    }
    std::fs::write(path, output).map_err(|e| format!("Write failed: {e}"))
}
