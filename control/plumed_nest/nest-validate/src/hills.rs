// SPDX-License-Identifier: AGPL-3.0-or-later

//! HILLS file parser — handles 1D and 2D PLUMED metadynamics output.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hills1D {
    pub time: Vec<f64>,
    pub centers: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub heights: Vec<f64>,
    pub biasfactor: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hills2D {
    pub time: Vec<f64>,
    pub centers_x: Vec<f64>,
    pub centers_y: Vec<f64>,
    pub sigmas_x: Vec<f64>,
    pub sigmas_y: Vec<f64>,
    pub heights: Vec<f64>,
    pub biasfactor: Vec<f64>,
}

impl Hills1D {
    pub fn n_gaussians(&self) -> usize {
        self.centers.len()
    }

    pub fn total_time_ps(&self) -> f64 {
        self.time.last().copied().unwrap_or(0.0)
    }

    pub fn mean_height(&self) -> f64 {
        if self.heights.is_empty() {
            return 0.0;
        }
        self.heights.iter().sum::<f64>() / self.heights.len() as f64
    }

    pub fn final_height_mean(&self, n: usize) -> f64 {
        let n = n.min(self.heights.len());
        let slice = &self.heights[self.heights.len() - n..];
        slice.iter().sum::<f64>() / slice.len() as f64
    }

    pub fn height_decay_ratio(&self) -> f64 {
        let n = 100.min(self.heights.len() / 2).max(1);
        let early = self.heights[..n].iter().sum::<f64>() / n as f64;
        let late = self.heights[self.heights.len() - n..].iter().sum::<f64>() / n as f64;
        if early.abs() < 1e-15 {
            0.0
        } else {
            late / early
        }
    }
}

impl Hills2D {
    pub fn n_gaussians(&self) -> usize {
        self.centers_x.len()
    }
}

pub fn parse_hills_1d(path: &Path) -> Result<Hills1D, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read HILLS: {e}"))?;

    let mut time = Vec::new();
    let mut centers = Vec::new();
    let mut sigmas = Vec::new();
    let mut heights = Vec::new();
    let mut biasfactor = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with('@') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }
        time.push(parts[0].parse::<f64>().map_err(|e| format!("Parse time: {e}"))?);
        centers.push(parts[1].parse::<f64>().map_err(|e| format!("Parse center: {e}"))?);
        sigmas.push(parts[2].parse::<f64>().map_err(|e| format!("Parse sigma: {e}"))?);
        heights.push(parts[3].parse::<f64>().map_err(|e| format!("Parse height: {e}"))?);
        biasfactor.push(parts[4].parse::<f64>().map_err(|e| format!("Parse biasf: {e}"))?);
    }

    Ok(Hills1D { time, centers, sigmas, heights, biasfactor })
}

pub fn parse_hills_2d(path: &Path) -> Result<Hills2D, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read HILLS: {e}"))?;

    let mut time = Vec::new();
    let mut centers_x = Vec::new();
    let mut centers_y = Vec::new();
    let mut sigmas_x = Vec::new();
    let mut sigmas_y = Vec::new();
    let mut heights = Vec::new();
    let mut biasfactor = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with('@') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 7 {
            continue;
        }
        time.push(parts[0].parse::<f64>().map_err(|e| format!("Parse time: {e}"))?);
        centers_x.push(parts[1].parse::<f64>().map_err(|e| format!("Parse cx: {e}"))?);
        centers_y.push(parts[2].parse::<f64>().map_err(|e| format!("Parse cy: {e}"))?);
        sigmas_x.push(parts[3].parse::<f64>().map_err(|e| format!("Parse sx: {e}"))?);
        sigmas_y.push(parts[4].parse::<f64>().map_err(|e| format!("Parse sy: {e}"))?);
        heights.push(parts[5].parse::<f64>().map_err(|e| format!("Parse height: {e}"))?);
        biasfactor.push(parts[6].parse::<f64>().map_err(|e| format!("Parse biasf: {e}"))?);
    }

    Ok(Hills2D { time, centers_x, centers_y, sigmas_x, sigmas_y, heights, biasfactor })
}

/// Detect if a HILLS file is 1D or 2D by checking column count.
pub fn detect_dimensionality(path: &Path) -> Result<usize, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read HILLS: {e}"))?;

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with('@') || line.is_empty() {
            continue;
        }
        let cols = line.split_whitespace().count();
        return Ok(if cols >= 7 { 2 } else { 1 });
    }
    Err("Empty HILLS file".to_string())
}
