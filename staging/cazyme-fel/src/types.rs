// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLandscapeReport {
    pub rmsd_kjmol: f64,
    pub max_diff_kjmol: f64,
    pub mean_diff_kjmol: f64,
    pub basin_diffs: Vec<BasinDiff>,
    pub verdict: CrossLandscapeVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossLandscapeVerdict {
    Distinct,
    SuspiciouslySimilar,
    IdenticalWithinNoise,
}

impl std::fmt::Display for CrossLandscapeVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Distinct => write!(f, "DISTINCT"),
            Self::SuspiciouslySimilar => write!(f, "SUSPICIOUS"),
            Self::IdenticalWithinNoise => write!(f, "FAIL_IDENTICAL"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasinDiff {
    pub label: String,
    pub theta_range: [f64; 2],
    pub free_energy_diff_kjmol: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingCheck {
    pub max_distance_nm: f64,
    pub mean_distance_nm: f64,
    pub final_distance_nm: f64,
    pub n_frames: usize,
    pub dissociated: bool,
    pub wall_active_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KsTestResult {
    pub statistic: f64,
    pub n_free: usize,
    pub n_bound: usize,
    pub critical_value_05: f64,
    pub distributions_same: bool,
}
