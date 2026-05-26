// SPDX-License-Identifier: AGPL-3.0-or-later

//! TOML-based target configuration.
//!
//! Allows targets to be defined declaratively rather than hardcoded in Rust.
//! Each target has a `target.toml` in its directory that specifies:
//! - Method and reference parameters
//! - Tolerance classes and pass/fail criteria
//! - File locations for HILLS/COLVAR/inputs
//! - Reference values for parity checks

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level target configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetConfig {
    pub target: TargetMeta,
    #[serde(default)]
    pub files: FileConfig,
    #[serde(default)]
    pub reference: ReferenceConfig,
    #[serde(default)]
    pub tolerances: ToleranceConfig,
    #[serde(default)]
    pub simulation: SimulationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetMeta {
    pub id: String,
    pub plum_id: String,
    pub method: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub temperature_k: f64,
    #[serde(default)]
    pub cv_names: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileConfig {
    #[serde(default = "default_hills")]
    pub hills: String,
    #[serde(default = "default_colvar")]
    pub colvar: String,
    #[serde(default = "default_plumed")]
    pub plumed_dat: String,
    #[serde(default = "default_tpr")]
    pub tpr: String,
}

fn default_hills() -> String { "output/HILLS".to_string() }
fn default_colvar() -> String { "COLVARb".to_string() }
fn default_plumed() -> String { "plumed/plumed_gromacs.dat".to_string() }
fn default_tpr() -> String { "inputs/topol.tpr".to_string() }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReferenceConfig {
    #[serde(default)]
    pub basins: Vec<BasinRef>,
    #[serde(default)]
    pub barrier_range_kjmol: Option<[f64; 2]>,
    #[serde(default)]
    pub folding_fe_range_kjmol: Option<[f64; 2]>,
    #[serde(default)]
    pub binding_fe_range_kjmol: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasinRef {
    pub name: String,
    pub x: f64,
    #[serde(default)]
    pub y: Option<f64>,
    #[serde(default)]
    pub tolerance_rad: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToleranceConfig {
    #[serde(default = "default_barrier_tol")]
    pub barrier_kjmol: f64,
    #[serde(default = "default_basin_tol")]
    pub basin_position_rad: f64,
    #[serde(default = "default_convergence_tol")]
    pub convergence_std_kjmol: f64,
    #[serde(default = "default_block_stderr_tol")]
    pub block_stderr_kjmol: f64,
    #[serde(default = "default_fe_tol")]
    pub folding_fe_kjmol: f64,
}

fn default_barrier_tol() -> f64 { 5.0 }
fn default_basin_tol() -> f64 { 0.5 }
fn default_convergence_tol() -> f64 { 3.0 }
fn default_block_stderr_tol() -> f64 { 5.0 }
fn default_fe_tol() -> f64 { 4.0 }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimulationConfig {
    #[serde(default)]
    pub nsteps: u64,
    #[serde(default)]
    pub ntmpi: u32,
    #[serde(default = "default_ntomp")]
    pub ntomp: u32,
}

fn default_ntomp() -> u32 { 8 }

/// Load target configuration from a TOML file.
pub fn load_config(target_dir: &Path) -> Option<TargetConfig> {
    let config_path = target_dir.join("target.toml");
    if !config_path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&config_path).ok()?;
    toml::from_str(&content).ok()
}

/// Generate a default target.toml for alanine dipeptide.
pub fn default_alanine_config() -> TargetConfig {
    TargetConfig {
        target: TargetMeta {
            id: "01_alanine_dipeptide".to_string(),
            plum_id: "19.009".to_string(),
            method: "well-tempered_metadynamics".to_string(),
            description: "Alanine dipeptide phi/psi FEL in vacuum".to_string(),
            temperature_k: 300.0,
            cv_names: vec!["phi".to_string(), "psi".to_string()],
        },
        files: FileConfig {
            hills: "output/HILLS".to_string(),
            colvar: "output/COLVAR".to_string(),
            plumed_dat: "plumed/plumed.dat".to_string(),
            tpr: "inputs/topol.tpr".to_string(),
        },
        reference: ReferenceConfig {
            basins: vec![
                BasinRef { name: "C7eq".to_string(), x: -1.4, y: Some(1.0), tolerance_rad: 0.5 },
                BasinRef { name: "C7ax".to_string(), x: 1.0, y: Some(-0.7), tolerance_rad: 0.5 },
            ],
            barrier_range_kjmol: Some([20.0, 45.0]),
            folding_fe_range_kjmol: None,
            binding_fe_range_kjmol: None,
        },
        tolerances: ToleranceConfig {
            barrier_kjmol: 5.0,
            basin_position_rad: 0.5,
            convergence_std_kjmol: 3.0,
            block_stderr_kjmol: 5.0,
            folding_fe_kjmol: 4.0,
        },
        simulation: SimulationConfig {
            nsteps: 5000000,
            ntmpi: 1,
            ntomp: 8,
        },
    }
}

/// Generate a default target.toml for chignolin.
pub fn default_chignolin_config() -> TargetConfig {
    TargetConfig {
        target: TargetMeta {
            id: "02_chignolin_opes".to_string(),
            plum_id: "24.029".to_string(),
            method: "OPES_METAD+OPES_METAD_EXPLORE".to_string(),
            description: "Chignolin protein folding via OPES".to_string(),
            temperature_k: 340.0,
            cv_names: vec!["hlda".to_string(), "rmsd_ca".to_string()],
        },
        files: FileConfig {
            hills: "Kernels.data".to_string(),
            colvar: "COLVARb".to_string(),
            plumed_dat: "plumed/plumed_gromacs.dat".to_string(),
            tpr: "inputs/topol.tpr".to_string(),
        },
        reference: ReferenceConfig {
            basins: vec![],
            barrier_range_kjmol: None,
            folding_fe_range_kjmol: Some([-12.0, -3.0]),
            binding_fe_range_kjmol: None,
        },
        tolerances: ToleranceConfig {
            barrier_kjmol: 5.0,
            basin_position_rad: 0.5,
            convergence_std_kjmol: 4.0,
            block_stderr_kjmol: 5.0,
            folding_fe_kjmol: 4.0,
        },
        simulation: SimulationConfig {
            nsteps: 25000000,
            ntmpi: 1,
            ntomp: 8,
        },
    }
}

/// Write a target configuration to TOML.
pub fn write_config(config: &TargetConfig, target_dir: &Path) -> Result<(), String> {
    let toml_str = toml::to_string_pretty(config)
        .map_err(|e| format!("TOML serialize error: {e}"))?;
    let config_path = target_dir.join("target.toml");
    std::fs::write(&config_path, toml_str)
        .map_err(|e| format!("Write error: {e}"))
}
