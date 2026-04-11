// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-hardware brain state persistence for NPU-steered RHMC.
//!
//! Serializes and deserializes ESN weights and optimal parameter snapshots
//! so that the NPU cortex can resume learning across sessions and hardware
//! changes.

use super::brain_rhmc::{RhmcConfigLive, TrajectoryObservation};
use crate::md::reservoir::npu::ExportedWeights;

use std::collections::VecDeque;
use std::path::PathBuf;

/// Persisted brain state for cross-hardware learning.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BrainState {
    pub gpu_name: String,
    pub esn_weights: SerializableWeights,
    pub optimal_dt: f64,
    pub optimal_n_md: usize,
    pub optimal_cg_tol: f64,
    pub trajectories_observed: usize,
    pub mean_acceptance: f64,
    pub mean_plaquette: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SerializableWeights {
    pub w_in: Vec<f32>,
    pub w_res: Vec<f32>,
    pub w_out: Vec<f32>,
    pub input_size: usize,
    pub reservoir_size: usize,
    pub output_size: usize,
    pub leak_rate: f32,
}

impl From<&ExportedWeights> for SerializableWeights {
    fn from(w: &ExportedWeights) -> Self {
        Self {
            w_in: w.w_in.clone(),
            w_res: w.w_res.clone(),
            w_out: w.w_out.clone(),
            input_size: w.input_size,
            reservoir_size: w.reservoir_size,
            output_size: w.output_size,
            leak_rate: w.leak_rate,
        }
    }
}

/// Directory for brain state persistence.
pub fn brain_state_dir() -> PathBuf {
    let base = dirs_fallback();
    let dir = base.join("hotspring").join("brain");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn dirs_fallback() -> PathBuf {
    std::env::var("XDG_DATA_HOME").map_or_else(
        |_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".local").join("share")
        },
        PathBuf::from,
    )
}

fn sanitize_gpu_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .to_lowercase()
}

/// Save brain state for a GPU.
pub fn save_brain_state(state: &BrainState) -> std::io::Result<()> {
    let dir = brain_state_dir();
    let filename = format!("{}_rhmc_brain.json", sanitize_gpu_name(&state.gpu_name));
    let path = dir.join(filename);
    let json = serde_json::to_string_pretty(state).map_err(std::io::Error::other)?;
    std::fs::write(&path, json)?;
    eprintln!(
        "[brain] Saved state for {} → {}",
        state.gpu_name,
        path.display()
    );
    Ok(())
}

/// Load brain state for a GPU (returns None if not found).
pub fn load_brain_state(gpu_name: &str) -> Option<BrainState> {
    let dir = brain_state_dir();
    let filename = format!("{}_rhmc_brain.json", sanitize_gpu_name(gpu_name));
    let path = dir.join(filename);
    let json = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&json).ok()
}

/// Build a `BrainState` snapshot from current runner state.
pub fn snapshot_brain_state(
    gpu_name: &str,
    weights: &ExportedWeights,
    config_live: &RhmcConfigLive,
    history: &VecDeque<TrajectoryObservation>,
) -> BrainState {
    let n = history.len().max(1) as f64;
    let mean_acceptance = history.iter().filter(|o| o.accepted).count() as f64 / n;
    let mean_plaquette = history.iter().map(|o| o.plaquette).sum::<f64>() / n;

    let cfg = config_live.config_for_gpu(gpu_name);

    BrainState {
        gpu_name: gpu_name.to_string(),
        esn_weights: SerializableWeights::from(weights),
        optimal_dt: cfg.dt,
        optimal_n_md: cfg.n_md_steps,
        optimal_cg_tol: cfg.cg_tol,
        trajectories_observed: history.len(),
        mean_acceptance,
        mean_plaquette,
    }
}
