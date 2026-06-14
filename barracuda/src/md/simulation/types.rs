// SPDX-License-Identifier: AGPL-3.0-or-later

//! Simulation result types: energy records, brain summary, and run output.

use crate::md::config::MdConfig;

/// Per-step energy record.
#[derive(Clone, Debug)]
pub struct EnergyRecord {
    /// MD step index.
    pub step: usize,
    /// Kinetic energy (reduced units).
    pub ke: f64,
    /// Potential energy (reduced units).
    pub pe: f64,
    /// Total energy (ke + pe).
    pub total: f64,
    /// Instantaneous temperature T* = 2 KE / (3 N `k_B`).
    pub temperature: f64,
}

/// Brain training summary from an MD run.
#[derive(Clone, Debug, Default)]
pub struct BrainSummary {
    /// Number of readout retrains (adaptive interval) during the run.
    pub retrain_count: usize,
    /// Number of trusted heads at run end (out of 12 MD heads).
    pub trusted_heads: usize,
    /// Overall brain confidence at run end (0.0-1.0).
    pub confidence: f64,
    /// Per-head R² scores at run end.
    pub head_r2: Vec<f64>,
    /// Whether the brain detected any anomalies during the run.
    pub anomaly_detected: bool,
    /// Nautilus shell JSON for cross-run cumulative learning. `None` if never evolved.
    pub nautilus_json: Option<String>,
    /// Number of cumulative Nautilus observations.
    pub nautilus_observations: usize,
    /// Number of Nautilus generations evolved.
    pub nautilus_generations: usize,
}

/// Simulation state and results.
#[derive(Debug)]
pub struct MdSimulation {
    /// MD configuration (N, box, Γ, κ, etc.).
    pub config: MdConfig,
    /// Energy records per dump step.
    pub energy_history: Vec<EnergyRecord>,
    /// Position snapshots for RDF/VACF post-processing.
    pub positions_snapshots: Vec<Vec<f64>>,
    /// Velocity snapshots for VACF and transport.
    pub velocity_snapshots: Vec<Vec<f64>>,
    /// RDF histogram (raw pair counts).
    pub rdf_histogram: Vec<u64>,
    /// Total wall time (seconds).
    pub wall_time_s: f64,
    /// Throughput (steps per second).
    pub steps_per_sec: f64,
    /// Brain training summary (None if brain was not used).
    pub brain_summary: Option<BrainSummary>,
}
