// SPDX-License-Identifier: AGPL-3.0-only

//! MD Brain Module — Nautilus evolutionary reservoir for molecular dynamics
//!
//! Single reservoir architecture: the Nautilus shell (evolutionary Bingo board
//! population + ridge regression readout) handles both within-run learning and
//! cross-run cumulative evolution. No separate ESN — all learning flows through
//! Nautilus.
//!
//! # Architecture
//!
//! ```text
//! MD simulation loop
//!   → MdStepEvent (energy, temperature, rebuild_count, algorithm, steps/s)
//!   → MdBrain (Nautilus reservoir + adaptive readout + confidence tracking)
//!   → MdSteering (skin fraction, dump interval, equilibrium detection)
//! ```
//!
//! # Adaptive readout retraining
//!
//! Instead of a fixed retrain interval, the readout retrains based on data
//! density: fast when target variance is high (early run, transitions),
//! slow when data is stable (equilibrated). Nautilus board evolution runs on
//! a slower cycle (every 500 observations) for structural adaptation.

use crate::md::neighbor::ForceAlgorithm;
use crate::tolerances::{
    BRAIN_EQUILIBRIUM_THRESHOLD, MD_ENERGY_FLOOR, MD_TARGET_TEMPERATURE_GUARD, VERLET_SKIN_FRACTION,
};
use barracuda::nautilus::{InstanceId, LinearReadout, NautilusShell, ReservoirInput, ShellConfig};

// ═══════════════════════════════════════════════════════════════════
//  MD Head Layout — 12 heads organized by domain
// ═══════════════════════════════════════════════════════════════════

/// Energy drift prediction (learned from trajectory stream).
pub const A0_ENERGY_DRIFT: usize = 0;
/// Temperature deviation from target.
pub const A1_TEMPERATURE_DEVIATION: usize = 1;
/// Equilibration progress (0 = not started, 1 = converged).
pub const A2_EQUILIBRIUM_PROGRESS: usize = 2;
/// Steps/second performance prediction.
pub const B0_STEPS_PER_SEC_PREDICT: usize = 3;
/// Verlet rebuild frequency prediction.
pub const B1_REBUILD_FREQUENCY: usize = 4;
/// GPU memory pressure estimate.
pub const B2_MEMORY_PRESSURE: usize = 5;
/// Optimal Verlet skin fraction steering.
pub const C0_OPTIMAL_SKIN: usize = 6;
/// Optimal dump interval steering.
pub const C1_OPTIMAL_DUMP_INTERVAL: usize = 7;
/// Algorithm effectiveness score (`AllPairs` vs `CellList` vs Verlet).
pub const C2_ALGORITHM_SCORE: usize = 8;
/// Energy anomaly detector (drift or explosion).
pub const D0_ENERGY_ANOMALY: usize = 9;
/// Force anomaly detector (divergence or NaN).
pub const D1_FORCE_ANOMALY: usize = 10;
/// Convergence flag (equilibrium reached).
pub const D2_CONVERGENCE_FLAG: usize = 11;
/// Total number of MD brain heads.
pub const MD_NUM_HEADS: usize = 12;
/// Input vector dimensionality for the MD brain reservoir.
pub const MD_INPUT_DIM: usize = 14;

// ═══════════════════════════════════════════════════════════════════
//  MD Step Event — streamed per dump interval
// ═══════════════════════════════════════════════════════════════════

/// Per-dump-interval observation from the MD simulation loop.
#[derive(Clone, Debug)]
pub struct MdStepEvent {
    /// Current MD step index.
    pub step: usize,
    /// Kinetic energy (reduced units).
    pub ke: f64,
    /// Potential energy (reduced units).
    pub pe: f64,
    /// Total energy (KE + PE).
    pub total_energy: f64,
    /// Instantaneous temperature T* = 2 KE / (3 N).
    pub temperature: f64,
    /// Target temperature T* = 1/Gamma.
    pub target_temperature: f64,
    /// Screening parameter kappa.
    pub kappa: f64,
    /// Coupling parameter Gamma.
    pub gamma: f64,
    /// Number of particles.
    pub n_particles: usize,
    /// Force computation algorithm in use.
    pub algorithm: ForceAlgorithm,
    /// Cumulative Verlet rebuild count.
    pub rebuild_count: usize,
    /// Current throughput (steps/second).
    pub steps_per_sec: f64,
    /// Elapsed wall time (seconds).
    pub wall_time_s: f64,
    /// Current Verlet skin fraction (skin / rc).
    pub skin_fraction: f64,
}

impl MdStepEvent {
    /// Encode into the 14D input vector for the reservoir.
    ///
    /// All values are normalized to approximately [-1, 1] for reservoir stability.
    #[must_use]
    pub fn to_input(&self) -> Vec<f64> {
        let algo_code = match self.algorithm {
            ForceAlgorithm::AllPairs => 0.0,
            ForceAlgorithm::CellList => 0.5,
            ForceAlgorithm::VerletList { .. } => 1.0,
        };
        vec![
            self.ke / self.n_particles as f64,
            self.pe / self.n_particles as f64,
            self.total_energy / self.n_particles as f64,
            self.temperature / self.target_temperature.max(MD_TARGET_TEMPERATURE_GUARD) - 1.0,
            self.kappa / 5.0,
            (self.gamma).ln() / 10.0,
            (self.n_particles as f64).ln() / 12.0,
            algo_code,
            (self.rebuild_count as f64).min(500.0) / 500.0,
            self.steps_per_sec.ln().max(0.0) / 10.0,
            self.wall_time_s.min(3600.0) / 3600.0,
            self.skin_fraction / 0.5,
            self.step as f64 / 100_000.0,
            (self.temperature - self.target_temperature).abs()
                / self.target_temperature.max(MD_TARGET_TEMPERATURE_GUARD),
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════
//  MD Steering — decisions output by the brain
// ═══════════════════════════════════════════════════════════════════

/// Steering decisions from the MD brain.
#[derive(Clone, Debug)]
pub struct MdSteering {
    /// Suggested Verlet skin fraction (0.1-0.5, default 0.2).
    pub skin_fraction: f64,
    /// Whether equilibration is detected as converged.
    pub equilibrium_converged: bool,
    /// Anomaly flag (energy explosion, force divergence).
    pub anomaly_detected: bool,
    /// Confidence level of the steering decision (0-1).
    pub confidence: f64,
}

impl Default for MdSteering {
    fn default() -> Self {
        Self {
            skin_fraction: VERLET_SKIN_FRACTION,
            equilibrium_converged: false,
            anomaly_detected: false,
            confidence: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  MD Head Confidence — rolling R² tracker
// ═══════════════════════════════════════════════════════════════════

struct MdHeadConfidence {
    predictions: Vec<Vec<f64>>,
    actuals: Vec<Vec<f64>>,
    trusted: Vec<bool>,
    r2: Vec<f64>,
    window: usize,
}

const CONFIDENCE_WINDOW: usize = 20;
const TRUST_THRESHOLD: f64 = 0.3;

impl MdHeadConfidence {
    fn new() -> Self {
        Self {
            predictions: (0..MD_NUM_HEADS)
                .map(|_| Vec::with_capacity(CONFIDENCE_WINDOW))
                .collect(),
            actuals: (0..MD_NUM_HEADS)
                .map(|_| Vec::with_capacity(CONFIDENCE_WINDOW))
                .collect(),
            trusted: vec![false; MD_NUM_HEADS],
            r2: vec![0.0; MD_NUM_HEADS],
            window: CONFIDENCE_WINDOW,
        }
    }

    fn record(&mut self, head: usize, predicted: f64, actual: f64) {
        if head >= MD_NUM_HEADS {
            return;
        }
        let pred_buf = &mut self.predictions[head];
        if pred_buf.len() >= self.window {
            pred_buf.remove(0);
        }
        pred_buf.push(predicted);

        let act_buf = &mut self.actuals[head];
        if act_buf.len() >= self.window {
            act_buf.remove(0);
        }
        act_buf.push(actual);

        self.recompute_r2(head);
    }

    fn recompute_r2(&mut self, head: usize) {
        let pred = &self.predictions[head];
        let actual = &self.actuals[head];
        let n = pred.len().min(actual.len());
        if n < 3 {
            self.trusted[head] = false;
            self.r2[head] = 0.0;
            return;
        }
        let offset_p = pred.len().saturating_sub(n);
        let offset_a = actual.len().saturating_sub(n);
        let mean_a: f64 = actual[offset_a..].iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = actual[offset_a..]
            .iter()
            .map(|a| (a - mean_a).powi(2))
            .sum();
        if ss_tot < MD_ENERGY_FLOOR {
            self.r2[head] = 0.0;
            self.trusted[head] = false;
            return;
        }
        let ss_res: f64 = pred[offset_p..]
            .iter()
            .zip(actual[offset_a..].iter())
            .map(|(p, a)| (a - p).powi(2))
            .sum();
        self.r2[head] = 1.0 - ss_res / ss_tot;
        self.trusted[head] = self.r2[head] >= TRUST_THRESHOLD;
    }

    fn is_trusted(&self, head: usize) -> bool {
        head < MD_NUM_HEADS && self.trusted[head]
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Nautilus reservoir input/target encoding
// ═══════════════════════════════════════════════════════════════════

fn md_event_to_reservoir_input(event: &MdStepEvent) -> ReservoirInput {
    ReservoirInput::Continuous(vec![
        event.ke / event.n_particles as f64,
        event.pe / event.n_particles as f64,
        event.temperature / event.target_temperature.max(MD_TARGET_TEMPERATURE_GUARD),
        event.kappa / 5.0,
        (event.gamma).ln() / 10.0,
        (event.n_particles as f64).ln() / 12.0,
        event.skin_fraction / 0.5,
        event.steps_per_sec.ln().max(0.0) / 10.0,
    ])
}

/// Reference memory for pressure scaling (4 GB).
const REF_MEMORY_BYTES: f64 = 4.0 * 1024.0 * 1024.0 * 1024.0;

/// Bytes per pair/neighbor (position + force components).
const BYTES_PER_ENTRY: f64 = 24.0;

/// Compute GPU memory pressure estimate from particle count and algorithm.
#[must_use]
fn memory_pressure(event: &MdStepEvent) -> f64 {
    let n = event.n_particles as f64;
    let bytes = match event.algorithm {
        ForceAlgorithm::AllPairs => n * n * BYTES_PER_ENTRY,
        ForceAlgorithm::CellList => n * 26.0 * BYTES_PER_ENTRY,
        ForceAlgorithm::VerletList { .. } => {
            let avg_neighbors = 26.0 * (1.0 + event.skin_fraction);
            n * avg_neighbors * BYTES_PER_ENTRY
        }
    };
    (bytes / REF_MEMORY_BYTES).clamp(0.0, 1.0)
}

/// Compute force anomaly from sudden energy jumps vs running stats over recent events.
#[must_use]
fn force_anomaly(event: &MdStepEvent, recent_energy_window: Option<&[f64]>) -> f64 {
    let current_ep = event.total_energy / event.n_particles as f64;
    let Some(window) = recent_energy_window else {
        return 0.0;
    };
    let take = window.len().min(10);
    if take < 3 {
        return 0.0;
    }
    let start = window.len().saturating_sub(take);
    let slice = &window[start..];
    let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
    let variance: f64 = slice.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / slice.len() as f64;
    let std = (variance + 1e-20).sqrt();
    let deviation = (current_ep - mean).abs();
    if deviation > 10.0 * std {
        1.0
    } else {
        0.0
    }
}

/// Build 12-dimensional target vector matching `MD_NUM_HEADS` for Nautilus readout.
fn md_event_to_target(
    event: &MdStepEvent,
    first_energy_per_particle: Option<f64>,
    recent_energy_window: Option<&[f64]>,
) -> Vec<f64> {
    let energy_drift = if let Some(first_e) = first_energy_per_particle {
        if first_e.abs() > MD_ENERGY_FLOOR {
            (event.total_energy / event.n_particles as f64 - first_e).abs() / first_e.abs()
        } else {
            0.0
        }
    } else {
        0.0
    };

    let temp_deviation = (event.temperature - event.target_temperature).abs()
        / event.target_temperature.max(MD_TARGET_TEMPERATURE_GUARD);

    let equil_progress = if temp_deviation < BRAIN_EQUILIBRIUM_THRESHOLD {
        1.0
    } else {
        (1.0 - temp_deviation).max(0.0)
    };

    let rebuild_norm = (event.rebuild_count as f64).min(500.0) / 500.0;
    let sps_norm = event.steps_per_sec.ln().max(0.0) / 10.0;

    vec![
        energy_drift.min(1.0),                    // A0: energy drift
        temp_deviation.min(1.0),                  // A1: temp deviation
        equil_progress,                           // A2: equilibrium progress
        sps_norm,                                 // B0: steps/s
        rebuild_norm,                             // B1: rebuild frequency
        memory_pressure(event),                   // B2: memory pressure
        event.skin_fraction / 0.5,                // C0: current skin
        (event.step as f64 / 100_000.0).min(1.0), // C1: dump interval
        match event.algorithm {
            // C2: algorithm effectiveness
            ForceAlgorithm::AllPairs => 0.0,
            ForceAlgorithm::CellList => 0.5,
            ForceAlgorithm::VerletList { .. } => 1.0,
        },
        (energy_drift * 100.0).min(1.0), // D0: energy anomaly
        force_anomaly(event, recent_energy_window), // D1: force anomaly
        equil_progress,                  // D2: convergence (mirrors A2)
    ]
}

// ═══════════════════════════════════════════════════════════════════
//  Adaptive Retrain Interval
// ═══════════════════════════════════════════════════════════════════

/// Compute target variance over recent observations to determine retrain pace.
fn target_variance(observations: &[(ReservoirInput, Vec<f64>)], window: usize) -> f64 {
    let n = observations.len();
    if n < 2 {
        return 1.0; // high variance default: retrain often when we have little data
    }
    let start = n.saturating_sub(window);
    let slice = &observations[start..];
    let count = slice.len() as f64;

    // Average variance across all target dimensions
    let n_dims = slice[0].1.len();
    if n_dims == 0 {
        return 0.0;
    }

    let mut total_var = 0.0;
    for d in 0..n_dims {
        let mean: f64 = slice.iter().map(|(_, t)| t[d]).sum::<f64>() / count;
        let var: f64 = slice
            .iter()
            .map(|(_, t)| (t[d] - mean).powi(2))
            .sum::<f64>()
            / count;
        total_var += var;
    }
    total_var / n_dims as f64
}

// ═══════════════════════════════════════════════════════════════════
//  MD Brain — Nautilus-only reservoir for MD simulations
// ═══════════════════════════════════════════════════════════════════

/// MD Brain: unified Nautilus evolutionary reservoir for molecular dynamics.
///
/// Maximum entries in the recent energy window for D1 force anomaly detection.
const RECENT_ENERGY_WINDOW_CAP: usize = 20;

/// MD Brain: unified Nautilus evolutionary reservoir for molecular dynamics.
///
/// Single learning system:
/// - **Nautilus** (cumulative): evolutionary reservoir computing with adaptive
///   readout retraining. Boards evolve structurally across generations.
///   Readout (ridge regression) retrains at adaptive intervals based on
///   data density — fast when data is changing, slow when stable.
pub struct MdBrain {
    confidence: MdHeadConfidence,
    /// Nautilus evolutionary shell — single reservoir for all learning.
    nautilus: NautilusShell,
    /// Accumulated observations (survives across runs via serialization).
    observations: Vec<(ReservoirInput, Vec<f64>)>,
    /// Number of Nautilus generations evolved.
    generations: usize,
    /// Number of readout retrains performed.
    readout_retrains: usize,
    /// Observation count at last readout retrain.
    last_retrain_at: usize,
    /// First energy/particle value (for drift calculation).
    first_energy_per_particle: Option<f64>,
    /// Whether the readout has been trained at least once.
    readout_trained: bool,
    /// Recent energy per particle (for D1 force anomaly: sudden jump detection).
    recent_energy_window: Vec<f64>,
}

impl MdBrain {
    /// Create a new MD brain with Nautilus-only configuration.
    #[must_use]
    pub fn new() -> Self {
        let config = ShellConfig {
            pop_size: 16,
            ..ShellConfig::default()
        };
        let mut nautilus = NautilusShell::from_seed(
            config.clone(),
            InstanceId("md-brain".to_string()),
            0xbadd_cafe,
        );
        let l2 = config.board_config.grid_size * config.board_config.grid_size;
        let input_dim = config.pop_size * l2;
        nautilus.readout = LinearReadout::new(input_dim, MD_NUM_HEADS, config.lambda);
        Self {
            confidence: MdHeadConfidence::new(),
            nautilus,
            observations: Vec::new(),
            generations: 0,
            readout_retrains: 0,
            last_retrain_at: 0,
            first_energy_per_particle: None,
            readout_trained: false,
            recent_energy_window: Vec::with_capacity(RECENT_ENERGY_WINDOW_CAP),
        }
    }

    /// Observe a step event and produce steering.
    ///
    /// The brain accumulates observations and retrains the Nautilus readout
    /// at adaptive intervals: fast when data variance is high, slow when stable.
    /// Board evolution runs on a slower cycle (every 500 observations).
    pub fn observe(&mut self, event: &MdStepEvent) -> MdSteering {
        // Track first energy for drift calculation
        if self.first_energy_per_particle.is_none() {
            self.first_energy_per_particle = Some(event.total_energy / event.n_particles as f64);
        }

        let naut_input = md_event_to_reservoir_input(event);
        let target = md_event_to_target(
            event,
            self.first_energy_per_particle,
            Some(&self.recent_energy_window),
        );
        self.observations.push((naut_input, target));

        // Update energy window for D1 force anomaly (use window from before this event)
        let current_ep = event.total_energy / event.n_particles as f64;
        if self.recent_energy_window.len() >= RECENT_ENERGY_WINDOW_CAP {
            self.recent_energy_window.remove(0);
        }
        self.recent_energy_window.push(current_ep);

        // Cap observations to prevent unbounded growth
        const MAX_OBS: usize = 2000;
        if self.observations.len() > MAX_OBS + 500 {
            let excess = self.observations.len() - MAX_OBS;
            let mut thinned: Vec<_> = self
                .observations
                .drain(..excess)
                .enumerate()
                .filter(|(i, _)| i % 3 == 0)
                .map(|(_, v)| v)
                .collect();
            thinned.append(&mut self.observations);
            self.observations = thinned;
        }

        // Evolve Nautilus boards periodically (structural adaptation)
        if self.observations.len() >= 20 && self.observations.len().is_multiple_of(500) {
            self.evolve_boards();
        }

        // Adaptive readout retrain
        if self.should_retrain() {
            self.retrain_readout();
        }

        // Predict and steer
        if self.readout_trained {
            let outputs = self.predict(event);
            let target = self.observations.last().map(|(_, t)| t.clone());
            if let Some(ref t) = target {
                self.update_confidence(t, &outputs);
            }
            return self.steer_from_outputs(&outputs, event);
        }

        Self::heuristic_steering(event)
    }

    /// Determine whether the readout should be retrained based on data density.
    fn should_retrain(&self) -> bool {
        let n = self.observations.len();
        if n < 20 {
            return false;
        }

        let since_last = n - self.last_retrain_at;

        // Immediate retrain on first 20 observations (cross-run bootstrap)
        if !self.readout_trained && n >= 20 {
            return true;
        }

        let variance = target_variance(&self.observations, 50);

        let interval = if variance > 0.1 {
            30 // dense/changing data: retrain often
        } else if variance > 0.01 {
            100 // moderate: standard pace
        } else {
            300 // stable/equilibrated: slow pace
        };

        since_last >= interval
    }

    /// Retrain only the Nautilus readout (ridge regression) on accumulated observations.
    fn retrain_readout(&mut self) {
        let (inputs, targets): (Vec<_>, Vec<_>) = self.observations.iter().cloned().unzip();
        let responses: Vec<Vec<f64>> = inputs
            .iter()
            .map(|inp| self.nautilus.population.respond_all(inp))
            .collect();

        if self.nautilus.readout.train(&responses, &targets).is_ok() {
            self.readout_retrains += 1;
            self.last_retrain_at = self.observations.len();
            self.readout_trained = true;
        }
    }

    /// Evolve the Nautilus board population (structural adaptation).
    fn evolve_boards(&mut self) {
        let (inputs, targets): (Vec<_>, Vec<_>) = self.observations.iter().cloned().unzip();
        for _ in 0..5 {
            if self.nautilus.evolve_generation(&inputs, &targets).is_ok() {
                self.generations += 1;
            }
        }
        self.readout_trained = true;
    }

    /// Predict via Nautilus shell for a given event.
    fn predict(&self, event: &MdStepEvent) -> Vec<f64> {
        let input = md_event_to_reservoir_input(event);
        self.nautilus
            .predict(&input)
            .unwrap_or_else(|| vec![0.0; MD_NUM_HEADS])
    }

    fn update_confidence(&mut self, target: &[f64], outputs: &[f64]) {
        for (h, (pred, actual)) in outputs.iter().zip(target.iter()).enumerate() {
            self.confidence.record(h, *pred, *actual);
        }
    }

    fn steer_from_outputs(&self, outputs: &[f64], event: &MdStepEvent) -> MdSteering {
        let skin = if self.confidence.is_trusted(C0_OPTIMAL_SKIN) && self.readout_retrains > 5 {
            (outputs[C0_OPTIMAL_SKIN] * 0.5).clamp(0.05, 0.5)
        } else {
            VERLET_SKIN_FRACTION
        };

        let equilibrium =
            if self.confidence.is_trusted(A2_EQUILIBRIUM_PROGRESS) && self.readout_retrains > 5 {
                outputs[A2_EQUILIBRIUM_PROGRESS] > 0.95
            } else {
                let temp_dev = (event.temperature - event.target_temperature).abs()
                    / event.target_temperature.max(MD_TARGET_TEMPERATURE_GUARD);
                temp_dev < BRAIN_EQUILIBRIUM_THRESHOLD
            };

        let anomaly = if self.confidence.is_trusted(D0_ENERGY_ANOMALY) && self.readout_retrains > 5
        {
            outputs[D0_ENERGY_ANOMALY] > 0.7
        } else {
            false
        };

        let trusted_count = (0..MD_NUM_HEADS)
            .filter(|&h| self.confidence.is_trusted(h))
            .count();

        MdSteering {
            skin_fraction: skin,
            equilibrium_converged: equilibrium,
            anomaly_detected: anomaly,
            confidence: trusted_count as f64 / MD_NUM_HEADS as f64,
        }
    }

    fn heuristic_steering(event: &MdStepEvent) -> MdSteering {
        let temp_dev = (event.temperature - event.target_temperature).abs()
            / event.target_temperature.max(MD_TARGET_TEMPERATURE_GUARD);

        MdSteering {
            skin_fraction: VERLET_SKIN_FRACTION,
            equilibrium_converged: temp_dev < BRAIN_EQUILIBRIUM_THRESHOLD,
            anomaly_detected: false,
            confidence: 0.0,
        }
    }

    /// Export the Nautilus shell + observations as JSON for cross-run persistence.
    #[must_use]
    pub fn export_nautilus_json(&self) -> Option<String> {
        serde_json::to_string(&(
            &self.nautilus,
            &self.observations,
            self.generations,
            self.readout_retrains,
        ))
        .ok()
    }

    /// Import Nautilus shell from JSON, restoring cumulative state.
    pub fn import_nautilus_json(&mut self, json: &str) -> bool {
        // Try new format first (with readout_retrains)
        if let Ok((shell, obs, gens, retrains)) =
            serde_json::from_str::<(NautilusShell, Vec<(ReservoirInput, Vec<f64>)>, usize, usize)>(
                json,
            )
        {
            self.nautilus = shell;
            self.observations = obs;
            self.generations = gens;
            self.readout_retrains = retrains;
            self.last_retrain_at = self.observations.len();
            self.readout_trained = retrains > 0;
            if let Some((_, ref t)) = self.observations.first() {
                if !t.is_empty() {
                    self.first_energy_per_particle = Some(t[A0_ENERGY_DRIFT]);
                }
            }
            return true;
        }
        // Backward compat: old format (shell, obs, gens) without retrains
        if let Ok((shell, obs, gens)) =
            serde_json::from_str::<(NautilusShell, Vec<(ReservoirInput, Vec<f64>)>, usize)>(json)
        {
            self.nautilus = shell;
            self.observations = obs;
            self.generations = gens;
            self.readout_retrains = 0;
            self.last_retrain_at = 0;
            self.readout_trained = gens > 0;
            return true;
        }
        false
    }

    /// Number of readout retrains completed (replaces old `trained_count`).
    #[must_use]
    pub const fn readout_retrain_count(&self) -> usize {
        self.readout_retrains
    }

    /// Number of Nautilus generations evolved.
    #[must_use]
    pub const fn nautilus_generations(&self) -> usize {
        self.generations
    }

    /// Number of cumulative observations (persists across runs).
    #[must_use]
    pub const fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Per-head R² confidence scores.
    #[must_use]
    pub fn head_confidence(&self) -> &[f64] {
        &self.confidence.r2
    }

    /// Per-head trust status.
    #[must_use]
    pub fn head_trust(&self) -> &[bool] {
        &self.confidence.trusted
    }
}

impl Default for MdBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MdBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MdBrain")
            .field("observations", &self.observations.len())
            .field("generations", &self.generations)
            .field("readout_retrains", &self.readout_retrains)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(step: usize, ke: f64, pe: f64, temp: f64) -> MdStepEvent {
        MdStepEvent {
            step,
            ke,
            pe,
            total_energy: ke + pe,
            temperature: temp,
            target_temperature: 0.01,
            kappa: 2.0,
            gamma: 100.0,
            n_particles: 2000,
            algorithm: ForceAlgorithm::VerletList { skin: 1.3 },
            rebuild_count: step / 10,
            steps_per_sec: 500.0,
            wall_time_s: step as f64 * 0.002,
            skin_fraction: 0.2,
        }
    }

    #[test]
    fn brain_accumulates_events() {
        let mut brain = MdBrain::new();
        for i in 0..10 {
            let event = make_event(i * 10, 1.0, -5.0, 0.01);
            let steering = brain.observe(&event);
            assert!(!steering.anomaly_detected);
        }
        assert_eq!(brain.observation_count(), 10);
    }

    #[test]
    fn input_encoding_dimensions() {
        let event = make_event(100, 1.0, -5.0, 0.01);
        let input = event.to_input();
        assert_eq!(input.len(), MD_INPUT_DIM);
    }

    #[test]
    fn heuristic_steering_converges() {
        let brain = MdBrain::new();
        let event = make_event(100, 1.0, -5.0, 0.01);
        let steering = MdBrain::heuristic_steering(&event);
        assert!(steering.equilibrium_converged);
        assert!(!steering.anomaly_detected);
        assert!((steering.skin_fraction - VERLET_SKIN_FRACTION).abs() < 1e-10);
    }

    #[test]
    fn brain_retrains_at_threshold() {
        let mut brain = MdBrain::new();
        for i in 0..50 {
            let event = make_event(i * 10, 1.0 + i as f64 * 0.01, -5.0, 0.01);
            brain.observe(&event);
        }
        assert!(brain.readout_retrain_count() >= 1);
    }

    #[test]
    fn target_has_12_dimensions() {
        let event = make_event(100, 1.0, -5.0, 0.01);
        let target = md_event_to_target(&event, Some(-4.0 / 2000.0), None);
        assert_eq!(target.len(), MD_NUM_HEADS);
    }

    #[test]
    fn adaptive_interval_fast_for_high_variance() {
        let var = target_variance(&[], 50);
        assert!(var >= 1.0, "empty obs should give high default variance");
    }
}
