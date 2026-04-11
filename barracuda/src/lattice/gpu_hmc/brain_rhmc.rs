// SPDX-License-Identifier: AGPL-3.0-or-later

//! Brain-steered RHMC: NPU cortex drives GPU physics parameters.
//!
//! The `NpuCortex` runs on a dedicated thread, observing trajectory outcomes
//! from both GPUs and producing parameter suggestions via the MultiHeadNpu
//! ESN.
//!
//! ## Hardware-first NPU
//!
//! `NpuInference` uses the Akida hardware NPU as the primary inference path
//! when available, falling back to software f32 only when no hardware is
//! present. By streaming observations from *both* GPUs through a single
//! unified physics stream, the NPU learns the physics parameters (dt,
//! n_md, CG tolerance) rather than hardware-specific quirks — and gets 2×
//! the learning opportunities per trajectory pair.
//!
//! ## Cross-GPU agreement
//!
//! When both GPUs run the same physics point (same beta, mass, lattice),
//! the cortex tracks whether they agree on acceptance and ΔH. Agreement
//! signals a robust parameter region; disagreement indicates sensitivity
//! to precision differences (native f64 vs DF64) and triggers conservative
//! steering.
//!
//! The `BrainRhmcRunner` orchestrates the loop: fire dual-GPU trajectories,
//! stream observations to the cortex, apply parameter suggestions with
//! safety clamps, and persist state for cross-hardware learning.

use crate::error::HotSpringError;
use crate::lattice::rhmc::RhmcConfig;
use crate::md::reservoir::heads;
use crate::md::reservoir::npu::{ExportedWeights, MultiHeadNpu};
use crate::tolerances::{ADAPTIVE_DT_MAX, ADAPTIVE_DT_MIN, ADAPTIVE_NMD_MAX, ADAPTIVE_NMD_MIN};

use super::brain_persistence::{BrainState, SerializableWeights, save_brain_state};
use super::unidirectional_cortex::{TrajectoryResult, UnidirectionalRhmc};

use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
//  Observable packet: GPU → NPU cortex
// ═══════════════════════════════════════════════════════════════════

/// Trajectory outcome streamed from a GPU to the NPU cortex.
#[derive(Debug, Clone)]
pub struct TrajectoryObservation {
    pub gpu_name: String,
    pub traj_idx: usize,
    pub accepted: bool,
    pub delta_h: f64,
    pub plaquette: f64,
    pub total_cg_iters: usize,
    pub elapsed_secs: f64,
    pub beta: f64,
    pub mass: f64,
    pub dt: f64,
    pub n_md_steps: usize,
    pub lattice_size: usize,
    /// Silicon routing state active during this trajectory.
    pub silicon_tags: SiliconRoutingTags,
}

/// Which silicon units were active for each trajectory phase.
///
/// Fed to the ESN so the NPU can learn performance correlations
/// between hardware routing decisions and physics outcomes.
#[derive(Debug, Clone, Default)]
pub struct SiliconRoutingTags {
    /// TMU-accelerated PRNG (Tier 0) was used for momentum generation.
    pub tmu_prng: bool,
    /// Subgroup reduce (Tier 4) was used for CG dot products.
    pub subgroup_reduce: bool,
    /// ROP atomic force accumulation (Tier 3) was active.
    pub rop_force_accum: bool,
    /// FP64 strategy: 0=native, 1=DF64, 2=hybrid, 3=concurrent.
    pub fp64_strategy_id: u8,
    /// Whether native SHADER_F64 was available.
    pub has_native_f64: bool,
}

// ═══════════════════════════════════════════════════════════════════
//  Parameter suggestion: NPU cortex → GPU
// ═══════════════════════════════════════════════════════════════════

/// Where a parameter suggestion originated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionSource {
    NpuModel,
    Heuristic,
    Default,
}

/// NPU cortex suggestion for RHMC parameters.
#[derive(Debug, Clone)]
pub struct RhmcParamSuggestion {
    pub dt: f64,
    pub n_md_steps: usize,
    pub cg_tol: f64,
    pub confidence: f64,
    pub source: SuggestionSource,
    pub gpu_name: String,
}

// ═══════════════════════════════════════════════════════════════════
//  NPU inference: hardware-first, software fallback
// ═══════════════════════════════════════════════════════════════════

/// Which backend actually performed the inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuBackend {
    /// Akida AKD1000 via PCIe (int8/int4).
    Hardware,
    /// NpuSimulator on CPU (f32).
    Software,
}

/// Metrics from a single NPU inference.
#[derive(Debug, Clone)]
pub struct NpuInferenceMetrics {
    pub backend: NpuBackend,
    pub latency_us: u64,
    pub energy_uj: f64,
    pub observation_count: usize,
}

/// Running NPU statistics accumulated over a run.
#[derive(Debug, Clone)]
pub struct NpuRunStats {
    pub inferences: usize,
    pub suggestions_applied: usize,
    pub mean_latency_us: f64,
    pub mean_energy_uj: f64,
    pub backend: NpuBackend,
    pub cross_gpu_agreements: usize,
    pub cross_gpu_disagreements: usize,
    pub total_observations: usize,
}

impl NpuRunStats {
    fn new(backend: NpuBackend) -> Self {
        Self {
            inferences: 0,
            suggestions_applied: 0,
            mean_latency_us: 0.0,
            mean_energy_uj: 0.0,
            backend,
            cross_gpu_agreements: 0,
            cross_gpu_disagreements: 0,
            total_observations: 0,
        }
    }

    fn update_inference(&mut self, metrics: &NpuInferenceMetrics) {
        self.inferences += 1;
        let n = self.inferences as f64;
        self.mean_latency_us += (metrics.latency_us as f64 - self.mean_latency_us) / n;
        self.mean_energy_uj += (metrics.energy_uj - self.mean_energy_uj) / n;
    }

    /// Format summary for terminal output.
    pub fn display_table(&self, traj_range: &str) -> String {
        let backend_label = match self.backend {
            NpuBackend::Hardware => "Akida (int8)",
            NpuBackend::Software => "CPU (f32)",
        };
        let agree_rate = if self.cross_gpu_agreements + self.cross_gpu_disagreements > 0 {
            self.cross_gpu_agreements as f64
                / (self.cross_gpu_agreements + self.cross_gpu_disagreements) as f64
        } else {
            1.0
        };
        format!(
            "\
╭─ NPU Brain ({traj_range}) ─────────────────────────────────────────╮
│ Backend             │ {:<50} │
│ Mean latency        │ {:<50} │
│ Mean energy/infer   │ {:<50} │
│ Inferences          │ {:<50} │
│ Observations (2/tr) │ {:<50} │
│ Suggestions applied │ {:<50} │
│ Cross-GPU agreement │ {:<50} │
╰─────────────────────┴────────────────────────────────────────────────╯",
            backend_label,
            format!("{:.0} us", self.mean_latency_us),
            format!("{:.2} uJ", self.mean_energy_uj),
            self.inferences,
            self.total_observations,
            self.suggestions_applied,
            format!(
                "{:.0}% ({}/{})",
                agree_rate * 100.0,
                self.cross_gpu_agreements,
                self.cross_gpu_agreements + self.cross_gpu_disagreements
            ),
        )
    }
}

/// Hardware-first NPU inference engine.
///
/// Uses Akida hardware as the primary inference path when the `npu-hw`
/// feature is active and an AKD1000 is present. Falls back to software
/// `MultiHeadNpu` otherwise. No dual-path overhead — one backend per run.
pub struct NpuInference {
    sw_npu: MultiHeadNpu,
    #[cfg(feature = "npu-hw")]
    hw_npu: Option<crate::md::npu_hw::NpuHardware>,
    cpu_tdp_per_core_w: f64,
}

impl NpuInference {
    /// Create the inference engine, preferring hardware.
    pub fn new(weights: &ExportedWeights) -> Self {
        let sw_npu = MultiHeadNpu::from_exported(weights);

        #[cfg(feature = "npu-hw")]
        let hw_npu = {
            use crate::md::npu_hw::NpuHardware;
            match NpuHardware::discover() {
                Some(info) => {
                    eprintln!(
                        "[brain] Akida NPU PRIMARY: {} NPUs, {} MB SRAM, PCIe {}",
                        info.npu_count, info.memory_mb, info.pcie_address
                    );
                    Some(NpuHardware::from_exported(weights, info))
                }
                None => {
                    eprintln!("[brain] No Akida NPU — falling back to software ESN");
                    None
                }
            }
        };

        Self {
            sw_npu,
            #[cfg(feature = "npu-hw")]
            hw_npu,
            cpu_tdp_per_core_w: 4.0,
        }
    }

    /// Which backend is active.
    pub fn active_backend(&self) -> NpuBackend {
        #[cfg(feature = "npu-hw")]
        if self.hw_npu.is_some() {
            return NpuBackend::Hardware;
        }
        NpuBackend::Software
    }

    /// Run inference on the active backend.
    pub fn infer(&mut self, input_sequence: &[Vec<f64>]) -> (Vec<f64>, NpuInferenceMetrics) {
        #[cfg(feature = "npu-hw")]
        if let Some(ref mut hw) = self.hw_npu {
            let start = Instant::now();
            let outputs = hw.predict(input_sequence);
            let latency_us = start.elapsed().as_micros() as u64;
            let energy_uj = 300.0 * latency_us as f64 / 1_000.0; // AKD1000 ~300mW
            return (
                outputs,
                NpuInferenceMetrics {
                    backend: NpuBackend::Hardware,
                    latency_us,
                    energy_uj,
                    observation_count: input_sequence.len(),
                },
            );
        }

        let start = Instant::now();
        let outputs = self.sw_npu.predict_all_heads(input_sequence);
        let latency_us = start.elapsed().as_micros() as u64;
        let energy_uj = latency_us as f64 * self.cpu_tdp_per_core_w / 1_000.0;
        (
            outputs,
            NpuInferenceMetrics {
                backend: NpuBackend::Software,
                latency_us,
                energy_uj,
                observation_count: input_sequence.len(),
            },
        )
    }
}

/// Cross-GPU agreement for a single trajectory pair.
#[derive(Debug, Clone)]
pub struct CrossGpuAgreement {
    pub traj_idx: usize,
    pub both_accepted: bool,
    pub both_rejected: bool,
    pub delta_h_spread: f64,
    pub plaquette_spread: f64,
}

impl CrossGpuAgreement {
    /// Both GPUs agree on acceptance → robust parameter region.
    pub fn acceptance_agrees(&self) -> bool {
        self.both_accepted || self.both_rejected
    }
}

// ═══════════════════════════════════════════════════════════════════
//  NPU Cortex: brain thread
// ═══════════════════════════════════════════════════════════════════

const TRUST_THRESHOLD: f64 = 0.3;
const OBSERVATION_WINDOW: usize = 20;
const CG_TOL_DEFAULT: f64 = 1e-10;
const CG_TOL_MIN: f64 = 1e-14;
const CG_TOL_MAX: f64 = 1e-6;

/// The NPU cortex runs on a dedicated thread, processing trajectory
/// observations from all GPUs through a unified physics stream.
///
/// By merging observations from both GPUs into one ESN input sequence,
/// the NPU sees the physics (beta, mass, plaquette, ΔH, acceptance)
/// rather than hardware identity — and gets 2× the learning rate.
pub struct NpuCortex {
    npu: NpuInference,
    observation_rx: mpsc::Receiver<TrajectoryObservation>,
    suggestion_tx: mpsc::Sender<RhmcParamSuggestion>,
    stats: NpuRunStats,
    /// Per-GPU history (for per-GPU acceptance rates).
    per_gpu_history: std::collections::HashMap<String, VecDeque<TrajectoryObservation>>,
    /// Unified physics stream — observations from ALL GPUs interleaved
    /// chronologically. The NPU learns physics, not hardware.
    unified_history: VecDeque<TrajectoryObservation>,
    /// Pending observations at the same traj_idx, waiting for the pair.
    pending_pair: Option<TrajectoryObservation>,
    /// Cross-GPU agreement log.
    agreement_log: Vec<CrossGpuAgreement>,
    report_interval: usize,
    checkpoint_interval: usize,
    esn_weights: ExportedWeights,
}

impl NpuCortex {
    /// Create cortex with hardware-first NPU.
    pub fn new(
        weights: &ExportedWeights,
        observation_rx: mpsc::Receiver<TrajectoryObservation>,
        suggestion_tx: mpsc::Sender<RhmcParamSuggestion>,
        report_interval: usize,
        checkpoint_interval: usize,
    ) -> Self {
        let npu = NpuInference::new(weights);
        let backend = npu.active_backend();
        Self {
            npu,
            observation_rx,
            suggestion_tx,
            stats: NpuRunStats::new(backend),
            per_gpu_history: std::collections::HashMap::new(),
            unified_history: VecDeque::new(),
            pending_pair: None,
            agreement_log: Vec::new(),
            report_interval,
            checkpoint_interval,
            esn_weights: weights.clone(),
        }
    }

    /// Run the cortex event loop (blocking — call from a dedicated thread).
    pub fn run(&mut self) {
        eprintln!(
            "[brain] NpuCortex started — backend={:?}, physics-unified stream",
            self.npu.active_backend()
        );

        loop {
            let Ok(obs) = self.observation_rx.recv() else {
                eprintln!("[brain] Observation channel closed — shutting down");
                self.checkpoint_all();
                break;
            };

            self.stats.total_observations += 1;

            // Track per-GPU history
            let gpu_history = self
                .per_gpu_history
                .entry(obs.gpu_name.clone())
                .or_default();
            gpu_history.push_back(obs.clone());
            if gpu_history.len() > OBSERVATION_WINDOW {
                gpu_history.pop_front();
            }

            // Track cross-GPU agreement for paired observations
            self.check_cross_gpu_agreement(&obs);

            // Push into unified physics stream (both GPUs interleaved)
            self.unified_history.push_back(obs.clone());
            if self.unified_history.len() > OBSERVATION_WINDOW * 2 {
                self.unified_history.pop_front();
            }

            // Run NPU inference on the unified physics stream.
            // The ESN sees physics from both GPUs — 2x learning opportunities.
            let input_seq = build_esn_input_sequence(&self.unified_history);
            let (outputs, metrics) = self.npu.infer(&input_seq);
            self.stats.update_inference(&metrics);

            // Produce physics-level suggestions (applied to both GPUs)
            if let Some(suggestion) = self.produce_suggestion(&outputs, &obs) {
                self.stats.suggestions_applied += 1;
                let _ = self.suggestion_tx.send(suggestion);
            }

            // Periodic reporting
            if self
                .stats
                .total_observations
                .is_multiple_of(self.report_interval)
            {
                let range = format!(
                    "obs {}-{}",
                    self.stats
                        .total_observations
                        .saturating_sub(self.report_interval)
                        + 1,
                    self.stats.total_observations
                );
                eprintln!("{}", self.stats.display_table(&range));
            }

            // Periodic checkpoint
            if self.checkpoint_interval > 0
                && self
                    .stats
                    .total_observations
                    .is_multiple_of(self.checkpoint_interval)
            {
                self.checkpoint_all();
            }
        }
    }

    fn check_cross_gpu_agreement(&mut self, obs: &TrajectoryObservation) {
        if let Some(ref pending) = self.pending_pair
            && pending.traj_idx == obs.traj_idx
            && pending.gpu_name != obs.gpu_name
        {
            let agreement = CrossGpuAgreement {
                traj_idx: obs.traj_idx,
                both_accepted: pending.accepted && obs.accepted,
                both_rejected: !pending.accepted && !obs.accepted,
                delta_h_spread: (pending.delta_h - obs.delta_h).abs(),
                plaquette_spread: (pending.plaquette - obs.plaquette).abs(),
            };
            if agreement.acceptance_agrees() {
                self.stats.cross_gpu_agreements += 1;
            } else {
                self.stats.cross_gpu_disagreements += 1;
            }
            self.agreement_log.push(agreement);
            self.pending_pair = None;
            return;
        }
        self.pending_pair = Some(obs.clone());
    }

    fn produce_suggestion(
        &self,
        outputs: &[f64],
        obs: &TrajectoryObservation,
    ) -> Option<RhmcParamSuggestion> {
        if obs.accepted && obs.delta_h.abs() > 5.0 {
            return None;
        }

        // Use the unified history length for confidence — more observations = more trust
        if self.unified_history.len() < 10 {
            return None;
        }

        // Physics-level acceptance rate (across all GPUs at this physics point)
        let acceptance_rate = self.unified_history.iter().filter(|o| o.accepted).count() as f64
            / self.unified_history.len() as f64;

        if acceptance_rate < 0.3 || obs.delta_h.abs() > 10.0 {
            return None;
        }

        let dt_head = if outputs.len() > heads::D1_OPTIMAL_DT {
            outputs[heads::D1_OPTIMAL_DT]
        } else {
            return None;
        };
        let nmd_head = if outputs.len() > heads::D2_OPTIMAL_NMD {
            outputs[heads::D2_OPTIMAL_NMD]
        } else {
            0.0
        };
        let anomaly_head = if outputs.len() > heads::B3_QCD_ANOMALY {
            outputs[heads::B3_QCD_ANOMALY]
        } else {
            0.0
        };
        let cg_head = if outputs.len() > heads::B0_QCD_CG_COST {
            outputs[heads::B0_QCD_CG_COST]
        } else {
            0.0
        };

        // Cross-GPU agreement boosts confidence; disagreement suppresses it
        let agree_total = self.stats.cross_gpu_agreements + self.stats.cross_gpu_disagreements;
        let cross_gpu_factor = if agree_total > 5 {
            self.stats.cross_gpu_agreements as f64 / agree_total as f64
        } else {
            0.5
        };

        if anomaly_head > 0.7 {
            let emergency_dt = (obs.dt * 0.5).max(ADAPTIVE_DT_MIN);
            return Some(RhmcParamSuggestion {
                dt: emergency_dt,
                n_md_steps: obs.n_md_steps,
                cg_tol: obs.dt.min(CG_TOL_DEFAULT),
                confidence: 0.9,
                source: SuggestionSource::NpuModel,
                gpu_name: obs.gpu_name.clone(),
            });
        }

        let dt_suggest = dt_head.abs().mul_add(0.05, ADAPTIVE_DT_MIN);
        if !(ADAPTIVE_DT_MIN..=ADAPTIVE_DT_MAX).contains(&dt_suggest) {
            return None;
        }

        let target_tau = obs.dt * obs.n_md_steps as f64;
        let n_md_suggest = if nmd_head.abs() > 0.01 {
            let raw = (nmd_head.abs() * 500.0).round() as usize;
            raw.clamp(ADAPTIVE_NMD_MIN, ADAPTIVE_NMD_MAX)
        } else {
            ((target_tau / dt_suggest).round() as usize).clamp(ADAPTIVE_NMD_MIN, ADAPTIVE_NMD_MAX)
        };

        let cg_tol_suggest = if cg_head.abs() > 0.01 {
            let log_tol = -10.0 + cg_head * 4.0;
            10.0_f64.powf(log_tol).clamp(CG_TOL_MIN, CG_TOL_MAX)
        } else {
            CG_TOL_DEFAULT
        };

        let confidence = acceptance_rate
            * (1.0 - anomaly_head).max(0.0)
            * (self.unified_history.len() as f64 / (OBSERVATION_WINDOW * 2) as f64).min(1.0)
            * cross_gpu_factor;

        if confidence < TRUST_THRESHOLD {
            return None;
        }

        Some(RhmcParamSuggestion {
            dt: dt_suggest,
            n_md_steps: n_md_suggest,
            cg_tol: cg_tol_suggest,
            confidence,
            source: SuggestionSource::NpuModel,
            gpu_name: obs.gpu_name.clone(),
        })
    }

    fn checkpoint_all(&self) {
        for (gpu_name, history) in &self.per_gpu_history {
            let state = BrainState {
                gpu_name: gpu_name.clone(),
                esn_weights: SerializableWeights::from(&self.esn_weights),
                optimal_dt: history.back().map_or(0.005, |o| o.dt),
                optimal_n_md: history.back().map_or(1, |o| o.n_md_steps),
                optimal_cg_tol: CG_TOL_DEFAULT,
                trajectories_observed: history.len(),
                mean_acceptance: history.iter().filter(|o| o.accepted).count() as f64
                    / history.len().max(1) as f64,
                mean_plaquette: history.iter().map(|o| o.plaquette).sum::<f64>()
                    / history.len().max(1) as f64,
            };
            let _ = save_brain_state(&state);
        }
    }

    /// Get the current run statistics.
    pub fn stats(&self) -> &NpuRunStats {
        &self.stats
    }
}

fn build_esn_input_sequence(history: &VecDeque<TrajectoryObservation>) -> Vec<Vec<f64>> {
    let window_size = 10;
    let start = history.len().saturating_sub(window_size);
    history
        .iter()
        .skip(start)
        .map(|obs| {
            crate::lattice::pseudofermion::npu_steering::npu_canonical_input_v2(
                obs.beta,
                obs.plaquette,
                obs.mass,
                obs.delta_h.abs().min(1000.0),
                if obs.accepted { 1.0 } else { 0.0 },
                obs.lattice_size,
                &obs.silicon_tags,
            )
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
//  Live config with safety clamps
// ═══════════════════════════════════════════════════════════════════

/// History entry for audit trail.
#[derive(Debug, Clone)]
pub struct AppliedSuggestion {
    pub traj_idx: usize,
    pub suggestion: RhmcParamSuggestion,
    pub old_dt: f64,
    pub old_n_md: usize,
}

/// Hot-updatable RHMC config with per-GPU specialization and safety clamps.
pub struct RhmcConfigLive {
    base_config: RhmcConfig,
    per_gpu_dt: std::collections::HashMap<String, f64>,
    per_gpu_nmd: std::collections::HashMap<String, usize>,
    per_gpu_cg_tol: std::collections::HashMap<String, f64>,
    suggestion_history: Vec<AppliedSuggestion>,
    suggestion_rx: mpsc::Receiver<RhmcParamSuggestion>,
}

impl RhmcConfigLive {
    pub fn new(
        base_config: RhmcConfig,
        suggestion_rx: mpsc::Receiver<RhmcParamSuggestion>,
    ) -> Self {
        Self {
            base_config,
            per_gpu_dt: std::collections::HashMap::new(),
            per_gpu_nmd: std::collections::HashMap::new(),
            per_gpu_cg_tol: std::collections::HashMap::new(),
            suggestion_history: Vec::new(),
            suggestion_rx,
        }
    }

    /// Drain pending suggestions and apply them.
    pub fn apply_pending(&mut self, traj_idx: usize) {
        while let Ok(suggestion) = self.suggestion_rx.try_recv() {
            let old_dt = self
                .per_gpu_dt
                .get(&suggestion.gpu_name)
                .copied()
                .unwrap_or(self.base_config.dt);
            let old_n_md = self
                .per_gpu_nmd
                .get(&suggestion.gpu_name)
                .copied()
                .unwrap_or(self.base_config.n_md_steps);

            // Rate-limit: max 50% change per suggestion to prevent cold-start ESN chaos
            let max_dt = (old_dt * 1.5).min(ADAPTIVE_DT_MAX);
            let min_dt = (old_dt * 0.5).max(ADAPTIVE_DT_MIN);
            let clamped_dt = suggestion.dt.clamp(min_dt, max_dt);
            let rate_max_nmd = ((old_n_md as f64 * 1.5).ceil() as usize).max(ADAPTIVE_NMD_MIN);
            let rate_min_nmd = ((old_n_md as f64 * 0.5).floor() as usize).max(1);
            let lo = rate_min_nmd.max(ADAPTIVE_NMD_MIN).min(rate_max_nmd);
            let hi = rate_max_nmd.min(ADAPTIVE_NMD_MAX).max(lo);
            let clamped_nmd = suggestion.n_md_steps.clamp(lo, hi);
            let clamped_cg = suggestion.cg_tol.clamp(CG_TOL_MIN, CG_TOL_MAX);

            self.per_gpu_dt
                .insert(suggestion.gpu_name.clone(), clamped_dt);
            self.per_gpu_nmd
                .insert(suggestion.gpu_name.clone(), clamped_nmd);
            self.per_gpu_cg_tol
                .insert(suggestion.gpu_name.clone(), clamped_cg);

            eprintln!(
                "[brain] Applied suggestion for {}: dt {old_dt:.6} → {clamped_dt:.6}, \
                 n_md {old_n_md} → {clamped_nmd}, cg_tol {clamped_cg:.2e} \
                 (confidence={:.2}, source={:?})",
                suggestion.gpu_name, suggestion.confidence, suggestion.source,
            );

            self.suggestion_history.push(AppliedSuggestion {
                traj_idx,
                suggestion,
                old_dt,
                old_n_md,
            });
        }
    }

    /// Get the config specialized for a given GPU.
    pub fn config_for_gpu(&self, gpu_name: &str) -> RhmcConfig {
        let mut cfg = self.base_config.clone();
        if let Some(&dt) = self.per_gpu_dt.get(gpu_name) {
            cfg.dt = dt;
        }
        if let Some(&nmd) = self.per_gpu_nmd.get(gpu_name) {
            cfg.n_md_steps = nmd;
        }
        if let Some(&tol) = self.per_gpu_cg_tol.get(gpu_name) {
            cfg.cg_tol = tol;
        }
        cfg
    }

    pub fn suggestion_history(&self) -> &[AppliedSuggestion] {
        &self.suggestion_history
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Brain RHMC runner: dual-GPU + NPU cortex orchestration
// ═══════════════════════════════════════════════════════════════════

/// Result from one brain-steered iteration (dual-GPU + NPU).
#[derive(Debug)]
pub struct BrainIterationResult {
    pub gpu_a: TrajectoryResult,
    pub gpu_b: TrajectoryResult,
    pub traj_idx: usize,
}

/// Orchestrates dual-GPU RHMC with NPU brain steering.
///
/// The runner owns:
/// - Two `UnidirectionalRhmc` instances (one per GPU)
/// - A live config that applies NPU suggestions
/// - Channels to the `NpuCortex` thread
pub struct BrainRhmcRunner {
    observation_tx: mpsc::Sender<TrajectoryObservation>,
    config_live: RhmcConfigLive,
    traj_count: usize,
    seed_a: u64,
    seed_b: u64,
}

impl BrainRhmcRunner {
    /// Create a new brain runner.
    ///
    /// Returns `(runner, observation_rx, suggestion_tx)` — the caller spawns
    /// the `NpuCortex` thread with the rx/tx endpoints.
    pub fn new(
        base_config: RhmcConfig,
        seed_a: u64,
        seed_b: u64,
    ) -> (
        Self,
        mpsc::Receiver<TrajectoryObservation>,
        mpsc::Sender<RhmcParamSuggestion>,
    ) {
        let (obs_tx, obs_rx) = mpsc::channel();
        let (sug_tx, sug_rx) = mpsc::channel();

        let runner = Self {
            observation_tx: obs_tx,
            config_live: RhmcConfigLive::new(base_config, sug_rx),
            traj_count: 0,
            seed_a,
            seed_b,
        };

        (runner, obs_rx, sug_tx)
    }

    /// Run one iteration: dual-GPU trajectories + observation streaming.
    ///
    /// # Errors
    ///
    /// Returns [`HotSpringError::ThreadPanicked`] if a worker thread panicked.
    pub fn run_iteration(
        &mut self,
        gpu_a: &mut UnidirectionalRhmc,
        gpu_b: &mut UnidirectionalRhmc,
    ) -> Result<BrainIterationResult, HotSpringError> {
        self.config_live.apply_pending(self.traj_count);

        let config_a = self.config_live.config_for_gpu(gpu_a.adapter_name());
        let config_b = self.config_live.config_for_gpu(gpu_b.adapter_name());

        let obs_beta_a = config_a.beta;
        let obs_mass_a = config_a.sectors.first().map_or(0.1, |s| s.mass);
        let obs_dt_a = config_a.dt;
        let obs_nmd_a = config_a.n_md_steps;
        let obs_beta_b = config_b.beta;
        let obs_mass_b = config_b.sectors.first().map_or(0.1, |s| s.mass);
        let obs_dt_b = config_b.dt;
        let obs_nmd_b = config_b.n_md_steps;

        let (result_a, result_b) = std::thread::scope(|scope| {
            let mut sa = self.seed_a;
            let mut sb = self.seed_b;
            let ga = &mut *gpu_a;
            let gb = &mut *gpu_b;

            let handle_a =
                scope.spawn(move || -> Result<(TrajectoryResult, u64), HotSpringError> {
                    let r = ga.run_trajectory(&config_a, &mut sa)?;
                    Ok((r, sa))
                });
            let handle_b =
                scope.spawn(move || -> Result<(TrajectoryResult, u64), HotSpringError> {
                    let r = gb.run_trajectory(&config_b, &mut sb)?;
                    Ok((r, sb))
                });

            let (ra, new_sa) = handle_a.join().map_err(|_| {
                HotSpringError::ThreadPanicked("GPU A trajectory thread panicked")
            })??;
            let (rb, new_sb) = handle_b.join().map_err(|_| {
                HotSpringError::ThreadPanicked("GPU B trajectory thread panicked")
            })??;
            self.seed_a = new_sa;
            self.seed_b = new_sb;
            Ok::<(TrajectoryResult, TrajectoryResult), HotSpringError>((ra, rb))
        })?;

        let lattice_size = gpu_a.state().gauge.gauge.dims[0];

        let obs_a = TrajectoryObservation {
            gpu_name: gpu_a.adapter_name().to_string(),
            traj_idx: self.traj_count,
            accepted: result_a.accepted,
            delta_h: result_a.delta_h,
            plaquette: result_a.plaquette,
            total_cg_iters: result_a.total_cg_iterations,
            elapsed_secs: result_a.elapsed_secs,
            beta: obs_beta_a,
            mass: obs_mass_a,
            dt: obs_dt_a,
            n_md_steps: obs_nmd_a,
            lattice_size,
            silicon_tags: gpu_a.silicon_routing_tags(),
        };
        let obs_b = TrajectoryObservation {
            gpu_name: gpu_b.adapter_name().to_string(),
            traj_idx: self.traj_count,
            accepted: result_b.accepted,
            delta_h: result_b.delta_h,
            plaquette: result_b.plaquette,
            total_cg_iters: result_b.total_cg_iterations,
            elapsed_secs: result_b.elapsed_secs,
            beta: obs_beta_b,
            mass: obs_mass_b,
            dt: obs_dt_b,
            n_md_steps: obs_nmd_b,
            lattice_size,
            silicon_tags: gpu_b.silicon_routing_tags(),
        };

        let _ = self.observation_tx.send(obs_a);
        let _ = self.observation_tx.send(obs_b);

        let idx = self.traj_count;
        self.traj_count += 1;

        Ok(BrainIterationResult {
            gpu_a: result_a,
            gpu_b: result_b,
            traj_idx: idx,
        })
    }

    pub fn traj_count(&self) -> usize {
        self.traj_count
    }

    pub fn config_live(&self) -> &RhmcConfigLive {
        &self.config_live
    }
}

// Cross-hardware state persistence extracted to `brain_persistence` module.
