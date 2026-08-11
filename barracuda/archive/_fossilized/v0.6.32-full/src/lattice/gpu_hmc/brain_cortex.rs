// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::md::reservoir::heads;
use crate::md::reservoir::npu::ExportedWeights;
use crate::tolerances::{ADAPTIVE_DT_MAX, ADAPTIVE_DT_MIN, ADAPTIVE_NMD_MAX, ADAPTIVE_NMD_MIN};

use super::brain_config::{
    CG_TOL_DEFAULT, CG_TOL_MAX, CG_TOL_MIN, RhmcParamSuggestion, SuggestionSource,
};
use super::brain_inference::{CrossGpuAgreement, NpuInference, NpuRunStats};
use super::brain_persistence::{BrainState, SerializableWeights, save_brain_state};
use super::brain_rhmc::TrajectoryObservation;

use std::collections::VecDeque;
use std::sync::mpsc;

const TRUST_THRESHOLD: f64 = 0.3;
const OBSERVATION_WINDOW: usize = 20;

pub struct NpuCortex {
    npu: NpuInference,
    observation_rx: mpsc::Receiver<TrajectoryObservation>,
    suggestion_tx: mpsc::Sender<RhmcParamSuggestion>,
    stats: NpuRunStats,
    per_gpu_history: std::collections::HashMap<String, VecDeque<TrajectoryObservation>>,
    unified_history: VecDeque<TrajectoryObservation>,
    pending_pair: Option<TrajectoryObservation>,
    agreement_log: Vec<CrossGpuAgreement>,
    report_interval: usize,
    checkpoint_interval: usize,
    esn_weights: ExportedWeights,
}

impl NpuCortex {
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

            let gpu_history = self
                .per_gpu_history
                .entry(obs.gpu_name.clone())
                .or_default();
            gpu_history.push_back(obs.clone());
            if gpu_history.len() > OBSERVATION_WINDOW {
                gpu_history.pop_front();
            }

            self.check_cross_gpu_agreement(&obs);

            self.unified_history.push_back(obs.clone());
            if self.unified_history.len() > OBSERVATION_WINDOW * 2 {
                self.unified_history.pop_front();
            }

            let input_seq = build_esn_input_sequence(&self.unified_history);
            let (outputs, metrics) = self.npu.infer(&input_seq);
            self.stats.update_inference(&metrics);

            if let Some(suggestion) = self.produce_suggestion(&outputs, &obs) {
                self.stats.suggestions_applied += 1;
                let _ = self.suggestion_tx.send(suggestion);
            }

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

        if self.unified_history.len() < 10 {
            return None;
        }

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
