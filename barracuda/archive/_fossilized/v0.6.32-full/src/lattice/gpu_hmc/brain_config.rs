// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::lattice::rhmc::RhmcConfig;
use crate::tolerances::{ADAPTIVE_DT_MAX, ADAPTIVE_DT_MIN, ADAPTIVE_NMD_MAX, ADAPTIVE_NMD_MIN};

use std::sync::mpsc;

pub(crate) const CG_TOL_DEFAULT: f64 = 1e-10;
pub(crate) const CG_TOL_MIN: f64 = 1e-14;
pub(crate) const CG_TOL_MAX: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionSource {
    NpuModel,
    Heuristic,
    Default,
}

#[derive(Debug, Clone)]
pub struct RhmcParamSuggestion {
    pub dt: f64,
    pub n_md_steps: usize,
    pub cg_tol: f64,
    pub confidence: f64,
    pub source: SuggestionSource,
    pub gpu_name: String,
}

#[derive(Debug, Clone)]
pub struct AppliedSuggestion {
    pub traj_idx: usize,
    pub suggestion: RhmcParamSuggestion,
    pub old_dt: f64,
    pub old_n_md: usize,
}

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
