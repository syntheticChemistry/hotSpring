// SPDX-License-Identifier: AGPL-3.0-only

//! NPU steering for adaptive dynamical HMC.
//!
//! Wraps a `MultiHeadNpu` to provide parameter suggestions (dt, n_md)
//! and anomaly detection during dynamical fermion thermalization.
//! The NPU is an apprentice — it defers to the heuristic adaptive
//! controller during crisis regimes and only steers when stable.

use crate::md::reservoir::heads;
use crate::md::reservoir::npu::MultiHeadNpu;
use crate::tolerances::{ADAPTIVE_DT_MAX, ADAPTIVE_DT_MIN, ADAPTIVE_NMD_MIN};

/// Build canonical 6D input for the NPU (Gen 1 format).
///
/// Matches the format in `production::trajectory_input::canonical_input`.
pub(super) fn npu_canonical_input(
    beta: f64,
    plaq: f64,
    mass: f64,
    chi: f64,
    acceptance: f64,
    lattice: usize,
) -> Vec<f64> {
    vec![
        (beta - 5.0) / 2.0,
        plaq,
        mass,
        chi / 1000.0,
        acceptance,
        lattice as f64 / 8.0,
    ]
}

/// Build canonical sequence (10 repeated frames) for the NPU.
pub(super) fn npu_canonical_seq(
    beta: f64,
    plaq: f64,
    mass: f64,
    chi: f64,
    acceptance: f64,
    lattice: usize,
) -> Vec<Vec<f64>> {
    vec![npu_canonical_input(beta, plaq, mass, chi, acceptance, lattice); 10]
}

/// NPU steering context for adaptive HMC.
///
/// Wraps a `MultiHeadNpu` and tracks trajectory state to provide parameter
/// suggestions during thermalization. The NPU can override the heuristic
/// controller when it has sufficient confidence in its predictions.
pub struct NpuSteering {
    /// The underlying multi-head NPU simulator.
    pub npu: MultiHeadNpu,
    /// How often (in trajectories) to query the NPU for parameter updates.
    pub feedback_interval: usize,
    last_plaquette: f64,
    last_acceptance: f64,
    last_chi: f64,
}

impl NpuSteering {
    /// Create a new NPU steering context.
    #[must_use]
    pub fn new(npu: MultiHeadNpu, feedback_interval: usize) -> Self {
        Self {
            npu,
            feedback_interval,
            last_plaquette: 0.5,
            last_acceptance: 0.0,
            last_chi: 0.0,
        }
    }

    /// Feed trajectory data to the NPU and optionally override controller params.
    ///
    /// Returns `Some((dt, n_md))` if the NPU suggests a parameter change,
    /// `None` if it defers to the heuristic controller.
    pub fn suggest_params(
        &mut self,
        beta: f64,
        mass: f64,
        lattice_size: usize,
        plaquette: f64,
        acceptance: f64,
        delta_h: f64,
    ) -> Option<(f64, usize)> {
        self.last_plaquette = plaquette;
        self.last_acceptance = acceptance;
        self.last_chi = delta_h.abs().min(1000.0);

        if acceptance < 0.40 || delta_h.abs() > 5.0 {
            return None;
        }

        let seq = npu_canonical_seq(
            beta,
            plaquette,
            mass,
            self.last_chi,
            acceptance,
            lattice_size,
        );

        let raw = self.npu.predict_head(&seq, heads::PARAM_SUGGEST);
        let dt_suggest = raw.abs().mul_add(0.05, ADAPTIVE_DT_MIN);

        if (ADAPTIVE_DT_MIN..=ADAPTIVE_DT_MAX).contains(&dt_suggest) {
            let target_tau = 0.02;
            let n_md = ((target_tau / dt_suggest).round() as usize).clamp(ADAPTIVE_NMD_MIN, 200);
            Some((dt_suggest, n_md))
        } else {
            None
        }
    }

    /// Query the NPU for a CG iteration estimate.
    #[must_use]
    pub fn predict_cg_iters(&mut self, beta: f64, mass: f64, lattice_size: usize) -> usize {
        let seq = npu_canonical_seq(
            beta,
            self.last_plaquette,
            mass,
            self.last_chi,
            self.last_acceptance,
            lattice_size,
        );
        let raw = self.npu.predict_head(&seq, heads::CG_ESTIMATE);
        (raw.abs() * 1000.0).round().max(10.0) as usize
    }

    /// Check for force anomaly using the NPU anomaly head.
    ///
    /// Returns `true` if the NPU's anomaly head fires (raw > 0.7),
    /// indicating the current trajectory has divergent forces and
    /// the controller should reduce dt aggressively.
    pub fn detect_force_anomaly(&mut self, beta: f64, mass: f64, lattice_size: usize) -> bool {
        let seq = npu_canonical_seq(
            beta,
            self.last_plaquette,
            mass,
            self.last_chi,
            self.last_acceptance,
            lattice_size,
        );
        let raw = self.npu.predict_head(&seq, heads::ANOMALY_DETECT);
        raw > 0.7
    }
}

/// Heuristic force anomaly detector for HMC trajectories.
///
/// Tracks a sliding window of |ΔH| values and flags when the current
/// value deviates by more than `threshold_sigma` standard deviations.
/// Delegates core statistics to `barracuda::nautilus::brain::force_anomaly`.
pub struct HmcForceAnomalyDetector {
    window: Vec<f64>,
    max_window: usize,
    threshold_sigma: f64,
}

impl HmcForceAnomalyDetector {
    /// Create a new detector with default parameters (window=20, threshold=10σ).
    #[must_use]
    pub fn new() -> Self {
        Self {
            window: Vec::new(),
            max_window: 20,
            threshold_sigma: 10.0,
        }
    }

    /// Record a |ΔH| value and return whether it's anomalous.
    pub fn observe(&mut self, abs_delta_h: f64) -> bool {
        let anomalous = if self.window.len() >= 3 {
            let mean: f64 = self.window.iter().sum::<f64>() / self.window.len() as f64;
            let variance: f64 = self.window.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / self.window.len() as f64;
            let std = (variance + 1e-20).sqrt();
            (abs_delta_h - mean).abs() > self.threshold_sigma * std
        } else {
            false
        };

        if self.window.len() >= self.max_window {
            self.window.remove(0);
        }
        self.window.push(abs_delta_h);

        anomalous
    }

    /// Reset the detector window.
    pub fn reset(&mut self) {
        self.window.clear();
    }
}

impl Default for HmcForceAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}
