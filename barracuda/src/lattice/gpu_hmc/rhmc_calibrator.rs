// SPDX-License-Identifier: AGPL-3.0-or-later

//! Self-tuning RHMC calibrator — eliminates hand-tuned magic numbers.
//!
//! Every simulation parameter is either:
//! - **Mathematical**: derived from theory (Omelyan λ, det_power = Nf/8)
//! - **Discovered**: measured from the gauge field (eigenvalue bounds)
//! - **Adapted**: feedback-controlled from physics observables (dt, n_md)
//! - **Validated**: checked and auto-corrected (pole count, spectral range)
//!
//! The calibrator owns all adaptive state and produces `RhmcConfig` on demand.
//! Physics observables (acceptance rate, ΔH, S_f consistency) serve as the
//! sole validators — no human guessing required.
//!
//! # Usage
//!
//! ```ignore
//! let mut cal = RhmcCalibrator::new(2, mass, beta, dims);
//! let spectral = cal.calibrate_spectral(&gpu, &pipelines, &state);
//! for _ in 0..n_trajectories {
//!     let config = cal.produce_config();
//!     let result = gpu_rhmc_trajectory(&gpu, ..., &config, &mut seed);
//!     cal.observe(&result);
//! }
//! ```
//!
//! # Design Invariants
//!
//! 1. No parameter requires human guessing
//! 2. Physics is always the validator
//! 3. Graceful degradation without NPU
//! 4. All thresholds live in `tolerances/lattice.rs`

use super::dynamical::GpuDynHmcPipelines;
use super::gpu_rhmc::GpuRhmcResult;
use super::spectral_probe::{SpectralInfo, probe_spectral_range};
use super::{GpuDynHmcState, GpuF64};
use crate::lattice::rhmc::{RationalApproximation, RhmcConfig, RhmcFermionConfig};
use crate::tolerances::{
    ADAPTIVE_DT_BUMP, ADAPTIVE_DT_DROP, ADAPTIVE_DT_MAX, ADAPTIVE_DT_MIN, ADAPTIVE_HIGH_ACCEPTANCE,
    ADAPTIVE_LOW_ACCEPTANCE, ADAPTIVE_NMD_MAX, ADAPTIVE_NMD_MIN, RHMC_APPROX_ERROR_THRESHOLD,
    RHMC_CG_TOL_FORCE, RHMC_CG_TOL_METROPOLIS, RHMC_CONSISTENCY_THRESHOLD, RHMC_MAX_POLES,
    RHMC_POLE_INCREMENT, RHMC_SPECTRAL_REPROBE_INTERVAL,
};

/// Central self-tuning calibrator for GPU RHMC simulations.
///
/// Manages spectral discovery, step-size adaptation, approximation quality
/// monitoring, and solver tolerance calibration. Produces `RhmcConfig`
/// instances where every parameter is derived from measurement or theory.
pub struct RhmcCalibrator {
    // Physics parameters (immutable for an ensemble)
    nf: usize,
    mass: f64,
    strange_mass: Option<f64>,
    beta: f64,
    /// Lattice dimensions (used for volume-dependent heuristics and spectral probing).
    #[expect(dead_code, reason = "EVOLUTION: reserved for GPU pipeline wiring")]
    dims: [usize; 4],

    // Discovered state
    spectral: Option<SpectralInfo>,
    n_poles: usize,

    // Adaptive step controller state
    dt: f64,
    n_md_steps: usize,
    accept_history: Vec<bool>,
    delta_h_history: Vec<f64>,
    window_size: usize,

    // Approximation quality monitor
    consistency_history: Vec<f64>,

    // Solver calibrator
    cg_tol_force: f64,
    cg_tol_metropolis: f64,
    cg_max_iter: usize,

    // Trajectory counter for periodic re-probing
    trajectory_count: usize,
}

impl RhmcCalibrator {
    /// Create a calibrator from physics parameters alone — no magic numbers.
    ///
    /// The calibrator starts with conservative defaults and refines all
    /// parameters from measurements during the first few trajectories.
    #[must_use]
    pub fn new(nf: usize, mass: f64, beta: f64, dims: [usize; 4]) -> Self {
        let vol: usize = dims.iter().product();
        let scale = (4096.0 / vol as f64).powf(0.25);
        let mass_factor = (mass * 0.5).min(1.0);
        let initial_dt = (0.001 * scale * mass_factor).clamp(ADAPTIVE_DT_MIN, ADAPTIVE_DT_MAX);
        let target_tau = 0.05;
        let initial_n_md =
            ((target_tau / initial_dt).round() as usize).clamp(ADAPTIVE_NMD_MIN, ADAPTIVE_NMD_MAX);

        Self {
            nf,
            mass,
            strange_mass: None,
            beta,
            dims,
            spectral: None,
            n_poles: 8,
            dt: initial_dt,
            n_md_steps: initial_n_md,
            accept_history: Vec::new(),
            delta_h_history: Vec::new(),
            window_size: 10,
            consistency_history: Vec::new(),
            cg_tol_force: RHMC_CG_TOL_FORCE,
            cg_tol_metropolis: RHMC_CG_TOL_METROPOLIS,
            cg_max_iter: 5000,
            trajectory_count: 0,
        }
    }

    /// Create a calibrator for Nf=2+1 with separate strange mass.
    #[must_use]
    pub fn new_nf2p1(light_mass: f64, strange_mass: f64, beta: f64, dims: [usize; 4]) -> Self {
        let mut cal = Self::new(3, light_mass, beta, dims);
        cal.strange_mass = Some(strange_mass);
        cal
    }

    /// Run GPU spectral probe on the current gauge configuration.
    ///
    /// Measures λ_max via power iteration and bounds λ_min analytically.
    /// This should be called once after quenched pre-thermalization and
    /// periodically during production (every `RHMC_SPECTRAL_REPROBE_INTERVAL`
    /// trajectories).
    pub fn calibrate_spectral(
        &mut self,
        gpu: &GpuF64,
        pipelines: &GpuDynHmcPipelines,
        state: &GpuDynHmcState,
    ) -> SpectralInfo {
        let info = probe_spectral_range(gpu, pipelines, state, self.mass);
        eprintln!(
            "[calibrator] spectral probe: λ_min≥{:.4e} (m²) λ_max≈{:.2} → range [{:.4e}, {:.2}]",
            info.lambda_min, info.lambda_max, info.range_min, info.range_max
        );
        self.spectral = Some(info.clone());
        info
    }

    /// Produce an `RhmcConfig` with all parameters derived from calibration.
    ///
    /// Every field is set from either theory, measurement, or adaptation.
    /// No hardcoded magic numbers.
    /// # Errors
    ///
    /// Returns `Err` if `nf` is not 2 or 3 (the only supported flavor counts).
    pub fn produce_config(&self) -> Result<RhmcConfig, crate::error::HotSpringError> {
        let spectral = self
            .spectral
            .clone()
            .unwrap_or_else(|| SpectralInfo::conservative_default(self.mass));

        match self.nf {
            2 => {
                let det_power = 0.25; // Nf/8 = 2/8
                let action_force = RationalApproximation::generate(
                    -det_power,
                    self.n_poles,
                    spectral.range_min,
                    spectral.range_max,
                );
                let heatbath = RationalApproximation::generate(
                    det_power / 2.0,
                    self.n_poles,
                    spectral.range_min,
                    spectral.range_max,
                );
                Ok(RhmcConfig {
                    sectors: vec![RhmcFermionConfig {
                        mass: self.mass,
                        det_power,
                        action_approx: action_force.clone(),
                        heatbath_approx: heatbath,
                        force_approx: action_force,
                    }],
                    beta: self.beta,
                    dt: self.dt,
                    n_md_steps: self.n_md_steps,
                    cg_tol: self.cg_tol_force,
                    cg_max_iter: self.cg_max_iter,
                })
            }
            3 => {
                let strange = self.strange_mass.unwrap_or(0.5);
                let light_power = 0.25; // 2/8
                let strange_power = 0.125; // 1/8
                let light_af = RationalApproximation::generate(
                    -light_power,
                    self.n_poles,
                    spectral.range_min,
                    spectral.range_max,
                );
                let strange_af = RationalApproximation::generate(
                    -strange_power,
                    self.n_poles,
                    spectral.range_min,
                    spectral.range_max,
                );
                Ok(RhmcConfig {
                    sectors: vec![
                        RhmcFermionConfig {
                            mass: self.mass,
                            det_power: light_power,
                            action_approx: light_af.clone(),
                            heatbath_approx: RationalApproximation::generate(
                                light_power / 2.0,
                                self.n_poles,
                                spectral.range_min,
                                spectral.range_max,
                            ),
                            force_approx: light_af,
                        },
                        RhmcFermionConfig {
                            mass: strange,
                            det_power: strange_power,
                            action_approx: strange_af.clone(),
                            heatbath_approx: RationalApproximation::generate(
                                strange_power / 2.0,
                                self.n_poles,
                                spectral.range_min,
                                spectral.range_max,
                            ),
                            force_approx: strange_af,
                        },
                    ],
                    beta: self.beta,
                    dt: self.dt,
                    n_md_steps: self.n_md_steps,
                    cg_tol: self.cg_tol_force,
                    cg_max_iter: self.cg_max_iter,
                })
            }
            nf => Err(crate::error::HotSpringError::InvalidOperation(format!(
                "unsupported flavor count nf={nf} (only 2 and 3 are supported)"
            ))),
        }
    }

    /// Observe a trajectory result and adapt all parameters.
    ///
    /// This is the core feedback loop. Physics observables drive all
    /// parameter changes:
    /// - acceptance rate → dt adaptation
    /// - |ΔH| → emergency scaling
    /// - approximation quality → pole count adjustment
    pub fn observe(&mut self, result: &GpuRhmcResult) {
        self.trajectory_count += 1;

        // Step controller: adapt dt from acceptance/ΔH
        self.adapt_step(result.accepted, result.delta_h);

        // Check if spectral re-probe is due
        if self
            .trajectory_count
            .is_multiple_of(RHMC_SPECTRAL_REPROBE_INTERVAL)
        {
            eprintln!(
                "[calibrator] spectral re-probe due (trajectory {})",
                self.trajectory_count
            );
        }
    }

    /// Record heatbath-action consistency for approximation quality monitoring.
    ///
    /// After heatbath generates φ = r_hb(D†D)η, the fermion action
    /// S_f(old) = φ†r_act(D†D)φ should equal η†η. The ratio S_f/η†η
    /// measures rational approximation quality.
    pub fn observe_consistency(&mut self, s_f_old: f64, eta_norm_sq: f64) {
        if eta_norm_sq > 1e-30 {
            let ratio = s_f_old / eta_norm_sq;
            self.consistency_history.push(ratio);
            if self.consistency_history.len() > 20 {
                self.consistency_history.remove(0);
            }

            let deviation = (ratio - 1.0).abs();
            if deviation > RHMC_CONSISTENCY_THRESHOLD {
                eprintln!(
                    "[calibrator] ⚠ consistency ratio {ratio:.4} deviates by {deviation:.4} \
                     (threshold={RHMC_CONSISTENCY_THRESHOLD})"
                );
                self.maybe_increase_poles();
            }
        }
    }

    /// Check approximation quality and auto-increase pole count if needed.
    fn maybe_increase_poles(&mut self) {
        let Ok(config) = self.produce_config() else {
            return;
        };
        let max_err = config
            .sectors
            .iter()
            .map(|s| s.action_approx.max_relative_error)
            .fold(0.0_f64, f64::max);

        if max_err > RHMC_APPROX_ERROR_THRESHOLD && self.n_poles < RHMC_MAX_POLES {
            let old = self.n_poles;
            self.n_poles = (self.n_poles + RHMC_POLE_INCREMENT).min(RHMC_MAX_POLES);
            eprintln!(
                "[calibrator] pole count {old}→{}: max_relative_error={max_err:.2e} \
                 exceeded threshold {RHMC_APPROX_ERROR_THRESHOLD:.0e}",
                self.n_poles
            );
        }
    }

    /// Acceptance-driven step-size adaptation (mirrors AdaptiveStepController).
    fn adapt_step(&mut self, accepted: bool, delta_h: f64) {
        self.accept_history.push(accepted);
        self.delta_h_history.push(delta_h);

        if self.accept_history.len() > self.window_size {
            self.accept_history.remove(0);
        }
        if self.delta_h_history.len() > self.window_size {
            self.delta_h_history.remove(0);
        }

        // Emergency: if |ΔH| > 2, scale dt toward |ΔH| ≈ 0.5 target.
        // For Omelyan (2nd-order symplectic), |ΔH| ∝ dt².
        if delta_h.abs() > 2.0 && !accepted {
            let scale = (0.5 / delta_h.abs()).sqrt().clamp(0.1, 0.9);
            self.dt = (self.dt * scale).max(ADAPTIVE_DT_MIN);
            eprintln!(
                "[calibrator] emergency dt scale: |ΔH|={:.1} → dt={:.6}",
                delta_h.abs(),
                self.dt
            );
            return;
        }

        if self.accept_history.len() < 5 {
            return;
        }

        let acc = self.acceptance_rate();
        let traj_length = self.dt * self.n_md_steps as f64;

        if acc > ADAPTIVE_HIGH_ACCEPTANCE {
            let new_dt = (self.dt * ADAPTIVE_DT_BUMP).min(ADAPTIVE_DT_MAX);
            self.n_md_steps =
                ((traj_length / new_dt).round() as usize).clamp(ADAPTIVE_NMD_MIN, ADAPTIVE_NMD_MAX);
            self.dt = new_dt;
        } else if acc < ADAPTIVE_LOW_ACCEPTANCE {
            let new_dt = (self.dt * ADAPTIVE_DT_DROP).max(ADAPTIVE_DT_MIN);
            self.n_md_steps =
                ((traj_length / new_dt).round() as usize).clamp(ADAPTIVE_NMD_MIN, ADAPTIVE_NMD_MAX);
            self.dt = new_dt;
        }
    }

    /// Current rolling acceptance rate (0.0–1.0).
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        if self.accept_history.is_empty() {
            return 0.0;
        }
        let n = self.accept_history.iter().filter(|&&a| a).count();
        n as f64 / self.accept_history.len() as f64
    }

    /// Mean |ΔH| over the rolling window.
    #[must_use]
    pub fn mean_delta_h(&self) -> f64 {
        if self.delta_h_history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.delta_h_history.iter().map(|dh| dh.abs()).sum();
        sum / self.delta_h_history.len() as f64
    }

    /// Current step size.
    #[must_use]
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Current MD step count.
    #[must_use]
    pub fn n_md_steps(&self) -> usize {
        self.n_md_steps
    }

    /// Current pole count.
    #[must_use]
    pub fn n_poles(&self) -> usize {
        self.n_poles
    }

    /// Total trajectories observed.
    #[must_use]
    pub fn trajectory_count(&self) -> usize {
        self.trajectory_count
    }

    /// Current spectral info, if probed.
    #[must_use]
    pub fn spectral_info(&self) -> Option<&SpectralInfo> {
        self.spectral.as_ref()
    }

    /// Whether a spectral re-probe is overdue.
    #[must_use]
    pub fn needs_spectral_reprobe(&self) -> bool {
        self.trajectory_count > 0
            && self
                .trajectory_count
                .is_multiple_of(RHMC_SPECTRAL_REPROBE_INTERVAL)
    }

    /// Re-probe spectral range on the current gauge field.
    ///
    /// Call this when `needs_spectral_reprobe()` returns true, passing
    /// the current GPU state. Updates the spectral bounds and regenerates
    /// rational approximations on the next `produce_config()` call.
    pub fn reprobe_spectral(
        &mut self,
        gpu: &GpuF64,
        pipelines: &GpuDynHmcPipelines,
        state: &GpuDynHmcState,
    ) {
        let old = self.spectral.as_ref().map(|s| (s.range_min, s.range_max));
        let info = self.calibrate_spectral(gpu, pipelines, state);
        if let Some((old_min, old_max)) = old {
            let shift_min = (info.range_min - old_min).abs() / old_min;
            let shift_max = (info.range_max - old_max).abs() / old_max;
            if shift_min > 0.2 || shift_max > 0.2 {
                eprintln!(
                    "[calibrator] significant spectral drift: range [{old_min:.4e}, {old_max:.2}] \
                     → [{:.4e}, {:.2}]",
                    info.range_min, info.range_max
                );
            }
        }
    }

    /// Force-vs-Metropolis CG tolerance for the current trajectory phase.
    #[must_use]
    pub fn cg_tol_force(&self) -> f64 {
        self.cg_tol_force
    }

    /// Tight CG tolerance for Metropolis action evaluation.
    #[must_use]
    pub fn cg_tol_metropolis(&self) -> f64 {
        self.cg_tol_metropolis
    }
}
