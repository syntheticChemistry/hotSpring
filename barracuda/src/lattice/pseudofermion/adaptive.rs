// SPDX-License-Identifier: AGPL-3.0-only

//! Adaptive step-size controller and thermalization for dynamical HMC.
//!
//! The controller adjusts dt and n_md to maintain target acceptance (~65%).
//! Works standalone with heuristic adaptation; optionally accepts NPU
//! parameter suggestions.  Includes mass annealing for warm-start
//! thermalization from quenched configurations.

use super::npu_steering::{HmcForceAnomalyDetector, NpuSteering, npu_canonical_seq};
use super::{DynamicalHmcConfig, dynamical_hmc_trajectory};
use crate::lattice::hmc::IntegratorType;
use crate::lattice::wilson::Lattice;
use crate::md::reservoir::heads;
use crate::md::reservoir::npu::MultiHeadNpu;
use crate::tolerances::{
    ADAPTIVE_DT_BUMP, ADAPTIVE_DT_DROP, ADAPTIVE_DT_MAX, ADAPTIVE_DT_MIN, ADAPTIVE_HIGH_ACCEPTANCE,
    ADAPTIVE_LOW_ACCEPTANCE, ADAPTIVE_NMD_MAX, ADAPTIVE_NMD_MIN,
};

/// Acceptance-driven adaptive step-size controller for dynamical HMC.
///
/// Adjusts `dt` and `n_md_steps` to maintain target acceptance (~65%).
/// Preserves trajectory length τ = dt × n_md when adjusting so the
/// autocorrelation properties of the Markov chain are roughly constant.
///
/// # Usage
///
/// ```ignore
/// let mut ctrl = AdaptiveStepController::for_dynamical([8,8,8,8], 5.4, 0.1);
/// for _ in 0..50 {
///     ctrl.apply_to(&mut config);
///     let result = dynamical_hmc_trajectory(&mut lattice, &mut config);
///     ctrl.update(result.accepted, result.delta_h);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct AdaptiveStepController {
    /// Current MD step size.
    pub dt: f64,
    /// Current number of MD steps.
    pub n_md_steps: usize,
    accept_history: Vec<bool>,
    delta_h_history: Vec<f64>,
    window_size: usize,
}

impl AdaptiveStepController {
    /// Create a controller with heuristic initial parameters for dynamical fermion HMC.
    ///
    /// Accounts for lattice volume (larger → smaller dt) and fermion mass
    /// (lighter → smaller dt due to stiffer force). Uses short initial
    /// trajectory (τ ≈ 0.05) to avoid catastrophic ΔH from quenched starts.
    #[must_use]
    pub fn for_dynamical(dims: [usize; 4], _beta: f64, mass: f64) -> Self {
        let vol: usize = dims.iter().product();
        let scale = (4096.0 / vol as f64).powf(0.25);
        let mass_factor = (mass * 0.5).min(1.0);
        let dt = (0.001 * scale * mass_factor).clamp(ADAPTIVE_DT_MIN, ADAPTIVE_DT_MAX);
        let target_tau = 0.05;
        let n_md_steps =
            ((target_tau / dt).round() as usize).clamp(ADAPTIVE_NMD_MIN, ADAPTIVE_NMD_MAX);

        Self {
            dt,
            n_md_steps,
            accept_history: Vec::new(),
            delta_h_history: Vec::new(),
            window_size: 10,
        }
    }

    /// Create a controller with NPU-suggested initial parameters.
    ///
    /// The NPU suggests a dt offset on top of the heuristic — untrained NPU
    /// (raw ≈ 0) preserves the heuristic, trained NPU fine-tunes it.
    #[must_use]
    pub fn for_dynamical_with_npu(
        dims: [usize; 4],
        beta: f64,
        mass: f64,
        npu: &mut MultiHeadNpu,
    ) -> Self {
        let heuristic = Self::for_dynamical(dims, beta, mass);
        let lattice_size = dims[0];
        let seq = npu_canonical_seq(beta, 0.5, mass, 0.0, 0.0, lattice_size);
        let raw = npu.predict_head(&seq, heads::PARAM_SUGGEST);

        let dt_suggest = raw.abs().mul_add(0.01, heuristic.dt);
        if (ADAPTIVE_DT_MIN..=ADAPTIVE_DT_MAX).contains(&dt_suggest) {
            let target_tau = 0.05;
            let n_md = ((target_tau / dt_suggest).round() as usize)
                .clamp(ADAPTIVE_NMD_MIN, ADAPTIVE_NMD_MAX);
            Self {
                dt: dt_suggest,
                n_md_steps: n_md,
                accept_history: Vec::new(),
                delta_h_history: Vec::new(),
                window_size: 10,
            }
        } else {
            heuristic
        }
    }

    /// Clear acceptance/ΔH history, keeping current dt and n_md_steps.
    pub fn reset_history(&mut self) {
        self.accept_history.clear();
        self.delta_h_history.clear();
    }

    /// Override parameters from an external source (e.g. NPU suggestion).
    pub fn set_params(&mut self, dt: f64, n_md_steps: usize) {
        self.dt = dt.clamp(ADAPTIVE_DT_MIN, ADAPTIVE_DT_MAX);
        self.n_md_steps = n_md_steps.clamp(ADAPTIVE_NMD_MIN, ADAPTIVE_NMD_MAX);
    }

    /// Apply current parameters to a `DynamicalHmcConfig`.
    pub fn apply_to(&self, config: &mut DynamicalHmcConfig) {
        config.dt = self.dt;
        config.n_md_steps = self.n_md_steps;
    }

    /// Record the result of a trajectory and adapt step size if needed.
    pub fn update(&mut self, accepted: bool, delta_h: f64) {
        self.accept_history.push(accepted);
        self.delta_h_history.push(delta_h);

        if self.accept_history.len() > self.window_size {
            self.accept_history.remove(0);
        }
        if self.delta_h_history.len() > self.window_size {
            self.delta_h_history.remove(0);
        }

        // Emergency: if |ΔH| > 2, scale dt toward |ΔH| ≈ 0.5 target.
        // For Omelyan (2nd-order symplectic), |ΔH| ∝ dt².  Keep n_md fixed
        // so trajectory shortens, avoiding compute-cost explosion.
        if delta_h.abs() > 2.0 && !accepted {
            let scale = (0.5 / delta_h.abs()).sqrt().clamp(0.1, 0.9);
            self.dt = (self.dt * scale).max(ADAPTIVE_DT_MIN);
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
        let n_accept = self.accept_history.iter().filter(|&&a| a).count();
        n_accept as f64 / self.accept_history.len() as f64
    }

    /// Mean |ΔH| over the window.
    #[must_use]
    pub fn mean_delta_h(&self) -> f64 {
        if self.delta_h_history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.delta_h_history.iter().map(|dh| dh.abs()).sum();
        sum / self.delta_h_history.len() as f64
    }

    /// Number of trajectories recorded.
    #[must_use]
    pub fn trajectory_count(&self) -> usize {
        self.accept_history.len()
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Thermalization result types
// ═══════════════════════════════════════════════════════════════════

/// Result of adaptive thermalization.
#[derive(Clone, Debug)]
pub struct AdaptiveThermalizationResult {
    /// Number of accepted trajectories.
    pub n_accepted: usize,
    /// Total trajectories attempted.
    pub n_total: usize,
    /// Final acceptance rate.
    pub acceptance_rate: f64,
    /// Final dt after adaptation.
    pub final_dt: f64,
    /// Final n_md after adaptation.
    pub final_n_md: usize,
    /// Total CG iterations.
    pub total_cg_iterations: usize,
    /// Mean |ΔH| over the last window.
    pub mean_delta_h: f64,
}

/// Mass annealing schedule for warm-start thermalization.
///
/// Starting from a heavy mass where the fermion force is gentle, we
/// gradually reduce to the target mass. At each stage, the adaptive
/// controller tunes dt/n_md for the current mass.
#[derive(Clone, Debug)]
pub struct MassAnnealingSchedule {
    /// Sequence of (mass, n_trajectories) stages.
    pub stages: Vec<(f64, usize)>,
}

impl MassAnnealingSchedule {
    /// Default 4-stage annealing from m=1.0 to target mass.
    #[must_use]
    pub fn default_for(target_mass: f64) -> Self {
        let mut stages = Vec::new();
        let masses = [1.0, 0.5, 0.2];
        for &m in &masses {
            if m > target_mass * 1.5 {
                stages.push((m, 15));
            }
        }
        stages.push((target_mass, 20));
        Self { stages }
    }

    /// Single-stage schedule (no annealing, for already-close masses).
    #[must_use]
    pub fn single(mass: f64, n_traj: usize) -> Self {
        Self {
            stages: vec![(mass, n_traj)],
        }
    }
}

/// Result of warm-start thermalization.
#[derive(Clone, Debug)]
pub struct WarmStartResult {
    /// Per-stage results.
    pub stage_results: Vec<StageResult>,
    /// Final acceptance rate (last stage).
    pub final_acceptance: f64,
    /// Final plaquette.
    pub final_plaquette: f64,
    /// Final dt.
    pub final_dt: f64,
    /// Final n_md.
    pub final_n_md: usize,
    /// Total CG iterations across all stages.
    pub total_cg_iterations: usize,
    /// Total trajectories across all stages.
    pub total_trajectories: usize,
}

/// Result of a single annealing stage.
#[derive(Clone, Debug)]
pub struct StageResult {
    /// Mass used in this stage.
    pub mass: f64,
    /// Number of trajectories in this stage.
    pub n_trajectories: usize,
    /// Acceptance rate for this stage.
    pub acceptance_rate: f64,
    /// Mean |ΔH| for this stage.
    pub mean_delta_h: f64,
    /// Final plaquette after this stage.
    pub plaquette: f64,
}

// ═══════════════════════════════════════════════════════════════════
//  Warm-start thermalization functions
// ═══════════════════════════════════════════════════════════════════

/// Warm-start dynamical thermalization with mass annealing.
pub fn dynamical_thermalize_warm_start(
    lattice: &mut Lattice,
    config: &mut DynamicalHmcConfig,
    schedule: &MassAnnealingSchedule,
    controller: &mut AdaptiveStepController,
) -> WarmStartResult {
    dynamical_thermalize_warm_start_inner(lattice, config, schedule, controller, None)
}

/// Warm-start with NPU steering.
pub fn dynamical_thermalize_warm_start_npu(
    lattice: &mut Lattice,
    config: &mut DynamicalHmcConfig,
    schedule: &MassAnnealingSchedule,
    controller: &mut AdaptiveStepController,
    npu: &mut NpuSteering,
) -> WarmStartResult {
    dynamical_thermalize_warm_start_inner(lattice, config, schedule, controller, Some(npu))
}

fn dynamical_thermalize_warm_start_inner(
    lattice: &mut Lattice,
    config: &mut DynamicalHmcConfig,
    schedule: &MassAnnealingSchedule,
    controller: &mut AdaptiveStepController,
    mut npu: Option<&mut NpuSteering>,
) -> WarmStartResult {
    config.integrator = IntegratorType::Omelyan;

    let lattice_size = lattice.dims[0];
    let beta = config.beta;
    let pre_plaq = lattice.average_plaquette();
    if (0.1..=0.999).contains(&pre_plaq) {
        println!("    Pre-annealing ⟨P⟩={pre_plaq:.6} (healthy)");
    } else {
        println!(
            "    ⚠ Pre-annealing plaquette implausible ({pre_plaq:.6}) — lattice may not be thermalized"
        );
    }
    let mut stage_results = Vec::new();
    let mut total_cg = 0;
    let mut total_traj = 0;
    let mut anomaly_detector = HmcForceAnomalyDetector::new();

    for (stage_idx, &(mass, n_traj)) in schedule.stages.iter().enumerate() {
        config.fermion.mass = mass;
        controller.reset_history();
        anomaly_detector.reset();

        let mut stage_accepted = 0;
        let mut stage_cg = 0;

        println!(
            "    [stage {}/{}] mass={mass:.2}, dt={:.4}, n_md={}",
            stage_idx + 1,
            schedule.stages.len(),
            controller.dt,
            controller.n_md_steps,
        );

        for i in 0..n_traj {
            controller.apply_to(config);
            let result = dynamical_hmc_trajectory(lattice, config);

            if i == 0 {
                println!(
                    "      [diag] S_gauge={:.1}, S_ferm={:.1}, cg_iters={}, ΔH={:.1}",
                    result.gauge_action,
                    result.fermion_action,
                    result.cg_iterations,
                    result.delta_h,
                );
            }

            if result.accepted {
                stage_accepted += 1;
            }
            stage_cg += result.cg_iterations;

            controller.update(result.accepted, result.delta_h);

            if anomaly_detector.observe(result.delta_h.abs()) {
                let emergency_dt = (controller.dt * 0.5).max(ADAPTIVE_DT_MIN);
                controller.set_params(emergency_dt, controller.n_md_steps);
                println!(
                    "      ⚠ Force anomaly detected (|ΔH|={:.1}), dt→{emergency_dt:.5}",
                    result.delta_h.abs()
                );
            }

            if let Some(ref mut steering) = npu
                && steering.feedback_interval > 0
                && (i + 1) % steering.feedback_interval == 0
            {
                if let Some((dt, n_md)) = steering.suggest_params(
                    beta,
                    mass,
                    lattice_size,
                    lattice.average_plaquette(),
                    controller.acceptance_rate(),
                    result.delta_h,
                ) {
                    println!(
                        "      [NPU] steer → dt={dt:.5}, n_md={n_md} (acc={:.0}%, |ΔH|={:.2})",
                        controller.acceptance_rate() * 100.0,
                        result.delta_h.abs(),
                    );
                    controller.set_params(dt, n_md);
                } else {
                    println!(
                        "      [NPU] defer to heuristic (acc={:.0}%)",
                        controller.acceptance_rate() * 100.0,
                    );
                }
            }

            if (i + 1) % 5 == 0 || i == n_traj - 1 {
                println!(
                    "      [{}/{}] acc={:.0}%, ⟨P⟩={:.6}, dt={:.4}, n_md={}, |ΔH|={:.2}",
                    i + 1,
                    n_traj,
                    controller.acceptance_rate() * 100.0,
                    lattice.average_plaquette(),
                    controller.dt,
                    controller.n_md_steps,
                    result.delta_h.abs(),
                );
            }
        }

        let stage_acc = stage_accepted as f64 / n_traj as f64;
        stage_results.push(StageResult {
            mass,
            n_trajectories: n_traj,
            acceptance_rate: stage_acc,
            mean_delta_h: controller.mean_delta_h(),
            plaquette: lattice.average_plaquette(),
        });

        total_cg += stage_cg;
        total_traj += n_traj;
    }

    let last_stage = stage_results.last().cloned().unwrap_or(StageResult {
        mass: config.fermion.mass,
        n_trajectories: 0,
        acceptance_rate: 0.0,
        mean_delta_h: 0.0,
        plaquette: 0.0,
    });

    WarmStartResult {
        stage_results,
        final_acceptance: last_stage.acceptance_rate,
        final_plaquette: last_stage.plaquette,
        final_dt: controller.dt,
        final_n_md: controller.n_md_steps,
        total_cg_iterations: total_cg,
        total_trajectories: total_traj,
    }
}

/// Run adaptive dynamical thermalization.
pub fn dynamical_thermalize_adaptive(
    lattice: &mut Lattice,
    config: &mut DynamicalHmcConfig,
    n_trajectories: usize,
    controller: &mut AdaptiveStepController,
) -> AdaptiveThermalizationResult {
    config.integrator = IntegratorType::Omelyan;

    let mut n_accepted = 0;
    let mut total_cg = 0;

    for i in 0..n_trajectories {
        controller.apply_to(config);
        let result = dynamical_hmc_trajectory(lattice, config);

        if result.accepted {
            n_accepted += 1;
        }
        total_cg += result.cg_iterations;

        controller.update(result.accepted, result.delta_h);

        if i > 0 && (i + 1) % 10 == 0 {
            println!(
                "    [{}/{}] acc={:.0}%, ⟨P⟩={:.6}, dt={:.4}, n_md={}, ⟨|ΔH|⟩={:.2}",
                i + 1,
                n_trajectories,
                controller.acceptance_rate() * 100.0,
                lattice.average_plaquette(),
                controller.dt,
                controller.n_md_steps,
                controller.mean_delta_h(),
            );
        }
    }

    AdaptiveThermalizationResult {
        n_accepted,
        n_total: n_trajectories,
        acceptance_rate: n_accepted as f64 / n_trajectories as f64,
        final_dt: controller.dt,
        final_n_md: controller.n_md_steps,
        total_cg_iterations: total_cg,
        mean_delta_h: controller.mean_delta_h(),
    }
}
