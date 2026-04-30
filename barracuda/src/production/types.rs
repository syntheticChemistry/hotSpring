// SPDX-License-Identifier: AGPL-3.0-or-later

/// NPU attention state for CG residual monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionState {
    /// Normal operation.
    Green,
    /// Elevated monitoring.
    Yellow,
    /// Critical — consider intervention.
    Red,
}

impl std::fmt::Display for AttentionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Green => write!(f, "GREEN"),
            Self::Yellow => write!(f, "YELLOW"),
            Self::Red => write!(f, "RED"),
        }
    }
}

/// Per-beta measurement result with NPU statistics.
#[derive(Clone, Debug)]
pub struct BetaResult {
    /// Coupling β value.
    pub beta: f64,
    /// Fermion mass (0 for quenched).
    pub mass: f64,
    /// Mean plaquette over measurement trajectories.
    pub mean_plaq: f64,
    /// Standard deviation of plaquette.
    pub std_plaq: f64,
    /// Average Polyakov loop magnitude.
    pub polyakov: f64,
    /// Plaquette susceptibility (variance × volume).
    pub susceptibility: f64,
    /// Mean action density.
    pub action_density: f64,
    /// HMC acceptance rate.
    pub acceptance: f64,
    /// Mean CG iterations per solve.
    pub mean_cg_iters: f64,
    /// Number of measurement trajectories.
    pub n_traj: usize,
    /// Total wall time in seconds.
    pub wall_s: f64,
    /// Detected thermodynamic phase.
    pub phase: &'static str,
    /// Thermalization trajectories actually run.
    pub therm_used: usize,
    /// Thermalization trajectory budget.
    pub therm_budget: usize,
    /// MD step size used.
    pub dt_used: f64,
    /// MD steps per trajectory.
    pub n_md_used: usize,
    /// NPU early-exit during thermalization.
    pub npu_therm_early_exit: bool,
    /// NPU quenched trajectory budget.
    pub npu_quenched_budget: usize,
    /// NPU quenched trajectories used.
    pub npu_quenched_used: usize,
    /// NPU early-exit during quenched phase.
    pub npu_quenched_early_exit: bool,
    /// NPU rejections predicted.
    pub npu_reject_predictions: usize,
    /// NPU rejections correctly predicted.
    pub npu_reject_correct: usize,
    /// NPU anomaly detections.
    pub npu_anomalies: usize,
    /// CG convergence check interval for NPU screening.
    pub npu_cg_check_interval: usize,
}

impl Default for BetaResult {
    fn default() -> Self {
        Self {
            beta: 0.0,
            mass: 0.0,
            mean_plaq: 0.0,
            std_plaq: 0.0,
            polyakov: 0.0,
            susceptibility: 0.0,
            action_density: 0.0,
            acceptance: 0.0,
            mean_cg_iters: 0.0,
            n_traj: 0,
            wall_s: 0.0,
            phase: "transition",
            therm_used: 0,
            therm_budget: 0,
            dt_used: 0.0,
            n_md_used: 0,
            npu_therm_early_exit: false,
            npu_quenched_budget: 0,
            npu_quenched_used: 0,
            npu_quenched_early_exit: false,
            npu_reject_predictions: 0,
            npu_reject_correct: 0,
            npu_anomalies: 0,
            npu_cg_check_interval: 0,
        }
    }
}

/// Which phase of the HMC pipeline produced this trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajectoryPhase {
    /// Pure-gauge trajectory (no fermions).
    Quenched,
    /// Thermalization trajectory — equilibrating before measurement.
    Therm,
    /// Production measurement trajectory — data collected for observables.
    Measurement,
}

/// Per-trajectory event streamed to the NPU for every single trajectory.
///
/// Replaces the summary-only data flow (one `BetaResult` per beta) with
/// fine-grained per-event observation (~60 events per beta point).
#[derive(Clone, Debug)]
pub struct TrajectoryEvent {
    /// Inverse coupling β = 6/g².
    pub beta: f64,
    /// Fermion mass parameter.
    pub mass: f64,
    /// Spatial lattice extent.
    pub lattice: usize,
    /// HMC pipeline phase that produced this trajectory.
    pub phase_tag: TrajectoryPhase,
    /// Sequential trajectory index within this beta point.
    pub traj_idx: usize,
    /// Average plaquette for this trajectory.
    pub plaquette: f64,
    /// Hamiltonian violation δH = `H_new` - `H_old`.
    pub delta_h: f64,
    /// Whether the Metropolis accept/reject accepted this trajectory.
    pub accepted: bool,
    /// CG solver iterations used in the fermion force computation.
    pub cg_iterations: usize,
    /// Real part of the Polyakov loop.
    pub polyakov_re: f64,
    /// Phase angle of the Polyakov loop.
    pub polyakov_phase: f64,
    /// Gauge action density `S_G` / (6 * volume).
    pub action_density: f64,
    /// Plaquette variance across the lattice.
    pub plaquette_var: f64,
    /// Wall-clock time for this trajectory in microseconds.
    pub wall_us: u64,
    /// Running Metropolis acceptance rate.
    pub running_acceptance: f64,
}

#[cfg(test)]
mod tests {
    use super::AttentionState;

    #[test]
    fn attention_state_display() {
        assert_eq!(format!("{}", AttentionState::Green), "GREEN");
        assert_eq!(format!("{}", AttentionState::Yellow), "YELLOW");
        assert_eq!(format!("{}", AttentionState::Red), "RED");
    }
}
