// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trajectory input construction and encoding for NPU.
//!
//! Gen 1: canonical 6D input for the multi-head ESN.
//! Gen 2: full-stream 21D input from per-trajectory events + optional proxy features.

use crate::production::{TrajectoryEvent, TrajectoryPhase};
use crate::proxy::ProxyFeatures;

/// Gen 1 canonical 6D input vector (kept for backward compat with existing ESN).
#[must_use]
pub(crate) fn canonical_input(
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

/// Gen 1 canonical sequence (10 repeated frames) for ESN input.
#[must_use]
pub(crate) fn canonical_seq(
    beta: f64,
    plaq: f64,
    mass: f64,
    chi: f64,
    acceptance: f64,
    lattice: usize,
) -> Vec<Vec<f64>> {
    vec![canonical_input(beta, plaq, mass, chi, acceptance, lattice); 10]
}

/// Gen 2 full-stream 21D input vector from a per-trajectory event + proxy context.
///
/// Dimensions 0-14: trajectory data (always available).
/// Dimensions 15-20: proxy features (0.0 when no proxy has been evaluated yet).
#[must_use]
pub fn trajectory_input(evt: &TrajectoryEvent) -> Vec<f64> {
    trajectory_input_with_proxy(evt, None)
}

/// Build the full input vector with optional proxy features appended.
#[must_use]
pub fn trajectory_input_with_proxy(
    evt: &TrajectoryEvent,
    proxy: Option<&ProxyFeatures>,
) -> Vec<f64> {
    vec![
        // 0-14: trajectory data
        (evt.beta - 5.0) / 2.0,
        evt.plaquette,
        evt.mass,
        evt.delta_h.clamp(-5.0, 5.0) / 5.0,
        evt.cg_iterations as f64 / 100_000.0,
        if evt.accepted { 1.0 } else { 0.0 },
        evt.polyakov_re.clamp(-1.0, 1.0),
        evt.polyakov_phase / std::f64::consts::PI,
        evt.action_density / 6.0,
        evt.plaquette_var.clamp(0.0, 0.1) * 10.0,
        evt.running_acceptance,
        evt.lattice as f64 / 12.0,
        evt.traj_idx as f64 / 60.0,
        evt.wall_us as f64 / 1_000_000.0,
        match evt.phase_tag {
            TrajectoryPhase::Quenched => 0.0,
            TrajectoryPhase::Therm => 0.5,
            TrajectoryPhase::Measurement => 1.0,
        },
        // 15-20: proxy features (Anderson + Potts)
        proxy.map_or(0.0, |p| p.level_spacing_ratio),
        proxy.map_or(0.0, |p| p.lambda_min.clamp(0.0, 1.0)),
        proxy.map_or(0.0, |p| p.ipr),
        proxy.map_or(0.0, |p| p.condition_number.clamp(0.0, 100.0) / 100.0),
        proxy.map_or(0.0, |p| p.potts_magnetization),
        proxy.map_or(0.0, |p| p.potts_susceptibility.clamp(0.0, 100.0) / 100.0),
    ]
}

/// Number of dimensions in the Gen 2 trajectory input vector (15 traj + 6 proxy).
pub const TRAJECTORY_INPUT_DIM: usize = 21;

/// Heuristic phase label from β (used when ESN head is untrusted).
#[must_use]
pub(crate) fn heuristic_phase(beta: f64) -> &'static str {
    if beta > 5.79 {
        "deconfined"
    } else if beta > 5.59 {
        "transition"
    } else {
        "confined"
    }
}
