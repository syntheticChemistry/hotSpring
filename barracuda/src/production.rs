// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared types and infrastructure for production lattice QCD binaries.
//!
//! Extracted from production_dynamical_mixed, production_mixed_pipeline, and
//! related binaries to reduce duplication and keep binaries under 1000 lines.
//!
//! # Contents
//!
//! - **MetaRow** — per-beta aggregate statistics from meta tables or trajectory logs
//! - **BetaResult** — per-beta measurement result with NPU stats
//! - **AttentionState** — NPU attention state machine (Green/Yellow/Red)
//! - **npu_worker** — 11-head NPU worker thread (NpuRequest, NpuResponse, spawn_npu_worker)
//! - **Helpers** — load_meta_table, plaquette_variance
//! - **ESN** — check_thermalization, predict_rejection, predict_beta_c,
//!   find_max_uncertainty_beta, build_training_data, bootstrap_esn_from_trajectory_log

pub mod beta_scan;
pub mod checkpoint;
pub mod cortex_worker;
pub mod dynamical_bootstrap;
pub mod dynamical_mixed_pipeline;
pub mod dynamical_summary;
pub mod mixed_summary;
pub mod npu_worker;
pub mod sub_models;
pub mod titan_validation;
pub mod titan_worker;
pub mod trajectory_input;
pub mod bootstrap;
pub mod esn_heuristics;
pub mod meta_table;
pub mod types;

pub use bootstrap::bootstrap_esn_from_trajectory_log;
pub use esn_heuristics::{
    build_training_data, check_thermalization, find_max_uncertainty_beta, plaquette_variance,
    predict_beta_c, predict_rejection,
};
pub use meta_table::{load_meta_table, MetaRow};
pub use types::{AttentionState, BetaResult, TrajectoryEvent, TrajectoryPhase};
