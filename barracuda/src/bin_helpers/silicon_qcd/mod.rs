// SPDX-License-Identifier: AGPL-3.0-or-later

//! Production silicon-instrumented QCD: budgets, telemetry, and GPU trajectory runners.

mod flow;
mod runner;
mod support;

pub use flow::{build_summary, run_gradient_flow_uni, run_quenched_gradient_flow};
pub use runner::{run_beta_point, run_quenched_beta, run_uni_traj};
pub use support::*;
