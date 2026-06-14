// SPDX-License-Identifier: AGPL-3.0-or-later

//! 3-month Chuna validation matrix: cell definitions, planning tables, and execution.

mod cells;
mod run;

pub use cells::{
    CellResult, CliArgs, MatrixCell, WilsonLoopResult, beta_scan_cells, custom_cell,
    dynamical_scaling_cells, estimate_wall_time, mass_scan_cells, parse_args, parse_csv_f64,
    parse_csv_usize, print_planning_matrix, quenched_ladder_cells,
};
pub use run::{hmc_dt_for_volume, run_cell, run_dynamical_cell, run_quenched_cell};
