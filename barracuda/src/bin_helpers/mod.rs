// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared helpers extracted from large validation binaries (harness setup, domain suites, reports).

#[cfg(feature = "sovereign-dispatch")]
pub mod coral_sovereign;
pub mod validation_matrix;
pub mod silicon_qcd;
pub mod chuna_overnight;
