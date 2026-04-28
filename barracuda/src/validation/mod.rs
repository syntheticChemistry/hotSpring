// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation harness for hotSpring binaries.
//!
//! Every validation binary follows the hotSpring pattern:
//!   - Hardcoded expected values with provenance
//!   - Explicit pass/fail checks against documented tolerances
//!   - Exit code 0 (all checks pass) or 1 (any check fails)
//!   - Machine-readable summary on stdout
//!
//! This module provides the shared infrastructure:
//! - [`ValidationHarness`] — physics validation (pass/fail, annotated checks)
//! - [`CompositionResult`] — NUCLEUS composition validation (bool/skip/latency)
//! - [`TelemetryWriter`] — JSONL telemetry for petalTongue integration
//! - [`OrExit`] — zero-panic exit for validation binaries
//! - [`ValidationSink`] enum (Stdout, Null, Ndjson) — zero `dyn` dispatch

mod composition;
mod harness;
mod telemetry;

pub use composition::{CheckOutcome, CompositionCheck, CompositionResult, OrExit, ValidationSink};
pub use harness::{Check, HardwareProfile, ToleranceMode, ValidationHarness};
pub use telemetry::TelemetryWriter;

#[cfg(test)]
mod tests;
