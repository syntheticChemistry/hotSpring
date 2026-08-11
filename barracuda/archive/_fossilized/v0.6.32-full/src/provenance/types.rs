// SPDX-License-Identifier: AGPL-3.0-or-later

/// A single provenance record tying a Rust reference value to its Python origin.
#[derive(Debug, Clone)]
#[must_use]
pub struct BaselineProvenance {
    /// Human-readable label (e.g. "L1 best chi2/datum")
    pub label: &'static str,
    /// Python script that produced the value (relative to control/)
    pub script: &'static str,
    /// Git commit hash of the control repo at time of run
    pub commit: &'static str,
    /// Date of the control run (ISO 8601)
    pub date: &'static str,
    /// Exact command used to produce the baseline
    pub command: &'static str,
    /// Python environment spec (conda env name or requirements file)
    pub environment: &'static str,
    /// The reference value itself
    pub value: f64,
    /// Unit or description of the value
    pub unit: &'static str,
}
