// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structured validation report types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSuite {
    pub plumed_version: Option<String>,
    pub gromacs_version: Option<String>,
    pub targets: Vec<(String, TargetReport)>,
    pub total_pass: usize,
    pub total_fail: usize,
    pub total_skip: usize,
    pub elapsed_ms: u64,
}

impl ValidationSuite {
    pub fn new() -> Self {
        Self {
            plumed_version: None,
            gromacs_version: None,
            targets: Vec::new(),
            total_pass: 0,
            total_fail: 0,
            total_skip: 0,
            elapsed_ms: 0,
        }
    }

    pub fn compute_summary(&mut self) {
        self.total_pass = self.targets.iter().filter(|(_, r)| r.industry_standard).count();
        self.total_fail = self.targets.iter().filter(|(_, r)| !r.industry_standard && !r.skipped).count();
        self.total_skip = self.targets.iter().filter(|(_, r)| r.skipped).count();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetReport {
    pub target_id: String,
    pub method: String,
    pub passes: Vec<String>,
    pub fails: Vec<String>,
    pub pass_rate: f64,
    pub industry_standard: bool,
    pub skipped: bool,
    pub metrics: serde_json::Value,
}

impl TargetReport {
    pub fn skipped(target_id: &str) -> Self {
        Self {
            target_id: target_id.to_string(),
            method: String::new(),
            passes: Vec::new(),
            fails: Vec::new(),
            pass_rate: 0.0,
            industry_standard: false,
            skipped: true,
            metrics: serde_json::Value::Null,
        }
    }

    pub fn compute_rates(&mut self) {
        let total = self.passes.len() + self.fails.len();
        self.pass_rate = if total > 0 {
            self.passes.len() as f64 / total as f64
        } else {
            0.0
        };
        self.industry_standard = self.fails.is_empty() && !self.passes.is_empty();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasinMatch {
    pub name: String,
    pub ref_x: f64,
    pub ref_y: Option<f64>,
    pub found_x: f64,
    pub found_y: Option<f64>,
    pub found_energy: f64,
    pub distance: f64,
    pub within_tolerance: bool,
}
