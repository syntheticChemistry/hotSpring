// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation scenarios for QCD campaigns.
//!
//! Defines pass/fail checks against known physics, used by the validation
//! binary suite.

use super::tolerances;

/// Validation result for a single observable.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub observable: String,
    pub measured: f64,
    pub expected: f64,
    pub tolerance: f64,
    pub passed: bool,
}

impl std::fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        write!(
            f,
            "[{}] {} = {:.6} (expected {:.6} ± {:.4})",
            status, self.observable, self.measured, self.expected, self.tolerance
        )
    }
}

/// Validate a plaquette measurement against known values.
#[must_use]
pub fn validate_plaquette(beta: f64, measured: f64) -> ValidationResult {
    let expected = match beta {
        b if (b - 5.70).abs() < 0.01 => tolerances::plaquette::BETA_5_70,
        b if (b - 5.90).abs() < 0.01 => tolerances::plaquette::BETA_5_90,
        b if (b - 6.00).abs() < 0.01 => tolerances::plaquette::BETA_6_00,
        b if (b - 6.20).abs() < 0.01 => tolerances::plaquette::BETA_6_20,
        _ => 0.0,
    };
    let tolerance = tolerances::plaquette::TOLERANCE;
    let passed = (measured - expected).abs() < tolerance;

    ValidationResult {
        observable: format!("plaquette(β={beta:.2})"),
        measured,
        expected,
        tolerance,
        passed,
    }
}

/// Validate that acceptance rate is in a healthy range.
#[must_use]
pub fn validate_acceptance(rate: f64) -> ValidationResult {
    let passed = rate >= tolerances::acceptance::MIN_PRODUCTION
        && rate <= tolerances::acceptance::MAX_HEALTHY;
    ValidationResult {
        observable: "acceptance_rate".into(),
        measured: rate,
        expected: tolerances::acceptance::TARGET,
        tolerance: 0.20,
        passed,
    }
}
