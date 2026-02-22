// SPDX-License-Identifier: AGPL-3.0-only

//! Validation harness for hotSpring binaries.
//!
//! Every validation binary follows the hotSpring pattern:
//!   - Hardcoded expected values with provenance
//!   - Explicit pass/fail checks against documented tolerances
//!   - Exit code 0 (all checks pass) or 1 (any check fails)
//!   - Machine-readable summary on stdout
//!
//! This module provides the shared infrastructure.

use std::process;

/// A single validation check with result tracking.
#[derive(Debug, Clone)]
pub struct Check {
    /// Human-readable label
    pub label: String,
    /// Whether this check passed
    pub passed: bool,
    /// Observed value
    pub observed: f64,
    /// Expected value
    pub expected: f64,
    /// Tolerance used
    pub tolerance: f64,
    /// How the tolerance was applied (absolute, relative, percentage)
    pub mode: ToleranceMode,
}

/// How a tolerance threshold is applied.
#[derive(Debug, Clone, Copy)]
pub enum ToleranceMode {
    /// |observed - expected| < tolerance
    Absolute,
    /// |observed - expected| / |expected| < tolerance
    Relative,
    /// |observed - expected| / |expected| * 100 < tolerance (percentage)
    Percentage,
    /// observed < threshold (upper bound only)
    UpperBound,
    /// observed > threshold (lower bound only)
    LowerBound,
}

impl std::fmt::Display for ToleranceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absolute => write!(f, "abs"),
            Self::Relative => write!(f, "rel"),
            Self::Percentage => write!(f, "pct"),
            Self::UpperBound => write!(f, "<"),
            Self::LowerBound => write!(f, ">"),
        }
    }
}

/// Accumulates validation checks and produces a summary with exit code.
#[derive(Debug, Default)]
#[must_use]
pub struct ValidationHarness {
    /// Name of the validation binary
    pub name: String,
    /// All checks performed
    pub checks: Vec<Check>,
}

impl ValidationHarness {
    /// Create a new harness for a named validation binary.
    #[must_use = "validation harness must be used to run checks"]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            checks: Vec::new(),
        }
    }

    /// Add an absolute tolerance check: |observed - expected| < tolerance
    pub fn check_abs(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let passed = (observed - expected).abs() < tolerance;
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed,
            expected,
            tolerance,
            mode: ToleranceMode::Absolute,
        });
    }

    /// Add a relative tolerance check: |observed - expected| / |expected| < tolerance
    pub fn check_rel(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let passed = if expected.abs() > f64::EPSILON {
            ((observed - expected) / expected).abs() < tolerance
        } else {
            observed.abs() < tolerance
        };
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed,
            expected,
            tolerance,
            mode: ToleranceMode::Relative,
        });
    }

    /// Add an upper-bound check: observed < threshold
    pub fn check_upper(&mut self, label: &str, observed: f64, threshold: f64) {
        self.checks.push(Check {
            label: label.to_string(),
            passed: observed < threshold,
            observed,
            expected: threshold,
            tolerance: threshold,
            mode: ToleranceMode::UpperBound,
        });
    }

    /// Add a lower-bound check: observed > threshold
    pub fn check_lower(&mut self, label: &str, observed: f64, threshold: f64) {
        self.checks.push(Check {
            label: label.to_string(),
            passed: observed > threshold,
            observed,
            expected: threshold,
            tolerance: threshold,
            mode: ToleranceMode::LowerBound,
        });
    }

    /// Add a combined absolute-or-relative check.
    ///
    /// Passes if EITHER |observed - expected| < tolerance
    /// OR |observed - expected| / |expected| < tolerance.
    /// Mirrors the two-mode acceptance used in most validation binaries.
    pub fn check_abs_or_rel(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let abs_err = (observed - expected).abs();
        let rel_err = if expected.abs() > crate::tolerances::NEAR_ZERO_EXPECTED {
            abs_err / expected.abs()
        } else {
            abs_err
        };
        let passed = abs_err < tolerance || rel_err < tolerance;
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed,
            expected,
            tolerance,
            mode: ToleranceMode::Absolute,
        });
    }

    /// Add a boolean pass/fail check.
    pub fn check_bool(&mut self, label: &str, passed: bool) {
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed: f64::from(u8::from(passed)),
            expected: 1.0,
            tolerance: 0.0,
            mode: ToleranceMode::Absolute,
        });
    }

    /// Number of checks that passed.
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.checks.iter().filter(|c| c.passed).count()
    }

    /// Total number of checks.
    #[must_use]
    pub const fn total_count(&self) -> usize {
        self.checks.len()
    }

    /// Whether all checks passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// Print summary and exit with appropriate code.
    ///
    /// Exit 0 if all checks pass, exit 1 if any fails.
    pub fn finish(&self) -> ! {
        println!();
        println!(
            "═══ {} validation: {}/{} checks passed ═══",
            self.name,
            self.passed_count(),
            self.total_count()
        );

        for check in &self.checks {
            let icon = if check.passed { "✓" } else { "✗" };
            println!(
                "  {icon} {}: observed={:.6e}, expected={:.6e}, tol={:.2e} ({})",
                check.label, check.observed, check.expected, check.tolerance, check.mode
            );
        }

        if self.all_passed() {
            println!("ALL CHECKS PASSED");
            process::exit(0);
        } else {
            let failed: Vec<&str> = self
                .checks
                .iter()
                .filter(|c| !c.passed)
                .map(|c| c.label.as_str())
                .collect();
            println!("FAILED CHECKS: {}", failed.join(", "));
            process::exit(1);
        }
    }
}

impl ValidationHarness {
    /// Format the validation summary as a string (for testing; `finish` prints and exits).
    #[cfg(test)]
    pub fn format_summary(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "═══ {} validation: {}/{} checks passed ═══",
            self.name,
            self.passed_count(),
            self.total_count()
        );
        for check in &self.checks {
            let icon = if check.passed { "✓" } else { "✗" };
            let _ = writeln!(
                s,
                "  {icon} {}: observed={:.6e}, expected={:.6e}, tol={:.2e} ({})",
                check.label, check.observed, check.expected, check.tolerance, check.mode
            );
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_tracks_pass_fail() {
        let mut h = ValidationHarness::new("test");
        h.check_abs("exact", 1.0, 1.0, 1e-10);
        h.check_abs("close", 1.0001, 1.0, 1e-3);
        h.check_abs("far", 2.0, 1.0, 1e-3);
        assert_eq!(h.passed_count(), 2);
        assert_eq!(h.total_count(), 3);
        assert!(!h.all_passed());
    }

    #[test]
    fn harness_all_pass() {
        let mut h = ValidationHarness::new("test");
        h.check_abs("a", 1.0, 1.0, 1e-10);
        h.check_upper("b", 0.5, 1.0);
        h.check_bool("c", true);
        assert!(h.all_passed());
    }

    #[test]
    fn relative_check_handles_zero() {
        let mut h = ValidationHarness::new("test");
        h.check_rel("near_zero", 1e-15, 0.0, 1e-10);
        assert!(h.checks[0].passed);
    }

    #[test]
    fn check_rel_large_values() {
        let mut h = ValidationHarness::new("test");
        h.check_rel("large", 1e10, 1e10, 1e-6); // exact match
        assert!(h.checks[0].passed);
        h.check_rel("large_close", 1e10 * 1.0001, 1e10, 1e-3); // within 0.1%
        assert!(h.checks[1].passed);
        h.check_rel("large_far", 2e10, 1e10, 1e-3); // 100% off
        assert!(!h.checks[2].passed);
    }

    #[test]
    fn check_rel_small_values() {
        let mut h = ValidationHarness::new("test");
        h.check_rel("small", 1e-15, 1e-15, 1e-6);
        assert!(h.checks[0].passed);
        h.check_rel("small_close", 1e-14, 1e-14, 1e-2);
        assert!(h.checks[1].passed);
    }

    #[test]
    fn check_rel_negative_values() {
        let mut h = ValidationHarness::new("test");
        h.check_rel("neg_exact", -16.0, -16.0, 1e-10);
        assert!(h.checks[0].passed);
        h.check_rel("neg_close", -15.97, -16.0, 0.02); // ~0.2% rel err
        assert!(h.checks[1].passed);
        h.check_rel("neg_sign_diff", 16.0, -16.0, 0.1); // wrong sign
        assert!(!h.checks[2].passed);
    }

    #[test]
    fn check_upper_exceeds_threshold() {
        let mut h = ValidationHarness::new("test");
        h.check_upper("below", 0.5, 1.0);
        assert!(h.checks[0].passed);
        h.check_upper("at", 1.0, 1.0);
        assert!(!h.checks[1].passed); // observed < threshold is pass, equal is fail
        h.check_upper("above", 1.5, 1.0);
        assert!(!h.checks[2].passed);
    }

    #[test]
    fn check_bool_false() {
        let mut h = ValidationHarness::new("test");
        h.check_bool("fail", false);
        assert!(!h.checks[0].passed);
        assert_eq!(h.passed_count(), 0);
    }

    #[test]
    fn format_summary_no_panic() {
        let mut h = ValidationHarness::new("my_validation");
        h.check_abs("a", 1.0, 1.0, 1e-10);
        h.check_abs("b", 2.0, 1.0, 0.1);
        let s = h.format_summary();
        assert!(!s.is_empty());
        assert!(s.contains("my_validation"));
        assert_eq!(h.passed_count(), 1);
        assert!(s.contains("1/2"));
    }

    #[test]
    fn harness_zero_checks() {
        let h = ValidationHarness::new("empty");
        assert_eq!(h.passed_count(), 0);
        assert_eq!(h.total_count(), 0);
        assert!(h.all_passed()); // vacuously true for empty
    }

    #[test]
    fn name_label_handling() {
        let mut h = ValidationHarness::new("validation_binary_name");
        h.check_abs("χ²/datum", 6.62, 6.62, 0.1);
        h.check_lower("E/A (MeV)", -16.0, -20.0);
        assert_eq!(h.name, "validation_binary_name");
        assert_eq!(h.checks[0].label, "χ²/datum");
        assert_eq!(h.checks[1].label, "E/A (MeV)");
    }

    #[test]
    fn check_abs_or_rel_abs_pass() {
        let mut h = ValidationHarness::new("test");
        h.check_abs_or_rel("abs_pass", 1.0, 1.0, 1e-10);
        assert!(h.checks[0].passed);
    }

    #[test]
    fn check_abs_or_rel_rel_pass() {
        let mut h = ValidationHarness::new("test");
        // abs_err = 100, tolerance = 1e-10 → abs fails
        // rel_err = 100/1e12 ≈ 1e-10 → rel passes with tol 1e-9
        h.check_abs_or_rel("rel_pass", 1e12 + 100.0, 1e12, 1e-9);
        assert!(h.checks[0].passed);
    }

    #[test]
    fn check_abs_or_rel_both_fail() {
        let mut h = ValidationHarness::new("test");
        h.check_abs_or_rel("both_fail", 10.0, 1.0, 0.1);
        assert!(!h.checks[0].passed);
    }

    #[test]
    fn check_lower_pass() {
        let mut h = ValidationHarness::new("test");
        h.check_lower("lower_pass", -15.0, -20.0);
        assert!(h.checks[0].passed);
    }

    #[test]
    fn check_lower_fail() {
        let mut h = ValidationHarness::new("test");
        h.check_lower("lower_fail", -25.0, -20.0);
        assert!(!h.checks[0].passed);
    }

    #[test]
    fn tolerance_mode_display_all_variants() {
        assert_eq!(ToleranceMode::Absolute.to_string(), "abs");
        assert_eq!(ToleranceMode::Relative.to_string(), "rel");
        assert_eq!(ToleranceMode::Percentage.to_string(), "pct");
        assert_eq!(ToleranceMode::UpperBound.to_string(), "<");
        assert_eq!(ToleranceMode::LowerBound.to_string(), ">");
    }

    #[test]
    fn format_summary_all_check_types() {
        let mut h = ValidationHarness::new("full_coverage");
        h.check_abs("abs", 1.0, 1.0, 1e-10);
        h.check_rel("rel", 1.0, 1.0, 1e-6);
        h.check_upper("upper", 0.5, 1.0);
        h.check_lower("lower", 2.0, 1.0);
        h.check_abs_or_rel("abs_or_rel", 1.0, 1.0, 0.1);
        h.check_bool("bool", true);
        let s = h.format_summary();
        assert!(s.contains("full_coverage"));
        assert!(s.contains("abs"));
        assert!(s.contains("rel"));
        assert!(s.contains("upper"));
        assert!(s.contains("lower"));
        assert!(s.contains("abs_or_rel"));
        assert!(s.contains("bool"));
        assert_eq!(h.passed_count(), 6);
        assert_eq!(h.total_count(), 6);
    }

    #[test]
    fn check_abs_or_rel_near_zero_expected() {
        let mut h = ValidationHarness::new("test");
        // When expected is near zero, rel_err = abs_err (avoids div by zero)
        h.check_abs_or_rel("tiny_obs_near_zero_exp", 1e-15, 0.0, 1e-10);
        assert!(h.checks[0].passed, "abs_err 1e-15 < 1e-10 should pass");
        h.check_abs_or_rel("larger_obs_near_zero_exp", 0.1, 1e-20, 0.01);
        assert!(!h.checks[1].passed, "abs_err 0.1 > 0.01 should fail");
    }

    #[test]
    fn check_rel_exact_zero_expected() {
        let mut h = ValidationHarness::new("test");
        h.check_rel("obs_small", 1e-16, 0.0, 1e-10);
        assert!(h.checks[0].passed, "|obs| < tol when expected=0");
        h.check_rel("obs_large", 1.0, 0.0, 1e-10);
        assert!(!h.checks[1].passed, "|obs| > tol when expected=0");
    }

    #[test]
    fn check_upper_boundary_equal_fails() {
        let mut h = ValidationHarness::new("test");
        h.check_upper("at_threshold", 1.0, 1.0);
        assert!(!h.checks[0].passed, "observed < threshold; equal fails");
    }

    #[test]
    fn check_lower_boundary_equal_fails() {
        let mut h = ValidationHarness::new("test");
        h.check_lower("at_threshold", 1.0, 1.0);
        assert!(!h.checks[0].passed, "observed > threshold; equal fails");
    }

    #[test]
    fn format_summary_includes_failed_icon() {
        let mut h = ValidationHarness::new("test");
        h.check_abs("pass", 1.0, 1.0, 0.1);
        h.check_abs("fail", 2.0, 1.0, 0.01);
        let s = h.format_summary();
        assert!(s.contains('✓') || s.contains("pass"));
        assert!(s.contains('✗') || s.contains("fail"));
        assert!(s.contains("1/2"));
    }
}
