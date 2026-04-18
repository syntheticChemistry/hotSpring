// SPDX-License-Identifier: AGPL-3.0-or-later

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
    h.check_rel("large", 1e10, 1e10, 1e-6);
    assert!(h.checks[0].passed);
    h.check_rel("large_close", 1e10 * 1.0001, 1e10, 1e-3);
    assert!(h.checks[1].passed);
    h.check_rel("large_far", 2e10, 1e10, 1e-3);
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
    h.check_rel("neg_close", -15.97, -16.0, 0.02);
    assert!(h.checks[1].passed);
    h.check_rel("neg_sign_diff", 16.0, -16.0, 0.1);
    assert!(!h.checks[2].passed);
}

#[test]
fn check_upper_exceeds_threshold() {
    let mut h = ValidationHarness::new("test");
    h.check_upper("below", 0.5, 1.0);
    assert!(h.checks[0].passed);
    h.check_upper("at", 1.0, 1.0);
    assert!(!h.checks[1].passed);
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
    assert!(h.all_passed());
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
    assert_eq!(h.passed_count(), 6);
    assert_eq!(h.total_count(), 6);
}

#[test]
fn check_abs_or_rel_near_zero_expected() {
    let mut h = ValidationHarness::new("test");
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

// ── Composition validation tests ────────────────────────────────

#[test]
fn composition_result_pass_fail() {
    let mut r = CompositionResult::new("test").with_sink(std::sync::Arc::new(ValidationSink::Null));
    r.check_bool("ok", true, "yes");
    r.check_bool("fail", false, "no");
    assert_eq!(r.passed, 1);
    assert_eq!(r.failed, 1);
    assert!(!r.all_passed());
    assert_eq!(r.exit_code(), 1);
}

#[test]
fn composition_result_all_pass() {
    let mut r = CompositionResult::new("test").with_sink(std::sync::Arc::new(ValidationSink::Null));
    r.check_bool("a", true, "ok");
    r.check_bool("b", true, "ok");
    assert!(r.all_passed());
    assert_eq!(r.exit_code(), 0);
}

#[test]
fn composition_result_skip() {
    let mut r = CompositionResult::new("test").with_sink(std::sync::Arc::new(ValidationSink::Null));
    r.check_skip("primal_health", "biomeOS not available");
    assert_eq!(r.skipped, 1);
    assert_eq!(r.passed, 0);
    assert_eq!(r.failed, 0);
    assert!(!r.all_passed());
}

#[test]
fn composition_result_exit_code_skip_aware() {
    let mut r = CompositionResult::new("test").with_sink(std::sync::Arc::new(ValidationSink::Null));
    r.check_skip("a", "no primals");
    assert_eq!(r.exit_code_skip_aware(), 2, "all skipped = exit 2");

    let mut r2 = CompositionResult::new("test").with_sink(std::sync::Arc::new(ValidationSink::Null));
    r2.check_bool("ok", true, "yes");
    r2.check_skip("b", "no primal");
    assert_eq!(r2.exit_code_skip_aware(), 0, "pass + skip = exit 0");

    let mut r3 = CompositionResult::new("test").with_sink(std::sync::Arc::new(ValidationSink::Null));
    r3.check_bool("fail", false, "no");
    r3.check_skip("b", "no primal");
    assert_eq!(r3.exit_code_skip_aware(), 1, "fail + skip = exit 1");
}

#[test]
fn composition_result_check_or_skip() {
    let mut r = CompositionResult::new("test").with_sink(std::sync::Arc::new(ValidationSink::Null));
    r.check_or_skip("with_value", Some(42), "no value", |val, cr| {
        cr.check_bool("got_42", val == 42, "expected 42");
    });
    assert_eq!(r.passed, 1);

    r.check_or_skip::<i32, _>("no_value", None, "not available", |_, _| {
        unreachable!();
    });
    assert_eq!(r.skipped, 1);
}

#[test]
fn composition_result_latency() {
    let mut r = CompositionResult::new("test").with_sink(std::sync::Arc::new(ValidationSink::Null));
    r.check_latency("fast", 100, 50_000);
    assert_eq!(r.passed, 1);
    r.check_latency("slow", 100_000, 50_000);
    assert_eq!(r.failed, 1);
}

#[test]
fn or_exit_result_ok() {
    let r: Result<i32, &str> = Ok(42);
    assert_eq!(r.or_exit("should not exit"), 42);
}

#[test]
fn or_exit_option_some() {
    let o: Option<i32> = Some(99);
    assert_eq!(o.or_exit("should not exit"), 99);
}

#[test]
fn check_outcome_equality() {
    assert_eq!(CheckOutcome::Pass, CheckOutcome::Pass);
    assert_ne!(CheckOutcome::Pass, CheckOutcome::Fail);
    assert_ne!(CheckOutcome::Fail, CheckOutcome::Skip);
}

#[test]
fn ndjson_sink_does_not_panic() {
    let sink = ValidationSink::ndjson();
    sink.on_check(CheckOutcome::Pass, "test", "ok");
    sink.on_check(CheckOutcome::Skip, "skipped", "no primal");
    sink.section("section_name");
    sink.write_summary(1, 0, 1);
}
