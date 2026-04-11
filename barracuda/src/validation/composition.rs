// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ecosystem-converged composition validation patterns.
//!
//! Absorbed from primalSpring / groundSpring / wetSpring / healthSpring.
//! These complement the physics-rich [`super::ValidationHarness`] with
//! patterns needed for NUCLEUS composition validation and CI pipelines.

use std::io::Write;
use std::process;

/// Zero-panic exit trait for validation binaries.
///
/// Replaces verbose `let Ok(v) = expr else { log::error!(...); process::exit(1); }`
/// boilerplate with a clean `.or_exit(msg)` call.
pub trait OrExit<T> {
    /// Unwrap the value or print `msg` to stderr and exit with code 1.
    fn or_exit(self, msg: &str) -> T;
}

impl<T, E: std::fmt::Display> OrExit<T> for Result<T, E> {
    fn or_exit(self, msg: &str) -> T {
        match self {
            Ok(v) => v,
            Err(e) => {
                log::error!("{msg}: {e}");
                process::exit(1);
            }
        }
    }
}

impl<T> OrExit<T> for Option<T> {
    fn or_exit(self, msg: &str) -> T {
        self.unwrap_or_else(|| {
            log::error!("{msg}");
            process::exit(1);
        })
    }
}

/// Outcome of a single composition validation check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckOutcome {
    Pass,
    Fail,
    Skip,
}

/// Pluggable output destination for composition validation checks.
pub trait ValidationSink: std::fmt::Debug + Send + Sync {
    /// Emit a single check result.
    fn on_check(&self, outcome: CheckOutcome, name: &str, detail: &str);
    /// Begin a named section of checks.
    fn section(&self, _name: &str) {}
    /// Write a summary footer after all checks are emitted.
    fn write_summary(&self, _passed: u32, _failed: u32, _skipped: u32) {}
}

/// Default sink that writes check results to stdout.
#[derive(Debug)]
pub struct StdoutSink;

impl ValidationSink for StdoutSink {
    fn on_check(&self, outcome: CheckOutcome, name: &str, detail: &str) {
        let tag = match outcome {
            CheckOutcome::Pass => "PASS",
            CheckOutcome::Fail => "FAIL",
            CheckOutcome::Skip => "SKIP",
        };
        println!("  [{tag}] {name}: {detail}");
    }

    fn section(&self, name: &str) {
        println!("\n--- {name} ---");
    }

    fn write_summary(&self, passed: u32, failed: u32, skipped: u32) {
        let total = passed + failed;
        print!("  Summary: {passed}/{total} passed");
        if skipped > 0 {
            print!(" ({skipped} skipped)");
        }
        println!();
    }
}

/// Sink that discards all output (useful for tests).
#[derive(Debug)]
pub struct NullSink;

impl ValidationSink for NullSink {
    fn on_check(&self, _: CheckOutcome, _: &str, _: &str) {}
}

/// Newline-delimited JSON sink for streaming validation output.
#[derive(Debug)]
pub struct NdjsonSink<W: std::io::Write + std::fmt::Debug + Send + Sync> {
    writer: std::sync::Mutex<W>,
}

impl<W: std::io::Write + std::fmt::Debug + Send + Sync> NdjsonSink<W> {
    /// Create a new NDJSON sink writing to the given destination.
    pub const fn new(writer: W) -> Self {
        Self {
            writer: std::sync::Mutex::new(writer),
        }
    }
}

impl NdjsonSink<std::io::Stdout> {
    /// Create a sink that writes NDJSON to stdout.
    #[must_use]
    pub fn stdout() -> Self {
        Self::new(std::io::stdout())
    }
}

impl<W: std::io::Write + std::fmt::Debug + Send + Sync> ValidationSink for NdjsonSink<W> {
    fn on_check(&self, outcome: CheckOutcome, name: &str, detail: &str) {
        let tag = match outcome {
            CheckOutcome::Pass => "pass",
            CheckOutcome::Fail => "fail",
            CheckOutcome::Skip => "skip",
        };
        let line = format!(
            "{{\"outcome\":\"{tag}\",\"name\":{},\"detail\":{}}}",
            serde_json::json!(name),
            serde_json::json!(detail),
        );
        if let Ok(mut w) = self.writer.lock() {
            let _ = writeln!(w, "{line}");
        }
    }

    fn section(&self, name: &str) {
        let line = format!("{{\"section\":{}}}", serde_json::json!(name));
        if let Ok(mut w) = self.writer.lock() {
            let _ = writeln!(w, "{line}");
        }
    }

    fn write_summary(&self, passed: u32, failed: u32, skipped: u32) {
        let line = format!(
            "{{\"summary\":{{\"passed\":{passed},\"failed\":{failed},\"skipped\":{skipped}}}}}"
        );
        if let Ok(mut w) = self.writer.lock() {
            let _ = writeln!(w, "{line}");
        }
    }
}

/// Composition validation result for NUCLEUS / primal integration tests.
#[derive(Debug)]
pub struct CompositionResult {
    /// Experiment or validation suite name.
    pub experiment: String,
    /// Number of checks that passed.
    pub passed: u32,
    /// Number of checks that failed.
    pub failed: u32,
    /// Number of checks that were skipped.
    pub skipped: u32,
    /// Individual check results in execution order.
    pub checks: Vec<CompositionCheck>,
    sink: std::sync::Arc<dyn ValidationSink>,
}

/// Result of a single named composition check.
#[derive(Debug)]
pub struct CompositionCheck {
    /// Check identifier.
    pub name: String,
    /// Whether this check passed, failed, or was skipped.
    pub outcome: CheckOutcome,
    /// Human-readable detail.
    pub detail: String,
}

impl CompositionResult {
    /// Create a new composition validation result.
    #[must_use]
    pub fn new(experiment: &str) -> Self {
        Self {
            experiment: experiment.to_owned(),
            passed: 0,
            failed: 0,
            skipped: 0,
            checks: Vec::new(),
            sink: std::sync::Arc::new(StdoutSink),
        }
    }

    /// Replace the output sink (builder-style).
    #[must_use]
    pub fn with_sink(mut self, sink: std::sync::Arc<dyn ValidationSink>) -> Self {
        self.sink = sink;
        self
    }

    /// Begin a named section of checks.
    pub fn section(&self, name: &str) {
        self.sink.section(name);
    }

    /// Record a boolean pass/fail check.
    pub fn check_bool(&mut self, name: &str, condition: bool, detail: &str) {
        let outcome = if condition {
            self.passed += 1;
            CheckOutcome::Pass
        } else {
            self.failed += 1;
            CheckOutcome::Fail
        };
        self.sink.on_check(outcome, name, detail);
        self.checks.push(CompositionCheck {
            name: name.to_owned(),
            outcome,
            detail: detail.to_owned(),
        });
    }

    /// Record a check that cannot be evaluated yet (needs live primals).
    pub fn check_skip(&mut self, name: &str, reason: &str) {
        self.skipped += 1;
        self.sink.on_check(CheckOutcome::Skip, name, reason);
        self.checks.push(CompositionCheck {
            name: name.to_owned(),
            outcome: CheckOutcome::Skip,
            detail: reason.to_owned(),
        });
    }

    /// Check that a latency measurement is within an acceptable bound.
    pub fn check_latency(&mut self, name: &str, actual_us: u64, max_us: u64) {
        let ok = actual_us <= max_us;
        let detail = format!("{actual_us}\u{03bc}s (max: {max_us}\u{03bc}s)");
        self.check_bool(name, ok, &detail);
    }

    /// Conditionally run a check or skip based on a prerequisite.
    pub fn check_or_skip<T, F>(
        &mut self,
        name: &str,
        prerequisite: Option<T>,
        skip_reason: &str,
        check: F,
    ) where
        F: FnOnce(T, &mut Self),
    {
        match prerequisite {
            Some(val) => check(val, self),
            None => self.check_skip(name, skip_reason),
        }
    }

    /// All non-skipped checks passed and at least one check was evaluated.
    #[must_use]
    pub const fn all_passed(&self) -> bool {
        self.failed == 0 && self.passed > 0
    }

    /// Print summary and delegate to sink.
    pub fn finish(&self) {
        println!(
            "\n{}: {}/{} checks passed{}",
            self.experiment,
            self.passed,
            self.passed + self.failed,
            if self.skipped > 0 {
                format!(" ({} skipped)", self.skipped)
            } else {
                String::new()
            }
        );
        self.sink
            .write_summary(self.passed, self.failed, self.skipped);
    }

    /// Process exit code: 0 = pass, 1 = fail.
    #[must_use]
    pub const fn exit_code(&self) -> i32 {
        if self.failed == 0 && self.passed > 0 {
            0
        } else {
            1
        }
    }

    /// Skip-aware exit code: 0 = pass, 1 = fail, 2 = all skipped.
    #[must_use]
    pub const fn exit_code_skip_aware(&self) -> i32 {
        if self.failed == 0 && self.passed > 0 {
            0
        } else if self.failed > 0 {
            1
        } else {
            2
        }
    }
}
