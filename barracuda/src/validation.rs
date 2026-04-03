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

use std::io::Write;
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
    /// GPU/substrate that produced this result (for cross-substrate comparison)
    pub substrate: Option<String>,
    /// Physics domain (e.g. "lattice_qcd", "dielectric", "kinetic_fluid")
    pub domain: Option<String>,
    /// Source paper citation
    pub paper: Option<String>,
    /// Unit or quantity kind
    pub unit: Option<String>,
    /// Why this tolerance value was chosen
    pub tolerance_justification: Option<String>,
    /// Wall-clock time for this check in milliseconds
    pub duration_ms: Option<u64>,
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

/// Hardware profile for a single GPU adapter (toadStool-compatible).
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Adapter name as reported by the driver.
    pub adapter: String,
    /// VRAM or max buffer size in bytes.
    pub vram_bytes: u64,
    /// Per-tier precision probe results: (tier_name, compiles, dispatch_us, max_ulp).
    pub precision_tiers: Vec<(String, bool, f64, f64)>,
    /// Domain → selected precision tier mapping.
    pub domain_routing: Vec<(String, String)>,
    /// Max lattice L dimension that fits in VRAM.
    pub max_lattice_l: usize,
}

/// Accumulates validation checks and produces a summary with exit code.
#[derive(Debug)]
#[must_use]
pub struct ValidationHarness {
    /// Name of the validation binary
    pub name: String,
    /// All checks performed
    pub checks: Vec<Check>,
    /// GPU adapter name (used in reports and comparison)
    gpu_name: Option<String>,
    /// Active substrate tag applied to subsequent checks
    active_substrate: Option<String>,
    /// Hardware profiles discovered during GPU validation
    pub hardware_profiles: Vec<HardwareProfile>,
    /// Run manifest for metadata embedding in JSON output
    pub run_manifest: Option<crate::lattice::measurement::RunManifest>,
}

impl ValidationHarness {
    /// Create a new harness for a named validation binary.
    ///
    /// # Example
    ///
    /// ```
    /// use hotspring_barracuda::validation::ValidationHarness;
    ///
    /// let mut harness = ValidationHarness::new("my_validation");
    /// harness.check_bool("feature works", true);
    /// harness.check_abs("value match", 3.14, 3.14, 0.01);
    /// assert_eq!(harness.passed_count(), 2);
    /// assert!(harness.all_passed());
    /// ```
    #[must_use = "validation harness must be used to run checks"]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            checks: Vec::new(),
            gpu_name: None,
            active_substrate: None,
            hardware_profiles: Vec::new(),
            run_manifest: None,
        }
    }

    /// Record the GPU adapter name for this validation run.
    pub fn set_gpu(&mut self, name: &str) {
        self.gpu_name = Some(name.to_string());
    }

    /// GPU adapter name, if set.
    #[must_use]
    pub fn gpu_name(&self) -> Option<&str> {
        self.gpu_name.as_deref()
    }

    /// Set the active substrate tag. All subsequent `check_*` calls carry it.
    pub fn set_substrate(&mut self, name: &str) {
        self.active_substrate = Some(name.to_string());
    }

    /// Clear the active substrate tag.
    pub fn clear_substrate(&mut self) {
        self.active_substrate = None;
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
            substrate: self.active_substrate.clone(),
            domain: None,
            paper: None,
            unit: None,
            tolerance_justification: None,
            duration_ms: None,
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
            substrate: self.active_substrate.clone(),
            domain: None,
            paper: None,
            unit: None,
            tolerance_justification: None,
            duration_ms: None,
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
            substrate: self.active_substrate.clone(),
            domain: None,
            paper: None,
            unit: None,
            tolerance_justification: None,
            duration_ms: None,
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
            substrate: self.active_substrate.clone(),
            domain: None,
            paper: None,
            unit: None,
            tolerance_justification: None,
            duration_ms: None,
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
            substrate: self.active_substrate.clone(),
            domain: None,
            paper: None,
            unit: None,
            tolerance_justification: None,
            duration_ms: None,
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
            substrate: self.active_substrate.clone(),
            domain: None,
            paper: None,
            unit: None,
            tolerance_justification: None,
            duration_ms: None,
        });
    }

    /// Annotate the most recent check with domain metadata.
    ///
    /// Call immediately after a `check_*` method to attach provenance
    /// (domain, paper citation, unit, tolerance justification).
    pub fn annotate(&mut self, domain: &str, paper: &str, unit: &str, justification: &str) {
        if let Some(last) = self.checks.last_mut() {
            last.domain = Some(domain.to_string());
            last.paper = Some(paper.to_string());
            last.unit = Some(unit.to_string());
            last.tolerance_justification = Some(justification.to_string());
        }
    }

    /// Annotate the most recent check with its wall-clock duration.
    pub fn annotate_duration(&mut self, ms: u64) {
        if let Some(last) = self.checks.last_mut() {
            last.duration_ms = Some(ms);
        }
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

    /// Write structured JSON results to a directory.
    ///
    /// Creates `<dir>/<harness_name>.json` with substrate metadata,
    /// per-check details including provenance annotations, and a summary.
    pub fn write_json(&self, dir: &str, total_duration_ms: u64) {
        let _ = std::fs::create_dir_all(dir);
        let path = format!("{}/{}.json", dir, self.name);
        let Ok(mut f) = std::fs::File::create(&path) else {
            eprintln!("warning: could not write {path}");
            return;
        };

        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("HOST"))
            .or_else(|_| std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_string()))
            .unwrap_or_else(|_| "unknown".into());

        let gpu_names: Vec<String> = self.hardware_profiles.iter()
            .map(|hp| format!("\"{}\"", escape_json(&hp.adapter)))
            .collect();
        let gpu_str = if gpu_names.is_empty() {
            String::new()
        } else {
            format!(",\n    \"gpus\": [{}]", gpu_names.join(", "))
        };

        let _ = write!(f, "{{\n  \"version\": \"1.1\",\n");
        let _ = write!(
            f,
            "  \"timestamp\": \"{}\",\n",
            chrono_rfc3339_utc()
        );
        let _ = write!(
            f,
            "  \"substrate\": {{\n    \"arch\": \"{}\",\n    \"os\": \"{}\",\n    \"hostname\": \"{}\"\n  }},\n",
            std::env::consts::ARCH,
            std::env::consts::OS,
            escape_json(&hostname)
        );
        let _ = write!(
            f,
            "  \"engine\": {{\n    \"primary\": \"cpu-native\"{gpu_str}\n  }},\n"
        );

        if let Some(ref manifest) = self.run_manifest {
            let _ = write!(f, "  \"run\": {},\n", manifest.to_json_value());
        }

        if !self.hardware_profiles.is_empty() {
            let _ = write!(f, "  \"hardware_profiles\": [\n");
            for (pi, hp) in self.hardware_profiles.iter().enumerate() {
                let pcomma = if pi + 1 < self.hardware_profiles.len() { "," } else { "" };
                let _ = write!(f, "    {{\n");
                let _ = write!(f, "      \"adapter\": \"{}\",\n", escape_json(&hp.adapter));
                let _ = write!(f, "      \"vram_bytes\": {},\n", hp.vram_bytes);
                let _ = write!(f, "      \"precision_tiers\": {{\n");
                for (ti, (name, compiles, dispatch, ulp)) in hp.precision_tiers.iter().enumerate() {
                    let tcomma = if ti + 1 < hp.precision_tiers.len() { "," } else { "" };
                    let _ = write!(f,
                        "        \"{name}\": {{\"compiles\": {compiles}, \"dispatch_us\": {}, \"max_ulp\": {}}}{tcomma}\n",
                        json_f64(*dispatch), json_f64(*ulp));
                }
                let _ = write!(f, "      }},\n");
                let _ = write!(f, "      \"domain_routing\": {{\n");
                for (di, (domain, tier)) in hp.domain_routing.iter().enumerate() {
                    let dcomma = if di + 1 < hp.domain_routing.len() { "," } else { "" };
                    let _ = write!(f, "        \"{domain}\": \"{tier}\"{dcomma}\n");
                }
                let _ = write!(f, "      }},\n");
                let _ = write!(f, "      \"sizing\": {{\"max_lattice_l\": {}, \"max_buffer_bytes\": {}}}\n",
                    hp.max_lattice_l, hp.vram_bytes);
                let _ = write!(f, "    }}{pcomma}\n");
            }
            let _ = write!(f, "  ],\n");
        }

        let _ = write!(f, "  \"checks\": [\n");

        for (i, c) in self.checks.iter().enumerate() {
            let comma = if i + 1 < self.checks.len() { "," } else { "" };
            let mode_str = match c.mode {
                ToleranceMode::Absolute => "absolute",
                ToleranceMode::Relative => "relative",
                ToleranceMode::Percentage => "percentage",
                ToleranceMode::UpperBound => "upper_bound",
                ToleranceMode::LowerBound => "lower_bound",
            };

            let _ = write!(f, "    {{\n");
            let _ = write!(f, "      \"label\": \"{}\",\n", c.label);
            let _ = write!(f, "      \"passed\": {},\n", c.passed);
            let _ = write!(f, "      \"observed\": {},\n", json_f64(c.observed));
            let _ = write!(f, "      \"expected\": {},\n", json_f64(c.expected));
            let _ = write!(f, "      \"tolerance\": {},\n", json_f64(c.tolerance));
            let _ = write!(f, "      \"mode\": \"{mode_str}\"");

            if let Some(ref sub) = c.substrate {
                let _ = write!(f, ",\n      \"substrate\": \"{}\"", escape_json(sub));
            }
            if let Some(ref d) = c.domain {
                let _ = write!(f, ",\n      \"domain\": \"{d}\"");
            }
            if let Some(ref p) = c.paper {
                let _ = write!(f, ",\n      \"paper\": \"{p}\"");
            }
            if let Some(ref u) = c.unit {
                let _ = write!(f, ",\n      \"unit\": \"{u}\"");
            }
            if let Some(ref tj) = c.tolerance_justification {
                let _ = write!(f, ",\n      \"tolerance_justification\": \"{}\"", escape_json(tj));
            }
            if let Some(ms) = c.duration_ms {
                let _ = write!(f, ",\n      \"duration_ms\": {ms}");
            }
            let _ = write!(f, "\n    }}{comma}\n");
        }

        let _ = write!(f, "  ],\n");
        let _ = write!(
            f,
            "  \"summary\": {{\n    \"total\": {},\n    \"passed\": {},\n    \"failed\": {},\n    \"duration_ms\": {}\n  }}\n",
            self.total_count(),
            self.passed_count(),
            self.total_count() - self.passed_count(),
            total_duration_ms
        );
        let _ = write!(f, "}}");
        println!("JSON results → {path}");
    }

    /// Print summary and exit with appropriate code.
    ///
    /// Exit 0 if all checks pass, exit 1 if any fails.
    /// Terminates the process; use only at the end of `main`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hotspring_barracuda::validation::ValidationHarness;
    ///
    /// let mut harness = ValidationHarness::new("example");
    /// harness.check_bool("ok", true);
    /// harness.finish();  // exits 0
    /// ```
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
    /// Print provenance records for all baselines used by this validation binary.
    ///
    /// Call this before running checks to produce a machine-readable
    /// provenance section linking every expected value to its origin.
    pub fn print_provenance(&self, baselines: &[&crate::provenance::BaselineProvenance]) {
        println!(
            "═══ {}: provenance ({} baselines) ═══",
            self.name,
            baselines.len()
        );
        for bp in baselines {
            println!(
                "  {}: value={:.6e} {}\n    script: {}\n    commit: {}\n    date: {}\n    command: {}",
                bp.label, bp.value, bp.unit, bp.script, bp.commit, bp.date, bp.command
            );
        }
        println!();
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

/// JSONL telemetry writer for structured validation output.
///
/// Writes one JSON object per line to a sidecar file. Each event has a
/// timestamp, section, observable name, and value. Designed for consumption
/// by petalTongue or any JSONL-aware tool.
pub struct TelemetryWriter {
    file: Option<std::io::BufWriter<std::fs::File>>,
    start: std::time::Instant,
    substrate: Option<String>,
}

impl TelemetryWriter {
    /// Create a new telemetry writer. If the file cannot be opened, logging
    /// is silently disabled (validation still runs).
    pub fn new(path: &str) -> Self {
        let file = std::fs::File::create(path)
            .ok()
            .map(std::io::BufWriter::new);
        if file.is_some() {
            log::info!("Telemetry → {path}");
        }
        Self {
            file,
            start: std::time::Instant::now(),
            substrate: None,
        }
    }

    /// Create a telemetry writer using discovery-based path resolution.
    ///
    /// Resolves the output directory via `discovery::telemetry_path()`,
    /// which checks `HOTSPRING_TELEMETRY_DIR`, then `<data_root>/telemetry/`,
    /// then falls back to CWD.
    pub fn discover(filename: &str) -> Self {
        let path = crate::discovery::telemetry_path(filename);
        let display = path.display().to_string();
        let file = std::fs::File::create(&path)
            .ok()
            .map(std::io::BufWriter::new);
        if file.is_some() {
            log::info!("Telemetry → {display}");
        }
        Self {
            file,
            start: std::time::Instant::now(),
            substrate: None,
        }
    }

    /// Create a no-op writer (for when telemetry is disabled).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            file: None,
            start: std::time::Instant::now(),
            substrate: None,
        }
    }

    /// Tag all subsequent telemetry events with the GPU/substrate name.
    #[must_use]
    pub fn with_substrate(mut self, name: String) -> Self {
        self.substrate = Some(name);
        self
    }

    /// Log a telemetry event. Fields with NaN/Inf are written as null.
    pub fn log(&mut self, section: &str, observable: &str, value: f64) {
        let sub = substrate_fragment(&self.substrate);
        let Some(ref mut f) = self.file else { return };
        let t = self.start.elapsed().as_secs_f64();
        let val = if value.is_finite() {
            format!("{value:.6e}")
        } else {
            "null".to_string()
        };
        let _ = writeln!(
            f,
            r#"{{"t":{t:.3},"section":"{section}","obs":"{observable}","val":{val}{sub}}}"#
        );
        let _ = f.flush();
    }

    /// Log a telemetry event with multiple key-value pairs.
    pub fn log_map(&mut self, section: &str, fields: &[(&str, f64)]) {
        let sub = substrate_fragment(&self.substrate);
        let Some(ref mut f) = self.file else { return };
        let t = self.start.elapsed().as_secs_f64();
        let pairs: Vec<String> = fields
            .iter()
            .map(|(k, v)| {
                let val = if v.is_finite() {
                    format!("{v:.6e}")
                } else {
                    "null".to_string()
                };
                format!(r#""{k}":{val}"#)
            })
            .collect();
        let _ = writeln!(
            f,
            r#"{{"t":{t:.3},"section":"{section}",{}{sub}}}"#,
            pairs.join(",")
        );
        let _ = f.flush();
    }
}

fn chrono_rfc3339_utc() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let rem = secs % 86400;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let s = rem % 60;
    // Convert days since epoch to Y-M-D (simplified Gregorian)
    let (y, mo, d) = epoch_days_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn epoch_days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Days since 1970-01-01
    let mut y = 1970;
    loop {
        let ylen = if is_leap(y) { 366 } else { 365 };
        if days < ylen {
            break;
        }
        days -= ylen;
        y += 1;
    }
    let leap = is_leap(y);
    let mdays = [
        31,
        if leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut mo = 0;
    for (i, &md) in mdays.iter().enumerate() {
        if days < md {
            mo = i as u64 + 1;
            break;
        }
        days -= md;
    }
    (y, mo, days + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn json_f64(v: f64) -> String {
    if v.is_finite() {
        // Preserve full precision, strip unnecessary trailing noise
        let s = format!("{v}");
        s
    } else if v.is_nan() {
        "null".into()
    } else {
        "null".into()
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn substrate_fragment(substrate: &Option<String>) -> String {
    match substrate {
        Some(s) => format!(r#","substrate":"{s}""#),
        None => String::new(),
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
