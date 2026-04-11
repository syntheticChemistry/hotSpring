// SPDX-License-Identifier: AGPL-3.0-or-later

//! [`ValidationHarness`] — accumulates validation checks and produces
//! a summary with exit code. Used by all `validate_*` binaries.

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

    fn push_check(&mut self, label: &str, passed: bool, observed: f64, expected: f64, tolerance: f64, mode: ToleranceMode) {
        self.checks.push(Check {
            label: label.to_string(),
            passed,
            observed,
            expected,
            tolerance,
            mode,
            substrate: self.active_substrate.clone(),
            domain: None,
            paper: None,
            unit: None,
            tolerance_justification: None,
            duration_ms: None,
        });
    }

    /// Add an absolute tolerance check: |observed - expected| < tolerance
    pub fn check_abs(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let passed = (observed - expected).abs() < tolerance;
        self.push_check(label, passed, observed, expected, tolerance, ToleranceMode::Absolute);
    }

    /// Add a relative tolerance check: |observed - expected| / |expected| < tolerance
    pub fn check_rel(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let passed = if expected.abs() > f64::EPSILON {
            ((observed - expected) / expected).abs() < tolerance
        } else {
            observed.abs() < tolerance
        };
        self.push_check(label, passed, observed, expected, tolerance, ToleranceMode::Relative);
    }

    /// Add an upper-bound check: observed < threshold
    pub fn check_upper(&mut self, label: &str, observed: f64, threshold: f64) {
        self.push_check(label, observed < threshold, observed, threshold, threshold, ToleranceMode::UpperBound);
    }

    /// Add a lower-bound check: observed > threshold
    pub fn check_lower(&mut self, label: &str, observed: f64, threshold: f64) {
        self.push_check(label, observed > threshold, observed, threshold, threshold, ToleranceMode::LowerBound);
    }

    /// Add a combined absolute-or-relative check.
    ///
    /// Passes if EITHER |observed - expected| < tolerance
    /// OR |observed - expected| / |expected| < tolerance.
    pub fn check_abs_or_rel(&mut self, label: &str, observed: f64, expected: f64, tolerance: f64) {
        let abs_err = (observed - expected).abs();
        let rel_err = if expected.abs() > crate::tolerances::NEAR_ZERO_EXPECTED {
            abs_err / expected.abs()
        } else {
            abs_err
        };
        let passed = abs_err < tolerance || rel_err < tolerance;
        self.push_check(label, passed, observed, expected, tolerance, ToleranceMode::Absolute);
    }

    /// Add a boolean pass/fail check.
    pub fn check_bool(&mut self, label: &str, passed: bool) {
        self.push_check(label, passed, f64::from(u8::from(passed)), 1.0, 0.0, ToleranceMode::Absolute);
    }

    /// Annotate the most recent check with domain metadata.
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

        let gpu_names: Vec<String> = self
            .hardware_profiles
            .iter()
            .map(|hp| format!("\"{}\"", escape_json(&hp.adapter)))
            .collect();
        let gpu_str = if gpu_names.is_empty() {
            String::new()
        } else {
            format!(",\n    \"gpus\": [{}]", gpu_names.join(", "))
        };

        let _ = write!(f, "{{\n  \"version\": \"1.1\",\n");
        let _ = writeln!(f, "  \"timestamp\": \"{}\",", chrono_rfc3339_utc());
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
            let _ = writeln!(f, "  \"run\": {},", manifest.to_json_value());
        }

        write_hardware_profiles(&mut f, &self.hardware_profiles);
        write_checks_json(&mut f, &self.checks);

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

    /// Print provenance records for all baselines used by this validation binary.
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

    /// Format the validation summary as a string (for testing; `finish` prints and exits).
    #[cfg(test)]
    pub fn format_summary(&self) -> String {
        use std::fmt::Write as FmtWrite;
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

fn write_hardware_profiles(f: &mut std::fs::File, profiles: &[HardwareProfile]) {
    if profiles.is_empty() {
        return;
    }
    let _ = writeln!(f, "  \"hardware_profiles\": [");
    for (pi, hp) in profiles.iter().enumerate() {
        let pcomma = if pi + 1 < profiles.len() { "," } else { "" };
        let _ = writeln!(f, "    {{");
        let _ = writeln!(f, "      \"adapter\": \"{}\",", escape_json(&hp.adapter));
        let _ = writeln!(f, "      \"vram_bytes\": {},", hp.vram_bytes);
        let _ = writeln!(f, "      \"precision_tiers\": {{");
        for (ti, (name, compiles, dispatch, ulp)) in hp.precision_tiers.iter().enumerate() {
            let tcomma = if ti + 1 < hp.precision_tiers.len() { "," } else { "" };
            let _ = writeln!(
                f,
                "        \"{name}\": {{\"compiles\": {compiles}, \"dispatch_us\": {}, \"max_ulp\": {}}}{tcomma}",
                json_f64(*dispatch),
                json_f64(*ulp)
            );
        }
        let _ = writeln!(f, "      }},");
        let _ = writeln!(f, "      \"domain_routing\": {{");
        for (di, (domain, tier)) in hp.domain_routing.iter().enumerate() {
            let dcomma = if di + 1 < hp.domain_routing.len() { "," } else { "" };
            let _ = writeln!(f, "        \"{domain}\": \"{tier}\"{dcomma}");
        }
        let _ = writeln!(f, "      }},");
        let _ = writeln!(
            f,
            "      \"sizing\": {{\"max_lattice_l\": {}, \"max_buffer_bytes\": {}}}",
            hp.max_lattice_l, hp.vram_bytes
        );
        let _ = writeln!(f, "    }}{pcomma}");
    }
    let _ = writeln!(f, "  ],");
}

fn write_checks_json(f: &mut std::fs::File, checks: &[Check]) {
    let _ = writeln!(f, "  \"checks\": [");
    for (i, c) in checks.iter().enumerate() {
        let comma = if i + 1 < checks.len() { "," } else { "" };
        let mode_str = match c.mode {
            ToleranceMode::Absolute => "absolute",
            ToleranceMode::Relative => "relative",
            ToleranceMode::Percentage => "percentage",
            ToleranceMode::UpperBound => "upper_bound",
            ToleranceMode::LowerBound => "lower_bound",
        };
        let _ = writeln!(f, "    {{");
        let _ = writeln!(f, "      \"label\": \"{}\",", c.label);
        let _ = writeln!(f, "      \"passed\": {},", c.passed);
        let _ = writeln!(f, "      \"observed\": {},", json_f64(c.observed));
        let _ = writeln!(f, "      \"expected\": {},", json_f64(c.expected));
        let _ = writeln!(f, "      \"tolerance\": {},", json_f64(c.tolerance));
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
    let _ = writeln!(f, "  ],");
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
    let (y, mo, d) = epoch_days_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn epoch_days_to_ymd(mut days: u64) -> (u64, u64, u64) {
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
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
}

fn json_f64(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
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
