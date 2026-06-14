// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parity validation — quantitative pass/fail against published reference values.
//!
//! Defines tolerance classes derived from PLUMED-NEST targets and provides
//! structured reporting for NUCLEUS parity evolution (barraCuda shaders).

use serde::{Deserialize, Serialize};

/// Tolerance class: defines acceptable deviation for a measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tolerance {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub justification: String,
    pub source_plum_id: Option<String>,
}

/// Single parity check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityCheck {
    pub metric: String,
    pub measured: f64,
    pub reference: f64,
    pub tolerance: f64,
    pub unit: String,
    pub deviation: f64,
    pub passed: bool,
}

impl ParityCheck {
    pub fn new(metric: &str, measured: f64, reference: f64, tolerance: f64, unit: &str) -> Self {
        let deviation = (measured - reference).abs();
        Self {
            metric: metric.to_string(),
            measured,
            reference,
            tolerance,
            unit: unit.to_string(),
            deviation,
            passed: deviation <= tolerance,
        }
    }

    pub fn range(metric: &str, measured: f64, range: (f64, f64), unit: &str) -> Self {
        let within = measured >= range.0 && measured <= range.1;
        let deviation = if within {
            0.0
        } else {
            (measured - range.0).abs().min((measured - range.1).abs())
        };
        Self {
            metric: metric.to_string(),
            measured,
            reference: (range.0 + range.1) / 2.0,
            tolerance: (range.1 - range.0) / 2.0,
            unit: unit.to_string(),
            deviation,
            passed: within,
        }
    }
}

/// Full parity report aggregating all checks across targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityReport {
    pub version: String,
    pub targets: Vec<TargetParity>,
    pub overall_pass_rate: f64,
    pub ready_for_nucleus: bool,
    pub tolerance_registry: Vec<Tolerance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetParity {
    pub target_id: String,
    pub method: String,
    pub checks: Vec<ParityCheck>,
    pub pass_rate: f64,
    pub all_pass: bool,
}

impl TargetParity {
    pub fn new(target_id: &str, method: &str, checks: Vec<ParityCheck>) -> Self {
        let n = checks.len();
        let passed = checks.iter().filter(|c| c.passed).count();
        let pass_rate = if n > 0 { passed as f64 / n as f64 } else { 0.0 };
        Self {
            target_id: target_id.to_string(),
            method: method.to_string(),
            checks,
            pass_rate,
            all_pass: passed == n && n > 0,
        }
    }
}

/// Industry-standard tolerance classes derived from PLUMED-NEST validation.
pub fn standard_tolerances() -> Vec<Tolerance> {
    vec![
        Tolerance {
            name: "fes_barrier_kjmol".to_string(),
            value: 5.0,
            unit: "kJ/mol".to_string(),
            justification: "Standard deviation of barrier height across block-averaged FES windows; accounts for sampling noise in well-tempered metadynamics".to_string(),
            source_plum_id: Some("19.009".to_string()),
        },
        Tolerance {
            name: "basin_position_rad".to_string(),
            value: 0.5,
            unit: "rad".to_string(),
            justification: "Grid spacing at 150-bin resolution over [-π,π]; minima within one grid cell of reference are acceptable".to_string(),
            source_plum_id: Some("19.009".to_string()),
        },
        Tolerance {
            name: "folding_fe_kjmol".to_string(),
            value: 4.0,
            unit: "kJ/mol".to_string(),
            justification: "Statistical uncertainty in protein folding ΔG from OPES reweighting with limited sampling; matches block averaging stderr".to_string(),
            source_plum_id: Some("24.029".to_string()),
        },
        Tolerance {
            name: "convergence_std_kjmol".to_string(),
            value: 3.0,
            unit: "kJ/mol".to_string(),
            justification: "Standard deviation of barrier height across last 3 convergence windows; below this indicates asymptotic convergence".to_string(),
            source_plum_id: Some("19.009".to_string()),
        },
        Tolerance {
            name: "block_stderr_kjmol".to_string(),
            value: 5.0,
            unit: "kJ/mol".to_string(),
            justification: "Maximum acceptable block averaging standard error; 5 kJ/mol ≈ 2 kT at 300K, below which thermal noise dominates".to_string(),
            source_plum_id: None,
        },
        Tolerance {
            name: "height_decay_ratio".to_string(),
            value: 0.15,
            unit: "dimensionless".to_string(),
            justification: "Ratio of final to initial Gaussian height in well-tempered metadynamics; values <0.15 indicate well-converged bias".to_string(),
            source_plum_id: Some("19.009".to_string()),
        },
        Tolerance {
            name: "puckering_theta_deg".to_string(),
            value: 10.0,
            unit: "degrees".to_string(),
            justification: "Acceptable deviation in Cremer-Pople θ for sugar pucker minima; within thermal fluctuation at 300K".to_string(),
            source_plum_id: Some("22.028".to_string()),
        },
        Tolerance {
            name: "binding_fe_kjmol".to_string(),
            value: 8.0,
            unit: "kJ/mol".to_string(),
            justification: "Acceptable deviation for absolute binding free energy; ±2 kcal/mol (±8.4 kJ/mol) is standard for computational methods".to_string(),
            source_plum_id: Some("23.004".to_string()),
        },
    ]
}

/// Generate a human-readable parity report string.
pub fn format_report(report: &ParityReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("{}\n", "═".repeat(72)));
    out.push_str("  PARITY VALIDATION REPORT — PLUMED-NEST × NUCLEUS Baseline\n");
    out.push_str(&format!("{}\n\n", "═".repeat(72)));
    out.push_str(&format!("  Version: {}\n", report.version));
    out.push_str(&format!("  Overall pass rate: {:.0}%\n", report.overall_pass_rate * 100.0));
    out.push_str(&format!("  NUCLEUS-ready: {}\n\n", if report.ready_for_nucleus { "YES" } else { "NO" }));

    for target in &report.targets {
        out.push_str(&format!("  ─── {} ({}) ───\n", target.target_id, target.method));
        for check in &target.checks {
            let status = if check.passed { "\x1b[32mPASS\x1b[0m" } else { "\x1b[31mFAIL\x1b[0m" };
            out.push_str(&format!(
                "    [{status}] {}: {:.2} {} (ref: {:.2}, tol: ±{:.2}, dev: {:.2})\n",
                check.metric, check.measured, check.unit,
                check.reference, check.tolerance, check.deviation
            ));
        }
        out.push_str(&format!("    Pass rate: {:.0}%\n\n", target.pass_rate * 100.0));
    }

    out.push_str("  ─── Tolerance Registry ───\n");
    for tol in &report.tolerance_registry {
        out.push_str(&format!(
            "    {:<25} ±{:<6.1} {} ({})\n",
            tol.name, tol.value, tol.unit,
            tol.source_plum_id.as_deref().unwrap_or("general")
        ));
    }

    out.push_str(&format!("\n{}\n", "═".repeat(72)));
    out
}
