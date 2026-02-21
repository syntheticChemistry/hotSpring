// SPDX-License-Identifier: AGPL-3.0-only

//! HotQCD EOS Table Validation (Paper 7).
//!
//! Validates the loaded HotQCD lattice QCD equation-of-state tables
//! against known physical constraints and demonstrates the computational
//! overlap between plasma MD and lattice QCD.
//!
//! # Validation targets
//!
//! | Check | Criterion | Basis |
//! |-------|-----------|-------|
//! | Pressure monotonicity | dp/dT > 0 | Thermodynamic stability |
//! | Asymptotic freedom | p/T^4 → SB at high T | QCD running coupling |
//! | Trace anomaly peak | near T_c | Lattice QCD consensus |
//! | Thermodynamic consistency | s ≈ (ε+p)/T | First law |
//!
//! # Provenance
//!
//! Bazavov et al., PRD 90, 094503 (2014) — HotQCD continuum EOS
//! Data: `github.com/jnoronhahostler/Equation-of-State`

use hotspring_barracuda::lattice::eos_tables::{computational_overlap_summary, HotQcdEos};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  HotQCD EOS Table Validation (Paper 7)                     ║");
    println!("║  Bazavov et al., PRD 90, 094503 (2014)                     ║");
    println!("║  Plasma ↔ QCD computational bridge                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("hotqcd_eos");

    // Load reference table
    let eos = HotQcdEos::reference_table();
    println!("  Loaded {} EOS data points:", eos.points.len());
    println!("{eos}");

    // ═══ Test 1: Pressure monotonicity ═══
    println!("═══ Pressure Monotonicity ═══");
    let mut pressure_ok = true;
    for w in eos.points.windows(2) {
        if w[1].pressure < w[0].pressure {
            println!(
                "  FAIL: p decreases at T/Tc={:.2}: {:.3} -> {:.3}",
                w[0].t_over_tc, w[0].pressure, w[1].pressure
            );
            pressure_ok = false;
        }
    }
    harness.check_bool("pressure monotonicity", pressure_ok);
    if pressure_ok {
        println!("  PASS: dp/dT > 0 across all temperature points");
    }
    println!();

    // ═══ Test 2: Asymptotic freedom ═══
    println!("═══ Asymptotic Freedom ═══");
    let af = eos.check_asymptotic_freedom();
    harness.check_bool("asymptotic freedom", af);
    println!("  SB limit: p/T⁴ = {:.3}", HotQcdEos::SB_PRESSURE_OVER_T4);
    if let Some(last) = eos.points.last() {
        let ratio = last.pressure / HotQcdEos::SB_PRESSURE_OVER_T4;
        println!(
            "  At T/T_c = {:.1}: p/T⁴ = {:.3} ({:.1}% of SB)",
            last.t_over_tc,
            last.pressure,
            ratio * 100.0
        );
    }
    println!();

    // ═══ Test 3: Trace anomaly peak ═══
    println!("═══ Trace Anomaly ═══");
    let max_ta = eos
        .points
        .iter()
        .max_by(|a, b| a.trace_anomaly.total_cmp(&b.trace_anomaly))
        .expect("HotQCD table non-empty");
    println!(
        "  Peak: (ε-3p)/T⁴ = {:.4} at T/T_c = {:.2}",
        max_ta.trace_anomaly, max_ta.t_over_tc
    );
    let ta_near_tc = max_ta.t_over_tc > 0.9 && max_ta.t_over_tc < 1.3;
    harness.check_bool("trace anomaly peak near T_c", ta_near_tc);
    println!();

    // ═══ Test 4: Thermodynamic consistency ═══
    println!("═══ Thermodynamic Consistency ═══");
    let violations = eos.check_thermodynamic_consistency(tolerances::HOTQCD_CONSISTENCY);
    if violations.is_empty() {
        println!("  PASS: s ≈ (ε+p)/T within 30% at all points");
    } else {
        for (t, diff) in &violations {
            println!("  Violation at T/T_c = {t:.2}: {:.1}% error", diff * 100.0);
        }
    }
    harness.check_bool(
        "thermodynamic consistency",
        violations.len() < tolerances::HOTQCD_MAX_VIOLATIONS,
    );
    println!();

    // ═══ Test 5: Interpolation ═══
    println!("═══ Interpolation Check ═══");
    if let Some(p) = eos.interpolate(1.25) {
        println!(
            "  At T/T_c = 1.25: p/T⁴ = {:.3}, ε/T⁴ = {:.3}",
            p.pressure, p.energy_density
        );
        let cs2 = p.speed_of_sound_sq_ideal();
        println!("  c_s² (ideal approx) = {cs2:.4}");
        harness.check_lower("interpolated pressure > 0", p.pressure, 0.0);
    }
    println!();

    // ═══ Computational overlap ═══
    println!("═══ Computational Overlap ═══");
    print!("{}", computational_overlap_summary());
    println!();

    println!("═══ Summary ════════════════════════════════════════════════");
    println!("  HotQCD EOS tables validated against physical constraints.");
    println!("  Plasma MD and lattice QCD share identical computational patterns.");
    println!("  No new GPU code required — reuse existing barracuda primitives.");
    println!();

    harness.finish();
}
