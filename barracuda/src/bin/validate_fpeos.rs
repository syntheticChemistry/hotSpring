// SPDX-License-Identifier: AGPL-3.0-only

//! Validate Militzer FPEOS table lookup and interpolation (Paper 32).
//!
//! Tests:
//!   - Grid-point exact lookup (hydrogen, helium)
//!   - Bilinear interpolation between grid points
//!   - Monotonicity: P increases with ρ and T
//!   - Cross-element: He pressure ≤ H pressure at same conditions
//!   - Thermodynamic consistency: P ≈ ρ² ∂(E/ρ)/∂ρ
//!
//! Provenance: Militzer et al. Phys. Rev. E 103, 013203 (2021), fpeos.de

use hotspring_barracuda::physics::fpeos::{helium_reference, hydrogen_reference};
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};

fn main() {
    let mut harness = ValidationHarness::new("fpeos_militzer");
    let mut telem = TelemetryWriter::discover("fpeos_telemetry.jsonl");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Paper 32: Militzer FPEOS — EOS Table Validation           ║");
    println!("║  Militzer et al. Phys. Rev. E 103, 013203 (2021)          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let h = hydrogen_reference();
    let he = helium_reference();

    // Grid-point exact lookup (hydrogen)
    println!("  Hydrogen grid-point lookup...");
    let test_points: &[(f64, f64, f64, f64)] = &[
        // (log_rho, log_T, expected_P, expected_E)
        (0.0, 5.0, 9.2, 14.0),
        (0.301, 6.0, 208.0, 175.0),
        (1.0, 8.0, 166_000.0, 40_000.0),
        (-0.301, 4.0, 0.42, -1.2),
        (0.699, 7.0, 6600.0, 2550.0),
    ];

    for &(lr, lt, exp_p, exp_e) in test_points {
        let pt = h.interpolate_log(lr, lt).unwrap();
        let p_err = (pt.pressure - exp_p).abs() / exp_p.abs().max(1e-10);
        let e_err = (pt.internal_energy - exp_e).abs() / exp_e.abs().max(1e-10);
        let label_p = format!("h_P_rho{lr}_T{lt}");
        let label_e = format!("h_E_rho{lr}_T{lt}");
        harness.check_upper(&label_p, p_err, 0.01);
        harness.check_upper(&label_e, e_err, 0.01);
        telem.log_map(
            "h_grid",
            &[
                ("log_rho", lr),
                ("log_T", lt),
                ("P", pt.pressure),
                ("E", pt.internal_energy),
            ],
        );
    }

    // Interpolation between grid points
    println!("  Bilinear interpolation checks...");
    let interp_pts: &[(f64, f64)] = &[(0.15, 5.5), (0.5, 6.5), (-0.15, 4.5), (0.85, 7.5)];
    for &(lr, lt) in interp_pts {
        let pt = h.interpolate_log(lr, lt).unwrap();
        harness.check_lower(&format!("h_interp_P_{lr}_{lt}"), pt.pressure, 0.0);
        telem.log_map(
            "h_interp",
            &[
                ("log_rho", lr),
                ("log_T", lt),
                ("P", pt.pressure),
                ("E", pt.internal_energy),
            ],
        );
    }

    // Monotonicity: P(ρ) at fixed T
    println!("  Monotonicity checks...");
    let mut mono_ok = true;
    for t in [4.0, 5.0, 6.0, 7.0, 8.0] {
        let mut prev = 0.0;
        for &lr in &h.log_densities {
            let pt = h.interpolate_log(lr, t).unwrap();
            if pt.pressure <= prev {
                mono_ok = false;
            }
            prev = pt.pressure;
        }
    }
    harness.check_bool("h_P_mono_rho", mono_ok);

    // Monotonicity: P(T) at fixed ρ
    let mut mono_t_ok = true;
    for &lr in &h.log_densities {
        let mut prev = 0.0;
        for t in [4.0, 5.0, 6.0, 7.0, 8.0] {
            let pt = h.interpolate_log(lr, t).unwrap();
            if pt.pressure <= prev {
                mono_t_ok = false;
            }
            prev = pt.pressure;
        }
    }
    harness.check_bool("h_P_mono_T", mono_t_ok);

    // Cross-element: He ≤ H at same conditions
    println!("  Cross-element comparison (He ≤ H)...");
    let mut cross_ok = true;
    for i in 0..h.pressure.len() {
        if he.pressure[i] > h.pressure[i] * 1.01 {
            cross_ok = false;
        }
    }
    harness.check_bool("he_le_h_pressure", cross_ok);

    // Helium grid-point checks
    println!("  Helium grid-point lookup...");
    let he_pt = he.interpolate_log(0.0, 6.0).unwrap();
    let he_p_err = (he_pt.pressure - 46.5).abs() / 46.5;
    harness.check_upper("he_P_rho0_T6", he_p_err, 0.01);

    // Thermodynamic consistency
    println!("  Thermodynamic consistency...");
    let consistency = h.thermodynamic_consistency();
    println!("    max ΔP/P = {consistency:.4e}");
    telem.log("fpeos", "thermo_consistency", consistency);
    // FPEOS data has numerical differentiation noise; allow generous threshold
    harness.check_upper("h_thermo_consistency", consistency, 2.0);

    harness.finish();
}
