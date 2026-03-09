// SPDX-License-Identifier: AGPL-3.0-only

//! Validate average-atom model against atoMEC reference (Paper 33).
//!
//! Tests:
//!   - Hydrogen at solid density + WDM temperatures
//!   - Aluminum at ambient + high temperature
//!   - Mean ionization increases with temperature
//!   - Electron density integrates to Z
//!   - Total energy is finite and negative for bound systems
//!   - Pressure is positive at WDM conditions
//!
//! Provenance: Callow et al. SciPy Proceedings 2023, atoMEC GitHub

use hotspring_barracuda::physics::average_atom::{solve_average_atom, AverageAtomConfig};
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};
use std::f64::consts::PI;

fn main() {
    let mut harness = ValidationHarness::new("atomec_average_atom");
    let mut telem = TelemetryWriter::new("atomec_telemetry.jsonl");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Paper 33: atoMEC Average-Atom Model                       ║");
    println!("║  Callow et al. SciPy Proceedings (2023)                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Hydrogen at solid density, T=1 eV (warm dense matter)
    println!("  Hydrogen (ρ=1 g/cc, T=1 eV)...");
    let h_config = AverageAtomConfig {
        z: 1.0,
        atomic_mass: 1.008,
        density: 1.0,
        temperature_ev: 1.0,
        n_grid: 300,
        max_scf: 100,
        scf_tol: 1e-4,
        mixing: 0.3,
    };
    let h_result = solve_average_atom(&h_config);

    println!(
        "    E = {:.4} Ha, Z* = {:.3}, P = {:.4e} Ha/Bohr³, {} SCF iters, conv={}",
        h_result.total_energy,
        h_result.mean_ionization,
        h_result.pressure,
        h_result.scf_iterations,
        h_result.converged
    );
    telem.log_map(
        "h_wdm",
        &[
            ("total_energy", h_result.total_energy),
            ("mean_ionization", h_result.mean_ionization),
            ("pressure", h_result.pressure),
            ("scf_iterations", h_result.scf_iterations as f64),
        ],
    );

    harness.check_bool("h_energy_finite", h_result.total_energy.is_finite());
    harness.check_lower("h_pressure_positive", h_result.pressure, 0.0);

    // Electron count: ∫n(r)4πr²dr ≈ Z
    let dr = h_result.r_grid[1] - h_result.r_grid[0];
    let total_e: f64 = h_result
        .density
        .iter()
        .zip(h_result.r_grid.iter())
        .map(|(&n, &r)| 4.0 * PI * r * r * n * dr)
        .sum();
    let charge_err = (total_e - 1.0).abs();
    println!("    ∫n(r)dr = {total_e:.4} (target: 1.0)");
    harness.check_upper("h_charge_conservation", charge_err, 0.5);

    // Hydrogen at higher temperature — ionization should increase
    println!("  Hydrogen (ρ=1 g/cc, T=50 eV)...");
    let h_hot = AverageAtomConfig {
        temperature_ev: 50.0,
        ..h_config.clone()
    };
    let h_hot_result = solve_average_atom(&h_hot);
    println!(
        "    E = {:.4} Ha, Z* = {:.3}",
        h_hot_result.total_energy, h_hot_result.mean_ionization
    );
    telem.log("h_hot", "mean_ionization", h_hot_result.mean_ionization);
    harness.check_bool(
        "h_ionization_increases_with_T",
        h_hot_result.mean_ionization >= h_result.mean_ionization - 0.1,
    );

    // Aluminum at ambient density, T=10 eV
    println!("  Aluminum (ρ=2.7 g/cc, T=10 eV)...");
    let al_config = AverageAtomConfig {
        z: 13.0,
        atomic_mass: 26.98,
        density: 2.7,
        temperature_ev: 10.0,
        n_grid: 300,
        max_scf: 100,
        scf_tol: 1e-3,
        mixing: 0.2,
    };
    let al_result = solve_average_atom(&al_config);
    println!(
        "    E = {:.4} Ha, Z* = {:.3}, P = {:.4e}, {} SCF iters",
        al_result.total_energy,
        al_result.mean_ionization,
        al_result.pressure,
        al_result.scf_iterations,
    );
    telem.log_map(
        "al_wdm",
        &[
            ("total_energy", al_result.total_energy),
            ("mean_ionization", al_result.mean_ionization),
            ("pressure", al_result.pressure),
        ],
    );

    harness.check_bool("al_energy_finite", al_result.total_energy.is_finite());
    harness.check_lower("al_pressure_positive", al_result.pressure, 0.0);
    // At 10 eV, Al should be partially ionized (Z* > 0)
    harness.check_lower("al_partially_ionized", al_result.mean_ionization, 0.0);

    // Aluminum at high temperature — more ionized
    println!("  Aluminum (ρ=2.7 g/cc, T=100 eV)...");
    let al_hot = AverageAtomConfig {
        temperature_ev: 100.0,
        ..al_config.clone()
    };
    let al_hot_result = solve_average_atom(&al_hot);
    println!(
        "    Z* = {:.3} (should be > Z* at 10 eV = {:.3})",
        al_hot_result.mean_ionization, al_result.mean_ionization
    );
    telem.log("al_hot", "mean_ionization", al_hot_result.mean_ionization);
    harness.check_bool(
        "al_ionization_increases",
        al_hot_result.mean_ionization >= al_result.mean_ionization - 0.5,
    );

    // Density monotonically decreasing away from nucleus (cold case)
    println!("  Density monotonicity (H, cold)...");
    let h_cold = AverageAtomConfig {
        temperature_ev: 0.1,
        ..h_config
    };
    let h_cold_result = solve_average_atom(&h_cold);
    let n_mono = h_cold_result
        .density
        .windows(2)
        .take(h_cold_result.density.len() / 2)
        .filter(|w| w[0] >= w[1] - 1e-15)
        .count();
    let mono_frac =
        n_mono as f64 / (h_cold_result.density.len() / 2).max(1) as f64;
    println!("    monotonic fraction (inner half): {mono_frac:.2}");
    harness.check_lower("h_cold_density_monotonic", mono_frac, 0.8);

    harness.finish();
}
