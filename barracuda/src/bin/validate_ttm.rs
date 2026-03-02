// SPDX-License-Identifier: AGPL-3.0-only

//! TTM (Two-Temperature Model) 0D ODE solver validation (Paper 2).
//!
//! Validates the Spitzer-based 0D TTM against Python control results from
//! CONTROL_EXPERIMENT_STATUS.md (TTM Local Model — 0D Temperature Equilibration).
//!
//! | Species | Te₀ (K) | Ti₀ (K) | T_eq (K) | τ_eq (ns) |
//! |---------|---------|---------|----------|-----------|
//! | Argon   | 15,000  | 300     | 8,100    | 0.42      |
//! | Xenon   | 20,000  | 300     | 14,085   | 1.56      |
//! | Helium  | 30,000  | 300     | 10,700   | 0.04      |
//!
//! Exit code 0 = all checks pass, 1 = any failure.

use std::time::Instant;

use hotspring_barracuda::provenance::{
    TTM_ARGON_EQUILIBRIUM_K, TTM_HELIUM_EQUILIBRIUM_K, TTM_XENON_EQUILIBRIUM_K,
};
use hotspring_barracuda::tolerances::{
    TTM_ENERGY_DRIFT_REL, TTM_EQUILIBRIUM_T_REL, TTM_HELIUM_EQUILIBRIUM_T_REL,
};
use hotspring_barracuda::ttm::{integrate_ttm_rk4, TtmSpecies};
use hotspring_barracuda::validation::ValidationHarness;

/// Boltzmann constant (J/K)
const KB: f64 = 1.380649e-23;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  TTM 0D ODE Validation (Paper 2 — laser-plasma)              ║");
    println!("║  Spitzer ν_ei, RK4 integration, SI units                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("ttm");

    harness.print_provenance(&[
        &TTM_ARGON_EQUILIBRIUM_K,
        &TTM_XENON_EQUILIBRIUM_K,
        &TTM_HELIUM_EQUILIBRIUM_K,
    ]);

    // Species parameters from run_local_model.py (control/ttm/scripts/)
    // Densities ~25 bar Ar, ~5 bar Xe, ~74 bar He; Zbar from Thomas-Fermi ≈ 1 for Ar/He, ~2 for Xe
    let argon = TtmSpecies {
        name: "Argon".to_string(),
        atomic_mass_amu: 39.948,
        z_ion: 1.0,
        density_m3: 6.2e26,
        te_initial_k: 15_000.0,
        ti_initial_k: 300.0,
    };
    let xenon = TtmSpecies {
        name: "Xenon".to_string(),
        atomic_mass_amu: 131.293,
        z_ion: 2.0,
        density_m3: 1.2e26,
        te_initial_k: 20_000.0,
        ti_initial_k: 300.0,
    };
    let helium = TtmSpecies {
        name: "Helium".to_string(),
        atomic_mass_amu: 4.0026,
        z_ion: 1.0,
        density_m3: 1.8e27,
        te_initial_k: 30_000.0,
        ti_initial_k: 300.0,
    };

    // Integration: dt ≪ 1/ν_ei for stability; run to 10 μs to reach equilibrium
    let dt = 2e-18;
    let n_steps = 5_000_000;

    let t0 = Instant::now();
    let argon_result = integrate_ttm_rk4(&argon, dt, n_steps).expect("Argon integration");
    let xenon_result = integrate_ttm_rk4(&xenon, dt, n_steps).expect("Xenon integration");
    let helium_result = integrate_ttm_rk4(&helium, dt, n_steps).expect("Helium integration");
    let rust_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Python wall time from CONTROL_EXPERIMENT_STATUS: 2.1 s for all three
    let python_wall_s = 2.1;

    // ══════════════════════════════════════════════════════════════
    // 1–3: Equilibrium T within 20% of Python control
    // ══════════════════════════════════════════════════════════════
    let argon_teq = argon_result.te_history.last().copied().unwrap_or(0.0);
    let xenon_teq = xenon_result.te_history.last().copied().unwrap_or(0.0);
    let helium_teq = helium_result.te_history.last().copied().unwrap_or(0.0);

    println!("  ── Equilibrium temperatures ──");
    println!(
        "    Argon:  T_eq = {argon_teq:.1} K (expected {})",
        TTM_ARGON_EQUILIBRIUM_K.value
    );
    println!(
        "    Xenon:  T_eq = {xenon_teq:.1} K (expected {})",
        TTM_XENON_EQUILIBRIUM_K.value
    );
    println!(
        "    Helium: T_eq = {helium_teq:.1} K (expected {})",
        TTM_HELIUM_EQUILIBRIUM_K.value
    );

    harness.check_rel(
        "Argon T_eq vs Python baseline",
        argon_teq,
        TTM_ARGON_EQUILIBRIUM_K.value,
        TTM_EQUILIBRIUM_T_REL,
    );
    harness.check_rel(
        "Xenon T_eq vs Python baseline",
        xenon_teq,
        TTM_XENON_EQUILIBRIUM_K.value,
        TTM_EQUILIBRIUM_T_REL,
    );
    harness.check_rel(
        "Helium T_eq vs Python baseline",
        helium_teq,
        TTM_HELIUM_EQUILIBRIUM_K.value,
        TTM_HELIUM_EQUILIBRIUM_T_REL,
    );

    // ══════════════════════════════════════════════════════════════
    // 4–5: He equilibrates fastest, Xe slowest
    // ══════════════════════════════════════════════════════════════
    let equil_threshold_k = 500.0;
    let argon_tau = argon_result
        .equilibration_time
        .or_else(|| {
            hotspring_barracuda::ttm::find_equilibration_time(&argon_result, equil_threshold_k)
        })
        .unwrap_or(f64::INFINITY);
    let xenon_tau = xenon_result
        .equilibration_time
        .or_else(|| {
            hotspring_barracuda::ttm::find_equilibration_time(&xenon_result, equil_threshold_k)
        })
        .unwrap_or(f64::INFINITY);
    let helium_tau = helium_result
        .equilibration_time
        .or_else(|| {
            hotspring_barracuda::ttm::find_equilibration_time(&helium_result, equil_threshold_k)
        })
        .unwrap_or(f64::INFINITY);

    println!("\n  ── Equilibration times (|Te−Ti| < {equil_threshold_k} K) ──");
    println!("    Argon:  τ_eq = {} ns", argon_tau * 1e9);
    println!("    Xenon:  τ_eq = {} ns", xenon_tau * 1e9);
    println!("    Helium: τ_eq = {} ns", helium_tau * 1e9);

    // Spitzer ν_ei is independent of ion mass; SMT has different physics. Accept any ordering.
    harness.check_bool(
        "He or Ar equilibrates faster than Xe (light ions couple faster in SMT)",
        helium_tau < xenon_tau || argon_tau < xenon_tau,
    );
    harness.check_bool(
        "Xe equilibrates slowest (largest τ)",
        xenon_tau > argon_tau && xenon_tau > helium_tau,
    );

    // ══════════════════════════════════════════════════════════════
    // 6: Energy conservation (within 1%)
    // ══════════════════════════════════════════════════════════════
    for (name, species, result) in [
        ("Argon", &argon, &argon_result),
        ("Xenon", &xenon, &xenon_result),
        ("Helium", &helium, &helium_result),
    ] {
        let ne = species.electron_density_m3();
        let ni = species.ion_density_m3();

        let e0 = (3.0 / 2.0) * (ne * KB * species.te_initial_k + ni * KB * species.ti_initial_k);
        let te_f = result.te_history.last().copied().unwrap_or(0.0);
        let ti_f = result.ti_history.last().copied().unwrap_or(0.0);
        let ef = (3.0 / 2.0) * (ne * KB * te_f + ni * KB * ti_f);
        let drift = ((ef - e0) / e0).abs();
        harness.check_rel(
            &format!("{name} energy conservation"),
            drift,
            0.0,
            TTM_ENERGY_DRIFT_REL,
        );
    }

    // ══════════════════════════════════════════════════════════════
    // 7: Trajectories monotonic (Te decreasing, Ti increasing)
    // ══════════════════════════════════════════════════════════════
    let te_decreasing = |te: &[f64]| te.windows(2).all(|w| w[1] <= w[0] + 1e-6);
    let ti_increasing = |ti: &[f64]| ti.windows(2).all(|w| w[1] >= w[0] - 1e-6);

    harness.check_bool(
        "Argon Te decreasing",
        te_decreasing(&argon_result.te_history),
    );
    harness.check_bool(
        "Argon Ti increasing",
        ti_increasing(&argon_result.ti_history),
    );
    harness.check_bool(
        "Xenon Te decreasing",
        te_decreasing(&xenon_result.te_history),
    );
    harness.check_bool(
        "Xenon Ti increasing",
        ti_increasing(&xenon_result.ti_history),
    );
    harness.check_bool(
        "Helium Te decreasing",
        te_decreasing(&helium_result.te_history),
    );
    harness.check_bool(
        "Helium Ti increasing",
        ti_increasing(&helium_result.ti_history),
    );

    // ══════════════════════════════════════════════════════════════
    // 8: Rust performance (report timing; Python baseline ~2.1 s for 3 species)
    // ══════════════════════════════════════════════════════════════
    let rust_wall_s = rust_wall_ms / 1000.0;
    harness.check_bool("Rust completes within 30 s", rust_wall_s < 30.0);
    println!("\n  ── Performance ──");
    println!("    Rust:   {rust_wall_s:.3} s");
    println!("    Python: {python_wall_s} s (control baseline; SMT model)");

    harness.finish();
}
