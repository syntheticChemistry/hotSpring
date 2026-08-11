// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation binary for multi-species kinetic-fluid coupling (Paper 45).
//!
//! Haack, Murillo, Sagert & Chuna, J. Comput. Phys. (2024), DOI:10.1016/j.jcp.2024.112908

use hotspring_barracuda::physics::kinetic_fluid::{
    run_bgk_relaxation, run_coupled_kinetic_fluid, run_sod_shock_tube,
};
use hotspring_barracuda::tolerances::{
    KINETIC_FLUID_BGK_MASS_REL, KINETIC_FLUID_BGK_MOMENTUM_REL,
    KINETIC_FLUID_COUPLED_MASS_ENERGY_REL, KINETIC_FLUID_COUPLED_MOMENTUM_REL,
    KINETIC_FLUID_EQUILIBRIUM_T_ABS, KINETIC_FLUID_INTERFACE_DENSITY_MATCH,
    KINETIC_FLUID_REGION_RHO_MAX, KINETIC_FLUID_REGION_RHO_MIN, KINETIC_FLUID_SOD_CONSERVATION_REL,
    KINETIC_FLUID_SOD_RHO_MAX, KINETIC_FLUID_SOD_RHO_MIN, KINETIC_FLUID_TEMP_RELAXATION_REL,
    TTM_ENERGY_DRIFT_REL,
};
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    let mut harness = ValidationHarness::new("validate_kinetic_fluid");

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Paper 45: Kinetic-Fluid Coupling (Haack et al. 2024)  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ── Phase 1: Homogeneous BGK Relaxation ──
    println!("── Phase 1: Multi-Species BGK Relaxation ──");
    let bgk = run_bgk_relaxation(3000, 0.005);

    harness.check_upper(
        "species_1_mass_conservation",
        bgk.mass_err_1,
        KINETIC_FLUID_BGK_MASS_REL,
    );
    harness.check_upper(
        "species_2_mass_conservation",
        bgk.mass_err_2,
        KINETIC_FLUID_BGK_MASS_REL,
    );
    harness.check_upper(
        "total_momentum_conservation",
        bgk.momentum_err,
        KINETIC_FLUID_BGK_MOMENTUM_REL,
    );
    harness.check_upper(
        "total_energy_conservation",
        bgk.energy_err,
        TTM_ENERGY_DRIFT_REL,
    );
    harness.check_bool("entropy_monotonic", bgk.entropy_monotonic);
    harness.check_upper(
        "temperature_relaxation",
        bgk.temp_relaxed,
        KINETIC_FLUID_TEMP_RELAXATION_REL,
    );

    println!(
        "  T₁={:.4}, T₂={:.4}, |ΔT|/T={:.2e}",
        bgk.t1_final, bgk.t2_final, bgk.temp_relaxed
    );
    let t_eq = f64::midpoint(bgk.t1_final, bgk.t2_final);
    harness.check_abs(
        "equilibrium_temperature",
        t_eq,
        1.25,
        KINETIC_FLUID_EQUILIBRIUM_T_ABS,
    );
    println!("  T_eq = {t_eq:.4} (expected ≈ 1.25)");

    println!();

    // ── Phase 2: Sod Shock Tube ──
    println!("── Phase 2: Sod Shock Tube ──");
    let sod = run_sod_shock_tube(400, 0.2);

    harness.check_upper(
        "sod_mass_conservation",
        sod.mass_err,
        KINETIC_FLUID_SOD_CONSERVATION_REL,
    );
    harness.check_upper(
        "sod_energy_conservation",
        sod.energy_err,
        KINETIC_FLUID_SOD_CONSERVATION_REL,
    );
    harness.check_bool("contact_in_range", sod.contact_in_range);
    harness.check_bool("shock_detected", sod.shock_detected);
    harness.check_lower(
        "density_min_physical",
        sod.rho_min,
        KINETIC_FLUID_SOD_RHO_MIN,
    );
    harness.check_upper(
        "density_max_physical",
        sod.rho_max,
        KINETIC_FLUID_SOD_RHO_MAX,
    );

    println!(
        "  contact at x={:.3}, shock at x={:.3}",
        sod.x_contact, sod.x_shock
    );
    println!("  ρ ∈ [{:.4}, {:.4}]", sod.rho_min, sod.rho_max);

    println!();

    // ── Phase 3: Coupled Kinetic-Fluid ──
    println!("── Phase 3: Coupled Kinetic-Fluid ──");
    let coupled = run_coupled_kinetic_fluid(30, 30, 81, 0.05);

    harness.check_upper(
        "coupled_mass_conservation",
        coupled.mass_err,
        KINETIC_FLUID_COUPLED_MASS_ENERGY_REL,
    );
    harness.check_upper(
        "coupled_momentum_conservation",
        coupled.momentum_err,
        KINETIC_FLUID_COUPLED_MOMENTUM_REL,
    );
    harness.check_upper(
        "coupled_energy_conservation",
        coupled.energy_err,
        KINETIC_FLUID_COUPLED_MASS_ENERGY_REL,
    );
    harness.check_upper(
        "interface_density_match",
        coupled.interface_density_match,
        KINETIC_FLUID_INTERFACE_DENSITY_MATCH,
    );
    harness.check_lower(
        "fluid_density_min",
        coupled.rho_fluid_min,
        KINETIC_FLUID_REGION_RHO_MIN,
    );
    harness.check_upper(
        "fluid_density_max",
        coupled.rho_fluid_max,
        KINETIC_FLUID_REGION_RHO_MAX,
    );
    harness.check_bool("simulation_completed", coupled.n_steps > 0);

    println!(
        "  mass err={:.4e}, mom err={:.4e}, energy err={:.4e}",
        coupled.mass_err, coupled.momentum_err, coupled.energy_err
    );
    println!(
        "  interface |Δρ|/ρ₀={:.4e}, {n} steps",
        coupled.interface_density_match,
        n = coupled.n_steps
    );
    println!(
        "  ρ_fluid ∈ [{:.4}, {:.4}]",
        coupled.rho_fluid_min, coupled.rho_fluid_max
    );

    println!();
    harness.finish();
}
