// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 45: kinetic-fluid coupling validation.

use hotspring_barracuda::physics::kinetic_fluid::{
    run_bgk_relaxation, run_coupled_kinetic_fluid, run_sod_shock_tube,
};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

pub fn paper_45_kinetic_fluid(harness: &mut ValidationHarness) {
    const PAPER: &str = "Haack et al., J. Comput. Phys. (2024), DOI:10.1016/j.jcp.2024.112908";
    const DOMAIN: &str = "kinetic_fluid";

    println!("━━━ Paper 45: Kinetic-Fluid Coupling ━━━");
    let t0 = Instant::now();

    // BGK relaxation
    let bgk = run_bgk_relaxation(3000, 0.005);

    harness.check_upper(
        "kf_mass_conservation_1",
        bgk.mass_err_1,
        tolerances::KINETIC_FLUID_BGK_MASS_REL,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "species 1 mass conserved to machine precision",
    );
    harness.check_upper(
        "kf_mass_conservation_2",
        bgk.mass_err_2,
        tolerances::KINETIC_FLUID_BGK_MASS_REL,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "species 2 mass conserved to machine precision",
    );
    harness.check_upper(
        "kf_momentum_conservation",
        bgk.momentum_err,
        tolerances::KINETIC_FLUID_BGK_MOMENTUM_REL,
    );
    harness.annotate(DOMAIN, PAPER, "relative_error", "total momentum conserved");
    harness.check_upper(
        "kf_energy_conservation",
        bgk.energy_err,
        tolerances::TTM_ENERGY_DRIFT_REL,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "total energy conserved within 1% (finite-step relaxation)",
    );
    harness.check_bool("kf_entropy_monotonic", bgk.entropy_monotonic);
    harness.annotate(
        DOMAIN,
        PAPER,
        "boolean",
        "H-theorem: entropy monotonically increases",
    );
    harness.check_upper(
        "kf_temperature_relaxation",
        bgk.temp_relaxed,
        tolerances::KINETIC_FLUID_TEMP_RELAXATION_REL,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "T₁ and T₂ converge within 1%",
    );

    let t_eq = f64::midpoint(bgk.t1_final, bgk.t2_final);
    harness.check_abs(
        "kf_equilibrium_temperature",
        t_eq,
        1.25,
        tolerances::KINETIC_FLUID_EQUILIBRIUM_T_ABS,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "temperature",
        "equilibrium T ≈ 1.25 from energy conservation (m₁T₁+m₂T₂)/(m₁+m₂)",
    );

    // Sod shock tube
    let sod = run_sod_shock_tube(400, 0.2);

    harness.check_upper(
        "kf_sod_mass",
        sod.mass_err,
        tolerances::KINETIC_FLUID_SOD_CONSERVATION_REL,
    );
    harness.annotate(DOMAIN, PAPER, "relative_error", "Sod shock: mass conserved");
    harness.check_upper(
        "kf_sod_energy",
        sod.energy_err,
        tolerances::KINETIC_FLUID_SOD_CONSERVATION_REL,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "Sod shock: energy conserved",
    );
    harness.check_bool("kf_sod_contact", sod.contact_in_range);
    harness.annotate(
        DOMAIN,
        PAPER,
        "boolean",
        "contact discontinuity in expected spatial range",
    );
    harness.check_bool("kf_sod_shock", sod.shock_detected);
    harness.annotate(DOMAIN, PAPER, "boolean", "shock front detected");
    harness.check_lower(
        "kf_sod_rho_min",
        sod.rho_min,
        tolerances::KINETIC_FLUID_SOD_RHO_MIN,
    );
    harness.annotate(DOMAIN, PAPER, "density", "density stays physical (>0.1)");
    harness.check_upper(
        "kf_sod_rho_max",
        sod.rho_max,
        tolerances::KINETIC_FLUID_SOD_RHO_MAX,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "density",
        "density bounded (Sod initial max=1.0)",
    );

    // Coupled kinetic-fluid
    let coupled = run_coupled_kinetic_fluid(30, 30, 81, 0.05);

    harness.check_upper(
        "kf_coupled_mass",
        coupled.mass_err,
        tolerances::KINETIC_FLUID_COUPLED_MASS_ENERGY_REL,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "coupled: mass conservation within 15% (interface flux mismatch at low resolution)",
    );
    harness.check_upper(
        "kf_coupled_momentum",
        coupled.momentum_err,
        tolerances::KINETIC_FLUID_COUPLED_MOMENTUM_REL,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "coupled: momentum conservation within 25% (operator splitting at low resolution)",
    );
    harness.check_upper(
        "kf_coupled_energy",
        coupled.energy_err,
        tolerances::KINETIC_FLUID_COUPLED_MASS_ENERGY_REL,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "coupled: energy conservation within 15%",
    );
    harness.check_upper(
        "kf_coupled_interface",
        coupled.interface_density_match,
        tolerances::KINETIC_FLUID_INTERFACE_DENSITY_MATCH,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "interface density match within 50% (coarse grid, operator splitting)",
    );
    harness.check_lower(
        "kf_coupled_rho_min",
        coupled.rho_fluid_min,
        tolerances::KINETIC_FLUID_REGION_RHO_MIN,
    );
    harness.annotate(DOMAIN, PAPER, "density", "fluid density stays physical");
    harness.check_upper(
        "kf_coupled_rho_max",
        coupled.rho_fluid_max,
        tolerances::KINETIC_FLUID_REGION_RHO_MAX,
    );
    harness.annotate(DOMAIN, PAPER, "density", "fluid density bounded");
    harness.check_bool("kf_simulation_completed", coupled.n_steps > 0);
    harness.annotate(DOMAIN, PAPER, "boolean", "simulation ran to completion");

    let dur = t0.elapsed().as_millis() as u64;
    println!("  Paper 45: checks complete ({dur}ms)\n");
}
