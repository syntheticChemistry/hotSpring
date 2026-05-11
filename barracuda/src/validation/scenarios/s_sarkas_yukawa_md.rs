// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Sarkas Yukawa MD — foundation-grade validation.
//!
//! Validates Sarkas Yukawa OCP transport physics and (when `barracuda-local`
//! is enabled) runs a CPU MD simulation with energy drift checks.
//!
//! This is the scenario referenced by projectNUCLEUS workloads and
//! foundation Thread 2 (Plasma Physics) validation targets.

use crate::md::config::quick_test_case;
use crate::md::transport::{d_star_daligault, sarkas_d_star_reference};
use crate::tolerances::md::{DALIGAULT_FIT_RMSE, TRANSPORT_D_STAR_VS_SARKAS};
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "sarkas-yukawa-md",
        track: Track::MolecularDynamics,
        tier: Tier::Rust,
        provenance_crate: "validate_md",
        provenance_date: "2026-05-11",
        description: "Sarkas Yukawa MD: Daligault D* fit + reference transport parity + optional CPU simulation",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    let config = quick_test_case(108);

    v.check_bool("sarkas:config_valid", config.n_particles == 108);
    v.check_bool("sarkas:kappa_positive", config.kappa > 0.0);
    v.check_bool("sarkas:gamma_positive", config.gamma > 0.0);
    v.check_bool("sarkas:dt_positive", config.dt > 0.0);
    v.check_bool("sarkas:box_side_positive", config.box_side() > 0.0);
    v.check_bool("sarkas:temperature_positive", config.temperature() > 0.0);

    let refs = sarkas_d_star_reference();
    let mut rmse_sum = 0.0;
    let mut rmse_count = 0;

    for (kappa, gamma, d_star_ref) in &refs {
        let d_star_fit = d_star_daligault(*gamma, *kappa);
        let rel_err = if d_star_ref.abs() > 1e-12 {
            ((d_star_fit - d_star_ref) / d_star_ref).abs()
        } else {
            0.0
        };
        rmse_sum += rel_err * rel_err;
        rmse_count += 1;

        let label = format!("sarkas:d_star_fit_k{kappa}_g{gamma}");
        v.check_upper(&label, rel_err, TRANSPORT_D_STAR_VS_SARKAS);
    }

    if rmse_count > 0 {
        let rmse = (rmse_sum / rmse_count as f64).sqrt();
        v.check_upper("sarkas:daligault_fit_rmse", rmse, DALIGAULT_FIT_RMSE);
    }

    v.check_bool("sarkas:reference_points_available", rmse_count == 12);

    #[cfg(feature = "barracuda-local")]
    run_cpu_simulation(v, &config);
}

#[cfg(feature = "barracuda-local")]
fn run_cpu_simulation(v: &mut ValidationHarness, config: &crate::md::config::MdConfig) {
    use crate::md::cpu_reference::run_simulation_cpu;
    use crate::md::observables::energy::validate_energy;
    use crate::tolerances::md::ENERGY_DRIFT_PCT;

    let sim = run_simulation_cpu(config);

    let energy_val = validate_energy(&sim.energy_history, config);
    v.check_upper(
        "sarkas:energy_drift_pct",
        energy_val.drift_pct,
        ENERGY_DRIFT_PCT,
    );
    v.check_bool("sarkas:energy_validation_pass", energy_val.passed);
    v.check_lower(
        "sarkas:mean_temperature_positive",
        energy_val.mean_temperature,
        0.0,
    );
    v.check_bool(
        "sarkas:simulation_completed",
        !sim.energy_history.is_empty(),
    );
    v.check_lower(
        "sarkas:steps_completed",
        sim.energy_history.len() as f64,
        10.0,
    );
}
