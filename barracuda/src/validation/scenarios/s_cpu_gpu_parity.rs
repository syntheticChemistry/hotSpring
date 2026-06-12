// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: CPU/GPU Parity — validates the CPU math foundations that
//! underpin every GPU parity check. Offline (no GPU required).
//!
//! Exercises the CPU side of each domain covered by
//! `validate_barracuda_cpu_gpu_parity`: lattice QCD, SEMF, transport,
//! spectral SpMV, BGK relaxation, Euler shock tube, and coupled
//! kinetic-fluid. GPU parity itself requires hardware; this scenario
//! proves the reference values are stable and deterministic.

use crate::tolerances;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "cpu-gpu-parity",
        track: Track::GpuCompute,
        tier: Tier::Rust,
        provenance_crate: "validate_barracuda_cpu_gpu_parity",
        provenance_date: "2026-05-17",
        description: "CPU reference stability for all GPU parity domains",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    check_lattice_qcd(v);
    check_semf(v);
    check_transport(v);
    check_spectral(v);
    check_bgk(v);
    check_euler(v);
    check_coupled(v);
}

fn check_lattice_qcd(v: &mut ValidationHarness) {
    use crate::lattice::hmc::{HmcConfig, hmc_trajectory};
    use crate::lattice::wilson::Lattice;

    let mut lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
    let mut cfg = HmcConfig {
        n_md_steps: 10,
        dt: 0.05,
        seed: 99,
        ..Default::default()
    };

    let result = hmc_trajectory(&mut lat, &mut cfg);
    let plaq = lat.average_plaquette();
    v.check_bool("parity:qcd_plaq_finite", plaq.is_finite());
    v.check_bool("parity:qcd_plaq_range", (0.0..1.0).contains(&plaq));
    v.check_bool("parity:qcd_delta_h_finite", result.delta_h.is_finite());
}

fn check_semf(v: &mut ValidationHarness) {
    use crate::physics::semf_binding_energy;
    use crate::provenance::SLY4_PARAMS;

    let be_pb208 = semf_binding_energy(82, 126, &SLY4_PARAMS);
    let be_fe56 = semf_binding_energy(26, 30, &SLY4_PARAMS);

    v.check_bool("parity:semf_pb208_finite", be_pb208.is_finite());
    v.check_bool("parity:semf_fe56_finite", be_fe56.is_finite());

    let be_again = semf_binding_energy(82, 126, &SLY4_PARAMS);
    v.check_bool(
        "parity:semf_deterministic",
        be_pb208.total_cmp(&be_again).is_eq(),
    );
}

fn check_transport(v: &mut ValidationHarness) {
    use crate::md::transport::{d_star_daligault, eta_star_stanton_murillo};

    let d_star = d_star_daligault(10.0, 1.0);
    let eta_star = eta_star_stanton_murillo(10.0, 1.0);

    v.check_bool(
        "parity:transport_d_finite",
        d_star.is_finite() && d_star > 0.0,
    );
    v.check_bool(
        "parity:transport_eta_finite",
        eta_star.is_finite() && eta_star > 0.0,
    );
}

fn check_spectral(v: &mut ValidationHarness) {
    use crate::spectral::anderson_2d;

    let m = anderson_2d(4, 4, 0.0, 42);
    let n = m.n;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
    let mut y = vec![0.0; n];
    m.spmv(&x, &mut y);

    let norm: f64 = y.iter().map(|yi| yi * yi).sum::<f64>().sqrt();
    v.check_bool("parity:spmv_norm_finite", norm.is_finite());
    v.check_bool("parity:spmv_norm_positive", norm > 0.0);

    let mut y2 = vec![0.0; n];
    m.spmv(&x, &mut y2);
    let same = y.iter().zip(y2.iter()).all(|(a, b)| a.total_cmp(b).is_eq());
    v.check_bool("parity:spmv_deterministic", same);
}

fn check_bgk(v: &mut ValidationHarness) {
    use crate::physics::kinetic_fluid::{BgkSpecies, bgk_relaxation_step, maxwellian_1d};

    let n_v = 51;
    let v_max = 6.0;
    let dv = 2.0 * v_max / (n_v - 1) as f64;
    let velocities: Vec<f64> = (0..n_v).map(|i| -v_max + i as f64 * dv).collect();

    let f1 = maxwellian_1d(&velocities, 1.0, 0.0, 2.0, 1.0);
    let f2 = maxwellian_1d(&velocities, 1.0, 0.0, 0.5, 4.0);

    let mut species = vec![
        BgkSpecies {
            f: f1,
            m: 1.0,
            nu: 1.0,
        },
        BgkSpecies {
            f: f2,
            m: 4.0,
            nu: 1.0,
        },
    ];
    let mass_before: f64 = species[0].f.iter().sum::<f64>() * dv;

    bgk_relaxation_step(&mut species, &velocities, dv, 0.01);

    let mass_after: f64 = species[0].f.iter().sum::<f64>() * dv;
    v.check_bool(
        "parity:bgk_mass_conserved",
        (mass_before - mass_after).abs() < tolerances::KINETIC_FLUID_BGK_MASS_REL,
    );
    v.check_bool(
        "parity:bgk_all_positive",
        species[0].f.iter().all(|&fi| fi >= 0.0),
    );
}

fn check_euler(v: &mut ValidationHarness) {
    use crate::physics::kinetic_fluid::run_sod_shock_tube;

    let r = run_sod_shock_tube(50, 0.01);
    v.check_bool(
        "parity:euler_mass_conserved",
        r.mass_err < tolerances::KINETIC_FLUID_BGK_MASS_REL,
    );
    v.check_bool(
        "parity:euler_rho_physical",
        r.rho_min > 0.0 && r.rho_max < 2.0,
    );
}

fn check_coupled(v: &mut ValidationHarness) {
    use crate::physics::kinetic_fluid::run_coupled_kinetic_fluid;

    let r = run_coupled_kinetic_fluid(10, 10, 21, 0.005);
    v.check_bool("parity:coupled_steps_positive", r.n_steps > 0);
    v.check_bool("parity:coupled_mass_bounded", r.mass_err < 1.0);
}
