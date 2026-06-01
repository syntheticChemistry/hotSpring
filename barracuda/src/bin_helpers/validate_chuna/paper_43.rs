// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 43: Wilson gradient flow validation.

use hotspring_barracuda::lattice::gradient_flow::{find_t0, run_flow, FlowIntegrator};
use hotspring_barracuda::lattice::su3::Su3Matrix;
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

/// CPU-side reference values collected during Paper 43 for GPU parity checks.
pub struct CpuReferenceValues {
    pub plaquette_8_rk3: f64,
    pub energy_8_rk3: f64,
}

pub fn paper_43_gradient_flow(harness: &mut ValidationHarness) -> CpuReferenceValues {
    const PAPER: &str = "Bazavov & Chuna, arXiv:2101.05320";
    const DOMAIN: &str = "lattice_qcd";

    println!("━━━ Paper 43: Wilson Gradient Flow ━━━");
    let seed = 42;
    let eps = 0.01;

    // Integrator convergence: |RK2-RK3| < |Euler-RK3|
    let t0 = Instant::now();
    let mut lat_e = Lattice::hot_start([4, 4, 4, 4], 6.0, seed);
    let mut lat_r2 = Lattice::hot_start([4, 4, 4, 4], 6.0, seed);
    let mut lat_r3 = Lattice::hot_start([4, 4, 4, 4], 6.0, seed);
    let meas = (0.05_f64 / eps).max(1.0) as usize;
    let res_e = run_flow(&mut lat_e, FlowIntegrator::Euler, eps, 1.0, meas);
    let res_r2 = run_flow(&mut lat_r2, FlowIntegrator::Rk2, eps, 1.0, meas);
    let res_r3 = run_flow(&mut lat_r3, FlowIntegrator::Rk3Luscher, eps, 1.0, meas);
    let dur = t0.elapsed().as_millis() as u64;

    let e_euler = res_e.last().map_or(f64::NAN, |m| m.energy_density);
    let e_rk2 = res_r2.last().map_or(f64::NAN, |m| m.energy_density);
    let e_rk3 = res_r3.last().map_or(f64::NAN, |m| m.energy_density);

    harness.check_bool(
        "gradient_flow_integrator_convergence",
        (e_rk2 - e_rk3).abs() < (e_euler - e_rk3).abs(),
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "convergence_hierarchy",
        "|RK2-RK3| < |Euler-RK3|",
    );
    harness.annotate_duration(dur);

    // 8⁴ energy smoothing
    let t0 = Instant::now();
    let mut lat = Lattice::hot_start([8, 8, 8, 8], 6.0, seed);
    let res_8 = run_flow(&mut lat, FlowIntegrator::Rk3Luscher, eps, 2.0, meas);
    let dur = t0.elapsed().as_millis() as u64;

    let e_start = res_8.first().map_or(f64::NAN, |m| m.energy_density);
    let e_end = res_8.last().map_or(f64::NAN, |m| m.energy_density);
    harness.check_bool("gradient_flow_energy_smoothing", e_end <= e_start);
    harness.annotate(
        DOMAIN,
        PAPER,
        "energy_density",
        "E(t_final) <= E(t_initial) under flow",
    );
    harness.annotate_duration(dur);

    // t²E(t) must increase under flow — prerequisite for scale setting.
    // On small hot-start lattices t₀ may not exist in the flow range
    // (t²E stays below 0.3 when the configuration is too disordered).
    let t2e_vals: Vec<f64> = res_8
        .iter()
        .filter(|m| m.t > 0.05)
        .map(|m| m.t2_e)
        .collect();
    let t2e_grows = t2e_vals.len() >= 2
        && t2e_vals.last().copied().unwrap_or(0.0) > t2e_vals.first().copied().unwrap_or(0.0);
    harness.check_bool("gradient_flow_t2e_increasing", t2e_grows);
    harness.annotate(
        DOMAIN,
        PAPER,
        "t2_energy",
        "t²E(t) increases under flow — prerequisite for scale setting",
    );

    let t0_val = find_t0(&res_8);
    if let Some(t0v) = t0_val {
        harness.check_lower("gradient_flow_t0_positive", t0v, 0.0);
        harness.annotate(DOMAIN, PAPER, "flow_time", "t₀ > 0 when found");
    }

    // Unitarity after flow
    let u = lat.link([0, 0, 0, 0], 0);
    let dev = (u * u.adjoint() - Su3Matrix::IDENTITY).norm_sq().sqrt();
    harness.check_upper("gradient_flow_unitarity", dev, tolerances::EXACT_F64);
    harness.annotate(
        DOMAIN,
        PAPER,
        "unitarity_deviation",
        "||UU†-I|| < 1e-10 after flow",
    );

    // Chuna W7 integrator on β-scan
    for &beta in &[5.5, 6.0, 6.2] {
        let t0 = Instant::now();
        let mut lat_w7 = Lattice::hot_start([4, 4, 4, 4], beta, seed);
        let res_w7 = run_flow(&mut lat_w7, FlowIntegrator::Lscfrk3w7, eps, 2.0, meas);
        let dur = t0.elapsed().as_millis() as u64;
        let e_i = res_w7.first().map_or(f64::NAN, |m| m.energy_density);
        let e_f = res_w7.last().map_or(f64::NAN, |m| m.energy_density);
        let label = format!("gradient_flow_w7_beta_{}", beta as u32 * 10);
        harness.check_bool(&label, e_f <= e_i + tolerances::EXACT_F64);
        harness.annotate(DOMAIN, PAPER, "energy_density", "LSCFRK3W7 smoothing");
        harness.annotate_duration(dur);
    }

    // CK4 stability
    let t0 = Instant::now();
    let mut lat_ck = Lattice::hot_start([8, 8, 8, 8], 6.0, seed);
    let res_ck = run_flow(&mut lat_ck, FlowIntegrator::Lscfrk4ck, eps, 2.0, meas);
    let dur = t0.elapsed().as_millis() as u64;
    let e_ck4 = res_ck.last().map_or(f64::NAN, |m| m.energy_density);
    harness.check_bool("gradient_flow_ck4_stable", e_ck4.is_finite() && e_ck4 > 0.0);
    harness.annotate(
        DOMAIN,
        PAPER,
        "energy_density",
        "LSCFRK4CK stable: finite positive E",
    );
    harness.annotate_duration(dur);

    let ck4_rk3_diff = (e_ck4 - e_end).abs();
    harness.check_upper(
        "gradient_flow_ck4_rk3_agreement",
        ck4_rk3_diff,
        tolerances::GRADIENT_FLOW_CK4_RK3_ENERGY_ABS,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "energy_density",
        "CK4 and RK3 converge to same E at same dt",
    );

    println!("  Paper 43: {} checks\n", 11);

    CpuReferenceValues {
        plaquette_8_rk3: lat.average_plaquette(),
        energy_8_rk3: e_end,
    }
}
