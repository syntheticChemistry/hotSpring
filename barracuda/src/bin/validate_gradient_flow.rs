// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validate Wilson gradient flow — Paper 43 (Bazavov & Chuna, 2021).
//!
//! Runs gradient flow on 4⁴ and 8⁴ lattices at several β values,
//! comparing Euler, RK2, and RK3 (Lüscher) integrators. Reports:
//!
//! - E(t) vs flow time t
//! - t²⟨E(t)⟩ for the t₀ scale
//! - Integrator convergence comparison
//! - Unitarity preservation
//!
//! Usage:
//!   cargo run --release --bin `validate_gradient_flow`

use hotspring_barracuda::lattice::gradient_flow::{
    FlowIntegrator, FlowMeasurement, energy_density, find_t0, run_flow,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn print_flow_table(measurements: &[FlowMeasurement]) {
    println!(
        "    {:>8} {:>12} {:>12} {:>12}",
        "t", "E(t)", "t²E(t)", "⟨P⟩"
    );
    println!("    {:>8} {:>12} {:>12} {:>12}", "---", "---", "---", "---");
    for m in measurements {
        println!(
            "    {:>8.4} {:>12.6} {:>12.6} {:>12.8}",
            m.t, m.energy_density, m.t2_e, m.plaquette
        );
    }
}

fn run_single(
    label: &str,
    dims: [usize; 4],
    beta: f64,
    integrator: FlowIntegrator,
    epsilon: f64,
    t_max: f64,
    seed: u64,
) -> Vec<FlowMeasurement> {
    let t0 = Instant::now();
    let mut lattice = Lattice::hot_start(dims, beta, seed);
    let p_init = lattice.average_plaquette();
    let e_init = energy_density(&lattice);

    let integrator_name = match integrator {
        FlowIntegrator::Euler => "Euler",
        FlowIntegrator::Rk2 => "RK2",
        FlowIntegrator::Rk3Luscher => "LSCFRK3W6 (Lüscher)",
        FlowIntegrator::Lscfrk3w7 => "LSCFRK3W7 (Chuna)",
        FlowIntegrator::Lscfrk4ck => "LSCFRK4CK (Carpenter-Kennedy)",
    };

    println!("\n  ── {label}: {integrator_name} ──");
    println!(
        "    Lattice: {}⁴, β={beta}, ε={epsilon}, t_max={t_max}",
        dims[0]
    );
    println!("    Initial: ⟨P⟩={p_init:.6}, E={e_init:.6}");

    let measure_interval = (0.05 / epsilon).max(1.0) as usize;
    let results = run_flow(&mut lattice, integrator, epsilon, t_max, measure_interval);

    let elapsed = t0.elapsed();
    let p_final = lattice.average_plaquette();
    println!(
        "    Final: ⟨P⟩={p_final:.6}, E={:.6}",
        results.last().unwrap().energy_density
    );
    println!("    Wall time: {:.2}s", elapsed.as_secs_f64());

    let u = lattice.link([0, 0, 0, 0], 0);
    let uu_dag = u * u.adjoint();
    let id = hotspring_barracuda::lattice::su3::Su3Matrix::IDENTITY;
    let dev = (uu_dag - id).norm_sq().sqrt();
    println!("    Unitarity deviation: {dev:.2e}");

    if let Some(t0_val) = find_t0(&results) {
        println!("    t₀ = {t0_val:.4}");
    } else {
        println!("    t₀ not found in range [0, {t_max}]");
    }

    print_flow_table(&results);
    results
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Wilson Gradient Flow — Paper 43 Validation                ║");
    println!("║  Bazavov & Chuna, arXiv:2101.05320 (2021)                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let mut harness = ValidationHarness::new("gradient_flow");
    // Provenance: Wilson gradient flow reference from Bazavov & Chuna, arXiv:2101.05320 (2021).
    // No Python baseline — analytical integrator convergence is the validation target.
    // t₀ scale definition: Lüscher, JHEP 1008:071 (2010).
    let total_t0 = Instant::now();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Phase 1: Integrator Comparison on 4⁴ (β=6.0)");
    println!("═══════════════════════════════════════════════════════════");

    let seed = 42;
    let eps = 0.01;
    let t_max = 1.0;

    let results_euler = run_single(
        "4⁴ β=6.0",
        [4, 4, 4, 4],
        6.0,
        FlowIntegrator::Euler,
        eps,
        t_max,
        seed,
    );
    let results_rk2 = run_single(
        "4⁴ β=6.0",
        [4, 4, 4, 4],
        6.0,
        FlowIntegrator::Rk2,
        eps,
        t_max,
        seed,
    );
    let results_rk3 = run_single(
        "4⁴ β=6.0",
        [4, 4, 4, 4],
        6.0,
        FlowIntegrator::Rk3Luscher,
        eps,
        t_max,
        seed,
    );

    println!("\n  ── Integrator Convergence at t=1.0 ──");
    let e_euler = results_euler.last().unwrap().energy_density;
    let e_rk2 = results_rk2.last().unwrap().energy_density;
    let e_rk3 = results_rk3.last().unwrap().energy_density;
    println!("    Euler:       E(1.0) = {e_euler:.8}");
    println!("    RK2:         E(1.0) = {e_rk2:.8}");
    println!("    RK3 Lüscher: E(1.0) = {e_rk3:.8}");
    println!("    |Euler-RK3|: {:.2e}", (e_euler - e_rk3).abs());
    println!("    |RK2-RK3|:   {:.2e}", (e_rk2 - e_rk3).abs());

    // Higher-order integrators must converge closer than Euler
    harness.check_bool(
        "RK2 closer to RK3 than Euler",
        (e_rk2 - e_rk3).abs() < (e_euler - e_rk3).abs(),
    );
    harness.check_bool("E(t) finite (Euler)", e_euler.is_finite());
    harness.check_bool("E(t) finite (RK3)", e_rk3.is_finite());

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Phase 2: β-scan with RK3 on 4⁴");
    println!("═══════════════════════════════════════════════════════════");

    for &beta in &[5.5, 5.7, 5.9, 6.0, 6.2] {
        run_single(
            &format!("4⁴ β={beta}"),
            [4, 4, 4, 4],
            beta,
            FlowIntegrator::Rk3Luscher,
            0.01,
            2.0,
            seed,
        );
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Phase 3: 8⁴ Production with RK3 (β=6.0)");
    println!("═══════════════════════════════════════════════════════════");

    run_single(
        "8⁴ β=6.0",
        [8, 8, 8, 8],
        6.0,
        FlowIntegrator::Rk3Luscher,
        0.01,
        2.0,
        seed,
    );

    let total_elapsed = total_t0.elapsed();
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Total wall time: {:.1}s", total_elapsed.as_secs_f64());
    println!("═══════════════════════════════════════════════════════════");

    harness.finish();
}
