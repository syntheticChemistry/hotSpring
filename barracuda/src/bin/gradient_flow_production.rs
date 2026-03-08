// SPDX-License-Identifier: AGPL-3.0-only

//! Production gradient flow scale-setting — Paper 43 extension.
//!
//! Thermalizes SU(3) pure gauge lattices via HMC, then measures gradient flow
//! observables (t₀, w₀) using the Lüscher (RK3) and Chuna (LSCFRK3W7) integrators.
//!
//! Reference: Bazavov & Chuna, arXiv:2101.05320 (2021)
//!
//! Data source: Self-generated via HMC thermalization (no external configs needed).
//! All gauge configurations are produced from hot starts with documented seeds.
//!
//! Usage:
//!   cargo run --release --bin gradient_flow_production

use hotspring_barracuda::lattice::gradient_flow::{
    find_t0, find_w0, run_flow, FlowIntegrator, FlowMeasurement,
};
use hotspring_barracuda::lattice::hmc::{hmc_trajectory, HmcConfig, HmcResult};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn thermalize(lattice: &mut Lattice, n_therm: usize) -> Vec<HmcResult> {
    let mut config = HmcConfig {
        n_md_steps: 20,
        dt: 0.05,
        seed: 12345,
        ..Default::default()
    };

    let mut results = Vec::with_capacity(n_therm);
    for _ in 0..n_therm {
        let r = hmc_trajectory(lattice, &mut config);
        results.push(r);
    }
    results
}

fn measure_flow(
    lattice: &mut Lattice,
    integrator: FlowIntegrator,
    epsilon: f64,
    t_max: f64,
) -> Vec<FlowMeasurement> {
    let measure_interval = (0.05 / epsilon).max(1.0) as usize;
    run_flow(lattice, integrator, epsilon, t_max, measure_interval)
}

fn main() {
    let mut harness = ValidationHarness::new("gradient_flow_production");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Production Gradient Flow Scale Setting — Paper 43         ║");
    println!("║  Bazavov & Chuna, arXiv:2101.05320 (2021)                  ║");
    println!("║  Data: Self-generated pure gauge SU(3) via HMC             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let total_start = Instant::now();

    // Production parameters — 8⁴ and 16⁴ lattices
    let configs = [
        ([8, 8, 8, 8], 5.9, 200, "8⁴ β=5.9 (confined)"),
        ([8, 8, 8, 8], 6.0, 200, "8⁴ β=6.0 (standard)"),
        ([8, 8, 8, 8], 6.2, 200, "8⁴ β=6.2 (weak)"),
        ([16, 16, 16, 16], 5.9, 500, "16⁴ β=5.9 (confined)"),
        ([16, 16, 16, 16], 6.0, 500, "16⁴ β=6.0 (standard)"),
        ([16, 16, 16, 16], 6.2, 500, "16⁴ β=6.2 (weak)"),
    ];

    for (dims, beta, n_therm, label) in &configs {
        println!("\n═══ {label} ═══");
        let start = Instant::now();

        // Thermalize from hot start
        let mut lattice = Lattice::hot_start(*dims, *beta, 42);
        let plaq_init = lattice.average_plaquette();
        println!("  Initial ⟨P⟩ = {plaq_init:.6}");

        let hmc_results = thermalize(&mut lattice, *n_therm);
        let plaq_therm = lattice.average_plaquette();
        let n_accept = hmc_results.iter().filter(|r| r.accepted).count();
        let acceptance = n_accept as f64 / hmc_results.len() as f64;
        println!(
            "  Thermalized: ⟨P⟩ = {plaq_therm:.6} ({n_therm} HMC, {:.0}% accept)",
            acceptance * 100.0
        );

        harness.check_lower(&format!("acceptance_{label}"), acceptance, 0.3);

        // Measure gradient flow with RK3 Lüscher integrator
        let flow_eps = 0.01;
        let flow_t_max = 4.0;

        let results_rk3 = measure_flow(
            &mut lattice,
            FlowIntegrator::Rk3Luscher,
            flow_eps,
            flow_t_max,
        );

        // Re-thermalize for second integrator comparison (fresh config)
        let mut lattice2 = Lattice::hot_start(*dims, *beta, 43);
        let _ = thermalize(&mut lattice2, *n_therm);
        let results_w7 = measure_flow(
            &mut lattice2,
            FlowIntegrator::Lscfrk3w7,
            flow_eps,
            flow_t_max,
        );

        // Extract t₀ and w₀
        let t0_rk3 = find_t0(&results_rk3);
        let t0_w7 = find_t0(&results_w7);
        let w0_rk3 = find_w0(&results_rk3);
        let w0_w7 = find_w0(&results_w7);

        println!("  ── Scale Setting ──");
        if let Some(t0) = t0_rk3 {
            println!("    t₀ (RK3):   {t0:.4}");
        } else {
            println!("    t₀ (RK3):   not found (extend t_max)");
        }
        if let Some(t0) = t0_w7 {
            println!("    t₀ (W7):    {t0:.4}");
        }
        if let Some(w0) = w0_rk3 {
            println!("    w₀ (RK3):   {w0:.4}");
        }
        if let Some(w0) = w0_w7 {
            println!("    w₀ (W7):    {w0:.4}");
        }

        // Compare integrators: t₀ from both should agree
        if let (Some(t0_a), Some(t0_b)) = (t0_rk3, t0_w7) {
            let rel = (t0_a - t0_b).abs() / t0_a.max(1e-30);
            harness.check_upper(&format!("t0_integrator_agreement_{label}"), rel, 0.05);
            println!("    |t₀(RK3) - t₀(W7)|/t₀ = {rel:.4e}");
        }

        // Energy density should be monotonically decreasing under flow
        let monotonic = results_rk3
            .windows(2)
            .all(|w| w[1].energy_density <= w[0].energy_density + 1e-10);
        harness.check_bool(&format!("flow_monotonic_{label}"), monotonic);

        // Unitarity after flow
        let u = lattice.link([0, 0, 0, 0], 0);
        let uu_dag = u * u.adjoint();
        let id = hotspring_barracuda::lattice::su3::Su3Matrix::IDENTITY;
        let dev = (uu_dag - id).norm_sq().sqrt();
        harness.check_upper(&format!("unitarity_{label}"), dev, 1e-10);

        let elapsed = start.elapsed();
        println!("  Wall time: {:.1}s", elapsed.as_secs_f64());
    }

    let total = total_start.elapsed();
    println!("\n  Total wall time: {:.1}s", total.as_secs_f64());

    harness.finish();
}
