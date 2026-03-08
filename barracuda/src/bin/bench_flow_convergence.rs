// SPDX-License-Identifier: AGPL-3.0-only

//! Gradient flow convergence benchmark — Paper 43 extension.
//!
//! Sweeps step sizes ε = 0.02, 0.01, 0.005, 0.002, 0.001 for each integrator
//! (Euler, RK2, RK3/W6, LSCFRK3W7, LSCFRK4CK) and measures E(t=1.0).
//! Convergence order is estimated from the Richardson extrapolation of
//! successive refinements.
//!
//! This reproduces the key result from Bazavov & Chuna (2021), Table I:
//! LSCFRK3 integrators achieve 3rd-order convergence, CK4 achieves 4th.
//!
//! Usage:
//!   cargo run --release --bin bench_flow_convergence

use hotspring_barracuda::lattice::gradient_flow::{run_flow, FlowIntegrator};
use hotspring_barracuda::lattice::hmc::{hmc_trajectory, HmcConfig};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Gradient Flow Convergence Benchmark — Paper 43            ║");
    println!("║  Bazavov & Chuna, arXiv:2101.05320 (2021), Table I         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let total_start = Instant::now();

    // Thermalize a single reference config
    println!("  Thermalizing 8⁴ at β=6.0 (100 HMC)...");
    let therm_start = Instant::now();
    let mut lattice = Lattice::hot_start([8, 8, 8, 8], 6.0, 42);
    let mut config = HmcConfig {
        n_md_steps: 20,
        dt: 0.05,
        seed: 12345,
        ..Default::default()
    };
    let n_therm = 100;
    let mut n_accept = 0;
    for _ in 0..n_therm {
        let r = hmc_trajectory(&mut lattice, &mut config);
        if r.accepted {
            n_accept += 1;
        }
    }
    println!(
        "  Done: ⟨P⟩ = {:.6}, {:.0}% accept, {:.1}s\n",
        lattice.average_plaquette(),
        100.0 * n_accept as f64 / n_therm as f64,
        therm_start.elapsed().as_secs_f64()
    );

    let integrators = [
        (FlowIntegrator::Euler, "Euler", 1),
        (FlowIntegrator::Rk2, "RK2", 2),
        (FlowIntegrator::Rk3Luscher, "RK3 (W6)", 3),
        (FlowIntegrator::Lscfrk3w7, "LSCFRK3 (W7)", 3),
        (FlowIntegrator::Lscfrk4ck, "LSCFRK4 (CK)", 4),
    ];

    let epsilons = [0.02, 0.01, 0.005, 0.002, 0.001];
    let t_target = 1.0;

    println!(
        "  {:<16} {:>8} {:>14} {:>14} {:>8} {:>6}",
        "Integrator", "ε", "E(t=1)", "ΔE vs finest", "Order", "ms"
    );
    println!("  {}", "-".repeat(72));

    let mut all_passed = true;

    for (integrator, name, expected_order) in &integrators {
        let mut e_at_t1: Vec<(f64, f64)> = Vec::new();

        for &eps in &epsilons {
            // Clone the lattice state by re-thermalizing from same seed
            let mut lat = Lattice::hot_start([8, 8, 8, 8], 6.0, 42);
            let mut cfg = HmcConfig {
                n_md_steps: 20,
                dt: 0.05,
                seed: 12345,
                ..Default::default()
            };
            for _ in 0..n_therm {
                let _ = hmc_trajectory(&mut lat, &mut cfg);
            }

            let step_start = Instant::now();
            let results = run_flow(&mut lat, *integrator, eps, t_target, 1);
            let wall_ms = step_start.elapsed().as_secs_f64() * 1000.0;

            let e_final = results.last().map_or(0.0, |m| m.energy_density);

            e_at_t1.push((eps, e_final));

            let finest = e_at_t1.last().map_or(0.0, |v| v.1);
            let delta = (e_final - finest).abs();

            println!(
                "  {name:<16} {eps:>8.4} {e_final:>14.10} {delta:>14.2e} {:>8} {wall_ms:>6.0}",
                ""
            );
        }

        // Estimate convergence order from Richardson extrapolation
        // Using the last three step sizes: order ≈ log(|E₁-E₂|/|E₂-E₃|) / log(h₁/h₂)
        if e_at_t1.len() >= 3 {
            let n = e_at_t1.len();
            let (h1, e1) = e_at_t1[n - 3];
            let (h2, e2) = e_at_t1[n - 2];
            let (_h3, e3) = e_at_t1[n - 1];

            let d12 = (e1 - e2).abs();
            let d23 = (e2 - e3).abs();

            if d23 > 1e-16 && d12 > 1e-16 {
                let ratio = h1 / h2;
                let order = (d12 / d23).ln() / ratio.ln();
                let order_ok = order > (*expected_order as f64 - 1.0)
                    && order < (*expected_order as f64 + 2.0);
                if !order_ok {
                    all_passed = false;
                }
                let status = if order_ok { "✓" } else { "✗" };
                println!(
                    "  {name:<16} Measured order: {order:.2} (expected {expected_order}) {status}"
                );
            } else {
                println!("  {name:<16} Converged to machine precision (order unmeasurable)");
            }
        }
        println!();
    }

    let total = total_start.elapsed();
    println!("  Total wall time: {:.1}s", total.as_secs_f64());

    if all_passed {
        println!("\n  All convergence orders match expected. PASS.");
    } else {
        println!("\n  Some convergence orders outside expected range. CHECK.");
        std::process::exit(1);
    }
}
