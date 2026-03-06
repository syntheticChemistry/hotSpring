// SPDX-License-Identifier: AGPL-3.0-only

//! Compare gradient flow integrators — Bazavov & Chuna (2021) reproduction.
//!
//! Runs all implemented flow integrators on the same gauge configuration
//! and compares E(t), t₀, w₀ at matched flow time. This directly reproduces
//! the integrator comparison from arXiv:2101.05320, Tables and Figures 2-6.
//!
//! Usage:
//!   cargo run --release --bin `compare_flow_integrators`

use hotspring_barracuda::lattice::gradient_flow::{find_t0, find_w0, run_flow, FlowIntegrator};
use hotspring_barracuda::lattice::hmc::{hmc_trajectory, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Gradient Flow Integrator Comparison                       ║");
    println!("║  Bazavov & Chuna, arXiv:2101.05320 (2021)                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let dims = [8, 8, 8, 8];
    let beta = 6.0;
    let seed = 42u64;
    let vol: usize = dims.iter().product();

    println!("\n  Lattice: {}⁴ ({} sites), β={}", dims[0], vol, beta);

    println!("\n  Phase 1: Generate thermalized config (100 HMC)...");
    let t0 = Instant::now();
    let mut lattice = Lattice::hot_start(dims, beta, seed);
    let mut hmc_cfg = HmcConfig {
        n_md_steps: 20,
        dt: 0.05,
        seed,
        integrator: IntegratorType::Omelyan,
    };
    for _ in 0..100 {
        hmc_trajectory(&mut lattice, &mut hmc_cfg);
    }
    println!(
        "    Done: ⟨P⟩={:.6} ({:.1}s)",
        lattice.average_plaquette(),
        t0.elapsed().as_secs_f64()
    );

    let integrators = [
        ("Euler", FlowIntegrator::Euler),
        ("RK2", FlowIntegrator::Rk2),
        ("LSCFRK3W6 (Lüscher)", FlowIntegrator::Rk3Luscher),
        ("LSCFRK3W7 (Chuna)", FlowIntegrator::Lscfrk3w7),
        ("LSCFRK4CK (CK45)", FlowIntegrator::Lscfrk4ck),
    ];

    let step_sizes = [0.02, 0.01, 0.005];
    let t_max = 2.0;

    println!("\n  Phase 2: Compare integrators at multiple step sizes");
    println!("  ═══════════════════════════════════════════════════════");

    for &epsilon in &step_sizes {
        println!("\n  ── Step size ε = {epsilon} ──");
        println!(
            "  {:30} {:>10} {:>10} {:>10} {:>8} {:>8}",
            "Integrator", "E(t=2)", "t₀", "w₀", "steps", "time(s)"
        );
        println!(
            "  {:30} {:>10} {:>10} {:>10} {:>8} {:>8}",
            "─".repeat(30),
            "─".repeat(10),
            "─".repeat(10),
            "─".repeat(10),
            "─".repeat(8),
            "─".repeat(8)
        );

        for (name, integrator) in &integrators {
            let mut flow_lat = Lattice {
                dims: lattice.dims,
                links: lattice.links.clone(),
                beta: lattice.beta,
            };

            let start = Instant::now();
            let measurements = run_flow(&mut flow_lat, *integrator, epsilon, t_max, 1);
            let elapsed = start.elapsed().as_secs_f64();

            let e_final = measurements.last().map_or(f64::NAN, |m| m.energy_density);
            let t0_val = find_t0(&measurements);
            let w0_val = find_w0(&measurements);
            let n_steps = (t_max / epsilon).round() as usize;

            println!(
                "  {:30} {:>10.6} {:>10} {:>10} {:>8} {:>8.3}",
                name,
                e_final,
                t0_val.map_or("N/A".to_string(), |v| format!("{v:.6}")),
                w0_val.map_or("N/A".to_string(), |v| format!("{v:.6}")),
                n_steps,
                elapsed,
            );
        }
    }

    println!("\n  Phase 3: Convergence test — E(t=1) vs step size");
    println!("  ═══════════════════════════════════════════════════════");

    let eps_scan = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002];
    let t_test = 1.0;

    let mut ref_lat = Lattice {
        dims: lattice.dims,
        links: lattice.links.clone(),
        beta: lattice.beta,
    };
    let ref_result = run_flow(&mut ref_lat, FlowIntegrator::Lscfrk4ck, 0.001, t_test, 1);
    let e_ref = ref_result.last().unwrap().energy_density;
    println!("  Reference: LSCFRK4CK at ε=0.001 → E(1) = {e_ref:.10}");

    println!(
        "\n  {:>8} {:>15} {:>15} {:>15}",
        "ε", "LSCFRK3W6", "LSCFRK3W7", "LSCFRK4CK"
    );
    println!(
        "  {:>8} {:>15} {:>15} {:>15}",
        "─".repeat(8),
        "─".repeat(15),
        "─".repeat(15),
        "─".repeat(15)
    );

    for &eps in &eps_scan {
        let mut errors = Vec::new();
        for integrator in [
            FlowIntegrator::Rk3Luscher,
            FlowIntegrator::Lscfrk3w7,
            FlowIntegrator::Lscfrk4ck,
        ] {
            let mut flow_lat = Lattice {
                dims: lattice.dims,
                links: lattice.links.clone(),
                beta: lattice.beta,
            };
            let result = run_flow(&mut flow_lat, integrator, eps, t_test, 1);
            let e = result.last().unwrap().energy_density;
            errors.push((e - e_ref).abs());
        }
        println!(
            "  {:>8.4} {:>15.2e} {:>15.2e} {:>15.2e}",
            eps, errors[0], errors[1], errors[2]
        );
    }

    println!("\n  Expected scaling: 3rd order → error ∝ ε³, 4th order → error ∝ ε⁴");
    println!("  LSCFRK3W7 should have smaller error constant than LSCFRK3W6");
}
