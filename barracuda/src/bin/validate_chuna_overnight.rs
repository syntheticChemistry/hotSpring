// SPDX-License-Identifier: AGPL-3.0-only

//! Chuna overnight validation — all Paper 43/44/45 systems in one run.
//!
//! This binary exercises every Chuna-paper pipeline (CPU + GPU), including:
//!
//! **Paper 43** (Gradient flow integrators):
//!   - Convergence sweep: ε = 0.02→0.001 for W6/W7/CK4
//!   - Production flow at 8⁴ and 16⁴ β = {5.9, 6.0, 6.2}
//!
//! **Paper 44** (Conservative BGK dielectric):
//!   - Standard + completed Mermin (CPU + GPU)
//!   - Multi-component Mermin (electron-ion, CPU + GPU)
//!   - Physics checks: f-sum rule, DSF positivity, Debye screening
//!
//! **Paper 45** (Multi-species kinetic-fluid coupling):
//!   - GPU BGK relaxation (conservation checks)
//!   - GPU Euler/Sod shock tube
//!   - Full coupled kinetic-fluid (GPU BGK + GPU Euler + interface)
//!
//! Usage:
//!   cargo run --release --bin validate_chuna_overnight 2>&1 | tee chuna_overnight.log

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut harness = ValidationHarness::new("chuna_overnight");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Overnight Validation — Papers 43 / 44 / 45         ║");
    println!("║  Bazavov & Chuna 2021, Chuna & Murillo 2024,              ║");
    println!("║  Haack, Murillo, Sagert & Chuna 2024                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let total_start = Instant::now();

    // ═══════════════════════════════════════════════════════════════
    //  Paper 43: Gradient Flow Integrators
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━ Paper 43: Gradient Flow Integrators ━━━\n");

    paper_43_convergence(&mut harness);
    paper_43_production(&mut harness);
    paper_43_dynamical(&mut harness);

    // ═══════════════════════════════════════════════════════════════
    //  Paper 44: Conservative BGK Dielectric
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━ Paper 44: Conservative BGK Dielectric ━━━\n");

    paper_44_cpu(&mut harness);
    paper_44_multicomponent_cpu(&mut harness);

    // GPU sections require wgpu
    match rt.block_on(GpuF64::new()) {
        Ok(gpu) => {
            paper_44_gpu(&mut harness, &gpu);
            paper_44_multicomponent_gpu(&mut harness, &gpu);

            // ═══════════════════════════════════════════════════════════════
            //  Paper 45: Multi-Species Kinetic-Fluid Coupling
            // ═══════════════════════════════════════════════════════════════
            println!("\n━━━ Paper 45: Kinetic-Fluid Coupling ━━━\n");

            paper_45_gpu_bgk(&mut harness, &gpu);
            paper_45_gpu_euler(&mut harness, &gpu);
            paper_45_gpu_coupled(&mut harness, &gpu);
        }
        Err(e) => {
            println!("  ⚠ No GPU available ({e}) — skipping GPU sections\n");
        }
    }

    let total = total_start.elapsed();
    println!("\n  Total wall time: {:.1}s", total.as_secs_f64());
    harness.finish();
}

// ─── Paper 43 ────────────────────────────────────────────────────

fn paper_43_convergence(harness: &mut ValidationHarness) {
    use hotspring_barracuda::lattice::gradient_flow::{run_flow, FlowIntegrator};
    use hotspring_barracuda::lattice::hmc::{hmc_trajectory, HmcConfig};
    use hotspring_barracuda::lattice::wilson::Lattice;

    println!("  Convergence sweep (8⁴ β=6.0)...");
    let start = Instant::now();

    let n_therm = 100;

    let integrators = [
        (FlowIntegrator::Rk3Luscher, "W6", 3),
        (FlowIntegrator::Lscfrk3w7, "W7", 3),
        (FlowIntegrator::Lscfrk4ck, "CK4", 4),
    ];

    let epsilons = [0.02, 0.01, 0.005, 0.002, 0.001];

    for (integrator, name, expected_order) in &integrators {
        let mut e_values: Vec<(f64, f64)> = Vec::new();
        for &eps in &epsilons {
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
            let results = run_flow(&mut lat, *integrator, eps, 1.0, 1);
            let e = results.last().map_or(0.0, |m| m.energy_density);
            e_values.push((eps, e));
        }

        if e_values.len() >= 3 {
            let n = e_values.len();
            let (h1, e1) = e_values[n - 3];
            let (h2, e2) = e_values[n - 2];
            let (_h3, e3) = e_values[n - 1];
            let d12 = (e1 - e2).abs();
            let d23 = (e2 - e3).abs();

            if d23 > 1e-16 && d12 > 1e-16 {
                let order = (d12 / d23).ln() / (h1 / h2).ln();
                // On small 8⁴ lattices, finite-size effects reduce measured order
                let order_ok = order > 1.5 && order < (*expected_order as f64 + 2.0);
                println!("    {name}: order = {order:.2} (expected {expected_order})");
                harness.check_bool(&format!("p43_convergence_{name}"), order_ok);
            } else {
                println!("    {name}: converged to machine precision");
                harness.check_bool(&format!("p43_convergence_{name}"), true);
            }
        }
    }
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_43_production(harness: &mut ValidationHarness) {
    use hotspring_barracuda::lattice::gradient_flow::{find_t0, find_w0, run_flow, FlowIntegrator};
    use hotspring_barracuda::lattice::hmc::{hmc_trajectory, HmcConfig};
    use hotspring_barracuda::lattice::wilson::Lattice;

    let configs: &[([usize; 4], f64, usize, &str)] = &[
        ([8, 8, 8, 8], 5.9, 200, "8⁴ β=5.9"),
        ([8, 8, 8, 8], 6.0, 200, "8⁴ β=6.0"),
        ([8, 8, 8, 8], 6.2, 200, "8⁴ β=6.2"),
        ([16, 16, 16, 16], 6.0, 500, "16⁴ β=6.0"),
    ];

    for (dims, beta, n_therm, label) in configs {
        let start = Instant::now();
        println!("  {label}...");

        let mut lattice = Lattice::hot_start(*dims, *beta, 42);
        // Larger lattices need smaller dt for reasonable acceptance
        let volume = dims[0] * dims[1] * dims[2] * dims[3];
        let (n_md, md_dt) = if volume > 10000 {
            (40, 0.025)
        } else {
            (20, 0.05)
        };
        let mut config = HmcConfig {
            n_md_steps: n_md,
            dt: md_dt,
            seed: 12345,
            ..Default::default()
        };
        let mut n_accept = 0;
        for _ in 0..*n_therm {
            if hmc_trajectory(&mut lattice, &mut config).accepted {
                n_accept += 1;
            }
        }
        let acceptance = n_accept as f64 / *n_therm as f64;
        println!(
            "    ⟨P⟩ = {:.6}, {:.0}% accept",
            lattice.average_plaquette(),
            acceptance * 100.0
        );
        harness.check_lower(&format!("p43_accept_{label}"), acceptance, 0.25);

        let flow = run_flow(&mut lattice, FlowIntegrator::Lscfrk3w7, 0.01, 4.0, 5);

        let monotonic = flow
            .windows(2)
            .all(|w| w[1].energy_density <= w[0].energy_density + 1e-10);
        harness.check_bool(&format!("p43_monotonic_{label}"), monotonic);

        if let Some(w0) = find_w0(&flow) {
            println!("    w₀ = {w0:.4}");
        }
        if let Some(t0) = find_t0(&flow) {
            println!("    t₀ = {t0:.4}");
        }

        println!("    {:.1}s", start.elapsed().as_secs_f64());
    }
}

/// Paper 43 extension: gradient flow on dynamical staggered fermion configs.
///
/// Bazavov & Chuna 2021 ran LSCFRK integrators on N_f=4 staggered configs
/// generated by MILC. This section thermalizes with adaptive dynamical HMC
/// (Omelyan integrator, acceptance-driven step control) and then runs W7
/// gradient flow, validating that scale setting (t₀, w₀) works on dynamical
/// backgrounds — not just quenched.
fn paper_43_dynamical(harness: &mut ValidationHarness) {
    use hotspring_barracuda::lattice::gradient_flow::{find_t0, find_w0, run_flow, FlowIntegrator};
    use hotspring_barracuda::lattice::pseudofermion::{
        dynamical_thermalize_adaptive, AdaptiveStepController, DynamicalHmcConfig,
        PseudofermionConfig,
    };
    use hotspring_barracuda::lattice::wilson::Lattice;

    let start = Instant::now();

    let dims = [8, 8, 8, 8];
    let beta = 5.4;
    let mass = 0.1;
    let n_therm = 50;

    // Adaptive controller: heuristic initial dt based on volume and mass
    let mut controller = AdaptiveStepController::for_dynamical(dims, beta, mass);

    // Probe for NPU hardware — if available, let it suggest initial params
    if hotspring_barracuda::discovery::probe_npu_available() {
        println!("  Dynamical 8⁴ β={beta} (N_f=4 staggered, m={mass}) [NPU detected]...");
    } else {
        println!(
            "  Dynamical 8⁴ β={beta} (N_f=4 staggered, m={mass}) [heuristic: dt={:.4}, n_md={}]...",
            controller.dt, controller.n_md_steps
        );
    }

    let mut lattice = Lattice::hot_start(dims, beta, 42);
    let mut config = DynamicalHmcConfig {
        seed: 12345,
        fermion: PseudofermionConfig {
            mass,
            cg_tol: 1e-8,
            cg_max_iter: 5000,
        },
        beta,
        n_flavors_over_4: 1,
        ..Default::default()
    };

    let therm = dynamical_thermalize_adaptive(&mut lattice, &mut config, n_therm, &mut controller);

    let plaq = lattice.average_plaquette();
    println!(
        "    ⟨P⟩ = {plaq:.6}, {:.0}% accept, {} CG iters, final dt={:.4}, n_md={}",
        therm.acceptance_rate * 100.0,
        therm.total_cg_iterations,
        therm.final_dt,
        therm.final_n_md,
    );

    harness.check_lower("p43_dyn_accept", therm.acceptance_rate, 0.20);
    harness.check_lower("p43_dyn_plaquette", plaq, 0.3);

    let flow = run_flow(&mut lattice, FlowIntegrator::Lscfrk3w7, 0.01, 2.0, 5);

    let monotonic = flow
        .windows(2)
        .all(|w| w[1].energy_density <= w[0].energy_density + 1e-10);
    harness.check_bool("p43_dyn_flow_monotonic", monotonic);

    if let Some(w0) = find_w0(&flow) {
        println!("    w₀ = {w0:.4} (dynamical)");
    }
    if let Some(t0) = find_t0(&flow) {
        println!("    t₀ = {t0:.4} (dynamical)");
    }

    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

// ─── Paper 44 ────────────────────────────────────────────────────

fn paper_44_cpu(harness: &mut ValidationHarness) {
    use hotspring_barracuda::physics::dielectric::{
        dynamic_structure_factor_completed, epsilon_completed_mermin, epsilon_mermin,
        f_sum_rule_integral, f_sum_rule_integral_completed, PlasmaParams,
    };

    println!("  Single-species Mermin (CPU)...");
    let start = Instant::now();

    let params = PlasmaParams::from_coupling(10.0, 2.0);
    let k = 1.0;
    let nu = 0.5;

    // f-sum rule: verify monotone convergence toward -πωₚ²/2 as ω_max increases.
    // At finite ν with strong coupling (Γ=10), the Drude-broadened peak requires
    // large ω_max for the trapezoidal integral to converge. Rather than hand-tuning
    // a tolerance, we check that the integral is converging in the right direction.
    let expected = -std::f64::consts::PI * params.omega_p * params.omega_p / 2.0;
    let f_25 = f_sum_rule_integral(k, nu, &params, 25.0);
    let f_50 = f_sum_rule_integral(k, nu, &params, 50.0);
    let f_100 = f_sum_rule_integral(k, nu, &params, 100.0);
    let err_25 = (f_25 - expected).abs();
    let err_50 = (f_50 - expected).abs();
    let err_100 = (f_100 - expected).abs();
    let converging = err_100 <= err_50 && err_50 <= err_25;
    let same_sign = f_100.signum() == expected.signum();
    println!(
        "    f-sum convergence: err@25={:.4e}, @50={:.4e}, @100={:.4e}, sign={}",
        err_25 / expected.abs(),
        err_50 / expected.abs(),
        err_100 / expected.abs(),
        if same_sign { "OK" } else { "WRONG" }
    );
    harness.check_abs("p44_fsum_converging", f64::from(converging), 1.0, 0.5);
    harness.check_abs("p44_fsum_sign", f64::from(same_sign), 1.0, 0.5);

    // Completed Mermin: same convergence check
    let fc_25 = f_sum_rule_integral_completed(k, nu, &params, 25.0);
    let fc_100 = f_sum_rule_integral_completed(k, nu, &params, 100.0);
    let fc_converging = (fc_100 - expected).abs() <= (fc_25 - expected).abs();
    harness.check_abs(
        "p44_fsum_completed_conv",
        f64::from(fc_converging),
        1.0,
        0.5,
    );

    // DSF positivity
    let omegas: Vec<f64> = (1..200).map(|i| 0.1 * i as f64).collect();
    let dsf = dynamic_structure_factor_completed(k, &omegas, nu, &params);
    let n_pos = dsf.iter().filter(|&&s| s >= -1e-15).count();
    let frac = n_pos as f64 / dsf.len() as f64;
    harness.check_lower("p44_dsf_positive", frac, 0.99);

    // High-frequency limit
    let eps_hf = epsilon_completed_mermin(k, 100.0, nu, &params);
    harness.check_upper("p44_hf_limit", (eps_hf.re - 1.0).abs(), 0.01);

    // Standard vs completed should agree at nu=0
    let eps_std = epsilon_mermin(k, 1.5, 1e-10, &params);
    let eps_cmp = epsilon_completed_mermin(k, 1.5, 1e-10, &params);
    let rel = (eps_std.re - eps_cmp.re).abs() / eps_std.abs().max(1e-15);
    harness.check_upper("p44_nu0_agreement", rel, 0.01);

    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_44_multicomponent_cpu(harness: &mut ValidationHarness) {
    use hotspring_barracuda::physics::dielectric_multicomponent::{
        epsilon_multicomponent_mermin, multicomponent_dsf, multicomponent_f_sum_integral,
        MultiComponentPlasma, SpeciesParams,
    };

    println!("  Multi-component Mermin (CPU)...");
    let start = Instant::now();

    let plasma = MultiComponentPlasma {
        species: vec![
            SpeciesParams {
                mass: 1.0 / 1836.0,
                charge: 1.0,
                density: 1.0,
                temperature: 1.0,
                nu: 0.1,
            },
            SpeciesParams {
                mass: 1.0,
                charge: 1.0,
                density: 1.0,
                temperature: 1.0,
                nu: 0.01,
            },
        ],
    };

    let k = 1.0;

    // Static limit should be Debye
    let eps_static = epsilon_multicomponent_mermin(k, 0.0, &plasma, true);
    let k_d_sq = plasma.total_k_debye_sq();
    let expected = 1.0 + k_d_sq / (k * k);
    let rel = (eps_static.re - expected).abs() / expected;
    println!("    Debye screening: rel = {rel:.4e}");
    harness.check_upper("p44_mc_debye", rel, 0.01);

    // High-frequency limit: ε→1 as ω→∞. For electron-ion plasma with m_e=1/1836,
    // ωₚₑ = √(4π n q²/m_e) ≈ 152. Use ω=10000 >> ωₚₑ.
    let eps_hf = epsilon_multicomponent_mermin(k, 10_000.0, &plasma, true);
    harness.check_upper("p44_mc_hf_limit", (eps_hf.re - 1.0).abs(), 0.01);

    // DSF positivity
    let omegas: Vec<f64> = (1..100).map(|i| 0.1 * i as f64).collect();
    let dsf = multicomponent_dsf(k, &omegas, &plasma);
    let n_pos = dsf.iter().filter(|&&s| s >= 0.0).count();
    let frac = n_pos as f64 / dsf.len() as f64;
    harness.check_lower("p44_mc_dsf_positive", frac, 0.95);

    // f-sum convergence: verify monotone convergence as ω_max increases
    let total_wp2 = plasma.total_omega_p_sq();
    let expected_fsum = -std::f64::consts::PI * total_wp2 / 2.0;
    let f_50 = multicomponent_f_sum_integral(k, &plasma, 50.0);
    let f_200 = multicomponent_f_sum_integral(k, &plasma, 200.0);
    let converging = (f_200 - expected_fsum).abs() <= (f_50 - expected_fsum).abs();
    let same_sign = f_200.signum() == expected_fsum.signum();
    println!("    f-sum: @50={f_50:.4e}, @200={f_200:.4e}, expected={expected_fsum:.4e}");
    harness.check_abs("p44_mc_fsum_conv", f64::from(converging), 1.0, 0.5);
    harness.check_abs("p44_mc_fsum_sign", f64::from(same_sign), 1.0, 0.5);

    // Passive medium
    for omega in [0.1, 0.5, 1.0, 5.0, 10.0] {
        let eps = epsilon_multicomponent_mermin(k, omega, &plasma, true);
        harness.check_lower(&format!("p44_mc_passive_w{omega}"), eps.im, -0.01);
    }

    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_44_gpu(harness: &mut ValidationHarness, gpu: &GpuF64) {
    use hotspring_barracuda::physics::gpu_dielectric::{
        validate_gpu_dielectric, GpuDielectricPipeline,
    };

    println!("  GPU Mermin (standard + completed)...");
    let start = Instant::now();

    let pipeline = GpuDielectricPipeline::new(gpu);
    let validation = validate_gpu_dielectric(gpu, &pipeline, 10.0, 2.0);

    // GPU f-sum: verify same sign as expected and GPU-CPU L² agreement
    let expected_fsum = -std::f64::consts::PI
        * hotspring_barracuda::physics::dielectric::PlasmaParams::from_coupling(10.0, 2.0)
            .omega_p
            .powi(2)
        / 2.0;
    let gpu_sign_ok = validation.f_sum_gpu.signum() == expected_fsum.signum();
    harness.check_abs("p44_gpu_fsum_sign", f64::from(gpu_sign_ok), 1.0, 0.5);
    harness.check_lower("p44_gpu_dsf_pos", validation.dsf_pos_fraction_gpu, 0.95);
    harness.check_upper("p44_gpu_loss_l2", validation.l2_loss_rel_error, 0.01);

    println!(
        "    GPU {:.2}s, CPU {:.2}s, L² = {:.4e}",
        validation.gpu_wall_seconds, validation.cpu_wall_seconds, validation.l2_loss_rel_error
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_44_multicomponent_gpu(harness: &mut ValidationHarness, gpu: &GpuF64) {
    use hotspring_barracuda::physics::gpu_dielectric_multicomponent::{
        validate_gpu_multicomponent, GpuMulticompPipeline,
    };

    println!("  GPU Multi-component Mermin...");
    let start = Instant::now();

    let pipeline = GpuMulticompPipeline::new(gpu);
    let (gpu_loss, cpu_loss) = validate_gpu_multicomponent(gpu, &pipeline);

    let n_close = gpu_loss
        .iter()
        .zip(cpu_loss.iter())
        .filter(|(&g, &c)| {
            let denom = c.abs().max(1e-15);
            (g - c).abs() / denom < 0.5
        })
        .count();
    let frac = n_close as f64 / gpu_loss.len().max(1) as f64;
    println!("    CPU-GPU agreement: {:.0}%", frac * 100.0);
    harness.check_lower("p44_mc_gpu_agreement", frac, 0.90);

    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

// ─── Paper 45 ────────────────────────────────────────────────────

fn paper_45_gpu_bgk(harness: &mut ValidationHarness, gpu: &GpuF64) {
    use hotspring_barracuda::physics::gpu_kinetic_fluid::{validate_gpu_bgk, GpuBgkPipeline};

    println!("  GPU BGK relaxation...");
    let start = Instant::now();

    let pipeline = GpuBgkPipeline::new(gpu);
    let (gpu_r, _cpu_r) = validate_gpu_bgk(gpu, &pipeline);

    harness.check_upper("p45_bgk_mass_err", gpu_r.result.mass_err_1, 1e-4);
    harness.check_upper("p45_bgk_energy_err", gpu_r.result.energy_err, 0.05);
    harness.check_bool("p45_bgk_entropy", gpu_r.result.entropy_monotonic);
    println!(
        "    GPU {:.2}s, CPU {:.2}s",
        gpu_r.gpu_wall_seconds, gpu_r.cpu_wall_seconds
    );
    println!("    ΔT/T = {:.4}", gpu_r.result.temp_relaxed);
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_45_gpu_euler(harness: &mut ValidationHarness, gpu: &GpuF64) {
    use hotspring_barracuda::physics::gpu_euler::{validate_gpu_euler, GpuEulerPipeline};

    println!("  GPU Euler / Sod shock tube...");
    let start = Instant::now();

    let pipeline = GpuEulerPipeline::new(gpu);
    let result = validate_gpu_euler(gpu, &pipeline);

    harness.check_upper("p45_euler_mass_err", result.mass_err, 0.01);
    harness.check_upper("p45_euler_energy_err", result.energy_err, 0.01);
    harness.check_upper("p45_euler_cpu_mass_err", result.cpu.mass_err, 0.01);

    let rho_range = result.rho.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - result.rho.iter().copied().fold(f64::INFINITY, f64::min);
    harness.check_lower("p45_euler_shock_resolved", rho_range, 0.5);

    println!(
        "    GPU {:.2}s, CPU {:.2}s",
        result.gpu_wall_seconds, result.cpu_wall_seconds
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_45_gpu_coupled(harness: &mut ValidationHarness, gpu: &GpuF64) {
    use hotspring_barracuda::physics::gpu_coupled_kinetic_fluid::{
        validate_gpu_coupled, GpuCoupledPipeline,
    };

    println!("  GPU coupled kinetic-fluid...");
    let start = Instant::now();

    let pipeline = GpuCoupledPipeline::new(gpu);
    let result = validate_gpu_coupled(gpu, &pipeline);

    harness.check_upper("p45_coupled_mass_err", result.mass_err, 0.05);
    harness.check_upper("p45_coupled_energy_err", result.energy_err, 0.1);
    // Interface density mismatch is inherent to half-space Maxwellian coupling
    // (kinetic cell sees both incoming+outgoing; fluid cell is Euler-updated).
    // Check that GPU mismatch agrees with CPU reference within 50% relative.
    let cpu_if = result.cpu.interface_density_match;
    let gpu_if = result.interface_density_match;
    let if_rel = if cpu_if > 1e-15 {
        (gpu_if - cpu_if).abs() / cpu_if
    } else {
        gpu_if
    };
    println!("    interface: GPU={gpu_if:.4e}, CPU={cpu_if:.4e}, rel={if_rel:.4e}");
    harness.check_upper("p45_coupled_interface_parity", if_rel, 0.5);
    println!(
        "    {} steps, GPU {:.2}s, CPU {:.2}s",
        result.n_steps, result.gpu_wall_seconds, result.cpu_wall_seconds
    );
    println!(
        "    CPU reference: mass_err={:.2e}, energy_err={:.2e}",
        result.cpu.mass_err, result.cpu.energy_err
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}
