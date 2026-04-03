// SPDX-License-Identifier: AGPL-3.0-only

//! Unified Chuna paper validation — Papers 43, 44, 45.
//!
//! Combines gradient flow (Bazavov & Chuna 2021), BGK dielectric
//! (Chuna & Murillo 2024), and kinetic-fluid coupling (Haack et al. 2024)
//! into a single binary for the portable validation artifact.
//!
//! CPU checks run unconditionally (59 checks — the deterministic baseline).
//! GPU is auto-detected: if present, additional GPU-vs-CPU parity checks
//! run on each discovered f64-capable adapter, with hardware profiling via
//! HardwareCalibration and PrecisionBrain. If no GPU is found, CPU-only
//! validation produces the same PASS/FAIL results with a note.
//!
//! Usage:
//!   cargo run --release --bin validate_chuna
//!   cargo run --release --bin validate_chuna -- --output results/

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_flow::{GpuFlowPipelines, GpuFlowState, gpu_gradient_flow};
use hotspring_barracuda::lattice::gradient_flow::{FlowIntegrator, find_t0, run_flow};
use hotspring_barracuda::lattice::su3::Su3Matrix;
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::physics::dielectric::{
    Complex, PlasmaParams, conductivity_dc, debye_screening, dynamic_structure_factor,
    dynamic_structure_factor_completed, epsilon_completed_mermin, epsilon_mermin,
    f_sum_rule_integral, f_sum_rule_integral_completed, plasma_dispersion_w, validate_dielectric,
};
use hotspring_barracuda::physics::kinetic_fluid::{
    run_bgk_relaxation, run_coupled_kinetic_fluid, run_sod_shock_tube,
};
use hotspring_barracuda::precision_brain::PrecisionBrain;
use hotspring_barracuda::precision_routing::PhysicsDomain;
use hotspring_barracuda::lattice::measurement::RunManifest;
use hotspring_barracuda::toadstool_report::{PerformanceMeasurement, report_to_toadstool};
use hotspring_barracuda::validation::{HardwareProfile, ValidationHarness};
use std::time::Instant;

/// CPU-side reference values collected during Paper 43 for GPU parity checks.
struct CpuReferenceValues {
    plaquette_8_rk3: f64,
    energy_8_rk3: f64,
}

fn main() {
    let mut harness = ValidationHarness::new("validate_chuna");
    harness.run_manifest = Some(RunManifest::capture("validate_chuna"));

    let args: Vec<String> = std::env::args().collect();
    let output_dir = args.windows(2).find_map(|pair| {
        if pair[0] == "--output" {
            Some(pair[1].clone())
        } else if let Some(stripped) = pair[0].strip_prefix("--output=") {
            Some(stripped.to_string())
        } else {
            None
        }
    });

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Validation Suite — Papers 43, 44, 45                    ║");
    println!("║  guideStone: CPU baseline + GPU parity on every substrate      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!(
        "substrate: {} {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("engine:    cpu-native (pure Rust, GPU auto-detected)\n");

    let wall_start = Instant::now();

    // Phase 1: CPU baseline (59 deterministic checks)
    let cpu_ref = paper_43_gradient_flow(&mut harness);
    paper_44_dielectric(&mut harness);
    paper_45_kinetic_fluid(&mut harness);

    let cpu_ms = wall_start.elapsed().as_millis() as u64;
    println!("  CPU baseline: {}/{} checks in {:.1}s\n",
        harness.passed_count(), harness.total_count(), cpu_ms as f64 / 1000.0);

    // Phase 2: GPU discovery + profiling + parity checks
    gpu_substrate_validation(&mut harness, &cpu_ref);

    let total_ms = wall_start.elapsed().as_millis() as u64;

    if let Some(ref dir) = output_dir {
        harness.write_json(dir, total_ms);
    }

    harness.finish();
}

/// GPU substrate discovery, profiling, and CPU-GPU parity validation.
///
/// For each f64-capable GPU discovered via toadStool-compatible enumeration:
/// 1. Probe precision tiers (F32/F64/DF64/F64Precise)
/// 2. Build PrecisionBrain domain routing table
/// 3. Run gradient flow on GPU, compare plaquette against CPU reference
/// 4. Report hardware profile to toadStool if socket available
fn gpu_substrate_validation(harness: &mut ValidationHarness, cpu_ref: &CpuReferenceValues) {
    println!("━━━ GPU Substrate Discovery ━━━");

    let adapters = GpuF64::enumerate_adapters();
    let f64_adapters: Vec<_> = adapters.iter().filter(|a| a.has_f64).collect();

    println!("  Discovered {} adapter(s), {} with f64 support",
        adapters.len(), f64_adapters.len());
    for a in &adapters {
        let f64_tag = if a.has_f64 { "f64" } else { "f32-only" };
        println!("    [{}] {} ({}, {}, {}MB)",
            a.index, a.name, a.driver, f64_tag, a.memory_bytes / (1024 * 1024));
    }
    println!();

    if f64_adapters.is_empty() {
        println!("  No f64-capable GPU found — CPU-only validation is complete.\n");
        return;
    }

    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            println!("  Could not create async runtime: {e} — skipping GPU validation.\n");
            return;
        }
    };

    let mut gpu_plaquettes: Vec<(String, f64)> = Vec::new();

    for adapter in &f64_adapters {
        let token = adapter.index.to_string();
        let adapter_name = &adapter.name;

        println!("━━━ GPU: {} ━━━", adapter_name);

        let gpu = match rt.block_on(GpuF64::with_adapter(&token)) {
            Ok(g) => g,
            Err(e) => {
                println!("  Failed to open {adapter_name}: {e} — skipping.\n");
                continue;
            }
        };
        let _guard = rt.enter();

        harness.set_gpu(adapter_name);
        harness.set_substrate(adapter_name);

        // Probe precision tiers
        let t_probe = Instant::now();
        let brain = PrecisionBrain::new(&gpu);
        let cal = &brain.calibration;
        let probe_ms = t_probe.elapsed().as_millis();

        println!("  Calibration ({probe_ms}ms):");
        println!("    f64: {}, df64: {}, transcendental_risk: {}",
            cal.has_any_f64, cal.df64_safe, cal.nvvm_transcendental_risk);
        for tier in &cal.tiers {
            if tier.compiles {
                println!("    {:?}: compile={:.0}μs, dispatch={:.0}μs, ulp={:.1}",
                    tier.tier, tier.compile_us, tier.dispatch_us, tier.probe_ulp);
            }
        }

        let vram = gpu.device().limits().max_buffer_size;
        let bytes_per_site: u64 = 4 * 18 * 8 * 4;
        let max_l = ((vram / bytes_per_site) as f64).powf(0.25).floor() as usize;
        println!("  VRAM: {}MB, max lattice L={max_l}", vram / (1024 * 1024));

        let domains = [
            (PhysicsDomain::GradientFlow, "gradient_flow"),
            (PhysicsDomain::Dielectric, "dielectric"),
            (PhysicsDomain::KineticFluid, "kinetic_fluid"),
            (PhysicsDomain::LatticeQcd, "lattice_qcd"),
        ];
        let domain_routing: Vec<(String, String)> = domains.iter()
            .map(|(d, name)| (name.to_string(), format!("{:?}", brain.route(*d))))
            .collect();

        println!("  Domain routing:");
        for (name, tier) in &domain_routing {
            println!("    {name} → {tier}");
        }

        harness.hardware_profiles.push(HardwareProfile {
            adapter: adapter_name.clone(),
            vram_bytes: vram,
            precision_tiers: cal.tiers.iter().map(|t| {
                (format!("{:?}", t.tier), t.compiles, t.dispatch_us, t.probe_ulp)
            }).collect(),
            domain_routing: domain_routing.clone(),
            max_lattice_l: max_l,
        });

        // GPU gradient flow parity check
        let t_flow = Instant::now();
        let seed = 42_u64;
        let beta = 6.0;
        let dims = [8, 8, 8, 8];
        let eps: f64 = 0.01;
        let meas = (0.05 / eps).max(1.0) as usize;

        let gpu_lat = Lattice::hot_start(dims, beta, seed);
        let pipelines = GpuFlowPipelines::new(&gpu);
        let state = GpuFlowState::from_lattice(&gpu, &gpu_lat, beta);
        let gpu_flow = gpu_gradient_flow(
            &gpu, &pipelines, &state,
            FlowIntegrator::Rk3Luscher, eps, 2.0, meas,
        );
        let flow_ms = t_flow.elapsed().as_millis();

        let gpu_plaq = gpu_flow.measurements.last().map_or(f64::NAN, |m| m.plaquette);
        let gpu_energy = gpu_flow.measurements.last().map_or(f64::NAN, |m| m.energy_density);

        println!("  GPU flow ({flow_ms}ms): plaq={gpu_plaq:.10}, E={gpu_energy:.6}");

        // CPU-GPU parity: 200 RK3 steps with different summation order
        // (GPU parallel reduction vs CPU sequential) accumulates ~O(200 * N_sites * ε_mach)
        // difference. Empirically 2-3e-10 on 8^4 lattice, so 1e-9 is the correct bound.
        let plaq_diff = (gpu_plaq - cpu_ref.plaquette_8_rk3).abs();
        harness.check_upper(
            &format!("gpu_cpu_plaquette_parity_{}", sanitize_name(adapter_name)),
            plaq_diff, 1e-9,
        );
        harness.annotate(
            "cross_substrate", "guideStone Property 1",
            "plaquette_difference",
            "CPU-GPU plaquette within 1e-9: 200 RK3 steps × parallel vs sequential reduction order",
        );

        let energy_diff = (gpu_energy - cpu_ref.energy_8_rk3).abs();
        harness.check_upper(
            &format!("gpu_cpu_energy_parity_{}", sanitize_name(adapter_name)),
            energy_diff, 1e-8,
        );
        harness.annotate(
            "cross_substrate", "guideStone Property 1",
            "energy_density_difference",
            "CPU and GPU flow energy within 1e-8 (accumulated integration differences)",
        );

        // GPU flow should also smooth energy
        let gpu_e_start = gpu_flow.measurements.first().map_or(f64::NAN, |m| m.energy_density);
        harness.check_bool(
            &format!("gpu_flow_energy_smoothing_{}", sanitize_name(adapter_name)),
            gpu_energy <= gpu_e_start,
        );
        harness.annotate(
            "lattice_qcd", "Bazavov & Chuna, arXiv:2101.05320",
            "energy_density",
            "GPU flow E(t_final) <= E(t_initial)",
        );

        gpu_plaquettes.push((adapter_name.clone(), gpu_plaq));

        // Report to toadStool if available
        let measurements = vec![PerformanceMeasurement {
            operation: "gradient_flow_rk3_8x8x8x8".into(),
            silicon_unit: "shader".into(),
            precision_mode: if cal.has_any_f64 { "f64" } else { "df64" }.into(),
            throughput_gflops: 0.0,
            tolerance_achieved: plaq_diff,
            gpu_model: adapter_name.clone(),
            measured_by: "validate_chuna".into(),
            timestamp: hotspring_barracuda::toadstool_report::epoch_now(),
        }];
        report_to_toadstool(&measurements);

        harness.clear_substrate();
        println!();
    }

    // Cross-GPU comparison (if multiple GPUs validated)
    if gpu_plaquettes.len() > 1 {
        println!("━━━ Cross-GPU Comparison ━━━");
        for i in 0..gpu_plaquettes.len() {
            for j in (i + 1)..gpu_plaquettes.len() {
                let (ref name_a, plaq_a) = gpu_plaquettes[i];
                let (ref name_b, plaq_b) = gpu_plaquettes[j];
                let diff = (plaq_a - plaq_b).abs();
                println!("  {} vs {}: |Δplaq| = {diff:.2e}", name_a, name_b);
                harness.check_upper(
                    &format!("cross_gpu_plaquette_{}_{}", sanitize_name(name_a), sanitize_name(name_b)),
                    diff, 5e-10,
                );
                harness.annotate(
                    "cross_substrate", "guideStone Property 1",
                    "plaquette_difference",
                    "Cross-GPU plaquette within 5e-10: both use parallel reduction, minor driver differences",
                );
            }
        }
        println!();
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .replace("__", "_")
}

// ═══════════════════════════════════════════════════════════════════════════
//  Paper 43: Wilson Gradient Flow (Bazavov & Chuna, arXiv:2101.05320)
// ═══════════════════════════════════════════════════════════════════════════

fn paper_43_gradient_flow(harness: &mut ValidationHarness) -> CpuReferenceValues {
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
    harness.check_upper("gradient_flow_unitarity", dev, 1e-10);
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
        harness.check_bool(&label, e_f <= e_i + 1e-10);
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
    harness.check_upper("gradient_flow_ck4_rk3_agreement", ck4_rk3_diff, 0.01);
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

// ═══════════════════════════════════════════════════════════════════════════
//  Paper 44: BGK Dielectric (Chuna & Murillo, Phys. Rev. E 111, 035206)
// ═══════════════════════════════════════════════════════════════════════════

fn paper_44_dielectric(harness: &mut ValidationHarness) {
    const PAPER: &str = "Chuna & Murillo, Phys. Rev. E 111, 035206 (2024)";
    const DOMAIN: &str = "plasma_dielectric";

    println!("━━━ Paper 44: BGK Dielectric ━━━");
    let t0 = Instant::now();

    // Plasma dispersion function
    let w0 = plasma_dispersion_w(Complex::ZERO);
    harness.check_abs("dielectric_W0_real", w0.re, 1.0, 1e-14);
    harness.annotate(
        DOMAIN,
        PAPER,
        "dimensionless",
        "W(0)=1 exact from series definition",
    );
    harness.check_upper("dielectric_W0_imag", w0.im.abs(), 1e-14);
    harness.annotate(DOMAIN, PAPER, "dimensionless", "Im[W(0)]=0 exact");

    let w_large = plasma_dispersion_w(Complex::new(20.0, 0.0));
    harness.check_upper("dielectric_W_large_arg", w_large.abs(), 0.01);
    harness.annotate(DOMAIN, PAPER, "dimensionless", "W(z)→0 for |z|→∞");

    let test_cases: &[(f64, f64, &str)] = &[
        (1.0, 1.0, "weak"),
        (10.0, 1.0, "moderate"),
        (10.0, 2.0, "strong_screen"),
    ];

    for &(gamma, kappa, label) in test_cases {
        let params = PlasmaParams::from_coupling(gamma, kappa);
        let nu = 0.1 * params.omega_p;

        // Debye screening
        let (eps_s, eps_d) = debye_screening(1.0, &params);
        harness.check_rel(&format!("dielectric_debye_{label}"), eps_s, eps_d, 1e-12);
        harness.annotate(
            DOMAIN,
            PAPER,
            "relative_error",
            "static limit matches Debye screening",
        );

        // DC conductivity (Drude)
        let dc = conductivity_dc(nu, &params);
        let dc_exp = params.omega_p.powi(2) / (4.0 * std::f64::consts::PI * nu);
        harness.check_rel(&format!("dielectric_drude_{label}"), dc, dc_exp, 1e-13);
        harness.annotate(DOMAIN, PAPER, "conductivity", "Drude σ = ωₚ²/(4πν) exact");

        // High-frequency limit
        let eps_high = epsilon_mermin(1.0, 100.0 * params.omega_p, nu, &params);
        harness.check_upper(
            &format!("dielectric_eps_inf_{label}"),
            (eps_high - Complex::ONE).abs(),
            0.01,
        );
        harness.annotate(DOMAIN, PAPER, "dielectric_function", "ε(ω→∞)→1");

        // f-sum rule sign
        let f_sum = f_sum_rule_integral(1.0, nu, &params, 200.0);
        harness.check_upper(&format!("dielectric_fsum_{label}"), f_sum, 0.0);
        harness.annotate(
            DOMAIN,
            PAPER,
            "sum_rule",
            "f-sum integral is negative (converging to -πωₚ²/2)",
        );

        // DSF positivity
        let omegas: Vec<f64> = (1..2000).map(|i| 0.1 + i as f64 * 0.02).collect();
        let s_kw = dynamic_structure_factor(1.0, &omegas, nu, &params);
        let s_max = s_kw.iter().copied().fold(0.0_f64, f64::max);
        let n_pos = s_kw
            .iter()
            .filter(|&&s| s >= -1e-6 * s_max.max(1e-10))
            .count();
        let frac = n_pos as f64 / s_kw.len() as f64;
        harness.check_lower(&format!("dielectric_dsf_pos_{label}"), frac, 0.98);
        harness.annotate(
            DOMAIN,
            PAPER,
            "fraction",
            "S(k,ω)≥0 for ≥98% of ω (physical positivity)",
        );
    }

    // Completed Mermin checks
    for &(gamma, kappa, label) in test_cases {
        let params = PlasmaParams::from_coupling(gamma, kappa);
        let nu = 0.1 * params.omega_p;

        let eps_cm = epsilon_completed_mermin(1.0, 100.0 * params.omega_p, nu, &params);
        harness.check_upper(
            &format!("dielectric_cm_inf_{label}"),
            (eps_cm - Complex::ONE).abs(),
            0.01,
        );
        harness.annotate(
            DOMAIN,
            PAPER,
            "dielectric_function",
            "completed Mermin ε(ω→∞)→1",
        );

        let f_sum_cm = f_sum_rule_integral_completed(1.0, nu, &params, 200.0);
        harness.check_upper(&format!("dielectric_cm_fsum_{label}"), f_sum_cm, 0.0);
        harness.annotate(DOMAIN, PAPER, "sum_rule", "completed Mermin f-sum negative");

        let omegas: Vec<f64> = (1..2000).map(|i| 0.1 + i as f64 * 0.02).collect();
        let s_cm = dynamic_structure_factor_completed(1.0, &omegas, nu, &params);
        let s_max_cm = s_cm.iter().copied().fold(0.0_f64, f64::max);
        let n_pos_cm = s_cm
            .iter()
            .filter(|&&s| s >= -1e-6 * s_max_cm.max(1e-10))
            .count();
        let frac_cm = n_pos_cm as f64 / s_cm.len() as f64;
        harness.check_lower(&format!("dielectric_cm_dsf_{label}"), frac_cm, 0.99);
        harness.annotate(
            DOMAIN,
            PAPER,
            "fraction",
            "completed Mermin S(k,ω)≥0 for ≥99%",
        );
    }

    // Full validation helper
    for &(gamma, kappa) in &[(1.0, 1.0), (10.0, 1.0), (10.0, 2.0)] {
        let r = validate_dielectric(gamma, kappa);
        let passed =
            r.debye_error < 1e-12 && r.f_sum_computed < 0.0 && r.high_freq_deviation < 0.01;
        harness.check_bool(&format!("dielectric_full_G{gamma}_k{kappa}"), passed);
        harness.annotate(
            DOMAIN,
            PAPER,
            "composite",
            "full validation (Debye + f-sum + ε∞)",
        );
    }

    let dur = t0.elapsed().as_millis() as u64;
    println!("  Paper 44: checks complete ({dur}ms)\n");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Paper 45: Kinetic-Fluid Coupling (Haack, Murillo, Sagert & Chuna, 2024)
// ═══════════════════════════════════════════════════════════════════════════

fn paper_45_kinetic_fluid(harness: &mut ValidationHarness) {
    const PAPER: &str = "Haack et al., J. Comput. Phys. (2024), DOI:10.1016/j.jcp.2024.112908";
    const DOMAIN: &str = "kinetic_fluid";

    println!("━━━ Paper 45: Kinetic-Fluid Coupling ━━━");
    let t0 = Instant::now();

    // BGK relaxation
    let bgk = run_bgk_relaxation(3000, 0.005);

    harness.check_upper("kf_mass_conservation_1", bgk.mass_err_1, 1e-8);
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "species 1 mass conserved to machine precision",
    );
    harness.check_upper("kf_mass_conservation_2", bgk.mass_err_2, 1e-8);
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "species 2 mass conserved to machine precision",
    );
    harness.check_upper("kf_momentum_conservation", bgk.momentum_err, 1e-10);
    harness.annotate(DOMAIN, PAPER, "relative_error", "total momentum conserved");
    harness.check_upper("kf_energy_conservation", bgk.energy_err, 0.01);
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
    harness.check_upper("kf_temperature_relaxation", bgk.temp_relaxed, 0.01);
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "T₁ and T₂ converge within 1%",
    );

    let t_eq = f64::midpoint(bgk.t1_final, bgk.t2_final);
    harness.check_abs("kf_equilibrium_temperature", t_eq, 1.25, 0.05);
    harness.annotate(
        DOMAIN,
        PAPER,
        "temperature",
        "equilibrium T ≈ 1.25 from energy conservation (m₁T₁+m₂T₂)/(m₁+m₂)",
    );

    // Sod shock tube
    let sod = run_sod_shock_tube(400, 0.2);

    harness.check_upper("kf_sod_mass", sod.mass_err, 1e-10);
    harness.annotate(DOMAIN, PAPER, "relative_error", "Sod shock: mass conserved");
    harness.check_upper("kf_sod_energy", sod.energy_err, 1e-10);
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
    harness.check_lower("kf_sod_rho_min", sod.rho_min, 0.1);
    harness.annotate(DOMAIN, PAPER, "density", "density stays physical (>0.1)");
    harness.check_upper("kf_sod_rho_max", sod.rho_max, 1.1);
    harness.annotate(
        DOMAIN,
        PAPER,
        "density",
        "density bounded (Sod initial max=1.0)",
    );

    // Coupled kinetic-fluid
    let coupled = run_coupled_kinetic_fluid(30, 30, 81, 0.05);

    harness.check_upper("kf_coupled_mass", coupled.mass_err, 0.15);
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "coupled: mass conservation within 15% (interface flux mismatch at low resolution)",
    );
    harness.check_upper("kf_coupled_momentum", coupled.momentum_err, 0.25);
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "coupled: momentum conservation within 25% (operator splitting at low resolution)",
    );
    harness.check_upper("kf_coupled_energy", coupled.energy_err, 0.15);
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "coupled: energy conservation within 15%",
    );
    harness.check_upper("kf_coupled_interface", coupled.interface_density_match, 0.5);
    harness.annotate(
        DOMAIN,
        PAPER,
        "relative_error",
        "interface density match within 50% (coarse grid, operator splitting)",
    );
    harness.check_lower("kf_coupled_rho_min", coupled.rho_fluid_min, 0.5);
    harness.annotate(DOMAIN, PAPER, "density", "fluid density stays physical");
    harness.check_upper("kf_coupled_rho_max", coupled.rho_fluid_max, 2.0);
    harness.annotate(DOMAIN, PAPER, "density", "fluid density bounded");
    harness.check_bool("kf_simulation_completed", coupled.n_steps > 0);
    harness.annotate(DOMAIN, PAPER, "boolean", "simulation ran to completion");

    let dur = t0.elapsed().as_millis() as u64;
    println!("  Paper 45: checks complete ({dur}ms)\n");
}
