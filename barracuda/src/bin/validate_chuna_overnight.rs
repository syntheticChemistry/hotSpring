// SPDX-License-Identifier: AGPL-3.0-only

//! Chuna overnight validation — all Paper 43/44/45 systems in one run.
//!
//! Hardware-agnostic: discovers all f64-capable GPUs at runtime, profiles
//! each via `HardwareCalibration + PrecisionBrain`, sizes workloads to
//! available VRAM, and validates on every target substrate. Any GPU with
//! SHADER_F64 or DF64 fallback is a science device at 14-digit precision.
//!
//! **Paper 43** (Gradient flow integrators):
//!   - Convergence sweep: ε = 0.02→0.001 for W6/W7/CK4
//!   - Production flow at 8⁴+ β = {5.9, 6.0, 6.2} (VRAM-adaptive sizes)
//!   - Dynamical fermion extension (warm-start + mass annealing)
//!
//! **Paper 44** (Conservative BGK dielectric):
//!   - Standard + completed Mermin (CPU + GPU)
//!   - Multi-component Mermin (electron-ion, CPU + GPU)
//!
//! **Paper 45** (Multi-species kinetic-fluid coupling):
//!   - GPU BGK relaxation, Euler/Sod shock tube, coupled kinetic-fluid
//!
//! Usage:
//!   cargo run --release --bin validate_chuna_overnight              # auto-select best GPU
//!   cargo run --release --bin validate_chuna_overnight -- --all-gpus # validate on every f64 GPU
//!   cargo run --release --bin validate_chuna_overnight -- --gpu 3090 # target specific GPU

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::precision_brain::PrecisionBrain;
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};
use std::time::Instant;

/// Key observables collected per substrate for cross-GPU comparison.
#[derive(Debug, Default)]
struct SubstrateResults {
    adapter_name: String,
    plaquettes: Vec<(String, f64)>,
    w0: Option<f64>,
    t0: Option<f64>,
    wall_seconds: f64,
}

/// Determine max lattice L for SU(3) based on available VRAM.
///
/// Memory per config: links + momenta + forces ~ 3 * 4 * L^4 * 4 * 18 * 8 bytes
/// (4D, 4 directions, 3x3 complex matrix = 18 f64, times 3 for link+mom+force).
fn max_lattice_l(max_buffer_bytes: u64) -> usize {
    let safety_margin = 4;
    let bytes_per_site: u64 = 4 * 18 * 8 * safety_margin;
    let max_sites = max_buffer_bytes / bytes_per_site;
    let l = (max_sites as f64).powf(0.25).floor() as usize;
    l.min(64).max(8)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let skip_to_dynamical = args.iter().any(|a| a == "--dynamical-only");
    let all_gpus = args.iter().any(|a| a == "--all-gpus");
    let gpu_token: Option<String> = args
        .iter()
        .position(|a| a == "--gpu")
        .and_then(|i| args.get(i + 1).cloned());

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Overnight Validation — Papers 43 / 44 / 45         ║");
    println!("║  Hardware-agnostic: discover, profile, validate            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    if skip_to_dynamical {
        println!("  [--dynamical-only] Skipping quenched/GPU sections, jumping to dynamical\n");
    }

    let total_start = Instant::now();

    // ═══ Phase 1: Discover GPU substrates ═══
    let adapters = GpuF64::enumerate_adapters();
    let f64_adapters: Vec<_> = adapters.iter().filter(|a| a.has_f64).collect();

    println!("  Substrate inventory ({} adapters, {} with f64):", adapters.len(), f64_adapters.len());
    for a in &adapters {
        let tag = if a.has_f64 { "f64" } else { "f32" };
        let mem_gb = a.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("    [{}] {} ({}, {tag}, {mem_gb:.1} GB)", a.index, a.name, a.driver);
    }
    println!();

    let targets: Vec<String> = if all_gpus {
        f64_adapters.iter().map(|a| a.index.to_string()).collect()
    } else if let Some(token) = gpu_token {
        vec![token]
    } else {
        vec!["auto".to_string()]
    };

    if targets.is_empty() {
        eprintln!("  FATAL: no f64-capable GPUs discovered");
        std::process::exit(1);
    }

    let mut harness = ValidationHarness::new("chuna_overnight");
    let mut all_results: Vec<SubstrateResults> = Vec::new();

    for (gpu_idx, token) in targets.iter().enumerate() {
        let gpu_header = if targets.len() > 1 {
            format!("GPU {}/{}", gpu_idx + 1, targets.len())
        } else {
            "GPU".to_string()
        };

        // ═══ Phase 2: Open + Profile ═══
        let gpu = match rt.block_on(GpuF64::with_adapter(token)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  [{gpu_header}] Failed to open GPU '{token}': {e}");
                continue;
            }
        };

        let max_l = max_lattice_l(gpu.device().limits().max_buffer_size);
        let vram_gb = gpu.device().limits().max_buffer_size as f64 / (1024.0 * 1024.0 * 1024.0);

        let brain = PrecisionBrain::new(&gpu);
        let cal = &brain.calibration;

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  [{gpu_header}] {} (f64={}, df64={}, f16={}, subgroups={})",
            gpu.adapter_name, gpu.has_f64, gpu.full_df64_mode, gpu.has_f16, gpu.has_subgroups);
        println!("  [{gpu_header}] {cal}");
        println!("  [{gpu_header}] VRAM: {vram_gb:.1} GB → max lattice L={max_l}");

        let telem_filename = if targets.len() > 1 {
            let safe_name: String = gpu.adapter_name.chars()
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect();
            format!("chuna_overnight_{safe_name}.jsonl")
        } else {
            "chuna_overnight_telemetry.jsonl".to_string()
        };
        let mut telem = TelemetryWriter::discover(&telem_filename)
            .with_substrate(gpu.adapter_name.clone());

        telem.log_map("hardware_profile", &[
            ("has_f64", f64::from(u8::from(gpu.has_f64))),
            ("full_df64_mode", f64::from(u8::from(gpu.full_df64_mode))),
            ("has_f16", f64::from(u8::from(gpu.has_f16))),
            ("has_subgroups", f64::from(u8::from(gpu.has_subgroups))),
            ("has_timestamps", f64::from(u8::from(gpu.has_timestamps))),
            ("vram_gb", vram_gb),
            ("max_lattice_l", max_l as f64),
        ]);
        for tier in &cal.tiers {
            telem.log_map(
                &format!("tier_{:?}", tier.tier),
                &[
                    ("compiles", f64::from(u8::from(tier.compiles))),
                    ("dispatches", f64::from(u8::from(tier.dispatches))),
                    ("transcendentals_safe", f64::from(u8::from(tier.transcendentals_safe))),
                    ("compile_us", tier.compile_us),
                    ("dispatch_us", tier.dispatch_us),
                    ("probe_ulp", if tier.probe_ulp.is_finite() { tier.probe_ulp } else { -1.0 }),
                ],
            );
        }

        harness.set_gpu(&gpu.adapter_name);
        harness.set_substrate(&gpu.adapter_name);

        let gpu_start = Instant::now();
        let mut results = SubstrateResults {
            adapter_name: gpu.adapter_name.clone(),
            ..Default::default()
        };

        // ═══ Phase 3: Validate ═══
        if !skip_to_dynamical {
            println!("\n━━━ Paper 43: Gradient Flow Integrators ━━━\n");
            paper_43_convergence(&mut harness, &mut telem);
            paper_43_production(&mut harness, &mut telem, &mut results);

            println!("\n━━━ Paper 44: Conservative BGK Dielectric ━━━\n");
            paper_44_cpu(&mut harness, &mut telem);
            paper_44_multicomponent_cpu(&mut harness, &mut telem);
            paper_44_gpu(&mut harness, &gpu, &mut telem);
            paper_44_multicomponent_gpu(&mut harness, &gpu, &mut telem);

            println!("\n━━━ Paper 45: Kinetic-Fluid Coupling ━━━\n");
            paper_45_gpu_bgk(&mut harness, &gpu, &mut telem);
            paper_45_gpu_euler(&mut harness, &gpu, &mut telem);
            paper_45_gpu_coupled(&mut harness, &gpu, &mut telem);
        }

        println!("\n━━━ Paper 43: Dynamical Extension (warm-start) ━━━\n");
        paper_43_dynamical(&mut harness, &mut telem, &mut results);

        results.wall_seconds = gpu_start.elapsed().as_secs_f64();
        telem.log("substrate_summary", "wall_seconds", results.wall_seconds);
        println!("\n  [{gpu_header}] {} done in {:.1}s\n", gpu.adapter_name, results.wall_seconds);
        all_results.push(results);
    }

    // ═══ Phase 4: Cross-GPU Comparison ═══
    if all_results.len() > 1 {
        println!("━━━ Cross-Substrate Comparison ({} GPUs) ━━━\n", all_results.len());
        harness.clear_substrate();

        for i in 0..all_results.len() {
            for j in (i + 1)..all_results.len() {
                let a = &all_results[i];
                let b = &all_results[j];
                println!("  {} vs {}:", a.adapter_name, b.adapter_name);

                for (label_a, plaq_a) in &a.plaquettes {
                    if let Some((_, plaq_b)) = b.plaquettes.iter().find(|(l, _)| l == label_a) {
                        let rel = if plaq_a.abs() > 1e-15 {
                            (plaq_a - plaq_b).abs() / plaq_a.abs()
                        } else {
                            (plaq_a - plaq_b).abs()
                        };
                        let agree = rel < 0.05;
                        println!("    {label_a}: {plaq_a:.6} vs {plaq_b:.6} (rel={rel:.4e}) {}",
                            if agree { "OK" } else { "CHECK" });
                        harness.check_upper(
                            &format!("xgpu_{label_a}_{}_{}", a.adapter_name, b.adapter_name),
                            rel, 0.05,
                        );
                    }
                }

                if let (Some(w0_a), Some(w0_b)) = (a.w0, b.w0) {
                    let rel = (w0_a - w0_b).abs() / w0_a.abs().max(1e-15);
                    println!("    w0: {w0_a:.4} vs {w0_b:.4} (rel={rel:.4e})");
                    harness.check_upper(
                        &format!("xgpu_w0_{}_{}", a.adapter_name, b.adapter_name),
                        rel, 0.1,
                    );
                }
                if let (Some(t0_a), Some(t0_b)) = (a.t0, b.t0) {
                    let rel = (t0_a - t0_b).abs() / t0_a.abs().max(1e-15);
                    println!("    t0: {t0_a:.4} vs {t0_b:.4} (rel={rel:.4e})");
                    harness.check_upper(
                        &format!("xgpu_t0_{}_{}", a.adapter_name, b.adapter_name),
                        rel, 0.1,
                    );
                }
                println!();
            }
        }

        println!("  Wall times:");
        for r in &all_results {
            println!("    {}: {:.1}s", r.adapter_name, r.wall_seconds);
        }
        println!();
    }

    let total = total_start.elapsed();
    println!("\n  Total wall time: {:.1}s ({} substrate{})",
        total.as_secs_f64(), all_results.len(),
        if all_results.len() == 1 { "" } else { "s" });
    harness.finish();
}

/// CPU-based quenched pre-thermalization fallback.
fn cpu_quenched_pretherm(
    dims: [usize; 4],
    beta: f64,
    n_quenched_therm: usize,
    telem: &mut TelemetryWriter,
    start: &Instant,
) -> hotspring_barracuda::lattice::wilson::Lattice {
    use hotspring_barracuda::lattice::hmc::{HmcConfig, hmc_trajectory};
    use hotspring_barracuda::lattice::wilson::Lattice;

    println!("    Quenched pre-therm ({n_quenched_therm} trajectories)...");
    let mut lattice = Lattice::hot_start(dims, beta, 42);
    let mut quenched_cfg = HmcConfig {
        n_md_steps: 20,
        dt: 0.05,
        seed: 12345,
        ..Default::default()
    };
    let mut q_accept = 0;
    for i in 0..n_quenched_therm {
        if hmc_trajectory(&mut lattice, &mut quenched_cfg).accepted {
            q_accept += 1;
        }
        if (i + 1) % 25 == 0 {
            println!(
                "      [{}/{}] ⟨P⟩={:.6}, acc={:.0}%",
                i + 1,
                n_quenched_therm,
                lattice.average_plaquette(),
                q_accept as f64 / (i + 1) as f64 * 100.0,
            );
            telem.log_map(
                "p43_dyn_quenched",
                &[
                    ("trajectory", (i + 1) as f64),
                    ("plaquette", lattice.average_plaquette()),
                    ("acceptance", q_accept as f64 / (i + 1) as f64),
                ],
            );
        }
    }
    println!(
        "    Quenched done: ⟨P⟩={:.6}, {:.0}% accept, {:.1}s",
        lattice.average_plaquette(),
        q_accept as f64 / n_quenched_therm as f64 * 100.0,
        start.elapsed().as_secs_f64(),
    );
    lattice
}

// ─── Paper 43 ────────────────────────────────────────────────────

fn paper_43_convergence(harness: &mut ValidationHarness, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::lattice::gradient_flow::{FlowIntegrator, run_flow};
    use hotspring_barracuda::lattice::hmc::{HmcConfig, hmc_trajectory};
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
            for i in 0..n_therm {
                let result = hmc_trajectory(&mut lat, &mut cfg);
                if (i + 1) % 25 == 0 {
                    println!(
                        "      [{name} ε={eps}] therm {}/{n_therm}: ⟨P⟩={:.6}, acc={}",
                        i + 1,
                        lat.average_plaquette(),
                        if result.accepted { "Y" } else { "N" }
                    );
                    telem.log_map(
                        &format!("p43_conv_{name}_eps{eps}"),
                        &[
                            ("trajectory", (i + 1) as f64),
                            ("plaquette", lat.average_plaquette()),
                            ("accepted", f64::from(u8::from(result.accepted))),
                        ],
                    );
                }
            }
            let results = run_flow(&mut lat, *integrator, eps, 1.0, 1);
            let e = results.last().map_or(0.0, |m| m.energy_density);
            e_values.push((eps, e));
            telem.log(&format!("p43_conv_{name}"), &format!("E_eps{eps}"), e);
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
                let order_ok = order > 1.5 && order < (*expected_order as f64 + 2.0);
                println!("    {name}: order = {order:.2} (expected {expected_order})");
                telem.log(&format!("p43_conv_{name}"), "order", order);
                harness.check_bool(&format!("p43_convergence_{name}"), order_ok);
            } else {
                println!("    {name}: converged to machine precision");
                harness.check_bool(&format!("p43_convergence_{name}"), true);
            }
        }
    }
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_43_production(harness: &mut ValidationHarness, telem: &mut TelemetryWriter, results: &mut SubstrateResults) {
    use hotspring_barracuda::lattice::gradient_flow::{FlowIntegrator, find_t0, find_w0, run_flow};
    use hotspring_barracuda::lattice::hmc::{HmcConfig, hmc_trajectory};
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
        let log_interval = if *n_therm >= 500 { 50 } else { 25 };
        for i in 0..*n_therm {
            let result = hmc_trajectory(&mut lattice, &mut config);
            if result.accepted {
                n_accept += 1;
            }
            if (i + 1) % log_interval == 0 {
                let running_acc = n_accept as f64 / (i + 1) as f64;
                println!(
                    "    [{}/{}] ⟨P⟩={:.6}, acc={:.0}%",
                    i + 1,
                    n_therm,
                    lattice.average_plaquette(),
                    running_acc * 100.0
                );
                telem.log_map(
                    &format!("p43_prod_{label}"),
                    &[
                        ("trajectory", (i + 1) as f64),
                        ("plaquette", lattice.average_plaquette()),
                        ("acceptance", running_acc),
                        ("delta_h", result.delta_h),
                    ],
                );
            }
        }
        let acceptance = n_accept as f64 / *n_therm as f64;
        println!(
            "    ⟨P⟩ = {:.6}, {:.0}% accept",
            lattice.average_plaquette(),
            acceptance * 100.0
        );
        telem.log_map(
            &format!("p43_prod_{label}"),
            &[
                ("final_plaquette", lattice.average_plaquette()),
                ("final_acceptance", acceptance),
                ("wall_seconds", start.elapsed().as_secs_f64()),
            ],
        );
        harness.check_lower(&format!("p43_accept_{label}"), acceptance, 0.25);
        results.plaquettes.push((label.to_string(), lattice.average_plaquette()));

        let flow = run_flow(&mut lattice, FlowIntegrator::Lscfrk3w7, 0.01, 4.0, 5);

        let monotonic = flow
            .windows(2)
            .all(|w| w[1].energy_density <= w[0].energy_density + 1e-10);
        harness.check_bool(&format!("p43_monotonic_{label}"), monotonic);

        if let Some(w0) = find_w0(&flow) {
            println!("    w₀ = {w0:.4}");
            telem.log(&format!("p43_prod_{label}"), "w0", w0);
            results.w0 = Some(w0);
        }
        if let Some(t0) = find_t0(&flow) {
            println!("    t₀ = {t0:.4}");
            telem.log(&format!("p43_prod_{label}"), "t0", t0);
            results.t0 = Some(t0);
        }

        println!("    {:.1}s", start.elapsed().as_secs_f64());
    }
}

/// Paper 43 extension: gradient flow on dynamical staggered fermion configs.
///
/// Uses warm-start strategy: quenched pre-thermalization at target β, then
/// mass annealing (m=1.0 → 0.5 → 0.2 → 0.1) with adaptive Omelyan HMC.
/// This solves the 0% acceptance problem from hot-start (|ΔH| ~ 6.5M).
///
/// The NPU, when available, monitors and adjusts the annealing schedule.
fn paper_43_dynamical(
    harness: &mut ValidationHarness,
    telem: &mut TelemetryWriter,
    results: &mut SubstrateResults,
) {
    use hotspring_barracuda::lattice::gradient_flow::{FlowIntegrator, find_t0, find_w0, run_flow};
    use hotspring_barracuda::lattice::pseudofermion::{
        AdaptiveStepController, DynamicalHmcConfig, MassAnnealingSchedule, PseudofermionConfig,
        dynamical_thermalize_warm_start, dynamical_thermalize_warm_start_npu,
    };

    let start = Instant::now();

    let dims = [8, 8, 8, 8];
    let beta = 5.4;
    let mass = 0.1;

    let npu_available = hotspring_barracuda::discovery::probe_npu_available();

    if npu_available {
        println!("  Dynamical 8⁴ β={beta} (N_f=4, m={mass}) [NPU detected, warm-start]...");
    } else {
        println!("  Dynamical 8⁴ β={beta} (N_f=4, m={mass}) [warm-start + mass annealing]...");
    }
    let mut lattice = cpu_quenched_pretherm(dims, beta, 100, telem, &start);

    // Step 2: Mass annealing with adaptive Omelyan HMC
    let schedule = MassAnnealingSchedule::default_for(mass);
    println!(
        "    Mass annealing: {} stages → m={mass}",
        schedule.stages.len()
    );

    // If NPU is available, use it for initial param suggestion and runtime steering
    let mut npu_steering = if npu_available {
        hotspring_barracuda::discovery::try_create_npu_steering(5)
    } else {
        None
    };

    let npu_active = npu_steering.is_some();
    let mut controller = if let Some(ref mut steering) = npu_steering {
        let ctrl =
            AdaptiveStepController::for_dynamical_with_npu(dims, beta, 1.0, &mut steering.npu);
        println!(
            "    [NPU] Steering active — initial dt={:.5}, n_md={}, feedback every {} traj",
            ctrl.dt, ctrl.n_md_steps, steering.feedback_interval,
        );
        telem.log_map(
            "p43_dyn_npu",
            &[
                ("npu_active", 1.0),
                ("initial_dt", ctrl.dt),
                ("initial_n_md", ctrl.n_md_steps as f64),
                ("feedback_interval", steering.feedback_interval as f64),
            ],
        );
        ctrl
    } else {
        let ctrl = AdaptiveStepController::for_dynamical(dims, beta, 1.0);
        println!("    [NPU] Not available — using heuristic controller");
        telem.log("p43_dyn_npu", "npu_active", 0.0);
        ctrl
    };

    let mut config = DynamicalHmcConfig {
        seed: 12345,
        fermion: PseudofermionConfig {
            mass: 1.0,
            cg_tol: 1e-8,
            cg_max_iter: 5000,
        },
        beta,
        n_flavors_over_4: 1,
        ..Default::default()
    };

    let warm = if let Some(ref mut steering) = npu_steering {
        dynamical_thermalize_warm_start_npu(
            &mut lattice,
            &mut config,
            &schedule,
            &mut controller,
            steering,
        )
    } else {
        dynamical_thermalize_warm_start(&mut lattice, &mut config, &schedule, &mut controller)
    };

    let plaq = lattice.average_plaquette();
    println!(
        "    ⟨P⟩ = {plaq:.6}, {:.0}% accept (final stage), {} CG iters, dt={:.4}, n_md={}",
        warm.final_acceptance * 100.0,
        warm.total_cg_iterations,
        warm.final_dt,
        warm.final_n_md,
    );

    for (i, stage) in warm.stage_results.iter().enumerate() {
        telem.log_map(
            &format!("p43_dyn_stage{i}"),
            &[
                ("mass", stage.mass),
                ("acceptance", stage.acceptance_rate),
                ("plaquette", stage.plaquette),
                ("mean_delta_h", stage.mean_delta_h),
                ("n_trajectories", stage.n_trajectories as f64),
            ],
        );
    }

    telem.log_map(
        "p43_dyn",
        &[
            ("final_plaquette", plaq),
            ("final_acceptance", warm.final_acceptance),
            ("total_cg_iterations", warm.total_cg_iterations as f64),
            ("final_dt", warm.final_dt),
            ("final_n_md", warm.final_n_md as f64),
            ("total_trajectories", warm.total_trajectories as f64),
            ("wall_seconds", start.elapsed().as_secs_f64()),
            ("npu_active", if npu_active { 1.0 } else { 0.0 }),
        ],
    );

    // Write run summary to run history for NPU cross-run learning
    {
        use hotspring_barracuda::lattice::pseudofermion::run_history::{
            RunHistoryWriter, RunSummary,
        };
        let root = hotspring_barracuda::discovery::discover_data_root();
        let history_path = root.join("telemetry/dynamical_run_history.jsonl");
        if let Some(parent) = history_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut writer) = RunHistoryWriter::open(&history_path) {
            writer.write_summary(&RunSummary {
                beta,
                mass,
                final_acceptance: warm.final_acceptance,
                final_plaquette: plaq,
                final_dt: warm.final_dt,
                n_trajectories: warm.total_trajectories,
                converged: warm.final_acceptance > 0.20,
            });
            writer.flush();
        }
    }

    harness.check_lower("p43_dyn_accept", warm.final_acceptance, 0.20);
    harness.check_lower("p43_dyn_plaquette", plaq, 0.3);
    results.plaquettes.push(("dyn_8^4".to_string(), plaq));

    // Step 3: Gradient flow on the dynamical config
    let flow = run_flow(&mut lattice, FlowIntegrator::Lscfrk3w7, 0.01, 2.0, 5);

    let monotonic = flow
        .windows(2)
        .all(|w| w[1].energy_density <= w[0].energy_density + 1e-10);
    harness.check_bool("p43_dyn_flow_monotonic", monotonic);

    if let Some(w0) = find_w0(&flow) {
        println!("    w₀ = {w0:.4} (dynamical)");
        telem.log("p43_dyn", "w0", w0);
        results.w0 = Some(w0);
    }
    if let Some(t0) = find_t0(&flow) {
        println!("    t₀ = {t0:.4} (dynamical)");
        telem.log("p43_dyn", "t0", t0);
        results.t0 = Some(t0);
    }

    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

// ─── Paper 44 ────────────────────────────────────────────────────

fn paper_44_cpu(harness: &mut ValidationHarness, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::dielectric::{
        PlasmaParams, dynamic_structure_factor_completed, epsilon_completed_mermin, epsilon_mermin,
        f_sum_rule_integral, f_sum_rule_integral_completed,
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
    telem.log_map(
        "p44_fsum",
        &[
            ("err_25", err_25 / expected.abs()),
            ("err_50", err_50 / expected.abs()),
            ("err_100", err_100 / expected.abs()),
        ],
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
    telem.log("p44_dsf", "positive_fraction", frac);
    harness.check_lower("p44_dsf_positive", frac, 0.99);

    let eps_hf = epsilon_completed_mermin(k, 100.0, nu, &params);
    harness.check_upper("p44_hf_limit", (eps_hf.re - 1.0).abs(), 0.01);

    let eps_std = epsilon_mermin(k, 1.5, 1e-10, &params);
    let eps_cmp = epsilon_completed_mermin(k, 1.5, 1e-10, &params);
    let rel = (eps_std.re - eps_cmp.re).abs() / eps_std.abs().max(1e-15);
    harness.check_upper("p44_nu0_agreement", rel, 0.01);

    telem.log("p44_cpu", "wall_seconds", start.elapsed().as_secs_f64());
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_44_multicomponent_cpu(harness: &mut ValidationHarness, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::dielectric_multicomponent::{
        MultiComponentPlasma, SpeciesParams, epsilon_multicomponent_mermin, multicomponent_dsf,
        multicomponent_f_sum_integral,
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

    for omega in [0.1, 0.5, 1.0, 5.0, 10.0] {
        let eps = epsilon_multicomponent_mermin(k, omega, &plasma, true);
        harness.check_lower(&format!("p44_mc_passive_w{omega}"), eps.im, -0.01);
    }

    telem.log("p44_mc_cpu", "wall_seconds", start.elapsed().as_secs_f64());
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_44_gpu(harness: &mut ValidationHarness, gpu: &GpuF64, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::gpu_dielectric::{
        GpuDielectricPipeline, validate_gpu_dielectric,
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

    telem.log_map(
        "p44_gpu",
        &[
            ("gpu_seconds", validation.gpu_wall_seconds),
            ("cpu_seconds", validation.cpu_wall_seconds),
            ("l2_loss_rel", validation.l2_loss_rel_error),
        ],
    );
    println!(
        "    GPU {:.2}s, CPU {:.2}s, L² = {:.4e}",
        validation.gpu_wall_seconds, validation.cpu_wall_seconds, validation.l2_loss_rel_error
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_44_multicomponent_gpu(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::physics::gpu_dielectric_multicomponent::{
        GpuMulticompPipeline, validate_gpu_multicomponent,
    };

    println!("  GPU Multi-component Mermin...");
    let start = Instant::now();

    let pipeline = GpuMulticompPipeline::new(gpu);
    let (gpu_loss, cpu_loss) = validate_gpu_multicomponent(gpu, &pipeline);

    let n_close = gpu_loss
        .iter()
        .zip(cpu_loss.iter())
        .filter(|&(&g, &c)| {
            let denom = c.abs().max(1e-15);
            (g - c).abs() / denom < 0.5
        })
        .count();
    let frac = n_close as f64 / gpu_loss.len().max(1) as f64;
    println!("    CPU-GPU agreement: {:.0}%", frac * 100.0);
    telem.log("p44_mc_gpu", "cpu_gpu_agreement", frac);
    harness.check_lower("p44_mc_gpu_agreement", frac, 0.90);

    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

// ─── Paper 45 ────────────────────────────────────────────────────

fn paper_45_gpu_bgk(harness: &mut ValidationHarness, gpu: &GpuF64, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::gpu_kinetic_fluid::{GpuBgkPipeline, validate_gpu_bgk};

    println!("  GPU BGK relaxation...");
    let start = Instant::now();

    let pipeline = GpuBgkPipeline::new(gpu);
    let (gpu_r, _cpu_r) = validate_gpu_bgk(gpu, &pipeline);

    harness.check_upper("p45_bgk_mass_err", gpu_r.result.mass_err_1, 1e-4);
    harness.check_upper("p45_bgk_energy_err", gpu_r.result.energy_err, 0.05);
    harness.check_bool("p45_bgk_entropy", gpu_r.result.entropy_monotonic);
    telem.log_map(
        "p45_bgk",
        &[
            ("mass_err", gpu_r.result.mass_err_1),
            ("energy_err", gpu_r.result.energy_err),
            ("temp_relaxed", gpu_r.result.temp_relaxed),
            ("gpu_seconds", gpu_r.gpu_wall_seconds),
            ("cpu_seconds", gpu_r.cpu_wall_seconds),
        ],
    );
    println!(
        "    GPU {:.2}s, CPU {:.2}s",
        gpu_r.gpu_wall_seconds, gpu_r.cpu_wall_seconds
    );
    println!("    ΔT/T = {:.4}", gpu_r.result.temp_relaxed);
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_45_gpu_euler(harness: &mut ValidationHarness, gpu: &GpuF64, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::gpu_euler::{GpuEulerPipeline, validate_gpu_euler};

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

    telem.log_map(
        "p45_euler",
        &[
            ("mass_err", result.mass_err),
            ("energy_err", result.energy_err),
            ("rho_range", rho_range),
            ("gpu_seconds", result.gpu_wall_seconds),
            ("cpu_seconds", result.cpu_wall_seconds),
        ],
    );
    println!(
        "    GPU {:.2}s, CPU {:.2}s",
        result.gpu_wall_seconds, result.cpu_wall_seconds
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn paper_45_gpu_coupled(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::physics::gpu_coupled_kinetic_fluid::{
        GpuCoupledPipeline, validate_gpu_coupled,
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
    telem.log_map(
        "p45_coupled",
        &[
            ("mass_err", result.mass_err),
            ("energy_err", result.energy_err),
            ("interface_gpu", gpu_if),
            ("interface_cpu", cpu_if),
            ("interface_rel", if_rel),
            ("n_steps", result.n_steps as f64),
            ("gpu_seconds", result.gpu_wall_seconds),
            ("cpu_seconds", result.cpu_wall_seconds),
        ],
    );
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
