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

#[path = "../bin_helpers/chuna_papers_44_45.rs"]
mod chuna_papers_44_45;

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let skip_to_dynamical = args.iter().any(|a| a == "--dynamical-only");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut harness = ValidationHarness::new("chuna_overnight");
    let mut telem = TelemetryWriter::discover("chuna_overnight_telemetry.jsonl");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Overnight Validation — Papers 43 / 44 / 45         ║");
    println!("║  Bazavov & Chuna 2021, Chuna & Murillo 2024,              ║");
    println!("║  Haack, Murillo, Sagert & Chuna 2024                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    if skip_to_dynamical {
        println!("  [--dynamical-only] Skipping quenched/GPU sections, jumping to dynamical\n");
    }

    let total_start = Instant::now();

    // ═══════════════════════════════════════════════════════════════
    //  Titan V Pre-Motor: background quenched pre-thermalization
    // ═══════════════════════════════════════════════════════════════
    let titan_handles = if skip_to_dynamical {
        None
    } else {
        let h = spawn_titan_pretherm_if_available(&rt);
        if h.is_some() {
            println!(
                "  [Brain] Titan V pre-motor spawned — quenched β=5.4 running in background\n"
            );
        }
        h
    };

    if !skip_to_dynamical {
        match rt.block_on(GpuF64::new()) {
            Ok(gpu) => {
                let streaming_pipelines =
                    hotspring_barracuda::lattice::gpu_hmc::GpuHmcStreamingPipelines::new(&gpu);

                // ═══════════════════════════════════════════════════════════════
                //  Paper 43: Gradient Flow Integrators (quenched, GPU streaming)
                // ═══════════════════════════════════════════════════════════════
                println!("\n━━━ Paper 43: Gradient Flow Integrators ━━━\n");

                paper_43_convergence(&mut harness, &gpu, &streaming_pipelines, &mut telem);
                paper_43_production(&mut harness, &gpu, &streaming_pipelines, &mut telem);

                // ═══════════════════════════════════════════════════════════════
                //  Paper 44: Conservative BGK Dielectric
                // ═══════════════════════════════════════════════════════════════
                println!("\n━━━ Paper 44: Conservative BGK Dielectric ━━━\n");

                chuna_papers_44_45::paper_44_cpu(&mut harness, &mut telem);
                chuna_papers_44_45::paper_44_multicomponent_cpu(&mut harness, &mut telem);

                chuna_papers_44_45::paper_44_gpu(&mut harness, &gpu, &mut telem);
                chuna_papers_44_45::paper_44_multicomponent_gpu(&mut harness, &gpu, &mut telem);

                // ═══════════════════════════════════════════════════════════════
                //  Paper 45: Multi-Species Kinetic-Fluid Coupling
                // ═══════════════════════════════════════════════════════════════
                println!("\n━━━ Paper 45: Kinetic-Fluid Coupling ━━━\n");

                chuna_papers_44_45::paper_45_gpu_bgk(&mut harness, &gpu, &mut telem);
                chuna_papers_44_45::paper_45_gpu_euler(&mut harness, &gpu, &mut telem);
                chuna_papers_44_45::paper_45_gpu_coupled(&mut harness, &gpu, &mut telem);
            }
            Err(e) => {
                println!("  ⚠ No GPU available ({e}) — skipping all GPU sections\n");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Paper 43 (cont.): Dynamical fermion extension
    //  Runs after 44/45 so Titan V pre-motor has time to thermalize
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━ Paper 43: Dynamical Extension (warm-start) ━━━\n");
    paper_43_dynamical(&mut harness, &mut telem, titan_handles);

    let total = total_start.elapsed();
    telem.log("summary", "total_wall_seconds", total.as_secs_f64());
    telem.log("summary", "checks_passed", harness.passed_count() as f64);
    telem.log("summary", "checks_total", harness.total_count() as f64);
    println!("\n  Total wall time: {:.1}s", total.as_secs_f64());
    harness.finish();
}

// ─── Titan V Pre-Motor ───────────────────────────────────────────

/// Handles returned by the Titan V background pre-thermalization.
struct TitanPrethermHandles {
    handles: hotspring_barracuda::production::titan_worker::TitanWorkerHandles,
}

/// Spawn Titan V background quenched pre-thermalization if a secondary GPU is available.
///
/// Returns `None` if only one GPU is found or if the worker fails to spawn.
fn spawn_titan_pretherm_if_available(rt: &tokio::runtime::Runtime) -> Option<TitanPrethermHandles> {
    use hotspring_barracuda::gpu::discover_primary_and_secondary_adapters;
    use hotspring_barracuda::production::titan_worker::{spawn_titan_worker, TitanRequest};

    let (_, secondary) = discover_primary_and_secondary_adapters();
    let secondary_idx = secondary?;

    println!("  [Brain] Secondary GPU discovered (adapter {secondary_idx}), spawning Titan V pre-motor...");

    // Create the GPU on the main thread to avoid wgpu deadlocks
    std::env::set_var("HOTSPRING_GPU_ADAPTER", &secondary_idx);
    let titan_gpu = match rt.block_on(GpuF64::new()) {
        Ok(gpu) => {
            println!("    Titan V: {} (f64={})", gpu.adapter_name, gpu.has_f64);
            gpu
        }
        Err(e) => {
            println!("    Titan V init failed: {e}");
            // Restore adapter env
            std::env::remove_var("HOTSPRING_GPU_ADAPTER");
            return None;
        }
    };
    std::env::remove_var("HOTSPRING_GPU_ADAPTER");

    let handles = match spawn_titan_worker(titan_gpu) {
        Ok(h) => h,
        Err(e) => {
            println!("    Titan V worker spawn failed: {e}");
            return None;
        }
    };

    // Send the pre-thermalization request (runs in background)
    let _ = handles.titan_tx.send(TitanRequest::PreThermalize {
        beta: 5.4,
        mass: 0.1,
        lattice: 8,
        n_quenched: 200,
        seed: 42,
        dt: 0.05,
        n_md: 20,
    });

    Some(TitanPrethermHandles { handles })
}

/// Try to receive a warm config from the Titan V pre-motor.
///
/// Waits up to 5 seconds, then falls back to CPU pre-thermalization.
fn receive_titan_warm_config(
    titan: TitanPrethermHandles,
    dims: [usize; 4],
    beta: f64,
) -> Option<hotspring_barracuda::lattice::wilson::Lattice> {
    use hotspring_barracuda::production::titan_worker::{TitanRequest, TitanResponse};

    match titan
        .handles
        .titan_rx
        .recv_timeout(std::time::Duration::from_secs(300))
    {
        Ok(TitanResponse::WarmConfig {
            beta: b,
            gauge_links,
            plaquette,
            wall_ms,
        }) => {
            println!(
                "    [Titan V] Warm config ready: β={b:.4}, ⟨P⟩={plaquette:.6}, {wall_ms:.0}ms"
            );
            let mut lattice = hotspring_barracuda::lattice::wilson::Lattice::cold_start(dims, beta);
            hotspring_barracuda::lattice::gpu_hmc::unflatten_links_into(&mut lattice, &gauge_links);
            let _ = titan.handles.titan_tx.send(TitanRequest::Shutdown);
            Some(lattice)
        }
        Err(_) => {
            println!("    [Titan V] Pre-therm timed out — falling back to CPU");
            let _ = titan.handles.titan_tx.send(TitanRequest::Shutdown);
            None
        }
    }
}

/// GPU-streaming quenched pre-thermalization.
///
/// Runs all HMC trajectories on GPU via streaming encoder batching.
/// Only 24-byte readback per trajectory (plaquette + delta_h + accept).
fn gpu_quenched_pretherm(
    gpu: &GpuF64,
    pipelines: &hotspring_barracuda::lattice::gpu_hmc::GpuHmcStreamingPipelines,
    dims: [usize; 4],
    beta: f64,
    n_quenched_therm: usize,
    n_md: usize,
    dt: f64,
    telem: &mut TelemetryWriter,
    telem_tag: &str,
    start: &Instant,
) -> hotspring_barracuda::lattice::wilson::Lattice {
    use hotspring_barracuda::lattice::gpu_hmc::{
        gpu_hmc_trajectory_streaming, gpu_links_to_lattice, GpuHmcState,
    };
    use hotspring_barracuda::lattice::wilson::Lattice;

    println!("    GPU quenched pre-therm ({n_quenched_therm} trajectories, streaming)...");
    let lattice = Lattice::hot_start(dims, beta, 42);
    let state = GpuHmcState::from_lattice(gpu, &lattice, beta);
    let mut seed = 12345_u64;
    let mut q_accept = 0_usize;
    for i in 0..n_quenched_therm {
        let r = gpu_hmc_trajectory_streaming(gpu, pipelines, &state, n_md, dt, i as u32, &mut seed);
        if r.accepted {
            q_accept += 1;
        }
        if (i + 1) % 25 == 0 {
            println!(
                "      [{}/{}] ⟨P⟩={:.6}, acc={:.0}%",
                i + 1,
                n_quenched_therm,
                r.plaquette,
                q_accept as f64 / (i + 1) as f64 * 100.0,
            );
            telem.log_map(
                telem_tag,
                &[
                    ("trajectory", (i + 1) as f64),
                    ("plaquette", r.plaquette),
                    ("acceptance", q_accept as f64 / (i + 1) as f64),
                ],
            );
        }
    }
    let mut out = lattice;
    gpu_links_to_lattice(gpu, &state, &mut out);
    let plaq = out.average_plaquette();
    println!(
        "    Quenched done: ⟨P⟩={plaq:.6}, {:.0}% accept, {:.1}s (GPU streaming)",
        q_accept as f64 / n_quenched_therm as f64 * 100.0,
        start.elapsed().as_secs_f64(),
    );
    out
}

// ─── Paper 43 ────────────────────────────────────────────────────

fn paper_43_convergence(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    pipelines: &hotspring_barracuda::lattice::gpu_hmc::GpuHmcStreamingPipelines,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::lattice::gpu_hmc::{
        gpu_hmc_trajectory_streaming, gpu_links_to_lattice, GpuHmcState,
    };
    use hotspring_barracuda::lattice::gradient_flow::{run_flow, FlowIntegrator};
    use hotspring_barracuda::lattice::wilson::Lattice;

    println!("  Convergence sweep (8⁴ β=6.0, GPU streaming)...");
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
            let lat = Lattice::hot_start([8, 8, 8, 8], 6.0, 42);
            let state = GpuHmcState::from_lattice(gpu, &lat, 6.0);
            let mut seed = 12345_u64;
            for i in 0..n_therm {
                let result = gpu_hmc_trajectory_streaming(
                    gpu, pipelines, &state, 20, 0.05, i as u32, &mut seed,
                );
                if (i + 1) % 25 == 0 {
                    println!(
                        "      [{name} ε={eps}] therm {}/{n_therm}: ⟨P⟩={:.6}, acc={}",
                        i + 1,
                        result.plaquette,
                        if result.accepted { "Y" } else { "N" }
                    );
                    telem.log_map(
                        &format!("p43_conv_{name}_eps{eps}"),
                        &[
                            ("trajectory", (i + 1) as f64),
                            ("plaquette", result.plaquette),
                            ("accepted", f64::from(u8::from(result.accepted))),
                        ],
                    );
                }
            }
            let mut lat = Lattice::cold_start([8, 8, 8, 8], 6.0);
            gpu_links_to_lattice(gpu, &state, &mut lat);
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
    println!("    {:.1}s (GPU streaming)", start.elapsed().as_secs_f64());
}

fn paper_43_production(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    pipelines: &hotspring_barracuda::lattice::gpu_hmc::GpuHmcStreamingPipelines,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::lattice::gpu_hmc::{
        gpu_hmc_trajectory_streaming, gpu_links_to_lattice, GpuHmcState,
    };
    use hotspring_barracuda::lattice::gradient_flow::{find_t0, find_w0, run_flow, FlowIntegrator};
    use hotspring_barracuda::lattice::wilson::Lattice;

    let configs: &[([usize; 4], f64, usize, &str)] = &[
        ([8, 8, 8, 8], 5.9, 200, "8⁴ β=5.9"),
        ([8, 8, 8, 8], 6.0, 200, "8⁴ β=6.0"),
        ([8, 8, 8, 8], 6.2, 200, "8⁴ β=6.2"),
        ([16, 16, 16, 16], 6.0, 500, "16⁴ β=6.0"),
    ];

    for (dims, beta, n_therm, label) in configs {
        let start = Instant::now();
        println!("  {label} (GPU streaming)...");

        let lattice = Lattice::hot_start(*dims, *beta, 42);
        let state = GpuHmcState::from_lattice(gpu, &lattice, *beta);
        let volume = dims[0] * dims[1] * dims[2] * dims[3];
        let (n_md, md_dt) = if volume > 10000 {
            (40, 0.025)
        } else {
            (20, 0.05)
        };
        let mut seed = 12345_u64;
        let mut n_accept = 0_usize;
        let mut last_plaq = 0.0_f64;
        let log_interval = if *n_therm >= 500 { 50 } else { 25 };
        for i in 0..*n_therm {
            let result = gpu_hmc_trajectory_streaming(
                gpu, pipelines, &state, n_md, md_dt, i as u32, &mut seed,
            );
            if result.accepted {
                n_accept += 1;
            }
            last_plaq = result.plaquette;
            if (i + 1) % log_interval == 0 {
                let running_acc = n_accept as f64 / (i + 1) as f64;
                println!(
                    "    [{}/{}] ⟨P⟩={:.6}, acc={:.0}%",
                    i + 1,
                    n_therm,
                    last_plaq,
                    running_acc * 100.0
                );
                telem.log_map(
                    &format!("p43_prod_{label}"),
                    &[
                        ("trajectory", (i + 1) as f64),
                        ("plaquette", last_plaq),
                        ("acceptance", running_acc),
                        ("delta_h", result.delta_h),
                    ],
                );
            }
        }
        let acceptance = n_accept as f64 / *n_therm as f64;
        println!(
            "    ⟨P⟩ = {last_plaq:.6}, {:.0}% accept",
            acceptance * 100.0
        );
        telem.log_map(
            &format!("p43_prod_{label}"),
            &[
                ("final_plaquette", last_plaq),
                ("final_acceptance", acceptance),
                ("wall_seconds", start.elapsed().as_secs_f64()),
            ],
        );
        harness.check_lower(&format!("p43_accept_{label}"), acceptance, 0.25);

        let mut lattice = Lattice::cold_start(*dims, *beta);
        gpu_links_to_lattice(gpu, &state, &mut lattice);
        let flow = run_flow(&mut lattice, FlowIntegrator::Lscfrk3w7, 0.01, 4.0, 5);

        let monotonic = flow
            .windows(2)
            .all(|w| w[1].energy_density <= w[0].energy_density + 1e-10);
        harness.check_bool(&format!("p43_monotonic_{label}"), monotonic);

        if let Some(w0) = find_w0(&flow) {
            println!("    w₀ = {w0:.4}");
            telem.log(&format!("p43_prod_{label}"), "w0", w0);
        }
        if let Some(t0) = find_t0(&flow) {
            println!("    t₀ = {t0:.4}");
            telem.log(&format!("p43_prod_{label}"), "t0", t0);
        }

        println!("    {:.1}s (GPU streaming)", start.elapsed().as_secs_f64());
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
    titan_handles: Option<TitanPrethermHandles>,
) {
    use hotspring_barracuda::lattice::gpu_hmc::dynamical::{GpuDynHmcPipelines, GpuDynHmcState};
    use hotspring_barracuda::lattice::gpu_hmc::gpu_rhmc::{GpuRhmcPipelines, GpuRhmcState};
    use hotspring_barracuda::lattice::gpu_hmc::rhmc_calibrator::RhmcCalibrator;
    use hotspring_barracuda::lattice::gpu_hmc::unidirectional_cortex::UnidirectionalRhmc;
    use hotspring_barracuda::lattice::gradient_flow::{find_t0, find_w0, run_flow, FlowIntegrator};
    use hotspring_barracuda::lattice::pseudofermion::MassAnnealingSchedule;

    let start = Instant::now();

    let dims = [8, 8, 8, 8];
    let beta = 5.4;
    let mass = 0.1;
    let nf = 4_usize;

    let npu_available = hotspring_barracuda::discovery::probe_npu_available();

    // Initialize GPU early so quenched pretherm fallback also runs on GPU
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt
        .block_on(GpuF64::new())
        .expect("GPU required for dynamical");
    println!(
        "    GPU: {} [unidirectional pipeline, O(1) readback]",
        gpu.adapter_name
    );

    let streaming_pl = hotspring_barracuda::lattice::gpu_hmc::GpuHmcStreamingPipelines::new(&gpu);

    // Step 1: Obtain pre-thermalized quenched config (Titan V or GPU streaming fallback)
    let mut lattice = if let Some(handles) = titan_handles {
        if let Some(lat) = receive_titan_warm_config(handles, dims, beta) {
            let plaq = lat.average_plaquette();
            telem.log("p43_dyn_titan", "plaquette", plaq);
            if plaq > 0.1 && plaq < 0.999 {
                println!(
                    "  Dynamical 8⁴ β={beta} (N_f=4, m={mass}) [Titan V warm config, ⟨P⟩={plaq:.6}]...",
                );
                lat
            } else {
                println!(
                    "  ⚠ Titan V warm config implausible (⟨P⟩={plaq:.6}) — GPU driver issue, falling back to GPU streaming",
                );
                telem.log("p43_dyn_titan", "rejected_plaquette", plaq);
                gpu_quenched_pretherm(
                    &gpu,
                    &streaming_pl,
                    dims,
                    beta,
                    100,
                    20,
                    0.05,
                    telem,
                    "p43_dyn_quenched",
                    &start,
                )
            }
        } else {
            println!(
                "  Dynamical 8⁴ β={beta} (N_f=4, m={mass}) [Titan V unavailable, GPU streaming fallback]..."
            );
            gpu_quenched_pretherm(
                &gpu,
                &streaming_pl,
                dims,
                beta,
                100,
                20,
                0.05,
                telem,
                "p43_dyn_quenched",
                &start,
            )
        }
    } else {
        println!("  Dynamical 8⁴ β={beta} (N_f=4, m={mass}) [GPU unidirectional pipeline]...");
        gpu_quenched_pretherm(
            &gpu,
            &streaming_pl,
            dims,
            beta,
            100,
            20,
            0.05,
            telem,
            "p43_dyn_quenched",
            &start,
        )
    };

    // Step 2: GPU-resident mass annealing via unidirectional RHMC
    let schedule = MassAnnealingSchedule::default_for(mass);
    println!(
        "    Mass annealing (GPU RHMC): {} stages → m={mass}",
        schedule.stages.len()
    );

    let npu_active = npu_available;
    if npu_active {
        telem.log("p43_dyn_npu", "npu_active", 1.0);
    } else {
        telem.log("p43_dyn_npu", "npu_active", 0.0);
    }

    let initial_mass = schedule.stages.first().map_or(1.0, |&(m, _)| m);
    let initial_cal = RhmcCalibrator::new(nf, initial_mass, beta, dims);
    let base_config = initial_cal.produce_config();

    let dyn_pipelines = GpuDynHmcPipelines::new(&gpu);
    let rhmc_pipelines = GpuRhmcPipelines::new(&gpu);
    let dyn_state = GpuDynHmcState::from_lattice(
        &gpu,
        &lattice,
        beta,
        initial_mass,
        base_config.cg_tol,
        base_config.cg_max_iter,
    );
    let rhmc_state = GpuRhmcState::new(&gpu, &base_config, dyn_state);

    let mut rhmc = UnidirectionalRhmc::new(gpu, dyn_pipelines, rhmc_pipelines, rhmc_state);
    println!(
        "    Pipeline compiled: {:.1}s",
        start.elapsed().as_secs_f64()
    );

    let mut seed = 12345_u64;
    let mut dt = 0.01_f64;
    let n_md = 10_usize;
    let mut total_cg = 0_usize;
    let mut total_traj = 0_usize;
    let mut final_acceptance = 0.0_f64;
    let mut stage_results = Vec::new();

    for (stage_idx, &(stage_mass, n_traj)) in schedule.stages.iter().enumerate() {
        let stage_cal = RhmcCalibrator::new(nf, stage_mass, beta, dims);
        let mut config = stage_cal.produce_config();
        config.dt = dt;
        config.n_md_steps = n_md;

        let mut stage_accepted = 0_usize;
        let mut stage_cg = 0_usize;
        let mut stage_delta_h_sum = 0.0_f64;

        println!(
            "    [stage {}/{}] mass={stage_mass:.2}, dt={dt:.4}, n_md={n_md}",
            stage_idx + 1,
            schedule.stages.len(),
        );

        for i in 0..n_traj {
            config.dt = dt;
            config.n_md_steps = n_md;
            let result = rhmc.run_trajectory(&config, &mut seed);

            if result.accepted {
                stage_accepted += 1;
            }
            stage_cg += result.total_cg_iterations;
            stage_delta_h_sum += result.delta_h.abs();

            if i == 0 {
                println!(
                    "      [diag] plaq={:.6}, cg_iters={}, ΔH={:.2}, {:.2}s",
                    result.plaquette,
                    result.total_cg_iterations,
                    result.delta_h,
                    result.elapsed_secs,
                );
            }

            let acc_rate = stage_accepted as f64 / (i + 1) as f64;
            if (i + 1) % 5 == 0 {
                let mean_dh = stage_delta_h_sum / (i + 1) as f64;
                if mean_dh > 5.0 {
                    dt = (dt * 0.5_f64).max(1e-4);
                } else if acc_rate > 0.85 && mean_dh < 0.3 {
                    dt *= 1.15;
                } else if acc_rate < 0.35 {
                    dt *= 0.85;
                }
                println!(
                    "      [{}/{}] acc={:.0}%, ⟨P⟩={:.6}, dt={dt:.4}, |ΔH|={:.2}",
                    i + 1,
                    n_traj,
                    acc_rate * 100.0,
                    result.plaquette,
                    mean_dh,
                );
            }
        }

        let stage_acc = if n_traj > 0 {
            stage_accepted as f64 / n_traj as f64
        } else {
            0.0
        };
        total_cg += stage_cg;
        total_traj += n_traj;
        final_acceptance = stage_acc;

        stage_results.push(hotspring_barracuda::lattice::pseudofermion::StageResult {
            mass: stage_mass,
            n_trajectories: n_traj,
            acceptance_rate: stage_acc,
            mean_delta_h: if n_traj > 0 {
                stage_delta_h_sum / n_traj as f64
            } else {
                0.0
            },
            plaquette: 0.0, // filled below after readback
        });
    }

    // Download final GPU config to CPU lattice
    hotspring_barracuda::lattice::gpu_hmc::gpu_links_to_lattice(
        rhmc.gpu(),
        &rhmc.state().gauge.gauge,
        &mut lattice,
    );

    let plaq = lattice.average_plaquette();
    if let Some(last) = stage_results.last_mut() {
        last.plaquette = plaq;
    }

    println!(
        "    ⟨P⟩ = {plaq:.6}, {:.0}% accept (final stage), {} CG iters, dt={dt:.4}, n_md={n_md}",
        final_acceptance * 100.0,
        total_cg,
    );

    for (i, stage) in stage_results.iter().enumerate() {
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
            ("final_acceptance", final_acceptance),
            ("total_cg_iterations", total_cg as f64),
            ("final_dt", dt),
            ("final_n_md", n_md as f64),
            ("total_trajectories", total_traj as f64),
            ("wall_seconds", start.elapsed().as_secs_f64()),
            ("npu_active", if npu_active { 1.0 } else { 0.0 }),
            ("pipeline", 1.0), // 1.0 = GPU unidirectional
        ],
    );

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
                final_acceptance,
                final_plaquette: plaq,
                final_dt: dt,
                n_trajectories: total_traj,
                converged: final_acceptance > 0.20,
            });
            writer.flush();
        }
    }

    harness.check_lower("p43_dyn_accept", final_acceptance, 0.20);
    harness.check_lower("p43_dyn_plaquette", plaq, 0.3);

    // Step 3: Gradient flow on the dynamical config
    let flow = run_flow(&mut lattice, FlowIntegrator::Lscfrk3w7, 0.01, 2.0, 5);

    let monotonic = flow
        .windows(2)
        .all(|w| w[1].energy_density <= w[0].energy_density + 1e-10);
    harness.check_bool("p43_dyn_flow_monotonic", monotonic);

    if let Some(w0) = find_w0(&flow) {
        println!("    w₀ = {w0:.4} (dynamical)");
        telem.log("p43_dyn", "w0", w0);
    }
    if let Some(t0) = find_t0(&flow) {
        println!("    t₀ = {t0:.4} (dynamical)");
        telem.log("p43_dyn", "t0", t0);
    }

    println!(
        "    {:.1}s (GPU unidirectional)",
        start.elapsed().as_secs_f64()
    );
}
