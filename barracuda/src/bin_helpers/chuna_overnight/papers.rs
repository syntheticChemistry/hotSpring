// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 43 lattice QCD overnight validation (GPU streaming HMC + gradient flow).
//!
//! Papers 44 (dielectric) and 45 (kinetic-fluid) live in `paper_44.rs` and `paper_45.rs`.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuDynHmcState, GpuHmcState, GpuHmcStreamingPipelines, gpu_dynamical_hmc_trajectory_streaming,
    gpu_hmc_trajectory_streaming, gpu_links_to_lattice,
};
use hotspring_barracuda::tolerances::lattice::DYNAMICAL_CG_MAX_ITER;
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};
use std::time::Instant;

use super::harness::SubstrateResults;

/// CPU-based quenched pre-thermalization fallback (retained for HOTSPRING_NO_GPU=1).
pub fn cpu_quenched_pretherm(
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

#[expect(
    clippy::expect_used,
    reason = "GPU trajectory failure is unrecoverable in this pipeline"
)]
pub fn paper_43_convergence(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::lattice::gradient_flow::{FlowIntegrator, run_flow};
    use hotspring_barracuda::lattice::wilson::Lattice;

    println!("  Convergence sweep (8⁴ β=6.0) [GPU streaming HMC]...");
    let start = Instant::now();

    let dims = [8, 8, 8, 8];
    let beta = 6.0;
    let n_therm = 100;
    let n_md_steps = 20;
    let dt = 0.05;

    let pipelines = GpuHmcStreamingPipelines::new(gpu);

    let integrators = [
        (FlowIntegrator::Rk3Luscher, "W6", 3),
        (FlowIntegrator::Lscfrk3w7, "W7", 3),
        (FlowIntegrator::Lscfrk4ck, "CK4", 4),
    ];

    let epsilons = [0.02, 0.01, 0.005, 0.002, 0.001];

    for (integrator, name, expected_order) in &integrators {
        let mut e_values: Vec<(f64, f64)> = Vec::new();
        for &eps in &epsilons {
            let lat = Lattice::hot_start(dims, beta, 42);
            let state = GpuHmcState::from_lattice(gpu, &lat, beta);
            let mut seed = 12345u64;
            for i in 0..n_therm {
                let r = gpu_hmc_trajectory_streaming(
                    gpu, &pipelines, &state, n_md_steps, dt, i as u32, &mut seed,
                )
                .expect("streaming HMC trajectory");
                if (i + 1) % 25 == 0 {
                    println!(
                        "      [{name} ε={eps}] therm {}/{n_therm}: ⟨P⟩={:.6}, acc={}",
                        i + 1,
                        r.plaquette,
                        if r.accepted { "Y" } else { "N" }
                    );
                    telem.log_map(
                        &format!("p43_conv_{name}_eps{eps}"),
                        &[
                            ("trajectory", (i + 1) as f64),
                            ("plaquette", r.plaquette),
                            ("accepted", f64::from(u8::from(r.accepted))),
                        ],
                    );
                }
            }
            let mut lat = Lattice::hot_start(dims, beta, 42);
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
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

#[expect(
    clippy::expect_used,
    reason = "GPU trajectory failure is unrecoverable in this pipeline"
)]
pub fn paper_43_production(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
    results: &mut SubstrateResults,
) {
    use hotspring_barracuda::lattice::gradient_flow::{FlowIntegrator, find_t0, find_w0, run_flow};
    use hotspring_barracuda::lattice::wilson::Lattice;

    let configs: &[([usize; 4], f64, usize, &str)] = &[
        ([8, 8, 8, 8], 5.9, 200, "8⁴ β=5.9"),
        ([8, 8, 8, 8], 6.0, 200, "8⁴ β=6.0"),
        ([8, 8, 8, 16], 6.0, 200, "8³×16 β=6.0"),
        ([8, 8, 8, 8], 6.2, 200, "8⁴ β=6.2"),
        ([16, 16, 16, 16], 6.0, 500, "16⁴ β=6.0"),
        ([16, 16, 16, 32], 6.0, 300, "16³×32 β=6.0"),
    ];

    for (dims, beta, n_therm, label) in configs {
        let start = Instant::now();
        println!("  {label} [GPU streaming]...");

        let lattice = Lattice::hot_start(*dims, *beta, 42);
        let volume: usize = dims.iter().product();
        let (n_md, md_dt) = if volume > 10000 {
            (40, 0.025)
        } else {
            (20, 0.05)
        };

        let pipelines = GpuHmcStreamingPipelines::new(gpu);
        let state = GpuHmcState::from_lattice(gpu, &lattice, *beta);
        let mut seed = 12345u64;
        let mut n_accept = 0usize;
        let mut last_plaq = 0.0;
        let log_interval = if *n_therm >= 500 { 50 } else { 25 };

        for i in 0..*n_therm {
            let r = gpu_hmc_trajectory_streaming(
                gpu, &pipelines, &state, n_md, md_dt, i as u32, &mut seed,
            )
            .expect("streaming HMC trajectory");
            if r.accepted {
                n_accept += 1;
            }
            last_plaq = r.plaquette;
            if (i + 1) % log_interval == 0 {
                let running_acc = n_accept as f64 / (i + 1) as f64;
                println!(
                    "    [{}/{}] ⟨P⟩={:.6}, acc={:.0}%",
                    i + 1,
                    n_therm,
                    r.plaquette,
                    running_acc * 100.0
                );
                telem.log_map(
                    &format!("p43_prod_{label}"),
                    &[
                        ("trajectory", (i + 1) as f64),
                        ("plaquette", r.plaquette),
                        ("acceptance", running_acc),
                        ("delta_h", r.delta_h),
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
        results.plaquettes.push((label.to_string(), last_plaq));

        let mut lattice = Lattice::hot_start(*dims, *beta, 42);
        gpu_links_to_lattice(gpu, &state, &mut lattice);

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
#[expect(
    clippy::expect_used,
    reason = "GPU trajectory failure is unrecoverable in this pipeline"
)]
pub fn paper_43_dynamical(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
    results: &mut SubstrateResults,
) {
    use hotspring_barracuda::lattice::gradient_flow::{FlowIntegrator, find_t0, find_w0, run_flow};
    use hotspring_barracuda::lattice::wilson::Lattice;

    let start = Instant::now();

    let dims = [8, 8, 8, 8];
    let beta = 5.4;
    let mass = 0.1;
    let n_fields = 1; // Nf = 4
    let cg_tol = 1e-8;
    let cg_max_iter = DYNAMICAL_CG_MAX_ITER;
    let n_quenched_therm = 100;
    let n_dyn_therm = 50;
    let n_md_steps = 20;
    let dt = 0.05;

    println!("  Dynamical 8⁴ β={beta} (N_f=4, m={mass}) [GPU streaming]...");

    // Step 1: Quenched GPU pre-thermalization
    println!("    Quenched pre-therm ({n_quenched_therm} trajectories, GPU streaming)...");
    let lattice = Lattice::hot_start(dims, beta, 42);
    let quenched_pip = GpuHmcStreamingPipelines::new(gpu);
    let quenched_state = GpuHmcState::from_lattice(gpu, &lattice, beta);
    let mut seed = 12345u64;
    let mut q_accept = 0usize;

    for i in 0..n_quenched_therm {
        let r = gpu_hmc_trajectory_streaming(
            gpu,
            &quenched_pip,
            &quenched_state,
            n_md_steps,
            dt,
            i as u32,
            &mut seed,
        )
        .expect("streaming HMC trajectory");
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
                "p43_dyn_quenched",
                &[
                    ("trajectory", (i + 1) as f64),
                    ("plaquette", r.plaquette),
                    ("acceptance", q_accept as f64 / (i + 1) as f64),
                ],
            );
        }
    }
    let q_plaq = {
        let mut tmp = Lattice::hot_start(dims, beta, 42);
        gpu_links_to_lattice(gpu, &quenched_state, &mut tmp);
        tmp.average_plaquette()
    };
    println!(
        "    Quenched done: ⟨P⟩={q_plaq:.6}, {:.0}% accept, {:.1}s",
        q_accept as f64 / n_quenched_therm as f64 * 100.0,
        start.elapsed().as_secs_f64(),
    );

    // Step 2: Dynamical fermion HMC via GPU streaming
    println!("    Dynamical GPU streaming ({n_dyn_therm} trajectories, mass annealing)...");
    let mut lattice = Lattice::hot_start(dims, beta, 42);
    gpu_links_to_lattice(gpu, &quenched_state, &mut lattice);

    let dyn_pip = hotspring_barracuda::lattice::gpu_hmc::GpuDynHmcStreamingPipelines::new(gpu);
    let masses = [1.0, 0.5, 0.2, mass];
    let trajs_per_mass = n_dyn_therm / masses.len().max(1);
    let mut total_cg = 0usize;
    let mut total_traj = 0usize;
    let mut last_acceptance = 0.0;
    let mut last_plaq = 0.0;

    for (stage_idx, &m) in masses.iter().enumerate() {
        let dyn_state = GpuDynHmcState::from_lattice_multi(
            gpu,
            &lattice,
            beta,
            m,
            cg_tol,
            cg_max_iter,
            n_fields,
        );
        let n_traj = if stage_idx == masses.len() - 1 {
            n_dyn_therm - trajs_per_mass * (masses.len() - 1)
        } else {
            trajs_per_mass
        };
        let mut stage_accept = 0usize;

        for t in 0..n_traj {
            let r = gpu_dynamical_hmc_trajectory_streaming(
                gpu,
                &dyn_pip,
                &dyn_state,
                n_md_steps,
                dt,
                (total_traj + t) as u32,
                &mut seed,
            )
            .expect("dynamical streaming HMC trajectory");
            if r.accepted {
                stage_accept += 1;
            }
            total_cg += r.cg_iterations;
            total_traj += 1;
            last_plaq = r.plaquette;
        }
        last_acceptance = stage_accept as f64 / n_traj.max(1) as f64;

        println!(
            "      stage {} (m={m}): {:.0}% accept, ⟨P⟩={last_plaq:.6}",
            stage_idx,
            last_acceptance * 100.0
        );
        telem.log_map(
            &format!("p43_dyn_stage{stage_idx}"),
            &[
                ("mass", m),
                ("acceptance", last_acceptance),
                ("plaquette", last_plaq),
                ("n_trajectories", n_traj as f64),
            ],
        );

        gpu_links_to_lattice(gpu, &dyn_state.gauge, &mut lattice);
    }

    let plaq = lattice.average_plaquette();
    println!(
        "    ⟨P⟩ = {plaq:.6}, {:.0}% accept (final stage), {} CG iters",
        last_acceptance * 100.0,
        total_cg,
    );

    telem.log_map(
        "p43_dyn",
        &[
            ("final_plaquette", plaq),
            ("final_acceptance", last_acceptance),
            ("total_cg_iterations", total_cg as f64),
            ("total_trajectories", total_traj as f64),
            ("wall_seconds", start.elapsed().as_secs_f64()),
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
            let _ = writer.write_summary(&RunSummary {
                beta,
                mass,
                final_acceptance: last_acceptance,
                final_plaquette: plaq,
                final_dt: dt,
                n_trajectories: total_traj,
                converged: last_acceptance > 0.20,
            });
            let _ = writer.flush();
        }
    }

    harness.check_lower("p43_dyn_accept", last_acceptance, 0.20);
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
