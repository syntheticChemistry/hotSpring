// SPDX-License-Identifier: AGPL-3.0-only
#![recursion_limit = "256"]

//! Production Dynamical + Multi-Head NPU Mixed Pipeline
//!
//! Exp 023: Full dynamical fermion HMC with 9-head NPU offloading for
//! pre/during/post computation screening. Builds on:
//! - Exp 022 (quenched NPU offload pipeline)
//! - production_dynamical_scan (resident CG dynamical HMC)
//! - Multi-head ESN (9 readout heads, zero-cost multi-output on Akida)
//!
//! # NPU Offload Architecture (11-head)
//!
//! Pre: β priority, param suggest, CG estimate, quenched length. During: therm, reject, phase.
//! Post: quality, anomaly, next-run.
//!
//! # Usage
//!
//! `HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_dynamical_mixed -- \
//!   --lattice=8 --betas=5.0,5.5,5.69,6.0 --mass=0.1 --therm=200 --meas=500 --seed=42`

use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_dynamical_hmc_trajectory_brain, gpu_hmc_trajectory_streaming, gpu_links_to_lattice,
    GpuDynHmcState, GpuDynHmcStreamingPipelines, GpuHmcState, GpuHmcStreamingPipelines,
    GpuResidentCgBuffers, GpuResidentCgPipelines,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::production::{
    dynamical_bootstrap::{
        acquire_dynamical_workers, run_bootstrap, run_post_computation, run_pre_computation,
        spawn_brain_residual_forwarder, DynamicalWorkers,
    },
    dynamical_summary::{
        create_trajectory_log_writer, hmc_auto_params, print_dynamical_startup_banner,
        print_dynamical_summary, write_dynamical_json, DynamicalBannerConfig, DynamicalNpuStats,
    },
    npu_worker::{NpuRequest, NpuResponse},
    plaquette_variance,
    titan_worker::{TitanRequest, TitanResponse},
    BetaResult, MetaRow,
};
use hotspring_barracuda::proxy::CortexRequest;

use std::io::Write as IoWrite;
use std::sync::mpsc;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
//  CLI
// ═══════════════════════════════════════════════════════════════════

struct CliArgs {
    lattice: usize,
    betas: Vec<f64>,
    mass: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    check_interval: usize,
    n_therm: usize,
    n_quenched_pretherm: usize,
    n_meas: usize,
    seed: u64,
    dt_override: Option<f64>,
    n_md_override: Option<usize>,
    output: Option<String>,
    trajectory_log: Option<String>,
    bootstrap_from: Option<String>,
    save_weights: Option<String>,
    no_titan: bool,
    no_npu_control: bool,
    max_adaptive: usize,
}

fn parse_args() -> CliArgs {
    let mut lattice = 8;
    let mut betas = vec![5.0, 5.5, 5.69, 6.0];
    let mut mass = 0.1;
    let mut cg_tol = 1e-8;
    let mut cg_max_iter = 5000;
    let mut check_interval = 10;
    let mut n_therm = 200;
    let mut n_quenched_pretherm = 50;
    let mut n_meas = 500;
    let mut seed = 42u64;
    let mut dt_override: Option<f64> = None;
    let mut n_md_override: Option<usize> = None;
    let mut output = None;
    let mut trajectory_log = None;
    let mut bootstrap_from = None;
    let mut save_weights = None;
    let mut no_titan = false;
    let mut no_npu_control = false;
    let mut max_adaptive: usize = 12;

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--lattice=") {
            lattice = val.parse().expect("--lattice=N");
        } else if let Some(val) = arg.strip_prefix("--betas=") {
            betas = val
                .split(',')
                .map(|s| s.parse().expect("beta float"))
                .collect();
        } else if let Some(val) = arg.strip_prefix("--mass=") {
            mass = val.parse().expect("--mass=F");
        } else if let Some(val) = arg.strip_prefix("--cg-tol=") {
            cg_tol = val.parse().expect("--cg-tol=F");
        } else if let Some(val) = arg.strip_prefix("--cg-max-iter=") {
            cg_max_iter = val.parse().expect("--cg-max-iter=N");
        } else if let Some(val) = arg.strip_prefix("--check-interval=") {
            check_interval = val.parse().expect("--check-interval=N");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            n_therm = val.parse().expect("--therm=N");
        } else if let Some(val) = arg.strip_prefix("--quenched-pretherm=") {
            n_quenched_pretherm = val.parse().expect("--quenched-pretherm=N");
        } else if let Some(val) = arg.strip_prefix("--meas=") {
            n_meas = val.parse().expect("--meas=N");
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            seed = val.parse().expect("--seed=N");
        } else if let Some(val) = arg.strip_prefix("--output=") {
            output = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--trajectory-log=") {
            trajectory_log = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--bootstrap-from=") {
            bootstrap_from = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--save-weights=") {
            save_weights = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--dt=") {
            dt_override = Some(val.parse().expect("--dt=F"));
        } else if let Some(val) = arg.strip_prefix("--n-md=") {
            n_md_override = Some(val.parse().expect("--n-md=N"));
        } else if arg == "--no-titan" {
            no_titan = true;
        } else if arg == "--no-npu-control" {
            no_npu_control = true;
        } else if let Some(val) = arg.strip_prefix("--max-adaptive=") {
            max_adaptive = val.parse().expect("--max-adaptive=N");
        }
    }

    CliArgs {
        lattice,
        betas,
        mass,
        cg_tol,
        cg_max_iter,
        check_interval,
        n_therm,
        n_quenched_pretherm,
        n_meas,
        seed,
        dt_override,
        n_md_override,
        output,
        trajectory_log,
        bootstrap_from,
        save_weights,
        no_titan,
        no_npu_control,
        max_adaptive,
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args = parse_args();
    let dims = [args.lattice, args.lattice, args.lattice, args.lattice];
    let vol: usize = dims.iter().product();

    print_dynamical_startup_banner(&DynamicalBannerConfig {
        lattice: args.lattice,
        vol,
        mass: args.mass,
        betas: &args.betas,
        cg_tol: args.cg_tol,
        cg_max_iter: args.cg_max_iter,
        check_interval: args.check_interval,
        n_therm: args.n_therm,
        n_quenched_pretherm: args.n_quenched_pretherm,
        n_meas: args.n_meas,
        seed: args.seed,
    });

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let Ok(workers) = acquire_dynamical_workers(&rt, args.no_titan) else {
        std::process::exit(1);
    };
    let DynamicalWorkers {
        gpu,
        npu_tx,
        npu_rx,
        brain_interrupt_rx,
        titan_handles,
        cortex_handles,
        npu_available,
    } = workers;

    let meta_context: Vec<MetaRow> =
        run_bootstrap(args.bootstrap_from.as_deref(), &npu_tx, &npu_rx);
    println!();

    let (auto_dt, auto_n_md) = hmc_auto_params(vol);
    let mut dt = args.dt_override.unwrap_or(auto_dt);
    let mut n_md = args.n_md_override.unwrap_or(auto_n_md);

    const DT_MIN: f64 = 0.001;
    const DT_MAX: f64 = 0.02;
    const NMD_MIN: usize = 20;
    const NMD_MAX: usize = 500;
    let npu_controls_params = !args.no_npu_control;
    println!(
        "  HMC:      dt={dt:.4}, n_md={n_md}, traj_length={:.3}, npu_control={}",
        dt * n_md as f64,
        npu_controls_params,
    );
    println!();

    let quenched_pipelines = GpuHmcStreamingPipelines::new(&gpu);
    let dyn_streaming_pipelines = GpuDynHmcStreamingPipelines::new(&gpu);
    let resident_cg_pipelines = GpuResidentCgPipelines::new(&gpu);

    let brain_residual_tx = spawn_brain_residual_forwarder(npu_tx.clone());

    let total_start = Instant::now();
    let mut results: Vec<BetaResult> = Vec::new();
    let mut npu_stats = DynamicalNpuStats::new();

    let mut traj_writer = create_trajectory_log_writer(args.trajectory_log.as_deref());

    let mut beta_order = run_pre_computation(
        &args.betas,
        &meta_context,
        args.mass,
        args.lattice,
        &npu_tx,
        &npu_rx,
        &mut npu_stats,
    );

    // ═══ Scan β points ═══
    println!(
        "═══ Dynamical β-Scan ({} points × {} meas) ═══",
        beta_order.len(),
        args.n_meas,
    );

    let mut bi = 0;
    while bi < beta_order.len() {
        let beta = beta_order[bi];
        println!(
            "── β = {:.4}, m = {} ({}/{}) ──",
            beta,
            args.mass,
            bi + 1,
            beta_order.len()
        );

        let start = Instant::now();
        let mut lat = Lattice::hot_start(dims, beta, args.seed + bi as u64);

        // Brain Layer 2: Check if Titan V has a warm config ready for this beta
        let mut titan_warm = false;
        if let Some(ref handles) = titan_handles {
            match handles.titan_rx.try_recv() {
                Ok(TitanResponse::WarmConfig {
                    beta: wb,
                    gauge_links,
                    plaquette,
                    wall_ms,
                }) => {
                    if (wb - beta).abs() < 0.001 {
                        hotspring_barracuda::lattice::gpu_hmc::unflatten_links_into(
                            &mut lat,
                            &gauge_links,
                        );
                        println!("  [Brain L2] Using Titan V warm config: P={plaquette:.6} ({wall_ms:.0}ms)");
                        titan_warm = true;
                    } else {
                        println!("  [Brain L2] Titan V config for β={wb:.4} (need {beta:.4}), discarding");
                    }
                }
                Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => {}
            }
        }

        if !titan_warm && vol <= 65536 {
            let mut cfg = HmcConfig {
                n_md_steps: n_md,
                dt,
                seed: args.seed + bi as u64 * 1000,
                integrator: IntegratorType::Omelyan,
            };
            for _ in 0..5 {
                hmc::hmc_trajectory(&mut lat, &mut cfg);
            }
        }

        // ─── NPU GPU-prep: predict optimal quenched length (Head 10) ───
        npu_tx
            .send(NpuRequest::PredictQuenchedLength {
                beta,
                mass: args.mass,
                lattice: args.lattice,
                meta_context: meta_context.clone(),
            })
            .ok();
        npu_stats.quenched_length_predictions += 1;
        npu_stats.total_npu_calls += 1;

        let npu_quenched_budget =
            if let Ok(NpuResponse::QuenchedLengthEstimate(est)) = npu_rx.recv() {
                println!(
                    "  NPU quenched-length prediction: {est} steps (budget was {})",
                    args.n_quenched_pretherm
                );
                est.min(args.n_quenched_pretherm)
            } else {
                args.n_quenched_pretherm
            };

        // ─── NPU GPU-prep: pipeline param suggestion during quenched upload ───
        npu_tx
            .send(NpuRequest::SuggestParameters {
                lattice: args.lattice,
                beta,
                mass: args.mass,
            })
            .ok();
        npu_stats.param_suggests += 1;
        npu_stats.total_npu_calls += 1;

        // Fire CG estimate for this β in parallel with quenched phase
        npu_tx
            .send(NpuRequest::PredictCgIters {
                beta,
                mass: args.mass,
                lattice: args.lattice,
            })
            .ok();
        npu_stats.cg_estimates += 1;
        npu_stats.total_npu_calls += 1;

        // Quenched pre-thermalization with NPU monitoring
        let quenched_state = GpuHmcState::from_lattice(&gpu, &lat, beta);
        let mut seed = args.seed * 100 + bi as u64;
        let quenched_check_interval = 10;
        let min_quenched = 5;
        let mut quenched_plaq_history: Vec<f64> = Vec::with_capacity(npu_quenched_budget);
        let mut quenched_used = 0;
        let mut quenched_early_exit = false;

        if npu_quenched_budget > 0 {
            print!("  Quenched pre-therm ({npu_quenched_budget} NPU-predicted)...");
            std::io::stdout().flush().ok();
            for i in 0..npu_quenched_budget {
                let r = gpu_hmc_trajectory_streaming(
                    &gpu,
                    &quenched_pipelines,
                    &quenched_state,
                    n_md,
                    dt,
                    i as u32,
                    &mut seed,
                );
                quenched_plaq_history.push(r.plaquette);
                quenched_used = i + 1;

                if let Some(ref mut w) = traj_writer {
                    let line = serde_json::json!({
                        "beta": beta, "mass": args.mass,
                        "traj_idx": i, "phase": "quenched_pretherm",
                        "accepted": r.accepted, "plaquette": r.plaquette,
                        "delta_h": r.delta_h, "cg_iters": 0,
                        "npu_budget": npu_quenched_budget,
                    });
                    writeln!(w, "{line}").ok();
                }

                if i >= min_quenched && (i + 1) % quenched_check_interval == 0 {
                    let window_start = quenched_plaq_history.len().saturating_sub(20);
                    npu_tx
                        .send(NpuRequest::QuenchedThermCheck {
                            plaq_window: quenched_plaq_history[window_start..].to_vec(),
                            beta,
                        })
                        .ok();
                    npu_stats.total_npu_calls += 1;

                    if let Ok(NpuResponse::QuenchedThermConverged(converged)) = npu_rx.recv() {
                        if converged {
                            quenched_early_exit = true;
                            break;
                        }
                    }
                }
            }
            if quenched_early_exit {
                npu_stats.quenched_early_exits += 1;
                npu_stats.quenched_steps_saved += npu_quenched_budget - quenched_used;
                print!(" early-exit@{quenched_used}");
            }
            println!(" done");
        }
        gpu_links_to_lattice(&gpu, &quenched_state, &mut lat);

        // Collect pipelined NPU responses (fired before quenched phase started)
        let npu_suggested_params = npu_rx.recv().ok();
        let npu_cg_estimate = npu_rx.recv().ok();

        if let Some(NpuResponse::ParameterSuggestion {
            dt: sdt,
            n_md: snmd,
        }) = npu_suggested_params
        {
            if npu_controls_params {
                let new_dt = sdt.clamp(DT_MIN, DT_MAX);
                let new_nmd = snmd.clamp(NMD_MIN, NMD_MAX);
                println!("  NPU param control: dt {dt:.4} → {new_dt:.4}, n_md {n_md} → {new_nmd}");
                dt = new_dt;
                n_md = new_nmd;
            } else {
                println!("  NPU param suggestion: dt={sdt:.4}, n_md={snmd} (--no-npu-control, keeping dt={dt:.4}, n_md={n_md})");
            }
        }

        // Use CG estimate to set adaptive check_interval
        let adaptive_check_interval = if let Some(NpuResponse::CgEstimate(est)) = npu_cg_estimate {
            let interval = if est < 200 {
                20
            } else if est < 1000 {
                10
            } else {
                5
            };
            println!("  NPU CG estimate: ~{est} iters → check_interval={interval}");
            interval
        } else {
            args.check_interval
        };

        // Brain Layer 3: Send cortex request for Anderson proxy at this beta
        let plaq_var_estimate = if results.is_empty() {
            0.05
        } else {
            let last = results.last().unwrap();
            last.std_plaq * last.std_plaq
        };
        cortex_handles
            .cortex_tx
            .send(CortexRequest {
                beta,
                mass: args.mass,
                lattice: args.lattice,
                plaq_var: plaq_var_estimate,
            })
            .ok();

        // Dynamical HMC setup
        let dyn_state = GpuDynHmcState::from_lattice(
            &gpu,
            &lat,
            beta,
            args.mass,
            args.cg_tol,
            args.cg_max_iter,
        );
        let cg_bufs = GpuResidentCgBuffers::new(
            &gpu,
            &dyn_streaming_pipelines.dyn_hmc,
            &resident_cg_pipelines,
            &dyn_state,
        );

        // ─── Thermalization with NPU early-exit (Head 3) ───
        let min_therm = 20;
        let therm_check_interval = 10;
        let mut plaq_history: Vec<f64> = Vec::with_capacity(args.n_therm);
        let mut therm_used = 0;
        let mut early_exit = false;

        print!("  Dynamical therm ({} max)...", args.n_therm);
        std::io::stdout().flush().ok();
        for i in 0..args.n_therm {
            let traj_idx = args.n_quenched_pretherm + i;
            let r = gpu_dynamical_hmc_trajectory_brain(
                &gpu,
                &dyn_streaming_pipelines,
                &resident_cg_pipelines,
                &dyn_state,
                &cg_bufs,
                n_md,
                dt,
                traj_idx as u32,
                &mut seed,
                adaptive_check_interval,
                &brain_residual_tx,
                &brain_interrupt_rx,
            );
            plaq_history.push(r.plaquette);
            therm_used = i + 1;

            if let Some(ref mut w) = traj_writer {
                let line = serde_json::json!({
                    "beta": beta, "mass": args.mass,
                    "traj_idx": traj_idx, "phase": "dynamical_therm",
                    "accepted": r.accepted, "plaquette": r.plaquette,
                    "delta_h": r.delta_h, "cg_iters": r.cg_iterations,
                });
                writeln!(w, "{line}").ok();
            }

            if i >= min_therm && (i + 1) % therm_check_interval == 0 {
                let window_start = plaq_history.len().saturating_sub(32);
                npu_tx
                    .send(NpuRequest::ThermCheck {
                        plaq_window: plaq_history[window_start..].to_vec(),
                        beta,
                    })
                    .ok();
                npu_stats.total_npu_calls += 1;

                if let Ok(NpuResponse::ThermConverged(converged)) = npu_rx.recv() {
                    if converged {
                        early_exit = true;
                        break;
                    }
                }
            }

            if (i + 1) % 50 == 0 {
                print!(" {}", i + 1);
                std::io::stdout().flush().ok();
            }
        }
        if early_exit {
            npu_stats.therm_early_exits += 1;
            npu_stats.therm_total_saved += args.n_therm - therm_used;
            print!(" early-exit@{therm_used}");
        }
        println!(" done");

        // ─── Measurement with NPU rejection prediction (Head 4) + anomaly detection (Head 8) ───
        let mut plaq_vals = Vec::with_capacity(args.n_meas);
        let mut poly_vals = Vec::new();
        let mut n_accepted = 0usize;
        let mut cg_total = 0usize;
        let mut reject_predictions = 0usize;
        let mut reject_correct = 0usize;
        let mut anomalies = 0usize;
        plaq_history.clear();

        print!("  Measuring ({} traj)...", args.n_meas);
        std::io::stdout().flush().ok();
        for i in 0..args.n_meas {
            let traj_idx = args.n_quenched_pretherm + args.n_therm + i;
            let traj_start = Instant::now();
            let r = gpu_dynamical_hmc_trajectory_brain(
                &gpu,
                &dyn_streaming_pipelines,
                &resident_cg_pipelines,
                &dyn_state,
                &cg_bufs,
                n_md,
                dt,
                traj_idx as u32,
                &mut seed,
                adaptive_check_interval,
                &brain_residual_tx,
                &brain_interrupt_rx,
            );
            let wall_us = traj_start.elapsed().as_micros() as u64;

            plaq_vals.push(r.plaquette);
            plaq_history.push(r.plaquette);
            if plaq_history.len() > 32 {
                plaq_history.remove(0);
            }
            cg_total += r.cg_iterations;
            if r.accepted {
                n_accepted += 1;
            }

            let running_acc = if i > 0 {
                n_accepted as f64 / (i + 1) as f64
            } else {
                0.5
            };

            // Head 4: Rejection prediction
            npu_tx
                .send(NpuRequest::RejectPredict {
                    beta,
                    plaquette: r.plaquette,
                    delta_h: r.delta_h,
                    acceptance_rate: running_acc,
                })
                .ok();
            npu_stats.reject_predictions += 1;
            npu_stats.total_npu_calls += 1;
            reject_predictions += 1;

            if let Ok(NpuResponse::RejectPrediction {
                likely_rejected,
                _confidence: _,
            }) = npu_rx.recv()
            {
                if likely_rejected != r.accepted {
                    npu_stats.reject_correct += 1;
                    reject_correct += 1;
                }
            }

            // Brain Layer 3: Check for proxy features from CPU cortex
            if i == 0 {
                if let Ok(features) = cortex_handles.proxy_rx.try_recv() {
                    npu_tx
                        .send(NpuRequest::ProxyFeatures {
                            beta: features.beta,
                            level_spacing_ratio: features.level_spacing_ratio,
                            lambda_min: features.lambda_min,
                            ipr: features.ipr,
                            tier: features.tier,
                        })
                        .ok();
                    println!(
                        "  [Brain L3] Proxy features: ⟨r⟩={:.3} |λ|_min={:.3} [{}]",
                        features.level_spacing_ratio, features.lambda_min, features.phase
                    );
                }
            }

            // Head 8: Anomaly detection (every 10 trajectories)
            if (i + 1) % 10 == 0 {
                npu_tx
                    .send(NpuRequest::AnomalyCheck {
                        plaq: r.plaquette,
                        delta_h: r.delta_h,
                        cg_iters: r.cg_iterations,
                        acceptance: running_acc,
                    })
                    .ok();
                npu_stats.anomaly_checks += 1;
                npu_stats.total_npu_calls += 1;

                if let Ok(NpuResponse::AnomalyFlag {
                    is_anomaly,
                    _score: _,
                }) = npu_rx.recv()
                {
                    if is_anomaly {
                        npu_stats.anomalies_found += 1;
                        anomalies += 1;
                    }
                }

                if npu_controls_params && i > 0 {
                    if running_acc > 0.85 {
                        let bump = (dt * 1.15).min(DT_MAX);
                        n_md = ((dt * n_md as f64 / bump).round() as usize).clamp(NMD_MIN, NMD_MAX);
                        dt = bump;
                        println!("  NPU mid-run: acc {running_acc:.0}% > 85%, dt → {dt:.4}, n_md → {n_md}");
                    } else if running_acc < 0.50 {
                        let drop = (dt * 0.85).max(DT_MIN);
                        n_md = ((dt * n_md as f64 / drop).round() as usize).clamp(NMD_MIN, NMD_MAX);
                        dt = drop;
                        println!("  NPU mid-run: acc {running_acc:.0}% < 50%, dt → {dt:.4}, n_md → {n_md}");
                    }
                }
            }

            // Polyakov readback
            let do_poly_readback = traj_writer.is_some() || (i + 1) % 100 == 0;
            let mut poly_mag = 0.0;
            let mut poly_phase = 0.0;
            if do_poly_readback {
                gpu_links_to_lattice(&gpu, &dyn_state.gauge, &mut lat);
                let (re, im) = lat.complex_polyakov_average();
                poly_mag = (re * re + im * im).sqrt();
                poly_phase = im.atan2(re);
                if (i + 1) % 100 == 0 {
                    poly_vals.push(poly_mag);
                }
            }

            if let Some(ref mut w) = traj_writer {
                let pvar = plaquette_variance(&plaq_history);
                let line = serde_json::json!({
                    "beta": beta, "mass": args.mass,
                    "traj_idx": traj_idx, "phase": "measurement",
                    "accepted": r.accepted, "plaquette": r.plaquette,
                    "delta_h": r.delta_h, "cg_iters": r.cg_iterations,
                    "polyakov_re": poly_mag, "polyakov_phase": poly_phase,
                    "action_density": 6.0 * (1.0 - r.plaquette),
                    "plaquette_var": pvar, "wall_us": wall_us,
                });
                writeln!(w, "{line}").ok();
            }

            if (i + 1) % 200 == 0 {
                print!(" {}", i + 1);
                std::io::stdout().flush().ok();
            }
        }
        println!(" done");

        if let Some(ref mut w) = traj_writer {
            w.flush().ok();
        }

        // Compute statistics
        let mean_plaq: f64 = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
        let var_plaq: f64 = plaq_vals
            .iter()
            .map(|p| (p - mean_plaq).powi(2))
            .sum::<f64>()
            / (plaq_vals.len() - 1).max(1) as f64;
        let std_plaq = var_plaq.sqrt();
        let mean_poly: f64 = if poly_vals.is_empty() {
            gpu_links_to_lattice(&gpu, &dyn_state.gauge, &mut lat);
            lat.average_polyakov_loop()
        } else {
            poly_vals.iter().sum::<f64>() / poly_vals.len() as f64
        };
        let susceptibility = var_plaq * vol as f64;
        let action_density = 6.0 * (1.0 - mean_plaq);
        let acceptance = n_accepted as f64 / args.n_meas as f64;
        let mean_cg = cg_total as f64 / args.n_meas as f64;
        let wall_s = start.elapsed().as_secs_f64();

        // Head 5: Phase classification
        npu_tx
            .send(NpuRequest::PhaseClassify {
                beta,
                plaquette: mean_plaq,
                polyakov: mean_poly,
                susceptibility,
            })
            .ok();
        npu_stats.phase_classifications += 1;
        npu_stats.total_npu_calls += 1;

        let phase = match npu_rx.recv() {
            Ok(NpuResponse::PhaseLabel(l)) => l,
            _ => {
                if beta < 5.6 {
                    "confined"
                } else if beta > 5.8 {
                    "deconfined"
                } else {
                    "transition"
                }
            }
        };

        // Head 7: Quality score
        let result = BetaResult {
            beta,
            mass: args.mass,
            mean_plaq,
            std_plaq,
            polyakov: mean_poly,
            susceptibility,
            action_density,
            acceptance,
            mean_cg_iters: mean_cg,
            n_traj: args.n_meas,
            wall_s,
            phase,
            therm_used,
            therm_budget: args.n_therm,
            dt_used: dt,
            n_md_used: n_md,
            npu_therm_early_exit: early_exit,
            npu_quenched_budget,
            npu_quenched_used: quenched_used,
            npu_quenched_early_exit: quenched_early_exit,
            npu_reject_predictions: reject_predictions,
            npu_reject_correct: reject_correct,
            npu_anomalies: anomalies,
            npu_cg_check_interval: adaptive_check_interval,
        };

        npu_tx
            .send(NpuRequest::QualityScore {
                result: result.clone(),
            })
            .ok();
        npu_stats.quality_scores += 1;
        npu_stats.total_npu_calls += 1;
        let quality = match npu_rx.recv() {
            Ok(NpuResponse::Quality(q)) => q,
            _ => 0.5,
        };

        results.push(result);

        let therm_info = if early_exit {
            format!("therm={therm_used}/{} (early-exit)", args.n_therm)
        } else {
            format!("therm={therm_used}")
        };
        println!(
            "  ⟨P⟩={:.6}±{:.6}  |L|={:.4}  χ={:.2}  acc={:.0}%  ⟨CG⟩={:.0}  Q={quality:.2}  {phase}  {therm_info}  dt={dt:.4} n_md={n_md}  ({wall_s:.1}s)",
            mean_plaq, std_plaq, mean_poly, susceptibility, acceptance * 100.0, mean_cg,
        );
        if anomalies > 0 {
            println!("  ⚠ {anomalies} anomalies detected by NPU");
        }
        println!();

        // Retrain after each β point
        npu_tx
            .send(NpuRequest::Retrain {
                results: results.clone(),
            })
            .ok();
        npu_stats.total_npu_calls += 1;
        if let Ok(NpuResponse::Retrained { beta_c }) = npu_rx.recv() {
            println!("  ESN retrained → β_c ≈ {beta_c:.4}");
        }

        // Gen 2: Query disagreement snapshot for concept edge detection
        npu_tx
            .send(NpuRequest::DisagreementQuery {
                beta,
                plaq: mean_plaq,
                mass: args.mass,
                chi: susceptibility,
                acceptance,
            })
            .ok();
        npu_stats.total_npu_calls += 1;
        if let Ok(NpuResponse::DisagreementSnapshot {
            delta_cg,
            delta_phase,
            delta_anomaly,
            delta_priority,
            urgency,
        }) = npu_rx.recv()
        {
            if urgency > 0.05 {
                println!("  [Concept Edge] β={beta:.4}: Δ_cg={delta_cg:.3} Δ_phase={delta_phase:.1} Δ_anom={delta_anomaly:.3} urgency={urgency:.3}");
            }
            if let Some(ref mut w) = traj_writer {
                let line = serde_json::json!({
                    "beta": beta, "phase": "disagreement_snapshot",
                    "delta_cg": delta_cg, "delta_phase": delta_phase,
                    "delta_anomaly": delta_anomaly, "delta_priority": delta_priority,
                    "urgency": urgency,
                    "mean_plaq": mean_plaq, "acceptance": acceptance,
                    "susceptibility": susceptibility,
                });
                writeln!(w, "{line}").ok();
            }
        }

        // Adaptive steering: after 3+ points, ask NPU to fill gaps
        if results.len() >= 3 && npu_stats.adaptive_inserted < args.max_adaptive {
            let measured: Vec<f64> = results.iter().map(|r| r.beta).collect();
            let remaining: Vec<f64> = if bi + 1 < beta_order.len() {
                beta_order[bi + 1..].to_vec()
            } else {
                vec![]
            };
            let beta_min = results.iter().map(|r| r.beta).fold(f64::INFINITY, f64::min) - 0.1;
            let beta_max = results
                .iter()
                .map(|r| r.beta)
                .fold(f64::NEG_INFINITY, f64::max)
                + 0.1;

            npu_tx
                .send(NpuRequest::SteerAdaptive {
                    measured_betas: measured.clone(),
                    queued_betas: remaining.clone(),
                    beta_min,
                    beta_max,
                    n_candidates: 80,
                })
                .ok();
            npu_stats.total_npu_calls += 1;

            npu_stats.adaptive_steered += 1;
            if let Ok(NpuResponse::AdaptiveSteered(Some(new_beta))) = npu_rx.recv() {
                println!("  NPU adaptive steer: inserting β={new_beta:.4} into scan queue ({}/{} adaptive budget)",
                    npu_stats.adaptive_inserted + 1, args.max_adaptive);
                beta_order.push(new_beta);
                npu_stats.adaptive_inserted += 1;
            }
        } else if npu_stats.adaptive_inserted >= args.max_adaptive && results.len() >= 3 {
            println!(
                "  NPU adaptive budget exhausted ({}/{})",
                npu_stats.adaptive_inserted, args.max_adaptive
            );
        }

        // Brain Layer 2: Kick off Titan V pre-therm for the NEXT beta point
        if let Some(ref handles) = titan_handles {
            if bi + 1 < beta_order.len() {
                let next_beta = beta_order[bi + 1];
                handles
                    .titan_tx
                    .send(TitanRequest::PreThermalize {
                        beta: next_beta,
                        mass: args.mass,
                        lattice: args.lattice,
                        n_quenched: args.n_quenched_pretherm,
                        seed: args.seed + (bi as u64 + 1) * 1000 + 500,
                        dt,
                        n_md,
                    })
                    .ok();
                println!("  [Brain L2] Titan V pre-thermalizing β={next_beta:.4} in background");
            }
        }

        bi += 1;
    }

    run_post_computation(
        &results,
        meta_context,
        args.save_weights.as_deref(),
        titan_handles.as_ref(),
        &mut traj_writer,
        &npu_tx,
        &npu_rx,
        &mut npu_stats,
    );
    drop(cortex_handles.cortex_tx);

    // ═══ Summary ═══
    let total_wall = total_start.elapsed().as_secs_f64();
    results.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap());

    let total_therm_budget: usize = results.iter().map(|r| r.therm_budget).sum();
    let total_therm_used: usize = results.iter().map(|r| r.therm_used).sum();
    let therm_savings_pct = if total_therm_budget > 0 {
        (1.0 - total_therm_used as f64 / total_therm_budget as f64) * 100.0
    } else {
        0.0
    };
    let quenched_savings_pct = if npu_stats.quenched_length_predictions > 0 {
        npu_stats.quenched_steps_saved as f64
            / (npu_stats.quenched_length_predictions as f64 * args.n_quenched_pretherm as f64)
                .max(1.0)
            * 100.0
    } else {
        0.0
    };

    print_dynamical_summary(
        &results,
        &npu_stats,
        args.lattice,
        args.mass,
        total_wall,
        quenched_savings_pct,
        therm_savings_pct,
        &gpu.adapter_name,
        args.trajectory_log.as_deref(),
    );

    if let Some(ref path) = args.output {
        let total_meas: usize = results.iter().map(|r| r.n_traj).sum();
        let npu_name = if npu_available {
            "AKD1000"
        } else {
            "MultiHeadNpu"
        };
        write_dynamical_json(
            path,
            &results,
            args.lattice,
            dims,
            vol,
            args.mass,
            args.cg_tol,
            args.cg_max_iter,
            args.check_interval,
            &gpu.adapter_name,
            npu_name,
            args.n_quenched_pretherm,
            args.n_therm,
            args.n_meas,
            args.seed,
            total_wall,
            total_meas,
            &npu_stats,
            quenched_savings_pct,
            therm_savings_pct,
        );
    }
}
