// SPDX-License-Identifier: AGPL-3.0-only

//! Per-β execution for dynamical mixed scan: quenched pre-therm, dynamical HMC, measurement, NPU steering.

use crate::lattice::gpu_hmc::{
    gpu_dynamical_hmc_trajectory_brain, gpu_hmc_trajectory_streaming, gpu_links_to_lattice,
    unflatten_links_into, GpuDynHmcState, GpuHmcState, GpuResidentCgBuffers,
};
use crate::lattice::hmc::{self, HmcConfig, IntegratorType};
use crate::lattice::wilson::Lattice;
use crate::production::{
    dynamical_summary::DynamicalNpuStats,
    npu_worker::{NpuRequest, NpuResponse},
    plaquette_variance,
    titan_worker::{TitanRequest, TitanResponse},
    BetaResult, MetaRow, TrajectoryEvent, TrajectoryPhase,
};
use crate::proxy::CortexRequest;
use std::io::Write;
use std::time::Instant;

use super::{DynamicalMixedConfig, DynamicalMixedScanContext};

const DT_MIN: f64 = 0.001;
const DT_MAX: f64 = 0.02;
const NMD_MIN: usize = 20;
const NMD_MAX: usize = 500;

/// Run a single β point: quenched pre-therm, dynamical therm, measurement, NPU post-processing.
pub(super) fn run_single_beta(
    config: &DynamicalMixedConfig,
    ctx: &DynamicalMixedScanContext<'_>,
    meta_context: &[MetaRow],
    dt: &mut f64,
    n_md: &mut usize,
    beta_order: &mut Vec<f64>,
    bi: usize,
    results: &mut Vec<BetaResult>,
    npu_stats: &mut DynamicalNpuStats,
    traj_writer: &mut Option<std::io::BufWriter<std::fs::File>>,
) {
    let beta = beta_order[bi];
    let dims = [config.lattice; 4];
    let vol: usize = dims.iter().product();
    let npu_controls_params = !config.no_npu_control;

    println!(
        "── β = {:.4}, m = {} ({}/{}) ──",
        beta,
        config.mass,
        bi + 1,
        beta_order.len()
    );

    let start = Instant::now();
    let mut lat = Lattice::hot_start(dims, beta, config.seed + bi as u64);

    let mut titan_warm = false;
    if let Some(handles) = ctx.titan_handles {
        let titan_result = if bi > 0 {
            handles
                .titan_rx
                .recv_timeout(std::time::Duration::from_secs(120))
                .ok()
        } else {
            None
        };
        if let Some(TitanResponse::WarmConfig {
            beta: wb,
            gauge_links,
            plaquette,
            wall_ms,
        }) = titan_result
        {
            if (wb - beta).abs() < 0.001 {
                unflatten_links_into(&mut lat, &gauge_links);
                println!(
                    "  [Brain L2] Using Titan V warm config: P={plaquette:.6} ({wall_ms:.0}ms)"
                );
                titan_warm = true;
            } else {
                println!("  [Brain L2] Titan V config for β={wb:.4} (need {beta:.4}), discarding");
            }
        }
    }

    if !titan_warm && vol <= 65536 {
        let mut cfg = HmcConfig {
            n_md_steps: *n_md,
            dt: *dt,
            seed: config.seed + bi as u64 * 1000,
            integrator: IntegratorType::Omelyan,
        };
        for _ in 0..5 {
            hmc::hmc_trajectory(&mut lat, &mut cfg);
        }
    }

    ctx.npu_tx
        .send(NpuRequest::PredictQuenchedLength {
            beta,
            mass: config.mass,
            lattice: config.lattice,
            meta_context: meta_context.to_vec(),
        })
        .ok();
    npu_stats.quenched_length_predictions += 1;
    npu_stats.total_npu_calls += 1;

    let npu_quenched_budget =
        if let Ok(NpuResponse::QuenchedLengthEstimate(est)) = ctx.npu_rx.recv() {
            println!(
                "  NPU quenched-length prediction: {est} steps (budget was {})",
                config.n_quenched_pretherm
            );
            est.min(config.n_quenched_pretherm)
        } else {
            config.n_quenched_pretherm
        };

    ctx.npu_tx
        .send(NpuRequest::SuggestParameters {
            lattice: config.lattice,
            beta,
            mass: config.mass,
        })
        .ok();
    npu_stats.param_suggests += 1;
    npu_stats.total_npu_calls += 1;

    ctx.npu_tx
        .send(NpuRequest::PredictCgIters {
            beta,
            mass: config.mass,
            lattice: config.lattice,
        })
        .ok();
    npu_stats.cg_estimates += 1;
    npu_stats.total_npu_calls += 1;

    let quenched_state = GpuHmcState::from_lattice(ctx.gpu, &lat, beta);
    let mut seed = config.seed * 100 + bi as u64;
    let quenched_check_interval = 10;
    let min_quenched = 5;
    let mut quenched_plaq_history: Vec<f64> = Vec::with_capacity(npu_quenched_budget);
    let mut quenched_used = 0;
    let mut quenched_early_exit = false;

    if npu_quenched_budget > 0 {
        print!("  Quenched pre-therm ({npu_quenched_budget} NPU-predicted)...");
        std::io::stdout().flush().ok();
        let mut quenched_accepted = 0usize;
        for i in 0..npu_quenched_budget {
            let traj_start = Instant::now();
            let r = gpu_hmc_trajectory_streaming(
                ctx.gpu,
                ctx.quenched_pipelines,
                &quenched_state,
                *n_md,
                *dt,
                i as u32,
                &mut seed,
            );
            let wall_us = traj_start.elapsed().as_micros() as u64;
            quenched_plaq_history.push(r.plaquette);
            quenched_used = i + 1;
            if r.accepted {
                quenched_accepted += 1;
            }
            let quenched_running_acc = if i > 0 {
                quenched_accepted as f64 / (i + 1) as f64
            } else {
                0.5
            };

            ctx.npu_tx
                .send(NpuRequest::TrajectoryEvent(TrajectoryEvent {
                    beta,
                    mass: config.mass,
                    lattice: config.lattice,
                    phase_tag: TrajectoryPhase::Quenched,
                    traj_idx: i,
                    plaquette: r.plaquette,
                    delta_h: r.delta_h,
                    accepted: r.accepted,
                    cg_iterations: 0,
                    polyakov_re: 0.0,
                    polyakov_phase: 0.0,
                    action_density: 6.0 * (1.0 - r.plaquette),
                    plaquette_var: plaquette_variance(&quenched_plaq_history),
                    wall_us,
                    running_acceptance: quenched_running_acc,
                }))
                .ok();

            if let Some(ref mut w) = traj_writer {
                writeln!(w, "{}", serde_json::json!({
                    "beta": beta, "mass": config.mass, "traj_idx": i, "phase": "quenched_pretherm",
                    "accepted": r.accepted, "plaquette": r.plaquette, "delta_h": r.delta_h,
                    "cg_iters": 0, "npu_budget": npu_quenched_budget,
                })).ok();
            }

            if i >= min_quenched && (i + 1) % quenched_check_interval == 0 {
                let window_start = quenched_plaq_history.len().saturating_sub(20);
                ctx.npu_tx
                    .send(NpuRequest::QuenchedThermCheck {
                        plaq_window: quenched_plaq_history[window_start..].to_vec(),
                        beta,
                        mass: config.mass,
                    })
                    .ok();
                npu_stats.total_npu_calls += 1;
                if let Ok(NpuResponse::QuenchedThermConverged(converged)) = ctx.npu_rx.recv() {
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
    gpu_links_to_lattice(ctx.gpu, &quenched_state, &mut lat);

    let npu_suggested_params = ctx.npu_rx.recv().ok();
    let npu_cg_estimate = ctx.npu_rx.recv().ok();

    if let Some(NpuResponse::ParameterSuggestion {
        dt: sdt,
        n_md: snmd,
    }) = npu_suggested_params
    {
        if npu_controls_params {
            let new_dt = sdt.clamp(DT_MIN, DT_MAX);
            let new_nmd = snmd.clamp(NMD_MIN, NMD_MAX);
            println!(
                "  NPU param control: dt {:.4} → {:.4}, n_md {} → {}",
                *dt, new_dt, *n_md, new_nmd
            );
            *dt = new_dt;
            *n_md = new_nmd;
        } else {
            println!("  NPU param suggestion: dt={:.4}, n_md={} (--no-npu-control, keeping dt={:.4}, n_md={})", sdt, snmd, *dt, *n_md);
        }
    }

    let adaptive_check_interval = if let Some(NpuResponse::CgEstimate(est)) = npu_cg_estimate {
        let interval = if est < 200 {
            20
        } else if est < 1000 {
            10
        } else {
            5
        };
        let npu_cap = (est * 4).max(500);
        let effective_cap = npu_cap.min(config.cg_max_iter);
        if effective_cap < config.cg_max_iter {
            println!("  NPU CG estimate: ~{est} iters → check_interval={interval}, cg_cap={effective_cap} (was {})", config.cg_max_iter);
        } else {
            println!("  NPU CG estimate: ~{est} iters → check_interval={interval}");
        }
        interval
    } else {
        config.check_interval
    };

    let plaq_var_estimate = results
        .last()
        .map_or(0.05, |last| last.std_plaq * last.std_plaq);
    ctx.cortex_handles
        .cortex_tx
        .send(CortexRequest {
            beta,
            mass: config.mass,
            lattice: config.lattice,
            plaq_var: plaq_var_estimate,
        })
        .ok();

    let dynamic_cg_max = if let Some(NpuResponse::CgEstimate(est)) = npu_cg_estimate.as_ref() {
        let cap = (*est * 3).max(500).min(config.cg_max_iter);
        if cap < config.cg_max_iter {
            eprintln!(
                "  [NPU] Dynamic CG cap: {cap} (predicted {est}, max {})",
                config.cg_max_iter
            );
        }
        cap
    } else {
        config.cg_max_iter
    };

    let dyn_state = GpuDynHmcState::from_lattice_multi(
        ctx.gpu,
        &lat,
        beta,
        config.mass,
        config.cg_tol,
        dynamic_cg_max,
        config.n_fields,
    );
    let cg_bufs = GpuResidentCgBuffers::new(
        ctx.gpu,
        &ctx.dyn_streaming_pipelines.dyn_hmc,
        ctx.resident_cg_pipelines,
        &dyn_state,
    );

    let min_therm = 8;
    let therm_check_interval = 5;
    let mut plaq_history: Vec<f64> = Vec::with_capacity(config.n_therm);
    let mut therm_used = 0;
    let mut early_exit = false;

    print!("  Dynamical therm ({} max)...", config.n_therm);
    std::io::stdout().flush().ok();
    let mut therm_accepted = 0usize;
    for i in 0..config.n_therm {
        let traj_idx = config.n_quenched_pretherm + i;
        let traj_start = Instant::now();
        let r = gpu_dynamical_hmc_trajectory_brain(
            ctx.gpu,
            ctx.dyn_streaming_pipelines,
            ctx.resident_cg_pipelines,
            &dyn_state,
            &cg_bufs,
            *n_md,
            *dt,
            traj_idx as u32,
            &mut seed,
            adaptive_check_interval,
            ctx.brain_residual_tx,
            ctx.brain_interrupt_rx,
        );
        let therm_wall_us = traj_start.elapsed().as_micros() as u64;
        plaq_history.push(r.plaquette);
        therm_used = i + 1;
        if r.accepted {
            therm_accepted += 1;
        }
        let therm_running_acc = if i > 0 {
            therm_accepted as f64 / (i + 1) as f64
        } else {
            0.5
        };

        ctx.npu_tx
            .send(NpuRequest::TrajectoryEvent(TrajectoryEvent {
                beta,
                mass: config.mass,
                lattice: config.lattice,
                phase_tag: TrajectoryPhase::Therm,
                traj_idx,
                plaquette: r.plaquette,
                delta_h: r.delta_h,
                accepted: r.accepted,
                cg_iterations: r.cg_iterations,
                polyakov_re: 0.0,
                polyakov_phase: 0.0,
                action_density: 6.0 * (1.0 - r.plaquette),
                plaquette_var: plaquette_variance(&plaq_history),
                wall_us: therm_wall_us,
                running_acceptance: therm_running_acc,
            }))
            .ok();

        if let Some(ref mut w) = traj_writer {
            writeln!(w, "{}", serde_json::json!({
                "beta": beta, "mass": config.mass, "traj_idx": traj_idx, "phase": "dynamical_therm",
                "accepted": r.accepted, "plaquette": r.plaquette, "delta_h": r.delta_h,
                "cg_iters": r.cg_iterations,
            })).ok();
        }

        if i >= min_therm && (i + 1) % therm_check_interval == 0 {
            let window_start = plaq_history.len().saturating_sub(32);
            ctx.npu_tx
                .send(NpuRequest::ThermCheck {
                    plaq_window: plaq_history[window_start..].to_vec(),
                    beta,
                    mass: config.mass,
                })
                .ok();
            npu_stats.total_npu_calls += 1;
            if let Ok(NpuResponse::ThermConverged(converged)) = ctx.npu_rx.recv() {
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
        npu_stats.therm_total_saved += config.n_therm - therm_used;
        print!(" early-exit@{therm_used}");
    }
    println!(" done");

    let mut plaq_vals = Vec::with_capacity(config.n_meas);
    let mut poly_vals = Vec::new();
    let mut n_accepted = 0usize;
    let mut cg_total = 0usize;
    let mut reject_predictions = 0usize;
    let mut reject_correct = 0usize;
    let mut anomalies = 0usize;
    plaq_history.clear();

    print!("  Measuring ({} traj)...", config.n_meas);
    std::io::stdout().flush().ok();
    for i in 0..config.n_meas {
        let traj_idx = config.n_quenched_pretherm + config.n_therm + i;
        let traj_start = Instant::now();
        let r = gpu_dynamical_hmc_trajectory_brain(
            ctx.gpu,
            ctx.dyn_streaming_pipelines,
            ctx.resident_cg_pipelines,
            &dyn_state,
            &cg_bufs,
            *n_md,
            *dt,
            traj_idx as u32,
            &mut seed,
            adaptive_check_interval,
            ctx.brain_residual_tx,
            ctx.brain_interrupt_rx,
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

        let (traj_poly_re, traj_poly_phase) = if r.accepted {
            let (re, im) = lat.complex_polyakov_average();
            (re.hypot(im), im.atan2(re))
        } else {
            (0.0, 0.0)
        };

        ctx.npu_tx
            .send(NpuRequest::TrajectoryEvent(TrajectoryEvent {
                beta,
                mass: config.mass,
                lattice: config.lattice,
                phase_tag: TrajectoryPhase::Measurement,
                traj_idx,
                plaquette: r.plaquette,
                delta_h: r.delta_h,
                accepted: r.accepted,
                cg_iterations: r.cg_iterations,
                polyakov_re: traj_poly_re,
                polyakov_phase: traj_poly_phase,
                action_density: 6.0 * (1.0 - r.plaquette),
                plaquette_var: plaquette_variance(&plaq_history),
                wall_us,
                running_acceptance: running_acc,
            }))
            .ok();

        ctx.npu_tx
            .send(NpuRequest::RejectPredict {
                beta,
                plaquette: r.plaquette,
                delta_h: r.delta_h,
                acceptance_rate: running_acc,
                mass: config.mass,
            })
            .ok();
        npu_stats.reject_predictions += 1;
        npu_stats.total_npu_calls += 1;
        reject_predictions += 1;

        if let Ok(NpuResponse::RejectPrediction {
            likely_rejected,
            confidence,
        }) = ctx.npu_rx.recv()
        {
            if likely_rejected != r.accepted {
                npu_stats.reject_correct += 1;
                reject_correct += 1;
            }
            if likely_rejected
                && confidence > 0.8
                && reject_correct > 3
                && i >= 2
                && n_accepted == 0
            {
                eprintln!(
                    "  [NPU] Reject streak: skipping remaining meas (conf={:.2}, 0/{} accepted)",
                    confidence,
                    i + 1
                );
                if r.delta_h.abs() >= 0.1 {
                    break;
                }
            }
        }

        if i == 0 {
            if let Ok(features) = ctx.cortex_handles.proxy_rx.try_recv() {
                ctx.npu_tx
                    .send(NpuRequest::ProxyFeatures {
                        beta: features.beta,
                        level_spacing_ratio: features.level_spacing_ratio,
                        lambda_min: features.lambda_min,
                        ipr: features.ipr,
                        bandwidth: features.bandwidth,
                        condition_number: features.condition_number,
                        phase: features.phase.clone(),
                        tier: features.tier,
                        potts_magnetization: features.potts_magnetization,
                        potts_susceptibility: features.potts_susceptibility,
                        potts_phase: features.potts_phase.clone(),
                    })
                    .ok();
                println!("  [Brain L3] Anderson: ⟨r⟩={:.3} |λ|_min={:.3} [{}] | Potts: mag={:.3} χ={:.1} [{}]",
                    features.level_spacing_ratio, features.lambda_min, features.phase,
                    features.potts_magnetization, features.potts_susceptibility, features.potts_phase);
            }
        }

        if (i + 1) % 10 == 0 {
            ctx.npu_tx
                .send(NpuRequest::AnomalyCheck {
                    beta,
                    plaq: r.plaquette,
                    delta_h: r.delta_h,
                    cg_iters: r.cg_iterations,
                    acceptance: running_acc,
                    mass: config.mass,
                })
                .ok();
            npu_stats.anomaly_checks += 1;
            npu_stats.total_npu_calls += 1;
            if let Ok(NpuResponse::AnomalyFlag { is_anomaly, .. }) = ctx.npu_rx.recv() {
                if is_anomaly {
                    npu_stats.anomalies_found += 1;
                    anomalies += 1;
                }
            }
            if npu_controls_params && i > 0 {
                if running_acc > 0.85 {
                    let bump = (*dt * 1.15).min(DT_MAX);
                    *n_md = ((*dt * *n_md as f64 / bump).round() as usize).clamp(NMD_MIN, NMD_MAX);
                    *dt = bump;
                    println!(
                        "  NPU mid-run: acc {running_acc:.0}% > 85%, dt → {:.4}, n_md → {}",
                        *dt, *n_md
                    );
                } else if running_acc < 0.50 {
                    let drop = (*dt * 0.85).max(DT_MIN);
                    *n_md = ((*dt * *n_md as f64 / drop).round() as usize).clamp(NMD_MIN, NMD_MAX);
                    *dt = drop;
                    println!(
                        "  NPU mid-run: acc {running_acc:.0}% < 50%, dt → {:.4}, n_md → {}",
                        *dt, *n_md
                    );
                }
            }
            if i > 0 && (i + 1) % 10 == 0 {
                let evt = TrajectoryEvent {
                    beta,
                    mass: config.mass,
                    lattice: config.lattice,
                    phase_tag: TrajectoryPhase::Measurement,
                    traj_idx,
                    plaquette: r.plaquette,
                    delta_h: r.delta_h,
                    accepted: r.accepted,
                    cg_iterations: r.cg_iterations,
                    polyakov_re: traj_poly_re,
                    polyakov_phase: traj_poly_phase,
                    action_density: 6.0 * (1.0 - r.plaquette),
                    plaquette_var: plaquette_variance(&plaq_history),
                    wall_us,
                    running_acceptance: running_acc,
                };
                ctx.npu_tx.send(NpuRequest::SubModelPredict(evt)).ok();
                if let Ok(NpuResponse::SubModelPredictions {
                    cg_cost,
                    steering,
                    phase: phase_pred,
                    ..
                }) = ctx.npu_rx.recv()
                {
                    if let Some(ref cg) = cg_cost {
                        if cg.len() >= 2 && cg[1] > 0.7 {
                            eprintln!(
                                "  [Sub-model] CG stall warning: P(stall)={:.2} at β={:.4}",
                                cg[1], beta
                            );
                        }
                    }
                    if let Some(ref ph) = phase_pred {
                        if !ph.is_empty() && (ph[0] > 0.8 || ph[0] < 0.2) {
                            eprintln!(
                                "  [Sub-model] Phase confidence: {:.2} at β={:.4}",
                                ph[0], beta
                            );
                        }
                    }
                    if let Some(ref steer) = steering {
                        if steer.len() >= 5 && steer[4] > 0.8 && i >= config.n_meas / 2 {
                            eprintln!("  [Sub-model] Steering: skip_decision={:.2}, saturation={:.2} → early-term meas", steer[4], steer[3]);
                            break;
                        }
                    }
                }
            }
        }

        let do_poly_readback = traj_writer.is_some() || (i + 1) % 100 == 0;
        let mut poly_mag = 0.0;
        let mut poly_phase = 0.0;
        if do_poly_readback {
            gpu_links_to_lattice(ctx.gpu, &dyn_state.gauge, &mut lat);
            let (re, im) = lat.complex_polyakov_average();
            poly_mag = re.hypot(im);
            poly_phase = im.atan2(re);
            if (i + 1) % 100 == 0 {
                poly_vals.push(poly_mag);
            }
        }

        if let Some(ref mut w) = traj_writer {
            let pvar = plaquette_variance(&plaq_history);
            writeln!(
                w,
                "{}",
                serde_json::json!({
                    "beta": beta, "mass": config.mass, "n_fields": config.n_fields,
                    "traj_idx": traj_idx, "phase": "measurement",
                    "accepted": r.accepted, "plaquette": r.plaquette,
                    "delta_h": r.delta_h, "cg_iters": r.cg_iterations,
                    "polyakov_re": poly_mag, "polyakov_phase": poly_phase,
                    "action_density": 6.0 * (1.0 - r.plaquette),
                    "plaquette_var": pvar, "wall_us": wall_us,
                })
            )
            .ok();
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

    let mean_plaq: f64 = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
    let var_plaq: f64 = plaq_vals
        .iter()
        .map(|p| (p - mean_plaq).powi(2))
        .sum::<f64>()
        / (plaq_vals.len() - 1).max(1) as f64;
    let std_plaq = var_plaq.sqrt();
    let mean_poly: f64 = if poly_vals.is_empty() {
        gpu_links_to_lattice(ctx.gpu, &dyn_state.gauge, &mut lat);
        lat.average_polyakov_loop()
    } else {
        poly_vals.iter().sum::<f64>() / poly_vals.len() as f64
    };
    let susceptibility = var_plaq * vol as f64;
    let action_density = 6.0 * (1.0 - mean_plaq);
    let acceptance = n_accepted as f64 / config.n_meas as f64;
    let mean_cg = cg_total as f64 / config.n_meas as f64;
    let wall_s = start.elapsed().as_secs_f64();

    ctx.npu_tx
        .send(NpuRequest::PhaseClassify {
            beta,
            plaquette: mean_plaq,
            polyakov: mean_poly,
            susceptibility,
            mass: config.mass,
            acceptance,
        })
        .ok();
    npu_stats.phase_classifications += 1;
    npu_stats.total_npu_calls += 1;

    let phase = match ctx.npu_rx.recv() {
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

    let result = BetaResult {
        beta,
        mass: config.mass,
        mean_plaq,
        std_plaq,
        polyakov: mean_poly,
        susceptibility,
        action_density,
        acceptance,
        mean_cg_iters: mean_cg,
        n_traj: config.n_meas,
        wall_s,
        phase,
        therm_used,
        therm_budget: config.n_therm,
        dt_used: *dt,
        n_md_used: *n_md,
        npu_therm_early_exit: early_exit,
        npu_quenched_budget,
        npu_quenched_used: quenched_used,
        npu_quenched_early_exit: quenched_early_exit,
        npu_reject_predictions: reject_predictions,
        npu_reject_correct: reject_correct,
        npu_anomalies: anomalies,
        npu_cg_check_interval: adaptive_check_interval,
    };

    ctx.npu_tx
        .send(NpuRequest::QualityScore {
            result: result.clone(),
        })
        .ok();
    npu_stats.quality_scores += 1;
    npu_stats.total_npu_calls += 1;
    let quality = match ctx.npu_rx.recv() {
        Ok(NpuResponse::Quality(q)) => q,
        _ => 0.5,
    };

    results.push(result);

    let therm_info = if early_exit {
        format!("therm={therm_used}/{} (early-exit)", config.n_therm)
    } else {
        format!("therm={therm_used}")
    };
    println!("  ⟨P⟩={:.6}±{:.6}  |L|={:.4}  χ={:.2}  acc={:.0}%  ⟨CG⟩={:.0}  Q={quality:.2}  {phase}  {therm_info}  dt={:.4} n_md={}  ({wall_s:.1}s)",
        mean_plaq, std_plaq, mean_poly, susceptibility, acceptance * 100.0, mean_cg, *dt, *n_md);
    if anomalies > 0 {
        println!("  ⚠ {anomalies} anomalies detected by NPU");
    }
    println!();

    ctx.npu_tx.send(NpuRequest::FlushTrajectoryBatch).ok();
    if let Ok(NpuResponse::TrajectoryBatchProcessed { n_events }) = ctx.npu_rx.recv() {
        if n_events > 0 {
            println!("  [NPU] Sub-models: {n_events} buffered events flushed");
        }
    }

    ctx.npu_tx
        .send(NpuRequest::Retrain {
            results: results.clone(),
        })
        .ok();
    npu_stats.total_npu_calls += 1;
    if let Ok(NpuResponse::Retrained { beta_c }) = ctx.npu_rx.recv() {
        println!("  ESN retrained → β_c ≈ {beta_c:.4}");
    }

    ctx.npu_tx.send(NpuRequest::SubModelMetrics).ok();
    if let Ok(NpuResponse::SubModelMetricsSnapshot(metrics)) = ctx.npu_rx.recv() {
        if let Some(ref mut w) = traj_writer {
            writeln!(w, "{}", serde_json::json!({"beta": beta, "phase": "sub_model_metrics", "bi": bi, "sub_models": metrics})).ok();
        }
    }

    ctx.npu_tx
        .send(NpuRequest::DisagreementQuery {
            beta,
            plaq: mean_plaq,
            mass: config.mass,
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
    }) = ctx.npu_rx.recv()
    {
        if urgency > 0.05 {
            println!(
                "  [Concept Edge] β={beta:.4}: Δ_cg={delta_cg:.3} Δ_phase={delta_phase:.1} Δ_anom={delta_anomaly:.3} urgency={urgency:.3}"
            );
        }
        if let Some(ref mut w) = traj_writer {
            writeln!(
                w,
                "{}",
                serde_json::json!({
                    "beta": beta, "phase": "disagreement_snapshot",
                    "delta_cg": delta_cg, "delta_phase": delta_phase,
                    "delta_anomaly": delta_anomaly, "delta_priority": delta_priority,
                    "urgency": urgency, "mean_plaq": mean_plaq, "acceptance": acceptance,
                    "susceptibility": susceptibility,
                })
            )
            .ok();
        }
    }

    if results.len() >= 3 && npu_stats.adaptive_inserted < config.max_adaptive {
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
        ctx.npu_tx
            .send(NpuRequest::SteerAdaptive {
                measured_betas: measured,
                queued_betas: remaining,
                beta_min,
                beta_max,
                n_candidates: 80,
            })
            .ok();
        npu_stats.total_npu_calls += 1;
        npu_stats.adaptive_steered += 1;
        match ctx.npu_rx.recv() {
            Ok(NpuResponse::AdaptiveSteered {
                suggestion: Some(new_beta),
                saturated,
            }) => {
                if saturated {
                    println!("  [NPU] Parameter set saturated — accepting final point β={new_beta:.4} then moving on");
                    beta_order.push(new_beta);
                    npu_stats.adaptive_inserted = config.max_adaptive;
                } else {
                    println!("  NPU adaptive steer: inserting β={:.4} into scan queue ({}/{} adaptive budget)", new_beta, npu_stats.adaptive_inserted + 1, config.max_adaptive);
                    beta_order.push(new_beta);
                    npu_stats.adaptive_inserted += 1;
                }
            }
            Ok(NpuResponse::AdaptiveSteered {
                suggestion: None,
                saturated,
            }) => {
                if saturated {
                    println!("  [NPU] Parameter set saturated — no novel points remain, moving on");
                    npu_stats.adaptive_inserted = config.max_adaptive;
                }
            }
            _ => {}
        }
    } else if npu_stats.adaptive_inserted >= config.max_adaptive && results.len() >= 3 {
        println!(
            "  NPU adaptive budget exhausted ({}/{})",
            npu_stats.adaptive_inserted, config.max_adaptive
        );
    }

    if let Some(handles) = ctx.titan_handles {
        if bi + 2 < beta_order.len() {
            let future_beta = beta_order[bi + 2];
            handles
                .titan_tx
                .send(TitanRequest::PreThermalize {
                    beta: future_beta,
                    mass: config.mass,
                    lattice: config.lattice,
                    n_quenched: config.n_quenched_pretherm,
                    seed: config.seed + (bi as u64 + 2) * 1000 + 500,
                    dt: *dt,
                    n_md: *n_md,
                })
                .ok();
            println!("  [Brain L2] Titan V pre-thermalizing β={future_beta:.4} in background");
        }
    }

    ctx.npu_tx.send(NpuRequest::FlushTrajectoryBatch).ok();
}
