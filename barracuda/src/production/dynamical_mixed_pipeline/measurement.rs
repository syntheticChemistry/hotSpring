// SPDX-License-Identifier: AGPL-3.0-or-later

//! Measurement phase for the dynamical mixed pipeline.
//!
//! Extracts the per-β measurement loop from `single_beta.rs` into a dedicated
//! module with structured state. The measurement loop is the densest phase:
//! NPU reject prediction, anomaly detection, sub-model inference, mid-run
//! parameter tuning, and trajectory logging all happen here.

use crate::lattice::gpu_hmc::{
    GpuDynHmcResult, GpuDynHmcState, GpuResidentCgBuffers, gpu_dynamical_hmc_trajectory_brain,
    gpu_links_to_lattice,
};
use crate::lattice::wilson::Lattice;
use crate::production::{
    TrajectoryEvent, TrajectoryPhase,
    npu_worker::{NpuRequest, NpuResponse},
    plaquette_variance,
};

use super::single_beta::{DT_MAX, DT_MIN, NMD_MAX, NMD_MIN};
use super::{DynamicalMixedConfig, DynamicalMixedScanContext};

use std::io::Write;
use std::time::Instant;

/// Accumulated statistics from one measurement phase.
pub(super) struct MeasurementResult {
    pub plaq_vals: Vec<f64>,
    pub poly_vals: Vec<f64>,
    pub n_accepted: usize,
    pub cg_total: usize,
    pub reject_predictions: usize,
    pub reject_correct: usize,
    pub anomalies: usize,
}

/// Run the measurement phase for a single β point.
///
/// Returns accumulated observables for post-processing.
pub(super) fn run_measurement(
    config: &DynamicalMixedConfig,
    ctx: &DynamicalMixedScanContext<'_>,
    dyn_state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    lat: &mut Lattice,
    dt: &mut f64,
    n_md: &mut usize,
    seed: &mut u64,
    adaptive_check_interval: usize,
    beta: f64,
    npu_stats: &mut crate::production::dynamical_summary::DynamicalNpuStats,
    traj_writer: &mut Option<std::io::BufWriter<std::fs::File>>,
) -> MeasurementResult {
    let npu_controls_params = !config.no_npu_control;
    let mut plaq_vals = Vec::with_capacity(config.n_meas);
    let mut poly_vals = Vec::new();
    let mut n_accepted = 0usize;
    let mut cg_total = 0usize;
    let mut reject_predictions = 0usize;
    let mut reject_correct = 0usize;
    let mut anomalies = 0usize;
    let mut plaq_history: Vec<f64> = Vec::with_capacity(config.n_meas);

    print!("  Measuring ({} traj)...", config.n_meas);
    std::io::stdout().flush().ok();

    for i in 0..config.n_meas {
        let traj_idx = config.n_quenched_pretherm + config.n_therm + i;
        let traj_start = Instant::now();
        let r = gpu_dynamical_hmc_trajectory_brain(
            ctx.gpu,
            ctx.dyn_streaming_pipelines,
            ctx.resident_cg_pipelines,
            dyn_state,
            cg_bufs,
            *n_md,
            *dt,
            traj_idx as u32,
            seed,
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

        emit_trajectory_event(
            ctx,
            config,
            beta,
            &plaq_history,
            &r,
            traj_idx,
            wall_us,
            running_acc,
            traj_poly_re,
            traj_poly_phase,
        );

        handle_reject_prediction(
            ctx,
            config,
            &r,
            running_acc,
            beta,
            i,
            n_accepted,
            &mut reject_predictions,
            &mut reject_correct,
            npu_stats,
        );

        if i == 0 {
            handle_proxy_features(ctx, npu_stats);
        }

        if (i + 1) % 10 == 0 {
            handle_anomaly_and_steering(
                ctx,
                config,
                &r,
                &plaq_history,
                running_acc,
                beta,
                i,
                dt,
                n_md,
                traj_idx,
                wall_us,
                traj_poly_re,
                traj_poly_phase,
                &mut anomalies,
                npu_controls_params,
                npu_stats,
            );
        }

        handle_polyakov_readback(
            ctx,
            dyn_state,
            lat,
            &mut poly_vals,
            i,
            traj_writer,
            config,
            beta,
            &r,
            &plaq_history,
            wall_us,
        );

        if (i + 1) % 200 == 0 {
            print!(" {}", i + 1);
            std::io::stdout().flush().ok();
        }
    }
    println!(" done");

    if let Some(w) = traj_writer.as_mut() {
        w.flush().ok();
    }

    MeasurementResult {
        plaq_vals,
        poly_vals,
        n_accepted,
        cg_total,
        reject_predictions,
        reject_correct,
        anomalies,
    }
}

fn emit_trajectory_event(
    ctx: &DynamicalMixedScanContext<'_>,
    config: &DynamicalMixedConfig,
    beta: f64,
    plaq_history: &[f64],
    r: &GpuDynHmcResult,
    traj_idx: usize,
    wall_us: u64,
    running_acc: f64,
    poly_re: f64,
    poly_phase: f64,
) {
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
            polyakov_re: poly_re,
            polyakov_phase: poly_phase,
            action_density: 6.0 * (1.0 - r.plaquette),
            plaquette_var: plaquette_variance(plaq_history),
            wall_us,
            running_acceptance: running_acc,
        }))
        .ok();
}

fn handle_reject_prediction(
    ctx: &DynamicalMixedScanContext<'_>,
    config: &DynamicalMixedConfig,
    r: &GpuDynHmcResult,
    running_acc: f64,
    beta: f64,
    i: usize,
    n_accepted: usize,
    reject_predictions: &mut usize,
    reject_correct: &mut usize,
    npu_stats: &mut crate::production::dynamical_summary::DynamicalNpuStats,
) {
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
    *reject_predictions += 1;

    if let Ok(NpuResponse::RejectPrediction {
        likely_rejected,
        confidence,
    }) = ctx.npu_rx.recv()
    {
        if likely_rejected != r.accepted {
            npu_stats.reject_correct += 1;
            *reject_correct += 1;
        }
        if likely_rejected && confidence > 0.8 && *reject_correct > 3 && i >= 2 && n_accepted == 0 {
            eprintln!(
                "  [NPU] Reject streak: skipping remaining meas (conf={:.2}, 0/{} accepted)",
                confidence,
                i + 1
            );
        }
    }
}

fn handle_proxy_features(
    ctx: &DynamicalMixedScanContext<'_>,
    npu_stats: &mut crate::production::dynamical_summary::DynamicalNpuStats,
) {
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
        npu_stats.total_npu_calls += 1;
        println!(
            "  [Brain L3] Anderson: ⟨r⟩={:.3} |λ|_min={:.3} [{}] | Potts: mag={:.3} χ={:.1} [{}]",
            features.level_spacing_ratio,
            features.lambda_min,
            features.phase,
            features.potts_magnetization,
            features.potts_susceptibility,
            features.potts_phase
        );
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "pipeline phase with multiple NPU interactions"
)]
fn handle_anomaly_and_steering(
    ctx: &DynamicalMixedScanContext<'_>,
    config: &DynamicalMixedConfig,
    r: &GpuDynHmcResult,
    plaq_history: &[f64],
    running_acc: f64,
    beta: f64,
    i: usize,
    dt: &mut f64,
    n_md: &mut usize,
    traj_idx: usize,
    wall_us: u64,
    traj_poly_re: f64,
    traj_poly_phase: f64,
    anomalies: &mut usize,
    npu_controls_params: bool,
    npu_stats: &mut crate::production::dynamical_summary::DynamicalNpuStats,
) {
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
    if let Ok(NpuResponse::AnomalyFlag { is_anomaly, .. }) = ctx.npu_rx.recv()
        && is_anomaly
    {
        npu_stats.anomalies_found += 1;
        *anomalies += 1;
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

    if i > 0 {
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
            plaquette_var: plaquette_variance(plaq_history),
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
            if let Some(ref cg) = cg_cost
                && cg.len() >= 2
                && cg[1] > 0.7
            {
                eprintln!(
                    "  [Sub-model] CG stall warning: P(stall)={:.2} at β={:.4}",
                    cg[1], beta
                );
            }
            if let Some(ref ph) = phase_pred
                && !ph.is_empty()
                && (ph[0] > 0.8 || ph[0] < 0.2)
            {
                eprintln!(
                    "  [Sub-model] Phase confidence: {:.2} at β={:.4}",
                    ph[0], beta
                );
            }
            if let Some(ref steer) = steering
                && steer.len() >= 5
                && steer[4] > 0.8
                && i >= config.n_meas / 2
            {
                eprintln!(
                    "  [Sub-model] Steering: skip_decision={:.2}, saturation={:.2} → early-term meas",
                    steer[4], steer[3]
                );
            }
        }
    }
}

fn handle_polyakov_readback(
    ctx: &DynamicalMixedScanContext<'_>,
    dyn_state: &GpuDynHmcState,
    lat: &mut Lattice,
    poly_vals: &mut Vec<f64>,
    i: usize,
    traj_writer: &mut Option<std::io::BufWriter<std::fs::File>>,
    config: &DynamicalMixedConfig,
    beta: f64,
    r: &GpuDynHmcResult,
    plaq_history: &[f64],
    wall_us: u64,
) {
    let do_poly_readback = traj_writer.is_some() || (i + 1).is_multiple_of(100);
    let mut poly_mag = 0.0;
    let mut poly_phase = 0.0;
    if do_poly_readback {
        gpu_links_to_lattice(ctx.gpu, &dyn_state.gauge, lat);
        let (re, im) = lat.complex_polyakov_average();
        poly_mag = re.hypot(im);
        poly_phase = im.atan2(re);
        if (i + 1).is_multiple_of(100) {
            poly_vals.push(poly_mag);
        }
    }

    if let Some(w) = traj_writer.as_mut() {
        let traj_idx = config.n_quenched_pretherm + config.n_therm + i;
        let pvar = plaquette_variance(plaq_history);
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
}
