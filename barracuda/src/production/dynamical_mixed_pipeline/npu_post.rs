// SPDX-License-Identifier: AGPL-3.0-or-later

//! NPU post-processing for a completed β-point: flush batch, retrain ESN,
//! sub-model metrics, disagreement query, adaptive β-insertion, Titan prefetch.

use std::io::Write as _;

use crate::production::{
    BetaResult,
    dynamical_summary::DynamicalNpuStats,
    npu_worker::{NpuRequest, NpuResponse},
    titan_worker::TitanRequest,
};

use super::{DynamicalMixedConfig, DynamicalMixedScanContext};

/// Per-β-point scalar context for NPU post-processing.
///
/// Bundles value-typed data that does not borrow from the scan pipeline,
/// keeping [`npu_post_process_beta`] under the argument-count threshold.
pub(super) struct NpuPostArgs {
    /// Index of the current β in `beta_order`.
    pub bi: usize,
    /// Current gauge coupling.
    pub beta: f64,
    /// Mean plaquette from measurement phase.
    pub mean_plaq: f64,
    /// Plaquette susceptibility (variance × volume).
    pub susceptibility: f64,
    /// Acceptance rate from measurement phase.
    pub acceptance: f64,
    /// Current MD timestep (read-only snapshot for Titan prefetch).
    pub dt: f64,
    /// Current number of MD steps (read-only snapshot for Titan prefetch).
    pub n_md: usize,
}

/// Run all NPU tasks that follow a completed β-point measurement.
///
/// - Flush buffered trajectory events
/// - Trigger ESN retrain
/// - Capture sub-model metrics
/// - Query disagreement snapshot
/// - Adaptive β-point steering and insertion
/// - Pre-thermalize the next β on Titan V (background)
pub(super) fn npu_post_process_beta(
    ctx: &DynamicalMixedScanContext<'_>,
    config: &DynamicalMixedConfig,
    args: &NpuPostArgs,
    results: &[BetaResult],
    npu_stats: &mut DynamicalNpuStats,
    beta_order: &mut Vec<f64>,
    traj_writer: &mut Option<std::io::BufWriter<std::fs::File>>,
) {
    let NpuPostArgs { bi, beta, mean_plaq, susceptibility, acceptance, dt, n_md } = *args;

    ctx.npu_tx.send(NpuRequest::FlushTrajectoryBatch).ok();
    if let Ok(NpuResponse::TrajectoryBatchProcessed { n_events }) = ctx.npu_rx.recv()
        && n_events > 0
    {
        println!("  [NPU] Sub-models: {n_events} buffered events flushed");
    }

    ctx.npu_tx
        .send(NpuRequest::Retrain {
            results: results.to_vec(),
        })
        .ok();
    npu_stats.total_npu_calls += 1;
    if let Ok(NpuResponse::Retrained { beta_c }) = ctx.npu_rx.recv() {
        println!("  ESN retrained → β_c ≈ {beta_c:.4}");
    }

    ctx.npu_tx.send(NpuRequest::SubModelMetrics).ok();
    if let Ok(NpuResponse::SubModelMetricsSnapshot(metrics)) = ctx.npu_rx.recv()
        && let Some(w) = traj_writer.as_mut()
    {
        writeln!(
            w,
            "{}",
            serde_json::json!({"beta": beta, "phase": "sub_model_metrics", "bi": bi, "sub_models": metrics})
        )
        .ok();
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
                "  [Concept Edge] β={beta:.4}: Δ_cg={delta_cg:.3} Δ_phase={delta_phase:.1} \
                 Δ_anom={delta_anomaly:.3} urgency={urgency:.3}"
            );
        }
        if let Some(w) = traj_writer.as_mut() {
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
                    println!(
                        "  [NPU] Parameter set saturated — accepting final point β={new_beta:.4} then moving on"
                    );
                    beta_order.push(new_beta);
                    npu_stats.adaptive_inserted = config.max_adaptive;
                } else {
                    println!(
                        "  NPU adaptive steer: inserting β={:.4} into scan queue ({}/{} adaptive budget)",
                        new_beta,
                        npu_stats.adaptive_inserted + 1,
                        config.max_adaptive
                    );
                    beta_order.push(new_beta);
                    npu_stats.adaptive_inserted += 1;
                }
            }
            Ok(NpuResponse::AdaptiveSteered {
                suggestion: None,
                saturated,
            }) if saturated => {
                println!("  [NPU] Parameter set saturated — no novel points remain, moving on");
                npu_stats.adaptive_inserted = config.max_adaptive;
            }
            _ => {}
        }
    } else if npu_stats.adaptive_inserted >= config.max_adaptive && results.len() >= 3 {
        println!(
            "  NPU adaptive budget exhausted ({}/{})",
            npu_stats.adaptive_inserted, config.max_adaptive
        );
    }

    if let Some(handles) = ctx.titan_handles
        && bi + 2 < beta_order.len()
    {
        let future_beta = beta_order[bi + 2];
        handles
            .titan_tx
            .send(TitanRequest::PreThermalize {
                beta: future_beta,
                mass: config.mass,
                lattice: config.lattice,
                n_quenched: config.n_quenched_pretherm,
                seed: config.seed + (bi as u64 + 2) * 1000 + 500,
                dt,
                n_md,
            })
            .ok();
        println!("  [Brain L2] Titan V pre-thermalizing β={future_beta:.4} in background");
    }

    ctx.npu_tx.send(NpuRequest::FlushTrajectoryBatch).ok();
}
