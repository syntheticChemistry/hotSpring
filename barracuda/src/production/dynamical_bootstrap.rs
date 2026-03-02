// SPDX-License-Identifier: AGPL-3.0-only

//! Bootstrap and pre-computation for the dynamical mixed pipeline.
//!
//! Extracted from production_dynamical_mixed to reduce binary size.

use crate::gpu::GpuF64;
use crate::lattice::gpu_hmc::{BrainInterrupt, CgResidualUpdate};
use crate::production::{
    cortex_worker::{spawn_cortex_worker, CortexWorkerHandles},
    dynamical_summary::DynamicalNpuStats,
    load_meta_table,
    npu_worker::{spawn_npu_worker, NpuRequest, NpuResponse, NpuWorkerHandles},
    titan_worker::{spawn_titan_worker, TitanRequest, TitanWorkerHandles},
    BetaResult, MetaRow,
};
use std::io::Write;
use std::process::ExitCode;
use std::sync::mpsc;

/// Bootstrap ESN from comma-separated paths: weights (.json) first, then meta/trajectory data.
/// Returns meta rows collected from non-weights paths (for use as meta_context).
pub fn run_bootstrap(
    paths_str: Option<&str>,
    npu_tx: &mpsc::Sender<NpuRequest>,
    npu_rx: &mpsc::Receiver<NpuResponse>,
) -> Vec<MetaRow> {
    let Some(paths_str) = paths_str else {
        return Vec::new();
    };

    let paths: Vec<&str> = paths_str.split(',').map(str::trim).collect();
    let mut all_meta_rows: Vec<MetaRow> = Vec::new();
    let mut weights_loaded = false;

    for path in &paths {
        let p = std::path::Path::new(path);
        let is_weights = p
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
            && !path.contains("jsonl");

        if is_weights && !weights_loaded {
            println!("  Bootstrap: loading ESN weights from {path}");
            npu_tx
                .send(NpuRequest::BootstrapFromWeights {
                    path: path.to_string(),
                })
                .ok();
            match npu_rx.recv() {
                Ok(NpuResponse::Bootstrapped { .. }) => {
                    println!("  Bootstrap: weights loaded");
                    weights_loaded = true;
                }
                _ => println!("  Bootstrap: weights failed, continuing..."),
            }
        } else if !is_weights {
            println!("  Bootstrap: loading data from {path}");
            match load_meta_table(path) {
                Ok(rows) => {
                    println!("  Bootstrap: got {} beta points", rows.len());
                    all_meta_rows.extend(rows);
                }
                Err(e) => eprintln!("  Warning: cannot read meta table {path}: {e}"),
            }
        }
    }

    if !all_meta_rows.is_empty() {
        let n = all_meta_rows.len();
        println!("  Bootstrap: training ESN from {n} combined beta points");
        npu_tx
            .send(NpuRequest::BootstrapFromMeta {
                rows: all_meta_rows.clone(),
            })
            .ok();
        match npu_rx.recv() {
            Ok(NpuResponse::Bootstrapped { n_points }) => {
                println!("  Bootstrap: trained on {n_points} data points");
            }
            _ => println!("  Bootstrap: training failed, using weights only"),
        }
    } else if !weights_loaded {
        println!("  Bootstrap: no data loaded, starting cold");
    }

    all_meta_rows
}

/// Run NPU β screening and CG estimate; returns beta order for the scan.
pub fn run_pre_computation(
    betas: &[f64],
    meta_context: &[MetaRow],
    mass: f64,
    lattice: usize,
    npu_tx: &mpsc::Sender<NpuRequest>,
    npu_rx: &mpsc::Receiver<NpuResponse>,
    npu_stats: &mut DynamicalNpuStats,
) -> Vec<f64> {
    println!("═══ Pre-Computation: NPU β Screening ═══");

    npu_tx
        .send(NpuRequest::PreScreenBeta {
            candidates: betas.to_vec(),
            meta_context: meta_context.to_vec(),
        })
        .ok();
    npu_stats.pre_screen_calls += 1;
    npu_stats.total_npu_calls += 1;

    let mut beta_order: Vec<f64> = betas.to_vec();
    if let Ok(NpuResponse::BetaPriorities(priorities)) = npu_rx.recv() {
        let mut sorted_priorities = priorities;
        sorted_priorities.sort_by(|a, b| b.1.total_cmp(&a.1));
        beta_order = sorted_priorities.iter().map(|p| p.0).collect();
        println!("  NPU priority order:");
        for (beta, score) in &sorted_priorities {
            println!("    β={beta:.4}: priority={score:.3}");
        }
    }
    println!();

    npu_tx
        .send(NpuRequest::PredictCgIters {
            beta: beta_order[0],
            mass,
            lattice,
        })
        .ok();
    npu_stats.cg_estimates += 1;
    npu_stats.total_npu_calls += 1;
    if let Ok(NpuResponse::CgEstimate(est)) = npu_rx.recv() {
        println!(
            "  NPU CG estimate for first β={:.4}: ~{est} iterations",
            beta_order[0]
        );
    }
    println!();

    beta_order
}

/// Post-computation: recommend next run, save weights, flush trajectory log, shutdown workers.
pub fn run_post_computation(
    results: &[BetaResult],
    meta_context: Vec<MetaRow>,
    save_weights: Option<&str>,
    titan_handles: Option<&TitanWorkerHandles>,
    traj_writer: &mut Option<impl Write>,
    npu_tx: &mpsc::Sender<NpuRequest>,
    npu_rx: &mpsc::Receiver<NpuResponse>,
    npu_stats: &mut DynamicalNpuStats,
) {
    println!();
    println!("═══ Post-Computation: NPU Recommendations ═══");
    npu_tx
        .send(NpuRequest::RecommendNextRun {
            all_results: results.to_vec(),
            meta_table: meta_context,
        })
        .ok();
    npu_stats.next_run_recommendations += 1;
    npu_stats.total_npu_calls += 1;

    if let Ok(NpuResponse::NextRunPlan {
        betas,
        mass,
        lattice,
    }) = npu_rx.recv()
    {
        println!("  NPU recommends next run:");
        println!("    lattice: {lattice}⁴");
        println!("    mass: {mass}");
        println!("    β values: {betas:?}");
    }

    if let Some(ref mut w) = traj_writer {
        w.flush().ok();
    }

    if let Some(path) = save_weights {
        npu_tx
            .send(NpuRequest::ExportWeights {
                path: path.to_string(),
            })
            .ok();
        match npu_rx.recv() {
            Ok(NpuResponse::WeightsSaved { path: saved }) if !saved.is_empty() => {
                println!("  ESN weights saved to: {saved}");
            }
            _ => eprintln!("  Warning: failed to save ESN weights"),
        }
    }

    if let Some(handles) = titan_handles {
        handles.titan_tx.send(TitanRequest::Shutdown).ok();
    }
    npu_tx.send(NpuRequest::Shutdown).ok();
}

/// Spawn the brain residual forwarder thread (CG residuals → NPU).
#[allow(clippy::expect_used)]
pub fn spawn_brain_residual_forwarder(
    npu_tx: mpsc::Sender<NpuRequest>,
) -> mpsc::Sender<CgResidualUpdate> {
    let (tx, rx) = mpsc::channel::<CgResidualUpdate>();
    std::thread::Builder::new()
        .name("brain-residual-fwd".into())
        .spawn(move || {
            for update in rx {
                npu_tx.send(NpuRequest::CgResidual(update)).ok();
            }
        })
        .expect("spawn residual forwarder");
    tx
}

/// Acquired GPU, NPU, Titan, and cortex workers for the dynamical pipeline.
pub struct DynamicalWorkers {
    /// Primary GPU handle (RTX 3090 or discovered).
    pub gpu: GpuF64,
    /// Sender for NPU worker requests.
    pub npu_tx: mpsc::Sender<NpuRequest>,
    /// Receiver for NPU worker responses.
    pub npu_rx: mpsc::Receiver<NpuResponse>,
    /// Receiver for brain-layer CG residual interrupts.
    pub brain_interrupt_rx: mpsc::Receiver<BrainInterrupt>,
    /// Titan V worker handles (None if unavailable).
    pub titan_handles: Option<TitanWorkerHandles>,
    /// CPU cortex worker handles.
    pub cortex_handles: CortexWorkerHandles,
    /// Whether hardware NPU is available.
    pub npu_available: bool,
}

/// Acquire GPU and spawn NPU, Titan, cortex workers. Returns Err on GPU failure.
pub fn acquire_dynamical_workers(
    rt: &tokio::runtime::Runtime,
    no_titan: bool,
) -> Result<DynamicalWorkers, ExitCode> {
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  GPU not available: {e}");
            return Err(ExitCode::from(1));
        }
    };
    println!("  GPU: {}", gpu.adapter_name);

    let npu_available = crate::discovery::probe_npu_available();
    println!(
        "  NPU: {} + 11-head worker thread (GPU prep + monitoring)",
        if npu_available {
            "AKD1000 (hardware)"
        } else {
            "MultiHeadNpu (ESN)"
        }
    );

    let NpuWorkerHandles {
        npu_tx,
        npu_rx,
        interrupt_rx: brain_interrupt_rx,
    } = spawn_npu_worker();
    println!(
        "  NPU worker: spawned (15-head cerebellum: 4 pre-GPU, 5 during, 3 post, 3 proxy, 1 brain)"
    );

    let titan_handles = if no_titan {
        None
    } else if let Ok(titan_gpu) = rt.block_on(GpuF64::from_adapter_name("titan")) {
        println!("  [Titan] GPU acquired: {}", titan_gpu.adapter_name);
        Some(spawn_titan_worker(titan_gpu))
    } else {
        eprintln!("  [Titan] No secondary GPU found");
        None
    };
    println!(
        "  Titan V:    {}",
        if titan_handles.is_some() {
            "pre-motor thread spawned"
        } else {
            "not available (Layer 2 disabled)"
        },
    );

    let cortex_handles = spawn_cortex_worker();
    println!("  CPU cortex: spawned (Anderson 3D proxy pipeline)");

    Ok(DynamicalWorkers {
        gpu,
        npu_tx,
        npu_rx,
        brain_interrupt_rx,
        titan_handles,
        cortex_handles,
        npu_available,
    })
}
