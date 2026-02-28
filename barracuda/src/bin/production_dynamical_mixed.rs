// SPDX-License-Identifier: AGPL-3.0-only

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
//! ```text
//! PRE-COMPUTATION (GPU prep)   DURING QUENCHED    DURING DYNAMICAL     POST-COMPUTATION
//! ──────────────────────────   ───────────────    ────────────────     ────────────────
//! Head 1: β priority           Head 11: q-therm  Head 3: therm detect Head 7: quality score
//! Head 2: param suggest                           Head 4: reject pred  Head 8: anomaly detect
//! Head 6: CG iter estimate                        Head 5: phase class  Head 9: next-run recommend
//! Head 10: quenched length
//! ```
//!
//! # Usage
//!
//! ```bash
//! HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_dynamical_mixed -- \
//!   --lattice=8 --betas=5.0,5.5,5.69,6.0 --mass=0.1 \
//!   --therm=200 --meas=500 --seed=42 \
//!   --bootstrap-from=meta_table_quenched.jsonl \
//!   --output=results/exp023.json \
//!   --trajectory-log=results/exp023_trajectories.jsonl \
//!   --save-weights=results/exp023_weights.json
//! ```

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_dynamical_hmc_trajectory_brain, gpu_hmc_trajectory_streaming, gpu_links_to_lattice,
    BrainInterrupt, CgResidualUpdate, GpuDynHmcState, GpuDynHmcStreamingPipelines, GpuHmcState,
    GpuHmcStreamingPipelines, GpuResidentCgBuffers, GpuResidentCgPipelines,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::md::reservoir::{
    heads, EchoStateNetwork, EsnConfig, ExportedWeights, MultiHeadNpu,
};
use hotspring_barracuda::proxy::{self, CortexRequest, ProxyFeatures};

use std::io::Write as IoWrite;
use std::sync::mpsc;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
//  Brain Layer 2: Titan V Pre-Motor Types
// ═══════════════════════════════════════════════════════════════════

/// Request sent to the Titan V pre-motor thread.
enum TitanRequest {
    PreThermalize {
        beta: f64,
        #[allow(dead_code)]
        mass: f64,
        lattice: usize,
        n_quenched: usize,
        seed: u64,
        dt: f64,
        n_md: usize,
    },
    Shutdown,
}

/// Response from the Titan V pre-motor thread.
enum TitanResponse {
    WarmConfig {
        beta: f64,
        gauge_links: Vec<f64>,
        plaquette: f64,
        wall_ms: f64,
    },
}

// ═══════════════════════════════════════════════════════════════════
//  Brain Layer 4: Attention State Machine
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionState {
    Green,
    Yellow,
    Red,
}

impl std::fmt::Display for AttentionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Green => write!(f, "GREEN"),
            Self::Yellow => write!(f, "YELLOW"),
            Self::Red => write!(f, "RED"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Meta Table Row
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct MetaRow {
    lattice: usize,
    beta: f64,
    mass: Option<f64>,
    mode: String,
    mean_plaq: f64,
    chi: f64,
    acceptance: f64,
    mean_cg_iters: f64,
    wall_s_per_traj: f64,
    n_meas: usize,
}

fn load_meta_table(path: &str) -> Vec<MetaRow> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Warning: cannot read meta table {path}: {e}");
            return Vec::new();
        }
    };
    content
        .lines()
        .filter_map(|line| serde_json::from_str::<MetaRow>(line).ok())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
//  NPU Worker — Expanded 9-Head Screening
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
struct BetaResult {
    beta: f64,
    mass: f64,
    mean_plaq: f64,
    std_plaq: f64,
    polyakov: f64,
    susceptibility: f64,
    action_density: f64,
    acceptance: f64,
    mean_cg_iters: f64,
    n_traj: usize,
    wall_s: f64,
    phase: &'static str,
    therm_used: usize,
    therm_budget: usize,
    npu_therm_early_exit: bool,
    npu_quenched_budget: usize,
    npu_quenched_used: usize,
    npu_quenched_early_exit: bool,
    npu_reject_predictions: usize,
    npu_reject_correct: usize,
    npu_anomalies: usize,
    npu_cg_check_interval: usize,
}

/// Message sent from the GPU/main thread to the NPU worker.
enum NpuRequest {
    // ─── Pre-computation ───
    PreScreenBeta {
        candidates: Vec<f64>,
        meta_context: Vec<MetaRow>,
    },
    SuggestParameters {
        lattice: usize,
        beta: f64,
        mass: f64,
    },
    PredictCgIters {
        beta: f64,
        mass: f64,
        lattice: usize,
    },
    PredictQuenchedLength {
        beta: f64,
        mass: f64,
        lattice: usize,
        meta_context: Vec<MetaRow>,
    },

    // ─── During computation (quenched phase) ───
    QuenchedThermCheck {
        plaq_window: Vec<f64>,
        beta: f64,
    },

    // ─── During computation (dynamical phase) ───
    ThermCheck {
        plaq_window: Vec<f64>,
        beta: f64,
    },
    RejectPredict {
        beta: f64,
        plaquette: f64,
        delta_h: f64,
        acceptance_rate: f64,
    },
    PhaseClassify {
        beta: f64,
        plaquette: f64,
        polyakov: f64,
        susceptibility: f64,
    },

    // ─── Post-computation ───
    QualityScore {
        result: BetaResult,
    },
    AnomalyCheck {
        plaq: f64,
        delta_h: f64,
        cg_iters: usize,
        acceptance: f64,
    },
    SteerAdaptive {
        measured_betas: Vec<f64>,
        beta_min: f64,
        beta_max: f64,
        n_candidates: usize,
    },
    RecommendNextRun {
        all_results: Vec<BetaResult>,
        meta_table: Vec<MetaRow>,
    },

    // ─── Brain Layer 1: CG residual monitoring ───
    CgResidual(CgResidualUpdate),

    // ─── Brain Layer 3: Proxy features from CPU cortex ───
    ProxyFeatures {
        beta: f64,
        level_spacing_ratio: f64,
        lambda_min: f64,
        ipr: f64,
        tier: u8,
    },

    // ─── Lifecycle ───
    Retrain {
        results: Vec<BetaResult>,
    },
    BootstrapFromMeta {
        rows: Vec<MetaRow>,
    },
    BootstrapFromWeights {
        path: String,
    },
    ExportWeights {
        path: String,
    },
    Shutdown,
}

/// Response from the NPU worker.
enum NpuResponse {
    BetaPriorities(Vec<(f64, f64)>),
    ParameterSuggestion { dt: f64, n_md: usize },
    CgEstimate(usize),
    QuenchedLengthEstimate(usize),
    QuenchedThermConverged(bool),
    ThermConverged(bool),
    RejectPrediction { likely_rejected: bool, _confidence: f64 },
    PhaseLabel(&'static str),
    Quality(f64),
    AnomalyFlag { is_anomaly: bool, _score: f64 },
    AdaptiveSteered(Option<f64>),
    NextRunPlan { betas: Vec<f64>, mass: f64, lattice: usize },
    Retrained { beta_c: f64 },
    Bootstrapped { n_points: usize },
    WeightsSaved { path: String },
    ResidualAck,
    ProxyFeaturesAck,
}

/// Spawn result includes the interrupt channel for brain-mode CG monitoring.
struct NpuWorkerHandles {
    npu_tx: mpsc::Sender<NpuRequest>,
    npu_rx: mpsc::Receiver<NpuResponse>,
    interrupt_rx: mpsc::Receiver<BrainInterrupt>,
}

fn spawn_npu_worker() -> NpuWorkerHandles {
    let (req_tx, req_rx) = mpsc::channel::<NpuRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<NpuResponse>();
    let (interrupt_tx_out, interrupt_rx_out) = mpsc::channel::<BrainInterrupt>();

    std::thread::Builder::new()
        .name("npu-cerebellum".into())
        .spawn(move || {
            let mut multi_npu: Option<MultiHeadNpu> = None;
            let interrupt_tx: Option<mpsc::Sender<BrainInterrupt>> = Some(interrupt_tx_out);

            // Brain Layer 1 + 4 state
            let mut residual_history: Vec<(usize, f64)> = Vec::new();
            let mut attention_state = AttentionState::Green;
            let mut green_count: usize = 0;
            let mut yellow_count: usize = 0;

            // Brain Layer 3 state
            #[allow(unused_assignments)]
            let mut latest_proxy: Option<ProxyFeatures> = None;

            let make_multi_esn = |seed: u64, results: &[BetaResult]| -> Option<MultiHeadNpu> {
                if results.is_empty() {
                    return None;
                }
                let (seqs, tgts) = build_training_data(results);
                let mut esn = EchoStateNetwork::new(EsnConfig {
                    input_size: 5,
                    reservoir_size: 50,
                    output_size: heads::NUM_HEADS,
                    spectral_radius: 0.95,
                    connectivity: 0.2,
                    leak_rate: 0.3,
                    regularization: 1e-3,
                    seed,
                });
                esn.train(&seqs, &tgts);
                esn.export_weights()
                    .map(|w| MultiHeadNpu::from_exported(&w))
            };

            for req in req_rx {
                match req {
                    // ─── Pre-computation heads ───
                    NpuRequest::PreScreenBeta { candidates, meta_context } => {
                        let priorities: Vec<(f64, f64)> = candidates
                            .iter()
                            .map(|&beta| {
                                let score = if let Some(ref mut npu) = multi_npu {
                                    let meta_plaq = meta_context
                                        .iter()
                                        .find(|r| (r.beta - beta).abs() < 0.01)
                                        .map(|r| r.mean_plaq)
                                        .unwrap_or(0.5);
                                    let meta_chi = meta_context
                                        .iter()
                                        .find(|r| (r.beta - beta).abs() < 0.01)
                                        .map(|r| r.chi)
                                        .unwrap_or(10.0);
                                    let input = vec![
                                        (beta - 5.0) / 2.0,
                                        meta_plaq,
                                        0.1,
                                        meta_chi / 1000.0,
                                        0.5,
                                    ];
                                    let seq = vec![input; 10];
                                    npu.predict_head(&seq, heads::BETA_PRIORITY)
                                } else {
                                    let near_bc = (-(beta - 5.69_f64).powi(2) / 0.3).exp();
                                    near_bc
                                };
                                (beta, score)
                            })
                            .collect();
                        resp_tx.send(NpuResponse::BetaPriorities(priorities)).ok();
                    }

                    NpuRequest::SuggestParameters { lattice, beta, mass } => {
                        let (dt, n_md) = if let Some(ref mut npu) = multi_npu {
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                lattice as f64 / 32.0,
                                mass,
                                0.5,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::PARAM_SUGGEST);
                            let dt_suggest = 0.001 + raw.abs() * 0.05;
                            let n_md_suggest = ((0.5 / dt_suggest).round() as usize).max(10);
                            (dt_suggest, n_md_suggest)
                        } else {
                            let vol = (lattice as f64).powi(4);
                            let scale = (4096.0 / vol).powf(0.25);
                            let dt = (0.05 * scale).max(0.001);
                            let n_md = ((0.5 / dt).round() as usize).max(20);
                            (dt, n_md)
                        };
                        resp_tx
                            .send(NpuResponse::ParameterSuggestion { dt, n_md })
                            .ok();
                    }

                    NpuRequest::PredictCgIters { beta, mass, lattice } => {
                        let est = if let Some(ref mut npu) = multi_npu {
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                mass,
                                lattice as f64 / 32.0,
                                0.0,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::CG_ESTIMATE);
                            (raw.abs() * 500.0).round() as usize
                        } else {
                            let vol = (lattice as f64).powi(4);
                            (100.0 + vol.sqrt() / mass.max(0.01)).round() as usize
                        };
                        resp_tx.send(NpuResponse::CgEstimate(est)).ok();
                    }

                    NpuRequest::PredictQuenchedLength { beta, mass, lattice, meta_context } => {
                        let steps = if let Some(ref mut npu) = multi_npu {
                            let meta_plaq = meta_context
                                .iter()
                                .find(|r| (r.beta - beta).abs() < 0.01)
                                .map(|r| r.mean_plaq)
                                .unwrap_or(0.5);
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                meta_plaq,
                                mass,
                                lattice as f64 / 32.0,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::QUENCHED_LENGTH);
                            (raw.abs() * 200.0).round().clamp(5.0, 200.0) as usize
                        } else {
                            let proximity = (-(beta - 5.69_f64).powi(2) / 0.3).exp();
                            let base = 20 + (80.0 * proximity) as usize;
                            base.min(lattice * 10)
                        };
                        resp_tx.send(NpuResponse::QuenchedLengthEstimate(steps)).ok();
                    }

                    // ─── During computation heads (quenched phase) ───
                    NpuRequest::QuenchedThermCheck { plaq_window, beta } => {
                        let converged = if let Some(ref mut npu) = multi_npu {
                            let var = plaquette_variance(&plaq_window);
                            let mean = plaq_window.iter().sum::<f64>()
                                / plaq_window.len().max(1) as f64;
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                mean,
                                var.sqrt(),
                                plaq_window.len() as f64 / 100.0,
                                1.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::QUENCHED_THERM);
                            raw > 0.5
                        } else {
                            check_thermalization_heuristic(&plaq_window)
                        };
                        resp_tx.send(NpuResponse::QuenchedThermConverged(converged)).ok();
                    }

                    // ─── During computation heads (dynamical phase) ───
                    NpuRequest::ThermCheck { plaq_window, beta } => {
                        let converged = if let Some(ref mut npu) = multi_npu {
                            let var = plaquette_variance(&plaq_window);
                            let mean = plaq_window.iter().sum::<f64>()
                                / plaq_window.len().max(1) as f64;
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                mean,
                                var.sqrt(),
                                plaq_window.len() as f64 / 200.0,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::THERM_DETECT);
                            raw > 0.5
                        } else {
                            check_thermalization_heuristic(&plaq_window)
                        };
                        resp_tx.send(NpuResponse::ThermConverged(converged)).ok();
                    }

                    NpuRequest::RejectPredict { beta, plaquette, delta_h, acceptance_rate } => {
                        let (likely_rejected, confidence) = if let Some(ref mut npu) = multi_npu {
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                plaquette,
                                delta_h / 10.0,
                                acceptance_rate,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::REJECT_PREDICT);
                            (raw > 0.5, raw.clamp(0.0, 1.0))
                        } else {
                            predict_rejection_heuristic(delta_h, acceptance_rate)
                        };
                        resp_tx
                            .send(NpuResponse::RejectPrediction { likely_rejected, _confidence: confidence })
                            .ok();
                    }

                    NpuRequest::PhaseClassify { beta, plaquette, polyakov, susceptibility } => {
                        let label = if let Some(ref mut npu) = multi_npu {
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                plaquette,
                                polyakov,
                                susceptibility / 1000.0,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::PHASE_CLASSIFY);
                            if raw > 0.6 {
                                "deconfined"
                            } else if raw > 0.3 {
                                "transition"
                            } else {
                                "confined"
                            }
                        } else if beta > 5.79 {
                            "deconfined"
                        } else if beta > 5.59 {
                            "transition"
                        } else {
                            "confined"
                        };
                        resp_tx.send(NpuResponse::PhaseLabel(label)).ok();
                    }

                    // ─── Post-computation heads ───
                    NpuRequest::QualityScore { result } => {
                        let score = if let Some(ref mut npu) = multi_npu {
                            let input = vec![
                                result.n_traj as f64 / 1000.0,
                                result.std_plaq / result.mean_plaq.abs().max(1e-10),
                                result.acceptance,
                                result.susceptibility / 1000.0,
                                result.mean_cg_iters / 500.0,
                            ];
                            let seq = vec![input; 10];
                            npu.predict_head(&seq, heads::QUALITY_SCORE).clamp(0.0, 1.0)
                        } else {
                            let acc_ok = if result.acceptance > 0.3 { 0.4 } else { 0.1 };
                            let stats_ok = if result.n_traj >= 100 { 0.3 } else { 0.1 };
                            let cg_ok = if result.mean_cg_iters < 1000.0 { 0.3 } else { 0.1 };
                            acc_ok + stats_ok + cg_ok
                        };
                        resp_tx.send(NpuResponse::Quality(score)).ok();
                    }

                    NpuRequest::AnomalyCheck { plaq, delta_h, cg_iters, acceptance } => {
                        let (is_anomaly, score) = if let Some(ref mut npu) = multi_npu {
                            let input = vec![
                                plaq,
                                delta_h / 10.0,
                                cg_iters as f64 / 500.0,
                                acceptance,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::ANOMALY_DETECT);
                            (raw > 0.7, raw.clamp(0.0, 1.0))
                        } else {
                            let anomaly =
                                delta_h.abs() > 50.0 || cg_iters > 4000 || plaq < 0.0 || plaq > 1.0;
                            let s = if anomaly { 0.9 } else { 0.1 };
                            (anomaly, s)
                        };
                        resp_tx
                            .send(NpuResponse::AnomalyFlag { is_anomaly, _score: score })
                            .ok();
                    }

                    NpuRequest::SteerAdaptive { measured_betas, beta_min, beta_max, n_candidates } => {
                        let suggestion = if let Some(ref mut npu) = multi_npu {
                            let mut best_beta = None;
                            let mut best_score = f64::NEG_INFINITY;
                            let step = (beta_max - beta_min) / (n_candidates as f64 + 1.0);
                            for ci in 1..=n_candidates {
                                let candidate = beta_min + ci as f64 * step;
                                let too_close = measured_betas
                                    .iter()
                                    .any(|&m| (m - candidate).abs() < step * 0.3);
                                if too_close {
                                    continue;
                                }
                                let input = vec![
                                    (candidate - 5.0) / 2.0,
                                    0.5,
                                    0.0,
                                    measured_betas.len() as f64 / 20.0,
                                    0.0,
                                ];
                                let seq = vec![input; 10];
                                let all = npu.predict_all_heads(&seq);
                                let priority = all[heads::BETA_PRIORITY];
                                let uncertainty = (all[heads::QUALITY_SCORE] - 0.5).abs();
                                let score = priority + uncertainty * 0.5;
                                if score > best_score {
                                    best_score = score;
                                    best_beta = Some(candidate);
                                }
                            }
                            best_beta
                        } else {
                            let gaps = find_largest_gaps(&measured_betas, beta_min, beta_max, 1);
                            gaps.into_iter().next()
                        };
                        resp_tx.send(NpuResponse::AdaptiveSteered(suggestion)).ok();
                    }

                    NpuRequest::RecommendNextRun { all_results, meta_table } => {
                        let plan = if let Some(ref mut npu) = multi_npu {
                            let last = all_results.last();
                            let input = vec![
                                all_results.len() as f64 / 10.0,
                                last.map(|r| r.mean_plaq).unwrap_or(0.5),
                                last.map(|r| r.acceptance).unwrap_or(0.5),
                                meta_table.len() as f64 / 100.0,
                                last.map(|r| r.mass).unwrap_or(0.1),
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::NEXT_RUN_RECOMMEND);
                            let suggested_beta = 5.0 + raw.abs() * 2.0;
                            let mass = last.map(|r| r.mass).unwrap_or(0.1);
                            let lattice = 8;
                            (vec![suggested_beta], mass, lattice)
                        } else {
                            let measured: Vec<f64> =
                                all_results.iter().map(|r| r.beta).collect();
                            let gaps = find_largest_gaps(&measured, 5.0, 7.0, 3);
                            let mass = all_results.first().map(|r| r.mass).unwrap_or(0.1);
                            (gaps, mass, 8)
                        };
                        resp_tx
                            .send(NpuResponse::NextRunPlan {
                                betas: plan.0,
                                mass: plan.1,
                                lattice: plan.2,
                            })
                            .ok();
                    }

                    // ─── Lifecycle ───
                    NpuRequest::Retrain { results } => {
                        let seed = 99 + results.len() as u64;
                        if let Some(new_npu) = make_multi_esn(seed, &results) {
                            multi_npu = Some(new_npu);
                        }
                        let beta_c = estimate_beta_c(&results);
                        resp_tx.send(NpuResponse::Retrained { beta_c }).ok();
                    }

                    NpuRequest::BootstrapFromMeta { rows } => {
                        let n = rows.len();
                        let results: Vec<BetaResult> = rows
                            .into_iter()
                            .map(|r| BetaResult {
                                beta: r.beta,
                                mass: r.mass.unwrap_or(0.0),
                                mean_plaq: r.mean_plaq,
                                std_plaq: 0.0,
                                polyakov: 0.0,
                                susceptibility: r.chi,
                                action_density: 6.0 * (1.0 - r.mean_plaq),
                                acceptance: r.acceptance,
                                mean_cg_iters: r.mean_cg_iters,
                                n_traj: r.n_meas,
                                wall_s: 0.0,
                                phase: if r.beta < 5.6 {
                                    "confined"
                                } else if r.beta > 5.8 {
                                    "deconfined"
                                } else {
                                    "transition"
                                },
                                therm_used: 0,
                                therm_budget: 0,
                                npu_therm_early_exit: false,
                                npu_quenched_budget: 0,
                                npu_quenched_used: 0,
                                npu_quenched_early_exit: false,
                                npu_reject_predictions: 0,
                                npu_reject_correct: 0,
                                npu_anomalies: 0,
                                npu_cg_check_interval: 10,
                            })
                            .collect();
                        if let Some(new_npu) = make_multi_esn(42, &results) {
                            multi_npu = Some(new_npu);
                        }
                        resp_tx.send(NpuResponse::Bootstrapped { n_points: n }).ok();
                    }

                    NpuRequest::BootstrapFromWeights { path } => {
                        match std::fs::read_to_string(&path) {
                            Ok(json) => match serde_json::from_str::<ExportedWeights>(&json) {
                                Ok(weights) => {
                                    multi_npu = Some(MultiHeadNpu::from_exported(&weights));
                                    resp_tx
                                        .send(NpuResponse::Bootstrapped { n_points: 0 })
                                        .ok();
                                }
                                Err(e) => {
                                    eprintln!("  Warning: failed to parse weights {path}: {e}");
                                    resp_tx
                                        .send(NpuResponse::Bootstrapped { n_points: 0 })
                                        .ok();
                                }
                            },
                            Err(e) => {
                                eprintln!("  Warning: cannot read {path}: {e}");
                                resp_tx
                                    .send(NpuResponse::Bootstrapped { n_points: 0 })
                                    .ok();
                            }
                        }
                    }

                    NpuRequest::ExportWeights { path } => {
                        let saved = if let Some(ref mut npu) = multi_npu {
                            let base = npu.base_mut();
                            let weights = ExportedWeights {
                                w_in: base.export_w_in(),
                                w_res: base.export_w_res(),
                                w_out: base.export_w_out(),
                                input_size: base.input_size(),
                                reservoir_size: base.reservoir_size(),
                                output_size: base.output_size(),
                                leak_rate: base.leak_rate(),
                            };
                            if let Some(parent) = std::path::Path::new(&path).parent() {
                                std::fs::create_dir_all(parent).ok();
                            }
                            std::fs::write(
                                &path,
                                serde_json::to_string(&weights).unwrap_or_default(),
                            )
                            .is_ok()
                        } else {
                            false
                        };
                        resp_tx
                            .send(NpuResponse::WeightsSaved {
                                path: if saved { path } else { String::new() },
                            })
                            .ok();
                    }

                    // ─── Brain Layer 1 + 4: CG residual monitoring + attention state machine ───
                    NpuRequest::CgResidual(update) => {
                        residual_history.push((update.iteration, update.rz_new));
                        if residual_history.len() > 50 {
                            residual_history.drain(..residual_history.len() - 50);
                        }

                        let anomaly_score = if let Some(ref mut npu) = multi_npu {
                            let window: Vec<f64> = residual_history
                                .iter()
                                .rev()
                                .take(10)
                                .map(|(_, rz)| rz.log10().max(-20.0) / 20.0)
                                .collect();
                            let input: Vec<f64> = window.iter().copied()
                                .chain(std::iter::repeat(0.0))
                                .take(5)
                                .collect();
                            let _ = npu.base_mut().predict_return_state(&[input]);
                            npu.base_mut().readout_head(heads::CG_RESIDUAL_MONITOR)
                        } else {
                            // Heuristic: check if residuals are increasing
                            if residual_history.len() >= 3 {
                                let recent: Vec<f64> = residual_history.iter().rev().take(3).map(|(_, rz)| *rz).collect();
                                if recent.windows(2).all(|w| w[0] >= w[1]) { 0.8 } else { 0.1 }
                            } else {
                                0.0
                            }
                        };

                        // Layer 4: Attention state transitions
                        match attention_state {
                            AttentionState::Green => {
                                if anomaly_score > 0.7 {
                                    attention_state = AttentionState::Red;
                                    yellow_count = 0;
                                    green_count = 0;
                                    // Check for actual divergence before killing
                                    if residual_history.len() >= 3 {
                                        let recent: Vec<f64> = residual_history.iter().rev().take(3).map(|(_, rz)| *rz).collect();
                                        if recent.windows(2).all(|w| w[0] >= w[1]) {
                                            if let Some(ref itx) = interrupt_tx {
                                                let _ = itx.send(BrainInterrupt::KillCg);
                                            }
                                        }
                                    }
                                } else if anomaly_score > 0.3 {
                                    attention_state = AttentionState::Yellow;
                                    green_count = 0;
                                    if let Some(ref itx) = interrupt_tx {
                                        let _ = itx.send(BrainInterrupt::AdjustCheckInterval(20));
                                    }
                                }
                            }
                            AttentionState::Yellow => {
                                if anomaly_score > 0.7 {
                                    yellow_count += 1;
                                    if yellow_count >= 2 {
                                        attention_state = AttentionState::Red;
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ = itx.send(BrainInterrupt::AdjustCheckInterval(5));
                                        }
                                    }
                                } else if anomaly_score < 0.3 {
                                    green_count += 1;
                                    if green_count >= 3 {
                                        attention_state = AttentionState::Green;
                                        green_count = 0;
                                        yellow_count = 0;
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ = itx.send(BrainInterrupt::AdjustCheckInterval(100));
                                        }
                                    }
                                } else {
                                    green_count = 0;
                                }
                            }
                            AttentionState::Red => {
                                if residual_history.len() >= 3 {
                                    let recent: Vec<f64> = residual_history.iter().rev().take(3).map(|(_, rz)| *rz).collect();
                                    let diverging = recent.windows(2).all(|w| w[0] >= w[1]);
                                    if diverging {
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ = itx.send(BrainInterrupt::KillCg);
                                        }
                                    }
                                }
                                if anomaly_score < 0.3 {
                                    green_count += 1;
                                    if green_count >= 3 {
                                        attention_state = AttentionState::Yellow;
                                        green_count = 0;
                                        yellow_count = 0;
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ = itx.send(BrainInterrupt::AdjustCheckInterval(20));
                                        }
                                    }
                                } else {
                                    green_count = 0;
                                }
                            }
                        }
                        resp_tx.send(NpuResponse::ResidualAck).ok();
                    }

                    // ─── Brain Layer 3: Proxy features from CPU cortex ───
                    NpuRequest::ProxyFeatures { beta, level_spacing_ratio, lambda_min, ipr, tier } => {
                        latest_proxy = Some(ProxyFeatures {
                            beta, level_spacing_ratio, lambda_min, ipr,
                            bandwidth: 0.0, phase: String::new(), tier, wall_ms: 0.0,
                        });
                        resp_tx.send(NpuResponse::ProxyFeaturesAck).ok();
                    }

                    NpuRequest::Shutdown => break,
                }
            }
        })
        .expect("spawn NPU worker thread");

    NpuWorkerHandles {
        npu_tx: req_tx,
        npu_rx: resp_rx,
        interrupt_rx: interrupt_rx_out,
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Brain Layer 2: Titan V Pre-Motor Worker
// ═══════════════════════════════════════════════════════════════════

struct TitanWorkerHandles {
    titan_tx: mpsc::Sender<TitanRequest>,
    titan_rx: mpsc::Receiver<TitanResponse>,
}

/// Create the Titan V GPU on the *calling* thread, then move it into a
/// worker. This avoids concurrent `wgpu::Instance` creation which can
/// deadlock the NVK/nouveau kernel driver when two GPUs are opened from
/// separate threads simultaneously.
fn spawn_titan_worker(titan_gpu: GpuF64) -> TitanWorkerHandles {
    let (req_tx, req_rx) = mpsc::channel::<TitanRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<TitanResponse>();

    let builder = std::thread::Builder::new().name("titan-premotor".into());
    builder.spawn(move || {
        let quenched_pipelines = GpuHmcStreamingPipelines::new(&titan_gpu);

        for req in req_rx {
            match req {
                TitanRequest::PreThermalize { beta, mass: _, lattice, n_quenched, seed, dt, n_md } => {
                    let t0 = Instant::now();
                    let dims = [lattice, lattice, lattice, lattice];
                    let mut lat = Lattice::hot_start(dims, beta, seed);

                    let mut cfg = HmcConfig {
                        n_md_steps: n_md,
                        dt,
                        seed: seed * 100,
                        integrator: IntegratorType::Omelyan,
                    };
                    for _ in 0..5 {
                        hmc::hmc_trajectory(&mut lat, &mut cfg);
                    }

                    let state = GpuHmcState::from_lattice(&titan_gpu, &lat, beta);
                    let mut titan_seed = seed * 200;

                    for i in 0..n_quenched {
                        let _ = gpu_hmc_trajectory_streaming(
                            &titan_gpu, &quenched_pipelines, &state,
                            n_md, dt, i as u32, &mut titan_seed,
                        );
                    }

                    gpu_links_to_lattice(&titan_gpu, &state, &mut lat);
                    let plaq = lat.average_plaquette();
                    let links = hotspring_barracuda::lattice::gpu_hmc::flatten_links(&lat);
                    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

                    eprintln!(
                        "  [Titan] Pre-therm β={beta:.4}: {n_quenched} quenched trajs, P={plaq:.6}, {wall_ms:.0}ms"
                    );

                    resp_tx.send(TitanResponse::WarmConfig {
                        beta,
                        gauge_links: links,
                        plaquette: plaq,
                        wall_ms,
                    }).ok();
                }
                TitanRequest::Shutdown => break,
            }
        }
    }).expect("spawn titan-premotor thread");

    TitanWorkerHandles { titan_tx: req_tx, titan_rx: resp_rx }
}

// ═══════════════════════════════════════════════════════════════════
//  Brain Layer 3: CPU Cortex Worker
// ═══════════════════════════════════════════════════════════════════

struct CortexWorkerHandles {
    cortex_tx: mpsc::Sender<CortexRequest>,
    proxy_rx: mpsc::Receiver<ProxyFeatures>,
}

fn spawn_cortex_worker() -> CortexWorkerHandles {
    let (req_tx, req_rx) = mpsc::channel::<CortexRequest>();
    let (feat_tx, feat_rx) = mpsc::channel::<ProxyFeatures>();

    std::thread::Builder::new()
        .name("cpu-cortex".into())
        .spawn(move || {
            let mut seed_counter = 42u64;
            for req in req_rx {
                seed_counter += 1;
                let features = proxy::anderson_3d_proxy(&req, seed_counter);
                eprintln!(
                    "  [Cortex] β={:.4}: ⟨r⟩={:.3} |λ|_min={:.3} [{}] ({:.0}ms)",
                    features.beta, features.level_spacing_ratio,
                    features.lambda_min, features.phase, features.wall_ms,
                );
                feat_tx.send(features).ok();
            }
        })
        .expect("spawn cortex worker");

    CortexWorkerHandles { cortex_tx: req_tx, proxy_rx: feat_rx }
}

// ═══════════════════════════════════════════════════════════════════
//  Heuristic fallbacks (when NPU has no trained model)
// ═══════════════════════════════════════════════════════════════════

fn check_thermalization_heuristic(plaq_window: &[f64]) -> bool {
    if plaq_window.len() < 10 {
        return false;
    }
    let n = plaq_window.len();
    let mean = plaq_window.iter().sum::<f64>() / n as f64;
    let var = plaq_window
        .iter()
        .map(|p| (p - mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;
    let half = n / 2;
    let mean_first = plaq_window[..half].iter().sum::<f64>() / half as f64;
    let mean_second = plaq_window[half..].iter().sum::<f64>() / (n - half) as f64;
    let drift = (mean_second - mean_first).abs();
    let relative_var = if mean.abs() > 1e-12 { var.sqrt() / mean.abs() } else { var.sqrt() };
    relative_var < 0.02 && drift < 0.005
}

fn predict_rejection_heuristic(delta_h: f64, acceptance_rate: f64) -> (bool, f64) {
    let rejection_score = if delta_h > 0.0 {
        1.0 - (-delta_h.abs()).exp()
    } else {
        0.0
    };
    let rate_factor = if acceptance_rate < 0.3 {
        1.2
    } else if acceptance_rate < 0.5 {
        1.0
    } else {
        0.8
    };
    let confidence = (rejection_score * rate_factor).clamp(0.0, 1.0);
    (confidence > 0.8, confidence)
}

fn estimate_beta_c(results: &[BetaResult]) -> f64 {
    if results.is_empty() {
        return 5.69;
    }
    let max_chi_r = results
        .iter()
        .max_by(|a, b| a.susceptibility.partial_cmp(&b.susceptibility).unwrap());
    max_chi_r.map(|r| r.beta).unwrap_or(5.69)
}

fn find_largest_gaps(measured: &[f64], min: f64, max: f64, n: usize) -> Vec<f64> {
    let mut sorted = measured.to_vec();
    sorted.push(min);
    sorted.push(max);
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();

    let mut gaps: Vec<(f64, f64)> = sorted.windows(2).map(|w| (w[1] - w[0], (w[0] + w[1]) / 2.0)).collect();
    gaps.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    gaps.iter().take(n).map(|g| g.1).collect()
}

fn plaquette_variance(history: &[f64]) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    history
        .iter()
        .map(|p| (p - mean).powi(2))
        .sum::<f64>()
        / (history.len() - 1) as f64
}

// ═══════════════════════════════════════════════════════════════════
//  Training data builder for 11-head ESN
// ═══════════════════════════════════════════════════════════════════

fn build_training_data(results: &[BetaResult]) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();
    let seq_len = 10;

    for r in results {
        let beta_norm = (r.beta - 5.0) / 2.0;
        let phase_val = match r.phase {
            "deconfined" => 1.0,
            "transition" => 0.5,
            _ => 0.0,
        };
        let proximity = (-(r.beta - 5.69).powi(2) / 0.1).exp();
        let anomaly_val = if r.acceptance < 0.05 || r.mean_cg_iters > 3000.0 {
            1.0
        } else {
            0.0
        };
        let quality = (r.acceptance * 0.4 + (1.0 - r.std_plaq / r.mean_plaq.abs().max(1e-10)).clamp(0.0, 1.0) * 0.3 + (r.n_traj as f64 / 1000.0).min(1.0) * 0.3).clamp(0.0, 1.0);

        let seq: Vec<Vec<f64>> = (0..seq_len)
            .map(|j| {
                let noise = 0.005 * ((j as f64) * 0.7).sin();
                vec![
                    beta_norm,
                    r.mean_plaq + noise * r.std_plaq,
                    r.mass,
                    r.susceptibility / 1000.0,
                    r.acceptance,
                ]
            })
            .collect();
        seqs.push(seq);

        let quenched_length_target = if r.npu_quenched_budget > 0 {
            r.npu_quenched_used as f64 / 200.0
        } else {
            proximity * 0.5
        };
        let quenched_therm_target = if r.npu_quenched_early_exit { 1.0 } else { 0.0 };

        // 15-head targets: one per head (matching heads::NUM_HEADS)
        targets.push(vec![
            proximity,                                     // head 0: beta priority
            0.01 + r.acceptance.abs() * 0.04,             // head 1: param suggest (dt proxy)
            if r.acceptance > 0.3 { 1.0 } else { 0.0 },  // head 2: therm converged
            1.0 - r.acceptance,                            // head 3: reject predict
            phase_val,                                     // head 4: phase classify
            r.mean_cg_iters / 500.0,                      // head 5: CG estimate
            quality,                                       // head 6: quality score
            anomaly_val,                                   // head 7: anomaly detect
            beta_norm,                                     // head 8: next-run recommend
            quenched_length_target,                        // head 9: quenched length
            quenched_therm_target,                         // head 10: quenched therm
            0.0,                                           // head 11: RMT spectral (proxy)
            phase_val,                                     // head 12: Potts phase (proxy)
            r.mean_cg_iters / 500.0,                      // head 13: Anderson CG (proxy)
            anomaly_val,                                   // head 14: CG residual monitor (brain)
        ]);
    }

    (seqs, targets)
}

fn complex_polyakov_average(lat: &Lattice) -> (f64, f64) {
    let ns = [lat.dims[0], lat.dims[1], lat.dims[2]];
    let spatial_vol = ns[0] * ns[1] * ns[2];
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for ix in 0..ns[0] {
        for iy in 0..ns[1] {
            for iz in 0..ns[2] {
                let c = lat.polyakov_loop([ix, iy, iz]);
                sum_re += c.re;
                sum_im += c.im;
            }
        }
    }
    let avg_re = sum_re / spatial_vol as f64;
    let avg_im = sum_im / spatial_vol as f64;
    ((avg_re * avg_re + avg_im * avg_im).sqrt(), avg_im.atan2(avg_re))
}

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

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--lattice=") {
            lattice = val.parse().expect("--lattice=N");
        } else if let Some(val) = arg.strip_prefix("--betas=") {
            betas = val.split(',').map(|s| s.parse().expect("beta float")).collect();
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
    }
}

// ═══════════════════════════════════════════════════════════════════
//  NPU Stats
// ═══════════════════════════════════════════════════════════════════

struct NpuStats {
    pre_screen_calls: usize,
    param_suggests: usize,
    cg_estimates: usize,
    quenched_length_predictions: usize,
    quenched_early_exits: usize,
    quenched_steps_saved: usize,
    therm_early_exits: usize,
    therm_total_saved: usize,
    reject_predictions: usize,
    reject_correct: usize,
    phase_classifications: usize,
    quality_scores: usize,
    anomaly_checks: usize,
    anomalies_found: usize,
    adaptive_steered: usize,
    adaptive_inserted: usize,
    next_run_recommendations: usize,
    total_npu_calls: usize,
}

impl NpuStats {
    fn new() -> Self {
        Self {
            pre_screen_calls: 0,
            param_suggests: 0,
            cg_estimates: 0,
            quenched_length_predictions: 0,
            quenched_early_exits: 0,
            quenched_steps_saved: 0,
            therm_early_exits: 0,
            therm_total_saved: 0,
            reject_predictions: 0,
            reject_correct: 0,
            phase_classifications: 0,
            quality_scores: 0,
            anomaly_checks: 0,
            anomalies_found: 0,
            adaptive_steered: 0,
            adaptive_inserted: 0,
            next_run_recommendations: 0,
            total_npu_calls: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args = parse_args();
    let dims = [args.lattice, args.lattice, args.lattice, args.lattice];
    let vol: usize = dims.iter().product();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Dynamical Mixed Pipeline: Resident CG + 11-Head NPU Offload  ║");
    println!("║  Experiment 023: Dynamical Fermion + NPU GPU-Prep Assist      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Lattice:  {}⁴ ({} sites)", args.lattice, vol);
    println!("  Mass:     {}", args.mass);
    println!("  β values: {:?}", args.betas);
    println!("  CG:       tol={:.0e}, max_iter={}, check_interval={}", args.cg_tol, args.cg_max_iter, args.check_interval);
    println!("  Therm:    {} dyn + {} quenched pre-therm", args.n_therm, args.n_quenched_pretherm);
    println!("  Meas:     {}", args.n_meas);
    println!("  Seed:     {}", args.seed);
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  GPU not available: {e}");
            std::process::exit(1);
        }
    };
    println!("  GPU: {}", gpu.adapter_name);

    let npu_available = hotspring_barracuda::discovery::probe_npu_available();
    println!(
        "  NPU: {} + 11-head worker thread (GPU prep + monitoring)",
        if npu_available { "AKD1000 (hardware)" } else { "MultiHeadNpu (ESN)" }
    );

    let npu_handles = spawn_npu_worker();
    let npu_tx = npu_handles.npu_tx;
    let npu_rx = npu_handles.npu_rx;
    let brain_interrupt_rx = npu_handles.interrupt_rx;
    println!("  NPU worker: spawned (15-head cerebellum: 4 pre-GPU, 5 during, 3 post, 3 proxy, 1 brain)");

    // Brain Layer 2: Titan V pre-motor
    // Create GPU device on main thread to avoid NVK driver deadlock from
    // concurrent wgpu::Instance creation across threads.
    let titan_handles = if args.no_titan {
        None
    } else {
        match rt.block_on(GpuF64::from_adapter_name("titan")) {
            Ok(titan_gpu) => {
                println!("  [Titan] GPU acquired: {}", titan_gpu.adapter_name);
                Some(spawn_titan_worker(titan_gpu))
            }
            Err(_) => {
                eprintln!("  [Titan] No secondary GPU found");
                None
            }
        }
    };
    let titan_available = titan_handles.is_some();
    println!(
        "  Titan V:    {}",
        if titan_available { "pre-motor thread spawned" } else { "not available (Layer 2 disabled)" },
    );

    // Brain Layer 3: CPU cortex (Anderson proxy)
    let cortex_handles = spawn_cortex_worker();
    println!("  CPU cortex: spawned (Anderson 3D proxy pipeline)");

    // ═══ Bootstrap from meta table or weights ═══
    if let Some(ref path) = args.bootstrap_from {
        let p = std::path::Path::new(path.as_str());
        let is_weights = p.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
            && !path.contains("jsonl");

        if is_weights {
            println!("  Bootstrap: loading ESN weights from {path}");
            npu_tx
                .send(NpuRequest::BootstrapFromWeights { path: path.clone() })
                .ok();
        } else {
            println!("  Bootstrap: training ESN from meta table {path}");
            let rows = load_meta_table(path);
            npu_tx.send(NpuRequest::BootstrapFromMeta { rows }).ok();
        }
        match npu_rx.recv() {
            Ok(NpuResponse::Bootstrapped { n_points }) => {
                println!("  Bootstrap: loaded {n_points} data points");
            }
            _ => println!("  Bootstrap: failed, starting cold"),
        }
    }
    println!();

    let vol_f = vol as f64;
    let scale = (4096.0_f64 / vol_f).powf(0.25);
    let mass_scale = (args.mass.max(0.01)).min(1.0);
    let auto_dt = (0.01 * scale * mass_scale.sqrt()).max(0.001);
    let auto_n_md = ((1.0 / auto_dt).round() as usize).max(20);
    let dt = args.dt_override.unwrap_or(auto_dt);
    let n_md = args.n_md_override.unwrap_or(auto_n_md);
    println!(
        "  HMC:      dt={dt:.4}, n_md={n_md}, traj_length={:.3}",
        dt * n_md as f64
    );
    println!();

    let quenched_pipelines = GpuHmcStreamingPipelines::new(&gpu);
    let dyn_streaming_pipelines = GpuDynHmcStreamingPipelines::new(&gpu);
    let resident_cg_pipelines = GpuResidentCgPipelines::new(&gpu);

    // Brain Layer 1: residual channel from CG solver → NPU forwarder thread
    let (brain_residual_tx, brain_residual_rx) = mpsc::channel::<CgResidualUpdate>();
    {
        let npu_fwd = npu_tx.clone();
        std::thread::Builder::new()
            .name("brain-residual-fwd".into())
            .spawn(move || {
                for update in brain_residual_rx {
                    npu_fwd.send(NpuRequest::CgResidual(update)).ok();
                }
            })
            .expect("spawn residual forwarder");
    }

    let total_start = Instant::now();
    let mut results: Vec<BetaResult> = Vec::new();
    let mut npu_stats = NpuStats::new();

    let mut traj_writer: Option<std::io::BufWriter<std::fs::File>> =
        args.trajectory_log.as_ref().map(|path| {
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let f = std::fs::File::create(path)
                .unwrap_or_else(|e| panic!("Cannot create trajectory log {path}: {e}"));
            std::io::BufWriter::new(f)
        });

    // ═══ Pre-computation: NPU screens β candidates ═══
    println!("═══ Pre-Computation: NPU β Screening ═══");
    let meta_context = args
        .bootstrap_from
        .as_ref()
        .map(|p| load_meta_table(p))
        .unwrap_or_default();

    npu_tx
        .send(NpuRequest::PreScreenBeta {
            candidates: args.betas.clone(),
            meta_context: meta_context.clone(),
        })
        .ok();
    npu_stats.pre_screen_calls += 1;
    npu_stats.total_npu_calls += 1;

    let mut beta_order: Vec<f64> = args.betas.clone();
    if let Ok(NpuResponse::BetaPriorities(priorities)) = npu_rx.recv() {
        let mut sorted_priorities = priorities;
        sorted_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        beta_order = sorted_priorities.iter().map(|p| p.0).collect();
        println!("  NPU priority order:");
        for (beta, score) in &sorted_priorities {
            println!("    β={beta:.4}: priority={score:.3}");
        }
    }
    println!();

    // ═══ Pre-computation: NPU suggests CG iterations ═══
    npu_tx
        .send(NpuRequest::PredictCgIters {
            beta: beta_order[0],
            mass: args.mass,
            lattice: args.lattice,
        })
        .ok();
    npu_stats.cg_estimates += 1;
    npu_stats.total_npu_calls += 1;
    if let Ok(NpuResponse::CgEstimate(est)) = npu_rx.recv() {
        println!("  NPU CG estimate for first β={:.4}: ~{est} iterations", beta_order[0]);
    }
    println!();

    // ═══ Scan β points ═══
    println!(
        "═══ Dynamical β-Scan ({} points × {} meas) ═══",
        beta_order.len(),
        args.n_meas,
    );

    let mut bi = 0;
    while bi < beta_order.len() {
        let beta = beta_order[bi];
        println!("── β = {:.4}, m = {} ({}/{}) ──", beta, args.mass, bi + 1, beta_order.len());

        let start = Instant::now();
        let mut lat = Lattice::hot_start(dims, beta, args.seed + bi as u64);

        // Brain Layer 2: Check if Titan V has a warm config ready for this beta
        let mut titan_warm = false;
        if let Some(ref handles) = titan_handles {
            match handles.titan_rx.try_recv() {
                Ok(TitanResponse::WarmConfig { beta: wb, gauge_links, plaquette, wall_ms }) => {
                    if (wb - beta).abs() < 0.001 {
                        hotspring_barracuda::lattice::gpu_hmc::unflatten_links_into(&mut lat, &gauge_links);
                        println!("  [Brain L2] Using Titan V warm config: P={plaquette:.6} ({wall_ms:.0}ms)");
                        titan_warm = true;
                    } else {
                        println!("  [Brain L2] Titan V config for β={wb:.4} (need {beta:.4}), discarding");
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {}
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

        let npu_quenched_budget = if let Ok(NpuResponse::QuenchedLengthEstimate(est)) = npu_rx.recv() {
            println!("  NPU quenched-length prediction: {est} steps (budget was {})", args.n_quenched_pretherm);
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
                    &gpu, &quenched_pipelines, &quenched_state, n_md, dt, i as u32, &mut seed,
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

        if let Some(NpuResponse::ParameterSuggestion { dt: sdt, n_md: snmd }) = npu_suggested_params {
            println!("  NPU param suggestion: dt={sdt:.4}, n_md={snmd} (using default dt={dt:.4}, n_md={n_md})");
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
        let plaq_var_estimate = if !results.is_empty() {
            let last = results.last().unwrap();
            last.std_plaq * last.std_plaq
        } else {
            0.05
        };
        cortex_handles.cortex_tx.send(CortexRequest {
            beta,
            mass: args.mass,
            lattice: args.lattice,
            plaq_var: plaq_var_estimate,
        }).ok();

        // Dynamical HMC setup
        let dyn_state = GpuDynHmcState::from_lattice(
            &gpu, &lat, beta, args.mass, args.cg_tol, args.cg_max_iter,
        );
        let cg_bufs = GpuResidentCgBuffers::new(
            &gpu, &dyn_streaming_pipelines.dyn_hmc, &resident_cg_pipelines, &dyn_state,
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
                &gpu, &dyn_streaming_pipelines, &resident_cg_pipelines,
                &dyn_state, &cg_bufs, n_md, dt, traj_idx as u32, &mut seed, adaptive_check_interval,
                &brain_residual_tx, &brain_interrupt_rx,
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
                &gpu, &dyn_streaming_pipelines, &resident_cg_pipelines,
                &dyn_state, &cg_bufs, n_md, dt, traj_idx as u32, &mut seed, adaptive_check_interval,
                &brain_residual_tx, &brain_interrupt_rx,
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

            let running_acc = if i > 0 { n_accepted as f64 / (i + 1) as f64 } else { 0.5 };

            // Head 4: Rejection prediction
            npu_tx
                .send(NpuRequest::RejectPredict {
                    beta, plaquette: r.plaquette,
                    delta_h: r.delta_h, acceptance_rate: running_acc,
                })
                .ok();
            npu_stats.reject_predictions += 1;
            npu_stats.total_npu_calls += 1;
            reject_predictions += 1;

            if let Ok(NpuResponse::RejectPrediction { likely_rejected, _confidence: _ }) = npu_rx.recv() {
                if likely_rejected != r.accepted {
                    npu_stats.reject_correct += 1;
                    reject_correct += 1;
                }
            }

            // Brain Layer 3: Check for proxy features from CPU cortex
            if i == 0 {
                match cortex_handles.proxy_rx.try_recv() {
                    Ok(features) => {
                        npu_tx.send(NpuRequest::ProxyFeatures {
                            beta: features.beta,
                            level_spacing_ratio: features.level_spacing_ratio,
                            lambda_min: features.lambda_min,
                            ipr: features.ipr,
                            tier: features.tier,
                        }).ok();
                        println!("  [Brain L3] Proxy features: ⟨r⟩={:.3} |λ|_min={:.3} [{}]",
                            features.level_spacing_ratio, features.lambda_min, features.phase);
                    }
                    Err(_) => {}
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

                if let Ok(NpuResponse::AnomalyFlag { is_anomaly, _score: _ }) = npu_rx.recv() {
                    if is_anomaly {
                        npu_stats.anomalies_found += 1;
                        anomalies += 1;
                    }
                }
            }

            // Polyakov readback
            let do_poly_readback = traj_writer.is_some() || (i + 1) % 100 == 0;
            let mut poly_mag = 0.0;
            let mut poly_phase = 0.0;
            if do_poly_readback {
                gpu_links_to_lattice(&gpu, &dyn_state.gauge, &mut lat);
                let (m, p) = complex_polyakov_average(&lat);
                poly_mag = m;
                poly_phase = p;
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
                beta, plaquette: mean_plaq, polyakov: mean_poly, susceptibility,
            })
            .ok();
        npu_stats.phase_classifications += 1;
        npu_stats.total_npu_calls += 1;

        let phase = match npu_rx.recv() {
            Ok(NpuResponse::PhaseLabel(l)) => l,
            _ => if beta < 5.6 { "confined" } else if beta > 5.8 { "deconfined" } else { "transition" },
        };

        // Head 7: Quality score
        let result = BetaResult {
            beta, mass: args.mass, mean_plaq, std_plaq, polyakov: mean_poly,
            susceptibility, action_density, acceptance, mean_cg_iters: mean_cg,
            n_traj: args.n_meas, wall_s, phase, therm_used, therm_budget: args.n_therm,
            npu_therm_early_exit: early_exit,
            npu_quenched_budget: npu_quenched_budget,
            npu_quenched_used: quenched_used,
            npu_quenched_early_exit: quenched_early_exit,
            npu_reject_predictions: reject_predictions,
            npu_reject_correct: reject_correct, npu_anomalies: anomalies,
            npu_cg_check_interval: adaptive_check_interval,
        };

        npu_tx
            .send(NpuRequest::QualityScore { result: result.clone() })
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
            "  ⟨P⟩={:.6}±{:.6}  |L|={:.4}  χ={:.2}  acc={:.0}%  ⟨CG⟩={:.0}  Q={quality:.2}  {phase}  {therm_info}  ({wall_s:.1}s)",
            mean_plaq, std_plaq, mean_poly, susceptibility, acceptance * 100.0, mean_cg,
        );
        if anomalies > 0 {
            println!("  ⚠ {anomalies} anomalies detected by NPU");
        }
        println!();

        // Retrain after each β point
        npu_tx
            .send(NpuRequest::Retrain { results: results.clone() })
            .ok();
        npu_stats.total_npu_calls += 1;
        if let Ok(NpuResponse::Retrained { beta_c }) = npu_rx.recv() {
            println!("  ESN retrained → β_c ≈ {beta_c:.4}");
        }

        // Adaptive steering: after 3+ points, ask NPU to fill gaps
        if results.len() >= 3 && bi + 1 < beta_order.len() {
            let measured: Vec<f64> = results.iter().map(|r| r.beta).collect();
            let remaining: Vec<f64> = beta_order[bi + 1..].to_vec();
            let beta_min = beta_order.iter().copied().fold(f64::INFINITY, f64::min) - 0.1;
            let beta_max = beta_order.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 0.1;

            npu_tx
                .send(NpuRequest::SteerAdaptive {
                    measured_betas: measured,
                    beta_min,
                    beta_max,
                    n_candidates: 40,
                })
                .ok();
            npu_stats.total_npu_calls += 1;

            npu_stats.adaptive_steered += 1;
            if let Ok(NpuResponse::AdaptiveSteered(Some(new_beta))) = npu_rx.recv() {
                let already_queued = remaining.iter().any(|&b| (b - new_beta).abs() < 0.02);
                let already_measured = results.iter().any(|r| (r.beta - new_beta).abs() < 0.02);
                if !already_queued && !already_measured {
                    println!("  NPU adaptive steer: inserting β={new_beta:.4} into scan queue");
                    beta_order.push(new_beta);
                    npu_stats.adaptive_inserted += 1;
                }
            }
        }

        // Brain Layer 2: Kick off Titan V pre-therm for the NEXT beta point
        if let Some(ref handles) = titan_handles {
            if bi + 1 < beta_order.len() {
                let next_beta = beta_order[bi + 1];
                handles.titan_tx.send(TitanRequest::PreThermalize {
                    beta: next_beta,
                    mass: args.mass,
                    lattice: args.lattice,
                    n_quenched: args.n_quenched_pretherm,
                    seed: args.seed + (bi as u64 + 1) * 1000 + 500,
                    dt,
                    n_md,
                }).ok();
                println!("  [Brain L2] Titan V pre-thermalizing β={next_beta:.4} in background");
            }
        }

        bi += 1;
    }

    // ═══ Post-computation: NPU recommends next run ═══
    println!();
    println!("═══ Post-Computation: NPU Recommendations ═══");
    npu_tx
        .send(NpuRequest::RecommendNextRun {
            all_results: results.clone(),
            meta_table: meta_context,
        })
        .ok();
    npu_stats.next_run_recommendations += 1;
    npu_stats.total_npu_calls += 1;

    if let Ok(NpuResponse::NextRunPlan { betas, mass, lattice }) = npu_rx.recv() {
        println!("  NPU recommends next run:");
        println!("    lattice: {}⁴", lattice);
        println!("    mass: {mass}");
        println!("    β values: {betas:?}");
    }

    // Flush trajectory log
    if let Some(ref mut w) = traj_writer {
        w.flush().ok();
    }

    // Save weights
    if let Some(ref path) = args.save_weights {
        npu_tx
            .send(NpuRequest::ExportWeights { path: path.clone() })
            .ok();
        match npu_rx.recv() {
            Ok(NpuResponse::WeightsSaved { path: saved }) if !saved.is_empty() => {
                println!("  ESN weights saved to: {saved}");
            }
            _ => eprintln!("  Warning: failed to save ESN weights"),
        }
    }

    // Shutdown brain threads
    if let Some(ref handles) = titan_handles {
        handles.titan_tx.send(TitanRequest::Shutdown).ok();
    }
    drop(cortex_handles.cortex_tx);
    npu_tx.send(NpuRequest::Shutdown).ok();

    // ═══ Summary ═══
    let total_wall = total_start.elapsed().as_secs_f64();
    results.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap());

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  Dynamical Mixed Pipeline Summary: {}⁴ SU(3), m={}",
        args.lattice, args.mass,
    );
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>6} {:>10}",
        "β", "⟨P⟩", "σ(P)", "|L|", "χ", "acc%", "⟨CG⟩", "phase", "time"
    );
    for r in &results {
        println!(
            "  {:>6.4} {:>10.6} {:>10.6} {:>10.4} {:>10.4} {:>7.1}% {:>8.0} {:>6} {:>9.1}s",
            r.beta, r.mean_plaq, r.std_plaq, r.polyakov, r.susceptibility,
            r.acceptance * 100.0, r.mean_cg_iters,
            &r.phase[..r.phase.len().min(6)], r.wall_s,
        );
    }
    println!();

    // NPU stats
    let total_therm_budget: usize = results.iter().map(|r| r.therm_budget).sum();
    let total_therm_used: usize = results.iter().map(|r| r.therm_used).sum();
    let therm_savings_pct = if total_therm_budget > 0 {
        (1.0 - total_therm_used as f64 / total_therm_budget as f64) * 100.0
    } else {
        0.0
    };

    let quenched_savings_pct = if npu_stats.quenched_length_predictions > 0 {
        npu_stats.quenched_steps_saved as f64 / (npu_stats.quenched_length_predictions as f64 * args.n_quenched_pretherm as f64).max(1.0) * 100.0
    } else {
        0.0
    };
    println!("  ┌──────────────────────────────────────────────────────────┐");
    println!("  │  11-Head NPU Offload Statistics                         │");
    println!("  ├──────────────────────────────────────────────────────────┤");
    println!("  │  PRE-COMPUTATION (GPU prep)                             │");
    println!("  │    β priority screens:   {:<6}                         │", npu_stats.pre_screen_calls);
    println!("  │    Parameter suggestions:{:<6}                         │", npu_stats.param_suggests);
    println!("  │    CG estimates:         {:<6}                         │", npu_stats.cg_estimates);
    println!("  │    Quenched-length preds:{:<6}                         │", npu_stats.quenched_length_predictions);
    println!("  │  DURING QUENCHED (NPU-monitored)                        │");
    println!("  │    Quenched early-exits: {:<6} ({:.1}% steps saved)    │", npu_stats.quenched_early_exits, quenched_savings_pct);
    println!("  │  DURING DYNAMICAL                                       │");
    println!("  │    Therm early-exits:    {:<6} / {:<6} ({:.1}% saved) │", npu_stats.therm_early_exits, results.len(), therm_savings_pct);
    println!("  │    Reject predictions:   {:<6} (correct: {:<6})       │", npu_stats.reject_predictions, npu_stats.reject_correct);
    println!("  │    Phase classifications:{:<6}                         │", npu_stats.phase_classifications);
    println!("  │  ADAPTIVE STEERING                                       │");
    println!("  │    Steer queries:        {:<6} (inserted: {:<6})      │", npu_stats.adaptive_steered, npu_stats.adaptive_inserted);
    println!("  │  POST-COMPUTATION                                       │");
    println!("  │    Quality scores:       {:<6}                         │", npu_stats.quality_scores);
    println!("  │    Anomaly checks:       {:<6} (found: {:<6})         │", npu_stats.anomaly_checks, npu_stats.anomalies_found);
    println!("  │    Next-run recommends:  {:<6}                         │", npu_stats.next_run_recommendations);
    println!("  │  TOTAL NPU calls:        {:<6}                         │", npu_stats.total_npu_calls);
    println!("  └──────────────────────────────────────────────────────────┘");
    println!();

    println!("  GPU: {}", gpu.adapter_name);
    println!(
        "  Total wall time: {:.1}s ({:.1} min)",
        total_wall, total_wall / 60.0,
    );
    if let Some(ref path) = args.trajectory_log {
        println!("  Trajectory log: {path}");
    }
    println!();

    if let Some(path) = args.output {
        let total_meas: usize = results.iter().map(|r| r.n_traj).sum();
        let json = serde_json::json!({
            "experiment": "023_DYNAMICAL_NPU_MIXED",
            "lattice": args.lattice,
            "dims": dims,
            "volume": vol,
            "mass": args.mass,
            "cg_tol": args.cg_tol,
            "cg_max_iter": args.cg_max_iter,
            "check_interval": args.check_interval,
            "gpu": gpu.adapter_name,
            "npu": if npu_available { "AKD1000" } else { "MultiHeadNpu" },
            "n_quenched_pretherm": args.n_quenched_pretherm,
            "n_therm_max": args.n_therm,
            "n_meas": args.n_meas,
            "seed": args.seed,
            "total_wall_s": total_wall,
            "total_measurements": total_meas,
            "npu_stats": {
                "heads": 11,
                "pre_screen_calls": npu_stats.pre_screen_calls,
                "param_suggests": npu_stats.param_suggests,
                "cg_estimates": npu_stats.cg_estimates,
                "quenched_length_predictions": npu_stats.quenched_length_predictions,
                "quenched_early_exits": npu_stats.quenched_early_exits,
                "quenched_steps_saved": npu_stats.quenched_steps_saved,
                "quenched_savings_pct": quenched_savings_pct,
                "therm_early_exits": npu_stats.therm_early_exits,
                "therm_savings_pct": therm_savings_pct,
                "reject_predictions": npu_stats.reject_predictions,
                "reject_correct": npu_stats.reject_correct,
                "phase_classifications": npu_stats.phase_classifications,
                "quality_scores": npu_stats.quality_scores,
                "anomaly_checks": npu_stats.anomaly_checks,
                "anomalies_found": npu_stats.anomalies_found,
                "adaptive_steered": npu_stats.adaptive_steered,
                "adaptive_inserted": npu_stats.adaptive_inserted,
                "next_run_recommendations": npu_stats.next_run_recommendations,
                "total_npu_calls": npu_stats.total_npu_calls,
            },
            "points": results.iter().map(|r| serde_json::json!({
                "beta": r.beta,
                "mass": r.mass,
                "mean_plaquette": r.mean_plaq,
                "std_plaquette": r.std_plaq,
                "polyakov": r.polyakov,
                "susceptibility": r.susceptibility,
                "action_density": r.action_density,
                "acceptance": r.acceptance,
                "mean_cg_iterations": r.mean_cg_iters,
                "n_trajectories": r.n_traj,
                "wall_s": r.wall_s,
                "phase": r.phase,
                "therm_used": r.therm_used,
                "therm_budget": r.therm_budget,
                "npu_therm_early_exit": r.npu_therm_early_exit,
                "npu_quenched_budget": r.npu_quenched_budget,
                "npu_quenched_used": r.npu_quenched_used,
                "npu_quenched_early_exit": r.npu_quenched_early_exit,
                "npu_reject_predictions": r.npu_reject_predictions,
                "npu_reject_correct": r.npu_reject_correct,
                "npu_anomalies": r.npu_anomalies,
                "npu_cg_check_interval": r.npu_cg_check_interval,
            })).collect::<Vec<_>>(),
        });
        if let Some(parent) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
            .unwrap_or_else(|e| eprintln!("  Failed to write {path}: {e}"));
        println!("  Results saved to: {path}");
    }
}
