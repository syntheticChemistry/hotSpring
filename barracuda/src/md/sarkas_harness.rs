// SPDX-License-Identifier: AGPL-3.0-only

//! Sarkas GPU validation harness.
//!
//! Shared logic for the `sarkas_gpu` binary: run single case with optional
//! brain state persistence, N-scaling summary printing.

use crate::bench::{peak_rss_mb, BenchReport, PhaseResult, PowerMonitor};
use crate::md::brain::MD_NUM_HEADS;
use crate::md::config::MdConfig;
use crate::md::neighbor::{AlgorithmSelector, ForceAlgorithm};
use crate::md::observables;
use crate::md::simulation::{self, BrainSummary};
use std::time::Instant;

/// Bundled brain state for cross-run transfer: Nautilus JSON only.
pub struct BrainState {
    /// Serialized Nautilus JSON for weight transfer.
    pub nautilus_json: String,
}

/// Run a single case with optional brain state import/export for cross-run persistence.
///
/// Returns `(passed, brain_summary)` so the caller can chain state.
pub async fn run_single_case_with_brain(
    config: &MdConfig,
    report: &mut BenchReport,
    brain_state: Option<&BrainState>,
) -> (bool, Option<BrainSummary>) {
    let monitor = PowerMonitor::start();
    let t0 = Instant::now();

    let selector = AlgorithmSelector::from_config(config);
    let algorithm = selector.select();
    let result = match &algorithm {
        ForceAlgorithm::AllPairs => {
            println!("  Using all-pairs mode (N={})", config.n_particles);
            simulation::run_simulation(config).await
        }
        ForceAlgorithm::CellList => {
            let box_side = config.box_side();
            let cells_per_dim = (box_side / config.rc).floor() as usize;
            println!(
                "  Using cell-list mode ({} cells/dim, N={})",
                cells_per_dim, config.n_particles
            );
            simulation::run_simulation_celllist(config).await
        }
        ForceAlgorithm::VerletList { skin } => {
            println!(
                "  Using Verlet neighbor list mode (skin={:.3}, N={})",
                skin, config.n_particles
            );
            let nautilus_json = brain_state.map(|bs| bs.nautilus_json.as_str());
            simulation::run_simulation_verlet_with_brain(config, *skin, nautilus_json).await
        }
    };

    let energy = monitor.stop();
    let wall_time = t0.elapsed().as_secs_f64();

    match result {
        Ok(sim) => {
            let energy_val = observables::validate_energy(&sim.energy_history, config);
            let ssf_device = match crate::gpu::GpuF64::new().await {
                Ok(gpu) if gpu.has_f64 => Some(gpu.to_wgpu_device()),
                _ => None,
            };
            observables::print_observable_summary_with_gpu(&sim, config, ssf_device.as_ref())
                .unwrap_or_else(|e| eprintln!("  VACF GPU failed: {e}"));

            if let Some(ref bs) = sim.brain_summary {
                if bs.retrain_count > 0 {
                    let r2_strs: Vec<String> = bs
                        .head_r2
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| **r > 0.1)
                        .map(|(i, r)| format!("H{i}={r:.3}"))
                        .collect();
                    println!(
                        "  Brain: {} retrains, {}/{} heads trusted, conf={:.2} [{}]",
                        bs.retrain_count,
                        bs.trusted_heads,
                        MD_NUM_HEADS,
                        bs.confidence,
                        if r2_strs.is_empty() {
                            "learning...".into()
                        } else {
                            r2_strs.join(", ")
                        },
                    );
                }
            }

            let total_steps = config.equil_steps + config.prod_steps;
            let brain_note = sim.brain_summary.as_ref().map_or(String::new(), |bs| {
                format!(
                    ", brain:{}/{}T conf={:.2}",
                    bs.trusted_heads, MD_NUM_HEADS, bs.confidence
                )
            });
            report.add_phase(PhaseResult {
                phase: format!("Sarkas GPU {}", config.label),
                substrate: "GPU f64 WGSL".into(),
                wall_time_s: wall_time,
                per_eval_us: wall_time * 1e6 / total_steps as f64,
                n_evals: total_steps,
                energy,
                peak_rss_mb: peak_rss_mb(),
                chi2: energy_val.drift_pct,
                precision_mev: 0.0,
                notes: format!(
                    "N={}, κ={}, Γ={}, {:.1} steps/s, drift={:.3}%{}",
                    config.n_particles,
                    config.kappa,
                    config.gamma,
                    sim.steps_per_sec,
                    energy_val.drift_pct,
                    brain_note,
                ),
            });

            (energy_val.passed, sim.brain_summary)
        }
        Err(e) => {
            println!("  ERROR: {e}");
            (false, None)
        }
    }
}

/// Run a single Sarkas case without brain persistence.
pub async fn run_single_case(config: &MdConfig, report: &mut BenchReport) -> bool {
    let (passed, _) = run_single_case_with_brain(config, report, None).await;
    passed
}

/// GPU result tuple: (N, steps/s, wall_s, drift%, watts_avg, watts_peak, joules, vram_mib, temp_c, samples).
pub type GpuNscaleRow = (usize, f64, f64, f64, f64, f64, f64, f64, f64, usize);

/// Print N-scaling summary tables (GPU performance, power/thermal, GPU vs CPU, scaling analysis).
pub fn print_n_scaling_summary(gpu_results: &[GpuNscaleRow], cpu_results: &[(usize, f64)]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  N-SCALING RESULTS                                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  GPU Performance (RTX 4070, κ=2, Γ=158, 30k production steps):");
    println!(
        "  {:>8} {:>10} {:>10} {:>8} {:>12} {:>8} {:>8} {:>10}",
        "N", "steps/s", "Wall (s)", "Drift%", "Pairs", "W(avg)", "W(peak)", "Total J"
    );
    println!(
        "  {:>8} {:>10} {:>10} {:>8} {:>12} {:>8} {:>8} {:>10}",
        "──", "──────", "───────", "─────", "─────", "──────", "──────", "──────"
    );
    for &(n, sps, wall, drift, w_avg, w_peak, joules, _vram, _temp, _samp) in gpu_results {
        println!(
            "  {:>8} {:>10.1} {:>10.1} {:>8.3} {:>12} {:>7.1}W {:>7.1}W {:>10.0}",
            n,
            sps,
            wall,
            drift,
            (n * (n - 1)) / 2,
            w_avg,
            w_peak,
            joules
        );
    }

    println!();
    println!("  GPU Power & Thermal Detail:");
    println!(
        "  {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "N", "W(avg)", "W(peak)", "Temp°C", "VRAM MB", "Samples", "J/step"
    );
    println!(
        "  {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "──", "──────", "──────", "──────", "──────", "───────", "──────"
    );
    for &(n, _sps, _wall, _drift, w_avg, w_peak, joules, vram, temp, samp) in gpu_results {
        let total_steps = 5_000 + 30_000;
        let j_per_step = if joules > 0.0 {
            joules / f64::from(total_steps)
        } else {
            0.0
        };
        println!(
            "  {n:>8} {w_avg:>7.1} {w_peak:>7.1} {temp:>7.0} {vram:>7.0} {samp:>10} {j_per_step:>8.3}"
        );
    }

    if !cpu_results.is_empty() {
        println!();
        println!("  GPU vs CPU:");
        println!(
            "  {:>8} {:>12} {:>12} {:>10}",
            "N", "GPU steps/s", "CPU steps/s", "Speedup"
        );
        println!(
            "  {:>8} {:>12} {:>12} {:>10}",
            "──", "──────────", "──────────", "───────"
        );
        for &(n, cpu_s) in cpu_results {
            if let Some(&(_, gpu_s, _, _, _, _, _, _, _, _)) = gpu_results
                .iter()
                .find(|&&(gn, _, _, _, _, _, _, _, _, _)| gn == n)
            {
                println!(
                    "  {:>8} {:>12.1} {:>12.1} {:>9.1}×",
                    n,
                    gpu_s,
                    cpu_s,
                    gpu_s / cpu_s
                );
            }
        }
    }

    if gpu_results.len() >= 2 {
        println!();
        println!("  Scaling analysis (time per step vs N):");
        let base_n = gpu_results[0].0 as f64;
        let base_sps = gpu_results[0].1;
        for &(n, sps, _, _, _, _, _, _, _, _) in gpu_results {
            let ratio_n = n as f64 / base_n;
            let ratio_time = base_sps / sps;
            let exponent = if ratio_n > 1.0 {
                ratio_time.log(ratio_n)
            } else {
                0.0
            };
            println!(
                "    N={n:>6}: {sps:.1} steps/s, time_ratio={ratio_time:.1}×, scaling exponent ~{exponent:.2}"
            );
        }
        println!("    (Perfect O(N²) = exponent 2.0)");
    }
}
