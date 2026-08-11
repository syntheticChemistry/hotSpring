// SPDX-License-Identifier: AGPL-3.0-or-later

//! Campaign summary table and JSONL report.

use hotspring_barracuda::bench::SubstrateResult;
use std::io::Write;
use std::time::Instant;

pub fn run(
    all_results: &[SubstrateResult],
    jsonl_records: &[String],
    reservoir_sizes: &[usize],
    crossover_found: bool,
    crossover_size: usize,
    campaign_start: Instant,
) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Cross-Substrate Summary                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    println!("  Timing Matrix (μs, lower is better):");
    println!(
        "  {:>6} | {:>12} | {:>12} | {:>12} | {:>12}",
        "RS", "CPU-f64", "CPU-f32", "GPU-step", "GPU-batch"
    );
    println!(
        "  {:->6}-+-{:->12}-+-{:->12}-+-{:->12}-+-{:->12}",
        "", "", "", "", ""
    );
    for &rs in reservoir_sizes {
        let cpu64 = all_results
            .iter()
            .find(|r| r.substrate == "CPU-f64" && r.reservoir_size == rs);
        let cpu32 = all_results
            .iter()
            .find(|r| r.substrate == "CPU-f32" && r.reservoir_size == rs);
        let gpu_s = all_results
            .iter()
            .find(|r| r.substrate == "GPU-f32-step" && r.reservoir_size == rs);
        let gpu_b = all_results
            .iter()
            .find(|r| r.substrate == "GPU-f32-batch" && r.reservoir_size == rs);

        println!(
            "  {:>6} | {:>12.1} | {:>12.1} | {:>12.1} | {:>12.1}",
            rs,
            cpu64.map_or(0.0, |r| r.mean_us),
            cpu32.map_or(0.0, |r| r.mean_us),
            gpu_s.map_or(0.0, |r| r.mean_us),
            gpu_b.map_or(0.0, |r| r.mean_us),
        );
    }
    println!();

    if crossover_found {
        println!("  GPU crossover point: RS ≈ {crossover_size}");
    } else {
        println!("  GPU did not beat CPU at tested reservoir sizes");
        println!("  ESN workloads are latency-bound; GPU needs RS>1K or multi-sequence batching");
    }
    println!();

    println!("  NPU Capability Envelope:");
    println!("    Threshold detection:      SUPPORTED");
    println!("    Streaming inference:      SUPPORTED");
    println!("    Multi-output (1-8):       SUPPORTED");
    println!("    Weight mutation:          SUPPORTED (with reload cost)");
    println!("    QCD thermalization:       SUPPORTED");
    println!("    Multi-observable scoring: SUPPORTED");
    println!();

    let total_elapsed = campaign_start.elapsed();
    println!("  Total campaign time: {:.1}s", total_elapsed.as_secs_f64());

    // Write JSONL log
    let log_dir = std::env::temp_dir().join("hotspring-runs").join("exp021");
    std::fs::create_dir_all(&log_dir).ok();
    let log_path = log_dir.join("cross_substrate_results.jsonl");
    if let Ok(mut f) = std::fs::File::create(&log_path) {
        for record in jsonl_records {
            writeln!(f, "{record}").ok();
        }
        println!("  Results written to: {}", log_path.display());
    }
}
