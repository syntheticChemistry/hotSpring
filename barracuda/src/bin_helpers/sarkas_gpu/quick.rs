// SPDX-License-Identifier: AGPL-3.0-or-later

//! Quick validation and full/long DSF sweeps.

use hotspring_barracuda::bench::BenchReport;
use hotspring_barracuda::md::config;
use hotspring_barracuda::md::sarkas_harness::run_single_case;
use hotspring_barracuda::validation::ValidationHarness;

use super::run_for_each_case;

pub async fn run_quick_validation(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Quick Validation: κ=2, Γ=158, N=500");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let config = config::quick_test_case(500);
    let passed = run_single_case(&config, report).await;
    harness.check_bool("quick_validation_k2_G158_N500", passed);
}

pub async fn run_full_sweep(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Full DSF Study: 9 PP Yukawa cases, N=2000");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let cases = config::dsf_pp_cases(2000, true);

    run_for_each_case(
        report,
        harness,
        &cases,
        "full_sweep_case",
        |i, total, case| format!("  Case {i}/{total}: {}", case.label),
        "SWEEP RESULTS",
    )
    .await;
}

pub async fn run_long_sweep(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  LONG Run: 9 PP Yukawa cases, N=2000, 80k production steps");
    println!("  Estimated: ~2.5 hours on RTX 4070");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let cases = config::dsf_pp_cases(2000, false);

    run_for_each_case(
        report,
        harness,
        &cases,
        "long_sweep_case",
        |i, total, case| format!("  Case {i}/{total}: {} (80k production steps)", case.label),
        "LONG SWEEP RESULTS",
    )
    .await;
}
