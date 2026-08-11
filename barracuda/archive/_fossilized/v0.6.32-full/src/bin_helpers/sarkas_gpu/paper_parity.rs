// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper-parity validation against Choi, Dharuman, Murillo (Phys. Rev. E 100, 013206, 2019).

use hotspring_barracuda::bench::BenchReport;
use hotspring_barracuda::md::config;
use hotspring_barracuda::validation::ValidationHarness;

use super::run_for_each_case;

pub async fn run_paper_parity(
    report: &mut BenchReport,
    harness: &mut ValidationHarness,
    extended: bool,
) {
    let (desc, cases) = if extended {
        (
            "PAPER PARITY (extended): 9 PP Yukawa, N=10000, 100k production steps",
            config::paper_parity_extended_cases(),
        )
    } else {
        (
            "PAPER PARITY: 9 PP Yukawa, N=10000, 80k production steps (matches database)",
            config::paper_parity_cases(),
        )
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  Experiment 003: {}  ║",
        if extended {
            "Paper Parity Extended"
        } else {
            "Paper Parity        "
        }
    );
    println!("║  Choi, Dharuman, Murillo — Phys. Rev. E 100, 013206 (2019) ║");
    println!("║  N=10,000, same physics, consumer GPU vs HPC cluster       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  {desc}");
    println!();

    let (passed, total) = run_for_each_case(
        report,
        harness,
        &cases,
        "paper_parity_case",
        |i, total, case| {
            format!(
                "  Case {i}/{total}: {} (N={}, {}k production)",
                case.label,
                case.n_particles,
                case.prod_steps / 1000
            )
        },
        "",
    )
    .await;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PAPER PARITY RESULTS: {passed}/{total} cases passed                    ║");
    if passed == total {
        println!("║  ✅ ALL CASES PASS — consumer GPU matches HPC physics      ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
}
