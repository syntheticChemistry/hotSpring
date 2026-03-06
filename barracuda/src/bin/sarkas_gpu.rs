// SPDX-License-Identifier: AGPL-3.0-only

//! Sarkas GPU — Yukawa OCP Molecular Dynamics on Consumer GPU
//!
//! Full GPU MD simulation matching Sarkas PP Yukawa DSF study:
//!   9 cases: κ=1,2,3 × Γ=weak,medium,strong
//!   Velocity-Verlet, PBC, Berendsen thermostat
//!   All physics in f64 WGSL (`SHADER_F64`)
//!
//! Validates against Python Sarkas baseline:
//!   - Energy conservation (drift < 5%)
//!   - RDF: peak position, g(r)→1 tail
//!   - VACF: diffusion coefficient
//!   - SSF: S(k→0) compressibility
//!
//! Run:
//!   cargo run --release --bin `sarkas_gpu`              # quick validation (N=500)
//!   cargo run --release --bin `sarkas_gpu` -- --full    # full 9-case sweep (N=2000)
//!   cargo run --release --bin `sarkas_gpu` -- --long    # long run: 9 cases, 80k steps (~71 min)
//!   cargo run --release --bin `sarkas_gpu` -- --paper   # PAPER PARITY: 9 cases, N=10000, 80k steps
//!   cargo run --release --bin `sarkas_gpu` -- --paper-ext # Extended paper: N=10000, 100k steps
//!   cargo run --release --bin `sarkas_gpu` -- --nscale  # N-scaling: 500→20000, GPU-only (~2-3h)
//!   cargo run --release --bin `sarkas_gpu` -- --scale   # quick scaling test (N=500,2000, GPU+CPU)
//!   cargo run --release --bin `sarkas_gpu` -- --brain-sweep  # persistent brain across 9 cases
//!   cargo run --release --bin `sarkas_gpu` -- --brain-nscale # persistent brain across N sizes
//!   cargo run --release --bin `sarkas_gpu` -- --brain-skin   # skin sweep to teach optimal skin

use hotspring_barracuda::bench::{
    peak_rss_mb, BenchReport, HardwareInventory, PhaseResult, PowerMonitor,
};
use hotspring_barracuda::discovery;
use hotspring_barracuda::md::brain::MD_NUM_HEADS;
use hotspring_barracuda::md::config::{self, MdConfig};
use hotspring_barracuda::md::observables;
use hotspring_barracuda::md::sarkas_harness::{
    print_n_scaling_summary, run_single_case, run_single_case_with_brain, BrainState,
};
use hotspring_barracuda::md::simulation;
use hotspring_barracuda::validation::ValidationHarness;

use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Sarkas GPU — Yukawa OCP Molecular Dynamics                ║");
    println!("║  f64 WGSL on Consumer GPU (SHADER_F64)                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("sarkas_gpu");
    let args: Vec<String> = std::env::args().collect();
    let full_sweep = args.iter().any(|a| a == "--full");
    let long_run = args.iter().any(|a| a == "--long");
    let paper_parity = args.iter().any(|a| a == "--paper");
    let paper_ext = args.iter().any(|a| a == "--paper-ext");
    let scale_test = args.iter().any(|a| a == "--scale");
    let nscale = args.iter().any(|a| a == "--nscale");
    let brain_sweep = args.iter().any(|a| a == "--brain-sweep");
    let brain_nscale = args.iter().any(|a| a == "--brain-nscale");
    let brain_skin = args.iter().any(|a| a == "--brain-skin");

    // ── Hardware inventory ──
    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string());
    let hw = HardwareInventory::detect(&hostname);
    println!("  Hardware: {} / {}", hw.cpu_model, hw.gpu_name);
    println!();

    let mut report = BenchReport::new(hw);

    if brain_sweep {
        run_brain_sweep(&mut report, &mut harness).await;
    } else if brain_nscale {
        run_brain_nscale(&mut report, &mut harness).await;
    } else if brain_skin {
        run_brain_skin_sweep(&mut report, &mut harness).await;
    } else if nscale {
        run_n_scaling(&mut report, &mut harness).await;
    } else if paper_parity {
        run_paper_parity(&mut report, &mut harness, false).await;
    } else if paper_ext {
        run_paper_parity(&mut report, &mut harness, true).await;
    } else if scale_test {
        run_scaling_test(&mut report, &mut harness).await;
    } else if long_run {
        run_long_sweep(&mut report, &mut harness).await;
    } else if full_sweep {
        run_full_sweep(&mut report, &mut harness).await;
    } else {
        run_quick_validation(&mut report, &mut harness).await;
    }

    // Save benchmark report (discovery module: HOTSPRING_DATA_ROOT / manifest parent / cwd)
    let report_dir = match discovery::benchmark_results_dir() {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(_) => discovery::paths::BENCHMARK_RESULTS.to_string(),
    };
    match report.save_json(&report_dir) {
        Ok(path) => println!("  Benchmark report saved: {path}"),
        Err(e) => println!("  Warning: could not save report: {e}"),
    }
    println!();
    report.print_summary();
    harness.finish();
}

/// Quick validation: single case, N=500, short run
async fn run_quick_validation(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Quick Validation: κ=2, Γ=158, N=500");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let config = config::quick_test_case(500);
    let passed = run_single_case(&config, report).await;
    harness.check_bool("quick_validation_k2_G158_N500", passed);
}

/// Full sweep: all 9 PP Yukawa cases at N=2000
async fn run_full_sweep(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Full DSF Study: 9 PP Yukawa cases, N=2000");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let cases = config::dsf_pp_cases(2000, true); // lite: 30k production steps

    let mut passed = 0;
    let total = cases.len();

    for (i, case) in cases.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Case {}/{}: {}", i + 1, total, case.label);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let case_passed = run_single_case(case, report).await;
        if case_passed {
            passed += 1;
        }
        harness.check_bool(&format!("full_sweep_case_{}", i + 1), case_passed);
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  SWEEP RESULTS: {passed}/{total} cases passed");
    println!("═══════════════════════════════════════════════════════════");
}

/// Long sweep: all 9 PP Yukawa cases at N=2000, 80k production steps
/// (~2.5 hours on RTX 4070, higher-fidelity observables)
async fn run_long_sweep(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  LONG Run: 9 PP Yukawa cases, N=2000, 80k production steps");
    println!("  Estimated: ~2.5 hours on RTX 4070");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let cases = config::dsf_pp_cases(2000, false); // full: 80k production steps

    let mut passed = 0;
    let total = cases.len();

    for (i, case) in cases.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  Case {}/{}: {} (80k production steps)",
            i + 1,
            total,
            case.label
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let case_passed = run_single_case(case, report).await;
        if case_passed {
            passed += 1;
        }
        harness.check_bool(&format!("long_sweep_case_{}", i + 1), case_passed);
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  LONG SWEEP RESULTS: {passed}/{total} cases passed");
    println!("═══════════════════════════════════════════════════════════");
}

/// Paper-parity validation: exact published parameters
/// Choi, Dharuman, Murillo (Phys. Rev. E 100, 013206, 2019)
/// N=10,000, 5k equil + 80k/100k production, 9 PP Yukawa cases
/// This is the headline comparison: consumer GPU vs HPC cluster
async fn run_paper_parity(
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

    let mut passed = 0;
    let total = cases.len();

    for (i, case) in cases.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  Case {}/{}: {} (N={}, {}k production)",
            i + 1,
            total,
            case.label,
            case.n_particles,
            case.prod_steps / 1000
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let case_passed = run_single_case(case, report).await;
        if case_passed {
            passed += 1;
        }
        harness.check_bool(&format!("paper_parity_case_{}", i + 1), case_passed);
        println!();
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PAPER PARITY RESULTS: {passed}/{total} cases passed                    ║");
    if passed == total {
        println!("║  ✅ ALL CASES PASS — consumer GPU matches HPC physics      ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
}

/// N-scaling experiment: GPU-only at N=500, 2000, 5000, 10000, 20000
/// Experiment 001: Demonstrates consumer GPU scaling to paper-parity (N=10k)
/// and beyond. CPU comparison at N=500 and N=2000 only (CPU is impractical
/// at larger N — hours per case on even a 24-thread workstation).
async fn run_n_scaling(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment 001: N-Scaling on Consumer GPU                  ║");
    println!("║  κ=2, Γ=158 — Paper parity (N=10k) and beyond              ║");
    println!("║  GPU-only: any gaming GPU is a science platform             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let gpu_sizes = [500, 2_000, 5_000, 10_000, 20_000];
    let cpu_sizes = [500, 2_000]; // CPU reference only where feasible

    // (N, steps/s, wall_s, drift%, watts_avg, watts_peak, gpu_joules, vram_mib, temp_c, samples)
    let mut gpu_results: Vec<(usize, f64, f64, f64, f64, f64, f64, f64, f64, usize)> = Vec::new();
    let mut cpu_results: Vec<(usize, f64)> = Vec::new(); // (N, steps/s)

    for &n in &gpu_sizes {
        let config = MdConfig {
            label: format!("nscale_N{n}"),
            n_particles: n,
            kappa: 2.0,
            gamma: 158.0,
            dt: 0.01,
            rc: 6.5,
            equil_steps: 5_000,
            prod_steps: 30_000,
            dump_step: 10,
            berendsen_tau: 5.0,
            rdf_bins: 500,
            vel_snapshot_interval: 100,
        };

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  GPU: N = {} (box = {:.1} a_ws, {} pairs)",
            n,
            config.box_side(),
            (n * (n - 1)) / 2
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let passed = run_single_case(&config, report).await;
        harness.check_bool(&format!("n_scaling_N{n}"), passed);

        if let Some(last) = report.phases.last() {
            let total_steps = config.equil_steps + config.prod_steps;
            let steps_per_sec = total_steps as f64 / last.wall_time_s;
            gpu_results.push((
                n,
                steps_per_sec,
                last.wall_time_s,
                last.chi2,
                last.energy.gpu_watts_avg,
                last.energy.gpu_watts_peak,
                last.energy.gpu_joules,
                last.energy.gpu_vram_peak_mib,
                last.energy.gpu_temp_peak_c,
                last.energy.gpu_samples,
            ));
        }
        println!();

        // CPU reference for small N only
        if cpu_sizes.contains(&n) {
            use hotspring_barracuda::md::cpu_reference;
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  CPU reference: N = {n}");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            let t0 = Instant::now();
            let monitor = PowerMonitor::start();
            let cpu_sim = cpu_reference::run_simulation_cpu(&config);
            let energy = monitor.stop();
            let wall_time = t0.elapsed().as_secs_f64();

            let total_steps = config.equil_steps + config.prod_steps;
            cpu_results.push((n, cpu_sim.steps_per_sec));

            let energy_val = observables::validate_energy(&cpu_sim.energy_history, &config);
            println!(
                "    CPU: {:.1} steps/s, drift={:.3}%, wall={:.1}s",
                cpu_sim.steps_per_sec, energy_val.drift_pct, wall_time
            );

            report.add_phase(PhaseResult {
                phase: format!("Sarkas CPU N={n}"),
                substrate: "CPU Rust f64".into(),
                wall_time_s: wall_time,
                per_eval_us: wall_time * 1e6 / total_steps as f64,
                n_evals: total_steps,
                energy,
                peak_rss_mb: peak_rss_mb(),
                chi2: energy_val.drift_pct,
                precision_mev: 0.0,
                notes: format!(
                    "N={}, {:.1} steps/s, drift={:.3}%",
                    n, cpu_sim.steps_per_sec, energy_val.drift_pct,
                ),
            });
            println!();
        }
    }

    print_n_scaling_summary(&gpu_results, &cpu_results);

    println!();
    println!("  Experiment 001 complete. See hotSpring/experiments/001_N_SCALING_GPU.md");
}

/// Scaling test: GPU vs CPU at increasing N
async fn run_scaling_test(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Scaling Test: κ=2, Γ=158 — GPU vs CPU");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let sizes = [500, 2000];

    let mut gpu_results: Vec<(usize, f64)> = Vec::new();
    let mut cpu_results: Vec<(usize, f64)> = Vec::new();

    for &n in &sizes {
        use hotspring_barracuda::md::cpu_reference;
        // Shorter run for scaling test
        let config = MdConfig {
            label: format!("k2_G158_N{n}"),
            n_particles: n,
            kappa: 2.0,
            gamma: 158.0,
            dt: 0.01,
            rc: 6.5,
            equil_steps: 500,
            prod_steps: 2_000,
            dump_step: 100,
            berendsen_tau: 5.0,
            rdf_bins: 500,
            vel_snapshot_interval: 100,
        };

        // ── GPU ──
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  GPU: N = {n}");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        let gpu_passed = run_single_case(&config, report).await;
        harness.check_bool(&format!("scaling_test_gpu_N{n}"), gpu_passed);
        // Get the last phase's steps/s
        if let Some(last) = report.phases.last() {
            let total_steps = config.equil_steps + config.prod_steps;
            let steps_per_sec = total_steps as f64 / last.wall_time_s;
            gpu_results.push((n, steps_per_sec));
        }
        println!();

        // ── CPU ──
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  CPU: N = {n}");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let t0 = Instant::now();
        let monitor = PowerMonitor::start();
        let cpu_sim = cpu_reference::run_simulation_cpu(&config);
        let energy = monitor.stop();
        let wall_time = t0.elapsed().as_secs_f64();

        let total_steps = config.equil_steps + config.prod_steps;
        cpu_results.push((n, cpu_sim.steps_per_sec));

        let energy_val = observables::validate_energy(&cpu_sim.energy_history, &config);
        println!("    Energy drift: {:.3}%", energy_val.drift_pct);

        report.add_phase(PhaseResult {
            phase: format!("Sarkas CPU N={n}"),
            substrate: "CPU Rust f64".into(),
            wall_time_s: wall_time,
            per_eval_us: wall_time * 1e6 / total_steps as f64,
            n_evals: total_steps,
            energy,
            peak_rss_mb: peak_rss_mb(),
            chi2: energy_val.drift_pct,
            precision_mev: 0.0,
            notes: format!(
                "N={}, {:.1} steps/s, drift={:.3}%",
                n, cpu_sim.steps_per_sec, energy_val.drift_pct,
            ),
        });
        println!();
    }

    // ── Summary ──
    println!("═══════════════════════════════════════════════════════════");
    println!("  SCALING SUMMARY");
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  {:>8} {:>12} {:>12} {:>10}",
        "N", "GPU steps/s", "CPU steps/s", "Speedup"
    );
    println!(
        "  {:>8} {:>12} {:>12} {:>10}",
        "──", "──────────", "──────────", "───────"
    );
    for i in 0..gpu_results.len() {
        let (n, gpu_s) = gpu_results[i];
        let (_, cpu_s) = cpu_results[i];
        let speedup = gpu_s / cpu_s;
        println!("  {n:>8} {gpu_s:>12.1} {cpu_s:>12.1} {speedup:>9.1}×");
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Experiment: Cross-case persistent brain
// ═══════════════════════════════════════════════════════════════════

/// Run all 9 Yukawa cases with brain weight persistence across cases.
/// The brain accumulates learning across different (kappa, gamma) regimes,
/// teaching it to predict regime-dependent behavior (performance, equilibration rate).
async fn run_brain_sweep(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment: Cross-Case Persistent Brain                   ║");
    println!("║  9 PP Yukawa cases, N=2000, brain weights carried forward  ║");
    println!("║  Goal: teach brain to predict across (κ,Γ) regimes         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let cases = config::dsf_pp_cases(2000, true);
    let mut brain_state: Option<BrainState> = None;
    let mut passed = 0;
    let total = cases.len();

    let mut evolution: Vec<(String, usize, f64, Vec<f64>, usize, usize)> = Vec::new();

    for (i, config) in cases.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  Case {}/{}: {} [brain: {}]",
            i + 1,
            total,
            config.label,
            if brain_state.is_some() {
                "inherited"
            } else {
                "fresh"
            }
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let (ok, summary) = run_single_case_with_brain(config, report, brain_state.as_ref()).await;
        harness.check_bool(&format!("brain_sweep_{}", config.label), ok);
        if ok {
            passed += 1;
        }

        if let Some(bs) = summary {
            evolution.push((
                config.label.clone(),
                bs.trusted_heads,
                bs.confidence,
                bs.head_r2.clone(),
                bs.nautilus_observations,
                bs.nautilus_generations,
            ));
            brain_state = bs.nautilus_json.map(|json| BrainState {
                nautilus_json: json,
            });
        }
        println!();
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BRAIN EVOLUTION ACROSS CASES                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  {:>12} {:>8} {:>8} {:>8} {:>6}  Best heads",
        "Case", "Trusted", "Conf", "NautObs", "Gens"
    );
    println!(
        "  {:>12} {:>8} {:>8} {:>8} {:>6}  ─────────",
        "────", "───────", "────", "───────", "────"
    );
    for (label, trusted, conf, r2s, n_obs, n_gens) in &evolution {
        let best: Vec<String> = r2s
            .iter()
            .enumerate()
            .filter(|(_, r)| **r > 0.1)
            .map(|(i, r)| format!("H{i}={r:.3}"))
            .collect();
        println!(
            "  {:>12} {:>5}/{} {:>8.3} {:>8} {:>6}  [{}]",
            label,
            trusted,
            MD_NUM_HEADS,
            conf,
            n_obs,
            n_gens,
            if best.is_empty() {
                "learning...".into()
            } else {
                best.join(", ")
            }
        );
    }
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  BRAIN SWEEP: {passed}/{total} cases passed");
    println!("═══════════════════════════════════════════════════════════");
}

// ═══════════════════════════════════════════════════════════════════
//  Experiment: N-scaling persistent brain
// ═══════════════════════════════════════════════════════════════════

/// Run `k2_G158` at increasing N with brain weight persistence.
/// Teaches the brain to predict throughput and rebuild frequency as a function of N.
async fn run_brain_nscale(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment: N-Scaling Persistent Brain                    ║");
    println!("║  κ=2, Γ=158 at N=500→10000, brain weights carried forward ║");
    println!("║  Goal: teach brain steps/s prediction and rebuild scaling  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n_sizes = [500, 1000, 2000, 5000, 10_000];
    let mut brain_state: Option<BrainState> = None;
    let mut passed = 0;
    let total = n_sizes.len();

    let mut evolution: Vec<(usize, usize, f64, Vec<f64>, usize, usize)> = Vec::new();

    for (i, &n) in n_sizes.iter().enumerate() {
        let config = MdConfig {
            label: format!("brain_nscale_N{n}"),
            n_particles: n,
            kappa: 2.0,
            gamma: 158.0,
            dt: 0.01,
            rc: 6.5,
            equil_steps: 5_000,
            prod_steps: 30_000,
            dump_step: 10,
            berendsen_tau: 5.0,
            rdf_bins: 500,
            vel_snapshot_interval: 100,
        };

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  N = {} ({}/{}) [brain: {}]",
            n,
            i + 1,
            total,
            if brain_state.is_some() {
                "inherited"
            } else {
                "fresh"
            }
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let (ok, summary) = run_single_case_with_brain(&config, report, brain_state.as_ref()).await;
        harness.check_bool(&format!("brain_nscale_N{n}"), ok);
        if ok {
            passed += 1;
        }

        if let Some(bs) = summary {
            evolution.push((
                n,
                bs.trusted_heads,
                bs.confidence,
                bs.head_r2.clone(),
                bs.nautilus_observations,
                bs.nautilus_generations,
            ));
            brain_state = bs.nautilus_json.map(|json| BrainState {
                nautilus_json: json,
            });
        }
        println!();
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BRAIN EVOLUTION ACROSS N                                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  {:>8} {:>8} {:>8} {:>8} {:>6}  Best heads",
        "N", "Trusted", "Conf", "NautObs", "Gens"
    );
    println!(
        "  {:>8} {:>8} {:>8} {:>8} {:>6}  ─────────",
        "──", "───────", "────", "───────", "────"
    );
    for (n, trusted, conf, r2s, n_obs, n_gens) in &evolution {
        let best: Vec<String> = r2s
            .iter()
            .enumerate()
            .filter(|(_, r)| **r > 0.1)
            .map(|(i, r)| format!("H{i}={r:.3}"))
            .collect();
        println!(
            "  {:>8} {:>5}/{} {:>8.3} {:>8} {:>6}  [{}]",
            n,
            trusted,
            MD_NUM_HEADS,
            conf,
            n_obs,
            n_gens,
            if best.is_empty() {
                "learning...".into()
            } else {
                best.join(", ")
            }
        );
    }
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  BRAIN N-SCALE: {passed}/{total} cases passed");
    println!("═══════════════════════════════════════════════════════════");
}

// ═══════════════════════════════════════════════════════════════════
//  Experiment: Skin fraction sweep
// ═══════════════════════════════════════════════════════════════════

/// Run `k2_G158` N=2000 with varying Verlet skin fractions.
/// Teaches the brain the relationship between skin, rebuilds, and throughput
/// so it can learn to steer C0 (optimal skin) at runtime.
async fn run_brain_skin_sweep(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment: Skin Fraction Sweep with Persistent Brain     ║");
    println!("║  κ=2, Γ=158, N=2000 — skin from 0.05 to 0.40             ║");
    println!("║  Goal: teach brain optimal skin steering (C0)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let skin_fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40];
    let mut brain_state: Option<BrainState> = None;
    let mut passed = 0;
    let total = skin_fractions.len();

    let mut evolution: Vec<(f64, f64, usize, usize, f64, Vec<f64>, usize, usize)> = Vec::new();

    for (i, &sf) in skin_fractions.iter().enumerate() {
        let rc = 6.5;
        let skin = sf * rc;
        let config = MdConfig {
            label: format!("brain_skin_{:.0}pct", sf * 100.0),
            n_particles: 2000,
            kappa: 2.0,
            gamma: 158.0,
            dt: 0.01,
            rc,
            equil_steps: 5_000,
            prod_steps: 30_000,
            dump_step: 10,
            berendsen_tau: 5.0,
            rdf_bins: 500,
            vel_snapshot_interval: 100,
        };

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "  skin_fraction={:.2} (skin={:.3} a_ws) [{}/{}] [brain: {}]",
            sf,
            skin,
            i + 1,
            total,
            if brain_state.is_some() {
                "inherited"
            } else {
                "fresh"
            }
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let monitor = PowerMonitor::start();
        let t0 = Instant::now();
        let nautilus_json = brain_state.as_ref().map(|bs| bs.nautilus_json.as_str());
        let result =
            simulation::run_simulation_verlet_with_brain(&config, skin, nautilus_json).await;
        let energy = monitor.stop();
        let wall_time = t0.elapsed().as_secs_f64();

        match result {
            Ok(sim) => {
                let energy_val = observables::validate_energy(&sim.energy_history, &config);
                let ssf_device = match hotspring_barracuda::gpu::GpuF64::new().await {
                    Ok(gpu) if gpu.has_f64 => Some(gpu.to_wgpu_device()),
                    _ => None,
                };
                observables::print_observable_summary_with_gpu(&sim, &config, ssf_device.as_ref())
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
                    evolution.push((
                        sf,
                        sim.steps_per_sec,
                        bs.retrain_count,
                        bs.trusted_heads,
                        bs.confidence,
                        bs.head_r2.clone(),
                        bs.nautilus_observations,
                        bs.nautilus_generations,
                    ));
                    brain_state = bs.nautilus_json.clone().map(|json| BrainState {
                        nautilus_json: json,
                    });
                }

                let total_steps = config.equil_steps + config.prod_steps;
                report.add_phase(PhaseResult {
                    phase: format!("Sarkas GPU skin={:.0}%", sf * 100.0),
                    substrate: "GPU f64 WGSL".into(),
                    wall_time_s: wall_time,
                    per_eval_us: wall_time * 1e6 / total_steps as f64,
                    n_evals: total_steps,
                    energy,
                    peak_rss_mb: peak_rss_mb(),
                    chi2: energy_val.drift_pct,
                    precision_mev: 0.0,
                    notes: format!(
                        "skin={:.2}, {:.1} steps/s, drift={:.3}%",
                        sf, sim.steps_per_sec, energy_val.drift_pct
                    ),
                });

                harness.check_bool(&format!("brain_skin_{sf:.2}"), energy_val.passed);
                if energy_val.passed {
                    passed += 1;
                }
            }
            Err(e) => {
                println!("  ERROR: {e}");
                harness.check_bool(&format!("brain_skin_{sf:.2}"), false);
            }
        }
        println!();
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BRAIN EVOLUTION ACROSS SKIN FRACTIONS                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  {:>6} {:>10} {:>8} {:>8} {:>8} {:>6}  Best heads",
        "Skin%", "Steps/s", "Trusted", "Conf", "NautObs", "Gens"
    );
    println!(
        "  {:>6} {:>10} {:>8} {:>8} {:>8} {:>6}  ─────────",
        "─────", "───────", "───────", "────", "───────", "────"
    );
    for (sf, sps, _retrains, trusted, conf, r2s, n_obs, n_gens) in &evolution {
        let best: Vec<String> = r2s
            .iter()
            .enumerate()
            .filter(|(_, r)| **r > 0.1)
            .map(|(i, r)| format!("H{i}={r:.3}"))
            .collect();
        println!(
            "  {:>5.0}% {:>10.1} {:>5}/{} {:>8.3} {:>8} {:>6}  [{}]",
            sf * 100.0,
            sps,
            trusted,
            MD_NUM_HEADS,
            conf,
            n_obs,
            n_gens,
            if best.is_empty() {
                "learning...".into()
            } else {
                best.join(", ")
            }
        );
    }
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  BRAIN SKIN SWEEP: {passed}/{total} cases passed");
    println!("═══════════════════════════════════════════════════════════");
}
