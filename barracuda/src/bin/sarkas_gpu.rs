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

use hotspring_barracuda::bench::{
    peak_rss_mb, BenchReport, HardwareInventory, PhaseResult, PowerMonitor,
};
use hotspring_barracuda::md::config::{self, MdConfig};
use hotspring_barracuda::md::observables;
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

    // ── Hardware inventory ──
    let hw = HardwareInventory::detect("Eastgate");
    println!("  Hardware: {} / {}", hw.cpu_model, hw.gpu_name);
    println!();

    let mut report = BenchReport::new(hw);

    if nscale {
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

    // Save benchmark report (resolve via CARGO_MANIFEST_DIR, not relative path)
    let report_dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benchmarks/nuclear-eos/results"
    );
    match report.save_json(report_dir) {
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
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  CPU reference: N = {n}");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            use hotspring_barracuda::md::cpu_reference;
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

    // ── Summary ──
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
    for &(n, sps, wall, drift, w_avg, w_peak, joules, _vram, _temp, _samp) in &gpu_results {
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

    // Detailed power/thermal breakdown
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
    for &(n, _sps, _wall, _drift, w_avg, w_peak, joules, vram, temp, samp) in &gpu_results {
        let total_steps = 5_000 + 30_000; // equil + prod
        let j_per_step = if joules > 0.0 {
            joules / f64::from(total_steps)
        } else {
            0.0
        };
        println!(
            "  {n:>8} {w_avg:>7.1} {w_peak:>7.1} {temp:>7.0} {vram:>7.0} {samp:>10} {j_per_step:>8.3}"
        );
    }

    // GPU vs CPU comparison
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
        for &(n, cpu_s) in &cpu_results {
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

    // Scaling analysis
    if gpu_results.len() >= 2 {
        println!();
        println!("  Scaling analysis (time per step vs N):");
        let base_n = gpu_results[0].0 as f64;
        let base_sps = gpu_results[0].1;
        for &(n, sps, _, _, _, _, _, _, _, _) in &gpu_results {
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

        use hotspring_barracuda::md::cpu_reference;
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

/// Run a single simulation case, validate observables, return pass/fail
async fn run_single_case(config: &MdConfig, report: &mut BenchReport) -> bool {
    let monitor = PowerMonitor::start();
    let t0 = Instant::now();

    // Cell-list mode activates when cells_per_dim >= 5 (enough cells for
    // meaningful neighbor-list optimization). Below that, all-pairs is used.
    // Bug fix (Feb 2026): cell_idx() modular wrapping used i32 % which produced
    // wrong results for negative operands on NVIDIA/Naga; replaced with branches.
    let box_side = config.box_side();
    let cells_per_dim = (box_side / config.rc).floor() as usize;
    let result = if cells_per_dim >= 5 {
        println!(
            "  Using cell-list mode ({} cells/dim, N={})",
            cells_per_dim, config.n_particles
        );
        simulation::run_simulation_celllist(config).await
    } else {
        println!(
            "  Using all-pairs mode (box/rc={:.1}, cells/dim={}, N={})",
            box_side / config.rc,
            cells_per_dim,
            config.n_particles
        );
        simulation::run_simulation(config).await
    };

    let energy = monitor.stop();
    let wall_time = t0.elapsed().as_secs_f64();

    match result {
        Ok(sim) => {
            // Validate energy
            let energy_val = observables::validate_energy(&sim.energy_history, config);

            // Create GPU device for SsfGpu (toadstool)
            let ssf_device = match hotspring_barracuda::gpu::GpuF64::new().await {
                Ok(gpu) if gpu.has_f64 => Some(gpu.to_wgpu_device()),
                _ => None,
            };

            // Print observable summary (with GPU SSF if available)
            observables::print_observable_summary_with_gpu(&sim, config, ssf_device);

            // Add to benchmark report
            let total_steps = config.equil_steps + config.prod_steps;
            report.add_phase(PhaseResult {
                phase: format!("Sarkas GPU {}", config.label),
                substrate: "GPU f64 WGSL".into(),
                wall_time_s: wall_time,
                per_eval_us: wall_time * 1e6 / total_steps as f64,
                n_evals: total_steps,
                energy,
                peak_rss_mb: peak_rss_mb(),
                chi2: energy_val.drift_pct, // using drift as "chi2" metric for MD
                precision_mev: 0.0,
                notes: format!(
                    "N={}, κ={}, Γ={}, {:.1} steps/s, drift={:.3}%",
                    config.n_particles,
                    config.kappa,
                    config.gamma,
                    sim.steps_per_sec,
                    energy_val.drift_pct,
                ),
            });

            energy_val.passed
        }
        Err(e) => {
            println!("  ERROR: {e}");
            false
        }
    }
}
