// SPDX-License-Identifier: AGPL-3.0-or-later

//! N-scaling and GPU-vs-CPU scaling experiments.

use std::time::Instant;

use hotspring_barracuda::bench::{BenchReport, PhaseResult, PowerMonitor, peak_rss_mb};
use hotspring_barracuda::md::config::MdConfig;
use hotspring_barracuda::md::observables;
use hotspring_barracuda::md::sarkas_harness::{print_n_scaling_summary, run_single_case};
use hotspring_barracuda::validation::ValidationHarness;

pub async fn run_n_scaling(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment 001: N-Scaling on Consumer GPU                  ║");
    println!("║  κ=2, Γ=158 — Paper parity (N=10k) and beyond              ║");
    println!("║  GPU-only: any gaming GPU is a science platform             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let gpu_sizes = [500, 2_000, 5_000, 10_000, 20_000];
    let cpu_sizes = [500, 2_000];

    let mut gpu_results: Vec<(usize, f64, f64, f64, f64, f64, f64, f64, f64, usize)> = Vec::new();
    let mut cpu_results: Vec<(usize, f64)> = Vec::new();

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

pub async fn run_scaling_test(report: &mut BenchReport, harness: &mut ValidationHarness) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Scaling Test: κ=2, Γ=158 — GPU vs CPU");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let sizes = [500, 2000];

    let mut gpu_results: Vec<(usize, f64)> = Vec::new();
    let mut cpu_results: Vec<(usize, f64)> = Vec::new();

    for &n in &sizes {
        use hotspring_barracuda::md::cpu_reference;
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

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  GPU: N = {n}");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        let gpu_passed = run_single_case(&config, report).await;
        harness.check_bool(&format!("scaling_test_gpu_N{n}"), gpu_passed);
        if let Some(last) = report.phases.last() {
            let total_steps = config.equil_steps + config.prod_steps;
            let steps_per_sec = total_steps as f64 / last.wall_time_s;
            gpu_results.push((n, steps_per_sec));
        }
        println!();

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
