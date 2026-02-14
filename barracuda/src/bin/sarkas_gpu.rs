//! Sarkas GPU — Yukawa OCP Molecular Dynamics on Consumer GPU
//!
//! Full GPU MD simulation matching Sarkas PP Yukawa DSF study:
//!   9 cases: κ=1,2,3 × Γ=weak,medium,strong
//!   Velocity-Verlet, PBC, Berendsen thermostat
//!   All physics in f64 WGSL (SHADER_F64)
//!
//! Validates against Python Sarkas baseline:
//!   - Energy conservation (drift < 5%)
//!   - RDF: peak position, g(r)→1 tail
//!   - VACF: diffusion coefficient
//!   - SSF: S(k→0) compressibility
//!
//! Run:
//!   cargo run --release --bin sarkas_gpu              # quick validation (N=500)
//!   cargo run --release --bin sarkas_gpu -- --full    # full 9-case sweep (N=2000)
//!   cargo run --release --bin sarkas_gpu -- --scale   # scaling test (N=500→10000)

use hotspring_barracuda::bench::{
    BenchReport, HardwareInventory, PhaseResult, PowerMonitor, peak_rss_mb,
};
use hotspring_barracuda::md::config::{self, MdConfig};
use hotspring_barracuda::md::simulation;
use hotspring_barracuda::md::observables;

use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Sarkas GPU — Yukawa OCP Molecular Dynamics                ║");
    println!("║  f64 WGSL on Consumer GPU (SHADER_F64)                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let args: Vec<String> = std::env::args().collect();
    let full_sweep = args.iter().any(|a| a == "--full");
    let scale_test = args.iter().any(|a| a == "--scale");

    // ── Hardware inventory ──
    let hw = HardwareInventory::detect("Eastgate");
    println!("  Hardware: {} / {}", hw.cpu_model, hw.gpu_name);
    println!();

    let mut report = BenchReport::new(hw);

    if scale_test {
        run_scaling_test(&mut report).await;
    } else if full_sweep {
        run_full_sweep(&mut report).await;
    } else {
        run_quick_validation(&mut report).await;
    }

    // Save benchmark report
    let report_dir = "../../benchmarks/nuclear-eos/results";
    match report.save_json(report_dir) {
        Ok(path) => println!("  Benchmark report saved: {}", path),
        Err(e) => println!("  Warning: could not save report: {}", e),
    }
    println!();
    report.print_summary();
}

/// Quick validation: single case, N=500, short run
async fn run_quick_validation(report: &mut BenchReport) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Quick Validation: κ=2, Γ=158, N=500");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let config = config::quick_test_case(500);
    run_single_case(&config, report).await;
}

/// Full sweep: all 9 PP Yukawa cases at N=2000
async fn run_full_sweep(report: &mut BenchReport) {
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

        if run_single_case(case, report).await {
            passed += 1;
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  SWEEP RESULTS: {}/{} cases passed", passed, total);
    println!("═══════════════════════════════════════════════════════════");
}

/// Scaling test: GPU vs CPU at increasing N
async fn run_scaling_test(report: &mut BenchReport) {
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
            label: format!("k2_G158_N{}", n),
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
        };

        // ── GPU ──
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  GPU: N = {}", n);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        run_single_case(&config, report).await;
        // Get the last phase's steps/s
        if let Some(last) = report.phases.last() {
            let total_steps = config.equil_steps + config.prod_steps;
            let steps_per_sec = total_steps as f64 / last.wall_time_s;
            gpu_results.push((n, steps_per_sec));
        }
        println!();

        // ── CPU ──
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  CPU: N = {}", n);
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
            phase: format!("Sarkas CPU N={}", n),
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
    println!("  {:>8} {:>12} {:>12} {:>10}", "N", "GPU steps/s", "CPU steps/s", "Speedup");
    println!("  {:>8} {:>12} {:>12} {:>10}", "──", "──────────", "──────────", "───────");
    for i in 0..gpu_results.len() {
        let (n, gpu_s) = gpu_results[i];
        let (_, cpu_s) = cpu_results[i];
        let speedup = gpu_s / cpu_s;
        println!(
            "  {:>8} {:>12.1} {:>12.1} {:>9.1}×",
            n, gpu_s, cpu_s, speedup
        );
    }
}

/// Run a single simulation case, validate observables, return pass/fail
async fn run_single_case(config: &MdConfig, report: &mut BenchReport) -> bool {
    let monitor = PowerMonitor::start();
    let t0 = Instant::now();

    // Use cell list only when it provides real benefit:
    // At least 5 cells per dim needed (box/rc >= 5), which requires N > ~5000
    let box_side = config.box_side();
    let cells_per_dim = (box_side / config.rc).floor() as usize;
    let result = if cells_per_dim >= 5 {
        println!("  Using cell-list mode ({} cells/dim)", cells_per_dim);
        simulation::run_simulation_celllist(config).await
    } else {
        println!("  Using all-pairs mode (box/rc={:.1}, cells/dim={})", box_side / config.rc, cells_per_dim);
        simulation::run_simulation(config).await
    };

    let energy = monitor.stop();
    let wall_time = t0.elapsed().as_secs_f64();

    match result {
        Ok(sim) => {
            // Validate energy
            let energy_val = observables::validate_energy(&sim.energy_history, config);

            // Print observable summary
            observables::print_observable_summary(&sim, config);

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
            println!("  ERROR: {}", e);
            false
        }
    }
}
