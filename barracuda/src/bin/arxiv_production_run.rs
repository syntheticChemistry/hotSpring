// SPDX-License-Identifier: AGPL-3.0-or-later

//! arXiv Production Run — SU(2) lattice QCD plaquette data at β=2.3
//!
//! Generates publication-quality data for:
//!   Section 3.2: Plaquette ⟨P⟩ (DF64 GPU vs f64 CPU)
//!   Section 3.5: Autocorrelation τ_int time series
//!
//! Outputs per-trajectory plaquette values for statistical analysis.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, ResidentObservableBuffers,
    gpu_hmc_trajectory_streaming_cpu_mom, plaquette_resident,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

struct CpuResult {
    plaqs: Vec<f64>,
    acceptance_rate: f64,
    delta_h_mean: f64,
}

fn production_run_cpu(dims: [usize; 4], beta: f64, n_therm: usize, n_prod: usize) -> CpuResult {
    let mut lat = Lattice::hot_start(dims, beta, 42);
    let cfg = &mut HmcConfig {
        n_md_steps: 20,
        dt: 0.02,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };

    // Thermalize
    let mut therm_accepted = 0usize;
    for _ in 0..n_therm {
        let r = hmc::hmc_trajectory(&mut lat, cfg);
        if r.accepted {
            therm_accepted += 1;
        }
    }
    let therm_rate = therm_accepted as f64 / n_therm as f64;
    println!("    Thermalization acceptance: {therm_accepted}/{n_therm} ({:.1}%)", therm_rate * 100.0);

    // Production
    let mut plaqs = Vec::with_capacity(n_prod);
    let mut accepted = 0usize;
    let mut delta_h_sum = 0.0f64;
    for _ in 0..n_prod {
        let r = hmc::hmc_trajectory(&mut lat, cfg);
        plaqs.push(r.plaquette);
        if r.accepted {
            accepted += 1;
        }
        delta_h_sum += r.delta_h.abs();
    }

    CpuResult {
        plaqs,
        acceptance_rate: accepted as f64 / n_prod as f64,
        delta_h_mean: delta_h_sum / n_prod as f64,
    }
}

struct GpuResult {
    plaqs: Vec<f64>,
    ms_per_traj: f64,
    acceptance_rate: f64,
}

fn production_run_gpu(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    dims: [usize; 4],
    beta: f64,
    n_therm: usize,
    n_prod: usize,
) -> GpuResult {
    let mut lat = Lattice::hot_start(dims, beta, 42);
    let cfg = &mut HmcConfig {
        n_md_steps: 20,
        dt: 0.02,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };

    // Thermalize on CPU (same starting point as CPU reference)
    for _ in 0..n_therm {
        hmc::hmc_trajectory(&mut lat, cfg);
    }

    // Upload thermalized lattice to GPU
    let state = GpuHmcState::from_lattice(gpu, &lat, beta);

    // Production on GPU with CPU-generated momenta (validated correct path)
    let mut plaqs = Vec::with_capacity(n_prod);
    let mut accepted = 0usize;
    let mut seed = 7777u64;
    let start = Instant::now();
    for _ in 0..n_prod {
        let r = gpu_hmc_trajectory_streaming_cpu_mom(gpu, pipelines, &state, 20, 0.02, &mut seed)
            .expect("GPU HMC trajectory failed");
        plaqs.push(r.plaquette);
        if r.accepted {
            accepted += 1;
        }
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    GpuResult {
        plaqs,
        ms_per_traj: elapsed_ms / n_prod as f64,
        acceptance_rate: accepted as f64 / n_prod as f64,
    }
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_error(data: &[f64]) -> f64 {
    let m = mean(data);
    let var: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    (var / data.len() as f64).sqrt()
}

fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let m = mean(data);
    let var: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    if var < 1e-15 {
        return vec![1.0; max_lag];
    }
    let n = data.len();
    (0..max_lag)
        .map(|t| {
            let sum: f64 = (0..n - t).map(|i| (data[i] - m) * (data[i + t] - m)).sum();
            sum / ((n - t) as f64 * var)
        })
        .collect()
}

fn integrated_autocorrelation(acf: &[f64]) -> f64 {
    // Madras-Sokal automatic windowing
    let mut tau_int = 0.5;
    for (t, &rho) in acf.iter().enumerate().skip(1) {
        if rho < 0.0 {
            break;
        }
        tau_int += rho;
        // Sokal criterion: cut when t > 6*tau_int
        if t as f64 > 6.0 * tau_int {
            break;
        }
    }
    tau_int
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  arXiv Production Run — SU(2) Plaquette at β=2.3           ║");
    println!("║  Omelyan integrator, n_md=20, dt=0.02                      ║");
    println!("║  Target: whitePaper/subGen/LATTICE_QCD_CONSUMER_GPU_ARXIV   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU not available: {e}");
            return;
        }
    };
    println!("  GPU: {}", gpu.adapter_name);
    println!();

    let beta = 2.3;
    let n_therm = 200;
    let n_prod = 200;

    let configs: Vec<(&str, [usize; 4])> = vec![("8⁴", [8, 8, 8, 8])];

    let pipelines = GpuHmcStreamingPipelines::new(&gpu);

    // ═══ DIAGNOSTIC: same-lattice plaquette comparison ═══
    println!("═══ DIAGNOSTIC: Same-lattice plaquette (CPU vs GPU, no HMC) ═══");
    {
        let test_lat = Lattice::cold_start([4, 4, 4, 4], beta);
        let cpu_plaq = test_lat.average_plaquette();
        let test_state = GpuHmcState::from_lattice(&gpu, &test_lat, beta);
        let obs = ResidentObservableBuffers::new(&gpu, &pipelines.reduce_pipeline, &test_state);
        let gpu_plaq = plaquette_resident(
            &gpu, &pipelines.hmc, &test_state, &pipelines.reduce_pipeline, &obs,
        ).expect("plaquette readback");
        println!("  Cold start: CPU={cpu_plaq:.15}, GPU={gpu_plaq:.15}");
        println!("  Δ = {:.2e}", (cpu_plaq - gpu_plaq).abs());
    }
    {
        let test_lat = Lattice::hot_start([4, 4, 4, 4], beta, 42);
        let cpu_plaq = test_lat.average_plaquette();
        let test_state = GpuHmcState::from_lattice(&gpu, &test_lat, beta);
        let obs = ResidentObservableBuffers::new(&gpu, &pipelines.reduce_pipeline, &test_state);
        let gpu_plaq = plaquette_resident(
            &gpu, &pipelines.hmc, &test_state, &pipelines.reduce_pipeline, &obs,
        ).expect("plaquette readback");
        println!("  Hot  start: CPU={cpu_plaq:.15}, GPU={gpu_plaq:.15}");
        println!("  Δ = {:.2e}", (cpu_plaq - gpu_plaq).abs());
    }
    {
        let mut test_lat = Lattice::hot_start([4, 4, 4, 4], beta, 42);
        let cfg = &mut HmcConfig { n_md_steps: 20, dt: 0.02, seed: 42, integrator: IntegratorType::Omelyan };
        for _ in 0..200 { hmc::hmc_trajectory(&mut test_lat, cfg); }
        let cpu_plaq = test_lat.average_plaquette();
        let test_state = GpuHmcState::from_lattice(&gpu, &test_lat, beta);
        let obs = ResidentObservableBuffers::new(&gpu, &pipelines.reduce_pipeline, &test_state);
        let gpu_plaq = plaquette_resident(
            &gpu, &pipelines.hmc, &test_state, &pipelines.reduce_pipeline, &obs,
        ).expect("plaquette readback");
        println!("  Thermalized: CPU={cpu_plaq:.15}, GPU={gpu_plaq:.15}");
        println!("  Δ = {:.2e}", (cpu_plaq - gpu_plaq).abs());
    }
    println!();

    for (label, dims) in &configs {
        let vol: usize = dims.iter().product();
        println!("═══ {label} (V={vol}, β={beta}, therm={n_therm}, prod={n_prod}) ═══");
        println!();

        // CPU reference (f64 native)
        println!("  Running CPU (f64 native)...");
        let cpu_start = Instant::now();
        let cpu_result = production_run_cpu(*dims, beta, n_therm, n_prod);
        let cpu_elapsed = cpu_start.elapsed().as_secs_f64();
        let cpu_mean = mean(&cpu_result.plaqs);
        let cpu_se = std_error(&cpu_result.plaqs);
        println!("  CPU: ⟨P⟩ = {cpu_mean:.10} ± {cpu_se:.2e} ({cpu_elapsed:.1}s)");
        println!("  CPU: acceptance = {:.1}%, ⟨|ΔH|⟩ = {:.4e}", cpu_result.acceptance_rate * 100.0, cpu_result.delta_h_mean);

        // GPU — corrected path (CPU momenta, GPU MD)
        println!("  Running GPU (CPU momenta, GPU MD)...");
        let gpu_result = production_run_gpu(&gpu, &pipelines, *dims, beta, n_therm, n_prod);
        let gpu_mean = mean(&gpu_result.plaqs);
        let gpu_se = std_error(&gpu_result.plaqs);
        println!("  GPU: ⟨P⟩ = {gpu_mean:.10} ± {gpu_se:.2e} ({:.1} ms/traj)", gpu_result.ms_per_traj);
        println!("  GPU: acceptance = {:.1}%", gpu_result.acceptance_rate * 100.0);

        // Agreement
        let delta = (gpu_mean - cpu_mean).abs();
        let combined_se = (cpu_se.powi(2) + gpu_se.powi(2)).sqrt();
        let sigma_ratio = if combined_se > 0.0 {
            delta / combined_se
        } else {
            0.0
        };
        println!("  |Δ|/σ = {sigma_ratio:.2}");
        println!();

        // Autocorrelation (GPU time series)
        let acf = autocorrelation(&gpu_result.plaqs, 100.min(n_prod / 4));
        let tau_int = integrated_autocorrelation(&acf);
        let n_eff = n_prod as f64 / (2.0 * tau_int);
        println!("  Autocorrelation: τ_int = {tau_int:.2}, N_eff = {n_eff:.0}");
        println!();

        // Output for paper
        println!("  ┌─ PAPER DATA (Section 3.2 + 3.5) ─────────────────────────┐");
        println!(
            "  │ {label:6} β={beta} │ ⟨P⟩_GPU = {gpu_mean:.8} ± {gpu_se:.2e}         │"
        );
        println!(
            "  │        │ ⟨P⟩_CPU = {cpu_mean:.8} ± {cpu_se:.2e}         │"
        );
        println!("  │        │ |Δ|/σ   = {sigma_ratio:.2}                             │");
        println!("  │        │ τ_int   = {tau_int:.2}                              │");
        println!("  │        │ N_eff   = {n_eff:.0}                               │");
        println!("  │        │ Trajs   = {n_prod} (therm={n_therm})             │");
        println!("  │        │ GPU     = {:.1} ms/traj                     │", gpu_result.ms_per_traj);
        println!("  │        │ CPU acc = {:.1}%  GPU acc = {:.1}%          │", cpu_result.acceptance_rate * 100.0, gpu_result.acceptance_rate * 100.0);
        println!("  └────────────────────────────────────────────────────────────┘");
        println!();
    }
}
