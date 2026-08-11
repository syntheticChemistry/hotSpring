// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dual-GPU Volume Scan — Cross-vendor parity at functional lattice sizes.
//!
//! Runs the same SU(3) pure gauge HMC from cached configs on BOTH GPUs
//! simultaneously (RTX 3090 + RX 6950 XT), producing cross-vendor parity
//! data at 16⁴ and 24⁴ volumes for the arXiv paper.
//!
//! Requires cached configs from `arxiv_thermalize_grid`.
//!
//! Usage:
//!   cargo run --release --bin arxiv_dual_gpu_scan --features barracuda-local

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming_cpu_mom,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

struct ScanResult {
    gpu_name: String,
    dims: [usize; 4],
    beta: f64,
    seed: u64,
    mean_plaq: f64,
    std_err: f64,
    acceptance_rate: f64,
    mean_delta_h: f64,
    tau_int: f64,
    ms_per_traj: f64,
    published: Option<f64>,
    source: &'static str,
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn integrated_autocorrelation(data: &[f64]) -> f64 {
    let n = data.len();
    let m = mean(data);
    let var: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-15 || n < 20 {
        return 0.5;
    }
    let mut tau = 0.5;
    let max_lag = n / 4;
    for t in 1..max_lag {
        let c_t: f64 = (0..n - t).map(|i| (data[i] - m) * (data[i + t] - m)).sum::<f64>()
            / ((n - t) as f64 * var);
        if c_t < 0.0 { break; }
        tau += c_t;
        if t as f64 > 6.0 * tau { break; }
    }
    tau
}

fn std_error_with_tau(data: &[f64], tau_int: f64) -> f64 {
    let m = mean(data);
    let var: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    (var * (2.0 * tau_int + 1.0) / data.len() as f64).sqrt()
}

fn run_cached_production(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    dims: [usize; 4],
    beta: f64,
    seed: u64,
    n_therm: usize,
    n_prod: usize,
    dt: f64,
    n_md_steps: usize,
    published: Option<f64>,
    source: &'static str,
) -> Option<ScanResult> {
    let l = dims[0];
    let cache_dir = Lattice::config_cache_dir();
    let cache_key = Lattice::cache_key(dims, beta, seed, n_therm, "omelyan");
    let cache_path = cache_dir.join(format!("{}.lat", &cache_key[..16]));
    let legacy_key = Lattice::legacy_cache_key(dims, beta, seed, n_therm, "omelyan");
    let legacy_path = Lattice::config_cache_root().join(format!("{}.lat", &legacy_key[..16]));

    let lat = match Lattice::load(&cache_path).or_else(|_| Lattice::load(&legacy_path)) {
        Ok(cached) => {
            println!("    [{}] {}⁴ β={beta:.1} seed={seed}: CACHE HIT", gpu.adapter_name, l);
            cached
        }
        Err(_) => {
            println!("    [{}] {}⁴ β={beta:.1} seed={seed}: CACHE MISS — skipping", gpu.adapter_name, l);
            return None;
        }
    };

    let state = GpuHmcState::from_lattice(gpu, &lat, beta);

    let mut plaqs = Vec::with_capacity(n_prod);
    let mut accepted = 0usize;
    let mut delta_h_sum = 0.0f64;
    let mut prod_seed = seed.wrapping_mul(7777);
    let start = Instant::now();

    for i in 0..n_prod {
        let r = gpu_hmc_trajectory_streaming_cpu_mom(gpu, pipelines, &state, n_md_steps, dt, &mut prod_seed)
            .expect("GPU HMC trajectory failed");
        plaqs.push(r.plaquette);
        if r.accepted { accepted += 1; }
        delta_h_sum += r.delta_h.abs();

        if (i + 1) % 200 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            eprint!("\r    [{}] {}⁴ β={beta:.1}: {}/{n_prod} ({rate:.1} traj/s)    ",
                gpu.adapter_name, l, i + 1);
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let tau_int = integrated_autocorrelation(&plaqs);
    let mean_plaq = mean(&plaqs);
    let std_err = std_error_with_tau(&plaqs, tau_int);
    let acceptance_rate = accepted as f64 / n_prod as f64;
    let mean_delta_h = delta_h_sum / n_prod as f64;
    let ms_per_traj = elapsed_ms / n_prod as f64;

    println!("\r    [{}] {}⁴ β={beta:.1} seed={seed}: ⟨P⟩={mean_plaq:.8} ± {std_err:.2e}, acc={:.0}%, τ={tau_int:.1}, {ms_per_traj:.1}ms/traj    ",
        gpu.adapter_name, l, acceptance_rate * 100.0);

    Some(ScanResult {
        gpu_name: gpu.adapter_name.clone(),
        dims, beta, seed, mean_plaq, std_err, acceptance_rate, mean_delta_h,
        tau_int, ms_per_traj, published, source,
    })
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Dual-GPU Volume Scan — Cross-Vendor Parity at 16⁴/24⁴         ║");
    println!("║  SU(3) pure gauge, GPU HMC (cpu_mom), Omelyan 2MN              ║");
    println!("║  Same cached configs → both GPUs → compare plaquettes          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    println!("  Discovering GPUs...");
    let gpu_nvidia = rt.block_on(GpuF64::from_adapter_name("3090"))
        .expect("RTX 3090 not found");
    println!("    GPU A: {}", gpu_nvidia.adapter_name);

    let gpu_amd = rt.block_on(GpuF64::from_adapter_name("6950"))
        .expect("RX 6950 XT not found");
    println!("    GPU B: {}", gpu_amd.adapter_name);
    println!();

    let pipelines_nvidia = GpuHmcStreamingPipelines::new(&gpu_nvidia);
    let pipelines_amd = GpuHmcStreamingPipelines::new(&gpu_amd);
    println!("  HMC pipelines compiled on both GPUs");
    println!();

    let cache_dir = Lattice::config_cache_dir();
    let cached_count = std::fs::read_dir(&cache_dir)
        .map(|rd| rd.filter(|e| e.as_ref().map(|e| e.path().extension().map(|ext| ext == "lat").unwrap_or(false)).unwrap_or(false)).count())
        .unwrap_or(0);
    println!("  Config cache: {} files in {}", cached_count, cache_dir.display());
    println!();

    if cached_count == 0 {
        println!("  [WARN] No cached configs found. Run arxiv_thermalize_grid first.");
        println!("  Exiting.");
        return;
    }

    let scan_points: Vec<([usize; 4], f64, u64, usize, usize, f64, usize, Option<f64>, &str)> = vec![
        ([16,16,16,16], 5.9, 42, 200, 1000, 0.01, 40, Some(0.5637), "GLS98"),
        ([16,16,16,16], 6.0, 42, 200, 1000, 0.01, 40, Some(0.5934), "GL98"),
        ([16,16,16,16], 6.2, 42, 200, 1000, 0.01, 40, Some(0.6136), "B00"),
    ];

    let total_start = Instant::now();
    let mut results: Vec<ScanResult> = Vec::new();

    for &(dims, beta, seed, n_therm, n_prod, dt, n_md, published, source) in &scan_points {
        println!("  ═══ {}⁴ β={beta:.1} seed={seed} ═══", dims[0]);

        // Run on NVIDIA
        if let Some(r) = run_cached_production(
            &gpu_nvidia, &pipelines_nvidia, dims, beta, seed,
            n_therm, n_prod, dt, n_md, published, source,
        ) {
            results.push(r);
        }

        // Run on AMD with same config
        if let Some(r) = run_cached_production(
            &gpu_amd, &pipelines_amd, dims, beta, seed,
            n_therm, n_prod, dt, n_md, published, source,
        ) {
            results.push(r);
        }
        println!();
    }

    // Cross-vendor parity summary
    let total_secs = total_start.elapsed().as_secs_f64();
    println!();
    println!("═══ Cross-Vendor Parity Summary ═══");
    println!();
    println!("| Lattice | β   | GPU                  | ⟨P⟩        | σ_stat     | Accept | τ_int | ms/traj |");
    println!("|---------|-----|----------------------|------------|------------|--------|-------|---------|");

    for r in &results {
        println!("| {}⁴     | {:.1} | {:<20} | {:.8} | {:.2e} | {:.0}%   | {:.1}   | {:.1}   |",
            r.dims[0], r.beta, r.gpu_name, r.mean_plaq, r.std_err,
            r.acceptance_rate * 100.0, r.tau_int, r.ms_per_traj);
    }

    // Compute inter-GPU agreement for each β
    println!();
    println!("  Cross-vendor agreement (same β, same cached config):");
    for &(_, beta, _, _, _, _, _, _, _) in &scan_points {
        let nvidia_plaq: Vec<f64> = results.iter()
            .filter(|r| (r.beta - beta).abs() < 0.01 && r.gpu_name.contains("NVIDIA"))
            .map(|r| r.mean_plaq)
            .collect();
        let amd_plaq: Vec<f64> = results.iter()
            .filter(|r| (r.beta - beta).abs() < 0.01 && !r.gpu_name.contains("NVIDIA"))
            .map(|r| r.mean_plaq)
            .collect();

        if let (Some(&pn), Some(&pa)) = (nvidia_plaq.first(), amd_plaq.first()) {
            let delta = (pn - pa).abs();
            let rel = delta / pn;
            println!("    β={beta:.1}: |ΔNVIDIA-AMD| = {delta:.2e} ({rel:.2e} relative)");
        }
    }

    println!();
    println!("  Total wall time: {total_secs:.1}s ({:.1} min)", total_secs / 60.0);
    println!();

    let json_results: Vec<serde_json::Value> = results.iter().map(|r| {
        serde_json::json!({
            "gpu": r.gpu_name,
            "lattice": format!("{}^4", r.dims[0]),
            "beta": r.beta,
            "seed": r.seed,
            "plaquette_mean": r.mean_plaq,
            "plaquette_stderr": r.std_err,
            "acceptance_rate": r.acceptance_rate,
            "mean_abs_delta_h": r.mean_delta_h,
            "tau_int": r.tau_int,
            "ms_per_traj": r.ms_per_traj,
            "published_value": r.published,
            "published_source": r.source,
        })
    }).collect();

    let output = serde_json::json!({
        "experiment": "arxiv-su3-dual-gpu-volume-scan",
        "gpus": [gpu_nvidia.adapter_name, gpu_amd.adapter_name],
        "integrator": "Omelyan 2MN",
        "momentum_source": "cpu_mom",
        "wall_seconds": total_secs,
        "results": json_results,
    });

    let receipt_path = "arxiv_dual_gpu_scan_results.json";
    match std::fs::write(receipt_path, serde_json::to_string_pretty(&output).unwrap_or_default()) {
        Ok(()) => println!("  Results written to {receipt_path}"),
        Err(e) => println!("  [WARN] Could not write results: {e}"),
    }
}
