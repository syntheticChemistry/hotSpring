// SPDX-License-Identifier: AGPL-3.0-or-later

//! arXiv Volume Scan — GPU production runs at functional lattice sizes.
//!
//! Runs SU(3) pure gauge HMC on 12⁴, 16⁴, and 24⁴ lattices at weak-coupling
//! β values where published data exists for direct comparison. Uses the
//! validated cpu_mom path (CPU-generated momenta, GPU molecular dynamics).
//!
//! Output: per-β plaquette with statistical error, acceptance rate, ΔH,
//! autocorrelation time, and comparison to published values.
//!
//! Expected runtime: ~2-6 hours depending on lattice size.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming_cpu_mom,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

struct VolumeScanResult {
    dims: [usize; 4],
    beta: f64,
    n_therm: usize,
    n_prod: usize,
    dt: f64,
    n_md_steps: usize,
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

fn std_error_with_tau(data: &[f64], tau_int: f64) -> f64 {
    let m = mean(data);
    let var: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    (var * (2.0 * tau_int + 1.0) / data.len() as f64).sqrt()
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
        if c_t < 0.0 {
            break;
        }
        tau += c_t;
        if t as f64 > 6.0 * tau {
            break;
        }
    }
    tau
}

fn run_gpu_volume_point(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    dims: [usize; 4],
    beta: f64,
    n_therm: usize,
    n_prod: usize,
    dt: f64,
    n_md_steps: usize,
    published: Option<f64>,
    source: &'static str,
) -> VolumeScanResult {
    let l = dims[0];
    let seed = 42u64;

    let cache_dir = Lattice::config_cache_dir();
    let cache_key = Lattice::cache_key(dims, beta, seed, n_therm, "omelyan");
    let cache_path = cache_dir.join(format!("{}.lat", &cache_key[..16]));
    let legacy_key = Lattice::legacy_cache_key(dims, beta, seed, n_therm, "omelyan");
    let legacy_path = Lattice::config_cache_root().join(format!("{}.lat", &legacy_key[..16]));

    let lat = if let Ok(cached) = Lattice::load(&cache_path).or_else(|_| Lattice::load(&legacy_path)) {
        println!("  {}⁴ β={beta:.1}: [CACHE HIT] loaded", l);
        cached
    } else {
        print!("  {}⁴ β={beta:.1}: [CACHE MISS] thermalizing on CPU ({n_therm} steps)... ", l);
        let therm_start = Instant::now();
        let mut lat = Lattice::hot_start(dims, beta, seed);
        let cfg = &mut HmcConfig {
            n_md_steps,
            dt,
            seed,
            integrator: IntegratorType::Omelyan,
        };
        for _ in 0..n_therm {
            hmc::hmc_trajectory(&mut lat, cfg);
        }
        let therm_secs = therm_start.elapsed().as_secs_f64();
        println!("{therm_secs:.1}s");

        if let Ok(hash) = lat.save(&cache_path) {
            println!("    [CACHED] {} → hash={}", cache_path.display(), &hash.to_hex()[..16]);
        }
        lat
    };

    // Upload to GPU
    let state = GpuHmcState::from_lattice(gpu, &lat, beta);

    // Production on GPU
    let mut plaqs = Vec::with_capacity(n_prod);
    let mut accepted = 0usize;
    let mut delta_h_sum = 0.0f64;
    let mut seed = 7777u64;
    let start = Instant::now();

    for i in 0..n_prod {
        let r = gpu_hmc_trajectory_streaming_cpu_mom(gpu, pipelines, &state, n_md_steps, dt, &mut seed)
            .expect("GPU HMC trajectory failed");
        plaqs.push(r.plaquette);
        if r.accepted {
            accepted += 1;
        }
        delta_h_sum += r.delta_h.abs();

        if (i + 1) % 200 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            let eta = (n_prod - i - 1) as f64 / rate;
            eprint!("\r  {}⁴ β={beta:.1}: {}/{n_prod} ({rate:.1} traj/s, ETA {eta:.0}s)    ", l, i + 1);
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let tau_int = integrated_autocorrelation(&plaqs);
    let mean_plaq = mean(&plaqs);
    let std_err = std_error_with_tau(&plaqs, tau_int);
    let acceptance_rate = accepted as f64 / n_prod as f64;
    let mean_delta_h = delta_h_sum / n_prod as f64;
    let ms_per_traj = elapsed_ms / n_prod as f64;

    let delta_str = match published {
        Some(pub_val) => format!("Δ={:.2}%", (mean_plaq - pub_val) / pub_val * 100.0),
        None => "—".to_string(),
    };

    println!("\r  {}⁴ β={beta:.1}: ⟨P⟩={mean_plaq:.8} ± {std_err:.2e}, acc={:.0}%, τ={tau_int:.1}, {ms_per_traj:.1}ms/traj, {delta_str}    ",
        l, acceptance_rate * 100.0);

    VolumeScanResult {
        dims, beta, n_therm, n_prod, dt, n_md_steps,
        mean_plaq, std_err, acceptance_rate, mean_delta_h, tau_int, ms_per_traj,
        published, source,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  arXiv Volume Scan — Functional Lattice Sizes              ║");
    println!("║  SU(3) pure gauge, GPU HMC (cpu_mom), Omelyan 2MN          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU initialization failed");
    println!("  GPU: {}", gpu.adapter_name);
    println!();

    let pipelines = GpuHmcStreamingPipelines::new(&gpu);
    println!("  HMC pipelines compiled");
    println!();

    let total_start = Instant::now();
    let mut results: Vec<VolumeScanResult> = Vec::new();

    let skip_12 = std::env::var("SKIP_12").is_ok();
    let skip_24 = std::env::var("SKIP_24").is_ok();

    if !skip_12 {
        // ═══ 12⁴ lattice ═══
        println!("═══ 12⁴ Lattice (20,736 sites) ═══");
        println!("  Params: 200 therm + 1000 prod, dt=0.015, N_md=30");
        println!();

        let scan_12 = [
            (5.7, Some(0.5464), "NS02"),
            (5.8, Some(0.5544), "NS02"),
            (5.9, Some(0.5637), "GLS98"),
            (6.0, Some(0.5934), "GL98"),
            (6.2, Some(0.6136), "B00"),
        ];

        for &(beta, published, source) in &scan_12 {
            let r = run_gpu_volume_point(&gpu, &pipelines, [12, 12, 12, 12], beta,
                200, 1000, 0.015, 30, published, source);
            results.push(r);
        }
        println!();
    } else {
        println!("  [SKIP] 12⁴ — already completed, set SKIP_12 to skip");
        println!();
    }

    // ═══ 16⁴ lattice ═══
    println!("═══ 16⁴ Lattice (65,536 sites) ═══");
    println!("  Params: 200 therm + 1000 prod, dt=0.01, N_md=40");
    println!();

    let scan_16 = [
        (5.9, Some(0.5637), "GLS98"),
        (6.0, Some(0.5934), "GL98"),
        (6.2, Some(0.6136), "B00"),
    ];

    for &(beta, published, source) in &scan_16 {
        println!("  Starting {}⁴ β={:.1} (CPU thermalization: 200 steps)...", 16, beta);
        let therm_start = Instant::now();
        let r = run_gpu_volume_point(&gpu, &pipelines, [16, 16, 16, 16], beta,
            200, 1000, 0.01, 40, published, source);
        println!("  Completed in {:.1}s", therm_start.elapsed().as_secs_f64());
        results.push(r);
    }

    println!();

    if !skip_24 {
        // ═══ 24⁴ lattice (stretch) ═══
        println!("═══ 24⁴ Lattice (331,776 sites) ═══");
        println!("  Params: 200 therm + 500 prod, dt=0.008, N_md=50");
        println!();

        let scan_24 = [
            (6.0, Some(0.5934), "GL98"),
        ];

        for &(beta, published, source) in &scan_24 {
            println!("  Starting {}⁴ β={:.1} (CPU thermalization: 200 steps)...", 24, beta);
            let therm_start = Instant::now();
            let r = run_gpu_volume_point(&gpu, &pipelines, [24, 24, 24, 24], beta,
                200, 500, 0.008, 50, published, source);
            println!("  Completed in {:.1}s", therm_start.elapsed().as_secs_f64());
            results.push(r);
        }
    } else {
        println!("  [SKIP] 24⁴ — set SKIP_24 to skip");
    }

    // ═══ Summary ═══
    let total_secs = total_start.elapsed().as_secs_f64();
    println!();
    println!("═══ Volume Scan Summary ═══");
    println!();
    println!("| Lattice | β   | ⟨P⟩        | σ_stat     | Accept | τ_int | ⟨|ΔH|⟩   | ms/traj | Published | Δ/%    |");
    println!("|---------|-----|------------|------------|--------|-------|----------|---------|-----------|--------|");

    for r in &results {
        let pub_str = match r.published {
            Some(v) => format!("{v:.4}"),
            None => "—".to_string(),
        };
        let delta_str = match r.published {
            Some(v) => format!("{:.2}%", (r.mean_plaq - v) / v * 100.0),
            None => "—".to_string(),
        };
        println!("| {}⁴     | {:.1} | {:.8} | {:.2e} | {:.0}%   | {:.1}   | {:.2e} | {:.1}   | {:<9} | {:<6} |",
            r.dims[0], r.beta, r.mean_plaq, r.std_err,
            r.acceptance_rate * 100.0, r.tau_int, r.mean_delta_h,
            r.ms_per_traj, pub_str, delta_str);
    }

    println!();
    println!("  Total wall time: {:.1}s ({:.1} hours)", total_secs, total_secs / 3600.0);
    println!("  GPU: {}", gpu.adapter_name);
    println!();

    // Write results to JSON for provenance
    let json_results: Vec<serde_json::Value> = results.iter().map(|r| {
        serde_json::json!({
            "lattice": format!("{}^4", r.dims[0]),
            "volume": r.dims[0].pow(4),
            "beta": r.beta,
            "plaquette_mean": r.mean_plaq,
            "plaquette_stderr": r.std_err,
            "acceptance_rate": r.acceptance_rate,
            "mean_abs_delta_h": r.mean_delta_h,
            "tau_int": r.tau_int,
            "ms_per_traj": r.ms_per_traj,
            "n_therm": r.n_therm,
            "n_prod": r.n_prod,
            "dt": r.dt,
            "n_md_steps": r.n_md_steps,
            "published_value": r.published,
            "published_source": r.source,
        })
    }).collect();

    let output = serde_json::json!({
        "experiment": "arxiv-su3-volume-scan",
        "gpu": gpu.adapter_name,
        "integrator": "Omelyan 2MN",
        "momentum_source": "cpu_mom",
        "wall_seconds": total_secs,
        "results": json_results,
    });

    let receipt_path = "arxiv_volume_scan_results.json";
    match std::fs::write(receipt_path, serde_json::to_string_pretty(&output).unwrap_or_default()) {
        Ok(()) => println!("  Results written to {receipt_path}"),
        Err(e) => println!("  [WARN] Could not write results: {e}"),
    }
}
