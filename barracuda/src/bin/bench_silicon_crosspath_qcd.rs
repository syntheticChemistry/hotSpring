// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-silicon-path QCD benchmark: same physics, different hardware paths.
//!
//! Runs identical QCD workloads through both GPUs with the same seed, beta,
//! and lattice dimensions, then compares:
//! - Correctness (do both cards give the same plaquette?)
//! - Performance (which silicon is faster for HMC?)
//! - Precision profile (rounding error accumulation differs by architecture)
//!
//! This exploits the sunMemo structure: by running the same thermalization on
//! different silicon, we get deeper and cheaper results — cross-validating
//! hardware correctness while generating publishable comparison data.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

struct CardResult {
    name: String,
    plaquettes: Vec<f64>,
    ms_per_traj: f64,
    acceptance: f64,
}

fn run_on_gpu(gpu: &GpuF64, dims: [usize; 4], beta: f64, seed: u64, n_therm: usize, n_meas: usize) -> CardResult {
    let name = gpu.adapter_name.clone();
    let pipelines = GpuHmcStreamingPipelines::new(gpu);
    let lat = Lattice::hot_start(dims, beta, seed);
    let hmc_state = GpuHmcState::from_lattice(gpu, &lat, beta);

    let n_md = 10;
    let dt = 0.1;
    let mut rng_seed = seed;
    let mut accepted = 0u32;

    let t0 = Instant::now();

    // Thermalize
    for i in 0..n_therm {
        let result = gpu_hmc_trajectory_streaming(
            gpu,
            &pipelines,
            &hmc_state,
            n_md,
            dt,
            i as u32,
            &mut rng_seed,
        )
        .expect("HMC trajectory");
        if result.accepted {
            accepted += 1;
        }
    }

    // Measure
    let mut plaquettes = Vec::with_capacity(n_meas);
    for j in 0..n_meas {
        let result = gpu_hmc_trajectory_streaming(
            gpu,
            &pipelines,
            &hmc_state,
            n_md,
            dt,
            (n_therm + j) as u32,
            &mut rng_seed,
        )
        .expect("HMC trajectory");
        plaquettes.push(result.plaquette);
        if result.accepted {
            accepted += 1;
        }
    }

    let elapsed = t0.elapsed();
    let total = (n_therm + n_meas) as f64;
    let ms_per_traj = elapsed.as_secs_f64() * 1000.0 / total;
    let acceptance = accepted as f64 / total;

    CardResult {
        name,
        plaquettes,
        ms_per_traj,
        acceptance,
    }
}

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Cross-Silicon-Path QCD Benchmark");
    println!("  Same physics → different silicon → compare values + performance");
    println!("  sunMemo: identical seed + β + dims = identical Markov chain");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let dims = [8, 8, 8, 8];
    let beta = 6.0;
    let seed = 20260808u64;
    let n_therm = 30;
    let n_meas = 10;

    println!("  Config: SU(3), {}⁴, β={}, seed={}", dims[0], beta, seed);
    println!("  Trajectories: {} therm + {} measure = {} total", n_therm, n_meas, n_therm + n_meas);
    println!("  HMC: n_md=10, dt=0.1 (Omelyan)");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let discrete: Vec<_> = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    println!("  GPUs: {} discrete cards", discrete.len());
    for a in &discrete {
        println!("    • {}", a.get_info().name);
    }
    println!();

    let mut results: Vec<CardResult> = Vec::new();

    for adapter in discrete {
        let info = adapter.get_info();
        let name = info.name.clone();
        println!("━━━ Running on: {} ━━━", name);

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  SKIP: {e}\n");
                continue;
            }
        };

        let r = run_on_gpu(&gpu, dims, beta, seed, n_therm, n_meas);
        let plaq_mean = r.plaquettes.iter().sum::<f64>() / r.plaquettes.len() as f64;

        println!("  ⟨P⟩ = {:.10}", plaq_mean);
        println!("  Acceptance: {:.0}%", r.acceptance * 100.0);
        println!("  Time: {:.2} ms/trajectory", r.ms_per_traj);
        println!();

        results.push(r);
    }

    if results.len() < 2 {
        println!("  Need 2+ GPUs for cross-silicon comparison.");
        println!("═══════════════════════════════════════════════════════════════════");
        return;
    }

    // Cross-comparison
    println!("━━━ Cross-Silicon Comparison ━━━");
    println!();

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let stderr = |v: &[f64]| {
        let m = mean(v);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0);
        (var / v.len() as f64).sqrt()
    };

    println!("  {:35} {:>14} {:>10} {:>12} {:>10}", "Card", "⟨P⟩", "± err", "ms/traj", "Accept%");
    println!("  {:35} {:>14} {:>10} {:>12} {:>10}", "─".repeat(35), "─".repeat(14), "─".repeat(10), "─".repeat(12), "─".repeat(10));

    for r in &results {
        let m = mean(&r.plaquettes);
        let e = stderr(&r.plaquettes);
        println!("  {:35} {:>14.10} {:>10.2e} {:>10.2}ms {:>8.0}%",
                 r.name, m, e, r.ms_per_traj, r.acceptance * 100.0);
    }
    println!();

    // Parity analysis
    let ref_plaq = mean(&results[0].plaquettes);
    let mut max_delta = 0.0f64;
    println!("  Cross-GPU parity (vs {}):", results[0].name);
    for r in results.iter().skip(1) {
        let m = mean(&r.plaquettes);
        let delta = (m - ref_plaq).abs();
        let relative = delta / ref_plaq;
        max_delta = max_delta.max(delta);
        println!("    Δ⟨P⟩ = {:.2e} (relative {:.2e}) — {}", delta, relative, r.name);
    }
    println!();

    // Same seed means same Markov chain → plaquettes should agree within
    // statistical error (both cards traverse identical path through config space)
    let combined_err = results.iter().map(|r| stderr(&r.plaquettes)).sum::<f64>();
    let n_sigma = max_delta / combined_err.max(1e-16);

    if n_sigma < 3.0 {
        println!("  ✓ SILICON PARITY CONFIRMED: cards agree within {:.1}σ", n_sigma);
        println!("    Same Markov chain → same physics → silicon routing is transparent");
    } else {
        println!("  ⚠ DIVERGENCE at {:.1}σ — investigating:", n_sigma);
        println!("    This may indicate precision differences in force accumulation");
        println!("    or subtle PRNG state divergence between DF64 implementations");
    }
    println!();

    // Performance routing recommendation
    let fastest = results.iter().min_by(|a, b| a.ms_per_traj.partial_cmp(&b.ms_per_traj).unwrap()).unwrap();
    let slowest = results.iter().max_by(|a, b| a.ms_per_traj.partial_cmp(&b.ms_per_traj).unwrap()).unwrap();
    let speedup = slowest.ms_per_traj / fastest.ms_per_traj;

    println!("  Performance routing:");
    println!("    Fastest: {} ({:.2} ms/traj)", fastest.name, fastest.ms_per_traj);
    println!("    Slowest: {} ({:.2} ms/traj)", slowest.name, slowest.ms_per_traj);
    println!("    Ratio: {:.2}×", speedup);
    println!();

    println!("  Silicon insight:");
    println!("    Same seed + same β + same volume = same Markov chain on both cards.");
    println!("    Performance delta reveals silicon architecture differences, not physics.");
    println!("    toadStool routes: faster card for production, slower for precision oracle.");
    println!();

    // Per-trajectory comparison (for the measurement phase)
    println!("  Per-trajectory plaquette (measurement phase):");
    println!("  {:4} {:>14} {:>14} {:>12}", "Traj", &results[0].name[..14.min(results[0].name.len())],
             &results[1].name[..14.min(results[1].name.len())], "Δ");
    for i in 0..results[0].plaquettes.len().min(results[1].plaquettes.len()) {
        let p0 = results[0].plaquettes[i];
        let p1 = results[1].plaquettes[i];
        let delta = (p1 - p0).abs();
        println!("  {:4} {:>14.10} {:>14.10} {:>12.2e}", i + 1, p0, p1, delta);
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Cross-Silicon QCD Complete — {} cards, {} trajectories each", results.len(), n_therm + n_meas);
    println!("  sunMemo: these configs feed the arXiv preprint validation table");
    println!("═══════════════════════════════════════════════════════════════════");
}
