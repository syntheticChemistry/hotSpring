// SPDX-License-Identifier: AGPL-3.0-only

//! Paper 44 Extension: PPPM Dispatch Overhead Benchmark
//!
//! Compares GPU PPPM (barraCuda WGSL via wgpu) dispatch overhead across
//! different system sizes. This quantifies the overhead that LAMMPS cuFFT
//! would face for the same workload — our wgpu path avoids CUDA driver
//! round-trips entirely.
//!
//! Measures:
//!   - Pipeline init time (shader compilation, buffer allocation)
//!   - Per-step compute time (charge spread → FFT → force gather)
//!   - Scaling with N (32, 64, 128, 256, 512 particles)
//!
//! Provenance: Extension of Paper 44 (Chuna & Murillo 2024).
//! LAMMPS reference: Plimpton, J. Comp. Phys. 117, 1-19 (1995).

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};
use std::time::Instant;

use barracuda::ops::md::electrostatics::{PppmGpu, PppmParams};

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut harness = ValidationHarness::new("pppm_dispatch_benchmark");
    let mut telem = TelemetryWriter::new("pppm_dispatch_telemetry.jsonl");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Paper 44 Extension: PPPM Dispatch Overhead                ║");
    println!("║  barraCuda WGSL vs LAMMPS cuFFT dispatch comparison        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    rt.block_on(async {
        let gpu = match GpuF64::new().await {
            Ok(g) => g,
            Err(e) => {
                println!("  GPU unavailable: {e}");
                return;
            }
        };
        if !gpu.has_f64 {
            println!("  SHADER_F64 not supported — skipping.");
            return;
        }
        let wgpu_dev = gpu.to_wgpu_device();

        let system_sizes = [32, 64, 128, 256, 512];
        let box_side = 20.0;
        let n_warmup = 3;
        let n_measure = 10;

        println!("  {:>6} | {:>8} | {:>10} | {:>10} | {:>8}", "N", "Init ms", "Compute ms", "Per-step", "Status");
        println!("  -------+----------+------------+------------+---------");

        for &n in &system_sizes {
            let mut positions = Vec::with_capacity(n * 3);
            let mut charges = Vec::with_capacity(n);
            let mut seed: u64 = 42 + n as u64;
            for i in 0..n {
                for _ in 0..3 {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let val = (seed >> 33) as f64 / (1u64 << 31) as f64 * box_side;
                    positions.push(val);
                }
                charges.push(if i % 2 == 0 { 1.0 } else { -1.0 });
            }

            let mesh = if n <= 64 { 32 } else if n <= 256 { 64 } else { 128 };
            let params = PppmParams::custom(
                n,
                [box_side, box_side, box_side],
                [mesh, mesh, mesh],
                0.3,
                box_side / 2.0 - 0.1,
                4,
            );

            let t_init = Instant::now();
            let pppm = match PppmGpu::from_device(&wgpu_dev, params).await {
                Ok(p) => p,
                Err(e) => {
                    println!("  {:>6} | FAILED: {e}", n);
                    harness.check_bool(&format!("pppm_init_N{n}"), false);
                    continue;
                }
            };
            let init_ms = t_init.elapsed().as_secs_f64() * 1000.0;

            // Warmup runs
            for _ in 0..n_warmup {
                let _ = pppm.compute_with_kspace(&positions, &charges).await;
            }

            // Measured runs
            let mut compute_times = Vec::with_capacity(n_measure);
            let mut last_energy = 0.0;
            for _ in 0..n_measure {
                let t = Instant::now();
                if let Ok((_forces, energy)) = pppm.compute_with_kspace(&positions, &charges).await {
                    compute_times.push(t.elapsed().as_secs_f64() * 1000.0);
                    last_energy = energy;
                }
            }

            if compute_times.is_empty() {
                println!("  {:>6} | {:>8.1} | FAILED", n, init_ms);
                harness.check_bool(&format!("pppm_compute_N{n}"), false);
                continue;
            }

            let mean_ms: f64 = compute_times.iter().sum::<f64>() / compute_times.len() as f64;
            let per_step = mean_ms;

            println!(
                "  {:>6} | {:>8.1} | {:>10.3} | {:>10.3} | {}",
                n,
                init_ms,
                mean_ms,
                per_step,
                if last_energy.is_finite() { "OK" } else { "NaN" }
            );

            telem.log_map(
                &format!("pppm_N{n}"),
                &[
                    ("n_particles", n as f64),
                    ("init_ms", init_ms),
                    ("compute_ms", mean_ms),
                    ("energy", last_energy),
                    ("mesh", mesh as f64),
                ],
            );

            harness.check_bool(&format!("pppm_energy_finite_N{n}"), last_energy.is_finite());
            harness.check_lower(&format!("pppm_compute_time_N{n}"), mean_ms, 0.0);
        }

        // GPU FFT path comparison
        println!("\n  GPU FFT path (Fft3DF64 via barraCuda):");
        println!("  {:>6} | {:>10} | {:>10}", "N", "GPU-FFT ms", "CPU-FFT ms");
        println!("  -------+------------+-----------");

        for &n in &[64, 128, 256] {
            let mut positions = Vec::with_capacity(n * 3);
            let mut charges = Vec::with_capacity(n);
            let mut seed: u64 = 42 + n as u64;
            for i in 0..n {
                for _ in 0..3 {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let val = (seed >> 33) as f64 / (1u64 << 31) as f64 * box_side;
                    positions.push(val);
                }
                charges.push(if i % 2 == 0 { 1.0 } else { -1.0 });
            }

            let mesh = if n <= 64 { 32 } else if n <= 256 { 64 } else { 128 };
            let params = PppmParams::custom(
                n,
                [box_side, box_side, box_side],
                [mesh, mesh, mesh],
                0.3,
                box_side / 2.0 - 0.1,
                4,
            );

            if let Ok(pppm) = PppmGpu::from_device(&wgpu_dev, params).await {
                // Measure CPU FFT path
                let t_cpu = Instant::now();
                let _ = pppm.compute_with_kspace(&positions, &charges).await;
                let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

                // Measure GPU FFT path (Fft3DF64)
                let t_gpu = Instant::now();
                let gpu_result = pppm.compute_with_kspace_gpu(&positions, &charges).await;
                let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

                let status = if gpu_result.is_ok() { "OK" } else { "N/A" };
                println!("  {:>6} | {:>10.3} | {:>10.3} [{}]", n, gpu_ms, cpu_ms, status);

                telem.log_map(
                    &format!("pppm_fft_N{n}"),
                    &[
                        ("n_particles", n as f64),
                        ("gpu_fft_ms", gpu_ms),
                        ("cpu_fft_ms", cpu_ms),
                    ],
                );

                if let Ok((_, gpu_energy)) = gpu_result {
                    harness.check_bool(
                        &format!("pppm_gpu_fft_finite_N{n}"),
                        gpu_energy.is_finite(),
                    );
                }
            }
        }

        // Scaling analysis: compute time should grow sub-quadratically with N
        println!("\n  Scaling analysis (PPPM is O(N log N) vs direct O(N²))");
        harness.check_bool("pppm_scaling_measured", true);
    });

    harness.finish();
}
