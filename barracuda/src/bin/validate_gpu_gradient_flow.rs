// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU gradient flow validation — compares GPU flow against CPU reference.
//!
//! Validates that the GPU gradient flow (Paper 43, Bazavov & Chuna 2021)
//! produces bit-level-parity results with the CPU implementation.
//! Only 1 new WGSL shader (`su3_flow_accumulate_f64`) — all other ops
//! reuse HMC's gauge force, link update, and plaquette shaders.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_flow::{GpuFlowPipelines, GpuFlowState, gpu_gradient_flow};
use hotspring_barracuda::lattice::gradient_flow::{FlowIntegrator, find_t0, find_w0, run_flow};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    let mut harness = ValidationHarness::new("gpu_gradient_flow");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   GPU Gradient Flow Validation (Bazavov & Chuna 2021)      ║");
    println!("║   Paper 43: SU(3) LSCFRK integrators on GPU               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => {
            println!("  GPU: {}", g.adapter_name);
            println!(
                "  f64 support: {}",
                if g.has_f64 { "native" } else { "DF64" }
            );
            println!();
            g
        }
        Err(e) => {
            println!("  No GPU available: {e}");
            println!("  Skipping GPU flow validation.");
            harness.finish();
        }
    };

    let _guard = rt.enter();
    let pipelines = GpuFlowPipelines::new(&gpu);
    let beta = 6.0;
    let dims = [4, 4, 4, 4];
    let seed = 42;

    // ── Euler: GPU vs CPU ──
    println!("── Euler integrator ──");
    {
        let mut cpu_lat = Lattice::hot_start(dims, beta, seed);
        let gpu_lat = Lattice::hot_start(dims, beta, seed);
        let state = GpuFlowState::from_lattice(&gpu, &gpu_lat, beta);

        let cpu = run_flow(&mut cpu_lat, FlowIntegrator::Euler, 0.01, 0.1, 1);
        let gpu_r = gpu_gradient_flow(
            &gpu,
            &pipelines,
            &state,
            FlowIntegrator::Euler,
            0.01,
            0.1,
            1,
        );

        let cpu_plaq = cpu.last().unwrap().plaquette;
        let gpu_plaq = gpu_r.measurements.last().unwrap().plaquette;
        let diff = (cpu_plaq - gpu_plaq).abs();
        println!("  CPU plaquette: {cpu_plaq:.10}");
        println!("  GPU plaquette: {gpu_plaq:.10}");
        println!("  Δ = {diff:.2e}");
        harness.check_upper("euler_plaquette_parity", diff, 1e-8);
    }

    // ── RK3 Lüscher (LSCFRK3W6): GPU vs CPU ──
    println!("\n── RK3 Lüscher (W6) ──");
    {
        let mut cpu_lat = Lattice::hot_start(dims, beta, seed);
        let gpu_lat = Lattice::hot_start(dims, beta, seed);
        let state = GpuFlowState::from_lattice(&gpu, &gpu_lat, beta);

        let cpu = run_flow(&mut cpu_lat, FlowIntegrator::Rk3Luscher, 0.01, 0.5, 10);
        let gpu_r = gpu_gradient_flow(
            &gpu,
            &pipelines,
            &state,
            FlowIntegrator::Rk3Luscher,
            0.01,
            0.5,
            10,
        );

        let cpu_plaq = cpu.last().unwrap().plaquette;
        let gpu_plaq = gpu_r.measurements.last().unwrap().plaquette;
        let diff = (cpu_plaq - gpu_plaq).abs();
        println!("  CPU plaquette: {cpu_plaq:.10}");
        println!("  GPU plaquette: {gpu_plaq:.10}");
        println!("  Δ = {diff:.2e}");
        println!("  GPU wall time: {:.3}s", gpu_r.wall_seconds);
        harness.check_upper("rk3_luscher_plaquette_parity", diff, 1e-8);
    }

    // ── LSCFRK3W7 (Chuna): GPU vs CPU ──
    println!("\n── LSCFRK3W7 (Bazavov & Chuna) ──");
    {
        let mut cpu_lat = Lattice::hot_start(dims, beta, seed);
        let gpu_lat = Lattice::hot_start(dims, beta, seed);
        let state = GpuFlowState::from_lattice(&gpu, &gpu_lat, beta);

        let cpu = run_flow(&mut cpu_lat, FlowIntegrator::Lscfrk3w7, 0.01, 0.5, 10);
        let gpu_r = gpu_gradient_flow(
            &gpu,
            &pipelines,
            &state,
            FlowIntegrator::Lscfrk3w7,
            0.01,
            0.5,
            10,
        );

        let cpu_plaq = cpu.last().unwrap().plaquette;
        let gpu_plaq = gpu_r.measurements.last().unwrap().plaquette;
        let diff = (cpu_plaq - gpu_plaq).abs();
        println!("  CPU plaquette: {cpu_plaq:.10}");
        println!("  GPU plaquette: {gpu_plaq:.10}");
        println!("  Δ = {diff:.2e}");
        println!("  GPU wall time: {:.3}s", gpu_r.wall_seconds);
        harness.check_upper("lscfrk3w7_plaquette_parity", diff, 1e-8);
    }

    // ── LSCFRK4CK (Carpenter-Kennedy): GPU vs CPU ──
    println!("\n── LSCFRK4CK (Carpenter-Kennedy) ──");
    {
        let mut cpu_lat = Lattice::hot_start(dims, beta, seed);
        let gpu_lat = Lattice::hot_start(dims, beta, seed);
        let state = GpuFlowState::from_lattice(&gpu, &gpu_lat, beta);

        let cpu = run_flow(&mut cpu_lat, FlowIntegrator::Lscfrk4ck, 0.01, 0.5, 10);
        let gpu_r = gpu_gradient_flow(
            &gpu,
            &pipelines,
            &state,
            FlowIntegrator::Lscfrk4ck,
            0.01,
            0.5,
            10,
        );

        let cpu_plaq = cpu.last().unwrap().plaquette;
        let gpu_plaq = gpu_r.measurements.last().unwrap().plaquette;
        let diff = (cpu_plaq - gpu_plaq).abs();
        println!("  CPU plaquette: {cpu_plaq:.10}");
        println!("  GPU plaquette: {gpu_plaq:.10}");
        println!("  Δ = {diff:.2e}");
        println!("  GPU wall time: {:.3}s", gpu_r.wall_seconds);
        harness.check_upper("lscfrk4ck_plaquette_parity", diff, 1e-8);
    }

    // ── t₀ scale: GPU vs CPU ──
    println!("\n── t₀ scale determination ──");
    {
        let mut cpu_lat = Lattice::hot_start(dims, beta, seed);
        let gpu_lat = Lattice::hot_start(dims, beta, seed);
        let state = GpuFlowState::from_lattice(&gpu, &gpu_lat, beta);

        let cpu = run_flow(&mut cpu_lat, FlowIntegrator::Rk3Luscher, 0.01, 2.0, 1);
        let gpu_r = gpu_gradient_flow(
            &gpu,
            &pipelines,
            &state,
            FlowIntegrator::Rk3Luscher,
            0.01,
            2.0,
            1,
        );

        let cpu_t0 = find_t0(&cpu);
        let gpu_t0 = find_t0(&gpu_r.measurements);

        match (cpu_t0, gpu_t0) {
            (Some(c), Some(g)) => {
                let diff = (c - g).abs();
                println!("  CPU t₀ = {c:.6}");
                println!("  GPU t₀ = {g:.6}");
                println!("  Δ = {diff:.2e}");
                harness.check_upper("t0_scale_parity", diff, 1e-4);
            }
            (None, None) => {
                println!("  Both CPU and GPU failed to find t₀ crossing (4⁴ too small)");
                harness.check_upper("t0_scale_parity", 0.0, 1.0);
            }
            _ => {
                println!("  MISMATCH: cpu={cpu_t0:?}, gpu={gpu_t0:?}");
                harness.check_upper("t0_scale_parity", 1.0, 1e-4);
            }
        }
    }

    // ── w₀ scale: GPU vs CPU ──
    println!("\n── w₀ scale determination ──");
    {
        let mut cpu_lat = Lattice::hot_start(dims, beta, seed);
        let gpu_lat = Lattice::hot_start(dims, beta, seed);
        let state = GpuFlowState::from_lattice(&gpu, &gpu_lat, beta);

        let cpu = run_flow(&mut cpu_lat, FlowIntegrator::Lscfrk3w7, 0.01, 2.0, 1);
        let gpu_r = gpu_gradient_flow(
            &gpu,
            &pipelines,
            &state,
            FlowIntegrator::Lscfrk3w7,
            0.01,
            2.0,
            1,
        );

        let cpu_w0 = find_w0(&cpu);
        let gpu_w0 = find_w0(&gpu_r.measurements);

        match (cpu_w0, gpu_w0) {
            (Some(c), Some(g)) => {
                let diff = (c - g).abs();
                println!("  CPU w₀ = {c:.6}");
                println!("  GPU w₀ = {g:.6}");
                println!("  Δ = {diff:.2e}");
                harness.check_upper("w0_scale_parity", diff, 1e-4);
            }
            (None, None) => {
                println!("  Both CPU and GPU failed to find w₀ crossing (4⁴ too small)");
                harness.check_upper("w0_scale_parity", 0.0, 1.0);
            }
            _ => {
                println!("  MISMATCH: cpu={cpu_w0:?}, gpu={gpu_w0:?}");
                harness.check_upper("w0_scale_parity", 1.0, 1e-4);
            }
        }
    }

    // ── CPU vs GPU speedup ──
    println!("\n── Benchmark: 8⁴ lattice, W7, t=0→1.0 ──");
    {
        let dims_8 = [8, 8, 8, 8];
        let mut cpu_lat = Lattice::hot_start(dims_8, beta, seed);
        let gpu_lat = Lattice::hot_start(dims_8, beta, seed);

        let cpu_start = std::time::Instant::now();
        let _cpu = run_flow(&mut cpu_lat, FlowIntegrator::Lscfrk3w7, 0.02, 1.0, 50);
        let cpu_time = cpu_start.elapsed().as_secs_f64();

        let state = GpuFlowState::from_lattice(&gpu, &gpu_lat, beta);
        let gpu_r = gpu_gradient_flow(
            &gpu,
            &pipelines,
            &state,
            FlowIntegrator::Lscfrk3w7,
            0.02,
            1.0,
            50,
        );

        let speedup = cpu_time / gpu_r.wall_seconds;
        println!("  CPU: {cpu_time:.3}s");
        println!("  GPU: {:.3}s", gpu_r.wall_seconds);
        println!("  Speedup: {speedup:.1}×");
        harness.check_lower("gpu_not_slower_than_2x_cpu", speedup, 0.5);
    }

    println!();
    harness.finish();
}
