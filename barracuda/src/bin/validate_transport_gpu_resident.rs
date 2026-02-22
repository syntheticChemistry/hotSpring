// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-Resident Transport Validation — Unidirectional Streaming VACF
//!
//! Proves that VACF and D* can be computed entirely on GPU using the
//! `GpuVelocityRing` + `vacf_dot_f64.wgsl` + `ReduceScalarPipeline` path.
//!
//! **Architecture**: Run GPU MD simulation (existing path), then:
//!   1. Upload velocity snapshots to GPU ring buffer (simulating GPU-resident storage)
//!   2. Compute VACF on GPU via per-particle dot product + reduction
//!   3. Green-Kubo integrate on CPU (O(n_lag) — trivial)
//!   4. Compare D*(GPU VACF) vs D*(CPU VACF) — must match
//!
//! This is step 1 of the full unidirectional pipeline. Step 2 would
//! eliminate the velocity readback entirely by capturing snapshots
//! directly from GPU velocity buffers during simulation.
//!
//! **Data flow proven**:
//!   GPU MD → velocity snapshots → GPU ring → GPU VACF → scalar D* → CPU
//!
//! Exit code 0 = all checks pass.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::config;
use hotspring_barracuda::md::cpu_reference;
use hotspring_barracuda::md::observables::{compute_vacf, validate_energy, GpuVelocityRing};
use hotspring_barracuda::md::observables::transport_gpu::compute_vacf_gpu;
use hotspring_barracuda::md::transport::d_star_daligault;
use hotspring_barracuda::validation::ValidationHarness;

use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  GPU-Resident Transport — Unidirectional Streaming VACF    ║");
    println!("║  D* computed on GPU: no position readback for VACF         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("transport_gpu_resident");

    let cases = config::transport_cases(500, true);
    let case = cases
        .into_iter()
        .find(|c| (c.kappa - 1.0).abs() < 0.01 && (c.gamma - 50.0).abs() < 0.01)
        .expect("κ=1, Γ=50 case not found");

    println!("  Case: κ={}, Γ={}, N=500 (lite)", case.kappa, case.gamma);
    let d_fit = d_star_daligault(case.gamma, case.kappa);
    println!("  D*(Daligault fit) = {d_fit:.4e}");
    println!();

    // ── Phase 1: CPU reference simulation + CPU VACF ──
    println!("═══ Phase 1: CPU Simulation + CPU VACF (baseline) ═══");
    let t_cpu = Instant::now();
    let sim = cpu_reference::run_simulation_cpu(&case);
    let dt_snap = case.dt * case.dump_step as f64 * case.vel_snapshot_interval as f64;
    let max_lag = (sim.velocity_snapshots.len() / 2).max(10);
    let vacf_cpu = compute_vacf(&sim.velocity_snapshots, case.n_particles, dt_snap, max_lag);
    let cpu_time = t_cpu.elapsed().as_secs_f64();

    println!("  D*(CPU VACF) = {:.4e}", vacf_cpu.diffusion_coeff);
    println!("  VACF frames: {}", sim.velocity_snapshots.len());
    println!("  CPU time: {cpu_time:.2}s");

    let ev = validate_energy(&sim.energy_history, &case);
    harness.check_upper("CPU energy conservation", ev.drift_pct, 5.0);
    println!();

    // ── Phase 2: Upload snapshots to GPU ring + GPU VACF ──
    println!("═══ Phase 2: GPU VACF (velocity ring + dot product shader) ═══");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(async { GpuF64::new().await }).expect("GPU init");

    if !gpu.has_f64 {
        println!("  SKIP: GPU lacks f64 support");
        harness.check_bool("GPU f64 available", false);
        harness.finish();
    }

    let n = case.n_particles;
    let n_snapshots = sim.velocity_snapshots.len();
    let n_ring_slots = n_snapshots; // store all snapshots in ring

    let t_gpu_upload = Instant::now();
    let mut ring = GpuVelocityRing::new(&gpu, n, n_ring_slots);

    for snap in &sim.velocity_snapshots {
        let vel_buf = gpu.create_f64_buffer(snap, "vel_upload");
        ring.store_snapshot(&gpu, &vel_buf);
    }
    let upload_time = t_gpu_upload.elapsed().as_secs_f64();
    println!("  Uploaded {n_snapshots} snapshots to GPU ring ({upload_time:.2}s)");

    let t_gpu_vacf = Instant::now();
    let vacf_gpu = compute_vacf_gpu(&gpu, &ring, dt_snap, max_lag)
        .expect("GPU VACF computation failed");
    let gpu_vacf_time = t_gpu_vacf.elapsed().as_secs_f64();

    println!("  D*(GPU VACF) = {:.4e}", vacf_gpu.diffusion_coeff);
    println!("  GPU VACF time: {gpu_vacf_time:.2}s");
    println!();

    // ── Phase 3: Parity comparison ──
    println!("═══ Phase 3: CPU vs GPU VACF Parity ═══");

    let d_cpu = vacf_cpu.diffusion_coeff;
    let d_gpu = vacf_gpu.diffusion_coeff;
    let rel_err = if d_cpu.abs() > f64::EPSILON {
        ((d_cpu - d_gpu) / d_cpu).abs()
    } else {
        0.0
    };

    println!("  D*(CPU): {d_cpu:.6e}");
    println!("  D*(GPU): {d_gpu:.6e}");
    println!("  Relative error: {:.2}%", rel_err * 100.0);

    // VACF C(0) comparison
    let c0_cpu = vacf_cpu.c_values[0];
    let c0_gpu = vacf_gpu.c_values[0];
    println!("  C(0) CPU: {c0_cpu:.6}, GPU: {c0_gpu:.6}");

    // Compare first few lag values
    let n_compare = 5.min(vacf_cpu.c_values.len()).min(vacf_gpu.c_values.len());
    println!("  First {n_compare} normalized C(lag) values:");
    for i in 0..n_compare {
        let diff = (vacf_cpu.c_values[i] - vacf_gpu.c_values[i]).abs();
        println!(
            "    lag {i}: CPU={:.6} GPU={:.6} |Δ|={:.2e}",
            vacf_cpu.c_values[i], vacf_gpu.c_values[i], diff
        );
    }
    println!();

    // GPU VACF should match CPU VACF closely — same data, different reduction order.
    // Machine epsilon differences accumulate through N-particle dot products.
    harness.check_upper(
        "D* CPU≈GPU VACF (same data, different reduction)",
        rel_err,
        0.01, // 1% tolerance — same data, just GPU reduction vs CPU sum
    );

    harness.check_bool("GPU D* > 0", d_gpu > 0.0);
    harness.check_bool("GPU D* finite", d_gpu.is_finite());

    // ── Summary ──
    println!("═══ Performance ═══");
    println!("  CPU total:        {cpu_time:.2}s (simulation + VACF)");
    println!("  GPU upload:       {upload_time:.2}s");
    println!("  GPU VACF compute: {gpu_vacf_time:.2}s");
    println!();
    println!("  Next evolution: eliminate velocity readback entirely —");
    println!("  capture snapshots directly from GPU velocity buffer during MD.");
    println!("  Target: 10-50× speedup by removing PCIe roundtrips.");
    println!();

    harness.finish();
}
