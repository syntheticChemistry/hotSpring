// SPDX-License-Identifier: AGPL-3.0-or-later

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;

pub fn bench_vacf_gpu_vs_cpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2a: VACF — GPU (barracuda::ops::md) vs CPU ═══");
    println!("  Provenance: hotSpring MD transport → toadStool S70+ (batched ACF shader)");
    println!();

    for &n_atoms in &[64, 256, 1024] {
        let n_frames = 200;
        let n_lags = 100;

        let velocities: Vec<f64> = (0..n_frames * n_atoms * 3)
            .map(|i| (i as f64 * 0.7).sin() * 2.0)
            .collect();

        // GPU path (upstream barracuda)
        let t = Instant::now();
        let gpu_result =
            barracuda::ops::md::compute_vacf_batch(device, &velocities, n_atoms, n_frames, n_lags);
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        // CPU path (hotSpring local)
        let vel_snapshots: Vec<Vec<f64>> = velocities
            .chunks(n_atoms * 3)
            .map(<[f64]>::to_vec)
            .collect();
        let t = Instant::now();
        let cpu_result = hotspring_barracuda::md::observables::compute_vacf(
            &vel_snapshots,
            n_atoms,
            0.01,
            n_lags,
        );
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let speedup = cpu_ms / gpu_ms.max(0.001);
        let gpu_ok = if let Ok(ref g) = gpu_result {
            format!("C(0)={:.4e}", g[0])
        } else {
            "FAIL".to_string()
        };

        println!(
            "  N={n_atoms:>5}, {n_frames} frames, {n_lags} lags: GPU={gpu_ms:.1}ms CPU={cpu_ms:.1}ms (×{speedup:.1}) [{gpu_ok}, CPU D*={:.4e}]",
            cpu_result.diffusion_coeff
        );
    }
    println!();
}

pub fn bench_stress_virial_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2d: Stress Virial GPU (barracuda::ops::md) ═══");
    println!("  Provenance: hotSpring MD transport → toadStool S70+ (ComputeDispatch)");
    println!();

    for &n_atoms in &[100, 500, 2000] {
        let positions: Vec<f64> = (0..n_atoms * 3)
            .map(|i| (i as f64 * 0.01).sin() * 5.0)
            .collect();
        let velocities: Vec<f64> = (0..n_atoms * 3)
            .map(|i| (i as f64 * 0.02).cos() * 0.5)
            .collect();
        let forces: Vec<f64> = (0..n_atoms * 3)
            .map(|i| (i as f64 * 0.03).sin() * 0.1)
            .collect();
        let masses: Vec<f64> = vec![1.0; n_atoms];
        let volume = 1000.0;

        let t = Instant::now();
        let result = barracuda::ops::md::compute_stress_virial(
            device,
            &positions,
            &velocities,
            &forces,
            &masses,
            volume,
        );
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match &result {
            Ok(sigma) => format!("σ_xx={:.4e}, σ_xy={:.4e}", sigma[0], sigma[3]),
            Err(e) => format!("ERR: {e}"),
        };

        println!("  N={n_atoms:>5}: {ms:.1}ms [{status}]");
    }
    println!();
}
