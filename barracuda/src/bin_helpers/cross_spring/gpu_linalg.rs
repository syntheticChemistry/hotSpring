// SPDX-License-Identifier: AGPL-3.0-or-later

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;

pub fn bench_gemm_transpose_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2f: GemmF64 Transpose (barracuda::ops::linalg::GemmF64) ═══");
    println!("  Provenance: neuralSpring Tikhonov/least-squares → barraCuda Sprint 6");
    println!("  Cross-spring: A^T*B without materializing transpose — Gram matrices,");
    println!("    normal equations, covariance. Used by neuralSpring regression,");
    println!("    hotSpring surrogate fitting, groundSpring least-squares");
    println!();

    use barracuda::ops::linalg::gemm_f64::GemmF64;

    for &(m, k, n) in &[(64_usize, 128, 32), (256, 512, 64), (512, 1024, 128)] {
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.01).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.02).cos()).collect();

        // Standard A*B
        let t = Instant::now();
        let result_ab = GemmF64::execute(Arc::clone(device), &a, &b, m, k, n, 1);
        let ms_ab = t.elapsed().as_secs_f64() * 1000.0;

        // A^T*B (storage: k×m, logically transposed to m×k, then multiplied by k×n)
        let a_for_trans: Vec<f64> = (0..k * m).map(|i| (i as f64 * 0.01).sin()).collect();
        let t = Instant::now();
        let result_atb = GemmF64::execute_gemm_ex(
            Arc::clone(device),
            &a_for_trans,
            &b,
            m,
            k,
            n,
            1,
            1.0,
            0.0,
            true,
            false,
        );
        let ms_atb = t.elapsed().as_secs_f64() * 1000.0;

        let ab_ok = result_ab
            .as_ref()
            .map_or_else(|e| format!("ERR: {e}"), |v| format!("ok, len={}", v.len()));
        let atb_ok = result_atb
            .as_ref()
            .map_or_else(|e| format!("ERR: {e}"), |v| format!("ok, len={}", v.len()));
        println!(
            "  {m}×{k} * {k}×{n}: A*B={ms_ab:.1}ms [{ab_ok}] | A^T*B={ms_atb:.1}ms [{atb_ok}]"
        );
    }
    println!();
}
