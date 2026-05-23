// SPDX-License-Identifier: AGPL-3.0-or-later

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;

pub async fn bench_nelder_mead_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2e: Batched Nelder-Mead GPU (barracuda::optimize) ═══");
    println!("  Provenance: neuralSpring parameter optimization → toadStool S79");
    println!("  Cross-spring: hotSpring HMC parameter tuning benefits from GPU batch optimizer");
    println!();

    use barracuda::optimize::{BatchNelderMeadConfig, batched_nelder_mead_gpu};

    for &(n_problems, dims) in &[(10_usize, 2_usize), (100, 3), (1000, 2)] {
        let config = BatchNelderMeadConfig {
            dims,
            max_iters: 200,
            tol: 1e-8,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        };

        let n_vertices = dims + 1;
        let simplices: Vec<f64> = (0..n_problems * n_vertices * dims)
            .map(|i| {
                let problem = i / (n_vertices * dims);
                let vertex = (i / dims) % n_vertices;
                let dim = i % dims;
                if vertex == 0 {
                    (problem as f64 * 0.1).sin()
                } else if dim == vertex - 1 {
                    1.0 + (problem as f64 * 0.1).sin()
                } else {
                    (problem as f64 * 0.1).sin()
                }
            })
            .collect();

        let t = Instant::now();
        let result = batched_nelder_mead_gpu(device, &config, n_problems, &simplices, |points| {
            points
                .chunks(dims)
                .map(|p| p.iter().map(|x| x * x).sum::<f64>())
                .collect()
        })
        .await;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match &result {
            Ok(results) => {
                let converged = results.iter().filter(|r| r.converged).count();
                let best = results.first().map_or(f64::NAN, |r| r.best_value);
                format!("{converged}/{n_problems} converged, best={best:.2e}")
            }
            Err(e) => format!("ERR: {e}"),
        };

        println!("  n={n_problems:>5}, dims={dims}: {ms:.1}ms [{status}]");
    }
    println!();
}
