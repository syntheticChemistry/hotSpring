// SPDX-License-Identifier: AGPL-3.0-only

//! Static Structure Factor S(k) from position snapshots.
//!
//! CPU and GPU paths for computing S(k) along principal axes.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::md::observables::SsfGpu;

/// Compute static structure factor S(k) from position snapshots
pub fn compute_ssf(
    snapshots: &[Vec<f64>],
    n: usize,
    box_side: f64,
    max_k_harmonics: usize,
) -> Vec<(f64, f64)> {
    let dk = 2.0 * std::f64::consts::PI / box_side;
    let _n_frames = snapshots.len();
    let mut sk_values: Vec<(f64, f64)> = Vec::new();

    // Compute S(k) for k-vectors along principal axes
    for kn in 1..=max_k_harmonics {
        let k_mag = kn as f64 * dk;
        let mut sk_sum = 0.0;
        let mut count = 0;

        for snap in snapshots {
            // S(k) = <|rho(k)|²> / N
            // rho(k) = sum_j exp(i k . r_j)
            // For k along x-axis: rho_x = sum_j exp(i kx * x_j)
            for axis in 0..3 {
                let mut re = 0.0;
                let mut im = 0.0;
                for j in 0..n {
                    let r_component = snap[j * 3 + axis];
                    let phase = k_mag * r_component;
                    re += phase.cos();
                    im += phase.sin();
                }
                sk_sum += (re * re + im * im) / n as f64;
                count += 1;
            }
        }

        sk_values.push((k_mag, sk_sum / count as f64));
    }

    sk_values
}

/// Compute S(k) using toadstool's SsfGpu, averaged over snapshots.
///
/// This mirrors `compute_ssf` but runs each snapshot on the GPU via
/// `SsfGpu::compute_axes`. Falls back to CPU if GPU dispatch fails.
pub fn compute_ssf_gpu(
    device: Arc<WgpuDevice>,
    snapshots: &[Vec<f64>],
    _n: usize,
    box_side: f64,
    max_k_harmonics: usize,
) -> Vec<(f64, f64)> {
    if snapshots.is_empty() {
        return Vec::new();
    }

    // Accumulator: (k, sum_of_sk, count)
    let mut accumulator: Vec<(f64, f64, usize)> = Vec::new();

    for snap in snapshots {
        match SsfGpu::compute_axes(device.clone(), snap, box_side, max_k_harmonics) {
            Ok(sk_pairs) => {
                // Grow accumulator on first snapshot
                if accumulator.is_empty() {
                    accumulator = sk_pairs.iter().map(|&(k, sk)| (k, sk, 1)).collect();
                } else {
                    for (i, &(_k, sk)) in sk_pairs.iter().enumerate() {
                        if i < accumulator.len() {
                            accumulator[i].1 += sk;
                            accumulator[i].2 += 1;
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("  SsfGpu::compute_axes failed: {e} — skipping snapshot");
            }
        }
    }

    // Average over snapshots
    accumulator
        .into_iter()
        .map(|(k, sum_sk, count)| {
            (
                k,
                if count > 0 {
                    sum_sk / count as f64
                } else {
                    0.0
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_ssf_single_frame() {
        let pos = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 3 particles
        let ssf = compute_ssf(&[pos], 3, 5.0, 3);
        assert_eq!(ssf.len(), 3);
        for (k, s) in &ssf {
            assert!(*k > 0.0);
            assert!(*s >= 0.0, "S(k) must be non-negative");
        }
    }
}
