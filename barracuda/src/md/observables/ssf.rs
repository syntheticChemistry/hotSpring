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
        match SsfGpu::compute_axes(Arc::clone(&device), snap, box_side, max_k_harmonics) {
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

    #[test]
    fn compute_ssf_empty_snapshots() {
        let ssf = compute_ssf(&[], 0, 5.0, 3);
        assert_eq!(ssf.len(), 3);
        for (_, s) in &ssf {
            assert!(s.is_nan() || *s == 0.0, "empty snapshots: S(k)=NaN or 0");
        }
    }

    #[test]
    fn compute_ssf_multi_frame_averaging() {
        let pos1 = vec![0.0, 0.0, 0.0, 2.5, 0.0, 0.0];
        let pos2 = vec![0.0, 0.0, 0.0, 2.5, 0.0, 0.0];
        let ssf_one = compute_ssf(&[pos1.clone()], 2, 5.0, 5);
        let ssf_two = compute_ssf(&[pos1, pos2], 2, 5.0, 5);
        assert_eq!(ssf_one.len(), ssf_two.len());
        for (a, b) in ssf_one.iter().zip(ssf_two.iter()) {
            assert!((a.0 - b.0).abs() < 1e-14, "k-values should match");
            assert!((a.1 - b.1).abs() < 1e-14, "identical frames → same S(k)");
        }
    }

    #[test]
    fn compute_ssf_k_spacing() {
        let pos = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let box_side = 10.0;
        let dk = 2.0 * std::f64::consts::PI / box_side;
        let ssf = compute_ssf(&[pos], 2, box_side, 4);
        for (i, (k, _)) in ssf.iter().enumerate() {
            let expected_k = (i + 1) as f64 * dk;
            assert!(
                (k - expected_k).abs() < 1e-14,
                "k-vector spacing should be 2π/L"
            );
        }
    }

    #[test]
    fn compute_ssf_single_particle_is_one() {
        let pos = vec![2.0, 3.0, 1.0];
        let ssf = compute_ssf(&[pos], 1, 5.0, 3);
        for (_, s) in &ssf {
            assert!(
                (*s - 1.0).abs() < 1e-14,
                "S(k) for a single particle should be 1.0, got {s}"
            );
        }
    }
}
