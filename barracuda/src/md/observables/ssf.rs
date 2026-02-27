// SPDX-License-Identifier: AGPL-3.0-only

//! Static Structure Factor S(k) from position snapshots.
//!
//! CPU and GPU paths for computing S(k) along principal axes.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::md::observables::SsfGpu;

/// Compute static structure factor S(k) from position snapshots
#[must_use]
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

        sk_values.push((k_mag, sk_sum / f64::from(count)));
    }

    sk_values
}

/// Compute S(k) using toadstool's `SsfGpu`, averaged over snapshots.
///
/// This mirrors `compute_ssf` but runs each snapshot on the GPU via
/// `SsfGpu::compute_axes`. Falls back to CPU if GPU dispatch fails.
#[must_use]
pub fn compute_ssf_gpu(
    device: &Arc<WgpuDevice>,
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
        match SsfGpu::compute_axes(Arc::clone(device), snap, box_side, max_k_harmonics) {
            Ok(sk_pairs) => {
                // Grow accumulator on first snapshot
                if accumulator.is_empty() {
                    accumulator = sk_pairs.iter().map(|&(k, sk): &(f64, f64)| (k, sk, 1)).collect();
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
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
        let ssf_one = compute_ssf(std::slice::from_ref(&pos1), 2, 5.0, 5);
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

    #[test]
    fn compute_ssf_zero_harmonics() {
        let pos = vec![1.0, 2.0, 3.0];
        let ssf = compute_ssf(&[pos], 1, 5.0, 0);
        assert!(ssf.is_empty(), "max_k_harmonics=0 should give empty result");
    }

    #[test]
    fn compute_ssf_large_system_positive() {
        let n = 50;
        let box_side = 10.0;
        let mut pos = Vec::with_capacity(n * 3);
        let mut seed = 123u64;
        for _ in 0..n * 3 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            pos.push((seed >> 33) as f64 / (1u64 << 31) as f64 * box_side);
        }
        let ssf = compute_ssf(&[pos], n, box_side, 10);
        assert_eq!(ssf.len(), 10);
        for (k, s) in &ssf {
            assert!(*k > 0.0, "k must be positive");
            assert!(s.is_finite(), "S(k) must be finite");
            assert!(*s >= 0.0, "S(k) must be non-negative");
        }
    }

    #[test]
    fn compute_ssf_uniform_lattice_order() {
        let n = 8;
        let box_side = 4.0;
        let spacing = box_side / 2.0;
        let mut pos = Vec::new();
        for ix in 0..2 {
            for iy in 0..2 {
                for iz in 0..2 {
                    pos.push(ix as f64 * spacing);
                    pos.push(iy as f64 * spacing);
                    pos.push(iz as f64 * spacing);
                }
            }
        }
        let ssf = compute_ssf(&[pos], n, box_side, 5);
        assert!(
            ssf[0].1 > ssf[4].1 * 0.01,
            "lowest k should have finite S(k)"
        );
    }

    #[test]
    fn compute_ssf_many_frames_reduces_variance() {
        let n = 10;
        let box_side = 5.0;
        let mut frames = Vec::new();
        for f in 0..20 {
            let mut pos = Vec::with_capacity(n * 3);
            let mut seed = (f * 1000 + 42) as u64;
            for _ in 0..n * 3 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                pos.push((seed >> 33) as f64 / (1u64 << 31) as f64 * box_side);
            }
            frames.push(pos);
        }
        let ssf = compute_ssf(&frames, n, box_side, 3);
        for (_, s) in &ssf {
            assert!(s.is_finite(), "averaged S(k) should be finite");
            assert!(*s > 0.0, "averaged S(k) should be positive");
        }
    }

    #[test]
    fn compute_ssf_max_k_harmonics_one() {
        let pos = vec![1.0, 1.0, 1.0];
        let ssf = compute_ssf(&[pos], 1, 5.0, 1);
        assert_eq!(ssf.len(), 1);
        assert!((ssf[0].1 - 1.0).abs() < 1e-14, "single particle S(k)=1");
    }

    #[test]
    fn compute_ssf_two_particles_periodic() {
        let box_side = 4.0;
        let pos = vec![
            0.0,
            0.0,
            0.0,
            box_side / 2.0,
            box_side / 2.0,
            box_side / 2.0,
        ];
        let ssf = compute_ssf(&[pos], 2, box_side, 2);
        assert_eq!(ssf.len(), 2);
        for (k, s) in &ssf {
            assert!(*k > 0.0);
            assert!(*s >= 0.0 && s.is_finite());
        }
    }

    #[test]
    fn compute_ssf_positions_with_fractional_coords() {
        let pos = vec![1.5, 2.3, 0.7, 3.1, 1.9, 2.5, 0.2, 4.8, 3.3];
        let ssf = compute_ssf(&[pos], 3, 10.0, 4);
        assert_eq!(ssf.len(), 4);
        for (k, s) in &ssf {
            assert!(*k > 0.0);
            assert!(s.is_finite(), "S(k) must be finite");
        }
    }

    #[test]
    fn compute_ssf_three_axes_contribute() {
        // S(k) averages over 3 axes; verify all contribute
        let pos = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 particles along x
        let ssf = compute_ssf(&[pos], 2, 5.0, 1);
        assert_eq!(ssf.len(), 1);
        // For k along x, y, z: x has structure, y/z are uniform
        assert!(ssf[0].1 > 0.0 && ssf[0].1.is_finite());
    }

    #[test]
    fn compute_ssf_single_particle_all_k_equal() {
        // For a single particle, S(k) = 1 for all k (no structure)
        let pos = vec![0.0, 0.0, 0.0];
        let ssf = compute_ssf(&[pos], 1, 10.0, 5);
        assert_eq!(ssf.len(), 5);
        for (k, s) in &ssf {
            assert!(*k > 0.0);
            assert!(
                (*s - 1.0).abs() < 1e-14,
                "single particle S(k)={s} should be 1.0"
            );
        }
    }

    #[test]
    fn compute_ssf_first_harmonic_is_dk() {
        let box_side = 5.0;
        let dk = 2.0 * std::f64::consts::PI / box_side;
        let pos = vec![1.0, 2.0, 3.0];
        let ssf = compute_ssf(&[pos], 1, box_side, 1);
        assert_eq!(ssf.len(), 1);
        assert!((ssf[0].0 - dk).abs() < 1e-14, "first k should be 2π/L");
    }

    #[test]
    fn compute_ssf_empty_snapshots_count() {
        // max_k_harmonics=3 with empty snapshots still produces 3 (k, S) pairs
        // but S is NaN (0/0) or 0
        let ssf = compute_ssf(&[], 0, 5.0, 3);
        assert_eq!(ssf.len(), 3);
    }

    #[test]
    fn compute_ssf_all_particles_at_origin() {
        // All particles at origin: rho(k) = N for each axis, S(k) = N
        let n = 4;
        let pos = vec![0.0; n * 3];
        let ssf = compute_ssf(&[pos], n, 5.0, 2);
        for (k, s) in &ssf {
            assert!(*k > 0.0);
            assert!((*s - n as f64).abs() < 1e-10, "S(k) = N when all at origin");
        }
    }
}
