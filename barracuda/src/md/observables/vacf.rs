// SPDX-License-Identifier: AGPL-3.0-only

//! Velocity Autocorrelation Function (VACF) and Mean-Square Displacement (MSD).
//!
//! Two VACF paths:
//! - `compute_vacf` — CPU post-process from `Vec<f64>` snapshots
//! - `compute_vacf_upstream_gpu` — GPU via barracuda's `compute_vacf_batch`
//!   (cross-spring: shader evolved from hotSpring's batched VACF design,
//!   absorbed into toadStool S70+ with wetSpring's ODE batch pattern)
//!
//! The upstream GPU path flattens hotSpring's `&[Vec<f64>]` into the
//! frame-major `[n_frames × N × 3]` layout that barracuda expects, runs
//! the GPU shader, then applies Green-Kubo integration + plateau detection.

use std::sync::Arc;

use barracuda::device::WgpuDevice;

use crate::tolerances::DIVISION_GUARD;

/// Plateau detection: D* considered converged after this many
/// seconds of non-increasing integral (expressed as time / `dt_dump`).
const PLATEAU_DETECTION_TIME: f64 = 20.0;

/// Velocity autocorrelation function result.
#[derive(Clone, Debug)]
pub struct Vacf {
    /// Lag times in units of ω_p⁻¹.
    pub t_values: Vec<f64>,
    /// Normalized C(t) / C(0).
    pub c_values: Vec<f64>,
    /// Self-diffusion coefficient D* = (1/3) ∫ C(t) dt.
    pub diffusion_coeff: f64,
}

/// Compute VACF from velocity snapshots (CPU post-process)
#[must_use]
pub fn compute_vacf(vel_snapshots: &[Vec<f64>], n: usize, dt_dump: f64, max_lag: usize) -> Vacf {
    let n_frames = vel_snapshots.len();
    let n_lag = max_lag.min(n_frames);
    let mut c_values = vec![0.0f64; n_lag];
    let mut counts = vec![0usize; n_lag];

    for t0 in 0..n_frames {
        for lag in 0..n_lag {
            let t1 = t0 + lag;
            if t1 >= n_frames {
                break;
            }
            let v0 = &vel_snapshots[t0];
            let v1 = &vel_snapshots[t1];

            let mut dot_sum = 0.0;
            for i in 0..n {
                dot_sum += v0[i * 3 + 2].mul_add(
                    v1[i * 3 + 2],
                    v0[i * 3 + 1].mul_add(v1[i * 3 + 1], v0[i * 3] * v1[i * 3]),
                );
            }
            c_values[lag] += dot_sum / n as f64;
            counts[lag] += 1;
        }
    }

    // Average over time origins
    for i in 0..n_lag {
        if counts[i] > 0 {
            c_values[i] /= counts[i] as f64;
        }
    }

    // Normalize by C(0)
    let c0 = c_values[0].max(DIVISION_GUARD);
    let c_normalized: Vec<f64> = c_values.iter().map(|&c| c / c0).collect();

    // Diffusion coefficient: D* = (1/3) integral_0^inf C(t) dt
    //
    // Use plateau detection: track the running integral D*(t) and stop at
    // its maximum. Beyond the correlation time, statistical noise in C(t)
    // causes D*(t) to random-walk, inflating the estimate. This is the
    // standard Green-Kubo cutoff from MD literature (e.g., Allen & Tildesley).
    let mut integral = 0.0;
    let mut d_star_max = 0.0;
    let mut plateau_count = 0;
    let plateau_window = (PLATEAU_DETECTION_TIME / dt_dump).ceil() as usize;

    for i in 1..n_lag {
        integral += (0.5 * dt_dump).mul_add(c_values[i - 1] + c_values[i], 0.0);
        let d_star_running = integral / 3.0;

        if d_star_running > d_star_max {
            d_star_max = d_star_running;
            plateau_count = 0;
        } else {
            plateau_count += 1;
            if plateau_count > plateau_window {
                break;
            }
        }
    }
    let diffusion_coeff = d_star_max;

    let t_values: Vec<f64> = (0..n_lag).map(|i| i as f64 * dt_dump).collect();

    Vacf {
        t_values,
        c_values: c_normalized,
        diffusion_coeff,
    }
}

/// Compute VACF on GPU via barracuda's upstream `compute_vacf_batch`.
///
/// Flattens hotSpring's snapshot format (`&[Vec<f64>]`) into the frame-major
/// `[n_frames × N × 3]` flat layout, dispatches one GPU kernel, then applies
/// the same Green-Kubo plateau detection as the CPU path.
///
/// Returns `None` if the GPU op fails (caller should fall back to CPU).
pub fn compute_vacf_upstream_gpu(
    device: &Arc<WgpuDevice>,
    vel_snapshots: &[Vec<f64>],
    n: usize,
    dt_dump: f64,
    max_lag: usize,
) -> Option<Vacf> {
    let n_frames = vel_snapshots.len();
    let n_lag = max_lag.min(n_frames);
    if n_lag < 2 || n_frames < 2 {
        return None;
    }

    let mut flat = Vec::with_capacity(n_frames * n * 3);
    for snap in vel_snapshots {
        flat.extend_from_slice(snap);
    }

    let raw_c = barracuda::ops::md::compute_vacf_batch(device, &flat, n, n_frames, n_lag).ok()?;

    let c0 = raw_c[0].max(DIVISION_GUARD);
    let c_normalized: Vec<f64> = raw_c.iter().map(|&c| c / c0).collect();

    // Green-Kubo integration with plateau detection (same as CPU path)
    let mut integral = 0.0;
    let mut d_star_max = 0.0;
    let mut plateau_count = 0;
    let plateau_window = (PLATEAU_DETECTION_TIME / dt_dump).ceil() as usize;

    for i in 1..n_lag {
        integral += (0.5 * dt_dump).mul_add(raw_c[i - 1] + raw_c[i], 0.0);
        let d_star_running = integral / 3.0;
        if d_star_running > d_star_max {
            d_star_max = d_star_running;
            plateau_count = 0;
        } else {
            plateau_count += 1;
            if plateau_count > plateau_window {
                break;
            }
        }
    }

    let t_values: Vec<f64> = (0..n_lag).map(|i| i as f64 * dt_dump).collect();

    Some(Vacf {
        t_values,
        c_values: c_normalized,
        diffusion_coeff: d_star_max,
    })
}

/// Compute D* from mean-square displacement (Einstein relation).
///
/// D* = lim_{t→∞} <|r(t) - r(0)|²> / (6 t)
///
/// More robust than Green-Kubo VACF for short runs and strongly-coupled systems,
/// because MSD is a cumulative quantity rather than an instantaneous correlation.
/// Uses unwrapped positions (correcting for PBC jumps via velocity integration).
#[must_use]
pub fn compute_d_star_msd(
    pos_snapshots: &[Vec<f64>],
    n: usize,
    box_side: f64,
    dt_snap: f64,
) -> f64 {
    let n_frames = pos_snapshots.len();
    if n_frames < 4 {
        return 0.0;
    }

    let mut unwrapped = pos_snapshots[0].clone();
    let mut all_unwrapped = Vec::with_capacity(n_frames);
    all_unwrapped.push(unwrapped.clone());

    for frame in 1..n_frames {
        let cur = &pos_snapshots[frame];
        let prev = &pos_snapshots[frame - 1];
        for i in 0..n {
            for d in 0..3 {
                let idx = i * 3 + d;
                let mut dr = cur[idx] - prev[idx];
                dr -= box_side * (dr / box_side).round();
                unwrapped[idx] += dr;
            }
        }
        all_unwrapped.push(unwrapped.clone());
    }

    // Compute MSD(lag) = <|r(t0+lag) - r(t0)|²> averaged over t0 and particles.
    // Fit D* from the linear regime: use the second quarter to third quarter of lags
    // (skip initial ballistic regime and noisy tail).
    let max_lag = n_frames / 2;
    let fit_start = max_lag / 4;
    let fit_end = 3 * max_lag / 4;
    if fit_end <= fit_start + 1 {
        return 0.0;
    }

    let mut msd_vals = Vec::with_capacity(fit_end - fit_start);
    let mut time_vals = Vec::with_capacity(fit_end - fit_start);

    for lag in fit_start..fit_end {
        let mut msd_sum = 0.0;
        let mut count = 0;
        for t0 in 0..(n_frames - lag) {
            let t1 = t0 + lag;
            let r0 = &all_unwrapped[t0];
            let r1 = &all_unwrapped[t1];
            for i in 0..n {
                let dx = r1[i * 3] - r0[i * 3];
                let dy = r1[i * 3 + 1] - r0[i * 3 + 1];
                let dz = r1[i * 3 + 2] - r0[i * 3 + 2];
                msd_sum += dz.mul_add(dz, dy.mul_add(dy, dx * dx));
            }
            count += n;
        }
        let msd = msd_sum / count as f64;
        let t = lag as f64 * dt_snap;
        msd_vals.push(msd);
        time_vals.push(t);
    }

    // Linear regression: MSD = 6D*t + offset
    let n_pts = msd_vals.len() as f64;
    let t_mean: f64 = time_vals.iter().sum::<f64>() / n_pts;
    let m_mean: f64 = msd_vals.iter().sum::<f64>() / n_pts;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..msd_vals.len() {
        let dt = time_vals[i] - t_mean;
        num += dt.mul_add(msd_vals[i] - m_mean, 0.0);
        den += dt.mul_add(dt, 0.0);
    }
    let slope = if den > DIVISION_GUARD { num / den } else { 0.0 };
    (slope / 6.0).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // exact known value (1.0)
    fn vacf_single_snapshot_returns() {
        let vel = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2 particles
        let vacf = compute_vacf(&[vel], 2, 0.01, 10);
        assert_eq!(vacf.c_values.len(), 1);
        assert!((vacf.c_values[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn vacf_multiple_frames() {
        // Constant velocities → C(t) = 1 for all t
        let vel = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2 particles
        let vel_snaps = vec![vel; 4];
        let vacf = compute_vacf(&vel_snaps, 2, 0.01, 4);
        assert_eq!(vacf.c_values.len(), 4);
        for (i, &c) in vacf.c_values.iter().enumerate() {
            assert!(
                (c - 1.0).abs() < 1e-10,
                "VACF[{i}] should be 1.0 for constant v, got {c}"
            );
        }
        assert!(vacf.diffusion_coeff >= 0.0);
    }

    #[test]
    fn vacf_diffusion_coeff_non_negative() {
        let vel_snaps = vec![
            vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            vec![0.9, 0.0, 0.0, -0.9, 0.0, 0.0],
        ];
        let vacf = compute_vacf(&vel_snaps, 2, 0.1, 2);
        assert!(vacf.diffusion_coeff >= 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known value (0.0)
    fn compute_d_star_msd_returns_zero_for_few_frames() {
        let pos = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 particles
        assert_eq!(
            compute_d_star_msd(std::slice::from_ref(&pos), 2, 5.0, 0.01),
            0.0
        );
        let snaps = vec![pos; 3];
        assert_eq!(compute_d_star_msd(&snaps[..2], 2, 5.0, 0.01), 0.0);
        assert_eq!(compute_d_star_msd(&snaps[..], 2, 5.0, 0.01), 0.0);
    }

    #[test]
    fn compute_d_star_msd_deterministic() {
        let mut snaps = Vec::new();
        for i in 0..20 {
            let pos = vec![
                f64::from(i) * 0.01,
                0.0,
                0.0,
                1.0 + f64::from(i) * 0.01,
                0.0,
                0.0,
            ]; // 2 particles diffusing
            snaps.push(pos);
        }
        let d1 = compute_d_star_msd(&snaps, 2, 10.0, 0.1);
        let d2 = compute_d_star_msd(&snaps, 2, 10.0, 0.1);
        assert_eq!(d1.to_bits(), d2.to_bits(), "MSD D* should be deterministic");
        assert!(d1 >= 0.0);
    }
}
