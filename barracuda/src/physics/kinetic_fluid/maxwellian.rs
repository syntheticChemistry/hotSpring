// SPDX-License-Identifier: AGPL-3.0-only

//! 1D Maxwellian distribution and velocity-space moment computation.
//!
//! Shared primitive used by BGK relaxation and kinetic-fluid coupling.

use std::f64::consts::PI;

/// 1D Maxwellian: `f(v) = n * sqrt(m/(2πT)) * exp(-m(v-u)²/(2T))`
#[must_use]
pub fn maxwellian_1d(v: &[f64], n: f64, u: f64, t: f64, m: f64) -> Vec<f64> {
    let t_safe = t.max(1e-12);
    let coeff = n * (m / (2.0 * PI * t_safe)).sqrt();
    v.iter()
        .map(|&vi| {
            let exponent = -m * (vi - u).powi(2) / (2.0 * t_safe);
            coeff * exponent.max(-500.0).exp()
        })
        .collect()
}

/// Moments of a 1D distribution: (density, velocity, temperature, energy).
///
/// In 1D: `E = (m/2) ∫ v² f dv = (n/2)(mu² + T)`, so `T = 2E/n - mu²`.
#[must_use]
pub fn compute_moments(f: &[f64], v: &[f64], dv: f64, m: f64) -> (f64, f64, f64, f64) {
    let n: f64 = f.iter().sum::<f64>() * dv;
    if n < 1e-30 {
        return (0.0, 0.0, 1e-12, 0.0);
    }
    let u: f64 = f.iter().zip(v).map(|(&fi, &vi)| fi * vi).sum::<f64>() * dv / n;
    let e: f64 = 0.5 * m * f.iter().zip(v).map(|(&fi, &vi)| fi * vi * vi).sum::<f64>() * dv;
    let t = (2.0 * e / n - m * u * u).max(1e-12);
    (n, u, t, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn velocity_grid(nv: usize, v_max: f64) -> (Vec<f64>, f64) {
        let v: Vec<f64> = (0..nv)
            .map(|i| -v_max + i as f64 * 2.0 * v_max / (nv - 1) as f64)
            .collect();
        let dv = v[1] - v[0];
        (v, dv)
    }

    #[test]
    fn maxwellian_normalizes() {
        let (v, dv) = velocity_grid(1001, 10.0);
        let f = maxwellian_1d(&v, 1.0, 0.0, 1.0, 1.0);
        let integral: f64 = f.iter().sum::<f64>() * dv;
        assert!((integral - 1.0).abs() < 0.01, "integral = {integral}");
    }

    #[test]
    fn maxwellian_mean_velocity() {
        let (v, dv) = velocity_grid(1001, 10.0);
        let u0 = 0.5;
        let f = maxwellian_1d(&v, 1.0, u0, 1.0, 1.0);
        let mean: f64 = f
            .iter()
            .zip(v.iter())
            .map(|(&fi, &vi)| fi * vi)
            .sum::<f64>()
            * dv;
        assert!((mean - u0).abs() < 0.02, "mean = {mean}, expected {u0}");
    }

    #[test]
    fn moments_roundtrip() {
        let (v, dv) = velocity_grid(501, 10.0);
        let (n0, u0, t0, m0) = (2.0, 0.3, 1.5, 1.0);
        let f = maxwellian_1d(&v, n0, u0, t0, m0);
        let (n, u, t, _) = compute_moments(&f, &v, dv, m0);
        assert!((n - n0).abs() < 0.05, "n={n}");
        assert!((u - u0).abs() < 0.05, "u={u}");
        assert!((t - t0).abs() < 0.1, "T={t}");
    }
}
