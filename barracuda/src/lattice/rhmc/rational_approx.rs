// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rational approximation to fractional powers: r(x) ≈ x^{p/q}.
//!
//! Uses a partial-fraction form r(x) = α₀ + Σ αᵢ / (x + σᵢ) fitted via
//! the Remez exchange algorithm with coordinate-descent pole optimisation.

use super::remez::remez_for_poles;

/// Partial-fraction representation of a rational approximation:
///   r(x) = `alpha_0` + sum_{i=1}^{n} `alpha_i` / (x + `sigma_i`)
///
/// Used to approximate x^{p/q} over [`lambda_min`, `lambda_max`].
#[derive(Clone, Debug)]
pub struct RationalApproximation {
    /// Constant term.
    pub alpha_0: f64,
    /// Residues (numerator coefficients).
    pub alpha: Vec<f64>,
    /// Shifts (denominator offsets, all positive).
    pub sigma: Vec<f64>,
    /// Power being approximated (e.g. 0.25 for x^{1/4}).
    pub power: f64,
    /// Spectral range lower bound.
    pub lambda_min: f64,
    /// Spectral range upper bound.
    pub lambda_max: f64,
    /// Maximum relative error over the spectral range.
    pub max_relative_error: f64,
}

impl RationalApproximation {
    /// Number of poles (shifts) in the approximation.
    #[must_use]
    pub const fn n_poles(&self) -> usize {
        self.alpha.len()
    }

    /// Evaluate the rational approximation at point x.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        let mut val = self.alpha_0;
        for (a, s) in self.alpha.iter().zip(self.sigma.iter()) {
            val += a / (x + s);
        }
        val
    }

    /// Generate an n-pole rational approximation to x^power on [`lambda_min`, `lambda_max`].
    ///
    /// Uses geometric pole initialization, Remez exchange for residue fitting,
    /// and coordinate descent for pole optimization.
    /// For production RHMC, replace with Remez-optimal coefficients.
    #[must_use]
    pub fn generate(power: f64, n_poles: usize, lambda_min: f64, lambda_max: f64) -> Self {
        let log_min = lambda_min.ln();
        let log_max = lambda_max.ln();

        let mut sigma: Vec<f64> = (0..n_poles)
            .map(|i| {
                let t = (i as f64 + 0.5) / n_poles as f64;
                (log_min + t * (log_max - log_min)).exp()
            })
            .collect();

        let n_eval = 4000;
        let eval_grid: Vec<f64> = (0..n_eval)
            .map(|i| {
                let t = f64::from(i) / f64::from(n_eval - 1);
                (log_min + t * (log_max - log_min)).exp()
            })
            .collect();

        let (mut best_coeffs, mut best_err) = remez_for_poles(&sigma, power, &eval_grid);

        // Coordinate descent on pole positions in log space
        for _ in 0..40 {
            let mut improved = false;
            for i in 0..n_poles {
                let log_s = sigma[i].ln();
                let lower = if i > 0 {
                    sigma[i - 1].ln() + 0.01
                } else {
                    log_min - 3.0
                };
                let upper = if i + 1 < n_poles {
                    sigma[i + 1].ln() - 0.01
                } else {
                    log_max + 3.0
                };

                // Golden section search for optimal pole position
                let gr = 0.5 * (5.0_f64.sqrt() - 1.0);
                let mut a = lower;
                let mut b = upper;
                let mut c = b - gr * (b - a);
                let mut d = a + gr * (b - a);

                sigma[i] = c.exp();
                let (_, mut fc) = remez_for_poles(&sigma, power, &eval_grid);
                sigma[i] = d.exp();
                let (_, mut fd) = remez_for_poles(&sigma, power, &eval_grid);

                for _ in 0..25 {
                    if (b - a).abs() < 0.005 {
                        break;
                    }
                    if fc < fd {
                        b = d;
                        d = c;
                        fd = fc;
                        c = b - gr * (b - a);
                        sigma[i] = c.exp();
                        let (_, e) = remez_for_poles(&sigma, power, &eval_grid);
                        fc = e;
                    } else {
                        a = c;
                        c = d;
                        fc = fd;
                        d = a + gr * (b - a);
                        sigma[i] = d.exp();
                        let (_, e) = remez_for_poles(&sigma, power, &eval_grid);
                        fd = e;
                    }
                }

                let (best_pos, best_pos_err) = if fc < fd { (c, fc) } else { (d, fd) };

                sigma[i] = log_s.exp();
                let (_, cur_err) = remez_for_poles(&sigma, power, &eval_grid);
                if best_pos_err < cur_err * 0.998 {
                    sigma[i] = best_pos.exp();
                    improved = true;
                }
            }

            let (c, e) = remez_for_poles(&sigma, power, &eval_grid);
            if e < best_err {
                best_err = e;
                best_coeffs = c;
            }
            if !improved {
                break;
            }
        }

        Self {
            alpha_0: best_coeffs[0],
            alpha: best_coeffs[1..].to_vec(),
            sigma,
            power,
            lambda_min,
            lambda_max,
            max_relative_error: best_err,
        }
    }

    /// 8-pole approximation to x^{1/4} on [0.01, 64].
    #[must_use]
    pub fn fourth_root_8pole() -> Self {
        Self::generate(0.25, 8, 0.01, 64.0)
    }

    /// 8-pole approximation to x^{-1/4} on [0.01, 64].
    #[must_use]
    pub fn inv_fourth_root_8pole() -> Self {
        Self::generate(-0.25, 8, 0.01, 64.0)
    }

    /// 8-pole approximation to x^{1/2} on [0.01, 64].
    #[must_use]
    pub fn sqrt_8pole() -> Self {
        Self::generate(0.5, 8, 0.01, 64.0)
    }

    /// 8-pole approximation to x^{-1/2} on [0.01, 64].
    #[must_use]
    pub fn inv_sqrt_8pole() -> Self {
        Self::generate(-0.5, 8, 0.01, 64.0)
    }
}
