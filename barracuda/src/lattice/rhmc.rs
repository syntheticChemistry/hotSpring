// SPDX-License-Identifier: AGPL-3.0-only

//! Rational Hybrid Monte Carlo (RHMC) infrastructure for fractional powers
//! of the fermion determinant.
//!
//! Enables simulation of non-multiples-of-4 flavor counts (Nf=2, 2+1)
//! via the "rooting trick": det(D†D)^{Nf/8} ≈ product of rational functions.
//!
//! # Components
//!
//! - `RationalApproximation`: Partial fraction coefficients for x^{p/q}
//! - `multi_shift_cg_solve`: Solves (D†D + σ_i)x_i = b for all shifts
//!   simultaneously with a single Krylov space
//! - `RhmcConfig`: Configuration for RHMC trajectories
//!
//! # References
//!
//! - Clark & Kennedy, NPB 552 (1999) 461 — RHMC algorithm
//! - Clark, "The Rational Hybrid Monte Carlo Algorithm" (2006)
//! - Remez algorithm for optimal rational approximation

use super::complex_f64::Complex64;
use super::dirac::{apply_dirac_sq, FermionField};
use super::wilson::Lattice;

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

/// Result of a multi-shift CG solve.
#[derive(Clone, Debug)]
pub struct MultiShiftCgResult {
    /// Number of CG iterations (shared across all shifts).
    pub iterations: usize,
    /// Final residual norm squared (for the base system, sigma=0).
    pub residual_sq: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

/// Multi-shift Conjugate Gradient: solve (D†D + `σ_i)x_i` = b for all shifts simultaneously.
///
/// All shifted systems share the same Krylov space. Only one matrix-vector
/// product (D†D·p) per iteration, regardless of the number of shifts.
///
/// Returns solution vectors `x_i` (one per shift) and iteration count.
#[must_use]
pub fn multi_shift_cg_solve(
    lattice: &Lattice,
    b: &FermionField,
    mass: f64,
    shifts: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<FermionField>, MultiShiftCgResult) {
    let vol = lattice.volume();
    let n_shifts = shifts.len();

    let mut x: Vec<FermionField> = (0..n_shifts).map(|_| FermionField::zeros(vol)).collect();
    let mut p: Vec<FermionField> = (0..n_shifts)
        .map(|_| FermionField {
            data: b.data.clone(),
            volume: vol,
        })
        .collect();
    let mut r = FermionField {
        data: b.data.clone(),
        volume: vol,
    };

    let b_norm_sq = b.dot(b).re;
    if b_norm_sq < 1e-30 {
        return (
            x,
            MultiShiftCgResult {
                iterations: 0,
                residual_sq: 0.0,
                converged: true,
            },
        );
    }

    let tol_sq = tol * tol * b_norm_sq;
    let mut rz = b_norm_sq;
    let mut zeta_prev: Vec<f64> = vec![1.0; n_shifts];
    let mut zeta_curr: Vec<f64> = vec![1.0; n_shifts];
    let mut beta_prev: Vec<f64> = vec![0.0; n_shifts];
    let mut alpha_prev = 0.0_f64;
    let mut active: Vec<bool> = vec![true; n_shifts];

    let mut iterations = 0;

    for _iter in 0..max_iter {
        iterations += 1;

        // A·p_0 = (D†D + σ_0)·p_0, but we apply (D†D)·p_0 and add σ_0·p_0 separately.
        // For the base system (shift 0), we track r (shared residual).
        let ap = apply_dirac_sq(lattice, &p[0], mass);

        let mut p_ap = p[0].dot(&ap).re;
        p_ap += shifts[0] * p[0].dot(&p[0]).re;

        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / p_ap;

        // Update x_0 and r
        for i in 0..vol {
            for c in 0..3 {
                x[0].data[i][c] += p[0].data[i][c].scale(alpha);
                r.data[i][c] -= (ap.data[i][c] + p[0].data[i][c].scale(shifts[0])).scale(alpha);
            }
        }

        let rz_new = r.dot(&r).re;

        // Update shifted systems
        for s in 1..n_shifts {
            if !active[s] {
                continue;
            }

            let ds = shifts[s] - shifts[0];
            let denom = alpha.mul_add(ds, 1.0)
                + alpha * alpha_prev * (1.0 - zeta_prev[s] / zeta_curr[s])
                    / beta_prev[s].max(1e-30);
            if denom.abs() < 1e-30 {
                active[s] = false;
                continue;
            }

            let zeta_next = zeta_curr[s] / denom;
            let alpha_s = alpha * zeta_next / zeta_curr[s];

            for i in 0..vol {
                for c in 0..3 {
                    x[s].data[i][c] += p[s].data[i][c].scale(alpha_s);
                }
            }

            let beta_s = if rz.abs() > 1e-30 {
                (zeta_next / zeta_curr[s]).powi(2) * (rz_new / rz)
            } else {
                0.0
            };

            for i in 0..vol {
                for c in 0..3 {
                    p[s].data[i][c] = r.data[i][c].scale(zeta_next) + p[s].data[i][c].scale(beta_s);
                }
            }

            zeta_prev[s] = zeta_curr[s];
            zeta_curr[s] = zeta_next;
            beta_prev[s] = beta_s;
        }

        let beta = if rz.abs() > 1e-30 { rz_new / rz } else { 0.0 };
        for i in 0..vol {
            for c in 0..3 {
                p[0].data[i][c] = r.data[i][c] + p[0].data[i][c].scale(beta);
            }
        }

        alpha_prev = alpha;
        rz = rz_new;

        if rz < tol_sq {
            break;
        }
    }

    (
        x,
        MultiShiftCgResult {
            iterations,
            residual_sq: rz / b_norm_sq,
            converged: rz < tol_sq,
        },
    )
}

/// RHMC configuration for a single fermion taste/flavor sector.
#[derive(Clone, Debug)]
pub struct RhmcFermionConfig {
    /// Mass for this sector.
    pub mass: f64,
    /// Power of the determinant (e.g. 1/4 for one rooted staggered taste).
    pub det_power: f64,
    /// Rational approximation for the action: `x^{det_power`}.
    pub action_approx: RationalApproximation,
    /// Rational approximation for the heatbath: x^{-det_power/2}.
    pub heatbath_approx: RationalApproximation,
    /// Rational approximation for the force: `x^{det_power`} (derivative form).
    pub force_approx: RationalApproximation,
}

/// RHMC HMC configuration.
#[derive(Clone, Debug)]
pub struct RhmcConfig {
    /// Fermion sectors (one per flavor group).
    pub sectors: Vec<RhmcFermionConfig>,
    /// Gauge coupling.
    pub beta: f64,
    /// MD step size.
    pub dt: f64,
    /// Number of MD steps.
    pub n_md_steps: usize,
    /// CG tolerance.
    pub cg_tol: f64,
    /// CG max iterations.
    pub cg_max_iter: usize,
}

impl RhmcConfig {
    /// Calibrated Nf=2 configuration using discovered spectral range.
    ///
    /// All parameters derived from measurements — no hardcoded spectral
    /// bounds. The spectral range comes from `SpectralInfo` (measured via
    /// GPU power iteration), and n_poles/dt/n_md are set by the calibrator.
    #[must_use]
    pub fn calibrated_nf2(
        mass: f64,
        beta: f64,
        n_poles: usize,
        range_min: f64,
        range_max: f64,
        dt: f64,
        n_md_steps: usize,
        cg_tol: f64,
        cg_max_iter: usize,
    ) -> Self {
        let det_power = 0.25; // Nf/8 = 2/8
        let action_force = RationalApproximation::generate(-det_power, n_poles, range_min, range_max);
        Self {
            sectors: vec![RhmcFermionConfig {
                mass,
                det_power,
                action_approx: action_force.clone(),
                heatbath_approx: RationalApproximation::generate(
                    det_power / 2.0,
                    n_poles,
                    range_min,
                    range_max,
                ),
                force_approx: action_force,
            }],
            beta,
            dt,
            n_md_steps,
            cg_tol,
            cg_max_iter,
        }
    }

    /// Nf=2 configuration: one rooted staggered field with det(D†D)^{1/4}.
    ///
    /// Staggered fermions produce 4 tastes, so Nf physical flavors needs
    /// det(D†D)^{Nf/8}. For Nf=2: α = 1/4.
    ///
    /// Pseudofermion setup (Clark & Kennedy, NPB 552):
    /// - Action:   S_f = φ† (D†D)^{-α} φ  →  r_act(x) ≈ x^{-1/4}
    /// - Heatbath: φ = (D†D)^{α/2} η       →  r_hb(x)  ≈ x^{+1/8}
    /// - Force:    dS_f/dU uses r_act
    ///
    /// Consistency: r_hb(x)² · r_act(x) = x^{1/4} · x^{-1/4} = 1.
    #[must_use]
    pub fn nf2(mass: f64, beta: f64) -> Self {
        let det_power = 0.25; // Nf/8 = 2/8
        let action_force = RationalApproximation::generate(-det_power, 8, 0.01, 64.0);
        Self {
            sectors: vec![RhmcFermionConfig {
                mass,
                det_power,
                action_approx: action_force.clone(),
                heatbath_approx: RationalApproximation::generate(det_power / 2.0, 8, 0.01, 64.0),
                force_approx: action_force,
            }],
            beta,
            dt: 0.01,
            n_md_steps: 50,
            cg_tol: 1e-8,
            cg_max_iter: 5000,
        }
    }

    /// Nf=2+1 configuration: light (u,d) det^{1/4} + strange det^{1/8}.
    ///
    /// Light sector: α = 1/4 (two degenerate flavors from 4-taste staggered)
    /// Strange sector: α = 1/8 (one flavor from 4-taste staggered)
    #[must_use]
    pub fn nf2p1(light_mass: f64, strange_mass: f64, beta: f64) -> Self {
        let light_power = 0.25;  // 2/8
        let strange_power = 0.125; // 1/8
        let light_af = RationalApproximation::generate(-light_power, 8, 0.01, 64.0);
        let strange_af = RationalApproximation::generate(-strange_power, 8, 0.01, 64.0);
        Self {
            sectors: vec![
                RhmcFermionConfig {
                    mass: light_mass,
                    det_power: light_power,
                    action_approx: light_af.clone(),
                    heatbath_approx: RationalApproximation::generate(
                        light_power / 2.0,
                        8,
                        0.01,
                        64.0,
                    ),
                    force_approx: light_af,
                },
                RhmcFermionConfig {
                    mass: strange_mass,
                    det_power: strange_power,
                    action_approx: strange_af.clone(),
                    heatbath_approx: RationalApproximation::generate(
                        strange_power / 2.0,
                        8,
                        0.01,
                        64.0,
                    ),
                    force_approx: strange_af,
                },
            ],
            beta,
            dt: 0.01,
            n_md_steps: 50,
            cg_tol: 1e-8,
            cg_max_iter: 5000,
        }
    }
}

/// RHMC pseudofermion heatbath: generate φ = (D†D)^{-p/2} η where η ~ N(0,1).
///
/// Uses the rational approximation for x^{-p/2} and multi-shift CG.
pub fn rhmc_heatbath(
    lattice: &Lattice,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
    seed: &mut u64,
) -> (FermionField, usize) {
    let vol = lattice.volume();
    let approx = &config.heatbath_approx;

    let mut eta = FermionField::zeros(vol);
    for site in &mut eta.data {
        for c in site.iter_mut() {
            let re = super::constants::lcg_gaussian(seed);
            let im = super::constants::lcg_gaussian(seed);
            *c = Complex64::new(re, im);
        }
    }

    // φ = r(-p/2)(D†D) η = α₀η + Σ αᵢ (D†D + σᵢ)⁻¹ η
    let (solutions, result) = multi_shift_cg_solve(
        lattice,
        &eta,
        config.mass,
        &approx.sigma,
        cg_tol,
        cg_max_iter,
    );

    let mut phi = FermionField::zeros(vol);
    for i in 0..vol {
        for c in 0..3 {
            let mut val = eta.data[i][c].scale(approx.alpha_0);
            for (s, x_s) in solutions.iter().enumerate() {
                val += x_s.data[i][c].scale(approx.alpha[s]);
            }
            phi.data[i][c] = val;
        }
    }

    (phi, result.iterations)
}

/// RHMC fermion action: `S_f` = φ† r(p)(D†D) φ.
///
/// Uses the rational approximation for x^p and multi-shift CG.
#[must_use]
pub fn rhmc_fermion_action(
    lattice: &Lattice,
    phi: &FermionField,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
) -> (f64, usize) {
    let approx = &config.action_approx;

    let (solutions, result) = multi_shift_cg_solve(
        lattice,
        phi,
        config.mass,
        &approx.sigma,
        cg_tol,
        cg_max_iter,
    );

    let mut action = approx.alpha_0 * phi.dot(phi).re;
    for (s, x_s) in solutions.iter().enumerate() {
        action += approx.alpha[s] * phi.dot(x_s).re;
    }

    (action, result.iterations)
}

/// RHMC fermion force: `dS_f/dU` = Σ αᵢ · d/dU [φ† (D†D + σᵢ)⁻¹ φ].
///
/// Each term in the sum is a standard pseudofermion force evaluated at the
/// shifted CG solution. Returns the total force summed over all poles.
#[must_use]
pub fn rhmc_fermion_force(
    lattice: &Lattice,
    phi: &FermionField,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
) -> (Vec<super::su3::Su3Matrix>, usize) {
    let vol = lattice.volume();
    let approx = &config.force_approx;

    let (solutions, result) = multi_shift_cg_solve(
        lattice,
        phi,
        config.mass,
        &approx.sigma,
        cg_tol,
        cg_max_iter,
    );

    let mut total_force = vec![super::su3::Su3Matrix::ZERO; vol * 4];

    for (s, x_s) in solutions.iter().enumerate() {
        let f_s = super::pseudofermion::pseudofermion_force(lattice, x_s, config.mass);
        for (tf, fs) in total_force.iter_mut().zip(f_s.iter()) {
            *tf = *tf + fs.scale(approx.alpha[s]);
        }
    }

    (total_force, result.iterations)
}

/// Remez exchange algorithm for fixed poles: finds residues that minimize max relative error.
///
/// Returns (coefficients, `max_relative_error`).
fn remez_for_poles(sigma: &[f64], power: f64, eval_grid: &[f64]) -> (Vec<f64>, f64) {
    let n_poles = sigma.len();
    let ncols = n_poles + 1;
    let n_sys = ncols + 1; // extra variable for equioscillation error E

    // Initial reference set: Chebyshev-like nodes from the eval grid
    let n_eval = eval_grid.len();
    let mut ref_indices: Vec<usize> = (0..n_sys)
        .map(|k| {
            let theta = std::f64::consts::PI * k as f64 / (n_sys - 1) as f64;
            let t = 0.5 * (1.0 - theta.cos());
            ((t * (n_eval - 1) as f64) as usize).min(n_eval - 1)
        })
        .collect();
    // Deduplicate
    ref_indices.sort_unstable();
    ref_indices.dedup();
    while ref_indices.len() < n_sys {
        for idx in 0..n_eval {
            if !ref_indices.contains(&idx) {
                ref_indices.push(idx);
                ref_indices.sort_unstable();
                break;
            }
        }
    }

    let mut best_coeffs = vec![0.0_f64; ncols];
    let mut best_err = f64::MAX;

    for _ in 0..60 {
        // Build Remez system: for reference point x_k,
        //   r(x_k)/target(x_k) - 1 = (-1)^k * E
        // => alpha_0/t_k + sum alpha_i/((x_k+sigma_i)*t_k) + (-1)^{k+1} E = 1
        let mut mat = vec![0.0_f64; n_sys * n_sys];
        let rhs = vec![1.0_f64; n_sys];
        for k in 0..n_sys {
            let x = eval_grid[ref_indices[k]];
            let t = x.powf(power);
            mat[k * n_sys] = 1.0 / t;
            for i in 0..n_poles {
                mat[k * n_sys + i + 1] = 1.0 / ((x + sigma[i]) * t);
            }
            mat[k * n_sys + ncols] = if k % 2 == 0 { -1.0 } else { 1.0 };
        }

        let solution = solve_linear_system(&mat, &rhs, n_sys);
        let coeffs: Vec<f64> = solution[..ncols].to_vec();

        // Evaluate signed relative error on the full grid
        let mut max_abs = 0.0_f64;
        let mut signed_err = Vec::with_capacity(n_eval);
        for (idx, &x) in eval_grid.iter().enumerate() {
            let exact = x.powf(power);
            let mut val = coeffs[0];
            for (a, s) in coeffs[1..].iter().zip(sigma.iter()) {
                val += a / (x + s);
            }
            let se = (val - exact) / exact;
            signed_err.push((idx, se));
            if se.abs() > max_abs {
                max_abs = se.abs();
            }
        }

        if max_abs < best_err {
            best_err = max_abs;
            best_coeffs = coeffs;
        }
        if max_abs < 1e-12 {
            break;
        }

        // Exchange: find local extrema of signed error with alternating signs
        let mut extrema: Vec<(usize, f64)> = Vec::new();

        // Always consider endpoints
        extrema.push(signed_err[0]);
        for i in 1..n_eval - 1 {
            let (_, ep) = signed_err[i - 1];
            let (idx, e) = signed_err[i];
            let (_, en) = signed_err[i + 1];
            if (e > ep && e > en) || (e < ep && e < en) {
                extrema.push((idx, e));
            }
        }
        extrema.push(signed_err[n_eval - 1]);

        if extrema.len() < n_sys {
            break;
        }

        // Select n_sys extrema with alternating signs, maximizing minimum |error|.
        // Greedy: walk through sorted-by-x extrema, pick alternating signs.
        let mut selected: Vec<(usize, f64)> = Vec::new();
        for &(idx, e) in &extrema {
            if selected.is_empty() {
                selected.push((idx, e));
            } else {
                let Some(last) = selected.last() else {
                    continue;
                };
                let last_sign = last.1.signum();
                if (e.signum() - last_sign).abs() > f64::EPSILON {
                    selected.push((idx, e));
                } else if selected.last().is_some_and(|l| e.abs() > l.1.abs()) {
                    if let Some(slot) = selected.last_mut() {
                        *slot = (idx, e);
                    }
                }
            }
        }

        // Trim to n_sys points, keeping the ones with largest |error|
        while selected.len() > n_sys {
            // Remove the extremum with smallest |error|
            let Some((min_idx, _)) = selected
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.1.abs().total_cmp(&b.1.abs()))
            else {
                break;
            };
            selected.remove(min_idx);
        }

        if selected.len() < n_sys {
            break;
        }

        let new_refs: Vec<usize> = selected.iter().map(|&(idx, _)| idx).collect();
        if new_refs == ref_indices {
            break; // converged
        }
        ref_indices = new_refs;
    }

    (best_coeffs, best_err)
}

/// Solve A·x = b via Gaussian elimination with partial pivoting.
fn solve_linear_system(ata: &[f64], atb: &[f64], n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; n * n];
    a.copy_from_slice(&ata[..n * n]);
    let mut b = atb[..n].to_vec();

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..n {
                a.swap(col * n + j, max_row * n + j);
            }
            b.swap(col, max_row);
        }
        let pivot = a[col * n + col];
        if pivot.abs() < 1e-30 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for j in col..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        let diag = a[i * n + i];
        x[i] = if diag.abs() > 1e-30 { sum / diag } else { 0.0 };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_approx_evaluates_correctly() {
        let r = RationalApproximation::fourth_root_8pole();
        let x = 1.0;
        let approx_val = r.eval(x);
        let exact_val = 1.0_f64; // x^{1/4} at x=1 = 1
        assert!(
            (approx_val - exact_val).abs() < 0.01,
            "fourth_root(1.0) = {approx_val}, expected ~1.0"
        );

        let x = 16.0;
        let approx_val = r.eval(x);
        let exact_val = 2.0; // 16^{1/4} = 2
        assert!(
            (approx_val - exact_val).abs() < 0.01,
            "fourth_root(16.0) = {approx_val}, expected ~2.0"
        );
    }

    #[test]
    fn inv_fourth_root_evaluates_correctly() {
        let r = RationalApproximation::inv_fourth_root_8pole();
        let x = 1.0;
        let approx_val = r.eval(x);
        let exact_val = 1.0;
        assert!(
            (approx_val - exact_val).abs() < 0.01,
            "inv_fourth_root(1.0) = {approx_val}, expected ~1.0"
        );
    }

    #[test]
    fn sqrt_evaluates_correctly() {
        let r = RationalApproximation::sqrt_8pole();
        let x = 4.0;
        let approx_val = r.eval(x);
        let exact_val = 2.0;
        let rel_err = (approx_val - exact_val).abs() / exact_val;
        assert!(
            rel_err < 0.02,
            "sqrt(4.0) = {approx_val}, expected ~2.0, rel_err={rel_err:.4}"
        );
        assert!(
            r.max_relative_error < 0.05,
            "max relative error {} too large for 8-pole sqrt",
            r.max_relative_error
        );
    }
}
