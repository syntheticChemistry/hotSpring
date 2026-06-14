// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative multi-species BGK kinetic relaxation.
//!
//! Reference: Haack, Hauck, Murillo, J. Stat. Phys. 168:826 (2017).

use super::maxwellian::{compute_moments, maxwellian_1d};

/// Species descriptor for BGK relaxation.
pub struct BgkSpecies {
    /// Mass.
    pub m: f64,
    /// Collision frequency.
    pub nu: f64,
    /// Distribution function (velocity-space discretization).
    pub f: Vec<f64>,
}

/// Compute conservation-preserving BGK target Maxwellians.
///
/// Returns `(n*, u*, T*)` for each species such that the BGK operator
/// conserves species mass, total momentum, and total kinetic energy.
///
/// In 1D: `T* = 2 * E_thermal / n_total` where
/// `E_thermal = E_total - (total_mass * u_bar²) / 2`.
#[must_use]
pub fn bgk_target_params(species: &[BgkSpecies], v: &[f64], dv: f64) -> Vec<(f64, f64, f64)> {
    let mut total_mom = 0.0;
    let mut total_mass = 0.0;
    let mut total_energy = 0.0;
    let mut moments = Vec::with_capacity(species.len());

    for sp in species {
        let (n, u, _t, e) = compute_moments(&sp.f, v, dv, sp.m);
        total_mom += sp.m * n * u;
        total_mass += sp.m * n;
        total_energy += e;
        moments.push((n, u));
    }

    let u_bar = if total_mass > 1e-30 {
        total_mom / total_mass
    } else {
        0.0
    };

    let n_total: f64 = moments.iter().map(|(n, _)| n).sum();
    let thermal = total_energy - 0.5 * total_mass * u_bar * u_bar;
    let t_star = if n_total > 1e-30 {
        (2.0 * thermal / n_total).max(1e-12)
    } else {
        1e-12
    };

    moments
        .iter()
        .map(|&(n_s, _)| (n_s, u_bar, t_star))
        .collect()
}

/// One forward-Euler BGK relaxation step for all species.
pub fn bgk_relaxation_step(species: &mut [BgkSpecies], v: &[f64], dv: f64, dt: f64) {
    let targets = bgk_target_params(species, v, dv);

    for (sp, &(n_star, u_star, t_star)) in species.iter_mut().zip(&targets) {
        let m_star = maxwellian_1d(v, n_star, u_star, t_star, sp.m);
        for (fi, mi) in sp.f.iter_mut().zip(&m_star) {
            *fi = (*fi + dt * sp.nu * (mi - *fi)).max(0.0);
        }
    }
}

/// Boltzmann H-function: `H = ∫ f ln(f) dv`.
#[must_use]
pub fn entropy_h(f: &[f64], dv: f64) -> f64 {
    f.iter()
        .map(|&fi| {
            let fp = fi.max(1e-300);
            fp * fp.ln()
        })
        .sum::<f64>()
        * dv
}

/// Result of BGK relaxation test.
pub struct BgkRelaxationResult {
    /// Species 1 density conservation error.
    pub mass_err_1: f64,
    /// Species 2 density conservation error.
    pub mass_err_2: f64,
    /// Total momentum conservation error.
    pub momentum_err: f64,
    /// Relative total energy conservation error.
    pub energy_err: f64,
    /// Whether entropy was monotonically non-increasing.
    pub entropy_monotonic: bool,
    /// Final temperature of species 1.
    pub t1_final: f64,
    /// Final temperature of species 2.
    pub t2_final: f64,
    /// Relative temperature difference at end.
    pub temp_relaxed: f64,
}

/// Run two-species BGK relaxation matching the Python control.
#[must_use]
pub fn run_bgk_relaxation(n_steps: usize, dt: f64) -> BgkRelaxationResult {
    let nv: usize = 201;
    let v_max = 8.0;
    let nv_m1 = (nv - 1) as f64;
    let v: Vec<f64> = (0..nv)
        .map(|i| -v_max + (i as f64) * 2.0 * v_max / nv_m1)
        .collect();
    let dv = v[1] - v[0];

    let (m1, m2) = (1.0, 4.0);
    let (n1, n2) = (1.0, 1.0);
    let (u1, u2) = (0.0, 0.0);
    let (t1, t2) = (2.0, 0.5);
    let (nu1, nu2) = (1.0, 1.0);

    let f1_init = maxwellian_1d(&v, n1, u1, t1, m1);
    let f2_init = maxwellian_1d(&v, n2, u2, t2, m2);

    let (n1_0, u1_0, _, e1_0) = compute_moments(&f1_init, &v, dv, m1);
    let (n2_0, u2_0, _, e2_0) = compute_moments(&f2_init, &v, dv, m2);
    let e_total_0 = e1_0 + e2_0;
    let mom_total_0 = m1 * n1_0 * u1_0 + m2 * n2_0 * u2_0;

    let mut species = vec![
        BgkSpecies {
            m: m1,
            nu: nu1,
            f: f1_init,
        },
        BgkSpecies {
            m: m2,
            nu: nu2,
            f: f2_init,
        },
    ];

    let mut h_prev = entropy_h(&species[0].f, dv) + entropy_h(&species[1].f, dv);
    let mut entropy_decreasing = false;

    for _ in 0..n_steps {
        bgk_relaxation_step(&mut species, &v, dv, dt);
        let h_curr = entropy_h(&species[0].f, dv) + entropy_h(&species[1].f, dv);
        if h_curr > h_prev + 1e-10 {
            entropy_decreasing = true;
        }
        h_prev = h_curr;
    }

    let (n1_f, _, t1_f, e1_f) = compute_moments(&species[0].f, &v, dv, m1);
    let (n2_f, u2_f, t2_f, e2_f) = compute_moments(&species[1].f, &v, dv, m2);
    let e_total_f = e1_f + e2_f;
    let (_, u1_f, _, _) = compute_moments(&species[0].f, &v, dv, m1);
    let mom_total_f = m1 * n1_f * u1_f + m2 * n2_f * u2_f;

    let temp_relaxed = if t1_f.max(t2_f) > 0.0 {
        (t1_f - t2_f).abs() / t1_f.max(t2_f)
    } else {
        0.0
    };

    BgkRelaxationResult {
        mass_err_1: (n1_f - n1_0).abs(),
        mass_err_2: (n2_f - n2_0).abs(),
        momentum_err: (mom_total_f - mom_total_0).abs(),
        energy_err: (e_total_f - e_total_0).abs() / e_total_0.abs().max(1e-30),
        entropy_monotonic: !entropy_decreasing,
        t1_final: t1_f,
        t2_final: t2_f,
        temp_relaxed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgk_mass_conservation() {
        let r = run_bgk_relaxation(1000, 0.005);
        assert!(
            r.mass_err_1 < 1e-8,
            "species 1 mass err = {:.2e}",
            r.mass_err_1
        );
        assert!(
            r.mass_err_2 < 1e-8,
            "species 2 mass err = {:.2e}",
            r.mass_err_2
        );
    }

    #[test]
    fn bgk_momentum_conservation() {
        let r = run_bgk_relaxation(1000, 0.005);
        assert!(
            r.momentum_err < 1e-10,
            "momentum err = {:.2e}",
            r.momentum_err
        );
    }

    #[test]
    fn bgk_energy_conservation() {
        let r = run_bgk_relaxation(1000, 0.005);
        assert!(r.energy_err < 0.01, "energy err = {:.2e}", r.energy_err);
    }

    #[test]
    fn bgk_h_theorem() {
        let r = run_bgk_relaxation(1000, 0.005);
        assert!(r.entropy_monotonic, "entropy should be non-increasing");
    }

    #[test]
    fn bgk_temperature_relaxation() {
        let r = run_bgk_relaxation(3000, 0.005);
        assert!(r.temp_relaxed < 0.01, "|T1-T2|/max = {:.4}", r.temp_relaxed);
    }
}
