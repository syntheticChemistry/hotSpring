// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-species kinetic-fluid coupling for HED simulations.
//!
//! Implements the conservative multi-species BGK kinetic equation,
//! 1D Euler fluid solver, and kinetic-fluid coupling interface.
//! Validates conservation of mass, momentum, and energy across
//! the coupling boundary.
//!
//! # References
//!
//! - Haack, Murillo, Sagert & Chuna, J. Comput. Phys. (2024), DOI:10.1016/j.jcp.2024.112908
//! - Haack, Hauck, Murillo, J. Stat. Phys. 168:826 (2017)

// Physics code uses standard single-letter variable names (n, u, T, m, v, p, E)
// and domain-specific naming (rho_init/rho_int, u_init/u_int) that clippy
// flags but are idiomatic in computational physics.
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::float_cmp)]

use std::f64::consts::PI;

/// Maximum sub-iterations for kinetic-fluid interface coupling (Haack et al. §3.2).
pub(crate) const INTERFACE_MAX_SUB_ITERATIONS: usize = 3;

/// Density mismatch tolerance for interface sub-iteration convergence.
pub(crate) const INTERFACE_CONVERGENCE_TOL: f64 = 0.01;

/// Upper fraction of the spatial domain searched for the contact discontinuity.
/// The contact is expected between nx/2 and nx*CONTACT_SEARCH_UPPER_FRAC.
const CONTACT_SEARCH_UPPER_FRAC: usize = 85;

/// Denominator for contact search fraction (85/100 = 85%).
const CONTACT_SEARCH_DENOM: usize = 100;

/// Adiabatic index for monatomic ideal gas.
const GAMMA: f64 = 5.0 / 3.0;

// ──────────────────────────────────────────────────────────
//  1D Maxwellian and moment computation
// ──────────────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────────────
//  Phase 1: Multi-species BGK relaxation
// ──────────────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────────────
//  Phase 2: 1D Euler (Sod shock tube)
// ──────────────────────────────────────────────────────────

/// Convert primitives (ρ, u, p) to conserved (ρ, ρu, E).
#[must_use]
fn euler_prim_to_cons(rho: f64, u: f64, p: f64) -> (f64, f64, f64) {
    let e = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
    (rho, rho * u, e)
}

/// Convert conserved to primitives.
#[must_use]
fn euler_cons_to_prim(rho: f64, rho_u: f64, e: f64) -> (f64, f64, f64) {
    let u = if rho.abs() > 1e-30 { rho_u / rho } else { 0.0 };
    let p = ((GAMMA - 1.0) * (e - 0.5 * rho * u * u)).max(1e-30);
    (rho, u, p)
}

/// Euler flux vector.
#[must_use]
fn euler_flux(rho: f64, u: f64, p: f64) -> (f64, f64, f64) {
    let e = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
    (rho * u, rho * u * u + p, (e + p) * u)
}

/// HLL approximate Riemann solver.
#[must_use]
fn hll_flux(rho_l: f64, u_l: f64, p_l: f64, rho_r: f64, u_r: f64, p_r: f64) -> (f64, f64, f64) {
    let c_l = if rho_l > 1e-30 {
        (GAMMA * p_l / rho_l).sqrt()
    } else {
        0.0
    };
    let c_r = if rho_r > 1e-30 {
        (GAMMA * p_r / rho_r).sqrt()
    } else {
        0.0
    };

    let s_l = (u_l - c_l).min(u_r - c_r);
    let s_r = (u_l + c_l).max(u_r + c_r);

    let f_l = euler_flux(rho_l, u_l, p_l);
    let f_r = euler_flux(rho_r, u_r, p_r);

    if s_l >= 0.0 {
        return f_l;
    }
    if s_r <= 0.0 {
        return f_r;
    }

    let u_l_cons = euler_prim_to_cons(rho_l, u_l, p_l);
    let u_r_cons = euler_prim_to_cons(rho_r, u_r, p_r);
    let denom = s_r - s_l;

    let hll = |fl: f64, fr: f64, ul: f64, ur: f64| -> f64 {
        (s_r * fl - s_l * fr + s_l * s_r * (ur - ul)) / denom
    };

    (
        hll(f_l.0, f_r.0, u_l_cons.0, u_r_cons.0),
        hll(f_l.1, f_r.1, u_l_cons.1, u_r_cons.1),
        hll(f_l.2, f_r.2, u_l_cons.2, u_r_cons.2),
    )
}

/// Result of Sod shock tube test.
pub struct SodResult {
    /// Relative mass conservation error.
    pub mass_err: f64,
    /// Relative energy conservation error.
    pub energy_err: f64,
    /// Contact discontinuity position.
    pub x_contact: f64,
    /// Whether contact is in expected range.
    pub contact_in_range: bool,
    /// Whether shock was detected.
    pub shock_detected: bool,
    /// Shock position.
    pub x_shock: f64,
    /// Min density.
    pub rho_min: f64,
    /// Max density.
    pub rho_max: f64,
}

/// Run the Sod shock tube problem.
///
/// # Panics
///
/// Panics if `nx` is zero (division by zero in grid spacing).
#[must_use]
#[allow(clippy::while_float)]
pub fn run_sod_shock_tube(nx: usize, t_final: f64) -> SodResult {
    let dx = 1.0 / nx as f64;
    let x: Vec<f64> = (0..nx).map(|i| (i as f64 + 0.5) * dx).collect();

    let mut rho: Vec<f64> = x
        .iter()
        .map(|&xi| if xi < 0.5 { 1.0 } else { 0.125 })
        .collect();
    let mut u = vec![0.0; nx];
    let mut p: Vec<f64> = x
        .iter()
        .map(|&xi| if xi < 0.5 { 1.0 } else { 0.1 })
        .collect();

    let total_mass_0: f64 = rho.iter().sum::<f64>() * dx;
    let (_, _, e0_vec): (Vec<f64>, Vec<f64>, Vec<f64>) = rho
        .iter()
        .zip(u.iter())
        .zip(p.iter())
        .map(|((&r, &ui), &pi)| euler_prim_to_cons(r, ui, pi))
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut a, mut b, mut c), (x, y, z)| {
                a.push(x);
                b.push(y);
                c.push(z);
                (a, b, c)
            },
        );
    let total_e_0: f64 = e0_vec.iter().sum::<f64>() * dx;

    let mut t = 0.0;
    while t < t_final {
        let max_speed = rho
            .iter()
            .zip(u.iter())
            .zip(p.iter())
            .map(|((&r, &ui), &pi)| {
                let c = if r > 1e-30 {
                    (GAMMA * pi / r).sqrt()
                } else {
                    0.0
                };
                ui.abs() + c
            })
            .fold(0.0_f64, f64::max);

        let dt = (0.4 * dx / max_speed.max(1e-30)).min(t_final - t);

        let mut rho_c: Vec<f64> = Vec::with_capacity(nx);
        let mut rhou_c: Vec<f64> = Vec::with_capacity(nx);
        let mut e_c: Vec<f64> = Vec::with_capacity(nx);
        for i in 0..nx {
            let (r, ru, e) = euler_prim_to_cons(rho[i], u[i], p[i]);
            rho_c.push(r);
            rhou_c.push(ru);
            e_c.push(e);
        }

        let mut flux_rho = vec![0.0; nx + 1];
        let mut flux_mom = vec![0.0; nx + 1];
        let mut flux_ene = vec![0.0; nx + 1];

        let f0 = euler_flux(rho[0], u[0], p[0]);
        flux_rho[0] = f0.0;
        flux_mom[0] = f0.1;
        flux_ene[0] = f0.2;

        for i in 1..nx {
            let fi = hll_flux(rho[i - 1], u[i - 1], p[i - 1], rho[i], u[i], p[i]);
            flux_rho[i] = fi.0;
            flux_mom[i] = fi.1;
            flux_ene[i] = fi.2;
        }

        let fn_last = euler_flux(rho[nx - 1], u[nx - 1], p[nx - 1]);
        flux_rho[nx] = fn_last.0;
        flux_mom[nx] = fn_last.1;
        flux_ene[nx] = fn_last.2;

        for i in 0..nx {
            rho_c[i] -= dt / dx * (flux_rho[i + 1] - flux_rho[i]);
            rhou_c[i] -= dt / dx * (flux_mom[i + 1] - flux_mom[i]);
            e_c[i] -= dt / dx * (flux_ene[i + 1] - flux_ene[i]);
            rho_c[i] = rho_c[i].max(1e-10);
        }

        for i in 0..nx {
            let (r, ui, pi) = euler_cons_to_prim(rho_c[i], rhou_c[i], e_c[i]);
            rho[i] = r;
            u[i] = ui;
            p[i] = pi;
        }

        t += dt;
    }

    let total_mass_f: f64 = rho.iter().sum::<f64>() * dx;
    let total_e_f: f64 = rho
        .iter()
        .zip(u.iter())
        .zip(p.iter())
        .map(|((&r, &ui), &pi)| euler_prim_to_cons(r, ui, pi).2)
        .sum::<f64>()
        * dx;

    // Contact detection: smoothed gradient, largest jump in [0.5, 0.85]
    let drho: Vec<f64> = rho.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let drho_smooth: Vec<f64> = (0..drho.len())
        .map(|i| {
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(drho.len());
            drho[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
        })
        .collect();

    let contact_lo = nx / 2;
    let contact_hi = (nx * CONTACT_SEARCH_UPPER_FRAC / CONTACT_SEARCH_DENOM).min(drho_smooth.len());
    let i_contact = (contact_lo..contact_hi)
        .max_by(|&a, &b| {
            drho_smooth[a]
                .partial_cmp(&drho_smooth[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(nx / 2);
    let x_contact = x[i_contact];

    let shock_lo = nx * 7 / 10;
    let shock_hi = (nx - 1).min(drho_smooth.len());
    let i_shock = (shock_lo..shock_hi).max_by(|&a, &b| {
        drho_smooth[a]
            .partial_cmp(&drho_smooth[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (shock_detected, x_shock) = match i_shock {
        Some(i) if drho_smooth[i] > 1e-4 => (true, x[i]),
        _ => (false, 0.0),
    };

    SodResult {
        mass_err: (total_mass_f - total_mass_0).abs() / total_mass_0,
        energy_err: (total_e_f - total_e_0).abs() / total_e_0,
        x_contact,
        contact_in_range: (0.6..0.8).contains(&x_contact),
        shock_detected,
        x_shock,
        rho_min: rho.iter().copied().fold(f64::INFINITY, f64::min),
        rho_max: rho.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

// ──────────────────────────────────────────────────────────
//  Phase 3: Coupled kinetic-fluid
// ──────────────────────────────────────────────────────────

/// Moments from kinetic distribution: (ρ, ρu, E) with mass m.
#[must_use]
fn kinetic_to_fluid(f: &[f64], v: &[f64], dv: f64, m: f64) -> (f64, f64, f64) {
    let n: f64 = f.iter().sum::<f64>() * dv;
    let rho = m * n;
    let rho_u = m * f.iter().zip(v).map(|(&fi, &vi)| fi * vi).sum::<f64>() * dv;
    let e = 0.5 * m * f.iter().zip(v).map(|(&fi, &vi)| fi * vi * vi).sum::<f64>() * dv;
    (rho, rho_u, e)
}

/// Result of the coupled kinetic-fluid test.
pub struct CoupledResult {
    /// Relative mass conservation error.
    pub mass_err: f64,
    /// Relative momentum conservation error.
    pub momentum_err: f64,
    /// Relative energy conservation error.
    pub energy_err: f64,
    /// Interface density mismatch (relative).
    pub interface_density_match: f64,
    /// Number of time steps taken.
    pub n_steps: usize,
    /// Min fluid density.
    pub rho_fluid_min: f64,
    /// Max fluid density.
    pub rho_fluid_max: f64,
}

/// Run the coupled kinetic-fluid test matching the Python control.
///
/// # Panics
///
/// Panics if `nx_kin`, `nx_fluid`, or `nv` is less than 2.
#[must_use]
#[allow(clippy::while_float)]
pub fn run_coupled_kinetic_fluid(
    nx_kin: usize,
    nx_fluid: usize,
    nv: usize,
    t_final: f64,
) -> CoupledResult {
    let dx = 1.0 / (nx_kin + nx_fluid) as f64;
    let v_max = 6.0;
    let v: Vec<f64> = (0..nv)
        .map(|i| -v_max + i as f64 * 2.0 * v_max / (nv - 1) as f64)
        .collect();
    let dv = v[1] - v[0];
    let m = 1.0;
    let nu = 10.0;

    let rho_init = 1.0;
    let u_init = 0.1;
    let p_init = 1.0;
    let t_init = p_init / (rho_init / m);

    let f_init = maxwellian_1d(&v, rho_init / m, u_init, t_init, m);
    let mut f_kin: Vec<Vec<f64>> = (0..nx_kin).map(|_| f_init.clone()).collect();

    let mut rho_fluid = vec![rho_init; nx_fluid];
    let mut u_fluid = vec![u_init; nx_fluid];
    let mut p_fluid = vec![p_init; nx_fluid];

    let total_mass_0: f64 = {
        let kin: f64 = (0..nx_kin)
            .map(|i| kinetic_to_fluid(&f_kin[i], &v, dv, m).0)
            .sum();
        let fluid: f64 = rho_fluid.iter().sum();
        (kin + fluid) * dx
    };
    let total_mom_0: f64 = {
        let kin: f64 = (0..nx_kin)
            .map(|i| kinetic_to_fluid(&f_kin[i], &v, dv, m).1)
            .sum();
        let fluid: f64 = rho_fluid
            .iter()
            .zip(u_fluid.iter())
            .map(|(&r, &u)| r * u)
            .sum();
        (kin + fluid) * dx
    };
    let total_e_0: f64 = {
        let kin: f64 = (0..nx_kin)
            .map(|i| kinetic_to_fluid(&f_kin[i], &v, dv, m).2)
            .sum();
        let fluid: f64 = rho_fluid
            .iter()
            .zip(u_fluid.iter())
            .zip(p_fluid.iter())
            .map(|((&r, &u), &p)| euler_prim_to_cons(r, u, p).2)
            .sum();
        (kin + fluid) * dx
    };

    let mut t = 0.0;
    let mut n_steps = 0;
    let max_steps = 5000;

    while t < t_final && n_steps < max_steps {
        let max_speed_fluid = rho_fluid
            .iter()
            .zip(u_fluid.iter())
            .zip(p_fluid.iter())
            .map(|((&r, &u), &pi)| {
                let c = if r > 1e-30 {
                    (GAMMA * pi / r).sqrt()
                } else {
                    0.0
                };
                u.abs() + c
            })
            .fold(0.0_f64, f64::max);
        let max_speed = max_speed_fluid.max(v_max);
        let dt = (0.3 * dx / max_speed.max(1e-30)).min(t_final - t);

        // Kinetic advection (first-order upwind)
        let mut f_new = f_kin.clone();
        for (j, &vj) in v.iter().enumerate() {
            if vj > 0.0 {
                for i in 1..nx_kin {
                    f_new[i][j] = f_kin[i][j] - dt * vj / dx * (f_kin[i][j] - f_kin[i - 1][j]);
                }
            } else {
                for i in 0..nx_kin - 1 {
                    f_new[i][j] = f_kin[i][j] - dt * vj / dx * (f_kin[i + 1][j] - f_kin[i][j]);
                }
            }
        }
        for row in &mut f_new {
            for fi in row.iter_mut() {
                *fi = fi.max(0.0);
            }
        }
        f_kin = f_new;

        // BGK collision (index needed for in-place mutation of nested Vec)
        #[allow(clippy::needless_range_loop)]
        for i in 0..nx_kin {
            let (ni, ui, ti, _) = compute_moments(&f_kin[i], &v, dv, m);
            if ni < 1e-30 {
                continue;
            }
            let m_local = maxwellian_1d(&v, ni, ui, ti, m);
            for j in 0..nv {
                f_kin[i][j] = (f_kin[i][j] + dt * nu * (m_local[j] - f_kin[i][j])).max(0.0);
            }
        }

        // Interface sub-iteration: converges the kinetic-fluid boundary
        // by repeating the interface exchange until density mismatch drops
        // below tolerance or max iterations reached (Haack et al. §3.2).
        let max_sub = INTERFACE_MAX_SUB_ITERATIONS;
        let sub_tol = INTERFACE_CONVERGENCE_TOL;
        let f_kin_boundary_save = f_kin[nx_kin - 1].clone();
        let rho_fluid_save = rho_fluid.clone();
        let u_fluid_save = u_fluid.clone();
        let p_fluid_save = p_fluid.clone();

        for _sub in 0..max_sub {
            // Interface: kinetic → fluid
            let (rho_int, rhou_int, e_int) = kinetic_to_fluid(&f_kin[nx_kin - 1], &v, dv, m);
            let u_int = if rho_int > 1e-30 {
                rhou_int / rho_int
            } else {
                0.0
            };
            let p_int = ((GAMMA - 1.0) * (e_int - 0.5 * rho_int * u_int * u_int)).max(1e-15);

            // Euler update (from saved state each sub-iteration)
            let mut rho_c: Vec<f64> = Vec::with_capacity(nx_fluid);
            let mut rhou_c: Vec<f64> = Vec::with_capacity(nx_fluid);
            let mut e_c: Vec<f64> = Vec::with_capacity(nx_fluid);
            for i in 0..nx_fluid {
                let (r, ru, e) =
                    euler_prim_to_cons(rho_fluid_save[i], u_fluid_save[i], p_fluid_save[i]);
                rho_c.push(r);
                rhou_c.push(ru);
                e_c.push(e);
            }

            let mut flux_rho = vec![0.0; nx_fluid + 1];
            let mut flux_mom = vec![0.0; nx_fluid + 1];
            let mut flux_ene = vec![0.0; nx_fluid + 1];

            let f_left = hll_flux(
                rho_int,
                u_int,
                p_int,
                rho_fluid_save[0],
                u_fluid_save[0],
                p_fluid_save[0],
            );
            flux_rho[0] = f_left.0;
            flux_mom[0] = f_left.1;
            flux_ene[0] = f_left.2;

            for i in 1..nx_fluid {
                let fi = hll_flux(
                    rho_fluid_save[i - 1],
                    u_fluid_save[i - 1],
                    p_fluid_save[i - 1],
                    rho_fluid_save[i],
                    u_fluid_save[i],
                    p_fluid_save[i],
                );
                flux_rho[i] = fi.0;
                flux_mom[i] = fi.1;
                flux_ene[i] = fi.2;
            }

            let f_right = euler_flux(
                rho_fluid_save[nx_fluid - 1],
                u_fluid_save[nx_fluid - 1],
                p_fluid_save[nx_fluid - 1],
            );
            flux_rho[nx_fluid] = f_right.0;
            flux_mom[nx_fluid] = f_right.1;
            flux_ene[nx_fluid] = f_right.2;

            for i in 0..nx_fluid {
                rho_c[i] -= dt / dx * (flux_rho[i + 1] - flux_rho[i]);
                rhou_c[i] -= dt / dx * (flux_mom[i + 1] - flux_mom[i]);
                e_c[i] -= dt / dx * (flux_ene[i + 1] - flux_ene[i]);
                rho_c[i] = rho_c[i].max(1e-10);
            }

            for i in 0..nx_fluid {
                let (r, ui, pi) = euler_cons_to_prim(rho_c[i], rhou_c[i], e_c[i]);
                rho_fluid[i] = r;
                u_fluid[i] = ui;
                p_fluid[i] = pi;
            }

            // Interface: fluid → kinetic (incoming distribution)
            let m_boundary = maxwellian_1d(
                &v,
                rho_fluid[0] / m,
                u_fluid[0],
                p_fluid[0] / (rho_fluid[0] / m),
                m,
            );
            f_kin[nx_kin - 1].clone_from(&f_kin_boundary_save);
            for j in 0..nv {
                if v[j] <= 0.0 {
                    f_kin[nx_kin - 1][j] = m_boundary[j];
                }
            }

            let rho_kin_if = kinetic_to_fluid(&f_kin[nx_kin - 1], &v, dv, m).0;
            let mismatch = (rho_kin_if - rho_fluid[0]).abs() / rho_init.max(1e-30);
            if mismatch < sub_tol {
                break;
            }
        }

        t += dt;
        n_steps += 1;
    }

    let total_mass_f: f64 = {
        let kin: f64 = (0..nx_kin)
            .map(|i| kinetic_to_fluid(&f_kin[i], &v, dv, m).0)
            .sum();
        let fluid: f64 = rho_fluid.iter().sum();
        (kin + fluid) * dx
    };
    let total_mom_f: f64 = {
        let kin: f64 = (0..nx_kin)
            .map(|i| kinetic_to_fluid(&f_kin[i], &v, dv, m).1)
            .sum();
        let fluid: f64 = rho_fluid
            .iter()
            .zip(u_fluid.iter())
            .map(|(&r, &u)| r * u)
            .sum();
        (kin + fluid) * dx
    };
    let total_e_f: f64 = {
        let kin: f64 = (0..nx_kin)
            .map(|i| kinetic_to_fluid(&f_kin[i], &v, dv, m).2)
            .sum();
        let fluid: f64 = rho_fluid
            .iter()
            .zip(u_fluid.iter())
            .zip(p_fluid.iter())
            .map(|((&r, &u), &p)| euler_prim_to_cons(r, u, p).2)
            .sum();
        (kin + fluid) * dx
    };

    let rho_if_kin = kinetic_to_fluid(&f_kin[nx_kin - 1], &v, dv, m).0;
    let interface_density_match = (rho_if_kin - rho_fluid[0]).abs() / rho_init.max(1e-30);

    CoupledResult {
        mass_err: (total_mass_f - total_mass_0).abs() / total_mass_0.max(1e-30),
        momentum_err: (total_mom_f - total_mom_0).abs() / total_mom_0.abs().max(1e-30),
        energy_err: (total_e_f - total_e_0).abs() / total_e_0.max(1e-30),
        interface_density_match,
        n_steps,
        rho_fluid_min: rho_fluid.iter().copied().fold(f64::INFINITY, f64::min),
        rho_fluid_max: rho_fluid.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

// ──────────────────────────────────────────────────────────
//  Combined validation
// ──────────────────────────────────────────────────────────

/// Full validation result for Paper 45.
pub struct KineticFluidValidation {
    /// BGK relaxation results.
    pub bgk: BgkRelaxationResult,
    /// Sod shock tube results.
    pub sod: SodResult,
    /// Coupled kinetic-fluid results.
    pub coupled: CoupledResult,
}

/// Run all three phases of the kinetic-fluid validation.
#[must_use]
pub fn validate_kinetic_fluid() -> KineticFluidValidation {
    let bgk = run_bgk_relaxation(3000, 0.005);
    let sod = run_sod_shock_tube(400, 0.2);
    let coupled = run_coupled_kinetic_fluid(30, 30, 81, 0.05);
    KineticFluidValidation { bgk, sod, coupled }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxwellian_normalizes() {
        let nv = 1001;
        let v_max = 10.0;
        let v: Vec<f64> = (0..nv)
            .map(|i| -v_max + i as f64 * 2.0 * v_max / (nv - 1) as f64)
            .collect();
        let dv = v[1] - v[0];
        let f = maxwellian_1d(&v, 1.0, 0.0, 1.0, 1.0);
        let integral: f64 = f.iter().sum::<f64>() * dv;
        assert!((integral - 1.0).abs() < 0.01, "integral = {integral}");
    }

    #[test]
    fn maxwellian_mean_velocity() {
        let nv = 1001;
        let v_max = 10.0;
        let v: Vec<f64> = (0..nv)
            .map(|i| -v_max + i as f64 * 2.0 * v_max / (nv - 1) as f64)
            .collect();
        let dv = v[1] - v[0];
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
        let nv = 501;
        let v_max = 10.0;
        let v: Vec<f64> = (0..nv)
            .map(|i| -v_max + i as f64 * 2.0 * v_max / (nv - 1) as f64)
            .collect();
        let dv = v[1] - v[0];
        let (n0, u0, t0, m0) = (2.0, 0.3, 1.5, 1.0);
        let f = maxwellian_1d(&v, n0, u0, t0, m0);
        let (n, u, t, _) = compute_moments(&f, &v, dv, m0);
        assert!((n - n0).abs() < 0.05, "n={n}");
        assert!((u - u0).abs() < 0.05, "u={u}");
        assert!((t - t0).abs() < 0.1, "T={t}");
    }

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

    #[test]
    fn sod_conservation() {
        let r = run_sod_shock_tube(200, 0.2);
        assert!(r.mass_err < 1e-10, "mass err = {:.2e}", r.mass_err);
        assert!(r.energy_err < 1e-10, "energy err = {:.2e}", r.energy_err);
    }

    #[test]
    fn sod_shock_structure() {
        let r = run_sod_shock_tube(400, 0.2);
        assert!(r.contact_in_range, "contact at {}", r.x_contact);
        assert!(r.shock_detected, "no shock detected");
    }

    #[test]
    fn coupled_mass_conservation() {
        let r = run_coupled_kinetic_fluid(30, 30, 81, 0.05);
        assert!(r.mass_err < 0.15, "mass err = {:.4e}", r.mass_err);
    }

    #[test]
    fn coupled_energy_conservation() {
        let r = run_coupled_kinetic_fluid(30, 30, 81, 0.05);
        assert!(r.energy_err < 0.15, "energy err = {:.4e}", r.energy_err);
    }

    #[test]
    fn coupled_interface_continuity() {
        let r = run_coupled_kinetic_fluid(30, 30, 81, 0.05);
        assert!(
            r.interface_density_match < 0.5,
            "interface match = {:.4e}",
            r.interface_density_match
        );
    }

    #[test]
    fn coupled_physical_density() {
        let r = run_coupled_kinetic_fluid(30, 30, 81, 0.05);
        assert!(r.rho_fluid_min > 0.5, "rho_min = {}", r.rho_fluid_min);
        assert!(r.rho_fluid_max < 2.0, "rho_max = {}", r.rho_fluid_max);
    }

    #[test]
    fn euler_prim_cons_roundtrip() {
        let (rho, u, p) = (1.5, 0.3, 2.0);
        let (r, ru, e) = euler_prim_to_cons(rho, u, p);
        let (r2, u2, p2) = euler_cons_to_prim(r, ru, e);
        assert!((r2 - rho).abs() < 1e-14);
        assert!((u2 - u).abs() < 1e-14);
        assert!((p2 - p).abs() < 1e-14);
    }

    #[test]
    fn hll_symmetry() {
        let (rho, u, p) = (1.0, 0.0, 1.0);
        let f = hll_flux(rho, u, p, rho, u, p);
        let ef = euler_flux(rho, u, p);
        assert!((f.0 - ef.0).abs() < 1e-14);
        assert!((f.1 - ef.1).abs() < 1e-14);
        assert!((f.2 - ef.2).abs() < 1e-14);
    }
}
