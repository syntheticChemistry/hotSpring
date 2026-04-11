// SPDX-License-Identifier: AGPL-3.0-or-later

//! 1D compressible Euler equations with HLL Riemann solver.
//!
//! Validates the fluid half of the kinetic-fluid coupling against
//! the classical Sod shock tube problem.

use super::GAMMA;

/// Upper fraction of the spatial domain searched for the contact discontinuity.
const CONTACT_SEARCH_UPPER_FRAC: usize = 85;

/// Denominator for contact search fraction (85/100 = 85%).
const CONTACT_SEARCH_DENOM: usize = 100;

/// Convert primitives (ρ, u, p) to conserved (ρ, ρu, E).
#[must_use]
pub(crate) fn prim_to_cons(rho: f64, u: f64, p: f64) -> (f64, f64, f64) {
    let e = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
    (rho, rho * u, e)
}

/// Convert conserved to primitives.
#[must_use]
pub(crate) fn cons_to_prim(rho: f64, rho_u: f64, e: f64) -> (f64, f64, f64) {
    let u = if rho.abs() > 1e-30 { rho_u / rho } else { 0.0 };
    let p = ((GAMMA - 1.0) * (e - 0.5 * rho * u * u)).max(1e-30);
    (rho, u, p)
}

/// Euler flux vector.
#[must_use]
pub(crate) fn flux(rho: f64, u: f64, p: f64) -> (f64, f64, f64) {
    let e = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
    (rho * u, rho * u * u + p, (e + p) * u)
}

/// HLL approximate Riemann solver.
#[must_use]
pub(crate) fn hll_flux(
    rho_l: f64,
    u_l: f64,
    p_l: f64,
    rho_r: f64,
    u_r: f64,
    p_r: f64,
) -> (f64, f64, f64) {
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

    let f_l = flux(rho_l, u_l, p_l);
    let f_r = flux(rho_r, u_r, p_r);

    if s_l >= 0.0 {
        return f_l;
    }
    if s_r <= 0.0 {
        return f_r;
    }

    let u_l_cons = prim_to_cons(rho_l, u_l, p_l);
    let u_r_cons = prim_to_cons(rho_r, u_r, p_r);
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
        .map(|((&r, &ui), &pi)| prim_to_cons(r, ui, pi))
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
    let max_steps = (t_final / 1e-6).ceil() as usize + 1;
    for _ in 0..max_steps {
        if t >= t_final {
            break;
        }
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
            let (r, ru, e) = prim_to_cons(rho[i], u[i], p[i]);
            rho_c.push(r);
            rhou_c.push(ru);
            e_c.push(e);
        }

        let mut flux_rho = vec![0.0; nx + 1];
        let mut flux_mom = vec![0.0; nx + 1];
        let mut flux_ene = vec![0.0; nx + 1];

        let f0 = flux(rho[0], u[0], p[0]);
        flux_rho[0] = f0.0;
        flux_mom[0] = f0.1;
        flux_ene[0] = f0.2;

        for i in 1..nx {
            let fi = hll_flux(rho[i - 1], u[i - 1], p[i - 1], rho[i], u[i], p[i]);
            flux_rho[i] = fi.0;
            flux_mom[i] = fi.1;
            flux_ene[i] = fi.2;
        }

        let fn_last = flux(rho[nx - 1], u[nx - 1], p[nx - 1]);
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
            let (r, ui, pi) = cons_to_prim(rho_c[i], rhou_c[i], e_c[i]);
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
        .map(|((&r, &ui), &pi)| prim_to_cons(r, ui, pi).2)
        .sum::<f64>()
        * dx;

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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn prim_cons_roundtrip() {
        let (rho, u, p) = (1.5, 0.3, 2.0);
        let (r, ru, e) = prim_to_cons(rho, u, p);
        let (r2, u2, p2) = cons_to_prim(r, ru, e);
        assert!((r2 - rho).abs() < 1e-14);
        assert!((u2 - u).abs() < 1e-14);
        assert!((p2 - p).abs() < 1e-14);
    }

    #[test]
    fn hll_symmetry() {
        let (rho, u, p) = (1.0, 0.0, 1.0);
        let f = hll_flux(rho, u, p, rho, u, p);
        let ef = flux(rho, u, p);
        assert!((f.0 - ef.0).abs() < 1e-14);
        assert!((f.1 - ef.1).abs() < 1e-14);
        assert!((f.2 - ef.2).abs() < 1e-14);
    }
}
