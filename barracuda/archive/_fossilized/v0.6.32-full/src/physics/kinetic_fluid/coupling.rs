// SPDX-License-Identifier: AGPL-3.0-or-later

//! Kinetic-fluid interface coupling with sub-iteration convergence.
//!
//! Implements the two-domain coupled solver from Haack, Murillo, Sagert & Chuna,
//! J. Comput. Phys. (2024): BGK kinetic domain ↔ 1D Euler fluid domain.

use super::euler::{cons_to_prim, flux, hll_flux, prim_to_cons};
use super::maxwellian::{compute_moments, maxwellian_1d};
use super::{GAMMA, INTERFACE_CONVERGENCE_TOL, INTERFACE_MAX_SUB_ITERATIONS};

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
    let mut f_buf: Vec<Vec<f64>> = f_kin.clone();

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
            .map(|((&r, &u), &p)| prim_to_cons(r, u, p).2)
            .sum();
        (kin + fluid) * dx
    };

    let mut t = 0.0;
    let mut n_steps = 0;
    let max_steps = 5000;
    let mut f_kin_boundary_save = vec![0.0; nv];
    let mut rho_fluid_save = vec![0.0; nx_fluid];
    let mut u_fluid_save = vec![0.0; nx_fluid];
    let mut p_fluid_save = vec![0.0; nx_fluid];

    for _ in 0..max_steps {
        if t >= t_final {
            break;
        }
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

        for (src, dst) in f_kin.iter().zip(f_buf.iter_mut()) {
            dst.copy_from_slice(src);
        }
        for (j, &vj) in v.iter().enumerate() {
            if vj > 0.0 {
                for i in 1..nx_kin {
                    f_buf[i][j] = f_kin[i][j] - dt * vj / dx * (f_kin[i][j] - f_kin[i - 1][j]);
                }
            } else {
                for i in 0..nx_kin - 1 {
                    f_buf[i][j] = f_kin[i][j] - dt * vj / dx * (f_kin[i + 1][j] - f_kin[i][j]);
                }
            }
        }
        for row in &mut f_buf {
            for fi in row.iter_mut() {
                *fi = fi.max(0.0);
            }
        }
        std::mem::swap(&mut f_kin, &mut f_buf);

        #[expect(
            clippy::needless_range_loop,
            reason = "index needed for f_kin[i] mutation"
        )]
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

        let max_sub = INTERFACE_MAX_SUB_ITERATIONS;
        let sub_tol = INTERFACE_CONVERGENCE_TOL;
        f_kin_boundary_save.copy_from_slice(&f_kin[nx_kin - 1]);
        rho_fluid_save.copy_from_slice(&rho_fluid);
        u_fluid_save.copy_from_slice(&u_fluid);
        p_fluid_save.copy_from_slice(&p_fluid);

        for _sub in 0..max_sub {
            let (rho_int, rhou_int, e_int) = kinetic_to_fluid(&f_kin[nx_kin - 1], &v, dv, m);
            let u_int = if rho_int > 1e-30 {
                rhou_int / rho_int
            } else {
                0.0
            };
            let p_int = ((GAMMA - 1.0) * (e_int - 0.5 * rho_int * u_int * u_int)).max(1e-15);

            let mut rho_c: Vec<f64> = Vec::with_capacity(nx_fluid);
            let mut rhou_c: Vec<f64> = Vec::with_capacity(nx_fluid);
            let mut e_c: Vec<f64> = Vec::with_capacity(nx_fluid);
            for i in 0..nx_fluid {
                let (r, ru, e) = prim_to_cons(rho_fluid_save[i], u_fluid_save[i], p_fluid_save[i]);
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

            let f_right = flux(
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
                let (r, ui, pi) = cons_to_prim(rho_c[i], rhou_c[i], e_c[i]);
                rho_fluid[i] = r;
                u_fluid[i] = ui;
                p_fluid[i] = pi;
            }

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
            .map(|((&r, &u), &p)| prim_to_cons(r, u, p).2)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
