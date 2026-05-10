// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hasenbusch mass preconditioning for HMC.
//!
//! Two-level split of the fermion determinant (Hasenbusch 2001, PLB 519, 177):
//!
//!   det(D†D(m_l)) = det(D†D(m_h)) × det(D†D(m_l)/D†D(m_h))
//!
//! Heavy sector is cheap (few CG iterations); ratio sector has smaller
//! condition number than the full light-mass operator.

use crate::lattice::cg::cg_solve;
use crate::lattice::dirac::{FermionField, apply_dirac_sq};
use crate::lattice::hmc::exp_su3_cayley_pub;
use crate::lattice::su3::Su3Matrix;
use crate::lattice::wilson::Lattice;

use super::action::{force_bilinear_ab, pseudofermion_force, pseudofermion_heatbath};
use super::config::{HasenbuschConfig, HasenbuschHmcConfig};

/// Heatbath for the heavy sector: `φ_H` = `D†(m_heavy)` η.
pub fn hasenbusch_heavy_heatbath(
    lattice: &Lattice,
    heavy_mass: f64,
    seed: &mut u64,
) -> FermionField {
    pseudofermion_heatbath(lattice, heavy_mass, seed)
}

/// Heatbath for the ratio sector: φ = `D†(m_heavy)` η.
///
/// Standard approximation (exact would use matrix square root).
pub fn hasenbusch_ratio_heatbath(
    lattice: &Lattice,
    heavy_mass: f64,
    seed: &mut u64,
) -> FermionField {
    pseudofermion_heatbath(lattice, heavy_mass, seed)
}

/// Action for the heavy sector: `S_H` = `φ†(D†D(m_heavy))⁻¹φ`.
///
/// Returns (action, CG result, solution x).
pub fn hasenbusch_heavy_action(
    lattice: &Lattice,
    phi: &FermionField,
    config: &HasenbuschConfig,
) -> (f64, crate::lattice::cg::CgResult, FermionField) {
    let vol = lattice.volume();
    let mut x = FermionField::zeros(vol);

    let cg_result = cg_solve(
        lattice,
        &mut x,
        phi,
        config.heavy_mass,
        config.cg_tol,
        config.cg_max_iter,
    );

    let action = phi.dot(&x).re;
    (action, cg_result, x)
}

/// Action for the ratio sector: `S_ratio` = `φ†(D†D(m_light))⁻¹D†D(m_heavy)φ`.
///
/// Solve (`D†D(m_light)`) x = φ, then S = φ†`D†D(m_heavy)`x.
/// Returns (action, CG result, x for force computation).
pub fn hasenbusch_ratio_action(
    lattice: &Lattice,
    phi: &FermionField,
    config: &HasenbuschConfig,
) -> (f64, crate::lattice::cg::CgResult, FermionField) {
    let vol = lattice.volume();
    let mut x = FermionField::zeros(vol);

    let cg_result = cg_solve(
        lattice,
        &mut x,
        phi,
        config.light_mass,
        config.cg_tol,
        config.cg_max_iter,
    );

    let ddx = apply_dirac_sq(lattice, &x, config.heavy_mass);
    let action = phi.dot(&ddx).re;

    (action, cg_result, x)
}

/// Force for the ratio sector.
///
/// `S_ratio` = φ†A⁻¹Bφ with A = `D†D(m_light)`, B = `D†D(m_heavy)`.
/// dS/dU = −y†(dA/dU)x + φ†A⁻¹(dB/dU)φ, where x = A⁻¹φ, y = A⁻¹Bφ.
#[must_use]
pub fn hasenbusch_ratio_force(
    lattice: &Lattice,
    phi: &FermionField,
    x: &FermionField,
    config: &HasenbuschConfig,
) -> Vec<Su3Matrix> {
    let vol = lattice.volume();
    let rhs = apply_dirac_sq(lattice, phi, config.heavy_mass);
    let mut y = FermionField::zeros(vol);

    let _cg = cg_solve(
        lattice,
        &mut y,
        &rhs,
        config.light_mass,
        config.cg_tol,
        config.cg_max_iter,
    );

    let f1 = force_bilinear_ab(lattice, &y, x, config.light_mass);
    let f2 = force_bilinear_ab(lattice, x, phi, config.heavy_mass);

    let mut force = vec![Su3Matrix::ZERO; vol * 4];
    for (out, (a, b)) in force.iter_mut().zip(f1.iter().zip(f2.iter())) {
        *out = *a - *b;
    }
    force
}

/// Result of a Hasenbusch HMC trajectory.
#[derive(Clone, Debug)]
pub struct HasenbuschHmcResult {
    /// Metropolis accept/reject decision.
    pub accepted: bool,
    /// Hamiltonian change ΔH.
    pub delta_h: f64,
    /// Mean plaquette.
    pub plaquette: f64,
    /// Wilson gauge action.
    pub gauge_action: f64,
    /// Fermion action (heavy + ratio sectors).
    pub fermion_action: f64,
    /// CG iterations for heavy sector.
    pub cg_iterations_heavy: usize,
    /// CG iterations for ratio sector.
    pub cg_iterations_ratio: usize,
}

/// Run one Hasenbusch-preconditioned HMC trajectory.
///
/// Two pseudofermion sectors: heavy (cheap) and ratio (expensive).
/// Uses multiple time-scale leapfrog: more MD steps for light sector.
pub fn hasenbusch_hmc_trajectory(
    lattice: &mut Lattice,
    config: &mut HasenbuschHmcConfig,
) -> HasenbuschHmcResult {
    let vol = lattice.volume();
    let old_links = lattice.links.clone();

    let phi_heavy =
        hasenbusch_heavy_heatbath(lattice, config.hasenbusch.heavy_mass, &mut config.seed);
    let phi_ratio =
        hasenbusch_ratio_heatbath(lattice, config.hasenbusch.heavy_mass, &mut config.seed);

    let gauge_action_before = lattice.wilson_action();
    let (s_heavy_before, cg_h, _) =
        hasenbusch_heavy_action(lattice, &phi_heavy, &config.hasenbusch);
    let (s_ratio_before, cg_r, _) =
        hasenbusch_ratio_action(lattice, &phi_ratio, &config.hasenbusch);

    let mut cg_heavy_total = cg_h.iterations;
    let mut cg_ratio_total = cg_r.iterations;
    let fermion_action_before = s_heavy_before + s_ratio_before;

    let mut momenta = vec![Su3Matrix::ZERO; vol * 4];
    for p in &mut momenta {
        *p = Su3Matrix::random_algebra(&mut config.seed);
    }
    let kinetic_before = kinetic_energy(&momenta);
    let h_old = kinetic_before + gauge_action_before + fermion_action_before;

    hasenbusch_leapfrog(
        lattice,
        &mut momenta,
        &phi_heavy,
        &phi_ratio,
        &config.hasenbusch,
        config.dt,
    );

    let gauge_action_after = lattice.wilson_action();
    let (s_heavy_after, cg_h2, _) =
        hasenbusch_heavy_action(lattice, &phi_heavy, &config.hasenbusch);
    let (s_ratio_after, cg_r2, _) =
        hasenbusch_ratio_action(lattice, &phi_ratio, &config.hasenbusch);

    cg_heavy_total += cg_h2.iterations;
    cg_ratio_total += cg_r2.iterations;

    let fermion_action_after = s_heavy_after + s_ratio_after;
    let kinetic_after = kinetic_energy(&momenta);
    let h_new = kinetic_after + gauge_action_after + fermion_action_after;
    let delta_h = h_new - h_old;

    let accept = if delta_h <= 0.0 {
        true
    } else {
        super::super::constants::lcg_uniform_f64(&mut config.seed) < (-delta_h).exp()
    };

    if !accept {
        lattice.links = old_links;
    }

    let plaquette = lattice.average_plaquette();

    HasenbuschHmcResult {
        accepted: accept,
        delta_h,
        plaquette,
        gauge_action: if accept { gauge_action_after } else { gauge_action_before },
        fermion_action: if accept { fermion_action_after } else { fermion_action_before },
        cg_iterations_heavy: cg_heavy_total,
        cg_iterations_ratio: cg_ratio_total,
    }
}

/// Multiple time-scale leapfrog: heavy sector (outer), ratio sector (inner).
fn hasenbusch_leapfrog(
    lattice: &mut Lattice,
    momenta: &mut [Su3Matrix],
    phi_heavy: &FermionField,
    phi_ratio: &FermionField,
    config: &HasenbuschConfig,
    dt: f64,
) {
    let vol = lattice.volume();
    let dt_heavy = dt / config.n_md_steps_heavy as f64;
    let dt_light = dt_heavy / config.n_md_steps_light as f64;
    let half_dt_heavy = 0.5 * dt_heavy;
    let half_dt_light = 0.5 * dt_light;

    for _ in 0..config.n_md_steps_heavy {
        update_momenta_gauge_heavy(lattice, momenta, phi_heavy, config, half_dt_heavy);

        for _ in 0..config.n_md_steps_light {
            update_momenta_ratio_only(lattice, momenta, phi_ratio, config, half_dt_light);

            for idx in 0..vol {
                let site = lattice.site_coords(idx);
                for mu in 0..4 {
                    let p = momenta[idx * 4 + mu];
                    let u = lattice.link(site, mu);
                    let exp_p = exp_su3_cayley_pub(&p, dt_light);
                    let new_u = (exp_p * u).reunitarize();
                    lattice.set_link(site, mu, new_u);
                }
            }

            update_momenta_ratio_only(lattice, momenta, phi_ratio, config, half_dt_light);
        }

        update_momenta_gauge_heavy(lattice, momenta, phi_heavy, config, half_dt_heavy);
    }
}

fn update_momenta_gauge_heavy(
    lattice: &Lattice,
    momenta: &mut [Su3Matrix],
    phi_heavy: &FermionField,
    config: &HasenbuschConfig,
    dt: f64,
) {
    let vol = lattice.volume();

    for idx in 0..vol {
        let site = lattice.site_coords(idx);
        for mu in 0..4 {
            let gf = lattice.gauge_force(site, mu);
            momenta[idx * 4 + mu] = momenta[idx * 4 + mu] + gf.scale(dt);
        }
    }

    let mut x_heavy = FermionField::zeros(vol);
    let _ = cg_solve(
        lattice,
        &mut x_heavy,
        phi_heavy,
        config.heavy_mass,
        config.cg_tol,
        config.cg_max_iter,
    );
    let f_heavy = pseudofermion_force(lattice, &x_heavy, config.heavy_mass);
    for (m, f) in momenta.iter_mut().zip(f_heavy.iter()) {
        *m = *m + f.scale(dt);
    }
}

fn update_momenta_ratio_only(
    lattice: &Lattice,
    momenta: &mut [Su3Matrix],
    phi_ratio: &FermionField,
    config: &HasenbuschConfig,
    dt: f64,
) {
    let vol = lattice.volume();

    let mut x_ratio = FermionField::zeros(vol);
    let _ = cg_solve(
        lattice,
        &mut x_ratio,
        phi_ratio,
        config.light_mass,
        config.cg_tol,
        config.cg_max_iter,
    );
    let f_ratio = hasenbusch_ratio_force(lattice, phi_ratio, &x_ratio, config);
    for (m, f) in momenta.iter_mut().zip(f_ratio.iter()) {
        *m = *m + f.scale(dt);
    }
}

/// Kinetic energy T(P) = −(1/2) Σ Tr(P²)
pub(super) fn kinetic_energy(momenta: &[Su3Matrix]) -> f64 {
    let mut t = 0.0;
    for p in momenta {
        let p2 = *p * *p;
        t -= 0.5 * p2.re_trace();
    }
    t
}
