// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dynamical fermion HMC trajectory (leapfrog + Omelyan integrators).
//!
//! Implements the full N_f-flavor HMC with staggered pseudofermions.
//! Total Hamiltonian: H = T(π) + S_G[U] + N_f/4 × S_F[U,φ]
//!
//! For 2-flavor QCD, `N_f/4` = 2. For non-integer rooting (2+1 flavor),
//! use RHMC (`lattice/gpu_hmc/gpu_rhmc.rs`).

use crate::lattice::cg::cg_solve;
use crate::lattice::dirac::FermionField;
use crate::lattice::hmc::{IntegratorType, exp_su3_cayley_pub};
use crate::lattice::su3::Su3Matrix;
use crate::lattice::wilson::Lattice;
use crate::tolerances::OMELYAN_LAMBDA;

use super::action::{pseudofermion_action, pseudofermion_force, pseudofermion_heatbath};
use super::config::{DynamicalHmcConfig, PseudofermionConfig};
use super::hasenbusch::kinetic_energy;

/// Result of a dynamical fermion HMC trajectory.
#[derive(Clone, Debug)]
pub struct DynamicalHmcResult {
    /// Metropolis accept/reject decision.
    pub accepted: bool,
    /// Hamiltonian change ΔH.
    pub delta_h: f64,
    /// Mean plaquette.
    pub plaquette: f64,
    /// Wilson gauge action.
    pub gauge_action: f64,
    /// Fermion action (pseudofermion sector).
    pub fermion_action: f64,
    /// Total CG iterations.
    pub cg_iterations: usize,
}

/// Run one dynamical fermion HMC trajectory.
///
/// Integrator is selected from `config.integrator` (Leapfrog or Omelyan).
pub fn dynamical_hmc_trajectory(
    lattice: &mut Lattice,
    config: &mut DynamicalHmcConfig,
) -> DynamicalHmcResult {
    let vol = lattice.volume();
    let old_links = lattice.links.clone();

    let phi_fields: Vec<FermionField> = (0..config.n_flavors_over_4)
        .map(|_| pseudofermion_heatbath(lattice, config.fermion.mass, &mut config.seed))
        .collect();

    let gauge_action_before = lattice.wilson_action();
    let mut fermion_action_before = 0.0;
    let mut total_cg_iters = 0;
    for phi in &phi_fields {
        let (sf, cg_res, _) = pseudofermion_action(lattice, phi, &config.fermion);
        fermion_action_before += sf;
        total_cg_iters += cg_res.iterations;
    }

    let mut momenta = vec![Su3Matrix::ZERO; vol * 4];
    for p in &mut momenta {
        *p = Su3Matrix::random_algebra(&mut config.seed);
    }
    let kinetic_before = kinetic_energy(&momenta);
    let h_old = kinetic_before + gauge_action_before + fermion_action_before;

    match config.integrator {
        IntegratorType::Leapfrog => {
            dynamical_leapfrog(lattice, &mut momenta, &phi_fields, config.n_md_steps, config.dt, &config.fermion);
        }
        IntegratorType::Omelyan => {
            dynamical_omelyan(lattice, &mut momenta, &phi_fields, config.n_md_steps, config.dt, &config.fermion);
        }
    }

    let gauge_action_after = lattice.wilson_action();
    let mut fermion_action_after = 0.0;
    for phi in &phi_fields {
        let (sf, cg_res, _) = pseudofermion_action(lattice, phi, &config.fermion);
        fermion_action_after += sf;
        total_cg_iters += cg_res.iterations;
    }
    let kinetic_after = kinetic_energy(&momenta);
    let h_new = kinetic_after + gauge_action_after + fermion_action_after;
    let delta_h = h_new - h_old;

    let accept = if delta_h <= 0.0 {
        true
    } else {
        let r = super::super::constants::lcg_uniform_f64(&mut config.seed);
        r < (-delta_h).exp()
    };

    if !accept {
        lattice.links = old_links;
    }

    let plaquette = lattice.average_plaquette();

    DynamicalHmcResult {
        accepted: accept,
        delta_h,
        plaquette,
        gauge_action: if accept { gauge_action_after } else { gauge_action_before },
        fermion_action: if accept { fermion_action_after } else { fermion_action_before },
        cg_iterations: total_cg_iters,
    }
}

/// Omelyan PEFRL integrator with combined gauge + fermion forces.
fn dynamical_omelyan(
    lattice: &mut Lattice,
    momenta: &mut [Su3Matrix],
    phi_fields: &[FermionField],
    n_steps: usize,
    dt: f64,
    fermion_config: &PseudofermionConfig,
) {
    let vol = lattice.volume();
    let lam = OMELYAN_LAMBDA;
    let half_dt = 0.5 * dt;

    for _step in 0..n_steps {
        update_total_momenta(lattice, momenta, phi_fields, lam * dt, fermion_config);

        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let p = momenta[idx * 4 + mu];
                let u = lattice.link(x, mu);
                let exp_p = exp_su3_cayley_pub(&p, half_dt);
                let new_u = (exp_p * u).reunitarize();
                lattice.set_link(x, mu, new_u);
            }
        }

        update_total_momenta(
            lattice,
            momenta,
            phi_fields,
            2.0f64.mul_add(-lam, 1.0) * dt,
            fermion_config,
        );

        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let p = momenta[idx * 4 + mu];
                let u = lattice.link(x, mu);
                let exp_p = exp_su3_cayley_pub(&p, half_dt);
                let new_u = (exp_p * u).reunitarize();
                lattice.set_link(x, mu, new_u);
            }
        }

        update_total_momenta(lattice, momenta, phi_fields, lam * dt, fermion_config);
    }
}

/// Leapfrog integrator with combined gauge + fermion forces.
fn dynamical_leapfrog(
    lattice: &mut Lattice,
    momenta: &mut [Su3Matrix],
    phi_fields: &[FermionField],
    n_steps: usize,
    dt: f64,
    fermion_config: &PseudofermionConfig,
) {
    let vol = lattice.volume();
    let half_dt = 0.5 * dt;

    update_total_momenta(lattice, momenta, phi_fields, half_dt, fermion_config);

    for step in 0..n_steps {
        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let p = momenta[idx * 4 + mu];
                let u = lattice.link(x, mu);
                let exp_p = exp_su3_cayley_pub(&p, dt);
                let new_u = (exp_p * u).reunitarize();
                lattice.set_link(x, mu, new_u);
            }
        }

        let p_dt = if step < n_steps - 1 { dt } else { half_dt };
        update_total_momenta(lattice, momenta, phi_fields, p_dt, fermion_config);
    }
}

/// Update momenta with combined gauge + fermion force.
fn update_total_momenta(
    lattice: &Lattice,
    momenta: &mut [Su3Matrix],
    phi_fields: &[FermionField],
    dt: f64,
    fermion_config: &PseudofermionConfig,
) {
    let vol = lattice.volume();

    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let gauge_f = lattice.gauge_force(x, mu);
            momenta[idx * 4 + mu] = momenta[idx * 4 + mu] + gauge_f.scale(dt);
        }
    }

    for phi in phi_fields {
        let mut x_field = FermionField::zeros(vol);
        let _cg = cg_solve(
            lattice,
            &mut x_field,
            phi,
            fermion_config.mass,
            fermion_config.cg_tol,
            fermion_config.cg_max_iter,
        );

        let ferm_force = pseudofermion_force(lattice, &x_field, fermion_config.mass);
        for (m, f) in momenta.iter_mut().zip(ferm_force.iter()) {
            *m = *m + f.scale(dt);
        }
    }
}
