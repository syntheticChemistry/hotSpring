// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pseudofermion action for dynamical fermion HMC (Papers 10-12).
//!
//! In lattice QCD with dynamical fermions, the fermion determinant
//! is represented via pseudofermion fields:
//!
//!   det(D†D) = ∫ Dφ†Dφ exp(−φ†(D†D)⁻¹φ)
//!
//! The pseudofermion action is `S_F` = φ†X where `(D†D)X = φ`, solved by CG.
//! The pseudofermion force (derivative of `S_F` with respect to gauge links)
//! drives the molecular dynamics evolution alongside the gauge force.
//!
//! # Algorithm
//!
//! 1. **Heat bath**: generate φ = D†η where η is Gaussian random
//! 2. **Action**: `S_F` = φ† (D†D)⁻¹ φ = X†(D†D)X where (D†D)X = φ
//! 3. **Force**: dS\_F/dU\_μ(x) computed from X via the Dirac operator derivative
//! 4. **HMC**: total force = gauge force + fermion force
//!
//! # References
//!
//! - Gottlieb et al., PRD 35, 2531 (1987) — pseudofermion HMC
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 8.1-8.3
//! - Clark & Kennedy, NPB Proc. Suppl. 129, 850 (2004) — RHMC
//! - Hasenbusch, PLB 519, 177 (2001) — mass preconditioning

pub mod adaptive;
pub mod hasenbusch;
pub mod npu_steering;
pub mod run_history;

#[cfg(test)]
mod tests;

pub use adaptive::{
    AdaptiveStepController, AdaptiveThermalizationResult, MassAnnealingSchedule, StageResult,
    WarmStartResult, dynamical_thermalize_adaptive, dynamical_thermalize_warm_start,
    dynamical_thermalize_warm_start_npu,
};
pub use npu_steering::{HmcForceAnomalyDetector, NpuSteering};
pub use hasenbusch::{
    HasenbuschConfig, HasenbuschHmcConfig, HasenbuschHmcResult, hasenbusch_heavy_action,
    hasenbusch_heavy_heatbath, hasenbusch_hmc_trajectory, hasenbusch_ratio_action,
    hasenbusch_ratio_force, hasenbusch_ratio_heatbath,
};

use super::cg::{CgResult, cg_solve};
use super::complex_f64::Complex64;
use super::dirac::{FermionField, apply_dirac, apply_dirac_adjoint};
use super::hmc::{IntegratorType, exp_su3_cayley_pub};
use super::su3::Su3Matrix;
use super::wilson::Lattice;
use crate::tolerances::OMELYAN_LAMBDA;

/// Configuration for pseudofermion HMC.
#[derive(Clone, Debug)]
pub struct PseudofermionConfig {
    /// Fermion mass (staggered)
    pub mass: f64,
    /// CG tolerance for inversions
    pub cg_tol: f64,
    /// CG maximum iterations
    pub cg_max_iter: usize,
}

impl Default for PseudofermionConfig {
    fn default() -> Self {
        Self {
            mass: 0.1,
            cg_tol: 1e-8,
            cg_max_iter: 5000,
        }
    }
}

/// Generate pseudofermion field via heat bath: φ = D†η.
///
/// η is drawn from a Gaussian distribution (each component independently).
/// The resulting φ samples the correct weight det(D†D) when integrated
/// with exp(−φ†(D†D)⁻¹φ).
pub fn pseudofermion_heatbath(lattice: &Lattice, mass: f64, seed: &mut u64) -> FermionField {
    let vol = lattice.volume();

    let mut eta = FermionField::zeros(vol);
    for site in &mut eta.data {
        for c in site.iter_mut() {
            let re = super::constants::lcg_gaussian(seed);
            let im = super::constants::lcg_gaussian(seed);
            *c = Complex64::new(re, im);
        }
    }

    apply_dirac_adjoint(lattice, &eta, mass)
}

/// Compute the pseudofermion action: `S_F` = φ†X where (D†D)X = φ.
///
/// Returns (action, CG result, solution X) for use in the fermion force.
pub fn pseudofermion_action(
    lattice: &Lattice,
    phi: &FermionField,
    config: &PseudofermionConfig,
) -> (f64, CgResult, FermionField) {
    let vol = lattice.volume();
    let mut x = FermionField::zeros(vol);

    let cg_result = cg_solve(
        lattice,
        &mut x,
        phi,
        config.mass,
        config.cg_tol,
        config.cg_max_iter,
    );

    let action = phi.dot(&x).re;

    (action, cg_result, x)
}

/// Compute the pseudofermion force: `dS_F/dU_μ(x)` for all links.
///
/// The force is derived from the variation of `S_F` = X†(D†D)X:
///
///   dS\_F/dU\_μ(x) = −Re Tr[dD/dU\_μ(x) · (X ⊗ Y† + Y ⊗ X†)]
///
/// where Y = D X (the intermediate field after one Dirac application).
///
/// For the staggered Dirac operator, dD/dU\_μ(x) acts only on the
/// link U\_μ(x) and its neighbor, giving a local expression involving
/// the color vectors at sites x and x+μ.
#[must_use]
pub fn pseudofermion_force(lattice: &Lattice, x_field: &FermionField, mass: f64) -> Vec<Su3Matrix> {
    let vol = lattice.volume();
    let y_field = apply_dirac(lattice, x_field, mass);

    let mut force = vec![Su3Matrix::ZERO; vol * 4];

    for idx in 0..vol {
        let site = lattice.site_coords(idx);
        for mu in 0..4 {
            let eta = staggered_phase_local(site, mu);
            let fwd = lattice.neighbor(site, mu, true);
            let fwd_idx = lattice.site_index(fwd);
            let u = lattice.link(site, mu);

            // Build the outer-product matrix M from fermion fields:
            //   M = −η_μ(x) × [X(x+μ) ⊗ Y†(x) − Y(x+μ) ⊗ X†(x)]
            //
            // Sign: matches gauge force convention where shader outputs ∂S/∂U
            // (positive gradient). The per-pole fermion contribution is
            // ∂S_ferm/∂U|_s = −α_s·d(x†D†Dx)/dU, applied as P += α_s·dt·F,
            // so F = −η·TA[U(x⊗y†−y⊗x†)].
            let neg_eta = -eta;
            let mut m_mat = Su3Matrix::ZERO;
            for a in 0..3 {
                for b in 0..3 {
                    let contrib = x_field.data[fwd_idx][a] * y_field.data[idx][b].conj()
                        - y_field.data[fwd_idx][a] * x_field.data[idx][b].conj();
                    m_mat.m[a][b] += contrib.scale(neg_eta);
                }
            }

            // Force = TA(U_μ(x) × M), matching the gauge force convention
            // where the link multiplication maps from algebra at identity
            // to the tangent space at U_μ(x).
            let w = u * m_mat;
            let wh = w.adjoint();
            let mut ta = Su3Matrix::ZERO;
            for a in 0..3 {
                for b in 0..3 {
                    ta.m[a][b] = (w.m[a][b] - wh.m[a][b]).scale(0.5);
                }
            }
            let tr = (ta.m[0][0] + ta.m[1][1] + ta.m[2][2]).scale(1.0 / 3.0);
            for a in 0..3 {
                ta.m[a][a] -= tr;
            }

            force[idx * 4 + mu] = ta;
        }
    }

    force
}

/// Compute the force from the bilinear form a†(D†D)b: returns d(a† D†D b)/dU.
///
/// Used for the Hasenbusch ratio sector. The standard pseudofermion force
/// is -d(x† D†D x)/dU for (D†D) x = φ; here we need cross terms.
fn force_bilinear_ab(
    lattice: &Lattice,
    a: &FermionField,
    b: &FermionField,
    mass: f64,
) -> Vec<Su3Matrix> {
    let vol = lattice.volume();
    let da = apply_dirac(lattice, a, mass);
    let db = apply_dirac(lattice, b, mass);

    let mut force = vec![Su3Matrix::ZERO; vol * 4];

    for idx in 0..vol {
        let site = lattice.site_coords(idx);
        for mu in 0..4 {
            let eta = staggered_phase_local(site, mu);
            let fwd = lattice.neighbor(site, mu, true);
            let fwd_idx = lattice.site_index(fwd);
            let u = lattice.link(site, mu);

            // M = η/2 × [a(x+μ) ⊗ (Db)†(x) − (Db)(x+μ) ⊗ a†(x)] + [b(x+μ) ⊗ (Da)†(x) − (Da)(x+μ) ⊗ b†(x)]
            // from d(a† D†D b)/dU = (Da)† (dD/dU) b + (Db)† (dD/dU) a
            let mut m_mat = Su3Matrix::ZERO;
            for i in 0..3 {
                for j in 0..3 {
                    let term1 = a.data[fwd_idx][i] * db.data[idx][j].conj()
                        - db.data[fwd_idx][i] * a.data[idx][j].conj();
                    let term2 = b.data[fwd_idx][i] * da.data[idx][j].conj()
                        - da.data[fwd_idx][i] * b.data[idx][j].conj();
                    m_mat.m[i][j] += (term1 + term2).scale(eta * 0.5);
                }
            }

            let w = u * m_mat;
            let wh = w.adjoint();
            let mut ta = Su3Matrix::ZERO;
            for i in 0..3 {
                for j in 0..3 {
                    ta.m[i][j] = (w.m[i][j] - wh.m[i][j]).scale(0.5);
                }
            }
            let tr = (ta.m[0][0] + ta.m[1][1] + ta.m[2][2]).scale(1.0 / 3.0);
            for i in 0..3 {
                ta.m[i][i] -= tr;
            }

            force[idx * 4 + mu] = ta;
        }
    }

    force
}

/// Staggered phase `η_μ(x)` = (−`1)^{x_0` + ... + x_{μ−1}}
fn staggered_phase_local(x: [usize; 4], mu: usize) -> f64 {
    let sum: usize = x.iter().take(mu).sum();
    if sum.is_multiple_of(2) { 1.0 } else { -1.0 }
}

/// Dynamical fermion HMC configuration.
#[derive(Clone, Debug)]
pub struct DynamicalHmcConfig {
    /// Number of MD steps per trajectory
    pub n_md_steps: usize,
    /// MD step size
    pub dt: f64,
    /// PRNG seed (mutated each trajectory)
    pub seed: u64,
    /// Pseudofermion configuration
    pub fermion: PseudofermionConfig,
    /// Gauge coupling (β)
    pub beta: f64,
    /// Number of staggered flavors (`N_f/4` for staggered; use 2 for 2-flavor)
    pub n_flavors_over_4: usize,
    /// Integrator type (default: Leapfrog for backward compat)
    pub integrator: IntegratorType,
}

impl Default for DynamicalHmcConfig {
    fn default() -> Self {
        Self {
            n_md_steps: 20,
            dt: 0.02,
            seed: 42,
            fermion: PseudofermionConfig::default(),
            beta: 5.5,
            n_flavors_over_4: 2,
            integrator: IntegratorType::Leapfrog,
        }
    }
}

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
/// Total Hamiltonian: H = T(π) + S\_G\[U\] + N\_f/4 × S\_F\[U,φ\]
///
/// The staggered determinant gives `det(D†D)^{N_f/4`}. For 2-flavor QCD,
/// `N_f/4` = 2 (one staggered field represents 4 tastes → 2 pseudofermion
/// fields for 2+1 would need RHMC; this implementation uses even `N_f` only).
pub fn dynamical_hmc_trajectory(
    lattice: &mut Lattice,
    config: &mut DynamicalHmcConfig,
) -> DynamicalHmcResult {
    let vol = lattice.volume();
    let old_links = lattice.links.clone();

    let phi_fields: Vec<FermionField> = (0..config.n_flavors_over_4)
        .map(|_| pseudofermion_heatbath(lattice, config.fermion.mass, &mut config.seed))
        .collect();

    // 2. Compute initial Hamiltonian
    let gauge_action_before = lattice.wilson_action();
    let mut fermion_action_before = 0.0;
    let mut total_cg_iters = 0;
    for phi in &phi_fields {
        let (sf, cg_res, _) = pseudofermion_action(lattice, phi, &config.fermion);
        fermion_action_before += sf;
        total_cg_iters += cg_res.iterations;
    }

    // Generate random momenta
    let mut momenta = vec![Su3Matrix::ZERO; vol * 4];
    for p in &mut momenta {
        *p = Su3Matrix::random_algebra(&mut config.seed);
    }
    let kinetic_before = kinetic_energy(&momenta);

    let h_old = kinetic_before + gauge_action_before + fermion_action_before;

    // 3. MD integration with combined gauge + fermion force
    match config.integrator {
        IntegratorType::Leapfrog => dynamical_leapfrog(
            lattice,
            &mut momenta,
            &phi_fields,
            config.n_md_steps,
            config.dt,
            &config.fermion,
        ),
        IntegratorType::Omelyan => dynamical_omelyan(
            lattice,
            &mut momenta,
            &phi_fields,
            config.n_md_steps,
            config.dt,
            &config.fermion,
        ),
    }

    // 4. Compute final Hamiltonian
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

    // 5. Metropolis accept/reject
    let accept = if delta_h <= 0.0 {
        true
    } else {
        let r = super::constants::lcg_uniform_f64(&mut config.seed);
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
        gauge_action: if accept {
            gauge_action_after
        } else {
            gauge_action_before
        },
        fermion_action: if accept {
            fermion_action_after
        } else {
            fermion_action_before
        },
        cg_iterations: total_cg_iters,
    }
}

/// Omelyan integrator with combined gauge + fermion forces.
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
        // P -> P + λ·dt·F(U)
        update_total_momenta(lattice, momenta, phi_fields, lam * dt, fermion_config);

        // U -> U + dt/2·P
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

        // P -> P + (1-2λ)·dt·F(U)
        update_total_momenta(
            lattice,
            momenta,
            phi_fields,
            2.0f64.mul_add(-lam, 1.0) * dt,
            fermion_config,
        );

        // U -> U + dt/2·P
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

        // P -> P + λ·dt·F(U)
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
        // Link update: U' = exp(dt·P) U
        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let p = momenta[idx * 4 + mu];
                let u = lattice.link(x, mu);
                let exp_p = super::hmc::exp_su3_cayley_pub(&p, dt);
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

    // Gauge force
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let gauge_f = lattice.gauge_force(x, mu);
            momenta[idx * 4 + mu] = momenta[idx * 4 + mu] + gauge_f.scale(dt);
        }
    }

    // Fermion force (one CG solve per pseudofermion field)
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

/// Kinetic energy T(P) = −(1/2) Σ Tr(P²)
fn kinetic_energy(momenta: &[Su3Matrix]) -> f64 {
    let mut t = 0.0;
    for p in momenta {
        let p2 = *p * *p;
        t -= 0.5 * p2.re_trace();
    }
    t
}
