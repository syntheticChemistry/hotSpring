// SPDX-License-Identifier: AGPL-3.0-only

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

use super::cg::{cg_solve, CgResult};
use super::complex_f64::Complex64;
use super::dirac::{apply_dirac, apply_dirac_adjoint, apply_dirac_sq, FermionField};
use super::hmc::{exp_su3_cayley_pub, IntegratorType};
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

/// Configuration for Hasenbusch mass preconditioning (Hasenbusch 2001, PLB 519, 177).
///
/// Two-level split: det(D†D(m_light)) = det(D†D(m_heavy)) × det(D†D(m_light)/D†D(m_heavy)).
/// Heavy sector is cheap (few CG iterations); ratio sector has smaller condition number than
/// the full light-mass operator → faster CG and smaller forces.
#[derive(Clone, Debug)]
pub struct HasenbuschConfig {
    /// Heavy (intermediate) mass — typically 0.3–0.5.
    pub heavy_mass: f64,
    /// Light (physical) mass — typically 0.01–0.1.
    pub light_mass: f64,
    /// CG tolerance for inversions.
    pub cg_tol: f64,
    /// CG maximum iterations per solve.
    pub cg_max_iter: usize,
    /// MD steps for the light (ratio) sector — more steps (expensive inversions).
    pub n_md_steps_light: usize,
    /// MD steps for the heavy sector — fewer steps (cheap inversions).
    pub n_md_steps_heavy: usize,
}

impl Default for HasenbuschConfig {
    fn default() -> Self {
        Self {
            heavy_mass: 0.4,
            light_mass: 0.1,
            cg_tol: 1e-8,
            cg_max_iter: 5000,
            n_md_steps_light: 16,
            n_md_steps_heavy: 4,
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
            //   M = η_μ(x)/2 × [X(x+μ) ⊗ Y†(x) − Y(x+μ) ⊗ X†(x)]
            let mut m_mat = Su3Matrix::ZERO;
            for a in 0..3 {
                for b in 0..3 {
                    let contrib = x_field.data[fwd_idx][a] * y_field.data[idx][b].conj()
                        - y_field.data[fwd_idx][a] * x_field.data[idx][b].conj();
                    m_mat.m[a][b] += contrib.scale(eta * 0.5);
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
    if sum.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Hasenbusch mass preconditioning (2-level split)
// ═══════════════════════════════════════════════════════════════════

/// Heatbath for the heavy sector: φ_H = D†(m_heavy) η.
pub fn hasenbusch_heavy_heatbath(
    lattice: &Lattice,
    heavy_mass: f64,
    seed: &mut u64,
) -> FermionField {
    pseudofermion_heatbath(lattice, heavy_mass, seed)
}

/// Heatbath for the ratio sector: φ = D†(m_heavy) η.
///
/// Standard approximation (exact would use matrix square root).
pub fn hasenbusch_ratio_heatbath(
    lattice: &Lattice,
    heavy_mass: f64,
    seed: &mut u64,
) -> FermionField {
    pseudofermion_heatbath(lattice, heavy_mass, seed)
}

/// Action for the heavy sector: S_H = φ†(D†D(m_heavy))⁻¹φ.
///
/// Returns (action, CG result, solution x).
pub fn hasenbusch_heavy_action(
    lattice: &Lattice,
    phi: &FermionField,
    config: &HasenbuschConfig,
) -> (f64, CgResult, FermionField) {
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

/// Action for the ratio sector: S_ratio = φ†(D†D(m_light))⁻¹ D†D(m_heavy) φ.
///
/// Solve (D†D(m_light)) x = φ, then S = φ† D†D(m_heavy) x.
/// Returns (action, CG result, x for force computation).
pub fn hasenbusch_ratio_action(
    lattice: &Lattice,
    phi: &FermionField,
    config: &HasenbuschConfig,
) -> (f64, CgResult, FermionField) {
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

    // S = φ† D†D(m_heavy) x
    let ddx = apply_dirac_sq(lattice, &x, config.heavy_mass);
    let action = phi.dot(&ddx).re;

    (action, cg_result, x)
}

/// Force for the ratio sector.
///
/// S_ratio = φ† A⁻¹ B φ with A = D†D(m_light), B = D†D(m_heavy).
/// dS/dU = -y†(dA/dU)x + φ†A⁻¹(dB/dU)φ, where x = A⁻¹φ, y = A⁻¹Bφ.
/// -dS/dU = +y†(dA/dU)x - x†(dB/dU)φ.
#[must_use]
pub fn hasenbusch_ratio_force(
    lattice: &Lattice,
    phi: &FermionField,
    x: &FermionField,
    config: &HasenbuschConfig,
) -> Vec<Su3Matrix> {
    // Solve (D†D(m_light)) y = D†D(m_heavy) φ
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

    // -dS/dU = +y†(dA/dU)x - x†(dB/dU)φ
    // = force_bilinear_ab(y,x,m_light) - force_bilinear_ab(x,phi,m_heavy)
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
#[allow(missing_docs)]
pub struct HasenbuschHmcResult {
    pub accepted: bool,
    pub delta_h: f64,
    pub plaquette: f64,
    pub gauge_action: f64,
    pub fermion_action: f64,
    pub cg_iterations_heavy: usize,
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

    // 1. Heatbath for both sectors
    let phi_heavy =
        hasenbusch_heavy_heatbath(lattice, config.hasenbusch.heavy_mass, &mut config.seed);
    let phi_ratio =
        hasenbusch_ratio_heatbath(lattice, config.hasenbusch.heavy_mass, &mut config.seed);

    // 2. Initial Hamiltonian
    let gauge_action_before = lattice.wilson_action();
    let (s_heavy_before, cg_h, _) =
        hasenbusch_heavy_action(lattice, &phi_heavy, &config.hasenbusch);
    let (s_ratio_before, cg_r, _) =
        hasenbusch_ratio_action(lattice, &phi_ratio, &config.hasenbusch);

    let mut cg_heavy_total = cg_h.iterations;
    let mut cg_ratio_total = cg_r.iterations;

    let fermion_action_before = s_heavy_before + s_ratio_before;

    // Random momenta
    let mut momenta = vec![Su3Matrix::ZERO; vol * 4];
    for p in &mut momenta {
        *p = Su3Matrix::random_algebra(&mut config.seed);
    }
    let kinetic_before = kinetic_energy(&momenta);
    let h_old = kinetic_before + gauge_action_before + fermion_action_before;

    // 3. Multiple time-scale leapfrog
    hasenbusch_leapfrog(
        lattice,
        &mut momenta,
        &phi_heavy,
        &phi_ratio,
        &config.hasenbusch,
        config.dt,
    );

    // 4. Final Hamiltonian
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

    // 5. Metropolis
    let accept = if delta_h <= 0.0 {
        true
    } else {
        super::constants::lcg_uniform_f64(&mut config.seed) < (-delta_h).exp()
    };

    if !accept {
        lattice.links = old_links;
    }

    let plaquette = lattice.average_plaquette();

    HasenbuschHmcResult {
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
        cg_iterations_heavy: cg_heavy_total,
        cg_iterations_ratio: cg_ratio_total,
    }
}

/// Configuration for Hasenbusch HMC.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct HasenbuschHmcConfig {
    pub dt: f64,
    pub seed: u64,
    pub hasenbusch: HasenbuschConfig,
    pub beta: f64,
}

impl Default for HasenbuschHmcConfig {
    fn default() -> Self {
        Self {
            dt: 0.02,
            seed: 42,
            hasenbusch: HasenbuschConfig::default(),
            beta: 5.5,
        }
    }
}

/// Multiple time-scale leapfrog: heavy sector (outer, fewer steps), ratio sector (inner, more steps).
///
/// The ratio (expensive, larger forces) gets n_md_steps_light sub-steps per heavy step.
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
        // Half kick: gauge + heavy only (ratio is on inner scale)
        hasenbusch_update_momenta_gauge_heavy(lattice, momenta, phi_heavy, config, half_dt_heavy);

        for _ in 0..config.n_md_steps_light {
            hasenbusch_update_momenta_ratio_only(
                lattice,
                momenta,
                phi_ratio,
                config,
                half_dt_light,
            );

            for idx in 0..vol {
                let site = lattice.site_coords(idx);
                for mu in 0..4 {
                    let p = momenta[idx * 4 + mu];
                    let u = lattice.link(site, mu);
                    let exp_p = super::hmc::exp_su3_cayley_pub(&p, dt_light);
                    let new_u = (exp_p * u).reunitarize();
                    lattice.set_link(site, mu, new_u);
                }
            }

            hasenbusch_update_momenta_ratio_only(
                lattice,
                momenta,
                phi_ratio,
                config,
                half_dt_light,
            );
        }

        hasenbusch_update_momenta_gauge_heavy(lattice, momenta, phi_heavy, config, half_dt_heavy);
    }
}

fn hasenbusch_update_momenta_gauge_heavy(
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

fn hasenbusch_update_momenta_ratio_only(
    lattice: &Lattice,
    momenta: &mut [Su3Matrix],
    phi_ratio: &FermionField,
    config: &HasenbuschConfig,
    dt: f64,
) {
    let vol = lattice.volume();

    // Only ratio sector force (no gauge, no heavy)
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
#[allow(missing_docs)]
pub struct DynamicalHmcResult {
    pub accepted: bool,
    pub delta_h: f64,
    pub plaquette: f64,
    pub gauge_action: f64,
    pub fermion_action: f64,
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
            (1.0 - 2.0 * lam) * dt,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heatbath_produces_nonzero_field() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let mut seed = 42u64;
        let phi = pseudofermion_heatbath(&lat, 0.1, &mut seed);
        let norm = phi.norm_sq();
        assert!(norm > 0.0, "pseudofermion should be nonzero: norm²={norm}");
    }

    #[test]
    fn pseudofermion_action_positive() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let mut seed = 42u64;
        let phi = pseudofermion_heatbath(&lat, 0.5, &mut seed);
        let config = PseudofermionConfig {
            mass: 0.5,
            cg_tol: 1e-8,
            cg_max_iter: 1000,
        };
        let (action, cg_res, _) = pseudofermion_action(&lat, &phi, &config);
        assert!(cg_res.converged, "CG should converge");
        assert!(
            action > 0.0,
            "pseudofermion action should be positive: {action}"
        );
    }

    #[test]
    fn fermion_force_is_traceless_antihermitian() {
        let lat = Lattice::hot_start([4, 4, 4, 4], 5.5, 42);
        let vol = lat.volume();
        let x_field = FermionField::random(vol, 99);
        let force = pseudofermion_force(&lat, &x_field, 0.1);

        for (i, f) in force.iter().enumerate().take(10) {
            let tr = f.m[0][0] + f.m[1][1] + f.m[2][2];
            assert!(
                tr.abs_sq() < 1e-20,
                "force[{i}] trace should be ~0: |tr|²={}",
                tr.abs_sq()
            );
            for a in 0..3 {
                for b in 0..3 {
                    let diff = f.m[a][b] + f.m[b][a].conj();
                    assert!(
                        diff.abs_sq() < 1e-20,
                        "force[{i}] ({a},{b}) should be anti-Hermitian: |f+f†|²={}",
                        diff.abs_sq()
                    );
                }
            }
        }
    }

    #[test]
    fn dynamical_hmc_cold_start_runs() {
        let mut lat = Lattice::cold_start([4, 4, 4, 4], 5.5);
        let mut config = DynamicalHmcConfig {
            n_md_steps: 5,
            dt: 0.01,
            seed: 42,
            fermion: PseudofermionConfig {
                mass: 0.5,
                cg_tol: 1e-6,
                cg_max_iter: 500,
            },
            beta: 5.5,
            n_flavors_over_4: 1,
            ..Default::default()
        };

        let result = dynamical_hmc_trajectory(&mut lat, &mut config);
        assert!(
            result.plaquette > 0.0 && result.plaquette <= 1.0,
            "plaquette should be in (0,1]: {}",
            result.plaquette
        );
        assert!(
            result.cg_iterations > 0,
            "should need at least 1 CG iteration"
        );
    }

    /// Compute squared Frobenius norm of a force field.
    fn force_norm_sq(force: &[Su3Matrix]) -> f64 {
        force.iter().map(|f| f.norm_sq()).sum()
    }

    #[test]
    fn hasenbusch_reduces_force_norm() {
        let lat = Lattice::hot_start([4, 4, 4, 4], 5.5, 42);
        let vol = lat.volume();
        let mut seed = 43u64;

        let phi = hasenbusch_ratio_heatbath(&lat, 0.4, &mut seed);

        let config = HasenbuschConfig {
            heavy_mass: 0.4,
            light_mass: 0.1,
            cg_tol: 1e-6,
            cg_max_iter: 500,
            n_md_steps_light: 4,
            n_md_steps_heavy: 2,
        };

        let mut x_heavy = FermionField::zeros(vol);
        let mut x_light = FermionField::zeros(vol);

        let _ = cg_solve(
            &lat,
            &mut x_heavy,
            &phi,
            config.heavy_mass,
            config.cg_tol,
            config.cg_max_iter,
        );
        let _ = cg_solve(
            &lat,
            &mut x_light,
            &phi,
            config.light_mass,
            config.cg_tol,
            config.cg_max_iter,
        );

        let force_heavy = pseudofermion_force(&lat, &x_heavy, config.heavy_mass);
        let force_light = pseudofermion_force(&lat, &x_light, config.light_mass);

        let norm_heavy = force_norm_sq(&force_heavy);
        let norm_light = force_norm_sq(&force_light);

        assert!(
            norm_heavy < norm_light,
            "heavy-mass force norm² ({norm_heavy}) should be smaller than light-mass ({norm_light})"
        );
    }

    #[test]
    fn hasenbusch_cg_converges_faster() {
        let lat = Lattice::hot_start([4, 4, 4, 4], 5.5, 42);
        let vol = lat.volume();
        let mut seed = 44u64;

        let phi = hasenbusch_ratio_heatbath(&lat, 0.4, &mut seed);

        let cg_tol = 1e-6;
        let cg_max = 500;

        let mut x_heavy = FermionField::zeros(vol);
        let mut x_light = FermionField::zeros(vol);

        let res_heavy = cg_solve(&lat, &mut x_heavy, &phi, 0.4, cg_tol, cg_max);
        let res_light = cg_solve(&lat, &mut x_light, &phi, 0.1, cg_tol, cg_max);

        assert!(
            res_heavy.converged,
            "CG at heavy mass should converge: {} iters",
            res_heavy.iterations
        );
        assert!(
            res_light.converged,
            "CG at light mass should converge: {} iters",
            res_light.iterations
        );
        assert!(
            res_heavy.iterations < res_light.iterations,
            "CG at m_heavy ({}) should need fewer iterations than at m_light ({})",
            res_heavy.iterations,
            res_light.iterations
        );
    }

    #[test]
    fn hasenbusch_hmc_runs() {
        let mut lat = Lattice::cold_start([4, 4, 4, 4], 5.5);
        let mut config = HasenbuschHmcConfig {
            dt: 0.02,
            seed: 42,
            hasenbusch: HasenbuschConfig {
                heavy_mass: 0.4,
                light_mass: 0.1,
                cg_tol: 1e-6,
                cg_max_iter: 500,
                n_md_steps_light: 4,
                n_md_steps_heavy: 2,
            },
            beta: 5.5,
        };

        let result = hasenbusch_hmc_trajectory(&mut lat, &mut config);

        assert!(
            result.plaquette > 0.0 && result.plaquette <= 1.0,
            "plaquette should be in (0,1]: {}",
            result.plaquette
        );
        assert!(
            result.cg_iterations_heavy > 0 || result.cg_iterations_ratio > 0,
            "should need CG iterations"
        );
        assert!(
            result.cg_iterations_heavy < result.cg_iterations_ratio,
            "heavy sector ({}) should use fewer CG iters than ratio ({})",
            result.cg_iterations_heavy,
            result.cg_iterations_ratio
        );
    }
}
