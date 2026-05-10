// SPDX-License-Identifier: AGPL-3.0-or-later

//! Core pseudofermion field operations: heat bath, action, and force.
//!
//! Implements the pseudofermion representation of the fermion determinant:
//!
//!   det(D†D) = ∫ Dφ†Dφ exp(−φ†(D†D)⁻¹φ)
//!
//! The heat bath generates φ = D†η from Gaussian noise η. The action
//! is `S_F` = φ†X where (D†D)X = φ. The force drives MD evolution.

use crate::lattice::cg::{CgResult, cg_solve};
use crate::lattice::complex_f64::Complex64;
use crate::lattice::dirac::{FermionField, apply_dirac, apply_dirac_adjoint};
use crate::lattice::su3::Su3Matrix;
use crate::lattice::wilson::Lattice;

use super::config::PseudofermionConfig;

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
            let re = super::super::constants::lcg_gaussian(seed);
            let im = super::super::constants::lcg_gaussian(seed);
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

            let neg_eta = -eta;
            let mut m_mat = Su3Matrix::ZERO;
            for a in 0..3 {
                for b in 0..3 {
                    let contrib = x_field.data[fwd_idx][a] * y_field.data[idx][b].conj()
                        - y_field.data[fwd_idx][a] * x_field.data[idx][b].conj();
                    m_mat.m[a][b] += contrib.scale(neg_eta);
                }
            }

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

/// Compute the force from the bilinear form a†(D†D)b: returns d(a†D†Db)/dU.
///
/// Used for the Hasenbusch ratio sector. Returns the gauge-projected
/// force contribution from the cross terms between fields `a` and `b`.
pub(super) fn force_bilinear_ab(
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

/// Staggered phase `η_μ(x)` = (−1)^{x_0 + ... + x_{μ−1}}
pub(super) fn staggered_phase_local(x: [usize; 4], mu: usize) -> f64 {
    let sum: usize = x.iter().take(mu).sum();
    if sum.is_multiple_of(2) { 1.0 } else { -1.0 }
}
