// SPDX-License-Identifier: AGPL-3.0-only

//! Data types for the GPU-resident spherical HFB solver.

/// Internal state for each nucleus during SCF iterations.
pub(super) struct NucleusState {
    pub(super) rho_p: Vec<f64>,
    pub(super) rho_n: Vec<f64>,
    pub(super) e_prev: f64,
    pub(super) converged: bool,
    pub(super) binding_energy: f64,
}

/// Work item for SCF iteration (nucleus index, group index, basis size).
pub(super) struct WorkItem {
    pub(super) si: usize,
    pub(super) gi: usize,
    pub(super) ns: usize,
}

/// Per-nucleus results from eigenvalue extraction and BCS occupation computation.
pub(super) struct EigenBcsResult {
    pub(super) si: usize,
    pub(super) gi: usize,
    pub(super) evals_p: Vec<f64>,
    pub(super) evecs_p: Vec<f64>,
    pub(super) evals_n: Vec<f64>,
    pub(super) evecs_n: Vec<f64>,
    pub(super) lambda_p: f64,
    pub(super) lambda_n: f64,
    pub(super) v2_p: Vec<f64>,
    pub(super) v2_n: Vec<f64>,
}

/// Results from the GPU-resident L2 HFB binding energy computation.
///
/// Contains per-nucleus binding energies, convergence flags, timing,
/// and dispatch statistics for performance analysis.
#[derive(Debug)]
pub struct GpuResidentL2Result {
    /// Per-nucleus results: `(Z, N, binding_energy_MeV, converged)`.
    pub results: Vec<(usize, usize, f64, bool)>,
    /// Wall-clock time for the HFB SCF computation (seconds).
    pub hfb_time_s: f64,
    /// GPU dispatches for eigensolve (one per SCF iteration with active nuclei).
    pub gpu_dispatches: usize,
    /// Total GPU dispatches including potentials, Hamiltonian, and eigensolve.
    pub total_gpu_dispatches: usize,
    /// Number of nuclei computed with full HFB (A > 20).
    pub n_hfb: usize,
    /// Number of nuclei computed with SEMF fallback (A <= 20).
    pub n_semf: usize,
}

/// Build initial Wood-Saxon density profiles for all solvers.
pub(super) fn build_initial_densities(
    solvers: &[(usize, usize, usize, crate::physics::hfb::SphericalHFB)],
) -> Vec<NucleusState> {
    solvers
        .iter()
        .map(|(z, n, _, hfb)| {
            let (rho_p, rho_n) =
                crate::physics::hfb_common::initial_wood_saxon_density(*z, *n, hfb.nr(), hfb.dr());
            NucleusState {
                rho_p,
                rho_n,
                e_prev: 1e10,
                converged: false,
                binding_energy: 0.0,
            }
        })
        .collect()
}

/// Check convergence and update nucleus state with new densities.
pub(super) fn check_convergence_and_update_state(
    state: &mut NucleusState,
    rho_p_mixed: Vec<f64>,
    rho_n_mixed: Vec<f64>,
    e_total: f64,
    tol: f64,
    iter: usize,
) -> bool {
    let de = (e_total - state.e_prev).abs();
    let binding_energy = e_total.abs();
    let converged = de < tol && iter > 5;

    state.rho_p = rho_p_mixed;
    state.rho_n = rho_n_mixed;
    state.e_prev = e_total;
    state.binding_energy = binding_energy;
    state.converged = converged;

    converged
}

/// Compute binding energy from single-particle energies and BCS occupations.
pub(super) fn compute_binding_energy(
    evals_p: &[f64],
    evals_n: &[f64],
    v2_p: &[f64],
    v2_n: &[f64],
    degs: &[f64],
    delta: f64,
) -> f64 {
    let ns = evals_p.len();
    let e_sp: f64 = (0..ns)
        .map(|i| (degs[i] * v2_p[i]).mul_add(evals_p[i], degs[i] * v2_n[i] * evals_n[i]))
        .sum();
    let e_pair: f64 = (0..ns)
        .map(|i| {
            let vu_p = (v2_p[i] * (1.0 - v2_p[i])).max(0.0).sqrt();
            let vu_n = (v2_n[i] * (1.0 - v2_n[i])).max(0.0).sqrt();
            -delta * degs[i] * (vu_p + vu_n)
        })
        .sum();
    e_sp + e_pair
}

/// Compute spin-orbit diagonal correction from densities.
pub(super) fn compute_spin_orbit_diagonal(
    hfb: &crate::physics::hfb::SphericalHFB,
    rho_p: &[f64],
    rho_n: &[f64],
    w0: f64,
) -> Vec<f64> {
    use crate::tolerances::SPIN_ORBIT_R_MIN;

    let ns = hfb.n_states();
    let mut so_diag = vec![0.0; ns];
    if w0 == 0.0 {
        return so_diag;
    }
    let cur_nr = hfb.nr();
    let cur_dr = hfb.dr();
    let rho_total: Vec<f64> = (0..cur_nr).map(|k| rho_p[k] + rho_n[k]).collect();
    let drho = barracuda::numerical::gradient_1d(&rho_total, cur_dr);
    let r = hfb.r_grid();
    let lj_qn = hfb.lj_quantum_numbers();

    for i in 0..ns {
        let (l, j) = lj_qn[i];
        if l == 0 {
            continue;
        }
        let ls = barracuda::ops::grid::compute_ls_factor(l as u32, j);
        let wf_i = hfb.wf_state(i);
        let so_integ: Vec<f64> = (0..cur_nr)
            .map(|k| wf_i[k].powi(2) * drho[k] / r[k].max(SPIN_ORBIT_R_MIN) * r[k].powi(2))
            .collect();
        let so_val = w0 * ls * barracuda::numerical::trapz(&so_integ, r).unwrap_or(0.0);
        so_diag[i] = so_val;
    }
    so_diag
}

/// Extract eigenvalues/eigenvectors from GPU readback and compute BCS occupations.
pub(super) fn extract_bcs_results(
    all_work: &[WorkItem],
    gpu_eigen: &(Vec<f64>, Vec<f64>),
    solvers: &[(usize, usize, usize, crate::physics::hfb::SphericalHFB)],
    global_max_ns: usize,
) -> Vec<EigenBcsResult> {
    use rayon::prelude::*;

    let (eigenvalues, eigenvectors) = gpu_eigen;
    let gm = global_max_ns * global_max_ns;

    all_work
        .par_iter()
        .enumerate()
        .map(|(wi, item): (usize, &WorkItem)| {
            let (z, n, _, ref hfb) = solvers[item.si];
            let ns = item.ns;

            let extract = |species: usize| {
                let eig_off = (wi * 2 + species) * global_max_ns;
                let vec_off = (wi * 2 + species) * gm;
                let mut evals: Vec<f64> = eigenvalues[eig_off..eig_off + global_max_ns]
                    .iter()
                    .copied()
                    .filter(|&e| e < 1e9)
                    .collect();
                evals.truncate(ns);
                if evals.len() < ns {
                    evals.resize(ns, 0.0);
                }
                let mut evecs = vec![0.0; ns * ns];
                for col in 0..ns {
                    if eigenvalues[eig_off + col] < 1e9 {
                        for row in 0..ns {
                            evecs[col * ns + row] =
                                eigenvectors[vec_off + col * global_max_ns + row];
                        }
                    }
                }
                (evals, evecs)
            };
            let (evals_p, evecs_p) = extract(0);
            let (evals_n, evecs_n) = extract(1);

            let (v2_p, lambda_p): (Vec<f64>, f64) =
                hfb.bcs_occupations_from_eigs(&evals_p, z, hfb.pairing_gap());
            let (v2_n, lambda_n): (Vec<f64>, f64) =
                hfb.bcs_occupations_from_eigs(&evals_n, n, hfb.pairing_gap());

            EigenBcsResult {
                si: item.si,
                gi: item.gi,
                evals_p,
                evecs_p,
                evals_n,
                evecs_n,
                lambda_p,
                lambda_n,
                v2_p,
                v2_n,
            }
        })
        .collect()
}
