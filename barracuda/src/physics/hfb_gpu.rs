//! GPU-Accelerated Spherical HFB Solver (Level 2)
//!
//! Uses `BatchedEighGpu` from toadstool to batch eigensolves across nuclei,
//! replacing the sequential `eigh_f64` loop with a single GPU dispatch per
//! SCF iteration per basis-size group.
//!
//! Architecture:
//!   1. Group nuclei by basis dimension (n_states)
//!   2. For each group, run lockstep SCF with batched GPU eigensolves
//!   3. Remove converged nuclei between iterations
//!
//! This is the critical L2 GPU wire: transforms O(N_nuclei) sequential
//! eigensolves into O(1) batched GPU dispatches per iteration.

use super::semf::semf_binding_energy;
use super::hfb::SphericalHFB;
use barracuda::device::WgpuDevice;
use barracuda::ops::linalg::BatchedEighGpu;
use std::sync::Arc;

/// Result from GPU-batched L2 evaluation across many nuclei
#[derive(Debug)]
pub struct BatchedL2Result {
    /// (Z, N, binding_energy_mev, converged) for each nucleus
    pub results: Vec<(usize, usize, f64, bool)>,
    /// Total wall time for HFB nuclei (seconds)
    pub hfb_time_s: f64,
    /// Number of GPU eigensolve dispatches
    pub gpu_dispatches: usize,
    /// Number of nuclei solved via HFB (A in 56..=132)
    pub n_hfb: usize,
    /// Number of nuclei solved via SEMF fallback
    pub n_semf: usize,
}

/// Evaluate binding energies for a list of nuclei using GPU-batched eigensolves.
///
/// Nuclei with 56 <= A <= 132 use the spherical HFB solver with GPU-batched
/// eigendecomposition. All others use the L1 SEMF formula.
///
/// # Arguments
/// * `device` - toadstool WgpuDevice (from `GpuF64::to_wgpu_device()`)
/// * `nuclei` - list of (Z, N) pairs
/// * `params` - 10-element Skyrme parameter vector
/// * `max_iter` - maximum SCF iterations (default: 200)
/// * `tol` - energy convergence tolerance in MeV (default: 0.05)
/// * `mixing` - density mixing factor (default: 0.3)
pub fn binding_energies_l2_gpu(
    device: Arc<WgpuDevice>,
    nuclei: &[(usize, usize)],
    params: &[f64],
    max_iter: usize,
    tol: f64,
    mixing: f64,
) -> BatchedL2Result {
    let t0 = std::time::Instant::now();
    let mut results: Vec<(usize, usize, f64, bool)> = Vec::with_capacity(nuclei.len());
    let mut gpu_dispatches = 0usize;
    let mut n_hfb = 0usize;
    let mut n_semf = 0usize;

    // Partition nuclei: HFB (56 <= A <= 132) vs SEMF
    let mut hfb_nuclei: Vec<(usize, usize, usize)> = Vec::new(); // (Z, N, original_index)
    for (idx, &(z, n)) in nuclei.iter().enumerate() {
        let a = z + n;
        if a >= 56 && a <= 132 {
            hfb_nuclei.push((z, n, idx));
        } else {
            let b = semf_binding_energy(z, n, params);
            results.push((z, n, b, true));
            n_semf += 1;
        }
    }

    if hfb_nuclei.is_empty() {
        return BatchedL2Result {
            results,
            hfb_time_s: t0.elapsed().as_secs_f64(),
            gpu_dispatches,
            n_hfb: 0,
            n_semf,
        };
    }

    // Build SphericalHFB instances and group by n_states
    let mut solvers: Vec<(usize, usize, usize, SphericalHFB)> = hfb_nuclei
        .iter()
        .map(|&(z, n, idx)| {
            let hfb = SphericalHFB::new_adaptive(z, n);
            (z, n, idx, hfb)
        })
        .collect();

    // Group by n_states for batched eigensolves
    let mut groups: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, (_, _, _, hfb)) in solvers.iter().enumerate() {
        groups.entry(hfb.n_states()).or_default().push(i);
    }

    // For each group, run lockstep SCF
    // We maintain per-nucleus state and batch eigensolves within each group
    let w0 = params[9];
    let ns_groups: Vec<(usize, Vec<usize>)> = groups.into_iter().collect();

    // Per-nucleus SCF state
    struct NucleusState {
        rho_p: Vec<f64>,
        rho_n: Vec<f64>,
        e_prev: f64,
        converged: bool,
        binding_energy: f64,
        iterations: usize,
    }

    let mut states: Vec<NucleusState> = solvers
        .iter()
        .map(|(z, n, _, hfb)| {
            let a = z + n;
            let r_nuc = 1.2 * (a as f64).powf(1.0 / 3.0);
            let rho0 = 3.0 * a as f64 / (4.0 * std::f64::consts::PI * r_nuc.powi(3));
            let nr = hfb.nr();

            // Initialize Wood-Saxon-like density profile
            let rho_p: Vec<f64> = (0..nr)
                .map(|k| {
                    let r = (k + 1) as f64 * (15.0 / nr as f64); // approximate r_max
                    if r < r_nuc {
                        (rho0 * *z as f64 / a as f64).max(1e-15)
                    } else {
                        1e-15
                    }
                })
                .collect();
            let rho_n: Vec<f64> = (0..nr)
                .map(|k| {
                    let r = (k + 1) as f64 * (15.0 / nr as f64);
                    if r < r_nuc {
                        (rho0 * *n as f64 / a as f64).max(1e-15)
                    } else {
                        1e-15
                    }
                })
                .collect();

            NucleusState {
                rho_p,
                rho_n,
                e_prev: 1e10,
                converged: false,
                binding_energy: 0.0,
                iterations: 0,
            }
        })
        .collect();

    // Run SCF loop per group (batched eigensolves within each)
    for &(ns, ref group_indices) in &ns_groups {
        let mat_size = ns * ns;

        for iter in 0..max_iter {
            // Collect indices of non-converged nuclei in this group
            let active: Vec<usize> = group_indices
                .iter()
                .copied()
                .filter(|&i| !states[i].converged)
                .collect();

            if active.is_empty() {
                break;
            }

            // For each active nucleus, build proton + neutron Hamiltonians
            // Pack them into a single buffer for batched GPU dispatch
            let batch_size = active.len() * 2; // proton + neutron per nucleus
            let mut packed_hamiltonians: Vec<f64> = vec![0.0; batch_size * mat_size];

            // Build Hamiltonians on CPU (this is the O(n_states^2) part)
            for (batch_idx, &solver_idx) in active.iter().enumerate() {
                let (z, n, _, ref hfb) = solvers[solver_idx];
                let state = &states[solver_idx];

                for (species_idx, is_proton) in [(0usize, true), (1usize, false)] {
                    let h_flat = hfb.build_hamiltonian(
                        &state.rho_p,
                        &state.rho_n,
                        is_proton,
                        params,
                        w0,
                    );
                    let offset = (batch_idx * 2 + species_idx) * mat_size;
                    packed_hamiltonians[offset..offset + mat_size]
                        .copy_from_slice(&h_flat);
                }
            }

            // GPU batched eigensolve — single dispatch for ALL active nuclei
            let (eigenvalues, eigenvectors) = match BatchedEighGpu::execute_f64(
                device.clone(),
                &packed_hamiltonians,
                ns,
                batch_size,
                30, // max Jacobi sweeps
            ) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("BatchedEighGpu failed: {}", e);
                    // Fall back to marking all as unconverged
                    for &i in &active {
                        states[i].converged = false;
                    }
                    break;
                }
            };
            gpu_dispatches += 1;

            // Unpack results and update densities (CPU)
            for (batch_idx, &solver_idx) in active.iter().enumerate() {
                let (z, n, _, ref hfb) = solvers[solver_idx];
                let state = &mut states[solver_idx];

                let nr = hfb.nr();
                let mut rho_p_new = vec![1e-15; nr];
                let mut rho_n_new = vec![1e-15; nr];

                // Proton eigendecomposition (species_idx = 0)
                let eig_offset_p = (batch_idx * 2) * ns;
                let vec_offset_p = (batch_idx * 2) * mat_size;
                // Neutron eigendecomposition (species_idx = 1)
                let eig_offset_n = (batch_idx * 2 + 1) * ns;
                let vec_offset_n = (batch_idx * 2 + 1) * mat_size;

                for (species_idx, is_proton) in [(0usize, true), (1usize, false)] {
                    let eig_offset = if is_proton { eig_offset_p } else { eig_offset_n };
                    let vec_offset = if is_proton { vec_offset_p } else { vec_offset_n };

                    let eigs = &eigenvalues[eig_offset..eig_offset + ns];
                    let vecs = &eigenvectors[vec_offset..vec_offset + mat_size];

                    let num_q = if is_proton { z } else { n };
                    let delta_q = hfb.pairing_gap();

                    // BCS occupations
                    let (v2, _lambda) =
                        hfb.bcs_occupations_from_eigs(eigs, num_q, delta_q);

                    // Build new density from BCS-weighted eigenstates
                    let rho_new =
                        hfb.density_from_eigenstates(vecs, &v2, ns);

                    if is_proton {
                        rho_p_new = rho_new;
                    } else {
                        rho_n_new = rho_new;
                    }
                }

                // Density mixing
                let alpha = if iter == 0 { 0.8 } else { mixing };
                for k in 0..nr {
                    state.rho_p[k] =
                        (alpha * rho_p_new[k] + (1.0 - alpha) * state.rho_p[k]).max(1e-15);
                    state.rho_n[k] =
                        (alpha * rho_n_new[k] + (1.0 - alpha) * state.rho_n[k]).max(1e-15);
                }

                // Compute total energy
                let e_total = hfb.compute_energy_from_densities(
                    &state.rho_p,
                    &state.rho_n,
                    &eigenvalues[eig_offset_p..eig_offset_p + ns],
                    &eigenvectors[vec_offset_p..vec_offset_p + mat_size],
                    &eigenvalues[eig_offset_n..eig_offset_n + ns],
                    &eigenvectors[vec_offset_n..vec_offset_n + mat_size],
                    params,
                );

                let de = (e_total - state.e_prev).abs();
                state.e_prev = e_total;
                state.iterations = iter + 1;
                state.binding_energy = if e_total < 0.0 { -e_total } else { e_total.abs() };

                if de < tol && iter > 5 {
                    state.converged = true;
                }
            }
        }
    }

    // Collect HFB results
    n_hfb = solvers.len();
    for (i, (z, n, _, _)) in solvers.iter().enumerate() {
        results.push((
            *z,
            *n,
            states[i].binding_energy,
            states[i].converged,
        ));
    }

    BatchedL2Result {
        results,
        hfb_time_s: t0.elapsed().as_secs_f64(),
        gpu_dispatches,
        n_hfb,
        n_semf,
    }
}
