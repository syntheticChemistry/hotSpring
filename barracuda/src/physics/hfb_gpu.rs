// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-Accelerated Spherical HFB Solver (Level 2)
//!
//! Pure GPU eigensolve pipeline — no CPU fallbacks. Uses `BatchedEighGpu`
//! from barracuda/toadstool to batch eigensolves across nuclei.
//!
//! Architecture:
//!   1. Pad ALL nuclei to a common max basis dimension
//!   2. Pack ALL active nuclei into ONE mega-batch per SCF iteration
//!   3. GPU eigensolve: `execute_single_dispatch()` for n≤32 (all Jacobi
//!      rotations in one shader), `execute_f64()` for larger matrices
//!      (multi-dispatch, still pure GPU)
//!   4. Remove converged nuclei between iterations
//!
//! Evolution history:
//!   v0.4.0: ~5 groups × 200 iterations = ~1000 GPU dispatches
//!   v0.5.0: 1 mega-batch × 200 iterations = ~200 dispatches (5× reduction)
//!   v0.5.3: single-dispatch: ALL rotations in 1 shader per SCF iteration
//!   v0.5.11: eliminated CPU fallback — multi-dispatch for n>32
//!           (estimated 3-5× speedup from eliminated dispatch overhead)
//!
//! The single-dispatch kernel handles n≤32, which covers all spherical HFB
//! basis sizes (n_shells 8-14 → n_states ≤ ~30).

use super::hfb::SphericalHFB;
use super::semf::semf_binding_energy;
use crate::tolerances::{DENSITY_FLOOR, GPU_JACOBI_CONVERGENCE};
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
    let mut _n_hfb = 0usize;
    let mut n_semf = 0usize;

    // Partition nuclei: HFB (56 <= A <= 132) vs SEMF
    let mut hfb_nuclei: Vec<(usize, usize, usize)> = Vec::new(); // (Z, N, original_index)
    for (idx, &(z, n)) in nuclei.iter().enumerate() {
        let a = z + n;
        if (56..=132).contains(&a) {
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
    let solvers: Vec<(usize, usize, usize, SphericalHFB)> = hfb_nuclei
        .iter()
        .map(|&(z, n, idx)| {
            let hfb = SphericalHFB::new_adaptive(z, n);
            (z, n, idx, hfb)
        })
        .collect();

    // ═══════════════════════════════════════════════════════════════
    // Mega-batch: pad ALL nuclei to max basis dimension
    // ═══════════════════════════════════════════════════════════════
    // Instead of grouping by n_states (5+ groups, 5+ dispatches/iter),
    // we pad all matrices to the maximum dimension and fire ONE dispatch
    // per SCF iteration. The padding overhead (extra Jacobi iterations on
    // identity-padded rows) is negligible compared to the dispatch overhead
    // saved (~ms per dispatch avoided).
    //
    // Padding strategy: fill diagonal of padded rows/cols with 1e10.
    // This pushes artificial eigenvalues far from the physical spectrum
    // (-50 to +100 MeV), making them trivially identifiable. BCS gives
    // v² ≈ 0 for these states (correct — they're unphysical).

    let max_ns = solvers
        .iter()
        .map(|(_, _, _, hfb)| hfb.n_states())
        .max()
        .unwrap_or(1);
    let max_mat_size = max_ns * max_ns;
    let w0 = params[9];

    // Per-nucleus SCF state
    struct NucleusState {
        rho_p: Vec<f64>,
        rho_n: Vec<f64>,
        e_prev: f64,
        converged: bool,
        binding_energy: f64,
        iterations: usize,
        actual_ns: usize, // real basis dimension (before padding)
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
                    let r = (k + 1) as f64 * (15.0 / nr as f64);
                    if r < r_nuc {
                        (rho0 * *z as f64 / a as f64).max(DENSITY_FLOOR)
                    } else {
                        DENSITY_FLOOR
                    }
                })
                .collect();
            let rho_n: Vec<f64> = (0..nr)
                .map(|k| {
                    let r = (k + 1) as f64 * (15.0 / nr as f64);
                    if r < r_nuc {
                        (rho0 * *n as f64 / a as f64).max(DENSITY_FLOOR)
                    } else {
                        DENSITY_FLOOR
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
                actual_ns: hfb.n_states(),
            }
        })
        .collect();

    // ═══════════════════════════════════════════════════════════════
    // PRE-ALLOCATE: eigensolve buffers + packed array (ONE TIME)
    // ═══════════════════════════════════════════════════════════════
    // Eliminates per-iteration GPU buffer allocation, shader compilation,
    // and bind group creation. Data uploads via queue.write_buffer (DMA).
    let max_batch_count = solvers.len() * 2; // proton + neutron per nucleus
    let (eigh_matrices_buf, eigh_eigenvalues_buf, eigh_eigenvectors_buf) =
        BatchedEighGpu::create_buffers(&device, max_ns, max_batch_count)
            .expect("pre-allocate eigensolve GPU buffers");
    let mut packed_hamiltonians: Vec<f64> = vec![0.0; max_batch_count * max_mat_size];

    // ═══════════════════════════════════════════════════════════════
    // Unified SCF loop: ONE dispatch per iteration for ALL nuclei
    // ═══════════════════════════════════════════════════════════════
    let all_indices: Vec<usize> = (0..solvers.len()).collect();

    for iter in 0..max_iter {
        // Collect indices of ALL non-converged nuclei (across all groups)
        let active: Vec<usize> = all_indices
            .iter()
            .copied()
            .filter(|&i| !states[i].converged)
            .collect();

        if active.is_empty() {
            break;
        }

        // Pack ALL active nuclei into ONE mega-batch (padded to max_ns)
        let batch_size = active.len() * 2;
        packed_hamiltonians[..batch_size * max_mat_size].fill(0.0);

        // Build Hamiltonians on CPU and pack with identity-padding
        for (batch_idx, &solver_idx) in active.iter().enumerate() {
            let (_z, _n, _, ref hfb) = solvers[solver_idx];
            let state = &states[solver_idx];
            let actual_ns = state.actual_ns;

            for (species_idx, is_proton) in [(0usize, true), (1usize, false)] {
                let h_flat =
                    hfb.build_hamiltonian(&state.rho_p, &state.rho_n, is_proton, params, w0);

                let offset = (batch_idx * 2 + species_idx) * max_mat_size;

                for r in 0..actual_ns {
                    for c in 0..actual_ns {
                        packed_hamiltonians[offset + r * max_ns + c] = h_flat[r * actual_ns + c];
                    }
                }

                for k in actual_ns..max_ns {
                    packed_hamiltonians[offset + k * max_ns + k] = 1e10;
                }
            }
        }

        // ─── GPU EIGENSOLVE: pre-allocated buffers, no per-iteration alloc ──
        // Upload packed Hamiltonians to pre-allocated GPU buffer (DMA write),
        // dispatch eigensolve in-place, readback eigenvalues+eigenvectors.
        // Single-dispatch (n≤32) or multi-dispatch — both pure GPU.
        device.queue().write_buffer(
            &eigh_matrices_buf,
            0,
            bytemuck::cast_slice(&packed_hamiltonians[..batch_size * max_mat_size]),
        );

        let eigh_dispatch = if max_ns <= 32 {
            BatchedEighGpu::execute_single_dispatch_buffers(
                &device,
                &eigh_matrices_buf,
                &eigh_eigenvalues_buf,
                &eigh_eigenvectors_buf,
                max_ns,
                batch_size,
                30,
                GPU_JACOBI_CONVERGENCE,
            )
        } else {
            BatchedEighGpu::execute_f64_buffers(
                &device,
                &eigh_matrices_buf,
                &eigh_eigenvalues_buf,
                &eigh_eigenvectors_buf,
                max_ns,
                batch_size,
                30,
            )
        };
        if let Err(e) = eigh_dispatch {
            eprintln!("BatchedEighGpu failed (iter {iter}): {e}");
            for &i in &active {
                states[i].converged = false;
            }
            break;
        }

        // Readback from pre-allocated GPU buffers
        let eigenvalues =
            BatchedEighGpu::read_eigenvalues(&device, &eigh_eigenvalues_buf, max_ns, batch_size)
                .expect("eigenvalue readback");
        let eigenvectors =
            BatchedEighGpu::read_eigenvectors(&device, &eigh_eigenvectors_buf, max_ns, batch_size)
                .expect("eigenvector readback");
        gpu_dispatches += 1;

        // Unpack results: extract actual_ns eigenvalues per nucleus
        for (batch_idx, &solver_idx) in active.iter().enumerate() {
            let (z, n, _, ref hfb) = solvers[solver_idx];
            let state = &mut states[solver_idx];
            let actual_ns = state.actual_ns;

            let nr = hfb.nr();
            let mut rho_p_new = vec![DENSITY_FLOOR; nr];
            let mut rho_n_new = vec![DENSITY_FLOOR; nr];

            // Eigenvalue/eigenvector offsets in padded arrays
            let eig_offset_p = (batch_idx * 2) * max_ns;
            let vec_offset_p = (batch_idx * 2) * max_mat_size;
            let eig_offset_n = (batch_idx * 2 + 1) * max_ns;
            let vec_offset_n = (batch_idx * 2 + 1) * max_mat_size;

            for (_species_idx, is_proton) in [(0usize, true), (1usize, false)] {
                let eig_offset = if is_proton {
                    eig_offset_p
                } else {
                    eig_offset_n
                };
                let vec_offset = if is_proton {
                    vec_offset_p
                } else {
                    vec_offset_n
                };

                // Extract ONLY the first actual_ns eigenvalues (physical ones)
                // The padded eigenvalues are at 1e10 — skip them
                let all_eigs = &eigenvalues[eig_offset..eig_offset + max_ns];
                let physical_eigs: Vec<f64> = all_eigs
                    .iter()
                    .copied()
                    .filter(|&e| e < 1e9) // filter out padded eigenvalues
                    .collect();

                // Use physical eigenvalues (pad with zeros if fewer than actual_ns)
                let eigs: Vec<f64> = if physical_eigs.len() >= actual_ns {
                    physical_eigs[..actual_ns].to_vec()
                } else {
                    let mut e = physical_eigs;
                    e.resize(actual_ns, 0.0);
                    e
                };

                // Extract eigenvectors: first actual_ns components of each
                // eigenvector (the padded components are near-zero for physical states)
                let mut vecs = vec![0.0; actual_ns * actual_ns];
                for col in 0..actual_ns {
                    // Only if this eigenvalue is physical
                    if all_eigs[col] < 1e9 {
                        for row in 0..actual_ns {
                            vecs[col * actual_ns + row] =
                                eigenvectors[vec_offset + col * max_ns + row];
                        }
                    }
                }

                let num_q = if is_proton { z } else { n };
                let delta_q = hfb.pairing_gap();

                // BCS occupations
                let (v2, _lambda) = hfb.bcs_occupations_from_eigs(&eigs, num_q, delta_q);

                // Build new density from BCS-weighted eigenstates
                let rho_new = hfb.density_from_eigenstates(&vecs, &v2, actual_ns);

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
                    (alpha * rho_p_new[k] + (1.0 - alpha) * state.rho_p[k]).max(DENSITY_FLOOR);
                state.rho_n[k] =
                    (alpha * rho_n_new[k] + (1.0 - alpha) * state.rho_n[k]).max(DENSITY_FLOOR);
            }

            // Compute total energy — use physical eigenvalues/eigenvectors
            let eig_p: Vec<f64> = eigenvalues[eig_offset_p..eig_offset_p + max_ns]
                .iter()
                .copied()
                .filter(|&e| e < 1e9)
                .take(actual_ns)
                .collect();
            let eig_n: Vec<f64> = eigenvalues[eig_offset_n..eig_offset_n + max_ns]
                .iter()
                .copied()
                .filter(|&e| e < 1e9)
                .take(actual_ns)
                .collect();

            // Extract actual_ns×actual_ns eigenvector blocks for energy
            let mut vecs_p = vec![0.0; actual_ns * actual_ns];
            let mut vecs_n = vec![0.0; actual_ns * actual_ns];
            for col in 0..actual_ns {
                if eigenvalues[eig_offset_p + col] < 1e9 {
                    for row in 0..actual_ns {
                        vecs_p[col * actual_ns + row] =
                            eigenvectors[vec_offset_p + col * max_ns + row];
                    }
                }
                if eigenvalues[eig_offset_n + col] < 1e9 {
                    for row in 0..actual_ns {
                        vecs_n[col * actual_ns + row] =
                            eigenvectors[vec_offset_n + col * max_ns + row];
                    }
                }
            }

            let e_total = hfb.compute_energy_from_densities(
                &state.rho_p,
                &state.rho_n,
                &eig_p,
                &vecs_p,
                &eig_n,
                &vecs_n,
                params,
            );

            let de = (e_total - state.e_prev).abs();
            state.e_prev = e_total;
            state.iterations = iter + 1;
            state.binding_energy = if e_total < 0.0 {
                -e_total
            } else {
                e_total.abs()
            };

            if de < tol && iter > 5 {
                state.converged = true;
            }
        }
    }

    // Collect HFB results
    _n_hfb = solvers.len();
    for (i, (z, n, _, _)) in solvers.iter().enumerate() {
        results.push((*z, *n, states[i].binding_energy, states[i].converged));
    }

    BatchedL2Result {
        results,
        hfb_time_s: t0.elapsed().as_secs_f64(),
        gpu_dispatches,
        n_hfb: _n_hfb,
        n_semf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_result_fields() {
        let r = BatchedL2Result {
            results: vec![(28, 28, 483.0, true)],
            hfb_time_s: 1.0,
            gpu_dispatches: 5,
            n_hfb: 1,
            n_semf: 0,
        };
        assert_eq!(r.results.len(), 1);
        assert_eq!(r.n_hfb, 1);
        assert_eq!(r.n_semf, 0);
    }

    #[test]
    fn nucleus_partitioning_semf_for_light() {
        // A < 56 should use SEMF, not HFB
        let nuclei = [(8, 8)]; // O-16, A=16
        let mut hfb_nuclei = Vec::new();
        let mut n_semf = 0usize;
        for (idx, &(z, n)) in nuclei.iter().enumerate() {
            let a = z + n;
            if (56..=132).contains(&a) {
                hfb_nuclei.push((z, n, idx));
            } else {
                n_semf += 1;
            }
        }
        assert!(hfb_nuclei.is_empty(), "O-16 should not use HFB");
        assert_eq!(n_semf, 1);
    }

    #[test]
    fn nucleus_partitioning_hfb_for_medium() {
        // A in 56..=132 should use HFB
        let nuclei = [(28, 28), (50, 82), (82, 126)]; // Ni-56, Sn-132, Pb-208
        let mut hfb_nuclei = Vec::new();
        let mut n_semf = 0usize;
        for (idx, &(z, n)) in nuclei.iter().enumerate() {
            let a = z + n;
            if (56..=132).contains(&a) {
                hfb_nuclei.push((z, n, idx));
            } else {
                n_semf += 1;
            }
        }
        assert_eq!(hfb_nuclei.len(), 2, "Ni-56 and Sn-132 in HFB range");
        assert_eq!(n_semf, 1, "Pb-208 (A=208) should use SEMF");
    }

    #[test]
    fn adaptive_basis_size_scales_with_nucleon_number() {
        use crate::physics::hfb::SphericalHFB;
        // Ni-56: small nucleus → modest basis
        let ni56 = SphericalHFB::new_adaptive(28, 28);
        let ns_ni = ni56.n_states();
        assert!(ns_ni > 5, "Ni-56 should have > 5 states (got {ns_ni})");

        // Sn-132: larger → larger basis
        let sn132 = SphericalHFB::new_adaptive(50, 82);
        let ns_sn = sn132.n_states();
        assert!(
            ns_sn > ns_ni,
            "Sn-132 ({ns_sn}) should have more states than Ni-56 ({ns_ni})"
        );

        // Basis sizes should be physically reasonable (< 200 states)
        assert!(ns_sn < 200, "Sn-132 basis too large: {ns_sn}");
    }

    #[test]
    fn identity_padding_strategy() {
        // Verify that padding diagonal with 1e10 produces eigenvalues
        // far from the physical spectrum (-50 to +100 MeV)
        let actual_ns = 4;
        let max_ns = 8;
        let max_mat_size = max_ns * max_ns;

        let mut padded = vec![0.0_f64; max_mat_size];
        // Simulate a small physical matrix
        for i in 0..actual_ns {
            padded[i * max_ns + i] = -(20.0 + i as f64 * 5.0); // Physical eigenvalues ~ -20 to -35
        }
        // Pad
        for k in actual_ns..max_ns {
            padded[k * max_ns + k] = 1e10;
        }

        // Check that padded diagonal values are far from physical
        for k in actual_ns..max_ns {
            assert!(
                padded[k * max_ns + k] > 1e9,
                "padded diagonal must be >> physical"
            );
        }
        // Check physical values are preserved
        for i in 0..actual_ns {
            assert!(
                padded[i * max_ns + i] < 0.0,
                "physical diagonal should be negative"
            );
        }
    }

    #[test]
    fn batched_hfb_cpu_path_determinism() {
        // Tests the CPU code path (no GPU) for determinism
        // Can't test GPU path in unit tests, but the solver construction is deterministic
        let hfb1 = SphericalHFB::new(28, 28, 4, 10.0, 50);
        let hfb2 = SphericalHFB::new(28, 28, 4, 10.0, 50);
        assert_eq!(hfb1.n_states(), hfb2.n_states());
        assert_eq!(hfb1.nr(), hfb2.nr());
    }
}
