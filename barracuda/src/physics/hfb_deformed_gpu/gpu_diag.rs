// SPDX-License-Identifier: AGPL-3.0-only

//! GPU batched eigensolve and BCS occupation computation for deformed HFB.

use super::types::NucleusSetup;
use crate::error::HotSpringError;
use crate::tolerances::{FERMI_SEARCH_MARGIN, GPU_JACOBI_CONVERGENCE, PAIRING_GAP_THRESHOLD};
use barracuda::device::WgpuDevice;
use barracuda::ops::linalg::BatchedEighGpu;
use rayon::prelude::*;
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
pub(super) fn diag_blocks_gpu(
    device: &Arc<WgpuDevice>,
    setup: &NucleusSetup,
    sorted_blocks: &[(i32, &Vec<usize>)],
    max_bs: usize,
    n_blocks: usize,
    v_potential: &[f64],
    wavefunctions: &[f64],
    n_particles: usize,
    delta_pair: f64,
    eigh_dispatches: &mut usize,
    total_gpu_dispatches: &mut usize,
    eigh_matrices_buf: &wgpu::Buffer,
    eigh_eigenvalues_buf: &wgpu::Buffer,
    eigh_eigenvectors_buf: &wgpu::Buffer,
) -> Result<(Vec<f64>, Vec<f64>), HotSpringError> {
    let n_states = setup.states.len();
    let n_grid = setup.n_grid;

    if max_bs < 2 || n_blocks == 0 {
        return Ok((vec![0.0; n_states], vec![0.0; n_states]));
    }

    let mat_size = max_bs * max_bs;

    let block_h_matrices: Vec<Vec<f64>> = sorted_blocks
        .par_iter()
        .map(|(_, block_indices)| {
            let bs = block_indices.len();
            let mut h = vec![0.0f64; max_bs * max_bs];

            for bi in 0..bs {
                let i = block_indices[bi];
                let si = &setup.states[i];

                let t_i = setup.hw_z.mul_add(
                    f64::from(si.n_z) + 0.5,
                    setup.hw_perp
                        * (2.0_f64.mul_add(f64::from(si.n_perp), f64::from(si.abs_lambda)) + 1.0),
                );
                h[bi * max_bs + bi] = t_i;

                for bj in bi..bs {
                    let j = block_indices[bj];

                    let mut v_ij = 0.0;
                    let base_i = i * n_grid;
                    let base_j = j * n_grid;
                    for i_rho in 0..setup.n_rho {
                        let dv = setup.volume_element(i_rho);
                        let row_start = i_rho * setup.n_z;
                        for i_z in 0..setup.n_z {
                            let k = row_start + i_z;
                            v_ij += wavefunctions[base_i + k]
                                * v_potential[k]
                                * wavefunctions[base_j + k]
                                * dv;
                        }
                    }

                    h[bi * max_bs + bj] += v_ij;
                    if bi != bj {
                        h[bj * max_bs + bi] += v_ij;
                    }
                }
            }
            for p in bs..max_bs {
                h[p * max_bs + p] = 1e6;
            }
            h
        })
        .collect();

    let mut packed_h = vec![0.0f64; n_blocks * mat_size];
    for (i, h) in block_h_matrices.into_iter().enumerate() {
        packed_h[i * mat_size..(i + 1) * mat_size].copy_from_slice(&h);
    }

    device
        .queue()
        .write_buffer(eigh_matrices_buf, 0, bytemuck::cast_slice(&packed_h));

    let dispatch_result = if max_bs <= 32 {
        BatchedEighGpu::execute_single_dispatch_buffers(
            &Arc::clone(device),
            eigh_matrices_buf,
            eigh_eigenvalues_buf,
            eigh_eigenvectors_buf,
            max_bs,
            n_blocks,
            50,
            GPU_JACOBI_CONVERGENCE,
        )
        .or_else(|_| {
            BatchedEighGpu::execute_f64_buffers(
                &Arc::clone(device),
                eigh_matrices_buf,
                eigh_eigenvalues_buf,
                eigh_eigenvectors_buf,
                max_bs,
                n_blocks,
                50,
            )
        })
    } else {
        BatchedEighGpu::execute_f64_buffers(
            &Arc::clone(device),
            eigh_matrices_buf,
            eigh_eigenvalues_buf,
            eigh_eigenvectors_buf,
            max_bs,
            n_blocks,
            50,
        )
    };
    dispatch_result
        .map_err(|e| HotSpringError::GpuCompute(format!("GPU eigensolve failed: {e}")))?;

    let eigenvalues_flat = BatchedEighGpu::read_eigenvalues(
        &Arc::clone(device),
        eigh_eigenvalues_buf,
        max_bs,
        n_blocks,
    )
    .map_err(|e| HotSpringError::GpuCompute(format!("eigenvalue readback: {e}")))?;
    *eigh_dispatches += 1;
    *total_gpu_dispatches += 1;

    let mut all_eigenvalues = vec![0.0; n_states];
    let mut all_occupations = vec![0.0; n_states];
    let mut block_eigs: Vec<(usize, f64)> = Vec::new();

    for (blk_idx, (_, indices)) in sorted_blocks.iter().enumerate() {
        for (bi, &gi) in indices.iter().enumerate() {
            let eval = eigenvalues_flat[blk_idx * max_bs + bi];
            all_eigenvalues[gi] = eval;
            block_eigs.push((gi, eval));
        }
    }

    block_eigs.sort_by(|a, b| a.1.total_cmp(&b.1));
    bcs_occupations(&block_eigs, n_particles, delta_pair, &mut all_occupations);

    Ok((all_eigenvalues, all_occupations))
}

pub(super) fn bcs_occupations(
    sorted_eigs: &[(usize, f64)],
    n_particles: usize,
    delta: f64,
    occ: &mut [f64],
) {
    if sorted_eigs.is_empty() {
        return;
    }
    if delta > PAIRING_GAP_THRESHOLD {
        let fermi = find_fermi_bcs(sorted_eigs, n_particles, delta);
        for &(si, eval) in sorted_eigs {
            let eps = eval - fermi;
            let e_qp = eps.hypot(delta);
            occ[si] = (0.5 * (1.0 - eps / e_qp)).clamp(0.0, 1.0);
        }
    } else {
        let mut left = n_particles as f64;
        for &(si, _) in sorted_eigs {
            if left >= 2.0 {
                occ[si] = 1.0;
                left -= 2.0;
            } else if left > 0.0 {
                occ[si] = left / 2.0;
                left = 0.0;
            }
        }
    }
}

pub(super) fn find_fermi_bcs(sorted_eigs: &[(usize, f64)], n_particles: usize, delta: f64) -> f64 {
    if sorted_eigs.is_empty() {
        return 0.0;
    }
    let n_t = n_particles as f64;
    let pn = |mu: f64| -> f64 {
        sorted_eigs
            .iter()
            .map(|&(_, e)| {
                let eps = e - mu;
                2.0 * 0.5 * (1.0 - eps / eps.hypot(delta))
            })
            .sum()
    };
    let (mut lo, mut hi) = (
        sorted_eigs[0].1 - FERMI_SEARCH_MARGIN,
        sorted_eigs[sorted_eigs.len() - 1].1 + FERMI_SEARCH_MARGIN,
    );
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if pn(mid) < n_t {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < PAIRING_GAP_THRESHOLD {
            break;
        }
    }
    0.5 * (lo + hi)
}
