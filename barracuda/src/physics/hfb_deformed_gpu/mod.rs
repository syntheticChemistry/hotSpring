// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-Accelerated Deformed HFB Solver (Level 3)
//!
//! Architecture: CPU (Rayon) builds block Hamiltonians + physics,
//! GPU (BatchedEighGpu) does ALL eigensolves in one batched dispatch.
//!
//! Pipeline per SCF iteration:
//!   CPU (Rayon, 24 threads): tau, J, Coulomb, potentials, H build, density, energy
//!   GPU (1 dispatch): batched eigensolve across all Omega blocks
//!   CPU (single): BCS bisection, convergence check, orchestration
//!
//! This combines the parallelism of Rayon for grid-based operations with
//! the massive throughput of GPU eigensolvers. The eigensolve (O(N³) per block,
//! ~10 blocks per nucleus, 200 iterations) is the ideal GPU target because
//! it's dense linear algebra with high arithmetic intensity.

mod gpu_diag;
mod physics;
pub(crate) mod types;

#[cfg(test)]
mod tests;

use super::hfb_deformed::DeformedHFBResult;
use super::hfb_deformed_common::{beta2_from_q20, rms_radius};
use crate::error::HotSpringError;
use crate::gpu::GpuF64;
use crate::tolerances::{BROYDEN_WARMUP, HFB_MAX_ITER, SCF_ENERGY_TOLERANCE};
use barracuda::device::WgpuDevice;
use barracuda::ops::linalg::BatchedEighGpu;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub use types::GpuResidentL3Result;

// EVOLUTION(GPU): will be used when deformed_*.wgsl shaders are wired for full GPU-resident pipeline
#[allow(dead_code)]
fn create_f64_storage_buf(device: &WgpuDevice, label: &str, data: &[f64]) -> wgpu::Buffer {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    device.create_storage_buffer(label, &bytes)
}

// EVOLUTION(GPU): will be used when deformed_*.wgsl shaders are wired for full GPU-resident pipeline
#[allow(dead_code)]
fn read_f64_from_gpu(device: &WgpuDevice, buf: &wgpu::Buffer, count: usize) -> Vec<f64> {
    device
        .read_buffer_f64(buf, count)
        .unwrap_or_else(|_| vec![0.0; count])
}

/// Batch compute L3 binding energies on GPU with deformed HFB.
///
/// # Errors
///
/// Returns [`HotSpringError::GpuCompute`] or [`HotSpringError::Barracuda`]
/// if GPU eigensolve or SCF iteration fails.
pub fn binding_energies_l3_gpu(
    device: &Arc<WgpuDevice>,
    nuclei: &[(usize, usize)],
    params: &[f64],
) -> Result<GpuResidentL3Result, HotSpringError> {
    let t0 = Instant::now();
    println!("    GPU: BatchedEighGpu for eigensolves, Rayon parallel across nuclei");
    println!(
        "    Processing {} nuclei on {} Rayon threads + GPU eigensolve",
        nuclei.len(),
        rayon::current_num_threads()
    );

    let eigh_count = AtomicUsize::new(0);
    let dispatch_count = AtomicUsize::new(0);
    let done_count = AtomicUsize::new(0);
    let total = nuclei.len();

    let results: Vec<(usize, usize, f64, bool, f64)> = nuclei
        .par_iter()
        .map(|&(z, n)| -> Result<(usize, usize, f64, bool, f64), HotSpringError> {
            let mut eigh_d = 0usize;
            let mut total_d = 0usize;
            let result = deformed_hfb_gpu_single(
                device,
                z,
                n,
                params,
                &mut eigh_d,
                &mut total_d,
            )?;
            eigh_count.fetch_add(eigh_d, Ordering::Relaxed);
            dispatch_count.fetch_add(total_d, Ordering::Relaxed);
            let idx = done_count.fetch_add(1, Ordering::Relaxed) + 1;
            let a = z + n;
            let status = if result.converged { "conv" } else { "NOCONV" };
            if idx <= 5 || idx.is_multiple_of(10) || idx == total || !result.converged {
                println!(
                    "    [{:>4}/{:>4}] Z={:>3} N={:>3} A={:>3} | BE={:>10.3} MeV β₂={:>6.3} {} iter={} {:.1}s elapsed",
                    idx, total, z, n, a,
                    result.binding_energy_mev,
                    result.beta2,
                    status,
                    result.iterations,
                    t0.elapsed().as_secs_f64()
                );
            }
            Ok((z, n, result.binding_energy_mev, result.converged, result.beta2))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(GpuResidentL3Result {
        n_nuclei: nuclei.len(),
        results,
        wall_time_s: t0.elapsed().as_secs_f64(),
        eigh_dispatches: eigh_count.load(Ordering::Relaxed),
        total_gpu_dispatches: dispatch_count.load(Ordering::Relaxed),
    })
}

/// Same as [`binding_energies_l3_gpu`] but auto-initializes the GPU device.
///
/// # Errors
///
/// Returns [`HotSpringError::GpuCompute`] if GPU init or runtime creation fails.
/// Returns [`HotSpringError::Barracuda`] if SCF iteration fails.
pub fn binding_energies_l3_gpu_auto(
    nuclei: &[(usize, usize)],
    params: &[f64],
) -> Result<GpuResidentL3Result, HotSpringError> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| HotSpringError::GpuCompute(format!("tokio: failed to create runtime: {e}")))?;
    let gpu = rt
        .block_on(GpuF64::new())
        .map_err(|e| HotSpringError::GpuCompute(format!("GPU init failed: {e}")))?;
    gpu.print_info();
    let device = gpu.to_wgpu_device();
    binding_energies_l3_gpu(&device, nuclei, params)
}

fn deformed_hfb_gpu_single(
    device: &Arc<WgpuDevice>,
    z: usize,
    n: usize,
    params: &[f64],
    eigh_dispatches: &mut usize,
    total_gpu_dispatches: &mut usize,
) -> Result<DeformedHFBResult, HotSpringError> {
    let setup = types::NucleusSetup::new(z, n);
    let n_grid = setup.n_grid;
    let n_states = setup.states.len();
    let max_iter = HFB_MAX_ITER;
    let tol = SCF_ENERGY_TOLERANCE;
    let broyden_warmup = BROYDEN_WARMUP;

    let wavefunctions = physics::precompute_wavefunctions(&setup);

    let mut sorted_blocks: Vec<(i32, &Vec<usize>)> =
        setup.omega_blocks.iter().map(|(k, v)| (*k, v)).collect();
    sorted_blocks.sort_by_key(|&(k, _)| k);
    let n_blocks = sorted_blocks.len();
    let max_bs = sorted_blocks
        .iter()
        .map(|(_, v)| v.len())
        .max()
        .unwrap_or(1);

    let (eigh_matrices_buf, eigh_eigenvalues_buf, eigh_eigenvectors_buf) =
        BatchedEighGpu::create_buffers(device, max_bs, n_blocks).map_err(|e| {
            HotSpringError::GpuCompute(format!("pre-allocate eigensolve GPU buffers: {e}"))
        })?;

    let mut rho_p = vec![0.0f64; n_grid];
    let mut rho_n = vec![0.0f64; n_grid];
    let mut e_prev = 0.0f64;
    let mut converged = false;
    let mut binding_energy = 0.0f64;
    let mut delta_e = 0.0f64;
    let mut iter = 0usize;
    let mut prev_occ_p = vec![0.0f64; n_states];
    let mut prev_occ_n = vec![0.0f64; n_states];
    let mut v_coulomb = vec![0.0f64; n_grid];

    let vec_dim = 2 * n_grid;
    let mut broyden_dfs: Vec<Vec<f64>> = Vec::new();
    let mut broyden_dus: Vec<Vec<f64>> = Vec::new();
    let mut prev_residual: Option<Vec<f64>> = None;
    let mut prev_input: Option<Vec<f64>> = None;

    for iteration in 0..max_iter {
        iter = iteration + 1;

        let rho_total: Vec<f64> = rho_p.iter().zip(&rho_n).map(|(&p, &nv)| p + nv).collect();

        let (tau_p, tau_n) =
            physics::compute_tau_rayon(&setup, &wavefunctions, &prev_occ_p, &prev_occ_n);
        let (_j_p, _j_n) =
            physics::compute_spin_current(&setup, &wavefunctions, &prev_occ_p, &prev_occ_n);
        physics::compute_coulomb_cpu(&setup, &rho_p, &mut v_coulomb);
        let d_rho_dr = physics::density_radial_derivative(&setup, &rho_total);

        let v_p = physics::mean_field_potential(
            &setup, params, &rho_p, &rho_n, &rho_total, true, &tau_p, &tau_n, &v_coulomb, &d_rho_dr,
        );
        let v_n = physics::mean_field_potential(
            &setup, params, &rho_p, &rho_n, &rho_total, false, &tau_p, &tau_n, &v_coulomb,
            &d_rho_dr,
        );

        let (_eigs_p, occ_p) = gpu_diag::diag_blocks_gpu(
            device,
            &setup,
            &sorted_blocks,
            max_bs,
            n_blocks,
            &v_p,
            &wavefunctions,
            setup.z,
            setup.delta_p,
            eigh_dispatches,
            total_gpu_dispatches,
            &eigh_matrices_buf,
            &eigh_eigenvalues_buf,
            &eigh_eigenvectors_buf,
        )?;
        let (_eigs_n, occ_n) = gpu_diag::diag_blocks_gpu(
            device,
            &setup,
            &sorted_blocks,
            max_bs,
            n_blocks,
            &v_n,
            &wavefunctions,
            setup.n_neutrons,
            setup.delta_n,
            eigh_dispatches,
            total_gpu_dispatches,
            &eigh_matrices_buf,
            &eigh_eigenvalues_buf,
            &eigh_eigenvectors_buf,
        )?;

        prev_occ_p.clone_from(&occ_p);
        prev_occ_n.clone_from(&occ_n);

        let (new_rho_p, new_rho_n) =
            physics::compute_densities_rayon(&setup, &wavefunctions, &occ_p, &occ_n);

        physics::density_mixing(
            &mut rho_p,
            &mut rho_n,
            &new_rho_p,
            &new_rho_n,
            iteration,
            broyden_warmup,
            n_grid,
            vec_dim,
            &mut broyden_dfs,
            &mut broyden_dus,
            &mut prev_residual,
            &mut prev_input,
        );

        binding_energy = physics::total_energy(&setup, params, &rho_p, &rho_n, &occ_p, &occ_n);

        if !binding_energy.is_finite() || binding_energy.abs() > 1e10 {
            break;
        }
        delta_e = (binding_energy - e_prev).abs();
        if iteration > broyden_warmup && delta_e < tol {
            converged = true;
            break;
        }
        e_prev = binding_energy;
    }

    let rho_total: Vec<f64> = rho_p.iter().zip(&rho_n).map(|(&p, &nv)| p + nv).collect();
    let q20 = physics::quadrupole(&setup, &rho_total);
    let beta2 = beta2_from_q20(setup.a, q20);
    let rms_r = rms_radius(
        &rho_total,
        setup.n_rho,
        setup.n_z,
        setup.d_rho,
        setup.d_z,
        setup.z_min,
    );

    Ok(DeformedHFBResult {
        binding_energy_mev: binding_energy,
        converged,
        iterations: iter,
        delta_e,
        beta2,
        q20_fm2: q20,
        rms_radius_fm: rms_r,
    })
}

pub const fn estimate_gpu_dispatches(n_nuclei: usize, avg_blocks: usize, max_iter: usize) -> usize {
    let per_iter = 4 + avg_blocks;
    n_nuclei * (1 + max_iter * per_iter)
}
