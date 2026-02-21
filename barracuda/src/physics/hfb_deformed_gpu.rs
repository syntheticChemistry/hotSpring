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

use super::constants::{E2, HBAR_C, M_NUCLEON};
use super::hfb_common::{factorial_f64, hermite_value};
use super::hfb_deformed::DeformedHFBResult;
use crate::gpu::GpuF64;
use crate::tolerances::{
    DEFORMATION_GUESS_GENERIC, DEFORMATION_GUESS_SD, DEFORMATION_GUESS_WEAK,
    DEFORMED_COULOMB_R_MIN, DENSITY_FLOOR, DIVISION_GUARD, GPU_JACOBI_CONVERGENCE,
    PAIRING_GAP_THRESHOLD, SCF_ENERGY_TOLERANCE, SPIN_ORBIT_R_MIN,
};
use barracuda::device::WgpuDevice;
use barracuda::ops::linalg::BatchedEighGpu;
use barracuda::special::{gamma, laguerre};
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

/// Result from GPU-resident L3 evaluation
#[derive(Debug)]
pub struct GpuResidentL3Result {
    pub results: Vec<(usize, usize, f64, bool, f64)>,
    pub wall_time_s: f64,
    pub eigh_dispatches: usize,
    pub total_gpu_dispatches: usize,
    pub n_nuclei: usize,
}

// ═══════════════════════════════════════════════════════════════════
// GPU Uniform Params — matches WGSL HamiltonianParams struct layout
// ═══════════════════════════════════════════════════════════════════

/// Matches the WGSL `HamiltonianParams` struct byte-for-byte.
/// Layout: 4×u32 (16 bytes) + 2×f64 (16 bytes) = 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HamiltonianParamsGpu {
    n_rho: u32,
    n_z: u32,
    block_size: u32, // max block size (for index array stride)
    n_blocks: u32,
    d_rho_lo: u32, // f64 as two u32 (WGSL f64 = 8 bytes)
    d_rho_hi: u32,
    d_z_lo: u32,
    d_z_hi: u32,
}

impl HamiltonianParamsGpu {
    #[allow(dead_code)] // EVOLUTION(GPU): used in test_params_gpu_layout; will wire to deformed_*.wgsl when GPU pipeline is complete
    const fn new(
        n_rho: u32,
        n_z: u32,
        block_size: u32,
        n_blocks: u32,
        d_rho: f64,
        d_z: f64,
    ) -> Self {
        let dr = d_rho.to_bits();
        let dz = d_z.to_bits();
        Self {
            n_rho,
            n_z,
            block_size,
            n_blocks,
            d_rho_lo: dr as u32,
            d_rho_hi: (dr >> 32) as u32,
            d_z_lo: dz as u32,
            d_z_hi: (dz >> 32) as u32,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Basis + Grid Setup
// ═══════════════════════════════════════════════════════════════════

struct BasisState {
    n_z: u32,
    n_perp: u32,
    abs_lambda: u32,
    lambda: i32,
    sigma: i32,
    omega_x2: i32,
    #[allow(dead_code)]
    // EVOLUTION(GPU): will be used when deformed_*.wgsl shaders need shell truncation
    _n_shell: u32,
}

struct NucleusSetup {
    z: usize,
    n_neutrons: usize,
    a: usize,
    n_rho: usize,
    n_z: usize,
    n_grid: usize,
    d_rho: f64,
    d_z: f64,
    z_min: f64,
    #[allow(dead_code)]
    // EVOLUTION(GPU): will be used when deformed_*.wgsl shaders wire grid bounds
    _rho_max: f64,
    hw_z: f64,
    hw_perp: f64,
    b_z: f64,
    b_perp: f64,
    delta_p: f64,
    delta_n: f64,
    states: Vec<BasisState>,
    omega_blocks: HashMap<i32, Vec<usize>>,
}

impl NucleusSetup {
    fn new(z: usize, n: usize) -> Self {
        let a = z + n;
        let a_f = a as f64;
        let hw0 = 41.0 * a_f.powf(-1.0 / 3.0);
        let beta2_init = Self::deformation_guess(z, n);
        let hw_z = hw0 * (1.0 - 2.0 * beta2_init / 3.0);
        let hw_perp = hw0 * (1.0 + beta2_init / 3.0);
        let b_z = HBAR_C / (M_NUCLEON * hw_z).sqrt();
        let b_perp = HBAR_C / (M_NUCLEON * hw_perp).sqrt();
        let r0 = 1.2 * a_f.powf(1.0 / 3.0);
        let rho_max = (r0 + 8.0).max(12.0);
        let z_max = (r0 * (1.0 + beta2_init.abs()) + 8.0).max(14.0);
        let n_rho = ((rho_max * 8.0) as usize).max(60);
        let n_z_val = ((2.0 * z_max * 8.0) as usize).max(80);
        let d_rho = rho_max / n_rho as f64;
        let d_z = 2.0 * z_max / n_z_val as f64;
        let delta = 12.0 / a_f.max(4.0).sqrt();
        let n_shells = ((2.0 * a_f.powf(1.0 / 3.0)) as usize + 5).clamp(10, 16);

        let mut setup = NucleusSetup {
            z,
            n_neutrons: n,
            a,
            n_rho,
            n_z: n_z_val,
            n_grid: n_rho * n_z_val,
            d_rho,
            d_z,
            z_min: -z_max,
            _rho_max: rho_max,
            hw_z,
            hw_perp,
            b_z,
            b_perp,
            delta_p: delta,
            delta_n: delta,
            states: Vec::new(),
            omega_blocks: HashMap::new(),
        };
        setup.build_basis(n_shells);
        setup
    }

    fn deformation_guess(z: usize, n: usize) -> f64 {
        let a = z + n;
        let magic = [2, 8, 20, 28, 50, 82, 126];
        let z_m = magic.iter().any(|&m| (z as i32 - m).unsigned_abs() <= 2);
        let n_m = magic.iter().any(|&m| (n as i32 - m).unsigned_abs() <= 2);
        if z_m && n_m {
            0.0
        } else if a > 222 {
            0.25
        } else if a > 150 && a < 190 {
            0.28
        } else if a > 20 && a < 28 {
            DEFORMATION_GUESS_SD
        } else if z_m || n_m {
            DEFORMATION_GUESS_WEAK
        } else {
            DEFORMATION_GUESS_GENERIC
        }
    }

    fn build_basis(&mut self, n_shells: usize) {
        for n_sh in 0..n_shells {
            for n_z_v in 0..=n_sh {
                let rem = n_sh - n_z_v;
                for n_perp in 0..=(rem / 2) {
                    let abs_l = rem - 2 * n_perp;
                    let lams = if abs_l == 0 {
                        vec![0i32]
                    } else {
                        vec![abs_l as i32]
                    };
                    for &lam in &lams {
                        for &sig in &[1i32, -1i32] {
                            let omega_x2 = 2 * lam + sig;
                            if omega_x2 <= 0 {
                                continue;
                            }
                            self.states.push(BasisState {
                                n_z: n_z_v as u32,
                                n_perp: n_perp as u32,
                                abs_lambda: abs_l as u32,
                                lambda: lam,
                                sigma: sig,
                                omega_x2,
                                _n_shell: n_sh as u32,
                            });
                        }
                    }
                }
            }
        }
        for (i, s) in self.states.iter().enumerate() {
            self.omega_blocks.entry(s.omega_x2).or_default().push(i);
        }
    }

    fn volume_element(&self, i_rho: usize) -> f64 {
        let rho = (i_rho + 1) as f64 * self.d_rho;
        2.0 * PI * rho * self.d_rho * self.d_z
    }

    const fn grid_idx(&self, i_rho: usize, i_z: usize) -> usize {
        i_rho * self.n_z + i_z
    }
}

// GpuPipelines removed — Hamiltonian now built on CPU with Rayon.
// GPU pipelines for H build retained in WGSL shaders for future full-GPU-resident version.
// Current architecture: CPU(Rayon) H build → GPU(BatchedEighGpu) eigensolve.

// ═══════════════════════════════════════════════════════════════════
// GPU buffer helpers
// ═══════════════════════════════════════════════════════════════════

// EVOLUTION(GPU): will be used when deformed_*.wgsl shaders are wired for full GPU-resident pipeline
#[allow(dead_code)] // EVOLUTION: deferred until deformed_*.wgsl pipeline wired
fn create_f64_storage_buf(device: &WgpuDevice, label: &str, data: &[f64]) -> wgpu::Buffer {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    device.create_storage_buffer(label, &bytes)
}

// EVOLUTION(GPU): will be used when deformed_*.wgsl shaders are wired for full GPU-resident pipeline
#[allow(dead_code)] // EVOLUTION: deferred until deformed_*.wgsl pipeline wired
fn read_f64_from_gpu(device: &WgpuDevice, buf: &wgpu::Buffer, count: usize) -> Vec<f64> {
    device
        .read_buffer_f64(buf, count)
        .unwrap_or_else(|_| vec![0.0; count])
}

// ═══════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════

pub fn binding_energies_l3_gpu(
    device: Arc<WgpuDevice>,
    nuclei: &[(usize, usize)],
    params: &[f64],
) -> GpuResidentL3Result {
    let t0 = Instant::now();
    println!("    GPU: BatchedEighGpu for eigensolves, Rayon parallel across nuclei");
    println!(
        "    Processing {} nuclei on {} Rayon threads + GPU eigensolve",
        nuclei.len(),
        rayon::current_num_threads()
    );

    // Process all nuclei in parallel using Rayon
    // Each nucleus builds H on its own CPU thread, then uses shared GPU for eigensolve
    use std::sync::atomic::{AtomicUsize, Ordering};
    let eigh_count = AtomicUsize::new(0);
    let dispatch_count = AtomicUsize::new(0);
    let done_count = AtomicUsize::new(0);
    let total = nuclei.len();

    let results: Vec<(usize, usize, f64, bool, f64)> = nuclei.par_iter().map(|&(z, n)| {
        let mut eigh_d = 0usize;
        let mut total_d = 0usize;
        let result = deformed_hfb_gpu_single(
            &device, z, n, params,
            &mut eigh_d, &mut total_d,
        );
        eigh_count.fetch_add(eigh_d, Ordering::Relaxed);
        dispatch_count.fetch_add(total_d, Ordering::Relaxed);
        let idx = done_count.fetch_add(1, Ordering::Relaxed) + 1;
        let a = z + n;
        let status = if result.converged { "conv" } else { "NOCONV" };
        if idx <= 5 || idx.is_multiple_of(10) || idx == total || !result.converged {
            println!("    [{:>4}/{:>4}] Z={:>3} N={:>3} A={:>3} | BE={:>10.3} MeV β₂={:>6.3} {} iter={} {:.1}s elapsed",
                idx, total, z, n, a,
                result.binding_energy_mev, result.beta2, status,
                result.iterations, t0.elapsed().as_secs_f64());
        }
        (z, n, result.binding_energy_mev, result.converged, result.beta2)
    }).collect();

    GpuResidentL3Result {
        n_nuclei: nuclei.len(),
        results,
        wall_time_s: t0.elapsed().as_secs_f64(),
        eigh_dispatches: eigh_count.load(Ordering::Relaxed),
        total_gpu_dispatches: dispatch_count.load(Ordering::Relaxed),
    }
}

pub fn binding_energies_l3_gpu_auto(
    nuclei: &[(usize, usize)],
    params: &[f64],
) -> GpuResidentL3Result {
    let rt = tokio::runtime::Runtime::new()
        .expect("tokio: failed to create runtime for GPU initialization");
    let gpu = rt
        .block_on(GpuF64::new())
        .expect("GPU init failed — pure GPU pipeline requires GPU adapter");
    gpu.print_info();
    let device = gpu.to_wgpu_device();
    binding_energies_l3_gpu(device, nuclei, params)
}

// ═══════════════════════════════════════════════════════════════════
// GPU-Resident SCF for a Single Nucleus
// ═══════════════════════════════════════════════════════════════════

fn deformed_hfb_gpu_single(
    device: &Arc<WgpuDevice>,
    z: usize,
    n: usize,
    params: &[f64],
    eigh_dispatches: &mut usize,
    total_gpu_dispatches: &mut usize,
) -> DeformedHFBResult {
    let setup = NucleusSetup::new(z, n);
    let n_grid = setup.n_grid;
    let n_states = setup.states.len();
    let max_iter = 200;
    let tol = SCF_ENERGY_TOLERANCE;
    let broyden_warmup = 50;

    // ── Step 1: Precompute wavefunctions (CPU Rayon) ──
    let wavefunctions = precompute_wavefunctions(&setup);

    // Precompute block structure
    let mut sorted_blocks: Vec<(i32, &Vec<usize>)> =
        setup.omega_blocks.iter().map(|(k, v)| (*k, v)).collect();
    sorted_blocks.sort_by_key(|&(k, _)| k);
    let n_blocks = sorted_blocks.len();
    let max_bs = sorted_blocks
        .iter()
        .map(|(_, v)| v.len())
        .max()
        .unwrap_or(1);

    // Pre-allocate eigensolve GPU buffers — reused every SCF iteration
    let (eigh_matrices_buf, eigh_eigenvalues_buf, eigh_eigenvectors_buf) =
        BatchedEighGpu::create_buffers(device, max_bs, n_blocks)
            .expect("pre-allocate eigensolve GPU buffers");

    // ── Step 2: SCF loop ──
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

    // Broyden mixing state
    let vec_dim = 2 * n_grid;
    let mut broyden_dfs: Vec<Vec<f64>> = Vec::new();
    let mut broyden_dus: Vec<Vec<f64>> = Vec::new();
    let mut prev_residual: Option<Vec<f64>> = None;
    let mut prev_input: Option<Vec<f64>> = None;

    for iteration in 0..max_iter {
        iter = iteration + 1;

        // ── CPU (Rayon): tau, spin-current, Coulomb, potentials ──
        let rho_total: Vec<f64> = rho_p.iter().zip(&rho_n).map(|(&p, &nv)| p + nv).collect();

        let (tau_p, tau_n) = compute_tau_rayon(&setup, &wavefunctions, &prev_occ_p, &prev_occ_n);
        let (_j_p, _j_n) = compute_spin_current(&setup, &wavefunctions, &prev_occ_p, &prev_occ_n);
        compute_coulomb_cpu(&setup, &rho_p, &mut v_coulomb);
        let d_rho_dr = density_radial_derivative(&setup, &rho_total);

        let v_p = mean_field_potential(
            &setup, params, &rho_p, &rho_n, &rho_total, true, &tau_p, &tau_n, &v_coulomb, &d_rho_dr,
        );
        let v_n = mean_field_potential(
            &setup, params, &rho_p, &rho_n, &rho_total, false, &tau_p, &tau_n, &v_coulomb,
            &d_rho_dr,
        );

        // ── CPU (Rayon) H build + GPU batched eigensolve (pre-allocated) ──
        let (_eigs_p, occ_p) = diag_blocks_gpu(
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
        );
        let (_eigs_n, occ_n) = diag_blocks_gpu(
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
        );

        prev_occ_p.clone_from(&occ_p);
        prev_occ_n.clone_from(&occ_n);

        // ── CPU (Rayon): density accumulation ──
        let (new_rho_p, new_rho_n) =
            compute_densities_rayon(&setup, &wavefunctions, &occ_p, &occ_n);

        // ── Density mixing (Broyden after warmup) ──
        density_mixing(
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

        // ── Energy functional ──
        binding_energy = total_energy(&setup, params, &rho_p, &rho_n, &occ_p, &occ_n);

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
    let q20 = quadrupole(&setup, &rho_total);
    let beta2 = beta2_from_q20(setup.a, q20);
    let rms_r = rms_radius(&setup, &rho_total);

    DeformedHFBResult {
        binding_energy_mev: binding_energy,
        converged,
        iterations: iter,
        delta_e,
        beta2,
        q20_fm2: q20,
        rms_radius_fm: rms_r,
    }
}

// ═══════════════════════════════════════════════════════════════════
// CPU (Rayon) H Build + GPU Batched Eigensolve
//
// Strategy: Build H on CPU with Rayon (24 threads, fast), then
// batch ALL omega-block eigensolves into ONE GPU dispatch via
// BatchedEighGpu. This eliminates per-iteration GPU dispatch
// overhead while keeping the eigensolve on GPU.
// ═══════════════════════════════════════════════════════════════════

fn diag_blocks_gpu(
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
) -> (Vec<f64>, Vec<f64>) {
    let n_states = setup.states.len();
    let n_grid = setup.n_grid;

    if max_bs < 2 || n_blocks == 0 {
        return (vec![0.0; n_states], vec![0.0; n_states]);
    }

    let mat_size = max_bs * max_bs;

    // ── CPU (Rayon): Build block Hamiltonians in parallel ──
    // Each block is independent → perfect Rayon parallelism
    let block_h_matrices: Vec<Vec<f64>> = sorted_blocks
        .par_iter()
        .map(|(_, block_indices)| {
            let bs = block_indices.len();
            let mut h = vec![0.0f64; max_bs * max_bs];

            for bi in 0..bs {
                let i = block_indices[bi];
                let si = &setup.states[i];

                // Kinetic diagonal
                let t_i = setup.hw_z * (f64::from(si.n_z) + 0.5)
                    + setup.hw_perp * (2.0 * f64::from(si.n_perp) + f64::from(si.abs_lambda) + 1.0);
                h[bi * max_bs + bi] = t_i;

                for bj in bi..bs {
                    let j = block_indices[bj];

                    // Potential matrix element: <ψ_i|V|ψ_j> = Σ_grid ψ_i·V·ψ_j·dV
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
            // Padding: large diagonal so padded eigenvalues sort high
            for p in bs..max_bs {
                h[p * max_bs + p] = 1e6;
            }
            h
        })
        .collect();

    // Pack into contiguous array for BatchedEighGpu
    let mut packed_h = vec![0.0f64; n_blocks * mat_size];
    for (i, h) in block_h_matrices.into_iter().enumerate() {
        packed_h[i * mat_size..(i + 1) * mat_size].copy_from_slice(&h);
    }

    // ── GPU: Batched eigensolve — pre-allocated buffers, no per-call alloc ──
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
    dispatch_result.expect("GPU eigensolve failed — pure GPU pipeline");

    let eigenvalues_flat = BatchedEighGpu::read_eigenvalues(
        &Arc::clone(device),
        eigh_eigenvalues_buf,
        max_bs,
        n_blocks,
    )
    .expect("eigenvalue readback");
    *eigh_dispatches += 1;
    *total_gpu_dispatches += 1;

    // Unpack eigenvalues + BCS occupations
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

    (all_eigenvalues, all_occupations)
}

fn bcs_occupations(sorted_eigs: &[(usize, f64)], n_particles: usize, delta: f64, occ: &mut [f64]) {
    if sorted_eigs.is_empty() {
        return;
    }
    if delta > PAIRING_GAP_THRESHOLD {
        let fermi = find_fermi_bcs(sorted_eigs, n_particles, delta);
        for &(si, eval) in sorted_eigs {
            let eps = eval - fermi;
            let e_qp = (eps * eps + delta * delta).sqrt();
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

fn find_fermi_bcs(sorted_eigs: &[(usize, f64)], n_particles: usize, delta: f64) -> f64 {
    if sorted_eigs.is_empty() {
        return 0.0;
    }
    let n_t = n_particles as f64;
    let pn = |mu: f64| -> f64 {
        sorted_eigs
            .iter()
            .map(|&(_, e)| {
                let eps = e - mu;
                2.0 * 0.5 * (1.0 - eps / (eps * eps + delta * delta).sqrt())
            })
            .sum()
    };
    let (mut lo, mut hi) = (
        sorted_eigs[0].1 - 50.0,
        sorted_eigs[sorted_eigs.len() - 1].1 + 50.0,
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

// ═══════════════════════════════════════════════════════════════════
// Wavefunction Precompute
// ═══════════════════════════════════════════════════════════════════

fn precompute_wavefunctions(setup: &NucleusSetup) -> Vec<f64> {
    let n_grid = setup.n_grid;
    let n_states = setup.states.len();

    // Compute each state's wavefunction in parallel using Rayon
    let wf_per_state: Vec<Vec<f64>> = (0..n_states)
        .into_par_iter()
        .map(|si| {
            let s = &setup.states[si];
            let mut wf = vec![0.0f64; n_grid];
            for i_rho in 0..setup.n_rho {
                for i_z in 0..setup.n_z {
                    let rho = (i_rho + 1) as f64 * setup.d_rho;
                    let z_c = setup.z_min + (i_z as f64 + 0.5) * setup.d_z;

                    let xi = z_c / setup.b_z;
                    let h_n = hermite_value(s.n_z as usize, xi);
                    let norm_z = 1.0
                        / (setup.b_z
                            * PI.sqrt()
                            * (1u64 << s.n_z) as f64
                            * factorial_f64(s.n_z as usize))
                        .sqrt();
                    let phi_z = norm_z * h_n * (-xi * xi / 2.0).exp();

                    let eta = (rho / setup.b_perp).powi(2);
                    let alpha = f64::from(s.abs_lambda);
                    let nf = factorial_f64(s.n_perp as usize);
                    let gv = gamma(f64::from(s.n_perp) + alpha + 1.0).unwrap_or(1.0);
                    let norm_rho = (nf / (PI * setup.b_perp * setup.b_perp * gv)).sqrt();
                    let lag = laguerre(s.n_perp as usize, alpha, eta);
                    let phi_rho = norm_rho
                        * (rho / setup.b_perp).powi(s.abs_lambda as i32)
                        * (-eta / 2.0).exp()
                        * lag;

                    wf[setup.grid_idx(i_rho, i_z)] = phi_z * phi_rho;
                }
            }
            // Normalize
            let norm2: f64 = (0..n_grid)
                .map(|k| {
                    let ir = k / setup.n_z;
                    wf[k] * wf[k] * setup.volume_element(ir)
                })
                .sum();
            if norm2 > DIVISION_GUARD {
                let sc = 1.0 / norm2.sqrt();
                for v in &mut wf {
                    *v *= sc;
                }
            }
            wf
        })
        .collect();

    // Flatten into [n_states × n_grid]
    let mut wavefunctions = vec![0.0; n_states * n_grid];
    for (si, wf) in wf_per_state.into_iter().enumerate() {
        wavefunctions[si * n_grid..(si + 1) * n_grid].copy_from_slice(&wf);
    }
    wavefunctions
}

// ═══════════════════════════════════════════════════════════════════
// Physics Kernels (CPU with Rayon — to be moved to GPU)
// ═══════════════════════════════════════════════════════════════════

fn compute_tau_rayon(
    setup: &NucleusSetup,
    wf: &[f64],
    occ_p: &[f64],
    occ_n: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ng = setup.n_grid;
    let n_rho = setup.n_rho;
    let n_z = setup.n_z;
    let d_rho = setup.d_rho;
    let d_z = setup.d_z;

    // Process each grid point in parallel
    let tau: Vec<(f64, f64)> = (0..ng)
        .into_par_iter()
        .map(|k| {
            let i_rho = k / n_z;
            let i_z = k % n_z;
            let mut tp = 0.0;
            let mut tn = 0.0;
            for (si, _s) in setup.states.iter().enumerate() {
                let op = occ_p[si] * 2.0;
                let on = occ_n[si] * 2.0;
                if op < DENSITY_FLOOR && on < DENSITY_FLOOR {
                    continue;
                }

                let base = si * ng;
                let dpsi_drho = if i_rho == 0 {
                    (wf[base + (n_z + i_z)] - wf[base + k]) / d_rho
                } else if i_rho == n_rho - 1 {
                    (wf[base + k] - wf[base + ((i_rho - 1) * n_z + i_z)]) / d_rho
                } else {
                    (wf[base + ((i_rho + 1) * n_z + i_z)] - wf[base + ((i_rho - 1) * n_z + i_z)])
                        / (2.0 * d_rho)
                };
                let dpsi_dz = if i_z == 0 {
                    (wf[base + (i_rho * n_z + 1)] - wf[base + k]) / d_z
                } else if i_z == n_z - 1 {
                    (wf[base + k] - wf[base + (i_rho * n_z + i_z - 1)]) / d_z
                } else {
                    (wf[base + (i_rho * n_z + i_z + 1)] - wf[base + (i_rho * n_z + i_z - 1)])
                        / (2.0 * d_z)
                };
                let grad2 = dpsi_drho * dpsi_drho + dpsi_dz * dpsi_dz;
                tp += op * grad2;
                tn += on * grad2;
            }
            (tp, tn)
        })
        .collect();

    let (mut tau_p, mut tau_n) = (vec![0.0; ng], vec![0.0; ng]);
    for (k, &(tp, tn)) in tau.iter().enumerate() {
        tau_p[k] = tp;
        tau_n[k] = tn;
    }
    (tau_p, tau_n)
}

fn compute_spin_current(
    setup: &NucleusSetup,
    wf: &[f64],
    occ_p: &[f64],
    occ_n: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ng = setup.n_grid;
    let (mut jp, mut jn) = (vec![0.0; ng], vec![0.0; ng]);
    for (si, s) in setup.states.iter().enumerate() {
        let op = occ_p[si] * 2.0;
        let on = occ_n[si] * 2.0;
        if op < DENSITY_FLOOR && on < DENSITY_FLOOR {
            continue;
        }
        let ls = f64::from(s.lambda) * f64::from(s.sigma) * 0.5;
        for k in 0..ng {
            let psi2 = wf[si * ng + k] * wf[si * ng + k];
            jp[k] += op * ls * psi2;
            jn[k] += on * ls * psi2;
        }
    }
    (jp, jn)
}

fn compute_coulomb_cpu(setup: &NucleusSetup, rho_p: &[f64], vc: &mut [f64]) {
    let ng = setup.n_grid;
    let mut shells: Vec<(f64, f64)> = Vec::with_capacity(ng);
    for (i, &rp) in rho_p.iter().enumerate().take(ng) {
        let ir = i / setup.n_z;
        let iz = i % setup.n_z;
        let rho = (ir + 1) as f64 * setup.d_rho;
        let z = setup.z_min + (iz as f64 + 0.5) * setup.d_z;
        let r = (rho * rho + z * z).sqrt();
        let dv = setup.volume_element(ir);
        shells.push((r, rp.max(0.0) * dv));
    }

    let mut sidx: Vec<usize> = (0..ng).collect();
    sidx.sort_by(|&a, &b| shells[a].0.total_cmp(&shells[b].0));

    let total_qr: f64 = shells
        .iter()
        .map(|(r, c)| {
            if *r > DEFORMED_COULOMB_R_MIN {
                c / r
            } else {
                0.0
            }
        })
        .sum();
    let mut cum_q = vec![0.0; ng];
    let mut cum_qr = vec![0.0; ng];
    let (mut aq, mut aqr) = (0.0, 0.0);
    for (k, &si) in sidx.iter().enumerate() {
        aq += shells[si].1;
        aqr += if shells[si].0 > DEFORMED_COULOMB_R_MIN {
            shells[si].1 / shells[si].0
        } else {
            0.0
        };
        cum_q[k] = aq;
        cum_qr[k] = aqr;
    }
    let mut rank = vec![0usize; ng];
    for (k, &si) in sidx.iter().enumerate() {
        rank[si] = k;
    }

    let tc: f64 = shells.iter().map(|(_, c)| c).sum();
    for i in 0..ng {
        let ri = shells[i].0.max(DEFORMED_COULOMB_R_MIN);
        let k = rank[i];
        let qi = if k > 0 { cum_q[k - 1] } else { 0.0 };
        let ext = total_qr - cum_qr[k];
        vc[i] = E2 * (qi / ri + ext) + super::hfb_common::coulomb_exchange_slater(rho_p[i]);
    }
    if tc < DIVISION_GUARD {
        for v in vc.iter_mut() {
            *v = 0.0;
        }
    }
}

fn density_radial_derivative(setup: &NucleusSetup, density: &[f64]) -> Vec<f64> {
    let ng = setup.n_grid;
    (0..ng)
        .into_par_iter()
        .map(|k| {
            let ir = k / setup.n_z;
            let iz = k % setup.n_z;
            let d_dr = if ir == 0 {
                (density[setup.n_z + iz] - density[k]) / setup.d_rho
            } else if ir == setup.n_rho - 1 {
                (density[k] - density[(ir - 1) * setup.n_z + iz]) / setup.d_rho
            } else {
                (density[(ir + 1) * setup.n_z + iz] - density[(ir - 1) * setup.n_z + iz])
                    / (2.0 * setup.d_rho)
            };
            let d_dz = if iz == 0 {
                (density[ir * setup.n_z + 1] - density[k]) / setup.d_z
            } else if iz == setup.n_z - 1 {
                (density[k] - density[ir * setup.n_z + iz - 1]) / setup.d_z
            } else {
                (density[ir * setup.n_z + iz + 1] - density[ir * setup.n_z + iz - 1])
                    / (2.0 * setup.d_z)
            };
            let rho_c = (ir + 1) as f64 * setup.d_rho;
            let z_c = setup.z_min + (iz as f64 + 0.5) * setup.d_z;
            let r = (rho_c * rho_c + z_c * z_c)
                .sqrt()
                .max(DEFORMED_COULOMB_R_MIN);
            (d_dr * rho_c + d_dz * z_c) / r
        })
        .collect()
}

fn mean_field_potential(
    setup: &NucleusSetup,
    params: &[f64],
    rho_p: &[f64],
    rho_n: &[f64],
    rho_total: &[f64],
    is_proton: bool,
    tau_p: &[f64],
    tau_n: &[f64],
    v_coulomb: &[f64],
    d_rho_dr: &[f64],
) -> Vec<f64> {
    let ng = setup.n_grid;
    let (t0, t1, t2, t3) = (params[0], params[1], params[2], params[3]);
    let (x0, x1, x2, x3) = (params[4], params[5], params[6], params[7]);
    let (alpha, w0) = (params[8], params[9]);
    let rho_q: &[f64] = if is_proton { rho_p } else { rho_n };
    let tau_q: &[f64] = if is_proton { tau_p } else { tau_n };
    let d_rq_dr = density_radial_derivative(setup, rho_q);

    (0..ng)
        .into_par_iter()
        .map(|i| {
            let rho = rho_total[i].max(0.0);
            let rq = rho_q[i].max(0.0);
            let tau_tot = tau_p[i] + tau_n[i];

            let vc = t0 * ((1.0 + x0 / 2.0) * rho - (0.5 + x0) * rq)
                + t3 / 12.0
                    * rho.powf(alpha)
                    * ((2.0 + alpha) * (1.0 + x3 / 2.0) * rho
                        - (2.0 * (0.5 + x3) * rq + alpha * (1.0 + x3 / 2.0) * rho));

            let ve = t1 / 4.0 * ((2.0 + x1) * tau_tot - (1.0 + 2.0 * x1) * tau_q[i])
                + t2 / 4.0 * ((2.0 + x2) * tau_tot + (1.0 + 2.0 * x2) * tau_q[i]);

            let ir = i / setup.n_z;
            let iz = i % setup.n_z;
            let rc = (ir + 1) as f64 * setup.d_rho;
            let zc = setup.z_min + (iz as f64 + 0.5) * setup.d_z;
            let r = (rc * rc + zc * zc).sqrt().max(SPIN_ORBIT_R_MIN);
            let vso = -w0 / 2.0 * (d_rho_dr[i] + d_rq_dr[i]) / r;

            let mut v = (vc + ve + vso).clamp(-5000.0, 5000.0);
            if is_proton {
                v += v_coulomb[i].clamp(-500.0, 500.0);
            }
            v
        })
        .collect()
}

fn compute_densities_rayon(
    setup: &NucleusSetup,
    wf: &[f64],
    occ_p: &[f64],
    occ_n: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ng = setup.n_grid;
    let rho: Vec<(f64, f64)> = (0..ng)
        .into_par_iter()
        .map(|k| {
            let (mut rp, mut rn) = (0.0, 0.0);
            for (i, _) in setup.states.iter().enumerate() {
                let psi2 = wf[i * ng + k] * wf[i * ng + k];
                rp += occ_p[i] * 2.0 * psi2;
                rn += occ_n[i] * 2.0 * psi2;
            }
            (rp, rn)
        })
        .collect();

    let (mut rho_p, mut rho_n) = (vec![0.0; ng], vec![0.0; ng]);
    for (k, &(rp, rn)) in rho.iter().enumerate() {
        rho_p[k] = rp;
        rho_n[k] = rn;
    }
    (rho_p, rho_n)
}

fn density_mixing(
    rho_p: &mut [f64],
    rho_n: &mut [f64],
    new_p: &[f64],
    new_n: &[f64],
    iteration: usize,
    warmup: usize,
    ng: usize,
    _vd: usize,
    broyden_dfs: &mut Vec<Vec<f64>>,
    broyden_dus: &mut Vec<Vec<f64>>,
    prev_r: &mut Option<Vec<f64>>,
    prev_u: &mut Option<Vec<f64>>,
) {
    let vd = 2 * ng;
    if iteration < warmup {
        let am = if iteration == 0 { 1.0 } else { 0.5 };
        for i in 0..ng {
            rho_p[i] = (1.0 - am) * rho_p[i] + am * new_p[i];
            rho_n[i] = (1.0 - am) * rho_n[i] + am * new_n[i];
        }
    } else {
        let am = 0.4;
        let input: Vec<f64> = rho_p.iter().chain(rho_n.iter()).copied().collect();
        let output: Vec<f64> = new_p.iter().chain(new_n.iter()).copied().collect();
        let residual: Vec<f64> = output.iter().zip(&input).map(|(&o, &i)| o - i).collect();

        if let (Some(pr), Some(pu)) = (prev_r.as_ref(), prev_u.as_ref()) {
            let df: Vec<f64> = residual.iter().zip(pr).map(|(&r, &p)| r - p).collect();
            let du: Vec<f64> = input.iter().zip(pu).map(|(&u, &p)| u - p).collect();
            if broyden_dfs.len() >= 8 {
                broyden_dfs.remove(0);
                broyden_dus.remove(0);
            }
            broyden_dfs.push(df);
            broyden_dus.push(du);
        }

        let mut mixed = vec![0.0; vd];
        for (m, (inp, res)) in input.iter().zip(residual.iter()).enumerate().take(vd) {
            mixed[m] = inp + am * res;
        }
        for m in 0..broyden_dfs.len() {
            let dfr: f64 = broyden_dfs[m]
                .iter()
                .zip(&residual)
                .map(|(&a, &b)| a * b)
                .sum();
            let dfd: f64 = broyden_dfs[m].iter().map(|&a| a * a).sum();
            if dfd > DIVISION_GUARD {
                let g = dfr / dfd;
                for i in 0..vd {
                    mixed[i] -= g * (broyden_dus[m][i] + am * broyden_dfs[m][i]);
                }
            }
        }
        *prev_r = Some(residual);
        *prev_u = Some(input);
        rho_p.copy_from_slice(&mixed[..ng]);
        rho_n.copy_from_slice(&mixed[ng..]);
        for v in rho_p.iter_mut() {
            *v = v.max(0.0);
        }
        for v in rho_n.iter_mut() {
            *v = v.max(0.0);
        }
    }
}

fn total_energy(
    setup: &NucleusSetup,
    params: &[f64],
    rho_p: &[f64],
    rho_n: &[f64],
    occ_p: &[f64],
    occ_n: &[f64],
) -> f64 {
    let mut e_kin = 0.0;
    for (i, s) in setup.states.iter().enumerate() {
        let t = setup.hw_z * (f64::from(s.n_z) + 0.5)
            + setup.hw_perp * (2.0 * f64::from(s.n_perp) + f64::from(s.abs_lambda) + 1.0);
        e_kin += 2.0 * (occ_p[i] + occ_n[i]) * t;
    }

    let (t0, t3) = (params[0], params[3]);
    let (x0, x3, alpha) = (params[4], params[7], params[8]);
    let (mut ec, mut ecl) = (0.0, 0.0);

    for ir in 0..setup.n_rho {
        let dv = setup.volume_element(ir);
        for iz in 0..setup.n_z {
            let idx = setup.grid_idx(ir, iz);
            let rho = (rho_p[idx] + rho_n[idx]).max(0.0);
            let rp = rho_p[idx].max(0.0);
            let rn = rho_n[idx].max(0.0);
            let h0 = t0 / 4.0 * ((2.0 + x0) * rho * rho - (1.0 + 2.0 * x0) * (rp * rp + rn * rn));
            let h3 = t3 / 24.0
                * rho.powf(alpha)
                * ((2.0 + x3) * rho * rho - (1.0 + 2.0 * x3) * (rp * rp + rn * rn));
            ec += (h0 + h3) * dv;
            ecl += super::hfb_common::coulomb_exchange_energy_density(rp) * dv;
        }
    }

    let rch = 1.2 * (setup.a as f64).powf(1.0 / 3.0);
    ecl += 0.6 * (setup.z as f64) * (setup.z as f64 - 1.0) * E2 / rch;
    let ld = setup.a as f64 / 28.0;
    let ep = if setup.delta_p > PAIRING_GAP_THRESHOLD {
        -setup.delta_p.powi(2) * ld / 4.0
    } else {
        0.0
    } + if setup.delta_n > PAIRING_GAP_THRESHOLD {
        -setup.delta_n.powi(2) * ld / 4.0
    } else {
        0.0
    };
    let hw0 = 41.0 * (setup.a as f64).powf(-1.0 / 3.0);
    e_kin + ec + ecl + ep - 0.75 * hw0
}

fn quadrupole(setup: &NucleusSetup, rt: &[f64]) -> f64 {
    let mut q = 0.0;
    for ir in 0..setup.n_rho {
        let dv = setup.volume_element(ir);
        for iz in 0..setup.n_z {
            let rho = (ir + 1) as f64 * setup.d_rho;
            let z = setup.z_min + (iz as f64 + 0.5) * setup.d_z;
            q += rt[setup.grid_idx(ir, iz)] * (2.0 * z * z - rho * rho) * dv;
        }
    }
    q
}

fn beta2_from_q20(a: usize, q20: f64) -> f64 {
    let af = a as f64;
    let r0 = 1.2 * af.powf(1.0 / 3.0);
    (5.0 * PI).sqrt() * q20 / (3.0 * af * r0 * r0)
}

fn rms_radius(setup: &NucleusSetup, rt: &[f64]) -> f64 {
    let (mut sr2, mut sr) = (0.0, 0.0);
    for ir in 0..setup.n_rho {
        let dv = setup.volume_element(ir);
        for iz in 0..setup.n_z {
            let rho = (ir + 1) as f64 * setup.d_rho;
            let z = setup.z_min + (iz as f64 + 0.5) * setup.d_z;
            let r2 = rho * rho + z * z;
            let idx = setup.grid_idx(ir, iz);
            sr2 += rt[idx] * r2 * dv;
            sr += rt[idx] * dv;
        }
    }
    if sr > 0.0 {
        (sr2 / sr).sqrt()
    } else {
        0.0
    }
}

pub const fn estimate_gpu_dispatches(n_nuclei: usize, avg_blocks: usize, max_iter: usize) -> usize {
    let per_iter = 4 + avg_blocks; // H build + eigh + BCS + density (GPU dispatches only)
    n_nuclei * (1 + max_iter * per_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nucleus_setup() {
        let s = NucleusSetup::new(8, 8);
        println!(
            "O-16: grid={}×{}={}, states={}, blocks={}",
            s.n_rho,
            s.n_z,
            s.n_grid,
            s.states.len(),
            s.omega_blocks.len()
        );
        assert!(s.n_grid > 4000);
        assert!(s.states.len() > 10);
    }

    #[test]
    fn test_params_gpu_layout() {
        let p = HamiltonianParamsGpu::new(60, 80, 20, 8, 0.2, 0.35);
        assert_eq!(std::mem::size_of_val(&p), 32);
        let d = f64::from_bits(u64::from(p.d_rho_hi) << 32 | u64::from(p.d_rho_lo));
        assert!((d - 0.2).abs() < 1e-15);
    }
}
