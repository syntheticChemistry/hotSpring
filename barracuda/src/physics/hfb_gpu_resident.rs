// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-Resident Spherical HFB Solver (Level 2)
//!
//! Per SCF iteration:
//!   1. CPU→GPU: write densities for all groups
//!   2. GPU compute: potentials + H-build (7 dispatches per group)
//!   3. CPU: compute spin-orbit diagonal (O(ns × nr) per nucleus — fast)
//!   4. GPU: spin-orbit diag add + pack H → eigensolve buffer (GPU shader)
//!   5. GPU: batched eigensolve (Jacobi rotation)
//!   6. GPU: density + mixing shader
//!   7. CPU (Rayon): BCS → energy
//!
//! H matrices never leave GPU — spin-orbit correction and eigensolve packing
//! happen in a single GPU shader (`spin_orbit_pack_f64.wgsl`), eliminating
//! the H staging readback entirely.
//!
//! Pre-allocated buffers eliminate per-iteration allocation.
//! GPU eigensolve (v0.5.4+) uses GPU Jacobi rotation: single-dispatch
//! (n≤32) or multi-dispatch (n>32), both pure GPU.

use super::hfb::SphericalHFB;
use super::hfb_gpu_types::{
    make_bind_group, make_pipeline, DensityParamsUniform, EnergyParamsUniform, GroupResources,
    HamiltonianDimsUniform, MixParamsUniform, PackParams, PotentialDimsUniform,
};
use super::semf::semf_binding_energy;
use crate::tolerances::{DENSITY_FLOOR, GPU_JACOBI_CONVERGENCE, RHO_POWF_GUARD, SPIN_ORBIT_R_MIN};
use barracuda::device::WgpuDevice;
use barracuda::ops::grid::compute_ls_factor;
use barracuda::ops::linalg::BatchedEighGpu;
use barracuda::shaders::precision::ShaderTemplate;
use rayon::prelude::*;
use std::sync::Arc;

const POTENTIALS_SHADER_BODY: &str = include_str!("shaders/batched_hfb_potentials_f64.wgsl");
const HAMILTONIAN_SHADER: &str = include_str!("shaders/batched_hfb_hamiltonian_f64.wgsl");
const DENSITY_SHADER: &str = include_str!("shaders/batched_hfb_density_f64.wgsl");
// EVOLUTION(GPU): Phase 4 stub — energy shader loaded but not yet dispatched.
// Wiring requires GPU SumReduceF64 + kinetic energy kernel. See EVOLUTION_READINESS.md Tier B.
#[allow(dead_code)] // EVOLUTION: will wire when SumReduceF64 + kinetic kernel available
const ENERGY_SHADER_BODY: &str = include_str!("shaders/batched_hfb_energy_f64.wgsl");
const SO_PACK_SHADER: &str = include_str!("shaders/spin_orbit_pack_f64.wgsl");

/// Internal state for each nucleus during SCF iterations.
struct NucleusState {
    rho_p: Vec<f64>,
    rho_n: Vec<f64>,
    e_prev: f64,
    converged: bool,
    binding_energy: f64,
}

/// Work item for SCF iteration (nucleus index, group index, basis size).
struct WorkItem {
    si: usize,
    gi: usize,
    ns: usize,
}

/// Per-nucleus results from eigenvalue extraction and BCS occupation computation.
struct EigenBcsResult {
    si: usize,
    gi: usize,
    evals_p: Vec<f64>,
    evecs_p: Vec<f64>,
    evals_n: Vec<f64>,
    evecs_n: Vec<f64>,
    lambda_p: f64,
    lambda_n: f64,
    v2_p: Vec<f64>,
    v2_n: Vec<f64>,
}

fn build_initial_densities(solvers: &[(usize, usize, usize, SphericalHFB)]) -> Vec<NucleusState> {
    solvers
        .iter()
        .map(|(z, n, _, hfb)| {
            let a = *z + *n;
            let r_nuc = 1.2 * (a as f64).powf(1.0 / 3.0);
            let rho0 = 3.0 * a as f64 / (4.0 * std::f64::consts::PI * r_nuc.powi(3));
            let nr = hfb.nr();
            let rho_p: Vec<f64> = (0..nr)
                .map(|k| {
                    let r = (k + 1) as f64 * hfb.dr();
                    if r < r_nuc {
                        (rho0 * *z as f64 / a as f64).max(DENSITY_FLOOR)
                    } else {
                        DENSITY_FLOOR
                    }
                })
                .collect();
            let rho_n: Vec<f64> = (0..nr)
                .map(|k| {
                    let r = (k + 1) as f64 * hfb.dr();
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
            }
        })
        .collect()
}

fn compute_spin_orbit_diagonal(
    hfb: &SphericalHFB,
    rho_p: &[f64],
    rho_n: &[f64],
    w0: f64,
) -> Vec<f64> {
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
        let ls = compute_ls_factor(l as u32, j);
        let wf_i = hfb.wf_state(i);
        let so_integ: Vec<f64> = (0..cur_nr)
            .map(|k| wf_i[k].powi(2) * drho[k] / r[k].max(SPIN_ORBIT_R_MIN) * r[k].powi(2))
            .collect();
        let so_val = w0 * ls * barracuda::numerical::trapz(&so_integ, r).unwrap_or(0.0);
        so_diag[i] = so_val;
    }
    so_diag
}

fn extract_bcs_results(
    all_work: &[WorkItem],
    gpu_eigen: &(Vec<f64>, Vec<f64>),
    solvers: &[(usize, usize, usize, SphericalHFB)],
    global_max_ns: usize,
) -> Vec<EigenBcsResult> {
    let (eigenvalues, eigenvectors) = gpu_eigen;
    let gm = global_max_ns * global_max_ns;

    all_work
        .par_iter()
        .enumerate()
        .map(|(wi, item)| {
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

            let (v2_p, lambda_p) = hfb.bcs_occupations_from_eigs(&evals_p, z, hfb.pairing_gap());
            let (v2_n, lambda_n) = hfb.bcs_occupations_from_eigs(&evals_n, n, hfb.pairing_gap());

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

fn compute_binding_energy(
    evals_p: &[f64],
    evals_n: &[f64],
    v2_p: &[f64],
    v2_n: &[f64],
    degs: &[f64],
    delta: f64,
) -> f64 {
    let ns = evals_p.len();
    let e_sp: f64 = (0..ns)
        .map(|i| degs[i] * v2_p[i] * evals_p[i] + degs[i] * v2_n[i] * evals_n[i])
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

fn check_convergence_and_update_state(
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

pub fn binding_energies_l2_gpu_resident(
    device: Arc<WgpuDevice>,
    nuclei: &[(usize, usize)],
    params: &[f64],
    max_iter: usize,
    tol: f64,
    mixing: f64,
) -> GpuResidentL2Result {
    let t0 = std::time::Instant::now();
    let mut results: Vec<(usize, usize, f64, bool)> = Vec::with_capacity(nuclei.len());
    let mut gpu_dispatches = 0usize;
    let mut total_gpu_dispatches = 0usize;
    let mut n_semf = 0usize;

    let mut hfb_nuclei: Vec<(usize, usize, usize)> = Vec::new();
    for (idx, &(z, n)) in nuclei.iter().enumerate() {
        let a = z + n;
        if (56..=132).contains(&a) {
            hfb_nuclei.push((z, n, idx));
        } else {
            results.push((z, n, semf_binding_energy(z, n, params), true));
            n_semf += 1;
        }
    }

    if hfb_nuclei.is_empty() {
        return GpuResidentL2Result {
            results,
            hfb_time_s: t0.elapsed().as_secs_f64(),
            gpu_dispatches,
            total_gpu_dispatches,
            n_hfb: 0,
            n_semf,
        };
    }

    let solvers: Vec<(usize, usize, usize, SphericalHFB)> = hfb_nuclei
        .iter()
        .map(|&(z, n, idx)| (z, n, idx, SphericalHFB::new_adaptive(z, n)))
        .collect();

    let mut groups_map: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, (_, _, _, hfb)) in solvers.iter().enumerate() {
        groups_map
            .entry((hfb.n_states(), hfb.nr()))
            .or_default()
            .push(i);
    }

    let (t0_p, t3, x0, x3) = (params[0], params[3], params[4], params[7]);
    let (t1, t2, x1, x2) = (params[1], params[2], params[5], params[6]);
    let alpha_skyrme = params[8];
    let w0 = params[9];
    let c0t = 0.25 * (t1 * (1.0 + x1 / 2.0) + t2 * (1.0 + x2 / 2.0));
    let c1n = 0.25 * (t1 * (0.5 + x1) - t2 * (0.5 + x2));
    let hbar2_2m = super::constants::HBAR2_2M;

    let raw_device = device.device();
    let raw_queue = device.queue();

    let potentials_shader = ShaderTemplate::for_device_auto(POTENTIALS_SHADER_BODY, &device);
    let pot_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("potentials"),
        source: wgpu::ShaderSource::Wgsl(potentials_shader.into()),
    });
    let ham_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hamiltonian"),
        source: wgpu::ShaderSource::Wgsl(HAMILTONIAN_SHADER.into()),
    });

    let mut states: Vec<NucleusState> = build_initial_densities(&solvers);

    let u = wgpu::BufferBindingType::Uniform;
    let sr = wgpu::BufferBindingType::Storage { read_only: true };
    let srw = wgpu::BufferBindingType::Storage { read_only: false };

    // ═══ PRE-ALLOCATE ALL GROUP RESOURCES (ONE TIME) ═══
    let ns_nr_groups: Vec<((usize, usize), Vec<usize>)> = groups_map.into_iter().collect();
    let mut all_groups: Vec<GroupResources> = Vec::new();

    for &((ns, nr), ref group_indices) in &ns_nr_groups {
        if group_indices.is_empty() {
            continue;
        }
        let dr = solvers[group_indices[0]].3.dr();
        let mat_size = ns * ns;
        let n_max = group_indices.len();

        use wgpu::util::DeviceExt;

        let mut all_wf = Vec::with_capacity(n_max * ns * nr);
        let mut all_dwf = Vec::with_capacity(n_max * ns * nr);
        let mut all_r_grid = Vec::with_capacity(n_max * nr);
        let mut all_lj = Vec::with_capacity(n_max * ns * ns);
        let mut all_ll1 = Vec::with_capacity(n_max * ns);

        for &si in group_indices {
            let hfb = &solvers[si].3;
            all_wf.extend_from_slice(&hfb.wf_flat());
            all_dwf.extend_from_slice(&hfb.dwf_flat());
            all_r_grid.extend_from_slice(hfb.r_grid());
            all_lj.extend_from_slice(&hfb.lj_same_flat());
            all_ll1.extend_from_slice(&hfb.ll1_values());
        }

        let wf_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wf"),
            contents: bytemuck::cast_slice(&all_wf),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dwf_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dwf"),
            contents: bytemuck::cast_slice(&all_dwf),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rgrid_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rgrid"),
            contents: bytemuck::cast_slice(&all_r_grid),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lj_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lj"),
            contents: bytemuck::cast_slice(&all_lj),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ll1_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ll1"),
            contents: bytemuck::cast_slice(&all_ll1),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let sky_data: [f64; 9] = [t0_p, t3, x0, x3, alpha_skyrme, dr, c0t, c1n, hbar2_2m];
        let sky_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky"),
            contents: bytemuck::cast_slice(&sky_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dr_data: [f64; 1] = [dr];
        let dr_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dr"),
            contents: bytemuck::cast_slice(&dr_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buf_nr = (n_max * nr * 8) as u64;
        let rho_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_p"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rho_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_n"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rho_alpha_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_alpha"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rho_alpha_m1_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_alpha_m1"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let u_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u_p"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let u_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u_n"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let fq_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fq_p"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let fq_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fq_n"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cenc_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cenc"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let phi_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phi"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let h_mat_bytes = (n_max * mat_size * 8) as u64;
        let h_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_p"),
            size: h_mat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let h_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_n"),
            size: h_mat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Spin-orbit diagonal buffer: ns values per nucleus, reused every iteration
        let so_diag_bytes = (n_max * ns * 8) as u64;
        let so_diag_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("so_diag"),
            size: so_diag_bytes.max(8), // min 8 bytes for wgpu
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pot_dims = PotentialDimsUniform {
            nr: nr as u32,
            batch_size: n_max as u32,
        };
        let pot_dims_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pot_dims"),
            contents: bytemuck::bytes_of(&pot_dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let ham_dims = HamiltonianDimsUniform {
            n_states: ns as u32,
            nr: nr as u32,
            batch_size: n_max as u32,
            _pad: 0,
        };
        let ham_dims_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ham_dims"),
            contents: bytemuck::bytes_of(&ham_dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (pot_layout, pot_bg) = make_bind_group(
            raw_device,
            "pot",
            &[
                (u, &pot_dims_buf),
                (sr, &rho_p_buf),
                (sr, &rho_n_buf),
                (srw, &u_p_buf),
                (srw, &u_n_buf),
                (sr, &rgrid_buf),
                (srw, &fq_p_buf),
                (srw, &fq_n_buf),
                (srw, &cenc_buf),
                (srw, &phi_buf),
                (sr, &sky_buf),
                (sr, &rho_alpha_buf),
                (sr, &rho_alpha_m1_buf),
            ],
        );
        let (hl_p, hbg_p) = make_bind_group(
            raw_device,
            "ham_p",
            &[
                (u, &ham_dims_buf),
                (sr, &wf_buf),
                (sr, &dwf_buf),
                (sr, &u_p_buf),
                (sr, &fq_p_buf),
                (sr, &rgrid_buf),
                (sr, &lj_buf),
                (sr, &ll1_buf),
                (srw, &h_p_buf),
                (sr, &dr_buf),
            ],
        );
        let (hl_n, hbg_n) = make_bind_group(
            raw_device,
            "ham_n",
            &[
                (u, &ham_dims_buf),
                (sr, &wf_buf),
                (sr, &dwf_buf),
                (sr, &u_n_buf),
                (sr, &fq_n_buf),
                (sr, &rgrid_buf),
                (sr, &lj_buf),
                (sr, &ll1_buf),
                (srw, &h_n_buf),
                (sr, &dr_buf),
            ],
        );

        let sky_pipe = make_pipeline(raw_device, &pot_module, "compute_skyrme", &[&pot_layout]);
        let cfwd = make_pipeline(
            raw_device,
            &pot_module,
            "compute_coulomb_forward",
            &[&pot_layout],
        );
        let cbwd = make_pipeline(
            raw_device,
            &pot_module,
            "compute_coulomb_backward",
            &[&pot_layout],
        );
        let fin_pipe = make_pipeline(
            raw_device,
            &pot_module,
            "finalize_proton_potential",
            &[&pot_layout],
        );
        let fq_pipe = make_pipeline(raw_device, &pot_module, "compute_f_q", &[&pot_layout]);
        let hp_pipe = make_pipeline(raw_device, &ham_module, "build_hamiltonian", &[&hl_p]);
        let hn_pipe = make_pipeline(raw_device, &ham_module, "build_hamiltonian", &[&hl_n]);

        // ── GPU density pipeline: BCS v² + density from eigenstates + mixing ──
        let density_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density"),
            source: wgpu::ShaderSource::Wgsl(DENSITY_SHADER.into()),
        });

        let buf_ns = (n_max * ns * 8) as u64;
        let buf_evecs = (n_max * ns * ns * 8) as u64;
        let buf_1 = (n_max * 8) as u64;

        let density_params_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density_params"),
            size: std::mem::size_of::<DensityParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let evals_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("evals_p"),
            size: buf_ns,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let evals_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("evals_n"),
            size: buf_ns,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lambda_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lambda_p"),
            size: buf_1,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lambda_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lambda_n"),
            size: buf_1,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let delta_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("delta"),
            size: buf_1,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v2_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2_p"),
            size: buf_ns,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v2_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2_n"),
            size: buf_ns,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let evecs_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("evecs_p"),
            size: buf_evecs,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let evecs_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("evecs_n"),
            size: buf_evecs,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rho_p_new_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_p_new"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let rho_n_new_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_n_new"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mix_params_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mix_params"),
            size: std::mem::size_of::<MixParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rho_p_staging = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_p_staging"),
            size: buf_nr,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rho_n_staging = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_n_staging"),
            size: buf_nr,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Degeneracy buffer (shared across batch — all nuclei in group have same basis)
        let degs: Vec<f64> = solvers[group_indices[0]].3.deg_values();
        let degs_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("degs"),
            contents: bytemuck::cast_slice(&degs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Bind groups for density shader (4 groups matching WGSL layout)
        let u = wgpu::BufferBindingType::Uniform;
        let (dp_layout, density_params_bg) =
            make_bind_group(raw_device, "density_g0", &[(u, &density_params_buf)]);
        // BCS v² bind groups (v2 is READ_WRITE for the BCS compute dispatch)
        let (bcs_write_layout, bcs_p_bg) = make_bind_group(
            raw_device,
            "bcs_p",
            &[
                (sr, &evals_p_buf),
                (sr, &lambda_p_buf),
                (sr, &delta_buf),
                (srw, &v2_p_buf),
            ],
        );
        let (_, bcs_n_bg) = make_bind_group(
            raw_device,
            "bcs_n",
            &[
                (sr, &evals_n_buf),
                (sr, &lambda_n_buf),
                (sr, &delta_buf),
                (srw, &v2_n_buf),
            ],
        );
        // Read-only BCS bind groups for density/mixing pipelines (v2 is READ only)
        let (bcs_read_layout, bcs_p_read_bg) = make_bind_group(
            raw_device,
            "bcs_p_read",
            &[
                (sr, &evals_p_buf),
                (sr, &lambda_p_buf),
                (sr, &delta_buf),
                (sr, &v2_p_buf),
            ],
        );
        let (_, bcs_n_read_bg) = make_bind_group(
            raw_device,
            "bcs_n_read",
            &[
                (sr, &evals_n_buf),
                (sr, &lambda_n_buf),
                (sr, &delta_buf),
                (sr, &v2_n_buf),
            ],
        );
        // Density write bind groups (rho_new is READ_WRITE for density compute)
        let (dens_write_layout, density_p_bg) = make_bind_group(
            raw_device,
            "dens_p",
            &[
                (sr, &evecs_p_buf),
                (sr, &v2_p_buf),
                (sr, &degs_buf),
                (sr, &wf_buf),
                (srw, &rho_p_new_buf),
            ],
        );
        let (_, density_n_bg) = make_bind_group(
            raw_device,
            "dens_n",
            &[
                (sr, &evecs_n_buf),
                (sr, &v2_n_buf),
                (sr, &degs_buf),
                (sr, &wf_buf),
                (srw, &rho_n_new_buf),
            ],
        );
        // Density read bind groups (rho_new is READ for mixing pass — avoids
        // buffer conflict when mix reads rho_new and density wrote it)
        let (dens_read_layout, density_p_read_bg) = make_bind_group(
            raw_device,
            "dens_p_read",
            &[
                (sr, &evecs_p_buf),
                (sr, &v2_p_buf),
                (sr, &degs_buf),
                (sr, &wf_buf),
                (sr, &rho_p_new_buf),
            ],
        );
        let (_, density_n_read_bg) = make_bind_group(
            raw_device,
            "dens_n_read",
            &[
                (sr, &evecs_n_buf),
                (sr, &v2_n_buf),
                (sr, &degs_buf),
                (sr, &wf_buf),
                (sr, &rho_n_new_buf),
            ],
        );
        let (mix_layout, mix_p_bg) = make_bind_group(
            raw_device,
            "mix_p",
            &[
                (u, &mix_params_buf),
                (sr, &rho_p_new_buf),
                (srw, &rho_p_buf),
            ],
        );
        let (_, mix_n_bg) = make_bind_group(
            raw_device,
            "mix_n",
            &[
                (u, &mix_params_buf),
                (sr, &rho_n_new_buf),
                (srw, &rho_n_buf),
            ],
        );

        let bcs_v2_pipe = make_pipeline(
            raw_device,
            &density_module,
            "compute_bcs_v2",
            &[&dp_layout, &bcs_write_layout],
        );
        let density_pipe = make_pipeline(
            raw_device,
            &density_module,
            "compute_density",
            &[&dp_layout, &bcs_read_layout, &dens_write_layout],
        );
        let mix_pipe = make_pipeline(
            raw_device,
            &density_module,
            "mix_density",
            &[&dp_layout, &bcs_read_layout, &dens_read_layout, &mix_layout],
        );

        // Energy pipeline (deferred to future PR — stubs for struct completeness)
        let energy_params_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("energy_params"),
            size: std::mem::size_of::<EnergyParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let energy_integrands_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("energy_integrands"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let e_pair_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("e_pair"),
            size: buf_1,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let energy_staging = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("energy_staging"),
            size: buf_nr,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let e_pair_staging = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("e_pair_staging"),
            size: buf_1,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Placeholder bind groups using density_params_buf for energy (will be rewired)
        let (ep_layout, energy_params_bg) = make_bind_group(
            raw_device,
            "energy_g0",
            &[(wgpu::BufferBindingType::Uniform, &energy_params_buf)],
        );
        let (_, energy_pair_bg) = make_bind_group(
            raw_device,
            "energy_pair",
            &[(sr, &energy_integrands_buf), (srw, &e_pair_buf)],
        );
        let _ = ep_layout;

        all_groups.push(GroupResources {
            ns,
            nr,
            group_indices: group_indices.clone(),
            n_max,
            mat_size,
            rho_p_buf,
            rho_n_buf,
            rho_alpha_buf,
            rho_alpha_m1_buf,
            pot_dims_buf,
            ham_dims_buf,
            h_p_buf,
            h_n_buf,
            so_diag_buf,
            pot_bg,
            hbg_p,
            hbg_n,
            sky_pipe,
            cfwd,
            cbwd,
            fin_pipe,
            fq_pipe,
            hp_pipe,
            hn_pipe,
            density_params_buf,
            density_params_bg,
            bcs_p_bg,
            bcs_n_bg,
            bcs_p_read_bg,
            bcs_n_read_bg,
            density_p_bg,
            density_n_bg,
            density_p_read_bg,
            density_n_read_bg,
            mix_p_bg,
            mix_n_bg,
            evals_p_buf,
            evals_n_buf,
            evecs_p_buf,
            evecs_n_buf,
            lambda_p_buf,
            lambda_n_buf,
            delta_buf,
            v2_p_buf,
            v2_n_buf,
            rho_p_new_buf,
            rho_n_new_buf,
            mix_params_buf,
            bcs_v2_pipe,
            density_pipe,
            mix_pipe,
            rho_p_staging,
            rho_n_staging,
            energy_params_bg,
            energy_pair_bg,
            energy_integrands_buf,
            e_pair_buf,
            energy_staging,
            e_pair_staging,
            nr_wg: (nr as u32).div_ceil(256),
            ns_wg: (ns as u32).div_ceil(16),
        });
    }

    let global_max_ns = all_groups.iter().map(|g| g.ns).max().unwrap_or(1);

    // ═══ PRE-ALLOCATE EIGENSOLVE BUFFERS (ONE TIME) ═══
    // Eliminates per-iteration GPU buffer allocation + shader/pipeline recreation.
    let max_eigh_batch = solvers.len() * 2; // proton + neutron per nucleus
    let (eigh_matrices_buf, eigh_eigenvalues_buf, eigh_eigenvectors_buf) =
        BatchedEighGpu::create_buffers(&device, global_max_ns, max_eigh_batch)
            .expect("pre-allocate eigensolve GPU buffers");

    // ═══ SPIN-ORBIT + PACK PIPELINE (GPU-RESIDENT H → EIGENSOLVE) ═══
    // Eliminates H staging readback: spin-orbit diag add + matrix packing
    // happen entirely on GPU, writing directly to the eigensolve buffer.
    let so_pack_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("so_pack"),
        source: wgpu::ShaderSource::Wgsl(SO_PACK_SHADER.into()),
    });
    let so_pack_layout = raw_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("so_pack_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let so_pack_pipe = make_pipeline(
        raw_device,
        &so_pack_module,
        "pack_with_spinorbit",
        &[&so_pack_layout],
    );

    // Per-group pack resources: bind groups referencing per-group H buffers + eigensolve buffer
    struct PackGroupResources {
        params_p_buf: wgpu::Buffer,
        params_n_buf: wgpu::Buffer,
        pack_p_bg: wgpu::BindGroup,
        pack_n_bg: wgpu::BindGroup,
    }
    let pack_resources: Vec<PackGroupResources> = all_groups
        .iter()
        .map(|g| {
            let params_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pack_params_p"),
                size: std::mem::size_of::<PackParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let params_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pack_params_n"),
                size: std::mem::size_of::<PackParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let pack_p_bg = raw_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pack_p"),
                layout: &so_pack_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_p_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: g.h_p_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: g.so_diag_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: eigh_matrices_buf.as_entire_binding(),
                    },
                ],
            });
            let pack_n_bg = raw_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pack_n"),
                layout: &so_pack_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_n_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: g.h_n_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: g.so_diag_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: eigh_matrices_buf.as_entire_binding(),
                    },
                ],
            });
            PackGroupResources {
                params_p_buf,
                params_n_buf,
                pack_p_bg,
                pack_n_bg,
            }
        })
        .collect();

    // ═══ UNIFIED SCF LOOP — ALL GROUPS IN ONE ENCODER ═══
    let mut t_gpu_total = 0.0f64;
    let t_poll_total = 0.0f64;
    let mut t_cpu_total = 0.0f64;
    let mut t_upload_total = 0.0f64;

    for iter in 0..max_iter {
        let any_active = all_groups
            .iter()
            .any(|g| g.group_indices.iter().any(|&si| !states[si].converged));
        if !any_active {
            break;
        }

        // Pre-compute active groups (shared across upload, H-build, pack, work items)
        let active_groups: Vec<(usize, Vec<usize>)> = all_groups
            .iter()
            .enumerate()
            .filter_map(|(gi, g)| {
                let active: Vec<usize> = g
                    .group_indices
                    .iter()
                    .copied()
                    .filter(|&si| !states[si].converged)
                    .collect();
                if active.is_empty() {
                    None
                } else {
                    Some((gi, active))
                }
            })
            .collect();

        let alpha_mix = if iter == 0 { 0.8 } else { mixing };

        // Build work items (no H data — H stays on GPU)
        let mut all_work: Vec<WorkItem> = Vec::new();
        for &(gi, ref active) in &active_groups {
            let g = &all_groups[gi];
            for &si in active {
                all_work.push(WorkItem { si, gi, ns: g.ns });
            }
        }

        // Step 1: Upload ALL data before recording the encoder.
        // write_buffer operations are guaranteed visible to the next queue.submit().
        let t_upload = std::time::Instant::now();
        for &(gi, ref active) in &active_groups {
            let g = &all_groups[gi];
            let n_nuclei = active.len();
            let bn = n_nuclei as u32;

            if n_nuclei != g.n_max || iter == 0 {
                raw_queue.write_buffer(
                    &g.pot_dims_buf,
                    0,
                    bytemuck::bytes_of(&PotentialDimsUniform {
                        nr: g.nr as u32,
                        batch_size: bn,
                    }),
                );
                raw_queue.write_buffer(
                    &g.ham_dims_buf,
                    0,
                    bytemuck::bytes_of(&HamiltonianDimsUniform {
                        n_states: g.ns as u32,
                        nr: g.nr as u32,
                        batch_size: bn,
                        _pad: 0,
                    }),
                );
            }

            let mut rho_p_flat = vec![0.0f64; n_nuclei * g.nr];
            let mut rho_n_flat = vec![0.0f64; n_nuclei * g.nr];
            for (bi, &si) in active.iter().enumerate() {
                rho_p_flat[bi * g.nr..(bi + 1) * g.nr].copy_from_slice(&states[si].rho_p);
                rho_n_flat[bi * g.nr..(bi + 1) * g.nr].copy_from_slice(&states[si].rho_n);
            }
            raw_queue.write_buffer(&g.rho_p_buf, 0, bytemuck::cast_slice(&rho_p_flat));
            raw_queue.write_buffer(&g.rho_n_buf, 0, bytemuck::cast_slice(&rho_n_flat));

            // CPU-computed rho_alpha and rho_alpha_m1 for full f64 precision
            let mut rho_alpha_flat = vec![0.0f64; n_nuclei * g.nr];
            let mut rho_alpha_m1_flat = vec![0.0f64; n_nuclei * g.nr];
            for k in 0..(n_nuclei * g.nr) {
                let rho = (rho_p_flat[k] + rho_n_flat[k]).max(RHO_POWF_GUARD);
                rho_alpha_flat[k] = rho.powf(alpha_skyrme);
                rho_alpha_m1_flat[k] = if rho_p_flat[k] + rho_n_flat[k] > DENSITY_FLOOR {
                    rho.powf(alpha_skyrme - 1.0)
                } else {
                    0.0
                };
            }
            raw_queue.write_buffer(&g.rho_alpha_buf, 0, bytemuck::cast_slice(&rho_alpha_flat));
            raw_queue.write_buffer(
                &g.rho_alpha_m1_buf,
                0,
                bytemuck::cast_slice(&rho_alpha_m1_flat),
            );

            // Spin-orbit diagonal: V_so[i] = w0 * ls_i * ∫ |ψ_i(r)|² (dρ/dr)/r r² dr
            // O(ns × nr) per nucleus — fast on CPU, tiny upload (ns f64s).
            let ns = g.ns;
            let mut so_diag_flat = vec![0.0f64; n_nuclei * ns];
            if w0 != 0.0 {
                for (bi, &si) in active.iter().enumerate() {
                    let (_, _, _, ref hfb) = solvers[si];
                    let so_diag =
                        compute_spin_orbit_diagonal(hfb, &states[si].rho_p, &states[si].rho_n, w0);
                    so_diag_flat[bi * ns..(bi + 1) * ns].copy_from_slice(&so_diag);
                }
            }
            raw_queue.write_buffer(&g.so_diag_buf, 0, bytemuck::cast_slice(&so_diag_flat));
        }

        // Upload pack params for spin-orbit + pack shader
        let n_work = all_work.len();
        let gns_wg = (global_max_ns as u32).div_ceil(16);
        {
            let mut cum_wi = 0usize;
            for &(gi, ref active) in &active_groups {
                let bn = active.len() as u32;
                let pr = &pack_resources[gi];

                raw_queue.write_buffer(
                    &pr.params_p_buf,
                    0,
                    bytemuck::bytes_of(&PackParams {
                        ns: all_groups[gi].ns as u32,
                        gns: global_max_ns as u32,
                        n_active: bn,
                        dst_start: (cum_wi * 2) as u32,
                        dst_stride: 2,
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    }),
                );

                raw_queue.write_buffer(
                    &pr.params_n_buf,
                    0,
                    bytemuck::bytes_of(&PackParams {
                        ns: all_groups[gi].ns as u32,
                        gns: global_max_ns as u32,
                        n_active: bn,
                        dst_start: (cum_wi * 2 + 1) as u32,
                        dst_stride: 2,
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    }),
                );

                cum_wi += active.len();
            }
        }
        t_upload_total += t_upload.elapsed().as_secs_f64();

        // Step 2: ONE encoder — H-build + spin-orbit pack in a single compute pass.
        // wgpu guarantees automatic memory barriers between dispatches within a pass,
        // so pack dispatches see the H matrices written by earlier H-build dispatches.
        let t_gpu = std::time::Instant::now();
        let mut encoder = raw_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scf_hbuild_pack"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hbuild_pack"),
                timestamp_writes: None,
            });

            for &(gi, ref active) in &active_groups {
                let g = &all_groups[gi];
                let bn = active.len() as u32;

                // H-build: 7 dispatches (Skyrme, Coulomb fwd/bwd/finalize, falloff, H_p, H_n)
                pass.set_pipeline(&g.sky_pipe);
                pass.set_bind_group(0, &g.pot_bg, &[]);
                pass.dispatch_workgroups(g.nr_wg, bn, 1);
                pass.set_pipeline(&g.cfwd);
                pass.set_bind_group(0, &g.pot_bg, &[]);
                pass.dispatch_workgroups(bn, 1, 1);
                pass.set_pipeline(&g.cbwd);
                pass.set_bind_group(0, &g.pot_bg, &[]);
                pass.dispatch_workgroups(bn, 1, 1);
                pass.set_pipeline(&g.fin_pipe);
                pass.set_bind_group(0, &g.pot_bg, &[]);
                pass.dispatch_workgroups(g.nr_wg, bn, 1);
                pass.set_pipeline(&g.fq_pipe);
                pass.set_bind_group(0, &g.pot_bg, &[]);
                pass.dispatch_workgroups(g.nr_wg, bn, 1);
                pass.set_pipeline(&g.hp_pipe);
                pass.set_bind_group(0, &g.hbg_p, &[]);
                pass.dispatch_workgroups(g.ns_wg, g.ns_wg, bn);
                pass.set_pipeline(&g.hn_pipe);
                pass.set_bind_group(0, &g.hbg_n, &[]);
                pass.dispatch_workgroups(g.ns_wg, g.ns_wg, bn);

                // Pack: spin-orbit add + ns×ns → gns×gns with padding into eigensolve buffer
                let pr = &pack_resources[gi];
                pass.set_pipeline(&so_pack_pipe);
                pass.set_bind_group(0, &pr.pack_p_bg, &[]);
                pass.dispatch_workgroups(gns_wg, gns_wg, bn);
                pass.set_bind_group(0, &pr.pack_n_bg, &[]);
                pass.dispatch_workgroups(gns_wg, gns_wg, bn);

                total_gpu_dispatches += 9; // 7 H-build + 2 pack
            }
        }

        // SINGLE submit — H-build + pack complete, eigensolve buffer ready
        raw_queue.submit(Some(encoder.finish()));
        gpu_dispatches += 1;
        t_gpu_total += t_gpu.elapsed().as_secs_f64();

        // Step 3: Eigensolve on packed GPU buffer
        let t_read = std::time::Instant::now();
        let gpu_eigen: Option<(Vec<f64>, Vec<f64>)> = if n_work > 0 {
            let batch_size = n_work * 2;

            // 4d. GPU eigensolve — operates on packed buffer (already populated by GPU)
            let dispatch_ok = if global_max_ns <= 32 {
                BatchedEighGpu::execute_single_dispatch_buffers(
                    &device,
                    &eigh_matrices_buf,
                    &eigh_eigenvalues_buf,
                    &eigh_eigenvectors_buf,
                    global_max_ns,
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
                    global_max_ns,
                    batch_size,
                    30,
                )
            };

            dispatch_ok.ok().map(|()| {
                let eigenvalues = BatchedEighGpu::read_eigenvalues(
                    &device,
                    &eigh_eigenvalues_buf,
                    global_max_ns,
                    batch_size,
                )
                .expect("eigenvalue readback");
                let eigenvectors = BatchedEighGpu::read_eigenvectors(
                    &device,
                    &eigh_eigenvectors_buf,
                    global_max_ns,
                    batch_size,
                )
                .expect("eigenvector readback");
                (eigenvalues, eigenvectors)
            })
        } else {
            None
        };

        // 5a. Rayon: eigensolve extraction + BCS λ (chemical potential) on CPU
        //
        // Eigenvalues/eigenvectors are extracted from the GPU readback for
        // CPU energy computation. BCS λ is computed on CPU (Brent root-finding).
        // v² will be computed on GPU via compute_bcs_v2 shader.
        let eigen_bcs: Vec<EigenBcsResult> = extract_bcs_results(
            &all_work,
            gpu_eigen
                .as_ref()
                .expect("GPU eigensolve must succeed — no CPU fallback"),
            &solvers,
            global_max_ns,
        );

        // 5b. GPU BCS v² + density + mixing pass
        //
        // Upload eigenvalues + λ + Δ + eigenvectors → dispatch compute_bcs_v2
        // → dispatch density + mixing → readback mixed densities.
        // BCS v² is computed entirely on GPU, eliminating CPU v² + upload.
        let mut density_encoder =
            raw_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bcs_density_all"),
            });
        let mut density_active_groups: Vec<usize> = Vec::new();

        for (gi, g) in all_groups.iter().enumerate() {
            let items: Vec<&EigenBcsResult> = eigen_bcs.iter().filter(|r| r.gi == gi).collect();
            if items.is_empty() {
                continue;
            }
            let bn = items.len() as u32;

            raw_queue.write_buffer(
                &g.mix_params_buf,
                0,
                bytemuck::bytes_of(&MixParamsUniform::new(bn * g.nr as u32, alpha_mix)),
            );
            raw_queue.write_buffer(
                &g.density_params_buf,
                0,
                bytemuck::bytes_of(&DensityParamsUniform {
                    n_states: g.ns as u32,
                    nr: g.nr as u32,
                    batch_size: bn,
                    _pad: 0,
                }),
            );

            // Upload eigenvalues + λ + Δ for GPU BCS v² computation
            let mut evals_p_flat = Vec::with_capacity(items.len() * g.ns);
            let mut evals_n_flat = Vec::with_capacity(items.len() * g.ns);
            let mut lambda_p_flat = Vec::with_capacity(items.len());
            let mut lambda_n_flat = Vec::with_capacity(items.len());
            let mut delta_flat = Vec::with_capacity(items.len());
            let mut evecs_p_flat = Vec::with_capacity(items.len() * g.ns * g.ns);
            let mut evecs_n_flat = Vec::with_capacity(items.len() * g.ns * g.ns);

            for d in &items {
                evals_p_flat.extend_from_slice(&d.evals_p);
                evals_n_flat.extend_from_slice(&d.evals_n);
                lambda_p_flat.push(d.lambda_p);
                lambda_n_flat.push(d.lambda_n);
                let (_, _, _, ref hfb) = solvers[d.si];
                delta_flat.push(hfb.pairing_gap());
                evecs_p_flat.extend_from_slice(&d.evecs_p);
                evecs_n_flat.extend_from_slice(&d.evecs_n);
            }

            raw_queue.write_buffer(&g.evals_p_buf, 0, bytemuck::cast_slice(&evals_p_flat));
            raw_queue.write_buffer(&g.evals_n_buf, 0, bytemuck::cast_slice(&evals_n_flat));
            raw_queue.write_buffer(&g.lambda_p_buf, 0, bytemuck::cast_slice(&lambda_p_flat));
            raw_queue.write_buffer(&g.lambda_n_buf, 0, bytemuck::cast_slice(&lambda_n_flat));
            raw_queue.write_buffer(&g.delta_buf, 0, bytemuck::cast_slice(&delta_flat));
            raw_queue.write_buffer(&g.evecs_p_buf, 0, bytemuck::cast_slice(&evecs_p_flat));
            raw_queue.write_buffer(&g.evecs_n_buf, 0, bytemuck::cast_slice(&evecs_n_flat));

            // BCS v² pass: write to v2 buffers (READ_WRITE bind groups)
            {
                let mut pass = density_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bcs_v2"),
                    timestamp_writes: None,
                });

                let ns_wg_bcs = (g.ns as u32).div_ceil(256);
                pass.set_pipeline(&g.bcs_v2_pipe);
                pass.set_bind_group(0, &g.density_params_bg, &[]);
                pass.set_bind_group(1, &g.bcs_p_bg, &[]);
                pass.dispatch_workgroups(ns_wg_bcs, bn, 1);
                pass.set_bind_group(1, &g.bcs_n_bg, &[]);
                pass.dispatch_workgroups(ns_wg_bcs, bn, 1);
            }

            // Density pass: read v2 buffers (READ-only), write rho_new
            {
                let mut pass = density_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("density"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&g.density_pipe);
                pass.set_bind_group(0, &g.density_params_bg, &[]);
                pass.set_bind_group(1, &g.bcs_p_read_bg, &[]);
                pass.set_bind_group(2, &g.density_p_bg, &[]);
                pass.dispatch_workgroups(g.nr_wg, bn, 1);

                pass.set_bind_group(1, &g.bcs_n_read_bg, &[]);
                pass.set_bind_group(2, &g.density_n_bg, &[]);
                pass.dispatch_workgroups(g.nr_wg, bn, 1);
            }

            // Mixing pass: read rho_new (now available), write rho in-place
            {
                let mut pass = density_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("density_mix"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&g.mix_pipe);
                pass.set_bind_group(0, &g.density_params_bg, &[]);
                pass.set_bind_group(1, &g.bcs_p_read_bg, &[]);
                pass.set_bind_group(2, &g.density_p_read_bg, &[]);
                pass.set_bind_group(3, &g.mix_p_bg, &[]);
                pass.dispatch_workgroups((bn * g.nr as u32).div_ceil(256), 1, 1);

                pass.set_bind_group(2, &g.density_n_read_bg, &[]);
                pass.set_bind_group(3, &g.mix_n_bg, &[]);
                pass.dispatch_workgroups((bn * g.nr as u32).div_ceil(256), 1, 1);
            }

            let rho_bytes = (items.len() * g.nr * 8) as u64;
            density_encoder.copy_buffer_to_buffer(&g.rho_p_buf, 0, &g.rho_p_staging, 0, rho_bytes);
            density_encoder.copy_buffer_to_buffer(&g.rho_n_buf, 0, &g.rho_n_staging, 0, rho_bytes);

            total_gpu_dispatches += 6; // 2 BCS v² + 2 density + 2 mixing
            density_active_groups.push(gi);
        }

        raw_queue.submit(Some(density_encoder.finish()));

        // Map density staging buffers
        let mut density_receivers = Vec::new();
        for &gi in &density_active_groups {
            let g = &all_groups[gi];
            let items_count = eigen_bcs.iter().filter(|r| r.gi == gi).count();
            let rho_bytes = (items_count * g.nr * 8) as u64;

            let slice_p = g.rho_p_staging.slice(..rho_bytes);
            let slice_n = g.rho_n_staging.slice(..rho_bytes);
            let (tx_p, rx_p) = std::sync::mpsc::channel();
            let (tx_n, rx_n) = std::sync::mpsc::channel();
            slice_p.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_p.send(r);
            });
            slice_n.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_n.send(r);
            });
            density_receivers.push((gi, rx_p, rx_n));
        }
        raw_device.poll(wgpu::Maintain::Wait);

        // Read back mixed densities per group
        let mut mixed_densities: std::collections::HashMap<usize, (Vec<f64>, Vec<f64>)> =
            std::collections::HashMap::new();
        for (gi, rx_p, rx_n) in density_receivers {
            rx_p.recv()
                .map_err(|_| "density rho_p map channel failed".to_string())
                .and_then(|r| r.map_err(|e| format!("density rho_p map: {e}")))
                .expect("GPU density staging map for rho_p");
            rx_n.recv()
                .map_err(|_| "density rho_n map channel failed".to_string())
                .and_then(|r| r.map_err(|e| format!("density rho_n map: {e}")))
                .expect("GPU density staging map for rho_n");

            let g = &all_groups[gi];
            let items_count = eigen_bcs.iter().filter(|r| r.gi == gi).count();
            let rho_bytes = (items_count * g.nr * 8) as u64;
            let rho_p_data: Vec<f64>;
            let rho_n_data: Vec<f64>;
            {
                let mapped = g.rho_p_staging.slice(..rho_bytes).get_mapped_range();
                rho_p_data =
                    bytemuck::cast_slice::<u8, f64>(&mapped)[..items_count * g.nr].to_vec();
            }
            g.rho_p_staging.unmap();
            {
                let mapped = g.rho_n_staging.slice(..rho_bytes).get_mapped_range();
                rho_n_data =
                    bytemuck::cast_slice::<u8, f64>(&mapped)[..items_count * g.nr].to_vec();
            }
            g.rho_n_staging.unmap();
            mixed_densities.insert(gi, (rho_p_data, rho_n_data));
        }

        // Build per-nucleus mixed density slices from GPU readback
        let mut group_offsets: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        // 5c. Update states from GPU-mixed densities + eigenvalue-based energy
        //
        // Energy uses BCS single-particle sum: E = Σ deg_i * v²_i * ε_i + E_pair
        // No eigenvectors needed — eigenvalues and v² from CPU suffice.
        // Convergence is consistent (same formula every iteration).
        for eb in &eigen_bcs {
            let (_, _, _, ref hfb) = solvers[eb.si];
            let nr = hfb.nr();
            let gi = eb.gi;

            let bi = {
                let offset = group_offsets.entry(gi).or_insert(0);
                let cur = *offset;
                *offset += 1;
                cur
            };

            let (rho_p_mixed, rho_n_mixed) =
                if let Some((ref rho_p_all, ref rho_n_all)) = mixed_densities.get(&gi) {
                    (
                        rho_p_all[bi * nr..(bi + 1) * nr].to_vec(),
                        rho_n_all[bi * nr..(bi + 1) * nr].to_vec(),
                    )
                } else {
                    (states[eb.si].rho_p.clone(), states[eb.si].rho_n.clone())
                };

            let degs: Vec<f64> = hfb.deg_values();
            let delta = hfb.pairing_gap();
            let e_total =
                compute_binding_energy(&eb.evals_p, &eb.evals_n, &eb.v2_p, &eb.v2_n, &degs, delta);

            check_convergence_and_update_state(
                &mut states[eb.si],
                rho_p_mixed,
                rho_n_mixed,
                e_total,
                tol,
                iter,
            );
        }
        t_cpu_total += t_read.elapsed().as_secs_f64();
    }

    #[cfg(debug_assertions)]
    eprintln!(
        "  [PROFILE] upload={:.3}s gpu={:.3}s poll={:.3}s cpu={:.3}s total={:.3}s",
        t_upload_total,
        t_gpu_total,
        t_poll_total,
        t_cpu_total,
        t0.elapsed().as_secs_f64()
    );

    let n_hfb = solvers.len();
    for (i, (z, n, _, _)) in solvers.iter().enumerate() {
        results.push((*z, *n, states[i].binding_energy, states[i].converged));
    }

    GpuResidentL2Result {
        results,
        hfb_time_s: t0.elapsed().as_secs_f64(),
        gpu_dispatches,
        total_gpu_dispatches,
        n_hfb,
        n_semf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_resident_result_fields() {
        let r = GpuResidentL2Result {
            results: vec![(28, 28, 483.0, true), (50, 82, 1095.0, true)],
            hfb_time_s: 2.5,
            gpu_dispatches: 10,
            total_gpu_dispatches: 70,
            n_hfb: 2,
            n_semf: 0,
        };
        assert_eq!(r.results.len(), 2);
        assert_eq!(r.n_hfb, 2);
        assert_eq!(r.total_gpu_dispatches, 70);
    }

    #[test]
    fn potential_dims_uniform_layout() {
        let dims = PotentialDimsUniform {
            nr: 100,
            batch_size: 4,
        };
        assert_eq!(dims.nr, 100);
        assert_eq!(dims.batch_size, 4);
        // Verify Pod safety — bytemuck::bytes_of requires this
        let bytes = bytemuck::bytes_of(&dims);
        assert_eq!(bytes.len(), 8); // 2 × u32 = 8 bytes
    }

    #[test]
    fn hamiltonian_dims_uniform_layout() {
        let dims = HamiltonianDimsUniform {
            n_states: 14,
            nr: 100,
            batch_size: 4,
            _pad: 0,
        };
        assert_eq!(dims.n_states, 14);
        let bytes = bytemuck::bytes_of(&dims);
        assert_eq!(bytes.len(), 16); // 4 × u32 = 16 bytes
    }

    #[test]
    fn nucleus_grouping_by_ns_nr() {
        use crate::physics::hfb::SphericalHFB;
        let nuclei: Vec<(usize, usize)> = vec![(28, 28), (40, 50), (50, 82)];
        let hfb_nuclei: Vec<(usize, usize)> = nuclei
            .iter()
            .copied()
            .filter(|&(z, n)| (56..=132).contains(&(z + n)))
            .collect();

        let mut groups: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();
        for (i, &(z, n)) in hfb_nuclei.iter().enumerate() {
            let hfb = SphericalHFB::new_adaptive(z, n);
            groups
                .entry((hfb.n_states(), hfb.nr()))
                .or_default()
                .push(i);
        }

        // Same-shell nuclei should be in the same group
        assert!(!groups.is_empty(), "should have at least one group");
        let total: usize = groups.values().map(std::vec::Vec::len).sum();
        assert_eq!(total, hfb_nuclei.len(), "all nuclei should be grouped");
    }

    #[test]
    fn density_initialization_positive() {
        // Verify initial density profile is positive and physically reasonable
        let z = 28;
        let n = 28;
        let a = z + n;
        let r_nuc = 1.2 * (a as f64).powf(1.0 / 3.0);
        let rho0 = 3.0 * a as f64 / (4.0 * std::f64::consts::PI * r_nuc.powi(3));
        let nr = 100;
        let dr = 15.0 / nr as f64;

        let rho_p: Vec<f64> = (0..nr)
            .map(|k| {
                let r = (k + 1) as f64 * dr;
                if r < r_nuc {
                    (rho0 * z as f64 / a as f64).max(DENSITY_FLOOR)
                } else {
                    DENSITY_FLOOR
                }
            })
            .collect();

        assert!(
            rho_p.iter().all(|&v| v > 0.0),
            "all densities must be positive"
        );
        assert!(rho_p[0] > 1e-3, "inner density should be substantial");
        assert!(rho_p[nr - 1] < 1e-10, "outer density should be near zero");
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known constant
    fn density_floor_applied() {
        use crate::tolerances::DENSITY_FLOOR;
        // Verify density floor logic
        let rho = -0.001_f64;
        let floored = rho.max(DENSITY_FLOOR);
        assert_eq!(floored, DENSITY_FLOOR);
        assert!(floored > 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known constant
    fn spin_orbit_r_min_guard() {
        use crate::tolerances::SPIN_ORBIT_R_MIN;
        let r = 0.001_f64;
        let guarded = r.max(SPIN_ORBIT_R_MIN);
        assert_eq!(guarded, SPIN_ORBIT_R_MIN);
        assert!(guarded >= 0.1);
    }
}
