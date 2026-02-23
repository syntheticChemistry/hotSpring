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
//!   7. GPU (when `gpu_energy`): energy integrands + pairing → readback; else CPU: BCS → energy
//!
//! H matrices never leave GPU — spin-orbit correction and eigensolve packing
//! happen in a single GPU shader (`spin_orbit_pack_f64.wgsl`), eliminating
//! the H staging readback entirely.
//!
//! Pre-allocated buffers eliminate per-iteration allocation.
//! GPU eigensolve (v0.5.4+) uses GPU Jacobi rotation: single-dispatch
//! (n≤32) or multi-dispatch (n>32), both pure GPU.

pub(crate) mod types;

mod dispatch;
mod pipelines;
mod resources;

#[cfg(test)]
mod tests;

use super::hfb::SphericalHFB;
use super::hfb_gpu_types::{make_pipeline, GroupResources, PackParams};
use super::semf::semf_binding_energy;
use crate::error::HotSpringError;
use crate::tolerances::GPU_JACOBI_CONVERGENCE;
use barracuda::device::WgpuDevice;
use barracuda::ops::linalg::BatchedEighGpu;
use barracuda::shaders::precision::ShaderTemplate;
use std::sync::Arc;

pub use types::GpuResidentL2Result;
use types::{
    build_initial_densities, check_convergence_and_update_state, compute_binding_energy,
    extract_bcs_results, WorkItem,
};

const POTENTIALS_SHADER_BODY: &str = include_str!("../shaders/batched_hfb_potentials_f64.wgsl");
const HAMILTONIAN_SHADER: &str = include_str!("../shaders/batched_hfb_hamiltonian_f64.wgsl");
const DENSITY_SHADER: &str = include_str!("../shaders/batched_hfb_density_f64.wgsl");
#[cfg(feature = "gpu_energy")]
const ENERGY_SHADER_BODY: &str = include_str!("../shaders/batched_hfb_energy_f64.wgsl");
const SO_PACK_SHADER: &str = include_str!("../shaders/spin_orbit_pack_f64.wgsl");

/// Batch compute L2 binding energies on GPU with full GPU-resident pipeline.
///
/// # Errors
///
/// Returns [`HotSpringError::GpuCompute`] if GPU buffer allocation, shader
/// compilation, or eigensolve fails.
#[allow(clippy::cast_possible_truncation, unused_assignments)]
pub fn binding_energies_l2_gpu_resident(
    device: &Arc<WgpuDevice>,
    nuclei: &[(usize, usize)],
    params: &[f64],
    max_iter: usize,
    tol: f64,
    mixing: f64,
) -> Result<GpuResidentL2Result, HotSpringError> {
    let t0 = std::time::Instant::now();
    let mut results: Vec<(usize, usize, f64, bool)> = Vec::with_capacity(nuclei.len());
    let mut gpu_dispatches = 0usize;
    let mut total_gpu_dispatches = 0usize;
    let mut n_semf = 0usize;

    let hfb_nuclei: Vec<(usize, usize, usize)> = nuclei
        .iter()
        .enumerate()
        .filter_map(|(idx, &(z, n))| {
            let a = z + n;
            if (56..=132).contains(&a) {
                Some((z, n, idx))
            } else {
                results.push((z, n, semf_binding_energy(z, n, params), true));
                n_semf += 1;
                None
            }
        })
        .collect();

    if hfb_nuclei.is_empty() {
        return Ok(GpuResidentL2Result {
            results,
            hfb_time_s: t0.elapsed().as_secs_f64(),
            gpu_dispatches,
            total_gpu_dispatches,
            n_hfb: 0,
            n_semf,
        });
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
    let c0t = 0.25 * (t1.mul_add(1.0 + x1 / 2.0, t2 * (1.0 + x2 / 2.0)));
    let c1n = 0.25 * (t1.mul_add(0.5 + x1, -(t2 * (0.5 + x2))));
    let hbar2_2m = super::constants::HBAR2_2M;

    let raw_device = device.device();
    let raw_queue = device.queue();

    let potentials_shader = ShaderTemplate::for_device_auto(POTENTIALS_SHADER_BODY, device);
    let pot_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("potentials"),
        source: wgpu::ShaderSource::Wgsl(potentials_shader.into()),
    });
    let ham_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hamiltonian"),
        source: wgpu::ShaderSource::Wgsl(HAMILTONIAN_SHADER.into()),
    });

    let mut states = build_initial_densities(&solvers);

    let ns_nr_groups: Vec<((usize, usize), Vec<usize>)> = groups_map.into_iter().collect();
    let density_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("density"),
        source: wgpu::ShaderSource::Wgsl(DENSITY_SHADER.into()),
    });
    #[cfg(feature = "gpu_energy")]
    let energy_module = {
        let energy_shader = ShaderTemplate::for_device_auto(ENERGY_SHADER_BODY, device);
        raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("energy"),
            source: wgpu::ShaderSource::Wgsl(energy_shader.into()),
        })
    };
    let sky_data: [f64; 9] = [t0_p, t3, x0, x3, alpha_skyrme, 0.0, c0t, c1n, hbar2_2m];

    let mut all_groups: Vec<GroupResources> = Vec::new();
    for ((ns, nr), group_indices) in ns_nr_groups {
        if group_indices.is_empty() {
            continue;
        }
        let dr = solvers[group_indices[0]].3.dr();
        let mut sky_data_mut = sky_data;
        sky_data_mut[5] = dr;
        let n_max = group_indices.len();
        all_groups.push(resources::allocate_group_resources(
            raw_device,
            &group_indices,
            ns,
            nr,
            n_max,
            &solvers,
            sky_data_mut,
            dr,
            &pot_module,
            &ham_module,
            &density_module,
            #[cfg(feature = "gpu_energy")]
            &energy_module,
        ));
    }

    let global_max_ns = all_groups.iter().map(|g| g.ns).max().unwrap_or(1);

    let max_eigh_batch = solvers.len() * 2;
    let (eigh_matrices_buf, eigh_eigenvalues_buf, eigh_eigenvectors_buf) =
        BatchedEighGpu::create_buffers(device, global_max_ns, max_eigh_batch).map_err(|e| {
            HotSpringError::GpuCompute(format!("pre-allocate eigensolve GPU buffers: {e}"))
        })?;

    // ═══ SPIN-ORBIT + PACK PIPELINE (GPU-RESIDENT H → EIGENSOLVE) ═══
    let so_pack_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("so_pack"),
        source: wgpu::ShaderSource::Wgsl(SO_PACK_SHADER.into()),
    });
    let so_pack_layout = raw_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("so_pack_layout"),
        entries: &[
            so_pack_layout_entry(0, wgpu::BufferBindingType::Uniform),
            so_pack_layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            so_pack_layout_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
            so_pack_layout_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });
    let so_pack_pipe = make_pipeline(
        raw_device,
        &so_pack_module,
        "pack_with_spinorbit",
        &[&so_pack_layout],
    );

    let pack_resources: Vec<resources::PackGroupResources> = all_groups
        .iter()
        .map(|g| build_pack_group(raw_device, g, &so_pack_layout, &eigh_matrices_buf))
        .collect();

    // ═══ UNIFIED SCF LOOP ═══
    #[allow(unused_variables)]
    let mut t_gpu_total = 0.0f64;
    #[allow(unused_variables)]
    let t_poll_total = 0.0f64;
    #[allow(unused_variables)]
    let mut t_cpu_total = 0.0f64;
    #[allow(unused_variables)]
    let mut t_upload_total = 0.0f64;

    for iter in 0..max_iter {
        let any_active = all_groups
            .iter()
            .any(|g| g.group_indices.iter().any(|&si| !states[si].converged));
        if !any_active {
            break;
        }

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

        let mut all_work: Vec<WorkItem> = Vec::new();
        for &(gi, ref active) in &active_groups {
            let g = &all_groups[gi];
            for &si in active {
                all_work.push(WorkItem { si, gi, ns: g.ns });
            }
        }

        let t_upload = std::time::Instant::now();
        let n_work = all_work.len();
        dispatch::upload_densities(
            raw_queue,
            &active_groups,
            &all_groups,
            &states,
            &solvers,
            &pack_resources,
            alpha_skyrme,
            w0,
            iter,
            global_max_ns,
        );
        t_upload_total += t_upload.elapsed().as_secs_f64();

        let t_gpu = std::time::Instant::now();
        dispatch::dispatch_hbuild_and_pack(
            raw_device,
            raw_queue,
            &active_groups,
            &all_groups,
            &pack_resources,
            &so_pack_pipe,
            global_max_ns,
            &mut total_gpu_dispatches,
        );
        gpu_dispatches += 1;
        t_gpu_total += t_gpu.elapsed().as_secs_f64();

        let t_read = std::time::Instant::now();
        let gpu_eigen: Option<(Vec<f64>, Vec<f64>)> = if n_work > 0 {
            let batch_size = n_work * 2;
            let dispatch_ok = if global_max_ns <= 32 {
                BatchedEighGpu::execute_single_dispatch_buffers(
                    device,
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
                    device,
                    &eigh_matrices_buf,
                    &eigh_eigenvalues_buf,
                    &eigh_eigenvectors_buf,
                    global_max_ns,
                    batch_size,
                    30,
                )
            };

            Some({
                dispatch_ok.map_err(|e| {
                    HotSpringError::GpuCompute(format!(
                        "GPU eigensolve must succeed — no CPU fallback: {e}"
                    ))
                })?;
                let eigenvalues = BatchedEighGpu::read_eigenvalues(
                    device,
                    &eigh_eigenvalues_buf,
                    global_max_ns,
                    batch_size,
                )
                .map_err(|e| HotSpringError::GpuCompute(format!("eigenvalue readback: {e}")))?;
                let eigenvectors = BatchedEighGpu::read_eigenvectors(
                    device,
                    &eigh_eigenvectors_buf,
                    global_max_ns,
                    batch_size,
                )
                .map_err(|e| HotSpringError::GpuCompute(format!("eigenvector readback: {e}")))?;
                (eigenvalues, eigenvectors)
            })
        } else {
            None
        };

        let eigen_bcs = extract_bcs_results(
            &all_work,
            gpu_eigen.as_ref().ok_or_else(|| {
                HotSpringError::GpuCompute(String::from(
                    "GPU eigensolve must succeed — no CPU fallback",
                ))
            })?,
            &solvers,
            global_max_ns,
        );

        let (mixed_densities, gpu_energies) = dispatch::run_density_mixing_pass(
            raw_device,
            raw_queue,
            &all_groups,
            &eigen_bcs,
            &solvers,
            alpha_mix,
            &mut total_gpu_dispatches,
            #[cfg(feature = "gpu_energy")]
            (t0_p, t3, x0, x3, alpha_skyrme),
        )?;

        let mut group_offsets: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

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

            let e_total = gpu_energies
                .as_ref()
                .and_then(|ge| ge.get(&gi))
                .map_or_else(
                    || {
                        let degs: Vec<f64> = hfb.deg_values();
                        let delta = hfb.pairing_gap();
                        compute_binding_energy(
                            &eb.evals_p,
                            &eb.evals_n,
                            &eb.v2_p,
                            &eb.v2_n,
                            &degs,
                            delta,
                        )
                    },
                    |energies| energies[bi],
                );

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

    Ok(GpuResidentL2Result {
        results,
        hfb_time_s: t0.elapsed().as_secs_f64(),
        gpu_dispatches,
        total_gpu_dispatches,
        n_hfb,
        n_semf,
    })
}

// ── Local helpers for SO-pack setup ──

const fn so_pack_layout_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn build_pack_group(
    raw_device: &wgpu::Device,
    g: &GroupResources,
    so_pack_layout: &wgpu::BindGroupLayout,
    eigh_matrices_buf: &wgpu::Buffer,
) -> resources::PackGroupResources {
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
        layout: so_pack_layout,
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
        layout: so_pack_layout,
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
    resources::PackGroupResources {
        params_p_buf,
        params_n_buf,
        pack_p_bg,
        pack_n_bg,
    }
}
