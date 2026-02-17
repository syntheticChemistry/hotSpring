// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-Resident Spherical HFB Solver (Level 2) — Optimized Hybrid
//!
//! Per SCF iteration — ONE encoder, ONE submit, ONE poll:
//!   1. CPU→GPU: write densities for all groups
//!   2. GPU compute: potentials + H-build (7 dispatches per group)
//!   3. GPU copy: H → staging buffers (in same encoder)
//!   4. Single queue.submit + single device.poll(Wait)
//!   5. CPU: read staging → spin-orbit correction
//!   6. GPU: `BatchedEighGpu::execute_single_dispatch` for ALL matrices
//!   7. CPU (Rayon): BCS → density → energy
//!
//! Pre-allocated staging buffers eliminate per-iteration allocation.
//! Single poll per iteration eliminates serial GPU sync overhead.
//! GPU eigensolve (v0.5.4) replaces CPU `eigh_f64` with single-dispatch
//! Jacobi rotation — all proton+neutron matrices in one shader invocation.
//! Falls back to CPU `eigh_f64` if `global_max_ns > 32` or GPU fails.

use super::hfb::SphericalHFB;
use super::semf::semf_binding_energy;
use crate::tolerances::{DENSITY_FLOOR, GPU_JACOBI_CONVERGENCE, RHO_POWF_GUARD, SPIN_ORBIT_R_MIN};
use barracuda::device::WgpuDevice;
use barracuda::linalg::eigh_f64;
use barracuda::ops::grid::{compute_ls_factor, SpinOrbitGpu};
use barracuda::ops::linalg::BatchedEighGpu;
use barracuda::shaders::precision::ShaderTemplate;
use rayon::prelude::*;
use std::sync::Arc;

const POTENTIALS_SHADER_BODY: &str = include_str!("shaders/batched_hfb_potentials_f64.wgsl");
const HAMILTONIAN_SHADER: &str = include_str!("shaders/batched_hfb_hamiltonian_f64.wgsl");

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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PotentialDimsUniform {
    nr: u32,
    batch_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HamiltonianDimsUniform {
    n_states: u32,
    nr: u32,
    batch_size: u32,
    _pad: u32,
}

fn make_bind_group(
    device: &wgpu::Device,
    label: &str,
    entries: &[(wgpu::BufferBindingType, &wgpu::Buffer)],
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = entries
        .iter()
        .enumerate()
        .map(|(i, (ty, _))| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: *ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        entries: &layout_entries,
    });
    let bg_entries: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .enumerate()
        .map(|(i, (_, buf))| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &layout,
        entries: &bg_entries,
    });
    (layout, bg)
}

fn make_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry_point),
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&pl),
        module,
        entry_point,
    })
}

struct GroupResources {
    ns: usize,
    nr: usize,
    group_indices: Vec<usize>,
    n_max: usize,
    mat_size: usize,

    rho_p_buf: wgpu::Buffer,
    rho_n_buf: wgpu::Buffer,
    rho_alpha_buf: wgpu::Buffer,
    rho_alpha_m1_buf: wgpu::Buffer,
    pot_dims_buf: wgpu::Buffer,
    ham_dims_buf: wgpu::Buffer,
    h_p_buf: wgpu::Buffer,
    h_n_buf: wgpu::Buffer,

    staging_p: wgpu::Buffer,
    staging_n: wgpu::Buffer,

    pot_bg: wgpu::BindGroup,
    hbg_p: wgpu::BindGroup,
    hbg_n: wgpu::BindGroup,

    sky_pipe: wgpu::ComputePipeline,
    cfwd: wgpu::ComputePipeline,
    cbwd: wgpu::ComputePipeline,
    fin_pipe: wgpu::ComputePipeline,
    fq_pipe: wgpu::ComputePipeline,
    hp_pipe: wgpu::ComputePipeline,
    hn_pipe: wgpu::ComputePipeline,

    nr_wg: u32,
    ns_wg: u32,
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

    let potentials_shader = ShaderTemplate::with_math_f64_auto(POTENTIALS_SHADER_BODY);
    let pot_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("potentials"),
        source: wgpu::ShaderSource::Wgsl(potentials_shader.into()),
    });
    let ham_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hamiltonian"),
        source: wgpu::ShaderSource::Wgsl(HAMILTONIAN_SHADER.into()),
    });

    struct NucleusState {
        rho_p: Vec<f64>,
        rho_n: Vec<f64>,
        e_prev: f64,
        converged: bool,
        binding_energy: f64,
    }

    let mut states: Vec<NucleusState> = solvers
        .iter()
        .map(|(z, n, _, hfb)| {
            let a = z + n;
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
        .collect();

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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rho_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_n"),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        // Pre-allocated staging buffers for readback (reused every iteration)
        let staging_p = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_p"),
            size: h_mat_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_n = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_n"),
            size: h_mat_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
            staging_p,
            staging_n,
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
            nr_wg: (nr as u32).div_ceil(256),
            ns_wg: (ns as u32).div_ceil(16),
        });
    }

    let global_max_ns = all_groups.iter().map(|g| g.ns).max().unwrap_or(1);

    // ═══ UNIFIED SCF LOOP — ALL GROUPS IN ONE ENCODER ═══
    let mut t_gpu_total = 0.0f64;
    let mut t_poll_total = 0.0f64;
    let _t_read_total = 0.0f64;
    let mut t_cpu_total = 0.0f64;
    let mut t_upload_total = 0.0f64;

    for iter in 0..max_iter {
        let any_active = all_groups
            .iter()
            .any(|g| g.group_indices.iter().any(|&si| !states[si].converged));
        if !any_active {
            break;
        }

        // Step 1: Upload densities for all groups
        let t_upload = std::time::Instant::now();
        for g in &all_groups {
            let active: Vec<usize> = g
                .group_indices
                .iter()
                .copied()
                .filter(|&si| !states[si].converged)
                .collect();
            if active.is_empty() {
                continue;
            }
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
        }

        t_upload_total += t_upload.elapsed().as_secs_f64();

        // Step 2: ONE encoder for ALL groups — compute + copy
        let t_gpu = std::time::Instant::now();
        let mut encoder = raw_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scf_all"),
        });

        let mut active_groups: Vec<(usize, Vec<usize>)> = Vec::new();

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("all_groups"),
                timestamp_writes: None,
            });

            for (gi, g) in all_groups.iter().enumerate() {
                let active: Vec<usize> = g
                    .group_indices
                    .iter()
                    .copied()
                    .filter(|&si| !states[si].converged)
                    .collect();
                if active.is_empty() {
                    continue;
                }
                let bn = active.len() as u32;

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

                total_gpu_dispatches += 7;
                active_groups.push((gi, active));
            }
        }

        // Copy H → staging (same encoder, no extra submit)
        for &(gi, ref active) in &active_groups {
            let g = &all_groups[gi];
            let byte_count = (active.len() * g.mat_size * 8) as u64;
            encoder.copy_buffer_to_buffer(&g.h_p_buf, 0, &g.staging_p, 0, byte_count);
            encoder.copy_buffer_to_buffer(&g.h_n_buf, 0, &g.staging_n, 0, byte_count);
        }

        // Step 3: SINGLE submit
        raw_queue.submit(Some(encoder.finish()));
        gpu_dispatches += 1;
        t_gpu_total += t_gpu.elapsed().as_secs_f64();

        // Step 4: Map ALL staging buffers, SINGLE poll
        let t_poll = std::time::Instant::now();
        let mut map_receivers = Vec::new();
        for &(gi, ref active) in &active_groups {
            let g = &all_groups[gi];
            let byte_count = (active.len() * g.mat_size * 8) as u64;
            let slice_p = g.staging_p.slice(..byte_count);
            let slice_n = g.staging_n.slice(..byte_count);
            let (tx_p, rx_p) = std::sync::mpsc::channel();
            let (tx_n, rx_n) = std::sync::mpsc::channel();
            slice_p.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_p.send(r);
            });
            slice_n.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_n.send(r);
            });
            map_receivers.push((gi, rx_p, rx_n));
        }

        raw_device.poll(wgpu::Maintain::Wait);
        t_poll_total += t_poll.elapsed().as_secs_f64();

        // Step 5: Read ALL staging buffers (sequential), then single Rayon pass
        let t_read = std::time::Instant::now();
        let alpha_mix = if iter == 0 { 0.8 } else { mixing };

        // 5a. Read all H matrices from staging buffers
        struct WorkItem {
            si: usize,
            ns: usize,
            h_p: Vec<f64>,
            h_n: Vec<f64>,
            rho_p_old: Vec<f64>,
            rho_n_old: Vec<f64>,
            e_prev: f64,
        }
        let mut all_work: Vec<WorkItem> = Vec::new();

        for (gi, rx_p, rx_n) in map_receivers {
            rx_p.recv()
                .map_err(|_| {
                    "GPU map callback: channel recv failed for H_p (sender dropped)".to_string()
                })
                .and_then(|r| r.map_err(|e| format!("GPU buffer map failed for H_p: {e}")))
                .expect("GPU staging buffer map for H_p");
            rx_n.recv()
                .map_err(|_| {
                    "GPU map callback: channel recv failed for H_n (sender dropped)".to_string()
                })
                .and_then(|r| r.map_err(|e| format!("GPU buffer map failed for H_n: {e}")))
                .expect("GPU staging buffer map for H_n");

            let g = &all_groups[gi];
            let active: Vec<usize> = g
                .group_indices
                .iter()
                .copied()
                .filter(|&si| !states[si].converged)
                .collect();
            let n_nuclei = active.len();
            let byte_count = (n_nuclei * g.mat_size * 8) as u64;

            let h_p_data: Vec<f64>;
            let h_n_data: Vec<f64>;
            {
                let mapped_p = g.staging_p.slice(..byte_count).get_mapped_range();
                h_p_data =
                    bytemuck::cast_slice::<u8, f64>(&mapped_p)[..n_nuclei * g.mat_size].to_vec();
            }
            g.staging_p.unmap();
            {
                let mapped_n = g.staging_n.slice(..byte_count).get_mapped_range();
                h_n_data =
                    bytemuck::cast_slice::<u8, f64>(&mapped_n)[..n_nuclei * g.mat_size].to_vec();
            }
            g.staging_n.unmap();

            for (bi, &si) in active.iter().enumerate() {
                all_work.push(WorkItem {
                    si,
                    ns: g.ns,
                    h_p: h_p_data[bi * g.mat_size..(bi + 1) * g.mat_size].to_vec(),
                    h_n: h_n_data[bi * g.mat_size..(bi + 1) * g.mat_size].to_vec(),
                    rho_p_old: states[si].rho_p.clone(),
                    rho_n_old: states[si].rho_n.clone(),
                    e_prev: states[si].e_prev,
                });
            }
        }

        // 5b. Apply spin-orbit corrections via SpinOrbitGpu
        //
        // Uses barracuda::ops::grid::SpinOrbitGpu which computes:
        //   V_so[i] = w0 * ls_i * ∫ |ψ_i(r)|² (dρ/dr)/r r² dr
        // Returns per-state diagonal contributions to H.
        let spin_orbit_gpu = if w0 == 0.0 {
            None
        } else {
            Some(SpinOrbitGpu::new(device.clone()))
        };
        for item in &mut all_work {
            if w0 == 0.0 {
                continue;
            }
            let (_, _, _, ref hfb) = solvers[item.si];
            let ns = item.ns;
            let cur_nr = hfb.nr();
            let cur_dr = hfb.dr();
            let rho_total: Vec<f64> = (0..cur_nr)
                .map(|k| item.rho_p_old[k] + item.rho_n_old[k])
                .collect();
            let drho = barracuda::numerical::gradient_1d(&rho_total, cur_dr);
            let r = hfb.r_grid();
            let lj_qn = hfb.lj_quantum_numbers();

            // Build flat arrays for SpinOrbitGpu
            let wf_flat = hfb.wf_flat();
            let wf_squared: Vec<f64> = wf_flat.iter().map(|v| v * v).collect();
            let ls_factors: Vec<f64> = lj_qn
                .iter()
                .map(|&(l, j)| compute_ls_factor(l as u32, j))
                .collect();

            match spin_orbit_gpu
                .as_ref()
                .expect("spin_orbit_gpu initialized when w0 != 0")
                .compute(&wf_squared, &drho, r, &ls_factors, cur_dr, w0)
            {
                Ok(so_diag) => {
                    for i in 0..ns {
                        item.h_p[i * ns + i] += so_diag[i];
                        item.h_n[i * ns + i] += so_diag[i];
                    }
                }
                Err(_) => {
                    // Fallback: CPU spin-orbit (same physics, no GPU)
                    for i in 0..ns {
                        let (l, j) = lj_qn[i];
                        if l == 0 {
                            continue;
                        }
                        let ls = compute_ls_factor(l as u32, j);
                        let wf_i = hfb.wf_state(i);
                        let so_integ: Vec<f64> = (0..cur_nr)
                            .map(|k| {
                                wf_i[k].powi(2) * drho[k] / r[k].max(SPIN_ORBIT_R_MIN)
                                    * r[k].powi(2)
                            })
                            .collect();
                        let so_val =
                            w0 * ls * barracuda::numerical::trapz(&so_integ, r).unwrap_or(0.0);
                        item.h_p[i * ns + i] += so_val;
                        item.h_n[i * ns + i] += so_val;
                    }
                }
            }
        }

        // 5c. GPU batched eigensolve — single-dispatch for ALL H_p + H_n
        let n_work = all_work.len();
        let gpu_eigen = if global_max_ns <= 32 && n_work > 0 {
            let batch_size = n_work * 2;
            let gm = global_max_ns * global_max_ns;
            let mut packed = vec![0.0f64; batch_size * gm];
            for (wi, item) in all_work.iter().enumerate() {
                let ns = item.ns;
                for (species, h) in [(0, &item.h_p), (1, &item.h_n)] {
                    let off = (wi * 2 + species) * gm;
                    for r in 0..ns {
                        for c in 0..ns {
                            packed[off + r * global_max_ns + c] = h[r * ns + c];
                        }
                    }
                    for k in ns..global_max_ns {
                        packed[off + k * global_max_ns + k] = 1e10;
                    }
                }
            }
            BatchedEighGpu::execute_single_dispatch(
                device.clone(),
                &packed,
                global_max_ns,
                batch_size,
                30,
                GPU_JACOBI_CONVERGENCE,
            )
            .ok()
        } else {
            None
        };

        // 5d. Rayon par_iter: BCS + density + energy
        let cpu_results: Vec<(usize, Vec<f64>, Vec<f64>, f64, f64, bool)> = all_work
            .par_iter()
            .enumerate()
            .map(|(wi, item)| {
                let (z, n, _, ref hfb) = solvers[item.si];
                let ns = item.ns;
                let cur_nr = hfb.nr();
                let gm = global_max_ns * global_max_ns;

                let (evals_p, evecs_p, evals_n, evecs_n) =
                    if let Some((ref eigenvalues, ref eigenvectors)) = gpu_eigen {
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
                        let (ep, vp) = extract(0);
                        let (en, vn) = extract(1);
                        (ep, vp, en, vn)
                    } else {
                        let eig_p = eigh_f64(&item.h_p, ns)
                            .expect("eigh_f64 failed for proton Hamiltonian");
                        let eig_n = eigh_f64(&item.h_n, ns)
                            .expect("eigh_f64 failed for neutron Hamiltonian");
                        (
                            eig_p.eigenvalues,
                            eig_p.eigenvectors,
                            eig_n.eigenvalues,
                            eig_n.eigenvectors,
                        )
                    };

                let (v2_p, _) = hfb.bcs_occupations_from_eigs(&evals_p, z, hfb.pairing_gap());
                let (v2_n, _) = hfb.bcs_occupations_from_eigs(&evals_n, n, hfb.pairing_gap());

                let rho_p_new = hfb.density_from_eigenstates(&evecs_p, &v2_p, ns);
                let rho_n_new = hfb.density_from_eigenstates(&evecs_n, &v2_n, ns);

                let mut rho_p_mixed = vec![0.0; cur_nr];
                let mut rho_n_mixed = vec![0.0; cur_nr];
                for k in 0..cur_nr {
                    rho_p_mixed[k] = (alpha_mix * rho_p_new[k]
                        + (1.0 - alpha_mix) * item.rho_p_old[k])
                        .max(DENSITY_FLOOR);
                    rho_n_mixed[k] = (alpha_mix * rho_n_new[k]
                        + (1.0 - alpha_mix) * item.rho_n_old[k])
                        .max(DENSITY_FLOOR);
                }

                let e_total = hfb.compute_energy_with_v2(
                    &rho_p_mixed,
                    &rho_n_mixed,
                    &evals_p,
                    &evecs_p,
                    &evals_n,
                    &evecs_n,
                    &v2_p,
                    &v2_n,
                    params,
                );
                let de = (e_total - item.e_prev).abs();
                let binding_energy = if e_total < 0.0 {
                    -e_total
                } else {
                    e_total.abs()
                };
                let converged = de < tol && iter > 5;

                (
                    item.si,
                    rho_p_mixed,
                    rho_n_mixed,
                    e_total,
                    binding_energy,
                    converged,
                )
            })
            .collect();

        for (si, rho_p_mixed, rho_n_mixed, e_total, binding_energy, converged) in cpu_results {
            states[si].rho_p = rho_p_mixed;
            states[si].rho_n = rho_n_mixed;
            states[si].e_prev = e_total;
            states[si].binding_energy = binding_energy;
            states[si].converged = converged;
        }
        t_cpu_total += t_read.elapsed().as_secs_f64();
    }

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
    fn density_floor_applied() {
        use crate::tolerances::DENSITY_FLOOR;
        // Verify density floor logic
        let rho = -0.001_f64;
        let floored = rho.max(DENSITY_FLOOR);
        assert_eq!(floored, DENSITY_FLOOR);
        assert!(floored > 0.0);
    }

    #[test]
    fn spin_orbit_r_min_guard() {
        use crate::tolerances::SPIN_ORBIT_R_MIN;
        let r = 0.001_f64;
        let guarded = r.max(SPIN_ORBIT_R_MIN);
        assert_eq!(guarded, SPIN_ORBIT_R_MIN);
        assert!(guarded >= 0.1);
    }
}
