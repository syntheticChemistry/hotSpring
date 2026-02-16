//! GPU-Resident Spherical HFB Solver (Level 2)
//!
//! Dispatches physics shaders directly on GPU: potentials + H-build.
//! Pipelines and buffers are allocated ONCE per basis-group, then reused
//! across SCF iterations. Only density data is re-uploaded per iteration.
//!
//! Architecture:
//!   1. Pre-compile shader pipelines (once at init)
//!   2. Pre-allocate GPU buffers sized for max batch (once per group)
//!   3. Per SCF iteration:
//!      a. Write densities to GPU (queue.write_buffer — no alloc)
//!      b. GPU: Potentials (Skyrme + Coulomb + f_q)   — 5 dispatches
//!      c. GPU: Hamiltonian H_p and H_n               — 2 dispatches
//!      d. Read back H matrices
//!      e. GPU: BatchedEighGpu eigensolve              — 1 dispatch
//!      f. CPU: BCS + Density + Energy + convergence
//!   4. CPU: Collect results

use super::semf::semf_binding_energy;
use super::hfb::SphericalHFB;
use barracuda::device::WgpuDevice;
use barracuda::ops::linalg::BatchedEighGpu;
use std::sync::Arc;

const POTENTIALS_SHADER: &str = include_str!("shaders/batched_hfb_potentials_f64.wgsl");
const HAMILTONIAN_SHADER: &str = include_str!("shaders/batched_hfb_hamiltonian_f64.wgsl");

#[derive(Debug)]
pub struct GpuResidentL2Result {
    pub results: Vec<(usize, usize, f64, bool)>,
    pub hfb_time_s: f64,
    pub gpu_dispatches: usize,
    pub total_gpu_dispatches: usize,
    pub n_hfb: usize,
    pub n_semf: usize,
}

/// Dimensions uniform for potentials shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PotentialDimsUniform {
    nr: u32,
    batch_size: u32,
}

/// Dimensions uniform for hamiltonian shader
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
                ty: *ty, has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        })
        .collect();

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{} layout", label)),
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label), layout: &layout, entries: &bg_entries,
    });

    (layout, bind_group)
}

fn make_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry_point),
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&pipeline_layout),
        module,
        entry_point,
    })
}

pub fn binding_energies_l2_gpu_resident(
    device: Arc<WgpuDevice>,
    nuclei: &[(usize, usize)],
    params: &[f64],
    max_iter: usize,
    tol: f64,
    mixing: f64,
) -> GpuResidentL2Result {
    let t0_wall = std::time::Instant::now();
    let mut results: Vec<(usize, usize, f64, bool)> = Vec::with_capacity(nuclei.len());
    let mut gpu_dispatches = 0usize;
    let mut total_gpu_dispatches = 0usize;
    let mut n_hfb = 0usize;
    let mut n_semf = 0usize;

    // Partition
    let mut hfb_nuclei: Vec<(usize, usize, usize)> = Vec::new();
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
        return GpuResidentL2Result {
            results, hfb_time_s: t0_wall.elapsed().as_secs_f64(),
            gpu_dispatches, total_gpu_dispatches, n_hfb: 0, n_semf,
        };
    }

    let solvers: Vec<(usize, usize, usize, SphericalHFB)> = hfb_nuclei
        .iter()
        .map(|&(z, n, idx)| (z, n, idx, SphericalHFB::new_adaptive(z, n)))
        .collect();

    // Group by (n_states, nr) for uniform GPU dispatch
    let mut groups: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, (_, _, _, hfb)) in solvers.iter().enumerate() {
        groups.entry((hfb.n_states(), hfb.nr())).or_default().push(i);
    }

    let (t0_p, t1, t2, t3) = (params[0], params[1], params[2], params[3]);
    let (x0, x1, x2, x3) = (params[4], params[5], params[6], params[7]);
    let alpha_skyrme = params[8];
    let c0t = 0.25 * (t1 * (1.0 + x1 / 2.0) + t2 * (1.0 + x2 / 2.0));
    let c1n = 0.25 * (t1 * (0.5 + x1) - t2 * (0.5 + x2));
    let hbar2_2m = super::constants::HBAR2_2M;

    let raw_device = device.device();
    let raw_queue = device.queue();

    // Compile shaders ONCE
    let pot_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("potentials"),
        source: wgpu::ShaderSource::Wgsl(POTENTIALS_SHADER.into()),
    });
    let ham_module = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hamiltonian"),
        source: wgpu::ShaderSource::Wgsl(HAMILTONIAN_SHADER.into()),
    });

    // Per-nucleus SCF state
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
            let rho_p: Vec<f64> = (0..nr).map(|k| {
                let r = (k + 1) as f64 * (15.0 / nr as f64);
                if r < r_nuc { (rho0 * *z as f64 / a as f64).max(1e-15) } else { 1e-15 }
            }).collect();
            let rho_n: Vec<f64> = (0..nr).map(|k| {
                let r = (k + 1) as f64 * (15.0 / nr as f64);
                if r < r_nuc { (rho0 * *n as f64 / a as f64).max(1e-15) } else { 1e-15 }
            }).collect();
            NucleusState { rho_p, rho_n, e_prev: 1e10, converged: false, binding_energy: 0.0 }
        })
        .collect();

    let ns_nr_groups: Vec<((usize, usize), Vec<usize>)> = groups.into_iter().collect();

    for &((ns, nr), ref group_indices) in &ns_nr_groups {
        if group_indices.is_empty() { continue; }

        let dr = solvers[group_indices[0]].3.dr();
        let mat_size = ns * ns;
        let n_max = group_indices.len();  // max batch size for this group

        // ═══════════════════════════════════════════════════════════
        // PRE-ALLOCATE: buffers sized for max batch, created ONCE
        // ═══════════════════════════════════════════════════════════
        use wgpu::util::DeviceExt;

        // Per-nucleus static data (packed for all nuclei in group)
        let mut all_wf = Vec::with_capacity(n_max * ns * nr);
        let mut all_dwf = Vec::with_capacity(n_max * ns * nr);
        let mut all_r_grid = Vec::with_capacity(n_max * nr);
        let mut all_lj_same = Vec::with_capacity(n_max * ns * ns);
        let mut all_ll1 = Vec::with_capacity(n_max * ns);

        for &si in group_indices {
            let hfb = &solvers[si].3;
            all_wf.extend_from_slice(&hfb.wf_flat());
            all_dwf.extend_from_slice(&hfb.dwf_flat());
            all_r_grid.extend_from_slice(hfb.r_grid());
            all_lj_same.extend_from_slice(&hfb.lj_same_flat());
            all_ll1.extend_from_slice(&hfb.ll1_values());
        }

        // Index map: solver_idx → position in group
        let group_idx_map: std::collections::HashMap<usize, usize> = group_indices
            .iter().enumerate().map(|(gi, &si)| (si, gi)).collect();

        // Static GPU buffers (never change)
        let wf_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wf"), contents: bytemuck::cast_slice(&all_wf),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dwf_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dwf"), contents: bytemuck::cast_slice(&all_dwf),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let r_grid_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("r_grid"), contents: bytemuck::cast_slice(&all_r_grid),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lj_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lj_same"), contents: bytemuck::cast_slice(&all_lj_same),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ll1_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ll1"), contents: bytemuck::cast_slice(&all_ll1),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Skyrme params (constant)
        let sky_data: [f64; 9] = [t0_p, t3, x0, x3, alpha_skyrme, dr, c0t, c1n, hbar2_2m];
        let sky_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky"), contents: bytemuck::cast_slice(&sky_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // dr as f64 storage for hamiltonian
        let dr_data: [f64; 1] = [dr];
        let dr_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dr"), contents: bytemuck::cast_slice(&dr_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Dynamic buffers — allocated for n_max, re-written each iteration
        let buf_size_nr = (n_max * nr * 8) as u64;
        let rho_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_p"), size: buf_size_nr,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rho_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rho_n"), size: buf_size_nr,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Intermediate GPU-only buffers (no CPU readback needed)
        let u_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u_p"), size: buf_size_nr,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let u_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u_n"), size: buf_size_nr,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let fq_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fq_p"), size: buf_size_nr,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let fq_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fq_n"), size: buf_size_nr,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let cenc_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cenc"), size: buf_size_nr,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let phi_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phi"), size: buf_size_nr,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });

        // H outputs — need COPY_SRC for readback
        let h_p_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_p"), size: (n_max * mat_size * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let h_n_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_n"), size: (n_max * mat_size * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Uniform buffers — updated when batch size changes
        let pot_dims = PotentialDimsUniform { nr: nr as u32, batch_size: n_max as u32 };
        let pot_dims_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pot_dims"), contents: bytemuck::bytes_of(&pot_dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let ham_dims = HamiltonianDimsUniform {
            n_states: ns as u32, nr: nr as u32, batch_size: n_max as u32, _pad: 0,
        };
        let ham_dims_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ham_dims"), contents: bytemuck::bytes_of(&ham_dims),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let u = wgpu::BufferBindingType::Uniform;
        let sr = wgpu::BufferBindingType::Storage { read_only: true };
        let srw = wgpu::BufferBindingType::Storage { read_only: false };

        // Create bind groups and pipelines ONCE
        let (pot_layout, pot_bg) = make_bind_group(raw_device, "pot", &[
            (u, &pot_dims_buf), (sr, &rho_p_buf), (sr, &rho_n_buf),
            (srw, &u_p_buf), (srw, &u_n_buf), (sr, &r_grid_buf),
            (srw, &fq_p_buf), (srw, &fq_n_buf),
            (srw, &cenc_buf), (srw, &phi_buf), (sr, &sky_buf),
        ]);
        let (ham_layout_p, ham_bg_p) = make_bind_group(raw_device, "ham_p", &[
            (u, &ham_dims_buf), (sr, &wf_buf), (sr, &dwf_buf),
            (sr, &u_p_buf), (sr, &fq_p_buf), (sr, &r_grid_buf),
            (sr, &lj_buf), (sr, &ll1_buf), (srw, &h_p_buf), (sr, &dr_buf),
        ]);
        let (ham_layout_n, ham_bg_n) = make_bind_group(raw_device, "ham_n", &[
            (u, &ham_dims_buf), (sr, &wf_buf), (sr, &dwf_buf),
            (sr, &u_n_buf), (sr, &fq_n_buf), (sr, &r_grid_buf),
            (sr, &lj_buf), (sr, &ll1_buf), (srw, &h_n_buf), (sr, &dr_buf),
        ]);

        let sky_pipe = make_pipeline(raw_device, &pot_module, "compute_skyrme", &[&pot_layout]);
        let cfwd_pipe = make_pipeline(raw_device, &pot_module, "compute_coulomb_forward", &[&pot_layout]);
        let cbwd_pipe = make_pipeline(raw_device, &pot_module, "compute_coulomb_backward", &[&pot_layout]);
        let fin_pipe = make_pipeline(raw_device, &pot_module, "finalize_proton_potential", &[&pot_layout]);
        let fq_pipe = make_pipeline(raw_device, &pot_module, "compute_f_q", &[&pot_layout]);
        let hp_pipe = make_pipeline(raw_device, &ham_module, "build_hamiltonian", &[&ham_layout_p]);
        let hn_pipe = make_pipeline(raw_device, &ham_module, "build_hamiltonian", &[&ham_layout_n]);

        let nr_wg = (nr as u32 + 255) / 256;
        let ns_wg = (ns as u32 + 15) / 16;

        // ═══════════════════════════════════════════════════════════
        // SCF LOOP — only write_buffer + dispatch + readback per iter
        // ═══════════════════════════════════════════════════════════
        for iter in 0..max_iter {
            let active: Vec<usize> = group_indices
                .iter().copied().filter(|&i| !states[i].converged).collect();
            if active.is_empty() { break; }

            let n_nuclei = active.len();
            let bn = n_nuclei as u32;

            // Update batch_size in uniforms if changed (nuclei converge → batch shrinks)
            if n_nuclei != n_max || iter == 0 {
                let dims_update = PotentialDimsUniform { nr: nr as u32, batch_size: bn };
                raw_queue.write_buffer(&pot_dims_buf, 0, bytemuck::bytes_of(&dims_update));
                let hdims_update = HamiltonianDimsUniform {
                    n_states: ns as u32, nr: nr as u32, batch_size: bn, _pad: 0,
                };
                raw_queue.write_buffer(&ham_dims_buf, 0, bytemuck::bytes_of(&hdims_update));
            }

            // Write densities (only active nuclei, packed contiguously)
            let mut rho_p_flat = vec![0.0f64; n_nuclei * nr];
            let mut rho_n_flat = vec![0.0f64; n_nuclei * nr];
            for (bi, &si) in active.iter().enumerate() {
                rho_p_flat[bi * nr..(bi + 1) * nr].copy_from_slice(&states[si].rho_p);
                rho_n_flat[bi * nr..(bi + 1) * nr].copy_from_slice(&states[si].rho_n);
            }
            raw_queue.write_buffer(&rho_p_buf, 0, bytemuck::cast_slice(&rho_p_flat));
            raw_queue.write_buffer(&rho_n_buf, 0, bytemuck::cast_slice(&rho_n_flat));

            // ─── GPU: 7 dispatches in single command encoder ─────
            let mut encoder = raw_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scf"),
            });
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("scf_pass"), timestamp_writes: None,
                });
                p.set_pipeline(&sky_pipe);
                p.set_bind_group(0, &pot_bg, &[]);
                p.dispatch_workgroups(nr_wg, bn, 1);

                p.set_pipeline(&cfwd_pipe);
                p.set_bind_group(0, &pot_bg, &[]);
                p.dispatch_workgroups(bn, 1, 1);

                p.set_pipeline(&cbwd_pipe);
                p.set_bind_group(0, &pot_bg, &[]);
                p.dispatch_workgroups(bn, 1, 1);

                p.set_pipeline(&fin_pipe);
                p.set_bind_group(0, &pot_bg, &[]);
                p.dispatch_workgroups(nr_wg, bn, 1);

                p.set_pipeline(&fq_pipe);
                p.set_bind_group(0, &pot_bg, &[]);
                p.dispatch_workgroups(nr_wg, bn, 1);

                p.set_pipeline(&hp_pipe);
                p.set_bind_group(0, &ham_bg_p, &[]);
                p.dispatch_workgroups(ns_wg, ns_wg, bn);

                p.set_pipeline(&hn_pipe);
                p.set_bind_group(0, &ham_bg_n, &[]);
                p.dispatch_workgroups(ns_wg, ns_wg, bn);
            }
            raw_queue.submit(Some(encoder.finish()));
            total_gpu_dispatches += 7;

            // Read back H matrices
            let h_p_data = device.read_buffer_f64(&h_p_buf, n_nuclei * mat_size)
                .expect("read H_p");
            let h_n_data = device.read_buffer_f64(&h_n_buf, n_nuclei * mat_size)
                .expect("read H_n");

            // Interleave for BatchedEighGpu: [p0, n0, p1, n1, ...]
            let batch_species = n_nuclei * 2;
            let mut packed_h = vec![0.0f64; batch_species * mat_size];
            for bi in 0..n_nuclei {
                let src = bi * mat_size;
                packed_h[(bi * 2) * mat_size..(bi * 2) * mat_size + mat_size]
                    .copy_from_slice(&h_p_data[src..src + mat_size]);
                packed_h[(bi * 2 + 1) * mat_size..(bi * 2 + 1) * mat_size + mat_size]
                    .copy_from_slice(&h_n_data[src..src + mat_size]);
            }

            // GPU eigensolve
            let (eigenvalues, eigenvectors) = match BatchedEighGpu::execute_f64(
                device.clone(), &packed_h, ns, batch_species, 30,
            ) {
                Ok(r) => r,
                Err(e) => { eprintln!("BatchedEighGpu failed: {}", e); break; }
            };
            gpu_dispatches += 1;
            total_gpu_dispatches += 1;

            // CPU: BCS + Density + Energy
            for (batch_idx, &solver_idx) in active.iter().enumerate() {
                let (z, n, _, ref hfb) = solvers[solver_idx];
                let state = &mut states[solver_idx];
                let cur_nr = hfb.nr();

                let eig_p = (batch_idx * 2) * ns;
                let vec_p = (batch_idx * 2) * mat_size;
                let eig_n = (batch_idx * 2 + 1) * ns;
                let vec_n = (batch_idx * 2 + 1) * mat_size;

                let mut rho_p_new = vec![1e-15; cur_nr];
                let mut rho_n_new = vec![1e-15; cur_nr];

                for (_, is_proton) in [(0usize, true), (1usize, false)] {
                    let eo = if is_proton { eig_p } else { eig_n };
                    let vo = if is_proton { vec_p } else { vec_n };
                    let eigs = &eigenvalues[eo..eo + ns];
                    let vecs = &eigenvectors[vo..vo + mat_size];
                    let num_q = if is_proton { z } else { n };
                    let (v2, _) = hfb.bcs_occupations_from_eigs(eigs, num_q, hfb.pairing_gap());
                    let rho_new = hfb.density_from_eigenstates(vecs, &v2, ns);
                    if is_proton { rho_p_new = rho_new; } else { rho_n_new = rho_new; }
                }

                let alpha = if iter == 0 { 0.8 } else { mixing };
                for k in 0..cur_nr {
                    state.rho_p[k] = (alpha * rho_p_new[k] + (1.0 - alpha) * state.rho_p[k]).max(1e-15);
                    state.rho_n[k] = (alpha * rho_n_new[k] + (1.0 - alpha) * state.rho_n[k]).max(1e-15);
                }

                let e_total = hfb.compute_energy_from_densities(
                    &state.rho_p, &state.rho_n,
                    &eigenvalues[eig_p..eig_p + ns],
                    &eigenvectors[vec_p..vec_p + mat_size],
                    &eigenvalues[eig_n..eig_n + ns],
                    &eigenvectors[vec_n..vec_n + mat_size],
                    params,
                );

                let de = (e_total - state.e_prev).abs();
                state.e_prev = e_total;
                state.binding_energy = if e_total < 0.0 { -e_total } else { e_total.abs() };
                if de < tol && iter > 5 { state.converged = true; }
            }
        }
    }

    n_hfb = solvers.len();
    for (i, (z, n, _, _)) in solvers.iter().enumerate() {
        results.push((*z, *n, states[i].binding_energy, states[i].converged));
    }

    GpuResidentL2Result {
        results,
        hfb_time_s: t0_wall.elapsed().as_secs_f64(),
        gpu_dispatches,
        total_gpu_dispatches,
        n_hfb,
        n_semf,
    }
}
