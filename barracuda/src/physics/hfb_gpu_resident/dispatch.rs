// SPDX-License-Identifier: AGPL-3.0-only

//! GPU dispatch routines for the HFB SCF loop.
//!
//! Three phases per SCF iteration:
//! 1. **Upload** — CPU densities + spin-orbit diag → GPU buffers
//! 2. **H-build + SO-pack** — potential → Hamiltonian → spin-orbit pack (single encoder)
//! 3. **Density mixing** — BCS v² → density → mix → staging readback

use super::super::hfb::SphericalHFB;
#[cfg(feature = "gpu_energy")]
use super::super::hfb_gpu_types::EnergyParamsUniform;
use super::super::hfb_gpu_types::{
    DensityParamsUniform, GroupResources, HamiltonianDimsUniform, MixParamsUniform, PackParams,
    PotentialDimsUniform,
};
use super::resources::PackGroupResources;
use super::types::{compute_spin_orbit_diagonal, EigenBcsResult, NucleusState};
use crate::error::HotSpringError;
use crate::tolerances::{DENSITY_FLOOR, RHO_POWF_GUARD};

/// Upload densities, spin-orbit diagonals, and pack params to GPU.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::too_many_arguments)]
pub(super) fn upload_densities(
    raw_queue: &wgpu::Queue,
    active_groups: &[(usize, Vec<usize>)],
    all_groups: &[GroupResources],
    states: &[NucleusState],
    solvers: &[(usize, usize, usize, SphericalHFB)],
    pack_resources: &[PackGroupResources],
    alpha_skyrme: f64,
    w0: f64,
    iter: usize,
    global_max_ns: usize,
) {
    for &(gi, ref active) in active_groups {
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

    let mut cum_wi = 0usize;
    for &(gi, ref active) in active_groups {
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

/// Dispatch H-build and SO-pack GPU passes in a single encoder.
#[allow(clippy::cast_possible_truncation)]
pub(super) fn dispatch_hbuild_and_pack(
    raw_device: &wgpu::Device,
    raw_queue: &wgpu::Queue,
    active_groups: &[(usize, Vec<usize>)],
    all_groups: &[GroupResources],
    pack_resources: &[PackGroupResources],
    so_pack_pipe: &wgpu::ComputePipeline,
    global_max_ns: usize,
    total_gpu_dispatches: &mut usize,
) {
    let gns_wg = (global_max_ns as u32).div_ceil(16);
    let mut encoder = raw_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("scf_hbuild_pack"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hbuild_pack"),
            timestamp_writes: None,
        });

        for &(gi, ref active) in active_groups {
            let g = &all_groups[gi];
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

            let pr = &pack_resources[gi];
            pass.set_pipeline(so_pack_pipe);
            pass.set_bind_group(0, &pr.pack_p_bg, &[]);
            pass.dispatch_workgroups(gns_wg, gns_wg, bn);
            pass.set_bind_group(0, &pr.pack_n_bg, &[]);
            pass.dispatch_workgroups(gns_wg, gns_wg, bn);

            *total_gpu_dispatches += 9;
        }
    }

    raw_queue.submit(Some(encoder.finish()));
}

/// Dispatch BCS v², density, mixing (and optional energy) passes, then
/// readback mixed densities via staging buffers.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::too_many_arguments)]
pub(super) fn run_density_mixing_pass(
    raw_device: &wgpu::Device,
    raw_queue: &wgpu::Queue,
    all_groups: &[GroupResources],
    eigen_bcs: &[EigenBcsResult],
    solvers: &[(usize, usize, usize, SphericalHFB)],
    alpha_mix: f64,
    total_gpu_dispatches: &mut usize,
    #[cfg(feature = "gpu_energy")] sky_params: (f64, f64, f64, f64, f64),
) -> Result<
    (
        std::collections::HashMap<usize, (Vec<f64>, Vec<f64>)>,
        Option<std::collections::HashMap<usize, Vec<f64>>>,
    ),
    HotSpringError,
> {
    let mut density_encoder = raw_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

        upload_eigen_bcs_data(raw_queue, g, &items, solvers);
        dispatch_bcs_density_mix(&mut density_encoder, g, bn);

        #[cfg(feature = "gpu_energy")]
        dispatch_energy_pass(
            raw_queue,
            &mut density_encoder,
            g,
            bn,
            solvers,
            sky_params,
            total_gpu_dispatches,
        );

        let rho_bytes = (items.len() * g.nr * 8) as u64;
        density_encoder.copy_buffer_to_buffer(&g.rho_p_buf, 0, &g.rho_p_staging, 0, rho_bytes);
        density_encoder.copy_buffer_to_buffer(&g.rho_n_buf, 0, &g.rho_n_staging, 0, rho_bytes);

        #[cfg(feature = "gpu_energy")]
        {
            let energy_bytes = (items.len() * g.nr * 8) as u64;
            let e_pair_bytes = (items.len() * 8) as u64;
            density_encoder.copy_buffer_to_buffer(
                &g.energy_integrands_buf,
                0,
                &g.energy_staging,
                0,
                energy_bytes,
            );
            density_encoder.copy_buffer_to_buffer(
                &g.e_pair_buf,
                0,
                &g.e_pair_staging,
                0,
                e_pair_bytes,
            );
        }

        *total_gpu_dispatches += 6;
        density_active_groups.push(gi);
    }

    raw_queue.submit(Some(density_encoder.finish()));
    readback_mixed_densities(raw_device, all_groups, eigen_bcs, &density_active_groups)
}

// ── Internal helpers ──

fn upload_eigen_bcs_data(
    raw_queue: &wgpu::Queue,
    g: &GroupResources,
    items: &[&EigenBcsResult],
    solvers: &[(usize, usize, usize, SphericalHFB)],
) {
    let mut evals_p_flat = Vec::with_capacity(items.len() * g.ns);
    let mut evals_n_flat = Vec::with_capacity(items.len() * g.ns);
    let mut lambda_p_flat = Vec::with_capacity(items.len());
    let mut lambda_n_flat = Vec::with_capacity(items.len());
    let mut delta_flat = Vec::with_capacity(items.len());
    let mut evecs_p_flat = Vec::with_capacity(items.len() * g.ns * g.ns);
    let mut evecs_n_flat = Vec::with_capacity(items.len() * g.ns * g.ns);

    for d in items {
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
}

fn dispatch_bcs_density_mix(encoder: &mut wgpu::CommandEncoder, g: &GroupResources, bn: u32) {
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
}

#[cfg(feature = "gpu_energy")]
#[allow(clippy::cast_possible_truncation)]
fn dispatch_energy_pass(
    raw_queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    g: &GroupResources,
    bn: u32,
    solvers: &[(usize, usize, usize, SphericalHFB)],
    sky_params: (f64, f64, f64, f64, f64),
    total_gpu_dispatches: &mut usize,
) {
    let (t0_p, t3, x0, x3, alpha_skyrme) = sky_params;
    let dr = solvers[g.group_indices[0]].3.dr();
    let to_pair = super::super::hfb_gpu_types::f64_to_u32_pair;
    let (t0_lo, t0_hi) = to_pair(t0_p);
    let (t3_lo, t3_hi) = to_pair(t3);
    let (x0_lo, x0_hi) = to_pair(x0);
    let (x3_lo, x3_hi) = to_pair(x3);
    let (alpha_lo, alpha_hi) = to_pair(alpha_skyrme);
    let (dr_lo, dr_hi) = to_pair(dr);
    let (hw_lo, hw_hi) = to_pair(0.0);
    raw_queue.write_buffer(
        &g.energy_params_buf,
        0,
        bytemuck::bytes_of(&EnergyParamsUniform {
            n_states: g.ns as u32,
            nr: g.nr as u32,
            batch_size: bn,
            _pad: 0,
            t0_lo,
            t0_hi,
            t3_lo,
            t3_hi,
            x0_lo,
            x0_hi,
            x3_lo,
            x3_hi,
            alpha_lo,
            alpha_hi,
            dr_lo,
            dr_hi,
            hw_lo,
            hw_hi,
        }),
    );
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("energy"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&g.energy_integrands_pipe);
    pass.set_bind_group(0, &g.energy_integrands_bg, &[]);
    pass.dispatch_workgroups(g.nr_wg, bn, 1);
    pass.set_pipeline(&g.pairing_energy_pipe);
    pass.set_bind_group(0, &g.energy_integrands_bg, &[]);
    pass.set_bind_group(1, &g.energy_pair_bg, &[]);
    pass.dispatch_workgroups(bn.div_ceil(256), 1, 1);
    *total_gpu_dispatches += 2;
}

fn readback_mixed_densities(
    raw_device: &wgpu::Device,
    all_groups: &[GroupResources],
    eigen_bcs: &[EigenBcsResult],
    density_active_groups: &[usize],
) -> Result<
    (
        std::collections::HashMap<usize, (Vec<f64>, Vec<f64>)>,
        Option<std::collections::HashMap<usize, Vec<f64>>>,
    ),
    HotSpringError,
> {
    let mut density_receivers: Vec<(
        usize,
        std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
        std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    )> = Vec::new();
    #[cfg(feature = "gpu_energy")]
    let mut energy_receivers = Vec::new();

    for &gi in density_active_groups {
        let g = &all_groups[gi];
        let items_count = eigen_bcs.iter().filter(|r| r.gi == gi).count();
        let rho_bytes = (items_count * g.nr * 8) as u64;

        let slice_p = g.rho_p_staging.slice(..rho_bytes);
        let slice_n = g.rho_n_staging.slice(..rho_bytes);
        let (tx_p, rx_p) = std::sync::mpsc::channel();
        let (tx_n, rx_n) = std::sync::mpsc::channel();
        slice_p.map_async(
            wgpu::MapMode::Read,
            move |r: Result<(), wgpu::BufferAsyncError>| {
                let _ = tx_p.send(r);
            },
        );
        slice_n.map_async(
            wgpu::MapMode::Read,
            move |r: Result<(), wgpu::BufferAsyncError>| {
                let _ = tx_n.send(r);
            },
        );
        density_receivers.push((gi, rx_p, rx_n));

        #[cfg(feature = "gpu_energy")]
        {
            let energy_bytes = (items_count * g.nr * 8) as u64;
            let e_pair_bytes = (items_count * 8) as u64;
            let (tx_e, rx_e) = std::sync::mpsc::channel();
            let (tx_pair, rx_pair) = std::sync::mpsc::channel();
            g.energy_staging.slice(..energy_bytes).map_async(
                wgpu::MapMode::Read,
                move |r: Result<(), wgpu::BufferAsyncError>| {
                    let _ = tx_e.send(r);
                },
            );
            g.e_pair_staging.slice(..e_pair_bytes).map_async(
                wgpu::MapMode::Read,
                move |r: Result<(), wgpu::BufferAsyncError>| {
                    let _ = tx_pair.send(r);
                },
            );
            energy_receivers.push((gi, items_count, g.nr, rx_e, rx_pair));
        }
    }
    raw_device.poll(wgpu::Maintain::Wait);

    let mut mixed_densities: std::collections::HashMap<usize, (Vec<f64>, Vec<f64>)> =
        std::collections::HashMap::new();
    for (gi, rx_p, rx_n) in density_receivers {
        rx_p.recv()
            .map_err(|_| {
                HotSpringError::GpuCompute(String::from("density rho_p map channel failed"))
            })?
            .map_err(|e| HotSpringError::GpuCompute(format!("density rho_p map: {e}")))?;
        rx_n.recv()
            .map_err(|_| {
                HotSpringError::GpuCompute(String::from("density rho_n map channel failed"))
            })?
            .map_err(|e| HotSpringError::GpuCompute(format!("density rho_n map: {e}")))?;

        let g = &all_groups[gi];
        let items_count = eigen_bcs.iter().filter(|r| r.gi == gi).count();
        let rho_bytes = (items_count * g.nr * 8) as u64;
        let rho_p_data: Vec<f64>;
        let rho_n_data: Vec<f64>;
        {
            let mapped = g.rho_p_staging.slice(..rho_bytes).get_mapped_range();
            rho_p_data = bytemuck::cast_slice::<u8, f64>(&mapped)[..items_count * g.nr].to_vec();
        }
        g.rho_p_staging.unmap();
        {
            let mapped = g.rho_n_staging.slice(..rho_bytes).get_mapped_range();
            rho_n_data = bytemuck::cast_slice::<u8, f64>(&mapped)[..items_count * g.nr].to_vec();
        }
        g.rho_n_staging.unmap();
        mixed_densities.insert(gi, (rho_p_data, rho_n_data));
    }

    #[cfg(feature = "gpu_energy")]
    let gpu_energies: Option<std::collections::HashMap<usize, Vec<f64>>> = {
        let mut energies = std::collections::HashMap::new();
        for (gi, items_count, nr, rx_e, rx_pair) in energy_receivers {
            rx_e.recv()
                .map_err(|_| {
                    HotSpringError::GpuCompute(String::from("energy integrands map channel failed"))
                })?
                .map_err(|e| HotSpringError::GpuCompute(format!("energy map: {e}")))?;
            rx_pair
                .recv()
                .map_err(|_| HotSpringError::GpuCompute(String::from("e_pair map channel failed")))?
                .map_err(|e| HotSpringError::GpuCompute(format!("e_pair map: {e}")))?;
            let g = &all_groups[gi];
            let energy_bytes = (items_count * nr * 8) as u64;
            let e_pair_bytes = (items_count * 8) as u64;
            let integrands: Vec<f64> = {
                let mapped = g.energy_staging.slice(..energy_bytes).get_mapped_range();
                bytemuck::cast_slice::<u8, f64>(&mapped)[..items_count * nr].to_vec()
            };
            g.energy_staging.unmap();
            let e_pair: Vec<f64> = {
                let mapped = g.e_pair_staging.slice(..e_pair_bytes).get_mapped_range();
                bytemuck::cast_slice::<u8, f64>(&mapped)[..items_count].to_vec()
            };
            g.e_pair_staging.unmap();
            let group_energies: Vec<f64> = e_pair
                .iter()
                .enumerate()
                .take(items_count)
                .map(|(bi, &ep)| {
                    let base = bi * nr;
                    let integrand_slice = &integrands[base..base + nr];
                    let trapz_sum = if nr <= 1 {
                        integrand_slice.first().copied().unwrap_or(0.0)
                    } else {
                        integrand_slice[0] / 2.0
                            + integrand_slice[1..nr - 1].iter().sum::<f64>()
                            + integrand_slice[nr - 1] / 2.0
                    };
                    trapz_sum + ep
                })
                .collect();
            energies.insert(gi, group_energies);
        }
        Some(energies)
    };

    #[cfg(not(feature = "gpu_energy"))]
    let gpu_energies: Option<std::collections::HashMap<usize, Vec<f64>>> = None;

    Ok((mixed_densities, gpu_energies))
}
