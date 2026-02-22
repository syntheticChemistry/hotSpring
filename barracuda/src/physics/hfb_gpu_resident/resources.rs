// SPDX-License-Identifier: AGPL-3.0-only

//! GPU buffer allocation and bind-group wiring for the HFB solver.
//!
//! Each "group" shares `(ns, nr)` dimensions and pre-allocates all wgpu
//! buffers, bind groups, and compute pipelines needed for the SCF loop.
//! This eliminates per-iteration allocation overhead.

use super::super::hfb::SphericalHFB;
#[cfg(feature = "gpu_energy")]
use super::super::hfb_gpu_types::EnergyParamsUniform;
use super::super::hfb_gpu_types::{
    make_bind_group, DensityParamsUniform, GroupResources, HamiltonianDimsUniform,
    MixParamsUniform, PotentialDimsUniform,
};
use super::pipelines;

/// Per-group SO-pack resources: bind groups referencing per-group H buffers
/// + the shared eigensolve buffer.
pub(super) struct PackGroupResources {
    pub(super) params_p_buf: wgpu::Buffer,
    pub(super) params_n_buf: wgpu::Buffer,
    pub(super) pack_p_bg: wgpu::BindGroup,
    pub(super) pack_n_bg: wgpu::BindGroup,
}

/// Allocate all GPU buffers and bind groups for one `(ns, nr)` group.
///
/// This is the heaviest one-time setup call: it creates ~30 buffers,
/// ~15 bind groups, and 8+ compute pipelines per group.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn allocate_group_resources(
    raw_device: &wgpu::Device,
    group_indices: &[usize],
    ns: usize,
    nr: usize,
    n_max: usize,
    solvers: &[(usize, usize, usize, SphericalHFB)],
    sky_data: [f64; 9],
    dr: f64,
    pot_module: &wgpu::ShaderModule,
    ham_module: &wgpu::ShaderModule,
    density_module: &wgpu::ShaderModule,
    #[cfg(feature = "gpu_energy")] energy_module: &wgpu::ShaderModule,
) -> GroupResources {
    use wgpu::util::DeviceExt;

    let u = wgpu::BufferBindingType::Uniform;
    let sr = wgpu::BufferBindingType::Storage { read_only: true };
    let srw = wgpu::BufferBindingType::Storage { read_only: false };

    let mat_size = ns * ns;

    // ── Gather per-nucleus basis data ──
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

    // ── Immutable basis buffers ──
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
    let sky_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sky"),
        contents: bytemuck::cast_slice(&sky_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dr_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dr"),
        contents: bytemuck::cast_slice(&[dr]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // ── Mutable density buffers (written each SCF iteration) ──
    let buf_nr = (n_max * nr * 8) as u64;
    let make_rho_rw = |label| {
        raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let make_rho_dst = |label| {
        raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let make_storage = |label| {
        raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buf_nr,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };

    let rho_p_buf = make_rho_rw("rho_p");
    let rho_n_buf = make_rho_rw("rho_n");
    let rho_alpha_buf = make_rho_dst("rho_alpha");
    let rho_alpha_m1_buf = make_rho_dst("rho_alpha_m1");
    let u_p_buf = make_storage("u_p");
    let u_n_buf = make_storage("u_n");
    let fq_p_buf = make_storage("fq_p");
    let fq_n_buf = make_storage("fq_n");
    let cenc_buf = make_storage("cenc");
    let phi_buf = make_storage("phi");

    // ── Hamiltonian + spin-orbit buffers ──
    let h_mat_bytes = (n_max * mat_size * 8) as u64;
    let make_h = |label| {
        raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: h_mat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let h_p_buf = make_h("H_p");
    let h_n_buf = make_h("H_n");
    let so_diag_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("so_diag"),
        size: ((n_max * ns * 8) as u64).max(8),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Uniform buffers ──
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

    // ── Potential bind group + pipelines ──
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

    // ── Hamiltonian bind groups (proton/neutron share layout) ──
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
    let (_hl_n, hbg_n) = make_bind_group(
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

    let (sky_pipe, cfwd, cbwd, fin_pipe, fq_pipe) =
        pipelines::create_potential_pipelines(raw_device, pot_module, &pot_layout);
    let (hp_pipe, hn_pipe) = pipelines::create_hamiltonian_pipelines(raw_device, ham_module, &hl_p);

    // ── BCS / density / mixing buffers ──
    let buf_ns = (n_max * ns * 8) as u64;
    let buf_evecs = (n_max * ns * ns * 8) as u64;
    let buf_1 = (n_max * 8) as u64;

    let make_storage_dst = |label, size| {
        raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };

    let density_params_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("density_params"),
        size: std::mem::size_of::<DensityParamsUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let evals_p_buf = make_storage_dst("evals_p", buf_ns);
    let evals_n_buf = make_storage_dst("evals_n", buf_ns);
    let lambda_p_buf = make_storage_dst("lambda_p", buf_1);
    let lambda_n_buf = make_storage_dst("lambda_n", buf_1);
    let delta_buf = make_storage_dst("delta", buf_1);
    let v2_p_buf = make_storage_dst("v2_p", buf_ns);
    let v2_n_buf = make_storage_dst("v2_n", buf_ns);
    let evecs_p_buf = make_storage_dst("evecs_p", buf_evecs);
    let evecs_n_buf = make_storage_dst("evecs_n", buf_evecs);

    let rho_p_new_buf = make_storage("rho_p_new");
    let rho_n_new_buf = make_storage("rho_n_new");
    let mix_params_buf = raw_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mix_params"),
        size: std::mem::size_of::<MixParamsUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let make_staging = |label, size| {
        raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let rho_p_staging = make_staging("rho_p_staging", buf_nr);
    let rho_n_staging = make_staging("rho_n_staging", buf_nr);

    let degs: Vec<f64> = solvers[group_indices[0]].3.deg_values();
    let degs_buf = raw_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("degs"),
        contents: bytemuck::cast_slice(&degs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // ── BCS / density / mixing bind groups ──
    let (dp_layout, density_params_bg) =
        make_bind_group(raw_device, "density_g0", &[(u, &density_params_buf)]);
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

    let (bcs_v2_pipe, density_pipe, mix_pipe) = pipelines::create_density_pipelines(
        raw_device,
        density_module,
        &dp_layout,
        &bcs_write_layout,
        &bcs_read_layout,
        &dens_write_layout,
        &dens_read_layout,
        &mix_layout,
    );

    // ── GPU energy pipeline (behind feature flag) ──
    #[cfg(feature = "gpu_energy")]
    let (
        energy_params_buf,
        energy_integrands_bg,
        energy_pair_bg,
        energy_integrands_pipe,
        pairing_energy_pipe,
        energy_integrands_buf,
        e_pair_buf,
        energy_staging,
        e_pair_staging,
    ) = {
        use super::super::hfb_gpu_types::make_pipeline;

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
        let energy_staging = make_staging("energy_staging", buf_nr);
        let e_pair_staging = make_staging("e_pair_staging", buf_1);

        let (energy_integrands_layout, energy_integrands_bg) = make_bind_group(
            raw_device,
            "energy_integrands_g0",
            &[
                (u, &energy_params_buf),
                (sr, &rho_p_buf),
                (sr, &rho_n_buf),
                (sr, &rgrid_buf),
                (sr, &cenc_buf),
                (srw, &energy_integrands_buf),
            ],
        );
        let (energy_pair_layout, energy_pair_bg) = make_bind_group(
            raw_device,
            "energy_pair_g1",
            &[
                (sr, &v2_p_buf),
                (sr, &v2_n_buf),
                (sr, &degs_buf),
                (sr, &delta_buf),
                (sr, &delta_buf),
                (srw, &e_pair_buf),
            ],
        );
        let energy_integrands_pipe = make_pipeline(
            raw_device,
            energy_module,
            "compute_energy_integrands",
            &[&energy_integrands_layout],
        );
        let pairing_energy_pipe = make_pipeline(
            raw_device,
            energy_module,
            "compute_pairing_energy",
            &[&energy_integrands_layout, &energy_pair_layout],
        );
        (
            energy_params_buf,
            energy_integrands_bg,
            energy_pair_bg,
            energy_integrands_pipe,
            pairing_energy_pipe,
            energy_integrands_buf,
            e_pair_buf,
            energy_staging,
            e_pair_staging,
        )
    };

    GroupResources {
        ns,
        nr,
        group_indices: group_indices.to_vec(),
        n_max,
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
        #[cfg(feature = "gpu_energy")]
        energy_params_buf,
        #[cfg(feature = "gpu_energy")]
        energy_integrands_bg,
        #[cfg(feature = "gpu_energy")]
        energy_pair_bg,
        #[cfg(feature = "gpu_energy")]
        energy_integrands_pipe,
        #[cfg(feature = "gpu_energy")]
        pairing_energy_pipe,
        #[cfg(feature = "gpu_energy")]
        energy_integrands_buf,
        #[cfg(feature = "gpu_energy")]
        e_pair_buf,
        #[cfg(feature = "gpu_energy")]
        energy_staging,
        #[cfg(feature = "gpu_energy")]
        e_pair_staging,
        nr_wg: (nr as u32).div_ceil(256),
        ns_wg: (ns as u32).div_ceil(16),
    }
}
