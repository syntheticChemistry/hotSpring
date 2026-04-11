// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unidirectional GPU RHMC pipeline — zero per-iteration readback.
//!
//! Transforms the RHMC trajectory from ~19,000 CPU-GPU syncs (sync-bound) to
//! ~100 syncs (compute-bound) by:
//!
//! 1. **Resident shifted CG** — batches of ~50 CG iterations per submission,
//!    one 8-byte convergence readback per batch.
//! 2. **GPU-resident Hamiltonian (B2)** — Wilson action, KE, and fermion
//!    action are all reduced and assembled into H on GPU. Zero readback for
//!    either H_old or H_new. Three WGSL kernels: `hamiltonian_assembly_f64`,
//!    `fermion_action_sum_f64`, plus existing reduce chains.
//! 3. **GPU-resident Metropolis (B3)** — accept/reject decision runs as a
//!    single-thread WGSL kernel (`metropolis_f64`) reading H_old/H_new from
//!    GPU buffers. Single 56-byte readback for all trajectory diagnostics.
//! 4. **Async cortex** — `UnidirectionalRhmc` struct with fire/poll API
//!    for non-blocking dual-GPU dispatch.
//!
//! Readback budget: ~100 × 8 bytes (CG convergence) + 56 bytes (Metropolis).
//! This eliminates B2 (CPU Hamiltonian assembly) and B3 (CPU Metropolis)
//! bottlenecks from `SILICON_TIER_ROUTING.md`.

use super::dynamical::{GpuDynHmcPipelines, GpuDynHmcState, gpu_axpy};
use super::gpu_rhmc::{
    GpuRhmcPipelines, GpuRhmcResult, GpuRhmcSectorBuffers, GpuRhmcState, dirac_dispatch,
    fermion_force_dispatch,
};
use super::resident_shifted_cg::{GpuResidentShiftedCgBuffers, gpu_multi_shift_cg_solve_resident};
use super::streaming::make_ferm_prng_params;
use super::true_multishift_cg::{TrueMultiShiftBuffers, gpu_true_multi_shift_cg_solve};
use super::uni_hamiltonian::{CG_CHECK_INTERVAL, compute_h_gpu};
pub use super::uni_hamiltonian::{UniHamiltonianBuffers, UniPipelines};
use super::{
    GpuF64, GpuHmcState, gpu_force_dispatch, gpu_link_update_dispatch, gpu_mom_update_dispatch,
    make_link_mom_params, make_prng_params,
};

use crate::lattice::rhmc::{RhmcConfig, RhmcFermionConfig};

/// Dispatch momenta PRNG via TMU path if available, else ALU fallback.
fn encode_uni_prng_dispatch(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    uni_pipelines: &UniPipelines,
    params_buf: &wgpu::Buffer,
    mom_buf: &wgpu::Buffer,
    wg_links: u32,
) {
    if let (Some(tmu_pl), Some(tables)) =
        (&uni_pipelines.tmu_prng_pipeline, &uni_pipelines.tmu_tables)
    {
        let bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uni_prng_tmu_bg"),
            layout: &tmu_pl.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mom_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&tables.log_table),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&tables.trig_table),
                },
            ],
        });
        GpuF64::encode_pass(enc, tmu_pl, &bg, wg_links);
    } else {
        let bg =
            gpu.create_bind_group(&uni_pipelines.momenta_prng_pipeline, &[params_buf, mom_buf]);
        GpuF64::encode_pass(enc, &uni_pipelines.momenta_prng_pipeline, &bg, wg_links);
    }
}

/// Build params for the Metropolis kernel: [rand, six_v] as raw f64.
fn make_metropolis_params(seed: &mut u64, six_v: f64) -> Vec<u8> {
    let r: f64 = crate::lattice::constants::lcg_uniform_f64(seed);
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&r.to_le_bytes());
    v.extend_from_slice(&six_v.to_le_bytes());
    v
}

/// GPU Metropolis: compute delta_H, accept/reject, write diagnostics.
///
/// Returns `GpuRhmcResult` from a single 56-byte readback.
///
/// # Errors
///
/// Returns `GpuCompute` if the Metropolis staging readback fails (device lost).
fn gpu_metropolis(
    gpu: &GpuF64,
    uni_pipelines: &UniPipelines,
    ham: &UniHamiltonianBuffers,
    gauge: &GpuHmcState,
    total_cg: usize,
    seed: &mut u64,
) -> Result<GpuRhmcResult, crate::error::HotSpringError> {
    let six_v = 6.0 * gauge.volume as f64;

    let mut enc = gpu.begin_encoder("b3_metropolis");
    let params = make_metropolis_params(seed, six_v);
    let pbuf = gpu.create_storage_buffer_init(&params, "uni_metro_p");

    let result_buf = gpu.create_f64_output_buffer(9, "uni_metro_result");

    let bg = gpu.create_bind_group(
        &uni_pipelines.metropolis_pipeline,
        &[
            &pbuf,
            &ham.h_old_buf,
            &ham.h_new_buf,
            &ham.plaq_sum_buf,
            &ham.diag_old_buf,
            &ham.diag_new_buf,
            &result_buf,
        ],
    );
    GpuF64::encode_pass(&mut enc, &uni_pipelines.metropolis_pipeline, &bg, 1);
    enc.copy_buffer_to_buffer(&result_buf, 0, &ham.metropolis_staging, 0, 9 * 8);
    gpu.submit_encoder(enc);

    // Single readback: 72 bytes (9 f64s)
    let data = gpu
        .read_staging_f64_n(&ham.metropolis_staging, 9)
        .map_err(|e| {
            crate::error::HotSpringError::GpuCompute(format!(
                "Metropolis readback failed (GPU lost?): {e}"
            ))
        })?;

    let accepted = data[0] > 0.5;

    if !accepted {
        let n_links = gauge.n_links;
        let mut enc = gpu.begin_encoder("uni_restore");
        enc.copy_buffer_to_buffer(
            &gauge.link_backup,
            0,
            &gauge.link_buf,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    Ok(GpuRhmcResult {
        accepted,
        delta_h: data[1],
        plaquette: data[2],
        total_cg_iterations: total_cg,
        s_gauge_old: data[3],
        s_gauge_new: data[4],
        t_old: data[5],
        t_new: data[6],
        s_ferm_old: data[7],
        s_ferm_new: data[8],
    })
}

// ── Full unidirectional RHMC trajectory (Phase 4) ────────────────

/// Run one RHMC trajectory using the fully GPU-resident pipeline (B2+B3).
///
/// All CG solves use GPU-resident shifted CG (zero per-iteration readback).
/// Hamiltonian assembly (H_old, H_new) is computed entirely on GPU — no
/// scalar readbacks for plaquette, KE, or fermion action components.
/// Metropolis accept/reject runs as a GPU kernel.
///
/// Total readback budget: ~100 × 8 bytes (CG convergence) + 56 bytes (Metropolis).
/// This eliminates 4+ sync points vs the prior CPU-assembly path.
///
/// # Errors
///
/// Returns `GpuCompute` if any GPU readback fails (device lost, timeout).
pub fn gpu_rhmc_trajectory_unidirectional(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    _rhmc_pipelines: &GpuRhmcPipelines,
    uni_pipelines: &UniPipelines,
    state: &GpuRhmcState,
    scg_bufs: &GpuResidentShiftedCgBuffers,
    ms_bufs: Option<&TrueMultiShiftBuffers>,
    ham_bufs: &UniHamiltonianBuffers,
    config: &RhmcConfig,
    seed: &mut u64,
) -> Result<GpuRhmcResult, crate::error::HotSpringError> {
    let gauge = &state.gauge.gauge;
    let n_links = gauge.n_links;
    let n_md_steps = config.n_md_steps;
    let dt = config.dt;

    // ── 1. GPU PRNG momenta + pseudofermion η + backup links ────
    {
        let mut enc = gpu.begin_encoder("uni_init");

        let mom_prng_params = make_prng_params(n_links as u32, 0, seed);
        let mom_prng_pbuf = gpu.create_uniform_buffer(&mom_prng_params, "uni_mom_p");
        encode_uni_prng_dispatch(
            &mut enc,
            gpu,
            uni_pipelines,
            &mom_prng_pbuf,
            &gauge.mom_buf,
            gauge.wg_links,
        );

        enc.copy_buffer_to_buffer(
            &gauge.link_buf,
            0,
            &gauge.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );

        gpu.submit_encoder(enc);
    }

    // ── 2. RHMC heatbath: generate φ for each sector ─────────────
    let mut total_cg: usize = 0;
    for (si, (sector, fconfig)) in state.sectors.iter().zip(config.sectors.iter()).enumerate() {
        let cg = uni_heatbath_sector(
            gpu,
            dyn_pipelines,
            uni_pipelines,
            &state.gauge,
            sector,
            fconfig,
            scg_bufs,
            ms_bufs,
            config.cg_tol,
            config.cg_max_iter,
            seed,
            si,
        );
        total_cg += cg;
    }

    // ── 3. H_old (GPU-resident, zero readback) ──────────────────
    total_cg += compute_h_gpu(
        gpu,
        dyn_pipelines,
        uni_pipelines,
        state,
        scg_bufs,
        ms_bufs,
        ham_bufs,
        config,
        &ham_bufs.h_old_buf,
        &ham_bufs.diag_old_buf,
    );

    // ── 4. Omelyan MD integration ────────────────────────────────
    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for _step in 0..n_md_steps {
        let cg = uni_total_force_dispatch(
            gpu,
            dyn_pipelines,
            uni_pipelines,
            &state.gauge,
            &state.sectors,
            &config.sectors,
            scg_bufs,
            ms_bufs,
            lam * dt,
            config.cg_tol,
            config.cg_max_iter,
        );
        total_cg += cg;
        gpu_link_update_dispatch(gpu, &dyn_pipelines.gauge, gauge, 0.5 * dt);

        let cg = uni_total_force_dispatch(
            gpu,
            dyn_pipelines,
            uni_pipelines,
            &state.gauge,
            &state.sectors,
            &config.sectors,
            scg_bufs,
            ms_bufs,
            2.0f64.mul_add(-lam, 1.0) * dt,
            config.cg_tol,
            config.cg_max_iter,
        );
        total_cg += cg;
        gpu_link_update_dispatch(gpu, &dyn_pipelines.gauge, gauge, 0.5 * dt);

        let cg = uni_total_force_dispatch(
            gpu,
            dyn_pipelines,
            uni_pipelines,
            &state.gauge,
            &state.sectors,
            &config.sectors,
            scg_bufs,
            ms_bufs,
            lam * dt,
            config.cg_tol,
            config.cg_max_iter,
        );
        total_cg += cg;
    }

    // ── 5. H_new (GPU-resident, zero readback) ──────────────────
    total_cg += compute_h_gpu(
        gpu,
        dyn_pipelines,
        uni_pipelines,
        state,
        scg_bufs,
        ms_bufs,
        ham_bufs,
        config,
        &ham_bufs.h_new_buf,
        &ham_bufs.diag_new_buf,
    );

    // ── 6. GPU Metropolis + single 56-byte readback ─────────────
    gpu_metropolis(gpu, uni_pipelines, ham_bufs, gauge, total_cg, seed)
}

// ── Heatbath + force helpers (internal) ──────────────────────────

/// RHMC heatbath for one sector using GPU PRNG + multi-shift CG.
fn uni_heatbath_sector(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    uni_pipelines: &UniPipelines,
    state: &GpuDynHmcState,
    sector: &GpuRhmcSectorBuffers,
    config: &RhmcFermionConfig,
    scg_bufs: &GpuResidentShiftedCgBuffers,
    ms_bufs: Option<&TrueMultiShiftBuffers>,
    cg_tol: f64,
    cg_max_iter: usize,
    seed: &mut u64,
    sector_idx: usize,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;
    let approx = &config.heatbath_approx;

    // GPU PRNG: Gaussian noise η → phi_buf
    {
        let wg_vol = (vol as u32).div_ceil(64);
        let ferm_params = make_ferm_prng_params(vol as u32, sector_idx as u32 * 1000, seed);
        let ferm_pbuf = gpu.create_uniform_buffer(&ferm_params, "uni_ferm_p");
        let ferm_bg = gpu.create_bind_group(
            &uni_pipelines.fermion_prng_pipeline,
            &[&ferm_pbuf, &sector.phi_buf],
        );
        gpu.dispatch(&uni_pipelines.fermion_prng_pipeline, &ferm_bg, wg_vol);
    }

    // Multi-shift CG: (D†D + σ_s) x_s = η
    let cg_iters = if let Some(ms) = ms_bufs {
        if approx.sigma.len() <= ms.n_shifts {
            gpu_true_multi_shift_cg_solve(
                gpu,
                dyn_pipelines,
                &uni_pipelines.true_ms_cg,
                state,
                ms,
                &sector.x_bufs,
                &sector.phi_buf,
                &approx.sigma,
                cg_tol,
                cg_max_iter,
                CG_CHECK_INTERVAL,
            )
        } else {
            gpu_multi_shift_cg_solve_resident(
                gpu,
                dyn_pipelines,
                &uni_pipelines.shifted_cg,
                state,
                scg_bufs,
                &sector.x_bufs,
                &sector.phi_buf,
                &approx.sigma,
                cg_tol,
                cg_max_iter,
                CG_CHECK_INTERVAL,
            )
        }
    } else {
        gpu_multi_shift_cg_solve_resident(
            gpu,
            dyn_pipelines,
            &uni_pipelines.shifted_cg,
            state,
            scg_bufs,
            &sector.x_bufs,
            &sector.phi_buf,
            &approx.sigma,
            cg_tol,
            cg_max_iter,
            CG_CHECK_INTERVAL,
        )
    };

    // Accumulate: temp = α₀·η + Σ αₛ·x_s → phi_buf
    gpu.zero_buffer(&state.x_buf, (n_flat * 8) as u64);
    gpu_axpy(
        gpu,
        &dyn_pipelines.axpy_pipeline,
        approx.alpha_0,
        &sector.phi_buf,
        &state.x_buf,
        n_flat,
    );
    for (s, a_s) in approx.alpha.iter().enumerate() {
        gpu_axpy(
            gpu,
            &dyn_pipelines.axpy_pipeline,
            *a_s,
            &sector.x_bufs[s],
            &state.x_buf,
            n_flat,
        );
    }
    {
        let mut enc = gpu.begin_encoder("uni_phi_copy");
        enc.copy_buffer_to_buffer(&state.x_buf, 0, &sector.phi_buf, 0, (n_flat * 8) as u64);
        gpu.submit_encoder(enc);
    }

    cg_iters
}

/// RHMC total force: gauge + Σ_sectors Σ_poles fermion force.
///
/// Uses true multi-shift CG (shared Krylov) when `ms_bufs` is available,
/// otherwise falls back to sequential shifted CG.
fn uni_total_force_dispatch(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    uni_pipelines: &UniPipelines,
    state: &GpuDynHmcState,
    sectors: &[GpuRhmcSectorBuffers],
    configs: &[RhmcFermionConfig],
    scg_bufs: &GpuResidentShiftedCgBuffers,
    ms_bufs: Option<&TrueMultiShiftBuffers>,
    dt: f64,
    cg_tol: f64,
    cg_max_iter: usize,
) -> usize {
    let gauge = &state.gauge;
    let n_links = gauge.n_links;

    // Gauge force → P += dt · F_gauge
    gpu_force_dispatch(gpu, &dyn_pipelines.gauge, gauge);
    gpu_mom_update_dispatch(gpu, &dyn_pipelines.gauge, gauge, dt);

    let mut total_cg = 0;

    for (sector, fconfig) in sectors.iter().zip(configs.iter()) {
        let approx = &fconfig.force_approx;

        let cg = if let Some(ms) = ms_bufs {
            if approx.sigma.len() <= ms.n_shifts {
                gpu_true_multi_shift_cg_solve(
                    gpu,
                    dyn_pipelines,
                    &uni_pipelines.true_ms_cg,
                    state,
                    ms,
                    &sector.x_bufs,
                    &sector.phi_buf,
                    &approx.sigma,
                    cg_tol,
                    cg_max_iter,
                    CG_CHECK_INTERVAL,
                )
            } else {
                gpu_multi_shift_cg_solve_resident(
                    gpu,
                    dyn_pipelines,
                    &uni_pipelines.shifted_cg,
                    state,
                    scg_bufs,
                    &sector.x_bufs,
                    &sector.phi_buf,
                    &approx.sigma,
                    cg_tol,
                    cg_max_iter,
                    CG_CHECK_INTERVAL,
                )
            }
        } else {
            gpu_multi_shift_cg_solve_resident(
                gpu,
                dyn_pipelines,
                &uni_pipelines.shifted_cg,
                state,
                scg_bufs,
                &sector.x_bufs,
                &sector.phi_buf,
                &approx.sigma,
                cg_tol,
                cg_max_iter,
                CG_CHECK_INTERVAL,
            )
        };
        total_cg += cg;

        // Per-pole: fermion force from x_s
        if let Some(rop) = &uni_pipelines.rop_accum {
            // ROP path (Tier 3): fuse force+momentum via atomicAdd(i32)
            let mut enc = gpu.begin_encoder("rop_zero_accum");
            rop.zero_accum(&mut enc);
            gpu.submit_encoder(enc);

            for (s, a_s) in approx.alpha.iter().enumerate() {
                {
                    let n_flat = gauge.volume * 6;
                    let mut enc = gpu.begin_encoder("rop_xcopy");
                    enc.copy_buffer_to_buffer(
                        &sector.x_bufs[s],
                        0,
                        &state.x_buf,
                        0,
                        (n_flat * 8) as u64,
                    );
                    gpu.submit_encoder(enc);
                }

                dirac_dispatch(
                    gpu,
                    dyn_pipelines,
                    gauge,
                    &state.phases_buf,
                    &state.x_buf,
                    &state.y_buf,
                    fconfig.mass,
                    1.0,
                );

                let mut enc = gpu.begin_encoder("rop_force_pole");
                rop.encode_pole_dispatch(
                    gpu,
                    &mut enc,
                    &gauge.link_buf,
                    &state.x_buf,
                    &state.y_buf,
                    &gauge.nbr_buf,
                    &state.phases_buf,
                    gauge.volume as u32,
                    *a_s * dt,
                );
                gpu.submit_encoder(enc);
            }

            // Single conversion: momentum += f64(accum) / scale
            let mut enc = gpu.begin_encoder("rop_convert");
            rop.encode_convert_to_momentum(gpu, &mut enc, &gauge.mom_buf);
            gpu.submit_encoder(enc);
        } else {
            // Standard path: separate force + momentum_update per pole
            for (s, a_s) in approx.alpha.iter().enumerate() {
                {
                    let n_flat = gauge.volume * 6;
                    let mut enc = gpu.begin_encoder("uni_xcopy");
                    enc.copy_buffer_to_buffer(
                        &sector.x_bufs[s],
                        0,
                        &state.x_buf,
                        0,
                        (n_flat * 8) as u64,
                    );
                    gpu.submit_encoder(enc);
                }

                dirac_dispatch(
                    gpu,
                    dyn_pipelines,
                    gauge,
                    &state.phases_buf,
                    &state.x_buf,
                    &state.y_buf,
                    fconfig.mass,
                    1.0,
                );
                fermion_force_dispatch(
                    gpu,
                    dyn_pipelines,
                    gauge,
                    &state.phases_buf,
                    &state.x_buf,
                    &state.y_buf,
                    &state.ferm_force_buf,
                );

                let ferm_mom_params = make_link_mom_params(n_links, *a_s * dt, gpu.full_df64_mode);
                let ferm_mom_pbuf = gpu.create_uniform_buffer(&ferm_mom_params, "uni_fmom_p");
                let ferm_mom_bg = gpu.create_bind_group(
                    &dyn_pipelines.gauge.momentum_pipeline,
                    &[&ferm_mom_pbuf, &state.ferm_force_buf, &gauge.mom_buf],
                );
                gpu.dispatch(
                    &dyn_pipelines.gauge.momentum_pipeline,
                    &ferm_mom_bg,
                    gauge.wg_links,
                );
            }
        }
    }

    total_cg
}
