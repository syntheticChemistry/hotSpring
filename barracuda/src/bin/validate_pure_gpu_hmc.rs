// SPDX-License-Identifier: AGPL-3.0-only

//! Pure GPU HMC validation: all lattice QCD math on GPU via fp64 WGSL.
//!
//! Tests each GPU shader individually against CPU reference:
//!
//! | Phase | GPU shader | CPU reference | Check |
//! |-------|-----------|---------------|-------|
//! | 1 | `wilson_plaquette_f64` | `average_plaquette()` | Machine-ε parity |
//! | 2 | `su3_gauge_force_f64` | `gauge_force()` | Component-wise parity |
//! | 3 | `su3_kinetic_energy_f64` | `kinetic_energy()` | Scalar parity |
//! | 4 | `su3_momentum_update_f64` | CPU P += dt*F | Component-wise parity |
//! | 5 | `su3_link_update_f64` | CPU Cayley exp | Component-wise parity |
//!
//! Links and momenta are GPU-resident. Only scalars stream back.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    build_neighbors, flatten_links, flatten_momenta, GpuHmcPipelines, WGSL_GAUGE_FORCE,
    WGSL_KINETIC_ENERGY, WGSL_LINK_UPDATE, WGSL_MOMENTUM_UPDATE, WGSL_WILSON_PLAQUETTE,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::su3::Su3Matrix;
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

fn read_f64(gpu: &GpuF64, buf: &wgpu::Buffer, count: usize) -> Vec<f64> {
    gpu.read_back_f64(buf, count)
        .unwrap_or_else(|e| panic!("GPU readback failed: {e}"))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Pure GPU HMC Validation — All Math on GPU (fp64 WGSL)     ║");
    println!("║  Unidirectional streaming: only scalars CPU←GPU            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("pure_gpu_hmc");
    let start_total = Instant::now();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| {
        panic!("tokio runtime failed: {e}");
    });
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("  No GPU with SHADER_F64 found: {e}");
            println!("  Skipping GPU validation");
            harness.check_bool("GPU available (SHADER_F64)", false);
            harness.finish();
        }
    };

    println!("  GPU: {}", gpu.adapter_name);
    println!();

    let pipelines = GpuHmcPipelines::new(&gpu);
    println!("  All 5 HMC shader pipelines compiled successfully");
    harness.check_bool("All shader pipelines compile", true);
    println!();

    // Prepare thermalized 4⁴ lattice
    let dims = [4, 4, 4, 4];
    let beta = 6.0;
    let vol: usize = dims.iter().product();
    let n_links = vol * 4;
    let wg = ((n_links + 63) / 64) as u32;
    let wg_vol = ((vol + 63) / 64) as u32;

    let mut lat = Lattice::hot_start(dims, beta, 42);
    let mut cfg = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };
    hmc::run_hmc(&mut lat, 20, 0, &mut cfg);

    let links_flat = flatten_links(&lat);
    let neighbors = build_neighbors(&lat);

    // ═══ Phase 1: GPU plaquette vs CPU ═══
    println!("═══ Phase 1: GPU plaquette vs CPU ═══");

    let cpu_plaq = lat.average_plaquette();

    let link_buf = gpu.create_f64_buffer(&links_flat, "links");
    let nbr_buf = gpu.create_u32_buffer(&neighbors, "nbr");

    let plaq_params: Vec<u8> = {
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&(vol as u32).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v
    };
    let plaq_param_buf = gpu.create_uniform_buffer(&plaq_params, "plaq_params");
    let plaq_out_buf = gpu.create_f64_output_buffer(vol, "plaq_out");

    let plaq_bg = gpu.create_bind_group(
        &pipelines.plaquette_pipeline,
        &[&plaq_param_buf, &link_buf, &nbr_buf, &plaq_out_buf],
    );

    gpu.dispatch(&pipelines.plaquette_pipeline, &plaq_bg, wg_vol);
    let per_site_plaq = read_f64(&gpu, &plaq_out_buf, vol);
    let gpu_plaq_sum: f64 = per_site_plaq.iter().sum();
    let gpu_plaq = gpu_plaq_sum / (6.0 * vol as f64);

    let plaq_err = (gpu_plaq - cpu_plaq).abs();
    println!("  CPU plaquette: {cpu_plaq:.15}");
    println!("  GPU plaquette: {gpu_plaq:.15}");
    println!("  |Δ|: {plaq_err:.2e}");

    harness.check_abs("GPU plaquette parity", gpu_plaq, cpu_plaq, 1e-10);
    println!();

    // ═══ Phase 2: GPU gauge force vs CPU ═══
    println!("═══ Phase 2: GPU gauge force vs CPU ═══");

    let force_params: Vec<u8> = {
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&(vol as u32).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&beta.to_le_bytes());
        v
    };
    let force_param_buf = gpu.create_uniform_buffer(&force_params, "force_params");
    let force_buf = gpu.create_f64_output_buffer(n_links * 18, "force");

    let force_bg = gpu.create_bind_group(
        &pipelines.force_pipeline,
        &[&force_param_buf, &link_buf, &nbr_buf, &force_buf],
    );

    gpu.dispatch(&pipelines.force_pipeline, &force_bg, wg);
    let gpu_force_flat = read_f64(&gpu, &force_buf, n_links * 18);

    let mut max_force_err: f64 = 0.0;
    for idx in 0..vol {
        let x = lat.site_coords(idx);
        for mu in 0..4 {
            let cpu_f = lat.gauge_force(x, mu);
            let base = (idx * 4 + mu) * 18;
            for row in 0..3 {
                for col in 0..3 {
                    max_force_err = max_force_err
                        .max(
                            (cpu_f.m[row][col].re - gpu_force_flat[base + row * 6 + col * 2]).abs(),
                        )
                        .max(
                            (cpu_f.m[row][col].im - gpu_force_flat[base + row * 6 + col * 2 + 1])
                                .abs(),
                        );
                }
            }
        }
    }
    println!("  Max |GPU-CPU| force error: {max_force_err:.2e}");
    harness.check_bool("GPU force parity < 1e-10", max_force_err < 1e-10);
    println!();

    // ═══ Phase 3: GPU kinetic energy vs CPU ═══
    println!("═══ Phase 3: GPU kinetic energy vs CPU ═══");

    let mut rng_seed = 99u64;
    let momenta: Vec<Su3Matrix> = (0..n_links)
        .map(|_| Su3Matrix::random_algebra(&mut rng_seed))
        .collect();
    let cpu_ke: f64 = momenta
        .iter()
        .map(|p| {
            let p2 = *p * *p;
            -0.5 * p2.re_trace()
        })
        .sum();

    let mom_flat = flatten_momenta(&momenta);
    let mom_buf = gpu.create_f64_buffer(&mom_flat, "momenta");

    let ke_params: Vec<u8> = {
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&(n_links as u32).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v
    };
    let ke_param_buf = gpu.create_uniform_buffer(&ke_params, "ke_params");
    let ke_out_buf = gpu.create_f64_output_buffer(n_links, "ke_out");

    let ke_bg = gpu.create_bind_group(
        &pipelines.kinetic_pipeline,
        &[&ke_param_buf, &mom_buf, &ke_out_buf],
    );

    gpu.dispatch(&pipelines.kinetic_pipeline, &ke_bg, wg);
    let ke_per_link = read_f64(&gpu, &ke_out_buf, n_links);
    let gpu_ke: f64 = ke_per_link.iter().sum();

    let ke_err = (gpu_ke - cpu_ke).abs();
    let ke_rel = ke_err / cpu_ke.abs().max(1e-30);
    println!("  CPU kinetic energy: {cpu_ke:.10}");
    println!("  GPU kinetic energy: {gpu_ke:.10}");
    println!("  |Δ|: {ke_err:.2e} (rel: {ke_rel:.2e})");

    harness.check_bool("GPU kinetic energy parity < 1e-10", ke_rel < 1e-10);
    println!();

    // ═══ Phase 4: GPU momentum update vs CPU ═══
    println!("═══ Phase 4: GPU momentum update P += dt*F ═══");

    let dt_test = 0.03;
    let mut cpu_mom = momenta.clone();
    for idx in 0..vol {
        let x = lat.site_coords(idx);
        for mu in 0..4 {
            let f = lat.gauge_force(x, mu);
            let p = &mut cpu_mom[idx * 4 + mu];
            for row in 0..3 {
                for col in 0..3 {
                    p.m[row][col].re += dt_test * f.m[row][col].re;
                    p.m[row][col].im += dt_test * f.m[row][col].im;
                }
            }
        }
    }

    let mom_rw_buf = gpu.create_f64_output_buffer(n_links * 18, "mom_rw");
    gpu.upload_f64(&mom_rw_buf, &mom_flat);

    let mom_update_params: Vec<u8> = {
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&(n_links as u32).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&dt_test.to_le_bytes());
        v
    };
    let mom_upd_param_buf = gpu.create_uniform_buffer(&mom_update_params, "mom_upd_params");

    let mom_upd_bg = gpu.create_bind_group(
        &pipelines.momentum_pipeline,
        &[&mom_upd_param_buf, &force_buf, &mom_rw_buf],
    );

    gpu.dispatch(&pipelines.momentum_pipeline, &mom_upd_bg, wg);
    let gpu_mom_after = read_f64(&gpu, &mom_rw_buf, n_links * 18);

    let cpu_mom_flat = flatten_momenta(&cpu_mom);
    let mut max_mom_err: f64 = 0.0;
    for (i, (&g, &c)) in gpu_mom_after.iter().zip(cpu_mom_flat.iter()).enumerate() {
        let _ = i;
        max_mom_err = max_mom_err.max((g - c).abs());
    }
    println!("  Max |GPU-CPU| momentum error: {max_mom_err:.2e}");
    harness.check_bool("GPU momentum update parity < 1e-10", max_mom_err < 1e-10);
    println!();

    // ═══ Phase 5: GPU link update (Cayley exp) vs CPU ═══
    println!("═══ Phase 5: GPU link update (Cayley exp) ═══");

    let link_rw_buf = gpu.create_f64_output_buffer(n_links * 18, "links_rw");
    gpu.upload_f64(&link_rw_buf, &links_flat);

    let link_upd_params: Vec<u8> = {
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&(n_links as u32).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&dt_test.to_le_bytes());
        v
    };
    let link_upd_param_buf = gpu.create_uniform_buffer(&link_upd_params, "link_upd_params");

    let link_upd_bg = gpu.create_bind_group(
        &pipelines.link_pipeline,
        &[&link_upd_param_buf, &mom_buf, &link_rw_buf],
    );

    gpu.dispatch(&pipelines.link_pipeline, &link_upd_bg, wg);
    let gpu_links_after = read_f64(&gpu, &link_rw_buf, n_links * 18);

    let mut max_link_err: f64 = 0.0;
    for idx in 0..vol {
        let x = lat.site_coords(idx);
        for mu in 0..4 {
            let p = momenta[idx * 4 + mu];
            let u = lat.link(x, mu);
            let exp_p = hmc::exp_su3_cayley_pub(&p, dt_test);
            let new_u = (exp_p * u).reunitarize();

            let base = (idx * 4 + mu) * 18;
            for row in 0..3 {
                for col in 0..3 {
                    max_link_err = max_link_err
                        .max(
                            (new_u.m[row][col].re - gpu_links_after[base + row * 6 + col * 2])
                                .abs(),
                        )
                        .max(
                            (new_u.m[row][col].im - gpu_links_after[base + row * 6 + col * 2 + 1])
                                .abs(),
                        );
                }
            }
        }
    }
    println!("  Max |GPU-CPU| link update error: {max_link_err:.2e}");
    harness.check_bool("GPU link update parity < 1e-10", max_link_err < 1e-10);
    println!();

    // ═══ Phase 6: Full GPU Omelyan HMC trajectory ═══
    println!("═══ Phase 6: Full GPU Omelyan HMC (all math on GPU) ═══");
    println!("  4⁴ lattice, β=6.0, dt=0.05, n_md=15, 10 trajectories");
    println!();

    use hotspring_barracuda::lattice::gpu_hmc::{gpu_hmc_trajectory, GpuHmcState};

    let mut lat_gpu = Lattice::hot_start(dims, beta, 42);
    let mut cfg_gpu = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };
    hmc::run_hmc(&mut lat_gpu, 20, 0, &mut cfg_gpu);

    let gpu_state = GpuHmcState::from_lattice(&gpu, &lat_gpu, beta);
    let mut seed = 1000u64;
    let n_traj = 10;
    let mut n_accepted = 0;
    let mut plaq_sum = 0.0;

    for i in 0..n_traj {
        let result = gpu_hmc_trajectory(&gpu, &pipelines, &gpu_state, 15, 0.05, &mut seed);
        let tag = if result.accepted { "✓" } else { "✗" };
        println!(
            "  traj {}: {} ΔH={:+.4e}  plaq={:.6}",
            i + 1,
            tag,
            result.delta_h,
            result.plaquette,
        );
        if result.accepted {
            n_accepted += 1;
        }
        plaq_sum += result.plaquette;
    }

    let accept_rate = n_accepted as f64 / n_traj as f64;
    let mean_plaq = plaq_sum / n_traj as f64;
    println!();
    println!(
        "  Acceptance: {n_accepted}/{n_traj} ({:.0}%)",
        accept_rate * 100.0,
    );
    println!("  Mean plaquette: {mean_plaq:.6}");

    harness.check_lower("GPU HMC acceptance > 30%", accept_rate, 0.30);
    harness.check_bool(
        "GPU HMC plaquette in physical range",
        mean_plaq > 0.45 && mean_plaq < 0.70,
    );
    println!();

    // ═══ Phase 7: Shader summary ═══
    println!("═══ Phase 7: Shader pipeline summary ═══");
    println!(
        "  wilson_plaquette_f64.wgsl    — {} bytes",
        WGSL_WILSON_PLAQUETTE.len()
    );
    println!(
        "  su3_gauge_force_f64.wgsl     — {} bytes",
        WGSL_GAUGE_FORCE.len()
    );
    println!(
        "  su3_momentum_update_f64.wgsl — {} bytes",
        WGSL_MOMENTUM_UPDATE.len()
    );
    println!(
        "  su3_link_update_f64.wgsl     — {} bytes",
        WGSL_LINK_UPDATE.len()
    );
    println!(
        "  su3_kinetic_energy_f64.wgsl  — {} bytes",
        WGSL_KINETIC_ENERGY.len()
    );
    println!();

    let elapsed = start_total.elapsed().as_secs_f64();
    println!("  Total wall time: {elapsed:.1}s");

    harness.finish();
}
