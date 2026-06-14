// SPDX-License-Identifier: AGPL-3.0-or-later

//! DF64 transcendental poisoning diagnostic: tests whether exp_df64/sqrt_df64
//! produce correct (non-zero) forces on each available GPU/driver combination.
//!
//! Bypasses the has_df64_spir_v_poisoning() safety fallback to test the
//! DF64 shader path directly. Compares DF64 vs native f64 forces to validate.
//!
//! Usage:
//!   HOTSPRING_GPU_ADAPTER=titan  cargo run --release --bin test_df64_nvk_poison
//!   HOTSPRING_GPU_ADAPTER=3090   cargo run --release --bin test_df64_nvk_poison
//!   cargo run --release --bin test_df64_nvk_poison  # default GPU

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::config::MdConfig;
use hotspring_barracuda::md::shaders;
use hotspring_barracuda::md::simulation::{init_fcc_lattice, init_velocities};
use hotspring_barracuda::tolerances::{DEFAULT_VELOCITY_SEED, MD_WORKGROUP_SIZE};

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(run_test());
}

async fn run_test() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  DF64 Transcendental Poisoning Diagnostic                  ║");
    println!("║  Tests: exp_df64 / sqrt_df64 on each driver                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU init failed: {e}");
            return;
        }
    };

    if !gpu.has_f64 {
        println!("  GPU lacks SHADER_F64 — cannot test.");
        return;
    }

    gpu.print_info();
    let caps = gpu.capabilities();
    let poisoning_risk = caps.has_df64_spir_v_poisoning();
    println!("  has_df64_spir_v_poisoning(): {poisoning_risk}");
    println!("  fp64_strategy(): {:?}", caps.fp64_strategy());
    println!();

    let n: usize = 500;
    let config = MdConfig {
        label: "df64_poison_test".to_string(),
        n_particles: n,
        kappa: 2.0,
        gamma: 158.0,
        dt: 0.01,
        rc: 6.5,
        equil_steps: 0,
        prod_steps: 0,
        dump_step: 1,
        berendsen_tau: 5.0,
        rdf_bins: 100,
        vel_snapshot_interval: 10,
    };

    let box_side = config.box_side();
    let prefactor = config.force_prefactor();
    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    let _velocities = init_velocities(
        n,
        config.temperature(),
        config.reduced_mass(),
        DEFAULT_VELOCITY_SEED,
    );

    let force_params: Vec<f64> = vec![
        n as f64,
        config.kappa,
        prefactor,
        config.rc * config.rc,
        box_side,
        box_side,
        box_side,
        0.0,
    ];

    let pos_buf = gpu.create_f64_output_buffer(n * 3, "positions");
    let force_buf = gpu.create_f64_output_buffer(n * 3, "forces");
    let pe_buf = gpu.create_f64_output_buffer(n, "pe");
    let params_buf = gpu.create_f64_buffer(&force_params, "params");
    let workgroups = n.div_ceil(MD_WORKGROUP_SIZE) as u32;

    gpu.upload_f64(&pos_buf, &positions);

    // ── Test 1: Native f64 force shader ──
    println!("  ── Test 1: Native f64 (SHADER_YUKAWA_FORCE) ──");
    let f64_pipeline = gpu.create_pipeline_f64(shaders::SHADER_YUKAWA_FORCE, "yukawa_f64");
    let f64_bg =
        gpu.create_bind_group(&f64_pipeline, &[&pos_buf, &force_buf, &pe_buf, &params_buf]);
    gpu.dispatch(&f64_pipeline, &f64_bg, workgroups);

    let forces_f64 = gpu.read_back_f64(&force_buf, n * 3).expect("read_back_f64");
    let f64_nonzero = forces_f64.iter().filter(|f| f.abs() > 1e-30).count();
    let f64_max = forces_f64
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let f64_sum: f64 = forces_f64.iter().copied().map(f64::abs).sum();

    println!("    Non-zero components: {f64_nonzero} / {}", n * 3);
    println!("    Max |force|: {f64_max:.6e}");
    println!("    Sum |force|: {f64_sum:.6e}");
    let f64_ok = f64_nonzero > n;
    println!(
        "    Status: {}",
        if f64_ok {
            "PASS"
        } else {
            "FAIL — forces are zero"
        }
    );
    println!();

    // ── Test 2: DF64 force shader (bypass safety) ──
    println!("  ── Test 2: DF64 (SHADER_YUKAWA_FORCE_DF64) — bypassing safety ──");

    let zeros = vec![0.0f64; n * 3];
    gpu.upload_f64(&force_buf, &zeros);

    let df64_pipeline = gpu.create_pipeline_df64(shaders::SHADER_YUKAWA_FORCE_DF64, "yukawa_df64");
    let df64_bg = gpu.create_bind_group(
        &df64_pipeline,
        &[&pos_buf, &force_buf, &pe_buf, &params_buf],
    );
    gpu.dispatch(&df64_pipeline, &df64_bg, workgroups);

    let forces_df64 = gpu
        .read_back_f64(&force_buf, n * 3)
        .expect("read_back_f64 df64");
    let df64_nonzero = forces_df64.iter().filter(|f| f.abs() > 1e-30).count();
    let df64_max = forces_df64
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let df64_sum: f64 = forces_df64.iter().copied().map(f64::abs).sum();

    println!("    Non-zero components: {df64_nonzero} / {}", n * 3);
    println!("    Max |force|: {df64_max:.6e}");
    println!("    Sum |force|: {df64_sum:.6e}");
    let df64_ok = df64_nonzero > n;
    println!(
        "    Status: {}",
        if df64_ok {
            "PASS"
        } else {
            "FAIL — DF64 transcendentals POISONED"
        }
    );
    println!();

    // ── Comparison ──
    if f64_ok && df64_ok {
        let mut max_reldiff = 0.0_f64;
        let mut sum_reldiff = 0.0_f64;
        let mut count = 0usize;
        for i in 0..forces_f64.len() {
            let a = forces_f64[i];
            let b = forces_df64[i];
            if a.abs() > 1e-20 {
                let rd = ((a - b) / a).abs();
                max_reldiff = max_reldiff.max(rd);
                sum_reldiff += rd;
                count += 1;
            }
        }
        let avg_reldiff = if count > 0 {
            sum_reldiff / count as f64
        } else {
            0.0
        };
        println!("  ── Force Comparison: f64 vs DF64 ──");
        println!("    Max relative diff: {max_reldiff:.6e}");
        println!("    Avg relative diff: {avg_reldiff:.6e}");
        println!("    Components compared: {count}");
        let precision_ok = max_reldiff < 1e-6;
        println!(
            "    Precision: {}",
            if precision_ok {
                "GOOD (< 1e-6)"
            } else {
                "DEGRADED (> 1e-6, expected for DF64)"
            }
        );
    } else if f64_ok && !df64_ok {
        println!("  ── DF64 POISONED: exp_df64/sqrt_df64 produce zero on this driver ──");
        println!("  Root cause: NVVM JIT corrupts f32-pair transcendentals in SPIR-V path");
        println!("  Workaround: coralReef sovereign compile (bypasses NVVM)");
    }

    // ── Summary ──
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GPU:         {}", gpu.adapter_name);
    println!("  Strategy:    {:?}", caps.fp64_strategy());
    println!("  NVVM risk:   {poisoning_risk}");
    println!("  f64 force:   {}", if f64_ok { "PASS" } else { "FAIL" });
    println!(
        "  DF64 force:  {}",
        if df64_ok { "PASS" } else { "POISONED" }
    );
    if f64_ok && df64_ok {
        println!("  Conclusion:  DF64 transcendentals SAFE on this driver — can use DF64 Yukawa");
    } else if !df64_ok {
        println!("  Conclusion:  DF64 transcendentals POISONED — sovereign bypass or NVK required");
    }
    println!();
}
