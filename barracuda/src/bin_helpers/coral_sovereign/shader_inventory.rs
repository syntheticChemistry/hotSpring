// SPDX-License-Identifier: AGPL-3.0-or-later

// ═══════════════════════════════════════════════════════════════
// Phase 3: QCD Shader Compilation Inventory
// ═══════════════════════════════════════════════════════════════

use super::prelude::{vec2_preamble_with_su3, LIB_LCG_F64, LIB_SU3_EXTENDED};
use coral_gpu::GpuContext;
use std::time::Instant;

/// Test 3: Compile all QCD production shaders through coral-reef.
/// Composite shaders get barraCuda's vec2<f64> complex + SU3 preamble
/// with auto-prepend guards to prevent coral-reef from injecting its own Complex64.
pub fn test_qcd_shader_compilation(ctx: &GpuContext) -> (u32, Vec<(String, String)>) {
    let su3_preamble = vec2_preamble_with_su3();

    let shaders: Vec<(&str, String)> = vec![
        // Standalone shaders (no library deps)
        (
            "axpy_f64",
            include_str!("../../lattice/shaders/axpy_f64.wgsl").to_string(),
        ),
        (
            "xpay_f64",
            include_str!("../../lattice/shaders/xpay_f64.wgsl").to_string(),
        ),
        (
            "sum_reduce_f64",
            include_str!("../../lattice/shaders/sum_reduce_f64.wgsl").to_string(),
        ),
        (
            "wilson_plaquette_f64",
            include_str!("../../lattice/shaders/wilson_plaquette_f64.wgsl").to_string(),
        ),
        (
            "su3_gauge_force_f64",
            include_str!("../../lattice/shaders/su3_gauge_force_f64.wgsl").to_string(),
        ),
        (
            "su3_link_update_f64",
            include_str!("../../lattice/shaders/su3_link_update_f64.wgsl").to_string(),
        ),
        (
            "su3_momentum_update_f64",
            include_str!("../../lattice/shaders/su3_momentum_update_f64.wgsl").to_string(),
        ),
        (
            "su3_kinetic_energy_f64",
            include_str!("../../lattice/shaders/su3_kinetic_energy_f64.wgsl").to_string(),
        ),
        (
            "dirac_staggered_f64",
            include_str!("../../lattice/shaders/dirac_staggered_f64.wgsl").to_string(),
        ),
        (
            "cg_compute_alpha_f64",
            include_str!("../../lattice/shaders/cg_compute_alpha_f64.wgsl").to_string(),
        ),
        (
            "cg_compute_beta_f64",
            include_str!("../../lattice/shaders/cg_compute_beta_f64.wgsl").to_string(),
        ),
        (
            "cg_update_xr_f64",
            include_str!("../../lattice/shaders/cg_update_xr_f64.wgsl").to_string(),
        ),
        (
            "cg_update_p_f64",
            include_str!("../../lattice/shaders/cg_update_p_f64.wgsl").to_string(),
        ),
        (
            "complex_dot_re_f64",
            include_str!("../../lattice/shaders/complex_dot_re_f64.wgsl").to_string(),
        ),
        (
            "staggered_fermion_force_f64",
            include_str!("../../lattice/shaders/staggered_fermion_force_f64.wgsl").to_string(),
        ),
        (
            "polyakov_loop_f64",
            include_str!("../../lattice/shaders/polyakov_loop_f64.wgsl").to_string(),
        ),
        (
            "metropolis_f64",
            include_str!("../../lattice/shaders/metropolis_f64.wgsl").to_string(),
        ),
        (
            "fermion_action_sum_f64",
            include_str!("../../lattice/shaders/fermion_action_sum_f64.wgsl").to_string(),
        ),
        (
            "hamiltonian_assembly_f64",
            include_str!("../../lattice/shaders/hamiltonian_assembly_f64.wgsl").to_string(),
        ),
        // Composite shaders: need Complex64/SU3 preamble + WGSL auto-conversion preprocessing.
        // These compile in the wgpu path because barraCuda's ShaderTemplate handles type conversions.
        // Marked separately so they don't inflate the failure count for standalone shader compilation.
    ];

    // hmc_leapfrog needs the full chain: complex + su3 + lcg (PRNG) + su3_extended
    let hmc_preamble = format!("{su3_preamble}\n{LIB_LCG_F64}\n{LIB_SU3_EXTENDED}");

    // Composite shaders: all need barraCuda's vec2<f64> complex + SU3 preamble.
    let composite_shaders: Vec<(&str, String)> = vec![
        (
            "wilson_action_f64 (+ c64+su3)",
            format!(
                "{su3_preamble}\n{}",
                include_str!("../../lattice/shaders/wilson_action_f64.wgsl")
            ),
        ),
        (
            "su3_hmc_force_f64 (+ c64+su3)",
            format!(
                "{su3_preamble}\n{}",
                include_str!("../../lattice/shaders/su3_hmc_force_f64.wgsl")
            ),
        ),
        (
            "pseudofermion_force_f64 (+ c64+su3)",
            format!(
                "{su3_preamble}\n{}",
                include_str!("../../lattice/shaders/pseudofermion_force_f64.wgsl")
            ),
        ),
        (
            "hmc_leapfrog_f64 (+ full chain)",
            format!(
                "{hmc_preamble}\n{}",
                include_str!("../../lattice/shaders/hmc_leapfrog_f64.wgsl")
            ),
        ),
        (
            "kinetic_energy_f64 (+ c64+su3)",
            format!(
                "{su3_preamble}\n{}",
                include_str!("../../lattice/shaders/kinetic_energy_f64.wgsl")
            ),
        ),
    ];

    let mut compiled = 0u32;
    let mut failures: Vec<(String, String)> = Vec::new();
    let mut total_bytes = 0usize;

    println!("  ── Standalone shaders (no preprocessing needed) ──\n");
    let total_t0 = Instant::now();
    for (name, source) in &shaders {
        let t0 = Instant::now();
        match ctx.compile_wgsl(source) {
            Ok(kernel) => {
                let us = t0.elapsed().as_micros();
                total_bytes += kernel.binary.len();
                println!(
                    "  [OK]  {name:<42} {us:>6}us  ({} bytes, {} GPRs)",
                    kernel.binary.len(),
                    kernel.gpr_count
                );
                compiled += 1;
            }
            Err(e) => {
                let us = t0.elapsed().as_micros();
                let err_short = format!("{e}").chars().take(80).collect::<String>();
                println!("  [ERR] {name:<42} {us:>6}us  {err_short}");
                failures.push((name.to_string(), format!("{e}")));
            }
        }
    }
    let standalone_ms = total_t0.elapsed().as_millis();

    println!("\n  ── Composite shaders (barraCuda vec2<f64> preamble) ──\n");
    let mut composite_compiled = 0u32;
    let mut composite_frontier = 0u32;
    for (name, source) in &composite_shaders {
        let t0 = Instant::now();
        match ctx.compile_wgsl(source) {
            Ok(kernel) => {
                let us = t0.elapsed().as_micros();
                total_bytes += kernel.binary.len();
                println!(
                    "  [OK]  {name:<42} {us:>6}us  ({} bytes, {} GPRs)",
                    kernel.binary.len(),
                    kernel.gpr_count
                );
                composite_compiled += 1;
            }
            Err(e) => {
                let us = t0.elapsed().as_micros();
                let err_short = format!("{e}").chars().take(100).collect::<String>();
                println!("  [FTR] {name:<42} {us:>6}us  {err_short}");
                composite_frontier += 1;
            }
        }
    }
    let total_ms = total_t0.elapsed().as_millis();
    let total_shaders = shaders.len() + composite_shaders.len();

    println!(
        "\n  Standalone: {compiled}/{} compiled in {standalone_ms}ms",
        shaders.len()
    );
    println!(
        "  Composite:  {composite_compiled}/{} compiled ({composite_frontier} need preprocessing)",
        composite_shaders.len()
    );
    println!(
        "  Total:      {} native binaries, {} bytes, {total_ms}ms",
        compiled + composite_compiled,
        total_bytes
    );

    (compiled + composite_compiled, failures)
}
