// SPDX-License-Identifier: AGPL-3.0-only

//! Sovereign Compiler FMA Fusion Benchmark
//!
//! Feeds all hotSpring WGSL shaders through barraCuda's SovereignCompiler
//! and reports FMA fusion and dead expression elimination statistics.
//! This quantifies the optimization impact of the sovereign pipeline
//! on physics, lattice, and MD shaders.
//!
//! Usage:
//!   cargo run --release --bin bench_sovereign_fma

use barracuda::shaders::sovereign::{dead_expr, fma_fusion};

struct ShaderEntry {
    name: &'static str,
    source: &'static str,
    category: &'static str,
}

macro_rules! shader {
    ($name:expr, $path:expr, $cat:expr) => {
        ShaderEntry {
            name: $name,
            source: include_str!($path),
            category: $cat,
        }
    };
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Sovereign Compiler FMA Fusion Benchmark                   ║");
    println!("║  barraCuda FMA fusion + dead-expr on hotSpring WGSL        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let shaders = vec![
        // ── Nuclear physics (HFB, SEMF, deformed) ──
        shader!(
            "batched_hfb_density_f64",
            "../physics/shaders/batched_hfb_density_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "batched_hfb_energy_f64",
            "../physics/shaders/batched_hfb_energy_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "batched_hfb_hamiltonian_f64",
            "../physics/shaders/batched_hfb_hamiltonian_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "batched_hfb_potentials_f64",
            "../physics/shaders/batched_hfb_potentials_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "bcs_bisection_f64",
            "../physics/shaders/bcs_bisection_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "chi2_batch_f64",
            "../physics/shaders/chi2_batch_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "deformed_density_energy_f64",
            "../physics/shaders/deformed_density_energy_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "deformed_gradient_f64",
            "../physics/shaders/deformed_gradient_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "deformed_hamiltonian_f64",
            "../physics/shaders/deformed_hamiltonian_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "deformed_potentials_f64",
            "../physics/shaders/deformed_potentials_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "deformed_wavefunction_f64",
            "../physics/shaders/deformed_wavefunction_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "semf_batch_f64",
            "../physics/shaders/semf_batch_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "semf_pure_gpu_f64",
            "../physics/shaders/semf_pure_gpu_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "spin_orbit_pack_f64",
            "../physics/shaders/spin_orbit_pack_f64.wgsl",
            "nuclear"
        ),
        // ── Plasma physics (dielectric, BGK) ──
        shader!(
            "dielectric_mermin_f64",
            "../physics/shaders/dielectric_mermin_f64.wgsl",
            "plasma"
        ),
        shader!(
            "bgk_relaxation_f64",
            "../physics/shaders/bgk_relaxation_f64.wgsl",
            "plasma"
        ),
        // ── Lattice QCD f64 ──
        shader!(
            "su3_gauge_force_f64",
            "../lattice/shaders/su3_gauge_force_f64.wgsl",
            "lattice"
        ),
        shader!(
            "su3_hmc_force_f64",
            "../lattice/shaders/su3_hmc_force_f64.wgsl",
            "lattice"
        ),
        shader!(
            "wilson_plaquette_f64",
            "../lattice/shaders/wilson_plaquette_f64.wgsl",
            "lattice"
        ),
        shader!(
            "hmc_leapfrog_f64",
            "../lattice/shaders/hmc_leapfrog_f64.wgsl",
            "lattice"
        ),
        shader!(
            "su3_link_update_f64",
            "../lattice/shaders/su3_link_update_f64.wgsl",
            "lattice"
        ),
        shader!(
            "su3_random_momenta_f64",
            "../lattice/shaders/su3_random_momenta_f64.wgsl",
            "lattice"
        ),
        shader!(
            "su3_flow_accumulate_f64",
            "../lattice/shaders/su3_flow_accumulate_f64.wgsl",
            "lattice"
        ),
        shader!(
            "wilson_action_f64",
            "../lattice/shaders/wilson_action_f64.wgsl",
            "lattice"
        ),
        shader!(
            "cg_kernels_f64",
            "../lattice/shaders/cg_kernels_f64.wgsl",
            "lattice"
        ),
        shader!(
            "dirac_staggered_f64",
            "../lattice/shaders/dirac_staggered_f64.wgsl",
            "lattice"
        ),
        shader!(
            "pseudofermion_force_f64",
            "../lattice/shaders/pseudofermion_force_f64.wgsl",
            "lattice"
        ),
        shader!(
            "su3_kinetic_energy_f64",
            "../lattice/shaders/su3_kinetic_energy_f64.wgsl",
            "lattice"
        ),
        shader!(
            "su3_momentum_update_f64",
            "../lattice/shaders/su3_momentum_update_f64.wgsl",
            "lattice"
        ),
        shader!(
            "sum_reduce_f64",
            "../lattice/shaders/sum_reduce_f64.wgsl",
            "lattice"
        ),
        shader!(
            "polyakov_loop_f64",
            "../lattice/shaders/polyakov_loop_f64.wgsl",
            "lattice"
        ),
        shader!(
            "lattice_init_f64",
            "../lattice/shaders/lattice_init_f64.wgsl",
            "lattice"
        ),
        shader!(
            "pseudofermion_heatbath_f64",
            "../lattice/shaders/pseudofermion_heatbath_f64.wgsl",
            "lattice"
        ),
        shader!(
            "staggered_fermion_force_f64",
            "../lattice/shaders/staggered_fermion_force_f64.wgsl",
            "lattice"
        ),
        shader!(
            "gaussian_fermion_f64",
            "../lattice/shaders/gaussian_fermion_f64.wgsl",
            "lattice"
        ),
        shader!(
            "higgs_u1_hmc_f64",
            "../lattice/shaders/higgs_u1_hmc_f64.wgsl",
            "lattice"
        ),
        // ── Lattice CG helper shaders ──
        shader!("axpy_f64", "../lattice/shaders/axpy_f64.wgsl", "lattice-cg"),
        shader!("xpay_f64", "../lattice/shaders/xpay_f64.wgsl", "lattice-cg"),
        shader!(
            "cg_update_p_f64",
            "../lattice/shaders/cg_update_p_f64.wgsl",
            "lattice-cg"
        ),
        shader!(
            "cg_compute_beta_f64",
            "../lattice/shaders/cg_compute_beta_f64.wgsl",
            "lattice-cg"
        ),
        shader!(
            "cg_update_xr_f64",
            "../lattice/shaders/cg_update_xr_f64.wgsl",
            "lattice-cg"
        ),
        shader!(
            "cg_compute_alpha_f64",
            "../lattice/shaders/cg_compute_alpha_f64.wgsl",
            "lattice-cg"
        ),
        shader!(
            "complex_dot_re_f64",
            "../lattice/shaders/complex_dot_re_f64.wgsl",
            "lattice-cg"
        ),
        // ── Lattice DF64 (double-float) shaders ──
        shader!(
            "su3_gauge_force_df64",
            "../lattice/shaders/su3_gauge_force_df64.wgsl",
            "lattice-df64"
        ),
        shader!(
            "su3_hmc_force_df64",
            "../lattice/shaders/su3_hmc_force_df64.wgsl",
            "lattice-df64"
        ),
        shader!(
            "wilson_plaquette_df64",
            "../lattice/shaders/wilson_plaquette_df64.wgsl",
            "lattice-df64"
        ),
        shader!(
            "wilson_action_df64",
            "../lattice/shaders/wilson_action_df64.wgsl",
            "lattice-df64"
        ),
        shader!(
            "su3_kinetic_energy_df64",
            "../lattice/shaders/su3_kinetic_energy_df64.wgsl",
            "lattice-df64"
        ),
        shader!(
            "kinetic_energy_df64",
            "../lattice/shaders/kinetic_energy_df64.wgsl",
            "lattice-df64"
        ),
        // ── Lattice utility shaders ──
        shader!("su3_f64", "../lattice/shaders/su3_f64.wgsl", "lattice-util"),
        shader!(
            "su3_math_f64",
            "../lattice/shaders/su3_math_f64.wgsl",
            "lattice-util"
        ),
        shader!(
            "complex_f64",
            "../lattice/shaders/complex_f64.wgsl",
            "lattice-util"
        ),
        shader!(
            "prng_pcg_f64",
            "../lattice/shaders/prng_pcg_f64.wgsl",
            "lattice-util"
        ),
        // ── MD shaders ──
        shader!(
            "yukawa_force_f64",
            "../md/shaders/yukawa_force_f64.wgsl",
            "md"
        ),
        shader!(
            "yukawa_force_verlet_f64",
            "../md/shaders/yukawa_force_verlet_f64.wgsl",
            "md"
        ),
        shader!(
            "yukawa_force_celllist_f64",
            "../md/shaders/yukawa_force_celllist_f64.wgsl",
            "md"
        ),
        shader!(
            "yukawa_force_celllist_v2_f64",
            "../md/shaders/yukawa_force_celllist_v2_f64.wgsl",
            "md"
        ),
        shader!(
            "yukawa_force_celllist_indirect_f64",
            "../md/shaders/yukawa_force_celllist_indirect_f64.wgsl",
            "md"
        ),
        shader!(
            "vv_half_kick_f64",
            "../md/shaders/vv_half_kick_f64.wgsl",
            "md"
        ),
        shader!(
            "vv_kick_drift_f64",
            "../md/shaders/vv_kick_drift_f64.wgsl",
            "md"
        ),
        shader!(
            "kinetic_energy_f64",
            "../md/shaders/kinetic_energy_f64.wgsl",
            "md"
        ),
        shader!("berendsen_f64", "../md/shaders/berendsen_f64.wgsl", "md"),
        shader!(
            "rdf_histogram_f64",
            "../md/shaders/rdf_histogram_f64.wgsl",
            "md"
        ),
        shader!("vacf_batch_f64", "../md/shaders/vacf_batch_f64.wgsl", "md"),
        shader!("vacf_dot_f64", "../md/shaders/vacf_dot_f64.wgsl", "md"),
        shader!(
            "stress_virial_f64",
            "../md/shaders/stress_virial_f64.wgsl",
            "md"
        ),
        shader!("verlet_build", "../md/shaders/verlet_build.wgsl", "md"),
        shader!(
            "verlet_check_displacement",
            "../md/shaders/verlet_check_displacement.wgsl",
            "md"
        ),
        shader!(
            "verlet_copy_ref",
            "../md/shaders/verlet_copy_ref.wgsl",
            "md"
        ),
        shader!("esn_readout", "../md/shaders/esn_readout.wgsl", "md-esn"),
        shader!(
            "esn_reservoir_update",
            "../md/shaders/esn_reservoir_update.wgsl",
            "md-esn"
        ),
        // ── MD DF64 (double-float) shaders ──
        shader!(
            "yukawa_force_df64",
            "../md/shaders/yukawa_force_df64.wgsl",
            "md-df64"
        ),
        shader!(
            "yukawa_force_verlet_df64",
            "../md/shaders/yukawa_force_verlet_df64.wgsl",
            "md-df64"
        ),
        shader!(
            "yukawa_force_celllist_indirect_df64",
            "../md/shaders/yukawa_force_celllist_indirect_df64.wgsl",
            "md-df64"
        ),
    ];

    let mut total_fma = 0;
    let mut total_dead = 0;
    let mut total_shaders = 0;
    let mut failed = 0;

    let mut by_category: std::collections::BTreeMap<&str, (usize, usize, usize)> =
        std::collections::BTreeMap::new();

    println!(
        "  {:<45} {:>6} {:>6} {:>8}",
        "Shader", "FMA", "Dead", "Status"
    );
    println!("  {}", "-".repeat(70));

    for entry in &shaders {
        match naga::front::wgsl::parse_str(entry.source) {
            Ok(mut module) => {
                let mut fma_count = 0usize;
                let mut dead_count = 0usize;

                for (_handle, func) in module.functions.iter_mut() {
                    fma_count += fma_fusion::fuse_multiply_add(&mut func.expressions);
                }
                for ep in &mut module.entry_points {
                    fma_count += fma_fusion::fuse_multiply_add(&mut ep.function.expressions);
                }
                for (_handle, func) in module.functions.iter_mut() {
                    dead_count += dead_expr::eliminate(&mut func.expressions, &func.body);
                }
                for ep in &mut module.entry_points {
                    dead_count +=
                        dead_expr::eliminate(&mut ep.function.expressions, &ep.function.body);
                }

                total_fma += fma_count;
                total_dead += dead_count;
                total_shaders += 1;

                let cat = by_category.entry(entry.category).or_insert((0, 0, 0));
                cat.0 += fma_count;
                cat.1 += dead_count;
                cat.2 += 1;

                let status = if fma_count > 0 { "optimized" } else { "clean" };
                println!(
                    "  {:<45} {:>6} {:>6} {:>8}",
                    entry.name, fma_count, dead_count, status
                );
            }
            Err(e) => {
                failed += 1;
                println!("  {:<45} {:>6} {:>6} {:>8}", entry.name, "-", "-", "FAIL");
                eprintln!("    Error: {e}");
            }
        }
    }

    println!("\n  ── Summary ──");
    println!("  Shaders analyzed: {total_shaders} ({failed} failed)");
    println!("  Total FMA fusions: {total_fma}");
    println!("  Total dead exprs eliminated: {total_dead}");
    println!(
        "  Avg FMA fusions per shader: {:.1}",
        if total_shaders > 0 {
            total_fma as f64 / total_shaders as f64
        } else {
            0.0
        }
    );

    println!("\n  ── By Category ──");
    println!(
        "  {:<12} {:>8} {:>8} {:>8} {:>10}",
        "Category", "Shaders", "FMA", "Dead", "FMA/shader"
    );
    for (cat, (fma, dead, count)) in &by_category {
        println!(
            "  {cat:<12} {count:>8} {fma:>8} {dead:>8} {:>10.1}",
            *fma as f64 / *count as f64
        );
    }
}
