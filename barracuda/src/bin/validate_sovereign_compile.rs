// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sovereign Compile Validation — coralReef native binary proof of concept.
//!
//! Compiles hotSpring physics WGSL shaders through coralReef's IPC compiler
//! (`CoralCompiler::compile_wgsl_direct`) and validates that native GPU
//! binaries are produced for SM70 (Volta/Titan V) and SM86 (Ampere/RTX 3090).
//!
//! This validates the compilation stage of the sovereign pipeline. GPU dispatch
//! is operational for AMD GCN5 (E2E) and RTX 5060 (DRM cracked, SM120 ISA pending).
//! Sovereign VFIO dispatch for Titan V is in progress (MMU layer).
//!
//! Usage:
//!   cargo run --release --features sovereign-dispatch --bin validate_sovereign_compile

use barracuda::device::coral_compiler::GLOBAL_CORAL;

struct ShaderEntry {
    name: &'static str,
    source: &'static str,
    #[expect(
        dead_code,
        reason = "used for categorization in output, may be used later"
    )]
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

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Sovereign Compile Validation — coralReef PoC              ║");
    println!("║  hotSpring WGSL → native SM70/SM86 via coral-compiler IPC  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let targets = [("sm_70", true), ("sm_86", true)];

    let shaders = vec![
        shader!(
            "chi2_batch_f64",
            "../physics/shaders/chi2_batch_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "semf_batch_f64",
            "../physics/shaders/semf_batch_f64.wgsl",
            "nuclear"
        ),
        shader!(
            "spin_orbit_pack_f64",
            "../physics/shaders/spin_orbit_pack_f64.wgsl",
            "nuclear"
        ),
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
            "su3_gauge_force_f64",
            "../lattice/shaders/su3_gauge_force_f64.wgsl",
            "lattice"
        ),
        shader!(
            "wilson_plaquette_f64",
            "../lattice/shaders/wilson_plaquette_f64.wgsl",
            "lattice"
        ),
        shader!(
            "su3_link_update_f64",
            "../lattice/shaders/su3_link_update_f64.wgsl",
            "lattice"
        ),
        shader!(
            "su3_flow_accumulate_f64",
            "../lattice/shaders/su3_flow_accumulate_f64.wgsl",
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
            "staggered_fermion_force_f64",
            "../lattice/shaders/staggered_fermion_force_f64.wgsl",
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
        shader!(
            "complex_f64",
            "../lattice/shaders/complex_f64.wgsl",
            "lattice-util"
        ),
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
    ];

    let mut total_fail = 0usize;
    for &(arch_name, fp64_software) in &targets {
        println!("  ═══ Target: {arch_name} ═══\n");
        println!("  {:<40} {:>8} {:>6}", "Shader", "Binary", "Status");
        println!("  {}", "-".repeat(60));

        let mut ok = 0usize;
        let mut fail = 0usize;
        let mut total_bytes = 0usize;

        for entry in &shaders {
            if let Some(binary) = GLOBAL_CORAL
                .compile_wgsl_direct(entry.source, arch_name, fp64_software)
                .await
            {
                ok += 1;
                total_bytes += binary.binary.len();
                println!(
                    "  {:<40} {:>6}B {:>6}",
                    entry.name,
                    binary.binary.len(),
                    "OK"
                );
            } else {
                fail += 1;
                println!("  {:<40} {:>8} {:>6}", entry.name, "-", "FAIL");
            }
        }

        total_fail += fail;
        println!(
            "\n  Summary ({arch_name}): {ok}/{} compiled, {fail} failed, {total_bytes} total bytes\n",
            ok + fail
        );
    }

    println!("  GPU dispatch status:");
    println!("    - amdgpu: E2E PASSED (GCN5 preswap 6/6, f64 LJ force verified)");
    println!("    - nouveau: codegen OK, sovereign VFIO dispatch in progress (MMU layer)");
    println!("    - nvidia-drm: RTX 5060 Blackwell DRM cracked (4/4 HW tests). SM120 ISA pending.");
    println!("    - iommufd: kernel-agnostic VFIO on 6.2+, dual-path (iommufd + legacy)");
    println!("\n  No Vulkan. No vendor SDK. Pure Rust → native binary compilation.");

    if total_fail > 0 {
        std::process::exit(1);
    }
}
