// SPDX-License-Identifier: AGPL-3.0-only

//! Sovereign Compile Validation — coralReef native binary proof of concept.
//!
//! Compiles hotSpring physics WGSL shaders through coralReef's in-process
//! compiler (`coral-gpu::GpuContext::compile_wgsl`) and validates that native
//! GPU binaries are produced for SM70 (Volta/Titan V) and SM86 (Ampere/RTX 3090).
//!
//! This validates the compilation stage of the sovereign pipeline. GPU dispatch
//! validation will follow when coral-driver DRM backends mature.
//!
//! Usage:
//!   cargo run --release --features sovereign-dispatch --bin validate_sovereign_compile

use barracuda::device::coral_gpu::{GpuTarget, NvArch};
use barracuda::device::CoralReefDevice;

struct ShaderEntry {
    name: &'static str,
    source: &'static str,
    #[expect(dead_code, reason = "used for categorization in output, may be used later")]
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
    println!("║  Sovereign Compile Validation — coralReef PoC              ║");
    println!("║  hotSpring WGSL → native SM70/SM86 via coral-gpu           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let dev = CoralReefDevice::new(GpuTarget::Nvidia(NvArch::Sm70))
        .expect("CoralReefDevice should init");

    let targets = [
        (GpuTarget::Nvidia(NvArch::Sm70), "sm_70"),
        (GpuTarget::Nvidia(NvArch::Sm86), "sm_86"),
    ];

    let shaders = vec![
        shader!("chi2_batch_f64", "../physics/shaders/chi2_batch_f64.wgsl", "nuclear"),
        shader!("semf_batch_f64", "../physics/shaders/semf_batch_f64.wgsl", "nuclear"),
        shader!("spin_orbit_pack_f64", "../physics/shaders/spin_orbit_pack_f64.wgsl", "nuclear"),
        shader!("batched_hfb_density_f64", "../physics/shaders/batched_hfb_density_f64.wgsl", "nuclear"),
        shader!("batched_hfb_energy_f64", "../physics/shaders/batched_hfb_energy_f64.wgsl", "nuclear"),
        shader!("batched_hfb_hamiltonian_f64", "../physics/shaders/batched_hfb_hamiltonian_f64.wgsl", "nuclear"),
        shader!("deformed_density_energy_f64", "../physics/shaders/deformed_density_energy_f64.wgsl", "nuclear"),
        shader!("deformed_gradient_f64", "../physics/shaders/deformed_gradient_f64.wgsl", "nuclear"),
        shader!("deformed_hamiltonian_f64", "../physics/shaders/deformed_hamiltonian_f64.wgsl", "nuclear"),
        shader!("deformed_potentials_f64", "../physics/shaders/deformed_potentials_f64.wgsl", "nuclear"),
        shader!("su3_gauge_force_f64", "../lattice/shaders/su3_gauge_force_f64.wgsl", "lattice"),
        shader!("wilson_plaquette_f64", "../lattice/shaders/wilson_plaquette_f64.wgsl", "lattice"),
        shader!("su3_link_update_f64", "../lattice/shaders/su3_link_update_f64.wgsl", "lattice"),
        shader!("su3_flow_accumulate_f64", "../lattice/shaders/su3_flow_accumulate_f64.wgsl", "lattice"),
        shader!("cg_kernels_f64", "../lattice/shaders/cg_kernels_f64.wgsl", "lattice"),
        shader!("dirac_staggered_f64", "../lattice/shaders/dirac_staggered_f64.wgsl", "lattice"),
        shader!("staggered_fermion_force_f64", "../lattice/shaders/staggered_fermion_force_f64.wgsl", "lattice"),
        shader!("su3_kinetic_energy_f64", "../lattice/shaders/su3_kinetic_energy_f64.wgsl", "lattice"),
        shader!("su3_momentum_update_f64", "../lattice/shaders/su3_momentum_update_f64.wgsl", "lattice"),
        shader!("sum_reduce_f64", "../lattice/shaders/sum_reduce_f64.wgsl", "lattice"),
        shader!("axpy_f64", "../lattice/shaders/axpy_f64.wgsl", "lattice-cg"),
        shader!("xpay_f64", "../lattice/shaders/xpay_f64.wgsl", "lattice-cg"),
        shader!("cg_update_p_f64", "../lattice/shaders/cg_update_p_f64.wgsl", "lattice-cg"),
        shader!("cg_compute_beta_f64", "../lattice/shaders/cg_compute_beta_f64.wgsl", "lattice-cg"),
        shader!("cg_update_xr_f64", "../lattice/shaders/cg_update_xr_f64.wgsl", "lattice-cg"),
        shader!("cg_compute_alpha_f64", "../lattice/shaders/cg_compute_alpha_f64.wgsl", "lattice-cg"),
        shader!("complex_dot_re_f64", "../lattice/shaders/complex_dot_re_f64.wgsl", "lattice-cg"),
        shader!("complex_f64", "../lattice/shaders/complex_f64.wgsl", "lattice-util"),
        shader!("yukawa_force_f64", "../md/shaders/yukawa_force_f64.wgsl", "md"),
        shader!("yukawa_force_verlet_f64", "../md/shaders/yukawa_force_verlet_f64.wgsl", "md"),
        shader!("yukawa_force_celllist_f64", "../md/shaders/yukawa_force_celllist_f64.wgsl", "md"),
        shader!("yukawa_force_celllist_v2_f64", "../md/shaders/yukawa_force_celllist_v2_f64.wgsl", "md"),
        shader!("yukawa_force_celllist_indirect_f64", "../md/shaders/yukawa_force_celllist_indirect_f64.wgsl", "md"),
        shader!("vv_half_kick_f64", "../md/shaders/vv_half_kick_f64.wgsl", "md"),
        shader!("vv_kick_drift_f64", "../md/shaders/vv_kick_drift_f64.wgsl", "md"),
        shader!("kinetic_energy_f64", "../md/shaders/kinetic_energy_f64.wgsl", "md"),
        shader!("berendsen_f64", "../md/shaders/berendsen_f64.wgsl", "md"),
        shader!("rdf_histogram_f64", "../md/shaders/rdf_histogram_f64.wgsl", "md"),
        shader!("vacf_batch_f64", "../md/shaders/vacf_batch_f64.wgsl", "md"),
        shader!("vacf_dot_f64", "../md/shaders/vacf_dot_f64.wgsl", "md"),
        shader!("stress_virial_f64", "../md/shaders/stress_virial_f64.wgsl", "md"),
        shader!("verlet_build", "../md/shaders/verlet_build.wgsl", "md"),
        shader!("verlet_check_displacement", "../md/shaders/verlet_check_displacement.wgsl", "md"),
        shader!("verlet_copy_ref", "../md/shaders/verlet_copy_ref.wgsl", "md"),
        shader!("esn_readout", "../md/shaders/esn_readout.wgsl", "md-esn"),
        shader!("esn_reservoir_update", "../md/shaders/esn_reservoir_update.wgsl", "md-esn"),
    ];

    for &(target, arch_name) in &targets {
        println!("  ═══ Target: {arch_name} ═══\n");
        println!(
            "  {:<40} {:>8} {:>6} {:>6} {:>8}",
            "Shader", "Binary", "GPR", "Instr", "Status"
        );
        println!("  {}", "-".repeat(74));

        let mut ok = 0usize;
        let mut fail = 0usize;
        let mut total_bytes = 0usize;

        for entry in &shaders {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                dev.compile_wgsl(entry.source, target)
            }));
            match result {
                Ok(Ok(kernel)) => {
                    ok += 1;
                    total_bytes += kernel.binary.len();
                    println!(
                        "  {:<40} {:>6}B {:>6} {:>6} {:>8}",
                        entry.name,
                        kernel.binary.len(),
                        kernel.gpr_count,
                        kernel.instr_count,
                        "OK"
                    );
                }
                Ok(Err(e)) => {
                    fail += 1;
                    let msg = format!("{e}");
                    let short = if msg.len() > 50 { &msg[..50] } else { &msg };
                    println!(
                        "  {:<40} {:>8} {:>6} {:>6} {:>8}",
                        entry.name, "-", "-", "-", "FAIL"
                    );
                    eprintln!("    {short}");
                }
                Err(_panic) => {
                    fail += 1;
                    println!(
                        "  {:<40} {:>8} {:>6} {:>6} {:>8}",
                        entry.name, "-", "-", "-", "PANIC"
                    );
                    eprintln!("    compiler panicked (coralReef gap)");
                }
            }
        }

        println!("\n  Summary ({arch_name}): {ok}/{} compiled, {fail} failed, {total_bytes} total bytes\n",
            ok + fail);
    }

    println!("  GPU dispatch validation pending coral-driver DRM backend maturity:");
    println!("    - amdgpu: E2E ready");
    println!("    - nouveau: codegen OK, dispatch pending kernel ioctl support");
    println!("    - nvidia-drm: pending UVM integration");
    println!("\n  No Vulkan. No vendor SDK. Pure Rust → native binary compilation.");
}
