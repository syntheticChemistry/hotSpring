// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validates the full NUCLEUS node atomic stack for QCD workloads.
//!
//! End-to-end path:
//!   NUCLEUS discovery → coralReef `shader.compile.wgsl` → toadStool
//!   `compute.dispatch.submit` / `compute.dispatch.result` → GPU readback →
//!   CPU parity (hotSpring `lattice/wilson`, oracle for `barracuda::ops::lattice::plaquette`)
//!   → BLAKE3 witness.
//!
//! Default workload: 4×4×4×4 cold-start Wilson plaquette (expected ⟨P⟩ = 1.0).
//!
//! ```text
//! cargo run --release --features sovereign-dispatch --bin validate_node_atomic
//! ```

use hotspring_barracuda::composition::{AtomicType, validate_atomic, validate_capability};
use hotspring_barracuda::dag_provenance::blake3_hex;
use hotspring_barracuda::lattice::dispatch_adapter::{
    LatticeDispatchAdapter, LatticeDispatchParams, make_plaq_uniform_params,
};
use hotspring_barracuda::lattice::gpu_hmc::{build_neighbors, flatten_links};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Node Atomic — End-to-End QCD Validation                   ║");
    println!("║  NUCLEUS → coralReef → toadStool → GPU → CPU parity        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("validate_node_atomic");
    let ctx = NucleusContext::detect();
    ctx.print_banner();
    println!();

    // ── Phase 1: NUCLEUS discovery (Node atomic = 5 primals) ──
    println!("═══ Phase 1: NUCLEUS Discovery (Node atomic) ═══");
    let node = validate_atomic(&ctx, AtomicType::Node, &mut harness);
    let required = AtomicType::Node.required_primals();
    println!("  Required primals ({}): {}", required.len(), required.join(", "));
    harness.check_bool("node:five_primals_required", required.len() == 5);
    println!();

    if ctx.discovered.is_empty() {
        eprintln!("  [SKIP] No primals discovered — set NUCLEUS_SOCKET_DIR or start stack.");
        harness.finish();
    }

    // ── Phase 2: Capability routing ──
    println!("═══ Phase 2: Capability Routing ═══");
    let compile_ok =
        validate_capability(&ctx, "shader", "shader.compile.wgsl", &mut harness);
    let submit_ok =
        validate_capability(&ctx, "compute", "compute.dispatch.submit", &mut harness);
    let result_ok =
        validate_capability(&ctx, "compute", "compute.dispatch.result", &mut harness);
    println!();

    if !(compile_ok && submit_ok && result_ok && node.passed) {
        eprintln!("  [SKIP] Node stack incomplete — cannot run GPU QCD dispatch.");
        harness.finish();
    }

    // ── Phase 3: Cold lattice setup (4⁴) ──
    println!("═══ Phase 3: Cold Lattice Setup (4×4×4×4) ═══");
    let dims_usize = [4_usize, 4, 4, 4];
    let dims_u32 = [4_u32, 4, 4, 4];
    let beta = 6.0_f64;
    let volume = dims_usize.iter().product::<usize>();

    let lat = Lattice::cold_start(dims_usize, beta);
    let cpu_plaq = lat.average_plaquette();
    println!(
        "  CPU reference ⟨P⟩ (oracle for barracuda::ops::lattice::plaquette): {cpu_plaq:.15}"
    );
    harness.check_abs(
        "cpu_cold_plaquette",
        cpu_plaq,
        1.0,
        tolerances::LATTICE_COLD_PLAQUETTE_ABS,
    );

    let action_density = barracuda::ops::lattice::action_density(cpu_plaq);
    harness.check_abs(
        "barracuda_action_density_cold",
        action_density,
        0.0,
        tolerances::LATTICE_COLD_PLAQUETTE_ABS,
    );

    let links = flatten_links(&lat);
    let neighbors = build_neighbors(&lat);
    println!(
        "  Links: {} f64, neighbors: {} u32",
        links.len(),
        neighbors.len()
    );
    harness.check_bool("lattice_links_packed", links.len() == volume * 4 * 18);
    harness.check_bool("lattice_neighbors_packed", neighbors.len() == volume * 8);
    println!();

    // ── Phase 4: GPU dispatch via LatticeDispatchAdapter ──
    println!("═══ Phase 4: GPU Dispatch (LatticeDispatchAdapter) ═══");
    let mut adapter = LatticeDispatchAdapter::new(ctx);

    let dispatch = adapter.dispatch(LatticeDispatchParams {
        shader_name: "wilson_plaquette_f64".into(),
        bdf: None,
        lattice_dims: dims_u32,
        links,
        neighbors,
        params: make_plaq_uniform_params(volume as u32),
        output_elements: volume,
    });

    let (gpu_plaq, job_id, elapsed_ms) = match dispatch {
        Ok(result) => {
            let sum: f64 = result.output.iter().sum();
            let avg = sum / (6.0 * volume as f64);
            let abs_err = (avg - cpu_plaq).abs();
            println!(
                "  GPU ⟨P⟩: {avg:.15} (abs_err: {abs_err:.2e}, job_id={}, {}ms)",
                result.job_id, result.elapsed_ms
            );
            harness.check_abs(
                "gpu_plaquette_parity",
                avg,
                cpu_plaq,
                tolerances::LATTICE_COLD_PLAQUETTE_ABS,
            );
            harness.check_bool("gpu_dispatch_complete", true);
            (avg, result.job_id, result.elapsed_ms)
        }
        Err(e) => {
            eprintln!("  GPU dispatch FAILED: {e}");
            harness.check_bool("gpu_dispatch_complete", false);
            harness.finish();
        }
    };
    println!();

    // ── Phase 5: BLAKE3 witness ──
    println!("═══ Phase 5: BLAKE3 Witness ═══");
    let witness_payload = format!(
        "node_atomic|dims={dims_usize:?}|beta={beta}|cpu={cpu_plaq:.17}|gpu={gpu_plaq:.17}|job={job_id}|ms={elapsed_ms}"
    );
    let witness = blake3_hex(witness_payload.as_bytes());
    println!("  witness: {witness}");
    harness.check_bool("witness_generated", witness.len() == 64);
    println!();

    // ── Summary ──
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Node atomic QCD validation complete");
    println!("  CPU ⟨P⟩ = {cpu_plaq:.15}");
    println!("  GPU ⟨P⟩ = {gpu_plaq:.15}");
    println!("  witness   = {witness}");
    println!("═══════════════════════════════════════════════════════════════");

    harness.finish();
}
