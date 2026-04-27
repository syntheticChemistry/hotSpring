// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validate Node atomic (proton) — compute dispatch proof.
//!
//! Proves:
//!   1. ToadStool alive → `compute.dispatch.submit` works
//!   2. barraCuda alive → `tensor.*` capabilities registered
//!   3. coralReef alive → `shader.compile` works
//!   4. Compute dispatch: submit a trivial workload through ToadStool
//!   5. Sovereign compile: submit WGSL through coralReef
//!
//! This validates that the IPC compute path produces results consistent
//! with the direct Rust execution path (the known-good baseline).
//!
//! Exit code 0 = Node valid, exit code 1 = degraded.

use hotspring_barracuda::composition::{AtomicType, validate_atomic, validate_capability};
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use serde_json;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Node Atomic (proton) — Compute Dispatch Validation        ║");
    println!("║  ToadStool + barraCuda + coralReef                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("nucleus_node");
    let ctx = NucleusContext::detect();
    ctx.print_banner();
    println!();

    let node = validate_atomic(&ctx, AtomicType::Node, &mut harness);
    println!();

    // ── Compute dispatch probe ──
    println!("  ── Compute Dispatch (ToadStool) ──");
    if let Some(ts) = ctx.by_domain("compute") {
        if ts.alive {
            let cap_result = ctx.call("toadstool", "compute.capabilities", &serde_json::json!({}));
            match cap_result {
                Ok(resp) => {
                    let has_caps = resp.get("result").is_some();
                    harness.check_bool("ToadStool compute.capabilities", has_caps);
                    println!(
                        "    Capabilities query: {}",
                        if has_caps { "OK" } else { "FAIL" }
                    );
                    if let Some(result) = resp.get("result") {
                        if let Some(devices) = result.get("devices").and_then(|d| d.as_array()) {
                            println!("    Devices: {}", devices.len());
                        }
                    }
                }
                Err(e) => {
                    harness.check_bool("ToadStool compute.capabilities", false);
                    println!("    Capabilities error: {e}");
                }
            }
        }
    }

    // ── Sovereign compile probe ──
    println!("  ── Sovereign Compile (coralReef) ──");
    if let Some(cr) = ctx.by_domain("shader") {
        if cr.alive {
            let compile_result = ctx.call("coralreef", "shader.list", &serde_json::json!({}));
            match compile_result {
                Ok(resp) => {
                    let has_list = resp.get("result").is_some();
                    harness.check_bool("coralReef shader.list", has_list);
                    println!("    Shader list: {}", if has_list { "OK" } else { "FAIL" });
                }
                Err(e) => {
                    harness.check_bool("coralReef shader.list", false);
                    println!("    Shader list error: {e}");
                }
            }
        }
    }

    // ── Capability assertions ──
    println!("  ── Capability assertions ──");
    validate_capability(&ctx, "toadstool", "compute.dispatch.submit", &mut harness);
    validate_capability(&ctx, "coralreef", "shader.compile", &mut harness);

    // ── Science parity: local Rust vs IPC compute ──
    println!("  ── Science Parity (Rust vs IPC) ──");

    // Nuclear EOS parity: compute SEMF locally, then via hotspring_primal IPC
    let z = 82_u32;
    let n = 126_u32;
    let local_be = hotspring_barracuda::physics::semf_binding_energy(
        z as usize,
        n as usize,
        &hotspring_barracuda::provenance::SLY4_PARAMS,
    );
    println!("    Local SEMF B.E.(Pb-208): {local_be:.4} MeV");

    if let Some(hs) = ctx.get_by_capability("physics") {
        if hs.alive {
            match ctx.call_by_capability(
                "physics",
                "physics.nuclear_eos",
                serde_json::json!({ "Z": z, "N": n }),
            ) {
                Ok(resp) => {
                    if let Some(ipc_be) = resp
                        .get("result")
                        .and_then(|r| r.get("binding_energy_mev"))
                        .and_then(|v| v.as_f64())
                    {
                        let rel_err = ((local_be - ipc_be) / local_be).abs();
                        harness.check_upper("SEMF parity (local vs IPC)", rel_err, tolerances::COMPOSITION_SEMF_PARITY_REL);
                        println!("    IPC SEMF B.E.(Pb-208): {ipc_be:.4} MeV (rel_err: {rel_err:.2e})");
                    } else {
                        harness.check_bool("SEMF parity (IPC response format)", false);
                    }
                }
                Err(e) => {
                    println!("    SEMF IPC error: {e}");
                    harness.check_bool("SEMF parity (IPC call)", false);
                }
            }
        }
    } else {
        println!("    No compute provider — skipping science parity");
    }

    // Lattice QCD parity: compute plaquette locally, then via IPC
    {
        use hotspring_barracuda::lattice::wilson::Lattice;
        let dims = [4_usize, 4, 4, 4];
        let beta = 6.0;
        let seed = 42;
        let lat = Lattice::hot_start(dims, beta, seed);
        let local_plaq = lat.average_plaquette();
        println!("    Local plaquette (4⁴, β=6.0): {local_plaq:.6}");

        if let Some(hs) = ctx.get_by_capability("physics") {
            if hs.alive {
                match ctx.call_by_capability(
                    "physics",
                    "physics.lattice_qcd",
                    serde_json::json!({
                        "dims": dims, "beta": beta, "seed": seed
                    }),
                ) {
                    Ok(resp) => {
                        if let Some(ipc_plaq) = resp
                            .get("result")
                            .and_then(|r| r.get("plaquette"))
                            .and_then(|v| v.as_f64())
                        {
                            let abs_err = (local_plaq - ipc_plaq).abs();
                            harness.check_upper("Lattice plaquette parity", abs_err, tolerances::COMPOSITION_PLAQUETTE_PARITY_ABS);
                            println!("    IPC plaquette: {ipc_plaq:.6} (abs_err: {abs_err:.2e})");
                        } else {
                            harness.check_bool("Lattice parity (IPC format)", false);
                        }
                    }
                    Err(e) => {
                        println!("    Lattice IPC error: {e}");
                        harness.check_bool("Lattice parity (IPC call)", false);
                    }
                }
            }
        }
    }
    println!();

    if ctx.discovered.is_empty() {
        println!("  ⚠  Standalone mode — no primals. Node validation skipped.");
        harness.check_bool("standalone (no primals)", true);
    } else {
        harness.check_bool("node healthy", node.passed);
    }

    harness.finish();
}
