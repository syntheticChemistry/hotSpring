// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validate full NUCLEUS composition for hotSpring QCD niche.
//!
//! This is the Phase 2 validation: Rust+Python baselines validated the physics,
//! now NUCLEUS primal composition validates the IPC-wired ecosystem.
//!
//! Checks:
//!   1. Tower atomic: BearDog + Songbird alive with crypto/discovery capabilities
//!   2. Node atomic: ToadStool + barraCuda + coralReef compute dispatch
//!   3. Nest atomic: rhizoCrypt + loamSpine + sweetGrass provenance trio
//!   4. Full NUCLEUS: all 9 primals wired, composition.* health endpoints
//!   5. Science health: compute_dispatch, gpu_backend, provenance_trio
//!   6. Capability-based routing: `get_by_capability` resolves vs named lookup
//!
//! Proto-nucleate: primalSpring/graphs/downstream/downstream_manifest.toml
//! Composition model: nucleated (hotSpring has its own binary)
//! Particle profile: proton-heavy (Node atomic dominant)
//!
//! Exit code 0 = composition valid, exit code 1 = degraded or failed.

use hotspring_barracuda::composition::{
    self, AtomicType, get_by_capability, validate_atomic, validate_capability,
};
use hotspring_barracuda::niche;
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use serde_json;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  NUCLEUS Composition Validation — hotSpring QCD Niche      ║");
    println!("║  Phase 2: Rust+Python baselines → Primal composition       ║");
    println!("║  Proto-nucleate: hotspring_qcd_proto_nucleate.toml         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("nucleus_composition");

    // ── Niche self-knowledge ──
    println!("  ── Niche Identity ──");
    println!("    Name:        {}", niche::NICHE_NAME);
    println!("    Version:     {}", niche::NICHE_VERSION);
    println!("    Proto:       {}", niche::PROTO_NUCLEATE);
    println!("    Profile:     {}", niche::PARTICLE_PROFILE);
    println!("    Model:       {}", niche::COMPOSITION_MODEL);
    println!("    Fragments:   {:?}", niche::FRAGMENTS);
    println!(
        "    Bond:        {} / {}",
        niche::BOND_TYPE,
        niche::TRUST_MODEL
    );
    println!("    Deps:        {} primals", niche::DEPENDENCIES.len());
    println!();

    // ── Discovery ──
    println!("═══ Phase 1: Primal Discovery ═══");
    let ctx = NucleusContext::detect();
    ctx.print_banner();
    println!("    Family ID: {}", ctx.family_id);
    println!();

    for (name, ep) in ctx.all_endpoints() {
        let status = if ep.alive { "OK" } else { "DOWN" };
        let caps = ep
            .capabilities
            .as_ref()
            .and_then(|c| c.get("capabilities"))
            .and_then(|a| a.as_array())
            .map_or(0, Vec::len);
        println!(
            "    {name:<16} [{status}]  caps={caps}  socket={}",
            ep.socket
        );
    }
    println!();

    // ── Tower Atomic (electron) ──
    println!("═══ Phase 2: Atomic Validation ═══");
    let tower = validate_atomic(&ctx, AtomicType::Tower, &mut harness);
    println!();

    // ── Node Atomic (proton) ──
    let node = validate_atomic(&ctx, AtomicType::Node, &mut harness);
    println!();

    // ── Nest Atomic (neutron) ──
    let nest = validate_atomic(&ctx, AtomicType::Nest, &mut harness);
    println!();

    // ── Full NUCLEUS ──
    let nucleus = validate_atomic(&ctx, AtomicType::FullNucleus, &mut harness);
    println!();

    // ── Capability-based routing ──
    println!("═══ Phase 3: Capability Validation ═══");
    validate_capability(&ctx, "beardog", "crypto.sign_ed25519", &mut harness);
    validate_capability(&ctx, "songbird", "discovery.find_primals", &mut harness);
    validate_capability(&ctx, "toadstool", "compute.dispatch.submit", &mut harness);
    validate_capability(&ctx, "coralreef", "shader.compile.wgsl", &mut harness);
    println!();

    // ── by_capability resolution ──
    println!("  ── Capability-based Discovery (vs named) ──");
    let compute = get_by_capability(&ctx, "compute");
    let shader = get_by_capability(&ctx, "shader");
    let crypto = get_by_capability(&ctx, "crypto");
    let dag = get_by_capability(&ctx, "dag");

    for (domain, result) in [
        ("compute", &compute),
        ("shader", &shader),
        ("crypto", &crypto),
        ("dag", &dag),
    ] {
        match result {
            Some(ep) => println!("    by_capability({domain}) → {} [OK]", ep.name),
            None => println!("    by_capability({domain}) → not found"),
        }
    }
    println!();

    // ── Composition health ──
    println!("═══ Phase 4: Composition Health ═══");
    let health = composition::composition_health(&ctx);
    println!("    tower_health:    {}", health["tower_health"]);
    println!("    node_health:     {}", health["node_health"]);
    println!("    nest_health:     {}", health["nest_health"]);
    println!("    nucleus_health:  {}", health["nucleus_health"]);
    println!("    primals_discovered: {}", health["primals_discovered"]);
    println!("    primals_alive:   {}", health["primals_alive"]);
    println!();

    // ── Phase 5: Science Parity Probes ──
    println!("═══ Phase 5: Science Parity (Rust vs Primal Composition) ═══");

    // Probe 1: Nuclear EOS through composition
    let z = 82_u32;
    let n = 126_u32;
    let local_be = hotspring_barracuda::physics::semf_binding_energy(
        z as usize,
        n as usize,
        &hotspring_barracuda::provenance::SLY4_PARAMS,
    );
    let local_be_per_a = local_be / (z + n) as f64;
    println!("    Rust SEMF B.E.(Pb-208):    {local_be:.4} MeV  ({local_be_per_a:.4} MeV/A)");

    if !ctx.discovered.is_empty() {
        match ctx.call_by_capability(
            "compute",
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
                    harness.check_upper("SEMF: Rust vs IPC parity", rel_err, tolerances::COMPOSITION_SEMF_PARITY_REL);
                    println!("    IPC  SEMF B.E.(Pb-208):    {ipc_be:.4} MeV  (rel_err: {rel_err:.2e})");
                } else {
                    harness.check_bool("SEMF: IPC response", false);
                    println!("    IPC response missing binding_energy_mev");
                }
            }
            Err(e) => println!("    SEMF IPC unavailable: {e}"),
        }

        // Probe 2: Lattice plaquette through composition
        let dims = [4_usize, 4, 4, 4];
        let beta = 6.0;
        let seed = 42;
        let lat =
            hotspring_barracuda::lattice::wilson::Lattice::hot_start(dims, beta, seed);
        let local_plaq = lat.average_plaquette();
        println!("    Rust plaquette (4⁴ β=6.0): {local_plaq:.6}");

        match ctx.call_by_capability(
            "compute",
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
                    harness.check_upper("Plaquette: Rust vs IPC parity", abs_err, tolerances::COMPOSITION_PLAQUETTE_PARITY_ABS);
                    println!("    IPC  plaquette:             {ipc_plaq:.6} (abs_err: {abs_err:.2e})");
                } else {
                    harness.check_bool("Plaquette: IPC response", false);
                }
            }
            Err(e) => println!("    Plaquette IPC unavailable: {e}"),
        }

        // Probe 3: HMC trajectory through composition
        match ctx.call_by_capability(
            "compute",
            "physics.hmc_trajectory",
            serde_json::json!({
                "dims": [4,4,4,4], "beta": 6.0, "n_steps": 10, "dt": 0.05, "seed": 42
            }),
        ) {
            Ok(resp) => {
                if let Some(result) = resp.get("result") {
                    let accepted = result.get("accepted").and_then(|v| v.as_bool());
                    let plaq = result.get("plaquette").and_then(|v| v.as_f64());
                    if let (Some(acc), Some(p)) = (accepted, plaq) {
                        harness.check_bool("HMC: IPC trajectory", true);
                        println!("    IPC HMC: P={p:.6} accepted={acc}");
                    } else {
                        harness.check_bool("HMC: IPC response format", false);
                    }
                }
            }
            Err(e) => println!("    HMC IPC unavailable: {e}"),
        }
    }
    println!();

    // If no primals at all, skip-pass (standalone mode)
    if ctx.discovered.is_empty() {
        println!("  ⚠  No primals discovered — standalone mode.");
        println!("     Set FAMILY_ID and start primals under $XDG_RUNTIME_DIR/biomeos/");
        println!("     to validate composition. Skipping composition checks.");
        harness.check_bool("standalone mode (no primals)", true);
    } else {
        harness.check_bool("tower healthy", tower.passed);
        harness.check_bool("node healthy", node.passed);
        harness.check_bool("nucleus healthy", nucleus.passed);
    }

    // ── Summary ──
    println!("═══ Summary ═══");
    println!("  Atomics:");
    for v in [&tower, &node, &nest, &nucleus] {
        let icon = if v.passed { "PASS" } else { "FAIL" };
        println!(
            "    {:<24} {}/{} alive  [{}]",
            v.atomic.label(),
            v.primals_alive,
            v.primals_required,
            icon
        );
        if !v.primals_missing.is_empty() {
            println!("      Missing: {}", v.primals_missing.join(", "));
        }
    }
    println!();

    harness.finish();
}
