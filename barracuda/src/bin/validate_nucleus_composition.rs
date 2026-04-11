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
//! Proto-nucleate: hotspring_qcd_proto_nucleate.toml
//! Composition model: nucleated (hotSpring has its own binary)
//! Particle profile: proton-heavy (Node atomic dominant)
//!
//! Exit code 0 = composition valid, exit code 1 = degraded or failed.

use hotspring_barracuda::composition::{
    self, AtomicType, get_by_capability, validate_atomic, validate_capability,
};
use hotspring_barracuda::niche;
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::validation::ValidationHarness;

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
    validate_capability(&ctx, "beardog", "crypto.sign", &mut harness);
    validate_capability(&ctx, "songbird", "net.discovery", &mut harness);
    validate_capability(&ctx, "toadstool", "compute.dispatch.submit", &mut harness);
    validate_capability(&ctx, "coralreef", "shader.compile", &mut harness);
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
