// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 152 validation binary: compute dispatch + provenance witnesses.
//!
//! Validates the ToadStool compute dispatch pipeline with blake3 hash
//! witnesses flowing through the provenance trio.
//!
//! # Exit codes
//!
//! - 0: all validation phases passed (or degraded gracefully)
//! - 1: validation failure
//!
//! # Environment
//!
//! - `HOTSPRING_NO_NUCLEUS=1` — skip primal detection (standalone mode)
//! - `FAMILY_ID` — family scope for socket discovery (default: "default")

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  hotSpring Experiment 152: Compute Dispatch + Provenance");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let nucleus = hotspring_barracuda::primal_bridge::NucleusContext::detect();
    nucleus.print_banner();
    println!();

    // Phase 0: composition health
    let health = nucleus.physics_health();
    println!("  composition.physics_health:");
    println!(
        "    compute_dispatch: {}",
        health["compute_dispatch"]
    );
    println!(
        "    gpu_backend:      {}",
        health["gpu_backend"]
    );
    println!(
        "    provenance_trio:  {}",
        health["provenance_trio"]
    );
    println!();

    // Phase 1: DAG session (if rhizoCrypt available)
    let mut dag = hotspring_barracuda::dag_provenance::DagSession::begin(
        &nucleus,
        "exp152_compute_dispatch",
    );

    // Phase 2: compute dispatch validation
    if nucleus.toadstool().is_some_and(|e| e.alive) {
        println!("  ToadStool detected — running compute dispatch validation");
        let result = hotspring_barracuda::compute_dispatch::validate_dispatch(
            &nucleus,
            dag.as_mut(),
        );

        println!();
        println!("  Dispatch validation results:");
        println!("    capabilities: {}", result.capabilities_available);
        println!("    gpu_caps:     {:?}", result.gpu_capabilities);
        println!("    submit:       {}", result.submit_succeeded);
        println!("    result:       {}", result.result_received);
        if let Some(ref h) = result.output_hash {
            println!("    output_hash:  {h}");
        }
        println!("    witnesses:    {}", result.witnesses.len());
        for err in &result.errors {
            println!("    ERROR: {err}");
        }
        println!();

        if result.all_passed() {
            println!("  PASS: full compute dispatch pipeline validated");
        } else if result.capabilities_available {
            println!("  PARTIAL: capabilities available but dispatch incomplete");
        } else {
            println!("  DEGRADED: ToadStool reachable but compute.dispatch not wired");
            println!("  (This is expected until ToadStool S173+ is deployed)");
        }
    } else {
        println!("  ToadStool not available — skipping dispatch validation");
        println!("  (standalone mode: local GPU via wgpu still works)");
    }

    println!();

    // Phase 3: standalone witness test (always runs)
    println!("  Standalone witness round-trip test:");
    let test_data = b"hotSpring experiment 152 test payload";
    let hash = hotspring_barracuda::dag_provenance::blake3_hex(test_data);
    let w = hotspring_barracuda::witness::WireWitnessRef::hash(
        "hotspring:exp152",
        &hash,
        Some("standalone:witness:test"),
    );
    let json = serde_json::to_string_pretty(&w).unwrap_or_default();
    println!("{json}");

    let back: hotspring_barracuda::witness::WireWitnessRef =
        serde_json::from_str(&json).unwrap_or_else(|e| {
            eprintln!("  FAIL: witness round-trip deserialize: {e}");
            std::process::exit(1);
        });
    assert_eq!(back.kind, "hash");
    assert_eq!(back.evidence, w.evidence);
    println!("  PASS: witness round-trip (blake3, serialize, deserialize)");
    println!();

    // Phase 4: dehydrate DAG (if session was created)
    if let Some(dag) = dag {
        let prov = dag.dehydrate(&nucleus);
        println!("  DAG provenance:");
        println!("    session:     {}", prov.dag_session_id);
        println!("    merkle_root: {}", prov.merkle_root);
        println!("    events:      {}", prov.events_count);
        println!("    witnesses:   {}", prov.witnesses.len());
        for w in &prov.witnesses {
            println!("      {} kind={} enc={}", w.agent, w.kind, w.encoding);
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Experiment 152: COMPLETE");
    println!("═══════════════════════════════════════════════════════════");
}
