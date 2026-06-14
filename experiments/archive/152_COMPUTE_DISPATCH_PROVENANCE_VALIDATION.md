# Experiment 152: Compute Dispatch + Provenance Witness Validation

**Date:** 2026-04-07
**Status:** FRAMEWORK — ready for live validation when ToadStool has GPU dispatch wired
**Track:** Science validation + ecosystem convergence
**Driven by:** primalSpring cross-evolution audit (Apr 7, 2026)

---

## Objective

Validate the full compute dispatch pipeline that esotericWebb's `webb_node.toml`
(Tower + ToadStool) will use for GPU workloads, and confirm that provenance
witnesses flow correctly through the trio.

Three audit items from primalSpring:
1. GPU shader absorption cycle (hotSpring → barraCuda → compute.dispatch)
2. ToadStool compute dispatch validation (submit/result/capabilities)
3. Provenance on compute results (blake3 hash witness via rhizoCrypt)

## Pipeline Under Test

```
hotSpring → compute.dispatch.submit → ToadStool
  ToadStool → barraCuda (shader execution) / coralReef (compilation)
  ToadStool ← result
hotSpring ← compute.dispatch.result
hotSpring → rhizoCrypt: kind:"hash" witness (blake3 of output tensor)
hotSpring → BearDog: sign merkle root (kind:"signature" witness)
hotSpring → loamSpine: permanent commit
hotSpring → sweetGrass: attribution braid
```

## What Changed (Framework)

### New modules in `barracuda/src/`:

- **`witness.rs`** — `WireWitnessRef` type per `ATTESTATION_ENCODING_STANDARD.md` v2.0.0.
  Builders: `hash()`, `checkpoint()`, `beardog_signature()`, `timestamp()`.

- **`compute_dispatch.rs`** — ToadStool dispatch validation:
  `query_capabilities()`, `submit_workload()`, `retrieve_result()`, `validate_dispatch()`.

- **`dag_provenance.rs`** (upgraded) — SHA-256 → blake3. Automatic hash witness emission
  on `DagEvent` outputs. BearDog signature witness on merkle root dehydration.
  `DagProvenance` now carries `Vec<WireWitnessRef>`.

- **`primal_bridge.rs`** (updated) — Added coralReef (glowplug) as discoverable endpoint.
  Added `composition.physics_health` per `COMPOSITION_HEALTH_STANDARD.md`.

### Dependencies

- `blake3 = "1"` added to Cargo.toml for provenance hashing.

## Validation Plan (When Live)

### Phase 1: Capabilities query
```
compute.dispatch.capabilities → expect: ["gpu.f64", "gpu.f32", "shader.wgsl", ...]
```

### Phase 2: Submit + retrieve
```
compute.dispatch.submit { shader: "vector_add_f64", input: [0..64] }
→ job_id
compute.dispatch.result { job_id }
→ output f64 array
→ blake3 hash witness emitted
```

### Phase 3: Provenance roundtrip
```
dag.create_session → session_id
dag.append_event (capabilities phase)
dag.append_event (submit phase)
dag.append_event (result phase + output_hash)
dag.dehydrate → merkle_root
crypto.sign_ed25519 → signature witness
→ DagProvenance with all witnesses
```

### Phase 4: Reproduce
```
Re-run dispatch with same input
Compare blake3 of output → must match original witness evidence
Verify signature witness via crypto.verify_ed25519
```

## Preconditions

- ToadStool must have `compute.dispatch.submit/result/capabilities` wired
  (currently "Method not found" on S168 binary — see Tier 10 compliance matrix)
- barraCuda must have GPU shaders registered with ToadStool
- coralReef glowplug must be running for sovereign shader compilation path
- Provenance trio (rhizoCrypt, loamSpine, sweetGrass) must be live

## Related

- `HOTSPRING_CORALREEF_SACRIFICIAL_EMBER_GPU_SOLVING_HANDOFF_APR06_2026.md`
- `PRIMALSPRING_TRIO_WITNESS_HARVEST_HANDOFF_APR07_2026.md`
- `ATTESTATION_ENCODING_STANDARD.md` v2.0.0
- `COMPOSITION_HEALTH_STANDARD.md`
- `SPRING_PROVENANCE_PATTERN.md`
- Experiment 150: crash vector hunt (sacrificial ember architecture)
- Experiment 151: revalidation and next stages
