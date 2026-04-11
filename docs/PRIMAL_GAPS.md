# hotSpring Primal Composition Gaps

**Spring:** hotSpring v0.6.32
**Proto-nucleate:** `hotspring_qcd_proto_nucleate.toml`
**Particle profile:** proton-heavy (Node atomic dominant)
**Date:** April 10, 2026
**License:** AGPL-3.0-or-later

---

## Purpose

This document tracks gaps discovered during hotSpring's NUCLEUS composition
wiring. Gaps are handed back to primalSpring for ecosystem-wide refinement
via PRs to `primalSpring/docs/PRIMAL_GAPS.md` and `graphs/downstream/`.

---

## Active Gaps

### GAP-HS-001: Squirrel Not Wired

- **Primal:** Squirrel
- **Severity:** Medium
- **Status:** Not started
- **Description:** The proto-nucleate does not include Squirrel in the
  composition. Adding Squirrel would give hotSpring access to
  `inference.complete`, `inference.embed`, and `inference.models`
  capabilities from neuralSpring's WGSL shader ML evolution.
- **Blocker:** No immediate physics need; hotSpring's ESN/NPU inference
  is currently in-repo via barraCuda. Adding Squirrel is a composition
  evolution, not a code change.
- **Action:** Propose Squirrel addition to proto-nucleate when AI-guided
  parameter tuning (e.g. HMC step size, thermostat coupling) is ready.

### GAP-HS-002: by_capability Discovery Not Pure

- **Primal:** biomeOS / primal_bridge
- **Severity:** Low
- **Status:** Partial
- **Description:** `primal_bridge.rs` uses named accessors (`toadstool()`,
  `beardog()`, `coralreef()`) alongside `capability.list` probing. The
  convenience API works but is not fully capability-addressed. The
  `fleet_client.rs` `route_by_capability()` method implements domain-based
  routing for the ember fleet but not for NUCLEUS socket routing.
- **Action:** Evolve `primal_bridge` to support `get_by_capability(domain)`
  that returns any primal advertising the requested capability, falling
  back to named lookup for compatibility.

### GAP-HS-003: MCP Tool Surface Missing

- **Primal:** petalTongue
- **Severity:** Medium
- **Status:** Not started
- **Description:** Sibling springs (airSpring, wetSpring) expose MCP tool
  definitions for AI/LLM integration. hotSpring has no MCP tooling.
- **Action:** Define MCP tools for validation-related queries:
  `hotspring.validate_status`, `hotspring.tolerance_check`,
  `hotspring.gpu_capability_report`.

### GAP-HS-004: health.readiness Not Implemented

- **Primal:** hotspring_primal
- **Severity:** Low
- **Status:** Fixed (April 2026) — added to handle_request dispatch.
- **Description:** `hotspring_primal.rs` served `health.liveness` but not
  `health.readiness`. Readiness should indicate whether GPU substrates
  are initialized and validation capabilities are available.

### GAP-HS-005: IONIC-RUNTIME Cross-Family GPU Lease

- **Primal:** BearDog / Songbird
- **Severity:** Medium (ecosystem-wide)
- **Status:** Blocked upstream
- **Description:** The proto-nucleate documents ionic bonding for cross-
  FAMILY_ID GPU lease (CERN-style deployment). BearDog's
  `crypto.sign_contract` and ionic propose/accept/seal protocol are not
  yet implemented. This blocks multi-family metallic fleet pooling.
- **Upstream ref:** `primalSpring/docs/PRIMAL_GAPS.md` IONIC-RUNTIME item.

### GAP-HS-006: BTSP-BARRACUDA-WIRE Session Crypto

- **Primal:** barraCuda / BearDog
- **Severity:** Medium (ecosystem-wide)
- **Status:** Blocked upstream
- **Description:** barraCuda session creation does not yet use full BTSP
  stream encryption (Phase 3). hotSpring's df64/tensor work is in-process
  via Rust crate import, so this gap only affects multi-process barraCuda
  IPC scenarios.
- **Upstream ref:** `primalSpring/docs/PRIMAL_GAPS.md` BTSP-BARRACUDA-WIRE.

### GAP-HS-007: TensorSession Not Adopted

- **Primal:** barraCuda
- **Severity:** Low
- **Status:** Deferred
- **Description:** hotSpring uses `TensorContext` from barraCuda but not
  the `TensorSession` fused multi-op pipeline API. Adopting TensorSession
  would enable fused GPU dispatch for multi-step physics pipelines
  (e.g. HMC trajectory = leapfrog + force + gauge update as single session).
- **Action:** Monitor barraCuda TensorSession stabilization; wire when
  the API is stable for lattice workloads.

### GAP-HS-008: Composition Validation Binaries Created

- **Primal:** hotspring (self)
- **Severity:** Informational (resolved)
- **Status:** Complete
- **Description:** Created `validate_nucleus_composition`, `validate_nucleus_tower`,
  `validate_nucleus_node`, `validate_nucleus_nest` binaries that prove NUCLEUS
  composition (IPC-wired primals) produces results consistent with direct Rust
  execution. This is Phase 2: Rust+Python baselines validate NUCLEUS patterns.
- **Pattern:** Same as Python→Rust: trusted baseline is local Rust, validation
  target is IPC composition. Standalone mode (no primals) skip-passes.
- **Handoff:** Pattern documented in wateringHole handoff for sibling springs.

### GAP-HS-009: ecoBin / plasmidBin Packaging

- **Primal:** hotspring
- **Severity:** Medium
- **Status:** Complete (harvest script created)
- **Description:** Created `scripts/harvest-ecobin.sh` for musl-static builds
  and plasmidBin submission. Generates `metadata.toml` per ecoBin v3.0 standard.
  Cross-compilation for aarch64 supported via `--cross-aarch64` flag.
- **Note:** Proto-nucleate states hotSpring binary is "not in plasmidBin" since
  it's a spring, not a primal. The harvest script enables opt-in submission
  when the ecosystem requires pre-built spring binaries for composition testing.

---

## Resolved Gaps

| ID | Description | Resolution | Date |
|----|-------------|------------|------|
| GAP-HS-004 | health.readiness missing | Added to hotspring_primal.rs | Apr 2026 |
| GAP-HS-008 | Composition validation binaries | 4 validate_nucleus_* binaries | Apr 2026 |
| GAP-HS-009 | ecoBin / plasmidBin packaging | harvest-ecobin.sh created | Apr 2026 |

---

## Handback Protocol

1. Document gap in this file with severity and upstream reference.
2. If the gap requires primal evolution: PR to `primalSpring/docs/PRIMAL_GAPS.md`.
3. If the gap requires graph evolution: PR to `primalSpring/graphs/downstream/`.
4. If the gap surfaced a new pattern: handoff to `ecoPrimals/infra/wateringHole/handoffs/`.
