# hotSpring Primal Composition Gaps

**Spring:** hotSpring v0.6.32
**Proto-nucleate:** `hotspring_qcd_proto_nucleate.toml`
**Particle profile:** proton-heavy (Node atomic dominant)
**Date:** April 10, 2026
**Last audited:** April 11, 2026
**License:** AGPL-3.0-or-later

---

## Purpose

This document tracks gaps discovered during hotSpring's NUCLEUS composition
wiring. Gaps are handed back to primalSpring for ecosystem-wide refinement
via PRs to `primalSpring/docs/PRIMAL_GAPS.md` and `graphs/downstream/`.

---

## Active Gaps

### GAP-HS-001: Squirrel Composition Wiring

- **Primal:** Squirrel
- **Severity:** Low (was Medium)
- **Status:** Proto-nucleate wired (optional node), code client exists
- **Description:** Squirrel added to `hotspring_qcd_proto_nucleate.toml`
  as an optional node (required=false). `squirrel_client.rs` provides
  `inference.complete`, `inference.embed`, `inference.models` routing.
  Remaining: validate inference round-trip when neuralSpring native
  inference is live (currently falls back to Ollama).
- **Action:** Integration test when neuralSpring WGSL ML is ready.

### GAP-HS-002: by_capability Discovery Evolution

- **Primal:** biomeOS / primal_bridge
- **Severity:** Low
- **Status:** Mostly resolved
- **Description:** `primal_bridge.rs` now has `by_domain(domain)` as the
  preferred entry point. Named accessors (`toadstool()`, `beardog()`, etc.)
  are retained for backward compatibility but internally route through
  `by_domain()` first, falling back to name-based lookup. Full migration
  to pure capability-based addressing requires downstream callers to
  switch from `.toadstool()` to `.by_domain("compute")`.
- **Action:** Migrate remaining call sites in bin/ targets over time.

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

### GAP-HS-010: Inline Validation Thresholds

- **Primal:** hotspring (self)
- **Severity:** Low
- **Status:** Open — migration in progress
- **Description:** Approximately 33 `validate_*` binaries use inline numeric
  thresholds instead of a centralized tolerances module. Centralizing improves
  auditability and keeps parity checks consistent across domains.
- **Action:** Continue migrating thresholds into the shared tolerances
  infrastructure as domains are touched.

---

## Resolved Gaps

| ID | Description | Resolution | Date |
|----|-------------|------------|------|
| GAP-HS-003 | MCP tool surface missing | `barracuda/src/mcp_tools.rs` defines 5 MCP tools; `hotspring_primal` serves `mcp.tools.list` with those tool schemas | Apr 2026 |
| GAP-HS-004 | health.readiness missing | Added to `hotspring_primal.rs` `handle_request` dispatch | Apr 2026 |
| GAP-HS-008 | Composition validation binaries | `validate_nucleus_composition`, `validate_nucleus_tower`, `validate_nucleus_node`, `validate_nucleus_nest` | Apr 2026 |
| GAP-HS-009 | ecoBin / plasmidBin packaging | `scripts/harvest-ecobin.sh` for musl-static builds and plasmidBin submission | Apr 2026 |
| GAP-HS-011 | JSON-RPC error encoding non-compliant | `hotspring_primal.rs` now uses `DispatchResult` enum with proper top-level JSON-RPC 2.0 `error` objects | Apr 11, 2026 |
| GAP-HS-012 | niche.rs missing biomeOS scheduling hints | Added `operation_dependencies()`, `cost_estimates()`, `SEMANTIC_MAPPINGS`, `socket_dirs()`, `resolve_server_socket()` (sibling spring pattern) | Apr 11, 2026 |
| GAP-HS-013 | Named accessors bypass capability routing | `primal_bridge.rs` named accessors now route through `by_domain()` first | Apr 11, 2026 |
| GAP-HS-014 | brain_rhmc.rs over 1000 LOC | Extracted `brain_persistence` module (serializable weights, save/load state) | Apr 11, 2026 |
| GAP-HS-015 | unsafe in bench_silicon_profile.rs | Replaced raw pointer with `std::thread::scope` (safe borrowing) | Apr 11, 2026 |
| GAP-HS-016 | Science composition probes missing | Added `validate_science_probes()` to `composition.rs` — Rust baseline validates NUCLEUS IPC parity | Apr 11, 2026 |

---

## Handback Protocol

1. Document gap in this file with severity and upstream reference.
2. If the gap requires primal evolution: PR to `primalSpring/docs/PRIMAL_GAPS.md`.
3. If the gap requires graph evolution: PR to `primalSpring/graphs/downstream/`.
4. If the gap surfaced a new pattern: handoff to `ecoPrimals/infra/wateringHole/handoffs/`.
