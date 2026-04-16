# hotSpring Primal Composition Gaps

**Spring:** hotSpring v0.6.32
**Proto-nucleate:** `hotspring_qcd_proto_nucleate.toml`
**Particle profile:** proton-heavy (Node atomic dominant)
**Date:** April 10, 2026
**Last audited:** April 16, 2026 (Sovereign pipeline complete + composition audit)
**License:** AGPL-3.0-or-later

---

## Purpose

This document tracks gaps discovered during hotSpring's NUCLEUS composition
wiring. Gaps are handed back to primalSpring for ecosystem-wide refinement
via PRs to `primalSpring/docs/PRIMAL_GAPS.md` and `graphs/downstream/`.

---

## Active Gaps

### GAP-HS-001: Squirrel End-to-End Validation

- **Primal:** Squirrel
- **Severity:** Low
- **Status:** Client wired, `validate_squirrel_roundtrip` binary added
- **Description:** Squirrel is in the proto-nucleate (optional node) and
  `squirrel_client.rs` routes `inference.*` capabilities. The
  `validate_squirrel_roundtrip` binary validates the round-trip when
  Squirrel is available. Remaining: confirm parity once neuralSpring
  WGSL shader ML replaces Ollama fallback.
- **Action:** Run `validate_squirrel_roundtrip` against live Squirrel
  once neuralSpring native inference is deployed.

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

### GAP-HS-026: Physics Dispatch Not Wired in Server

- **Primal:** hotSpring (self)
- **Severity:** Medium
- **Status:** Active — registered-but-pending dispatch returns -32001
- **Description:** `hotspring_primal` server advertises physics/compute
  capabilities via `capabilities.list` and `niche::LOCAL_CAPABILITIES`,
  but the `handle_request` dispatch only implements health, composition,
  capabilities, compute.status, and mcp.tools.list. Physics methods
  return a structured `-32001` error indicating dispatch is pending.
  Full physics execution dispatch requires wiring validation binary
  logic into the JSON-RPC server.
- **Action:** Incrementally wire physics dispatch methods into
  `hotspring_primal` as physics pipelines stabilize.

### GAP-HS-027: TensorSession Adoption

- **Primal:** barraCuda
- **Severity:** Low
- **Status:** Deferred (supersedes GAP-HS-007)
- **Description:** barraCuda's `TensorSession` fused multi-op pipeline
  API is not yet adopted in hotSpring. GPU HMC trajectory (leapfrog +
  force + gauge update) is the natural first candidate. Blocked on
  TensorSession API stabilization for lattice workloads.
- **Action:** Wire `TensorSession` into `gpu_hmc/mod.rs` when barraCuda
  API stabilizes.

### GAP-HS-028: LIME/ILDG Zero-Copy I/O

- **Primal:** hotSpring (self)
- **Severity:** Low
- **Status:** Active
- **Description:** `lattice/lime.rs` and `lattice/ildg.rs` allocate
  `Vec<u8>` and copy binary payloads. Zero-copy I/O via `mmap` or
  streaming parsers would reduce memory pressure for large gauge configs.
- **Action:** Evaluate `memmap2` (behind feature gate) or streaming
  record parsers for LIME binary payload path.

---

## Resolved Gaps

| ID | Description | Resolution | Date |
|----|-------------|------------|------|
| GAP-HS-003 | MCP tool surface missing | `barracuda/src/mcp_tools.rs` defines 5 MCP tools; `hotspring_primal` serves `mcp.tools.list` with those tool schemas | Apr 2026 |
| GAP-HS-004 | health.readiness missing | Added to `hotspring_primal.rs` `handle_request` dispatch | Apr 2026 |
| GAP-HS-008 | Composition validation binaries | `validate_nucleus_composition`, `validate_nucleus_tower`, `validate_nucleus_node`, `validate_nucleus_nest` | Apr 2026 |
| GAP-HS-009 | ecoBin / plasmidBin packaging | `scripts/harvest-ecobin.sh` for musl-static builds and plasmidBin submission | Apr 2026 |
| GAP-HS-010 | Inline validation thresholds | Cost estimate literals extracted to `tolerances::cost` module; capability cost constants now named and sourced | Apr 11, 2026 |
| GAP-HS-011 | JSON-RPC error encoding non-compliant | `hotspring_primal.rs` now uses `DispatchResult` enum with proper top-level JSON-RPC 2.0 `error` objects | Apr 11, 2026 |
| GAP-HS-012 | niche.rs missing biomeOS scheduling hints | Added `operation_dependencies()`, `cost_estimates()`, `SEMANTIC_MAPPINGS`, `socket_dirs()`, `resolve_server_socket()` (sibling spring pattern) | Apr 11, 2026 |
| GAP-HS-013 | Named accessors bypass capability routing | `primal_bridge.rs` named accessors now route through `by_domain()` first | Apr 11, 2026 |
| GAP-HS-014 | brain_rhmc.rs over 1000 LOC | Extracted `brain_persistence` module (serializable weights, save/load state) | Apr 11, 2026 |
| GAP-HS-015 | unsafe in bench_silicon_profile.rs | Replaced raw pointer with `std::thread::scope` (safe borrowing) | Apr 11, 2026 |
| GAP-HS-016 | Science composition probes missing | Added `validate_science_probes()` to `composition.rs` — Rust baseline validates NUCLEUS IPC parity | Apr 11, 2026 |
| GAP-HS-017 | Flat CAPABILITIES array | Niche split into `LOCAL_CAPABILITIES` (served) + `ROUTED_CAPABILITIES` (proxied with canonical provider); disjoint-set test | Apr 11, 2026 |
| GAP-HS-018 | No biomeOS registration | `register_with_target()` sends `lifecycle.register` + `capability.register` over JSON-RPC; graceful degradation | Apr 11, 2026 |
| GAP-HS-019 | plasmidBin metadata incomplete | `metadata.toml` upgraded to full schema: `[provenance]`, `[compatibility]`, `[builds.*]`, `[genomeBin]`; `manifest.lock` entry added | Apr 11, 2026 |
| GAP-HS-020 | Validation harness missing Skip / NDJSON | `CompositionResult` with `check_skip()`, `exit_code_skip_aware()` (0/1/2), `ValidationSink` trait, `NdjsonSink`, `StdoutSink`, `NullSink` | Apr 11, 2026 |
| GAP-HS-021 | OrExit trait for binaries | `OrExit<T>` trait on `Result<T,E>` and `Option<T>` — replaces verbose unwrap/expect patterns with `.or_exit(msg)` | Apr 11, 2026 |
| GAP-HS-022 | No capability registry TOML | `config/capability_registry.toml` — authoritative TOML registry with bidirectional sync test vs niche.rs | Apr 11, 2026 |
| GAP-HS-023 | No standalone mode | `HOTSPRING_NO_NUCLEUS=1` skips registration and IPC; physics runs locally without biomeOS | Apr 11, 2026 |
| GAP-HS-024 | Clippy errors in test/bin targets | All `#[cfg(test)]` modules carry `#[allow(clippy::unwrap_used, clippy::expect_used)]`; `cargo clippy --all-targets` clean | Apr 11, 2026 |
| GAP-HS-025 | 13+ rustdoc warnings | Fixed unresolved links, HTML tags, bare URLs; `cargo doc --lib --no-deps` clean | Apr 11, 2026 |
| GAP-HS-007 | TensorSession not adopted | Superseded by GAP-HS-027 (deferred) | Apr 11, 2026 |
| — | Socket naming mismatch | `hotspring_primal` uses `niche::resolve_server_socket()` for family-scoped names | Apr 11, 2026 |
| — | biomeOS registration not wired | `register_with_target()` called on server startup after socket bind | Apr 11, 2026 |
| — | barraCuda pin drift | `Cargo.toml` reconciled to `b95e9c59` matching CHANGELOG v0.6.32 | Apr 11, 2026 |
| — | DAG method name drift | `dag_provenance.rs` aligned to `dag.session.create`, `dag.event.append`, `dag.merkle.root` | Apr 11, 2026 |
| — | receipt_signing crypto method | `crypto.sign` → `crypto.sign_ed25519` matching registry | Apr 11, 2026 |
| — | validation.rs over 1000 LOC | Split into `validation/` module: harness, telemetry, composition, tests | Apr 11, 2026 |
| — | Capability validation stale names | `validate_nucleus_*` binaries use canonical names from proto-nucleate | Apr 11, 2026 |
| — | No deploy graphs | `graphs/hotspring_qcd_deploy.toml` created from proto-nucleate | Apr 11, 2026 |
| — | discover_capabilities() duplicated niche | Delegates to `niche::all_capabilities()` as source of truth | Apr 11, 2026 |

### GAP-HS-029: Fork Isolation Pattern Not in Ecosystem Standard

- **Primal:** coralReef / toadStool
- **Severity:** Low (pattern works, not yet standardized)
- **Status:** Implemented in coral-driver, needs ecosystem documentation
- **Description:** The fork-isolation pattern (`fork_isolated_raw` +
  `MappedBar::isolated_*` safe wrappers) is a reusable primitive for any
  hardware operation that might hang. Currently lives only in coral-driver.
  Should be documented in `SPRING_COMPOSITION_PATTERNS.md` or
  `ECOBIN_ARCHITECTURE_STANDARD.md` as a recommended pattern for hardware
  fault containment.
- **Action:** PR to primalSpring documenting fork-isolation as ecosystem pattern.

### GAP-HS-030: Ember Absorption into toadStool

- **Primal:** toadStool / coralReef
- **Severity:** Medium (architectural)
- **Status:** Deferred until sovereign pipeline fully validated on all GPUs
- **Description:** Per NUCLEUS design, ember (per-GPU MMIO daemon) should be
  absorbed into toadStool after the sovereign GPU solve. The sovereign_init
  pipeline, fork isolation, and MMIO gateway modules are the primary
  absorption targets. The `handlers_mmio.rs` RPC surface becomes part of
  toadStool's hardware orchestration layer.
- **Action:** Track in toadStool's roadmap. Coordinate with coralReef team on
  module boundaries.

### GAP-HS-031: K80 VFIO Legacy Group EBUSY

- **Primal:** coralReef (ember)
- **Severity:** Medium (hardware-specific)
- **Status:** Blocked — kernel issue
- **Description:** Tesla K80 VFIO groups report EBUSY when ember tries to
  open them. Root cause is likely iommufd cdev reference leak on AMD IOMMU.
  The swap logic is validated on Titan V; K80 path is architecturally
  identical but blocked by this kernel issue.
- **Action:** Test after reboot with `iommu=pt` or legacy-only VFIO config.

---

## Handback Protocol

1. Document gap in this file with severity and upstream reference.
2. If the gap requires primal evolution: PR to `primalSpring/docs/PRIMAL_GAPS.md`.
3. If the gap requires graph evolution: PR to `primalSpring/graphs/downstream/`.
4. If the gap surfaced a new pattern: handoff to `ecoPrimals/infra/wateringHole/handoffs/`.
