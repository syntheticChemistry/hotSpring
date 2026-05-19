# hotSpring Primal Composition Gaps

**Spring:** hotSpring v0.6.32
**Proto-nucleate:** `downstream_manifest.toml` (spring_name = "hotspring")
**Particle profile:** proton-heavy (Node atomic dominant)
**Date:** April 10, 2026 (created), May 18, 2026 (last audited)
**Last audited:** May 19, 2026 (GPC boundary analysis: 210 experiments, sovereignty tier model, CE validation, 183ms warm pipeline, 7 deploy graphs)
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

### GAP-HS-002: by_capability Discovery Evolution — RESOLVED

- **Primal:** biomeOS / primal_bridge
- **Severity:** Low
- **Status:** RESOLVED (April 27, 2026)
- **Description:** All production callers migrated to `by_domain()`. Named
  accessors deprecated since v0.6.33 with `#[deprecated]` annotations.
  `composition.rs` now derives primal requirements from `niche::DEPENDENCIES`
  (single source of truth) via `required_domains()`. Hardcoded name→domain
  map removed.
- **Resolution:** Capability-based discovery is now the default. Legacy
  accessors remain for backward compat but emit deprecation warnings.

### GAP-HS-005: IONIC-RUNTIME Cross-Family GPU Lease

- **Primal:** BearDog / Songbird
- **Severity:** Medium (ecosystem-wide)
- **Status:** Blocked upstream
- **Description:** The proto-nucleate documents ionic bonding for cross-
  FAMILY_ID GPU lease (CERN-style deployment). BearDog's
  `crypto.sign_contract` and ionic propose/accept/seal protocol are not
  yet implemented. This blocks multi-family metallic fleet pooling.
- **Upstream ref:** `primalSpring/docs/PRIMAL_GAPS.md` IONIC-RUNTIME item.

### GAP-HS-006: BTSP-BARRACUDA-WIRE Session Crypto — RESOLVED

- **Primal:** barraCuda / BearDog
- **Severity:** Medium (ecosystem-wide)
- **Status:** RESOLVED (May 6, 2026)
- **Description:** barraCuda session creation now uses full BTSP stream
  encryption (Phase 3, Sprint 43). barraCuda Sprint 51-53 pulled (May 6) —
  includes BufReader fix, client_nonce HKDF, encrypted frame loop E2E tests.
  toadStool S218 also verified Phase 3 transport switch.
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

### GAP-HS-026: Physics Dispatch Not Wired in Server — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Medium
- **Status:** **Resolved** (April 17, 2026)
- **Resolution:** All 13 physics/compute methods in `LOCAL_CAPABILITIES` are
  now wired in `hotspring_primal.rs`. Physics methods (`physics.*`) execute
  real library code with `catch_unwind` safety. Compute methods (`compute.*`)
  delegate to GPU capability detection. The `is_registered_but_pending`
  fallback and `-32001` code have been removed. NUCLEUS composition validators
  now include science parity probes comparing local Rust results against
  IPC-routed results within documented tolerances.

### GAP-HS-027: TensorSession Adoption

- **Primal:** barraCuda
- **Severity:** Low
- **Status:** Active — upstream unblocked (barraCuda Sprint 66 shipped `sub`/`negate`)
- **Description:** barraCuda's `TensorSession` fused multi-op pipeline
  API is not yet adopted in hotSpring. GPU HMC trajectory (leapfrog +
  force + gauge update) is the natural first candidate. Sprint 66
  shipped `TensorSession::sub()` and `TensorSession::negate()`, completing
  the momentum-update primitives (`p_new = p - dt * force`). IPC batch
  path (`tensor.batch.submit`) also handles `sub` and `negate` ops.
  Sprint 64 added GEMM routing with `MatmulPrecision` and tensor-core
  hints. OOM detection (`is_oom()`, `is_retriable()`) also available.
- **Action:** Wire `TensorSession` into `gpu_hmc/mod.rs` for a single
  HMC trajectory as proof-of-concept, using `sub`/`negate` for leapfrog.

### GAP-HS-028: LIME/ILDG Zero-Copy I/O — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low
- **Status:** **Resolved** (May 7, 2026)
- **Description:** `lattice/lime.rs` and `lattice/ildg.rs` previously
  allocated `Vec<u8>` for all record payloads via `read_all()`.
- **Resolution:** Added streaming API to `LimeReader`:
  - `next_header()` — reads header without buffering payload
  - `copy_payload_into(dest, data_length)` — streams payload to `W: Write`
  - `skip_payload()` — discards pending payload (no allocation)
  `read_gauge_config()` in `ildg.rs` now uses the streaming API for
  metadata records and buffers only the binary-data record once (no clone).
  Unknown LIME record types are skipped via `skip_payload()` without
  any intermediate allocation. `read_all()` retained for backward compat
  with a clear doc note directing large-file callers to the streaming API.

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
| GAP-HS-007 | TensorSession not adopted | Superseded by GAP-HS-027 (now active — barraCuda Sprint 56d) | Apr 11, 2026 |
| — | Socket naming mismatch | `hotspring_primal` uses `niche::resolve_server_socket()` for family-scoped names | Apr 11, 2026 |
| — | biomeOS registration not wired | `register_with_target()` called on server startup after socket bind | Apr 11, 2026 |
| — | barraCuda pin drift | `Cargo.toml` reconciled to `b95e9c59` matching CHANGELOG v0.6.32 | Apr 11, 2026 |
| — | DAG method name drift | `dag_provenance.rs` aligned to `dag.session.create`, `dag.event.append`, `dag.merkle.root` | Apr 11, 2026 |
| — | receipt_signing crypto method | `crypto.sign` → `crypto.sign_ed25519` matching registry | Apr 11, 2026 |
| — | validation.rs over 1000 LOC | Split into `validation/` module: harness, telemetry, composition, tests | Apr 11, 2026 |
| — | Capability validation stale names | `validate_nucleus_*` binaries use canonical names from proto-nucleate | Apr 11, 2026 |
| — | No deploy graphs | `graphs/hotspring_qcd_deploy.toml` created from proto-nucleate | Apr 11, 2026 |
| — | discover_capabilities() duplicated niche | Delegates to `niche::all_capabilities()` as source of truth | Apr 11, 2026 |

### GAP-HS-032: guideStone Binary Unification — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low
- **Status:** **Resolved** (April 18, 2026)
- **Resolution:** Created `hotspring_guidestone` binary using
  `primalspring::composition::{CompositionContext, validate_parity, validate_liveness}`
  API. Bare guideStone validates Properties 1-5 without primals; NUCLEUS additive
  layer tests scalar parity (stats.mean), vector parity (tensor.matmul), SEMF
  end-to-end, crypto.hash provenance witness, and compute.dispatch via IPC.
  `validate_primal_proof` retained for backward compatibility and extended IPC
  probes using hotSpring's own `NucleusContext`.

### GAP-HS-033: primalSpring Composition API Adoption — RESOLVED

- **Primal:** hotSpring (self) / primalSpring
- **Severity:** Low
- **Status:** **Resolved** (April 18, 2026)
- **Resolution:** Added `primalspring` as a path dependency.
  `hotspring_guidestone` binary uses `CompositionContext::from_live_discovery_with_fallback()`,
  `validate_parity()`, `validate_parity_vec()`, `validate_liveness()`, and
  `primalspring::tolerances` for all IPC parity checks. hotSpring's own
  `NucleusContext` retained for the server binary and Tier 2 validators.

### GAP-HS-037: Property 3 BLAKE3 CHECKSUMS Manifest — RESOLVED

- **Primal:** hotSpring (self) / primalSpring
- **Severity:** Medium
- **Status:** **Resolved** (April 17, 2026)
- **Resolution:** Generated BLAKE3 CHECKSUMS manifest for 15 validation-critical
  source files (guideStone binary, physics, provenance, tolerances, composition,
  niche, lib, Cargo.toml, validate-primal-proof.sh). Property 3 now fully passes
  (30/30 bare checks). Also fixed `deny.toml` lookup to check both `deny.toml`
  and `barracuda/deny.toml`, and updated script to build from barracuda/ then
  run from repo root so CHECKSUMS resolves correctly.

### GAP-HS-036: primalSpring v0.9.17 Absorption — RESOLVED

- **Primal:** hotSpring (self) / primalSpring
- **Severity:** Low
- **Status:** **Resolved** (April 20, 2026)
- **Resolution:** Absorbed v0.9.17 patterns:
  1. **primalspring dep**: v0.9.16 → v0.9.17 (backward-compatible, compiles clean)
  2. **guideStone standard v1.2.0**: Reference updated in binary module doc
  3. **Deployment env vars**: `validate-primal-proof.sh` auto-sets
     `BEARDOG_FAMILY_SEED`, `SONGBIRD_SECURITY_PROVIDER`, `NESTGATE_JWT_SECRET`
     when `FAMILY_ID` is provided
  4. **genomeBin v5.1**: 46 binaries across 6 target triples acknowledged
  5. **coralReef iter84**: `--port` → `--rpc-bind` noted for any future deploy scripts

### GAP-HS-035: primalSpring v0.9.16 Pattern Absorption — RESOLVED

- **Primal:** hotSpring (self) / primalSpring
- **Severity:** Low
- **Status:** **Resolved** (April 20, 2026)
- **Resolution:** Absorbed three v0.9.16 patterns into `hotspring_guidestone`:
  1. **Property 3 BLAKE3**: Replaced manual CHECKSUMS file-exists check with
     `primalspring::checksums::verify_manifest()` — per-file BLAKE3 hash verification.
  2. **Protocol tolerance**: Added `is_protocol_error()` arms in crypto and compute
     IPC handlers — HTTP-on-UDS classified as SKIP (Songbird/petalTongue pattern).
  3. **Family-aware discovery**: Inherited automatically via `CompositionContext` —
     `{capability}-{FAMILY_ID}.sock` resolved before `{capability}.sock`.

### GAP-HS-038: Phase 46 Composition Template Absorption — RESOLVED

- **Primal:** hotSpring (self) / primalSpring
- **Severity:** Low
- **Status:** **Resolved** (April 27, 2026)
- **Resolution:** Absorbed Phase 46 composition template:
  1. **`nucleus_composition_lib.sh`**: Copied to `tools/`, verified all 41 functions source correctly
  2. **`hotspring_composition.sh`**: Domain-specific composition with 5 hooks, async tick model, DAG memoization, scientific provenance braids, compute dispatch to barraCuda/toadStool
  3. **Bare mode verified**: All capabilities degrade gracefully (no crash without NUCLEUS)
  4. **DAG memoization wrapper**: `memo_check_vertex`/`memo_store_result`/`memo_get_result` — candidate for upstream promotion

### GAP-HS-039: rhizoCrypt DAG Graceful Degradation (PG-45)

- **Primal:** rhizoCrypt
- **Severity:** Medium
- **Status:** Active — mitigated locally
- **Description:** rhizoCrypt may accept UDS connection with no JSON-RPC response,
  causing DAG calls to timeout. Composition script wraps all DAG calls with
  `cap_available dag` guards and local memo cache fallback.
- **Action:** Upstream rhizoCrypt should implement connection-level healthcheck.

### GAP-HS-040: toadStool Short Timeout Sensitivity (PG-46)

- **Primal:** toadStool
- **Severity:** Low
- **Status:** Active — mitigated locally, toadStool S241 audited
- **Description:** toadStool IPC is slow on short timeouts (< 5s). Composition
  script uses >= 10s for real compute dispatch. Background validation via
  `dispatch_background_validation()` uses async polling to avoid blocking.
  Confirmed: 10s+ needed for real dispatch workloads. toadStool S241
  audited discovery timeout and documented device pool acquire semantics.
  hotSpring's `fleet_toadstool.rs` maps `TOADSTOOL_DISPATCH_TIMEOUT = 30s`
  for sovereign dispatch (GAP-HS-040 confirmed this is necessary).
- **Action:** Upstream toadStool should document minimum recommended timeout
  in public API docs.

### GAP-HS-041: barraCuda stats.entropy Missing (PG-47)

- **Primal:** barraCuda
- **Severity:** Low
- **Status:** RESOLVED — `stats.entropy` registered as alias of `stats.shannon`
  since barraCuda Sprint 50. Confirmed available in Sprint 64-67 method surface
  (72 methods). Safe to call directly via IPC.
- **Resolved:** May 13, 2026

### GAP-HS-042: petalTongue plasmidBin Threading (PG-48)

- **Primal:** petalTongue
- **Severity:** Low
- **Status:** Active — documented
- **Description:** petalTongue musl-static build from plasmidBin may have winit
  threading issues. Use local (non-plasmidBin) build for live visualization mode.
  Composition script's `push_scene` degrades to no-op when visualization is offline.
- **Action:** Upstream petalTongue should test musl-static winit compatibility.

### GAP-HS-043: Deep Debt Evolution — Capability-Based Discovery — RESOLVED

- **Severity:** Medium
- **Status:** RESOLVED (April 27, 2026)
- **Description:** `composition.rs` and `primal_bridge.rs` had hardcoded primal
  name→domain mappings that duplicated `niche::DEPENDENCIES`. Named accessors
  required knowledge of specific primal names. Large files (rhmc 989L,
  nuclear_eos_helpers 978L) exceeded 800-line threshold.
- **Resolution:** composition.rs uses `niche::DEPENDENCIES` as single source of
  truth. All production callers migrated to `by_domain()`. Named accessors
  deprecated. rhmc.rs split into mod.rs + remez.rs. nuclear_eos_helpers split
  into mod.rs + objectives.rs. Pre-existing `nuclear_eos_l2_*` compile errors
  fixed (upstream barraCuda `DiscoveredDevice` API). 993 tests pass.

### GAP-HS-029: Fork Isolation Pattern Not in Ecosystem Standard

- **Primal:** coralReef / toadStool
- **Severity:** Low (pattern works, not yet standardized)
- **Status:** Implemented in coral-driver, Phase C absorption target
- **Description:** The fork-isolation pattern (`fork_isolated_raw` +
  `MappedBar::isolated_*` safe wrappers) is a reusable primitive for any
  hardware operation that might hang. Currently lives only in coral-driver.
  toadStool's `PHASE_C_CORAL_DRIVER_SPLIT_PLAN.md` lists `isolation.rs`
  under VFIO absorption targets — this pattern should move with the
  hardware code into toadStool.
- **Action:** Confirm hotSpring's fork-isolation version matches or
  supersedes coral-driver's version for the Phase C handshake. Document
  in primalSpring as ecosystem pattern.

### GAP-HS-030a: Ember Absorption into toadStool — RESOLVED

- **Primal:** toadStool / coralReef
- **Severity:** Medium (architectural)
- **Status:** **RESOLVED** (toadStool Phase A+B, May 2026)
- **Description:** Per NUCLEUS design, ember (per-GPU MMIO daemon) should be
  absorbed into toadStool after the sovereign GPU solve. The sovereign_init
  pipeline, fork isolation, and MMIO gateway modules are the primary
  absorption targets.
- **Resolution:** toadStool Phase A absorbed ember into `VfioResourceHandle`
  (`crates/core/ember/src/vfio_handle.rs`). Phase B absorbed glowplug into
  `SwapOrchestrator<SysfsSwapExecutor>` (`crates/core/glowplug/src/swap.rs`).
  Legacy `coral-ember` / `coral-glowplug` crates are `#[deprecated(since = "0.2.0")]`
  pointing to toadStool (`toadstool-ember`, `toadstool` glowplug / diesel ECU). hotSpring's `fleet_ember.rs` now has a parallel
  `toadstool-dispatch` feature flag for migration to `ToadStoolDispatchClient`.
  Phase C (cylinder/coral-driver absorption) is planned and documented in
  `PHASE_C_CORAL_DRIVER_SPLIT_PLAN.md`.

### GAP-HS-030b: K80 VFIO Legacy Group EBUSY — RESOLVED

- **Primal:** toadStool (ember hardware path; issue catalogued during coralReef-era deployments)
- **Severity:** Medium (hardware-specific)
- **Status:** RESOLVED (May 6, 2026)
- **Description:** Tesla K80 VFIO groups previously reported EBUSY when ember
  tried to open them. Root cause was redundant `drivers_probe` in udev rules
  triggering a reset through the PLX PEX 8747 PCIe switch, leaving the device
  in a broken state. Fix: removed `drivers_probe` writes, added
  `d3cold_allowed=0`, kernel cmdline `vfio-pci.ids=10de:102d` handles binding.
  VFIO groups 35/36 now accessible with 0666 permissions after cold boot.
- **Resolution:** K80 boots cleanly to vfio-pci. `/dev/vfio/35` and
  `/dev/vfio/36` open without EBUSY. Validated with full power drain.

### GAP-HS-073: GV100 FECS Secure Boot Without WPR — RESOLVED

- **Primal:** toadStool (diesel sovereign pipeline; historically coral-driver / `sovereign_init` under coralReef)
- **Severity:** Critical (blocks GV100 sovereign FECS boot)
- **Status:** **RESOLVED** (May 11, 2026) — binary-patched nouveau warm-catch
  bypasses the firmware barrier entirely. FECS RUNNING via ACR/SEC2 natively.
- **Description:** Experiment 173 proved the nvidia-535 closed driver does NOT
  configure WPR (Write-Protected Region) on GV100 Titan V. WPR registers
  (PFB_WPR1/WPR2 at 0x100CE4-CF0) remain zero. GV100 is pre-GSP: the RM
  runs on the CPU and does not need WPR hardware protection.
  
  **Resolution (Exp 190):** Binary-patched nouveau warm-catch (4 teardown
  functions NOP'd at ELF level via `coral-driver::tools::elf_patcher`).
  Nouveau loads, trains HBM2, and initializes FECS via ACR/SEC2 natively.
  The patched module cannot tear down — warm state persists across vfio-pci
  rebind. FECS_MC = 0x0c060006 (running), PGRAPH enabled, 1 GPC active.
  PMU absence is non-fatal (nouveau skips PMU but ACR/SEC2 still loads FECS).
  
  Pipeline now in pure Rust: `toadstool device warm-catch <BDF> --memory-type hbm2`.
  Era-aware settle: HBM2=12s. RAII `ModuleCleanupGuard` ensures stock
  `nouveau.ko` restored even on panic.
  
  Previous approaches (direct PIO, ACR solver, nvidia-470 VM warm-handoff)
  are superseded by the binary-patched nouveau warm-catch.
- **Action:** Dispatch validation on warm-caught Titan V is the next step
  (user has 1-2 cards from most NVIDIA generations).

### GAP-HS-031: Blackwell SM Warp Exception — Invalid Address Space (Exp 175-177) (RESOLVED)

- **Primal:** coralReef (coral-driver / coral-kmod / uvm_compute)
- **Severity:** Critical (was blocking sovereign dispatch on RTX 5060)
- **Status:** RESOLVED (April 19, 2026) — RTX 5060 sovereign VFIO dispatch LIVE. f64 div/sqrt MUFU polyfills, semaphore fence ordering, UVM write access, QMD v5.0 all proven.
- **Description:** Experiment 175-177 evolved the Blackwell dispatch investigation.
  The full WGSL→SM120 SASS compile pipeline works on RTX 5060 (GB206). Channel
  creation, GPFIFO allocation, and doorbell mechanism all work via coral-kmod.
  
  **Resolved sub-issues:**
  - Channel class: fixed to BLACKWELL_CHANNEL_GPFIFO_A (0xC96F, matches CUDA)
  - QMD v5.0: 384-byte layout implemented, 40+ unit tests pass
  - UVM_PAGEABLE_MEM_ACCESS: ABI struct was 4 bytes (kernel expects 8) — field
    misread caused false "failure" reports; actually always succeeded
  - VRAM page size: kmod used 2MB huge page attrs for 4KB buffers → FAULT_PDE;
    fixed to PAGE_SIZE_4KB, eliminating page directory faults
  - FAULT_PDE → FAULT_PTE progression: page directory correct, page table closer
  
  **Root cause chain (current):**
  1. `UVM_REGISTER_GPU_VASPACE` fails with `GPU_IN_FULL` (0x5D) on desktop Blackwell
  2. Without UVM VA space registration, replayable page faults can't be serviced
  3. `GPU_PROMOTE_CTX` returns `INSUFFICIENT_PERMISSIONS` from userspace (kernel-only)
  4. `GR_CTXSW_SETUP_BIND` with `vMemPtr=0` relies on demand-paging via UVM (step 1)
  5. SM hits "Invalid Address Space" (ESR 0x10) on first CBUF/context access
  
  CUDA avoids this because it has working UVM fault handling. coral-kmod has kernel
  privilege to call GPU_PROMOTE_CTX and eagerly allocate GR context buffers.
  
- **Action:** Re-enable GPU_PROMOTE_CTX in coral-kmod for Blackwell. Allocate
  context buffers from kernel context to bypass demand-paging dependency on UVM.
  
  PCI device ID 0x2d05 was NOT in the original Blackwell range
  (0x2900..=0x2999). Fixed to include 0x2B00..=0x2DFF.
- **Action:** Test on hardware. If NOP timeout persists, investigate
  channel scheduling, USERD VRAM allocation size, or error notifier state.

### GAP-HS-074: Generation-Branched FalconBootSolver (Resolved)

- **Primal:** coralReef (coral-driver / acr_boot / solver)
- **Severity:** Enhancement
- **Status:** Resolved
- **Description:** The FalconBootSolver now classifies GPUs into three
  generations via `GpuGeneration` enum: Kepler (no ACR), CpuRm (Volta/Pascal,
  no WPR), GspRm (Turing+, full ACR). The `boot_for_generation()` method
  dispatches to the correct strategy cascade. Kepler skips ACR entirely
  (sovereign_init handles direct PIO boot). Volta logs a CPU-RM advisory
  before falling through to legacy ACR (golden-state replay is the primary
  path via sovereign_init's GR init stage). Turing+ runs the full cascade.

### GAP-HS-075: coralReef f64 Transcendental Lowering for SM32 (Resolved)

- **Primal:** coralReef (coral-reef / codegen / lower_f64)
- **Severity:** Critical (blocked Kepler sovereign compile)
- **Status:** Resolved (April 18, 2026)
- **Description:** The `lower_f64_function` pass had an early return
  (`if !is_amd && sm.sm() < 70 { return; }`) that skipped f64 transcendental
  lowering for all NVIDIA GPUs below SM70, leaving `f64rcp`/`f64exp2` ops in
  the IR. The SM32 encoder panicked on these unrecognized ops. Since MUFU is
  f32-only on ALL NVIDIA generations, the lowering pass must run everywhere.
  
  Fix applied in three parts:
  1. Removed the SM < 70 guard in `lower_f64/mod.rs`
  2. Added SM-aware helpers: `emit_iadd` (IAdd2 for SM32, IAdd3 for SM70+)
     and `emit_shl_imm` (OpShl for SM32, OpShf for SM70+)
  3. Fixed `as_imm_not_i20`/`as_imm_not_f20` in `codegen/ir/src.rs` to
     return None instead of asserting when source modifiers are on immediates
  
  Result: 10/10 HMC pipeline shaders now compile to native SASS on SM35 (Kepler).
  All 1314 coral-reef unit tests pass.
- **Action:** None — resolved. Pattern documented for other SM-specific encoder gaps.

### GAP-HS-034: Sovereign Compile Parity Across GPU Generations (Resolved)

- **Primal:** coralReef (coral-reef + coral-gpu), barraCuda
- **Severity:** Milestone
- **Status:** Resolved (April 18, 2026)
- **Description:** Full HMC pipeline (10 shaders: wilson_plaquette, sum_reduce,
  cg_compute_alpha, su3_gauge_force, metropolis, dirac_staggered,
  staggered_fermion_force, fermion_action_sum, hamiltonian_assembly, cg_kernels)
  compiles from WGSL to native SASS on:
  - SM35 (Kepler/Tesla K80): 10/10
  - SM70 (Volta/Titan V): 10/10
  - SM120 (Blackwell/RTX 5060): 10/10
  
  Validated via `bench_sovereign_parity` and `validate_pure_gauge --features sovereign-dispatch`
  (16/16 checks pass). QMD v5.0 implemented for Blackwell.
  
  GAP-HS-031 RESOLVED — RTX 5060 sovereign VFIO dispatch LIVE (April 19, 2026).
- **Action:** None — both compile parity and dispatch parity resolved.

### GAP-HS-044: Deploy Graph Order Conflict — RESOLVED

- **Primal:** hotSpring (self) / biomeOS
- **Severity:** Medium (biomeOS deploy ordering)
- **Status:** **Resolved** (May 7, 2026)
- **Description:** `graphs/hotspring_qcd_deploy.toml` had `squirrel` and
  `sweetgrass` both assigned `order = 9`, creating an ambiguous deploy
  ordering for biomeOS. `hotspring` was at `order = 10`, which might
  conflict with squirrel concurrency.
- **Resolution:** `squirrel` → `order = 10`, `petaltongue` added at
  `order = 11`, `hotspring` moved to `order = 12`. Strict monotonic
  ordering enforced for all meta-tier nodes.

### GAP-HS-045: petalTongue Not in niche.rs — RESOLVED

- **Primal:** petalTongue
- **Severity:** Low
- **Status:** **Resolved** (May 7, 2026)
- **Description:** `niche.rs` `DEPENDENCIES` and `ROUTED_CAPABILITIES`
  did not list petalTongue, though the deploy graph included it. Any
  biomeOS pathway analysis that reads niche dependencies would miss
  visualization routing.
- **Resolution:** Added `petaltongue` to `DEPENDENCIES` (`required = false`,
  `capability_domain = "visualization"`) and five visualization/interaction
  capabilities to `ROUTED_CAPABILITIES`.

### GAP-HS-046: Clippy Dead Threshold in clippy.toml — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low (code hygiene)
- **Status:** **Resolved** (May 7, 2026)
- **Description:** `clippy.toml` had `too-many-lines-threshold = 500`
  but `Cargo.toml` globally allows the `too_many_lines` lint, making the
  threshold a dead configuration that created false expectation of
  enforcement.
- **Resolution:** Removed the dead threshold; added a clear comment
  explaining the lint is intentionally suppressed (physics modules justify
  long files within the 1000-line ecosystem limit).

### GAP-HS-047: Titan V PMU Firmware Extraction Tool — DEPRIORITIZED

- **Primal:** coralReef (acr_boot) / hotSpring (sovereign pipeline)
- **Severity:** Medium (nvidia-470 warm-handoff bypasses this for compute)
- **Status:** **Deprioritized** — benchScale warm-handoff is the faster path
- **Description:** GV100 PMU firmware is missing from `linux-firmware`.
  nvidia-470 embeds it in `nv-kernel.o_binary`. Without it, SEC2 ACR
  boot loader starts but never completes.
- **Resolution (partial):** `exp168_pmu_firmware_probe.rs` scans for Falcon
  UC firmware blobs. However, the nvidia-470 warm-handoff (via benchScale
  VM isolation) achieves full FECS boot without extracting PMU — it lets
  nvidia-470 do the work inside a VM, then hands the warm GPU to VFIO.
  PMU extraction remains valuable for cold sovereign boot but is no longer
  the critical path for Titan V compute.
- **Reference:** `wateringHole/handoffs/HOTSPRING_CORALREEF_TITANV_WARM_DMATRF_HANDOFF_MAY07_2026.md`

### GAP-HS-048: gpu/mod.rs Pipeline Creation Duplication — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low (code quality)
- **Status:** **Resolved** (May 7, 2026)
- **Description:** `gpu/mod.rs` (797L) had near-duplicate `validate_pipeline`
  + `validate_pipeline_entry` and `build_pipeline` + `build_pipeline_entry`
  functions. The pipeline creation impl block duplicated the `entry_point`
  pattern with no consolidation.
- **Resolution:** Smart refactoring: DF64 wire helpers moved to `buffers.rs`,
  pipeline creation impl (with merged `validate_pipeline_inner` +
  `build_pipeline_inner`) moved to `dispatch.rs`, device constructor helpers
  (`negotiate_features`, `open_from_adapter_inner`, `finalize_device`) moved
  to `adapter.rs`. `mod.rs` reduced from 797 → 333 lines. All submodules
  under 460 lines.

### GAP-HS-049: exp070 Raw Pointer in Enum Variant — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low (safe Rust)
- **Status:** **Resolved** (May 7, 2026)
- **Description:** `exp070_register_dump.rs` had `AccessMode::DirectMmap { base: *const u8 }`
  — a raw pointer in an enum variant that was visible outside the unsafe block
  where it was created. The `read_bar0_mmap` function took a `*const u8`
  argument, spreading unsafe pointer use across the file.
- **Resolution:** Introduced `Bar0View` RAII struct that encapsulates
  `rustix::mm::mmap`, provides safe bounds-checked `read_u32()` via
  volatile reads, and calls `munmap` in `Drop`. The `AccessMode` enum now
  holds `DirectMmap(Bar0View)` — no raw pointers outside the struct.
  Unsafe surface reduced to two `read_volatile` lines inside `Bar0View`.

### GAP-HS-050: Large File Smart Refactoring (Evolution Pass)

- **Primal:** hotSpring (self)
- **Severity:** Low
- **Status:** **Resolved** (May 7, 2026)
- **Summary of files refactored:**
  - `lattice/pseudofermion/mod.rs` 926L → 76L (extracted `config.rs` 117L, `action.rs` 175L, `hasenbusch.rs` 327L, `dynamics.rs` 227L)
  - `production/npu_worker/handlers.rs` 839L → 105L (extracted `handlers_screening.rs` 156L, `handlers_steering.rs` 266L, `handlers_inference.rs` 144L)
  - `gpu/mod.rs` 797L → 333L (split across existing `adapter`, `buffers`, `dispatch` submodules)
  - All 14 submodule files stay ≤ 460 lines; no file added exceeds 800L.

### GAP-HS-051: BAR0 MMIO Unsafe Consolidation — RESOLVED

- **Primal:** hotSpring (self) — sovereign GPU pipeline experiments
- **Severity:** Low (code quality, unsafe surface area)
- **Status:** **Resolved** (May 8, 2026)
- **Description:** Four experiment binaries (`exp070`, `exp169`, `exp170`, `exp171`)
  each contained a full duplicate copy of the same ~60-line BAR0 MMIO helper
  (`Bar0Map`/`Bar0View` structs with `mmap`/`munmap` RAII wrappers and
  `read_volatile`/`write_volatile` register accessors). Any bug or
  improvement had to be applied in four places. The library's
  `#![forbid(unsafe_code)]` prevented housing the unsafe code there.
- **Resolution:** Created `src/low_level/bar0.rs` — a single-source-of-truth
  file containing documented, bounds-checked `Bar0View` (read-only) and
  `Bar0Map` (read-write) RAII types. All four binaries now include it via
  `#[path = "../low_level/bar0.rs"] mod bar0_mmio;` with `#[allow(unsafe_code)]`
  scoped to the module declaration. The unsafe surface is confined to two
  `read_volatile` / `write_volatile` call sites inside one audited file.
  `#![allow(dead_code)]` in the shared file suppresses false warnings from
  binaries that only use one of the two types.

### GAP-HS-052: capability_registry.toml Missing petalTongue Entries — RESOLVED

- **Primal:** hotSpring (self) / petalTongue
- **Severity:** Low (registry drift)
- **Status:** **Resolved** (May 8, 2026)
- **Description:** `config/capability_registry.toml` was missing 5 petalTongue
  capability entries (`visualization.render`, `visualization.render.scene`,
  `visualization.render.stream`, `interaction.subscribe`, `interaction.poll`)
  that were present in `niche.rs`'s `ROUTED_CAPABILITIES`. The registry
  diverged from the code.
- **Resolution:** Added the 5 missing entries to `capability_registry.toml`
  with correct `served = "routed"`, `provider = "petaltongue"`,
  `domain = "visualization"` fields. Registry and niche.rs are now in sync.

### GAP-HS-053: Large File Smart Refactoring — Phase 2 (RESOLVED)

- **Primal:** hotSpring (self)
- **Severity:** Low (code quality, maintainability)
- **Status:** **Resolved** (May 8, 2026)
- **Files refactored (this session):**
  - `lattice/rhmc/mod.rs` 802L → 479L: Extracted `RationalApproximation` to
    `rational_approx.rs` (181L) and `multi_shift_cg_solve` to
    `multi_shift.rs` (162L). `mod.rs` is now a coordinating hub.
  - `nuclear_eos_helpers/mod.rs` 821L → 227L: Extracted analysis logic to
    `analysis.rs` (259L) and all print/report functions to `reporting.rs`
    (386L). `mod.rs` retains core types and fundamental math.
  - `production/dynamical_mixed_pipeline/single_beta.rs` 953L → 825L:
    Extracted NPU post-processing phase to `npu_post.rs` (208L) containing
    `NpuPostArgs` struct and `npu_post_process_beta` function.
- **Combined with Phase 1 (May 7):** `pseudofermion/mod.rs` 926L → 76L,
  `npu_worker/handlers.rs` 839L → 105L, `gpu/mod.rs` 797L → 333L.
- **Note:** `single_beta.rs` (825L) remains above the 800L audit threshold.
  It is a complex, stateful pipeline where further splitting would introduce
  fragile parameter passing. Tracked for future evolution when the pipeline
  is next refactored for TensorSession adoption.

### GAP-HS-054: Hostname Provenance in Benchmark Binaries — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low (provenance quality)
- **Status:** **Resolved** (May 8, 2026)
- **Description:** `nuclear_eos_gpu.rs` and `sarkas_gpu.rs` used
  `std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string())`
  for the `gate_name` field in `HardwareInventory`. Benchmarks run in
  environments without `$HOSTNAME` set (e.g. CI, containers, WSL) would
  report `"unknown"`, breaking provenance traceability.
- **Resolution:** Added `pub fn resolve_gate_name() -> String` to
  `src/bench/hardware.rs` with priority chain: `$HOSTNAME` → `$COMPUTERNAME`
  → `/etc/hostname` → `niche::NICHE_NAME`. Never returns `"unknown"`.
  Added `HardwareInventory::detect_local()` convenience constructor. Both
  benchmark binaries updated to use `detect_local()`. Re-exported as
  `pub use hardware::resolve_gate_name` from `bench/mod.rs`.

### GAP-HS-055: clippy --all-targets Warnings — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low (code hygiene)
- **Status:** **Resolved** (May 8, 2026)
- **Description:** `cargo clippy --all-targets` revealed one unfulfilled lint
  expectation: `#[expect(clippy::float_cmp)]` in `gpu/mod.rs:325` on a test
  that uses `assert!(... < 1e-7)` rather than `==`, so `float_cmp` never
  fired. Separately, `exp168_pmu_firmware_probe.rs` had `manual_range_contains`,
  `case_sensitive_file_extension_comparison`, and `map_or_false` lints.
  Hardware register constants in experiment binaries caused dead-code warnings.
- **Resolution:** Removed unfulfilled `#[expect(clippy::float_cmp)]`.
  Applied all clippy suggestions in `exp168`. Added targeted
  `#[allow(dead_code)]` to GPU register reference constants (intentional
  full maps for hardware documentation). `cargo clippy --all-targets`
  now exits zero with no warnings.

### GAP-HS-056: External C Dependency Assessment — DOCUMENTED

- **Primal:** hotSpring (self)
- **Severity:** Low (ecoBin compliance review)
- **Status:** **Documented** (May 8, 2026)
- **Description:** `cudarc = { version = "0.19.3", optional = true }` is a
  Rust wrapper around the CUDA C runtime (C/FORTRAN). Reviewed for
  ecoBin compliance per "zero C dependencies in application code" standard.
- **Finding:** `cudarc` is **ecoBin-compliant** because:
  1. It is `optional = true` — not in the default build.
  2. Only activated by `--features cuda-validation` (feature `cuda-validation = ["dep:cudarc"]`).
  3. Only two binaries require it (`validate_5060_dual_use`,
     `validate_cross_vendor_dispatch`), both with `required-features = ["cuda-validation"]`.
  4. Both binaries declare `#![expect(unsafe_code, reason = "CUDA kernel launch via cudarc...")]`.
  5. The default `cargo build` / `cargo check` / `cargo clippy` produce
     zero CUDA / C linkage.
- **Action:** None — current containment is correct. Document in `deny.toml`
  under a skip-list comment explaining the optional C dep exception.

### GAP-HS-057: Tier 4 IPC-First Rewiring — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** High (ecosystem-wide Tier 4 target)
- **Status:** **Resolved** (May 10, 2026)
- **Description:** primalSpring post-interstadial guidance requires Tier 4
  IPC-first defaults: `barracuda` optional, feature-gated behind
  `#[cfg(feature = "barracuda-local")]`, `CompositionContext` for cross-primal calls.
- **Resolution:**
  - `primal-proof` feature flag added to `Cargo.toml`.
  - 25+ modules gated behind `#[cfg(feature = "barracuda-local")]`.
  - Local fallbacks for `Complex64`, `bisect`, `hermite`, `factorial`,
    `lu_solve`, `MD_WORKGROUP_SIZE`, GPU adapter enumeration.
  - `primal-proof` build compiles clean (`--no-default-features --features primal-proof`).
  - `low_level` module registered with `#[cfg(feature = "low-level")]` gate.
- **Verified:** `cargo check --lib --no-default-features --features primal-proof` — zero errors.

### GAP-HS-058: Deploy Graph Coverage — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Medium (audit cited "1 TOML" vs 7+ elsewhere)
- **Status:** **Resolved** (May 10, 2026)
- **Description:** Audit identified hotSpring with only 1 deploy graph
  (`hotspring_qcd_deploy.toml`) vs 7+ in other springs. Each major physics
  pipeline should have a dedicated deploy graph.
- **Resolution:** Created 3 new deploy graphs covering all major pipelines:
  `hotspring_md_deploy.toml` (Yukawa OCP), `hotspring_nuclear_eos_deploy.toml`
  (Skyrme HFB), `hotspring_plasma_deploy.toml` (dense plasma). All include
  skunkBat, provenance trio, and full NUCLEUS composition.

### GAP-HS-059: guideStone L6 NUCLEUS Deployment — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Medium (certification level advancement)
- **Status:** **Resolved** (May 10, 2026)
- **Description:** primalSpring guidance to push toward guideStone L6
  (NUCLEUS deployment validation) given test coverage supports it.
- **Resolution:** `certification/deployment.rs` implements Layer 6 validation:
  deploy graph coverage, biomeOS `composition.status` probing,
  `method.register` dynamic registration, skunkBat audit wiring.
  `MAX_LAYER` advanced from 5 to 6.

### GAP-HS-060: biomeOS v3.51 Absorption — RESOLVED

- **Primal:** biomeOS
- **Severity:** Medium (composition API absorption)
- **Status:** **Resolved** (May 10, 2026)
- **Description:** biomeOS v3.51 provides `composition.status` and
  `method.register` APIs for health monitoring and dynamic method registration.
- **Resolution:**
  - `ipc/biome_status.rs`: `CompositionStatus` struct, `query_composition_status`,
    health validation integration.
  - `ipc/method_register.rs`: 24 hotSpring physics/compute methods defined for
    dynamic registration.
  - `ipc/provenance/`: Per-trio modules — `rhizocrypt.rs`, `loamspine.rs`,
    `sweetgrass.rs`.

### GAP-HS-061: IPC Error Typing — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low (code quality — `Result<_, String>` elimination)
- **Status:** **Resolved** (May 10, 2026)
- **Description:** `send_jsonrpc` returned `Result<_, String>`, propagating
  opaque string errors through the IPC layer. primalSpring exemplar targets
  zero `Result<_, String>` in production paths.
- **Resolution:** Added `HotSpringError::Ipc(String)` variant. `send_jsonrpc`
  now returns `Result<_, HotSpringError>` with typed error variants for IO,
  JSON, and IPC failures. Legacy callers bridge via `.map_err(|e| e.to_string())`.

### GAP-HS-062: Deep Typed Error Propagation — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Medium (code quality)
- **Status:** **Resolved** (May 10, 2026)
- **Description:** 60+ `pub fn` across `fleet_ember.rs`, `fleet_client.rs`,
  `compute_dispatch.rs`, and `NucleusContext` returned `Result<_, String>`.
- **Resolution:** Evolved to `Result<_, HotSpringError>` with typed variants.
  Added `impl From<HotSpringError> for String` for binary boundary ergonomics.
  Remaining `Result<_, String>` confined to binary helpers and hardware-access.

### GAP-HS-063: Hostname Hardcoding — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low (portability)
- **Status:** **Resolved** (May 10, 2026)
- **Description:** 3 production files read `/etc/hostname` directly, which is
  Linux-specific and fails on macOS/BSDs.
- **Resolution:** Consolidated into `niche::hostname()` with env var chain
  (`$HOSTNAME` → `$HOST` → `$COMPUTERNAME` → `/etc/hostname`).

### GAP-HS-064: `#![forbid(unsafe_code)]` vs `low_level` — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Medium (build correctness)
- **Status:** **Resolved** (May 10, 2026)
- **Description:** `lib.rs` has `#![forbid(unsafe_code)]` but upstream added
  `pub mod low_level` containing `unsafe` mmap/MMIO blocks.
- **Resolution:** Removed `pub mod low_level` from lib.rs. Binaries access it
  via `#[path]` inclusion (as designed). `forbid` applies library-wide.

### GAP-HS-065: Chuna Papers Monolith — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low (maintainability)
- **Status:** **Resolved** (May 10, 2026)
- **Description:** `bin_helpers/chuna_overnight/papers.rs` at 831 lines mixed
  3 distinct paper domains (lattice QCD, dielectric, kinetic-fluid).
- **Resolution:** Extracted `paper_44.rs` (220L, dielectric) and `paper_45.rs`
  (132L, kinetic-fluid). Residual `papers.rs` covers paper 43 at 490L.

### GAP-HS-066: 19 dead_code Warnings — RESOLVED

- **Primal:** hotSpring (self)
- **Severity:** Low (cleanliness)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** 19 `dead_code` warnings from superseded NPU handler files
  (`handlers_inference.rs`, `handlers_screening.rs`, `handlers_steering.rs`)
  that were replaced by the `handlers/` subdirectory module structure.
- **Resolution:** Removed 3 dead files and their `mod` declarations. Zero warnings.

### GAP-HS-067: skunkBat IPC — Graph-Only — RESOLVED

- **Primal:** skunkBat (JH-5)
- **Severity:** Medium (composition completeness)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** skunkBat was wired in 4 deploy graphs but had no Rust
  IPC client module. Could not query audit events programmatically.
- **Resolution:** Added `src/ipc/skunkbat.rs` — cursor-based `security.audit_log`
  client. 6 tests. When Phase 3 ships, audit events auto-forward to rhizoCrypt DAG.

### GAP-HS-068: Foundation Seeding — Plasma Thread — RESOLVED

- **Primal:** foundation (sporeGarden)
- **Severity:** Medium (provenance completeness)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** Thread 2 (Plasma Physics) had data sources but no validation
  targets. Sarkas Yukawa MD results were validated in hotSpring but not
  contributed to the foundation geological layer.
- **Resolution:** Created `data/targets/thread02_plasma_targets.toml` with 12
  validated targets. Provenance manifest and validation summary committed.
  Thread 2 upgraded from "mapped" to "active" in THREAD_INDEX.toml.

### GAP-HS-076: K80 GK210 Nouveau Chipset ID Missing — RESOLVED

- **Primal:** toadStool (sovereign pipeline / warm-catch) / upstream kernel (nouveau)
- **Severity:** Critical (blocks K80 warm-catch and nouveau-assisted sovereign dispatch)
- **Status:** **PROVEN** — patched nouveau successfully initialized K80 GR (Exp 188)
- **Description:** Upstream nouveau has no `case 0x0f2:` in the device chipset
  switch table (`drivers/gpu/drm/nouveau/nvkm/engine/device/base.c`). K80 BOOT0
  reads `0x0f22d0a1` (chip ID `0xf2`, GK210), which falls through to
  "unknown chipset" → `-ENODEV`. No subdevices initialize — no GR, no PMU, no FIFO.
  This is why the warm-catch pipeline cannot provide live GPCs: nouveau never
  touches the GPU at all. The GK210 is architecturally identical to GK110B
  (`nvf1_chipset`), differing only in VRAM addressing width. The fix is
  `case 0x0f2: device->chip = &nvf1_chipset; break;` — discussed on nouveau
  mailing list (April 2024, Ilia Mirkin) but never merged upstream.
- **Risk:** FB/memory controller differences between GK210 and GK110B may cause
  GDDR5 training failures even with the chipset patch. PMU VBIOS tables may
  also differ.
- **Resolution (May 10, 2026):** Binary-patched `nouveau.ko` (`cmp $0xf1` → `cmp $0xf2`
  at offset `b76f8`) successfully boots K80 as GK110B. Output: `NVIDIA GK110B (0f22d0a1)`,
  `fb: 12288 MiB GDDR5`, `VRAM: 12288 MiB`, DRM initialized (card1, renderD129).
  5 GPCs enrolled (`0x022430=5`), 6 TPCs per GPC (`0x022438=6`), 30 TPCs total.
  Post-VFIO-rebind GPC stations initially power-gated (livepatch load failure
  on 6.17). **Resolved by binary ELF patching** — `coral-driver::tools::elf_patcher`
  NOPs 4 teardown functions at machine-code level, replacing the livepatch approach.
  Patched nouveau trains GDDR5 (12 GiB), initializes 5 GPCs, and warm state persists
  across vfio-pci rebind. Pipeline: `toadstool device warm-catch <BDF> --memory-type gddr5`.
- **Status:** **RESOLVED** (May 11, 2026, Exp 190). K80 GDDR5 trained, 5 GPCs active,
  FECS_MC = 0x00060005 (running). PMC_ENABLE = 0xfc37b1ef (pop=22).
  Upstream one-line patch still needed for unmodified nouveau (`case 0x0f2`).
- **Ref:** Exp 185, Exp 188, Exp 190, Exp 171, Exp 178, Exp 181

### GAP-HS-069: single_beta.rs Production Pipeline > 800L — RESOLVED

- **Primal:** N/A (local code quality)
- **Severity:** Low (maintainability)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** `single_beta.rs` was 826 lines — a single `run_single_beta`
  function spanning 780 lines with embedded measurement loop, NPU interactions,
  parameter tuning, anomaly detection, and trajectory logging.
- **Resolution:** Extracted 273-line measurement loop into `measurement.rs` (423L).
  `single_beta.rs` reduced to 553L. Measurement phase structured as
  `MeasurementResult` with sub-functions for reject prediction, anomaly
  detection, sub-model steering, and Polyakov readback.

### GAP-HS-070: DOWNSTREAM_PATTERNS.md Stale P0 — RESOLVED

- **Primal:** N/A (documentation alignment)
- **Severity:** Low (documentation accuracy)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** `DOWNSTREAM_PATTERNS.md` listed P0 "fix workload binary name
  (`validate_sarkas_md` → `validate_md`)" as pending, but projectNUCLEUS
  already uses `hotspring_unibin validate --scenario sarkas_yukawa_md`.
- **Resolution:** Refreshed entire document — updated integration points,
  patterns absorbed, and action items to reflect current state.

### GAP-HS-071: PAPER_REVIEW_QUEUE Tier 3b Contradiction — RESOLVED

- **Primal:** N/A (documentation accuracy)
- **Severity:** Low (spec hygiene)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** Papers 43-45 (Chuna) listed as "Queued" in Tier 3b table
  but documented as "COMPLETE" in summary section. Stats header stale (1002
  tests vs actual 1,025).
- **Resolution:** Fixed Tier 3b table to show "COMPLETE". Updated header stats
  to 1,025 tests, 155 binaries.

### GAP-HS-072: Foundation Thread 2 Expression Missing — RESOLVED

- **Primal:** foundation (sporeGarden)
- **Severity:** Medium (cross-repo alignment)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** `thread02_plasma.toml` referenced
  `expressions/PLASMA_QCD_SOVEREIGN_GPU.md` but the file did not exist.
  `THREAD_INDEX.toml` had `expression = ""`.
- **Resolution:** Created `PLASMA_QCD_SOVEREIGN_GPU.md` covering validation
  chain, Sarkas MD, lattice QCD, Kokkos parity, and cross-thread connections.
  Aligned `THREAD_INDEX.toml` to reference it. Added 2 foundation workloads.

### GAP-HS-077: Duplicate GAP IDs (030/032/033/057) — RESOLVED

- **Primal:** hotSpring (documentation)
- **Severity:** Low (registry hygiene)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** Four GAP IDs were reused for different entries: GAP-HS-030
  (standalone collided with 030a/030b), GAP-HS-032 (guideStone vs FalconBoot),
  GAP-HS-033 (composition API vs f64 transcendental), GAP-HS-057 (Tier 4 vs
  K80 GK210). Upstream audit flagged GAP-HS-030 specifically.
- **Resolution:** Renumbered newer duplicates to GAP-HS-073 through GAP-HS-076.
  Zero duplicate IDs remain.

### GAP-HS-078: primal-proof Test Coverage Incomplete — RESOLVED

- **Primal:** hotSpring (Tier 4)
- **Severity:** Medium (test configuration)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** `cargo test --lib --no-default-features --features primal-proof`
  failed due to ungated `barracuda::` imports in test modules (`error.rs` tests,
  `md/reservoir/tests.rs`). Production build was clean but test surface was not.
- **Resolution:** Moved barracuda-dependent tests in `error.rs` to
  `#[cfg(feature = "barracuda-local")] mod tests_barracuda`. Gated
  `head_group_layout_matches_toadstool_head_group` behind `barracuda-local`.
  576 tests pass in default (IPC-first) mode; 1,025 pass with barracuda-local enabled.

### GAP-HS-079: skunkBat Missing from 3 Deploy Graphs — RESOLVED

- **Primal:** hotSpring (deploy)
- **Severity:** Low (JH-5 readiness)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** skunkBat node was present in 4/7 local deploy graphs but
  missing from `spectral`, `plasma_md`, and `sovereign_gpu` graphs, and from
  the plasmidBin `hotspring_cell.toml`.
- **Resolution:** Added skunkBat node to all 3 missing graphs (7/7 complete).
  Added to `hotspring_cell.toml` with `security.audit_log` capabilities.

### GAP-HS-080: Upstream Scorecard L5/L6 Discrepancy — DOCUMENTED

- **Primal:** primalSpring (scorecard)
- **Severity:** Low (reporting)
- **Status:** **Documented** — owner is L2 (primalSpring)
- **Description:** `CROSS_SPRING_PARITY_SCORECARD.md` row correctly shows
  hotSpring as **L6 (certified)**, but the "Guidestone Level" summary table
  groups hotSpring under L5. Next scorecard audit should reconcile.
- **Owner layer:** L2 (primalSpring)

### GAP-HS-081: plasmidBin Manifest Test Count Stale — RESOLVED

- **Primal:** plasmidBin (manifest)
- **Severity:** Low (metadata)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** `manifest.toml [springs.hotspring]` showed `tests = 1040`
  (stale). Verified count is 1,025.
- **Resolution:** Updated to `tests = 1025` and added Tier 4 IPC-first note.

### GAP-HS-082: NUCLEUS Workload Scenario ID Mismatch — RESOLVED

- **Primal:** projectNUCLEUS (workload)
- **Severity:** Critical (NUCLEUS dispatch broken)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** `hotspring-md-validation.toml` referenced `--scenario
  sarkas_yukawa_md` but no scenario with that ID was registered. The existing
  MD scenario was `md-yukawa-ocp` (config smoke test only). NUCLEUS workload
  dispatch would find zero matching scenarios.
- **Resolution:** Created `sarkas-yukawa-md` scenario (`s_sarkas_yukawa_md.rs`)
  with foundation-grade validation: Daligault D* fit across 12 reference points,
  RMSE check, plus CPU MD simulation with energy drift validation when
  `barracuda-local` is enabled. Updated NUCLEUS TOML and foundation workload
  to use `sarkas-yukawa-md`. 7 registered scenarios total (6 default + 1
  barracuda-local).

### GAP-HS-083: Foundation Thread 2 Workload Missing — RESOLVED

- **Primal:** foundation (workload pipeline)
- **Severity:** Medium (foundation_validate.sh gap)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** `foundation_validate.sh --thread plasma` found zero workloads
  because `workloads/thread02_plasma/` did not exist. Foundation targets existed
  (12 targets, all validated) but had no execution path via the validate script.
- **Resolution:** Created `workloads/thread02_plasma/hs-sarkas-md.toml` pointing
  to UniBin `validate --scenario sarkas-yukawa-md`. Also fixed foundation
  workload `hs-sarkas-md-validation.toml` scenario ID. Fixed targets file
  `[meta].expression` to reference `PLASMA_QCD_SOVEREIGN_GPU.md`.

### GAP-HS-084: Fleet Discovery legacy `/run/coralreef/` paths — RESOLVED

- **Primal:** toadStool (fleet discovery; superseded coralReef-era `/run/coralreef` hardcoding)
- **Severity:** Low (portability)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** `discover_diesel_ember_socket()` in `fleet_client.rs`
  hardcoded `/run/coralreef` as the ember socket directory. Not portable
  to non-standard installations or containerized environments.
- **Resolution:** Added runtime-dir discovery with cascade:
  `$TOADSTOOL_RUN_DIR` → `$XDG_RUNTIME_DIR/toadstool` → `/run/toadstool`.
  Legacy `$CORALREEF_RUN_DIR` → `$XDG_RUNTIME_DIR/coralreef` → `/run/coralreef` retained as deprecated fallbacks (prefer `TOADSTOOL_*`).
  Consistent with XDG Base Directory Specification.

### GAP-HS-085: DOWNSTREAM_PATTERNS.md Stale — RESOLVED

- **Primal:** hotSpring (documentation)
- **Severity:** Low (docs alignment)
- **Status:** **Resolved** (May 11, 2026)
- **Description:** Several items stale: expression doc listed as "needed"
  (was created), scenario name mismatched code, foundation workload listed
  as "Pending" (was created), no mention of spectral scenario barracuda-local
  requirement.
- **Resolution:** Updated all stale items to current status. Added scenario
  registry listing with tier requirements.

### GAP-HS-086: Deep Debt Consolidation — Magic Numbers + Path Agnosticism — RESOLVED

- **Primal:** hotSpring (code hygiene)
- **Severity:** Low (maintainability)
- **Status:** **Resolved** (May 12, 2026)
- **Description:** Audit found 3 duplicated `CG_BACKOFF_CAP = 2000` constants,
  5 hardcoded timeout literals, hardcoded `/proc/` paths, and repeated
  benchmark report save logic across multiple binaries.
- **Resolution:** `CG_BACKOFF_CAP` consolidated to `tolerances/lattice.rs`.
  Timeout constants extracted: `EMBER_ADOPT_TIMEOUT`, `EMBER_STATUS_TIMEOUT`,
  `GPU_POLL_INTERVAL`, `TITAN_WARM_RECV_TIMEOUT`. `/proc/` paths overridable
  via `PROC_CPUINFO`, `PROC_MEMINFO`, `PROC_SELF_STATUS` env vars.
  `BenchReport::save_and_print()` eliminates duplicated discovery+save pattern.
  579 tests (default) / 1,028 (barracuda-local) — zero clippy warnings.

### GAP-HS-087: Compute Trio Rewire Sprint

- **Primal:** toadStool / barraCuda / coralReef (cross-primal)
- **Severity:** Medium (architectural alignment)
- **Status:** Active — sprint in progress (May 12, 2026)
- **Description:** hotSpring's interfaces with the compute trio
  (toadStool, barraCuda, coralReef) need rewiring after the trio's
  May 2026 evolution: Phase A+B ember/glowplug absorption, barraCuda
  Sprint 56d PrecisionTier/PhysicsDomain formalization, coralReef
  Iter 100 PTX atomics/barriers, and Phase C coral-driver split plan.
- **Completed:**
  - Local `PrecisionTier` (4 variants) and `PhysicsDomain` (12 variants)
    replaced with re-exports from `barracuda::device::precision_tier`
    (15-tier and 15-variant upstream enums). All 10 consumer files compile
    clean with upstream types.
  - `FmaPolicy` and `domain_requires_separate_fma` re-exported from
    `barracuda::device::fma_policy`.
  - `toadstool-dispatch` feature flag added with `ToadStoolDispatchClient`
    in `fleet_toadstool.rs` — parallel IPC path alongside direct ember
    sockets, preparing for Phase C cutover.
  - `HardwareHint` field added to `PrecisionRoute` with domain-based
    default routing (`hardware_hint_for_domain()`). Inference/Training
    route to `TensorCore`; all physics domains route to `Compute`.
  - `validate_compute_trio_pipeline` binary created: Yukawa force +
    Wilson plaquette through full barraCuda→coralReef→toadStool→hardware
    chain with CPU reference parity checks.
  - Barrier shader compilation validation: 9 WGSL shaders using
    `workgroupBarrier()` cataloged with `validate_barrier_shaders()`
    for coralReef's `membar.{cta,gl}` emitter validation.
- **Remaining:**
  - Wire `TensorSession` into `gpu_hmc/mod.rs` (GAP-HS-027).
  - Cut over default dispatch path from ember to toadStool after Phase C.
  - Cross-generation validation: run `validate_compute_trio_pipeline`
    on K80 (SM35), Titan V (SM70), RTX 5060 (SM120), RTX 4070 (Ada),
    MI50 (GFX906) once trio daemons are available on all hardware.

### GAP-HS-088: Deep Debt Rewire — Capability Discovery & Code Health

- **Primal:** hotSpring (self)
- **Severity:** Low (code health / architectural alignment)
- **Status:** **RESOLVED** (May 12, 2026)
- **Description:** Systematic evolution of hardcoded primal knowledge,
  non-idiomatic Rust patterns, and unsafe code to modern capability-based
  discovery and safe abstractions.
- **Completed:**
  - `detect_sovereign_available()` in `precision_brain.rs`: Inverted
    discovery order — NUCLEUS `by_domain("shader")` is now primary, env
    vars (`CORALREEF_SOCKET`, `CORALREEF_MANIFEST` — deprecated; prefer `TOADSTOOL_SOCKET` / manifest equivalents) are CI/lab fallbacks.
    Removed XDG_DATA_DIRS coralReef manifest filesystem scan.
  - IPC provenance clients (`sweetgrass.rs`, `rhizocrypt.rs`,
    `loamspine.rs`): Replaced `niche::socket_dirs()` + hardcoded
    `biomeos/*.sock` path construction with `by_domain("attribution")`
    / `by_domain("dag")` / `by_domain("ledger")` NUCLEUS discovery.
  - `skunkbat.rs`: Replaced hardcoded `skunkbat/skunkbat.sock` path
    construction with `by_domain("security")` NUCLEUS discovery.
  - `certification/deployment.rs`: Replaced `REQUIRED_PRIMALS` hardcoded
    9-entry name list with `required_primals()` derived from
    `niche::DEPENDENCIES` (single source of truth for required primals).
  - `compute_dispatch.rs`: Evolved barrier shader validation from direct
    `send_jsonrpc` on coralReef socket to `call_by_capability("shader",
    "shader.compile.wgsl", ...)` NUCLEUS routing.
  - `toadstool_report.rs`: Added NUCLEUS `by_domain("compute")` as
    primary discovery for `toadstool_socket()`. Evolved
    `report_to_toadstool_with_nucleus()` to prefer `call_by_capability`
    over direct socket IPC.
  - `fleet_client.rs`: Fixed `Vec<&String>` → `Vec<&str>` with
    `sort_unstable()`.
  - `low_level/bar0.rs`: BAR0 map size now discovered from file metadata
    (`bar0_map_size()`) instead of hardcoded 16 MiB. Sysfs PCI base path
    overridable via `HOTSPRING_SYSFS_PCI` env var.
  - `register_maps/mod.rs`: Extracted PCI vendor IDs to named constants
    (`PCI_VENDOR_NVIDIA`, `PCI_VENDOR_AMD`).
- **Validation:** 1,031 library tests pass, zero clippy warnings.

### GAP-HS-089: River Delta Upstream Audit Response (May 12, 2026)

- **Primal:** hotSpring (self) / cross-primal
- **Severity:** Medium (convergence readiness)
- **Status:** **RESOLVED** (May 12, 2026)
- **Description:** Response to primalSpring L2 gate "River Delta Evolution"
  upstream audit. hotSpring is gS L6 (highest delta spring), Tier 4
  IPC-first, LTEE B2 complete. Audit identified three gaps:
  (1) `--format json` for Tier 2 `toadstool.validate` readiness,
  (2) standard `control/ltee_b2_anderson/` layout for lithoSpore handoff,
  (3) upstream PRIMAL_GAPS staleness (B2 listed as "started" vs "complete").
- **Completed:**
  - `--format json` flag added to `hotspring_unibin validate` CLI. New
    `finish_json()` method on `ValidationHarness` produces structured JSON
    to stdout: `{"status":"PASS","checks":N,"passed":N,"values":[...]}`.
    Compatible with `toadstool.validate` Tier 2 ingestion (Pass 14).
  - `control/ltee_b2_anderson/` directory created with `README.md` and
    symlink to `experiments/results/ltee/ltee_b2_anderson_expected.json`
    as `expected_values.json` for standard lithoSpore consumption layout.
  - Upstream audit status corrections documented: LTEE B2 is **COMPLETE**
    (Tier 1 + Tier 2), not "started" as in primalSpring PRIMAL_GAPS snapshot.
    Thread 2 (plasma/QCD) is **ACTIVE** with sources + targets validated
    (GAP-HS-068/072/083 resolved). Titan V FECS + K80 livepatch remain
    blocked on coralReef Pass 12 (upstream sentinel).
- **Upstream action:** primalSpring PRIMAL_GAPS Layer 3 row for hotSpring
  should update LTEE column from "B2 started" to "B2 complete" and note
  `--format json` readiness for Pass 14.
- **Validation:** 1,031 library tests pass, `hotspring_unibin` compiles
  with `--format json` flag.

---

### GAP-HS-090 — Downstream Repo Audit + Scenario Registry Expansion (May 12 2026)

- **Severity:** Medium
- **Classification:** Evolution → registry expansion + cross-repo alignment
- **Description:** Cloned and reviewed projectNUCLEUS, foundation, and lithoSpore
  from sporeGarden. Identified hotSpring-side gaps and downstream staleness.
- **Completed:**
  - **Scenario registry expanded** from 7 → 11 (+ barracuda-local gated):
    `screened-coulomb`, `gradient-flow`, `dielectric-mermin`, `transport-stanton-murillo`
    added to `build_registry()`. Each scenario validates core physics with published
    fits/limits. gradient-flow and dielectric-mermin require `barracuda-local` feature
    (modules are feature-gated).
  - **biomeOS IPC capability evolution:** `ipc/biome_status.rs` and `ipc/method_register.rs`
    evolved from hardcoded `biomeos/biomeos.sock` path to `by_domain("composition")`
    capability discovery, with `BIOMEOS_SOCKET` env var and socket-dir scanning as
    fallbacks. Last two hardcoded socket paths in library IPC code.
  - **validate_all.rs** tier range comment corrected (58–62 → 58–64 for NUCLEUS suites).
  - **Downstream repos cloned** to `ecoPrimals/gardens/`:
    projectNUCLEUS, foundation, lithoSpore.
- **Downstream findings (not hotSpring code gaps — cross-repo process gaps):**
  - lithoSpore `crates/ltee-anderson` module 7 still SKIP-stubbed with
    `validation/expected/module7_anderson.json` path that does not exist.
    hotSpring's `control/ltee_b2_anderson/expected_values.json` is the correct
    artifact; litho needs to either symlink or reference it.
  - foundation `expressions/LTEE_EVOLUTION.md` still marks hotSpring B2 as "STARTED"
    (should be "COMPLETE"). lithoSpore `docs/UPSTREAM_GAPS.md` lists B2 as "QUEUED".
  - foundation `foundation_validate.sh` Phase 5 only scans `workloads/thread*`
    directories — `workloads/hotspring/` (containing Chuna validation) is orphaned
    from automated runs. Either consolidate under `thread02_plasma/` or broaden scan.
  - foundation Phase 6 target comparison expects `metric` field but
    `thread02_plasma_targets.toml` uses `expected_value` — structural mismatch.
- **Upstream actions:**
  - lithoSpore: unblock module 7 with hotSpring B2 artifacts.
  - foundation: update LTEE narrative docs, fix Phase 5/6 tooling alignment.
  - projectNUCLEUS: workload TOMLs for new scenarios (screened-coulomb, transport,
    gradient-flow, dielectric) pending hotSpring scenario stabilization.
- **Benchmarks:** Python baselines exist for all major physics domains in
  `control/*/scripts/` (Sarkas, nuclear EOS, lattice QCD, BGK, kinetic-fluid,
  gradient flow, spectral, TTM, Abelian Higgs, screened Coulomb, transport).
  Kokkos/LAMMPS parity wired for 9 Yukawa MD cases via `benchmarks/kokkos-lammps/`.
  No Galaxy/OpenMM/GROMACS benchmarks (not applicable to our physics domains).
- **Paper queue gaps:** Papers 25–31 (Folding@home, SETI@home, BOINC), 32–42
  (Tier 4 warm-dense-matter / NIF roadmap), B9 (DFE evolution LTEE) remain queued.
- **Dataset gaps:** Dense Plasma Properties Database (off-repo download),
  Zenodo surrogate archive (optional 6GB), full Militzer FPEOS corpus (partial),
  atoMEC (partial at 7/9), Sulfolobus genomes (wetSpring pipeline queued).

---

### GAP-HS-091 — Tier 2 Live Science API Convergence (May 12 2026)

- **Severity:** Medium → Low (wired, awaiting upstream method availability)
- **Classification:** Evolution → Tier 2 IPC convergence
- **Trigger:** Ecosystem Wave Sync May 12 — toadStool S250 shipped `toadstool.validate`,
  barraCuda shipped `precision.route`, Tier 2 declared UNBLOCKED.
- **Completed:**
  - **`ipc/tier2.rs` created:** Tier 2 Live Science API client module with:
    - `workload_preflight()` — calls `toadstool.validate` for workload pre-flight
      (GPU availability, precision tier, dispatch time estimate, warnings)
    - `list_workloads()` — calls `toadstool.list_workloads` for workload catalog
    - `precision_advisory()` — calls `precision.route` for barraCuda precision
      routing advisory (domain + operation → tier/hardware/notes)
    - `tier2_status()` — probes both Tier 2 services, reports readiness
    - `Tier2Status::check()` — records readiness on `ValidationHarness`
  - **`niche.rs` ROUTED_CAPABILITIES updated:** Added `toadstool.validate`,
    `toadstool.list_workloads`, `precision.route` to routed capability table.
  - **`capability_registry.toml` synced:** 3 new entries matching niche.
  - **5 new tests** (584 total lib tests, up from 579).
- **Remaining:**
  - Upstream `toadstool.validate` JSON-RPC handler completion (S250 shipped
    CLI preflight; JSON-RPC handler may still be pending per `LIVE_SCIENCE_API.md`
    which describes it as aspirational contract).
  - `barracuda.precision.route` may be library-level only (not yet a JSON-RPC
    method in barraCuda's `REGISTERED_METHODS`); client will degrade gracefully.
  - ~~Wire `tier2_status()` into `hotspring_unibin status` subcommand.~~ **DONE** (May 12)
  - ~~Wire `workload_preflight()` into scenario runner pre-check.~~ **DONE** (May 12)
- **Validation:** 590/590 lib tests pass. Zero clippy warnings.

---

### GAP-HS-092 — IPC Transport Evolution: call_by_capability Proliferation (May 12 2026)

- **Severity:** Low (all modules already had NUCLEUS discovery; transport now unified)
- **Classification:** Deep Debt → IPC transport evolution
- **Trigger:** Continued deep debt pass — IPC modules were split between
  `by_domain()` discovery and `send_jsonrpc()` transport. The `call_by_capability()`
  API unifies both in a single call, reducing coupling and socket-path leakage.
- **Completed:**
  - **`ipc/biome_status.rs`**: `query_composition_status()` now uses
    `call_by_capability("composition", "composition.status", ...)` as primary
    transport, falling back to env-var and socket-dir scan.
  - **`ipc/method_register.rs`**: `register_rpc()` helper uses
    `call_by_capability("composition", "method.register", ...)`, replacing the
    `biomeos_socket()` + `send_jsonrpc()` two-step pattern.
  - **`ipc/skunkbat.rs`**: `query_audit_log()` uses
    `call_by_capability("security", "security.audit_log", ...)` with direct
    socket fallback.
  - **`ipc/provenance/sweetgrass.rs`**: `submit_braid()` uses
    `call_by_capability("attribution", ...)` with direct socket fallback.
  - **`ipc/provenance/rhizocrypt.rs`**: `submit_witness()` uses
    `call_by_capability("dag", ...)` with direct socket fallback.
  - **`ipc/provenance/loamspine.rs`**: `record_entry()` uses
    `call_by_capability("ledger", ...)` with direct socket fallback.
  - **`fleet_toadstool.rs`**: `capabilities()` and `submit()` now try
    `call_by_capability("compute", ...)` before falling back to cached
    socket transport.
  - **`fleet_client.rs`**: `discover_diesel_ember_socket()` now tries NUCLEUS
    `by_domain("ember")` before falling back to filesystem diesel layout scan.
  - **`hardware_calibration.rs`**: `TierCapability::failed()` and
    `TierCapability::compiled_only()` constructors eliminate ~50 lines of
    repeated boilerplate.
- **Pattern:** Every IPC client module now follows the same 3-tier resolution:
  1. `call_by_capability(domain, method, params)` — unified discovery + transport
  2. Direct `send_jsonrpc` to discovered socket — when NUCLEUS routing unavailable
     but endpoint discovered
  3. Env var / socket-dir fallback — for CI/lab environments
- **Validation:** 590/590 lib tests pass. Zero clippy warnings.

---

### GAP-HS-093 — Sovereign GPU Validation Niche Wiring (May 12 2026)

- **Severity:** Medium (new functionality for upstream audit compliance)
- **Classification:** Feature evolution → sovereign GPU validation
- **Trigger:** primalSpring upstream audit designating hotSpring as "Sovereign GPU
  Validation Niche" — requiring validation of warm/cold boot, Phase D local dispatch,
  and compute trio E2E through Tier 2 Science API.
- **Completed:**
  - **`hotspring_unibin status`**: now includes Tier 2 readiness section
    (`toadstool.validate`, `precision.route`, fully_wired status).
  - **Scenario runner pre-check**: `workload_preflight("hotspring-scenarios")`
    called before running scenarios — reports toadStool preflight status.
  - **`s_sovereign_dispatch` scenario**: new `GpuCompute`-track Live-tier scenario
    exercising: `tier2_status`, `workload_preflight`, `precision_advisory`,
    `compute.dispatch.submit` probe, `ember.fecs.state` probe, and
    `ember.warm_cycle` routable check. Gracefully degrades in standalone mode.
  - **`ROUTED_CAPABILITIES` expanded**: `compute.dispatch.result`, `ember.status`,
    `ember.warm_cycle`, `ember.adopt_device`, `ember.fecs.state` added to niche
    routing table + `capability_registry.toml` (lockstep test passes).
  - **`FecsState` typed struct**: replaces `serde_json::Value` return from
    `fecs_state()` — typed `running`, `pc`, `cpuctl`, `mailbox0`, `sctl`,
    `error`, `timed_out` fields with `is_faulted()` helper. 3 new tests.
  - **`fleet_ember.rs` call_by_capability evolution**: `status()`, `warm_cycle()`,
    `adopt_device()`, and `fecs_state()` now prefer NUCLEUS
    `call_by_capability("compute", ...)` with direct socket fallback.
  - **Phase D `try_local_dispatch()`**: `fleet_toadstool.rs` gains
    `try_local_dispatch()` function + `LocalDispatchResult` struct.
    Sets `local_dispatch: true` and `phase_d: true` in dispatch params.
    `local-dispatch` feature flag added to `Cargo.toml`.
  - **6 new tests** (590 total lib tests, up from 584).
  - **`s_cold_boot_sentinel` scenario**: Second `GpuCompute`-track scenario
    exercising typed `FecsState` response parsing, device health/recovery
    routing, and dispatch result routable checks.
  - **`FusedPipeline` concept** (`compute_dispatch.rs`): Multi-op session
    dispatch via `compute.dispatch.submit_fused` with sequential fallback.
    `FusedOp` with dependency graph, `FusedResult` with per-op outcomes.
    3 new tests.
  - **`fleet_ember.rs`**: `device_health()` and `device_recover()` evolved
    to `call_by_capability` with socket fallback.
  - **lithoSpore B2 handoff**: `expected_values.json` symlink verified intact.
- **Remaining:**
  - Exercise `try_local_dispatch()` on biomeGate hardware (RTX 3090 / Titan V)
    and compare parity with coralReef-forwarded dispatch.
  - Exercise cold boot validation via `falcon_boot()` on K80, capture structured
    `FecsState` errors, and hand back to coralReef.
  - Expand `s_sovereign_dispatch` with real workload dispatch once toadStool
    Phase D `try_local_dispatch()` becomes production default.
  - Sovereign dispatch timeout exercise with cold boot structured errors.
  - TensorSession: upstream barraCuda adoption still deferred (GAP-HS-027);
    `FusedPipeline` is the hotSpring-side wiring. Wire into lattice
    `resident_CG` / Hasenbusch paths once upstream ships.
- **Validation:** 590/590 lib tests pass. Zero clippy warnings.

---

### GAP-HS-094 — Compute Trio Rewire: GlowplugClient + Dispatch Pipeline (May 12 2026)

- **Severity:** Medium (architectural evolution — glowplug→toadStool transition)
- **Classification:** Deep Debt → compute trio IPC transport evolution
- **Trigger:** toadStool Phase C absorbed glowplug/cylinder/ember from coralReef.
  hotSpring's `GlowplugClient` was the last major IPC module using pre-toadStool
  direct socket dispatch without NUCLEUS `call_by_capability` preference.
- **Completed:**
  - **`GlowplugClient` NUCLEUS evolution:** Added `call_with_nucleus_fallback()`
    helper method that attempts `call_by_capability` routing before falling back
    to direct socket RPC. Applied to 7 methods:
    - `list_devices()` → `device.list` via compute domain
    - `dispatch_with_options()` → `device.dispatch` via compute domain
    - `device_swap()` → `device.swap` via compute domain
    - `device_health()` → `device.health` via compute domain
    - `device_resurrect()` → `device.resurrect` via compute domain
    - `sovereign_boot()` → `sovereign.boot` via compute domain
    - All with glowplug socket fallback when NUCLEUS unavailable.
  - **ROUTED_CAPABILITIES expanded:** Added `ember.device.health`,
    `ember.device.recover`, `device.list`, `sovereign.boot` to `niche.rs`
    and `capability_registry.toml`.
  - **`s_compute_trio` scenario:** New `GpuCompute`-track Live-tier scenario
    exercising the full barraCuda→coralReef→toadStool pipeline:
    - Phase 1: precision advisory via `precision.route`
    - Phase 2: shader compilation via `shader.compile.wgsl`
    - Phase 3: toadStool workload preflight
    - Phase 4: toadStool dispatch probe (dry-run)
    - Phase 5: per-GPU hardware readiness (RTX 5060, Titan V, K80)
  - **`s_hotqcd_dispatch` scenario:** New `GpuCompute`-track Live-tier scenario
    exercising the full lattice QCD dispatch pipeline:
    - 6 core QCD shaders compiled through coralReef IPC
    - 3 silicon routing (barrier-heavy) shaders compiled
    - Precision advisory confirms f64/mixed/df64 tier for lattice QCD
    - toadStool dispatch probe with QCD-specific workload metadata
- **Per-GPU dispatch status (biomeGate):**
  | GPU | Warm Boot | Engine State | Dispatch | Blocker |
  |-----|-----------|-------------|----------|---------|
  | RTX 5060 (SM120) | nvidia shared | All engines | PROVEN (12/12 wgpu) | None |
  | Titan V (SM70) | warm-catch FECS running | PMC/FECS/GPC | BLOCKED | wgpu can't enumerate VFIO device |
  | K80 (SM37) | warm-catch GDDR5 trained | PMC/FECS/5 GPCs | BLOCKED | wgpu can't enumerate VFIO device |
- **Remaining:**
  - `GlowplugClient` low-level methods (`register_dump`, `register_snapshot`,
    `read_bar0_range`, `oracle_capture`, `capture_training`,
    `experiment_start/end`, `device_reset`) intentionally remain on direct
    socket — these are device-specific hardware operations that cannot be
    load-balanced or routed through toadStool.
  - Titan V + K80 dispatch validation blocked on wgpu VFIO adapter enumeration.
    Terminal architecture: coralReef SM70/SM37 wgpu backend rebuild removes
    VFIO adapter dependency entirely.
  - `fleet_client.rs` FleetRouter still reads glowplug fleet JSON — not yet
    NUCLEUS compute domain topology. This is expected until Phase C
    consolidates all ember routing through toadStool.
- **Validation:** 590/590 lib tests pass. 13/16 scenarios (default/barracuda-local).

---

### GAP-HS-095 — VFIO Sovereign Dispatch Wiring (May 12 2026)

- **Severity:** Critical (unblocks Titan V + K80 sovereign dispatch)
- **Classification:** Feature evolution → in-process VFIO dispatch
- **Trigger:** GAP-HS-094 identified that wgpu cannot enumerate VFIO-bound GPUs
  (no Vulkan ICD). coral-gpu's `vfio` feature provides an alternative path:
  `GpuContext::from_vfio(bdf)` opens `NvVfioComputeDevice` directly, bypassing
  wgpu and Vulkan entirely.
- **Root cause:** Three dispatch paths exist, none fully connected for VFIO GPUs:
  1. **wgpu** — blind to VFIO GPUs (no Vulkan adapter exposed to host)
  2. **coral-gpu VFIO** — works but `vfio` feature was not enabled in hotSpring's
     `coral-gpu` dependency
  3. **toadStool IPC** — forwards `compute.dispatch.submit` to
     `compute.dispatch.execute`, a method coralReef does not implement
- **Completed:**
  - **`barracuda/Cargo.toml`**: `coral-gpu` dependency now includes
    `features = ["vfio"]`, unlocking `discover_vfio_nvidia_bdf()`,
    `GpuContext::from_vfio(bdf)`, `GpuContext::from_vfio_with_sm(bdf, sm)`,
    and VFIO backend preference in `GpuContext::auto()`.
  - **`validate_vfio_sovereign` binary**: New validation binary exercising
    VFIO dispatch on all known biomeGate GPUs:
    - Phase 1: VFIO GPU discovery via sysfs probing
    - Phase 2: Per-GPU validation (Titan V SM70 at `02:00.0`,
      K80 die0 SM37 at `4b:00.0`, K80 die1 SM37 at `4c:00.0`):
      - `GpuContext::from_vfio_with_sm()` open
      - WGSL → native SASS compile (`write_constant`, `wilson_plaquette_f64`,
        `su3_gauge_force_f64`)
      - Dispatch + readback with sentinel verification
    - CLI: `--bdf <BDF>` and `--sm <SM>` for targeting specific devices
  - **`s_vfio_dispatch` scenario**: New `GpuCompute`-track Live-tier scenario
    in the validation harness:
    - VFIO driver presence check
    - Per-target sysfs presence + vfio-pci binding verification
    - Full dispatch pipeline when `sovereign-dispatch` feature enabled:
      VFIO open → target detect → WGSL compile → alloc → upload →
      dispatch → readback → value verification (42)
    - Graceful degradation: reports `sovereign_dispatch_feature = false`
      without the feature, skips unbound GPUs
  - **Scenario registry**: 17 scenarios total (14 default / 17 with
    `barracuda-local` + `sovereign-dispatch`).
- **Upstream Phase C/D gaps blocking IPC dispatch for other springs:**
  1. **toadStool `compute.dispatch.execute` miswiring:** `compute.dispatch.submit`
     handler in toadStool forwards to `compute.dispatch.execute` on coralReef,
     but coralReef's `REGISTERED_METHODS` does not include that method.
     Phase C should absorb `NvVfioComputeDevice` from `coral-driver` into
     toadStool so dispatch executes locally without IPC forwarding.
  2. **toadStool `VfioResourceHandle` is metadata-only:** `VfioResourceHandle`
     (`crates/core/ember/src/vfio_handle.rs`) stores BDF, IOMMU group, and
     resource token but does not open real VFIO file descriptors or create
     compute contexts. Phase D should integrate the `NvVfioComputeDevice::open()`
     stack (VFIO group fd → device fd → BAR0 mmap → DMA context → GPFIFO)
     into the absorbed toadStool device factory.
  3. **coralReef `enumerate_all()` is DRM-only:** Even with the `vfio` feature
     enabled, `GpuContext::enumerate_all()` uses DRM render nodes
     (`/dev/dri/renderD*`). VFIO-bound GPUs have no DRM node. Needs either
     `enumerate_all_with_vfio()` or merge of VFIO discovery into the unified
     enumeration path so `auto()` sees all available GPUs regardless of backend.
  4. **Dispatch ownership confusion:** hotSpring uses in-process `coral-gpu`
     (Option A, proven on RTX 5060 via DRM, now wired for VFIO on Titan V/K80).
     Other springs using toadStool IPC (Option B) cannot dispatch until
     Phase C/D completes. Both paths must converge post-Phase D so that
     `compute.dispatch.submit` works for all springs uniformly.
- **Evolution (May 12, 2026 — warm API + kernel-module roadmap):**
  - **Cold vs warm init mismatch fixed:** `GpuContext::from_vfio()` called
    `NvVfioComputeDevice::open()` which runs cold init — destroying warm
    state from `toadstool device warm-catch`. Added three warm-aware entry points
    to `coral-gpu/src/context.rs`:
    - `from_vfio_warm(bdf)` — auto-detect SM, warm open with deferred bus-master
    - `from_vfio_warm_with_sm(bdf, sm)` — explicit SM, warm handoff
    - `from_vfio_warm_legacy(bdf, sm)` — legacy VFIO group path for K80
      (no iommufd FLR that kills PLX 8747 bridge)
  - **Validation binary updated:** `validate_vfio_sovereign` defaults to warm
    mode, `--cold` flag for cold init, K80 uses legacy variant automatically.
  - **Scenario updated:** `s_vfio_dispatch` uses `from_vfio_warm_with_sm()` for
    Titan V and `from_vfio_warm_legacy()` for K80.
  - **Hardware validation blocked:** Ember subprocess crashes on startup
    (zombie `[toadstool-ember] <defunct>` processes). `toadstool device warm-fecs` cannot
    relay to ember. GPUs are cold (FECS PRI timeout). Warm API wiring is
    correct but hardware E2E requires upstream ember fix.
  - **Kernel-module evolution roadmap:** `coral-kmod` already proxies nvidia
    RM via `/dev/coral-rm` for Blackwell. Long-term: expand to standalone
    GPU compute driver — graduate `vfio_compute/` init + dispatch from
    userspace MMIO into kernel module. Module binds PCI device directly,
    exposes `/dev/coral-gpu{N}`, GPU stays visible without VFIO isolation.
    Warm boot experiments are R&D for register sequences that move into module.
    Pattern: hotSpring solves locally, hands patterns upstream, primals absorb
    and abstract, hotSpring resolves with their new abstraction.
- **Validation:** 590/590 lib tests pass. `cargo check --features sovereign-dispatch`
  compiles clean with VFIO feature and warm API.
- **Ownership audit (May 12, 2026 — ember/glowplug dual-existence resolved):**
  - **Root cause of ember crash:** Stale `/usr/local/bin/toadstool-ember` binary
    (pre-absorption build) caused zombie `<defunct>` processes. Fixed by
    rebuilding from current toadStool / plasmidBin ecoBin sources.
  - **Cylinder device.swap translation bug FIXED:** In diesel engine mode,
    the ECU forwards `device.swap` to cylinder→ember, but ember only knows
    `ember.swap`. Added translation layer in `cylinder.rs` to convert
    `device.*` methods to `ember.*` before forwarding to ember.
  - **Titan V warm dispatch results (post-fix):**
    - VFIO warm open: PASS (Nvidia Sm70)
    - WGSL compile (write_constant): PASS (192 bytes, 22 GPRs)
    - WGSL compile (wilson_plaquette_f64): PASS (6000 bytes, 38 GPRs)
    - WGSL compile (su3_gauge_force_f64): PASS (20096 bytes, 54 GPRs)
    - Dispatch + readback: FAIL (sentinel 0xDEADBEEF — FECS compute context
      not fully initialized from single nouveau round-trip)
  - **Dual-existence (May 12 snapshot; superseded May 13):** Diesel runtime was split between legacy coralReef-hosted binaries (`coral-ember`, `coral-glowplug`, `coralctl` → now **`toadstool-ember`**, **`toadstool`**, **`toadstool device`**) and early toadStool library crates. **Resolved:** diesel daemon + sockets consolidated under toadStool (`/run/toadstool`); coralReef is shader-compile-only (see Evolution Pass below).

---

### GAP-HS-096 — Ember/Glowplug Ownership: Dual-Existence Audit (May 12 2026)

- **Severity:** Critical (architectural — blocks toadStool as sole hardware lifecycle daemon)
- **Classification:** Ownership evolution → primal boundary resolution
- **Trigger:** Ember subprocess crashes traced to stale diesel-stack binary install (historically coralReef-built paths; use **plasmidBin ecoBins** / toadStool rebuilds).
  Investigation revealed ember/glowplug/cylinder exist in both primals
  with different maturity levels.
- **Ownership map (May 12 audit — superseded May 13; see Evolution Pass below):**
  - **coralReef:** Shader compilation primal (`shader.compile.*`) only; diesel stack excised from coralReef.
  - **toadStool:** Diesel engine (ember / glowplug / cylinder / driver), hardware lifecycle IPC, **`toadstool device`** CLI — deploy via **plasmidBin ecoBins**; runtime dirs **`/run/toadstool`** (legacy **`/run/coralreef`** / **`CORALREEF_*`** env vars deprecated → **`TOADSTOOL_*`**).
  - **Post-audit resolution:** Phase C/D items below were tracked May 12; **toadStool S243/S254** (sections below) record absorption completion (`toadstool-ember.service`, `toadstool-glowplug.service`, warm-catch, VFIO parity).
- **Fixes applied:**
  1. Rebuilt diesel stack binaries from source (toadStool / plasmidBin ecoBins), reinstalled
  2. Fixed cylinder `device.*` → `ember.*` method translation in `cylinder.rs`
  3. Titan V warm VFIO open + compilation proven (3/4 tests PASS)
- **Remaining gaps (May 12 snapshot — largely cleared May 13; see S243/S254 below):**
  - RPC/service parity items below were tracked during dual-existence; treat as **historical** unless reopened upstream.
  - **`toadstool device`** CLI parity (~20 legacy `coralctl` subcommands) — shipped S253+
  - systemd: **`toadstool-glowplug.service`**, **`toadstool-ember.service`**
- **Validation:** 590/590 lib tests pass.

---

### Evolution Pass — May 13, 2026 (updated: post-excision alignment)

All three GAPs (094, 095, 096) have been consolidated into a per-primal
evolution pass handoff:
`infra/wateringHole/handoffs/HOTSPRING_SOVEREIGN_COMPUTE_EVOLUTION_PASS_MAY13_2026.md`

**Post-excision update**: coralReef Sprint 9 excised the entire diesel engine
stack. `coral-gpu` path dependency broken and commented out in `Cargo.toml`.
`sovereign-dispatch` feature is now a stub. 590/590 lib tests pass.

Updated directives:

- **coralReef** (E1–E2 SUPERSEDED, E3 RESOLVED, E4 DONE): Diesel stack fully
  excised. coralReef is a pure compiler primal. No remaining hardware work.
- **toadStool** (E1–E3, CRITICAL PATH): Phase C is now sole critical path for
  sovereign compute. No coralReef fallback exists. Parity gaps and C1–C7
  execution plan remain the roadmap.
- **hotSpring** (self): Rewire sovereign-dispatch to toadStool post-Phase C,
  validate each C-item on hardware, adopt barraCuda v0.4.0 dispatch wire,
  migrate fleet discovery from coralReef to toadStool paths.

The upstream evolution pass document has been updated with post-excision state:
`infra/wateringHole/handoffs/UPSTREAM_PRIMAL_EVOLUTION_PASS_12_14_MAY12_2026.md`

---

### Rewire Pass — May 13, 2026

hotSpring rewired to fully modern architecture post-excision:

- **`glowplug_client.rs`**: `from_nucleus` now resolves `compute` domain
  (toadStool) first, falls back to `shader` (legacy). Error variant renamed
  `NoCoralreefEndpoint` → `NoComputeEndpoint`. Module doc updated.
- **Sovereign bins** (`validate_vfio_sovereign`, `validate_coral_sovereign`,
  `bench_sovereign_parity`): Docs updated to reflect coral-gpu excision and
  toadStool Phase C dependency. All behind `sovereign-dispatch` feature gate.
- **`s_vfio_dispatch.rs`**: Description and scenario doc updated.
- **`hardware_calibration.rs`**: Reference updated from `coral-driver` to
  `toadStool Phase C`.
- **Compile IPC**: `GLOBAL_CORAL` / `compile_wgsl_direct` path confirmed
  correct — still talks to coralReef as pure compiler.
- **Capability registry + niche**: Already modern (`compute.dispatch.*` →
  toadstool, `shader.compile.*` → coralreef, `precision.route` → barracuda).
- **Fleet discovery**: Already toadStool-first (NUCLEUS → compute.sock → legacy).
- **590/590** lib tests pass.

---

### toadStool S243 Audit + Rewire — May 13, 2026

Deep audit of toadStool S237–S243 with hotSpring capability alignment.

**toadStool state (S243, 22,843+ tests, ~83.6% coverage):**

- **Phase A (ember absorption): COMPLETE** — `toadstool-ember`,
  `VfioResourceHandle`, vendor lifecycle, sysfs, device pool, dispatch
  path. `compute.dispatch.capabilities` reports `ember.phase: "B"`.
- **Phase B (glowplug absorption): COMPLETE** — `SwapOrchestrator` with
  real `SysfsSwapExecutor` (PCI unbind/rebind), `GpuPersonality`,
  `GlowPlugClient` using orchestrator, 7-step swap+boot.
- **Phase C (cylinder + coral-driver): PLANNING ONLY** — Split plan
  created (S241), recon complete (S243), legacy `swap_device` removed,
  but no `toadstool-cylinder` crate yet. No coral-driver absorption.
  VFIO PBDMA dispatch WIRED (S258) — full `ComputeDevice` trait impl.
- **Phase D (local dispatch): WIRED** (S254 AMD, S258 NVIDIA PBDMA)

**67 live JSON-RPC methods** confirmed in handler, including:
- `compute.dispatch.*` (submit, status, result, forward, capabilities)
- `compute.dispatch.pipeline.submit/status` — multi-stage ordered dispatch
- `compute.performance_surface.report/query/list` — silicon performance
- `compute.route.multi_unit` — multi-GPU routing
- `compute.hardware.*` (observe, status, vfio_devices, distill, apply,
  share_recipe, auto_init, auto_init_all) — full HW learning surface
- `shader.dispatch` — compiled binary dispatch
- `gpu.query_info/memory/telemetry` — GPU introspection
- `ember.list/status` — sysfs-based device listing (Phase A/B)
- `auth.check/mode/peer_info` — MethodGate (JH-0+JH-2)
- `provenance.query` — local provenance

**Phase C pending (confirmed NOT served as JSON-RPC):**
- `ember.swap` / `device.swap` — internal via SwapOrchestrator only
- `ember.warm_cycle` / `ember.adopt_device` — not present
- `ember.fecs.state` / `ember.device.health` / `ember.device.recover`
- `device.list` — use `ember.list` instead
- `sovereign.boot` — Rust API only (`SwapOrchestrator::execute_boot`)

**VFIO reality:** `VfioResourceHandle` is metadata + optional fd number;
ember crate does **not** own ioctl/open path. `compute.hardware.vfio_devices`
is sysfs/metadata listing, not VFIO fd acquisition.

**hotSpring rewire:**
- Capability registry + niche expanded: +15 new toadStool methods
  (pipeline, performance surface, HW learning expanded, auth, provenance)
- `ipc/tier2.rs`: Added `dispatch_capabilities()` helper for S243 response
- `glowplug_client.rs`: Phase C pending annotations on `device.swap`,
  `device.list`, `sovereign.boot`, `device.dispatch`
- `fleet_client.rs`: Ember phase B documentation
- Registry↔niche lockstep test passes
- **591/591** lib tests pass (up from 590)

**Previously listed blockers (all RESOLVED by toadStool S245-S254):**
1. ~~`toadstool-cylinder` crate~~ — **DONE (S245)**
2. ~~Coral-driver absorption~~ — **DONE (S246-S248)**
3. ~~Real VFIO fd holding~~ — **DONE (S253, OwnedFd)**
4. ~~`ember.swap` / `sovereign.boot`~~ — **DONE: `device.swap` + `device.warm_catch` (S252)**
5. ~~Warm-catch pipeline~~ — **DONE (S252)**
6. ~~CLI parity~~ — **DONE: `toadstool device` CLI (S253)**

All Phase C blockers RESOLVED. Phase D (LocalDeviceFactory) WIRED (S254).

---

### toadStool S254 Phase C+D Rewire — May 13, 2026

toadStool shipped Phase C complete (S245-S253) and Phase D factory (S254).
hotSpring rewired to match:

- **Env vars:** Prefer **`TOADSTOOL_*`** (`TOADSTOOL_RUN_DIR`, `TOADSTOOL_SOCKET`, manifest discovery). Legacy **`CORALREEF_RUN_DIR`**, **`CORALREEF_SOCKET`**, **`CORALREEF_MANIFEST`**, **`CORALREEF_GLOWPLUG_SOCKET`** are deprecated compatibility aliases only.
- **Socket paths**: `/run/toadstool` default (legacy `/run/coralreef` retained only as fallback). Discovery accepts both `toadstool-ember-*` and legacy `ember-*` socket names.
- **Capability registry**: +10 new methods (device.swap, device.warm_catch,
  device.list/status/reacquire, ember.reacquire, mmio.read32/write32/batch,
  mmio.pramin.read32, mmio.bar0.probe, mmio.falcon.status). All
  phase-c-pending entries promoted to routed.
- **`glowplug_client.rs`**: Phase C complete annotations — device management
  and dispatch all now served by toadStool.
- **74 toadStool JSON-RPC methods**, 8,832+ lib tests upstream.
- **Phase D**: AMD sovereign dispatch live (GEM/PM4/fence). NVIDIA
  `NvVfioComputeDevice` FECS-gated (returns Unsupported until firmware bridge).
- **591/591** hotSpring lib tests pass.

**NVIDIA PBDMA dispatch WIRED** (toadStool S258): `NvVfioComputeDevice` now
implements full `ComputeDevice` trait: `alloc()` → VFIO DMA, `upload()` →
host memcpy, `dispatch()` → GPFIFO pushbuf + doorbell, `readback()` → host read,
`sync()` → poll GP_GET. FECS-gated (warm probe S256). AMD DRM path live.
Next: hardware validation on Titan V and K80.

---

### GAP-HS-097 — Deep Debt Resolution + Evolution Sprint (May 13 2026)

- **Severity:** Low (code health / ecosystem compliance)
- **Classification:** Deep Debt → evolution sprint
- **Trigger:** primalSpring "Deep Debt Resolution + Evolution Sprint" directive.
- **Completed:**
  1. **println/eprintln migration:** 10 library-core modules migrated from
     `eprintln!`/`println!` to `log::info!`/`log::warn!`/`log::error!`/`log::debug!`:
     `precision_brain`, `hardware_calibration`, `compute_dispatch`, `composition`,
     `gpu/adapter`, `low_level/bar0`, `certification`, `receipt_signing`,
     `dag_provenance`, `data`, `validation/harness`.
  2. **Hardcoded lab BDFs evolved:** `s_compute_trio`, `s_vfio_dispatch`,
     `s_hotqcd_dispatch` — all hardcoded PCI BDFs replaced with env var
     overrides (`HOTSPRING_RTX5060_BDF`, `HOTSPRING_TITAN_V_BDF`,
     `HOTSPRING_K80_BDF`) with lab defaults. Shader paths resolved via
     `CARGO_MANIFEST_DIR` with `HOTSPRING_QCD_SHADER_DIR` override.
  3. **blake3 pure Rust:** `default-features = false` drops `cc` C build
     dependency. Default build has zero C dependencies.
  4. **Boot scripts migrated:** Created `toadstool-ember.service` and
     `toadstool-glowplug.service`. `k80-wake-and-run.sh` updated to use
     `TOADSTOOL_*` vars and `toadstoolctl`. 9 coral-era scripts archived
     to `scripts/archive/`.
  5. **unwrap() evolution:** Top 10 concerning binary targets evolved:
     `validate_gpu_gradient_flow` (8), `validate_precision_matrix` (4),
     `validate_gradient_flow` (4), `gpu_physics_proxy` (3),
     `validate_sovereign_roundtrip` (3), `meta_table_scan` (2),
     `validate_multi_observable_npu` (2), `validate_silicon_science` (2),
     `compare_flow_integrators` (2). Patterns: `.unwrap()` → `let Some(x) = .. else`,
     `.map_or(NAN, ..)`, `.expect("context")`, `.is_ok_and(..)`.
  6. **CI gate:** `.github/workflows/ci.yml` — `cargo check`, `cargo clippy`,
     `cargo test --lib`, `cargo fmt --check`. No prior CI workflow existed.
  7. **Dependency audit:** `docs/DEPENDENCY_AUDIT.md` — full ecoBin compliance
     assessment. Default build: zero C deps. wgpu/tokio documented as
     ecosystem boundaries. cudarc/rustix feature-gated.
- **Validation:** 591/591 lib tests pass. Zero clippy warnings.
- **Handoff:** `wateringHole/handoffs/HOTSPRING_DEEP_DEBT_SPRINT_MAY13_2026.md`

### GAP-HS-098 — Niche Convergence: Wire Hygiene + Node Atomic Scenario (May 13 2026)

- **Severity:** Low (wire correctness + scenario coverage)
- **Classification:** Niche convergence → atomic deployment prep
- **Trigger:** primalSpring "Delta Spring Directive — Niche Convergence → Atomic Deployment"
  (May 13, 2026). ludoSpring found bearDog wire name discrepancy; all springs
  directed to verify. Node atomic live validation path needed scenario in
  UniBin registry.
- **Completed:**
  1. **bearDog wire name fix:** `receipt_signing.rs` corrected from `payload`/`encoding`
     params to canonical `message` (base64) per bearDog `handle_sign_ed25519` spec.
     `validate_nucleus_tower.rs` also fixed to base64-encode probe message.
  2. **`base64_encode` crate module:** Extracted from `dag_provenance.rs` inline module
     to shared `src/base64_encode.rs` for reuse across receipt signing and tower validation.
  3. **skunkBat wire verified:** Already using `security.audit_log` — correct.
  4. **`s_node_atomic` scenario:** Structural validation of Node atomic (proton) added to
     UniBin registry — 5 domain checks, Tower superset, standalone behavior, SEMF/plaquette
     science baselines (15 checks).
  5. **Harvest script evolved:** `scripts/harvest-ecobin.sh` updated from retired
     `hotspring_primal` binary to `hotspring_unibin`. All UniBin modes registered.
  6. **8 clippy errors resolved:** 5 default + 3 barracuda-local (doc paragraph, if_not_else,
     bool_to_int_with_if, manual_let_else, map_unwrap_or, unwrap_used).
- **Validation:** 595 (default) / 1,041 (barracuda-local) lib tests pass. Zero clippy warnings.

### GAP-HS-099 — Deep Debt Resolution + Evolution Sprint (May 13 2026)

- **Severity:** Low (code health / ecosystem compliance)
- **Classification:** Deep debt → evolution audit + resolution
- **Trigger:** primalSpring "Deep Debt Resolution + Evolution Sprint" directive.
- **Full audit completed across all 7 dimensions:**
  1. **TODO/FIXME/HACK:** Zero markers in library or binary code.
  2. **Modern Rust idioms:** `#![forbid(unsafe_code)]`, `#![deny(clippy::unwrap_used)]`,
     zero clippy, `#[expect]` over `#[allow]`, `let...else`, `is_ok_and()`.
  3. **External deps:** Zero C deps on default build. `deny.toml` bans 12 categories.
  4. **Large files:** All 17 files >800L are standalone bin targets. Library max is
     `niche.rs` (846L) — proper structure, not a split candidate.
  5. **Unsafe:** 2 files, both feature-gated (`low-level`, `cuda-validation`).
     Library has `#![forbid(unsafe_code)]`.
  6. **Hardcoding:** coralReef socket paths evolved to env-var discovery.
     8 binaries migrated. All `/proc/`, `/sys/`, BDF, NPU paths already discoverable.
  7. **Mocks:** Zero production mocks. All outside `#[cfg(test)]`.
- **Resolved:**
  1. `fleet_client::ember_socket_candidates(bdf)` + `glowplug_socket_path()` with
     `TOADSTOOL_GLOWPLUG_SOCKET` / `TOADSTOOL_RUN_DIR` env-var discovery.
  2. 8 experiment binaries migrated from `/run/coralreef/` hardcoded paths.
  3. `gpu_flow.rs` placeholder buffer labels → `*_unused`.
  4. `silicon_qcd/flow.rs` removed `_uni: ()` placeholder parameter.
  5. Clippy `collapsible_str_replace` fix.
- **Documented for future evolution:**
  25 binary targets use `panic!` in unrecoverable paths (tokio runtime, GPU init,
  weight export). Library has zero panic paths. Evolution to `Result` mains is polish.
- **Validation:** 595 (default) / 1,041 (barracuda-local). Zero clippy. Zero TODO.
- **Handoff:** `wateringHole/handoffs/HOTSPRING_DEEP_DEBT_SPRINT_MAY13_2026.md`

---

### GAP-HS-098 — Compute Trio Pipeline Rewire (May 13, 2026)

- **Scope:** Pull + review + rewire from toadStool S255-S257, barraCuda Sprint 64-67,
  coralReef Sprint 5-6.
- **What was rewired:**
  1. **toadStool S255-S257:**
     - Capability registry: added `ember.swap`, `sovereign.boot` semantic aliases
       (both → `device_swap`), `auth.peer_info`, `provenance.get`.
     - `niche.rs` ROUTED_CAPABILITIES aligned to 77 toadStool methods.
     - FECS warm-state detection (`probe_warm_fecs()`) now in `NvVfioComputeDevice`:
       BAR0 probe for PMC_ENABLE, FECS CPUCTL HALTED (bit 5 — reconciled from
       incorrect bit 4), FECS MAILBOX0.
     - CPUCTL HALTED bit fix: bit 5 (0x20) is HALTED, bit 4 was HRESET. Fixed
       in toadStool `mmio.rs` and `firmware.rs`.
  2. **barraCuda Sprint 64-67:**
     - `PrecisionAdvisory` struct: added `dispatch_path` field
       (`"wgpu"` | `"sovereign"` | `"unavailable"`).
     - `dispatch_path` consumed in `s_compute_trio` and `s_hotqcd_dispatch`
       validation scenarios.
     - GAP-HS-041 RESOLVED: `stats.entropy` available (alias of `stats.shannon`).
     - GAP-HS-027 updated: `TensorSession::sub()` and `negate()` shipped (Sprint 66).
       IPC `tensor.batch.submit` handles `sub`/`negate` ops. GEMM routing wired
       with `MatmulPrecision` and tensor-core hints. OOM detection available.
  3. **coralReef Sprint 5-6:**
     - **IPC param fix:** hotSpring was sending `"source"` to `shader.compile.wgsl`;
       coralReef expects `"wgsl_source"`. Fixed in `s_compute_trio.rs`,
       `s_hotqcd_dispatch.rs` (2 call sites), and `compute_dispatch.rs`.
     - Capability registry: added `shader.compile.wgsl.multi`,
       `shader.compile.status`, `shader.compile.capabilities` (new coralReef methods).
     - `naga::Module` direct ingest (`compile_module`/`compile_module_full`) available.
     - PTX SM120: switch lowering, math builtins, atomics/barriers/subgroups shipped.
       Full SM120 texture instruction set still in progress.
     - **HMMA GEMM codegen SHIPPED**: `shader.compile.gemm` endpoint generates PTX
       using `mma.sync.aligned` for SM80+ (f16→f32, f16→f16, TF32). `compile_gemm()`
       produces native tensor-core kernels. Wire compat aliases added (`source` →
       `wgsl_source`, `binary` → `binary_b64`, `info` → `shader_info`).
       `CompilationInfoResponse` wire fields: `gprs`, `shared_memory`, `barriers`,
       `workgroup`, `wave_size`, `local_memory`.
  4. **Env var deprecation:** Legacy **`CORALREEF_SOCKET`** (and related **`CORALREEF_*`** overrides) in `precision_brain.rs` emit warnings — prefer **`TOADSTOOL_SOCKET`**. **`CORALREEF_MANIFEST`** fallback removed.
- **Remaining:**
  - Wire `TensorSession` into `gpu_hmc/mod.rs` (GAP-HS-027 — upstream unblocked).
  - **Hardware validation**: Run DMA roundtrip, GPFIFO NOP submission, warm channel
    creation, and sync timing on Titan V and K80 with toadStool S258 PBDMA dispatch.
  - Run `s_compute_trio` and `s_vfio_dispatch` live with all three GPUs.
  - **HMMA GEMM E2E**: coralReef `shader.compile.gemm` → PTX with `mma.sync.aligned`
    → toadStool PBDMA dispatch. The compile side is ready (SM80+). E2E requires
    Compute QMD construction: coralReef builds Volta compute class method headers
    (SET_SHADER_PROGRAM, LAUNCH); toadStool submits via PBDMA GPFIFO.
  - `CompileResponse` wire compat: coralReef now serves both canonical (`binary_b64`,
    `shader_info`, `gprs`) and legacy aliases (`binary`, `info`, `gpr_count`).
    hotSpring parsers handle both forms.
- **Validation:** 591/591 lib tests pass. Zero clippy warnings.

---

### GAP-HS-100 — Local Debt Resolution + Composition Evolution (May 14, 2026)

- **Scope:** Seven-item sprint resolving fragile composition patterns.
- **What was implemented:**
  1. **Compile-then-dispatch pipeline:** `compile_and_submit()` chains coralReef compile → toadStool dispatch.
  2. **Circuit-breaker discovery:** `PrimalEndpoint` tracks failures, marks dead after 3, re-probes after 30s.
  3. **Dispatch unification:** `compute_dispatch.rs` canonical; `fleet_toadstool.rs` submit/dispatch deprecated.
  4. **FusedPipeline typed errors:** `FusedSubmitReport` with per-op `Submitted`/`Failed` outcomes.
  5. **`parse_jsonrpc_response()`:** Centralized JSON-RPC envelope parsing.
  6. **TOML-loaded aliases:** `[primal_aliases]` in `capability_registry.toml` at runtime.
  7. **Tiered validation:** `validate_all --tier smoke|nucleus|silicon` (65 suites: 35/7/23).
- **Upstream gaps discovered:**
  - **coralReef:** `sum_reduce_subgroup_f64.wgsl` causes assertion panic killing daemon (GAP for coralReef team).
  - **toadStool:** `compute.dispatch.submit` returns no structured error on invalid binary (empty result, no error field).
  - **plasmidBin:** ecoBin pipeline lag — pre-subgroup-ops / pre-S259 harvests. Automated CI harvest would prevent.
- **Validation:** 595/595 lib tests pass (default). Zero clippy warnings.
- **Handoff:** `wateringHole/handoffs/HOTSPRING_LOCAL_DEBT_COMPOSITION_EVOLUTION_HANDOFF_MAY14_2026.md`

---

### GAP-HS-101 — plasmidBin Local Debt Resolution (May 14, 2026)

- **Scope:** plasmidBin ecoBin harvest lag, tooling bugs, deployment gaps.
- **What was implemented:**
  1. **Release cascade in `fetch.sh`:** Cascades through 5 most recent releases when a binary is missing from `latest`, solving the incremental-release problem.
  2. **Symlink-aware `doctor.sh`:** Fixed `file` → `file -L` and `du -h` → `du -hL` to correctly identify `static-pie` ecoBins behind backward-compat symlinks.
  3. **Generalized `upgrade-primal.sh`:** Unified upgrade script supporting `--all`, `--trio`, `--status`, `--check`, with automatic rollback on service failure.
  4. **User-mode systemd services:** `barracuda-user.service` and `coralreef-user.service` for `systemctl --user` deployment.
  5. **Full NUCLEUS deployment:** 13/13 primals deployed to `/usr/local/bin/`, all ecoBin (static-pie, stripped).
- **Upstream gaps discovered:**
  - **GAP-PB-001:** `skunkbat` missing checksum entry in `checksums.toml` for `x86_64-unknown-linux-musl` (plasmidBin/primalSpring).
  - **GAP-PB-002:** barracuda does not implement `health.version` RPC (barraCuda team).
- **Validation:** 595/595 lib tests pass. Zero clippy warnings. 13/13 primals deployed. 3/3 compute trio live on IPC.
- **Handoff:** `wateringHole/handoffs/HOTSPRING_PLASMIDBIN_LOCAL_OWNERSHIP_HANDOFF_MAY14B_2026.md`

### GAP-HS-102 — Upstream Absorption + Deep Debt Sprint (May 14, 2026)

- **Severity:** Low (lint hygiene + composition alignment)
- **Classification:** Upstream absorption + deep debt lint evolution
- **Trigger:** primalSpring "Ecosystem Status Update — May 14, 2026" directive.
  barraCuda v0.4.0 released back, coralReef v0.1.0 released back as pure sovereign compiler,
  plasmidBin composition evolved with skunkBat per atomic model.
- **Completed:**
  1. **Clippy zero warnings restored** across `--all-targets --features barracuda-local`:
     ~25 warnings fixed across bins, tests, and lib (auto-fix + manual).
  2. **Test lint expectations aligned:** 5 `#[cfg(test)]` modules gained `#![expect]`,
     3 unfulfilled expectations removed, 4 integration tests migrated `#[allow]` → `#[expect]`.
  3. **plasmidBin composition:** `[niches.hotspring]` and `ports.env` updated to include
     skunkBat per Tower = bearDog + songBird + skunkBat atomic model. `COMP_TOWER` aligned.
  4. **barraCuda v0.4.0 alignment confirmed:** precision/E2E, VFIO dispatch, health.version
     already wired by biomeGate. No stale `*_cpu` scalar names in codebase.
  5. **coralReef v0.1.0 alignment confirmed:** IPC-only (no direct Rust dependency).
     Blackwell SM120, naga::Module ingest, dual-vendor all accessible via IPC.
- **Validation:** 595/595 lib tests pass. Zero clippy warnings (all-targets barracuda-local).
  Zero unfulfilled lint expectations. `cargo fmt --check` clean.

### GAP-HS-103 — Wave 17 Signal Adoption + Deep Debt Sprint (May 16, 2026)

- **Severity:** Low (forward evolution + structural improvements, no regressions)
- **Classification:** Neural API signal adoption + deep debt refactoring audit

#### Signal adoption (per primalSpring Wave 17)
  1. **`primal.announce` registration**: `niche.rs` refactored — `register_with_target()`
     now tries `primal.announce` first, falling back to legacy multi-call for older biomeOS.
  2. **`node.compute` signal dispatch**: `dispatch_node_compute()` in `compute_dispatch.rs`.
  3. **`tower.publish` signed publication**: `publish_result()` in `compute_dispatch.rs`.
  4. **Capability registry signals**: `[signals]` section with adopted/candidate tiers.
- **Signal candidates remaining:** `nest.store` (awaiting nestGate evolution).
  `nest.commit` was promoted to adopted in Wave 20 (GAP-HS-104).

#### Deep debt refactoring
  5. **`niche.rs` (932L) → `niche/mod.rs` (516L) + `niche/tables.rs` (394L)**: Table
     extraction. Registration logic visible without scrolling past 400L of data.
  6. **`compute_dispatch.rs` (926L) → `compute_dispatch/mod.rs` (592L) + `fused.rs` (335L)**:
     FusedPipeline and 5 types extracted with their own tests.

#### Comprehensive audit results
- **TODO/FIXME/HACK:** Zero in codebase.
- **`unsafe`:** 10 sites, all necessary (8 BAR0 MMIO mmap/volatile, 2 `Send` impls, 1 CUDA FFI).
- **Large files (>800L):** 19 files. 2 refactored this sprint (`niche.rs`, `compute_dispatch.rs`).
  Remaining 17 are experiment binaries (biomeGate-owned sovereign compute):
  `sarkas_gpu.rs` (856L), `nuclear_eos_l2_hetero.rs` (858L), `nuclear_eos_gpu.rs` (870L),
  `validate_silicon_capabilities.rs` (867L), `validate_precision_matrix.rs` (885L),
  `validate_chuna.rs` (887L), `bench_qcd_silicon.rs` (908L), `bench_silicon_saturation.rs` (923L),
  `bench_silicon_budget.rs` (845L), `validate_silicon_science.rs` (844L),
  `bench_cross_spring_evolution.rs` (934L), `cross_substrate_esn_benchmark.rs` (942L),
  `celllist_diag.rs` (951L), `exp184_k80_gr_sovereign.rs` (932L),
  `glowplug_client.rs` (938L), and 2 fossilized bins.
- **External deps:** 14 crates, all pure Rust except `cudarc` (CUDA FFI, feature-gated) and
  `wgpu` (inherent to GPU mission). `blake3` configured C-free.
- **Mock leakage:** None. `NpuSimulator` is intentional production simulation.
- **Hardcoded paths:** ~45 socket path literals. Production paths use env-var fallback chains
  (`BIOMEOS_SOCKET_DIR` → `XDG_RUNTIME_DIR` → `/run/toadstool` → `temp_dir`). Test paths
  are appropriately isolated in `#[cfg(test)]` modules.

#### Benchmark & paper audit
- **Python CPU benchmarks:** Yes — `control/` + `validate_*` + `run_all_parity.py`, 9/9 papers green.
- **Kokkos/LAMMPS GPU parity:** Yes — `bench_md_parity`, `bench_kokkos_complexity`. Gap 27× → 3.7×.
- **OpenMM/GROMACS/Galaxy:** Not applicable per project physics scope.
- **Papers remaining:** ~25 queued (Paper 23, Track 5 [25-31], Tier 4 [32-42], LTEE B9).
- **Datasets partial:** Militzer FPEOS (partial), atoMEC (7/9), Zenodo surrogate (optional 6GB).

- **Validation:** 595/595 lib tests pass. Zero clippy warnings. `cargo fmt --check` clean.

### GAP-HS-104 — Wave 20 Schema Standardization (May 16, 2026)

- **Severity:** Low (schema alignment, no regressions)
- **Classification:** primalSpring Wave 20 absorption
- **Trigger:** primalSpring "Wave 20 Delta Spring Evolution — Schema Standardization + E2E Validation"
  directive. Registry expanded to 452 methods. `capability.list` canonical envelope defined.
  `nest.commit` E2E validation scenario added upstream.
- **Completed:**
  1. **`capabilities_list_response()`**: New canonical response builder in `niche/tables.rs`.
     Returns `{ "capabilities": [...], "count": N, "primal": "hotspring" }` per Wave 20 schema.
  2. **`primal.list` registered**: Added to `capability_registry.toml` as routed capability
     (biomeOS-served) and to `ROUTED_CAPABILITIES` in `niche/tables.rs`.
  3. **`nest.commit` signal dispatch**: `commit_provenance()` in `dag_provenance.rs` dispatches
     via `signal.dispatch("nest.commit", ...)` with fallback to direct `ledger.record` +
     `attribution.braid`. Signal promoted from candidate to adopted in `[signals]`.
  4. **`s_schema_standard` scenario**: Validates capability.list envelope shape (array, count,
     primal), signal registry presence (3 adopted, 1 candidate), niche identity constants.
- **Validation:** 596 (default) / 1,045 (barracuda-local) lib tests pass. Zero clippy warnings.
  `cargo fmt --check` clean.

### GAP-HS-105 — Deep Debt Sprint: glowplug_client Refactor + Full Audit (May 16, 2026)

- **Severity:** Low (refactoring, no regressions)
- **Classification:** Deep debt resolution
- **Trigger:** Reiterated deep debt directive — modernize, refactor large files, audit all debt dimensions.
- **Completed:**
  1. **`glowplug_client.rs` refactored (938L → 647 + 221):** Split into `glowplug_client/mod.rs`
     (client impl, free functions, tests — 647L) and `glowplug_client/types.rs` (protocol types,
     error variants, response structs — 221L). **Zero library files >800L remain** — highest is
     `gpu_rhmc.rs` at 796L.
  2. **`#[allow(deprecated)]` → `#[expect(deprecated)]`** in `compute_dispatch/mod.rs:359`:
     upgraded to idiomatic Rust 1.81+ lint expectation.
  3. **Full audit re-confirmed:**
     - **TODO/FIXME/HACK:** Zero in all library code.
     - **`unsafe`:** 9 blocks in `low_level/bar0.rs` (MMIO — inherently unsafe, well-audited),
       1 in `validate_5060_dual_use.rs` (CUDA FFI, binary only). No evolution possible.
     - **External deps:** All pure Rust except `cudarc` (CUDA FFI, feature-gated) and `wgpu`
       (Vulkan backend, unavoidable for GPU). `blake3` explicitly avoids C build path.
     - **Hardcoding:** Production paths (`/run/toadstool`) have env-var fallbacks. Routing tables
       in `niche/tables.rs` are compile-time self-knowledge by design.
     - **Mocks:** `NpuSimulator` is intentional cross-substrate math, not test leakage.
     - **`.unwrap()`:** `#![deny(clippy::unwrap_used)]` enforced at crate level — zero in lib code.
     - **`Box<dyn>`:** Zero in hot paths.
  4. **Benchmark status confirmed:**
     - Python CPU parity: 10 papers wired (6, 8–13, 43–45), `run_all_parity.py` green.
     - Kokkos GPU parity: `bench_md_parity`, `bench_kokkos_complexity` — gap 27× → 3.7×.
     - OpenMM/GROMACS/Galaxy: Not applicable per physics scope.
  5. **Paper queue:** ~25 queued (Paper 23/Sulfolobus, Track 5 [25–31], Tier 4 [32–42], LTEE B9).
     LTEE B1–B7 complete per primalSpring `LTEE_PAPER_QUEUE_TRACKER.md`.
  6. **Dataset status:** Militzer FPEOS (partial), atoMEC (7/9), Zenodo surrogate (optional 6GB),
     Dense Plasma Properties DB (off-repo), Sulfolobus genomes (wetSpring queued).
- **Validation:** 596 (default) / 1,045 (barracuda-local) lib tests pass. Zero clippy warnings.
  `cargo fmt --check` clean.

### GAP-HS-106 — Wave 20 Debt Resolution (May 17, 2026)

- **Severity:** Low (documentation drift + fossilized binary alignment)
- **Classification:** primalSpring Wave 20 debt resolution audit
- **Trigger:** `WAVE20_DEBT_RESOLUTION_MAY17_2026.md` — primalSpring audited all delta springs
  post-Wave 20 and identified 4 hotSpring-specific debt items.
- **Completed:**
  1. **Fossilized RPC shape fixed**: `hotspring_primal.rs` `capability.list` handler now returns
     canonical `{ "capabilities": [...], "count": N, "primal": "hotspring" }` envelope.
  2. **`nest.commit` doc drift resolved**: Removed `nest.commit` from candidate lists in
     `PRIMAL_GAPS.md` (GAP-HS-103), Wave 17 handoff, and doc evolution handoff.
     Only `nest.store` remains as candidate (awaiting nestGate).
  3. **`commit_provenance()` documented as scaffolding**: Added wiring status doc to
     `dag_provenance.rs` — ready for Titan V pipeline session finalization or any
     `validate_*` binary running a full DAG lifecycle.
  4. **Test count verified**: 596 (default lib) / 1,045 (barracuda-local lib) confirmed.
     Workspace `cargo test` compile errors in biomeGate binaries (feature-gate debt) —
     not strandGate-side issue. primalSpring's cited 1,607 figure includes binary tests
     that require biomeGate's `barracuda-local` feature gating fix.
- **Validation:** 596 (default) / 1,045 (barracuda-local) lib tests pass. Zero clippy warnings.

### GAP-HS-107 — Tier 2 Sovereign Compute Blocker: GPC Power Domain Wall (May 19, 2026)

- **Severity:** High (blocks all sovereign shader execution on Titan V)
- **Classification:** Hardware power domain boundary — blocks vendor-atheistic compute
- **Root Cause:** After nouveau unbind, all GPU engine domains (GR, CE, NVDEC) are
  power-gated. The PRI ring to these domains is dead — reads return `0xbadfXXXX`
  fault values. This is a hardware power domain boundary, not a software/configuration
  issue. The chicken-and-egg: need PRI ring to write power registers, but PRI ring
  requires the target domain to be powered.
- **Evidence:**
  - CE0 at `0x104000` → `0xbadf3000` (PRI fault)
  - GPCCS at `0x41A004` → `0xbadf5545` (PRI fault)
  - PBDMA DEVICE error (intr_0 bit 28) when dispatching to gated engines
  - PGRAPH_STATUS at `0x400700` → `0x00000000` (no activity)
- **Impact:** Sovereign VFIO infrastructure (Tier 1) is fully validated, but sovereign
  compute dispatch (Tier 2) is completely blocked on Titan V. RTX 5060 has Tier 2
  via DRM path (proprietary driver), but the VFIO sovereign path is the atheistic target.
- **Proposed Solutions (prioritized):**
  1. **PMU Mailbox Protocol** (Exp 211): PMU falcon at `0x10A000+` is alive post-unbind.
     Send PG_CTRL command via MBOX0/MBOX1 to ungate GPC domain. Success criterion:
     `GPC_ENABLES` at `0x41A004` returns non-fault value.
  2. **Kernel Patch**: Modify nouveau `gv100_gr_fini()` to skip GPC power-down during unbind.
  3. **nvidia-470 Handoff**: Use proprietary driver as warm handoff source (keeps GPCs powered).
  4. **K80 Cross-Gen**: Incoming K80 has unsigned falcons (no ACR) — may reach Tier 2
     via direct PIO falcon upload before Volta.
- **Cross-References:** Exp 210 (`experiments/210_SOVEREIGN_GPC_BOUNDARY.md`),
  Exp 211 (`experiments/211_PMU_MAILBOX_TIER2_INVESTIGATION.md`),
  GAP-HS-047 (PMU firmware extraction — deprioritized, may be relevant),
  `infra/whitePaper/gen4/architecture/SILICON_DEISM.md`
- **Status:** OPEN — awaiting PMU mailbox investigation (Exp 211)

---

## Handback Protocol

1. Document gap in this file with severity and upstream reference.
2. If the gap requires primal evolution: PR to `primalSpring/docs/PRIMAL_GAPS.md`.
3. If the gap requires graph evolution: PR to `primalSpring/graphs/downstream/`.
4. If the gap surfaced a new pattern: handoff to `ecoPrimals/infra/wateringHole/handoffs/`.
