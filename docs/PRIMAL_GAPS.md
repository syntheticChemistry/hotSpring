# hotSpring Primal Composition Gaps

**Spring:** hotSpring v0.6.32
**Proto-nucleate:** `downstream_manifest.toml` (spring_name = "hotspring")
**Particle profile:** proton-heavy (Node atomic dominant)
**Date:** April 10, 2026
**Last audited:** May 11, 2026 (sovereign barrier resolution + Post-Interstadial Evolution: Volta ACR skip, HBM2 warm-handoff, benchScale VM isolation, K80 PCIe link, skunkBat IPC, Tier 4 IPC-first)
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
- **Status:** Deferred (supersedes GAP-HS-007)
- **Description:** barraCuda's `TensorSession` fused multi-op pipeline
  API is not yet adopted in hotSpring. GPU HMC trajectory (leapfrog +
  force + gauge update) is the natural first candidate. Blocked on
  TensorSession API stabilization for lattice workloads.
- **Action:** Wire `TensorSession` into `gpu_hmc/mod.rs` when barraCuda
  API stabilizes.

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
- **Status:** Active — mitigated locally
- **Description:** toadStool IPC is slow on short timeouts (< 5s). Composition
  script uses >= 10s for real compute dispatch. Background validation via
  `dispatch_background_validation()` uses async polling to avoid blocking.
- **Action:** Upstream toadStool should document minimum recommended timeout.

### GAP-HS-041: barraCuda stats.entropy Missing (PG-47)

- **Primal:** barraCuda
- **Severity:** Low
- **Status:** Active — mitigated locally
- **Description:** barraCuda may lack `stats.entropy` method. Composition script
  uses `stats.mean` as a proxy for tensor IPC validation and computes entropy
  locally when needed.
- **Action:** Monitor barraCuda for `stats.entropy` addition.

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
- **Status:** Implemented in coral-driver, needs ecosystem documentation
- **Description:** The fork-isolation pattern (`fork_isolated_raw` +
  `MappedBar::isolated_*` safe wrappers) is a reusable primitive for any
  hardware operation that might hang. Currently lives only in coral-driver.
  Should be documented in `SPRING_COMPOSITION_PATTERNS.md` or
  `ECOBIN_ARCHITECTURE_STANDARD.md` as a recommended pattern for hardware
  fault containment.
- **Action:** PR to primalSpring documenting fork-isolation as ecosystem pattern.

### GAP-HS-030a: Ember Absorption into toadStool

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

### GAP-HS-030b: K80 VFIO Legacy Group EBUSY — RESOLVED

- **Primal:** coralReef (ember)
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

### GAP-HS-030: GV100 FECS Secure Boot Without WPR — PARTIALLY RESOLVED

- **Primal:** coralReef (coral-driver / sovereign_init)
- **Severity:** Critical (blocks GV100 sovereign FECS boot)
- **Status:** Partially resolved — ACR solver bypass done, warm-handoff via
  benchScale/agentReagents is the production path
- **Description:** Experiment 173 proved the nvidia-535 closed driver does NOT
  configure WPR (Write-Protected Region) on GV100 Titan V. WPR registers
  (PFB_WPR1/WPR2 at 0x100CE4-CF0) remain zero. GV100 is pre-GSP: the RM
  runs on the CPU and does not need WPR hardware protection.
  
  **May 10-11, 2026 progress:**
  - `sovereign_stages.rs`: Added Volta CpuRm early-exit that detects
    `AcrSec2 + !wpr_configured + SM 70-74` and skips the ACR solver,
    going directly to PIO FECS bootstrap. Eliminates >5min hang (now 4s).
  - Sovereign init stages 1-3 all pass on warm GPU: bar0_probe OK,
    pmc_enable OK (0x5fecdff1), hbm2_training SKIPPED (warm detected).
  - HBM2 warm-handoff validated: nouveau trains 12GB HBM2, PRAMIN fully
    accessible after `reset_method=none` VFIO rebind.
  - FECS remains blocked: secure Falcon v5 rejects unsigned PIO upload
    (cpuctl=0x12, mb0=0x0, running=false). nouveau skips GR entirely on
    GV100 (PMU firmware unavailable). Direct PIO and ACR are both dead ends.
  
  **Production path:** nvidia-470 warm-handoff via benchScale VM isolation.
  nvidia-470 is the only driver that initializes GR/FECS on GV100 (with
  its embedded PMU firmware). The handoff runs inside a benchScale/
  agentReagents VM to avoid crashing the host DRM. The host DRM (nvidia-580
  serving RTX 5060 display) stays completely uninterrupted.
- **Action:** Wire `agentReagents/templates/reagent-nvidia470-titanv.yaml`
  for benchScale-driven warm-handoff. Physical card swaps also viable
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

### GAP-HS-032: Generation-Branched FalconBootSolver (Resolved)

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

### GAP-HS-033: coralReef f64 Transcendental Lowering for SM32 (Resolved)

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

### GAP-HS-057: K80 GK210 Nouveau Chipset ID Missing — PROVEN FIX (P0)

- **Primal:** coralReef (sovereign pipeline) / upstream kernel (nouveau)
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
  Post-VFIO-rebind GPC stations power-gated (livepatch load failure on 6.17) —
  need PLX keepalive fix + livepatch rebuild for full warm-catch.
- **Remaining:** Livepatch module format incompatible with kernel 6.17 strict relocation
  enforcement. Need either kprobe-based approach or kernel source rebuild.
- **Ref:** Exp 185, Exp 188, Exp 171, Exp 178, Exp 181

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

---

## Handback Protocol

1. Document gap in this file with severity and upstream reference.
2. If the gap requires primal evolution: PR to `primalSpring/docs/PRIMAL_GAPS.md`.
3. If the gap requires graph evolution: PR to `primalSpring/graphs/downstream/`.
4. If the gap surfaced a new pattern: handoff to `ecoPrimals/infra/wateringHole/handoffs/`.
