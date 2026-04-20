# hotSpring Primal Composition Gaps

**Spring:** hotSpring v0.6.32
**Proto-nucleate:** `downstream_manifest.toml` (spring_name = "hotspring")
**Particle profile:** proton-heavy (Node atomic dominant)
**Date:** April 10, 2026
**Last audited:** April 20, 2026 (primalSpring v0.9.17 absorption — genomeBin v5.1, guideStone v1.2.0, deployment-validated end-to-end, v0.9.17 env var requirements documented)
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

### GAP-HS-030: GV100 WPR Not Used by Closed Driver (Exp 173)

- **Primal:** coralReef (coral-driver / sovereign_init)
- **Severity:** Critical (blocks GV100 sovereign ACR boot)
- **Status:** Open — approach pivot required
- **Description:** Experiment 173 proved the nvidia-535 closed driver does NOT
  configure WPR (Write-Protected Region) on GV100 Titan V. WPR registers
  (PFB_WPR1/WPR2 at 0x100CE4-CF0) remain zero while nvidia-smi shows a
  fully functional GPU. GV100 is pre-GSP: the RM runs on the CPU and does
  not need WPR hardware protection. The "vendor UEFI WPR capture" approach
  is a dead end for Volta.
  
  The ACR chain in coral-driver's `FalconBootSolver` expects WPR to be
  configured (Turing+ assumption). On GV100, SEC2 cannot enter HS mode
  because the WPR-based signature verification path doesn't apply. The
  no-ACR warm handoff (Exp 172: nouveau→vfio swap preserving HBM2) remains
  the best achievable state, but FECS/GPCCS remain halted because they
  require ACR bootstrapping that depends on WPR.
  
  Possible paths: (a) reverse-engineer nvidia-535 mmiotrace to find how RM
  initializes FECS without ACR on Volta, (b) focus sovereign boot on K80
  (Kepler, no ACR) and Turing+ (where WPR is actually used), (c) accept
  vendor-in-VM as GV100 compute path.
- **Action:** Pivot coral-driver `FalconBootSolver` to support a Volta-specific
  path that bypasses WPR/ACR. Analyze mmiotrace from Exp 173 artifacts.

### GAP-HS-031: Blackwell SM Warp Exception — Invalid Address Space (Exp 175-177)

- **Primal:** coralReef (coral-driver / coral-kmod / uvm_compute)
- **Severity:** Critical (blocks sovereign dispatch on RTX 5060)
- **Status:** Root cause identified — fix requires kmod GPU_PROMOTE_CTX
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
  
  Remaining blocker for dispatch: GAP-HS-031 (GPFIFO NOP timeout on Blackwell).
- **Action:** None — compile parity resolved. Dispatch parity tracked in GAP-HS-031.

---

## Handback Protocol

1. Document gap in this file with severity and upstream reference.
2. If the gap requires primal evolution: PR to `primalSpring/docs/PRIMAL_GAPS.md`.
3. If the gap requires graph evolution: PR to `primalSpring/graphs/downstream/`.
4. If the gap surfaced a new pattern: handoff to `ecoPrimals/infra/wateringHole/handoffs/`.
