# Changelog

All notable changes to hotSpring.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file covers the spring as a whole. For crate-level details see
`barracuda/CHANGELOG.md`.

## Unreleased — primalSpring Audit Absorption (May 17, 2026 PM)

### Added (lithoSpore R1–R4 absorption)
- **`s_anderson_parity` validation scenario**: Cross-tier parity for Anderson
  module (Python spectral_control.py vs Rust validate_spectral). 6 checks:
  Herman/Lyapunov, level statistics, 3D bandwidth, GOE→Poisson, dimensional
  hierarchy, mobility edge. `barracuda-local` feature gate.
- **`spectral_parity.py`**: Python-side cross-tier parity checker comparing
  spectral_control.json against Rust reference values. 6/6 ALL PASS.
- **`docs/IPC_DEGRADATION_BEHAVIOR.md`**: Consolidated degradation behavior
  for all 14 primal interactions. Documents what happens when each primal is
  down (all return `Err`/`None`, never panic), circuit breaker (3 failures →
  dead, 30s reprobe), and validation degradation (honest failures, skip-pass
  for standalone mode).
- **Stability tier annotations**: All 117 capabilities in
  `barracuda/config/capability_registry.toml` now have `stability` field
  (`stable`/`evolving`) aligned with primalSpring upstream registry.

### Changed
- **`commit_provenance()`**: Now reports `primals_reached` in return value
  listing which trio components (rhizoCrypt/loamSpine/sweetGrass) were
  successfully contacted. Partial completion is explicit per trio semantics.
- **`dag_provenance.rs` module docs**: Added trio transaction semantics
  documentation (non-atomic commit, partial states, no rollback, domain
  logic never gates on provenance).
- **`spectral_control.py`**: Fixed numpy bool JSON serialization for `--json`
  output mode.

## Unreleased — Wave 20 Experiment Buildouts + Compute Parity (May 17, 2026)

### Added (4-phase experiment + compute evolution sprint)
- **Experiment 197 + 198 standalone files**: `experiments/197_SOVEREIGN_INIT_RPC_WARM_COLD.md`
  and `experiments/198_VENDOR_AGNOSTIC_BOOT_PIPELINE.md` — extracted from EXPERIMENT_INDEX
  entries into full dedicated experiment journals.
- **`s_cpu_gpu_parity` validation scenario**: CPU reference stability checks across
  7 physics domains (QCD, SEMF, Transport, Spectral SpMV, BGK, Euler, Coupled
  Kinetic-Fluid). `barracuda-local` feature gate.
- **`s_toadstool_dispatch` validation scenario**: Offline toadStool dispatch validation
  (parameter assembly, input hashing, barrier shader paths, witness construction,
  serialization, `commit_provenance` parameter checks). No IPC required.
- **`s_mixed_hardware` validation scenario**: forge dispatch routing, pipeline topology
  construction, NUCLEUS atomic composition, and biomeOS graph coordination. Offline.
  `barracuda-local` feature gate.
- **`dispatch_cpu_fallback()`** in `compute_dispatch/mod.rs`: Local CPU execution for
  `vector_add` and `semf_batch` workloads when toadStool is unavailable.
- **`forge::nucleus` module**: NUCLEUS atomic types (Tower/Node/Nest/FullNucleus) with
  domain mappings, substrate compatibility, and `AtomicBinding` for substrate dispatch.
- **`ChannelKind::PcieDirect`**: GPU→NPU PCIe peer-to-peer (no CPU roundtrip) in
  `metalForge/forge/src/pipeline.rs`. New topologies: `mixed_pcie_direct()`,
  `nucleus_atomic()`.
- **`forge::biome_graph` module**: NUCLEUS atomic coordination as directed graph —
  nodes are `(AtomicType, SubstrateKind)` instances, edges are `ChannelKind` connections.
  Pathfinding, reachability, `pcie_direct_hops()`. Standard and PCIe-direct graph
  constructors.

### Changed
- **`validate_mixed_substrate` binary**: Extended with NUCLEUS atomic binding checks,
  PCIe direct topology validation, and biomeOS graph coordination tests.
- **Parity greenboard**: Regenerated to ALL GREEN (10/10 papers). Paper 45 kinetic-fluid
  gap resolved (control JSON already existed, greenboard was stale).
- **`PAPER_REVIEW_QUEUE.md`**: GPU coverage clarified — papers 2,7 CPU-only by design;
  6,17,21,22 CPU-natural; 20 GPU-promotable via SpMV+Lanczos.
- **Doc metrics normalized**: 198 experiments, 596/1,045 lib tests, 22 validation
  scenarios (17 default + 5 barracuda-local) across README, specs, experiments,
  whitePaper/baseCamp docs.

### Metrics
- Validation scenarios: 22 (17 default + 5 barracuda-local)
- Lib tests: 596 (default) / 1,045 (barracuda-local) — all pass
- Clippy: zero warnings
- metalForge/forge tests: 32/32 non-GPU pass (4 wgpu-backend expected in headless)

## Unreleased — Wave 20 Debt Resolution (May 17, 2026)

### Fixed (primalSpring Wave 20 audit response)
- **Fossilized RPC canonical envelope**: `hotspring_primal.rs` `capability.list` handler
  now returns `"count"` and `"primal"` fields per Wave 20 schema standard.
- **`nest.commit` doc drift**: Removed from candidate lists in PRIMAL_GAPS.md, Wave 17
  handoff, and doc evolution handoff. Signal was promoted to adopted in Wave 20.
- **`commit_provenance()` wiring doc**: Added scaffolding status documentation to
  `dag_provenance.rs` — ready for Titan V pipeline integration.

### Metrics
- Lib tests: 596 (default) / 1,045 (barracuda-local)
- Clippy: zero warnings
- Workspace binary compile: biomeGate feature-gate debt (not strandGate scope)

## Unreleased — Deep Debt Sprint: glowplug_client Refactor (May 16, 2026)

### Changed (deep debt — large file refactoring)
- **`glowplug_client.rs` (938L) → `glowplug_client/mod.rs` (647L) + `glowplug_client/types.rs`
  (221L)**: Protocol types (GlowplugDispatchOptions, GlowplugDeviceSummary, GlowplugDeviceDetail,
  GlowplugDaemonHealth, GlowplugError, CaptureTrainingResult, WarmCatchResult,
  SovereignBootResult, BootStepResult) extracted to `types.rs`. Client impl, free functions, and
  tests remain in `mod.rs`. **Zero library files >800L remain.**
- `#[allow(deprecated)]` upgraded to `#[expect(deprecated, reason = "...")]` in
  `compute_dispatch/mod.rs` for idiomatic Rust 1.81+ lint expectations.
- **Full deep debt audit re-confirmed:** zero TODO/FIXME/HACK, zero `.unwrap()` in lib,
  zero `Box<dyn>` in hot paths, `unsafe` confined to `bar0.rs` MMIO + CUDA FFI binary.

### Documentation evolution (same sprint)
- All docs normalized to 596/1,045 test counts, 198 experiments
- README directory tree updated for `niche/`, `compute_dispatch/`, `glowplug_client/` modules
- `hotspring_primal.rs` → `_fossilized/` path corrected in tree
- Notebook upgraded to Level 6 CERTIFIED, stale counts fixed
- whitePaper/README.md date updated to May 16, 2026; experiment count 190 → 198
- whitePaper/baseCamp/README.md date updated to May 16, 2026
- 5 May 13–14 handoffs archived to `wateringHole/handoffs/archive/`
- scripts/README.md: `tools/` path clarified, archive date description improved
- Upstream handoff: `infra/wateringHole/handoffs/HOTSPRING_DEEP_DEBT_DOC_EVOLUTION_MAY16_2026.md`

### Metrics
- Lib tests: 596 (default) / 1,045 (barracuda-local)
- Clippy: zero warnings
- Library files >800L: **zero** (highest: gpu_rhmc.rs 796L)

## Unreleased — Wave 20 Schema Standardization (May 16, 2026)

### Added (Wave 20 absorption — primalSpring schema standard)
- **`capabilities_list_response()`** in `niche/tables.rs` — canonical response
  builder for `capability.list` per Wave 20 schema: `{ "capabilities": [...],
  "count": N, "primal": "hotspring" }`. All downstream consumers get the required
  canonical subset.
- **`primal.list`** added to `capability_registry.toml` as routed capability
  (biomeOS-served). Corresponding entry in `ROUTED_CAPABILITIES`. Ready for
  biomeOS Wave 20 rollout.
- **`commit_provenance()`** in `dag_provenance.rs` — `nest.commit` signal
  dispatch (Wave 20). Dispatches `nest.commit` signal which biomeOS decomposes
  into event.append → crypto.sign → content.put → session.commit → braid.create.
  Falls back to direct `ledger.record` + `attribution.braid` multi-call for
  pre-v3.57 biomeOS.
- **`s_schema_standard` scenario** — validates Wave 20 canonical response shapes:
  capability.list envelope (capabilities array, count, primal), signal registry
  presence (adopted + candidate), niche identity constants.
- **`nest.commit` promoted** from signal candidate to adopted in
  `capability_registry.toml` `[signals]`.

### Metrics
- 596 (default) / 1,045 (barracuda-local) lib tests pass
- Zero clippy warnings
- Zero format drift
- 18 registered scenarios (default) / 21 (barracuda-local)

## Unreleased — Post-BootPipeline Documentation + Cross-Team Handoff (May 16, 2026)

### Changed (documentation)
- **Test counts updated**: 591→606 (cylinder), experiment count 196→198 across
  README, EXPERIMENT_INDEX, experiments/README, whitePaper/baseCamp.
- **Science ladder**: Added Sovereign Init RPC (Exp 197) + Vendor-Agnostic
  BootPipeline (Exp 198) to README progression.
- **sovereign_gpu_compute.md**: Added BootPipeline section with trait signature,
  DeviceTopology vocabulary, VBIOS interpreter fix details, updated next steps.
- **nucleus_composition_evolution.md**: Updated test counts, added BootPipeline
  validation status.

### Added (handoffs)
- **`HOTSPRING_BOOT_PIPELINE_VBIOS_HANDOFF_MAY16C_2026.md`**: BootPipeline trait,
  VBIOS fixes, DeviceTopology, VegaInit stub — upstream asks for toadStool and
  coralReef teams.
- **`HOTSPRING_PRIMALS_SPRINGS_EVOLUTION_HANDOFF_MAY16D_2026.md`**: Cross-team
  handoff covering primal use/evolution review, NUCLEUS composition patterns,
  neuralAPI signal adoption, sovereign boot insights, atomic instantiation
  patterns, and upstream asks for all primal and spring teams.

### Remaining Sovereign Boot Issues
- **Titan V warm**: GspBridge stub halts at stage 4. Needs real FECS state
  capture or coralReef IPC bridge.
- **K80 cold**: VBIOS interpreter correct, BAR access blocked (vfio-pci binding
  requires iommufd FDs, not sysfs resource0).
- **K80 warm**: PGRAPH clock-gated after driver teardown — needs ungating.

## Sovereign Boot: Hardware-Agnostic BootPipeline + VBIOS Interpreter Fixes (Exp 198, May 16, 2026)

### Added (toadStool cylinder — `hardware.rs`)
- **`BootPipeline` trait**: Vendor-agnostic boot pipeline trait using `&dyn RegisterAccess`.
  Associated types `ProbeResult`/`InitResult` allow vendor-specific detail while the
  trait signature is universal: `probe → is_warm → devinit → engine_init → verify`.
- **`DeviceTopology` + `DeviceFunction`**: Replace NVIDIA-specific `DeviceInit`/`DieInfo`
  with vocabulary that works for AMD chiplets, Intel tiles, FPGAs. `single()`, `dual()`,
  `with_firmware()` builders.
- **`BootProbeInfo`, `BootInitInfo`, `DeviceBootResult`, `FunctionBootResult`**:
  Cross-vendor summary types bridging vendor-specific results to universal consumers.

### Added (toadStool cylinder — `init_kepler.rs`, `init_volta.rs`)
- **`BootPipeline` impl for `KeplerInit` + `VoltaInit`**: NVIDIA pipelines now implement
  both `InitPipeline` (NV-specific, `&MappedBar`) and `BootPipeline` (vendor-agnostic,
  `&dyn RegisterAccess`). Warm path fully functional; cold path returns `Unsupported`
  via `BootPipeline` (requires fork-isolated MMIO via `InitPipeline`).

### Added (toadStool cylinder — `amd_metal.rs`)
- **`VegaInit` BootPipeline stub**: AMD Vega 20 (MI50) `BootPipeline` implementation
  using GRBM_STATUS/SRBM_STATUS register map. Probe detects warm/cold, verify checks
  engine idle bits. Proves the trait works cross-vendor without AMD hardware.
  8 new tests with `FakeBar` mock.

### Fixed (toadStool cylinder — VBIOS interpreter)
- **Opcode `0x50` (`INIT_IO_RESTRICT_PROG`)**: Wrong format `4 + count*2` corrected to
  `11 + count*4` per nouveau `init_io_restrict_prog`. Root cause of K80 Script 1 going
  off-script and hitting "too many unknown opcodes" error.
- **Opcode `0x88` (`INIT_RAM_RESTRICT_ZM_REG_GROUP`)**: Added missing opcode for
  ram-restrict register group writes needed by GDDR5 memory training tables.
- **`ram_restrict_group_count()`**: Was reading from raw BIT M data offset `M+2`;
  corrected to dereference the M table pointer and read `snr` field from rammap table
  header at offset +4.
- **Opcode `0x70` (`INIT_EON`)**: Added end-of-nested-condition complement.

### Validated (Experiment 198)
- **Titan V (GV100, warm)**: `sovereign.init` re-confirmed: `compute_ready=true`,
  `all_ok=true`, 6 stages (3 ok + 3 skipped), 101ms total. PTIMER alive, VRAM ok,
  FECS warm-preserved.
- **toadStool cylinder**: 591 → 606 tests (15 new: 7 hardware, 8 AMD stub).
- **VBIOS interpreter**: K80 Script 1 opcode stream now parses correctly through
  INIT_IO_RESTRICT_PROG blocks. K80 bound to vfio-pci (BARs unassigned) — hardware
  validation pending proper VFIO device open.

## Sovereign Init RPC — Warm/Cold Cross-Hardware (Exp 197, May 16, 2026)

### Added (toadStool server — `sovereign.rs`)
- **`sovereign.init` JSON-RPC method**: First direct diesel engine invocation over IPC.
  Opens BAR0 via sysfs, runs the staged `sovereign_init` pipeline, returns per-stage
  results with timing. Params: `bdf` (required), `halt_before`, `skip_gr_init`,
  `golden_state_path`, `vbios_rom_path`, `sm_version`, `fbpa_count`.

### Added (toadStool cylinder — `mapped_bar.rs`)
- **`MappedBar::from_sysfs_rw(bdf, size)`**: Opens BAR0 via sysfs `resource0` with
  read-write mmap. Enables sovereign init without a full VFIO device open (iommufd FDs
  not needed for BAR0-only stages).

### Validated (Experiment 197)
- **Titan V (GV100, warm)**: BAR0 probe OK (12ms), PMC enable OK (75ms),
  memory training skipped (warm detected), falcon boot halted at StubGspBridge.
  Pipeline total: 88ms. Stages 1-3 proven sovereign.
- **K80 GPU0 + GPU1 (GK210, cold VFIO)**: BAR0 probe OK (12ms), PMC enable OK (75ms),
  memory training failed ("PRAMIN dead" — GDDR5 DEVINIT replay needed). 206-208ms total.
- **Next steps**: K80 VBIOS ROM extraction + DEVINIT replay. Titan V: real GspBridge
  (coralReef IPC or warm-handoff FECS state).

## Unreleased — Documentation Evolution + Upstream Handoff (May 16, 2026)

### Changed (documentation normalization)
- **Test/binary/suite counts**: All living docs updated 592→595 (default),
  64→65 (validation suites), 166→167 (binaries).
- **`hotspring_primal` → `hotspring_unibin`**: Living docs normalized.
  Historical entries in PRIMAL_GAPS.md and CHANGELOG.md kept as fossil record.
- **experiments/README.md**: Experiments 192–196 added to Active table.
- **wateringHole archival**: 7 May 12 handoffs moved to `archive/`.
  Titan V DMATRF May 7 status corrected to `upstream`.
- **scripts/README.md**: Archived script paths corrected `lab/` → `archive/`.
- **PRIMAL_GAPS.md**: Stale handoff filename and test count fixed.
- **EXPERIMENT_INDEX.md**: Stray `||` typo on Exp 194 corrected.
- **notebook**: `01-composition-validation.ipynb` counts and deploy node updated.

### Added
- **Upstream handoff**: `HOTSPRING_DOC_EVOLUTION_UPSTREAM_HANDOFF_MAY16_2026.md`
  — composition patterns, signal adoption summary, deep debt status, upstream
  asks for primalSpring + sibling springs.

## Unreleased — Wave 17 Signal Adoption + Deep Debt Refactoring (May 16, 2026)

### Added (Neural API signals)
- **`primal.announce`** registration in `niche.rs` — single atomic call replaces
  the legacy `lifecycle.register` + N × `capability.register` + `method.register`
  pattern. Automatic fallback to legacy multi-call for older biomeOS.
- **`dispatch_node_compute()`** in `compute_dispatch.rs` — dispatches GPU workloads
  via the `node.compute` signal. biomeOS decomposes compile → submit → execute
  through the graph. Falls back to `compile_and_submit()` for older biomeOS.
- **`publish_result()`** in `compute_dispatch.rs` — publishes signed results via
  `tower.publish` signal (sign → announce → audit). Falls back to direct
  `crypto.sign_ed25519` + `discovery.announce` for older biomeOS.
- **Signal tier annotations** in `capability_registry.toml` — declares `node.compute`
  and `tower.publish` as adopted, `nest.store` and `nest.commit` as candidates.

### Refactored (deep debt — large file evolution)
- **`niche.rs` (932L) → `niche/mod.rs` (516L) + `niche/tables.rs` (394L)**:
  Static capability tables, dependency declarations, semantic mappings, and cost
  estimates extracted to `tables.rs`. Runtime logic (socket resolution, registration,
  standalone mode) remains in `mod.rs`. Registration logic is now visible without
  scrolling past 400 lines of data tables.
- **`compute_dispatch.rs` (926L) → `compute_dispatch/mod.rs` (592L) + `compute_dispatch/fused.rs` (335L)**:
  `FusedPipeline` and its 5 types (`FusedOp`, `FusedOpSubmitOutcome`, `FusedSubmitReport`,
  `FusedResult`, `FusedOpResult`) extracted to `fused.rs` with their own tests. Dispatch
  validation and signal APIs remain in `mod.rs`.

### Deep Debt Audit Results
- **Zero TODO/FIXME/HACK** markers in codebase
- **10 `unsafe` sites** — all necessary (BAR0 MMIO mmap/volatile in `low_level/bar0.rs` + 1 CUDA FFI)
- **Zero mock leakage** into production code (`NpuSimulator` is intentional)
- **19 files >800L** — 2 refactored this sprint, remainder are experiment binaries (biomeGate)
- **14 external deps** — all pure Rust except `cudarc` (CUDA FFI) and `wgpu` (GPU drivers)

### Metrics
- 595 lib tests pass, zero clippy warnings, zero format drift
- niche.rs: 932 → 516 + 394 (table extraction, not just splitting)
- compute_dispatch.rs: 926 → 592 + 335 (FusedPipeline extraction)

## Diesel Engine Capability Abstraction (May 16, 2026)

### Changed (toadStool ember — `plx_keepalive.rs`)
- **`PlxKeepalive` → `PcieBridgeKeepalive`**: Generalized naming. PLX discovery retained
  as a priority hint, not the identity of the subsystem. `PlxKeepalive` preserved as a
  type alias for backward compatibility.
- **`detect_pcie_bridges()`**: Returns all upstream bridge BDFs with PLX bridges first.
  `detect_plx_bridge()` is now a thin wrapper.
- **`PLX_VENDOR_ID`**: Consolidated as a shared `pub const` — server no longer has its own copy.

### Changed (toadStool glowplug — `plx.rs`)
- **`PlxGuardian` → `BridgeGuardian`**: Generalized to protect any PCIe-bridged device.
  `PlxGuardian` preserved as type alias. `scan_and_protect()` now uses
  `detect_pcie_bridges()` with PLX hint logging.

### Added (toadStool cylinder — `gsp_bridge.rs`)
- **`GspBridge` capability queries**: `supports_acr()`, `supports_pgob()`, `supports_pmu()`,
  `supports_gr_init()` — all default to `false`. Enables callers to introspect bridge
  capabilities instead of matching on `BootStrategy` externally.
- **`GspBridge::pmu_boot()`**: Default method for PMU falcon bootstrap (returns `Unsupported`).

### Added (toadStool cylinder — `sovereign_stages.rs`)
- **`MemoryTrainingStrategy`**: Enum dispatch keyed by `MemoryType`. Maps GDDR5 → DEVINIT,
  HBM2 → typestate controller, GDDR6/GDDR6X/GDDR7/HBM3 → explicit `Unsupported`.
- **`dispatch_memory_training()`**: Centralizes warm-detection + training execution. The
  100+ lines of if/else in `sovereign_init` collapsed to a single `match`.

### Changed (toadStool cylinder — `sovereign_init.rs`)
- **`engine_ungate()`**: Generalized from Kepler-only `kepler_pgraph_ungate` to support
  arbitrary engines. Takes `engine_name` and optional `status_reg` for post-replay
  validation. Stage names are now `engine_ungate:PGRAPH`, `engine_ungate:CE`, etc.
- **`SovereignInitOptions::engine_init_sequences`**: Vec of (name, sequence, status_reg)
  tuples for per-engine ungating. Legacy `kepler_gr_init` preserved as fallback.

### Added (toadStool cylinder — `pmu_init.rs`)
- **`PmuBootstrap::for_chip(ChipFamily)`**: Parametric constructor — makes PMU bootstrap
  available beyond Kepler.

### Added (toadStool glowplug — `warm_init.rs`)
- **`DriverLabExecutor`**: Callback-driven executor for `DriverLabPlan`. Orchestrates
  power cycle → swap → settle → capture → persist → pairwise diff. Composable with
  any swap mechanism (bare-metal, VM, manual).
- **`LabExecutionResult`**, **`TrialExecutionResult`**, **`DiffSummary`**: Structured
  result types for lab execution reporting.

### Validated
- **6,989 tests pass**, 0 failures, 0 new warnings
- Deployed to `toadstool-ember.service` — PCIe bridge keepalive confirmed:
  3 PLX bridges discovered, 2 K80 GPUs protected, 8 hierarchies pinned
- All 3 GPUs alive: K80 (D0, vfio-pci), Titan V (D0, vfio-pci), RTX 5060 (nvidia)

## PLX Keepalive Boot-Catch + Event-Driven Evolution (May 16, 2026)

### Fixed (toadStool server — `pcie_keepalive.rs`)
- **PCI class code extraction bug**: `(class >> 8) & 0xFFFFFF` produced 24-bit values
  compared against 16-bit constants (`0x0604`, `0x0300`, `0x0302`) — bridge and GPU
  class checks **never matched**. Fixed to `(class >> 16) & 0xFFFF` via new
  `pci_base_subclass()` helper with named constants (`PCI_CLASS_BRIDGE_PCI`,
  `PCI_CLASS_VGA`, `PCI_CLASS_3D`).
- **PLX bridge discovery at boot**: Class scan now correctly discovers PLX PEX 8747
  bridges (vendor `0x10b5`, class `0x0604`). Three-phase discovery: class scan →
  GPU ancestry walk (handles dead config space) → retry with delay (3 attempts, 1s each).
- **`pci0000:40` root complex inclusion**: Ancestry walk now uses `is_pci_bdf()` (checks
  for both `:` and `.`) to reject PCI domain roots.

### Added (toadStool ember — `plx_keepalive.rs`)
- **`ActivityTracker`**: Shared `Arc<AtomicU64>` for PCIe activity-aware backpressure.
  Keepalive skips synthetic heartbeats when real PCIe traffic was recent. Uses
  `epoch_ms()` from `observation.rs` instead of inline `SystemTime` boilerplate.
- **`is_pci_bdf(name)`**: Shared BDF format validator — rejects PCI domain roots
  like `pci0000:40` that contain a colon but no dot.
- **`PlxKeepalive::with_activity_tracker()`**: Builder method to attach an activity
  tracker for backpressure.

### Changed
- **`tokio::time::sleep` → `tokio::time::interval`**: Both server `pcie_keepalive` and
  ember `PlxKeepalive` now use `interval` with `MissedTickBehavior::Skip`. First
  heartbeat fires immediately at t=0 (no initial delay).
- **Server activity tracking**: Replaced static `LAST_PCIE_ACTIVITY_MS` + `record_pcie_activity()`
  with ember's `ActivityTracker` via `OnceLock<ActivityTracker>`. Single implementation,
  no duplicated timestamp logic.
- **Dead-bridge recovery**: Ancestry walk adds parent bridges with dead config space (`0xFFFF`)
  as keepalive targets and pins their power — handles the case where the PLX fabric is
  already in D3cold when the service starts.

### Metrics
- 17 new tests across `pcie_keepalive.rs` and `plx_keepalive.rs` — all pass
- Full workspace: `cargo check` clean, `cargo test` 0 failures
- PLX keepalive **validated at boot** — 3 PLX bridges discovered, 2 K80 GPUs
  downstream, 8 hierarchy nodes pinned, first heartbeat at t=0

## Unreleased — Diesel Engine Driver Sketch + PRI Refactor (May 16, 2026)

### Added (toadStool cylinder)
- **`nv::pri` module**: Single source of truth for PRI fault detection (`is_pri_fault`,
  `is_error_or_zero`, `domain_for_offset`). Eliminates 4 duplicate implementations
  across `gr_init`, `driver_probe`, `pmu_init`, `warm_capture`, and `sovereign_init`.
- **`GrInitSequence::apply(bar0)`**: Replays captured init sequence onto hardware
  via BAR0 MMIO writes. Handles read-modify-write for masked registers.
- **`GrInitSequence::validate(bar0)`**: Reads back registers and returns mismatches
  against expected values for post-replay verification.
- **`GrInitSequence`**: Ordered register write list built from BAR0 cold/warm diffs.
  Serialize/deserialize via JSON for cross-session capture and replay. Domain
  filtering, merge, and summary methods.
- **`WarmStateCapture`**: Automated cold/warm snapshot pipeline — captures two
  `Bar0Snapshot`s, computes `Bar0Diff`, derives `GrInitSequence`. Bridges
  glowplug orchestration with cylinder hardware capture.
- **`DriverProbe` + `TrialResult`**: Multi-driver comparison tool. `FalconState`
  enum (NotStarted/Halted/Running/HsLocked/PriGated) probes falcons at arbitrary
  BAR0 bases. `DriverProbe` tracks trials across drivers with analysis methods
  (`best_by_engines`, `pgraph_alive_trials`, `fecs_uploadable_trials`).
- **`PmuBootstrap`**: Kepler PMU falcon bootstrap — reset, IMEM/DMEM upload via
  PIO, start with mailbox handshake, PFIFO writability test. `PmuSnapshot`
  captures PMU register state with `falcon_state()` bridge to shared `FalconState`.
- **Kepler PGRAPH ungating stage**: `sovereign_init.rs` Stage 3b replays
  `GrInitSequence` to ungate PGRAPH before falcon boot on NoAcr GPUs. Now
  delegates to `GrInitSequence::apply()` instead of inline write logic.

### Changed
- **`ChipFamily::from_sm`** now delegates to `profile_for_sm()` — eliminates
  SM→family range drift between `ChipFamily` and `GenerationProfile`.
- **`kepler_pgraph_ungate`** refactored from 35-line inline write loop to 10-line
  delegation to `GrInitSequence::apply()` + shared `is_pri_fault()`.
- **`PmuSnapshot` gains `falcon_state()`** bridge method — reuses `FalconState`
  from `driver_probe` instead of independent boolean `is_running`.

### Metrics
- 40 new tests across `nv::pri`, `nv::gr_init`, `nv::driver_probe`, `nv::pmu_init`,
  `vfio::warm_capture` — all pass
- Full workspace: 585+ tests pass, zero lint errors, zero new warnings
- Zero code duplication for PRI fault detection (was 4 copies)

## Unreleased — Upstream Absorption + Deep Debt Sprint (May 14, 2026)

### Fixed
- **Clippy zero warnings restored** across `--all-targets --features barracuda-local`:
  `div_ceil`, `needless_borrow`, `map_unwrap_or`, `unnecessary_ne`, `unnested_or_patterns`,
  `redundant_closure`, `needless_range_loop`, `unnecessary_raw_string_hashes`,
  `variables_in_format_string`, `multiplication_by_neg_one`, `unused_import`.
- **Test module lint expectations** updated: 5 `#[cfg(test)]` modules gained proper
  `#![expect(clippy::expect_used)]` / `#![expect(clippy::unwrap_used)]`.
  3 unfulfilled expectations removed (`hfb/tests.rs`, `hfb_deformed/potentials.rs`).
- **Integration test `#[allow]` → `#[expect]`**: 4 test crates migrated to Rust 2024 idiom.
- **`validate_streaming_pipeline.rs`**: `needless_pass_by_ref_mut` expectation restored,
  unused `unwrap_used` expectation in `integration_prescreen` removed.

### Changed
- **plasmidBin composition aligned with atomic model:** `[niches.hotspring]` in
  `manifest.toml` and `ports.env` now include `skunkbat` per Tower = bearDog + songBird
  + skunkBat definition. `COMP_TOWER` also updated.

### Metrics
- 595/595 lib tests pass (default features)
- 65 validation suites in 3 tiers (35 smoke / 7 nucleus / 23 silicon)
- 0 clippy warnings (`--all-targets --features barracuda-local`)
- 0 unfulfilled lint expectations

## Unreleased — plasmidBin Local Debt Resolution + Full Deployment (May 14, 2026)

### Added
- **Release cascade in `fetch.sh`:** plasmidBin cascades through 5 recent releases when
  a binary is missing from `latest`, solving the incremental-release ecoBin harvest lag.
- **Generalized `upgrade-primal.sh`:** Unified upgrade script replacing `upgrade-toadstool.sh`.
  Supports `--all`, `--trio`, `--status`, `--check`, `--force` with automatic rollback.
- **User-mode systemd services:** `barracuda-user.service` and `coralreef-user.service`.
- **Full NUCLEUS deployment:** 13/13 primals deployed to `/usr/local/bin/` from plasmidBin ecoBins.

### Fixed
- **`doctor.sh` symlink detection:** `file` → `file -L`, `du -h` → `du -hL` to correctly
  identify `static-pie` ecoBins behind backward-compat symlinks (was reporting DYNAMIC/0).

### Metrics
- 595/595 lib tests pass (default features)
- 65 validation suites in 3 tiers (35 smoke / 7 nucleus / 23 silicon)
- 0 clippy warnings
- 13/13 NUCLEUS primals deployed, 3/3 compute trio IPC live

---

## Unreleased — Local Debt Resolution + Composition Evolution (May 14, 2026)

### Added
- **Compile-then-dispatch pipeline:** `compile_and_submit()` in `compute_dispatch.rs` chains
  coralReef `shader.compile.wgsl` → toadStool `compute.dispatch.submit` with compiled binary.
  `submit_binary()` for pre-compiled payloads.
- **Circuit-breaker discovery:** `PrimalEndpoint` tracks `fail_count`/`dead_since`.
  `NucleusContext` gains `record_failure()`/`record_success()`/`maybe_reprobe()`/`refresh()`.
  `call_tracked()` applies circuit-breaker logic (3 failures = mark dead, 30s cooldown).
- **`parse_jsonrpc_response()` helper:** Centralized JSON-RPC envelope parsing with typed errors.
- **`FusedSubmitReport`/`FusedOpSubmitOutcome`:** Typed error handling for batched submissions.
- **TOML-loaded primal aliases:** `[primal_aliases]` in `capability_registry.toml` loaded at runtime.
- **Tiered validation:** `validate_all --tier smoke|nucleus|silicon` (65 suites: 35/7/23).

### Changed
- `fleet_toadstool.rs` `submit()`/`dispatch()` deprecated → use `compute_dispatch.rs`.
- `glowplug_client.rs` docs clarify device-management-only scope.
- `validate_compute_trio_pipeline` uses `compile_and_submit()` for yukawa/plaquette.
- Experiment 190 archived (coral-ember era); 191 is active toadStool-era journal.

### Metrics
- **Tests:** 595 (default) / 1,041 (barracuda-local)
- **Clippy:** zero warnings
- **TODO/FIXME/HACK:** zero
- **Validation suites:** 65 (3 tiers: smoke/nucleus/silicon)

## Unreleased — Deep Debt Resolution + Evolution Sprint (May 13, 2026)

### Changed
- **coralReef socket discovery evolved:** `fleet_client::ember_socket_candidates(bdf)`
  and `fleet_client::glowplug_socket_path()` added with env-var discovery
  (`TOADSTOOL_GLOWPLUG_SOCKET`, `TOADSTOOL_RUN_DIR` fallback chain).
  8 experiment binaries migrated from hardcoded `/run/coralreef/` paths.
- **`gpu_flow.rs`** buffer labels `*_placeholder` → `*_unused` (accurate naming).
- **`silicon_qcd/flow.rs`** removed `_uni: ()` placeholder parameter +
  updated caller in `runner.rs`.
- **`collapsible_str_replace`** fix in `ember_socket_candidates`.

### Metrics
- **Tests:** 592 (default) / 1,041 (barracuda-local)
- **Clippy:** zero warnings (both feature sets)
- **TODO/FIXME/HACK:** zero
- **Production mocks:** zero
- **Library unsafe:** zero (`#![forbid(unsafe_code)]`)
- **C deps (default):** zero

## Unreleased — Niche Convergence → Atomic Deployment (May 13, 2026)

### Fixed
- **bearDog wire name hygiene:** `receipt_signing.rs` corrected from
  `payload`/`encoding` params to canonical `message` (base64) per bearDog
  upstream spec. `validate_nucleus_tower.rs` also updated to base64-encode
  the probe message. Extracted `base64_encode` as shared crate module
  (previously duplicated in `dag_provenance.rs`).
- **5 clippy errors** resolved: `too_long_first_doc_paragraph`
  (`compute_dispatch.rs`), `if_not_else` (`toadstool_report.rs`),
  `bool_to_int_with_if` (`harness.rs`), `manual_let_else`
  (`s_hotqcd_dispatch.rs`), `map_unwrap_or` (`bench/report.rs`),
  `unwrap_used` (`s_gradient_flow.rs`).

### Added
- **`s_node_atomic` validation scenario:** Node atomic (proton) structural
  validation in UniBin registry — domain composition, Tower superset,
  standalone behavior, SEMF/plaquette science baselines. 17 registered
  scenarios (default) / 20 (barracuda-local+sovereign-dispatch).
- **`base64_encode` crate module:** Minimal RFC 4648 encoder extracted from
  `dag_provenance.rs` for reuse across `receipt_signing.rs` and
  `validate_nucleus_tower.rs`.

### Changed
- **Harvest script updated:** `scripts/harvest-ecobin.sh` now builds
  `hotspring_unibin` (the canonical UniBin binary) instead of the removed
  `hotspring_primal`. All `unibin_modes` listed: certify, validate, serve,
  status, version.

### Metrics
- **Tests:** 592 (default) / 1,041 (barracuda-local)
- **Clippy:** zero warnings (both feature sets)
- **Scenarios:** 17 (default) / 20 (barracuda-local+sovereign-dispatch)

## Unreleased — Compute Trio Rewire + Deep Debt Capability Evolution (May 12, 2026)

### Changed (May 12 — IPC Transport Evolution: call_by_capability Proliferation)
- **GAP-HS-092:** All IPC client modules now prefer `call_by_capability(domain,
  method, params)` for unified discovery + transport, falling back to direct
  socket RPC and env-var/socket-dir scan. Affected modules: `biome_status`,
  `method_register`, `skunkbat`, `sweetgrass`, `rhizocrypt`, `loamspine`,
  `fleet_toadstool` (`capabilities`, `submit`), `fleet_client`
  (`discover_diesel_ember_socket` tries NUCLEUS `by_domain("ember")` first).
- **`hardware_calibration.rs`:** `TierCapability::failed()` and
  `TierCapability::compiled_only()` constructors eliminate ~50 lines of
  repeated failed-tier boilerplate.

### Added
- **Compute Trio Rewire Sprint (GAP-HS-087):** `PrecisionTier`/`PhysicsDomain`
  re-exported from upstream barraCuda (15-tier/15-variant canonical enums).
  `toadstool-dispatch` feature flag with `ToadStoolDispatchClient` for Phase C
  ember→toadStool cutover. `HardwareHint` in `PrecisionRoute`. `validate_compute_trio_pipeline`
  binary (Yukawa force + Wilson plaquette end-to-end). Barrier shader validation
  for coralReef `membar.{cta,gl}` emitter.
- **Deep Debt Capability Discovery (GAP-HS-088):** IPC provenance clients
  evolved to NUCLEUS `by_domain()` discovery. `detect_sovereign_available()`
  inverted to capability-first. Barrier validation uses `call_by_capability`.
  `toadstool_report.rs` uses `by_domain("compute")`. Deploy graph order dedup.

### Changed
- **CG_BACKOFF_CAP** (2000) consolidated from 3 local consts in
  `resident_cg.rs`, `resident_shifted_cg.rs`, `true_multishift_cg.rs` to a
  single source of truth in `tolerances/lattice.rs`.
- **Timeout constants extracted:** `EMBER_ADOPT_TIMEOUT` (30 s),
  `EMBER_STATUS_TIMEOUT` (5 s) in `fleet_ember.rs`; `GPU_POLL_INTERVAL`
  (200 ms) in `bench/telemetry.rs`; `TITAN_WARM_RECV_TIMEOUT` (120 s) in
  `single_beta.rs`.
- **`/proc/` paths agnostic:** `bench/hardware.rs` (`PROC_CPUINFO`,
  `PROC_MEMINFO`) and `bench/report.rs` (`PROC_SELF_STATUS`) now accept
  environment variable overrides for cross-platform/CI testing.
- **`BenchReport::save_and_print()`** consolidates repeated
  `discovery::benchmark_results_dir()` + `save_json()` + println pattern.
  `nuclear_eos_gpu.rs` and `sarkas_gpu.rs` simplified (unused `discovery`
  and `PathBuf` imports removed).
- **Deploy graphs:** Fixed duplicate `order = 9` (sweetgrass/skunkbat) in md,
  plasma, nuclear_eos, qcd graphs — skunkbat now `order = 10`.
- **`low_level/bar0.rs`**: BAR0 map size discovered from file metadata; sysfs
  path overridable via `HOTSPRING_SYSFS_PCI`.
- **`fleet_client.rs`**: `Vec<&String>` → `Vec<&str>` with `sort_unstable()`.
- **PCI vendor IDs** extracted to named constants in `register_maps/mod.rs`.

### Added (May 12 — Scenario Expansion + Downstream Audit)
- **Scenario registry expansion** (7 → 9 default / 12 with barracuda-local):
  `screened-coulomb` (Yukawa eigenvalues), `transport-stanton-murillo` (η*/λ*
  fits), `gradient-flow` (Wilson flow on SU(3), barracuda-local),
  `dielectric-mermin` (Mermin static/high-freq limits, barracuda-local).
- **biomeOS IPC capability evolution:** `ipc/biome_status.rs` and
  `ipc/method_register.rs` evolved from hardcoded `biomeos/biomeos.sock` to
  `by_domain("composition")` discovery with `BIOMEOS_SOCKET` env fallback.
  Last hardcoded socket paths in library IPC code eliminated.
- **Downstream repos cloned** (`gardens/projectNUCLEUS`, `gardens/foundation`,
  `gardens/lithoSpore`) and audited for hotSpring integration patterns.
  GAP-HS-090 documents findings in `docs/PRIMAL_GAPS.md`.
- **`validate_all.rs`** tier range comment corrected (58–62 → 58–64).

### Added (May 12 — Tier 2 Live Science API Convergence)
- **`ipc/tier2.rs`:** Tier 2 client wiring for `toadstool.validate` (workload
  pre-flight), `toadstool.list_workloads` (catalog), `precision.route`
  (barraCuda precision advisory). `tier2_status()` + `Tier2Status::check()`
  for harness integration. Degrades gracefully when primals unavailable.
- **`niche.rs`:** 3 new routed capabilities — `toadstool.validate`,
  `toadstool.list_workloads`, `precision.route`.
- **`capability_registry.toml`:** 3 new entries synced with niche.

### Metrics
- **584** lib tests (default) / **1,036** (barracuda-local + toadstool-dispatch) — zero clippy warnings
- **190** experiments | **166** binaries | **64/64** validation suites
- **9** validation scenarios (default) / **12** (barracuda-local)

## Unreleased — LTEE B2 Complete + Exp 190 Reconciliation (May 11, 2026)

### Added
- **LTEE B2 Tier 2 Rust scenario** (`s_ltee_anderson.rs`): Self-contained
  validation scenario implementing Wiser et al. 2013 power-law fitness model,
  Anderson-like Hamiltonian from fitness increments, QL tridiagonal
  eigensolver, and level spacing ratio diagnostics. 18 validation checks
  covering power-law fidelity, diminishing returns, GOE/Poisson bounds,
  sliding-window localization, and 12-population variance. Available in
  default (IPC-first) build — no `barracuda-local` dependency. B2 marked
  COMPLETE in PAPER_REVIEW_QUEUE.
- **Exp 190 in EXPERIMENT_INDEX**: Three-GPU sovereign validation now indexed
  (was in `experiments/` but missing from index). RTX 5060 12/12, Titan V
  warm-catch, K80 warm-catch all documented.
- **Titan V / K80 benchScale needs**: Documented in Exp 190 — nvidia-470 VM
  images, QEMU passthrough configs, multi-GPU coexistence script, firmware
  archive requirements, coralReef SM rebuild dependency.
- **Expected values JSON**: `experiments/results/ltee/ltee_b2_anderson_expected.json`
  for lithoSpore module 7 absorption.

### Verification
- `cargo test --lib` — 579 passed, 0 failed (IPC-first default; +3 LTEE)
- `cargo test --lib --features barracuda-local` — 1,028 passed, 0 failed, 6 ignored (+3 LTEE)
- `cargo clippy --lib` — zero warnings (default)
- `cargo fmt --check` — clean
- 8 registered validation scenarios total (was 7)

## Unreleased — Sovereign Rust Evolution (May 11, 2026)

### Added
- **Pure Rust ELF patcher** (`coral-driver/src/tools/elf_patcher.rs`):
  Replaces `patch_nouveau_teardown.py`. Uses the `object` crate for ELF
  parsing — zero subprocess calls. `KmodPatcher::default_nouveau_targets()`
  patches 4 teardown functions. Vendor-agnostic via `PatchTarget` struct.
- **Standalone warm probe** (`coral-driver/src/vfio/warm_probe.rs`):
  `WarmStateSnapshot` struct (PMC, PRAMIN, FECS, GPC) extracted from
  `sovereign_stages.rs` as reusable public API.
- **Warm-catch orchestrator** (`coral-ember/src/ipc/handlers_warm_catch.rs`):
  `ember.warm_catch` JSON-RPC handler — full pipeline: patch → swap nouveau
  → settle → swap vfio → probe. Era-aware settle durations from `MemoryType`
  (GDDR5=10s, HBM2=12s, GDDR6=8s). Supports `--dry-run` and `--settle`.
- **`coralctl warm-catch <BDF>`**: New CLI subcommand wired through glowplug →
  ember RPC. Replaces `k80_warm_catch.sh` and `titanv_warm_handoff.sh`.
- **Sovereign auto-warm pre-check**: `warm_catch_pre_check()` in
  `sovereign_init.rs` detects cold GPU + available warm-catch infrastructure.
  `handlers_sovereign.rs` logs opportunity before `sovereign_init`.

### Changed
- **Jelly strings archived**: `k80_warm_catch.sh`, `titanv_warm_handoff.sh`,
  `patch_nouveau_teardown.py`, `bpf_warm_catch_guard.py` moved to
  `scripts/archive/`. All warm-catch functionality now in pure Rust.
- **scripts/README.md**: Updated archive table with pure Rust replacements.
## Unreleased — Sovereign Warm-Catch Breakthrough (May 11, 2026)

### Added
- **Binary-patched nouveau warm-catch**: `patch_nouveau_teardown.py` patches
  4 teardown functions (gf100_gr_fini, nvkm_pmu_fini, nvkm_mc_disable,
  nvkm_fifo_fini) in stock `nouveau.ko` to NOP. Replaces broken livepatch
  approach (kernel 6.17 R_X86_64_64 relocation check blocks out-of-tree
  livepatch/kprobe modules entirely).
- **K80 warm-catch script** (`k80_warm_catch.sh`): Full warm-catch
  orchestration using patched nouveau. GDDR5 trained, GPCs active, PMC
  preserved across unbind.
- **Titan V warm-handoff script** (`titanv_warm_handoff.sh`): nouveau
  initializes FECS via ACR/SEC2 natively. FECS RUNNING post-VFIO rebind.

### Fixed
- **GAP-HS-073 (Titan V FECS)**: RESOLVED — patched nouveau warm-handoff
  brings FECS up via ACR/SEC2 firmware (PMU absence is non-fatal). FECS_MC
  = 0x0c060006 (running), PGRAPH enabled, 1 GPC active.
- **GAP-HS-076 (K80 GPCs)**: RESOLVED — patched nouveau trains GDDR5
  (12288 MiB), initializes GPCs. PMC_ENABLE = 0xfc37b1ef (pop=22), FECS_MC
  = 0x00060005 (running), GPC_MASK = 0x10, WARM=True.
- **Livepatch kernel 6.17 incompatibility**: Bypassed entirely. Binary
  patching stock .ko avoids module loader relocation checks.

## Unreleased — Three-GPU Sovereign Validation (May 11, 2026)

### Added
- **Exp 190**: Three-GPU sovereign validation sprint — post-power-cycle
  validation across RTX 5060, Titan V, K80. RTX 5060: 12/12 sovereign
  roundtrip PASS, 154.2 steps/s Yukawa OCP MD. Titan V: warm detected
  (HBM2 from BIOS POST), FECS blocked (Falcon v5 HS mode). K80: PLX
  alive (rev ca → first time since D3cold), PMC enabled (0xfc37b1ef),
  GPCs power-gated (cold GDDR5).

### Fixed
- **k80-wake-and-run.sh**: `tomllib` → `tomli` fallback for Python 3.10
  compatibility under sudo.

## Unreleased — Deep Debt Evolution + Infra Handoff (May 11, 2026)

### Changed
- **Deprecated primal accessors removed**: `toadstool()`, `beardog()`,
  `rhizocrypt()`, `loamspine()`, `sweetgrass()`, `coralreef()` convenience
  methods deleted from `primal_bridge.rs` — zero callers remained. All primal
  resolution now routes through `by_domain()` capability-based discovery.

### Added
- **Ecosystem handoff**: `wateringHole/handoffs/INFRA_MATURITY_ECOSYSTEM_HANDOFF_MAY11_2026.md`
  — comprehensive handoff for primals/springs teams covering benchScale +
  agentReagents maturity, NUCLEUS composition patterns, hardware interaction
  lessons, Neural API integration, and gaps for upstream audit.

### Infra (benchScale + agentReagents, pushed separately)
- **benchScale**: LibvirtConfig→BenchScaleConfig migration complete, `cp`→`std::fs::copy`,
  SSH interface discovery via russh, boot diagnostics async + configurable users,
  DHCP FFI consolidated, `VfioPassthrough` with QEMU commandline injection.
- **agentReagents**: `InstallingCosmic`→`InstallingDesktop`, verification.rs
  smart refactor (1061→3 modules), cosmic-specific strings removed, desktop
  verification now distro-agnostic.

## Unreleased — Interstadial Sprint (May 11, 2026)

### Changed
- **Tier 4 IPC-first defaults**: `default = ["barracuda-local"]` → `default = []`.
  Bare `cargo build` now produces IPC-only binary. Local compute opt-in via
  `--features barracuda-local`. 576 tests pass in default mode, 1,025 with
  barracuda-local. hotSpring now qualifies for the Pillar 5 Tier 4 exit gate.
- **metalForge/forge decoupled**: barracuda dependency made optional behind
  `barracuda-local` feature. `bridge` module conditionally compiled. Forge
  builds and clips clean with `default = []`.
- **Clippy warnings resolved**: `suspicious_arithmetic_impl` in `complex_f64.rs`
  and `unnecessary_wraps` in `nuclear_matter.rs` fixed (zero warnings in
  both default and barracuda-local builds).

### Added
- **LTEE B2 (Exp 189)**: Tier 1 Python baseline for Wiser et al. 2013 —
  Anderson disorder analogy for fitness dynamics. Notebook at
  `notebooks/papers/13-ltee-anderson-fitness.ipynb`. Power-law fitness model,
  fitness-increment Anderson Hamiltonian, sliding-window ⟨r⟩ localization
  analysis, 12-population variance study. Expected values JSON for lithoSpore
  module 7 (anderson). B2 marked STARTED in PAPER_REVIEW_QUEUE.

### Verification
- `cargo clippy --lib` — zero warnings (default)
- `cargo clippy --lib --features barracuda-local` — zero warnings
- `cargo test --lib` — 576 passed, 0 failed (IPC-first default)
- `cargo test --lib --features barracuda-local` — 1,025 passed, 0 failed, 6 ignored
- `cargo check` metalForge/forge — clean (both default and barracuda-local)

## Unreleased — Post-Interstadial Push 3: Deep Debt (May 11, 2026)

### Added
- **`sarkas-yukawa-md` validation scenario** (`s_sarkas_yukawa_md.rs`):
  Foundation-grade scenario with Daligault D* fit across 12 reference points,
  RMSE validation, and CPU MD simulation with energy drift checks (when
  `barracuda-local` enabled). 7 registered scenarios total.
- **Foundation Thread 2 workload**: `workloads/thread02_plasma/hs-sarkas-md.toml`
  enables `foundation_validate.sh --thread plasma` execution path.

### Changed
- **NUCLEUS workload fix**: `hotspring-md-validation.toml` scenario ID corrected
  from `sarkas_yukawa_md` to `sarkas-yukawa-md` (was broken — no matching
  scenario existed).
- **Fleet discovery evolution**: `discover_diesel_ember_socket()` now uses
  `coralreef_run_dir()` cascade (`$CORALREEF_RUN_DIR` → `$XDG_RUNTIME_DIR/coralreef`
  → `/run/coralreef`) instead of hardcoded path.
- **Foundation targets metadata**: Fixed `thread02_plasma_targets.toml`
  `[meta].expression` to reference `PLASMA_QCD_SOVEREIGN_GPU.md`.
- **DOWNSTREAM_PATTERNS.md**: Refreshed all stale items, added scenario registry
  listing with tier requirements.

### Metrics
- Tests: 576 (default / IPC-first) / 1,025 (barracuda-local) — both configurations clean
- Clippy: zero warnings
- Scenarios: 7 registered (6 default + 1 barracuda-local)
- Gaps: 4 new resolved (GAP-HS-082 through GAP-HS-085)

## Post-Interstadial Push 2 (May 11, 2026)

### Changed
- **primal-proof test coverage**: Moved barracuda-dependent tests in
  `error.rs` behind `#[cfg(feature = "barracuda-local")]` and gated
  `head_group_layout_matches_toadstool_head_group` in `reservoir/tests.rs`.
  576 tests now pass in primal-proof mode (up from build-only validation).
- **Deploy graphs 7/7 skunkBat**: Added skunkBat node to `spectral`,
  `plasma_md`, and `sovereign_gpu` deploy graphs (was 4/7). All 7 deploy
  graphs now include defense/audit node for JH-5 readiness.
- **plasmidBin cell graph**: Added skunkBat node to `hotspring_cell.toml`
  with `security.audit_log` capabilities.
- **GAP ID collision fix**: Renumbered 4 duplicate gap IDs (GAP-HS-030/032/
  033/057) to GAP-HS-073 through GAP-HS-076. Zero duplicates remain.
- **plasmidBin manifest**: Fixed stale test count (1040→1025), added Tier 4
  IPC-first note.

### Metrics
- Tests: 576 (default / IPC-first) / 1,025 (barracuda-local) — both configurations clean
- Clippy: zero warnings
- Deploy graphs: 7 (all with skunkBat)
- Gap IDs: zero duplicates (5 new gaps resolved: GAP-HS-077 through GAP-HS-081)

## Post-Interstadial Evolution (May 11, 2026)

### Added
- **skunkBat IPC module** (`src/ipc/skunkbat.rs`): Rust client for
  `security.audit_log` cursor-based audit event polling via JSON-RPC.
  6 new tests. JH-5 forwarding ready — when Phase 3 ships, audit events
  propagate to rhizoCrypt DAG + sweetGrass braid automatically.
- **Foundation seeding**: Contributed 12 Sarkas Yukawa MD validation targets
  to `sporeGarden/foundation` Thread 2 (Plasma Physics). Energy drift,
  RDF structure, self-diffusion D*, viscosity, and Daligault fit parity —
  all validated and traceable to published papers.
- **Foundation Thread 2 expression doc**: Created
  `PLASMA_QCD_SOVEREIGN_GPU.md` covering validation chain, Sarkas MD,
  lattice QCD, Kokkos/LAMMPS parity, and cross-thread connections.
- **Foundation workloads**: Two workload TOMLs for foundation validation
  pipeline — Sarkas MD (`hs-sarkas-md-validation.toml`) and Chuna papers
  (`hs-chuna-validation.toml`).

### Changed
- **19 dead_code warnings eliminated**: Removed 3 superseded handler files
  (`handlers_inference.rs`, `handlers_screening.rs`, `handlers_steering.rs`)
  that were replaced by `handlers/` subdirectory during prior NPU refactor.
  Zero clippy warnings remaining.
- **Smart refactor `single_beta.rs`** (826L→553L): Extracted 273-line
  measurement loop into `measurement.rs` (423L). NPU reject prediction,
  anomaly detection, sub-model steering, and Polyakov readback are now
  separate functions with structured `MeasurementResult` return.

### Infrastructure
- **UniBin release binary built**: 2.4M stripped binary (`hotspring_unibin`),
  ready for plasmidBin GitHub Releases and NUCLEUS workload dispatch.
- **NUCLEUS workload updated**: `hotspring-md-validation.toml` evolved from
  hardcoded `/home/irongate/` path to gate-agnostic `$SPRINGS_ROOT` + UniBin
  `validate` subcommand.

### Metrics
- Tests: 1,025 (up from 1,019 — skunkBat IPC tests)
- Clippy: zero warnings (down from 19)
- Build configs: default, `primal-proof` (IPC-only), all-targets — all clean

## Sovereign Barrier Resolution + Docs Cleanup (May 11, 2026)

### Added
- **wateringHole handoff**: `HOTSPRING_CORALREEF_SOVEREIGN_BARRIERS_HANDOFF_MAY11_2026.md` — Volta ACR skip, HBM2 warm-handoff proof, benchScale VM isolation path, K80 PCIe diagnosis.

### Changed
- **docs/PRIMAL_GAPS.md**: Updated GAP-HS-030 (GV100 FECS) to Partially Resolved (ACR solver bypass + benchScale path). Deprioritized GAP-HS-047 (PMU extraction). Fixed duplicate gap IDs (030a/030b). Updated audit date.
- **README.md**: Status block updated to 188 experiments, Titan V HBM2 warm-handoff proven, Diesel Engine validated. Sovereign GPU status row updated with current barrier states.
- **wateringHole/README.md**: Added On Disk column, added May 10-11 handoff entries.
- **scripts/lab/titanv_nvidia470_warm_handoff.sh**: Marked DEPRECATED — production path is benchScale VM isolation (host DRM stays uninterrupted).

### Architecture
- **Warm-handoff strategy pivot**: Direct host-side driver swaps (nvidia-470 ↔ nvidia-580) can crash the kernel DRM. Production path uses benchScale + agentReagents to run nvidia-470 inside a VM with Titan V VFIO passthrough, keeping the host display driver completely uninterrupted. Physical card swaps also supported (1-2 cards from most NVIDIA generations available for profiling).

## Deep Debt Evolution Phase 4 (May 10, 2026)

### Changed
- **Typed IPC errors**: `fleet_ember.rs` (24 pub fns), `fleet_client.rs` (5),
  `compute_dispatch.rs` (3), `NucleusContext::call/call_by_capability` (2) all
  evolved from `Result<_, String>` to `Result<_, HotSpringError>`. Remaining
  `Result<_, String>` now confined to binary helpers and hardware-access code.
  `impl From<HotSpringError> for String` enables clean `?` at binary boundaries.
- **Hostname consolidation**: `niche::hostname()` centralized from 3 separate
  `/etc/hostname` reads (`bench/hardware.rs`, `validation/harness.rs`,
  `lattice/measurement/time_host.rs`).
- **`#![forbid(unsafe_code)]` fixed**: `low_level/bar0.rs` removed from `pub mod`
  in `lib.rs` (only used via `#[path]` inclusion from binaries); `forbid` now
  applies correctly library-wide.
- **Smart refactor `chuna_overnight/papers.rs`**: 831L → 490L via extraction
  of `paper_44.rs` (220L, dielectric) and `paper_45.rs` (132L, kinetic-fluid).

### Documentation
- **All docs aligned**: README.md, whitePaper/README.md, whitePaper/baseCamp/README.md,
  EXPERIMENT_INDEX.md, sporeprint/ — unified to canonical 1,025 tests / 155 binaries /
  7 deploy graphs / 188 experiments / guideStone L6 CERTIFIED (numbers as of May 11, 2026).
- **Upstream handoff**: `HOTSPRING_DEEP_DEBT_PHASE4_UPSTREAM_HANDOFF_MAY10_2026.md` —
  patterns for primalSpring, barraCuda, coralReef, toadStool, projectNUCLEUS, foundation.
- **wateringHole/README.md** cleaned: removed stale handoff table, mmiotraces reference.
- **Deprecated scripts archived**: `scripts/warm_handoff.sh` and `scripts/manual_warm_handoff.sh`
  moved to `scripts/archive/`.

### Verified
- `cargo fmt --check` — zero drift
- `cargo clippy --lib` — zero new warnings (19 pre-existing dead_code)
- `cargo test --lib` — 1,019 passed, 0 failed, 6 ignored (at time of phase; now 1,025)
- `primal-proof` build — compiles clean
- Full `cargo check` (all binaries) — clean

## Unreleased — Post-Interstadial Spring Evolution (May 10, 2026)

### Added
- **guideStone L6 certification**: NUCLEUS deployment validation layer —
  deploy graph coverage, biomeOS `composition.status` probing,
  `method.register` dynamic registration, skunkBat audit wiring.
- **`primal-proof` feature flag**: IPC-first Tier 4 rewiring. 25+ modules
  gated behind `#[cfg(feature = "barracuda-local")]`. Library compiles
  without barraCuda for IPC-only builds (`--no-default-features --features primal-proof`).
- **Local `Complex64` fallback**: lattice QCD core compiles without barraCuda
  via self-contained complex arithmetic implementation.
- **`ipc::biome_status`**: biomeOS v3.51 `composition.status` IPC client
  with `CompositionStatus` struct and health validation integration.
- **`ipc::method_register`**: biomeOS v3.51 `method.register` IPC client.
  24 hotSpring physics/compute methods defined for dynamic registration.
- **`ipc::provenance/`**: Per-trio modules — `rhizocrypt.rs` (DAG witnesses),
  `loamspine.rs` (ledger entries), `sweetgrass.rs` (attribution braids).
- **Deploy graphs**: `hotspring_md_deploy.toml` (Yukawa OCP),
  `hotspring_nuclear_eos_deploy.toml` (Skyrme HFB), `hotspring_plasma_deploy.toml`
  (dense plasma). All include skunkBat, provenance trio, and full NUCLEUS.
- **skunkBat** node added to all deploy graphs (defense/audit capability).

### Changed
- `certification/mod.rs`: `MAX_LAYER` 5→6, L6 deployment validation integrated.
- `physics/nuclear_matter.rs`: local bisect fallback when barracuda-local disabled.
- `physics/hfb_common.rs`: local Hermite/factorial fallbacks for IPC-only builds.
- `tolerances/md.rs`: `MD_WORKGROUP_SIZE` compile-time fallback (64) for IPC builds.
- `error.rs`: `HotSpringError::Ipc` variant added; `Barracuda` variant gated behind `barracuda-local`.
- `primal_bridge::send_jsonrpc`: returns `Result<_, HotSpringError>` (was `String`).
- `lib.rs`: `pub mod low_level` registered with `#[cfg(feature = "low-level")]` gate
  (upstream created module but missed the registration).
- Upstream merge resolved: pseudofermion submodule refactor (action/config/dynamics)
  + barracuda-local gates preserved cleanly.

### Verified
- `cargo fmt --check` — zero drift
- `cargo clippy --lib` — zero new warnings (19 pre-existing upstream)
- `cargo test --lib` — 1,019 passed, 0 failed, 6 ignored (at time of phase; now 1,025)
- `primal-proof` build (no barracuda) — compiles clean
- Cross-sync: zero drift against primalSpring canonical 413 (was 403, +10 `game.*`)
## Sovereign Pipeline Hardening + Docs Cleanup (May 10, 2026)

### Added
- **Experiments 182–184**: K80 FECS PIO boot, K80 FECS interrupt boot, K80 GR sovereign (ember-wired). Added to EXPERIMENT_INDEX.md and experiments/README.md.
- **wateringHole/README.md**: Index of handoffs, mmiotraces, and deprecated lab scripts.
- **Upstream handoff**: `HOTSPRING_CORALREEF_SOVEREIGN_KEEPALIVE_HANDOFF_MAY10_2026.md` — documents all coral-ember/glowplug hardening, upstream debt, and composition patterns for NUCLEUS/neuralAPI.

### Changed
- **coral-ember**: BDF validation (`validate_bdf`), keepalive interval clamping (≥250ms), COMMAND register + endpoint device reads in keepalive loop, lock-poison returns JSON-RPC error, `last_error` and `endpoint_alive` in `SwitchHealth`.
- **coral-glowplug**: Config validation on load (BDF format + cross-ref), `query_switch_health` wrapped in `spawn_blocking` (2s timeout), sovereign pre-flight switch check, log level promotion for switch health failures.
- **k80-wake-and-run.sh**: BDFs extracted from `glowplug.toml` via `tomllib` (zero hardcoded addresses), socket-readiness poll loops replace fixed sleeps, DRM isolation rules generated from config.
- **k80-sovereign-wake.service**: Orders after `coral-glowplug.service` (was ember only).
- **coral-ember.service**: Added `StartLimitIntervalSec=300 / StartLimitBurst=3`.
- **install-boot-config.sh**: Disables deprecated `plx-keepalive.service` instead of installing it.
- **post-boot-oracle-capture.sh**: Fixed stale VFIO target BDF (4a→4b).
- **README.md**: Fixed version ref (v0.6.17→v0.6.32), directory tree (added wateringHole, sporeprint, tools, notebooks, scripts/boot), unsafe claims aligned with actual `#![forbid(unsafe_code)]` + low-level bin exceptions, experiment count 181→184, plasmidBin path corrected.

### Deprecated
- `wateringHole/warm_handoff.sh` — marked as legacy ad-hoc lab script, violates coralctl-only policy.
- `scripts/boot/plx-keepalive.sh` — already deprecated (kept as fossil record).
- `fleet_mode` and `standby_pool_size` keys removed from `/etc/coralreef/glowplug.toml`.

## Unreleased — Deep Debt Evolution Phase 3 (May 9, 2026)

### Changed
- **Hardcoded `/tmp` fallbacks evolved**: `primal_bridge.rs` now uses
  `niche::socket_dirs()` multi-path discovery; `toadstool_report.rs`,
  `fleet_client.rs`, `brain_persistence.rs`, `validate_cross_vendor_dispatch.rs`
  all use `std::env::temp_dir()` instead of hardcoded `/tmp`.
- **Smart refactor `rhmc/mod.rs`**: 802L → 363L mod.rs + 215L `rational.rs`
  (partial-fraction math) + 210L `multishift_cg.rs` (solver). Config builders
  deduplicated via `RhmcFermionConfig::from_spectral()` helper.
- **API signatures evolved**: `niche::set_family_id(String)` →
  `set_family_id(impl Into<String>)`; `TelemetryWriter::with_substrate(String)`
  → `with_substrate(impl Into<String>)`.
- **Dependency updates**: `cudarc` 0.19.3→0.19.4, `tokio` 1.50→1.52.3.
- **Production error handling**: `production_dynamical.rs` and `validate_fpeos.rs`
  unwrap() calls replaced with proper error paths and harness reporting.
- **Validation matrix updated**: `cells.rs` now reflects ILDG/Lime and
  autocorrelation as done (previously marked todo).
- **NaN-safe sort**: `gpu_physics_proxy.rs` partial_cmp with Ordering::Equal fallback.

### Verified
- `cargo fmt --check` — zero formatting drift
- `cargo clippy --lib` — zero warnings
- `cargo test --lib` — 1002 tests pass, 0 failures

## Unreleased — Interstadial Eukaryotic Evolution (May 9, 2026)

### Added
- **Eukaryotic UniBin** (`hotspring_unibin`): Single binary with `certify`,
  `validate`, `status`, `version` subcommands. Absorbs guideStone certification
  (L0–L5) and 6 validation scenarios into one binary.
- **`barracuda/src/certification/`** organelle module: `bare.rs` (Properties 1-5),
  `composition_probes.rs` (scalar/vector parity, SEMF E2E, crypto witness, compute
  dispatch). Library API `certification::certify(max_layer)` returns `ValidationResult`.
- **`barracuda/src/validation/scenarios/`** with `ScenarioMeta` registry:
  6 absorbed scenarios across 6 tracks (nuclear-physics, lattice-qcd,
  spectral-theory, molecular-dynamics, composition-parity, domain-science).
  Each carries provenance (original binary name, date).
- **`barracuda/src/ipc/`** consolidated IPC module: unified namespace re-exporting
  from `primal_bridge`, `composition`, `glowplug_client`, `fleet_ember`,
  `fleet_client`, `squirrel_client`, `toadstool_report`, `receipt_signing`.
- **`docs/PRIMAL_PROOF_IPC_MAPPING.md`**: Maps every `barracuda::` library call
  to its JSON-RPC equivalent for IPC-first composition validation.
- **`fossilRecord/experiments_prokaryotic_may2026/`**: Dated snapshot of 8 experiment
  binaries (exp070, exp154–exp158, exp167) with provenance README.

### Changed
- All 9 bare `#[allow(...)]` annotations now include `reason = "..."` — zero bare
  suppressions in active code (metalForge/forge tests, integration tests, dual_dispatch).
- primalSpring dependency updated to path → v0.9.25 (interstadial eukaryotic).
- `clap` added as dependency for UniBin CLI argument parsing.

### Verified
- `cargo fmt --check` — zero formatting drift
- `cargo clippy --lib` — zero warnings
- `cargo test --lib` — 1002 tests pass, 0 failures
- Zero `#[deprecated]` without `note =`
- Zero `#[allow(...)]` without `reason =` in active code
- Zero TODO/FIXME/HACK/DEBT markers in active code

## Unreleased — Deep Debt Evolution Phase 2 (May 8, 2026)

### Added
- **3 integration tests**: `integration_dielectric.rs` (5 tests: Drude,
  Mermin, conductivity, Debye screening), `integration_spectral.rs` (3 tests:
  Lanczos eigenvalues, SpMV, DOS), `integration_lattice.rs` (4 tests: cold/hot
  start, HMC trajectory)
- **9 unit tests**: `primal_bridge.rs` (6 tests: empty context, by_domain,
  get_by_capability, manual endpoint construction), `receipt_signing.rs`
  (3 tests: unavailable, embed passthrough, serde round-trip)
- **2 CPU/GPU parity domains**: Nuclear EOS (GPU SEMF batch vs CPU, 20 nuclei)
  and Spectral (GPU SpMV vs CPU SpMV, Anderson 2D L=8) added to
  `validate_barracuda_cpu_gpu_parity` (now 8 domains total)
- **`docs/DOWNSTREAM_PATTERNS.md`**: Integration audit of `projectNUCLEUS`
  and `foundation` repos (workload TOMLs, deploy graphs, lineage threads)
- **Paper 45 `kinetic_fluid_control.json`**: Frozen reference results
  (18/18 checks) committed to `control/kinetic_fluid/results/`
- **`experiments/results/papers/`**: Frozen parity greenboard JSON snapshot
  (25 papers, all substrates)

### Changed
- **Refactored `pseudofermion/mod.rs`** (926L → ~540L): Hasenbusch
  mass-preconditioning extracted to `hasenbusch.rs` submodule
- **Refactored `npu_worker/handlers.rs`** (839L → handlers/ directory):
  Split into `precompute.rs`, `thermalization.rs`, `inference.rs`, `proxy.rs`
- **Refactored `nuclear_eos_helpers/mod.rs`** (821L → ~440L): Display/print
  functions extracted to `display.rs` submodule
- **`exp070_register_dump.rs`**: Wrapped raw mmap in `SafeBarMapping` struct
  with `Drop` impl for RAII munmap and bounds-checked register accessors
- **`toadstool_report.rs`**: Socket resolution now uses `niche::socket_dirs()`
  with live path probing instead of hardcoded `/tmp` fallback
- **`primal_bridge.rs`**: Deprecated named accessors stripped of hardcoded
  primal name fallbacks — now pure `by_domain()` delegates

### Verified
- **Tier 4 binaries**: `validate_fpeos` 18/19 (advisory thermo consistency),
  `validate_atomec` 7/9 (average-atom SCF needs charge conservation tuning)
- **1002/1002** lib tests pass (up from 993)

---

## Phase 60 Absorption: Cross-Spring Parity (May 8, 2026)

### Added
- **Deploy graphs (1 → 5)**: 4 new domain-specific NUCLEUS deployment profiles:
  `hotspring_plasma_md_deploy.toml` (Tower + Node, no shader),
  `hotspring_nuclear_eos_deploy.toml` (Tower + Node + Nest provenance),
  `hotspring_spectral_deploy.toml` (Tower + barraCuda minimal),
  `hotspring_sovereign_gpu_deploy.toml` (full NUCLEUS with coralReef)
- **`tools/check_method_strings.sh`**: Method string drift detector — local
  registry check (source vs `capability_registry.toml`) + cross-registry
  check (hotSpring vs primalSpring canonical 389-method registry)
- **`tests/integration_registry_sync.rs`**: Rust integration tests for
  registry validation — `local_registry_parses_cleanly`,
  `deploy_graphs_reference_only_registered_capabilities`,
  `cross_registry_sync_with_primalspring` (ignored: 13 methods pending
  upstream addition)

### Changed
- **barraCuda optional**: `barracuda` dependency is now `optional = true`
  with `barracuda-local` default feature. Build with `--no-default-features`
  for IPC-only NUCLEUS deployment mode. Declaration of intent — all existing
  code continues to work with default features enabled.
## Unreleased — Paper Baseline Notebooks (May 7, 2026)

### Added
- **`notebooks/papers/`**: 12 publishable-grade Jupyter notebooks reproducing 22 peer-reviewed physics papers from Python baselines.
  - 01-semf-binding-energy (live SEMF for 2042 nuclei, Chabanat/AME2020)
  - 02-yukawa-screening (live Yukawa eigenvalues, Murillo & Weisheit 1998)
  - 03-sarkas-yukawa-md (live small-N MD + Daligault fit, Stanton & Murillo 2016)
  - 04-ttm-laser-plasma (live TTM ODE for 3 noble gases, Chen et al. 2001)
  - 05-stanton-murillo-transport (live Daligault analytical model)
  - 06-surrogate-learning (live RBF surrogate demo, Diaw et al. 2024)
  - 07-quenched-qcd (live SU(3) HMC beta-scan on 4^4, Wilson/Creutz)
  - 08-dynamical-fermions (staggered Dirac + phase diagram, Gottlieb 1987)
  - 09-abelian-higgs (live U(1) Higgs HMC on 8x8, Bazavov 2015)
  - 10-spectral-theory (Anderson 1D/2D/3D + Hofstadter + Herman, 6 papers)
  - 11-gradient-flow (LSCFRK3 W6/W7 integrators on 4^4, Luscher/Chuna)
  - 12-plasma-dielectric (completed Mermin + BGK relaxation, Chuna & Murillo 2024)
- **`notebooks/papers/PAPER_NOTEBOOK_GUIDE.md`**: Collaborator pattern doc (cell structure, data loading, evolution tiers)
- **`experiments/results/papers/`**: Directory for frozen production data (transport grids, HMC trajectories)

## Unreleased — sporePrint Tier 2 Notebooks (May 7, 2026)

### Added
- **sporePrint Tier 2 content**: 5 public notebooks + 6 frozen JSON validation data files.
  - `notebooks/01-composition-validation.ipynb` — guideStone Level 5, deploy graph, capability routing
  - `notebooks/02-benchmark-comparison.ipynb` — Python vs Rust, GPU vs CPU, DF64, energy/cost
  - `notebooks/03-experiment-evidence.ipynb` — 181 experiments, science ladder, evolution timeline
  - `notebooks/04-cross-spring-connections.ipynb` — primal consumption, patterns handed back
  - `notebooks/05-physics-deep-dive.ipynb` — nuclear EOS, lattice QCD, sovereign GPU, code safety
- **`experiments/results/`**: 6 frozen JSON files (composition_validation, test_suite_report, experiment_catalog, benchmark_timing, cross_spring_matrix, security_convergence)
- **`notebooks/NOTEBOOK_PATTERN.md`**: Adapted from primalSpring/wetSpring pattern for physics domains
- **`sporeprint/validation-summary.md`**: Updated with current numbers (993 tests, 181 experiments, 5 notebooks)

## Titan V Warm Handoff DMATRF Breakthrough + Docs Sweep (May 7, 2026)

### Added
- **Titan V warm handoff pipeline** (`volta_warm_pipeline.rs`): Direct `resource0` BAR0 mapping preserves nouveau warm state. DMATRF to FECS IMEM: 101 blocks (25632B) in 192µs, PRAMIN staging verified, DMEM PIO verified. Approach C (full firmware load) replaces BL-only approach.
- **Falcon v5 HS ROM security gate discovery**: All falcon v5 boots (SEC2, FECS, GPCCS) go through on-die ROM that validates IMEM contents against WPR-authenticated signatures. FECS ROM runs to PC=0x1161 with `exci=0x04070000` (security trap) when loaded with unsigned firmware. `sctl=0x3000` (HS mode 3) confirmed mandatory. This is the architectural gate between warm DMA capability and actual code execution.
- **SEC2 DMEM scan**: Full 64KB DMEM scan in warm audit reveals ACR firmware partial initialization (non-zero regions at 0x0000, 0x0200, 0x0B00, 0x0F00, 0xFE00 after IRQ poke).
- **nvidia-470 firmware extraction attempt**: Downloaded `nvidia-kernel-source-470` (470.256.02). `nv-kernel.o_binary` (40MB) has obfuscated symbols (`_nvNNNNNNrm`). PMU firmware embedded in RM binary, not extractable by header magic alone.

### Changed
- **Warm pipeline streamlined**: Removed ineffective approaches A (mailbox+IRQ) and B (blind CMDQ write) in favor of direct DMATRF FECS loading. SEC2 ACR at mb0=1 confirmed non-responsive to host-initiated commands.
- **FECS ENGCTL reset**: Added ENGCTL reset (pulse 0x01→0x00) before DMATRF loading for clean falcon state.
- **PRAMIN multi-page staging**: Firmware staging now handles multi-page PRAMIN writes (64KB window at BAR0+0x700000) for full fecs_inst.bin (25632B).

### Findings
- **Root blocker**: GV100 PMU firmware absent from `linux-firmware` (only `gm20b`/`gp10b` Tegra chips have PMU FW). SEC2 ACR BL starts (mb0=1) but never completes authentication — PMU manages power/clock domains required by ACR. nvidia-470 embeds PMU FW in its kernel module binary.
- **SEC2 warm state**: Running at PC=0x1161, TRACEPC shows deep ACR execution (0x2D07→0x4E5A), CMDQ header non-zero (h=0x8010) but DMEM queues never initialized. ACR BL started but stalled.
- **DMATRF timing**: 192µs for 101 blocks — physical VRAM DMA path is fully functional on warm GPU.

## K80 Warm FECS/PFIFO Pipeline + Checkpoint (April 30, 2026)

### Added
- **K80 warm FECS boot** (coralReef coral-driver): Internal firmware protocol for GK210B Kepler — FECS/GPCCS firmware loaded via Falcon v3 PIO, IMEM tags, csdata in DMEM. FECS boots and reaches idle state (CPUCTL=0x20). Internal firmware context size read from `0x409804`. Fire-and-forget channel binding for internal protocol.
- **Kepler PFIFO pipeline** (coralReef coral-driver): PFIFO scheduler sub-block (`0x2500-0x26FF`) permanently PRI-faulted after VFIO FLR — discovered accessible registers (`0x2270` RUNLIST_BASE, `0x2274` RUNLIST_SUBMIT, `0x2390+` PBDMA assignment table). Dynamic PBDMA-to-runlist discovery from `0x2390` (hardware shows PBDMA 0 → runlist 1, not 0). Runlist completes successfully.
- **Kepler doorbell mechanism**: Architecture-specific doorbell at `0x3000 + channel_id * 8` for GPFIFO notification.
- **GK104 runlist entry format**: Corrected to `(channel_id, 0x00000004)` matching Nouveau's `gk104_fifo_runlist_chan`.
- **Experiment 179**: K80 warm FECS dispatch pipeline — full debug journal.
- **K80 FECS/PFIFO handoff**: `wateringHole/handoffs/HOTSPRING_CORALREEF_K80_FECS_PFIFO_HANDOFF_APR30_2026.md`.

### Changed
- **`kepler_fecs_boot.rs`**: Tight PMC GR reset + CG disable sequence. Removed per-GPC ITFEN/WDT writes. Restored MC_UNK260 bracket. Falcon PC register reads corrected (0x0A4 not 0x0A8).
- **`fecs_method.rs`**: Wake trigger at `0x409840`. Internal firmware method interface (`fecs_internal_method`, `fecs_internal_bind_channel`, `fecs_internal_save_context`).
- **`warm_channel.rs`**: Internal firmware reads context size from `0x409804` (not FECS method 0x10). `fecs_set_watchdog_timeout` made non-fatal.
- **`kepler_csdata.rs`**: Fixed AINCW+star (was AINCW+starstar) — prevented FECS DMEM overwrite.
- **`gr_engine_status.rs`**: `fecs_halted()` correctly interprets CPUCTL 0x20 as idle (not halted). Added `ctxsw_mailbox0` field.
- **`pfifo.rs`**: Reads PBDMA→runlist assignment from `0x2390` (accessible on GK210B). Skips writes to PRI-faulted registers. PBDMA stale state clearing added.
- **`channel/mod.rs`**: `create_kepler` uses discovered `target_runlist` from `init_pfifo_engine_kepler` instead of hardcoded 0.
- **`page_tables.rs`**: Kepler runlist entry format corrected.
- **`registers.rs`**: `gk104_doorbell`, corrected runlist encoding functions, PCCSR field definitions.
- **`submission.rs`**: Architecture-specific doorbell for Kepler.
- **`device_open.rs`**: Fine-grained PFIFO register probe (0x2200–0x254C) for accessibility mapping.

### Findings
- **PFIFO scheduler sub-block dead after VFIO FLR**: Registers `0x2004`, `0x2200-0x2253`, `0x22C0`, `0x2300`, `0x2504`, `0x2600` consistently PRI-fault (`0xbad0011f`). Not recoverable by PMC/PRI resets. However, `0x2270` (RUNLIST_BASE), `0x2274` (RUNLIST_SUBMIT), `0x2390+` (PBDMA assignment) are accessible and functional.
- **Runlist ID mismatch**: Nouveau programs PBDMA 0 → runlist 1 at `0x2390`. Previous hardcoded runlist 0 caused silent stall. After fix, runlist completes.
- **SCHED_ERROR code=32**: Remaining issue — scheduler reports error after runlist completion. PBDMA 0 doesn't pick up the channel. Next: investigate PCCSR binding and PBDMA stale state clearing.

### Status
- K80 FECS boot: **OPERATIONAL** (firmware loads, boots, reaches idle)
- K80 PFIFO runlist: **OPERATIONAL** (completes on correct runlist ID)
- K80 GPFIFO dispatch: **IN PROGRESS** (SCHED_ERROR code=32, PBDMA stale state)

## Unreleased — K80 PGOB Binary Analysis + GPU Checkpoint (April 29, 2026)

### Added
- **nvidia-470 PGOB binary analysis**: Static disassembly of `nv-kernel.o_binary` revealed PSW-only PGOB sequence at `0x10a78c` — nvidia-470 skips `0x0205xx` power domain steps entirely. Two functions identified: `_nv029216rm` (ungate) and `_nv029114rm` (gate). Documented in `agentReagents/tools/k80-sovereign/nvidia470_pgob_analysis.md`.
- **`nvidia470_pgob_disable()`**: New PSW-only PGOB function in `coral-driver/pgob.rs`, derived from nvidia-470 binary analysis. Integrated into `kepler_warm.rs` as first-attempt before Nouveau fallback.
- **`nvidia470_pgob_enable()`**: Inverse function (re-gate GPCs) in `coral-driver/pgob.rs`.
- **Proprietary nvidia-470 build recipe**: `agentReagents/tools/k80-sovereign/build_nvidia470_kernel617.sh` — compiles proprietary nvidia-470 for kernel 6.17 in `/tmp` (zero host contamination). Applied Pop!_OS compat patches for `del_timer_sync`, `follow_pfn`, `__vma_start_write`, `drm_fb_create`.
- **QEMU VM reagent for K80**: Built and tested direct QEMU VM with K80 VFIO passthrough, host kernel, proprietary nvidia-470 module. Module successfully probed K80 (`NVRM: loading 470.256.02`). mmiotrace empty due to VFIO BAR mapping bypass — pivoted to static binary analysis.
- **K80 GPU solve status handoff**: `wateringHole/handoffs/HOTSPRING_CORALREEF_K80_PGOB_NVIDIA470_HANDOFF_APR29_2026.md`.

### Changed
- **`coral-driver/pgob.rs`**: Added nvidia-470 PSW-only PGOB functions alongside existing Nouveau-derived `gk110_pgob_disable`. Both approaches now available — PSW-only tried first, Nouveau fallback if GPCs remain gated.
- **`coral-driver/kepler_warm.rs`**: POST-done path and cold-recovery path both try `nvidia470_pgob_disable` first. Fallback to `gk110_pgob_disable` if GPCs still show `0xbadf` pattern.

### Findings
- **PSW-only requires running PMU firmware**: nvidia-470's PSW handshake at `0x10a78c` needs PMU falcon actively processing commands. Without loaded firmware, register writes are no-ops. The `0x0205xx` power steps succeed on GK210B (no PRIVRING faults as previously reported) but GPC PRI routes remain broken.
- **Root cause narrowed**: PRI ring shows `pri_gpc_cnt=0` — zero GPC stations enrolled. GPCs aren't just power-gated, they're absent from PRI topology. Two paths forward: PRI ring GPC enrollment, or PMU firmware load for PSW processing.

## Unreleased — Documentation Sweep + Handoff (April 27, 2026)

### Changed
- **Test counts**: 990→993 across all documentation (README, EXPERIMENT_INDEX, barracuda/README, whitePaper/*, specs/*, experiments/README, ABSORPTION_MANIFEST).
- **Dates**: All audit/status dates updated to April 27, 2026.
- **`whitePaper/baseCamp/nucleus_composition_evolution.md`**: Added Phase 46 composition template (§7) and deep debt evolution (§8) sections.
- **`whitePaper/README.md`**: Updated header to reflect Phase 46 + deep debt completion.
- **`whitePaper/baseCamp/README.md`**: Status line updated with Phase 46 + deep debt.
- **`specs/README.md`**: Date and test count updated.
- **`scripts/boot/install-coralreef-perms.sh`**: Hardcoded `/home/biomegate/` replaced with `$REEF_ROOT` env var with fallback.

### Added
- **`experiments/README.md`**: Deep debt evolution session entry (April 27).
- **`wateringHole/NUCLEUS_SPRING_ALIGNMENT.md`**: Phase 46 Composition Evolution section with lane assignments, hotSpring deep debt summary, updated spring pinning.
- **`wateringHole/handoffs/HOTSPRING_V0632_DEEPDEBT_PHASE46_HANDOFF_APR27_2026.md`**: Comprehensive handoff documenting capability-based discovery patterns, smart refactoring guidance, EVOLUTION markers for upstream, active PRIMAL_GAPS, composition patterns for NUCLEUS deployment.

## Deep Debt Evolution (April 27, 2026)

### Changed
- **Capability-based primal discovery**: `composition.rs` now derives primal requirements from `niche::DEPENDENCIES` (single source of truth) instead of duplicating name→domain mappings. `AtomicType` exposes `required_domains()` as the primary API; `required_primals()` is derived. Removed redundant `capability_domain_for_required_primal()`.
- **Deprecated named accessors**: `primal_bridge.rs` named methods (`toadstool()`, `beardog()`, `coralreef()`, etc.) deprecated in favor of `by_domain("compute")`, `by_domain("crypto")`, `by_domain("shader")` etc. All production callers migrated.
- **Data-driven alias resolution**: `primal_bridge.rs` hardcoded `if primal == "coralreef"` replaced with data-driven `PRIMAL_ALIASES` table.

### Refactored
- **`lattice/rhmc.rs` (989→802+190)**: Extracted Remez exchange algorithm and Gaussian elimination solver to `lattice/rhmc/remez.rs`. Physics types and RHMC functions remain in `mod.rs`.
- **`nuclear_eos_helpers.rs` (978→824+174)**: Extracted L1/L2 optimization objectives to `nuclear_eos_helpers/objectives.rs`. Residual metrics, reporting, and analysis remain in `mod.rs`.

### Fixed
- **Pre-existing `nuclear_eos_l2_*` compile errors**: Updated `nuclear_eos_l2_ref.rs` and `nuclear_eos_l2_hetero.rs` to handle upstream barraCuda `DiscoveredDevice` API change (`Auto::new()` → `.wgpu_device()`).

### Assessed (no action needed)
- **Unsafe code**: Only in `exp070_register_dump.rs` (BAR0 mmap) and `validate_5060_dual_use.rs` (CUDA launch) — both feature-gated with documented `#[expect]` reasons. `lib.rs` enforces `#![forbid(unsafe_code)]`.
- **Dependencies**: All healthy — no deprecated crates, no OpenSSL/ring. `blake3`, `wgpu`, `cudarc`, `rustix` are appropriate for their domains.
- **dyn dispatch**: Benchmark traits (`MdBenchmarkBackend`, `ComputeBackend`) retain `dyn` — open extensibility across lib/bin boundary, not on hot physics paths.
- **async_trait**: Zero uses. Clean.
- **`#[allow]` in production**: Only in `_fossilized/` (archived) binaries.
- **NpuSimulator**: Software NPU simulator used in production pipeline — legitimate fallback for non-NPU hardware, not a mock.

## Phase 46 Composition Template Absorption (April 27, 2026)

### Added
- **`tools/hotspring_composition.sh`**: Domain-specific NUCLEUS composition script implementing event-driven QCD computation with DAG memoization. Covers all 5 composition hooks (`domain_init`, `domain_render`, `domain_on_key`, `domain_on_click`, `domain_on_tick`). Async tick model uses convergence-based progression instead of fixed-rate polling. Parameter sweeps (beta values) run as DAG vertex walks with memoization to skip recomputed configurations. Ledger spines seal reproducible runs. Braids carry peer-review provenance (DOIs, lattice dimensions, hardware IDs). Compute dispatch routes to barraCuda/toadStool via IPC with local fallback.
- **`tools/nucleus_composition_lib.sh`**: Copied from primalSpring Phase 46 — reusable NUCLEUS composition library (41 functions: discovery, transport, DAG, ledger, braids, petalTongue, sensor streams).
- **DAG memoization wrapper**: `memo_check_vertex`, `memo_store_result`, `memo_get_result` — wraps rhizoCrypt DAG with SHA-256 parameter hashing. Candidate for upstream promotion to `nucleus_composition_lib.sh`.
- **Scientific provenance schema**: Braid metadata includes paper DOI, coupling constant (beta), lattice dimensions (L³×T), algorithm (HMC/RHMC), trajectory count, hardware ID, Rust version. Satisfies peer-review audit requirements.

### Verified
- Bare mode: all lib functions degrade gracefully (no NUCLEUS → no crash, capability checks return offline)
- Memoization: cache miss on first call, cache hit after store, DAG backing offline = local-only memo
- Compute dispatch: tensor probe skips when offline, SEMF falls back to local computation, background validation launches and polls correctly

### Documentation
- CHANGELOG, README, PRIMAL_GAPS, experiments/README, scripts/README: Phase 46 absorption
- wateringHole handoff: async computation patterns, DAG memoization, scientific provenance schema

## GPU Solve Tighten and Refactor (April 27, 2026)

### Changed (coralReef coral-driver)
- **init.rs split**: Monolithic 5466-LOC `vfio_compute/init.rs` split into 11 focused modules: `gr_bar0.rs`, `warm_channel.rs`, `kepler_cold.rs`, `kepler_warm.rs`, `kepler_recovery.rs`, `kepler_fecs_boot.rs`, `pmu.rs`, `pgob.rs`, `pri.rs`, `quiesce.rs`, `vbios_devinit.rs`. Original file reduced to 19-line re-export facade.
- **Shared abstractions**: `write_kepler_hub_station_params()` deduplicated across 3 files; `PGOB_POWER_STEPS` table deduplicated in `pgob.rs`; dead code (`kepler_pclock_pre_init`, `kepler_pri_station_probe`) removed.
- **kepler_csdata.rs**: `pub const` narrowed to `pub(crate) const`, `debug_assert!` for xfer==0 edge case, precondition docs added.
- **hardware_guard.rs**: Magic numbers replaced with named constants (`PMC_ENABLE`, `PGRAPH_BIT`, `DEAD_SENTINEL`).

### Changed (hotSpring barracuda)
- **IPC dedup**: Shared `primal_bridge::jsonrpc_request()` envelope builder; `glowplug_client` and `niche::send_registration` refactored to use shared transport.
- **GPU module DRY**: `gpu/mod.rs` extracted `open_from_adapter_inner()`; `hardware_calibration.rs` extracted `summarize_tiers()`; `precision_brain.rs` extracted shared `finish()` bootstrap.
- **Experiment bin hygiene**: 6 experiment bins now use `HOTSPRING_BDF` env var fallback instead of hardcoded BDF; `exp154`/`exp158` cross-referenced; `dual_dispatch.rs` `#[allow(dead_code)]` replaced with `#[expect(dead_code, reason="...")]`.

### Verified
- `cargo fmt` clean on both repos
- `cargo clippy -- -W clippy::pedantic -W clippy::nursery` passes (warnings only, no errors)
- `#![forbid(unsafe_code)]` holds in hotSpring library code
- No `#[allow()]` in production code (only in fossilized archives)

## Property 3 CHECKSUMS + Script Fix (April 17, 2026)

### Added
- **BLAKE3 CHECKSUMS manifest** (`validation/CHECKSUMS`): 15 validation-critical source files hashed with BLAKE3, following primalSpring's source-integrity pattern. Covers guideStone binary, physics modules, provenance, tolerances, composition, niche, lib, Cargo.toml, and validate-primal-proof.sh.
- Old binary-artifact CHECKSUMS preserved as `validation/CHECKSUMS.v0631-binaries`.

### Changed
- **Property 3 (Self-Verifying)**: `deny.toml` lookup now checks both `deny.toml` (barracuda/) and `barracuda/deny.toml` (repo root), so the guideStone passes from either CWD.
- **`validate-primal-proof.sh`**: Now builds from `barracuda/` then runs the binary from the repo root, so `validation/CHECKSUMS` resolves correctly. BLAKE3 checksums verify on every invocation.

### Verified
- Bare guideStone: **30/30 checks pass** (3 SKIP — expected NUCLEUS liveness only). Property 3 now fully green with all 15 file hashes verified.
- 993 unit tests pass (0 failures, 6 ignored)
- guideStone runs correctly from both barracuda/ and repo root

## Blackwell Dispatch Live (April 16, 2026)

### Fixed (coralReef Iter 85)
- **f64 division on Blackwell**: `MUFU.RCP64H` returns 0 on SM120 hardware — coralReef now uses F2F(f64→f32) + MUFU.RCP + F2F(f32→f64) seed on SM ≥ 100, 2 Newton-Raphson iterations for full precision. Same fix applied to f64 sqrt (RSQ64H → F2F+RSQ).
- **`@builtin(num_workgroups)` on Blackwell**: S2R NCTAID_X/Y/Z returns [0,0,0] on SM120 — coralReef now emits LDC c[7][0/4/8] from driver constants CBUF on SM ≥ 100. coral-driver populates CBUF 7 with grid dimensions at dispatch time.
- **Semaphore fence ordering**: `submit_fence_release` uses compute engine's SET_REPORT_SEMAPHORE (subchannel 1) instead of PBDMA — ensures fence completes only after compute work
- **UVM write access**: `map_external_allocation` now sets `gpu_mapping_type = 1` (ReadWriteAtomic)
- **QMD v5.0 completeness**: GRID_*_RESUME fields, SM_CONFIG_SHARED_MEM_SIZE, QMD_GROUP_ID = 0x1f

### Documentation
- wateringHole handoff: `HOTSPRING_BLACKWELL_DISPATCH_LIVE_HANDOFF_APR16_2026.md`

## primalSpring v0.9.17 Absorption (April 20, 2026)

### Changed
- **primalspring dependency**: v0.9.16 → v0.9.17 (genomeBin v5.1, 46 cross-arch binaries, deployment-validated end-to-end, guideStone standard v1.2.0)
- **Doc reference**: v0.9.16 → v0.9.17 in guideStone binary module doc
- **`validate-primal-proof.sh`**: Auto-sets required NUCLEUS env vars when FAMILY_ID is provided — `BEARDOG_FAMILY_SEED` (derived from FAMILY_ID), `SONGBIRD_SECURITY_PROVIDER=beardog`, `NESTGATE_JWT_SECRET` (random Base64). Header updated for v0.9.17 genomeBin depot workflow.

### Verified
- Bare guideStone: 14/14 checks pass against primalSpring v0.9.17 (backward-compatible API)
- All v0.9.17 known issues addressed in script: beardog seed, songbird security provider, nestgate JWT, coralReef `--rpc-bind`

### Documentation
- CHANGELOG, README, PRIMAL_GAPS: v0.9.17 absorption session
- wateringHole handoffs: comprehensive evolution patterns + v0.9.17 absorption

## primalSpring v0.9.16 Absorption (April 20, 2026)

### Added
- `scripts/validate-primal-proof.sh` — end-to-end primal proof validation script. Bare mode (domain only) and `--full` mode (pre-flight `primalspring_guidestone` + domain `hotspring_guidestone`). Detects bare-only vs live NUCLEUS automatically.
- plasmidBin deployment workflow documented in README Quick Start

### Changed
- **Property 3 (Self-Verifying)**: Upgraded from manual CHECKSUMS file-exists check to `primalspring::checksums::verify_manifest()` — BLAKE3 per-file hash verification with PASS/FAIL/SKIP semantics (v0.9.16 pattern)
- **Protocol tolerance**: Added `is_protocol_error()` arms to `validate_provenance_witness` and `validate_compute_dispatch` — Songbird/petalTongue HTTP-on-UDS classified as SKIP (reachable but incompatible), matching v0.9.16 liveness semantics
- **primalspring dependency**: Auto-updated v0.9.15 → v0.9.16 (BLAKE3 checksums module, family-aware discovery, protocol error classification)
- **Doc reference**: v0.9.15 → v0.9.16 in guideStone binary module doc

### Verified
- Bare guideStone: 14/14 checks pass (4 SKIP expected — no CHECKSUMS manifest, no NUCLEUS primals)
- Known v0.9.16 issues handled: `is_protocol_error()` → SKIP for HTTP-on-UDS, `is_connection_error()` → SKIP for BearDog BTSP reset
- plasmidBin ecoBin present: `hotspring_primal` (v0.6.32, x86_64 musl-static)

### Documentation
- CHANGELOG, README, PRIMAL_GAPS: v0.9.16 absorption session
- wateringHole handoff: `HOTSPRING_V0632_V0916_ABSORPTION_HANDOFF_APR20_2026.md`

## Sovereign Compute Parity + guideStone (April 18-19, 2026)

### Added
- `bench_sovereign_parity` binary — dual-path QCD benchmark (coral-gpu sovereign vs wgpu vendor)
- Experiments 173–176: VM reagent WPR capture, K80 sovereign boot, RTX 5060 shared compute, QCD parity benchmark
- Full HMC pipeline (10 shaders) compiles to native SASS on SM35/SM70/SM120 via coralReef
- `validate_pure_gauge` sovereign GPU compile integration (16/16 checks pass)
- QMD v5.0 support for Blackwell (SM120+) in coral-driver
- **`hotspring_guidestone` binary** — self-validating NUCLEUS deployable using `primalspring::composition` API
- `deny.toml` for barracuda + metalForge/forge — ecoBin C-dep bans
- `validate_primal_proof` binary — Level 5 primal proof harness (10 manifest capabilities)
- wateringHole handoffs: sovereign compile parity, Level 5 composition proof, Blackwell dispatch handoff

### Fixed
- coralReef `UvmPageableMemAccessParams` ABI bug: struct was 4 bytes, kernel expects 8 — `pageable_mem_access` field was missing, causing false "failure" reports (ioctl was always succeeding)
- coralReef coral-kmod VRAM alloc flags: 2MB huge page attrs on 4KB buffers caused FAULT_PDE; fixed to PAGE_SIZE_4KB
- coralReef f64 transcendental lowering: removed SM < 70 guard, SM32 now lowers via MUFU seed + Newton-Raphson
- coralReef SM32 encoder: `emit_iadd` (IAdd2 for Kepler), `emit_shl_imm` (OpShl for Kepler)
- coralReef `as_imm_not_i20`/`as_imm_not_f20`: graceful fallback when source modifiers on immediates
- Broken doc links: `CONTROL_EXPERIMENT_STATUS.md` and `NUCLEAR_EOS_STRATEGY.md` marked superseded
- GAP-HS-026 resolved: All 13 physics/compute methods wired in `hotspring_primal.rs` server dispatch
- Unsafe elimination: `niche::set_family_id()` with `OnceLock` replaces `set_var`
- `dyn` dispatch eliminated, `#[allow]` → `#[expect]` migration

### Changed
- guideStone Level 5 CERTIFIED (all 5 guideStone properties satisfied)
- `validate_all.rs` expanded from 37 to 64 suites
- Binary count: 140 → 166
- GAP-HS-031 root cause identified: GR context buffers not eagerly promoted on Blackwell (UVM VA space registration fails, GPU_PROMOTE_CTX requires kernel privilege)

## v0.6.32 — Composition Audit + Doc Cleanup (April 11, 2026)

### Added
- `graphs/` directory with deploy TOMLs aligned to proto-nucleate
- `validate_squirrel_roundtrip` binary for end-to-end inference validation
- Root-level CHANGELOG (this file)
- wateringHole handoff: `HOTSPRING_V0632_COMPOSITION_AUDIT_PRIMAL_EVOLUTION_HANDOFF_APR11_2026.md`

### Fixed
- **Socket naming mismatch**: `hotspring_primal` now uses `niche::resolve_server_socket()` for family-scoped socket names (`hotspring-physics-{family_id}.sock`) — previously hardcoded `hotspring-physics.sock`
- **biomeOS registration**: `register_with_target()` now called on server startup — previously existed but was never invoked
- **barraCuda pin reconciled**: `Cargo.toml` now matches CHANGELOG v0.6.32 pin (`b95e9c59`)
- **DAG method names aligned**: `dag_provenance.rs` uses canonical wire names (`dag.session.create`, `dag.event.append`, `dag.merkle.root`) matching `capability_registry.toml`
- **Crypto method name**: `receipt_signing.rs` uses `crypto.sign_ed25519` (was `crypto.sign`)
- **Capability validation binaries**: `validate_nucleus_composition` and `validate_nucleus_tower` use canonical capability names (`crypto.sign_ed25519`, `crypto.verify_ed25519`, `discovery.find_primals`, `shader.compile.wgsl`)
- **Registered-but-pending dispatch**: `hotspring_primal` returns structured `-32001` error for registered physics methods not yet dispatched (was generic `-32601 Method not found`)
- **discover_capabilities()**: Now delegates to `niche::all_capabilities()` as source of truth

### Changed
- `validation.rs` split into `validation/` module (harness, telemetry, composition, tests) — was 1392 lines, now all files under 520 LOC
- `--family-id` CLI argument now sets `FAMILY_ID` env var (was silently ignored)

### Documentation
- Root README: fixed baseCamp listing (5→17 docs), barraCuda pin ref (`fbad3c0a`→`b95e9c59`), added `graphs/` and `CHANGELOG.md` to directory tree, expanded composition narrative
- `whitePaper/README.md`: status promoted from "Working draft" to "Current", codebase health table updated (Feb 26→Apr 11, v0.6.14→v0.6.32)
- `whitePaper/baseCamp/` listing in README/whitePaper now reflects full 17-doc set
- `experiments/README.md`: added NUCLEUS composition section, noted Exp 164 journal gap, disambiguated Exp 103a/103b
- Archived superseded handoff `HOTSPRING_V0632_COMPOSITION_EVOLUTION_HANDOFF_APR11_2026`

## v0.6.32 — Composition Evolution (April 10-11, 2026)

See `barracuda/CHANGELOG.md` for full details.

## License

AGPL-3.0-or-later
