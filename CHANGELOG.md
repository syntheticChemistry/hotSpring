# Changelog

All notable changes to hotSpring.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file covers the spring as a whole. For crate-level details see
`barracuda/CHANGELOG.md`.

## Unreleased — K80 Warm FECS/PFIFO Pipeline + Checkpoint (April 30, 2026)

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
