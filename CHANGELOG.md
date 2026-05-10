# Changelog

All notable changes to hotSpring.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file covers the spring as a whole. For crate-level details see
`barracuda/CHANGELOG.md`.

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
- `cargo test --lib` — 1,019 passed, 0 failed, 6 ignored
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
