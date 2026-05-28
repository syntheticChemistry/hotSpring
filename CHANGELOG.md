# Changelog

All notable changes to hotSpring.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file covers the spring as a whole. For crate-level details see
`barracuda/CHANGELOG.md`.

## Diesel Engine Silicon Deistic Abstraction — Exp 230/231 (May 28, 2026)

### Added
- **`InterruptProfile` struct** in `pmc.rs` — per-generation interrupt register
  semantics (direct-write for Kepler/Maxwell/Pascal, SET/CLEAR pair for Volta+).
  `for_sm()` factory, `disable_offset()` / `disable_value()` helpers.
- **`HandoffCapabilityProfile` struct** — generation-aware hardware profile with
  GPC count, TPC base/stride, PCCSR base, FECS/GPCCS/PMU bases, BAR0 domain map,
  PMC warm threshold. Built from SM version via `GenerationProfile`.
- **`PatchSet::from_recipe_toml()`** — loads patch sets from catalyst recipe TOML
  files (e.g. `gv100_nvidia470.toml`). Enables new GPU+driver combos without
  recompiling cylinder.
- **`PatchSet::by_profile()`** — dispatches patch sets from `(ChipFamily, driver, strategy)`
  instead of magic string names. Keeps compiled-in defaults with TOML override.
- **`PatchStrategy::from_str()`** — parser for TOML string format
  (`"RetAtEntry"`, `"NopCallAt(0x7f)"`, etc.).
- **`execute_handoff_with_heartbeat()`** — accepts heartbeat callback for external
  watchdog integration. 13 heartbeat calls wired at pipeline step boundaries.
- **Experiment 230**: Diesel Abstraction Revalidation — validation plan for Titan V.
- **Experiment 231**: K80 Cross-Gen Quench Probe — Kepler interrupt quench test plan.

### Changed
- **`GenerationProfile`** — added `interrupt_profile: InterruptProfile` field to
  all 11 generation consts (Kepler through Blackwell B).
- **`HandoffConfig`** — added `sm_version: Option<u32>` field. All 7 config
  presets now specify SM version (70 for Titan V, 35 for K80).
- **Pipeline** — replaced ~15 hardcoded Volta register offsets with
  `HandoffCapabilityProfile` lookups (GPC topology, PCCSR scan, FECS capture,
  PMC warm threshold, firmware artifact naming, BAR0 domain map).
- **`post_exit_quench`** / **`emergency_quench`** / **`quench_gpu_interrupts`** —
  refactored to use shared `pmc::quench_interrupts()` driven by `InterruptProfile`.
- **`rm_trigger` binary** — accepts `--bdf` CLI arg. All 6 hardcoded
  `"0000:49:00.0"` references replaced with parameterized BDF.
- **`catalyst_watchdog::activate()`** — accepts `InterruptProfile` + configurable
  timeout (was hardcoded 120s, now 450s for catalyst).
- **`pri_recovery::recover_pri_ring()`** — accepts `chip_name` parameter for
  generation-aware firmware artifact naming.

## Catalyst Channel + Lockup Forensics — Exp 229 (May 28, 2026)

### Added
- **Experiment 229**: Catalyst Channel — RM Compute Channel Before Warm Swap.
  Full end-to-end Tier 2 WarmCompute handoff achieved (Run #9: `success=true`,
  `total_ms=80886`, 20/20 patches, GPU warm at 23 engines throughout).
- **`nv_close_device` RetAtEntry patch** — prevents nvidia_close from
  tearing down RM thread stacks, IRQ handler, and MSI while RM threads
  are still running. Root cause of lockups #4–#7 (interrupt storm + stack
  use-after-free).
- **`nv_pci_remove` RetAtEntry patch** — prevents PCI unbind callback from
  hanging in `os_delay` polling loop (GPU not quiesced). Enables clean
  nvsov→vfio-pci warm swap.
- **`post_exit_quench()`** in `trigger_rm_init` — writes 0xFFFFFFFF to
  `NV_PMC_INTR_EN_CLEAR_0` (BAR0+0x180) AFTER `nvidia_close` completes,
  catching re-enabled interrupts. Belt-and-suspenders defense.
- **`post_exit_intx_disable()`** — sets PCI CMD bit 10 (INTx Disable) after
  nvidia_close MSI teardown, preventing legacy INTx storm.
- **`catalyst_watchdog`** background thread — monitors handoff liveness,
  performs emergency interrupt quench + process kill if pipeline hangs.
- **`catalyst-sentinel.sh` v2** — BAR0/sysfs-only forensic logger (removed
  PCI config reads that acquired `pci_lock` and were a secondary lockup
  vector). Captured evidence for all 9 runs across power cycles.
- **`docs/exp229-lockup-analysis.md`** — complete forensic record: 5 distinct
  lockup vectors identified, mitigated, and proven across 9 runs.

### Fixed
- **System lockup on catalyst handoff** — 7 lockups triaged across power
  cycles using diesel engine sentinel data. Five distinct vectors cataloged:
  1. `pci_lock` deadlock from keepalive reads (exclusion guard)
  2. INTR_EN quench to read-only register 0x140 (fixed: CLEAR@0x180)
  3. nvidia_close re-enables INTR_EN after quench (post-exit pipeline quench)
  4. `nv_dev_free_stacks` use-after-free on RM thread stacks (nv_close_device NOP)
  5. `nv_pci_remove` os_delay hang on unbind (nv_pci_remove NOP)

### Changed
- **`trigger_rm_init()`** now accepts `bdf` parameter for post-exit quench.
- **`nvidia_catalyst_handoff` patch set** — 20 targets (was 18): added
  `nv_close_device` and `nv_pci_remove` RetAtEntry.

## SBR Bus Reset Suppression — Exp 226 (May 26, 2026)

### Added
- **Experiment 226**: SBR Bus Reset Suppression — Exp 225 FLR-first fix
  suppressed per-device resets via `reset_method`, but the kernel's
  `vfio_pci_dev_set_try_reset()` fires `pci_reset_bus()` (Secondary Bus
  Reset) when all devices in the dev_set have `open_count==0`. SBR resets
  everything behind the PCIe bridge (`00:01.3`), bypassing per-device
  `reset_method` entirely.
- **`suppress_bus_reset()`** in `guarded_sysfs/driver_ops.rs` (was `guarded_sysfs.rs` — split S276) — compiles a tiny GPL
  kernel module (`no_bus_reset.ko`) via kbuild that sets
  `PCI_DEV_FLAGS_NO_BUS_RESET` on the target device, making
  `pci_bus_resetable()` return false and `pci_reset_bus()` return
  `-ENOTTY`.
- **`restore_bus_reset()`** in `guarded_sysfs/driver_ops.rs` — unloads the module
  and cleans up build artifacts.
- **Three-layer reset defense** in `prepare_anchor_release()`: (1) bridge
  power pinning, (2) per-device FLR/PM suppression, (3) bus-level SBR
  suppression via `no_bus_reset.ko`.

### Changed
- **Step 0e diagnostic** updated to reference SBR/Exp 226 instead of
  FLR/Exp 225.
- **Step 9** now calls `restore_bus_reset()` after `restore_flr()` to
  unload the module and re-enable SBR for subsequent operations.

## Diesel Engine Evolution: Reset-on-Release Fix (May 26, 2026)

### Added
- **Experiment 225**: Catalyst TPC Persistence Test — tested whether TPC PRI
  stations survive nvidia-470 catalyst unbind → vfio-pci rebind. **RESULT:**
  nvidia RM failed to initialize GPU; `vfio-pci` reset-on-release destroyed
  VBIOS warm state (`PMC_ENABLE 0x5fecdff1 → 0x40000020`). Titan V #1 ended
  at Tier 0 (Cold). Root cause: FLR suppression applied too late in handoff
  pipeline — after `VfioAnchor` drop triggered `vfio_pci_core_release()`.
- **`prepare_anchor_release()`** in `guarded_sysfs.rs` — composable helper
  that pins bridges, disables FLR on target + IOMMU siblings before anchor
  drop. Foundation for the FLR-first anchor release pattern.
- **`VfioAnchor::release_prepared()`** in `vfio_anchor.rs` — self-consuming
  release method with debug assertion that `reset_method` is empty, catching
  callers that skip `prepare_anchor_release()`.
- **Step 0e anchor release guard** in `execute_handoff` — reads PMC_ENABLE
  after anchor release and halts with clear diagnostic if GPU went cold
  (popcount < 10). Prevents wasting 60s on doomed catalyst settle.
- **Post-settle RM health check** — after seeder settle period, verifies
  nvidia RM completed DEVINIT. Logs failure but continues for forensics.

### Changed
- **RPC handler**: Both `sovereign_warm_handoff` and `sovereign_catalyst_boot`
  now call `prepare_anchor_release()` BEFORE `VfioAnchor` drop, and use
  `release_prepared()` instead of implicit drop. This is the Exp 225 fix.
- **Step 5 FLR** in `execute_handoff` documented as safety belt (idempotent
  for direct callers; RPC handler already suppresses FLR before anchor release).
- **`is_catalyst`** detection moved earlier in `execute_handoff` (before
  Step 0e) to enable anchor release guard; duplicate definition removed.

## Sovereignty Audit + PostPrimordial Checkpoint (May 26, 2026)

### Changed
- **Sovereignty audit**: `sovereign.classify_tier` confirmed both Titan Vs at
  **Tier 1 (WarmInfrastructure)**: `tpc_alive=false`, `tpc_status=0xBADF5040`,
  `gpc_enables=0x00000000`. Tier 2 sovereign compute NOT achieved.
- **Documentation corrected**: README, EXPERIMENT_INDEX, Exp 223 all updated to
  reflect honest Tier 1 status. `compute_ready` documented as init health check,
  not dispatch readiness. VBIOS POST is the source of warm state, not toadStool.
- **PostPrimordial transition**: local wateringHole archived to infra hub,
  hotSpring aligned to live plasmidBin deployments (except toadStool = local build).

## ACR Sovereign Boot Catalyst — Infrastructure Hardening (May 26, 2026)

### Added
- **Experiment 223**: ACR Sovereign Boot Catalyst — HS mode architecture mapped,
  ENGCTL destruction identified, CPUCTL_ALIAS boot path validated. Python
  catalyst superseded by Rust `exp224_pmu_acr_catalyst`. **Tier 1 confirmed**
  via `classify_tier` (TPC wall intact, GPCCS HS fuse-locked).
- **`low_level/falcon.rs`** — shared Falcon v5 register map module: 24 canonical
  register offsets, 5 engine bases (PMU/FECS/GPCCS/SEC2/NVDEC), `FalconSnapshot`
  struct with `sec_mode()`/`cpu_state()`/`imem_size()`/`dmem_size()`, PIO
  upload/verify helpers, `Bar0Domain` presets.
- **`Bar0Error` enum** — typed errors for BAR0 access: `DeadLink`, `Unaligned`,
  `OutOfDomain`, `DenyListed`, `Overflow`, `OutOfBounds`.
- **`Bar0Map::r32_checked`** — dead-link sentinel detection (`0xFFFF_FFFF`).
- **`Bar0Map::open_bdf`** — BDF-based open with `HOTSPRING_SYSFS_PCI` env support.
- **`SafeBar0::with_deny_list`** — write deny-list for dangerous registers
  (ENGCTL destroys falcon security state irreversibly).
- **16 unit tests** for `bar0.rs` (alignment, domain, deny-list, sentinel) and
  `falcon.rs` (SEC_MODE decode, CPUCTL bits, IMEM/DMEM sizing, engine offsets).
- **`pub mod low_level`** exported from crate lib behind `low-level` feature gate.

### Changed
- **`#![forbid(unsafe_code)]` → `#![deny(unsafe_code)]`** on barracuda lib to
  allow `#[allow(unsafe_code)]` on the `low_level` module. Unsafe remains
  confined to BAR0 MMIO (mmap/volatile) and not allowed elsewhere.
- **Alignment checks** added to `Bar0Map::r32`/`w32` — panics on non-4-byte-aligned offsets.
- **exp224 rewired** to use `hotspring_barracuda::low_level::{bar0::*, falcon::*}`
  instead of inline `#[path]` inclusion and duplicated register constants.
- **Superseded Python scripts** (`acr_sovereign_boot.py`, `post_reboot_acr_boot.py`)
  documented as fossils in `infra/catalysts/reagents/`.

### Discovered
- Direct host PIO to PMU is **blocked in HS mode 2** (VBIOS-initialized state).
  The PMU is a firmware fortress — the correct boot path uses an intermediate
  Boot Falcon (NVDEC/SEC2) for ACR, which is what toadStool's `sovereign.init`
  implements.
- ENGCTL (0x3C0) toggle is an irreversible engine reset (HS→NS), not an "HS
  unlock". CPU execution permanently disabled. Only recoverable via power cycle.
- CPUCTL_ALIAS (0x130) is the correct host control register for HS falcons, but
  it is unresponsive in the pre-DEVINIT VBIOS state.

## UEFI Model GPU Sovereignty — PRI Ring Recovery (May 25, 2026)

### Added
- **Experiment 221**: UEFI Model GPU Sovereignty — tested "Firmware as Boot
  Service" hypothesis. PRI ring recovery proven: PGRAPH re-enable
  (PMC_ENABLE bit 12) + PRI ring master enumerate restores routing after
  kernel PCI framework destroys it. Falcon registers accessible post-recovery.
- **`recover_pri_ring()`** in `sovereign_handoff.rs` — integrated into handoff
  pipeline as step 6c (between warm_swap and tier_classify).
- **Diagnostic probe enhancements** — PCI config space reads (command register,
  PM state), correct falcon PC register (`0x40911c`), PGRAPH and PRI ring
  master status capture.
- **GAP-TS-221-A through D**: Runtime Services model, GPC sub-ring recovery,
  PCI bus master disable, firmware extraction from nvidia.ko.
- **Handoff**: `HOTSPRING_UEFI_MODEL_PRI_RING_RECOVERY_EXP221_MAY25_2026.md`
  posted to wateringHole.

### Changed
- **`nvidia_boot_services` patch set** — no longer uses `RetAtEntry` on
  `nv_pci_remove` (leaks iomem without preserving PRI ring). Now delegates to
  `nvidia_catalyst_handoff` (clean unbind + post-swap recovery).
- **`PriRingAnchor` health classification** — probes post-recovery BAR0 state
  instead of pre-swap catalyst_tier. Correctly classifies as Degraded.
- **Falcon register addresses** — `FALCON_PC` corrected to `0x11c` (hardware
  PC). Added `FALCON_BOOTVEC` (`0x104`) and `FALCON_STATUS` (`0x108`).
- **Sovereignty Tier Model** — added Tier 1+ (PRI Recovery) between Tier 1
  and Tier 2. Tier 2 now documented as requiring Runtime Services model.

### Discovered
- PRI ring destruction occurs in **kernel PCI framework** (`pci_device_remove`
  clears PMC_ENABLE), not in nvidia's `nv_pci_remove`.
- `RetAtEntry` on `nv_pci_remove` is a dead end — still kills PRI ring AND
  leaks iomem (`request_mem_region` stale claims).
- FECS/GPCCS on GV100 are **fuse-enforced HS (high-security)** mode — host
  IMEM PIO upload blocked by hardware fuses.
- FECS IMEM is genuinely wiped during PCI unbind, not just PRI-gated.
- ACR boot from vfio-pci not viable — WPR not configured on GV100 (pre-GSP).
- **Architecture pivot**: Tier 2 (WarmCompute) requires nvidia as persistent
  Runtime Service, not exitable boot service.

### Validated
- Both Titan V cards (02:00.0, 49:00.0): PRI ring recovery stable and
  repeatable. Zero D-state threads. Zero iomem leaks. No system instability.

## CAZyme FEL — Phase 0.5 Validated (May 24, 2026)

### Added
- **Phase 0.5 complete**: Free beta-D-xylopyranose ring puckering FEL.
  CHARMM36 force field (carb.rtp BXYL), TIP3P water (863 molecules),
  10 ns WTMetaD on Cremer-Pople theta (PLUMED PUCKERING CV).
  Three basins resolved: two chairs (theta~8°, ~172°) + boat (~91°).
  Barriers 40-54 kJ/mol consistent with pyranose literature.
  Full carbohydrate MD pipeline validated: RDKit → pdb2gmx → solvation →
  EM → NVT/NPT → PLUMED puckering CV → WTMetaD → sum_hills → FEL.
  Report: `control/gromacs_fel/cazyme_gh10/VALIDATION_REPORT.md`
- **CHARMM36-jul2022**: Downloaded and validated carbohydrate force field
  (364 carbohydrate residue definitions including AXYL, BXYL, ALXYL, BLXYL).
- **PDB 1E0X**: GH10 xylanase (Xyl10A, Streptomyces lividans) structure
  downloaded and analyzed. Xylobiosyl-enzyme intermediate with X2F + XYP
  sugars covalently linked to Glu-236.

## CAZyme FEL — Phase 0.4 Validated (May 24, 2026)

### Added
- **Phase 0.4 complete**: Alanine dipeptide well-tempered metadynamics tutorial
  executed end-to-end. 10 ns production run, GROMACS 2026.0 + PLUMED 2.9.2.
  C7eq global minimum at phi=-81.2°, psi=52.9°; C7ax at phi=+60°, ΔF=5.57 kJ/mol.
  Converged within ±0.5 kJ/mol. Matches literature (AMBER99SB-ILDN).
  Full report: `control/gromacs_fel/tutorial/alanine_dipeptide/wtmetad/VALIDATION_REPORT.md`
- **PLUMED 2.9.2** installed in gromacs-fel environment (PLUMED_KERNEL linkage).
- **Tutorial workspace**: `control/gromacs_fel/tutorial/alanine_dipeptide/wtmetad/`
  with md.mdp, plumed.dat, HILLS, COLVAR, fes_2d.dat, fes_phi.dat, fes_psi.dat.
## CAZyme FEL — Biomolecular MD Evolution (May 24, 2026)

### Added
- **Experiment 220**: CAZyme Conformational Energy Landscapes — GROMACS 2026.0
  installed as industry control (CUDA, PLUMED, Colvars built-in). 4-phase plan:
  GROMACS tutorial → barraCuda bonded FF shaders → hotSpring topology/MD loop →
  metadynamics bias → parity validation.
- **GROMACS 2026.0 control environment**: `conda activate gromacs-fel` on
  strandGate RTX 3090. GPU-accelerated, CUDA 12.9.
- **GAP-HS-111**: Biomolecular force field evolution — bonded FF terms
  (harmonic bond/angle, dihedral torsion, improper), topology reader
  (GROMOS 45a4 / GLYCAM06), metadynamics bias layer.
- **GAP-HS-112**: petalTongue FEL visualization evolution — 2D/3D FEL surfaces,
  Cremer-Pople CV plots, interactive ring puckering viz.
- **helixVision downstream feed**: Documented predict→validate→confirm→visualize
  loop (coralForge predicts → hotSpring MD validates → GROMACS confirms →
  petalTongue renders).
- **`control/gromacs_fel/`**: GROMACS industry control workspace.
- **Handoff**: `HOTSPRING_CAZYME_FEL_EVOLUTION_MAY24_2026.md` posted to local
  and upstream wateringHole.

### Context
- Collaborators: Alistaire (domain expert — CAZyme biochemistry, QM/MM),
  Mark (NSF HPC — Texas A&M ACES, A100 GPUs).
- Scientific question: Do AutoDock Vina docking scores correlate with
  metadynamics FEL in CAZyme active sites? (5 min docking vs 12–48 hr FEL)
- Pilot system: GH10 β-xylanase (PDB 1E0X).
- Domain: specs/ updated with biomolecular MD scope.

## Covalent Gate Deployment — Wave 46+ (May 23, 2026)

### Added
- **biomeGate deployment handoff**: `HOTSPRING_GATE_DEPLOYMENT_MAY23_2026.md` —
  confirms sole-tenant assignment on biomeGate (TR 3970X, 2× Titan V, 256GB),
  documents proto-nucleate composition (9 primals), deployment flow, validation
  status, and contention assessment (none — sole tenant).
- **GAP-HS-108**: Hardware doc drift (K80 still listed upstream, actually retired).
- **GAP-HS-109**: skunkBat niche-hotspring vs proto-nucleate mismatch.
- **GAP-HS-110**: Sovereign stack not integrated into validate-primal-proof.sh.

### Validated
- Deploy graphs (7) pass Dark Forest Glacial Gate (all 5 pillars).
- 24 validation scenarios compile clean.
- Proto-nucleate composition wired: 9 primals, proton_heavy profile.
- Sovereign GPU stack operational: VFIO boot, ember daemon, catalyst pipeline.
- Deployment path confirmed: `fetch_primals.sh` → `nucleus_launcher.sh` →
  `validate_composition.sh` → `validate-primal-proof.sh --full`.

### Status
- **Live NUCLEUS validation**: PENDING — requires biomeGate plasmidBin pull.
- **Rewiring tier**: 3 (~15-25% IPC coverage).
- **Next**: Pull v2026.05.23 binaries on biomeGate, start NUCLEUS, run full chain.

## Catalyst Driver Pattern HW Validated — Exp 219 (May 24, 2026)

### Changed
- **Domain-scoped BAR0 capture**: `Bar0Snapshot::capture_domains()` reads only
  known Volta BAR0 domains (641K registers) instead of full 16 MiB linear scan
  (4.2M registers). **515× faster** — 897ms vs 462s.
- **Surgical `nv_pci_remove` patches**: Replaced blanket `RetAtEntry` with four
  `NopCallAt` patches at offsets 0x374/0x3a0/0x1fe/0x2a0 — allows PCI resource
  cleanup (`__release_region`, `pci_disable_device`) while NOP-ing GPU teardown
  functions (`nv_shutdown_adapter`, `rm_disable_gpu_state`, etc.).
- **Pipeline reordering**: BAR0 capture moved before sibling rebind to avoid
  PCI device lock contention with NVIDIA RM teardown.
- **Fire-and-poll unbind**: `sysfs_unbind_fire_and_poll` replaces blocking
  unbind for the 7s NVIDIA RM teardown, keeping toadstool-ember responsive.

### Added
- `Bar0Snapshot::capture_domains()` in `warm_capture.rs`.
- SBR bridge reset recovery path — `setpci BRIDGE_CONTROL.W` bit 6 toggle
  recovers GPUs from dirty catalyst states without full power cycle.
- `profile-catalyst-teardown.sh` ftrace script for kernel function profiling.

### Validated
- Catalyst pipeline completes in **26s** (first clean end-to-end success).
- 83,623 alive registers captured across 22 Volta BAR0 domains.
- Tier 1 (WarmInfrastructure) confirmed on live Titan V hardware.
- 3-layer preservation operational: frozen .ko (41MB) + recipe JSON + replay
  sequence (83K writes) persisted.
- Zero TODO/FIXME in changed hotSpring files.
## nvidia-470 nvsov Dual-Load Injection — Exp 218 (May 21, 2026)

### Added
- **Co-load isolation NOP set**: 5 new targets (`nv_cap_init`, `nv_cap_drv_init`,
  `nv_procfs_init`, `nvidia_register_module`, `nv_cap_procfs_init`) in
  `PatchSet::nvidia_warm_handoff()` — prevent procfs/chardev/capabilities
  conflicts when loading `nvsov` alongside host nvidia-580.
- **`strip_ksymtab()`**: ELF section zeroing for `__ksymtab`, `__kcrctab`,
  `__ksymtab_strings` to prevent kernel "exports duplicate symbol" rejection.
- **`objcopy`-based ksymtab stripping**: Post-patch `objcopy --remove-section`
  in the DKMS handoff path — robust section removal before `insmod`.
- **`sovereign.snapshot` RPC**: Read-only GPU state capture endpoint.
- **`sovereign.compare` RPC**: Twin-card structured diff (captures two
  snapshots, returns `SnapshotDelta` list).
- **`SnapshotDelta` + `diff_structured()`**: Structured comparison of
  `SovereignSnapshot` pairs.
- **`SymbolResolver` trait + `NmResolver`**: Abstracted symbol resolution
  from patch application.
- **`PatchSet::from_json()`**: Runtime patch set definition via JSON.
- **`HandoffConfig` extensions**: `patch_set_override`, `skip_preflight`.
- **`WarmInitPlan::from_handoff_config()`**: Unified config derivation.
- **Generation-dispatched experiment stages**: Stages 4-6 accept optional
  `chip` parameter with auto-detection from BOOT0.

### Fixed
- **R_X86_64_PC32/PLT32 normalization**: `normalize_relocations()` now handles
  type 2 (PC32) and type 4 (PLT32) 32-bit relocations — required for
  kernel 6.17+ which rejects nonzero values at these target locations.
- **`RetAtEntry` relocation conflict**: Patches at offset+5 (after ftrace
  `call __fentry__` preamble) to avoid clobbering PLT32 relocation
  displacement at bytes 1-4.
- **`RetAtEntry` return value**: Now emits `xor eax,eax; ret` (return 0)
  instead of bare `ret`, fixing nvidia init failures from garbage `eax`.

### Validated
- `nvsov` module loads alongside nvidia-580 (ksymtab stripped, co-load
  isolation NOPs applied, PC32/PLT32 relocations normalized). Module enters
  `nvidia_init_module` without kernel oops.
- `sovereign.snapshot` on both Titan Vs: consistent Tier 1.
- `sovereign.compare` on twin Titan Vs: one register delta (`THERM_GATE`).
- `sovereign.warm_handoff` on both Titan Vs: consistent 6/8 patches, Tier 1.
- 700 cylinder + 109 glowplug unit tests pass.

## Driver Infra Evolution — Exp 217 (May 21, 2026)

### Added
- **Exp 217 — TPC PRI Station Creation**: BAR0 path definitively CLOSED.
  Full ungating sequence + sw_nonctx.bin replay + PGRAPH reset all fail to
  create TPC PRI ring stations — confirmed firmware-mediated (GPCCS required).

### Validated
- BAR0-only Tier 2 sovereignty is impossible on Volta. Strategic pivot to
  nvidia-470 `nvsov` dual-load injection path (Exp 218).

## D-State Hardening & Kernel Health — Exps 214-216 (May 20-21, 2026)

### Added
- **`guarded_sysfs` module**: All sysfs writes through timeout-guarded
  helpers preventing D-state hangs. FLR disable/restore for warm swap.
- **`kernel_health.rs`**: 3-layer detection for kernel header corruption
  (autoconf.h freshness, struct layout probe, reference module cross-check).
- **`sovereign.kernel_health` RPC**: Exposes kernel health check via RPC.

### Fixed
- **Exp 216**: Corrupted `autoconf.h` (out-of-tree build shifted
  `struct module` by 24 bytes) — root-caused and auto-repairable.

## Live Hardware Warm Handoff & IOMMU Fix — Exp 213 (May 20, 2026)

### Added
- **`sovereign.classify_tier` RPC**: Generation-aware tier classification endpoint.
  Auto-detects SM version from BOOT0, looks up `GenerationProfile`, returns full
  evidence with profile metadata (CE class, register offsets, tier level).
- **IOMMU group sibling handling**: `iommu_group_siblings()` discovers sibling BDFs
  in the same IOMMU group. `rebind_siblings_to_vfio()` restores siblings after
  warm swap. Handoff Step 2 now unbinds all siblings before seeder bind.
- **VFIO anchor release**: `sovereign_warm_handoff` RPC releases VFIO anchor and
  cached device before spawning the handoff pipeline. Prevents IOMMU group
  deadlock from daemon-held FDs.

### Fixed
- **systemd `/tmp` access**: Added `/tmp` to `ReadWritePaths` in
  `toadstool-ember.service` for patched module staging.

### Validated
- `sovereign.classify_tier` on 2× Titan V: SM 70 auto-detected, CE class
  0xC3B5 (VOLTA_DMA_COPY_A), correct profile offsets, Tier 0 (cold) confirmed.
- `sovereign.init` cold start: 203-205ms on both cards.

### Discovered
- Cascading kernel failure: stuck nouveau probe → zombie VFIO thread → IOMMU
  group lock. Root cause: daemon held anchor FDs during handoff. Fixed.

## Sovereignty Consolidation Sprint — Exp 212 (May 20, 2026)

### Added
- **Warm capture → engine ungate wire**: `SovereignStrategy` now accepts golden-state
  `GrInitSequence` via `with_golden_sequences()`. `engine_ungate_sequences()` returns
  them for replay. `SovereignInitOptions.engine_init_path` loads golden-state JSON.
  RPC handler `sovereign.init` deserializes `GrInitSequence` from file path.
  Silicon-deistic bridge: learn from vendor driver once, replay forever.
- **`classify_tier_for_profile()`**: Generation-aware tier classification using offsets
  from `GenerationProfile` instead of hardcoded Volta values. Backward-compatible
  `classify_tier()` preserved as convenience.
- **`validate_ce_with_profile()`**: CE validation with generation-aware DMA class
  from `GenerationProfile`.
- **`GenerationProfile` tier fields**: 5 new fields (`fecs_pc_offset`,
  `gpc_broadcast_offset`, `ce0_base_offset`, `pgraph_status_offset`, `ce_class`)
  across all 11 generation profiles.
- **`ChipFamily::engine_label()`**: Default engine name for golden sequences.

### Changed
- **`pipeline_for_family()`** returns `Option<Box<dyn InitPipeline>>` — unknown
  families get `None` instead of silently falling back to VoltaInit.
- **`BootPipeline` warm heuristic** fixed from `count_ones() > 8` to
  `count_ones() >= 8 && pramin_accessible`, matching `is_warm_gpu()`.
  Applied to both `init_volta.rs` and `init_kepler.rs`.
- **Init pipeline hierarchy documented**: module-level doc clarifies that
  `InitPipeline`/`BootPipeline` is the thin cross-vendor probe surface,
  `sovereign_init` is the production NVIDIA orchestrator.

## Sovereign Driver Rotation Codified (May 20, 2026)

### Added
- **`sovereign.warm_handoff` RPC endpoint**: Single JSON-RPC call orchestrates the
  full driver rotation pipeline — find stock `.ko` → binary-patch → insmod → seeder
  bind → settle → warm swap to vfio-pci → tier classification → rmmod + cleanup.
  Per-GPU granularity. Operator never touches the kernel.
- **`cylinder::vfio::kmod`**: Kernel module lifecycle management via `Command`
  (`insmod`/`rmmod`/`modinfo`). Module presence check via `/sys/module/`.
- **`cylinder::vfio::module_patch`**: Binary NOP patcher with predefined `PatchSet`s
  (volta_warm_handoff, kepler_warm_handoff). Resolves ELF symbols via `nm`, patches
  function prologues at offset +5 (ret after ftrace — proven technique from Exp 211).
- **`cylinder::vfio::sovereign_handoff`**: 8-step pipeline orchestrator composing kmod,
  module_patch, sysfs bind/unbind, bridge pinning, FLR disable, and tier classification.
- **`glowplug::warm_init::ModuleSource`**: New enum on `SeederDriver` — `System` (stock
  module) or `Patched { stock_module, patch_set }` (diesel engine patches at runtime).
  `nouveau_titanv()` now uses `Patched` with `volta_warm_handoff`; K80 stays `System`.
- **Experiment index updated**: Exp 211 status → Complete. 17 RPC methods.

### Changed
- **Sovereign compute philosophy confirmed**: "with the primals we never need to touch
  the kernel of the OS ever." Driver rotation is a diesel engine operation, not a manual
  shell script. Display GPU (nvidia-580) is never disturbed. Conflicting drivers
  (nvidia-470) go in agentReagents VMs, never on bare metal.

## Warm Handoff Executed — Binary Patch Proven (May 20, 2026)

### Added
- **Binary-patch technique for kernel modules**: Patches function prologues at
  offset +5 (after ftrace call site) to bypass kernel 6.17 strict ELF relocation
  checks. Proven on stock `nouveau.ko` — source-patch and livepatch approaches
  both fail on 6.17+ due to `Invalid relocation target` errors.
- **Warm handoff execution results**: PMC_ENABLE preserved at 0x5fecdff1 (23 engines)
  through full nouveau→unbind→vfio-pci cycle. GPC broadcast fabric preserved.
  TPC/CE sub-units remain gated because nouveau lacks signed PMU firmware for Volta.
- **Phase C: DMEM Access Probe** — Volta HS lock is total: DMEM returns `0xDEAD5EC2`
  sentinel, writes silently dropped, queue HEAD read-only, interrupts blocked.
  PMU software path conclusively closed.

### Fixed
- **Dev environment stabilized**: cleaned ~470GB of cargo debug artifacts across
  5 projects. Fixed rustup proxy issue (Cursor AppImage `argv[0]` leaking into
  rustup). Killed 3 duplicate toadstool daemons.

### Changed
- **Exp 211 status**: All phases complete. Binary-patch technique validated.
  Warm handoff proven (PMC preserved, TPC gated on Volta due to missing firmware).
  K80 cross-gen promoted to Priority 1 (unsigned falcons — nouveau fully initializes).
  nvidia-470 handoff deprioritized (requires DRM contamination — violates
  non-disruption principle).

## Unreleased — River Delta Audit + Dark Forest Gate (May 19, 2026)

### Added
- **`s_dark_forest_gate` validation scenario**: Structural validation of all 7
  deploy graphs against the five Dark Forest Glacial Gate pillars (zero port
  exposure, BTSP enforced, secure_by_default, Songbird-only network, nucleated
  composition). 7×7 = 49 assertions. Default (no feature gate).
- **`secure_by_default = true`** added to all 7 deploy graph metadata sections,
  resolving the Dark Forest PENDING on the cross-spring parity scorecard.

### Changed
- **Scenario count**: 17/23 → 18/24 (18 default + 6 barracuda-local). Updated
  across README, specs, whitePaper, EXPERIMENT_INDEX, sporeprint, DOWNSTREAM_PATTERNS.
- **Experiment count drift fixed**: 207/208 → 210 in experiments/README.md,
  README.md directory tree, whitePaper/README.md.
- **May 17 handoffs archived** (3 files) per 48h rule.

## Fossilize & Reframe Sovereign Evolution (May 19, 2026)

### Added
- **Experiment 211 stub — PMU Mailbox Tier 2 Investigation**: Forward-looking
  experiment framing the PMU mailbox protocol at 0x10A000+ as the primary path
  to Tier 2 sovereign compute. Five-phase investigation plan (liveness, commands,
  injection, PGOB fallback, kernel patch fallback).
- **GAP-HS-107**: Tier 2 sovereign compute blocker — GPC power domain wall.
  Highest severity gap, cross-references Exp 210/211.

### Changed
- **Exp 210 reframed**: Added fleet sovereign status matrix (Titan V / RTX 5060 /
  K80 incoming / AMD RDNA2), four-priority remaining-work roadmap, K80 unsigned-falcon
  angle (may reach Tier 2 before Volta), VBIOS interpreter progress (422 ops,
  ~100 unknown opcodes).
- **README sovereign GPU row**: Reframed with evolution ladder reference
  (agnostic → atheistic → deistic), PMU mailbox as Tier 2 unlock, K80 incoming
  angle, SILICON_DEISM.md cross-reference.
- **EXPERIMENT_INDEX**: Narrative updated with sovereign evolution ladder framing,
  Exp 211 row added, experiment count 210 → 211.
- **wateringHole handoff**: Added evolution ladder section, fleet sovereign status
  matrix, and upstream next-steps broken down by team (toadStool, coralReef,
  primalSpring/composition).
- **PRIMAL_GAPS.md**: GAP-HS-107 added for Tier 2 blocker.

## Sovereign GPC Boundary Analysis (May 19, 2026)

### Added
- **Experiment 210 — Sovereign GPC Boundary Analysis**: Systematic analysis of
  hardware power domain boundaries after nouveau unbind. Fixed PTOP_DEVICE_INFO_V2
  parser for GV100 (runlist in kind==2 entries at bits [17:14]).
- **`discover_ce_runlist()`**: Standalone topology parser to find CE engine runlist
  ID from PTOP table at 0x22700+. Correctly handles GV100 multi-kind entry format.
- **`find_pbdma_for_runlist()`**: Runlist-indexed PBDMA lookup from 0x2390 table
  (bitmask extraction, returns lowest PBDMA for a given runlist).
- **`validate_ce()`**: End-to-end CE DMA validation — discovers CE runlist, creates
  channel on non-GR runlist, allocates DMA buffers, builds CE pushbuffer, submits
  via GPFIFO with CE PBDMA force-programming.
- **`sovereign.ce_validate` RPC**: New JSON-RPC method to trigger CE validation
  from server handler.
- **`SovereignTier` enum**: Tier 0 (Cold), Tier 1 (WarmInfrastructure), Tier 2
  (WarmCompute), Tier 3 (FullSovereign) — with `classify_tier()`, `TierCapabilities`,
  and `TierEvidence` structs.
- **Tier classification in warm_status**: `sovereign.warm_status` now returns
  `sovereign_tier` (level + name) based on live register probing.

### Validated
- **CE runlist discovery**: Runlist 10 identified for CE engine via fixed PTOP parser.
- **PBDMA mapping**: PBDMA 9 correctly mapped to CE runlist 10 from 0x2390 table.
- **CE channel creation**: Non-GR channel created on runlist 10 (first non-GR
  sovereign channel).
- **Tier 1 (Warm Infrastructure)**: VFIO bind, DMA, PFIFO scheduling, FECS liveness,
  channel creation, pushbuffer encoding all validated as sovereign.

### Known Gap
- **All engine domains power-gated**: CE0 (0x104000), GPCCS, PGRAPH, NVDEC all
  return PRI faults (0xbadfXXXX) after nouveau unbind. Tier 2 blocked by GPC
  power domain. PMU mailbox or kernel-level clock gating override required.
- **NVK does not support Volta**: nouveau Vulkan driver requires SM75+ (Turing).
  No local compute path for Titan V via DRM/wgpu.

## Sovereign VFIO Dispatch Bridge (May 18, 2026)

### Added
- **Experiment 209 — Sovereign VFIO Dispatch Bridge**: anchor-fd adoption
  bridges ember→dispatch gap. `ComputeDevice::adopt_anchor_fds()` trait
  method + `NvVfioComputeDevice::open_vfio_from_received()` enable VFIO
  dispatch when ember holds the VFIO group (EBUSY workaround).
- **`dup_received_fds_from_anchor()`**: Server helper that extracts
  anchor VFIO fds (device + iommufd/container) into `ReceivedVfioFds`.
- **FECS setup in adopted path**: `fecs_setup_channel()` called after
  `open_vfio_from_received()` to send INIT_CTXSW/BIND_CHANNEL/COMMIT.

### Validated
- **VFIO device from anchor fds**: `VfioDevice::from_received()` on dup'd
  anchor fds — BAR0 mmap, DMA backend, PFIFO channel, all working.
- **PBDMA pushbuffer submission on warm Titan V**: GP_PUT advances, GPFIFO
  entry ingested by hardware.
- **coralReef SM70 compile**: WGSL → SM70 SASS (240B, 15 instructions, 30ms).
- **compute.dispatch.submit → local_cylinder**: Full dispatch pipeline
  (alloc → upload → dispatch → sync → readback) returns `completed`.

### Known Gap
- **PGRAPH power gating**: FECS method mailbox (0x409xxx) returns PRI fault
  (0xbadf5545) after nouveau→vfio-pci handoff. FECS alive but GR engine
  power-gated — context switching blocked. PMC toggle insufficient.

## Reboot-Efficient Sovereign Evolution (May 18, 2026)

### Added
- **Experiment 208 — Reboot-Efficient Sovereign Evolution**: fd store chain
  proven end-to-end. Warm keepalive validated across daemon restarts.
- **`sovereign.warm_status` RPC**: Lightweight method reporting anchor state,
  boot state probe (via sysfs BAR0), PMC_ENABLE, PRAMIN sentinel, and fd store
  capability per GPU. No pipeline execution required.
- **Cold pipeline early-exit**: New `skip_cold_memory_training` flag on
  `SovereignInitOptions`. Cold GPUs skip doomed `memory_training` stage,
  reducing pipeline from ~14s to ~200ms (70× faster).
- **Anchor confirmation logging**: `sovereign.init` and `sovereign.profile`
  handlers now log `anchor_held` status at pipeline completion.

### Validated
- **systemd FileDescriptorStore**: 4 VFIO fds (2 device + 2 iommufd) stored
  on SIGTERM, retrieved on startup, anchors reconstructed — proven with cold
  and warm GPUs.
- **Post-reboot warm keepalive**: Both Titan Vs warm after power cycle
  (`compute_ready: true`, 3.9s pipeline). Warm state **persists across
  `systemctl restart`** — first-ever warm keepalive on sovereign hardware.
- **Twin-card warm profiles**: falcon_boot ACR dominates at 94.5% (3.7s).

### Optimized
- **Falcon warm preservation — VALIDATED**: Early falcon probe before
  `pgraph_reset` detects `WarmPreserved`/`WarmRunning` FECS state. When
  detected, skips `pgraph_reset` + `falcon_boot`. **Warm pipeline: 183ms**
  (76× faster than cold, 21× faster than pre-falcon-fix warm). Boot ROM
  FECS firmware survives VFIO device open (PC=0xB0+), so the first
  sovereign.init after power cycle benefits — no ACR boot needed.
- **FLR PC threshold (>= 0x40)**: Prevents false-positive warm detection
  from Function Level Reset residual PCs (0x00–0x10). ACR firmware entry
  points are 0x80+ (code section). Correctly classifies post-FLR cold
  state vs genuine running firmware.

## Unreleased — Sovereign Boot Abstraction + Profiling (May 18, 2026)

### Added
- **Experiment 207 — Sovereign Boot Abstraction + Twin Profiling**: Unified
  warm/cold boot model with `SovereignBootState` enum, `WarmKeepalive` facade,
  `sovereign.profile` RPC method, and twin-card profiling experiments.
- **`boot_state.rs`** (cylinder): `SovereignBootState::Warm`/`Cold`,
  `ColdBootReason` enum (PowerOnReset/BusReset/D3Cold/FdLost/Unknown),
  `BootCapability` flags, `probe_boot_state()` — single source of truth
  replacing scattered `is_warm_gpu()` + `FalconWarmState` + `warm_detected`.
  Module-level docs codify the **hardware line**: cold boot = power-on reset
  = boot ROM trains HBM2 = same wall NVIDIA faces. 10 unit tests.
- **`warm_keepalive.rs`** (ember): `WarmKeepalive` owning wrapper,
  `WarmKeepaliveRef` non-owning view, `KeepaliveStore`, `DmaSpec` —
  facade over VfioAnchor + Clutch + systemd fd store. 5 unit tests.
- **`sovereign_profile.rs`** (cylinder): `SovereignProfile` struct with
  per-stage µs timings, `RegisterSnapshot` (BOOT0/PMC/PTIMER/FECS/GPCCS),
  pre/post-pipeline snapshots. 2 serde tests.
- **`sovereign.profile` RPC**: JSON-RPC method returning detailed profiling
  data. Enables twin-card experimentation without log scraping.
- **Twin-card cold profiling**: Both Titan Vs profiled in cold state.
  Card 1: falcon=3697ms, memory=5419ms (11.3s total).
  Card 2: falcon=224ms, memory=10537ms (13.0s total).
  Hardware line confirmed — HBM2 untrained, only power cycle recovers.

### Changed
- **`sovereign_init.rs`**: Pipeline now runs `boot_state_probe` stage,
  `SovereignInitResult` gains `boot_state: Option<SovereignBootState>`.
  `warm_detected` kept for backward compatibility.
- **`FalconWarmState`**: Added `Serialize`/`Deserialize` derives for
  embedding in `SovereignBootState::Warm`.
- **Dispatch handler**: `try_engage_clutch` simplified using `WarmKeepaliveRef`.
- **Method list**: `sovereign.profile` added to core method registry.

## Unreleased — Falcon ACR DMA Boot Solved (May 17, 2026 PM)

### Added
- **Experiment 206 — Falcon ACR DMA Boot Solved**: Wired DMA backend to
  sovereign falcon boot, resolving the ACR HS boot blocker since Exp 080.
  `boot_falcon_hs` loads GR firmware via iommufd-mapped DMA buffers,
  GPCCS+FECS boot successfully. FECS cpuctl=0x10 (halted in command-wait).

### Changed
- **toadStool `sovereign.rs`**: Stateless handler now acquires `DmaBackend`
  from VFIO device. `EmberGateBypass` entered for self-opens. Sysfs path
  probes iommufd cdev for DMA as fallback.
- **toadStool `sovereign_init.rs`**: `gr_init` skipped after successful
  ACR boot (FECS already running, PIO re-upload would conflict).
- **toadStool `open.rs`**: iommufd/cdev failure log promoted to `warn`.

### Fixed
- **02:00.0 unbound**: Titan V #1 had no driver; bound to vfio-pci via
  `driver_override` enabling iommufd cdev path.
- **Stale coral-ember**: Legacy coral-ember processes holding VFIO cdev fds
  blocked iommufd bind (EINVAL). Killed and disabled coral services.

## Unreleased — Dual Titan V Twin Study Baseline (May 17, 2026 PM)

### Added
- **Experiment 205 — Dual Titan V Twin Study Baseline**: Second Titan V
  installed in former K80 PCIe slot. Both GV100 cards validated through
  sovereign.init pipeline with identical results: boot0=0x140000a1,
  PMC_ENABLE=0x5fecdff1, all registers match except PTIMER (expected).
  VBIOS ROM byte-identical (sha256=af04a2c6). IOMMU groups 65+32 provide
  clean isolation for independent VFIO access.

### Changed
- **`/etc/toadstool/glowplug.toml`**: K80 device entries replaced with
  Titan V #2 (`0000:49:00.0`, name: `titan-v-2`, role: `compute`,
  health_policy: `active`). Daemon header updated from "Dual Titan V + K80"
  to "Dual Titan V + RTX 5060".

## Unreleased — primalSpring Audit Absorption (May 17, 2026 PM)

### Added (lithoSpore R1–R4 absorption)
- **`s_anderson_parity` validation scenario**: Cross-tier parity for Anderson
  module (Python spectral_control.py vs Rust validate_spectral). 6 checks:
  Herman/Lyapunov, level statistics, 3D bandwidth, GOE→Poisson, dimensional
  hierarchy, mobility edge. `barracuda-local` feature gate.
- **`spectral_parity.py`**: Python-side cross-tier parity checker comparing
  spectral_control.json against Rust reference values. 6/6 ALL PASS.
- **`docs/DEGRADATION_BEHAVIOR.md`**: Consolidated degradation behavior
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

## Unreleased — VBIOS Interpreter Live Validation (Exp 204, May 17, 2026)

### Fixed (toadStool cylinder — VBIOS interpreter `opcodes.rs`)
- **`0x56` INIT_CONDITION_TIME stride**: 5 → 3. Our Maxwell+ path assumed an
  extended u16 delay field; nouveau uses stride 3 for all generations.
- **`0x3A` INIT_GENERIC_CONDITION stride**: Now uses `3 + size` for unknown
  conditions (sovereign mode has no display connector info).
- **`0x4F` INIT_TMDS stride**: 9 → 5. Corrected to match nouveau.

### Added (toadStool cylinder — VBIOS interpreter `opcodes.rs`)
- **Volta-specific opcodes**: `0xAC` (stride 13), `0xB0` (stride 10),
  `0xB1` (stride 3), `0x9E` (stride 1 prefix). Not in upstream nouveau.
- **Consecutive `0xFF` end-of-script**: Erased ROM regions terminate scripts
  gracefully instead of desyncing.
- **Graceful desync recovery**: 100 unknown opcodes → clean script termination
  with warning, allowing pipeline to continue.

### Validated (Experiment 204)
- **Cold Titan V VBIOS execution**: 422 ops, 231 BAR0 writes including PLL
  programming on real GV100 hardware.
- **PMC cold state confirmed**: PMC_ENABLE=0x00000000 on VFIO-bound GPU.
- **PGRAPH status improved**: 0x00000000 after PGOB (engines responding after
  PLL init).

## Unreleased — Warm/Cold Boot Convergence (Exp 203, May 17, 2026)

### Added (toadStool cylinder — `sovereign_strategy.rs`)
- **`FalconWarmState` enum**: `Cold`, `WarmPreserved`, `WarmRunning`, `Inconsistent` —
  classifies falcon thermal state for warm/cold dispatch.
- **`detect_falcon_warm_state()`**: Default trait method reads FECS CPUCTL/MAILBOX0/PC.
  `NvKeplerStrategy` overrides to always return `Cold`.
- **`pfifo_config()`**: Default trait method selects `PfifoInitConfig` based on
  `FalconWarmState` (warm_fecs_alive / warm_handoff / default).

### Changed (toadStool cylinder — `sovereign_stages.rs`)
- **`falcon_boot()` signature**: `warm_detected: bool` replaced with
  `warm_state: FalconWarmState`. Dispatches on enum variants instead of
  inline BAR0 register reads.

### Added (toadStool cylinder — `pfifo.rs`)
- **`PfifoInitConfig::for_thermal_state(warm, fecs_preserved)`**: Unified config
  selection method replacing scattered if/else branches.

### Changed (toadStool cylinder — VBIOS interpreter `opcodes.rs`)
- **6 PLL opcodes activated**: 0x79 (INIT_PLL), 0x4B (INIT_PLL_INDIRECT),
  0x34 (INIT_RAM_RESTRICT_PLL), 0x4A (variant), 0x59 (INIT_PLL2),
  0x87 (INIT_RAM_RESTRICT_ZM_REG) — now perform actual BAR0 writes with
  pre-computed coefficient words from the VBIOS ROM.
- **4 register copy opcodes activated**: 0x88 (INIT_RAM_RESTRICT_ZM_REG_GROUP),
  0x8F (variant), 0x90 (INIT_COPY_ZM_REG), 0x5F (INIT_COPY_NV_REG) —
  previously stub-handled, now perform register reads/writes.

### Documentation (toadStool cylinder)
- **`NvGspBridge`** and **`GspBridge` trait**: Documented as frozen dependency —
  firmware blobs are pinned artifacts, upload mechanisms are hardware-defined,
  Rust code evolves glacially. Future bridges (AMD, NPU) follow the same pattern.

## Unreleased — Experiment Surface Rewire (Exp 202, May 17, 2026)

### Added (toadStool cylinder — `sovereign_strategy.rs`)
- **`ProbeIdentity`**: Vendor-neutral struct for device identity probing.
- **`probe_identity()`**: Default trait method on `SovereignStrategy` — delegates
  to `bar0_probe()` for NVIDIA, overrideable for AMD/NPU.
- **`verify_device()`**: Default trait method — delegates to existing `verify()`
  for NVIDIA, overrideable for vendor-specific health checks.
- **`pre_channel_init()`**: Default trait method — no-op for Kepler, runs
  CG sweep + PRI recovery + PGOB ungating for Volta+ (NvAcrStrategy).

### Changed (toadStool cylinder — `sovereign_stages.rs`)
- **`falcon_boot()`**: Now accepts `FalconBootStyle` parameter and dispatches
  on enum variant instead of internally re-deriving generation via
  `profile_for_sm()` / `is_kepler()`.
- **`NoFalcons` variant**: Immediate success path for hardware without falcon
  microcontrollers.

### Changed (toadStool cylinder — `sovereign_types.rs`)
- **`SovereignInitResult` fields renamed**: `chip_id` → `identity_chip`,
  `boot0` → `identity_raw`, `hbm2_writes` → `training_writes`. Serde aliases
  preserve backward compatibility with persisted JSON.
- **`HaltBefore` expanded**: `CgSweep` and `PgobUngate` variants added between
  `PmcEnable` and `MemoryTraining`, enabling observation of raw post-PMC
  state and post-CG-sweep state during experiments.

### Changed (toadStool cylinder — `sovereign_init.rs`)
- Pipeline stage 1 renamed from `bar0_probe` to `identity_probe`, delegates
  to `strategy.probe_identity()`.
- Verify stage delegates to `strategy.verify_device()`.
- New halt points wired for `CgSweep` and `PgobUngate`.

### Changed (toadStool server — `dispatch/mod.rs`)
- `sovereign_init_ember` calls `strategy.pre_channel_init()` before
  `sovereign_init`, running CG sweep on raw BAR0 before factory channel
  creation.

## Unreleased — Volta Cold Boot CG Sweep (Exp 201, May 17, 2026)

### Added (toadStool cylinder — `sovereign_stages.rs`)
- **`cg_sweep()`**: Disables ELCG/BLCG/SLCG across PTHERM, PMC CG slots,
  PRIV_RING, PFB, PCLOCK, per-FBPA (4), and per-LTC (6) domains. Writes
  `CG_DISABLE` (0x0) to each register. Returns `CgSweepResult` with change
  and fault counts.
- **`pri_bus_recover()`**: Probes all PRI domains via `PriBusMonitor`,
  acknowledges PRIV_RING faults, and re-probes. Clears stale backpressure
  after CG transition.
- **`pgob_ungating()`**: Delegates to `bridge.pgob_disable()` for PGRAPH
  GPC broadcast ungating. Previously only called from Kepler's falcon_boot.

### Changed (toadStool cylinder — `sovereign_init.rs`)
- **Pipeline stages 2b/2c/2d**: New `cg_sweep` → `pri_recovery` →
  `pgob_ungating` stages run for all non-NoAcr generations (Volta+) between
  `pmc_enable` and `memory_training`. Unblocks cold HBM2 training and
  falcon DMA boot by clearing `0xBADF` PRI faults from clock-gated domains.

### Changed (toadStool cylinder — `registers.rs`)
- **`cg` module**: Removed `#[expect(dead_code)]` — constants now actively
  used by `sovereign_stages::cg_sweep`.

### Architecture Note
The CG sweep, PRI recovery, and PGOB ungating were already implemented in
glowplug's warm path (`warm.rs::run_step_clock_gating`). This change
extracts the same register-level logic into `MappedBar`-only functions
(no GlowPlug dependency) and wires them into the sovereign cold pipeline.
This is the first step toward warm/cold convergence — both paths now share
the same CG ungating constants and PGOB sequence.

## Diesel Engine Power Safety (Exp 200, May 17, 2026)

### Added (toadStool cylinder — `generation.rs`)
- **`PowerSafetyProfile` struct**: Generation-aware PMC_ENABLE sequencing
  policy with `initial_pmc_mask`, `full_enable_after_devinit`, and
  `rollback_on_devinit_failure` fields.
- **`PowerSafetyProfile::PRE_FIRMWARE`**: Conservative mask (0xC000_2030) for
  Kepler/Maxwell — enables only PPCI + PBUS + PTIMER + PFIFO. Rolls back
  PMC_ENABLE on devinit failure.
- **`PowerSafetyProfile::FIRMWARE_MANAGED`**: Full ungating (0xFFFF_FFFF)
  for Pascal+ where firmware manages power rails.
- **`GenerationProfile.power_safety`**: New field on all 10 generation
  profiles, wired to the appropriate safety level.

### Changed (toadStool cylinder — `sovereign_stages.rs`)
- **`pmc_enable()`**: Now accepts `&PowerSafetyProfile` and writes only the
  profile's `initial_pmc_mask` instead of blanket 0xFFFF_FFFF. Returns
  `PmcEnableResult` for rollback tracking.
- **`pmc_enable_rollback()`**: New function — restores PMC_ENABLE to its
  pre-pipeline value when devinit fails on pre-firmware GPUs.
- **`pmc_enable_full()`**: New function — writes 0xFFFF_FFFF post-devinit
  for firmware-managed generations only.

### Changed (toadStool cylinder — `sovereign_init.rs`)
- **Pipeline restructured**: Generation profile now resolved before stage 2
  (was after stage 3). PMC_ENABLE is staged: conservative mask in stage 2,
  full ungating in new stage 3b (post-devinit, firmware-managed only).
- **Devinit failure rollback**: If memory training fails on a
  `rollback_on_devinit_failure` generation, PMC_ENABLE is restored to its
  pre-pipeline value before returning.

### Root Cause — K80 Fire (Exp 199 Post-Mortem)
Writing `0xFFFF_FFFF` to PMC_ENABLE on a cold Kepler GPU with uninitialised
GDDR5 instantly ungated all engine clock domains, causing inrush current
beyond the aged VRM's capacity. The new staged approach prevents this by
limiting initial PMC_ENABLE to essential buses and rolling back on failure.

## Diesel Engine Sovereign Boot (Exp 199, May 16, 2026)

### Added (toadStool cylinder — `lib.rs`)
- **`ComputeDevice::bar0()`**: Returns `Option<&MappedBar>` to expose cached
  BAR0 mapping from VFIO devices. Default `None`.
- **`ComputeDevice::dma_backend()`**: Returns `Option<&DmaBackend>` for ACR
  falcon boot. Default `None`.

### Added (toadStool cylinder — `compute_device.rs`)
- **`NvVfioComputeDevice::bar0()`**: Returns `&VfioDispatchState.bar0`.
- **`NvVfioComputeDevice::dma_backend()`**: Returns `&VfioDispatchState.dma_backend`.

### Added (toadStool server — `dispatch/mod.rs`)
- **`DispatchHandler::sovereign_init_ember()`**: Runs sovereign pipeline using
  diesel engine's cached `MappedBar` + `DmaBackend`. Dynamically selects
  `NvGspBridge` vs `StubGspBridge` based on firmware availability.

### Changed (toadStool server — `handler/mod.rs`)
- **`sovereign.init` routing**: When `bar0_source=ember`, routes to
  `DispatchHandler::sovereign_init_ember()` instead of stateless handler.

### Changed (toadStool cylinder — `nv_gsp_bridge.rs`)
- **`NvGspBridge::acr_boot()`**: Real implementation using `boot_falcon_hs`
  for GPCCS then FECS when DMA backend is provided. Replaces stub.
- **`NvGspBridge::supports_acr()`**: Returns `true` when firmware is available.

### Experiment 199 Results
- **Titan V**: bar0_probe/pmc_enable OK, memory_training FAILED (PGRAPH CG
  gated, PRAMIN PRI faults). FECS HS boot times out at pc=0x8 (FBIF PRI).
- **K80 x2**: bar0_probe OK (GK210 identified), PMC_ENABLE 0xc0002020→0xfc37b1ef,
  VBIOS interpreter ran 268 writes but PRAMIN still dead.
- **Architecture validated**: `bar0_source=ember` pipeline borrows BAR0/DMA
  from diesel engine. No EBUSY conflicts.

### Remaining Sovereign Boot Issues
- **Titan V cold**: PGRAPH CG ungating needed before falcon DMA boot.
  HBM2 controller inaccessible post-FLR.
- **K80 cold**: VBIOS script 1 hits unknown opcode 0x0a. GDDR5 training
  incomplete — PRAMIN inaccessible after 268 register writes.
- **coral-ember coexistence**: Both services cannot hold VFIO devices
  simultaneously. Needs lifecycle coordination or shared VFIO ownership.

## Post-BootPipeline Documentation + Cross-Team Handoff (May 16, 2026)

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
