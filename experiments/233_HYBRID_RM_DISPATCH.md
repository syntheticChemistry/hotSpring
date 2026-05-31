# Experiment 233: Hybrid RM Dispatch — Channel Retry with Diesel Engine

**Date:** 2026-05-29
**Status:** IN PROGRESS
**Hardware:** Dual Titan V (GV100, SM70) — 0000:02:00.0 + 0000:49:00.0
**Prerequisite:** Exp 229 Run #9 (Tier 2 WarmCompute proven), Exp 232 (crash defense matrix)

## Objective

Retry RM compute channel creation (Exp 229 `--channel` mode) under diesel
engine crash protections. The only attempt with `--channel` (Exp 229 Attempt #2)
had failed DEVINIT (PMC_ENABLE=0x40001121, 5 engines). Run #9 proved full
DEVINIT works (23 engines) but never retried `--channel`.

Hypothesis: with diesel engine protections (IRQ storm detector, kernel sentinel,
catalyst watchdog, 20-patch build), a clean power state produces full DEVINIT,
and RM channel creation succeeds. If the channel is established before warm swap,
FECS ctx-switch state machine should process our dispatch post-swap.

## Strategy

### Phase 1: Validate Crash Protections
1. Confirm toadstool-ember service starts cleanly
2. Confirm kernel sentinel + catalyst watchdog active
3. Run `sovereign.warm_handoff` on Titan V #1 — verify Tier 2 with no lockups

### Phase 2: Channel Creation Probe
1. The pipeline already passes `create_channel=true` for catalyst strategies
2. Monitor rm_trigger output for `device_alloc` status
3. If channel succeeds: capture channel_id, work_submit_token, PCCSR status
4. If channel fails with 0x22: probe RM state to identify missing infrastructure

### Phase 3: Post-Swap Dispatch
1. If channel was established: warm swap to vfio-pci
2. Create sovereign channel and attempt dispatch
3. Check PCCSR status: PENDING (0x1) vs ACTIVE (0x5+) vs ON_PBDMA (0x6)
4. Submit coralReef SM70 shader, readback buffer

## Success Criteria
- RM channel creation: `device_alloc status=0x00` (not 0x22 or 0xdeadbeef)
- Post-swap PCCSR: channel status >= 5 (ACTIVE/ON_PBDMA)
- Non-zero shader readback

## Risk Mitigation
- Diesel engine protections live (Exp 232 defense matrix)
- If lockup: kernel sentinel captures forensics before system loss
- Watchdog 450s timeout prevents infinite hangs
- IRQ storm detector pre-emptive quench

## Files Changed
(none yet — using existing infrastructure)

## Results

### Phase 1: Crash Protection Validation

**Pre-reboot state (2026-05-29 07:50):**

- toadstool-ember: `active (running)` for 13h
- Kernel sentinel: active (test crash report captured at startup)
- PCIe keepalive: 20 bridges, 5 GPUs pinned
- Both Titan Vs: `vfio-pci` bound
- RTX 5060: `nvidia` bound (host GPU)

**Zombie module detected:**
- `nvsov 35635200 -1 - Unloading` — B2 crash vector from previous session
- Preflight guard correctly halted pipeline: "module 'nvsov' is stuck"
- `rmmod -f nvsov` → EBUSY (expected — kernel module state machine is terminal)
- **Requires reboot cycle to clear**

Diesel engine working as designed — preflight caught zombie, prevented lockup.

### Run #1 — Post-Reboot (2026-05-29 08:20)

**Boot state (5min uptime):**
- nvsov: CLEAR (zombie purged by reboot)
- toadstool-ember: `active (running)` since 08:15
- Both Titan Vs: `vfio-pci` bound
- RTX 5060: `nvidia` bound (host GPU)
- PCIe keepalive: 20 bridges, 5 GPUs pinned

**Pipeline execution:** `sovereign.warm_handoff` on 0000:02:00.0

**Results:**
- **Preflight:** CLEAN — no zombie, no stuck modules, kernel healthy
- **Module prep:** 20/20 patches applied (nvidia→nvsov rename + catalyst NOP set)
- **DEVINIT:** PMC_ENABLE 0x40000121 → 0x5fecdff1 (popcount 4→23) — FULL INIT
- **RM trigger:** channel=true, exit=0, 7.2s
- **Settle:** 60s, PMC_ENABLE=0x5fecdff1 (stable)
- **Warm swap:** nvsov → vfio-pci (7s poll)
- **Catalyst capture:** 63,310 alive registers captured
- **PRI ring recovery:** PMC preserved, PGRAPH=ON, ring_status=0x1, FECS/GPCCS accessible, IMEM 8/8 alive
- **Tier:** `warm_compute` (Tier 2) — CONFIRMED
- **Total time:** 79.3s
- **System lockup:** NONE — system survived entire pipeline

**RM Channel Evidence:**
- `steps_completed: 3` (out of ~18 channel creation steps)
- `channel_id: null` — channel was NOT established
- `work_submit_token: null`
- `all_ok: false`

**Interpretation:** The 20-patch NOP set allows full DEVINIT (23 engines) but blocks
RM's internal device registration. The NOP'd functions (`nv_cap_init`, `nv_procfs_init`,
`nv_acpi_init`, `nv_cap_create_dir_entry`, etc.) are needed by RM to populate its GPU
manager bookkeeping. Without them, `device_alloc` (step 5) fails with status=0x22
(NV_ERR_OBJECT_NOT_FOUND) — same as Exp 229 Attempt #2. The first 3 steps succeed:
`card_info`, `root_client`, `gpu_attach_ids_ctrl` — these are RM-level operations that
don't require device-specific infrastructure.

**Crash protection performance:**
- **Kernel sentinel:** Caught 20 crash signatures during module_cleanup (A6 vector —
  `_nv011358rm` page fault in `rm_shutdown_rm` path). Sentinel correctly auto-stopped
  after 20+ signatures. Non-fatal — kernel stayed alive.
- **Catalyst watchdog:** Detected zombie (B2 vector — `Unloading, refcnt=-1`), logged
  and deactivated gracefully.
- **IRQ quench:** GPU interrupts quenched pre-close, preventing IRQ storm.
- **System stability:** Fully operational post-handoff despite kernel oops.

**FECS context switch init:**
- `status=0, mb0=0x00000001, tpc0=0x000000cf, gpc_en=0x00000000`
- FECS acknowledged ctxsw init (status=0) but GPC enables=0x0 — no GPCs enrolled
- `tpc0=0x000000cf` — TPC status shows data but no active context

### Analysis: The Patch Set Dilemma

The 20-patch NOP set creates an inherent contradiction:
1. We NOP `nv_cap_init`/`nv_procfs_init`/etc. to prevent the renamed module from
   colliding with the host nvidia driver's procfs and capability entries
2. But RM needs these init paths to populate its device registry, which is required
   for `device_alloc` and everything downstream (subdevice, VA space, channel, etc.)

**Options for Exp 234:**
1. **Selective un-NOP:** Remove NOP patches for `nv_cap_init` and `nv_cap_create_dir_entry`
   only — these create `/dev/nvidia-caps/` entries that may not collide if we namespace
   the module name. Risk: procfs collision with host nvidia.
2. **Post-swap RM channel injection:** Instead of creating the channel pre-swap, create
   it post-swap using direct BAR0 register writes to the PCCSR/PBDMA/runlist state
   machine. This bypasses RM entirely. Risk: FECS ACR still validates.
3. **RM state injection:** After RM initializes the GPU (DEVINIT succeeds), manually
   populate the missing RM data structures by writing to kernel memory (via /dev/mem
   or module parameter). Risk: kernel memory corruption, ABI instability.
4. **Dual-module cooperation:** Keep the NOP'd nvsov for DEVINIT, then use a second
   micro-module that opens the HOST nvidia's device nodes to create a channel on the
   same GPU. Risk: host nvidia may reject the GPU since it's not bound to it.
5. **os_is_administrator + nv_cap bypass:** The existing catalyst patch set already
   includes `os_is_administrator` Ret1AtEntry (returns admin=true) and
   `nv_cap_validate_and_dup_fd` Ret1AtEntry (bypasses cap validation). The
   `nv_cap_create_dir_entry` and `nv_cap_create_file_entry` now return Ret1AtEntry
   (non-NULL fake handle). The root cause of 0x22 may not be caps alone — it could
   be the GPU's internal device registry populated during `nv_pci_probe`, which runs
   within `init_module` and needs the cap infrastructure to register the device.
   Investigate whether the device_alloc failure is actually
   `NV_ERR_OBJECT_NOT_FOUND` because the GPU was never registered in RM's device table
   (not because of capability denial).

### Post-Run #1 Code Revalidation (2026-05-31)

**Crash protection rewiring (S283):**
- Module-cleanup watchdog signals wired: `cleanup.rs` now emits
  `PipelineSignal::EnterModuleCleanup` / `ExitModuleCleanup` to the catalyst
  watchdog, activating high-frequency zombie module monitoring
- `catalyst_boot.rs` now activates watchdog (120s timeout) with heartbeats —
  previously ran unprotected
- `warm_handoff.rs` upgraded to `execute_handoff_with_signals` with closures
  calling `catalyst_watchdog::enter/exit_module_cleanup`
- Dead `experiment_stage_4`/`experiment_stage_6` wrappers removed (superseded
  by `_with_chip` generation-aware variants)
- Patch set test updated: `nvidia_warm_handoff` 17→19 targets
- Full workspace validation: 0 errors, 0 clippy warnings, 9,175 tests passed

**Crash protection layer integrity (verified post-refactor):**

| Layer | Status |
|-------|--------|
| IRQ clutch | INTACT — engaged pre-unbind, MISFIRE fallback on failure |
| PRI fault guard | INTACT — RPC-level SBR + pipeline write gate |
| Zombie module | INTACT — rmmod skip for all catalyst patch sets |
| Catalyst watchdog | FULLY WIRED — heartbeats + module-cleanup signals connected |
| Kernel sentinel | INTACT — crash reports, emergency quench cross-trigger |
| Forensic breadcrumbs | INTACT — correct paths, sync'd writes |

All crash protection code paths preserved through the Deep Debt Evolution Pass
(6-wave refactoring, May 28-31). No functional regressions.

