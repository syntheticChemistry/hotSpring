# Experiment 232: Crash Vector Reprofile — Diesel Engine Defense Matrix

**Date:** 2026-05-28
**Status:** COMPLETE
**Hardware:** Dual Titan V (GV100, SM70) — 0000:02:00.0 + 0000:49:00.0
**Prerequisite:** Exp 230 (Diesel Abstraction Revalidation — COMPLETE)

## Objective

Systematically reprofile every known and potential crash vector through the
abstracted diesel engine. Categorize each by: severity, current defense status,
remaining exposure, and diesel engine coverage.

## Crash Vector Taxonomy

### Category A: CONFIRMED KILLS (caused hard lockup, fix proven)

| # | Vector | Mechanism | Fix | Defense Layer | Proven Run |
|---|--------|-----------|-----|---------------|------------|
| A1 | IRQ storm (INTR_EN quench to RO register) | Wrote 0 to 0x140 (read-only on Volta) → no effect → level-triggered INTx storm | Write 0xFFFFFFFF to INTR_EN_CLEAR@0x180 | `InterruptProfile.disable_offset()` | Exp 229 #5→#6 |
| A2 | nvidia_close re-enables INTR_EN | rm_trigger quench succeeds, then nvidia_close teardown re-enables all sources | Post-exit dual quench from pipeline (after `cmd.output()` returns) | `pmc::quench_interrupts()` + `pmc::intx_disable()` | Exp 229 #6→#7 |
| A3 | nv_dev_free_stacks use-after-free | rm_disable/shutdown NOP'd but nv_close_device still frees RM thread stacks → kernel corruption | `nv_close_device` RetAtEntry (skip all per-device teardown) | Patch set target #1 | Exp 229 #7→#8 |
| A4 | nv_pci_remove os_delay hang | PCI unbind callback polls for GPU quiescence forever (nv_close_device NOP'd) | `nv_pci_remove` RetAtEntry | Patch set target #2 | Exp 229 #8→#9 |
| A5 | pci_lock deadlock (keepalive) | Bridge keepalive config reads take pci_lock while pipeline holds GPU | `HandoffExclusionGuard` — excludes BDF+siblings+bridges from keepalive reads | Keepalive exclusion RAII | Exp 229 #1-3 |
| A6 | irq_domain_remove crash (cleanup_module NOP) | cleanup_module RetAtEntry prevents pci_unregister_driver teardown → kernel oopses in irq_domain_remove + msi_device_data_release during unbind | **REVERTED** — do NOT patch cleanup_module | Zombie B2 accepted as non-fatal | Exp 232 Probe 2 |

### Category B: CONFIRMED HANGS (caused D-state / degraded, fix proven)

| # | Vector | Mechanism | Fix | Defense Layer |
|---|--------|-----------|-----|---------------|
| B1 | VFIO anchor session leak | Previous VFIO anchor session holds kernel refs → unbind enters D-state | Release all BAR0 fds before pipeline start | `released leaked BAR0 resource0 fds` |
| B2 | nvsov module zombie (refcount -1) | Patched cleanup_module cannot properly tear down → module stuck in Unloading | Preflight detection + halt with clear message | Pipeline preflight check |
| B3 | driver_override sysfs hang | D-state processes from B1/B2 block sysfs writes during rebind | RAII + fire-and-poll unbind with deadline | `fire-and-poll unbind` (330s deadline) |

### Category C: POTENTIAL VECTORS (theorized, not yet triggered)

| # | Vector | Mechanism | Current Defense | Exposure |
|---|--------|-----------|-----------------|----------|
| C1 | Watchdog false-positive kill | Watchdog times out during legitimate long settle → emergency quench mid-pipeline | 450s timeout + heartbeat callbacks at 13 step boundaries | LOW — heartbeats reset timer |
| C2 | DRM/5060 interrupt cross-contamination | Catalyst pipeline somehow infects host GPU (5060 on nvidia driver) | Separate PCI domains, no shared IRQ lines | LOW — not observed since Exp 229 #3 |
| C3 | PRI ring fault cascade | Post-swap BAR0 reads trigger PRI faults that propagate to host bus | Read-only BAR0 access, PRI recovery step | MEDIUM — faulted_domains=1 seen in Exp 230 |
| C4 | Kernel OOM from BAR0 snapshot | 25MB catalyst snapshot allocation under memory pressure | No defense | LOW — system has 256GB RAM |
| C5 | PCIe bridge power state change | Bridge enters D3 during handoff → GPU unreachable | Bridge hierarchy pinning at startup | LOW — 10 hierarchies pinned |
| C6 | Module compilation failure on kernel upgrade | kbuild breaks (missing cpufeaturemasks.awk), DKMS fallback may not exist | Layer 2 DKMS fallback | MEDIUM — seen in Exp 230 (DKMS worked) |
| C7 | INTR_PENDING accumulation | Pending interrupts (currently 0x11100000) with no handler → fire on next MSI enable | Quench reads INTR@0x100 to clear edge-triggered | MEDIUM — 3 sources pending now |
| C8 | Concurrent handoff race | Two handoff RPCs for same BDF simultaneously | Global handoff mutex (per-BDF) | LOW — mutex exists |
| C9 | RM channel partial failure → stale GPU state | rm_trigger device_alloc fails (status 0x22) leaving RM in half-initialized state | GPU warm via rm_init_adapter regardless of channel status | LOW — GPU warms even on partial RM init |
| C10 | Firmware IMEM capture during falcon execution | Reading FECS IMEM while falcon is running → inconsistent snapshot | Post-PRI-recovery capture (falcon halted at known PC) | LOW — current approach is sound |

### Category D: CROSS-GENERATION VECTORS (Kepler/Maxwell/Pascal specific)

| # | Vector | Mechanism | Defense | Status |
|---|--------|-----------|---------|--------|
| D1 | PMC_ENABLE inrush (K80 fire, Exp 199) | Full PMC_ENABLE on cold Kepler ungates all clocks → VRM overload | `PowerSafetyProfile.initial_pmc_mask` | Proven (Exp 200) |
| D2 | Pre-Volta INTR_EN is direct-writable | 0x140 is R/W (not RO like Volta) → quench writes to 0x140 directly | `InterruptProfile::PRE_VOLTA` dispatches to 0x140 | UNTESTED on hardware |
| D3 | No SET/CLEAR register pair | Pre-Volta lacks 0x160/0x180 → writing to CLEAR register is undefined | `InterruptProfile.disable_offset()` returns 0x140 for PRE_VOLTA | UNTESTED on hardware |
| D4 | VBIOS devinit is power sequencer | No hardware power sequencing on Kepler/Maxwell | `PowerSafetyProfile` gates PMC_ENABLE writes | Proven (Exp 200) |

## Current System State (post Exp 232 Probe 2 — fresh boot)

```
GPU 0000:49:00.0 (Titan V / GV100):
  Driver:       nvidia (fresh boot, not yet handed off)
  Build:        20 patches (cleanup_module REVERTED after A6)

Module nvsov:
  State:        NOT LOADED (clean boot after power cycle)
  Zombie kill:  force rmmod (O_NONBLOCK|O_TRUNC) available in preflight
  Prevention:   cleanup_module patch abandoned — zombie accepted as non-fatal
```

## Reprobing Plan

### Phase 1: Pending Interrupt Stress (C7)
Fire a second handoff on the same boot cycle while INTR_PENDING=0x11100000.
Validates that the pipeline's post-exit quench handles accumulated pending IRQs.

### Phase 2: Watchdog Timeout Test (C1)
Inject an artificial delay (>450s) to verify watchdog emergency kill path.
Should quench GPU and kill pipeline without system lockup.

### Phase 3: Rapid Handoff Cycling
Fire 3 handoffs back-to-back (with reboot between each to clear zombie module).
Stress-tests the full lifecycle including module load/unload/zombie detection.

### Phase 4: Cross-Gen Probe (Exp 231 — K80)
Test `InterruptProfile::PRE_VOLTA` on actual Kepler hardware:
- Verify quench writes to 0x140 directly (not 0x180)
- Verify `PowerSafetyProfile` prevents PMC_ENABLE inrush
- Validate GPC topology dispatch (2 GPCs for GK210 vs 6 for GV100)

## Probe Results

### Probe 1: Zombie module detection (B2 + C7)

**Fired**: Second handoff RPC on same boot, while INTR_PENDING=0x11100000 (3 sources)
**Result**: Preflight halt in 0ms — `"module 'nvsov' is stuck (Unloading/negative refcount) — reboot required"`
**System**: Stable, no lockup, kernel clean (no dmesg errors)
**INTR analysis**: Pending bits 20 (FIFO), 24 (PGRAPH), 28 (CE) — none in INTR_EN mask (0x200 = bit9/PBDMA only) → safe, will not fire
**Verdict**: B2 properly detected. C7 not a risk while INTR_EN masks pending sources.

### Probe 2: cleanup_module RetAtEntry patch — NEW LOCKUP VECTOR (A6)

**Fired**: Clean boot (no zombie), handoff with `cleanup_module` added as RetAtEntry patch (21 patches total)
**Result**: HARD LOCKUP — required full power cycle

**Timeline reconstruction from `journalctl -b -1`:**
```
14:58:05  sovereign.warm_handoff RPC received (nvsov, 21 patches applied)
14:58:06  module patched: applied=21, total=21
14:58:06  nvsov loaded, chardev major=507
14:58:11  rm_trigger completed (exit=0, GPU warmed: PMC 0x5fecdff1)
14:58:11  interrupt quench complete (INTR_EN 0x7fffffff → 0x0, pending=0x0)
14:59:14  catalyst capture complete, tier=WarmCompute
14:59:14  fire-and-poll unbind initiated
14:59:14  KERNEL OOPS #1: irq_domain_remove+0xdc/0x100
14:59:14  KERNEL OOPS #2: msi_device_data_release+0x42/0x50
14:59:16  unbind "completes" (2s) — zombie kernel now
14:59:16  drivers_probe sent (vfio-pci rebind attempted)
<LOCKUP — no further log entries>
```

**Root cause**: The `cleanup_module` RetAtEntry patch prevents nvidia's
`pci_unregister_driver()` from ever running. When the kernel unbinds the PCI
device (fire-and-poll writes to `sysfs/unbind`), it tries to tear down the MSI
IRQ domain that nvidia created during `pci_enable_msi()`. But the nvidia
teardown functions that prepare the IRQ domain for clean removal (`nv_pci_remove`,
`free_irq`, `pci_disable_msi`) were NOP'd by patches A1-A5 — so the
`irq_domain` is in an inconsistent state and the kernel page-faults trying to
walk it.

**Key insight**: `cleanup_module` is the *module exit* path (called by
`delete_module(2)` syscall, i.e. `rmmod`). It's NOT called during PCI unbind.
Patching it has NO EFFECT on the unbind path — the zombie still happens because
the PCI remove callback (already NOP'd) means `nvidia_close` never properly
decrements the module refcount. But patching cleanup_module *adds* a new crash:
when the kernel eventually calls `pci_unregister_driver()` during module
unload, the MSI/IRQ domain is corrupt.

**Fix**: Reverted cleanup_module patch (back to 20 targets). The zombie module
(B2) is an accepted non-fatal state — the preflight guard detects it and the
force rmmod (O_NONBLOCK|O_TRUNC) handles burial when possible. A reboot clears
permanent zombies.

**New vector cataloged**: A6 — irq_domain_remove crash from cleanup_module NOP

### Probe 4: C7 pending interrupt analysis + C3 PRI fault analysis (post Probe 3 boot)

**Method**: Analyzed BAR0 catalyst snapshot (62,571 alive regs, 641,088 total) captured
during Probe 3 handoff, plus live PCIe/kernel state.

**C7 — Pending Interrupt Accumulation:**
```
INTR_0:    0x10000000 (bit 28 = CE pending)
INTR_EN:   0x00000200 (bit 9 = PBDMA only)
Overlap:   NONE — pending sources fully masked by INTR_EN
```
**Verdict**: C7 is SAFE. Only 1 pending source (CE, bit 28), and it's not in the
INTR_EN mask. Even if MSI/INTx were re-enabled, PBDMA (bit 9) is the only armed
source and has no pending interrupt. vfio-pci handles its own interrupt registration.

**C3 — PRI Ring Fault Cascade:**
```
Total registers scanned: 630,784
PRI fault registers:     255,023 (40.4%)
Healthy alive registers:  97,056
Fault types:
  0xbadf5040: 179,531  (PRI timeout — expected for ungated domains)
  0xbadf1100:  71,300  (PRI ring error — hub decode fault)
  0xbadf1300:   2,274  (PRI ring — invalid register)
Faulted domains: GR/PGRAPH (27k), FB/FBPA (15k), CE (9k), FIFO/PBDMA (3k), THERM, PTIMER
```
**Host-side impact**: ZERO
- PCIe AER: no errors (CorrErr=0, NonFatalErr=0, FatalErr=0)
- PCIe link: downgraded to 2.5GT/s x8 (expected — no link retrain post-swap)
- 5060 host GPU: healthy (52C, 25W, driver 580.126.18)
- Kernel dmesg: clean — no oops, no MCE, no NMI

**Verdict**: C3 is SAFE. PRI faults are GPU-internal and do NOT propagate to the
PCIe bus or host. The IOMMU/vfio-pci isolation contains them. The 40% fault rate
is expected — most GPU domains are behind PRI ring arbitration and return 0xbadf*
when the domain hasn't been re-initialized by a guest driver.

### Probe 3: Revalidation with reverted 20-patch build (A6 fix confirmed)

**Fired**: Clean boot, 20 patches (cleanup_module REMOVED), full handoff RPC
**Result**: FULL SUCCESS — no lockup, no kernel oops, 79.3s total

**Timeline:**
```
17:27:52  RPC received, preflight pass (no zombie, module clean)
17:27:53  module patched: applied=20, total=20
17:27:54  nvsov loaded (finit_module, 400ms)
17:27:58  rm_trigger completed (GPU warmed: PMC 0x5fecdff1)
17:27:58  interrupt quench complete (INTR_EN → 0x0, pending=0x0)
17:28:01  RM init triggered, entering 60s settle
17:29:01  settle complete, tier=WarmCompute
17:29:02  fire-and-poll unbind (2s) — NO KERNEL OOPS
17:29:04  driver_override=vfio-pci, drivers_probe sent
17:29:09  vfio-pci bound (7s poll)
17:29:10  BAR0 snapshot: 62571 alive regs (1015ms capture)
17:29:11  fecs_init_ctxsw, pri_ring_recovery complete
17:29:11  module_cleanup: normal rmmod failed, zombie killer failed (EBUSY)
17:29:12  sovereign.warm_handoff: complete, success=true, total_ms=79272
```

**Key observations:**
- All 18 pipeline steps completed (17 ok, 1 expected fail: module_cleanup)
- Unbind path clean — no `irq_domain_remove` crash (A6 fix confirmed)
- Zombie module still occurs (B2) but is non-fatal — pipeline reports success
- 5060 host GPU unaffected — no DRM/cross-contamination (C2 clear)
- Kernel clean: `dmesg` shows zero oopses/panics/faults
- Watchdog deactivated cleanly after handoff completion

## Diesel Engine Improvements (post-probe)

### Kernel Oops Sentinel (`kernel_sentinel.rs`)
- Background OS thread monitoring `/dev/kmsg` in real-time
- Detects crash signatures: Oops, BUG, RIP, panic, irq_domain, msi_device_data
- On crash detection: saves full triage report to `/var/lib/toadstool/crash-reports/`
- Report includes: GPU BAR0 registers (both GPUs), module state, PCI config, kernel log context
- If handoff is active: triggers emergency interrupt quench via watchdog
- Requires `CAP_SYSLOG` capability (added to systemd unit)
- Tested: NVRM warning → logged as WARN, BUG → saves full crash report

### IRQ Storm Detector (watchdog enhancement)
- During active handoff, watchdog samples INTR_EN_0 via BAR0 mmap at 500ms intervals
- If INTR_EN has unexpected hot bits (anything beyond PBDMA bit 9), auto-quenches
- Pre-emptive defense: catches IRQ re-enable that nvidia_close may do before the
  pipeline's post-exit quench runs
- Also exports `is_active()` and `force_emergency_quench()` for cross-module use

## Success Criteria

- [x] All Category A vectors remain defended (no regression) — Exp 230 Run 2, Probe 3
- [x] A6 (cleanup_module NOP) identified, triggered, and reverted — Probe 2→3
- [x] Category B vectors detected and reported gracefully — Probe 1 (B2)
- [x] B2 zombie accepted as non-fatal, pipeline succeeds despite it — Probe 3
- [x] At least 2 Category C vectors probed and characterized — Probe 4 (C3, C7 both SAFE)
- [x] Kernel oops sentinel deployed and tested — crash reports with GPU forensics
- [x] IRQ storm detector active during handoff — pre-emptive quench capability
