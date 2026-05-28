# Exp 229: Catalyst Channel — System Lockup Analysis

## Status: FULL END-TO-END SUCCESS — Tier 2 WarmCompute handoff completed

Run #9 (May 28 11:07): **success=true, tier=warm_compute, total_ms=80886**
- 20/20 patches applied (nv_close_device + nv_pci_remove + all isolation NOPs)
- GPU warm throughout: PMC_ENABLE=0x5fecdff1 (23 engines), TPC alive
- warm_swap nvsov→vfio-pci completed in 7.1s
- 62,571 alive BAR0 registers captured post-swap
- PRI ring recovery succeeded
- Post-handoff: driver=vfio-pci, PMC=0x5fecdff1, INTR_EN=0x00000200
- Only `module_cleanup` (rmmod nvsov) failed — non-critical

### Lockup Mechanism (confirmed across 5 incidents)

**Kill chain**: When `rm_trigger` exits and closes its nvidia file descriptors,
`nvidia_close` runs the teardown path. Because `rm_disable_adapter` and
`rm_shutdown_adapter` are NOP'd (to keep the GPU warm), the teardown skips GPU
register quiescing but STILL runs `free_irq()` + `pci_disable_msi()`.

Result: GPU is warm with 23 active engines, `INTR_EN=0x7fffffff` (all interrupt
sources enabled), MSI handler removed, MSI disabled at PCI level → GPU falls
back to legacy INTx → level-triggered INTx fires with no handler to ACK →
**infinite interrupt storm** → all CPUs saturated → system lockup requiring
power cycle.

### Evidence Trail

#### Sentinel v2 data (lockup #5, 2026-05-28 08:19)

```
TICK 15: pmc=0x5fecdff1 pop=23 intr_en=0x7fffffff  driver=nvsov  mods=nvsov(6)
TICK 16: pmc=0x5fecdff1 pop=23 intr_en=0x7fffffff  driver=nvsov  mods=nvsov(6)
TICK 17: pmc=0x5fecdff1 pop=23 intr_en=0x7fffffff  driver=nvsov  mods=nvsov(6)
TICK 18: pmc=0x5fecdff1 pop=23 intr_en=0x7fffffff  driver=nvsov  mods=nvsov(0) ← rm_trigger exited
[NO TICK 19 — SYSTEM DEAD]
```

#### Why the quench fix from lockup #4 failed

```
[QUENCH] NV_PMC_INTR_EN_0: 0x7fffffff → 0x7fffffff
```

On Volta (GV100), `NV_PMC_INTR_EN(0)` at BAR0 offset **0x140 is READ-ONLY**.
NVIDIA uses a SET/CLEAR register pair pattern:
- `0x140` — INTR_EN: read current enable mask (READ-ONLY)
- `0x160` — INTR_EN_SET: write 1 bits to enable interrupt sources (WRITE-ONLY)
- `0x180` — INTR_EN_CLEAR: write 1 bits to disable interrupt sources (WRITE-ONLY)

The fix wrote 0 to 0x140 (no-op). Must write `0xFFFFFFFF` to **0x180** instead.

### Fix Applied

In `rm_trigger.rs`, `quench_gpu_interrupts()`:
- Write `0xFFFFFFFF` to `NV_PMC_INTR_EN_CLEAR(0)` at BAR0+0x180
- Read back `NV_PMC_INTR_EN(0)` at 0x140 to verify it becomes 0
- Read `NV_PMC_INTR(0)` at 0x100 to ACK any pending interrupts

### Timeline of all lockups

| # | Date | Cause | Fix attempted | Fix worked? |
|---|------|-------|---------------|-------------|
| 1 | May 28 early | Unknown (pre-sentinel) | Added keepalive exclusion | No |
| 2 | May 28 early | Unknown (pre-sentinel) | Extended exclusion to bridges | No |
| 3 | May 28 early | Unknown (pre-sentinel) | Removed nv_pci_remove NOPs | No |
| 4 | May 28 07:39 | IRQ storm after rm_trigger exit | Write 0 to INTR_EN@0x140 | No — 0x140 is read-only |
| 5 | May 28 08:19 | Same IRQ storm, quench confirmed failed | Write 0xFFFFFFFF to INTR_EN_CLEAR@0x180 | **No** — nvidia_close re-enables |
| 6 | May 28 08:54 | nvidia_close RE-ENABLES INTR_EN after rm_trigger quench | Post-exit dual quench from pipeline + PCI INTx disable | **Yes** — quench works, but new vector |
| 7 | May 28 09:13 | RM thread stack use-after-free (INTR_EN=0, NOT interrupt storm) | RetAtEntry on nv_close_device (skip ALL per-device teardown) | **YES — first survival!** |
| 8 | May 28 09:34 | (no lockup) nv_pci_remove hangs in os_delay loop on unbind | Added RetAtEntry on nv_pci_remove | **YES — full success** |
| 9 | May 28 11:07 | **FULL SUCCESS** — warm_compute tier, 80.9s total, 20/20 patches | All defenses active | **END-TO-END PASS** |

### Lockup #6: nvidia_close re-enables interrupts

The INTR_EN_CLEAR quench in rm_trigger SUCCEEDED (`0x7fffffff → 0x00000000`).
But `nvidia_close` (triggered by fd close AFTER the quench) RE-ENABLED all
interrupt sources. Sentinel evidence:

```
Tick 19: intr_en=0x7fffffff  mods=nvsov(6)  ← GPU warm, rm_trigger holding fds
Tick 22: intr_en=0x7fffffff  mods=nvsov(0)  ← rm_trigger exited, nvidia_close RE-ENABLED
[NO TICK 23 — DEAD]
```

rm_trigger's QUENCH log confirmed success: `INTR_EN: 0x7fffffff → 0x00000000`.
But between that quench and the sentinel read 196ms later, nvidia_close put
INTR_EN back to 0x7fffffff. The quench was correct but too early.

**Fix**: Added dual-layer post-exit quench in the pipeline (`trigger_rm_init`):
1. `post_exit_quench(bdf)` — writes 0xFFFFFFFF to INTR_EN_CLEAR@0x180 AFTER
   nvidia_close has fully completed (process has exited, `cmd.output()` returned)
2. `post_exit_intx_disable(bdf)` — sets PCI CMD bit 10 (INTx Disable) as
   belt-and-suspenders defense

### Key registers (Volta GV100)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x000 | NV_PMC_BOOT_0 | RO | Chip identification |
| 0x100 | NV_PMC_INTR(0) | R/W | Pending interrupts (read clears edge) |
| 0x140 | NV_PMC_INTR_EN(0) | RO | Current interrupt enable mask |
| 0x160 | NV_PMC_INTR_EN_SET(0) | WO | Write 1 to enable interrupt source |
| 0x180 | NV_PMC_INTR_EN_CLEAR(0) | WO | Write 1 to disable interrupt source |
| 0x200 | NV_PMC_ENABLE | RW | Engine enable (warmness indicator) |

### Lockup #7: NOT interrupt storm — RM thread stack corruption

Post-exit quench CONFIRMED working: sentinel shows `INTR_EN=0x00000000` at
tick 37 (after nvidia_close, after pipeline quench). System still froze ~1s later.

```
Tick 36 @ 09:13:31: intr_en=0x7fffffff  nvsov(6)  ← rm_trigger holding
Tick 37 @ 09:13:32: intr_en=0x00000000  nvsov(0)  ← QUENCH CONFIRMED, nvidia_close done
[NO TICK 38 — DEAD]
```

Root cause: `nv_close_device` still runs `nv_dev_free_stacks()` which frees
kernel thread stacks used by RM threads. With `rm_disable_adapter` and
`rm_shutdown_adapter` NOP'd, RM threads are NOT stopped before their stacks
are freed → use-after-free on kernel stacks → silent kernel corruption → lockup.

**Fix**: RetAtEntry on `nv_close_device` (the per-device close function).
Prevents ALL per-device teardown (stack free, free_irq, pci_disable_msi).
The outer `nvidia_close` still calls `rm_free_unused_clients` for safe
per-client RM object cleanup. GPU stays fully managed with live IRQ handler.

### Run #8: FIRST SURVIVAL — nv_pci_remove hang (no lockup)

With `nv_close_device` NOP'd, the system survived the entire catalyst
handoff cycle for the first time. Sentinel ran 224 ticks (~3:45 min) with
GPU warm (`PMC_ENABLE=0x5fecdff1`, 23 engines) and no lockup.

The GPU survived fd close, nvidia_close, 60s settle, and unbind with full
warmth preserved. However, `nv_pci_remove` (PCI unbind callback) hangs in
an `os_delay` polling loop waiting for GPU quiescence that never comes
(because `nv_close_device` was NOP'd).

```
kern: nv_pci_remove+0x18e/0x3b0 [nvsov]
      os_delay+0xf8/0x270 [nvsov]
```

This causes `driver_override` sysfs writes to hang during rebind (D-state
processes). The system is alive but degraded — requires reboot to clear
D-state processes. Fix: add `nv_pci_remove` RetAtEntry to catalyst patch set.

**Failure type catalog update:**
- Lockups #1-3: pci_lock / keepalive (mitigated by exclusion guard)
- Lockups #4-5: INTR_EN quench to wrong register (fixed: CLEAR@0x180)
- Lockup #6: nvidia_close re-enables INTR_EN (fixed: post-exit quench)
- Lockup #7: nv_dev_free_stacks use-after-free (fixed: nv_close_device NOP)
- Run #8: nv_pci_remove os_delay hang (fix: nv_pci_remove NOP — deployed)

### Safety architecture (diesel engine)

1. **Pre-exit quench**: `quench_gpu_interrupts()` in rm_trigger writes to
   INTR_EN_CLEAR before closing nvidia fds (necessary but not sufficient —
   nvidia_close can re-enable)
2. **Post-exit quench** (NEW): `post_exit_quench()` + `post_exit_intx_disable()`
   in the pipeline, runs AFTER nvidia_close has fully completed. This is the
   critical defense — catches re-enabled interrupts.
3. **Watchdog**: `catalyst-watchdog` thread in toadstool monitors handoff
   liveness, performs emergency quench + process kill if pipeline hangs
4. **Sentinel**: External bash script logs BAR0 state every second (no PCI
   config reads — those take pci_lock and can deadlock)
5. **Keepalive exclusion**: RAII guard excludes target BDF + bridges from
   keepalive config reads during handoff
