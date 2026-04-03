# Experiment 138: D-State Root Cause Analysis and Rewire Plan

**Date**: 2026-04-02
**Status**: ANALYSIS COMPLETE — pre-reboot review, identifies code changes needed
**Predecessor**: Exp 136 (Dual GPU Boot Iteration), Exp 137 (SEC2 DMA Reconstruction)

## Executive Summary

Experiments 136-137 ended with both GPUs unrecoverable without reboot: K80 die #1
in PCIe D-state from Exp 136's cold-boot, and Titan V's daemon stack crashed from
Exp 137's warm-fecs after register manipulation. This document traces the exact
failure chain, identifies five architectural gaps in coralReef, and specifies the
code changes needed before the next iteration.

## D-State Failure Chain — Titan V

### Timeline

```
19:05  ACR boot Strategy 8 runs → SEC2 briefly starts FECS → DMA fails
       (ember child PID 162863 enters D-state from earlier reset attempt)

19:25  Exp 137 register manipulation phase:
       • BIND_INST changed: 0x00010040 → 0x00020041
       • FECS HRESET (cpuctl 0x08) × 3
       • FECS STARTCPU (cpuctl 0x02) × 4
       • GPCCS HRESET + STARTCPU × 2
       • FECS SCTL write (0x00 — ignored, read-only)
       • PRAMIN window shifted (0x1700 ← 0x05)
       • SEC2 DMEMC sequential reads (256 iterations)

19:27  coralctl warm-fecs requested
       → warm_handoff.rs calls slot.swap_traced("nouveau", ...)
       → swap/mod.rs: handle_swap_device_with_journal
         Phase 1: prepare_for_unbind → pin_power, clear reset_method → OK
         Phase 2: drop(held_device) → closes VFIO fd → kernel vfio_pci_core_disable
                  → sysfs_write("driver/unbind", bdf)
                  → isolated_sysfs_write forks child PID 198112
                  → child writes to .../driver/unbind
                  → kernel tries to unbind vfio-pci
                  → vfio-pci teardown accesses PCI config space
                  ╔══════════════════════════════════════════════╗
                  ║  PCIe completion timeout → CHILD IN D-STATE  ║
                  ╚══════════════════════════════════════════════╝
                  → parent waitpid times out after 10s
                  → parent SIGKILL's child (ignored — D-state)
                  → returns error: "timed out after 10s (child likely in D-state)"
       → warm_handoff returns error to glowplug
       → device left UNBOUND (vfio-pci unbound but nouveau never bound)

19:28  coralctl swap vfio-pci requested (attempt to recover)
       → glowplug dispatches to ember
       → ember bind_vfio tries sysfs operations
       → ALSO HANGS (D-state cascades)
       → glowplug daemon (PID 1776) crashes

19:30  systemd restarts coral-glowplug → PID 202165
       → new daemon tries to re-initialize managed devices
       → hits D-state on the same device → PID 202165 enters D-state
       → daemon socket unresponsive
```

### Why Did vfio-pci Unbind Enter D-State?

The D-state occurred in the kernel's `vfio_pci_core_disable()` path:

```
vfio_pci_core_release()
  → __vfio_pci_disable()
    → pci_reset_function()  ← even with reset_method="" cleared,
                                the kernel may still attempt FLR
    → pci_disable_device()  ← accesses PCI config space (command register)
    → pci_set_power_state(D3hot) ← config space write
```

**The GPU's PCIe endpoint was confused by our register writes:**

| Register Mutation | Impact on PCIe Endpoint |
|---|---|
| BIND_INST 0x00020041 | Falcon DMA engine now has a valid but empty page table — any pending DMA operations from SEC2 may generate poisoned TLPs |
| FECS HRESET × 3 | Each HRESET stops FECS mid-execution, potentially leaving GR engine in partial state |
| FECS/GPCCS STARTCPU × 4 | HS-mode STARTCPU rejection leaves falcons in 0x12 (halted+start) — ambiguous state |
| PRAMIN window 0x1700 ← 0x05 | PRAMIN now points to VRAM 0x20000 — any kernel access to BAR0 0x700000+ reads from unexpected VRAM |

The critical one is likely the **PRAMIN window shift**. When vfio-pci teardown
reads BAR0 offset 0x700000+ (which it may do for PCI capability chain walking
or VFIO region accounting), it's now reading from VRAM at 0x20000 instead of
the expected PRAMIN content. If the GPU's internal routing gets confused by
this mismatch during teardown, it can cause a PCIe completion timeout.

## D-State Failure Chain — K80

The K80 D-state (PID in earlier session) occurred during Exp 136's cold-boot:

```
coralctl cold-boot with gk210_full_bios_recipe.json
  → Devinit scripts applied (clocks, power)
  → FECS boot reported "success"
  → BUT: subsequent BAR0 reads return 0xFFFFFFFF
  → PCIe link hung (completion timeout on every BAR0 access)
  → kernel processes accessing this device enter D-state
  → coralctl reset --method remove-rescan also hangs (sysfs remove on hung device)
```

Root cause: the cold-boot devinit sequence wrote hardware registers that the
GPU's PCIe PHY didn't expect without a proper VBIOS POST. The PCIe link
dropped and never recovered.

## Architectural Gaps

### Gap 1: No BAR0 Health Gate Before Swap

**Current**: `preflight_device_check()` in `swap/preflight.rs` checks:
- sysfs path exists
- power state not D3cold
- vendor ID not 0xFFFF
- config space readable
- cold-hardware detection (PTIMER frozen + empty reset_method)

**Missing**: No check that BAR0 MMIO registers are actually responsive.
After our register manipulation, PCI config space may read fine (0x10DE vendor)
while BAR0 is partially non-responsive. The preflight passes, but the subsequent
kernel driver operations (which touch BAR0 during init/teardown) hang.

**Fix**: Add a BAR0 register read (BOOT0 + PMC_ENABLE) to preflight when the
device is vfio-pci bound. If BOOT0 returns 0xFFFFFFFF or PMC_ENABLE has PRI
faults, reject the swap.

```rust
// In preflight_device_check, after config space check:
if current_driver.as_deref() == Some("vfio-pci") {
    if let Some(held) = held_devices.get(bdf) {
        let boot0 = held.device.bar0_read_u32(0x0);
        let pmc = held.device.bar0_read_u32(0x200);
        if boot0 == 0xFFFFFFFF || (pmc & 0xBADF0000) == 0xBADF0000 {
            return Err("preflight FAILED: BAR0 not responsive after experiment");
        }
    }
}
```

### Gap 2: PRAMIN Window Not Restored After Experiments

**Current**: Register manipulation via `coralctl mmio write` can change the
PRAMIN window (`NV_PBUS_BAR0_WINDOW` at 0x1700) to arbitrary addresses.
Nothing restores it.

**Missing**: When kernel drivers access BAR0 space 0x700000-0x7FFFFF (the
PRAMIN aperture), they expect it to point to the instmem area. If we've
redirected it to arbitrary VRAM, kernel operations get wrong data.

**Fix**: 
1. `coralctl mmio write` should warn when writing to 0x1700 (PRAMIN window)
2. Any command that modifies PRAMIN should restore it before returning
3. The `prepare_for_unbind` lifecycle hook should restore PRAMIN to 0x0 or
   the original value

### Gap 3: VFIO fd Close Not Isolated

**Current** in `swap/mod.rs` lines 181-199:
```rust
if let Some(device) = held.remove(bdf) {
    drop(device);  // ← runs in main ember thread
}
```

The VFIO device `drop` closes the device fd, which triggers kernel-side
`vfio_pci_core_disable()`. If the device is in a bad state, this can hang
the ember thread.

**Missing**: The fd close is done inline, not isolated. The `sysfs_write`
that follows IS isolated (via `isolated_sysfs_write`), but the `drop` that
precedes it is not.

**Fix**: Move the VFIO fd close to an isolated thread, similar to
`guarded_vfio_open`:

```rust
fn guarded_vfio_close(device: HeldDevice, timeout: Duration) -> Result<(), String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("vfio-close".to_string())
        .spawn(move || {
            drop(device);
            let _ = tx.send(());
        })
        .map_err(|e| format!("spawn vfio-close thread: {e}"))?;
    
    match rx.recv_timeout(timeout) {
        Ok(()) => Ok(()),
        Err(_) => {
            // Thread is in D-state — leaked, but ember stays alive
            Err("VFIO fd close timed out (device likely in bad state)")
        }
    }
}
```

### Gap 4: No "Experiment Mode" Safety Boundary

**Current**: `coralctl mmio write` has unrestricted access to any BAR0 register.
There's no tracking of which registers were modified or whether the device is
in a "dirty" state that's unsafe for driver swaps.

**Missing**: After register experiments (BIND_INST, CPUCTL HRESET, PRAMIN),
the device is in a state that's potentially unsafe for kernel driver
transitions. Nothing prevents a subsequent `warm-fecs` from triggering D-state.

**Fix**: Two-layer approach:
1. **Dirty flag**: Track whether BAR0 has been written since last clean state.
   Set a `device.experiment_dirty = true` flag on any `mmio write`.
2. **Pre-swap gate**: If `experiment_dirty`, require either:
   - A PCI-level reset (SBR via bridge) to restore clean state, OR
   - Explicit `--force` flag acknowledging the risk
3. **PRAMIN restore**: Automatically restore PRAMIN window to 0x0 when the
   dirty flag is set and a swap is requested.

### Gap 5: SEC2 DMEM Not Captured During Warm-FECS

**Current**: `warm_handoff.rs` captures FECS state, PFIFO snapshot, and some
engine registers during the nouveau phase. But it does NOT capture:
- SEC2 DMEM (init message with CMDQ/MSGQ offsets)
- SEC2 CMDQ/MSGQ register state
- SEC2 BIND_INST (valid instance block address)
- Full oracle snapshot (MMU page tables)

**Missing**: After swap-back to VFIO, SEC2's CMDQ is consumed and BIND_INST
is invalid. Without the captured state, we can't reconstruct the command
interface to send BOOTSTRAP_FALCON.

**Fix**: Add to `warm_handoff.rs` between Step 6c (PFIFO snapshot) and
Step 7 (swap back):

```rust
// Step 6d: Capture SEC2 state for post-swap ACR reconstruction
let sec2_snapshot = if fecs_usable {
    let sec2_dmem = ember.sec2_dmem_dump(&bdf, 0, 4096)?;
    let sec2_cmdq = ember.sec2_queue_probe(&bdf)?;
    let sec2_bind_inst = ember.mmio_read(&bdf, 0x87090)?;
    let sec2_bind_contents = ember.pramin_read(&bdf, bind_inst_addr, 256)?;
    Some(Sec2Snapshot { dmem, cmdq, bind_inst, bind_contents })
} else {
    None
};
```

## Rewire Plan — Priority Order

### Before Reboot (documentation only)

1. ✅ This document (Exp 138)
2. ✅ Exp 137 document (completed)

### After Reboot — Immediate Safety Fixes

| # | Change | File | Risk Mitigation |
|---|--------|------|-----------------|
| 1 | VFIO fd close isolation | `swap/mod.rs` | Prevents D-state from poisoning ember main thread |
| 2 | PRAMIN restore in prepare_for_unbind | `vendor_lifecycle/nvidia.rs` | Prevents PRAMIN mismatch during kernel teardown |
| 3 | BAR0 health check in preflight | `swap/preflight.rs` | Rejects swap when device is in bad state |
| 4 | Experiment dirty flag + pre-swap gate | `hold.rs` + `swap/mod.rs` | Requires reset before swap after register writes |

### After Safety Fixes — SEC2 DMA Reconstruction

| # | Change | File | Purpose |
|---|--------|------|---------|
| 5 | SEC2 DMEM capture in warm_handoff | `warm_handoff.rs` | Captures CMDQ offsets for post-swap use |
| 6 | Oracle capture during warm_handoff | `warm_handoff.rs` | Preserves MMU page table state |
| 7 | SEC2 CMDQ reconstruction command | `coralctl` new subcommand | Writes BOOTSTRAP_FALCON to CMDQ + pokes IRQ |
| 8 | End-to-end warm-fecs-acr pipeline | `coralctl` new subcommand | Single command: warm-fecs → capture → swap → reconstruct → bootstrap |

## Register Manipulation Safety Rules

Going forward, any experiment that writes to these "danger registers" must
follow a recovery protocol before allowing driver swaps:

| Register | Address | Why Dangerous |
|----------|---------|---------------|
| PRAMIN_WINDOW | 0x1700 | Redirects kernel BAR0 reads to wrong VRAM |
| SEC2_BIND_INST | 0x87090 | Can trigger DMA on invalid page tables |
| FECS_CPUCTL | 0x409100 | HRESET stops GR engine mid-context |
| GPCCS_CPUCTL | 0x41a100 | Same — GPC scheduler disrupted |
| PMC_ENABLE | 0x200 | Disabling engines causes PRI faults |
| PFB_* | 0x100000+ | Memory controller — can hang VRAM |

**Recovery protocol**: Before any driver swap after touching these registers:
1. Restore PRAMIN window to 0x0
2. Verify BOOT0 reads valid chipset ID
3. Verify PMC_ENABLE is not PRI-faulting
4. If any check fails, attempt bridge SBR before swap

## Findings Summary for Exp 136-137-138

| Discovery | Impact | Source |
|-----------|--------|--------|
| BIND_INST bit 0 = 0 after nouveau teardown | SEC2 DMA completely disabled | Exp 137 |
| FECS/GPCCS IMEM has firmware (loaded by nouveau) | Don't need to re-load — just need to START | Exp 137 |
| HS mode hardware locks STARTCPU from host | Only SEC2 ACR can start falcons | Exp 137 |
| SEC2 uses CMDQ/MSGQ, not mailbox | MB0/MB1 writes ignored | Exp 137 |
| CMDQ registers all zero after swap | Must capture during nouveau phase | Exp 137 |
| PRAMIN shift causes D-state on unbind | Must restore before swap | Exp 138 |
| VFIO fd close not D-state isolated | Can poison main daemon thread | Exp 138 |
| No BAR0 health gate in preflight | Preflight passes on corrupted devices | Exp 138 |
| Strategy 8 got FECS briefly running (10ms) | SEC2 CAN bootstrap — just needs DMA fix | Exp 136 |
| K80 cold-boot hangs PCIe (no VBIOS POST) | Needs proper GDDR5 training | Exp 136 |
