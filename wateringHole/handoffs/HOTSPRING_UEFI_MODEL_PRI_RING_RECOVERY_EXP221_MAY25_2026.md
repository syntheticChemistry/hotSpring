# UEFI Model GPU Sovereignty — PRI Ring Recovery — hotSpring Handoff

**Date:** May 25, 2026
**From:** hotSpring
**To:** primalSpring (audit), toadStool (upstream code landed)
**Status:** Experiment 221 complete — PRI ring recovery validated, falcon HS boundary mapped
**Experiments:** 221 (builds on Exp 219 catalyst HW validation)
**Previous:** `HOTSPRING_CATALYST_HW_VALIDATED_EXP219_MAY24_2026.md`

## Summary

Exp 221 tested the "UEFI Model" hypothesis: treat nvidia RM as UEFI-like
boot services that initialize GPU compute engines, then transition to
sovereign control. Four phases executed over May 24-25, 2026.

**Core finding:** PRI ring destruction during PCI unbind occurs in the
**kernel's PCI framework** (PMC_ENABLE cleared), not in nvidia's
`nv_pci_remove`. The PRI ring IS recoverable from cold via direct BAR0
writes. However, falcon firmware (FECS/GPCCS IMEM) is wiped during unbind
and cannot be replayed — FECS/GPCCS are fuse-enforced HS (high-security)
mode on GV100, blocking direct host IMEM upload.

## Key Results

| Finding | Detail |
|---------|--------|
| PRI ring death location | Kernel PCI framework (`pci_device_remove`), NOT `nv_pci_remove` |
| RetAtEntry on `nv_pci_remove` | Dead end — still kills PRI ring AND leaks iomem |
| PRI ring recovery | **Works**: re-enable PGRAPH (PMC_ENABLE bit 12) + PRI ring master enumerate |
| Falcon accessibility | FECS/GPCCS cpuctl readable after recovery (halted at PC=0) |
| Falcon IMEM | Wiped during unbind, not recoverable |
| Falcon security | Fuse-enforced HS mode — no host IMEM PIO upload |
| ACR boot from vfio-pci | Not viable — WPR not configured on GV100 (pre-GSP) |
| PRI ring anchor health | Degraded (PGRAPH on, falcons accessible, TPC sub-ring not functional) |
| System stability | Both cards clean after handoff, zero D-state, zero iomem leaks |

## What Changed (toadStool)

### `sovereign_handoff.rs`
- **`recover_pri_ring()`** — new function: re-enables PGRAPH in PMC_ENABLE,
  acknowledges pending PRI ring interrupts, enumerates PRI ring stations,
  starts ring, verifies falcon accessibility. Runs as step 6c between
  warm_swap and tier_classify.
- **Diagnostic probe enhanced** — PCI config space reads (command register,
  PM state) between unbind and rebind. Shows `bus_master=false` after unbind.
  Correct falcon PC register (`0x40911c` hardware PC, not `0x409624` CTXSW).
  Added `pmc_enable` and `pri_ring_master` status to probe.
- **Post-recovery IMEM probe** — attempts FECS IMEM PIO read after PGRAPH
  is re-enabled. Confirms IMEM is genuinely empty (not just PRI-gated).
- **PRI ring anchor health** — now probes actual post-recovery BAR0 state
  instead of using pre-swap tier. Correctly classifies as Degraded when
  PGRAPH is on and falcons are accessible but TPC sub-ring is down.

### `module_patch.rs`
- **`nvidia_boot_services`** — no longer uses RetAtEntry on `nv_pci_remove`
  (leaks iomem without preserving PRI ring). Now delegates to
  `nvidia_catalyst_handoff` (clean unbind + post-swap PRI ring recovery).

### `firmware.rs` (runtime/gpu)
- **Falcon register correction** — `FALCON_PC` changed from `0x104`
  (BOOTVEC) to `0x11c` (hardware PC). Added `FALCON_BOOTVEC` and
  `FALCON_STATUS` constants. `read_falcon` now reads both BOOTVEC and PC.
- **`exit_boot_services()`** — enhanced with PGRAPH status and PRI ring
  master status capture in boot service evidence.

### `pri_ring_anchor.rs` (ember)
- No structural changes; health classification logic moved to
  `sovereign_handoff.rs` for post-recovery accuracy.

## Architecture Discovery: The Three Boundaries

```
┌─────────────────────────────────────────────────────┐
│ Boundary 1: PCI Framework                            │
│ Kernel clears PMC_ENABLE during standard unbind.     │
│ PGRAPH disabled → PRI ring routing fails.            │
│ RECOVERABLE via BAR0 write to PMC_ENABLE bit 12.     │
├─────────────────────────────────────────────────────┤
│ Boundary 2: Falcon HS Fuses                          │
│ FECS/GPCCS are fuse-locked to HS (high-security).    │
│ Host IMEM PIO upload blocked by hardware.            │
│ NOT RECOVERABLE from userspace.                      │
├─────────────────────────────────────────────────────┤
│ Boundary 3: ACR Secure Boot Chain                    │
│ Firmware loaded only via SEC2→ACR→FECS/GPCCS.        │
│ Requires WPR (not configured on GV100 pre-GSP).     │
│ NOT AVAILABLE from vfio-pci.                         │
└─────────────────────────────────────────────────────┘
```

Tier 1 (WarmInfrastructure) operates above Boundary 1: recoverable.
Tier 2 (WarmCompute) requires crossing Boundaries 2 and 3: not viable
via the ExitBootServices model on GV100.

## Sovereign Tier Landscape (Updated)

| Tier | Status | What Works | What's Missing |
|------|--------|-----------|----------------|
| Cold | Baseline | PMC, PTIMER | Everything |
| Tier 1 (WarmInfra) | **HW validated** | PFIFO, DMA, VRAM, PRI ring (recovered) | Falcon firmware |
| Tier 1+ (PRI Recovery) | **NEW — validated** | Above + PGRAPH, falcon registers accessible | FECS/GPCCS firmware execution |
| Tier 2 (WarmCompute) | Blocked | — | Fuse-enforced HS, no WPR, no ACR from vfio |

## Upstream Gaps for primalSpring

### Resolved (from Exp 219/221)
- PRI ring destruction root cause identified (kernel PCI framework)
- PRI ring recovery implemented and validated
- RetAtEntry approach conclusively eliminated
- Falcon register addresses corrected
- Diagnostic probe comprehensive (PCI config + BAR0 + IMEM)

### Still Open
- **GAP-TS-221-A: Runtime Services model** — Tier 2 requires nvidia as
  persistent runtime service (not exited). Architecture TBD.
- **GAP-TS-221-B: GPC sub-ring recovery** — PRI ring master recovers
  top-level routing, but GPC sub-ring status=0xcf (error). GPC-routed
  registers (CTXSW, TPC) still PRI-fault after recovery.
- **GAP-TS-221-C: PCI bus master disable** — kernel clears PCI command
  register bit 2 during unbind. Could we patch the kernel PCI unbind path
  to skip this for sovereign GPU devices?
- **GAP-TS-221-D: Firmware extraction from nvidia.ko** — FECS/GPCCS
  firmware is embedded in the nvidia kernel module. Extracting and
  loading via a custom ACR chain is unexplored.

## Test Results

- 20 sovereign_handoff tests: all pass
- 121 ember tests: all pass
- 6 gpu firmware tests: all pass
- Hardware: both Titan V cards validated (02:00.0, 49:00.0)
- Binary deployed to `/usr/local/bin/toadstool`

## References

- Experiment doc: `experiments/221_UEFI_MODEL_GPU_SOVEREIGNTY.md`
- Plan: `.cursor/plans/uefi_model_gpu_sovereignty_5ead68c1.plan.md`
- Previous: `experiments/219_CATALYST_DRIVER_PATTERN.md`
- Previous: `experiments/217_TPC_PRI_STATION_CREATION.md`
