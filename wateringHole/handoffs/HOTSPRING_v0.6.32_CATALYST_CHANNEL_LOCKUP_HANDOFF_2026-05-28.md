# hotSpring Handoff: Catalyst Channel + Lockup Forensics — Exp 229

**Date:** May 28, 2026
**Sprint:** S279 (continued)
**Status:** COMPLETE — Tier 2 WarmCompute handoff achieved
**Hardware:** Titan V (GV100) @ 0000:49:00.0

## Summary

Full end-to-end catalyst warm handoff completed for the first time
(Run #9). GPU maintains 23 active engines (PMC_ENABLE=0x5fecdff1)
through the entire nvsov→vfio-pci driver swap cycle.

Seven system lockups triaged across nine runs using diesel engine
sentinel data. Five distinct lockup vectors cataloged, mitigated,
and proven. The sentinel (`catalyst-sentinel.sh` v2) and watchdog
(`catalyst_watchdog`) captured forensic data across power cycles,
enabling root cause analysis without reproducing conditions.

## Key Results

- `success=true`, `tier=warm_compute`, `total_ms=80,886`
- 20/20 nvidia_catalyst_handoff patches applied
- 62,571 alive BAR0 registers captured post-swap
- FECS, GPCCS, PMU all running post-swap
- PRI ring recovery succeeded

## Lockup Vector Catalog

| # | Vector | Root Cause | Fix |
|---|--------|-----------|-----|
| 1-3 | pci_lock deadlock | keepalive config reads during SBR | HandoffExclusionGuard |
| 4-5 | Interrupt storm | Quench wrote to read-only INTR_EN@0x140 | Write CLEAR register@0x180 |
| 6 | Interrupt storm (variant) | nvidia_close re-enables INTR_EN after quench | post_exit_quench from pipeline |
| 7 | Kernel corruption | nv_dev_free_stacks frees RM thread stacks while running | nv_close_device RetAtEntry |
| 8 | Unbind hang | nv_pci_remove os_delay polling loop | nv_pci_remove RetAtEntry |

## Patches Added to nvidia_catalyst_handoff

- `nv_close_device` — RetAtEntry (prevents ALL per-device teardown on fd close)
- `nv_pci_remove` — RetAtEntry (prevents PCI unbind hang)
- `post_exit_quench()` — pipeline-side INTR_EN_CLEAR after nvidia_close
- `post_exit_intx_disable()` — PCI CMD INTx disable after MSI teardown

## Files Changed (primals/toadStool)

- `cylinder/src/vfio/module_patch/patch_sets/nvidia.rs` — 20 targets (was 18)
- `cylinder/src/vfio/sovereign_handoff/rm_trigger.rs` — post_exit_quench, bdf param
- `cylinder/src/bin/rm_trigger.rs` — corrected quench_gpu_interrupts (CLEAR@0x180)
- `server/src/background/catalyst_watchdog.rs` — new watchdog module
- `server/src/background/mod.rs` — catalyst_watchdog integration
- `server/src/unibin/mod.rs` — watchdog thread spawn

## Files Changed (springs/hotSpring)

- `docs/exp229-lockup-analysis.md` — complete forensic record
- `experiments/229_CATALYST_RM_CHANNEL.md` — experiment journal
- `scripts/catalyst-sentinel.sh` — v2 (BAR0-only, no pci_lock)
- `CHANGELOG.md` — Exp 229 entry
- `EXPERIMENT_INDEX.md` — Exp 229 status updated

## Remaining Work

- RM channel creation still partial (`device_alloc` status=0x22) — Exp 230+
- `rmmod nvsov` fails (non-critical, module stays at refcount 0)
- Volta INTR_EN register quirks documented for upstream reference

## Upstream Gaps for Primal Teams

- **cylinder**: `trigger_rm_init` now has 3 params — callers need updating if
  any exist outside `pipeline.rs`
- **catalyst_watchdog**: uses `unsafe` MMIO — `#![allow(unsafe_code)]` override
  in server crate module
- **sentinel script**: lives in hotSpring/scripts, should be wired into diesel
  engine as a first-class background service for production lockup detection
