# hotSpring → primalSpring Handoff: Crash Vector Reprofile + Diesel Engine Defense Matrix

**Date:** May 28, 2026
**Experiment:** 232 (Crash Vector Reprofile)
**Sprint:** S280
**Status:** COMPLETE

## Summary

Systematically reprobed every known crash vector through the abstracted diesel
engine. Cataloged 23 vectors across 4 categories (A: confirmed kills, B: confirmed
hangs, C: potential, D: cross-generation). Added three new diesel engine defense
layers: kernel oops sentinel, IRQ storm detector, and zombie module defense.

## Key Results

- **4 probes executed**, all success criteria met
- **NEW crash vector A6** discovered and fixed: `cleanup_module` RetAtEntry patch
  causes kernel oops in `irq_domain_remove` during PCI unbind
- **Revalidation clean**: 20-patch build, 79.3s handoff, no lockup, kernel clean
- **C3 (PRI faults) + C7 (pending IRQs) confirmed safe**: GPU-internal, zero PCIe AER

## Diesel Engine Improvements

### Kernel Oops Sentinel (kernel_sentinel.rs)
- Monitors `/dev/kmsg` for crash signatures (Oops, BUG, RIP, panic)
- Saves crash reports to `/var/lib/toadstool/crash-reports/` with:
  - GPU BAR0 registers (both GPUs)
  - Kernel module state (nvsov, nvidia, vfio_pci, etc.)
  - PCI config space
  - Last 50 kernel log lines
- Triggers emergency quench if handoff active during crash
- Requires `CAP_SYSLOG` (added to systemd unit)

### IRQ Storm Detector (watchdog enhancement)
- Samples INTR_EN_0 via BAR0 mmap at 500ms during active handoff
- Pre-emptively quenches if unexpected hot bits detected
- Catches IRQ re-enable from nvidia_close before pipeline quench runs

### Zombie Module Defense
- `rmmod_guarded` retries with `O_NONBLOCK|O_TRUNC` zombie-killer flags
- Pipeline preflight detects zombie, attempts burial, halts gracefully if permanent
- `ModuleSnapshot` struct + high-frequency polling during cleanup phase
- `PipelineSignal` lifecycle events for watchdog coordination

## Crash Vector Taxonomy

| Category | Count | Status |
|----------|-------|--------|
| A: Confirmed Kills | 6 (A1–A6) | All defended or reverted |
| B: Confirmed Hangs | 3 (B1–B3) | Detected and handled gracefully |
| C: Potential | 10 (C1–C10) | C3, C7 probed — SAFE |
| D: Cross-Generation | 4 (D1–D4) | Awaiting K80 hardware (Exp 231) |

## Files Changed (primals/toadStool)

- `crates/server/src/background/kernel_sentinel.rs` — NEW: /dev/kmsg sentinel
- `crates/server/src/background/catalyst_watchdog.rs` — IRQ detector + cross-module API
- `crates/server/src/background/mod.rs` — sentinel module wiring
- `crates/server/src/unibin/mod.rs` — sentinel startup in prod path
- `crates/server/Cargo.toml` — added libc dependency
- `crates/core/cylinder/src/vfio/guarded_sysfs/proc_scan.rs` — ModuleSnapshot
- `crates/core/cylinder/src/vfio/guarded_sysfs/kmod_build.rs` — force rmmod
- `crates/core/cylinder/src/vfio/sovereign_handoff/pipeline.rs` — signal callbacks + forensics
- `crates/core/cylinder/src/vfio/module_patch/patch_sets/nvidia.rs` — cleanup_module reverted

## Files Changed (springs/hotSpring)

- `CHANGELOG.md` — Exp 232 entry
- `EXPERIMENT_INDEX.md` — Exp 230 COMPLETE, Exp 232 entry, count 232
- `experiments/232_CRASH_VECTOR_REPROFILE.md` — full experiment journal
- `experiments/230_DIESEL_ABSTRACTION_REVALIDATION.md` — Run 2 results logged

## Upstream Gaps for Primals Teams

- **toadStool**: Sentinel crash reports could be forwarded to a remote endpoint
  for fleet-wide crash aggregation.
- **primalSpring**: The crash vector taxonomy (A1–A6, B1–B3, C1–C10, D1–D4) is a
  reusable pattern for any GPU driver rotation pipeline. Consider extracting as a
  standard.
- **Service unit**: `CAP_SYSLOG` now required. Update any deployment automation
  that provisions the systemd unit.
