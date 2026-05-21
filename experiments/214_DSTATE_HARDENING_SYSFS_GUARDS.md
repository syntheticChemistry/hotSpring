# Experiment 214 — D-State Hardening, Sysfs Guards, and Testing

**Date**: 2026-05-20
**Status**: COMPLETE
**Hardware**: 2x NVIDIA Titan V (GV100), BDFs `0000:02:00.0`, `0000:49:00.0`
**Dependency**: Exp 213 (Live Hardware Warm Handoff)

## Objective

Harden the sovereign handoff pipeline against kernel D-state deadlocks discovered
in Exp 213. Build infrastructure to prevent, detect, and recover from kernel-level
hangs during sysfs and kernel module operations.

## Root Cause Analysis

Exp 213's cascading kernel failure revealed five systemic weaknesses:

1. **Zero timeouts on kernel-touching operations** — `sysfs_write`, `insmod`, `rmmod`,
   and `drivers_probe` used blocking calls with no timeout, allowing a single stuck
   kernel thread to freeze the entire daemon.
2. **Fragmented sysfs helper implementations** — Three separate sysfs write
   implementations with inconsistent error handling.
3. **Global GPU mutex** — A global `HANDOFF_LOCKS` HashMap with manual
   `acquire`/`release` calls. Panics or abandoned threads would leak the lock.
4. **No pre-flight checks** — `nouveau` in `Unloading` state (refcount=-1) was
   not detected before attempting `insmod`.
5. **No rollback on partial failure** — A failed handoff left the device unbound
   from both `vfio-pci` and `nouveau`, requiring manual recovery.

## Implementation

### Phase 1: `guarded_sysfs` Primitives

New module `cylinder/src/vfio/guarded_sysfs.rs`:

- **Child-process isolation**: All sysfs writes and `kmod` operations run in a
  forked child process. The parent monitors with a configurable timeout.
- **`reap_or_orphan` pattern**: After timeout + SIGKILL, polls for 2s then
  orphans the child rather than blocking indefinitely on `wait()`. Prevents
  the D-state cascade where a killed child inherits the parent's kernel stuck state.
- **`is_module_stuck` pre-flight**: Checks `/proc/modules` for `Unloading` state
  before attempting any module operation.
- **`unbind_iommu_siblings`**: Uses guarded writes with per-sibling timeout.

### Phase 2: RAII `HandoffGuard`

Replaced the global `HANDOFF_LOCKS` HashMap with an RAII `HandoffGuard` struct
in `sovereign_handoff.rs`:

```rust
struct HandoffGuard { bdf: String }
impl HandoffGuard {
    fn acquire(bdf: &str) -> Result<Self, String> { /* ... */ }
}
impl Drop for HandoffGuard {
    fn drop(&mut self) { /* releases lock even on panic */ }
}
```

All 19 explicit `acquire_handoff_lock`/`release_handoff_lock` calls removed.

### Phase 3: Module Patch Policy

Enhanced `module_patch.rs`:

- **`min_applied` policy**: `PatchSet` gains a `min_applied: usize` field. The
  patch operation fails if fewer than `min_applied` patches succeed.
- **Multi-byte NOP acceptance**: Added `0x0F` (multi-byte NOP lead) to accepted
  ftrace call site patterns alongside `0xE8`, `0x90`, `0x00`. Required for
  `CONFIG_DYNAMIC_FTRACE` kernels.
- **New `InsufficientPatches` error variant**.

### Phase 4: Guarded Executor in GlowPlug

Updated `glowplug/src/sysfs_executor.rs`:

- All `unbind` and `drivers_probe` calls now use `sysfs_write_guarded` with
  configurable timeouts (`UNBIND_TIMEOUT`, `PROBE_TIMEOUT`).
- Fixed silent error swallowing where `drivers_probe` failure was ignored
  with `let _ = ...`.

### Phase 5: `halt_result` Rollback Evolution

Updated `sovereign_handoff.rs`:

- `halt_result` signature extended with `module_name: &str` and
  `needs_device_rollback: bool`.
- Rollback now triggers even when no module was loaded, if the device was
  already unbound from `vfio-pci`.
- All 17 call sites updated.

## Testing

32 new tests added across `toadstool-cylinder` and `toadstool-glowplug`:

| Test Area | Count | Key Validations |
|-----------|-------|-----------------|
| `guarded_sysfs` | 8 | `parse_module_stuck` synthetic cases, timeout behavior |
| `sovereign_handoff` | 6 | RAII guard, rollback with siblings, rollback with device flag |
| `module_patch` | 5 | `min_applied` policy, multi-byte NOP, insufficient patches error |
| `sysfs_executor` | 4 | Guarded write delegation, error propagation |
| `sovereign_tiers` | 4 | Tier ordering, capabilities, display format |

All tests pass: `39 sovereign tests passed; 0 failed`.

## Validation on Live Hardware

After power cycle:

1. `sovereign.warm_handoff` on Titan V #1 — pipeline completed successfully
2. Module patching accepted `0x0F` lead bytes (kernel 6.17.9 with `CONFIG_DYNAMIC_FTRACE`)
3. `nouveau` loaded, probed, settled, and swapped back to `vfio-pci`
4. Both Titan Vs classified as Tier 1 (WarmInfrastructure)
5. No D-state incidents during or after handoff

## Key Metrics

| Metric | Before | After |
|--------|--------|-------|
| Sysfs write timeout | ∞ (blocking) | 5-30s configurable |
| Module stuck detection | none | pre-flight `/proc/modules` check |
| Lock leak on panic | yes | no (RAII) |
| Rollback on partial failure | incomplete | full device + module + sibling |
| Ftrace NOP patterns | 2 (0xE8, 0x90) | 4 (0xE8, 0x90, 0x00, 0x0F) |

## Files Changed

| File | Changes |
|------|---------|
| `cylinder/src/vfio/guarded_sysfs.rs` | `reap_or_orphan`, `parse_module_stuck`, guarded siblings |
| `cylinder/src/vfio/sovereign_handoff.rs` | RAII `HandoffGuard`, extended `halt_result`, rollback flags |
| `cylinder/src/vfio/module_patch.rs` | `min_applied` policy, `0x0F` NOP lead, `InsufficientPatches` |
| `glowplug/src/sysfs_executor.rs` | Guarded writes for unbind/probe, error propagation fix |
