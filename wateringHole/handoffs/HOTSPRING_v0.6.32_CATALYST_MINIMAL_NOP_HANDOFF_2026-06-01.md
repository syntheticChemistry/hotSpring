# hotSpring Handoff — Exp 234 Catalyst Minimal NOP Checkpoint

**Date:** 2026-06-01
**Spring:** hotSpring
**Session:** S284 (Exp 234 — Catalyst Minimal NOP)
**Status:** PAUSED — system lockup pattern during warm handoff teardown/rebind

---

## Context

Exp 233 identified `device_alloc 0x22` (NV_ERR_OBJECT_NOT_FOUND) as the blocker
for RM channel creation. The NOP'd cap system prevents GPU registration in RM's
internal device table. Exp 234 created `nvidia_catalyst_minimal_nop` — a refined
patch set targeting the teardown/rebind path to allow selective un-NOPing.

## Progress (5 Runs)

| Run | Result | Root Cause |
|-----|--------|------------|
| 1 | Kernel crash (proc_register) | Cap functions use hardcoded MODULE_NAME="nvidia" → procfs collision |
| 2 | finit_module ENODEV | discover_dkms_version picked nvidia-580 instead of nvidia-470 |
| 3 | BAR0 iomem leak | NOP'd nv_pci_remove prevents release_mem_region |
| 4 | driver_override D-state | NOP'd cleanup_module leaves PCI driver registered |
| 5 | 331s busy-wait + exit hang | NOP'd nv_close_device prevents usage_count decrement; nv_kthread_q_stop hangs |

**GPU fully initializes** in Runs 3-5 (PMC_ENABLE=0x5fecdff1, 23 engines, FECS/GPCCS firmware captured). The problem is the teardown/rebind path, not GPU initialization.

## Fixes Applied (toadStool)

### Patch Set: nvidia_catalyst_minimal_nop (21 targets)

**NOT NOP'd (must run for clean teardown):**
- `nv_close_device` — usage_count decrement (prevents nv_pci_remove busy-wait)
- `nv_pci_remove` — release_mem_region for BAR0
- `cleanup_module` — nv_pci_unregister_driver for PCI driver cleanup

**NOP'd (prevents hangs/crashes):**
- `nv_kthread_q_stop` — prevents module exit hang (kthread deadlock)
- `rm_disable_adapter`, `rm_shutdown_adapter` — preserves GPU state
- `rm_shutdown_rm`, `nv_destroy_rsync_info` — closed-binary hang prevention
- Namespace collision NOPs (procfs, cap system, acpi, nvlink, nvswitch)
- Access control bypass (os_is_administrator, cap_validate_and_dup_fd)
- init_module PatchByteAt ×3 (chrdev isolation)

### Infrastructure Fix: fire-and-poll child wait

`sysfs_unbind_fire_and_poll` in `guarded_sysfs/driver_ops.rs` now waits for
the child process to exit after the driver symlink disappears. In the kernel,
`driver_sysfs_remove()` runs BEFORE the `.remove` callback — the symlink
disappears while `nv_pci_remove` still holds `device_lock`. Waiting for child
exit guarantees `device_lock` is released.

### DKMS Version Pinning

`NVIDIA_470_DKMS_VERSION = "470.256.02"` hardcoded in `sovereign_handoff/config.rs`.
The `discover_dkms_version` function selected nvidia-580 (lexicographic max) instead
of nvidia-470 for Titan V.

## Key Findings

1. **MODULE_NAME is baked into .rodata** — renaming module to nvsov only changes
   ELF module name, not C string literals. Cap/procfs functions create
   `/proc/driver/nvidia/` paths that collide with host driver.

2. **nvidia_exit_module disassembly** (0x11f0, 168 bytes) reveals exact call chain:
   `nv_uvm_exit` → `nv_pci_unregister_driver` → `nvidia_unregister_module` →
   `nv_teardown_pat_support` → 2× `nv_kthread_q_stop` → `rm_destroy_event_locks` →
   `rm_shutdown_rm` → `nv_destroy_rsync_info` → `nvswitch_exit` → `nvlink_core_exit` →
   `nv_cap_drv_exit` → `nv_module_resources_exit` → `nv_cap_destroy_entry` →
   `nv_procfs_exit` → `nv_memdbg_exit`.

3. **nv_close_device contains usage_count decrement** — NOPing it prevents
   nv_pci_remove from ever completing (busy-wait for usage_count==0).

## Test Status

- 720/720 lib tests pass (toadstool-cylinder)
- 0 clippy warnings
- All patch set structure tests updated

## For Next Agent

Run #6 is ready. Binaries installed. System needs reboot (stuck nvsov module from Run #5).
After reboot, restart toadstool-ember and execute:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"sovereign.warm_handoff","params":{"bdf":"0000:02:00.0","strategy":"nvidia_catalyst_minimal_nop_titanv"}}' | socat -t 420 - TCP:127.0.0.1:<PORT>,connect-timeout=10
```

If Run #6 succeeds (driver_override + drivers_probe + rmmod all clean):
- Proceed to `channel-adoption` (adopt_rm_channel post-swap)
- Then `shader-dispatch` (sovereign shader execution)

If Run #6 still locks up: the teardown/rebind path may need further analysis
of `nv_stop_device` internals (rm_ref_dynamic_power, nv_acpi_unregister_notifier).
