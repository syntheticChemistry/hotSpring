# Experiment 234: Catalyst Minimal NOP — Device Registration Unblock

**Date:** 2026-05-31
**Status:** IN PROGRESS
**Hardware:** Titan V (GV100, SM70) — 0000:02:00.0
**Prerequisite:** Exp 233 Run #1 (device_alloc 0x22 failure characterized)

## Objective

Unblock RM device_alloc (step 5/26 of the RM object tree) by restoring the
nvidia cap subsystem that was NOP'd in the catalyst_handoff patch set. Exp 233
proved the NOP set allows full DEVINIT (23 engines) but blocks RM from
populating its GPU manager device table, causing device_alloc to return 0x22
(NV_ERR_OBJECT_NOT_FOUND).

## Root Cause Analysis (from nvidia-470 source trace)

### Module Init Chain
```
nvidia_frontend_init_module() → nvidia_init_module():
  1. nv_procfs_init()         — /proc/driver/nvidia/ (collides with host)
  2. nv_caps_root_init()      — calls nv_cap_init("driver/MODULE_NAME")
  3. nv_module_init()         — nv_cap_drv_init, rm_init_rm
  4. nv_drivers_init()        — nvidia_register_module, nv_pci_register_driver
  5. __register_chrdev(major)  — (patched to dynamic major 0)
```

### PCI Probe Chain (per-GPU)
```
nv_pci_probe():
  1. pci_enable_device, BAR setup
  2. rm_init_private_state(sp, nv)  — CLOSED BINARY, NOT NOP'd
  3. nv_linux_add_device_locked(nvl)
  4. nvidia_frontend_add_device()   — assigns minor number
  5. nv_procfs_add_gpu(nvl)         — NOT NOP'd, uses proc_nvidia
```

### Device Open Chain (first open)
```
nvidia_open() → nv_open_device() → nv_start_device():
  1. MSI/MSI-X setup, request_threaded_irq
  2. rm_init_adapter(sp, nv)  — CLOSED BINARY, runs DEVINIT (PMC_ENABLE 4→23)
  3. nv_acpi_register_notifier
```

### Key Finding
The cap subsystem (`nv_cap_init`, `nv_cap_drv_init`, `nv_cap_create_dir_entry`,
`nv_cap_create_file_entry`) initializes data structures that `rm_init_private_state`
and `rm_init_adapter` (in the closed RM binary) may use for GPU manager registration.
With Ret1AtEntry (fake handles), these structures are fake — RM's NULL checks pass
but the underlying data is not functional.

### Collision Analysis
- `nv_cap_init("driver/nvsov/capabilities")` uses MODULE_NAME → no collision
- `nv_cap_drv_init` → `alloc_chrdev_region(... "nvidia-caps")` → dynamic major → no collision
- `nv_cap_procfs_init` → creates `/proc/driver/nvidia-caps/` → **COLLIDES** (hardcoded)
- `nv_procfs_init` → creates `/proc/driver/nvidia/` → **COLLIDES** (hardcoded)

## Strategy

### `nvidia_catalyst_minimal_nop` Patch Set

**Un-NOP'd (restored to original code):**
- `nv_cap_init` — creates /proc/driver/nvsov/capabilities/ (nvsov namespace)
- `nv_cap_drv_init` — initializes cap hash tables, cdev, internal structures
- `nv_cap_create_dir_entry` — creates real cap directory entries
- `nv_cap_create_file_entry` — creates real cap file entries

**Changed to Ret0AtEntry:**
- `nv_cap_procfs_init` — stub returns 0 (skip /proc/driver/nvidia-caps/ creation)
- `nv_procfs_init` — stub returns 0 (skip /proc/driver/nvidia/ creation)
- `nv_acpi_init` — stub returns 0 (skip ACPI handler registration)

**Unchanged from catalyst_handoff:**
- All teardown NOPs (nv_close_device, nv_pci_remove, rm_disable_adapter, etc.)
- os_is_administrator → Ret1AtEntry (admin bypass)
- nv_cap_validate_and_dup_fd → Ret1AtEntry (cap validation bypass)
- nvlink_core_init, nvswitch_init → Ret1AtEntry
- init_module patches (dynamic chrdev major, return 0)
- rm_shutdown_rm, nv_destroy_rsync_info → RetAtEntry (exit crash prevention)

### Expected Outcome

If the cap system was the blocker:
1. `nv_cap_drv_init` initializes hash tables and cdev → `g_nv_cap_drv.initialized = true`
2. `nv_cap_init` creates real cap root directory → `nvidia_caps_root` is a valid pointer
3. `rm_init_private_state` uses real cap infrastructure → registers GPU in device table
4. `GPU_GET_PROBED_IDS` returns the GPU → `device_alloc` succeeds with status=0x00
5. Full RM channel tree (device → subdevice → VA → memory → TSG → GPFIFO → compute)
6. Post-swap `adopt_rm_channel()` picks up the RM channel

If the cap system is NOT the blocker:
- device_alloc still returns 0x22 → need to investigate rm_init_private_state via
  binary disassembly or alternative approach (Vector B: golden context path)

## Code Changes

### New PatchStrategy: Ret0AtEntry
- `types.rs`: Added `Ret0AtEntry` variant — `xor eax,eax; ret` (3 bytes at entry+5)
- `apply.rs`: Added application logic and reapply_nops support
- Used for functions returning `int` where 0 = success and undefined rax is unsafe

### New Patch Set: nvidia_catalyst_minimal_nop
- `patch_sets/nvidia.rs`: `PatchSet::nvidia_catalyst_minimal_nop()`
- `patch_sets/mod.rs`: Added to `by_name()` and `by_profile()` dispatchers
- 19 targets (vs 22 in catalyst_handoff — 4 cap functions removed, 3 changed strategy)

### Tests
- `tests.rs`: `nvidia_catalyst_minimal_nop_patch_set_structure` validates structure
- `tests.rs`: `apply_single_patch_ret0_at_entry` validates Ret0AtEntry application
- 720 lib tests pass, 0 clippy warnings

## Pre-Run Validation
- `cargo check --workspace`: PASS (0 errors)
- `cargo test -p toadstool-cylinder`: PASS (720 tests + 2 ignored doctests)
- `cargo clippy -p toadstool-cylinder`: PASS (0 warnings)
- No zombie modules present
- toadstool-ember service: checked

## Results

### Run #1 — Kernel Crash (proc_dir_entry collision)

Un-NOPing cap functions caused `proc_register` crash. `nv_cap_init("driver/" MODULE_NAME)`
uses hardcoded `MODULE_NAME="nvidia"` in .rodata — renaming to `nvsov.ko` only changes
ELF module name, not string literals. Collision with host `/proc/driver/nvidia/capabilities/`.

**Fix:** Reverted to NOP all cap functions. Cap system cannot be un-NOP'd without patching
.rodata strings.

### Run #2 — Wrong Module (nvidia-580 instead of nvidia-470)

`discover_dkms_version("nvidia")` returned `"580.126.18"` (lexicographic max) instead of
`"470.256.02"`. Pipeline patched the nvidia-580 host driver module (18.7 MB) instead of
nvidia-470 (42.8 MB). nvidia-580 dropped GV100 support — different symbols, different
init_module layout, missing `nv_cap_procfs_init` and `nv_acpi_init` symbols.

5 patches failed:
- `nv_cap_procfs_init` — symbol not found (doesn't exist in nvidia-580)
- `nv_acpi_init` — symbol not found (renamed to `nv_acpi_methods_init` in nvidia-580)
- 3 `init_module` PatchByteAt — byte mismatches (different binary layout)

**Fix:** Hardcoded `NVIDIA_470_DKMS_VERSION = "470.256.02"` for all Titan V catalyst configs.

### Run #3 — 22/22 Patches OK, Seeder Bind Failed

All patches applied successfully. Module loaded. But GPU probe failed:
- `request_mem_region failed for 16M @ 0xd8000000` — BAR0 still claimed by STUCK nvsov
  module from Run #2's failed rollback
- `nv_pci_remove` was NOP'd (RetAtEntry) so `release_mem_region` was never called during
  previous rollback unbind → BAR0 leaked
- Module rmmod hung in "going" state (refcount=-1) because exit function tried to clean up
  uninit'd state (chrdev, kthreads, caps)

**Root cause (2 bugs):**
1. **BAR0 iomem leak:** `nv_pci_remove` NOP prevents `release_mem_region` during unbind
2. **Module exit hang:** `cleanup_module` → `nvidia_exit_module` tries to stop kthreads,
   unregister chrdev, etc. that were never properly initialized

**Fix (for Run #4):**
1. UN-NOP `nv_pci_remove` — let it run but `rm_disable_adapter` and `rm_shutdown_adapter`
   are separately NOP'd, so GPU state is preserved while BAR0 is properly released
2. NOP `cleanup_module` with RetAtEntry — prevents exit function from hanging on uninit'd state
3. Reboot required to clear stuck module from Run #3

### Run #4 — driver_override D-state Hang (cleanup_module NOP regression)

22/22 patches applied. Module loaded correctly with nvidia-470. GPU fully initialized
(PMC_ENABLE=0x5fecdff1, 23 engines warm). Firmware captured (FECS/GPCCS IMEM).
Unbind from nvsov completed in 2s.

**Crash point:** Writing to `driver_override` for vfio-pci rebind hung in D-state.

**Root cause:** `cleanup_module` NOP'd (RetAtEntry) leaves the PCI driver registered
after module free. The sysfs `driver_override` write needs `device_lock`, which is held
by the stale PCI driver registration.

**Fix:** Removed `cleanup_module` NOP — let `nvidia_exit_module` run so
`nv_pci_unregister_driver()` properly deregisters the PCI driver.

### Run #5 — nv_pci_remove busy-wait (331s timeout, module exit hang)

`cleanup_module` NOP removed. Pipeline reached same point: GPU init, firmware capture,
unbind. Child process stayed in `nv_pci_remove` for 331 seconds, exceeding the handoff
deadline.

**Root cause (2 bugs):**
1. **usage_count busy-wait:** `nv_close_device` NOP'd with RetAtEntry — prevents
   `usage_count` decrement when `rm_trigger` closes `/dev/nvidia0`. `nv_pci_remove`
   busy-waits forever for `usage_count == 0`.
2. **Module exit hang:** `nvidia_exit_module` calls `nv_kthread_q_stop()` twice
   (for `nv_deferred_close_kthread_q` and `nv_kthread_q`). Kthread stop hangs
   because queues have pending work from the catalyst pipeline.

**Fix:**
1. Un-NOP'd `nv_close_device` — lets `usage_count` decrement. The dangerous calls
   inside `nv_stop_device` (`rm_disable_adapter`, `rm_shutdown_adapter`) are separately
   NOP'd.
2. Added `nv_kthread_q_stop` as RetAtEntry NOP target (GLOBAL FUNC at 0x16340, 160 bytes)
   — prevents both kthread stop calls from hanging during module exit.
3. Fixed fire-and-poll race: `sysfs_unbind_fire_and_poll` now waits for the child
   process to exit (ensuring `nv_pci_remove` completes and `device_lock` is released)
   before proceeding with `driver_override` writes.

### Run #6 — (pending reboot + execution)

System locked during Run #5 (stuck nvsov refcnt=-1). All three fixes installed.

### Fire-and-Poll Race Fix

The `sysfs_unbind_fire_and_poll` mechanism detected driver symlink removal BEFORE
`nv_pci_remove` completed. In the kernel, `driver_sysfs_remove()` runs before the
`.remove` callback. The parent saw "driver cleared" and immediately wrote to
`driver_override`, which needs `device_lock` still held by `nv_pci_remove`.

Fixed in `guarded_sysfs/driver_ops.rs`: after driver symlink clears, poll `waitpid`
on the child until it exits (guaranteeing `device_release_driver()` completed and
`device_lock` is free).

## Status: PAUSED — System Lockup Pattern

Exp 234 is paused due to repeated system lockups during the warm handoff pipeline.
The experiment has made significant progress:
- GPU fully initializes (23 engines warm, firmware captured)
- All patches apply correctly (22/22 with correct nvidia-470 module)
- Unbind from nvsov completes (BAR0 released)
- GPU state survives the swap (PMC_ENABLE preserved post-unbind)

But three sequential bugs in the teardown/rebind path caused system lockups:
1. `cleanup_module` NOP → dangling PCI driver → D-state on `driver_override`
2. `nv_close_device` NOP → usage_count leak → 331s busy-wait → module exit hang
3. Fire-and-poll race → `device_lock` contention → D-state

All three are now fixed. Awaiting clean execution (Run #6) after reboot.

## Files Changed (toadStool)
- `crates/core/cylinder/src/vfio/module_patch/types.rs` — Ret0AtEntry strategy
- `crates/core/cylinder/src/vfio/module_patch/apply.rs` — Ret0AtEntry application
- `crates/core/cylinder/src/vfio/module_patch/patch_sets/nvidia.rs` — new patch set
  + Run #3: NOP cleanup_module, un-NOP nv_pci_remove
  + Run #4: un-NOP cleanup_module (PCI driver must unregister)
  + Run #5: un-NOP nv_close_device (usage_count decrement), NOP nv_kthread_q_stop
- `crates/core/cylinder/src/vfio/module_patch/patch_sets/mod.rs` — dispatcher registration
- `crates/core/cylinder/src/vfio/module_patch/tests.rs` — new tests (updated per run)
- `crates/core/cylinder/src/vfio/module_patch/mod.rs` — doctest fence fix (ignore)
- `crates/core/cylinder/src/vfio/sovereign_handoff/config.rs` — hardcoded nvidia-470 DKMS version
- `crates/core/cylinder/src/vfio/sovereign_handoff/tests.rs` — updated version assertions
- `crates/core/cylinder/src/vfio/guarded_sysfs/driver_ops.rs` — fire-and-poll child wait fix

## Current Patch Set: nvidia_catalyst_minimal_nop (21 targets)

**NOT NOP'd (must run):**
- `nv_close_device` — decrements usage_count
- `nv_pci_remove` — calls release_mem_region for BAR0
- `cleanup_module` — calls nv_pci_unregister_driver for PCI driver cleanup

**NOP'd (RetAtEntry / Ret0AtEntry / Ret1AtEntry):**
- `rm_disable_adapter`, `rm_shutdown_adapter` — preserve GPU state during teardown
- `nv_kthread_q_stop` — prevent module exit hang (kthread stop deadlock)
- `rm_shutdown_rm`, `nv_destroy_rsync_info` — closed-binary hang prevention
- `nv_procfs_init`, `nv_cap_procfs_init`, `nv_acpi_init` — Ret0AtEntry (namespace collision)
- `nvlink_core_init`, `nvswitch_init` — Ret1AtEntry (not needed)
- `nv_cap_init`, `nv_cap_drv_init`, `nv_cap_create_dir_entry`, `nv_cap_create_file_entry` — Ret1AtEntry (MODULE_NAME collision)
- `nv_cap_validate_and_dup_fd`, `nv_cap_close_fd`, `nv_cap_destroy_entry` — access control bypass
- `os_is_administrator` — Ret1AtEntry (admin bypass)
- `init_module` PatchByteAt ×3 — chrdev isolation (dynamic major 0)

## Sovereign Compute Validation (June 1, 2026 — S284)

While Exp 234 Run #6 remains pending reboot, the sovereign compute pipeline was
validated end-to-end on the existing fleet using the post-primordial QCD evolution
stack (Phases 1-8 complete):

- **Lockup defense matrix**: 11/11 passed — all 5 diesel-engine defenses available
- **Compute trio pipeline**: 16/19 — Yukawa MD + Wilson plaquette compiled via
  coralReef, dispatched via toadStool, results received. 7/9 barrier shaders compiled.
- **Compute dispatch (Exp 152)**: FULL PASS — BLAKE3 witness pipeline live
- **Silicon capabilities (RTX 5060)**: 9/12 — FMA, DF64, workgroup reduction pass
- **MMIO probes**: Both Titan Vs alive (BOOT0 0x140000a1, 23 engines, PTIMER ticking)

Remaining failures are upstream: coralReef compiler bugs (2 shaders), songbird
capability advertisement gap, barracuda ReduceScalarPipeline readback on Blackwell.

Status: PAUSED — Run #6 cleared, reboot required to clear stuck nvsov module.

### Run #6 — Hard Lockup at rm_trigger (June 1, 2026)

Post-reboot to clear Run #5 nvsov. Clean slate: no nvidia/vfio modules, both Titan Vs
on vfio-pci, VRAM alive. Ember healthy (3 GPUs, all Alive).

**Kill point: `rm_trigger` chardev open (17:04:08.868)**

Timeline:
1. 17:04:07 — `sovereign.warm_handoff` RPC received (strategy: nvidia_catalyst_minimal_nop_titanv)
2. 17:04:07 — RACE: second RPC received 500ms later — both trigger anchor release
3. 17:04:07 — First RPC fails fast (success:false, total_ms:0). Second continues.
4. 17:04:07 — GPU COLD (PMC popcount=4). SBR unsuppressed for 02:00.0.
5. 17:04:08 — nvsov.ko patched: 21/21 patches applied (nvidia-470.256.02 correct)
6. 17:04:08 — **PCI disable I/O error** — config space already bad before insmod
7. 17:04:08 — nvsov insmod: 400ms, module loads, probes 02:00.0
8. 17:04:08 — `rm_trigger` spawned (major=507, channel=true, bdf=02:00.0)
9. SILENCE — hard lockup. No kernel or userspace output after rm_trigger spawn.
10. Forced reboot to recover. FAT-fs volumes not properly unmounted.

**Root cause analysis — what got past the clutch:**

1. **rm_trigger enters kernel RM init**: chardev open triggers `nv_open → rm_init_adapter
   → rm_init_private_state`. This runs inside nvidia's closed-source kernel blob. The
   lockup is a hard kernel deadlock or infinite loop inside RM GPU initialization.
   The 450s catalyst watchdog is powerless — it runs in userspace and gets frozen by
   the kernel lockup.

2. **Double RPC race**: Two warm_handoff RPCs 500ms apart both trigger anchor release
   (SBR unsuppression, no_bus_reset cycling). The first fails fast but may have left
   PCI bus state inconsistent. Evidence: `enable` sysfs I/O error at 17:04:08.305.

3. **GPU was COLD**: PMC popcount=4 means only basic engines alive. RM must do full
   cold init (VBIOS, falcon boot, HBM2 training, RM channel tree). Something in this
   full init path hangs the PCIe bus.

4. **Lock debugging disabled**: Kernel tainting by nvsov disables lockdep, so no
   deadlock detection available.

**Evolution targets for toadStool:**

- [ ] Mutex/gate `sovereign.warm_handoff` — reject concurrent RPCs for same BDF
- [ ] Pre-rm_trigger PCI health check — abort if config space returns I/O errors
- [ ] rm_trigger timeout: spawn in cgroup with hardware watchdog NMI as backstop
- [ ] Investigate: can we skip rm_trigger entirely and use Tier 1 (MMIO-only) path?
- [ ] NMI watchdog capture: enable `nmi_watchdog=1` + `softlockup_panic=1` to get
  a stack trace on next lockup instead of silent hang
- [ ] Consider: warm the GPU via nouveau HBM2 training BEFORE loading nvsov
  (cold GPU init is the suspected kill vector)
