# Experiment 218 — nvidia-470 nvsov Dual-Load Injection

**Date**: 2026-05-21
**Status**: IN PROGRESS — co-load isolation solved, module loads alongside nvidia-580, reboot needed to clear zombie module from test oops
**Hardware**: 2x NVIDIA Titan V (GV100), BDFs `0000:02:00.0`, `0000:49:00.0`
**Dependency**: Exp 217 (TPC wall confirmed firmware-dependent), Exp 216 (kernel health clean)

## Objective

Load nvidia-470 as a renamed module (`nvsov`) alongside the host nvidia-580
display driver, use it as the warm handoff seeder for Titan V, and verify that
TPC PRI ring stations survive the `nvsov` → `vfio-pci` warm swap.

nvidia-470 fully initializes the compute domain on Volta (SEC2→ACR→PMU→GPCCS→
FECS→TPC PRI stations). If TPC stations survive the warm swap, `classify_tier()`
should return `WarmCompute` with `tpc_alive = true` — breaking the TPC wall.

## Why This Works (Theory)

nvidia-470 is the last proprietary driver that supports Volta without GSP
firmware requirements. It performs full GR initialization including:
- PMU firmware load via ACR/SEC2
- GPCCS boot with signed firmware
- TPC PRI ring station creation as part of GPCCS init
- FECS boot + GR engine context setup

After init, unbinding nvidia-470 (with teardown NOPs applied) should preserve
all this state — including TPC PRI stations — through the vfio-pci bind.

## Existing Infrastructure

### Already Implemented

| Component | Location | Status |
|-----------|----------|--------|
| DKMS module discovery | `kmod::find_dkms_module("nvidia", "470.256.02")` | Ready |
| Binary patching | `PatchSet::nvidia_warm_handoff()` — 6 NOP targets | Ready |
| Module rename | `module_patch::rename_module_identity()` nvidia→nvsov | Ready |
| Handoff config | `HandoffConfig::nvidia_patched_titanv()` | Ready |
| Handoff pipeline | `execute_handoff()` — 8-step diesel engine | Ready |
| RPC entry | `sovereign.warm_handoff` strategy `nvidia_patched_titanv` | Ready |
| Build recipe | `build_nvidia470_kernel617.sh` (isolated /tmp) | Ready |
| Warm-preserving swap | `nvsov` in glowplug `is_warm_preserving_swap()` | Ready (Exp 217) |
| Tier classifier | `classify_tier()` with `tpc_alive` check | Ready (Exp 217) |

### Blocked / Needs Work

| Blocker | Description | Resolution |
|---------|-------------|------------|
| Symbol collisions | `nvidia_*` exported symbols conflict with nvidia-580 | Compile-time rename or expanded NOP set |
| procfs conflict | `/proc/driver/nvidia/` registration | NOP `nv_procfs_init` / `nv_register_procfs` |
| chardev conflict | `/dev/nvidia*` character device registration | NOP `nvidia_register_module` / `nv_register_chrdev` |
| UVM conflict | `/dev/nvidia-uvm` device | NOP UVM init (not needed for warm handoff) |

## Trials

### Trial 1: Compile-Time Rename via agentReagents Build

Use `build_nvidia470_kernel617.sh` pattern to build nvidia-470 with
`MODULE_BASE_NAME=nvsov` set in the kernel module Makefile. This changes the
module identity at compile time, avoiding all runtime rename issues.

```bash
# In agentReagents build environment:
cd /tmp/nvidia-470.256.02-build
sed -i 's/MODULE_BASE_NAME.*/MODULE_BASE_NAME = nvsov/' nvidia/Makefile
make -j$(nproc) module
# Output: nvsov.ko with all symbols prefixed differently
```

**Test:** `insmod /tmp/nvsov.ko` alongside loaded nvidia-580. Check `lsmod`
for both `nvidia` and `nvsov`. If loaded, bind Titan V.

### Trial 2: Expanded Runtime NOP Set

If compile-time rename is insufficient (some symbols are still exported with
`nvidia_` prefix regardless of `MODULE_BASE_NAME`), expand the runtime patch
set to NOP all conflicting registration:

```rust
PatchSet {
    name: "nvidia_warm_handoff_isolated",
    targets: vec![
        // Existing teardown NOPs
        ("nv_pci_remove", RetAtEntry),
        ("gpuStateUnload_IMPL", RetAtEntry),
        ("gpuStateDestroy_IMPL", RetAtEntry),
        // Co-load isolation NOPs
        ("nv_procfs_init", RetAtEntry),       // /proc/driver/nvidia/
        ("nv_register_chrdev", RetAtEntry),    // /dev/nvidia*
        ("nvidia_uvm_init", RetAtEntry),       // /dev/nvidia-uvm
        ("rm_init_adapter", RetAtEntry),       // RM control device
    ],
}
```

**Test:** Patch, rename, insmod, bind Titan V, warm swap to vfio-pci, classify.

### Trial 3: Full Warm Handoff Pipeline

Once `nvsov` loads alongside nvidia-580, run the diesel engine pipeline:

```bash
# Via RPC:
toadstool-rpc sovereign.warm_handoff '{"bdf":"0000:49:00.0","strategy":"nvidia_patched_titanv"}'
```

**Success criteria:**
- `nvsov` loads without kernel taint/oops
- Titan V binds to `nvsov` and completes init (dmesg shows GR init)
- Warm swap to `vfio-pci` preserves state (FLR disabled)
- `classify_tier()` returns `WarmCompute` with `tpc_alive = true`
- `sovereign.experiment --stage 1` shows TPC0 control ≠ `0xBADF5040`

## Success Criteria

| Metric | Threshold |
|--------|-----------|
| `nvsov` co-loads with nvidia-580 | No kernel oops/taint |
| Titan V binds to nvsov | dmesg: GR init complete |
| TPC0 control (0x504000) | Non-`0xBADF` value |
| `tpc_alive` | `true` |
| `classify_tier()` | `WarmCompute` (Tier 2) |
| PBDMA DEVICE error | bit 28 clear |
| (stretch) FECS context switch | completes |
| (stretch) QMD dispatch + readback | non-zero |

## Fallback

If nvidia-470 co-load proves impossible on bare metal:
1. **Exclusive nvidia-470 session:** Unload nvidia-580, load nvidia-470 for
   Titan V init, warm swap to vfio-pci, then reload nvidia-580 for display.
   Requires X11/Wayland restart — acceptable for experiment.
2. **agentReagents VM capture:** Run nvidia-470 in VM, capture full register
   state at TPC-alive moment via BAR0 snapshot. Replay captured state post-
   nouveau warm handoff. (mmiotrace in VM + register bisection)

## Files to Change

**toadStool:**
- `crates/core/cylinder/src/vfio/module_patch.rs` — Expanded NOP set for co-load isolation
- `crates/core/cylinder/src/vfio/kmod.rs` — Build-from-source support for agentReagents path
- `crates/core/glowplug/src/warm_init.rs` — Update `nvidia_patched_titanv` plan docs

**agentReagents:**
- `tools/k80-sovereign/build_nvidia470_kernel617.sh` — Add `MODULE_BASE_NAME=nvsov`

**hotSpring:**
- `experiments/218_NVIDIA470_NVSOV_DUAL_LOAD.md` — This document
- `EXPERIMENT_INDEX.md` — Add Exp 218

## Related Experiments

| Exp | Relevance |
|-----|-----------|
| 217 | BAR0 path CLOSED — TPC stations are firmware-mediated |
| 216 | Kernel build environment healthy (autoconf.h clean) |
| 211 | PMU software path CLOSED (HS-locked) |
| 215 | TPC wall identified, experiment infrastructure created |
| 206 | ACR DMA boot proven — falcon firmware upload works |
| 195 | Driver Lab: Mesa vs Vendor analysis |
| 194 | Cold/warm boot architecture framework |
