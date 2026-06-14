# Experiment 218 — nvidia-470 nvsov Dual-Load Injection

**Date**: 2026-05-21 → 2026-05-22
**Status**: COMPLETE — dual-load co-existence achieved, warm handoff succeeds, Tier 0 (Cold) result
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

## Results

### Trial 3 — Full Warm Handoff (2026-05-22)

**Outcome: SUCCESS (infrastructure) / NEGATIVE (TPC sovereignty)**

The patched nvidia-470 module (`nvso13`) loaded alongside nvidia-580 and completed
a full warm handoff cycle. All 8 pipeline steps passed. Key findings:

```
module_prep:      ✅ 12/17 patches applied, renamed nvidia→nvso13
unbind_current:   ✅ vfio-pci unbound (+ audio sibling)
deferred_insmod:  ✅ dual-load module loaded + bound via driver_override
seeder_bind:      ✅ driver=nvso13 bound to 0000:49:00.0
seeder_settle:    ✅ 10s settle
warm_swap:        ✅ nvso13 → vfio-pci (warm_preserved=true)
tier_classify:    ✅ Tier 0: Cold boot — tpc_alive=false
module_cleanup:   ✅ rmmod nvso13 clean
```

**Tier Classification:** `cold` — PMC_ENABLE shows 4 engines, PRAMIN inaccessible,
TPC stations not alive. The 470 driver loaded and probed the GPU but our NOP set
prevented the full RM initialization (caps, procfs, nvlink, nvswitch, acpi, chardev
all stubbed). Without RM init completing, the GPU compute domain was not initialized.

### Bugs Found & Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `Invalid module format` (type 11, 32S) | Kernel 6.17 rejects nonzero relocation targets | `normalize_relocations` + type 11 handling |
| Post-objcopy corruption | objcopy re-created relocation conflicts | Moved objcopy BEFORE patching (pre-patch pipeline) |
| Offset domain mismatch | `nullify_relocations_at` compared file offsets vs section-relative | Added `target_sh_offset` resolution via `sh_info` |
| Wrong file offsets for all patches | `nm` returns section-relative VAs, not file offsets | `resolve_symbol_file_offsets()` — pure ELF parser with per-section offset |
| `.init.text` symbols misplaced | Single `.text` offset assumed for all symbols | Section-aware ELF symbol resolver |
| Ftrace clobber | `ret` at entry+0 destroyed ftrace call site | Ftrace-aware `RetAtEntry` — patches at entry+5 when preamble detected |
| `register_chrdev` conflict | Host nvidia owns major 195 | `NopCallAt(0x7f)` on `init_module` to skip `__register_chrdev` |
| `nvidia_register_module` NOPed | Module instance table empty → probe fails | Removed from NOP set |
| `No such device` | insmod before GPU unbound from vfio-pci | Deferred insmod after unbind + `driver_override` |
| procfs conflicts (caps, nvlink, nvswitch) | Duplicate `/proc/driver/nvidia-*` entries | NOPed `nv_cap_*_init`, `nvlink_core_init`, `nvswitch_init` |
| `nv_cap_alloc` NULL deref | Cap table uninitialized after NOPing init | NOPed `nv_cap_create_dir_entry`, `nv_cap_create_file_entry`, `nv_cap_destroy_entry` |
| LOCAL symbols not found | ELF parser filtered to `STB_GLOBAL` only | Accept all `STT_FUNC` symbols regardless of binding |
| `nopcall` relocation not nullified | `patch_ranges` filter excluded `nopcall` prefix | Added `nopcall` to filter predicate |

### Final NOP Set (17 targets)

| Symbol | Strategy | Section | Purpose |
|--------|----------|---------|---------|
| `nv_pci_remove` | RetAtEntry | .text | Preserve GPU state on unbind |
| `nv_cap_init` | Ret1AtEntry | .text | Skip cap table init (returns "success") |
| `nv_cap_drv_init` | Ret1AtEntry | .text | Skip cap driver init |
| `nv_procfs_init` | RetAtEntry | .text | Skip /proc/driver/nvidia/ |
| `nv_cap_procfs_init` | RetAtEntry | .text | Skip cap procfs |
| `nvlink_core_init` | Ret1AtEntry | .init.text | Skip nvlink subsystem |
| `nvswitch_init` | Ret1AtEntry | .text | Skip nvswitch subsystem |
| `nv_acpi_init` | RetAtEntry | .text | Skip ACPI handler duplication |
| `nv_cap_create_dir_entry` | RetAtEntry | .text | Stub cap dir creation |
| `nv_cap_create_file_entry` | RetAtEntry | .text | Stub cap file creation |
| `nv_cap_destroy_entry` | RetAtEntry | .text | Stub cap cleanup |
| `init_module` | NopCallAt(0x7f) | .init.text | Skip __register_chrdev call |

### Analysis

The dual-load succeeds mechanically but the nvidia-470 driver's RM initialization
is too deeply coupled to the subsystems we NOPed. The driver loaded, registered its
PCI driver, probed the GPU, but without caps/procfs/nvlink/chardev the RM core
(`rm_init_rm`) likely skipped GPU engine initialization. The result is equivalent
to a partial-init state — the driver owns the device but hasn't booted the compute
pipeline (SEC2→ACR→PMU→GPCCS→FECS→TPC).

**Next steps for TPC sovereignty require one of:**
1. **Selective RM enablement** — Allow more RM subsystems to init while still
   preventing host conflicts. Needs careful bisection of which NOPs can be removed.
2. **Exclusive nvidia-470 session** — Unload nvidia-580 entirely, load nvidia-470
   for full init, warm swap, then reload nvidia-580. Requires display restart.
3. **Register state capture** — Run full nvidia-470 in VM, capture complete BAR0
   register state at TPC-alive moment, replay captured state post-warm-handoff.

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
