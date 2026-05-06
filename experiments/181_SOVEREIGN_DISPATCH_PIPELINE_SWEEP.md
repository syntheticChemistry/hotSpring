# Experiment 181: Sovereign Dispatch Pipeline Sweep

**Date:** May 2, 2026
**Status:** Two-layer root cause: subchannel mismatch FIXED + GR engine never initializes on K80/nouveau/kernel 6.17
**Hardware:** RTX 5060 (GB206, SM120), Titan V (GV100, SM70), Tesla K80 (GK210B, SM37)
**Primal:** coralReef (coral-driver, coral-reef)
**Predecessor:** Exp 180 (Three-GPU HW validation)

## Objective

Validate the full sovereign pipeline (WGSL → coral-reef compile → SASS → dispatch → readback)
on all three GPU generations. For Titan V and K80, the goal is pure Rust/ecoPrimals dispatch
WITHOUT proprietary nvidia driver dependency.

## Hardware Configuration (post-BIOS update)

| GPU | BDF | Driver | SM |
|-----|-----|--------|----|
| RTX 5060 | `0000:21:00.0` | nvidia | SM120 |
| Titan V | `0000:02:00.0` | vfio-pci | SM70 |
| K80 GPU0 | `0000:4b:00.0` | vfio-pci | SM37 |
| K80 GPU1 | `0000:4c:00.0` | vfio-pci | SM37 |

## RTX 5060 (Blackwell SM120) — 8/8 PASS

Full sovereign pipeline validated via CUDA driver path:

| Test | Result |
|------|--------|
| `cuda_device_opens` | PASS |
| `cuda_e2e_write_42` | PASS — WGSL → SM120 SASS → dispatch → readback = 42 |
| `cuda_cubin_sass_write_42_sm120` | PASS |
| `cuda_ptx_write_42_direct` | PASS |
| `cuda_e2e_array_length` | PASS |
| `cuda_e2e_multi_binding_copy` | PASS |
| `cuda_e2e_multi_binding_array_length` | PASS |
| `cuda_e2e_num_workgroups` | PASS |

Pipeline: WGSL → `coral-reef compile_wgsl_full(Sm120)` → SASS → `NvUvmComputeDevice::dispatch` → readback verified.

## Titan V (Volta SM70) — BLOCKED

### Approach 1: Nouveau DRM Dispatch

nouveau + nvidia coexistence confirmed (kernel 6.17.9). Titan V bound to nouveau at `renderD129`.

| Stage | Result |
|-------|--------|
| `NvDevice::open()` | PASS — SM70 detected, compute_class=0xC3C0, new_uapi=true |
| Memory alloc/upload/readback | PASS — 3/3 basic ops |
| Dispatch (`nouveau_exec_signal`) | FAIL — errno 19 (ENODEV) |
| dmesg | `pmu: firmware unavailable` + `fifo:...errored - disabling channel` |

**Root cause**: GV100 has no PMU firmware for nouveau. FECS/GPCCS load via SEC2/ACR, but
compute channels error without PMU managing power/clocks. Channel disabled by kernel before
exec ioctl returns.

### Approach 2: VFIO Warm Handoff (nouveau → vfio-pci)

Livepatch `livepatch_nvkm_mc_reset.ko` loaded (kernel 6.17.9 match). GV100 has no FLR
(`FLReset-` in PCI capabilities). Manual warm handoff executed:

1. Bound to nouveau — GR initialized, no errors during init
2. Set `driver_override=vfio-pci`, unbound from nouveau, bound to vfio-pci
3. Ran `vfio_dispatch_warm_handoff` with `CORALREEF_VFIO_BDF=0000:02:00.0`

| Stage | Result |
|-------|--------|
| VFIO open (warm) | PASS — SM70 detected, BAR0 mapped, bus master enabled |
| PFIFO init (warm) | PASS — liveness probe: preempt ACK received |
| FECS state post-rebind | **HRESET** — cpuctl=0x00000012, hs_mode=true |
| STARTCPU recovery | FAIL — "HS mode prevents host restart" |
| FECS method interface | FAIL — timeout: status2=0xffffffff |
| Dispatch + sync | FAIL — fence timeout after 5000ms |

**Root cause**: On Volta, FECS and GPCCS are **HS-mode (Hub Secure) falcons**. The host CPU
cannot restart them via STARTCPU — only SEC2/ACR can boot them through the secure boot chain.
The livepatch blocks `nvkm_mc_reset` but nouveau's GR fini path still resets individual falcons.
All four falcons (FECS, GPCCS, PMU, SEC2) found in HRESET after rebind.

### Titan V — What Works

- VFIO device open (cold and warm)
- PFIFO channel creation (V2 5-level page tables, GV100 per-runlist format)
- PFIFO liveness (preempt ACK)
- GPFIFO doorbell + GP_GET poll infrastructure
- WGSL → SM70 SASS compilation

### Titan V — What's Needed

**SEC2 → ACR → FECS cold boot chain** (GAP-HS-030): Implement the secure falcon boot
sequence in `coral-driver` that boots SEC2 from firmware blobs, SEC2 configures WPR and
loads ACR, ACR boots FECS/GPCCS with signed firmware. All firmware blobs are available
in `/lib/firmware/nvidia/gv100/`.

## Tesla K80 (Kepler SM37) — BLOCKED

### Approach 1: Nouveau DRM Dispatch

K80 bound to nouveau at `renderD130`. Clean init: NVIDIA GK110B, 12GB GDDR5, DRM initialized.

| Stage | Result |
|-------|--------|
| `NvDevice::open()` | PASS — SM37 detected, compute_class=0xA1C0, new_uapi=true |
| Memory alloc/free | PASS |
| Dispatch (new UAPI) | FAIL — `syncobj_wait` errno 62 (ETIME) after 45s |
| Dispatch (legacy UAPI) | FAIL — `gem_cpu_prep` errno 16 (EBUSY) after 85s |
| dmesg | `SCHED_ERROR 0a [CTXSW_TIMEOUT]` — context switch to compute channel fails |

**Root cause**: Kepler compute context switches fail on kernel 6.17. FECS can't complete the
switch to our compute channel. This happens on BOTH new UAPI (EXEC ioctl) and legacy UAPI
(pushbuf_submit). After several failed attempts, GPU timer stalls (`timer: stalled at ffffffffffffffff`).

The `hw_nv_nouveau` test also has a bug: `compile_for_sm` maps SM37 to `NvArch::Sm70`
(generates wrong ISA), though the CTXSW_TIMEOUT occurs before any shader execution.

### K80 — What Works (from Exp 180)

- VFIO device open (legacy path)
- Kepler 2-level page table + RAMFC creation
- PFIFO engine init + runlist submission
- Channel binding (PCCSR PENDING state)
- FECS firmware PIO upload to IMEM (verified)

### K80 — What's Needed

**GPC PGOB ungating** that survives through the full initialization path. The PGOB disable
code runs but GPCs remain power-gated. With nouveau DRM, the CTXSW_TIMEOUT indicates
GR engine context switching is broken at a lower level than the VFIO path attempts.

## Findings Summary

| Pipeline Stage | RTX 5060 (SM120) | Titan V (SM70) | K80 (SM37) |
|---------------|-------------------|----------------|------------|
| WGSL → SASS compile | DONE | DONE | DONE |
| VFIO/driver open | DONE | DONE | DONE |
| FECS/GR boot | KmodPromote (nvidia) | BLOCKED (HS requires SEC2/ACR) | BLOCKED (PGOB gates GPCs) |
| PFIFO channel | DONE | DONE | DONE |
| GPFIFO dispatch | DONE | DONE (infra) | DONE (infra) |
| Completion + readback | DONE | DONE (infra) | DONE (infra) |
| **End-to-end dispatch** | **PROVEN** | **BLOCKED** | **BLOCKED** |

## Key Discovery

**nouveau + nvidia kernel module coexistence works on kernel 6.17.9**. Both modules loaded
simultaneously, each binding to separate GPUs. This enables using nouveau for initialization
while nvidia handles production dispatch on other GPUs.

## Diagnostic Deep Dive

### K80 Push Buffer Subchannel Mismatch (FIXED)

`create_channel()` binds compute to subchannel **0** (single SubchanSpec entry). But
`PushBuf::compute_init()` sent `SET_OBJECT` on subchannel **1**. On nouveau, the kernel only
allocates engine resources for the subchannel index specified in `create_channel()` (index 0).
Methods sent to subchannel 1 route to an unbound slot — the PFIFO can't context-switch
the channel, causing `CTXSW_TIMEOUT` (SCHED_ERROR 0a).

**Root cause of CTXSW_TIMEOUT confirmed.** Fix: changed `compute_init()` and
`compute_dispatch_with_launch()` from `sub = 1` to `sub = 0`, matching `create_channel()`.
Also added missing `driver_const_handle` GEM to legacy pushbuf BO list (CBUF7 reference
without pinning could cause MMU fault). Unit tests updated. 743/743 pass.

### K80 `compile_for_sm` Bug (FIXED)

`hw_nv_nouveau.rs::compile_for_sm()` mapped SM37 → `NvArch::Sm70` (wrong ISA). Fixed to
use the full arch detection from `hw_sovereign_e2e.rs` — SM37 now correctly maps to `Sm35`.

### K80 Card Lockup

After repeated CTXSW_TIMEOUT failures, the K80's GPU timer stalled
(`timer: stalled at ffffffffffffffff`). Subsequent rebind attempts failed. The PLX PCIe
switch (PEX 8747) went to `rev ff` (all-ones = unresponsive). GPU0 disappeared from PCI
entirely. **System reboot required to recover.**

### Architecture Strategy Clarification

The long-term approach:
1. **nouveau/DRM** = reference solver for sovereign dispatch
2. **VFIO sovereign** = ultimate goal, informed by nouveau internals
3. **Proprietary drivers** (CUDA/UVM) = benchmarking infrastructure via `infra/benchScale/`

## Fixes Applied

| Fix | File | Detail |
|-----|------|--------|
| Subchannel 1→0 | `pushbuf.rs` | `compute_init` + `compute_dispatch_with_launch` now use SC 0 |
| Legacy BO list | `mod.rs` | Added `driver_const_handle` GEM to pushbuf BO list |
| compile_for_sm | `hw_nv_nouveau.rs` | SM37 → `NvArch::Sm35` (was incorrectly Sm70) |
| Unit tests | `pushbuf.rs` | Updated assertions from SC 1 to SC 0 |

## Deep Dive: K80 GR Engine Not Initializing (May 2, 2026)

After full power cycle and fresh nouveau bind, the subchannel fix was validated but
CTXSW_TIMEOUT persists. Investigation revealed a **deeper issue**: nouveau on kernel
6.17.9 does NOT initialize the GR engine for K80 at all.

### Evidence

1. **Zero GR messages** in dmesg: no `gr:`, `fecs:`, `gpc:`, `falcon:` messages.
   Init sequence: `GK110B → bios → fb → drm` — GR completely absent.
2. **debugfs `internal_clients`** shows only `fbdev` — no GR, CE, or compute engines.
3. **Chip ID mismatch**: boot0 `0x0f22d0a1` decodes to chip `0xf2`. Module has
   `nvf0_chipset` (GK110=0xf0) and `nvf1_chipset` (GK110B=0xf1) but **no `nvf2`**.
   Nouveau prints "GK110B" (falls back to nvf1) but GR may not init for chip 0xf2.
4. **Built-in firmware exists**: `gk110b_gr_new`, `gk110b_gr_fwif`, and compiled-in
   FECS/GPCCS microcode (`gk110_grhub_code/data`, `gk110_grgpc_code/data`) are all
   present in the module binary (confirmed via `nm`/`strings`).
5. **Channel creation succeeds**: kernel accepts `KEPLER_COMPUTE_B` (0xA1C0) without
   error, but the engine backing it is dead.

### PLX PCIe Switch Fragility

The K80 behind PLX PEX 8747 is extremely fragile:
- `rmmod nouveau` triggers D3cold transition → PLX switch wedges to `rev ff`
- Any CTXSW_TIMEOUT cascade eventually wedges the PLX switch
- Recovery requires **full AC power cycle** (software reboot insufficient)
- Ember's D-state isolation helps for MMIO/sysfs but cannot prevent PCIe fabric wedge

### Hypothesis

GR engine init is **silently skipping** for K80 on kernel 6.17 because:
- The `nvf1_chipset` entry may not include `.gr = gk110b_gr_new` (most likely)
- OR `gk110b_gr_fwif` firmware probe fails silently for chip 0xf2
- OR GR init defers to first channel use but the channel doesn't trigger it

### Workaround Paths

1. **VFIO sovereign path** — bypass nouveau entirely. `coral-driver` already has
   FECS/GPCCS firmware loading for K80. Address PGOB power gating directly.
2. **Kernel patch** — add `nvf2_chipset` entry (or fix `nvf1` to cover 0xf2)
   with `.gr = gk110b_gr_new`. Requires kernel rebuild.
3. **Older kernel** — K80 compute on nouveau worked on kernel 5.x. Test 5.15 LTS.

## Next Steps (Priority Order)

1. **K80 VFIO sovereign path** — bypass nouveau. Fix PGOB GPC power gating in
   `coral-driver` to enable direct FECS boot. Firmware blobs available at
   `/lib/firmware/nvidia/gk210/`.
2. **Titan V SEC2/ACR boot chain** — implement cold sovereign FECS boot using
   available firmware (`sec2/`, `acr/`, `gr/` blobs in `/lib/firmware/nvidia/gv100/`).
3. **K80 kernel patch** — submit patch to add `nvf2_chipset` with GR support to
   upstream nouveau. This would fix nouveau compute for K80/GK210 on kernel 6.17+.
4. **PLX power management fix** — disable D3 transitions for K80 via kernel param
   or PCI quirk to prevent PLX switch wedge during driver swaps.
