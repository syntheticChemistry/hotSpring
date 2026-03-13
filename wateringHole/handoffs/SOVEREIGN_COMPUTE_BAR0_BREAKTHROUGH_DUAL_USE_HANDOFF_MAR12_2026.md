# Sovereign Compute — BAR0 Breakthrough, Dual-Use Architecture & Remaining Gaps

**Date**: March 12, 2026
**From**: hotSpring (hardware investigation + implementation in coralReef)
**To**: coralReef, toadStool, barraCuda, all springs
**Type**: Architecture evolution + actionable gap list
**Status**: BAR0 GR init **IMPLEMENTED**, three dispatch paths wired, validation pending

---

## Executive Summary

hotSpring investigated the CTXNOTVALID blocker that prevented GPU compute
dispatch on both Titan V (GV100) and RTX 3090 (GA102). The root cause was
firmware register init data being misrouted to push buffer channels instead
of BAR0 MMIO. **This is now fixed.**

The sovereign compute stack now has **three dispatch paths** — all pure Rust
userspace, differing only in which kernel driver handles memory and submission:

| Path | Kernel Driver | GR Init | Root? | Status |
|------|--------------|---------|-------|--------|
| **Sovereign BAR0** | nouveau (open source) | BAR0 MMIO | Yes* | **IMPLEMENTED** |
| **UVM** | nvidia (proprietary) | RM handles it | No | **IMPLEMENTED** |
| **VFIO** (future) | vfio-pci (generic) | BAR0 MMIO | No** | Design phase |

\* Root needed for BAR0 sysfs — solvable with udev rules, capabilities, or toadStool daemon.
\** VFIO uses IOMMU group permissions, not root.

The **VFIO path** is the endgame: zero GPU-specific drivers, full userspace
hardware ownership, and **dual-use** — the same machine runs Steam games on
nvidia and ecoPrimals science on VFIO, switching on demand.

---

## What's Sovereign Now (Pure Rust, No External Dependencies)

| Layer | Component | Status |
|-------|-----------|--------|
| Shader compilation | WGSL → SASS (SM20-SM89), GFX (RDNA2) | **Done** |
| Math engine | DF64, Yukawa, Bessel, spectral, precision routing | **Done** |
| QMD construction | Volta v2.1, Turing v2.2, Ampere v3.0 | **Done** |
| Push buffer | SET_OBJECT, SEND_PCAS, CBUF descriptors | **Done** |
| Firmware parsing | sw_bundle_init, sw_method_init, sw_ctx | **Done** |
| GR context init | BAR0 MMIO register writes from firmware | **Done** (Mar 12) |
| Address-aware split | Firmware → BAR0 vs FECS routing | **Done** (Mar 12) |
| BAR0 MMIO access | sysfs resource0 mmap, volatile read/write | **Done** (Mar 12) |
| UVM dispatch | RM objects, GPFIFO, USERD doorbell | **Done** |
| DRM dispatch | nouveau GEM, VM_BIND, EXEC, syncobj | **Done** |

## What Still Needs a Kernel Driver

| Function | Why | Path to Sovereign |
|----------|-----|-------------------|
| GPU memory allocation | DMA needs IOMMU/kernel | VFIO DMA buffers |
| Virtual address space | Page tables are kernel-managed | VFIO IOMMU mapping |
| Command submission | GPFIFO ring needs GPU VA | Direct GPFIFO via VFIO BAR |
| Synchronization | Interrupts are kernel-routed | VFIO eventfd interrupts |

---

## The BAR0 Breakthrough — What Changed

### Problem
`CHANNEL_ALLOC` creates a nouveau channel but doesn't initialize the PGRAPH
compute engine registers. First dispatch gets `CTXNOTVALID` from PBDMA.

### Root Cause
`sw_method_init.bin` firmware entries have addresses >= 0x00400000 (BAR0
PGRAPH register space). Push buffer method encoding uses 13 bits (max
offset 0x7FFC). coralReef's `split_for_application()` was routing ALL
`MethodInit` entries to FECS channel submission → they got filtered as
invalid → zero GR init → CTXNOTVALID.

### Fix (committed to coralReef as `23ed6f8`)
1. **`nv/bar0.rs`** (NEW): BAR0 MMIO via sysfs `resource0` mmap
2. **`gsp/applicator.rs`**: Address-aware split (offset > 0x7FFC → BAR0)
3. **`nv/mod.rs`**: Phased device open:
   - Phase 0: `try_bar0_gr_init()` — BAR0 writes BEFORE channel creation
   - Phase 1: VM_INIT
   - Phase 2: CHANNEL_ALLOC (benefits from Phase 0)
   - Phase 3: `try_fecs_channel_init()` — remaining low-address FECS methods

### Firmware Analysis

| GPU | BAR0 register writes | FECS channel methods | Total |
|-----|---------------------|---------------------|-------|
| RTX 3090 (GA102, SM86) | 2972 | 0 | 2972 |
| Titan V (GV100, SM70) | 1570 | 927 | 2497 |

GA102 is 100% BAR0 — ALL init goes through sovereign register writes.
GV100 is hybrid — 63% BAR0 + 37% FECS channel methods.

---

## Remaining Gaps — Ownership & Actions

### Gap A: BAR0 Validation (needs sudo)

**Owner**: hotSpring (testing) + toadStool (permission management)
**Effort**: Small
**Status**: Code done, needs root to validate

```bash
# One-time: enable BAR0 access
sudo chmod 666 /sys/class/drm/renderD*/device/resource0

# Validate sovereign BAR0 init
ARGV0=cargo cargo test --test hw_nv_nouveau --features nouveau -- \
  --ignored --nocapture nouveau_sovereign_bar0_diagnostic

# Test full dispatch
ARGV0=cargo cargo test --test hw_nv_nouveau --features nouveau -- \
  --ignored --nocapture nouveau_dispatch_diagnostic
```

### Gap B: Root Access Solutions (toadStool)

**Owner**: toadStool
**Effort**: Small per solution, pick one or more

| Solution | Complexity | Persistence | Agentic? |
|----------|-----------|-------------|----------|
| **udev rule** | Low | Survives reboot | Yes — auto-applied |
| **setcap** | Low | Per-binary | Partial |
| **toadStool daemon** | Medium | Always-on | **Most agentic** |
| **VFIO group perms** | Medium | Survives reboot | Yes |

**Recommended**: udev rule for immediate use, toadStool daemon for long-term.

```bash
# /etc/udev/rules.d/99-ecoprimals-bar0.rules
SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", RUN+="/bin/chmod 0660 %S%p/resource0"
SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", RUN+="/bin/chgrp gpu-mmio %S%p/resource0"
```

### Gap C: UVM Path Activation

**Owner**: hotSpring (testing) + system config
**Effort**: Small — code is done, needs driver switch
**Status**: `NvUvmComputeDevice` fully implemented, both GPUs on nouveau

The intended config (from modprobe.d) was:
- Titan V → nouveau (sovereign path)
- RTX 3090 → nvidia (UVM compatibility path)

Currently both are on nouveau. To test UVM:
1. Unbind RTX 3090 from nouveau
2. Load nvidia + nvidia-uvm modules
3. Run UVM tests (already written, marked `#[ignore]`)

### Gap D: VFIO GPU Backend (coralReef + toadStool)

**Owner**: coralReef (dispatch) + toadStool (device management)
**Effort**: Medium-Large — but pattern proven on Akida NPU
**Status**: Design phase

toadStool has a **complete VFIO backend** for the Akida NPU:
- `crates/neuromorphic/akida-driver/src/backends/vfio/` — VFIO open, BAR mmap, DMA
- `scripts/setup-akida-vfio.sh` — device unbind/rebind, permission setup

Extending this to NVIDIA GPUs requires:
1. VFIO device open + BAR0/BAR1 mapping (reuse Akida pattern)
2. DMA buffer allocation for GEM-equivalent memory
3. GPFIFO ring setup via mapped USERD page
4. Push buffer submission via GPFIFO (already in `uvm_compute.rs`)
5. Interrupt/completion via VFIO eventfd

The UVM dispatch path (`NvUvmComputeDevice`) already does steps 3-4 via
the proprietary RM — for VFIO we replace RM with direct BAR/DMA access.

### Gap E: Remaining Trio Wiring (from previous handoff)

See `SOVEREIGN_COMPUTE_TRIO_WIRING_GAPS_HANDOFF_MAR12_2026.md` for full details.
Summary of open items:

| Gap | Description | Owner | Status |
|-----|-------------|-------|--------|
| 1 | dispatch_binary → coral cache | barraCuda | **CLOSED** |
| 2 | UVM GPFIFO + USERD | coralReef | Partially closed |
| 3 | FECS GR context | coralReef | **CLOSED** (BAR0 path) |
| 4 | RegisterAccess bridge | toadStool | **CLOSED** |
| 5 | Knowledge → init path | coralReef + toadStool | Open |
| 6 | Error recovery | All three | Open |

---

## Dual-Use Architecture: Gaming + Science on One Machine

### The Vision

The same computer runs Steam games (nvidia proprietary driver) during
personal time and ecoPrimals sovereign compute (VFIO) during science time.
No reboot required — just a mode switch.

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    HARDWARE                              │
│  ┌────────────┐    ┌────────────┐                       │
│  │  RTX 3090  │    │  Titan V   │                       │
│  │  (21:00.0) │    │  (4b:00.0) │                       │
│  └─────┬──────┘    └─────┬──────┘                       │
│        │                 │                               │
└────────┼─────────────────┼───────────────────────────────┘
         │                 │
    ┌────┴────┐       ┌────┴────┐
    │ GAMING  │       │ SCIENCE │
    │  MODE   │       │  MODE   │
    └────┬────┘       └────┬────┘
         │                 │
    nvidia driver     vfio-pci
    (proprietary)     (generic)
         │                 │
    Steam/Proton     ecoPrimals
    Vulkan games     sovereign
                     compute
```

### Multi-GPU Split (current hardware)

| GPU | Gaming Mode | Science Mode |
|-----|------------|-------------|
| RTX 3090 | nvidia driver (Steam) | vfio-pci (ecoPrimals) |
| Titan V | nouveau (display fallback) | vfio-pci (ecoPrimals) |

### Single-GPU Dynamic Switch

For machines with one GPU:

```bash
# Switch to science mode
ecoprimals-mode science    # stops display server, unbinds nvidia, binds vfio
                           # runs ecoPrimals workloads
                           # toadStool manages the transition

# Switch to gaming mode
ecoprimals-mode gaming     # unbinds vfio, loads nvidia, restarts display
```

toadStool is the natural owner of this mode switch — it already handles
device discovery, driver detection, and hardware management.

### Implementation Plan

**Phase 1** (now): Validate sovereign BAR0 + nouveau on Titan V
**Phase 2** (next): Validate UVM on RTX 3090 with nvidia driver
**Phase 3**: Build VFIO GPU backend (coralReef + toadStool)
**Phase 4**: Build `ecoprimals-mode` CLI tool (toadStool)
**Phase 5**: Dynamic GPU arbitration — toadStool detects idle GPU, claims for science

### toadStool Setup Script Template

Modeled on existing `scripts/setup-akida-vfio.sh`:

```bash
#!/bin/bash
# setup-gpu-sovereign.sh — bind GPU to vfio-pci for sovereign compute
#
# Usage: sudo ./setup-gpu-sovereign.sh [PCI_ADDR]
# Example: sudo ./setup-gpu-sovereign.sh 0000:4b:00.0

ADDR=${1:-"0000:4b:00.0"}

# 1. Unbind from current driver (nouveau/nvidia)
echo "$ADDR" > /sys/bus/pci/devices/$ADDR/driver/unbind 2>/dev/null

# 2. Load VFIO modules
modprobe vfio-pci

# 3. Get vendor:device for VFIO binding
VENDOR=$(cat /sys/bus/pci/devices/$ADDR/vendor)
DEVICE=$(cat /sys/bus/pci/devices/$ADDR/device)
echo "${VENDOR#0x} ${DEVICE#0x}" > /sys/bus/pci/drivers/vfio-pci/new_id

# 4. Bind to vfio-pci
echo "$ADDR" > /sys/bus/pci/drivers/vfio-pci/bind

# 5. Set group permissions
IOMMU_GROUP=$(basename $(readlink /sys/bus/pci/devices/$ADDR/iommu_group))
chmod 666 /dev/vfio/$IOMMU_GROUP

echo "GPU $ADDR bound to vfio-pci (IOMMU group $IOMMU_GROUP)"
echo "ecoPrimals can now use this GPU without root"
```

---

## Cross-Platform Sovereignty Path

The VFIO approach isn't NVIDIA-specific. The same pattern works for:

| Vendor | BAR0 Init Data | Driver Replaced | VFIO Support |
|--------|---------------|-----------------|-------------|
| NVIDIA | sw_method_init.bin + sw_bundle_init.bin | nouveau / nvidia | Yes (IOMMU) |
| AMD | amdgpu open-source init (observable) | amdgpu | Yes (IOMMU) |
| Intel | i915 open-source init (observable) | i915 | Yes (IOMMU) |

coralReef already has `AmdDevice` for amdgpu. The sovereign GSP's learning
architecture (observe → distill → apply) works across all vendors.

---

## File Reference

| File | Repo | Description |
|------|------|-------------|
| `nv/bar0.rs` | coralReef | BAR0 MMIO access via sysfs |
| `gsp/applicator.rs` | coralReef | Address-aware firmware split |
| `nv/mod.rs` | coralReef | Phased device open (BAR0 → channel → FECS) |
| `nv/uvm_compute.rs` | coralReef | Full UVM dispatch pipeline |
| `backends/vfio/` | toadStool | VFIO backend (Akida — model for GPU) |
| `setup-akida-vfio.sh` | toadStool | VFIO device setup script |
| `nvpmu/src/bar0.rs` | toadStool | BAR0 access (original impl) |

## Commits

| Hash | Repo | Description |
|------|------|-------------|
| `23ed6f8` | coralReef | Sovereign BAR0 GR init implementation |
| `e160d89` | coralReef | Filter BAR0 addresses from FECS channel |
| `996b7c1` | coralReef | Wire FECS GR context init |
| `a691023` | coralReef | QMD, CBUF, syncobj, dispatch fixes |

---

*hotSpring sovereign compute investigation — March 12, 2026*
