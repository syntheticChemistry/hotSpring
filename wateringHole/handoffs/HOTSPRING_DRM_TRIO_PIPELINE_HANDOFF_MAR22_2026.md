# hotSpring → Compute Trio: DRM Pipeline + Sovereign VFIO Progress

**Date:** March 22, 2026
**From:** hotSpring
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** Three-GPU DRM validation — MI50 E2E complete, RTX 5060 Blackwell DRM cracked, Titan V VFIO staged, iommufd/cdev kernel-agnostic VFIO

---

## Executive Summary

- **MI50 (AMD GCN5)**: Full E2E pipeline PASSES — WGSL → coral-reef → GCN5 ISA → PM4 → DRM → MI50 → readback. 6/6 phases, f64 LJ force verified.
- **RTX 5060 (NVIDIA Blackwell SM120)**: NvUvmComputeDevice fully operational — open/alloc/free/bind all pass. Two Blackwell-specific bugs found and fixed:
  1. **Single mmap context**: Blackwell 580.x nvidiactl fd supports only ONE `rm_map_memory` context. Fixed with combined USERD+GPFIFO allocation.
  2. **Per-buffer fd**: User buffer allocations need a dedicated nvidiactl fd per `rm_map_memory` call.
- **Titan V (NVIDIA Volta SM70)**: Sovereign VFIO path staged. PTE encoding verified correct (nouveau GP100-style matches Volta NV_MMU_VER2). VFIO group EBUSY after hard reboot — needs PCI reset with root.
- **4/4 NVIDIA DRM hardware tests pass** on RTX 5060.
- **Closed-source Volta UVM driver captured** (65 files, 1.7 MB) for reverse engineering.

---

## Part 1: MI50 AMD DRM — Complete

### NOP Dispatch (Exp 072)
```
Phase 1: AmdDevice::open().............. OK ✓
Phase 2: Alloc shader buffer............ OK ✓
Phase 3: Upload s_endpgm................ OK ✓
Phase 4: Dispatch (1×1×1 workgroups).... OK ✓
Phase 5: Sync (fence wait).............. OK ✓
→ AMD DRM NOP DISPATCH: ALL PHASES PASSED
```

### GCN5 E2E (6/6 PASS)
```
[PASS] A: f64 Write (64/64 = 42.0)
[PASS] B: f64 Arithmetic (64/64 = 42.0, 6.0*7.0)
[PASS] C: Multi-Workgroup (1024/1024)
[PASS] D: GLOBAL_LOAD diagnostic (STORE-then-LOAD ✓)
[PASS] F: HBM2 Bandwidth (1M elements, 4.47 GB/s)
[PASS] E: f64 Lennard-Jones (Newton's 3rd law verified)
```

---

## Part 2: RTX 5060 NVIDIA Blackwell DRM — Pipeline Cracked

### Blackwell-Specific Bugs Fixed

**Bug 1: `NV_ERR_NOT_SUPPORTED` (0x56) on compute engine alloc**
- RTX 5060 is SM 12.0 (Blackwell B / GB206), not Ampere
- Added `BLACKWELL_COMPUTE_B = 0xCEC0` and `BLACKWELL_CHANNEL_GPFIFO_B = 0xCA6F`
- Extended `GpuGen` enum with Ada, Hopper, BlackwellA, BlackwellB

**Bug 2: `NV_ERR_STATE_IN_USE` (0x63) on `RM_MAP_MEMORY`**
- Root cause: Blackwell 580.x nvidiactl fd supports only ONE active mmap context
- Diagnostic proved: 1st `rm_map_memory` succeeds, 2nd returns 0x63 on same fd
- **Fix for control plane**: Combined USERD (4KB) + GPFIFO (4KB) into single 8KB allocation, one `rm_map_memory` call, subdivided via pointer offsets
- **Fix for data plane**: Each user buffer `alloc()` opens a fresh `/dev/nvidiactl` fd for its mmap context, stored in `UvmBuffer.mmap_fd`

### Test Results (4/4 PASS)
```
uvm_compute_bind ................ ok  (channel=0xCA6F, compute=0xCEC0)
uvm_compute_device_open ......... ok  (SM120 fully initialized)
uvm_compute_alloc_free .......... ok  (per-buffer fd mmap)
uvm_map_memory_single_context ... ok  (combined allocation validation)
```

### Remaining for Full Dispatch
- coral-reef `NvArch` enum needs `Sm120` variant for Blackwell ISA compilation
- QMD v3.0 builder already works (SM 80+ catch-all)
- All RM/UVM plumbing operational — dispatch is ISA-gated only

---

## Part 3: Titan V Sovereign VFIO — Staged

### PTE Encoding Analysis
The current page table encoding in `page_tables.rs` uses nouveau GP100-style:
```rust
encode_pte(phys) = (phys >> 4) | FLAGS  // FLAGS = VALID|COH|VOL
```

This is **mathematically equivalent** to the Volta `make_pte` from `uvm_volta_mmu.c`:
```
address >> 12 → bits 53:8, KIND(0x00) → bits 63:56
```
For 4K-aligned system memory addresses, both produce identical 64-bit PTEs.

### VA Coverage
- Static identity map: first 2 MiB (GPU VA 0x1000 → 0x1FF000)
- User buffers start at `0x10_0000` (1 MiB) → 1 MiB usable for dispatch
- Sufficient for simple compute (shader + QMD + pushbuf + buffer < 1 MiB)

### Current Blocker
- VFIO group 73 is EBUSY after hard reboot — needs PCI unbind/rebind with root
- Command to run: `sudo sh -c 'echo 0000:03:00.1 > /sys/bus/pci/drivers/vfio-pci/unbind && echo 0000:03:00.0 > /sys/bus/pci/drivers/vfio-pci/unbind && sleep 1 && echo 0000:03:00.0 > /sys/bus/pci/drivers/vfio-pci/bind && echo 0000:03:00.1 > /sys/bus/pci/drivers/vfio-pci/bind'`
- After reset, tests run with: `CORALREEF_VFIO_BDF=0000:03:00.0 CORALREEF_VFIO_SM=70 cargo test -p coral-driver --features vfio --test hw_nv_vfio -- --ignored`

### GR Engine Status
- FECS falcon halted on cold VFIO boot (no signed firmware loaded)
- GlowPlug oracle warm-up or FECS firmware load needed before compute dispatch
- Closed-source Volta driver reference available at `wateringHole/nvidia-closed-volta-580/`

---

## Part 4: Files Modified

### coral-driver (coralReef)
| File | Change |
|------|--------|
| `src/nv/uvm_compute.rs` | Single-alloc mmap, per-buffer fd, Blackwell GpuGen |
| `src/nv/uvm/mod.rs` | Blackwell class IDs (COMPUTE_A/B, CHANNEL_A/B) |
| `src/nv/uvm/rm_client/alloc.rs` | Blackwell compute class names in error messages |
| `src/nv/uvm/rm_client_tests.rs` | Auto-detect SM for Blackwell-aware compute bind test |

### hotSpring (wateringHole)
| File | Purpose |
|------|---------|
| `nvidia-closed-volta-580/` | 65 files, 1.7 MB closed-source UVM reference |
| `nvidia-closed-volta-580/REVERSE_ENGINEERING_GUIDE.md` | Volta PTE, doorbell, class reference |

---

## Part 5: Next Steps

### Immediate (require root)
1. **Titan V VFIO reset** — PCI unbind/rebind to clear EBUSY
2. **Titan V VFIO open test** — validate BAR0 read, DMA alloc, upload/readback

### Short-term
3. **coral-reef SM120** — Add `NvArch::Sm120` to enable Blackwell ISA compilation
4. **RTX 5060 full dispatch** — WGSL → SM120 → QMD → GPFIFO → execute → readback
5. **Titan V GR warm-up** — GlowPlug oracle or FECS firmware for compute dispatch

### Medium-term
6. **Three-GPU simultaneous dispatch** — MI50 + RTX 5060 + Titan V running concurrently
7. **DF64 physics on all GPUs** — Lennard-Jones, Yukawa, Wilson plaquette

---

## Part 4 (Addendum): iommufd/cdev Backend — Kernel-Agnostic VFIO

**Added:** March 22, 2026

The Titan V VFIO EBUSY blocker (Part 3) was resolved by implementing the modern
`iommufd`/`cdev` VFIO path alongside the legacy container/group API. On kernel 6.17,
the legacy group API returns EBUSY when the companion HDA audio device shares the
IOMMU group. The iommufd/cdev path operates per-device, bypassing group viability.

### Changes (38 files, +1643/-406 lines)

**coral-driver:**
- iommufd ABI types + 5 ioctl wrappers (bind, attach, ioas alloc/map/unmap)
- `VfioBackendKind` enum, `ReceivedVfioFds` enum, `sendable_fds()`, `from_received()`
- `DmaBackend` dispatch for both legacy and iommufd DMA mapping
- `sysfs_vfio_cdev_name()` for `/dev/vfio/devices/` node discovery

**coral-ember:**
- Backend-aware logging (no panicking container_fd/group_fd on iommufd)
- `SCM_RIGHTS` sends 2 fds (iommufd) or 3 fds (legacy) + JSON metadata

**coral-glowplug:**
- `ReceivedVfioFds` from coral-driver replaces local `EmberFds`
- All 6 downstream callers updated to `VfioDevice::from_received()`

### Validation
- 607 tests pass across all three crates
- Hardware validated on Titan V: ember → SCM_RIGHTS → client → BAR0 + DMA

See `HOTSPRING_IOMMUFD_EMBER_EVOLUTION_HANDOFF_MAR22_2026.md` for full details.
