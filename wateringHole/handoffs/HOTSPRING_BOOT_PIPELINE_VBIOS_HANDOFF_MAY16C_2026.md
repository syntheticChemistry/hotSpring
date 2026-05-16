# SPDX-License-Identifier: AGPL-3.0-only

# hotSpring Handoff: Vendor-Agnostic BootPipeline + VBIOS Interpreter Fixes

**Date:** 2026-05-16  
**From:** hotSpring (Exp 198)  
**To:** toadStool, coralReef, primalSpring, sibling springs  
**Scope:** Sovereign boot abstraction, K80 VBIOS interpreter debugging, cross-vendor boot trait

---

## What Was Done

### 1. `BootPipeline` Trait — Vendor-Agnostic Boot Abstraction

**File:** `toadstool-cylinder/src/hardware.rs`

A new trait that captures the universal cold/warm boot sequence for any
PCIe compute device, independent of vendor:

```
probe → is_warm → devinit → engine_init → verify
```

Key design decisions:
- Uses `&dyn RegisterAccess` (not `&MappedBar`) — works with any BAR0 accessor
- Associated types `ProbeResult`/`InitResult` preserve vendor-specific detail
- Summary types (`BootProbeInfo`, `BootInitInfo`) bridge to universal consumers
- `Send + Sync + Debug` bounds enable concurrent fleet boot orchestration

### 2. `DeviceTopology` — Multi-Die/Multi-Function Vocabulary

**File:** `toadstool-cylinder/src/hardware.rs`

Replaces NVIDIA-specific `DeviceInit`/`DieInfo` with structures that work
for K80 (2 GK210 dies on one PCIe card), AMD chiplets, Intel tiles, FPGAs.

- `DeviceTopology::single("Titan V")` for single-function devices
- `DeviceTopology::dual("Tesla K80", bdf0, bdf1)` for multi-die
- `.with_firmware(blob)` for shared firmware images
- `FunctionBootResult` / `DeviceBootResult` aggregate per-function outcomes

### 3. NVIDIA BootPipeline Implementations

**Files:** `init_kepler.rs`, `init_volta.rs`

`KeplerInit` and `VoltaInit` now implement both:
- `InitPipeline` — NV-specific, uses `&MappedBar`, full cold-boot capable
- `BootPipeline` — vendor-agnostic, uses `&dyn RegisterAccess`, warm path functional

The warm path reads BOOT0, PMC_ENABLE, PTIMER via `RegisterAccess`.
The cold path returns `DriverError::Unsupported` via `BootPipeline` because
VBIOS DEVINIT and falcon boot require fork-isolated MMIO through `InitPipeline`.

### 4. AMD VegaInit BootPipeline Stub

**File:** `amd_metal.rs`

Proves the trait works cross-vendor without AMD hardware:
- Probe reads GRBM_STATUS, GRBM_STATUS2, SRBM_STATUS
- Warm detection via GRBM GUI_ACTIVE + SRBM GFX_RQ_PENDING
- Verify checks both status registers for idle
- 8 unit tests with `FakeBar` mock

### 5. K80 VBIOS Interpreter Fixes

**Files:** `opcodes.rs`, `interpreter/mod.rs`

Root cause of K80 Script 1 going off-script ("too many unknown opcodes"):

| Bug | Fix |
|-----|-----|
| Opcode 0x50 (`INIT_IO_RESTRICT_PROG`) stride `4+count*2` | Corrected to `11+count*4` per nouveau `init_io_restrict_prog` |
| Missing opcode 0x88 (`INIT_RAM_RESTRICT_ZM_REG_GROUP`) | Added: `6 + count * n * 4` where n = `ram_restrict_group_count` |
| `ram_restrict_group_count()` reading from raw BIT M data offset | Fixed to dereference M table pointer, read `snr` from rammap header at +4 |
| Missing opcode 0x70 (`INIT_EON`) | Added: 1-byte advance, sets `vm.execute = true` |

---

## Test Results

| Area | Before | After |
|------|--------|-------|
| toadstool-cylinder tests | 591 | **606** (+15) |
| New hardware.rs tests | — | 7 (DeviceTopology, BootProbeInfo, BootInitInfo, results) |
| New amd_metal.rs tests | — | 8 (VegaInit probe, warm, cold, verify, FakeBar) |
| Titan V warm revalidation | ✅ | ✅ (101ms, compute_ready=true) |

---

## Upstream Asks

### For toadStool Team
1. **Absorb `BootPipeline` into public API**: The trait is in `hardware.rs` alongside
   existing `Vendor`, `Capability`, etc. Consider re-exporting from `toadstool-core`.
2. **Wire `sovereign.probe` / `sovereign.verify` RPCs**: Expose BootPipeline's probe
   and verify methods as JSON-RPC for biomeOS fleet orchestration.
3. **K80 VFIO device open**: The interpreter fixes are correct but K80 BAR access
   currently fails because the device is bound to `vfio-pci`. Need
   `VfioDevice::open()` (iommufd path) for proper BAR0 access, not `MappedBar::from_sysfs_rw()`.

### For coralReef Team
1. **GspBridge for Titan V warm**: FECS is preserved through warm-catch. The
   `BootPipeline` probe confirms warm state. What's needed is a real `GspBridge`
   implementation (not `StubGspBridge`) that can leverage the warm FECS state for
   compute dispatch without re-running the full SEC2/ACR chain.
2. **VBIOS interpreter coverage**: The fixes handle K80 Script 1. Remaining scripts
   (Script 2+) may hit additional unhandled opcodes — needs hardware validation.

### For primalSpring + Sibling Springs
1. **Composition pattern**: `BootPipeline` follows the same
   "trait with associated types → vendor impl → universal consumer" pattern as
   NUCLEUS composition. Springs that need device-level boot awareness can depend
   on `toadstool-cylinder` and use `&dyn BootPipeline` without knowing the vendor.
2. **neuralAPI integration**: `sovereign.probe` → JSON-RPC → biomeOS → fleet-level
   warm/cold assessment across heterogeneous hardware. Same IPC patterns as
   `node.compute` and `tower.publish`.

---

## Remaining Sovereign Boot Issues

### Warm Boot
- **Titan V**: warm-catch preserves FECS/GPCCS/PMC state. `BootPipeline::probe`
  correctly detects warm. **Blocker**: GspBridge stub halts falcon boot at stage 4.
  Need real FECS state capture or coralReef IPC bridge.
- **K80**: warm-catch via patched nouveau works. Post-rebind PMC shows 22 engines.
  **Blocker**: PGRAPH clock-gated after driver teardown → needs ungating before dispatch.
- **RTX 5060**: sovereign from boot, no warm-catch needed.

### Cold Boot
- **K80**: VBIOS interpreter now parses Script 1 correctly. **Blocker**: BAR access
  returns 0xFFFFFFFF because K80 is bound to vfio-pci (sysfs resource0 not available).
  Need VFIO device open with iommufd for proper BAR0 mmap.
- **Titan V**: cold boot requires full SEC2/ACR/FECS chain. `InitPipeline` handles
  this but depends on firmware blobs + fork-isolated MMIO. Not attempted via
  `BootPipeline` (returns Unsupported).
- **Any cold GPU**: `BootPipeline::devinit` + `engine_init` return Unsupported for
  cold paths on both NVIDIA implementations. This is by design — cold init requires
  vendor-specific machinery (VBIOS replay, falcon firmware, memory training) that
  `RegisterAccess` alone cannot express.

---

## Files Changed

```
primals/toadStool/crates/core/cylinder/src/hardware.rs          — BootPipeline, DeviceTopology, tests
primals/toadStool/crates/core/cylinder/src/vfio/init_kepler.rs  — BootPipeline impl for KeplerInit
primals/toadStool/crates/core/cylinder/src/vfio/init_volta.rs   — BootPipeline impl for VoltaInit
primals/toadStool/crates/core/cylinder/src/vfio/amd_metal.rs    — VegaInit + FakeBar + 8 tests
primals/toadStool/crates/core/cylinder/src/vfio/channel/devinit/script/interpreter/opcodes.rs — opcode fixes
primals/toadStool/crates/core/cylinder/src/vfio/channel/devinit/script/interpreter/mod.rs     — ram_restrict fix
```
