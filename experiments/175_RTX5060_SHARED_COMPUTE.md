# Experiment 175: RTX 5060 Shared Compute Prototype

**Date**: 2026-04-16
**Track**: 3 — Sovereign Compute Evolution Strategy
**GPU**: RTX 5060 (GB206, Blackwell, SM 120)
**BDF**: 0000:21:00.0
**Goal**: Prototype SHARED compute on the display GPU alongside gaming

## Changes Made

### 1. glowplug.toml: Promoted to `role = "shared"`
- Changed RTX 5060 from `role = "display"` to `role = "shared"`
- Fixed model name from "rtx-5070" to "rtx-5060"
- ember treats shared as protected (no VFIO hold, nvidia driver stays bound)
- Compute dispatches via nvidia-drm render node + UVM

### 2. coral-driver identity fix: SM 120 detection for PCI ID 0x2d05
- RTX 5060 PCI device ID is `0x2d05`, not in the original `0x2900..=0x2999` range
- Extended Blackwell range to `0x2900..=0x2999 | 0x2B00..=0x2DFF`
- Verified: `nvidia_sm()` now returns `Some(120)` for 0x2d05

### 3. coral-driver nvidia_drm: Multi-GPU index derivation
- `NvDrmDevice::open_path()` previously hardcoded `gpu_index = 0`
- Added `nvidia_gpu_index_from_render_node()` that resolves:
  render node → sysfs BDF → `/proc/driver/nvidia/gpus/<BDF>/information` → Device Minor
- Correctly maps render nodes to NVIDIA device ordinals in multi-GPU systems

## Test Results

### WGSL Compilation (PASS)
```
Discovered GPU: nvidia sm120
compiled 160 bytes for sm120
```
- `GpuContext::auto()` selects nvidia-drm backend
- WGSL → naga → coral-reef → SM 120 SASS pipeline fully functional
- Integration tests: 4/4 pass (discover, enumerate, compile, identity)

### UVM Dispatch (PARTIAL — GPFIFO timeout)
```
SM 120: channel=0xCA6F compute=0xCEC0
GPFIFO completion timeout: GP_GET=0 GP_PUT=1
```
- RM object chain succeeds: ROOT → DEVICE → SUBDEVICE → UUID → UVM_REGISTER → VA_SPACE → CHANNEL_GROUP → GPFIFO → COMPUTE
- `BLACKWELL_CHANNEL_GPFIFO_B` (0xCA6F) and `BLACKWELL_COMPUTE_B` (0xCEC0) accepted
- NOP smoke test fails: GPU never picks up GPFIFO entry
- Root cause: likely Blackwell-specific USERD layout or doorbell mechanism

### UVM lower-level tests (47/50 pass)
- All allocation, memory mapping, channel creation, and compute binding tests pass
- Only GPFIFO submission-dependent tests fail

## Architecture Validated

```
WGSL source
    ↓ naga parse
SPIR-V
    ↓ coral-reef compile (target=SM120)
SASS binary (160 bytes)
    ↓ coral-gpu context (nvidia-drm backend)
NvDrmDevice → NvUvmComputeDevice
    ↓ RM + UVM ioctls
GPFIFO submission ← BLOCKED (Blackwell USERD/doorbell)
```

## Next Steps

1. **GPFIFO doorbell investigation**: Compare Blackwell USERD layout with Ada/Ampere.
   The USERMODE doorbell offset (0x90) and GP_PUT/GP_GET offsets may differ.
2. **CUDA reference**: Use cudarc/CUDA to capture a working Blackwell GPFIFO
   submission trace for comparison.
3. **SharedQuota enforcement**: Wire `nvidia-smi` power/compute limits once dispatch works.
4. **DRM isolation verification**: Confirm compute workloads don't disrupt compositor.

## Key Finding

The RTX 5060 (GB206) uses PCI device ID `0x2d05`, which falls outside the initial
Blackwell range in coral-driver's identity table. This is consistent with NVIDIA's
practice of using multiple PCI ID ranges for the same architecture generation.
