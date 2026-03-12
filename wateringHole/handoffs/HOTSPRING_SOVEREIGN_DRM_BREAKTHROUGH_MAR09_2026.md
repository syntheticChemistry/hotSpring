# hotSpring — Sovereign DRM Dispatch Breakthrough

**Date**: March 9, 2026  
**Version**: hotSpring v0.6.30 | coralReef Iter 35 (patched) | kernel 6.17.9

---

## What Happened

During testing to validate the sovereign compute stack as a replacement for
naga/NVK/wgpu, we discovered and fixed 3 critical bugs in `coral-driver` that
had been blocking DRM channel allocation on all NVIDIA GPUs since coralReef's
inception.

The root cause was an **ioctl number off-by-two**: `DRM_NOUVEAU_CHANNEL_ALLOC`
was using offset 0x00 (GETPARAM) instead of 0x02 (the actual CHANNEL_ALLOC).
Two secondary bugs (VA space collision with kernel-managed region, and broken
gem_info query) were also fixed.

## Results

| Component | Before | After |
|-----------|--------|-------|
| VM_INIT | PASS | PASS |
| CHANNEL_ALLOC | **EINVAL on all GPUs** | **PASS on all GPUs** |
| GEM alloc + VM_BIND | FAIL (cascade) | PASS |
| Upload + readback | FAIL (cascade) | PASS |
| Dispatch (EXEC) | Never reached | Runs without error |
| Compute output | N/A | 0 (QMD tuning needed) |

## Sovereign Stack Status

```
WGSL source
  ↓ coralReef compiler (1616 tests, 46/46 shaders) ✅
Native SASS binary
  ↓ coral-driver NvDevice (GEM + VM_BIND + QMD + pushbuf) ✅
DRM dispatch (EXEC)
  ↓ nouveau kernel module (VM_INIT + CHANNEL_ALLOC + EXEC) ✅
GPU hardware
  ↓ Compute execution (QMD field alignment pending) ⚠️
Result readback ✅
```

## Deprecation Implications for hotSpring

Once compute execution is confirmed:

- **Remove wgpu dependency** — sovereign `NvDevice` replaces `wgpu::Device`
- **Remove naga codegen path** — coralReef compiler produces correct SASS directly
- **DF64 poisoning becomes irrelevant** — that was a naga SPIR-V bug
- **NAK crashes become irrelevant** — coralReef doesn't use NAK
- **ReduceScalarPipeline regression resolves** — native f64 reduction works in SASS

## Files Modified in coralReef

- `crates/coral-driver/src/nv/ioctl/mod.rs` — ioctl numbers, domain flags, gem_new API
- `crates/coral-driver/src/nv/mod.rs` — VA space, alloc path
- `crates/coral-driver/tests/hw_nv_nouveau.rs` — updated for new API
