# hotSpring ↔ toadStool — Brain Architecture Sync

**Date:** February 28, 2026
**From:** hotSpring (brain architecture, dual-GPU production)
**To:** toadStool / barracuda core team
**toadStool HEAD:** `ad9e9de` (S68+++ fix: box_muller_cos)
**hotSpring HEAD:** post-brain-architecture
**License:** AGPL-3.0-only

---

## Summary

hotSpring built a concurrent "brain architecture" — 4 hardware threads
(RTX 3090, Titan V, CPU, NPU) running simultaneously during dynamical
HMC production. Initial launch hung due to concurrent `wgpu::Instance`
creation causing NVK/nouveau driver deadlock. Fixed by serializing
device creation on the main thread before spawning workers.

toadStool already has `MultiDevicePool` / `GpuPool` / `DeviceLease`
infrastructure in `barracuda::multi_gpu`. hotSpring should migrate to
this instead of its custom `GpuF64::from_adapter_name()` approach.

---

## Bug Found: NVK Dual-GPU Deadlock

**Symptom:** Process hung at `drm_syncobj_array_wait_timeout` with GPU
at idle temperature (42°C). All 138 worker threads in `futex_do_wait`.
No output for 35+ minutes.

**Root cause:** Two threads concurrently called `wgpu::Instance::new()`
and `enumerate_adapters()`, hitting NVK/nouveau kernel driver contention.
Both GPUs (RTX 3090 card0, Titan V card1) use the same nouveau DRM
subsystem. Concurrent Vulkan initialization caused a DRM-level deadlock.

**Fix:** Create both `GpuF64` devices sequentially on the main thread
before spawning any worker threads. The Titan V `GpuF64` is passed
into the worker via ownership transfer (`spawn_titan_worker(titan_gpu)`).

**Upstream implication:** toadStool's `MultiDevicePool::new()` creates
all devices in one pass and should not be affected. But any future
code that creates `WgpuDevice` instances from multiple threads under NVK
should be aware of this driver-level serialization requirement.

---

## What hotSpring Built (Brain Architecture)

| Layer | Thread | Hardware | Work |
|-------|--------|----------|------|
| 1 | Main | RTX 3090 | Dynamical HMC + CG residual streaming |
| 2 | titan-premotor | Titan V | Quenched pre-therm for next β |
| 3 | cpu-cortex | CPU | Anderson 3D + Potts Z(3) proxy |
| 4 | npu-cerebellum | AKD1000 | 15-head ESN, attention state machine |

Communication: `mpsc` channels with typed request/response enums.
NPU monitors CG residuals via Head 15 and can issue `KillCg` or
`AdjustCheckInterval` interrupts.

---

## Rewiring Status (from S68 handoff)

### Done

- [x] toadStool S61–S68+++ pulled and building
- [x] `gaussian_fermion_f64.wgsl` cos bug fixed
- [x] `spectral::*` re-exported from upstream
- [x] `Fp64Strategy` and `su3_df64_preamble` consumed from upstream

### Next (deferred — brain architecture took priority)

| Item | toadStool API | Effort | Priority |
|------|---------------|--------|----------|
| Local lattice shaders → upstream | `ops::lattice::absorbed_shaders` | Medium | High |
| Manual DF64 preamble → `compile_shader_df64()` | `shaders::precision` | Low | High |
| `GpuF64` → `MultiDevicePool` | `multi_gpu::MultiDevicePool` | High | Medium |
| CG solver → upstream | `ops::lattice::cg` | Medium | Medium |
| `NeighborMode::PrecomputedBuffer` | `ops::lattice` | Low | Medium |
| ESN → upstream v2 (keep multi-head local) | `esn_v2::ESN` | Medium | Low |

### New: toadStool Capabilities hotSpring Should Track

| Capability | Session | Impact |
|-----------|---------|--------|
| `MultiDevicePool` + `DeviceLease` | S52+ | Replace custom dual-GPU code |
| `UnidirectionalPipeline` + `GpuRingBuffer` | S52+ | GPU-resident streaming for CG |
| Universal precision (`compile_op_shader`) | S68 | Single-source shaders |
| Sovereign compiler (naga IR, FMA fusion) | S68 | Automatic optimization |
| `BatchIprGpu` (batch IPR on GPU) | S54+ | GPU-accelerated Anderson proxy |
| `PcieBridge` + `BandwidthTier` | S52+ | Informed dispatch decisions |
| Device registry (dedup by physical device) | S52+ | Clean multi-adapter enumeration |

---

## Validated

- [x] Dual-GPU pilot: RTX 3090 + Titan V concurrent (serialized creation)
- [x] All 4 brain layers: NPU cerebellum, 3090 motor, Titan V pre-motor, CPU cortex
- [x] Exit code 0, 9.2 min for 1-β pilot (3 therm + 3 meas)
- [x] `--no-titan` fallback flag available

---

## For toadStool Team

1. **NVK multi-GPU:** If `MultiDevicePool` is ever called under NVK,
   ensure device creation is serialized. Add a note to docs.
2. **Brain pattern:** The 4-thread brain with typed channels is a useful
   pattern for multi-fidelity physics. Consider generalizing in toadStool
   as a `ConcurrentPipeline` abstraction.
3. **CG residual streaming:** hotSpring's `CgResidualUpdate` +
   `BrainInterrupt` pattern (streaming residuals to NPU during GPU CG
   solve) could be generalized for any iterative GPU solver.
