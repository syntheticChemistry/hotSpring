# hotSpring Multi-GPU Benchmark & Cooperative Dispatch
> **SUPERSEDED** by `HOTSPRING_BARRACUDA_FULL_GPU_HANDOFF_FEB17_2026.md`

## Date: 2026-02-17
## From: hotSpring → toadstool (barracuda)
## Hardware: RTX 4070 (Ada, proprietary) + Titan V (Volta GV100, NVK open-source)

---

## 1. Validation Matrix — BarraCUDA Ops on Both Cards

| Binary | Workload | RTX 4070 | Titan V | Notes |
|--------|----------|----------|---------|-------|
| validate_barracuda_hfb | BCS + eigensolve | **16/16 PASS** | **16/16 PASS** | Identical results on both cards |
| validate_barracuda_pipeline | Yukawa MD (f64) | **12/12 PASS** | **CRASH** | NAK compiler assertion (exp f64) |
| validate_md | LJ/Coulomb/Morse/VV (f32) | **20/20 PASS** | **20/20 PASS** | Uses WgpuDevice::new() directly |
| validate_pppm | PPPM Coulomb | **4/8 PASS** | **CRASH** | 4070: PPPM energy sign bug; Titan: NAK crash |
| f64_builtin_test | f64 builtins | **9/12 PASS** | **CRASH** | 4070: exp precision; Titan: NAK crash on exp |
| nuclear_eos_gpu | L1 SEMF GPU | PASS (long) | **partial** | Titan: L1 basic works, crashes on math_f64 exp |

### NVK NAK Compiler Bug (Critical for toadstool)

All Titan V crashes share the same root cause:
```
assertion failed: vec.len() == bits.div_ceil(32)
at ../src/nouveau/compiler/nak/from_nir.rs:430
```

**Trigger**: Any WGSL shader using native `exp()` on f64 values.
**Unaffected**: sqrt, abs, floor, ceil, round, comparisons, arithmetic on f64.
**Workaround**: Use barracuda's software `exp_f64()` from `math_f64.wgsl` instead of native `exp()`.

**Recommended fix for barracuda**: In `YukawaForceF64::new()`, `PppmGpu`, and any f64 shader using `exp()`:
1. Detect NVK backend: `adapter_info.driver.contains("NVK")` or `adapter_info.driver.contains("Mesa")`
2. Replace `exp(` with `exp_f64(` in shader source before compilation
3. Ensure `math_f64.wgsl` preamble is prepended (already done for Yukawa)

The `exp_f64()` software implementation in `math_f64.wgsl` works correctly on NVK — only the native WGSL `exp()` builtin triggers the NAK bug.

---

## 2. Side-by-Side FP64 Performance

### BCS Bisection (pure fp64 streaming compute)

| Batch | RTX 4070 | Titan V | Titan V speedup |
|-------|----------|---------|-----------------|
| 128 | 5.17 ms | 1.46 ms | **3.5×** |
| 2048 | 12.35 ms | 1.90 ms | **6.5×** |
| 8192 | 10.93 ms | 2.78 ms | **3.9×** |

**Peak: Titan V = 2.95M bisections/s (3.9× vs 4070's 749K/s)**

### Batched Eigensolve (Jacobi rotations, dispatch-heavy)

| Batch×Dim | RTX 4070 | Titan V | RTX 4070 speedup |
|-----------|----------|---------|-------------------|
| 128×20 | 10.30 ms | 28.26 ms | **2.7×** |
| 512×20 | 21.51 ms | 94.01 ms | **4.4×** |
| 512×30 | 48.02 ms | 306.77 ms | **6.4×** |

**RTX 4070 wins 2-6× on eigensolve due to lower dispatch latency.**

### Interpretation

- **Streaming compute** (BCS): Hardware fp64 throughput dominates. Titan V's 1:2 fp64 rate wins decisively.
- **Iterative/dispatch-heavy** (eigensolve): Per-dispatch overhead dominates. NVIDIA proprietary driver has far lower dispatch latency than NVK.
- **Driver maturity** matters as much as hardware for latency-sensitive workloads.

---

## 3. Multi-GPU Cooperative Dispatch Results

### Naive data splitting (same task, split batch)

| Mode | BCS 8192 | Eigensolve 256×20 |
|------|----------|-------------------|
| Single card (4070) | 8.29 ms | 13.33 ms |
| 50/50 split | 12.20 ms | 28.10 ms |
| **Speedup** | **0.68×** (slower) | **0.47×** (slower) |

Naive splitting is SLOWER because:
- Thread synchronization overhead exceeds compute savings at small batch sizes
- The slower card (Titan V for eigensolve) becomes the bottleneck
- wgpu instance overhead per device is non-trivial

### Specialized routing (different tasks, each on best card)

| Mode | BCS 4096 + Eigen 128×20 |
|------|-------------------------|
| Sequential (4070 only) | 16.35 ms |
| Specialized (BCS→Titan, Eigen→4070) | 11.02 ms |
| **Speedup** | **1.48×** |

**Task-level routing works.** Each card runs what it's best at, simultaneously.

---

## 4. Recommendations for toadstool GpuPool / MultiDevicePool

### Scheduling strategy
1. **Task-level routing** over data-parallel splitting
2. **Workload classification**: tag each kernel as "streaming" or "iterative"
3. **Route streaming kernels** (BCS, forces, density, reduction) → highest fp64 card
4. **Route iterative kernels** (eigensolve, SCF convergence) → lowest-latency driver
5. **Batch size threshold**: only split data across GPUs for batches > 10K elements

### NVK compatibility
6. **Detect NAK/NVK** at device init time
7. **Auto-replace native exp()** with software exp_f64() for NVK targets
8. **Test matrix**: maintain a per-driver shader compatibility table

### Device API additions
9. `WgpuDevice::is_nvk()` → bool (check driver string)
10. `ShaderTemplate::for_device(shader, device)` → auto-patches for driver quirks
11. `GpuPool::route(WorkloadType, batch_size)` → selects optimal device

---

## 5. hotSpring Local Fixes (for toadstool to absorb)

### Fix 1: Device limits for HFB potentials shader
**File**: `hotspring/barracuda/src/gpu.rs` (GpuF64::new)
**Issue**: Default `max_storage_buffers_per_shader_stage = 8`, HFB needs 12
**Fix**: Request `max_storage_buffers_per_shader_stage = 12` in device limits
**Absorb**: `science_limits()` should include this, or make it configurable

### Fix 2: Adapter selector numeric fallthrough
**File**: `hotspring/barracuda/src/gpu.rs` (GpuF64::new)
**Issue**: "4070" parsed as index 4070 (out of bounds), not as name substring
**Fix**: If numeric index exceeds adapter count, fall through to name matching
**Absorb**: `WgpuDevice::with_adapter_selector()` should already do this

### Fix 3: Naga bitcast<f64>(vec2<u32>) not supported
**File**: `shaders/batched_hfb_density_f64.wgsl`
**Issue**: `bitcast<f64>(vec2<u32>)` rejected by Naga shader validator
**Fix**: Replaced with integer-ratio encoding (alpha_num/alpha_den)
**Absorb**: Any shader using bitcast<f64> needs this pattern

### Fix 4: f64 literal type inference
**File**: `shaders/spin_orbit_pack_f64.wgsl`
**Issue**: `1e10` inferred as f32, assigned to f64 buffer → type mismatch
**Fix**: `f64(1e10)` explicit cast
**Absorb**: All f64 shaders should use explicit f64() casts for literals

### Fix 5: Buffer usage conflicts in compute passes
**File**: `physics/hfb_gpu_resident.rs`
**Issue**: Same buffer bound as READ_WRITE (BCS) and READ (density) in same dispatch
**Fix**: Split into 3 separate compute passes (BCS, density, mixing) with dedicated bind groups
**Absorb**: wgpu requires exclusive access for READ_WRITE within a dispatch

---

## 6. New Files

- `src/bin/bench_multi_gpu.rs` — Multi-GPU cooperative benchmark (comparative + cooperative + specialized routing)
- This handoff document

## Validation State

- `cargo fmt`: clean
- `cargo clippy --lib -- -W clippy::pedantic`: 0 warnings
- `cargo test`: 199 passed, 5 ignored, 0 failed
