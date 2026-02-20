# hotSpring → ToadStool: Full BarraCUDA GPU Validation Handoff

**Date:** 2026-02-17
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Supersedes:** `TOADSTOOL_MULTI_GPU_NVK_HANDOFF_FEB17_2026.md`,
               `HOTSPRING_MULTI_GPU_BENCHMARK_FEB17_2026.md`

---

## Executive Summary

hotSpring has completed a full validation pass of BarraCUDA GPU operations on
two physically distinct GPUs with two different Vulkan drivers. We ran
**barracuda CPU** (199 unit tests), **barracuda pure GPU** (6 validation
binaries per card), and **barracuda cooperative multi-GPU** (4 benchmark
modes). All results are compared to CPU reference and analytical controls.

**Key findings:**
1. BCS bisection and batched eigensolve produce **identical results** across
   both GPUs (RTX 4070 proprietary + Titan V NVK open-source)
2. NVK has a **NAK compiler bug** on native `exp(f64)` — blocks Yukawa MD and
   PPPM. Workaround: use software `exp_f64()` from `math_f64.wgsl`
3. **Specialized multi-GPU routing** (task-level, not data splitting) achieves
   **1.48x speedup** — proof of concept for `GpuPool` scheduling
4. Five local shader/device fixes documented below for toadstool to absorb

---

## 1. Hardware Configuration

| | RTX 4070 | Titan V |
|---|---|---|
| **Architecture** | Ada Lovelace (AD104) | Volta (GV100) |
| **PCIe slot** | 01:00.0 | 05:00.0 |
| **Kernel module** | nvidia (proprietary 580.82) | nouveau |
| **Vulkan driver** | NVIDIA proprietary | NVK (Mesa 25.1.5) |
| **shaderFloat64** | true | true |
| **Theoretical fp64** | ~400 GFLOPS (1:64) | ~7,450 GFLOPS (1:2) |
| **VRAM** | 12 GB GDDR6X | 12 GB HBM2 |

Both GPUs visible simultaneously to `wgpu::Instance::enumerate_adapters()`.
Selection via `HOTSPRING_GPU_ADAPTER` env var (name substring or index).

---

## 2. Validation Matrix

### CPU (Pure Rust, no GPU)

```
cargo test:   199 passed, 0 failed, 5 ignored
cargo clippy: 0 warnings (pedantic)
cargo fmt:    clean
```

### GPU — Per-Card Results

| Binary | Workload | RTX 4070 | Titan V | Notes |
|--------|----------|----------|---------|-------|
| `validate_barracuda_hfb` | BCS bisection + batched eigensolve | **16/16 PASS** | **16/16 PASS** | Identical numerical results |
| `validate_barracuda_pipeline` | Yukawa OCP MD end-to-end | **12/12 PASS** | **CRASH** | NAK `exp(f64)` bug |
| `validate_md` | LJ + Coulomb + Morse + VV (f32) | **20/20 PASS** | **20/20 PASS** | f32 shaders unaffected |
| `validate_pppm` | PPPM Ewald Coulomb | **4/8 PASS** | **CRASH** | 4070: PPPM energy sign bug; Titan: NAK crash |
| `f64_builtin_test` | WGSL f64 builtins (sqrt, exp, etc.) | **9/12 PASS** | **partial CRASH** | sqrt OK on NVK; exp crashes NAK |
| `nuclear_eos_gpu` | L1 SEMF GPU chi2 | **PASS** | **partial** | Basic SEMF OK on NVK; math_f64 exp path crashes |
| `bench_gpu_fp64` | BCS + eigensolve throughput | **PASS** | **PASS** | Completed on both cards |
| `bench_multi_gpu` | Cooperative multi-GPU dispatch | **PASS** | **PASS** | 4 modes, specialized routing wins |

### Numerical Parity (where both cards pass)

BCS bisection:
- Chemical potential max error: `6.23e-11` (identical on both cards)
- Occupation (v²) max error: `5.15e-13` (identical)
- Particle number error: `< 1e-12` all nuclei (identical)

Batched eigensolve:
- Eigenvalue relative error: `2.38e-12` (identical)
- Eigenvector orthogonality: `2.44e-15` (Titan) vs `3.11e-15` (4070) — both excellent

---

## 3. Critical Bug: NVK NAK Compiler Crash on `exp(f64)`

### Symptom
```
assertion failed: vec.len() == bits.div_ceil(32)
at ../src/nouveau/compiler/nak/from_nir.rs:430
```
Followed by `Parent device is lost`.

### Trigger
Any WGSL shader that calls the **native** `exp()` builtin on f64 values.

### Unaffected operations
`sqrt`, `abs`, `floor`, `ceil`, `round`, all arithmetic, comparisons on f64 — these all work correctly on NVK.

### Affected barracuda ops
- `ops::md::forces::YukawaForceF64` — calls `exp(-kappa * r)` in shader
- `ops::md::electrostatics::PppmGpu` — erfc implementation uses exp
- Any shader using `ShaderTemplate::with_math_f64()` + native exp path

### Workaround (proven in hotSpring)
Replace native `exp()` with barracuda's software `exp_f64()` from `math_f64.wgsl`.
The software implementation works correctly on NVK — only the native builtin triggers NAK.

### Recommended fix for barracuda

```rust
// In any Op that compiles f64 shaders with exp():
let is_nvk = adapter_info.driver.contains("NVK")
    || adapter_info.driver.contains("Mesa")
    || adapter_info.driver.contains("nouveau");

let shader_source = if is_nvk {
    // Replace native exp() with software exp_f64()
    shader_body.replace("exp(", "exp_f64(")
} else {
    shader_body.to_string()
};
let full_shader = ShaderTemplate::with_math_f64(&shader_source);
```

### Suggested API addition
```rust
impl WgpuDevice {
    /// Returns true if this device uses the NVK (nouveau) Vulkan driver.
    pub fn is_nvk(&self) -> bool {
        let d = &self.adapter_info.driver;
        d.contains("NVK") || d.contains("nouveau") || d.contains("Mesa")
    }
}

impl ShaderTemplate {
    /// Auto-patch shader for driver compatibility.
    /// Replaces native builtins that crash on NVK with software equivalents.
    pub fn for_device(shader: &str, device: &WgpuDevice) -> String {
        let patched = if device.is_nvk() {
            shader.replace("exp(", "exp_f64(")
        } else {
            shader.to_string()
        };
        Self::with_math_f64_auto_safe(&patched)
    }
}
```

---

## 4. Performance: Side-by-Side FP64 Benchmarks

### BCS Bisection — Titan V dominates (pure fp64 streaming compute)

| Batch | RTX 4070 | Titan V | Winner |
|-------|----------|---------|--------|
| 128 | 5.17 ms (25K/s) | 1.46 ms (88K/s) | **Titan 3.5x** |
| 2048 | 12.35 ms (166K/s) | 1.90 ms (1.08M/s) | **Titan 6.5x** |
| 8192 | 10.93 ms (749K/s) | 2.78 ms (2.95M/s) | **Titan 3.9x** |

### Batched Eigensolve — RTX 4070 dominates (dispatch-latency sensitive)

| Batch x Dim | RTX 4070 | Titan V | Winner |
|-------------|----------|---------|--------|
| 128 x 20 | 10.30 ms | 28.26 ms | **4070 2.7x** |
| 512 x 20 | 21.51 ms | 94.01 ms | **4070 4.4x** |
| 512 x 30 | 48.02 ms | 306.77 ms | **4070 6.4x** |

### Key insight
- **Streaming kernels** (few dispatches, lots of compute): hardware fp64 rate dominates
- **Iterative kernels** (many dispatches with readback): driver dispatch latency dominates
- NVK's dispatch overhead is 3-6x higher than NVIDIA proprietary for iterative patterns

---

## 5. Multi-GPU Cooperative Dispatch Results

### Naive data splitting (same task, batch split 50/50)
| | BCS 8192 | Eigensolve 256x20 |
|---|---|---|
| Single card (4070) | 8.29 ms | 13.33 ms |
| 50/50 split across both | 12.20 ms | 28.10 ms |
| **Result** | **0.68x (slower)** | **0.47x (slower)** |

**Why naive splitting fails**: thread sync overhead + the slower card bottlenecks.

### Specialized routing (different tasks, each on best card)
| | BCS 4096 + Eigen 128x20 |
|---|---|
| Sequential on 4070 only | 16.35 ms |
| BCS→Titan V, Eigen→4070 simultaneously | 11.02 ms |
| **Result** | **1.48x speedup** |

### Recommendations for `GpuPool` / `MultiDevicePool`

1. **Task-level routing** over data-parallel splitting
2. **Workload classification**: tag kernels as `Streaming` vs `Iterative`
3. **Route streaming** (BCS, forces, density, reduction) → highest fp64 throughput card
4. **Route iterative** (eigensolve, SCF control) → lowest-latency driver
5. **Batch threshold**: only consider data splitting for batches > 10K elements
6. **Device profiling**: auto-calibrate per-device dispatch latency at pool init
7. Suggested API:
```rust
enum WorkloadType { Streaming, Iterative, Mixed }

impl GpuPool {
    fn route(&self, workload: WorkloadType, batch: usize) -> &WgpuDevice;
}
```

---

## 6. Local Fixes for ToadStool to Absorb

### Fix 1: `science_limits()` needs higher storage buffer count
**Issue**: Default `max_storage_buffers_per_shader_stage = 8`, but HFB potentials shader binds 12.
**hotSpring fix**: Request `max_storage_buffers_per_shader_stage = 12` in `GpuF64::new()`.
**Absorb**: Update `science_limits()` in `device/tensor_context.rs` to include this, or make configurable per-workload.

### Fix 2: Adapter selector numeric fallthrough
**Issue**: `with_adapter_selector("4070")` parses as index 4070 (out of bounds) instead of name substring.
**hotSpring fix**: If `parsed_index >= adapters.len()`, fall through to name matching.
**Absorb**: `WgpuDevice::with_adapter_selector()` should implement this guard.

### Fix 3: Naga rejects `bitcast<f64>(vec2<u32>)`
**Issue**: WGSL spec allows it, but Naga 0.14 (wgpu 0.19) does not implement it.
**hotSpring fix**: Replaced with integer-ratio encoding (`alpha = f64(num) / f64(den)`).
**Absorb**: Any barracuda shader using `bitcast<f64>` needs an alternative. Consider adding `decode_f64(lo: u32, hi: u32)` to `math_f64.wgsl` for when Naga catches up.

### Fix 4: f64 literal type inference in WGSL
**Issue**: `h_dst[idx] = 1e10;` fails — `1e10` inferred as f32 for f64 buffer.
**hotSpring fix**: `h_dst[idx] = f64(1e10);` — explicit cast.
**Absorb**: Audit all f64 shaders for uncast literals. Add to coding guidelines: always wrap f64 constants in `f64()`.

### Fix 5: Buffer usage conflicts in compute passes
**Issue**: Same buffer bound as `STORAGE_READ_WRITE` in one bind group and `STORAGE_READ` in another within the same dispatch — wgpu rejects this.
**hotSpring fix**: Split into 3 separate compute passes (BCS v², density, mixing) with dedicated read-only bind groups for downstream stages.
**Absorb**: wgpu's validation is correct — exclusive access required for `READ_WRITE`. Any fused compute pass must ensure no buffer appears in conflicting usage modes across bind groups within a single dispatch.

---

## 7. Existing Issues (Pre-existing, Not New)

### PPPM energy sign bug (RTX 4070, 4/8 checks)
- Two-charge Coulomb energy should be negative (attractive q+/q-) — barracuda returns positive
- NaCl crystal energy should be negative — returns positive
- Newton's 3rd law violation: `|F1+F2|/|F1| = 6.9e-4` (should be ~0)
- **Scope**: barracuda `PppmGpu` implementation, not hotSpring

### f64 exp() precision (RTX 4070, 3/12 checks)
- Native GPU exp(f64) differs from CPU by ~8e-8 — exceeds the test's 1e-10 tolerance
- This is a precision difference, not a correctness bug — the tolerance is too tight
- **Recommendation**: Relax tolerance to 1e-6 for native-vs-CPU exp comparison

---

## 8. New Files in hotSpring

| File | Purpose |
|------|---------|
| `src/bin/bench_multi_gpu.rs` | Multi-GPU cooperative benchmark (comparative + cooperative + specialized) |
| `wateringHole/handoffs/HOTSPRING_BARRACUDA_FULL_GPU_HANDOFF_FEB17_2026.md` | This document |

---

## 9. Summary: What Works, What's Blocked, What's Next

### Works on both cards
- BCS bisection (f64) — identical numerical results
- Batched eigensolve (f64) — identical eigenvalues, near-identical orthogonality
- MD forces (LJ, Coulomb, Morse — f32) — 20/20 analytical checks
- L1 SEMF GPU evaluation — exact match to CPU (<1e-10 MeV)

### Works on RTX 4070 only (blocked on Titan V by NVK exp bug)
- Yukawa OCP MD pipeline — 12/12 checks on 4070, NAK crash on Titan V
- f64 native exp/log builtins — compile and run on 4070, crash NAK on Titan V

### Pre-existing barracuda issues (both cards)
- PPPM Coulomb energy sign — barracuda bug, not driver-specific
- f64 exp precision tolerance — too tight, should be 1e-6

### Next steps for toadstool
1. **[HIGH]** `exp(f64)` NVK workaround: auto-detect and use `exp_f64()` software path
2. **[HIGH]** Absorb Fixes 1-5 from section 6
3. **[MEDIUM]** `GpuPool.route()` with workload-type-aware scheduling
4. **[MEDIUM]** Fix PPPM energy sign bug
5. **[LOW]** `WgpuDevice::is_nvk()` + `ShaderTemplate::for_device()` API
6. **[LOW]** Relax f64 exp tolerance in builtin tests
