# hotSpring → ToadStool: GPU Sovereignty First Solution & Evolution Path

**Date:** 2026-02-18
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Builds on:** `HOTSPRING_BARRACUDA_FULL_GPU_HANDOFF_FEB17_2026.md`,
              `NVK_EIGENSOLVE_PERF_ANALYSIS_FEB18_2026.md`

---

## Executive Summary

hotSpring has completed a first solution for sovereign FP64 GPU compute
on mixed-driver multi-GPU systems. We proved that open-source drivers
(NVK/nouveau) expose full FP64 hardware capabilities, identified and
mitigated the primary performance bottleneck (NAK shader compiler code
quality, not hardware), and established persistent driver coexistence
for NVIDIA proprietary + nouveau running side-by-side.

**This is a first solution.** ToadStool should evolve a pure Rust native
solution for driver/GPU handling that internalizes these lessons. The
compiler bottleneck we found (NAK generating ~149x less efficient code
for loop-heavy f64 patterns) is solvable — NAK is written in Rust.

### What Was Delivered

| Deliverable | Status | Impact |
|-------------|--------|--------|
| Warp-packed eigensolve shader | **Done** | 2.2-5.2x NVK speedup |
| NVK `exp(f64)` crash workaround | **Done** | Unblocks Yukawa MD, PPPM |
| Driver persistence (3 layers) | **Done** | Survives reboot/power loss |
| 149x gap decomposition | **Done** | 5 specific NAK deficiencies identified |
| NAK contribution plan | **Done** | Phase 1-4 with effort estimates |
| CPU crossover analysis | **Done** | GPU lessons → CPU SIMD optimization |
| Benchmark reproduction suite | **Done** | `bench_wgsize_nvk.rs`, `bench_gpu_fp64.rs` |

---

## First Solution: What We Built

### 1. Warp-Packed Eigensolve (Shader Level)

Changed from `@workgroup_size(1,1,1)` (1 thread per matrix, 31/32 SIMD
lanes wasted) to `@workgroup_size(32,1,1)` with 32 independent matrices
per workgroup (no barriers, no cooperation — each thread owns its matrix).

```
Titan V NVK, batch=512, dim=30, 200 sweeps:
  Before (wg1):  152.8ms
  After (wp32):   69.8ms  → 2.2x speedup
```

RTX 4070 proprietary: **no regression**. The proprietary scheduler already
handles wg1 efficiently; warp-packing is neutral on good compilers.

**For toadstool to absorb:**
- `batched_eigh_single_dispatch_f64.wgsl`: change `@workgroup_size(1,1,1)`
  to `@workgroup_size(32,1,1)`, compute `batch_idx = wg_id.x * 32 + local_id.x`
- `batched_eigh_gpu.rs`: dispatch `batch.div_ceil(32)` instead of `batch`
- Add early-return guard: `if batch_idx >= batch_size { return; }`

### 2. NVK exp(f64) Workaround (Already in toadstool)

toadstool's `ShaderTemplate::for_device()` and `WgpuDevice::is_nvk()`
handle this correctly. No additional changes needed.

### 3. Driver Persistence (System Level)

Three persistence layers installed on the hotSpring development machine:

**Layer 1: modprobe override** — `/etc/modprobe.d/hotspring-nouveau-titanv.conf`
```conf
# Override nvidia's "alias nouveau off" blacklist
remove nouveau /bin/true
install nouveau /sbin/modprobe --ignore-install nouveau
softdep nouveau pre: drm_display_helper ttm drm_gpuvm drm_exec \
    drm_ttm_helper gpu-sched i2c-algo-bit mxm-wmi wmi video
```

**Layer 2: systemd service** — `hotspring-titanv-compute.service`
Runs before display-manager. Loads nouveau deps, insmod nouveau.ko,
binds Titan V. Ensures compute GPU available even on headless boots.

**Layer 3: Vulkan ICD environment** — `~/.config/environment.d/hotspring-vulkan.conf`
```conf
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json:/opt/mesa-nvk/share/vulkan/icd.d/nouveau_icd.x86_64.json
```

**Hotswap without reboot** (if nouveau is loaded but Titan V unbound):
```bash
echo 0000:05:00.0 | sudo tee /sys/bus/pci/drivers/nouveau/bind
```

### Boot Sequence (Discovered)

```
t=0s    PCI enumerates: RTX 4070 (01:00.0) + Titan V (05:00.0)
t=7s    nvidia.ko probes both GPUs
        → RTX 4070: SUCCESS (has GSP)
        → Titan V: FAIL (no GSP in open nvidia.ko for Volta)
t=~8s   nvidia-drm initializes for RTX 4070 only
t=~550s GDM/Xorg starts, auto-detects orphaned Titan V
        → Loads nouveau X11 DDX → triggers kernel nouveau.ko load
        → nouveau deps loaded, Titan V bound → /dev/dri/card1 appears
```

The `alias nouveau off` blocks `modprobe nouveau` by name but NOT loading
via the DRM/Xorg subsystem. Our modprobe override makes manual loading
work too.

---

## The 149x Gap: Where Sovereign FP64 Performance Lives

### Hardware Reality

| GPU | FP64 TFLOPS | FP64:FP32 |
|-----|-------------|-----------|
| RTX 4070 (Ada) | ~0.45 | 1:64 |
| Titan V (Volta) | ~7.45 | 1:2 |
| **Titan V advantage** | **16.6x** | |

The Titan V should be 16.6x FASTER. Instead it is 9x SLOWER (after
warp-packing). Total compiler efficiency gap: **~149x**.

### The Five NAK Deficiencies

| # | Deficiency | Est. factor | NAK status |
|---|-----------|-------------|------------|
| 1 | No SM70 instruction scheduling | ~3-4x | Only SM32 (Kepler) has real scheduling |
| 2 | No dual-issue exploitation | ~2x | Not implemented for any arch |
| 3 | Basic loop unrolling | ~1.5-2x | MR 26626 (Dec 2023), may miss nested loops |
| 4 | Missing FMA selection for f64 | ~1.3-1.5x | Not confirmed, needs IR dump |
| 5 | Generic shared memory scheduling | ~1.5-2x | No bank-conflict awareness |

### The Compilation Stack (All Modifiable)

```
WGSL (our shaders)           ← hotSpring/toadstool owns this
  ↓ naga (Rust)              ← gfx-rs, contribute upstream
SPIR-V
  ↓ spirv_to_nir (C)        ← Mesa, contribute upstream
NIR
  ↓ nak_from_nir (Rust)     ← Mesa/NAK, contribute upstream
NAK IR (SSA)
  ↓ NAK opt passes (Rust)   ← Mesa/NAK, PRIMARY TARGET
SASS machine code
```

**NAK is written entirely in Rust.** Same language as our codebase. Same
build toolchain. Same development philosophy. Contributing to NAK is
directly aligned with ecoPrimals sovereignty goals.

---

## Evolution Path for ToadStool

### Phase 1: Absorb First Solution (This Handoff)

Absorb the warp-packed eigensolve change into `batched_eigh_single_dispatch_f64.wgsl`.
This is a one-file shader change + one-line dispatch change. No risk to
other ops. Immediate 2.2x NVK improvement.

### Phase 2: Pure Rust GPU Discovery & Driver Handling

toadstool already has the foundation:
- `WgpuDevice::is_nvk()`, `is_radv()`, `is_nvidia_proprietary()`
- `DeviceCapabilities::from_device()` with per-vendor workgroup sizes
- `GpuCalibration` auto-tuning framework
- `BARRACUDA_GPU_ADAPTER` environment variable for selection

**Evolve this into a full native driver awareness layer:**

```rust
pub struct GpuDriverProfile {
    pub driver: DriverKind,           // NVK, RADV, NvidiaProprietary, ...
    pub compiler: CompilerKind,       // NAK, ACO, NvidiaPtxas, ...
    pub arch: GpuArch,               // Volta, Ada, RDNA3, CDNA2, ...
    pub fp64_rate: Fp64Rate,         // Full, Throttled(ratio), Software
    pub scheduling_quality: Quality,  // Known compiler quality for our patterns
    pub workarounds: Vec<Workaround>, // exp_f64 substitution, etc.
}

impl GpuDriverProfile {
    pub fn optimal_eigensolve_strategy(&self) -> EigensolveStrategy {
        match (self.compiler, self.arch) {
            (NAK, Volta) => EigensolveStrategy::WarpPacked { wg_size: 32 },
            (NAK, _)     => EigensolveStrategy::WarpPacked { wg_size: 32 },
            (ACO, RDNA3) => EigensolveStrategy::WavePacked { wave_size: 64 },
            _            => EigensolveStrategy::Standard,
        }
    }
}
```

This makes shader specialization data-driven rather than string-matching.

### Phase 3: NAK Contribution (Medium-Term)

Clone Mesa, build debug NAK, contribute upstream:

1. **SM70 latency tables** in `calc_instr_deps.rs` — use envytools data
2. **f64 FMA pattern matching** in `from_nir.rs` — fold `mul + add` → `DFMA`
3. **Loop unrolling** for bounded nested loops — our Jacobi pattern
4. **Dual-issue** for Volta — paired execution units

Every NAK improvement benefits ALL NVK users, not just us. This is the
open-source multiplier effect.

### Phase 4: Specialized Codegen (Long-Term, Optional)

If NAK upstream moves too slowly, build a specialized Rust codegen:

```rust
pub trait NumericalCodegen {
    fn emit_jacobi_kernel(&self, arch: GpuArch, n: usize) -> Vec<u8>;
    fn emit_reduction(&self, arch: GpuArch, op: ReduceOp) -> Vec<u8>;
    fn emit_gemm_f64(&self, arch: GpuArch, tile: TileSize) -> Vec<u8>;
}
```

- Target SM70 (Volta) and SM89 (Ada) initially
- Generate SASS directly from pattern templates
- Use envytools ISA documentation for encoding
- Bypass naga + NAK entirely for compute-only pipelines
- 100% Rust, AGPL-3.0, owned by ecoPrimals

This is the "rebuild from first principles" option — extreme but aligned
with toadStool's 480+ WGSL shaders, pure Rust, zero C deps philosophy.

### Phase 5: AMD as Second Open-Source Target

AMD RADV/ACO is more mature than NVK/NAK for compute:
- ACO matches or beats AMD proprietary driver on many workloads
- Wave64 mode: 64-wide SIMD (vs 32 on NVIDIA)
- Larger L2 cache: 6MB on RX 7900 XTX
- Infinity Cache reduces memory latency for Jacobi rotation patterns

toadstool already has `WgpuDevice::is_radv()` and vendor-specific
workgroup sizes for AMD. Adding an AMD GPU to the test matrix would
provide a second data point for compiler quality comparison.

---

## CPU Crossover: GPU Lessons Apply to CPU

| GPU lesson | CPU application |
|-----------|----------------|
| Warp-packing (SIMD lane fill) | AVX-512: 8 f64 lanes per vector |
| Manual loop unrolling | `#[unroll]` hints, loop tiling for L1 |
| FMA selection | `f64::mul_add()` maps to hardware FMA |
| Shared memory bank conflicts | Cache-line alignment, false sharing |
| Batched independent work | `rayon` parallel iterators on matrix batches |

The Jacobi eigensolve on CPU can be optimized with the same principles:
pack 8 independent matrices across AVX-512 lanes, ensure LLVM sees FMA
opportunities, tile loops for L1 cache residency. This is the
"lessons from GPU apply to CPU" principle — and vice versa.

---

## Files Delivered

### New Files (Untracked)

| File | Purpose |
|------|---------|
| `barracuda/src/bin/bench_wgsize_nvk.rs` | Workgroup-size diagnostic (wg1, wg32, wp32) |
| `barracuda/src/md/shaders/yukawa_force_f64_nvk_safe.wgsl` | NVK-safe Yukawa shader |
| `barracuda/src/md/yukawa_nvk_safe.rs` | NVK-safe Yukawa module |
| `NVK_EIGENSOLVE_PERF_ANALYSIS_FEB18_2026.md` | Full 149x gap analysis |
| This document | Handoff + evolution path |

### Modified Files

| File | Change |
|------|--------|
| `barracuda/Cargo.toml` | Added `bench_wgsize_nvk` binary |
| `barracuda/src/gpu.rs` | `HOTSPRING_GPU_ADAPTER` env var |
| `barracuda/src/physics/bcs_gpu.rs` | wgpu 22 API update |
| `barracuda/src/physics/hfb_gpu_resident.rs` | wgpu 22 API update |
| `barracuda/src/tolerances.rs` | Added GPU-specific tolerance constants |
| `barracuda/src/bin/bench_gpu_fp64.rs` | Multi-GPU adapter selection |
| `barracuda/src/bin/bench_multi_gpu.rs` | Multi-GPU adapter selection |
| `barracuda/src/bin/f64_builtin_test.rs` | NVK exp(f64) crash detection |
| `barracuda/src/bin/validate_barracuda_pipeline.rs` | Adapter selection |
| `barracuda/src/bin/validate_special_functions.rs` | Minor fix |
| `barracuda/src/md/mod.rs` | NVK-safe module registration |

### System Persistence Files (Development Machine)

| File | Purpose |
|------|---------|
| `/etc/modprobe.d/hotspring-nouveau-titanv.conf` | Override nouveau blacklist |
| `/etc/systemd/system/hotspring-titanv-compute.service` | Boot-time Titan V binding |
| `~/.config/environment.d/hotspring-vulkan.conf` | Persistent VK_DRIVER_FILES |

---

## Benchmark Data (Post-Power-Loss Validated)

All benchmarks re-run and confirmed identical after power loss recovery.

### Warp-Packed Eigensolve (batch=512, 200 sweeps)

| Config | RTX 4070 (nvidia) | Titan V (NVK) wg1 | Titan V (NVK) wp32 | NVK speedup |
|--------|:--:|:--:|:--:|:--:|
| dim=20 | 3.5ms | 68.7ms | 31.5ms | **2.2x** |
| dim=30 | 7.4ms | 152.8ms | 69.9ms | **2.2x** |

### BCS Bisection (Titan V Wins)

| batch | RTX 4070 | Titan V | Ratio |
|-------|:--:|:--:|:--:|
| 8 | 2.38ms | 2.23ms | **Titan 1.07x faster** |
| 8192 | 3.36ms | 2.96ms | **Titan 1.14x faster** |

### Dispatch-Dominated (batch=512, 5 sweeps)

| Config | RTX 4070 wg1 | Titan V wp32 | Ratio |
|--------|:--:|:--:|:--:|
| dim=20 | 0.16ms | 0.93ms | 5.8x gap |
| dim=30 | 0.26ms | 1.88ms | 7.2x gap |

The dispatch-dominated case shows a smaller gap because dispatch
overhead (not compute) dominates — and NVK dispatch is only ~5-7x
slower, not 20x.

---

## Reproduction

```bash
cd hotSpring/barracuda
export VK_DRIVER_FILES="/usr/share/vulkan/icd.d/nvidia_icd.json:\
/opt/mesa-nvk/share/vulkan/icd.d/nouveau_icd.x86_64.json"

# Verify both GPUs visible
vulkaninfo --summary | grep -E "deviceName|driverName"

# Side-by-side eigensolve diagnostic
HOTSPRING_GPU_ADAPTER=4070  cargo run --release --bin bench_wgsize_nvk
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_wgsize_nvk

# Full GPU FP64 benchmark
HOTSPRING_GPU_ADAPTER=4070  cargo run --release --bin bench_gpu_fp64
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_gpu_fp64
```

---

## Summary for ToadStool

**Absorb now:**
1. Warp-packed eigensolve shader change (one file, one dispatch line)
2. `bench_wgsize_nvk.rs` as a permanent diagnostic binary

**Plan for next evolution:**
1. `GpuDriverProfile` — data-driven shader specialization
2. NAK contribution — SM70 latency tables, f64 FMA, loop unrolling
3. AMD RADV as second open-source target
4. CPU SIMD optimization using GPU lessons

**The 149x gap is a compiler problem in a Rust codebase.** We can solve it.
Every improvement we contribute to NAK benefits the entire open-source
GPU compute ecosystem. This is sovereignty in practice — not just
avoiding proprietary dependencies, but building the tools ourselves.
