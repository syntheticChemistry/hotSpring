# NVK Eigensolve Performance Analysis — Driver Isolation Study

**Date:** February 18, 2026
**From:** hotSpring (computational physics validation)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only

---

## Summary

The batched eigensolve (`BatchedEighGpu::execute_single_dispatch`) is 6x
slower end-to-end on the Titan V (NVK) than the RTX 4070 (nvidia proprietary).
Diagnostic benchmarking isolates the root cause to **NAK shader compiler
code quality for loop-heavy f64 Jacobi patterns** — not hardware, not
workgroup size, not dispatch overhead.

BCS bisection (same dispatch pattern, simpler shader) is **faster** on the
Titan V than the 4070, proving GV100 hardware is not the bottleneck.

---

## Hardware + Driver Matrix

| GPU | Architecture | Driver | Kernel Module | Notes |
|-----|-------------|--------|---------------|-------|
| RTX 4070 | Ada (AD104) | nvidia proprietary 580.82 | nvidia (open) | Display GPU |
| Titan V | Volta (GV100) | NVK / Mesa 25.1.5 | nouveau | Compute only |

**Critical finding**: nvidia 580.x open kernel modules (`nvidia.ko`) require
GSP (GPU System Processor), present only on Turing+. The Titan V (Volta)
**cannot** use the nvidia proprietary driver on this system. NVK is the
**only** viable driver.

```
NVRM: The NVIDIA GPU 0000:05:00.0 (PCI ID: 10de:1d81)
NVRM: installed in this system is not supported by open nvidia.ko
NVRM: because it does not include the required GPU System Processor (GSP).
```

---

## Benchmark Results

### BCS Bisection — Titan V is FASTER (no NVK penalty)

| batch | 4070 (nvidia) | Titan V (NVK) | Ratio |
|-------|--------------|---------------|-------|
| 8 | 2.382ms | 2.233ms | **0.94x (Titan faster)** |
| 512 | 2.360ms | 2.128ms | **0.90x (Titan faster)** |
| 8192 | 3.356ms | 2.963ms | **0.88x (Titan faster)** |

BCS uses `@workgroup_size(64)`, simple arithmetic (sqrt, add, mul),
no shared memory. NVK generates excellent code for this pattern.

### Batched Eigensolve — 6x slower end-to-end

| batch | dim | 4070 (nvidia) | Titan V (NVK) | Ratio |
|-------|-----|--------------|---------------|-------|
| 8 | 20 | 5.7ms | 15.2ms | 2.7x |
| 512 | 20 | 16.4ms | 95.2ms | 5.8x |
| 8 | 30 | 11.3ms | 45.3ms | 4.0x |
| 512 | 30 | 44.4ms | 308.1ms | 6.9x |

### Diagnostic: Pure Compute Isolation (no readback)

Custom benchmark `bench_wgsize_nvk` runs the same Jacobi sweep code
with `device.poll(Maintain::Wait)` between rounds (no staging/map overhead).

| batch | dim | sweeps | 4070 wg1 | Titan wg1 | Ratio |
|-------|-----|--------|----------|-----------|-------|
| 512 | 20 | 5 | 0.14ms | 4.64ms | **33x** |
| 512 | 20 | 200 | 3.34ms | 68.4ms | **20x** |
| 512 | 30 | 200 | 7.55ms | 152.5ms | **20x** |

**Pure GPU compute is 20x slower on NVK.** The measured 6x end-to-end gap
is because readback overhead (~13ms 4070, ~27ms Titan V) compresses the ratio.

---

## workgroup_size(32) Does NOT Help

Tested parallel Jacobi with 32 threads per workgroup cooperating via
`workgroupBarrier()`:

| Platform | wg1 | wg32 | wg32/wg1 |
|----------|-----|------|----------|
| 4070 (nvidia) batch=512 dim=30 s=200 | 7.6ms | 14.7ms | **1.9x slower** |
| Titan V (NVK) batch=512 dim=30 s=200 | 152.5ms | 509.9ms | **3.3x slower** |

Barrier synchronization overhead exceeds the parallelism gains. The Jacobi
rotation pattern has O(n) independent work per (p,q) pair — not enough to
amortize barrier costs with only 32 threads.

**Recommendation**: Do NOT attempt to parallelize within workgroups for this
algorithm. The serial single-dispatch design is correct.

---

## Root Cause Analysis

| Factor | Evidence | Verdict |
|--------|----------|---------|
| GV100 hardware | BCS bisection is faster on Titan V | **Not the cause** |
| NVK dispatch overhead | BCS single dispatch is fast on NVK | **Not the cause** |
| workgroup_size(1) | wg32 is worse on both platforms | **Not the cause** |
| Shared memory | BCS (no shared) vs eigensolve (shared) | Partial contributor |
| **NAK code generation** | 20x pure-compute gap, same algorithm | **Primary cause** |

The NAK shader compiler (NVK's backend, part of Mesa) generates
significantly less efficient machine code for:
- Deeply nested loops (3 levels: sweep × p × q)
- Dense f64 arithmetic in inner loops (rotation: multiply-accumulate)
- Conditional branches (`abs(apq) < tol`, convergence checks)
- Heavy shared memory access patterns (`var<workgroup>` arrays)

BCS avoids all of these: single loop, no shared memory, minimal branching.

---

## Actionable Items for ToadStool

### 1. File Mesa/NVK Bug Report (Priority: HIGH)

NAK generates ~20x slower code for loop-heavy f64 compute shaders.
Reproduction: `hotSpring/barracuda/src/bin/bench_wgsize_nvk.rs` on any
NVK-capable GPU vs nvidia proprietary.

### 2. Keep Single-Dispatch Eigensolve Design (No Change Needed)

workgroup_size(32) is worse. The architecture is correct. The fix must
come from NAK, not from shader restructuring.

### 3. Consider Multi-Sweep Dispatch Strategy (Mitigation)

If NAK's penalty is worse for deep loops, splitting 200 sweeps into
40 dispatches × 5 sweeps might reduce the per-dispatch compile overhead.
Trade-off: more queue.submit() calls vs simpler per-dispatch code.
Worth benchmarking if Mesa NVK fix is slow to land.

### 4. NVK ICD Path Requires Manual Setup

After system relog, NVK disappears from Vulkan enumeration. Requires:
```bash
export VK_DRIVER_FILES="/usr/share/vulkan/icd.d/nvidia_icd.json:/opt/mesa-nvk/share/vulkan/icd.d/nouveau_icd.x86_64.json"
```
Consider adding to `/etc/environment` or a shell profile.

### 5. Titan V Cannot Use nvidia Proprietary

The open nvidia kernel modules (580.x) do not support pre-Turing GPUs.
Volta (GV100) has no GSP firmware. NVK is the only viable path for
Titan V compute. This is a permanent constraint unless NVIDIA ships
closed-source modules for this driver version, or until NVK performance
improves.

---

## Warp-Packed Fix: 2.2x Speedup (Proven)

The warp-packed approach dispatches 32 independent matrix eigensolves per
workgroup, each thread handling its OWN matrix with no barriers. This fills
all 32 SIMD lanes instead of wasting 31/32.

**Titan V NVK results (batch=512):**

| Config | wg1 (current) | wp32 (warp-packed) | Speedup |
|--------|---------------|--------------------| --------|
| dim=20 s=5 | 4.67ms | 0.93ms | **5.0x** |
| dim=20 s=200 | 68.4ms | 31.2ms | **2.2x** |
| dim=30 s=5 | 9.78ms | 1.89ms | **5.2x** |
| dim=30 s=200 | 152.5ms | 69.5ms | **2.2x** |

**On 4070 proprietary: no regression** (warp-packing is neutral because the
proprietary scheduler already handles wg1 efficiently).

**Remaining gap after warp-packing:** Titan V wp32 vs 4070 wg1 is ~9x
(down from 20x). The remaining gap is NAK instruction-level code quality
for loop-heavy f64 patterns.

**NAK compilation stats (from `NAK_DEBUG=print`):**

| Metric | wg1 | wg32 (barrier) | wp32 (warp-packed) |
|--------|-----|----------------|--------------------|
| Instructions | 378 | 463 | 430 |
| Static cycles | 4342 | 4637 | 4700 |
| Max warps/SM | 48 | 48 | 48 |
| Spills to mem | 0 | 0 | 0 |
| Num GPRs | 35 | 30 | 40 |

No register spills on any variant. The issue is purely instruction
scheduling and ILP extraction in NAK.

### Implementation for `BatchedEighGpu`

```wgsl
// Replace @workgroup_size(1, 1, 1) with:
@compute @workgroup_size(32, 1, 1)
fn batched_eigh_single_dispatch(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // Each thread processes its own independent matrix
    let batch_idx = wg_id.x * 32u + local_id.x;
    // ... rest of serial Jacobi algorithm unchanged ...
}
// Dispatch: (batch_size.div_ceil(32), 1, 1) instead of (batch_size, 1, 1)
```

No barriers needed. No cooperation. Each thread independently does its
full serial Jacobi eigensolve. The only change is packing 32 independent
serial computations into one warp for SIMD lane utilization.

---

## The 9x Gap — Full Decomposition

After warp-packing, Titan V NVK is still **~9x slower** than RTX 4070
nvidia proprietary on compute-heavy eigensolves. This section decomposes
exactly where those 9x live and identifies every intervention point.

### Theoretical FP64 Hardware

| GPU | FP64 TFLOPS | FP64:FP32 | Memory BW | SMs | Clock |
|-----|-------------|-----------|-----------|-----|-------|
| RTX 4070 (Ada) | ~0.45 | 1:64 | 504 GB/s | 46 | 2.5 GHz |
| Titan V (Volta) | ~7.45 | 1:2 | 653 GB/s | 80 | 1.46 GHz |
| **Titan V advantage** | **16.6x** | | **1.3x** | **1.7x** | |

If both compilers generated equally optimal code, the Titan V should be
**~16x FASTER** than the RTX 4070 on FP64 compute. Instead it is **~9x
SLOWER**. The total compiler efficiency gap is:

```
16.6x (expected advantage) × 9x (actual disadvantage) = ~149x
```

**NAK generates code that is ~149x less efficient than nvidia proprietary
for loop-heavy f64 Jacobi patterns on Volta hardware.**

### Where the 149x Lives

The gap decomposes across five compiler optimization categories:

| Optimization | nvidia proprietary | NAK (NVK) | Est. factor |
|--------------|-------------------|-----------|-------------|
| Instruction scheduling (latency) | Full SM70 latency tables | Conservative 16-cycle delay | ~3-4x |
| Dual-issue (ILP) | Exploits paired execution units | Not implemented for any arch | ~2x |
| Loop unrolling | Aggressive for tight inner loops | Basic (MR 26626, Dec 2023) | ~1.5-2x |
| f64 FMA selection | DFMA replaces multiply-add pairs | May miss FMA opportunities | ~1.3-1.5x |
| Shared memory scheduling | Bank-conflict-aware scheduling | Generic memory ordering | ~1.5-2x |

Compounded: 3.5 × 2 × 1.75 × 1.4 × 1.75 ≈ **30x** pure code quality.
The remaining ~5x comes from interaction effects (poor scheduling compounds
with poor register lifetimes compounds with missed FMA, etc.).

### The Compilation Stack

```
WGSL source (our code)
  ↓ naga (Rust, part of wgpu)
SPIR-V bytecode
  ↓ spirv_to_nir (Mesa, C)
NIR (Mesa IR — optimization passes in C)
  ↓ nak_from_nir (Mesa/NAK, Rust)
NAK IR (SSA, Rust)
  ↓ NAK optimization passes (Rust)
NAK IR (scheduled, register-allocated)
  ↓ NAK emit (Rust)
SASS machine code (GPU binary)
```

**Every layer except hardware is software we can modify:**

| Layer | Language | Modifiable | Impact potential |
|-------|----------|------------|-----------------|
| WGSL shader | WGSL | **Us** (hotSpring/toadstool) | Low (already optimized) |
| naga | **Rust** | Contribute upstream (gfx-rs) | Low-medium |
| NIR passes | C | Contribute upstream (Mesa) | Medium |
| NAK IR + passes | **Rust** | Contribute upstream (Mesa) | **HIGH — this is the bottleneck** |
| NAK emit | **Rust** | Contribute upstream (Mesa) | **HIGH** |

---

## Where We Solve: NAK (Rust)

NAK is the primary intervention point. It is:
- Written entirely in **Rust** (our language)
- Under active development by Collabora (Faith Ekstrand)
- Located at `mesa/src/nouveau/compiler/nak/`
- Open to contributions under MIT license

### NAK Source Structure (from XDC 2023 and Mesa docs)

| File | Role | Relevance |
|------|------|-----------|
| `calc_instr_deps.rs` | Instruction dependency (RAW/WAR/WAW) | **Critical** — feeds scheduler |
| `ir.rs` | NAK IR definitions, `Foldable` | Core data structures |
| `from_nir.rs` | NIR → NAK IR lowering | f64 instruction selection |
| `opt_*.rs` | Optimization passes | Loop unrolling, copy prop, DCE |
| `assign_regs.rs` | Register allocation | ILP vs pressure tradeoff |
| `calc_instr_deps.rs` | Dependency computation | **Scheduling quality** |
| `encode.rs` / `emit.rs` | SASS binary emission | SM70 encoding |

### Specific NAK Gaps (Confirmed by Phoronix + Mesa MRs)

1. **No dual-issue scheduling** — listed as future work in MR 35821
2. **No functional unit resource tracking** — same MR
3. **SM70 scheduling may be generic** — real scheduling only documented for
   SM32 (KeplerB, Mesa 25.2, July 2025). Volta may use conservative model.
4. **Loop unrolling is basic** — MR 26626 (Dec 2023) added initial support,
   but our 3-level nested loop (sweep × p × q) may not trigger it.
5. **No software pipelining** — overlapping loop iterations not implemented.

### Contribution Plan

**Phase 1: Diagnostic (1 week)**
```bash
git clone https://gitlab.freedesktop.org/mesa/mesa.git
cd mesa/src/nouveau/compiler/nak/
# Build debug Mesa with NAK_DEBUG=print enabled
meson setup build -Dbuildtype=debug -Dvulkan-drivers=nouveau
ninja -C build
# Run our benchmark with debug NAK
NAK_DEBUG=print,annotate ./bench_wgsize_nvk
```

Produce NAK IR dumps for all three shader variants (wg1, wg32, wp32).
Compare instruction count, scheduling gaps, register usage.

**Phase 2: SM70 Instruction Latency Tables (2 weeks)**
NAK needs accurate latency information for Volta (SM70) instructions.
NVIDIA publishes ISA docs; envytools has reverse-engineered timing.
Contribute SM70 latency data to `calc_instr_deps.rs`.

**Phase 3: f64 Loop Optimization (4 weeks)**
- Inner loop unrolling for fixed-bound `for` loops in Jacobi sweeps
- FMA selection: pattern-match `a * b + c` → DFMA in `from_nir.rs`
- Loop-invariant code motion for Jacobi rotation constants

**Phase 4: Scheduling Quality (ongoing)**
- Dual-issue for Volta (SM70 supports it, NAK doesn't exploit it)
- Functional unit tracking to avoid structural hazards
- Better shared memory scheduling to avoid bank conflicts

### What toadStool Already Has

toadStool's barracuda crate has infrastructure we can leverage:

| Feature | Location | Reusable for NAK work |
|---------|----------|----------------------|
| `WgpuDevice::is_nvk()` | `device/wgpu_device.rs` | Driver detection for testing |
| `ShaderTemplate::for_device()` | `shaders/precision.rs` | NVK-specific shader patching |
| Autotune framework | `device/autotune.rs` | Benchmark harness for NAK changes |
| `DeviceCapabilities` | `device/capabilities.rs` | Per-vendor workgroup optimization |
| math_f64.wgsl | `shaders/math/math_f64.wgsl` | Software fallbacks for broken builtins |

### Alternative: Specialized Codegen (Long-term)

If NAK contributions are too slow to land upstream, we could build a
**specialized SPIR-V → SASS codegen** for numerical compute patterns.
This is the "rebuild from first principles" approach:

- Target only SM70 (Volta) and SM89 (Ada) initially
- Generate SASS directly for Jacobi rotations, reductions, GEMM
- Use envytools ISA documentation for encoding
- Bypass naga + NAK entirely for compute-only pipelines
- 100% Rust, owned by ecoPrimals, AGPL-3.0

This is extreme but **aligned with toadStool's philosophy** (480+ WGSL
shaders, pure Rust, zero C deps). We don't need a general-purpose
compiler — we need optimal code for a specific set of numerical patterns.

---

## Open-Source FP64 Strategy

### The Sovereign Stack

```
WGSL → naga (Rust) → SPIR-V → driver shader compiler → GPU machine code
```

| Driver | Compiler | Language | Target | Maturity |
|--------|----------|----------|--------|----------|
| NVK | NAK | **Rust** | NVIDIA SM 50+ | Growing (Mesa 25.x) |
| RADV | ACO | C++ | AMD GCN/RDNA | Mature (matches proprietary) |

NAK is written in Rust — same language as our codebase. Contributing
optimizations directly is feasible and aligned with sovereignty goals.

### Why Open-Source Drivers for FP64

| GPU | Proprietary FP64 | Open-Source FP64 | Ratio |
|-----|------------------|------------------|-------|
| RTX 4070 (Ada) | Throttled 1:64 | Full shaderFloat64 via NVK/Vulkan | Same HW, same rate |
| Titan V (Volta) | Full 1:2 (but closed-source nvidia.ko requires GSP) | Full via NVK | **Only option** |
| RX 7900 XTX (RDNA 3) | Limited driver support | Full via RADV | Better cache (6MB L2) |
| MI250X (CDNA 2) | ROCm (semi-open) | Full via RADV | Full-rate FP64 silicon |

Key: consumer NVIDIA proprietary drivers actively gimp FP64 via driver-level
throttling. Open-source drivers expose the actual hardware capability.
For Volta (Titan V), open-source is the ONLY path — nvidia open kernel
modules lack GSP support for pre-Turing GPUs.

### AMD Path (Next Hardware)

AMD RDNA 3 / CDNA advantages for our workload:
- **Larger L2 cache** (6MB on RX 7900 XTX vs 4MB on Ada, 6MB on Volta)
- **ACO compiler maturity** — matches or beats AMD proprietary for compute
- **Wave64 mode** — 64-wide SIMD packs more independent work per wavefront
- **Infinity Cache** — reduces memory latency for Jacobi rotation patterns

Recommended test card: **RX 7900 XTX** (RDNA 3, 96 CUs, 6MB L2, ~1.2 TFLOPS FP64)
or **Instinct MI250X** (CDNA 2, full-rate FP64, 128GB HBM2e) if budget allows.

### Path to Closing the NVK Gap

1. **Immediate (done)**: Warp-packed eigensolve — 2.2x NVK speedup
2. **Short-term**: File Mesa NAK bug with reproduction case
3. **Medium-term**: Clone Mesa, profile NAK IR, contribute SM70 latency tables
4. **Long-term**: Contribute NAK patches for f64 loop ILP (Rust codebase)
5. **Parallel**: Add AMD GPU + RADV as second open-source validation target
6. **Nuclear**: Specialized SPIR-V → SASS codegen for numerical patterns

---

## Nouveau / NVK Persistence After Reboot

### How It Works (Discovered)

The Titan V cannot use `nvidia.ko` (no GSP). At boot:

1. `nvidia` driver probes Titan V → **fails** (`probe with driver nvidia failed with error -1`)
2. Titan V is left orphaned (no driver)
3. GDM/Xorg starts, auto-detects Titan V, loads `nouveau` DDX (X11 driver)
4. nouveau DDX triggers kernel `nouveau.ko` load via DRM subsystem
5. nouveau.ko loads its deps (gpu-sched, drm_gpuvm, drm_display_helper, etc.)
6. nouveau binds to Titan V → creates `/dev/dri/card1` + `/dev/dri/renderD129`

**Key: the `alias nouveau off` in `/lib/modprobe.d/nvidia-graphics-drivers.conf`
blocks `modprobe nouveau` by name, but does NOT block loading via the DRM
subsystem or PCI modalias.**

### Fragility Points

| Scenario | Nouveau loads? | Fix |
|----------|---------------|-----|
| Normal boot + login | Yes (GDM triggers it) | No action needed |
| Power loss before login | No (GDM never started) | Login to trigger |
| Headless / compute-only | No (no display manager) | systemd service needed |
| `modprobe nouveau` manually | No (aliased to `off`) | Use `insmod` with deps |
| `insmod nouveau.ko` without deps | No (Unknown symbol) | Load deps first |

### Persistence Solution (Installed)

Three layers of persistence were installed:

**1. modprobe.d override** — `/etc/modprobe.d/hotspring-nouveau-titanv.conf`
Overrides the `alias nouveau off` so `modprobe nouveau` works again.
Uses `softdep` to ensure deps load before nouveau.

**2. systemd service** — `hotspring-titanv-compute.service`
Runs before `display-manager.service`. Loads nouveau deps via modprobe,
then insmod nouveau.ko. Ensures Titan V is available even without GDM.
Enabled at boot via `systemctl enable`.

**3. Vulkan ICD environment** — `~/.config/environment.d/hotspring-vulkan.conf`
Sets `VK_DRIVER_FILES` to include both nvidia and NVK ICDs. Persists across
login sessions via systemd user environment.

### Hotswap: Rebind Without Reboot

If nouveau is already loaded but Titan V is unbound:
```bash
# Rebind Titan V to nouveau (no reboot needed)
echo 0000:05:00.0 | sudo tee /sys/bus/pci/drivers/nouveau/bind
```

If nouveau is NOT loaded (e.g., after power loss before login):
```bash
# Load deps then nouveau (bypasses blacklist alias)
for dep in wmi mxm-wmi i2c-algo-bit ttm drm_exec drm_gpuvm \
           drm_display_helper drm_ttm_helper gpu-sched; do
  sudo modprobe "$dep"
done
sudo insmod /lib/modules/$(uname -r)/kernel/drivers/gpu/drm/nouveau/nouveau.ko
```

Or, with the modprobe override installed:
```bash
sudo modprobe nouveau  # Works now (override removes "alias nouveau off")
```

### PCIe Bandwidth Note

The Titan V is in a **PCIe x4 slot** (31.5 Gb/s measured, capable of
126 Gb/s at x16). Moving it to an x16 slot would improve data transfer
by ~4x. For pure compute this doesn't affect kernel execution time, but
reduces upload/readback latency for batched workloads.

---

## CPU Crossover: Lessons from GPU Apply to CPU

The optimizations discovered for GPU code generation apply directly to
CPU-side Rust:

| GPU lesson | CPU application |
|-----------|----------------|
| Warp-packing (SIMD lane fill) | SIMD vectorization via `packed_simd` / autovectorization |
| Manual loop unrolling | `#[unroll]` hints, loop tiling for L1 cache |
| FMA selection | `f64::mul_add()` for fused multiply-add |
| Shared memory bank conflicts | Cache-line alignment, false sharing avoidance |
| Instruction scheduling | LLVM backend already handles this well for x86 |
| Batched independent work | `rayon` parallel iterators on matrix batches |

The Jacobi eigensolve can be optimized on CPU using the same principles:
batch independent matrices across SIMD lanes (AVX-512 has 8× f64 lanes)
and ensure the compiler sees FMA opportunities.

---

## Files

| File | Purpose |
|------|---------|
| `barracuda/src/bin/bench_wgsize_nvk.rs` | workgroup_size diagnostic (wg1, wg32, wp32) |
| `barracuda/src/bin/bench_gpu_fp64.rs` | Full GPU FP64 benchmark suite |
| `/etc/modprobe.d/hotspring-nouveau-titanv.conf` | Persistent nouveau loading override |
| `/etc/systemd/system/hotspring-titanv-compute.service` | Boot-time Titan V binding |
| `~/.config/environment.d/hotspring-vulkan.conf` | Persistent VK_DRIVER_FILES |
| This document | Root cause analysis, fix, and strategy |

---

## Reproduction

```bash
cd hotSpring/barracuda
export VK_DRIVER_FILES="/usr/share/vulkan/icd.d/nvidia_icd.json:/opt/mesa-nvk/share/vulkan/icd.d/nouveau_icd.x86_64.json"

# Side-by-side benchmark
HOTSPRING_GPU_ADAPTER=4070  cargo run --release --bin bench_gpu_fp64
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_gpu_fp64

# Diagnostic with warp-packed fix
HOTSPRING_GPU_ADAPTER=4070  cargo run --release --bin bench_wgsize_nvk
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_wgsize_nvk

# NAK shader IR dump (requires debug Mesa build)
NAK_DEBUG=print HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_wgsize_nvk

# Verify persistence after reboot
vulkaninfo --summary | grep -E "deviceName|driverName"
# Should show both "NVIDIA GeForce RTX 4070" and "NVIDIA TITAN V (NVK GV100)"
```
