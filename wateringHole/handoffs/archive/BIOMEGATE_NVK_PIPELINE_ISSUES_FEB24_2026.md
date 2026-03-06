# NVK Pipeline Issues + Open-Driver Evolution Strategy

**Date:** February 24, 2026
**From:** hotSpring (biomeGate compute campaign, Day 2)
**To:** ToadStool core team
**License:** AGPL-3.0-only
**Purpose:** Document NVK pipeline failures discovered during production runs,
propose an open-driver evolution path, and provide timing comparisons for
streaming optimization.

---

## Executive Summary

During the biomeGate Week-of-Runs campaign, we hit three classes of NVK issues:

1. **Device loss at 32^4 lattice** — Titan V (NVK) crashes with "Parent device
   is lost" on production-size lattices (~600 MB storage buffers)
2. **NAK exp/log f64 crash** — Known, workaround in place, but limits native builtins
3. **Dispatch limit workaround needed** — wgpu's 65535 workgroup limit required
   2D dispatch refactor across 15 WGSL shaders

These are solvable. NVIDIA has already dropped Titan V from proprietary driver
support. Our long-term compute strategy **must** be open-driver (NVK/nouveau).
This document proposes the path.

---

## 1. NVK Pipeline Failures (Production Campaign)

### 1.1 Device Loss at 32^4 (CRITICAL)

**Symptom:** `wgpu::Queue::submit` panics with "Parent device is lost" when
running streaming HMC on 32^4 lattice (1,048,576 sites).

**Context:**
- 16^4 (65,536 sites): works perfectly, all 9 beta points completed
- 32^4 (1,048,576 sites): device lost during first GPU thermalization
- Buffer sizes: link buffer = 603 MB, momentum buffer = 603 MB, total ~1.8 GB
- Titan V has 12 GB VRAM — memory is NOT the issue
- RTX 3090 (nvidia proprietary) handles 32^4 without issue

**Kernel log (from `journalctl -k`):**
```
nouveau 4b:00.0: fifo: fault 00 [VIRT_READ] at 0000003f87c54000
  engine 40 [gr] client 04 [GPC4/T1_4] reason 02 [PTE]
  on channel 3 [production_beta[809246]]
nouveau 4b:00.0: gr: GPC0/TPC5/SM1 trap: global [...] MULTIPLE_WARP_ERRORS
  (repeated across all 6 GPCs, 24+ SMs)
nouveau 4b:00.0: fifo: channel 3 killed!
```

**Root cause:** **PTE fault** — the GPU attempted a virtual memory read at an
address without a valid page table entry. This is a nouveau kernel module
virtual memory management bug, not a user-space NVK issue. The fault address
`0x3f87c54000` (~254 GB into the VA space) suggests nouveau's BO (buffer
object) mapping failed for the large combined allocation (~1.8 GB across
link + momentum + force + plaquette buffers).

**Binary search results (Feb 24):**
- 20^4 (160,000 sites, ~83 MB link buf): OK
- 24^4 (331,776 sites, ~172 MB link buf): OK
- 28^4 (614,656 sites, ~318 MB link buf): OK
- 30^4 (810,000 sites, ~418 MB link buf): OK
- **31^4 (923,521 sites, ~477 MB link buf): CRASH (device lost)**
- 32^4 (1,048,576 sites, ~603 MB link buf): CRASH (device lost)

The threshold is between 30^4 and 31^4. Total VRAM at 31^4 is ~1.4 GB
(link + momentum + force + plaquette). The Titan V has 12 GB — this is
not a capacity issue but a virtual address space or BO mapping limit
in nouveau's `drm_gpuvm`.

**Investigation path:**
1. ~~Binary search lattice size: 16^4 OK → try 20^4, 24^4, 28^4~~ DONE (30^4 OK, 31^4 crash)
2. Test with fewer concurrent buffers (reduce pipeline to minimal 2-buffer HMC)
3. Check nouveau's `drm_gpuvm` BO mapping limit on GV100
4. Test with Mesa git HEAD (25.1.5 is 6 months old; nouveau kernel fixes land separately)
5. Test with smaller encoders (split 40-step trajectory into 4×10-step sub-encoders)
6. Enable `VK_MESA_NVK_DEBUG=*` for NVK-side debugging
7. File upstream Mesa/nouveau bug with this exact kernel trace

### 1.2 NAK exp(f64) / log(f64) Assertion (MEDIUM)

**Symptom:** `nak/from_nir.rs:430: assertion failed: vec.len() == bits.div_ceil(32)`
when compiling WGSL shaders that use native `exp()` or `log()` on `f64`.

**Current workaround:** `ShaderTemplate::for_driver_auto()` detects NVK via
`GpuDriverProfile` and applies polynomial approximation workarounds
(`Workaround::NvkExpF64Crash`, `NvkLogF64Crash`).

**Impact:** validate_all scores 38/39 (only "WGSL f64 Builtins" fails).
All physics-producing shaders use the workaround. No data quality impact.

**Upstream path:** The fix is in Mesa's `src/nouveau/compiler/nak/from_nir.rs` —
the 128-bit f64 return value from transcendental builtins is not correctly
split for the NAK register allocator. This is a compiler bug, not a hardware
limitation (Volta SM70 supports `MUFU.EX2` and `MUFU.LG2` natively).

### 1.3 Workgroup Dispatch Limit (RESOLVED)

**Symptom:** `dispatch_workgroups([65536, 1, 1])` exceeds wgpu's 65535 limit
for 32^4 lattices with workgroup_size(64).

**Fix applied:** Refactored `barracuda/src/gpu/dispatch.rs` to use 2D dispatch
via `split_workgroups()`, and updated all 15 lattice WGSL shaders to linearize
`global_invocation_id` using `@builtin(num_workgroups)`:

```wgsl
let idx = gid.x + gid.y * nwg.x * 64u;
```

Backward-compatible: with 1D dispatch, `gid.y = 0` so `idx = gid.x`.

**Verified:** `validate_gpu_beta_scan` passes 6/6 after refactor.

---

## 2. Timing Comparison: Day 1-2 Results

### 2.1 Day 1: Deferred Runs

| Run | GPU | Wall Time | Key Result |
|-----|-----|-----------|------------|
| sarkas_gpu --paper (9 cases) | RTX 3090 | ~3.5 hrs | 9/9 paper parity |
| validate_all (39 suites) | Titan V (NVK) | 6089.6s (1.7 hrs) | 38/39 PASS |
| nuclear_eos_l1_ref --pareto | CPU (64t) | ~73 min | λ=100 best: 4/5 2σ |
| nuclear_eos_l2_gpu --phase1 | RTX 3090 | 1930.2s (32 min) | 791 HFB nuclei |
| bench_gpu_fp64 | Titan V (NVK) | ~15s | 1.95M BCS/s |

### 2.2 Day 2: Production Beta-Scans

| Run | GPU | Lattice | Points | Wall Time | Per-Point | Acceptance |
|-----|-----|---------|--------|-----------|-----------|------------|
| quenched scan | Titan V (NVK) | 16^4 | 9 | 47.4 min | 316s | 53-66% |
| quenched scan | RTX 3090 | 32^4 | 12/12 (complete) | 13.6 hrs ($0.58) | 4082s avg | 15-24% |
| quenched scan | Titan V (NVK) | 32^4 | CRASH | — | — | — |

### 2.3 Per-Trajectory Timing

| Lattice | GPU | dt | n_md | Time/Traj | Notes |
|---------|-----|----|------|-----------|-------|
| 8^4 | RTX 3090 | 0.050 | 10 | ~0.35s | validate_gpu_beta_scan |
| 16^4 | Titan V (NVK) | 0.025 | 20 | ~1.26s | production scan |
| 32^4 | RTX 3090 | 0.0125 | 40 | ~15.5s | production scan |
| 32^4 | Titan V (NVK) | 0.0125 | 40 | CRASH | device lost |

**Scaling analysis:** 8^4→16^4 = 16× volume, 2× MD steps = 32× work, actual
3.6× slower (0.35→1.26s) = 8.9× streaming efficiency gain. The streaming
HMC encoder batches all 20 MD dispatches into a single GPU submission, which
eliminates dispatch overhead that would otherwise dominate at small batch sizes.

8^4→32^4 = 256× volume, 4× MD steps = 1024× work, actual 44× slower
(0.35→15.5s) = 23× streaming efficiency gain on the RTX 3090.

### 2.4 Bottleneck Identification

1. **32^4 acceptance (20%)**: Too low for production statistics. The step size
   dt=0.0125 is adequate for 100% acceptance at β=6.0 but only 20% at β=4.0
   (strong coupling). Need β-adaptive step size or multi-scale integrators.

2. **NVK dispatch latency**: bench_multi_gpu showed NVK cooperative dispatch at
   0.95× (slower than single-GPU). The device-loss at 32^4 compounds this.
   The streaming pipeline is the key optimization: 200 dispatches per encoder
   vs 200 individual submissions.

3. **VRAM-limited scaling**: 48^4 dynamical needs 16.9 GB — Titan V (12 GB)
   cannot participate. Only the 3090 (24 GB) can run production dynamical QCD.

---

## 3. Open-Driver Evolution Strategy

### 3.1 Why Open Drivers

NVIDIA dropped Titan V from proprietary driver support in late 2025. The 500-series
drivers are the last to include Volta (GV100). Future proprietary releases will
not support GV100 at all. Meanwhile:

- NVK is **actively developed** in Mesa (commits weekly to `src/nouveau/`)
- NVK supports Vulkan 1.3 on Volta, Turing, Ampere, Ada, Blackwell
- NAK (the shader compiler) improves with every Mesa release
- The community is fixing fp64 issues (our NAK deficiency is known upstream)

For hotSpring's compute needs, proprietary driver lock-in is a strategic risk.
A 3070 or 3060 can run NVK today. Our physics code should work identically on
both driver stacks.

### 3.2 Target Hardware for NVK Development

| Card | VRAM | fp64 | Arch | NVK Status | Role |
|------|------|------|------|------------|------|
| Titan V | 12 GB | 1:2 native | Volta | Works ≤16^4, crashes ≥32^4 | Development/verification |
| RTX 3070 | 8 GB | 1:64 emulated | Ampere | Should work (same as 3090) | Budget NVK target |
| RTX 3060 | 12 GB | 1:64 emulated | Ampere | Should work | 12 GB budget NVK |
| RTX 3090 | 24 GB | 1:64 emulated | Ampere | Works via proprietary | Production (transition to NVK) |

**Recommended next purchase:** RTX 3060 12GB (~$200 used). Same VRAM as Titan V,
same Ampere arch as 3090, NVK support should be identical to 3090. This gives
us a second NVK-capable GPU for verification without the Volta-specific NAK bugs.

### 3.3 ToadStool Work Items (Prioritized)

#### P0: Investigate NVK device loss at 32^4

1. Add `VK_MESA_NVK_DEBUG=*` logging to isolate the failure point
2. Binary-search the lattice size: 20^4, 24^4, 28^4
3. Test splitting the streaming encoder (smaller command buffers)
4. Test with Mesa git HEAD NVK
5. File upstream Mesa bug if reproducible with a minimal test case

#### P1: Build Mesa NVK from git HEAD

Mesa 25.1.5 is from August 2025. NVK has had 6 months of fixes. Build from
`main` and re-test:
- 32^4 device loss
- NAK exp/log f64 assertion
- Dispatch performance vs 25.1.5

#### P2: Software-emulated GPU dispatch fallback

For cases where NVK hardware dispatch fails, implement a software fallback that:
- Runs the same WGSL shaders through naga→SPIR-V→LLVM→CPU (via wgpu's CPU backend)
- Accepts identical `GpuF64` API calls
- Useful for verification and debugging (CPU results = ground truth)
- Already partially available via `llvmpipe` (Mesa software Vulkan)

#### P2.5: Hybrid Core Streaming — DF64 on FP32 Cores (NEW)

**The discovery:** On consumer GPUs with 1:64 fp64 hardware, double-float (f32-pair)
arithmetic delivers **9.9× the f64-equivalent throughput** by streaming to the FP32
cores instead of waiting for the few dedicated FP64 units.

`bench_fp64_ratio` results on RTX 3090:

| Path | Throughput | Precision | Use For |
|------|-----------|-----------|---------|
| FP32 | 14.98 TFLOPS | 7 digits | N/A for physics |
| **DF64 (f32-pair)** | **3.24 TFLOPS** | **14 digits** | Bulk SU(3), force computation |
| FP64 native | 0.33 TFLOPS | 16 digits | Reductions, convergence, link updates |

On Titan V (1:2 native fp64): DF64 is 0.5× SLOWER than native f64. Use native f64 only.

**Architecture for toadStool `ShaderTemplate`:**

1. **GPU capability detection**: `GpuDriverProfile` already identifies Volta/Ampere/Ada.
   Add an `fp64_strategy` field: `Native` (Titan V, V100, A100) vs `Hybrid` (consumer).

2. **DF64 core library**: `df64_core.wgsl` (already prototyped in barracuda) provides
   `Df64` struct and `df64_add`, `df64_mul`, `df64_sub`, `df64_div` using Dekker
   splitting (no FMA required, pure f32 ALU operations).

3. **Hybrid gauge force kernel**: For consumer GPUs, generate:
   - `mul_su3_df64()` for the 7 staple matrix multiplications (95% of compute)
   - Native `f64` for the traceless anti-Hermitian projection + trace (5%)
   - `df64_to_f64()` conversion at the precision boundary

4. **Precision analysis per kernel**:
   | Kernel | % Compute | DF64-safe? | Notes |
   |--------|-----------|------------|-------|
   | `su3_gauge_force` | 40% of HMC | YES | 7 SU(3) muls, projection needs f64 |
   | `su3_momentum_update` | 5% | YES | Simple multiply-add |
   | `su3_link_update` (Cayley) | 10% | PARTIAL | Cayley exponential needs f64 |
   | `wilson_plaquette` | 15% | YES | 4 SU(3) muls per plaquette |
   | `su3_kinetic_energy` | 5% | YES | Trace + reduction needs f64 for sum |
   | `su3_random_momenta` | 5% | NO | RNG precision matters |
   | CG solver inner products | 20% | NO | Convergence-critical |

5. **Combined throughput estimate**: If 70% of HMC compute moves to DF64:
   - Effective throughput: 0.7 × 3.24 + 0.3 × 0.33 = **2.37 TFLOPS**
   - vs current all-f64: 0.33 TFLOPS
   - **7.2× HMC speedup on consumer GPUs** without losing physics accuracy

6. **Concurrent execution**: On Ampere, FP32 and FP64 units execute simultaneously
   within the same SM. A shader mixing DF64 and native f64 operations naturally
   utilizes both execution unit types — **true core streaming**.

**Implementation path:**
1. Validate DF64 SU(3) multiply precision (14 digits sufficient for gauge force)
2. Implement hybrid `su3_gauge_force_hybrid.wgsl`
3. Run `validate_gpu_beta_scan` to confirm physics parity
4. If parity holds, convert remaining kernels per the table above
5. Benchmark full HMC trajectory: hybrid vs pure-f64 on RTX 3090

#### P3: NAK exp/log f64 upstream fix

1. Isolate the minimal WGSL shader that triggers the assertion
2. File Mesa issue with reproducer (SPIR-V + NAK crash trace)
3. Propose fix in `nak/from_nir.rs` (128-bit split for transcendental returns)

#### P4: NVK streaming dispatch optimization

The streaming encoder (batching N dispatches into one command buffer) is our
key performance advantage. Investigate:
- Does NVK properly pipeline compute dispatches within a single encoder?
- Are there NVK-specific barriers or flushes we can remove?
- What is the maximum command buffer size before NVK falls over?

---

## 4. Run Status Summary (End of Day 2)

| Task | Status | Notes |
|------|--------|-------|
| NVK handoff doc | DONE | wateringHole/handoffs/BIOMEGATE_NVK_TITAN_V_SETUP_FEB23_2026.md |
| sarkas_gpu --paper | DONE | 9/9 paper parity on RTX 3090 |
| validate_all NVK | DONE | 38/39 PASS on Titan V (NVK exp crash is known) |
| nuclear_eos_l1_ref | DONE | Pareto sweep, λ=100 best |
| nuclear_eos_l2_gpu | DONE | Phase 1 SLy4 baseline, 32 min on RTX 3090 |
| 16^4 β-scan Titan V | DONE | 9 points, 47 min, physics validated |
| 32^4 β-scan RTX 3090 | COMPLETE | 12/12 points, 13.6 hrs, χ=40.1 at β=5.69 |
| 32^4 β-scan Titan V | FAILED | NVK device loss — needs investigation |
| 48^4 β-scan | BLOCKED | Waiting for 32^4 to finish on 3090 |
| Dynamical fermion | BLOCKED | Waiting for quenched scan completion |
| 2D dispatch refactor | DONE | 15 shaders + dispatch.rs, backward-compatible |
| Buffer size limits | DONE | Increased to 2 GB binding / 4 GB total |

### What to Examine

- 32^4 β-scan on 3090: **COMPLETE** (12/12 points, 13.6 hrs, $0.58). Peak χ=40.1 at β=5.69 matches β_c=5.692
- 30^4 β-scan on Titan V: **FAILED** (NVK PTE fault — documented above)
- Acceptance rate at strong coupling (β=4.0): only 20% — need adaptive step size
- 48^4 and dynamical runs deferred until quenched scan completes
- NVK 32^4 device loss: top priority for ToadStool investigation

---

## Addendum: FP64:FP32 Definitive Benchmark (Feb 24, 2026)

### Correction: The "1:2 on consumer cards" claim was wrong

The `bench_fp64_ratio` micro-benchmark (pure FMA chain, 4M threads × 4096 ops,
no memory bottleneck) provides definitive measurements:

| Path | fp32 TFLOPS | fp64 TFLOPS | fp64:fp32 |
|------|----------:|----------:|:----------|
| RTX 3090 — Vulkan/wgpu (nvidia proprietary) | 14.05 | 0.33 | **1:43** |
| RTX 3090 — CUDA (nvcc -O3) | 22.07 | 0.29 | **1:77** |
| Titan V — Vulkan/wgpu (NVK/NAK) | 0.33 | 0.25 | **1:1.3** |
| RTX 3090 hardware spec | 35.6 | 0.56 | 1:64 |
| Titan V hardware spec | 14.9 | 7.45 | 1:2 |

**Key findings:**

1. Consumer Ampere fp64:fp32 is hardware ~1:64 (164 FP64 units / 10496 FP32 cores).
   Both CUDA and Vulkan confirm this — the fp64 TFLOPS are nearly identical
   (0.29 vs 0.33). The ratio difference (1:77 vs 1:43) is CUDA's more aggressive
   FP32 loop unrolling, not Vulkan FP64 improvement.

2. Titan V (GV100) has genuine 1:2 hardware (2,560 dedicated FP64 cores), confirmed
   at 1:1.3. BUT NVK/NAK achieves only 3.4% of hardware peak (0.25 vs 7.45 TFLOPS).
   This is the **#1 NVK compiler optimization target** for toadStool.

3. The original "1:2" evidence was flawed: native builtins being 1.5-2.2× faster
   than `math_f64` software emulation only proves the software was slow, and the
   eigensolve comparison was confounded by NVK dispatch overhead.

### Implications for toadStool

- **NAK compiler optimization** is the highest-leverage improvement. The Titan V
  has 7.45 TFLOPS of fp64 hardware; NAK currently extracts 0.25 TFLOPS (3.4%).
  Even reaching 30% would give 2.2 TFLOPS — 7× better than the RTX 3090 via CUDA.
- **Consumer GPUs** provide exact fp64 for correctness verification but not for
  throughput-critical compute. The Titan V is the compute workhorse.
- **barracuda's advantages over CUDA** are NOT about fp64 ratio — they are about
  streaming pipeline (zero CPU↔GPU round-trips), zero external dependencies,
  cross-vendor compatibility, and open-driver access to compute-class hardware
  (Titan V via NVK) that CUDA can no longer use.
