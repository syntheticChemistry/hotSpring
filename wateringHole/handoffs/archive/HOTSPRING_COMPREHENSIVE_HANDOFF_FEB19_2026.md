# hotSpring → ToadStool/BarraCUDA: Comprehensive Evolution Handoff

**Date:** 2026-02-19
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Supersedes:** All previous hotSpring handoffs (consolidation)
**Previous handoffs:** 12 documents in `wateringHole/handoffs/` (retained as fossil record)

---

## Executive Summary

hotSpring has completed its primary mission: validating that consumer GPU hardware
can reproduce published computational physics across four domains — plasma MD,
nuclear structure, transport coefficients, and lattice gauge theory. The codebase
is at v0.5.14 with 281 passing unit tests (286 total, 5 GPU-ignored), 16/16
validation suites, zero clippy warnings, zero unsafe code, and zero doc warnings.

**v0.5.12 rewire**: `ReduceScalarPipeline` rewire — first complete feedback loop.
**v0.5.13**: GPU-resident cell-list (3-pass bin→scan→scatter) + indirect force shader.
  Discovered ToadStool `CellListGpu` prefix_sum binding mismatch; built corrected
  implementation locally. VACF particle-identity bug fixed by indirect indexing.
**v0.5.14**: Daligault D* model evolved with κ-dependent `C_w(κ)`, reducing
  crossover-regime errors from 44–63% to <10%. 12 Sarkas D_MKS provenance values
  stored; transport grid expanded to 20 (κ,Γ) points.

See `HOTSPRING_TOADSTOOL_ABSORPTION_HANDOFF_FEB19_2026.md` for v0.5.13+v0.5.14
details and absorption candidates.

This document consolidates everything the ToadStool/BarraCUDA team needs to
absorb hotSpring's work, evolve the primitives it uses, and close the remaining
gaps for full GPU sovereignty.

---

## 1. What hotSpring Proved

| Claim | Evidence | Files |
|-------|----------|-------|
| Consumer GPU produces correct f64 physics | 4.55e-13 MeV max error vs CPU | `bin/validate_cpu_gpu_parity.rs` |
| RTX 4070 sustains production MD | 9/9 cases, N=10k, 80k steps, 0.000% drift | `bin/sarkas_gpu.rs` |
| Titan V (NVK open-source) matches proprietary | Identical physics to 1e-15 | `experiments/006_GPU_FP64_COMPARISON.md` |
| 478× faster than Python at L1 nuclear EOS | 2.27 chi2 in 2.3s vs 6.62 in 184s | `bin/nuclear_eos_l1_ref.rs` |
| 44.8× less energy than Python | 126 J vs 5,648 J for 100k evals | `benchmarks/` |
| Full AME2020 (2,042 nuclei) on single GPU | L1 Pareto, L2 HFB, L3 deformed | `bin/nuclear_eos_l2_gpu.rs` |
| Lattice QCD validated on CPU | 12/12 SU(3) pure gauge checks | `bin/validate_pure_gauge.rs` |
| Transport coefficients validated | 13/13 Stanton-Murillo checks | `bin/validate_stanton_murillo.rs` |
| Unidirectional pipeline works for MD | 10,000× bandwidth reduction | `md/simulation.rs` |
| Python → Rust → GPU path generalizes | 4 physics domains validated | All validation binaries |

---

## 2. BarraCUDA Primitives Used — Complete Catalog

### 2.1 Primitives hotSpring Depends On

| BarraCUDA Module | hotSpring Usage | Satisfaction |
|------------------|-----------------|-------------|
| `barracuda::linalg::eigh_f64` | CPU symmetric eigendecomposition | Excellent |
| `barracuda::ops::linalg::BatchedEighGpu` | GPU-batched eigensolve (HFB, NAK) | Excellent |
| `barracuda::ops::grid::SpinOrbitGpu` | GPU spin-orbit correction in HFB | Excellent (v0.5.6) |
| `barracuda::ops::grid::compute_ls_factor` | Canonical l·s factor | Excellent |
| `barracuda::numerical::{trapz, gradient_1d}` | Radial integration, gradient | Good |
| `barracuda::optimize::{bisect, brent}` | Root-finding (BCS, NMP) | Excellent |
| `barracuda::optimize::{nelder_mead, bfgs}` | Multi-start optimization | Good |
| `barracuda::sample::{latin_hypercube, sobol_sequence, direct}` | Sampling strategies | Good |
| `barracuda::surrogate::RBFSurrogate` | RBF surrogate for optimization | Good |
| `barracuda::stats::{chi2_decomposed_weighted, bootstrap_ci}` | Statistical analysis | Good |
| `barracuda::special::{gamma, laguerre, hermite, bessel, erf, legendre, factorial}` | Special functions | Excellent |
| `barracuda::ops::md::{YukawaForceF64, VelocityVerletKickDrift, BerendsenThermostat, KineticEnergy}` | MD pipeline | Excellent |
| `barracuda::ops::ssf::SsfGpu` | Static structure factor | Good |
| `barracuda::ops::pppm::PppmGpu` | PPPM Coulomb (κ=0) | Good |
| `barracuda::device::{WgpuDevice, TensorContext}` | GPU device bridge | Excellent |
| `barracuda::ShaderTemplate::with_math_f64_auto()` | Driver-aware WGSL injection | Excellent (v0.5.8) |
| `GpuF64::create_pipeline_f64()` | Driver-aware shader compilation | Excellent (v0.5.11) |

### 2.2 Primitives hotSpring Needs But Doesn't Have

| Need | Why | Current Workaround | Status |
|------|-----|-------------------|--------|
| ~~`SumReduceF64::sum_from_buffer()`~~ | ~~GPU-buffer → scalar~~ | ~~Local WGSL copy~~ | ✅ **DONE (v0.5.12)** — `ReduceScalarPipeline` absorbed and rewired |
| ~~`StatefulPipeline`~~ | Iterative GPU-resident simulations | Manual encoder batching | ✅ Available in ToadStool; hotSpring deferred (architectural change) |
| ~~GPU-resident cell-list~~ | Eliminate CPU readback for neighbor rebuild | CPU-side sort every 20 steps | ✅ `CellListGpu` available in ToadStool; hotSpring not yet rewired |
| Complex f64 GPU primitive | Lattice QCD, FFT, quantum mechanics | `lattice/complex_f64.rs` (CPU) + WGSL template | ToadStool has f32 complex; f64 not yet |
| GPU-resident VACF | Velocity autocorrelation without snapshot readback | CPU post-process on snapshots | Keep velocity ring buffer on GPU |
| FFT (real + complex) | Full QCD, spectral analysis, PPPM improvements | None (blocking gap) | ToadStool has `fft_1d_f64`; hotSpring not yet integrated |

### 2.3 Zero Duplicate Math (Verified)

hotSpring delegates ALL mathematical operations to BarraCUDA primitives.
No duplications remain — the last one (`SHADER_SUM_REDUCE`) was eliminated
in v0.5.12 by rewiring to `barracuda::pipeline::ReduceScalarPipeline`.
   This is the #1 API gap to close.

2. **Inlined WGSL math in deformed shaders** — WGSL has no `#include` mechanism.
   Functions like `abs_f64` are duplicated in deformed_*.wgsl files. The
   `ShaderTemplate::with_math_f64_auto()` pattern handles this for new shaders
   but legacy deformed shaders predate it.

---

## 3. Evolution Lessons — What We Learned

### 3.1 The Complexity Boundary (Amdahl's Law Applied)

**Finding**: GPU eigensolve is ~1% of HFB SCF iteration time for 4×4 to 12×12
matrices. Moving ONLY the eigensolve to GPU makes it SLOWER (dispatch overhead).

| Matrix dim | GPU compute | Dispatch overhead | GPU wins? |
|:----------:|:----------:|:-----------------:|:---------:|
| 12×12 | ~8 ms | ~50 ms | No (14%) |
| 30×30 | ~125 ms | ~50 ms | Marginal |
| **50×50** | **~580 ms** | **~50 ms** | **Yes (92%)** |
| 100×100+ | >4.6 s | ~50 ms | Dominant |

**Lesson**: GPU profiling must track dispatch count and round-trip overhead, not
just utilization percentage. 79.3% GPU utilization with 145,000 synchronous
dispatches is a false positive for efficiency.

**The fix**: Move ALL physics to GPU (GPU-resident SCF loop). Eliminate CPU↔GPU
round-trips entirely. Upload initial state, run N iterations as GPU dispatches
within one compute pass, read back only convergence scalar.

### 3.2 The f64 Breakthrough

**Discovery**: The true fp64:fp32 throughput ratio on consumer GPUs via wgpu/Vulkan
is ~1:2, not CUDA's reported 1:64. The 1:64 penalty only applies to CUDA's native
fp64 units, which wgpu bypasses by routing through the shader ALU.

**Impact**: This makes EVERY consumer GPU a viable f64 compute node for science.
The performance ceiling is bandwidth, not compute — same as CPU at this precision.
Native WGSL builtins (`sqrt`, `exp`, `round`, `floor` on f64 types) gave 2-6×
throughput improvement over software-emulated f64.

### 3.3 NAK Shader Compiler Gap (NVK/nouveau)

Five specific deficiencies in the NAK (Mesa/nouveau) shader compiler for f64
loop-heavy kernels:

| NAK Deficiency | Impact | Known Solution |
|----------------|--------|----------------|
| No loop unrolling | ~4× of the 9× gap | LLVM, GCC both do this |
| Register spills to local memory | Latency | Register pressure analysis |
| Source-order instruction scheduling | No latency hiding | Instruction reordering pass |
| No f64 FMA fusion | Wasted throughput | FMA pattern matching |
| No shared memory bank conflict avoidance | Bandwidth waste | Access swizzling |

**Workarounds delivered**: `batched_eigh_nak_optimized_f64.wgsl` — manual 4×
unroll, hoisted locals, interleaved loads, explicit `fma()`, branchless `select()`.
Neutral on proprietary, 2-4× improvement on NVK.

**Universal benefit**: Every Spring benefits from NAK improvements. Loop unrolling
alone would close ~4× of the gap. NAK is written in Rust (Mesa) — we can contribute.

### 3.4 Cayley Transform > Taylor for Matrix Exponentials

In lattice QCD HMC, second-order Taylor approximation for SU(3) matrix exponential
caused 0% acceptance rate with no error message. The Cayley transform
`(I + X/2)(I - X/2)^{-1}` is exactly unitary for anti-Hermitian input.

**Lesson**: Unitarity preservation is binary — approximate methods accumulate
error that compounds catastrophically in long Markov chains. Any GPU implementation
of matrix exponentiation for group elements MUST use Cayley (or equivalent exact
method), not Taylor.

### 3.5 Sign Conventions in Lattice QCD

The gauge force `dP/dt = -(β/3) Proj_TA(U × V)` requires:
- V is the staple sum (NOT V†)
- Traceless anti-Hermitian projection uses `(M - M†)/2` (NOT `(M + M†)/2`)
- Getting ANY sign wrong causes 0% HMC acceptance with NO obvious error

**Lesson**: Lattice QCD implementations need comprehensive acceptance-rate
validation, not just plaquette checks. A plaquette can look reasonable while
the HMC dynamics are completely wrong.

### 3.6 LCG Determinism for Reproducibility

All lattice modules share a centralized Knuth MMIX LCG (multiplier
6364136223846793005, increment 1442695040888963407) via `lattice/constants.rs`.
This ensures bitwise determinism across runs, which is essential for:
- Debugging (reproduce exact failure)
- Validation (compare CPU vs GPU at specific configurations)
- Regression testing (detect numerical changes from code changes)

**Lesson**: Never use `rand` or system RNG for scientific simulations that need
reproducibility. A simple LCG with known good constants is sufficient and
perfectly portable.

### 3.7 Green-Kubo Transport is Fragile

Dimensional analysis in transport coefficient extraction must be perfect — any
missing factor shows up as orders-of-magnitude error, not a subtle drift. The
VACF → D* path works; stress ACF → η* and heat ACF → λ* required Sarkas
recalibration because the original Daligault coefficients assumed a different
reduced-unit normalization than our Python baseline.

**Lesson**: Always calibrate analytical fits against actual simulation data,
not just published tables. Unit conventions vary between groups and papers.

---

## 4. Design Feedback for BarraCUDA/ToadStool

### 4.1 StatefulPipeline Proposal

For iterative GPU-resident simulations, the unidirectional streaming pattern
doesn't fit. The data uploads once and stays GPU-resident. hotSpring's MD and
HFB loops both follow this pattern:

```
1. Upload initial state → GPU buffers
2. Stream N iterations as dispatches within one compute pass
3. Dispatch reduction (sum, max, norm) within same pass
4. Copy scalar result to staging
5. Read back minimal bytes (convergence flag)
6. Repeat
```

**Proposed API**:

```rust
pub struct StatefulPipeline {
    device: Arc<WgpuDevice>,
    resident_buffers: Vec<wgpu::Buffer>,
    convergence_staging: wgpu::Buffer,
    config: StatefulConfig,
}

impl StatefulPipeline {
    pub fn run_iterations(
        &self,
        chain: &[(&ComputePipeline, &BindGroup, u32)],
        iterations: usize,
    ) -> Result<Vec<f64>>;
}
```

### 4.2 SumReduceF64 Buffer-to-Buffer

The most impactful single API change: let `SumReduceF64` accept a GPU buffer
instead of `&[f64]`. This pattern appears in every physics workload:

- **MD**: Total KE, total PE → temperature, energy conservation
- **HFB**: Binding energy convergence (norm of density change)
- **PPPM**: Ewald sum total energy, net force magnitude
- **Eigensolve**: Residual norm for convergence
- **Lattice QCD**: Plaquette average, action, Dirac residual

A helper like `pipeline.reduce_sum(buffer, n) -> wgpu::Buffer` that returns a
buffer containing the scalar would eliminate 12+ lines of boilerplate per use
and is the only remaining duplicated shader in hotSpring.

### 4.3 Complex f64 as First-Class Primitive

`lattice/complex_f64.rs` defines `Complex64` with full arithmetic, conjugate,
inverse, exponential, polar form, and magnitude. The `WGSL_COMPLEX64` string
constant contains equivalent WGSL functions.

**Recommendation**: Absorb into `barracuda::numerical` or `barracuda::linalg`.
Complex f64 is needed by: FFT, quantum mechanics, signal processing, lattice
QCD, transfer matrices, spectral methods. Every Spring doing physics will
eventually need this.

### 4.4 GPU-Resident Cell-List Construction

The cell-list rebuild currently requires CPU readback every 20 steps:
1. Read back all positions from GPU (N×24 bytes)
2. CPU sorts particles into cells
3. Re-upload sorted positions + cell metadata

**Recommendation**: GPU-resident construction via:
1. Atomic cell assignment: one thread per particle, `atomicAdd` to cell count
2. Prefix sum: parallel scan to compute `cell_start` offsets
3. Scatter: each particle writes to its cell position

barracuda already has `prefix_sum.wgsl`. Combined with an atomic scatter shader,
the entire cell-list rebuild stays on GPU. This would make MD simulations 100%
unidirectional during production.

---

## 5. Lattice QCD GPU Promotion Roadmap

### Phase 1: SU(3) on GPU (WGSL templates exist)

1. Compile `WGSL_COMPLEX64` and `WGSL_SU3` via `ShaderTemplate`
2. Create plaquette computation shader (input: link buffer, output: average plaquette)
3. Validate GPU plaquette vs CPU plaquette on cold and hot lattices
4. **Data layout**: SoA for coalesced access — each of the 18 f64 components
   of a link matrix in a separate buffer

**Memory requirements**:

| Lattice | Sites | Links | Memory | Fits in |
|---------|-------|-------|--------|---------|
| 4^4 | 256 | 1,024 | 147 KB | L1 cache |
| 8^4 | 4,096 | 16,384 | 2.4 MB | L2 cache |
| 16^4 | 65,536 | 262,144 | 37.7 MB | VRAM easily |
| 32^4 | 1,048,576 | 4,194,304 | 603 MB | 12 GB VRAM |

### Phase 2: HMC on GPU (new shaders needed)

1. Port Cayley exponential to WGSL (3×3 inverse via cofactor — exact, no iteration)
2. Port leapfrog link update (`exp(εP) × U` per link)
3. Keep Metropolis accept/reject on CPU (single random number per trajectory)
4. Validate GPU HMC acceptance rate matches CPU (96-100% at β=5.5-6.0)

### Phase 3: Dirac CG on GPU (dominant cost in full QCD)

1. Port staggered Dirac operator to WGSL (sparse matrix-vector product)
2. Port CG iteration to GPU (dot products via SumReduceF64)
3. Memory layout: SoA for coalesced access across lattice sites
4. This is the single highest-impact GPU kernel for lattice QCD

### Phase 4: Multi-GPU Temperature Scan

1. Each GPU runs independent HMC at different β (temperature)
2. Embarrassingly parallel — no inter-GPU communication needed
3. Natural fit for basement HPC mesh (9 GPUs, 10G backbone)

---

## 6. Codebase Health at Handoff

| Metric | Value |
|--------|-------|
| Unit tests | **283** pass (5 ignored, GPU-required) |
| Validation suites | **16/16** pass |
| Clippy (default + pedantic) | **0** warnings |
| Doc warnings | **0** |
| Unsafe blocks | **0** |
| TODO/FIXME/HACK markers | **0** |
| Centralized tolerances | **58** constants in `tolerances.rs` |
| Centralized provenance | All validation targets traced (Python origins, DOIs) |
| AGPL-3.0 compliance | All 69 `.rs` files, 28 `.wgsl` shaders |
| Test coverage | ~48% line / ~65% function (GPU modules require hardware) |
| Active `.rs` files | 69 (22 library modules + 30 binaries + 17 lattice/physics) |
| Active `.wgsl` shaders | 28 (10 physics + 8 MD production + 4 MD reference + 6 diagnostic) |
| Total library code | ~15,000 lines |
| Total binary code | ~12,000 lines |

### Files Over 1000 Lines (8 files, all justified)

All are physics-coherent modules where splitting would create artificial
boundaries that harm understanding. Down from 9 after `md/simulation.rs`
dropped below 1000 lines via `ReduceScalarPipeline` rewire (v0.5.12).

| File | Lines | Why |
|------|-------|-----|
| `physics/hfb_gpu_resident.rs` | 1,701 | GPU pipeline with 14 buffers, 3 compute stages |
| `physics/hfb.rs` | 1,405 | Core spherical HFB solver |
| `physics/hfb_deformed.rs` | 1,290 | Deformed HFB (Nilsson basis, 2D grid) |
| `physics/hfb_deformed_gpu.rs` | 1,227 | GPU deformed HFB |
| `bin/nuclear_eos_l1_ref.rs` | 1,250 | L1 reference pipeline |
| `bin/nuclear_eos_l2_hetero.rs` | 1,207 | Heterogeneous L2 cascade |
| `bin/celllist_diag.rs` | 1,158 | Cell-list diagnostic (6-phase) |
| `bin/nuclear_eos_gpu.rs` | 1,039 | GPU nuclear EOS pipeline |

---

## 7. What to Absorb Immediately

### Priority 1: Buffer-to-Buffer SumReduce — ✅ DONE

ToadStool shipped `ReduceScalarPipeline` (Feb 19 2026). hotSpring rewired
to it in v0.5.12 — local `SHADER_SUM_REDUCE` removed, zero regressions.
**First complete feedback → absorption → rewire loop.**

### Priority 2: Complex f64 Primitive

Absorb `lattice/complex_f64.rs` (316 lines) + `WGSL_COMPLEX64` into barracuda.
This unblocks FFT, lattice QCD GPU, and any future Spring doing quantum physics.

### Priority 3: NAK-Optimized Eigensolve Shader

`batched_eigh_nak_optimized_f64.wgsl` is a drop-in replacement for the baseline
eigensolve shader with 5 NAK workarounds. Same API, same bind group layout. On
proprietary: neutral. On NVK: 2-4× faster. Ship alongside the baseline for
automatic driver detection.

### Priority 4: Cayley Matrix Exponential

The Cayley transform pattern from `lattice/hmc.rs` is reusable for any unitary
group element update. Consider adding to `barracuda::linalg` for general N×N
anti-Hermitian → unitary mapping.

---

## 8. What NOT to Change

1. **Do not split the >1000-line physics files.** They are coherent units where
   every function shares state with its neighbors. Splitting creates import
   tangles and hides the data flow.

2. **Do not replace the LCG with `rand`.** Bitwise determinism across platforms
   is more valuable than statistical quality for validation workloads. The Knuth
   MMIX LCG is centralized in `lattice/constants.rs` and tested.

3. **Do not generalize the MD engine.** The Yukawa OCP specialization (hardcoded
   force law, reduced units, Berendsen thermostat) is intentional — it matches
   the exact paper configuration. Generalization loses validation coverage.

4. **Do not refactor `hfb_gpu_resident.rs` without domain expert review.** The
   14-buffer pipeline with interleaved CPU/GPU stages was debugged through four
   1,000×-improvement cycles. The buffer layout encodes hard-won knowledge about
   GPU staging, density mixing order, and BCS bisection boundaries.

---

## 9. Remaining Infrastructure Tasks

These are NOT code tasks — they are environment/tooling tasks:

| Task | Why | Effort |
|------|-----|--------|
| GPU-in-CI for test coverage | 48% → 90% requires running GPU tests | Infrastructure |
| Python baseline rerun | Verify Sarkas calibration at N=2000 | Needs conda + Sarkas install |
| `cargo-llvm-cov` in CI | Track coverage regression | CI config |
| Titan V proprietary driver test | Unlock 6.9 TFLOPS fp64, fair comparison | Driver install |
| NAK loop unrolling patch | Close 4× of the 9× NVK gap | Mesa contribution |

---

## 10. Summary for Quick Reference

**hotSpring is done as a validation Spring.** The platform is proven: consumer GPU
reproduces published physics across plasma MD, nuclear structure, transport, and
lattice QCD. 283 tests, 16/16 suites, zero warnings, zero unsafe, zero debt.

**What BarraCUDA needs to evolve:**

1. Buffer-to-buffer `SumReduceF64` (small, high impact)
2. `StatefulPipeline` for iterative GPU-resident workloads (medium, architectural)
3. Complex f64 as first-class primitive (small, enables FFT/QCD)
4. GPU-resident cell-list construction (medium, eliminates last CPU readback)
5. FFT (large, enables full QCD and spectral analysis)

**What hotSpring hands off:**

- 69 validated `.rs` modules + 28 `.wgsl` shaders
- Complete physics documentation (`PHYSICS.md`, 12 sections, 24 references)
- 7 experiment journals documenting discoveries and debugging
- 12 previous handoff documents (fossil record of evolution)
- Lattice QCD infrastructure ready for GPU promotion
- NAK-optimized eigensolve shader (drop-in)
- Calibrated transport coefficient fits (Sarkas-validated)

---

*License: AGPL-3.0-only. All discoveries, code, and documentation are
sovereign community property. No proprietary dependency required for any
physics result produced by this codebase.*
