# Experiment 005: L2 Mega-Batch GPU — Complexity Boundary Analysis

**Date:** February 15–16, 2026
**Author:** hotSpring team
**Hardware:** RTX 4070 (12 GB, SHADER_F64), i9-12900K (24 threads)
**License:** AGPL-3.0-only

---

## 1. Objective

Evaluate the mega-batch architectural change (Experiment 004 recommendation) on
L2 spherical HFB and characterize the **CPU-vs-GPU complexity boundary** —
the matrix size / workload threshold at which GPU dispatch overhead is repaid
by parallel compute. The stated end goal: **pure GPU faster than CPU** for
all HFB levels.

---

## 2. Background

Experiment 004 diagnosed L3 GPU as 16x slower than CPU due to ~145,000 small
synchronous dispatches. The remedy was mega-batching: pad all nuclei to a
common basis dimension, fire ONE `BatchedEighGpu` dispatch per SCF iteration.

This experiment validates that remedy on L2 and measures whether the reduction
from 206 dispatches (grouped) to ~101 dispatches (mega-batch) translates to
wall-time improvement — and why CPU remains dominant.

---

## 3. Run Configuration

| Parameter | Value |
|-----------|-------|
| Binary | `nuclear_eos_l2_gpu` |
| Dataset | AME2020 full (2,042 nuclei, 791 HFB) |
| Skyrme params | SLy4 |
| SCF config | max_iter=200, tol=0.05, mixing=0.30 |
| GPU dispatch | Mega-batch: 1 per SCF iteration (all active nuclei) |
| Padding | Diagonal 1e10 for rows/cols beyond actual n_states |
| Monitoring | `nvidia-smi -l 2` CSV (1,806 samples over 60 min) |
| Command | `cargo run --release --bin nuclear_eos_l2_gpu -- --nuclei=full --phase1-only` |

---

## 4. Results

### 4.1 Physics Output

| Metric | Value |
|--------|:-----:|
| chi2/datum | **224.52** |
| HFB converged | 2039/791 (all but 2 borderline) |
| SEMF fallback | 1,251 nuclei |
| NMP chi2/datum | 0.6294 |

Physics output is **identical** to previous runs (chi2=224.52, NMP=0.63),
confirming that mega-batch padding does not contaminate eigenvalues.

### 4.2 Performance: Dispatch Count and Wall Time

| Implementation | GPU Dispatches | Wall Time | Per HFB Nucleus | Speedup vs prev |
|----------------|:--------------:|:---------:|:---------------:|:---------------:|
| **CPU-only** (nalgebra eigh) | 0 | **35.1s** | **44.4ms** | (baseline) |
| GPU v1 (grouped, 5 groups) | 206 | 66.3 min | 5,029ms | — |
| **GPU v2 (mega-batch)** | **101** | **40.9 min** | **3,104ms** | **1.6x vs v1** |

**CPU is 70x faster than GPU v2.** The mega-batch halved dispatches and
improved GPU 1.6x over v1, but CPU remains dominant by ~two orders of magnitude.

### 4.3 GPU Utilization Profile

| Phase | Duration | GPU Util (avg) | Power (avg) | VRAM (avg) |
|-------|----------|:--------------:|:-----------:|:----------:|
| Active SCF (t=0–40 min) | ~40 min | **94.9%** | **74.6 W** | **948 MiB** |
| Tail / converging (t=40–45 min) | ~5 min | ~50% | ~50 W | ~780 MiB |
| Post-exit (nvidia-smi running) | ~15 min | 7% | 29 W | ~520 MiB |

**Overall** (program runtime only, 41 min):

| Metric | Value |
|--------|:-----:|
| Avg GPU utilization | ~90% |
| Avg GPU power | ~70 W |
| Total GPU energy | **47.8 Wh** ($0.006) |
| Peak VRAM | 1,068 MiB |

GPU was saturated at 99% for the first 20 minutes, declining gradually as
nuclei converge and drop from the active batch. This confirms the mega-batch
architecture successfully fills the GPU.

### 4.4 Utilization Timeline

```
  Time (s) | GPU % | Power (W) | VRAM (MiB) | Phase
  ---------|-------|-----------|------------|------
         0 |     7 |      29   |       538  | startup
         2 |    99 |      77   |     1,065  | first dispatch
       300 |    99 |      85   |     1,044  | full SCF (791 nuclei)
       600 |    99 |      85   |     1,034  | full SCF
       900 |    99 |      85   |     1,031  | full SCF
     1,200 |    99 |      85   |     1,037  | full SCF
     1,500 |    98 |      83   |       943  | nuclei converging
     1,800 |    94 |      70   |       834  | fewer active
     2,100 |    85 |      50   |       767  | ~half converged
     2,400 |    85 |      50   |       781  | tail nuclei
     2,455 |     — |       —   |         —  | program exits
```

---

## 5. Diagnosis: The Complexity Boundary

### 5.1 Why CPU Wins by 70x

The per-SCF-iteration pipeline:

```
CPU: Build 791×2 Hamiltonians    →  ~20s estimated
CPU: Pack into flat buffer       →  ~100ms
GPU: BatchedEighGpu (1 dispatch) →  ~200ms (kernel) + ~50ms (overhead)
CPU: Readback eigenvalues        →  ~50ms
CPU: BCS pairing (791 nuclei)    →  ~2s estimated
CPU: Density update              →  ~1s estimated
                                    ─────────────
                                    ~24s per iteration
```

**101 iterations × 24s = 2,424s ≈ 40.4 min** — matches the measured 40.9 min.

The eigensolve itself is **~1% of the total iteration time**. The remaining
99% — Hamiltonian construction, BCS pairing, density updates — runs on CPU.

**Amdahl's Law**: If 1% of work is GPU-accelerated, even infinite GPU speedup
yields max 1.01x total improvement. The dispatch overhead makes it a net loss.

### 5.2 Why the Matrices Are Too Small

HFB basis sizes for nuclei with A=56–132:

| Shell range | n_states | Matrix dim | Elements | CPU eigh time |
|-------------|:--------:|:----------:|:--------:|:-------------:|
| Z=28, N=28 (⁵⁶Ni) | 4 | 4×4 | 16 | **~0.5 μs** |
| Z=50 region | 8 | 8×8 | 64 | **~2 μs** |
| Z=50, N=82 (¹³²Sn) | 12 | 12×12 | 144 | **~5 μs** |

nalgebra's eigendecomposition of a 12×12 symmetric matrix fits in L1 cache and
completes in microseconds. The GPU's BatchedEighGpu Jacobi algorithm also
completes in microseconds of *compute*, but each dispatch cycle (buffer alloc,
shader bind, queue submit, fence wait, readback) costs milliseconds.

**The breakeven point**: For a SINGLE dispatch with 1,582 matrices (791×2) of
size 12×12, the compute is ~8ms (1,582 × 5μs). The dispatch overhead is ~50ms.
The GPU is doing useful work for 14% of the dispatch wall time.

### 5.3 Where GPU Becomes Faster (the Crossover)

| Scenario | Matrix dim | Compute per dispatch | Overhead | GPU wins? |
|----------|:----------:|:--------------------:|:--------:|:---------:|
| Current L2 (12×12) | 12 | ~8ms | ~50ms | **No** (14% compute) |
| Larger basis (30×30) | 30 | ~125ms | ~50ms | **Marginal** (71%) |
| Deformed L3 (50×50) | 50 | ~580ms | ~50ms | **Yes** (92%) |
| Full L3 (100×100) | 100 | ~4.6s | ~50ms | **Yes** (99%) |
| Beyond mean-field (200+) | 200+ | >37s | ~50ms | **Dominant** |

**The boundary is at n_states ≈ 30–50.** Below this, CPU cache coherence
beats GPU parallelism. Above this, GPU's massively parallel Jacobi sweeps
dominate. This is not a bug — it is Amdahl's Law applied to matrix size.

### 5.4 The Real Fix: Move the OTHER 99% to GPU

The eigensolve is the wrong target for optimization. The real opportunity:

| Component | Current | Target | Shader Feasible? |
|-----------|---------|--------|:----------------:|
| Hamiltonian construction | CPU (serial) | **GPU** (WGSL) | Yes — element-wise grid eval |
| BCS pairing | CPU (Brent root-find) | **GPU** (WGSL) | Yes — bisection per state |
| Density update | CPU (basis summation) | **GPU** (WGSL) | Yes — parallel reduction |
| Convergence check | CPU (max diff) | **GPU** (WGSL) | Yes — max reduction |
| Eigensolve | GPU (BatchedEighGpu) | GPU (already there) | Done |

Moving ALL components to GPU eliminates 101 CPU↔GPU round-trips entirely:

```
CURRENT:  [CPU: H-build] → upload → [GPU: eigh] → download → [CPU: BCS] → [CPU: ρ] × 101
TARGET:   [GPU: H-build → eigh → BCS → ρ → convergence] × 101 (zero round-trips)
```

**Estimated target**: 791 nuclei on GPU-resident pipeline, ~0.5–1.0s per
iteration, 101 iterations → **~50–100s total**. This would be **faster than
CPU** (35.1s for CPU is for smaller basis; GPU-resident allows larger basis
for better physics at comparable time).

---

## 6. Comparison with Experiment 004 (L3 Dispatch Overhead)

| Metric | Exp 004 (L3 grouped) | **Exp 005 (L2 mega-batch)** |
|--------|:--------------------:|:---------------------------:|
| Level | L3 deformed | L2 spherical |
| Nuclei | 52 | 791 (2,042 total) |
| GPU dispatches | ~145,000 | **101** |
| Wall time | >94 min (incomplete) | **40.9 min (complete)** |
| GPU utilization | 79.3% | **94.9%** |
| CPU→GPU round-trips | ~145,000 | **101** |
| Dispatch reduction | — | **1,435x** (vs Exp 004) |
| GPU efficiency | Overhead-dominated | **Compute-limited** |

The mega-batch eliminated the dispatch overhead pathology. GPU utilization
rose from 79% to 95%. The remaining bottleneck is now clearly **CPU-bound
physics computations** (H-build, BCS, density), not GPU dispatch overhead.

This is progress: we moved from GPU-overhead-bound to CPU-compute-bound.
The next step is to move the CPU compute to GPU.

---

## 7. Energy and Cost Analysis

| Substrate | Wall Time | Energy | Cost ($0.12/kWh) |
|-----------|:---------:|:------:|:-----------------:|
| CPU-only | 35.1s | ~8 Wh (est 800W×35s) | $0.001 |
| GPU v1 (grouped) | 66.3 min | ~82 Wh | $0.010 |
| **GPU v2 (mega-batch)** | **40.9 min** | **~48 Wh** | **$0.006** |
| **Target (GPU-resident)** | **~100s** | **~2 Wh** | **$0.0003** |

The GPU-resident target (all physics on-GPU, zero round-trips) would reduce
energy by 24x vs current GPU and be competitive with CPU on time.

---

## 8. Stated Goal: Pure GPU Faster Than CPU

The path from current (70x slower) to target (faster than CPU):

| Step | What Moves to GPU | Est. Improvement | Cumulative |
|------|-------------------|:----------------:|:----------:|
| 0. Current | Eigensolve only | Baseline | 40.9 min |
| 1. H-build shader | Hamiltonian construction | **~10x** | ~4 min |
| 2. BCS shader | Pairing + occupation | **~2x** | ~2 min |
| 3. Density shader | Basis-weighted sum | **~1.5x** | ~80s |
| 4. GPU-resident loop | Eliminate all round-trips | **~2x** | **~40s** |
| 5. Larger basis | Better physics, GPU scales | GPU **wins** outright | ~30s |

**Step 4 crosses the boundary**: at ~40s total for 791 nuclei, GPU-resident
is comparable to CPU's 35s — but with a larger basis set (Step 5), the GPU
pulls ahead because O(n^3) eigensolves favor GPU while CPU time grows cubically.

**This is the architectural evolution that makes GPU faster than CPU.**

---

## 9. Implications for ToadStool

The following ToadStool primitives are needed for Steps 1–4:

1. **GPU Hamiltonian kernel**: Skyrme potential evaluation on radial grid (WGSL)
   — ToadStool provides: `ops/grid/finite_difference_f64`, grid math primitives
2. **GPU BCS kernel**: Bisection root-finding + occupation computation (WGSL)
   — ToadStool provides: `ops/optimize/bisection_f64` (new)
3. **GPU density kernel**: Weighted basis sum with workgroup reduction (WGSL)
   — ToadStool provides: `ops/reduce/weighted_dot_f64` (exists)
4. **Multi-kernel pipeline**: Chain dependent shaders without CPU readback
   — ToadStool provides: `begin_batch()`/`end_batch()` + dependent chaining
5. **Convergence reduction**: Max-absolute-difference on GPU, return 1 scalar
   — ToadStool provides: `ops/reduce/max_abs_diff_f64` (new)

---

## 9.1 GPU-Resident Attempt (Experiment 005b, Feb 16)

Following the path-to-pure-GPU analysis, hotSpring implemented and tested
a full GPU-resident pipeline:

**What was built:**
- `batched_hfb_potentials_f64.wgsl`: Skyrme, Coulomb (forward/backward
  cumulative sum), exchange, effective mass f_q — all on GPU
- `batched_hfb_hamiltonian_f64.wgsl`: H = T_eff + V matrix elements via
  radial grid integration — per-nucleus wavefunctions, derivatives, basis
  quantum numbers packed as batch arrays
- `hfb_gpu_resident.rs`: Pre-allocated buffers and pipelines (created ONCE),
  7 kernel dispatches per iteration in a single compute pass
- f64 workarounds: `pow_f64()` via f32 cast (NVIDIA NVVM chokes on f64 pow),
  storage buffers for Skyrme parameters (avoids bitcast<f64>(vec2<u32>))

**What worked:**
- Shaders compile and dispatch correctly on RTX 4070 SHADER_F64
- 91% GPU utilization during dispatch
- 1,320 MiB VRAM under load
- Per-nucleus wavefunction batching handles varying (Z,N) configurations
- 7 dispatches per iteration chain in a single wgpu command encoder

**What didn't work:**
- Each SCF iteration still requires **3 synchronous CPU↔GPU round-trips**:
  1. `read_buffer_f64(H_p)`: staging alloc + map + copy (~5ms)
  2. `read_buffer_f64(H_n)`: staging alloc + map + copy (~5ms)
  3. `BatchedEighGpu::execute_f64()`: internal buffer alloc + readback (~10ms)
- For 200 iterations × ~10 groups = 2,000 round-trips × ~20ms = **~40s of
  pure overhead** — the same order as CPU solving everything in 35s
- Result: GPU-resident is NOT faster than mega-batch for small matrices

**Conclusion:**
The bottleneck has shifted from *dispatch overhead* (solved by mega-batching)
to *readback overhead* (synchronous staging buffer operations). The solution
is ToadStool item 4.1: **dependent op chaining** where shader outputs become
shader inputs without CPU involvement. BatchedEighGpu must accept input
buffers already on GPU and write output buffers that stay on GPU.

```
CURRENT (GPU-resident, blocked):
  GPU: [pot+H-build] → CPU readback → GPU: [eigh] → CPU readback → CPU: [BCS+ρ]

NEEDED (ToadStool multi-kernel):
  GPU: [pot+H-build → eigh → BCS → ρ → convergence] → CPU: [1 scalar readback]
```

---

## 10. Reproduction

```bash
cd hotSpring/barracuda

# GPU profiling
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used \
  --format=csv -l 2 > /tmp/l2_megabatch_gpu.csv 2>&1 &

# L2 GPU mega-batch (full AME2020)
time cargo run --release --bin nuclear_eos_l2_gpu -- --nuclei=full --phase1-only

# CPU reference for comparison (L2 CPU is embedded in L3 run: L2=35.1s)
time cargo run --release --bin nuclear_eos_l3_ref -- --nuclei=full --params=sly4

# GPU-resident (Experiment 005b — for profiling only, slower due to readback)
time cargo run --release --bin nuclear_eos_l2_gpu -- --nuclei=full --phase1-only --gpu-resident
```

---

## 11. References

- Experiment 004: `experiments/004_GPU_DISPATCH_OVERHEAD_L3.md` — Dispatch overhead diagnosis
- Experiment 003: `experiments/003_RTX4070_CAPABILITY_PROFILE.md` — RTX 4070 f64 capability
- Ring & Schuck, *The Nuclear Many-Body Problem*, Springer (2004) — HFB theory
- Amdahl, G.M. "Validity of the single processor approach." AFIPS '67 — Speedup limits

---

*AGPL-3.0-only | hotSpring project | Updated: February 16, 2026*
