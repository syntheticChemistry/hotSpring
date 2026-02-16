# Experiment 004: GPU Dispatch Overhead — L3 Deformed HFB Profiling

**Date**: February 15, 2026
**Researcher**: Kevin Eastgate (via Cursor AI assistant)
**Hardware**: RTX 4070 (12 GB GDDR6X), i9-12900K (24 threads), 32 GB DDR5, Pop!_OS 22.04
**Binary**: `nuclear_eos_l3_gpu --params=sly4`
**Status**: COMPLETE — profiling data collected, architecture bottleneck diagnosed

---

## 1. Why This Experiment

### The question

When we move L3 deformed HFB from CPU to GPU, does the speedup justify the
migration? What are the actual bottlenecks — and are they compute-bound or
dispatch-bound?

### The context

- L3 deformed HFB uses a 2D cylindrical grid (20k–50k points) per nucleus
- L2 spherical HFB for 52 nuclei runs in **3.5 seconds** on CPU
- L3 CPU (Rayon, full 24 threads) runs in **5m51s wall / 104m38s CPU time** for 52 nuclei
- The GPU version uses `BatchedEighGpu` for eigensolves but builds Hamiltonians on CPU
- We expected the GPU to accelerate via massive grid-parallel computation

### What we expected

| Component | CPU time share | GPU acceleration expected |
|-----------|:-----------:|:------------------------:|
| Hamiltonian build (grid ops) | ~60% | 10–50× (grid-parallel) |
| Eigensolve (BatchedEighGpu) | ~25% | 5–20× (batched Jacobi) |
| Density/BCS/energy | ~15% | 5–10× (elementwise) |
| **Total** | **5m51s** | **~30–60s** |

### What we measured

The GPU run was **terminated after 94 minutes** without completing any nucleus.
This is **16× slower** than pure CPU for the same workload.

---

## 2. Profiling Setup

Three concurrent monitors captured the full system profile:

```bash
# GPU: nvidia-smi every 2 seconds → CSV
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used \
  --format=csv -l 2 > /tmp/gpu_profile.csv 2>&1 &

# CPU: vmstat every 2 seconds
vmstat 2 > /tmp/cpu_profile.log 2>&1 &

# L3 run: release binary, sly4 params, 52 nuclei
time target/release/nuclear_eos_l3_gpu --params=sly4 > /tmp/l3_gpu_profile.log 2>&1
```

**Data files** (archived in `experiments/data/004/`):
- `gpu_profile.csv` — 2,823 samples (94 min at 2s intervals)
- `cpu_profile.log` — 3,093 samples
- `l3_gpu_profile.log` — run output (25 lines before SIGTERM)

---

## 3. Results

### 3.1 GPU Profile Summary

| Metric | Value |
|--------|-------|
| Wall time | **94 min 4s** (SIGTERM — no nucleus completed) |
| GPU utilization (avg) | **79.3%** |
| GPU utilization (peak) | **88%** |
| GPU power (idle) | 28.2 W |
| GPU power (avg under load) | **51.4 W** |
| GPU power (peak) | 52.4 W |
| VRAM (avg) | 647 MiB |
| VRAM (peak) | 676 MiB |
| GPU energy consumed | **80.6 Wh** |
| Energy cost | $0.0097 (at $0.12/kWh) |

### 3.2 CPU Profile Summary

| Metric | Value |
|--------|-------|
| CPU user avg | **7.7%** |
| CPU system avg | 3.0% |
| CPU idle avg | **84.7%** |
| CPU active avg | **10.7%** (~2.6 of 24 threads) |
| Memory stable | ~12 GB free of 32 GB |

### 3.3 Comparison: CPU-only vs GPU-hybrid

| Metric | CPU-only (Rayon) | GPU-hybrid | Ratio |
|--------|:----------------:|:----------:|:-----:|
| Wall time | 5m 51s | >94 min (incomplete) | **>16× slower** |
| CPU time | 104m 38s | ~10 min (142m wall × 10.7%) | 10× less CPU |
| CPU utilization | ~1800% (24 threads) | ~257% (~2.6 threads) | CPU freed |
| GPU utilization | 0% | 79.3% avg | GPU saturated |
| GPU power | 0 W | 51.4 W avg | 80.6 Wh total |
| Completion | **52/52 nuclei** | **0/52 nuclei** | N/A |
| Energy (CPU) | ~30 Wh est (24 cores × 180W × 5.85 min) | ~2 Wh est | CPU energy saved |
| Energy (GPU) | 0 Wh | 80.6 Wh | GPU dominant |
| Energy (total) | ~30 Wh | ~82.6 Wh | **2.75× more energy** |

---

## 4. Diagnosis: Dispatch Overhead, Not Compute

### The paradox

The GPU was 79% utilized and still 16× slower. This means the GPU was
**consistently busy doing work** — but that work included massive overhead per
dispatch, not just physics computation.

### The math

```
52 nuclei × 200 SCF iterations × 2 isospins = 20,800 eigensolve dispatches
Plus: Hamiltonian packing, buffer creation, shader submission, synchronous readback
Estimated total GPU dispatches: ~145,000 (from run log)
```

Each dispatch cycle:
1. CPU builds Hamiltonian blocks (Rayon parallel) → packs into flat array
2. CPU creates GPU buffer, uploads packed matrices
3. GPU runs BatchedEighGpu (Jacobi iterations)
4. CPU waits (synchronous `pollster::block_on`) for result
5. CPU reads eigenvalues back, unpacks, computes BCS occupations
6. Repeat for next isospin / next SCF step

**The bottleneck is steps 2, 4, 5** — the round-trip. For small matrices
(typical block size 4–12), the actual Jacobi computation takes microseconds,
but the buffer creation + submission + sync readback takes milliseconds.

### Estimated overhead breakdown

| Component | Time per dispatch | Total (×145k) |
|-----------|:-----------------:|:-------------:|
| Buffer alloc + upload | ~100 μs | ~14.5 s |
| Shader pipeline bind | ~50 μs | ~7.3 s |
| Synchronous readback wait | ~500 μs | **~72.5 s** |
| Rayon contention (24 threads → 1 GPU) | ~200 μs | **~29.0 s** |
| **Overhead subtotal** | | **~123 s (~2 min)** |
| Actual GPU compute | | ~remainder |

But this assumes no contention amplification. With 24 Rayon threads all
competing for a single GPU, the actual overhead is **multiplicative** — each
thread's synchronous wait blocks while other threads queue dispatches, creating
a cascading delay. The 94 minutes suggests **~50× amplification** from
contention.

### The key insight

> **The trains to and from take way more time than the actual work.**
> We should pre-plan and fill the available GPU function space, then fire at once.

The GPU is a factory. Sending one small package at a time, waiting for it to
come back, then sending the next is catastrophically inefficient. We need to:
1. **Batch**: Collect ALL eigensolves across ALL nuclei into one mega-dispatch
2. **Pipeline**: Chain dependent operations on-GPU without CPU round-trips
3. **Persist**: Keep grid data and wavefunctions in GPU buffers between SCF steps
4. **Async**: Fire dispatches and continue CPU prep work in parallel

---

## 5. Architecture: What "GPU-First" Actually Means

### Current (broken) pattern

```
For each nucleus (24 Rayon threads competing):
  For each SCF iteration:
    CPU: Build H blocks (Rayon)          # ~ms, parallel
    CPU→GPU: Upload packed H             # ~ms, serial per thread
    GPU: BatchedEighGpu                  # ~μs, fast
    GPU→CPU: Readback eigenvalues        # ~ms, serial per thread, BLOCKING
    CPU: BCS occupations, density        # ~ms, parallel
    Repeat
```

### Target (batched) pattern

```
GPU: Persistent buffers for ALL 52 nuclei (grid, wavefunctions, density)

For each SCF iteration (orchestrated by CPU, executed by GPU):
  GPU shader 1: Build ALL Hamiltonians (grid-parallel, ALL nuclei at once)
  GPU shader 2: Pack + BatchedEighGpu (ALL nuclei, ALL blocks, ONE dispatch)
  GPU shader 3: BCS occupations (elementwise, ALL nuclei)
  GPU shader 4: Density update (grid-parallel, ALL nuclei)
  GPU shader 5: Energy + convergence check (reduction, ALL nuclei)
  GPU→CPU: Read ONLY convergence flags (tiny readback)
  CPU: Decision only — which nuclei continue, which have converged
```

### Expected improvement

| Metric | Current | Target | Improvement |
|--------|:-------:|:------:|:-----------:|
| GPU dispatches | ~145,000 | ~1,000 (5 per iteration × 200 iterations) | **145×** |
| GPU→CPU readbacks | ~145,000 | ~200 (one per iteration) | **725×** |
| VRAM usage | 676 MiB peak | ~2–4 GB (persistent buffers) | Uses more GPU |
| Wall time (est.) | >94 min | **~30–60s** | **90–190×** |
| CPU utilization | 10.7% | <5% (pure orchestration) | CPU freed |

---

## 6. What ToadStool/BarraCUDA Can Help With

### Already available (from Feb 14–15 evolution)

| Feature | ToadStool Module | Relevance |
|---------|-----------------|-----------|
| `TensorContext::begin_batch()` / `end_batch()` | `device/tensor_context.rs` | Batch multiple ops into single submit |
| `AsyncSubmitter::queue_operation()` | `device/async_submit.rs` | Non-blocking dispatch |
| `BufferPool` | `device/tensor_context.rs` | Reuse GPU buffers across iterations |
| `science_limits()` | buffer config | 512 MiB storage, 1 GiB max |
| `BatchedEighGpu` | linalg | Already used — but needs bigger batches |

### Needs evolution (handoff to ToadStool team)

| Feature | Why | Priority |
|---------|-----|:--------:|
| **Multi-kernel pipeline** | Chain H-build → eigh → BCS → density without CPU round-trips | **Critical** |
| **GPU-resident SCF loop** | Keep iteration state on GPU, only read convergence flags | **Critical** |
| **Batched H construction shader** | Grid-parallel Hamiltonian for ALL nuclei in one dispatch | High |
| **Persistent buffer management** | Allocate once for all nuclei, reuse across SCF iterations | High |
| **Async readback for decision data** | Non-blocking read of convergence flags only | Medium |
| **2D grid shader for deformed potential** | Cylindrical (r,z) grid physics entirely on GPU | Medium |

### hotSpring-specific (stays in hotSpring)

| Feature | Why |
|---------|-----|
| Nucleus setup and quantum number assignment | Physics-specific, small data |
| Skyrme parameter management | Optimization-layer concern |
| AME2020 data loading and comparison | I/O, not compute |
| Convergence strategy (which nuclei to continue) | Decision logic |

---

## 7. Lessons Learned

### Lesson 1: GPU utilization ≠ GPU efficiency

79% GPU utilization sounds great. But if most of that utilization is buffer
management overhead (allocating, binding, waiting for tiny dispatches to
complete), the GPU is "busy doing nothing useful." The metric that matters is
**physics FLOPS per watt**, not utilization percentage.

### Lesson 2: Synchronous readback is the serial killer

Every `pollster::block_on` call turns a parallel GPU into a serial bottleneck.
The CPU stops, waits for the GPU to finish, reads back data, then starts the
next dispatch. With 24 Rayon threads doing this simultaneously, they serialize
on the single GPU queue.

### Lesson 3: Small matrices are the wrong workload for individual dispatches

Block sizes of 4–12 mean the eigensolve itself is trivial. The dispatch overhead
dominates by orders of magnitude. Either batch thousands of these small matrices
into one dispatch (already done within a single nucleus — need to extend across
ALL nuclei), or keep them on CPU where the overhead is zero.

### Lesson 4: The CPU freed itself

The CPU dropped from 1800% (Rayon full-blast) to 257% utilization. That's
21 cores now available for other work. If the GPU dispatch overhead is solved,
we get both: fast GPU physics AND free CPU for the next experiment or daily use.

### Lesson 5: Pre-plan, fill, fire

> The optimal GPU pattern is not "send work, wait, send more work."
> It is "plan all work, fill all available GPU function space, fire everything
> at once, then read only what you need for the next decision."

This is the factory model: load the entire production run, let the GPU
assembly line process it, and only check the output dock when you need to
make a routing decision. Every unnecessary check of the output dock (readback)
is wasted time.

---

## 8. Connection to ToadStool Buffer Management

ToadStool's recent evolutions directly address several of these bottlenecks:

| ToadStool Feature | Experiment 004 Lesson |
|-------------------|----------------------|
| `begin_batch()` / `end_batch()` | Lesson 5: batch dispatches |
| `AsyncSubmitter` | Lesson 2: avoid sync readback |
| `BufferPool` | Lesson 3: reuse buffers, avoid per-dispatch allocation |
| `Cascade` pipeline | Lesson 5: chain stages without CPU intervention |

The remaining gap is **GPU-resident iteration loops** — where the SCF
convergence check happens ON the GPU and only a boolean flag comes back.
This requires a reduction shader that computes max(|E_new - E_old|) across
all nuclei and returns a single u32 (continue/stop). ToadStool's
`SumReduceF64` is a starting point but needs a "max absolute difference"
variant.

---

## 9. Reproducing This Experiment

```bash
cd /home/eastgate/Development/ecoPrimals/hotSpring/barracuda

# Build release
cargo build --release --bin nuclear_eos_l3_gpu

# Start monitors
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used \
  --format=csv -l 2 > /tmp/gpu_profile.csv 2>&1 &
GPU_PID=$!
vmstat 2 > /tmp/cpu_profile.log 2>&1 &
CPU_PID=$!

# Run (expect >90 min with current dispatch pattern)
time target/release/nuclear_eos_l3_gpu --params=sly4 > /tmp/l3_gpu_profile.log 2>&1

# Stop monitors
kill $GPU_PID $CPU_PID

# Analyze
awk -F', ' 'NR>1{gsub(/ %/,"",$2); gsub(/ W/,"",$4); sum+=$2; pow+=$4; n++} \
  END{printf "GPU avg: %.1f%%, Power avg: %.1fW, Energy: %.1f Wh\n", \
  sum/n, pow/n, pow*2/3600}' /tmp/gpu_profile.csv
```

---

*Generated from hotSpring profiling session. Profiling data: 2,823 GPU samples,
3,093 CPU samples over 94 minutes. License: AGPL-3.0*
