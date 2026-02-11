# Benchmark Protocol

**Purpose**: Ensure reproducible, comparable results across gates.

---

## General Rules

1. **Same input, same parameters, same code** - only the hardware changes
2. **3 runs minimum** - report median wall time (avoids JIT and caching outliers)
3. **Record everything**: wall time, time per step, peak memory, CPU threads, GPU utilization
4. **First run is warmup** - Numba JIT compiles on first call. Record it separately.
5. **Pin Numba threads** to match core count: `NUMBA_NUM_THREADS=N`

---

## Sarkas CPU Benchmarks

### Test Matrix

| Variable | Values |
|----------|--------|
| **Gate** | Eastgate (dev), Strandgate (64c EPYC), Northgate (i9-14900K), Southgate (5800X3D) |
| **Particles** | 1,000 / 10,000 / 100,000 |
| **Timesteps** | 1,000 (fixed) |
| **Potential** | Yukawa (κ=2, Γ=10) |
| **Threads** | Max available per gate |

### Input File

Use a single Sarkas YAML input file. Only change `num_particles`. Everything else identical.

Save as `hotSpring/benchmarks/sarkas-cpu/yukawa_benchmark.yaml`.

### Recording Template

```
Gate:           [name]
CPU:            [model]
Cores/Threads:  [N]
RAM:            [GB]
NUMBA_NUM_THREADS: [N]
Python:         [version]
Sarkas:         [version]
Numba:          [version]

Particles: 1000
  Run 1: [time]s (warmup/JIT)
  Run 2: [time]s
  Run 3: [time]s
  Run 4: [time]s
  Median (2-4): [time]s
  Peak memory: [MB]

Particles: 10000
  ...

Particles: 100000
  ...
```

Save as `hotSpring/benchmarks/sarkas-cpu/[gate]-results.md`.

### Profiling

On one gate (Eastgate is fine), run cProfile on the 10K particle case:

```python
import cProfile
import pstats

# Run simulation with profiling
cProfile.run('run_sarkas_simulation()', 'sarkas_profile.prof')

# Analyze
stats = pstats.Stats('sarkas_profile.prof')
stats.sort_stats('cumulative')
stats.print_stats(30)
```

Save profile to `hotSpring/benchmarks/sarkas-cpu/profile/`.

Key question: what fraction of time is:
- Force calculation (pairwise + PPPM)?
- FFT (pyfftw)?
- Neighbor list construction?
- Integration (Velocity-Verlet)?
- I/O and diagnostics?

This directly informs which BarraCUDA shaders are highest priority.

---

## Surrogate Learning GPU Benchmarks

### Test Matrix

| Variable | Values |
|----------|--------|
| **Gate** | Eastgate (RTX 4070), Northgate (RTX 5090), Strandgate (RTX 3090) |
| **Task** | Benchmark functions (Rastrigin, Rosenbrock), Nuclear EOS |
| **Framework** | Whatever Code Ocean capsule uses (likely PyTorch) |

### Recording

```
Gate:        [name]
GPU:         [model]
VRAM:        [GB]
Framework:   [PyTorch/TF version]
CUDA:        [version]

Task: Rastrigin surrogate
  Training time: [time]s
  Final accuracy: [metric]
  GPU utilization: [%]
  VRAM peak: [GB]

Task: Nuclear EOS surrogate
  ...
```

---

## Cross-Substrate Benchmarks (Future)

Same workload on every available compute substrate:
- CPU (EPYC, i9, 5800X3D)
- GPU (RTX 5090, RTX 3090, RTX 4070, RX 6950 XT)
- NPU (Akida)
- BarraCUDA (when ready)

This is the headline comparison. It comes after Phase A validation is complete.
