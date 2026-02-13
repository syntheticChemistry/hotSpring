# Benchmark Protocol

**Purpose**: Ensure reproducible, comparable results across gates.

**Why hardware matters**: Identical physics computations cost different amounts
of time, energy, and money depending on the substrate (Python vs Rust vs GPU).
This protocol captures the full cost picture so that "same physics, different
cost" becomes a quantitative, citable claim.

---

## General Rules

1. **Same input, same parameters, same code** — only the hardware changes
2. **3 runs minimum** — report median wall time (avoids JIT and caching outliers)
3. **Record everything**: wall time, time per step, peak memory, CPU threads, GPU utilization, **CPU energy (RAPL), GPU power (nvidia-smi)**
4. **First run is warmup** — Numba JIT compiles on first call. Record it separately.
5. **Pin Numba threads** to match core count: `NUMBA_NUM_THREADS=N`

---

## Automated Measurement System

As of February 2026, hotSpring includes an automated benchmark harness that
captures time, energy, and hardware context for every validation run.

### What's measured

| Metric | Source | Unit |
|--------|--------|------|
| **Wall time** | `Instant::now()` (Rust) / `time.monotonic()` (Python) | seconds |
| **Per-eval time** | wall / n_evals | microseconds |
| **CPU energy** | Intel RAPL `/sys/class/powercap/intel-rapl:0/energy_uj` | Joules |
| **GPU power** | `nvidia-smi -lms 100` (background thread, 100ms polling) | Watts |
| **GPU energy** | Trapezoidal integration of power samples | Joules |
| **GPU temperature** | nvidia-smi | Celsius |
| **GPU VRAM** | nvidia-smi | MiB |
| **Process RSS** | `/proc/self/status` VmHWM (Rust) / `resource.getrusage` (Python) | MB |
| **Hardware inventory** | `/proc/cpuinfo`, `/proc/meminfo`, nvidia-smi | — |

### How to run

```bash
# Rust (L1 + L2 with full benchmark harness)
cd hotSpring/barracuda
cargo run --release --bin nuclear_eos_gpu

# Python (surrogate workflow with benchmark harness)
cd hotSpring/control/surrogate/nuclear-eos
micromamba run -n surrogate python3 scripts/run_surrogate.py --level 1
```

Both produce JSON reports automatically.

### Where results go

```
benchmarks/nuclear-eos/results/
├── eastgate_2026-02-12T19-30-00.json   (Rust)
├── eastgate_2026-02-12T19-45-00_3.11.json (Python)
└── ...
```

### JSON schema

Each report contains:
- `timestamp` — ISO 8601
- `hardware` — full `HardwareInventory` (CPU, GPU, RAM, kernel, etc.)
- `phases[]` — array of `PhaseResult` objects:
  - `phase`, `substrate`, `wall_time_s`, `per_eval_us`, `n_evals`
  - `energy` — `{ cpu_joules, gpu_joules, gpu_watts_avg, gpu_watts_peak, gpu_temp_peak_c, gpu_vram_peak_mib, gpu_samples }`
  - `peak_rss_mb`, `chi2`, `precision_mev`, `notes`

### Comparing across gates

Load any two JSON files and diff on any axis:
- Time: `phase.wall_time_s`
- Energy: `phase.energy.cpu_joules` or `phase.energy.gpu_joules`
- Throughput: `1e6 / phase.per_eval_us` (evals/second)
- Physics: `phase.chi2` (must match within precision)

### Source code

- **Rust**: `barracuda/src/bench.rs` — `HardwareInventory`, `PowerMonitor`, `BenchReport`
- **Python**: `control/surrogate/nuclear-eos/scripts/bench_wrapper.py` — compatible implementation

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

## Cross-Substrate Benchmarks

Same workload on every available compute substrate:
- **Python** — scipy/numpy (reference)
- **BarraCUDA CPU** — native Rust (L1 + L2)
- **BarraCUDA GPU** — WGSL f64 shaders via wgpu/Vulkan (L1, evolving to L2+)
- **NPU (Akida)** — future

### Nuclear EOS substrate matrix

| Phase | Python | BarraCUDA CPU | BarraCUDA GPU |
|-------|--------|---------------|---------------|
| L1 SEMF (100k) | ✅ 1,143 us/eval, 5,648 J | ✅ 72.7 us/eval, 374 J | ✅ 39.7 us/eval, 126 J |
| L1 LHS sweep (512 pts) | ✅ 147.2s, 7,420 J | ✅ 5.07s, 265 J | ✅ 3.75s, 122 J |
| L1 DirectSampler | ✅ 6.62 chi2 (1008 evals) | ✅ 1.52 chi2 (48 evals) | ✅ GPU-accelerated eval |
| L2 HFB (SLy4 baseline) | ✅ | ✅ 54.91 chi2 | CPU (GPU evolving) |
| L2 DirectSampler | ✅ 1.93 chi2 (SparsitySampler) | ✅ 23.09 chi2, 32,500 J | CPU (GPU evolving) |

Each cell generates a `PhaseResult` in the benchmark JSON, capturing time,
energy, and physics agreement.  This is the headline comparison: **same
physics, different cost**.

### Latest results (February 13, 2026, Eastgate)

```
L1 SEMF (100k iterations, 52 nuclei):
  Python:          chi2 = 4.99,  1,143 us/eval,  5,648 J,   49 W CPU
  BarraCUDA CPU:   chi2 = 4.99,   72.7 us/eval,    374 J,   51 W CPU
  BarraCUDA GPU:   chi2 = 4.99,   39.7 us/eval,    126 J,   32 W GPU
  GPU precision:   Max |B_cpu - B_gpu| = 4.55e-13 MeV

L1 LHS Sweep (512 parameter sets):
  Python:          chi2 = 6.87 (best/512), 147.2s, 7,420 J
  BarraCUDA CPU:   chi2 = 5.69 (best/512),   5.07s,  265 J
  BarraCUDA GPU:   chi2 = 5.69 (best/512),   3.75s,  122 J

L2 HFB DirectSampler (12 evals):
  BarraCUDA CPU:   chi2 = 23.09,  252s, 32,500 J (135W avg)
```

### Gate matrix

| Gate | CPU | GPU | RAM |
|------|-----|-----|-----|
| **Eastgate** (dev) | i9-12900K (16c/24t) | RTX 4070 (12GB) | 64GB |
| **Strandgate** | EPYC 7742 (64c) | — | 256GB |
| **Northgate** | i9-14900K | RTX 5090 (32GB) | 128GB |
| **Southgate** | 5800X3D | — | 32GB |
