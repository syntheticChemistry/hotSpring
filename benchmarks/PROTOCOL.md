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

This directly informs which BarraCuda shaders are highest priority.

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
- **BarraCuda CPU** — native Rust (L1 + L2)
- **BarraCuda GPU** — WGSL f64 shaders via wgpu/Vulkan (L1, evolving to L2+)
- **NPU (Akida)** — future

### Nuclear EOS substrate matrix

| Phase | Python | BarraCuda CPU | BarraCuda GPU |
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
  BarraCuda CPU:   chi2 = 4.99,   72.7 us/eval,    374 J,   51 W CPU
  BarraCuda GPU:   chi2 = 4.99,   39.7 us/eval,    126 J,   32 W GPU
  GPU precision:   Max |B_cpu - B_gpu| = 4.55e-13 MeV

L1 LHS Sweep (512 parameter sets):
  Python:          chi2 = 6.87 (best/512), 147.2s, 7,420 J
  BarraCuda CPU:   chi2 = 5.69 (best/512),   5.07s,  265 J
  BarraCuda GPU:   chi2 = 5.69 (best/512),   3.75s,  122 J

L2 HFB DirectSampler (12 evals):
  BarraCuda CPU:   chi2 = 23.09,  252s, 32,500 J (135W avg)
```

### Gate matrix

| Gate | CPU | GPU | RAM |
|------|-----|-----|-----|
| **Eastgate** (dev) | i9-12900K (16c/24t) | RTX 4070 (12GB) | 64GB |
| **Strandgate** | EPYC 7742 (64c) | — | 256GB |
| **Northgate** | i9-14900K | RTX 5090 (32GB) | 128GB |
| **Southgate** | 5800X3D | — | 32GB |

---

## GPU Molecular Dynamics — Sarkas on Consumer GPU

### Overview

The `sarkas_gpu` binary runs full Yukawa OCP molecular dynamics entirely on
GPU using f64 WGSL shaders (`SHADER_F64`). This reproduces the Sarkas PP
Yukawa DSF study (9 cases) on consumer hardware.

Run: `cargo run --release --bin sarkas_gpu`

| Mode | Flag | Description |
|------|------|-------------|
| Quick | (default) | κ=2, Γ=158, N=500, 1k equil + 5k prod |
| Full sweep | `--full` | All 9 PP Yukawa cases, N=2000 |
| Scaling | `--scale` | κ=2 Γ=158 at N=500, 2000, 5000, 10000 |

### Full Sweep Results (Eastgate, February 13, 2026)

**9/9 PP Yukawa cases PASSED** at N=2000 on RTX 4070 (f64 WGSL):

| κ | Γ    | Drift | RDF Tail Err | D*      | steps/s | Time   | GPU Energy |
|---|------|-------|-------------|---------|---------|--------|------------|
| 1 |   14 | 0.000% | 0.0001     | 1.35e-1 | 74.0    | 7.9min | 25.6 kJ    |
| 1 |   72 | 0.000% | 0.0004     | 2.40e-2 | 76.7    | 7.6min | 24.6 kJ    |
| 1 |  217 | 0.004% | 0.0009     | 8.18e-3 | 84.0    | 6.9min | 22.5 kJ    |
| 2 |   31 | 0.000% | 0.0001     | 6.10e-2 | 78.7    | 7.4min | 23.7 kJ    |
| 2 |  158 | 0.000% | 0.0003     | 5.49e-3 | 90.2    | 6.5min | 20.7 kJ    |
| 2 |  476 | 0.000% | 0.0014     | 2.76e-5 | 96.9    | 6.0min | 19.3 kJ    |
| 3 |  100 | 0.000% | 0.0001     | 2.28e-2 | 85.5    | 6.8min | 21.7 kJ    |
| 3 |  503 | 0.000% | 0.0001     | 1.73e-3 | 100.0   | 5.8min | 18.7 kJ    |
| 3 | 1510 | 0.000% | 0.0014     | 1.00e-4 | 120.3   | 4.9min | 15.5 kJ    |

**Total sweep: 60 minutes, 53W average GPU, ~192 kJ total.**

### GPU vs CPU Scaling

| N    | GPU steps/s | CPU steps/s | Speedup | GPU J/step | CPU J/step |
|------|-------------|-------------|---------|------------|------------|
|  500 |       521.5 |       608.1 |    0.9× | 0.081      | 0.071      |
| 2000 |       240.5 |        64.8 |  **3.7×** | 0.207    | 0.712      |

GPU advantage grows with N² (force computation is O(N²) all-pairs).
At N=2000, GPU uses **3.4× less energy per step** and runs **3.7× faster**.

### Acceptance Criteria

| Observable | Criterion | Status |
|------------|-----------|--------|
| Energy drift | < 5% | ✅ All ≤ 0.004% |
| RDF tail | \|g(∞)−1\| < 0.15 | ✅ All ≤ 0.0014 |
| RDF peak | Increases with Γ | ✅ Verified across all 9 |
| VACF D* | Decreases with Γ | ✅ Verified |
| SSF S(k→0) | Compressibility trends | ✅ Verified |

### DSF Study Matrix (PP Yukawa)

| κ | Γ values | rc/a_ws | Method | Status |
|---|----------|---------|--------|--------|
| 1 | 14, 72, 217 | 8.0 | All-pairs GPU (N=2000) | ✅ 3/3 |
| 2 | 31, 158, 476 | 6.5 | All-pairs GPU (N=2000) | ✅ 3/3 |
| 3 | 100, 503, 1510 | 6.0 | All-pairs GPU (N=2000) | ✅ 3/3 |

### Phase D: Native f64 Builtins + N-Scaling (Feb 14-15, 2026)

After replacing software-emulated f64 transcendentals with native WGSL builtins:

| N | steps/s | Wall (35k) | Energy (J) | Method |
|---|---------|-----------|-----------|--------|
| 500 | 998.1 | 35s | 1,655 | all-pairs |
| 2,000 | 361.5 | 97s | 5,108 | all-pairs |
| 5,000 | 134.9 | 259s | 16,745 | all-pairs |
| 10,000 | 110.5 | 317s | 19,351 | cell-list |
| 20,000 | 56.1 | 624s | 39,319 | cell-list |

**2-6× improvement** over software-emulated baseline. Paper parity (N=10k) in **5.3 min**.
GPU now wins at every N including N=500 (1.8× vs CPU).

### Flagged for Future

- **PPPM/Ewald** (κ=0 Coulomb cases): needs 3D FFT pipeline on GPU
- **MSU HPC comparison**: CPU baseline at N=10,000+ for headline number
- **Extended N-scaling** (N=50k-400k): see `experiments/003_RTX4070_CAPABILITY_PROFILE.md`
- **Parameter space sweep** (36 κ,Γ cases): ~3 hrs for 4× the Murillo study
- **Nuclei scaling** (52→2457): full AME2020 dataset, ~3 min on GPU
