# Experiment 052: NVK/Vulkan Baseline + Kokkos Parity Benchmark

**Date:** March 10, 2026
**hotSpring:** v0.6.28
**barraCuda:** v0.3.4 (`a012076`)
**toadStool:** S145
**coralReef:** Phase 10, Iteration 30
**Status:** ACTIVE

---

## Objective

Quantify the performance gap between hotSpring's pure Rust GPU pipeline
(barraCuda via wgpu/Vulkan) and Kokkos/CUDA (via LAMMPS) on identical
physics workloads. This gap is the primary metric driving optimization of
both the wgpu/Vulkan path and coralReef's sovereign dispatch.

Secondary goal: establish NVK/Mesa performance baselines on the Titan V
as the floor that coralReef's sovereign dispatch must meet.

See `specs/MULTI_BACKEND_DISPATCH.md` for the three-tier dispatch architecture.

---

## Background

### The "Unknown Solution" (March 10, 2026)

Hardware diagnostics for coralReef Iteration 30 (Experiment 051) revealed
that Mesa's NVK Vulkan driver (25.1.5) provides full compute dispatch on
the Titan V (GV100, SM70). This was previously assumed to be limited due
to NVK instability on Volta hardware. The discovery means:

- **Both GPUs dispatch through wgpu/Vulkan** (Titan V via NVK, RTX 3090
  via NVIDIA proprietary)
- **Full 4-tier precision** (F32, F64, F64Precise, DF64) on Titan V
- **Dual-GPU cooperative dispatch** is functional (Split BCS, Split HMC,
  Redundant validation, PCIe 1.2 GB/s)

### Current Performance Gap

From `specs/CROSS_SPRING_EVOLUTION.md` (RTX 3090, DF64, N=2000):

| Case | barraCuda steps/s | Est. Kokkos-CUDA | Gap |
|------|-------------------|------------------|-----|
| k1_G14 | 181 (AllPairs) | ~720 | 4.0x |
| k2_G31 | 368 (Verlet) | ~1100 | 3.0x |
| k2_G158 | 846 (Verlet) | ~3050 | 3.6x |
| k3_G100 | 977 (Verlet) | ~3130 | 3.2x |
| k3_G1510 | 992 (Verlet) | ~3670 | 3.7x |

**Note:** Kokkos numbers are estimates from published data scaled to RTX 3090.
This experiment replaces estimates with measured values.

---

## Test Matrix

### Phase 1: NVK Baseline (Titan V via wgpu/Vulkan)

```bash
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_md_parity -- --output=nvk_baseline.json
```

Captures: steps/s, energy drift, wall time, force method for all 9 cases
on Titan V (NVK, native f64 at 1:2 throughput).

### Phase 2: Proprietary Baseline (RTX 3090 via wgpu/Vulkan)

```bash
HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin bench_md_parity -- --output=prop_baseline.json
```

Captures: same metrics on RTX 3090 (NVIDIA proprietary, DF64 on f32 cores).

### Phase 3: Kokkos/CUDA Reference (RTX 3090)

**Prerequisite:** LAMMPS compiled with Kokkos-CUDA.

```bash
# Build LAMMPS with Kokkos-CUDA
cd /opt/lammps
cmake -C ../cmake/presets/kokkos-cuda.cmake \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE86=ON \
      ../cmake
make -j$(nproc) lmp

# Run parity benchmark (auto-detects LAMMPS in PATH)
cargo run --release --bin bench_md_parity -- --output=kokkos_parity.json
```

### Phase 4: N-Scaling Comparison

Run all three backends at N=500, 2000, 5000, 10000 to understand scaling:

```bash
for n in 500 2000 5000 10000; do
    cargo run --release --bin bench_md_parity -- --particles=$n --quick --output=scaling_N${n}.json
done
```

---

## Metrics

| Metric | Source | Comparison Axis |
|--------|--------|----------------|
| Steps/s (production) | MD simulation | Primary performance |
| Energy drift % | Thermodynamic validation | Physics correctness |
| Wall time (total) | Benchmark harness | End-to-end cost |
| Force method | Algorithm selector | Fair comparison |
| GPU adapter + driver | wgpu/Vulkan enumeration | Hardware identification |

---

## Expected Outcomes

### NVK Baseline

The Titan V should show competitive or superior steps/s compared to the
RTX 3090 (DF64) for:
- AllPairs workloads (FP64 throughput-dominated, Titan V has 1:2 vs 1:64)
- Small N (dispatch overhead-dominated)

The RTX 3090 may win on:
- CellList/Verlet workloads at large N (higher raw FLOPS on f32 cores)
- Memory bandwidth-limited workloads (936 GB/s vs 653 GB/s)

### Kokkos Gap Analysis

Expect 3-4x gap, decomposed into:
1. **Dispatch overhead** (15-20%): wgpu round-trip vs CUDA kernel launch
2. **Memory access** (20-30%): WGSL vs hand-tuned CUDA memory patterns
3. **Arithmetic throughput** (10-15%): FMA fusion, register pressure
4. **Compilation quality** (5-10%): naga→SPIR-V→ptxas vs direct ptxas

### coralReef Targets

Once coralReef sovereign dispatch is unblocked:
- **Floor:** Must match NVK/Vulkan steps/s on same hardware
- **Ceiling:** Should approach Kokkos-CUDA steps/s
- **Advantage:** FMA policy control, register allocation, instruction scheduling

---

## Infrastructure

### New Files

| File | Purpose |
|------|---------|
| `barracuda/src/bench/md_backend.rs` | `MdBenchmarkBackend` trait + implementations |
| `barracuda/src/bin/bench_md_parity.rs` | 9-case benchmark binary |
| `specs/MULTI_BACKEND_DISPATCH.md` | Three-tier dispatch strategy |

### Existing Files Used

| File | Purpose |
|------|---------|
| `barracuda/src/bench/compute_backend.rs` | `BackendKind` enum (extended) |
| `barracuda/src/md/config.rs` | `dsf_pp_cases()` — the 9-case matrix |
| `barracuda/src/md/sarkas_harness.rs` | Existing GPU MD execution logic |
| `experiments/040_KOKKOS_LAMMPS_VALIDATION.md` | LAMMPS input template |
| `experiments/051_CORALREEF_ITER30_HARDWARE_DATA.md` | Hardware diagnostics |

---

## Results

### Phase 1: NVK Baseline

**Status:** PENDING — run `bench_md_parity` with `HOTSPRING_GPU_ADAPTER=titan`

### Phase 2: Proprietary Baseline

**Status:** PENDING — run `bench_md_parity` with `HOTSPRING_GPU_ADAPTER=3090`

### Phase 3: Kokkos/CUDA

**Status:** BLOCKED — LAMMPS+Kokkos not yet installed on test rig

### Phase 4: N-Scaling

**Status:** PENDING

---

## Gap Tracking

| Date | Avg Gap | Best Case | Worst Case | Notes |
|------|---------|-----------|------------|-------|
| Mar 5, 2026 | ~3.5x | 3.0x (k2_G31) | 4.0x (k1_G14) | Estimated from published data |
| Mar 10, 2026 | — | — | — | bench_md_parity created, awaiting first run |

Target trajectory:
- **v0.6.28**: Establish measured baseline (this experiment)
- **v0.6.30**: Reduce to 2.0-2.5x via workgroup tuning + shared memory
- **v0.7.x**: Reduce to 1.5x via coralReef sovereign dispatch
- **v0.8.x**: Parity (1.0x) or better via full sovereign optimization

---

## References

- Experiment 040: Kokkos/LAMMPS Validation Baseline (queued plan)
- Experiment 051: coralReef Iter 30 Hardware Data Capture (completed)
- specs/MULTI_BACKEND_DISPATCH.md: Three-tier dispatch architecture
- Choi, Dharuman, Murillo: Phys. Rev. E 100, 013206 (2019) — PP Yukawa DSF
- Edwards et al.: JPDC 74(12), 2014 — Kokkos
- Trott et al.: IEEE CiSE 24(4), 2022 — Kokkos 3
