# Spec: sunMemo Configuration Archive Structure

**Location**: `~/.local/share/hotspring/configs/`  
**Binaries**: `arxiv_thermalize_sun`, `arxiv_measure_battery`, `milc_validation_loop`  
**Target**: All three reviewers (different views of same data)

---

## Purpose

The SU(N) memo table holds thermalized gauge configurations and their
measured observables. It serves three audiences:

- **Chuna**: MILC-compatible SU(3) configs for cross-validation
- **Kachkovskiy**: Eigenvalue/spectral data from gauge backgrounds
- **Murillo**: Provenance chain demonstrating reproducibility infrastructure

---

## Directory Structure

```
~/.local/share/hotspring/configs/
├── index.toml                    ← Master index (queryable)
├── su2/
│   ├── beta2.3_8x8x8x8/
│   │   ├── config_0001.bin       ← Internal binary format
│   │   ├── config_0001.milc      ← MILC v5 export
│   │   ├── observables.json      ← Measured ⟨P⟩, |L|, W(R,T), χ(R,T)
│   │   └── provenance.blake3    ← Content hash + DAG ref
│   ├── beta2.3_16x16x16x16/
│   └── beta2.3_24x24x24x24/
├── su3/
│   ├── beta6.0_8x8x8x8/
│   ├── beta6.0_16x16x16x16/     ← Primary validation point
│   │   ├── config_0001.bin
│   │   ├── config_0001.milc      ← For Chuna/Bazavov
│   │   ├── observables.json
│   │   └── provenance.blake3
│   ├── beta6.0_32x32x32x32/     ← Large-volume (Rung 1 production)
│   └── beta6.2_16x16x16x16/     ← Cross-vendor validated point
├── su4/
│   └── beta10.0_12x12x12x12/
├── su5/
│   └── beta16.0_8x8x8x8/
├── su6/
│   └── beta24.0_8x8x8x8/
├── su8/
│   └── beta40.0_8x8x8x8/
├── milc_import/                  ← Configs received from external (MILC/NERSC)
│   └── README.md
└── spectral/                     ← Eigenvalue data from Dirac/Wilson operators
    └── README.md
```

---

## Index Format (index.toml)

```toml
[metadata]
version = 1
last_updated = "2026-08-07T12:00:00Z"
gate = "strandGate"

[[configs]]
gauge_group = "SU(3)"
beta = 6.0
dims = [16, 16, 16, 16]
n_configs = 200
thermalization = 100
plaquette_mean = 0.59342
plaquette_err = 0.00003
path = "su3/beta6.0_16x16x16x16"
milc_exported = true
provenance = "blake3:a1b2c3d4..."

[[configs]]
gauge_group = "SU(4)"
beta = 10.0
dims = [12, 12, 12, 12]
n_configs = 50
thermalization = 200
plaquette_mean = 0.4521
plaquette_err = 0.0012
path = "su4/beta10.0_12x12x12x12"
milc_exported = false
provenance = "blake3:e5f6g7h8..."
```

---

## Observable JSON Format

```json
{
  "gauge_group": "SU(3)",
  "beta": 6.0,
  "dims": [16, 16, 16, 16],
  "n_configs": 200,
  "thermalization": 100,
  "observables": {
    "plaquette": {
      "mean": 0.59342,
      "error": 0.00003,
      "jackknife_bins": 10,
      "tau_int": 2.3
    },
    "polyakov_loop": {
      "re_mean": 0.0012,
      "im_mean": -0.0003,
      "abs_mean": 0.0041
    },
    "wilson_loops": {
      "W_1_1": 0.5934,
      "W_1_2": 0.3521,
      "W_2_2": 0.2103
    },
    "creutz_ratios": {
      "chi_1_1": 0.0287,
      "chi_2_2": 0.0291
    },
    "topological_charge": {
      "mean": 0.02,
      "susceptibility": 1.23
    }
  },
  "hardware": {
    "gpu": "NVIDIA GeForce RTX 3090",
    "precision": "DF64",
    "ms_per_trajectory": 617
  },
  "provenance": {
    "config_hash": "blake3:...",
    "code_commit": "7561b4c",
    "timestamp": "2026-08-07T12:34:56Z"
  }
}
```

---

## MILC Export Convention

Every SU(3) config with ≥100 post-thermalization trajectories gets an
automatic MILC v5 export. The export file sits alongside the internal
binary:

- `config_NNNN.bin` — Internal format (fast read, platform-native endian)
- `config_NNNN.milc` — MILC v5 format (big-endian, CRC32, portable)

The MILC file is what we hand to Chuna/Bazavov. It contains its own
plaquette in the header — they can verify it matches their measurement.

---

## Production Campaigns (current targets)

| Gauge Group | β | Volume | N_therm | N_prod | Status | Method |
|---|---|---|---|---|---|---|
| SU(2) | 2.2-2.5 | 16⁴-32⁴ | 200-400 | 200-1000 | ✅ Complete (36 configs) | CPU |
| SU(3) | 5.9-6.2 | 16⁴ | 200 | 200 | ✅ Complete (9 configs) | **GPU** (25s each) |
| SU(3) | 5.9-6.2 | 24⁴ | 200 | 200 | ✅ Complete (9 configs) | **GPU** (169s each) |
| SU(3) | 5.9-6.2 | 32⁴ | 300 | 200 | ✅ Complete (3 configs) | CPU |
| SU(4) | 10.5-11.0 | 16⁴ | 200 | 500 | ✅ Complete (3 configs) | CPU |
| SU(4) | 10.5-11.0 | 24⁴ | 300 | 500 | **IN PROGRESS** (100/300) | CPU |
| SU(5) | 16.5-17.5 | 16⁴ | 200 | 500 | **TODO** | CPU (GPU needs SU(N) shaders) |
| SU(6) | 24.0-26.0 | 16⁴ | 200 | 500 | **TODO** | CPU (GPU needs SU(N) shaders) |
| SU(8) | 44.0-46.0 | 16⁴ | 200 | 200 | **TODO** | CPU (GPU needs SU(N) shaders) |

**Total cached: 105 configs** (57 SU(3) + 36 SU(2) + 3 SU(4) + 9 root)

### GPU Pipeline Status (Aug 9, 2026)

| Stage | Status | Notes |
|-------|--------|-------|
| Thermalize (SU(3) ≤ 24⁴) | **GPU-NATIVE** | AMD 19× faster, 25s/config at 16⁴ |
| Thermalize (SU(3) 32⁴) | **GPU-READY** | Guard bypass → NVIDIA 24GB handles 51⁴ max |
| Thermalize (SU(N≥4)) | CPU only | Needs N×N shader generalization |
| Measure (plaquette) | **GPU-NATIVE** | Cross-GPU parity: Δ = 10⁻¹⁰ |
| Measure (Polyakov) | **GPU-NATIVE** | Built into HMC state |
| Measure (Wilson loops) | CPU | Complex multi-hop path, planned for GPU |
| Cross-validate | **GPU×2** | NVIDIA + AMD produce identical observables |

### Lattice Capacity (Aug 9, 2026)

Previous max was 22⁴ (software guard limited to 805 MB).

| Strategy | Max L⁴ | Sites | VRAM | vs Previous |
|----------|--------|-------|------|-------------|
| Previous (guard-limited) | 22⁴ | 234K | 0.8 GB | 1× |
| Guard bypass (NVIDIA) | 51⁴ | 6.8M | 23.9 GB | 29× |
| ROP offload (3 bufs) | 61⁴ | 13.8M | 24.9 GB | 59× |
| Precision folded + ROP | 64⁴ | 16.8M | 24.8 GB | 72× |
| Multi-GPU + folded | **73⁴** | **28.4M** | 41.9 GB | **121×** |

Per-site VRAM: 3,536 bytes (current) → 1,488 bytes (fully folded).

Silicon offloading saves 23-27% of trajectory time by moving force
accumulation to ROPs, reductions to subgroup shuffles, and
interpolation to TMU hardware.

---

## Cross-Silicon Profiling (Aug 8-9, 2026)

Same seed + same β + same dims produces identical Markov chain on both GPUs.
This validates hardware correctness and quantifies silicon routing benefit.

### Volume Scaling (SU(3), β=6.0, DF64, n_md=10) — Updated Aug 9

| Volume | RTX 3090 | RX 6950 XT | AMD Speedup | Plaquette Δ |
|--------|----------|-----------|-------------|-------------|
| 4⁴ | 10.54 ms | 5.00 ms | 2.1× | 1.82e-7 |
| 6⁴ | 10.67 ms | 3.90 ms | 2.7× | 6.22e-7 |
| 8⁴ | 28.05 ms | 5.32 ms | 5.3× | 7.40e-8 |
| 10⁴ | 94.04 ms | 6.81 ms | 13.8× | 1.57e-8 |
| 12⁴ | 198.61 ms | 11.61 ms | 17.1× | 1.28e-8 |
| 16⁴ | 625.70 ms | 31.09 ms | **20.1×** | 6.11e-8 |

**AMD scales nearly linearly** (6.2× time for 256× volume). Infinity Cache dominance.
**NVIDIA dispatch-bound** at small volumes (10 ms floor at 4⁴-6⁴).

### NPU Cross-PCIe Pattern — Aug 9

Three-substrate orchestration validated:
- GPU (RTX 3090) → HMC production
- GPU (RX 6950 XT) → f64 oracle / cross-validation
- NPU (AKD1000 sim) → ESN phase classification (0.02% overhead)

| Metric | Value |
|--------|-------|
| ESN f32 vs CPU f64 error | 3.91×10⁻⁷ |
| Monitoring overhead | 15.7 µs / 68.69 ms = 0.02% |
| Adaptive β_c accuracy | 0.013 from known |
| Compute savings (NPU steering) | 62% fewer evaluations |

### Silicon Routing (toadStool-consumable)

| Task | Card | Rationale |
|------|------|-----------|
| HMC production (any volume) | AMD RX 6950 XT | 5-20× faster, same physics |
| Precision oracle / f64 | NVIDIA RTX 3090 | Native SHADER_F64 |
| ESN phase classification | AKD1000 NPU | 30 mW, zero GPU interference |
| Large reservoir ESN (RS>768) | GPU (either) | GPU crossover at RS=768 |
| Adaptive β steering | NPU → GPU | NPU predicts, GPU validates |
| TMU multigrid | RTX 3090 | Texture cache hierarchy |
| ROP force atomics | AMD | 6.35× faster |
| Cross-silicon validation | Both GPUs | Same config, compare plaquettes |

### Key Properties

- **Bitwise reproducible**: same seed → Δ = 0.00 on repeat
- **Cross-GPU parity**: Δ ≤ 10⁻⁷ (DF64 rounding, not physics)
- **Linear MD scaling on AMD**: time ∝ n_md (compute-bound, IC-resident)
- **NPU overhead negligible**: 0.02% of trajectory time
- **AMD buffer fix**: `max_buffer_size = 2^31-1` (RADV i32::MAX constraint)
- **20× root cause**: intra-dispatch IC absorption (34-113 MB working set vs 6 MB L2)
- **Generation-specific**: NVIDIA wins raw (1.3-1.9×), AMD wins lattice QCD (IC effect)
- **RT Cores accessible**: EXPERIMENTAL_RAY_QUERY=YES on both cards (wgpu 28)
- **F16 available**: SHADER_F16=YES — enables DF32 precision tier
- **14/15 silicon units accessible** (only tensor cores remain driver-blocked)

---

## Files

- `~/.local/share/hotspring/configs/` — Config archive
- `src/bin/arxiv_thermalize_sun.rs` — Thermalization binary
- `src/bin/arxiv_thermalize_gpu.rs` — GPU-accelerated thermalizer
- `src/bin/arxiv_measure_battery.rs` — Observable measurement
- `src/bin/arxiv_measure_gpu.rs` — GPU-accelerated measurement
- `src/bin/milc_validation_loop.rs` — MILC export/import
- `src/bin/bench_silicon_crosspath_qcd.rs` — Cross-GPU same-seed comparison
- `src/bin/bench_silicon_volume_scaling.rs` — Volume scaling benchmark
- `src/bin/bench_silicon_force_paths.rs` — Force accumulation profiling
- `src/bin/bench_precision_ladder.rs` — Precision/reproducibility validation
- `src/bin/bench_gpu_pcie_stream.rs` — GPU-to-GPU PCIe streaming
- `src/bin/validate_gpu_cpu_therm_parity.rs` — GPU vs CPU thermalization parity
- `src/bin/validate_three_substrate.rs` — GPU + NPU + validation GPU orchestration
- `src/bin/validate_hetero_monitor.rs` — Heterogeneous monitoring (9/9 pass)
- `src/bin/validate_streaming_pipeline.rs` — GPU streaming pipeline validation
- `src/bin/cross_substrate_esn_benchmark.rs` — CPU × GPU × NPU ESN comparison (35/35 pass)
- `src/bin/production_mixed_pipeline.rs` — Full production: 3090 + NPU + AMD oracle
- `src/bin/sun_npu_metalforge.rs` — SU(N) NPU metalForge phase classification
- `src/bin/profile_lattice_capacity.rs` — Max lattice size + silicon offloading profiler
- `src/bin/pseudospore_manifest.rs` — BLAKE3 manifest + DAG generator for pseudoSpore bundles
- `src/bin/bench_silicon_genealogy.rs` — Full SiliconProfile: cache, dispatch, FP32, atomics per card
- `src/bin/bench_access_pattern_era.rs` — Linear vs strided access per generation
- `src/bin/bench_dispatch_count_scaling.rs` — Dispatch count scaling (proves linear, disproves eviction)
- `src/bin/probe_rt_tensor_features.rs` — wgpu feature census (RT, F16, F64, subgroup per card)
- `scripts/validate.sh` — POSIX sh validation script (BLAKE3 + DAG + Ed25519)
- `specs/SUNMEMO_STRUCTURE.md` — This document
