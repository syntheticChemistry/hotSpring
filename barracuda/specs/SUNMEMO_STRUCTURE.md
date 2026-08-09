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

| Gauge Group | β | Volume | N_therm | N_prod | Status |
|---|---|---|---|---|---|
| SU(2) | 2.3 | 8⁴ | 100 | 1000 | ✅ Complete |
| SU(2) | 2.3 | 16⁴ | 200 | 500 | ✅ Complete |
| SU(2) | 2.3 | 24⁴ | 200 | 200 | ✅ Complete |
| SU(3) | 6.0 | 8⁴ | 50 | 200 | ✅ Complete |
| SU(3) | 6.0 | 16⁴ | 100 | 200 | ✅ Complete |
| SU(3) | 6.2 | 16⁴ | 100 | 200 | ✅ Complete |
| SU(3) | 6.0 | 32⁴ | 100 | 200 | **TODO** |
| SU(4) | 10.0 | 12⁴ | 200 | 500 | **IN PROGRESS** |
| SU(5) | 16.0 | 8⁴ | 300 | 500 | **TODO** |
| SU(6) | 24.0 | 8⁴ | 400 | 500 | **TODO** |
| SU(8) | 40.0 | 8⁴ | 500 | 200 | **TODO** |

---

## Cross-Silicon Profiling (Aug 8, 2026)

Same seed + same β + same dims produces identical Markov chain on both GPUs.
This validates hardware correctness and quantifies silicon routing benefit.

### Volume Scaling (SU(3), β=6.0, DF64, n_md=10)

| Volume | RTX 3090 | RX 6950 XT | AMD Speedup | Plaquette Δ |
|--------|----------|-----------|-------------|-------------|
| 4⁴ | 9.49 ms | 4.93 ms | 1.92× | 1.82e-7 |
| 8⁴ | 25.69 ms | 5.31 ms | 4.84× | 7.40e-8 |
| 12⁴ | 190.93 ms | 11.26 ms | 16.95× | 1.28e-8 |
| 16⁴ | 601.27 ms | 31.32 ms | 19.19× | 6.11e-8 |

### Silicon Routing (toadStool-consumable)

| Task | Card | Rationale |
|------|------|-----------|
| HMC production (≥ 8⁴) | AMD | 5-19× faster |
| Precision oracle | NVIDIA | Independent silicon = independent cross-check |
| ESN classification | AKD1000 NPU | 2 µs/sample |
| TMU multigrid | RTX 3090 | Texture cache hierarchy |
| ROP force atomics | AMD | 6.35× faster |

### Key Properties

- **Bitwise reproducible**: same seed → Δ = 0.00 on repeat
- **Cross-GPU parity**: Δ ≤ 10⁻⁶ (DF64 rounding, not physics)
- **Linear MD scaling on AMD**: time ∝ n_md (compute-bound)

---

## Files

- `~/.local/share/hotspring/configs/` — Config archive
- `src/bin/arxiv_thermalize_sun.rs` — Thermalization binary
- `src/bin/arxiv_measure_battery.rs` — Observable measurement
- `src/bin/milc_validation_loop.rs` — MILC export/import
- `src/bin/bench_silicon_crosspath_qcd.rs` — Cross-GPU same-seed comparison
- `src/bin/bench_silicon_volume_scaling.rs` — Volume scaling benchmark
- `src/bin/bench_silicon_force_paths.rs` — Force accumulation profiling
- `src/bin/bench_precision_ladder.rs` — Precision/reproducibility validation
- `specs/SUNMEMO_STRUCTURE.md` — This document
