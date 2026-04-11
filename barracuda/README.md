# hotspring-barracuda

**Validation crate for hotSpring — computational physics reproduction on consumer hardware.**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](../LICENSE)

---

## What This Is

`hotspring-barracuda` is the Rust crate that drives all validation, physics
computation, and GPU acceleration for the [hotSpring](../README.md) project.

**Dependency split:**

| Primal | Provides | Path |
|--------|----------|------|
| [barraCuda](../../barraCuda/) | Math, shaders, compilation, ESN, NPU math | `../../barraCuda/crates/barracuda` |
| [toadStool](../../phase1/toadStool/) | NPU hardware (akida-driver, akida-models) | `../../phase1/toadStool/crates/neuromorphic/` |
| [coralReef](../../coralReef/) | Sovereign shader compiler (WGSL→native SASS/GFX) | `../../coralReef/crates/coral-gpu` |

barraCuda knows WHAT to compute. toadStool exposes WHERE hardware exists.
coralReef compiles HOW (WGSL→native, bypassing wgpu/Vulkan). hotSpring
consumes all three for domain-specific physics: nuclear structure, molecular
dynamics, lattice QCD, spectral theory, and transport coefficients.

```
hotSpring (this repo)
  └── barracuda/              ← you are here (hotspring-barracuda v0.6.31)
       ├── src/lib.rs         ← crate root (v0.6.31)
       ├── src/physics/       ← nuclear structure (L1/L2/L3 HFB, SEMF)
       ├── src/md/            ← GPU molecular dynamics (Yukawa OCP)
       ├── src/lattice/       ← lattice QCD (SU(3), HMC, Dirac, CG, Abelian Higgs)
       ├── src/spectral/      ← re-exports from upstream barracuda::spectral
       ├── src/gpu/           ← GPU FP64 device wrapper
       ├── src/tolerances/    ← ~170 centralized validation thresholds
       ├── src/provenance.rs  ← baseline + analytical provenance (DOIs, Python origins)
       ├── src/discovery.rs   ← capability-based data path + NPU discovery
       └── src/bin/           ← 111+ validation/benchmark binaries
```

---

## Quick Start

```bash
cd barracuda

cargo test --lib          # 956 library tests
cargo clippy --all-targets  # 0 warnings (pedantic + nursery)
cargo doc --no-deps       # Full API docs, 0 warnings

cargo run --release --bin validate_all  # 39/39 validation suites
```

Requires the [barraCuda](../../barraCuda/) primal at `../../barraCuda/crates/barracuda`.

---

## Architecture

### Zero Unsafe Code

The entire crate has zero `unsafe` blocks. `#![deny(clippy::expect_used, clippy::unwrap_used)]`
enforced crate-wide — all fallible operations use `?` propagation via `HotSpringError`.

### Dependency on barraCuda

hotspring-barracuda depends on the standalone `barracuda` crate (v0.3.3) for:

| Primitive | Usage |
|-----------|-------|
| `barracuda::ops::linalg::lu_solve` | Linear system solving (replaced local Gauss-Jordan) |
| `barracuda::ops::md::CellListGpu` | GPU cell-list spatial decomposition |
| `barracuda::spectral::*` | Spectral theory (Anderson, Lanczos, Hofstadter) |
| `barracuda::gpu::*` | GPU adapter, shader compilation, WgslOptimizer |
| `barracuda::ops::*` | Linear algebra, quadrature, optimization, sampling |

### Evolution Cycle

```
hotSpring implements physics locally (Rust + WGSL templates)
  → validates against Python baselines
    → hands off to toadStool/barraCuda via wateringHole/handoffs/
      → barraCuda absorbs as shared GPU primitives
        → hotSpring rewires to lean on upstream, deletes local code
```

### Module Map

| Module | Domain | Tests | WGSL Shaders |
|--------|--------|:-----:|:------------:|
| `physics/` | Nuclear structure (L1/L2/L3 HFB, SEMF) | ~180 | 14 |
| `md/` | GPU molecular dynamics (Yukawa OCP) | ~120 | 11 |
| `lattice/` | Lattice QCD (SU(3), HMC, Dirac, CG, asymmetric Ns³×Nt) | ~200 | 7 |
| `spectral/` | Re-exports from upstream barracuda | — | — |
| `gpu/` | FP64 device wrapper, telemetry | ~30 | — |
| `tolerances/` | ~170 centralized thresholds | ~20 | — |
| `bench/` | Benchmark harness (RAPL, nvidia-smi) | ~10 | — |

### Hardware-Agnostic GPU Discovery

Any GPU with SHADER_F64 or DF64 fallback is a science device at 14-digit precision.
The overnight validation binary (`validate_chuna_overnight`) uses a four-phase pattern:

1. **Discover** — `GpuF64::enumerate_adapters()` finds all GPUs, filters by f64 capability (wrapped in `catch_unwind` + `HOTSPRING_NO_GPU` escape hatch for headless HPC)
2. **Profile** — `PrecisionBrain::new()` probes each tier (F32/F64/DF64/F64Precise) for compilation, dispatch, transcendental safety, and ULP accuracy
3. **Size** — `max_lattice_l()` derives workload dimensions from VRAM (`max_buffer_size`), so a 3050 with 8 GB runs 16^4, a 3090 with 24 GB runs 24^4+
4. **Validate** — runs the full Paper 43/44/45 suite on each substrate with tagged telemetry, then cross-compares physics observables across GPUs

```bash
cargo run --release --bin validate_chuna_overnight              # auto-select best GPU
cargo run --release --bin validate_chuna_overnight -- --all-gpus # validate every f64 GPU
cargo run --release --bin validate_chuna_overnight -- --gpu 3090 # target specific GPU
```

`TelemetryWriter` and `ValidationHarness` support per-substrate tagging, so every
JSONL event and JSON report records which GPU produced the result. When benchScale
deploys this in a container with one GPU passed through, the binary discovers that
GPU, profiles it, sizes workloads, validates — zero hardcoding.

This is smart **single-substrate** usage, distinct from brain architecture
(cross-substrate work-sharing, a metalForge concern).

**Migration for other binaries:** The 48+ other `GpuF64::new()` binaries already
work via env-driven discovery (`HOTSPRING_GPU_ADAPTER`). The discover-profile-size
pattern in the overnight binary is the reference for multi-GPU + profiling adoption.

### Key Properties

- **AGPL-3.0-or-later** on all `.rs` and `.wgsl` files
- **Provenance**: all validation constants traced to Python origins or DOIs
- **Tolerances**: ~170+ centralized constants with physical justification
- **ValidationHarness**: structured pass/fail with exit code 0/1
- **Capability discovery**: GPU adapter by name/index/auto, NPU via sysfs
- **NaN-safe**: all float sorting uses `f64::total_cmp()`
- **Zero external commands**: pure-Rust timestamps, graceful nvidia-smi fallback

---

## Validation Suites

39 validation suites covering 22 papers across 4 substrates:

| Substrate | Coverage |
|-----------|----------|
| Python Control | 18/22 papers |
| BarraCuda CPU | 22/22 papers (COMPLETE) |
| BarraCuda GPU | 20/22 papers |
| metalForge (GPU+NPU+CPU) | 9/22 papers |

---

## Version History

See [CHANGELOG.md](CHANGELOG.md) for full version history.

Current: **v0.6.29** (March 11, 2026)
- 847 lib tests, 112+ binaries, 84 WGSL shaders, 0 clippy warnings
- barraCuda v0.3.5 (`8d63c77`), toadStool S146, coralReef Phase 10 Iter 31
- Chuna Papers 43-45: 44/44 overnight checks pass
- coralReef sovereign compile: **45/46** shaders to native SM70/SM86 SASS
- 12/12 NVVM bypass patterns compile (all 3 poisoning patterns × 6 targets)
- Self-routing `PrecisionBrain`: hardware calibration, NVVM poisoning gated, sovereign bypass integrated
- Live Kokkos parity: 9/9 cases, 12.4× gap (DF64 transcendental fix applied)
- Deep technical debt resolution: zero files >1000 lines, zero unsafe

---

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
