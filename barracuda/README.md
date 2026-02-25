# hotspring-barracuda

**Validation crate for hotSpring — computational physics reproduction on consumer hardware.**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](../LICENSE)

---

## What This Is

`hotspring-barracuda` is the Rust crate that drives all validation, physics
computation, and GPU acceleration for the [hotSpring](../README.md) project.
It leans on ToadStool's shared [BarraCUDA](../../phase1/toadstool/crates/barracuda/)
library for scientific computing primitives (linear algebra, GPU dispatch, shaders)
and adds domain-specific physics: nuclear structure, molecular dynamics, lattice
QCD, spectral theory, and transport coefficients.

```
hotSpring (this repo)
  └── barracuda/              ← you are here (hotspring-barracuda v0.6.14)
       ├── src/lib.rs         ← crate root
       ├── src/physics/       ← nuclear structure (L1/L2/L3 HFB, SEMF)
       ├── src/md/            ← GPU molecular dynamics (Yukawa OCP)
       ├── src/lattice/       ← lattice QCD (SU(3), HMC, Dirac, CG, Abelian Higgs)
       ├── src/spectral/      ← re-exports from upstream barracuda::spectral
       ├── src/gpu/           ← GPU FP64 device wrapper
       ├── src/tolerances/    ← ~150 centralized validation thresholds
       ├── src/provenance.rs  ← baseline + analytical provenance (DOIs, Python origins)
       ├── src/discovery.rs   ← capability-based data path + NPU discovery
       └── src/bin/           ← 76 validation/benchmark binaries
```

---

## Quick Start

```bash
cd barracuda

cargo test --lib          # 629 library tests (~700s)
cargo test                # 664 total (629 lib + 31 integration + 4 doc)
cargo clippy --all-targets  # 0 warnings (pedantic + nursery)
cargo doc --no-deps       # Full API docs, 0 warnings

cargo run --release --bin validate_all  # 39/39 validation suites
```

Requires the shared BarraCUDA crate at `../../phase1/toadstool/crates/barracuda`.

---

## Architecture

### Zero Unsafe Code

The entire crate has zero `unsafe` blocks. `#![deny(clippy::expect_used, clippy::unwrap_used)]`
enforced crate-wide — all fallible operations use `?` propagation via `HotSpringError`.

### Dependency on Shared BarraCUDA

hotspring-barracuda depends on the shared `barracuda` crate for:

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
    → hands off to toadStool via wateringHole/handoffs/
      → toadStool absorbs as shared GPU primitives
        → hotSpring rewires to lean on upstream, deletes local code
```

### Module Map

| Module | Domain | Tests | WGSL Shaders |
|--------|--------|:-----:|:------------:|
| `physics/` | Nuclear structure (L1/L2/L3 HFB, SEMF) | ~180 | 14 |
| `md/` | GPU molecular dynamics (Yukawa OCP) | ~120 | 11 |
| `lattice/` | Lattice QCD (SU(3), HMC, Dirac, CG) | ~200 | 7 |
| `spectral/` | Re-exports from upstream barracuda | — | — |
| `gpu/` | FP64 device wrapper, telemetry | ~30 | — |
| `tolerances/` | ~150 centralized thresholds | ~20 | — |
| `bench/` | Benchmark harness (RAPL, nvidia-smi) | ~10 | — |

### Key Properties

- **AGPL-3.0-only** on all 135 `.rs` files and 25 `.wgsl` shaders
- **Provenance**: all validation constants traced to Python origins or DOIs
- **Tolerances**: ~150 centralized constants with physical justification
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

Current: **v0.6.14** (February 25, 2026)
- 0 clippy warnings (library + 76 binaries, pedantic + nursery)
- Cross-primal discovery (`probe_npu_available()`, no hardcoded paths)
- β_c provenance (Bali, Engels, Creutz citations in `provenance.rs`)
- WGSL PRNG deduplication (`prng_pcg_f64.wgsl` shared library)
- 4 new GPU validation tolerances
- `gpu_hmc.rs` refactored into `gpu_hmc/` module (5 files, all <1000 LOC)

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
