# Chuna Engine — Production QCD Workbench

> **Previous version**: This document supersedes the March 11 fossil (v0.6.29).
> The original reproduction data sheet is preserved in `whitePaper/baseCamp/chuna_*.md`.
> For the full paper queue across 25 papers, see [`specs/PAPER_REVIEW_QUEUE.md`](specs/PAPER_REVIEW_QUEUE.md).

**Date**: April 2026 (v0.6.32 — hotSpring-guideStone-v0.7.0)
**Author**: Kevin Mok (mokkevin@msu.edu)
**Hardware**: biomeGate — Threadripper 3970X, RTX 3090 (24 GB), RX 6950 XT (16 GB), ~$4K used parts
**License**: AGPL-3.0

---

## Summary

Three of Chuna's published papers reproduced and validated across **three
independent substrates** (Python, Rust CPU, Rust GPU). The Chuna Engine
is a production tool for lattice QCD research — not a paper demo.

**59/59 validation checks pass.** Cross-vendor GPU parity: 2.59e-11 delta (NVIDIA vs AMD).
**8 production binaries** via `validation/chuna-engine`.
**CPU-only path**: All validation works without a GPU (`HOTSPRING_NO_GPU=1`).
**Python bridge**: `control/hotspring_reader/` loads all output into NumPy arrays.
**guideStone certified**: 5 properties + cross-substrate parity + optional NUCLEUS provenance.

| Paper | Citation | What we reproduced | Detailed artifact |
|-------|----------|--------------------|----|
| 43 | Bazavov & Chuna, arXiv:2101.05320 | Gradient flow w/ LSCFRK — coefficients derived independently, convergence verified, t₀/w₀ scale setting | [`whitePaper/baseCamp/chuna_gradient_flow.md`](whitePaper/baseCamp/chuna_gradient_flow.md) |
| 44 | Chuna & Murillo, Phys. Rev. E 111, 035206 | Standard, completed, multi-component Mermin dielectric. DSF vs MD validated against Murillo Group plasma DB | [`whitePaper/baseCamp/chuna_bgk_dielectric.md`](whitePaper/baseCamp/chuna_bgk_dielectric.md) |
| 45 | Haack, Murillo, Sagert & Chuna, J. Comput. Phys. (2024) | BGK relaxation + Euler/HLL + coupled kinetic-fluid interface. Conservation laws, H-theorem, Sod shock | [`whitePaper/baseCamp/chuna_kinetic_fluid.md`](whitePaper/baseCamp/chuna_kinetic_fluid.md) |

---

## Chuna Engine — 8 Binaries

| Binary | Purpose |
|--------|---------|
| `chuna_generate` | ILDG gauge config generation (CPU default, `--gpu` optional) |
| `chuna_flow` | Gradient flow on any ILDG config |
| `chuna_measure` | Observable suite: plaquette, Polyakov, Wilson loops, HVP, flow, condensate |
| `chuna_analyze` | Jackknife + autocorrelation + susceptibilities. `--format=tsv` for paper tables |
| `chuna_convert` | ILDG/LIME/QCDml conversion and CRC verification |
| `chuna_benchmark_flow` | Integrator efficiency workbench (W6, W7, CK4 comparison) |
| `chuna_matrix` | Task orchestration for production parameter sweeps |
| `chuna_validate_shader` | WGSL shader cross-path validation (CPU + NagaExecutor + GPU + coralReef) |

Entry point: `validation/chuna-engine <subcommand>` (POSIX wrapper with integrity checks).

## Quick Start

```bash
git clone https://github.com/syntheticChemistry/hotSpring
cd hotSpring/barracuda

# Full validation (59 checks, CPU-only, ~5 hours)
cargo run --release --bin validate_chuna_overnight

# Generate an ensemble
cargo run --release --bin chuna_generate -- \
    --beta=6.0 --dims=8,8,8,8 --n-configs=20 --outdir=data/b6.0_L8

# Measure observables
cargo run --release --bin chuna_measure -- \
    --dir=data/b6.0_L8/ --mass=0.1 --hvp --outdir=data/b6.0_L8/measurements

# Analyze with TSV export (paper-ready table)
cargo run --release --bin chuna_analyze -- \
    --dir=data/b6.0_L8/measurements/ --format=tsv --output=analysis.tsv

# CG convergence history for solver comparison
cargo run --release --bin chuna_measure -- \
    --dir=data/b6.0_L8/ --hvp --cg-history --outdir=measurements
```

**Requirements**: Rust (stable). No GPU required — set `HOTSPRING_NO_GPU=1` to
skip GPU discovery entirely. No manual dependency setup — `cargo build` handles everything.

## Python Interface

```python
import hotspring_reader as hs

# Load measurements into NumPy arrays
meas = hs.load_measurements("data/b6.0_L8/measurements/")
print(meas["plaquette"])     # float64 array
print(meas["flow_t0"])       # float64 array (NaN where absent)

# Export analysis to TSV for paper tables
ana = hs.load_analysis("analysis.json")
hs.to_tsv(ana, "table.tsv")  # observable, mean, error, tau_int, ...
```

Located at `control/hotspring_reader/`. Requires only NumPy.

---

## Paper 43: Gradient Flow Integrators

5 integrators implemented (Euler, RK2, W6, W7, CK4). The 3-stage 3rd-order
LSCFRK coefficients were **derived from first principles** — a `const fn`
solves the four Taylor order conditions at compile time. Our values match
the paper exactly.

| Result | Value |
|--------|-------|
| W6/W7/CK4 convergence order | 2.06/2.08/2.11 on 8⁴ (finite-size suppressed; threshold >1.5) |
| β=6.0 plaquette | ⟨P⟩ = 0.5929 (literature: ~0.594) |
| 8⁴ + 16⁴ thermalization | 100-500 HMC trajectories |
| Flow energy | Monotonically decreasing at all β |
| GPU speedup | **38.5×** (RTX 3090 vs single-core CPU) |
| W7 vs W6 for w₀ | ~2× more efficient (confirms paper) |

**Full details**: [`whitePaper/baseCamp/chuna_gradient_flow.md`](whitePaper/baseCamp/chuna_gradient_flow.md)

---

## Paper 44: Conservative BGK Dielectric

Standard, completed (Eq. 26), and multi-component Mermin dielectric functions.
All three on CPU (Rust) and GPU (WGSL f64 shaders).

| Result | Value |
|--------|-------|
| Debye screening | Exact to 1e-12 |
| DSF positivity | ≥98% (standard), ≥99% (completed) |
| f-sum rule | Converging to −πωₚ²/2 |
| GPU-CPU L² parity | 5.5e-7 (standard), 100% (multi-component) |
| Rust vs Python speedup | **322×** (same algorithm) |
| DSF vs MD (κ=2, Γ=31, q=0.54) | Completed Mermin within **0.8%** of MD peak |

**Bug found**: 6 `cscale` shader instances using element-wise `vec2 * vec2`
instead of complex×scalar — WGSL language footgun. Fix: 4% → 100% GPU agreement.

**Full details**: [`whitePaper/baseCamp/chuna_bgk_dielectric.md`](whitePaper/baseCamp/chuna_bgk_dielectric.md)

---

## Paper 45: Multi-Species Kinetic-Fluid Coupling

Multi-species BGK relaxation, 1D Euler/HLL shock tube, and fully coupled
kinetic-fluid interface with physics-based sub-iteration convergence.

| Result | Value |
|--------|-------|
| BGK mass conservation | Exact (Δm = 0) |
| Euler mass conservation | 1.4e-15 |
| Sod shock front | Contact + shock resolved |
| H-theorem | Entropy monotonically non-decreasing |
| Interface GPU-CPU parity | ~15% (physics-limited, not hand-tuned) |

**Full details**: [`whitePaper/baseCamp/chuna_kinetic_fluid.md`](whitePaper/baseCamp/chuna_kinetic_fluid.md)

---

## Cost

| | ICER | biomeGate |
|--|------|-----------|
| Hardware | HPC cluster ($M+) | ~$4,000 used parts |
| Software | MILC (C) + Python + Fortran + CUDA | Rust + WGSL (`cargo build`) |
| Gradient flow (8⁴ W7) | MILC production | 0.14s, 38.5× GPU speedup |
| Kinetic-fluid coupling | Python | 322× Rust speedup |
| Full validation (3 papers) | Batch queue | ~5 hours, single binary (8⁴ papers in ~1 hr) |
| Vendor lock-in | CUDA (NVIDIA only) | Vulkan (any GPU) |

---

## DF64: Additional Precision Tier

| Tier | Digits | Throughput (RTX 3090) |
|------|:------:|:---------------------:|
| f32 | 7 | ~29 TFLOPS |
| **DF64** | **14** | **~3.2 TFLOPS** |
| f64 | 15-16 | ~0.3 TFLOPS |

DF64 uses f32-pair arithmetic on the abundant FP32 cores. 9.9× native f64
throughput at 14-digit precision. Stability audited across 9 cancellation
families (Experiment 046).

---

## What We Extended Beyond the Papers

1. GPU acceleration for all three papers (none used GPU in the originals)
2. Multi-component Mermin (Paper 44) — electron-ion extension
3. **Dynamical N_f=4 staggered HMC (Paper 43) — COMPLETE**: 85% acceptance via warm-start mass annealing (m: 1.0 → 0.5 → 0.2 → 0.1), NPU-steered adaptive Omelyan, Akida AKD1000 feedback
4. Cross-paper validation — single binary validates all three together
5. DSF vs MD cross-validation — against Murillo Group open plasma database
6. Precision stability audit — stable at f32/DF64/f64
7. NPU steering with Akida AKD1000 — adaptive parameter control with learned trust thresholds
8. **Cross-spring shader evolution**: barraCuda's 791 WGSL shaders include contributions
   from all 5 springs — hotSpring precision/physics, wetSpring bio-stats,
   neuralSpring ML ops, groundSpring validation, airSpring hydrology — all
   available to hotSpring through a single `barracuda` dependency

---

## NPU Steering

The Akida AKD1000 NPU monitors the dynamical HMC and adjusts step size (dt)
and trajectory count (n_md) in real time.  Key design: the NPU is an
**apprentice, not a dictator** — it defers to the heuristic adaptive
controller during crisis regimes (acceptance < 40% or |ΔH| > 5) and only
steers when the system has stabilized.

See [`NPU_STEERING_LESSONS.md`](NPU_STEERING_LESSONS.md) for full writeup.

---

## Three-Substrate Parity

Every check is validated across Python, Rust CPU, and Rust GPU independently.
Cross-substrate comparison tool: `control/hotspring_reader/compare_substrates.py`.

| Substrate | Papers Covered | Entry Point |
|-----------|:--------------:|-------------|
| Python Control | 21/25 | `control/*/scripts/*_control.py` |
| Rust CPU | 25/25 | `cargo run --release --bin validate_chuna_overnight` |
| Rust GPU | 20/25 | `--gpu` flag on `chuna_generate` |

Full tolerance table with physical derivations: `validation/GUIDESTONE.md`.

---

## guideStone Certification

This artifact satisfies all 5 guideStone properties plus cross-substrate parity.
Optional NUCLEUS provenance layer (bearDog signing, rhizoCrypt DAG, toadStool
integration) activates when primals are detected, degrades gracefully when absent.

Certification: `validation/GUIDESTONE.md`
Integrity: `validation/CHECKSUMS` (SHA-256)

---

## Next Steps

1. **Validate** — Run `validation/run` on your machine (CPU-only, ~5 hours)
2. **Load** — Use `hotspring_reader.py` to inspect results in NumPy
3. **Generate** — Create ensembles at any beta/L/Nf with `chuna_generate`
4. **Compare** — Cross-check with MILC configs using matched parameters
5. **Prototype** — Design new integrators in `chuna_benchmark_flow`

---

## References

- Bazavov, A. & Chuna, T. arXiv:2101.05320 (2021)
- Chuna, T. & Murillo, M.S. Phys. Rev. E 111, 035206 (2024)
- Haack, J.R., Murillo, M.S., Sagert, I. & Chuna, T. J. Comput. Phys. (2024)
