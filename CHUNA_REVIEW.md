# Chuna Paper Reproduction — Review Package

**Date**: March 10, 2026 (v0.6.26 — upstream primal rewire + coralReef Iter 29)
**Author**: Kevin Mok (mokkevin@msu.edu)
**Hardware**: biomeGate — Threadripper 3970X, RTX 3090 (24 GB), Titan V (12 GB HBM2), ~$4K used parts
**License**: AGPL-3.0

---

## Summary

Three of Chuna's published papers reproduced in pure Rust + WGSL GPU shaders,
validated on consumer hardware. No ICER, no MILC, no CUDA, no vendor SDK.

**Core paper reproduction: 41/41 checks pass** (11 quenched flow + 20 dielectric + 10 kinetic-fluid).
**Dynamical N_f=4 extension: 3/3 pass** — warm-start with mass annealing, NPU-guided adaptive Omelyan HMC, 85% acceptance at target mass m=0.1.

| Paper | Citation | What we reproduced | Detailed artifact |
|-------|----------|--------------------|----|
| 43 | Bazavov & Chuna, arXiv:2101.05320 | Gradient flow w/ LSCFRK — coefficients derived independently, convergence verified, t₀/w₀ scale setting | [`whitePaper/baseCamp/chuna_gradient_flow.md`](whitePaper/baseCamp/chuna_gradient_flow.md) |
| 44 | Chuna & Murillo, Phys. Rev. E 111, 035206 | Standard, completed, multi-component Mermin dielectric. DSF vs MD validated against Murillo Group plasma DB | [`whitePaper/baseCamp/chuna_bgk_dielectric.md`](whitePaper/baseCamp/chuna_bgk_dielectric.md) |
| 45 | Haack, Murillo, Sagert & Chuna, J. Comput. Phys. (2024) | BGK relaxation + Euler/HLL + coupled kinetic-fluid interface. Conservation laws, H-theorem, Sod shock | [`whitePaper/baseCamp/chuna_kinetic_fluid.md`](whitePaper/baseCamp/chuna_kinetic_fluid.md) |

---

## Architecture

hotSpring is the validation application. The compute engine is
[**barraCuda**](https://github.com/ecoPrimals/barraCuda) (v0.3.3, `27011af`) — a standalone
Rust crate providing SU(3) lattice math, 791 WGSL shaders in 3-tier precision
(f32/DF64/f64), GPU pipeline dispatch, and cross-spring evolved physics ops.
[**toadStool**](https://github.com/ecoPrimals/toadStool) (S138) provides hardware
discovery, NPU dispatch, and shader proxy to [**coralReef**](https://github.com/ecoPrimals/coralReef)
(Phase 10, Iter 25) — the sovereign WGSL→native compiler (AMD E2E proven, NVIDIA pending).
barraCuda is pulled automatically via `Cargo.toml` (git dependency).

## Quick Start

```bash
git clone https://github.com/syntheticChemistry/hotSpring
cd hotSpring/barracuda

# All three papers in one run (~5 hours; 8⁴ + Papers 44/45 in ~1 hour, 16⁴ adds ~4 hours)
cargo run --release --bin validate_chuna_overnight

# Dynamical fermion extension only (~3.5 hours, skips quenched/GPU sections)
cargo run --release --bin validate_chuna_overnight -- --dynamical-only

# Individual papers
cargo run --release --bin validate_gradient_flow      # Paper 43
cargo run --release --bin validate_dielectric          # Paper 44
cargo run --release --bin validate_kinetic_fluid       # Paper 45

# Paper 44 extension: DSF vs Murillo Group MD data
cargo run --release --bin validate_dsf_vs_md
```

Requirements: Rust (stable), any Vulkan GPU with `SHADER_F64` support.
No manual dependency setup — `cargo build` fetches barraCuda from GitHub.

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

## Questions for Review

1. **Does the physics look right?** Do our convergence data, scale-setting
   values, dielectric curves, and conservation results match ICER/MILC?

2. **Did we derive the integrators correctly?** We solved the four Taylor
   order conditions independently. Our coefficients match the paper.

3. **Is the approach viable?** Pure Rust + WGSL, dispatched through Vulkan.
   No MILC, no Fortran, no CUDA.

---

## References

- Bazavov, A. & Chuna, T. arXiv:2101.05320 (2021)
- Chuna, T. & Murillo, M.S. Phys. Rev. E 111, 035206 (2024)
- Haack, J.R., Murillo, M.S., Sagert, I. & Chuna, T. J. Comput. Phys. (2024)
