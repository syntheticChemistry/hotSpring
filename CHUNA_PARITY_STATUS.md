# Chuna Papers — Parity & Extensions Status

**Last Updated**: March 11, 2026 (v0.6.28 — upstream primal sync + coralReef Iter 30 + live Kokkos parity benchmark)
**Crate**: hotspring-barracuda v0.6.28 (847 lib tests, 112+ binaries)
**Rewired to**: barraCuda v0.3.4 (`a012076`), toadStool S145, coralReef Phase 10 Iter 30
**Sovereign pipeline**: coralReef live — **45/46** standalone shaders compile to native SM70/SM86 SASS (Iter 30). 12/12 NVVM bypass patterns. Full `GpuBackend` impl. IPC discovery wired. `sovereign-dispatch` feature available.
**Demo-ready**: **44/44 overnight checks pass** — core paper reproduction 41/41 (11 quenched flow + 20 dielectric + 10 kinetic-fluid). **Dynamical N_f=4 extension: 3/3 pass** (flow monotonic, acceptance 85%, plaquette 0.470). NPU-steered warm-start with mass annealing.
**Handoff**: `wateringHole/handoffs/HOTSPRING_V0628_UPSTREAM_SYNC_HANDOFF_MAR10_2026.md`

---

## Paper 43: SU(3) Gradient Flow Integrators

**Ref**: Bazavov & Chuna, arXiv:2101.05320 (2021)

| Tier | Status | Detail |
|------|:------:|--------|
| Python Control | ✅ | `gradient_flow_control.py` — 3 integrators, t₀ + w₀ |
| BarraCuda CPU | ✅ | 5 integrators, 14/14 tests, imports `barracuda::numerical::lscfrk` |
| BarraCuda GPU | ✅ | `gpu_flow.rs`, 7/7, **38.5× speedup** |
| Production 8⁴ | ✅ | `gradient_flow_production` — β=5.9,6.0,6.2 |
| Production 16⁴ | ✅ | Extended to 16⁴ lattices, 500 HMC thermalization |
| Convergence | ✅ | `bench_flow_convergence` — ε sweep 0.02→0.001, order verification |

### Completed (all)

- [x] Production scale: 8⁴ HMC thermalization + gradient flow (Exp 048)
- [x] Convergence benchmark: ε=0.02→0.001 for W6/W7/CK4 with order extraction
- [x] Extend to 16⁴ lattices with 500-step thermalization
- [x] **Dynamical N_f=4 staggered** (extension, not paper requirement): 8⁴ β=5.4,
      m=0.1. Warm-start mass annealing (m: 1.0→0.5→0.2→0.1), adaptive Omelyan
      with NPU steering. 85% acceptance, |ΔH|<0.5, 3/3 checks pass.
      See `NPU_STEERING_LESSONS.md` for full analysis
      produce 0% acceptance from hot start. Root cause: hot-start ↛ dynamical
      equilibrium without intermediate warm-up. |ΔH| ~ 6.5×10⁶ even at smallest dt.
      **Next step**: warm-start from thermalized quenched config (β=5.4 quenched ⟨P⟩~0.54),
      then enable fermion coupling with mass preconditioning. This is standard practice
      in production lattice QCD (MILC, CL2QCD).

---

## Paper 44: Conservative BGK Dielectric Functions

**Ref**: Chuna & Murillo, Phys. Rev. E 111, 035206 (2024), arXiv:2405.07871

| Tier | Status | Detail |
|------|:------:|--------|
| Python Control | ✅ | `bgk_dielectric_control.py` — completed Mermin |
| BarraCuda CPU | ✅ | `dielectric.rs` — 25 tests, standard + completed Mermin |
| BarraCuda GPU | ✅ | `dielectric_mermin_f64.wgsl` — stable W(z) |
| Completed Mermin | ✅ | Momentum conservation via Eq. 26 correction |
| Multi-component | ✅ | `dielectric_multicomponent.rs` — electron-ion, 6 tests |
| Multi-comp GPU | ✅ | `dielectric_multicomponent_f64.wgsl` + `gpu_dielectric_multicomponent.rs` |

### What's Done

**Standard Mermin** (number conservation only):
- f-sum rule sign correct, Debye screening exact (1e-12)
- DSF ≥98% positive, high-freq ε→1, dispersion monotonic
- GPU: stable W(z) via direct asymptotic expansion

**Completed Mermin** (number + momentum conservation):
- Momentum correction factor G_p = R × ω(ω+iν)/(k²v_th²)
- DSF ≥99% positive across all coupling regimes
- 7 new unit tests

**Multi-Component Mermin** (electron-ion):
- Species-indexed susceptibility with independent mass, charge, density, T, ν
- Reduces to single-species at self-consistent κ = √(3Γ)
- Debye screening correct for total k_D², f-sum rule validated
- DSF positivity ≥95%, passive medium compliance
- GPU shader with per-species loop and configurable momentum conservation
- 6 new unit tests

### Completed (all)

- [x] Compare analytical DSF vs MD S(k,ω) — Exp 047, 14/14 ✓
- [x] Multi-component extension (electron-ion) — CPU + GPU

---

## Paper 45: Multi-Species Kinetic-Fluid Coupling

**Ref**: Haack, Murillo, Sagert & Chuna, J. Comput. Phys. (2024)

| Tier | Status | Detail |
|------|:------:|--------|
| Python Control | ✅ | `kinetic_fluid_control.py` — 18/18 |
| BarraCuda CPU | ✅ | `kinetic_fluid.rs` — 16 tests + 20/20, 322× faster |
| GPU BGK | ✅ | `gpu_kinetic_fluid.rs` + `bgk_relaxation_f64.wgsl` |
| GPU Euler | ✅ | `gpu_euler.rs` + `euler_hll_f64.wgsl` (HLL Riemann solver) |
| GPU Coupled | ✅ | `gpu_coupled_kinetic_fluid.rs` (BGK + Euler + interface) |

### What's Done

**CPU**: Full 3-phase validation:
- Phase 1: Multi-species BGK relaxation (mass/momentum/energy, H-theorem)
- Phase 2: 1D Euler/Sod shock tube (HLL Riemann solver, shock detection)
- Phase 3: Coupled kinetic-fluid interface (density continuity)

**GPU BGK**: Two-pass pipeline:
- Pass 1: Velocity-space moment contributions (parallel per v-point)
- Pass 2: Maxwellian target + BGK relaxation
- Uses `math_f64.wgsl` polyfills for `exp_f64`

**GPU Euler** (NEW):
- `euler_hll_f64.wgsl`: HLL flux computation + conservative update
- Two-pass: flux at interfaces → conservative cell update
- Validates against CPU Sod shock tube (mass/energy conservation)

**GPU Coupled** (NEW):
- Full pipeline: kinetic advection (CPU) → BGK collision (GPU) →
  interface (CPU) → Euler update (GPU) → boundary injection (CPU)
- Validates mass, momentum, energy conservation against CPU reference
- Interface density matching between kinetic and fluid domains

### Completed (all)

- [x] GPU BGK relaxation
- [x] GPU Euler update shader (HLL Riemann solver)
- [x] Full coupled kinetic-fluid GPU pipeline

---

## Overnight Validation — 44/44 (Core 41/41, Extension 3/3)

```bash
cargo run --release --bin validate_chuna_overnight 2>&1 | tee chuna_overnight.log
```

Runs all Paper 43/44/45 systems in one binary:
- P43: convergence sweep + production flow (8⁴ + 16⁴) + dynamical N_f=4 staggered flow
- P44: standard/completed Mermin (CPU+GPU) + multi-component (CPU+GPU)
- P45: GPU BGK + GPU Euler/Sod + GPU coupled kinetic-fluid

**Result (v0.6.24, March 9 2026)**: **44/44 checks pass** (11+20+10 core + 3/3 dynamical ext), exit code 0. Dynamical N_f=4 extension complete — warm-start mass annealing (m: 1.0→0.5→0.2→0.1), NPU-steered adaptive Omelyan, 85% acceptance at target mass m=0.1.

### Critical Fixes in v0.6.22→v0.6.23

1. **`cscale` shader fix**: 6 instances of incorrect complex scalar multiplication in `dielectric_multicomponent_f64.wgsl` were zeroing imaginary parts. Multi-component agreement went from 4% → 100%.
2. **Interface sub-iteration**: 3-iteration convergence loop in CPU + GPU kinetic-fluid coupling (physics-based, not hand-tuned).
3. **Precise pipeline routing**: Multi-component Mermin uses `create_pipeline_f64_entry_precise` (no FMA fusion).
4. **Deep debt**: Zero clippy warnings, zero library panics, `log` crate for GPU diagnostics, named constants throughout.

---

## Overall Progress

```
Paper 43: [██████████] 100% — COMPLETE
Paper 44: [██████████] 100% — COMPLETE (incl. multi-component)
Paper 45: [██████████] 100% — COMPLETE (incl. GPU Euler + coupled)
```

### Test & Binary Counts

| Component | Count |
|-----------|:-----:|
| Lib tests (total) | 769 |
| Dielectric (P44) | 25 + 6 multicomp |
| Kinetic-fluid (P45) | 16 |
| Gradient flow (P43) | 14 |
| Euler layout (P45) | 1 |
| Binaries | 101+ |

### coralReef Sovereign Compilation Coverage

| Category | Result |
|----------|--------|
| Standalone-parseable shaders | **45/46** compile to SM70/SM86 SASS (Iter 30) |
| Template-dependent shaders | 28 (expected — preprocessed inline) |
| Compilation gap | `complex_f64` CoralReefDevice compile failure (1 shader) |
| NVVM bypass patterns | 12/12 pass (3 poisoning patterns × 6 GPU targets) |
| `ComputeDevice: Send + Sync` | ✅ Resolved (Iter 26) — full `GpuBackend` impl active |
| DRM dispatch | Nouveau EINVAL on GV100 channel creation; amdgpu E2E ready |

*AGPL-3.0-only*
