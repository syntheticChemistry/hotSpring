# hotSpring → toadStool / barraCuda Handoff: Gradient Flow, Science Ladder, Derived Integrators

**Date:** March 6, 2026
**From:** hotSpring v0.6.17+ (Exp 032 + gradient flow + Chuna reproduction)
**To:** toadStool, barraCuda, coralReef teams
**License:** AGPL-3.0-only
**Covers:** Asymmetric lattice HMC, Wilson gradient flow, LSCFRK integrators, N_f=4 dynamical infrastructure, RHMC, science ladder

---

## Executive Summary

- **Asymmetric lattices** (N_s³ × N_t) validated on GPU with 26-36× speedup across 8 geometries
- **Wilson gradient flow** implemented with t₀ (Lüscher) and w₀ (BMW) scale setting
- **LSCFRK integrators derived from first principles** — `const fn derive_lscfrk3(c2, c3)` solves order conditions at compile time; zero magic numbers for 3rd-order schemes
- **5 integrators** (Euler, RK2, LSCFRK3W6, LSCFRK3W7, LSCFRK4CK) validated, convergence scaling matches Chuna arXiv:2101.05320
- **N_f=4 staggered dynamical fermion HMC** infrastructure complete on GPU (Dirac, CG, pseudofermion, trajectory loop)
- **RHMC** (rational approximation + multi-shift CG) implemented on CPU for fractional flavors
- **7 new binaries** registered: `bench_backends`, `production_finite_temp`, `production_gradient_flow`, `compare_flow_integrators`, `validate_gradient_flow`, `production_dynamical`
- **Science ladder**: Quenched → Flow → Integrators → N_f=4 → N_f=2 → N_f=2+1 → N_f=2+1+1

---

## Part 1: Asymmetric Lattice GPU HMC

hotSpring's `Lattice` struct already supported `dims: [Nx, Ny, Nz, Nt]`, but production
binaries and GPU HMC were only tested on hypercubic (L⁴). Exp 032 extends to physically
meaningful finite-temperature geometries.

### GPU Scaling (verified March 6 2026)

| Lattice | Volume | CPU ms/traj | GPU ms/traj | Speedup |
|---------|:------:|:-----------:|:-----------:|:-------:|
| 4⁴ | 256 | 71.7 | 18.7 | 3.8× |
| 8³×4 | 2,048 | 580.6 | 18.4 | 31.5× |
| 16³×4 | 16,384 | 4,708.4 | 173.6 | 27.1× |
| 16³×8 | 32,768 | 9,293.9 | 359.6 | 25.8× |
| 32³×4 | 131,072 | 37,768.7 | 1,451.1 | 26.0× |
| 32³×8 | 262,144 | 75,124.8 | 2,840.2 | 26.5× |

### barraCuda action: Asymmetric lattice support

The WGSL shaders use `dims[0..4]` uniforms throughout — no hardcoded assumption
of cubic geometry. barraCuda's lattice shaders should validate asymmetric dispatch
if they haven't already. Key shaders affected:

- `su3_hmc_force_f64.wgsl` — site-indexing uses `dims`
- `su3_plaquette_f64.wgsl` — loop over 4D neighbors
- `su3_hmc_leapfrog_f64.wgsl` — momentum update
- `polyakov_loop_f64.wgsl` — N_t-direction product

---

## Part 2: Wilson Gradient Flow — Derived Integrators

### What was implemented

File: `barracuda/src/lattice/gradient_flow.rs`

- **Wilson flow**: evolves gauge field under gradient flow dV/dt = -g²₀ ∂S/∂V
- **E(t)**: action density at flow time t
- **t²⟨E(t)⟩**: dimensionless observable for scale setting
- **t₀**: Lüscher scale (t²⟨E⟩ = 0.3)
- **w₀**: BMW scale (W(t) = t d/dt [t²⟨E⟩] = 0.3)
- **W(t) function**: computed via numerical derivative of flow measurements

### LSCFRK Derivation (the key insight)

**Problem**: The coefficients A = [0, -17/32, -32/27] and B = [1/4, 8/9, 3/4]
for Lüscher's integrator appeared as magic numbers copied from the paper.

**Solution**: These are the unique solutions to four algebraic constraints
(Taylor series order conditions for 3rd-order accuracy):

1. b₁ + b₂ + b₃ = 1
2. b₂c₂ + b₃c₃ = 1/2
3. b₂c₂² + b₃c₃² = 1/3
4. b₃ a₃₂ c₂ = 1/6

Given two free parameters (c₂, c₃), everything else follows by algebra.
The 2N-storage (Williamson) constraint converts from Butcher tableau to (A, B) form.

```rust
const fn derive_lscfrk3(c2: f64, c3: f64) -> ([f64; 3], [f64; 3]) {
    let b3 = (1.0/3.0 - c2/2.0) / (c3 * (c3 - c2));
    let b2 = (0.5 - b3*c3) / c2;
    let a32 = 1.0 / (6.0 * b3 * c2);
    let a31 = c3 - a32;
    let a21 = c2;
    let big_b1 = a21;
    let big_b2 = a32;
    let big_b3 = b3;
    let big_a1 = 0.0;
    let big_a2 = (a31 - big_b1) / big_b2;
    let big_a3 = (b2 - big_b2) / big_b3;
    ([big_a1, big_a2, big_a3], [big_b1, big_b2, big_b3])
}
```

**Tests verify**: derivation reproduces published W6/W7 coefficients exactly,
and all four order conditions are satisfied to machine precision.

### barraCuda action: Absorb gradient flow primitives

1. `derive_lscfrk3` as a `const fn` utility — any spring doing ODE integration benefits
2. Generic `lscfrk_step` function — one code path for any 2N-storage scheme
3. Flow measurement pipeline (E(t), t₀, w₀) — reusable for any gauge theory

### Note on CK4

The 4th-order Carpenter-Kennedy (LSCFRK4CK) coefficients cannot be derived in closed
form — the order conditions are nonlinear at 4th order. The integer ratios
(e.g. 567301805773/1357537059087) are exact representations from numerical
root-finding (NASA TM-109112, 1994). This is documented in the code.

---

## Part 3: N_f=4 Staggered Dynamical Fermion Infrastructure

### What exists (CPU + GPU)

| Component | File | Status |
|-----------|------|--------|
| Staggered Dirac operator | `lattice/dirac.rs` + GPU shader | ✅ CPU + GPU |
| CG solver for D†D | `lattice/cg.rs` + GPU shaders | ✅ CPU + GPU (15,360× readback reduction) |
| Pseudofermion heat bath | `lattice/pseudofermion/` | ✅ CPU |
| Fermion force | `lattice/gpu_hmc/dynamical.rs` | ✅ GPU |
| Dynamical HMC trajectory | `lattice/gpu_hmc/dynamical.rs` | ✅ GPU (`gpu_dynamical_hmc_trajectory`) |
| Production binary | `src/bin/production_dynamical.rs` | ✅ Compiled |

### What exists for fractional flavors

| Component | File | Status |
|-----------|------|--------|
| Rational approximation | `lattice/rhmc.rs` | ✅ CPU |
| Multi-shift CG | `lattice/rhmc.rs` | ✅ CPU |
| N_f=2 via rooting trick | — | Pending (wire RHMC into HMC) |

### barraCuda action: GPU RHMC

The multi-shift CG solver `(D†D + σ_i)x_i = b` is CPU-only. GPU acceleration
would enable N_f=2, 2+1 production runs. The solver structure is:
- One matrix-vector product per iteration (same as standard CG)
- Multiple solution vectors updated simultaneously (one per shift σ_i)
- Convergence checked independently per shift

---

## Part 4: Science Ladder — Where hotSpring Stands

| Level | Physics | hotSpring Status | What barraCuda Needs |
|-------|---------|-----------------|---------------------|
| 0 | Quenched HMC | ✅ 32⁴, 32³×8, 64³×8 | Nothing — done |
| 1 | Gradient flow (t₀, w₀) | ✅ 5 integrators, 14/14 tests | Absorb `derive_lscfrk3`, `lscfrk_step` |
| 2 | Chuna integrator convergence | ✅ Matches paper | Nothing — validated |
| 3 | N_f=4 staggered dynamical | ✅ Infrastructure complete | Nothing — runs on existing GPU CG |
| 4 | N_f=2 dynamical (RHMC) | Pending | GPU multi-shift CG |
| 5 | N_f=2+1 (strange quark) | Pending | GPU RHMC + mass tuning |
| 6 | N_f=2+1+1 HISQ | Long-term | HISQ action (Naik term) |

---

## Part 5: Binaries Created This Session

| Binary | Purpose | Cargo.toml |
|--------|---------|-----------|
| `bench_backends` | CPU vs GPU backend comparison | ✅ Added |
| `production_finite_temp` | Asymmetric N_s³×N_t β-scan with adaptive dt | ✅ Added |
| `production_gradient_flow` | Quenched gradient flow (t₀, w₀ measurement) | ✅ Added |
| `compare_flow_integrators` | 5-integrator convergence comparison | ✅ Added |
| `validate_gradient_flow` | Gradient flow validation (Euler/RK2/RK3 cross-check) | ✅ Added |
| `production_dynamical` | N_f=4 staggered dynamical fermion HMC on GPU | ✅ Added |

---

## Part 6: Lessons for barraCuda/toadStool Evolution

### 1. Derive, don't copy

The LSCFRK coefficients taught us: if a number has an algebraic derivation,
encode the derivation, not the result. `const fn` in Rust makes this zero-cost.
barraCuda should adopt this pattern for any integrator coefficients, quadrature
weights, or numerical method parameters that have closed-form solutions.

### 2. Precision-per-observable

Gradient flow and scale setting are precision-sensitive — the W(t) function
requires numerical differentiation of t²⟨E⟩. DF64 is essential here. The
`compile_shader_universal()` precision system should support flow observables.

### 3. Adaptive dt for HMC

`production_finite_temp.rs` implements adaptive dt adjustment based on acceptance
rate (reduce if <55%, increase if >85%). This pattern should be a first-class
barraCuda feature, not per-spring reimplementation.

### 4. Resume capability

`production_finite_temp` supports `--resume-from` to continue interrupted runs.
Long-running lattice QCD production needs checkpoint/resume as infrastructure.

---

## Action Items

### toadStool action:
- [ ] Validate asymmetric lattice dispatch in GPU HMC shaders
- [ ] Add `--resume-from` checkpoint/resume infrastructure

### barraCuda action:
- [ ] Absorb `derive_lscfrk3` as a `const fn` utility
- [ ] Absorb generic `lscfrk_step` for 2N-storage Lie group integrators
- [ ] Absorb flow measurement pipeline (E(t), t₀, w₀)
- [ ] GPU multi-shift CG solver for RHMC (N_f=2, 2+1)
- [ ] Adaptive HMC dt as first-class feature

### coralReef action:
- [ ] When coralDriver lands, gradient flow is a natural Titan V target (f64-heavy, embarrassingly parallelizable flow force computation)

---

*hotSpring proves consumer hardware can reproduce Chuna's academic gradient flow
integrators — derived from first principles, validated against the paper, running
on a $1,500 GPU. The science ladder from quenched through N_f=4 dynamical is
built. The coefficients are not magic — they are algebra.*
