# Experiment 044: Chuna BGK Dielectric Functions

**Date**: March 6, 2026
**Paper**: Chuna & Murillo, Phys. Rev. E 111, 035206 (2024), arXiv:2405.07871
**Status**: ✅ COMPLETED MERMIN — Python → CPU → GPU (19/19 Py, 25 CPU tests, GPU standard+completed, **144× CPU vs Py**)
**Priority**: P2 — extends Papers 1/5 (Sarkas MD, transport)

---

## Objective

Reproduce the conservative dielectric functions from the multicomponent
BGK (Bhatnagar-Gross-Krook) kinetic equation. Validate against known
plasma physics limits (Debye screening, Drude conductivity, f-sum rule).

## Evolution Path

```
Python (bgk_dielectric_control.py) → BarraCuda CPU (dielectric.rs) → BarraCuda GPU → sovereign
```

## What Was Implemented

### Python Control (19/19 checks)

| Check | Γ=1,κ=1 | Γ=10,κ=1 | Γ=10,κ=2 |
|-------|---------|----------|----------|
| Debye screening ε(k,0) = 1 + (k_D/k)² | ✓ exact | ✓ exact | ✓ exact |
| f-sum rule sign (∫ω Im[1/ε]dω < 0) | ✓ | ✓ | ✓ |
| Landau damping sign (Im[ε] ≥ 0) | ✓ 99.4% | ✓ 99.8% | ✓ 99.8% |
| DSF positivity S(k,ω) ≥ 0 | ✓ 98.6% | ✓ 99.7% | ✓ 99.7% |
| Drude conductivity σ(0) | ✓ exact | ✓ exact | ✓ exact |
| High-freq limit ε → 1 | ✓ | ✓ | ✓ |
| Dispersion (loss peak vs k) | ✓ monotonic | | |

### Physics Implemented

1. **Plasma dispersion function Z(z)**: Pure NumPy (no scipy), power
   series for |z|<6, asymptotic for |z|≥6
2. **Free-particle (Vlasov) susceptibility**: χ₀(k,ω) for Maxwellian
3. **Standard Mermin dielectric function**: With relaxation rate ν
4. **Dynamic structure factor**: S(k,ω) via fluctuation-dissipation
5. **Drude conductivity**: σ(ω) = ωₚ²/(4π(ν-iω))

### Known Limitations (Standard Mermin)

The standard Mermin function conserves particle number but NOT momentum.
This manifests as:
- **f-sum violation**: 65-97% depending on coupling (Chuna 2024, Sec. III)
- **S(k,ω) < 0** at ~1-2% of frequencies near plasma resonance

These are precisely the pathologies that Chuna's "completed Mermin"
(BGK with both number and momentum conservation) resolves.

## Rust Module: `dielectric.rs`

BarraCuda CPU implementation with:
- `plasma_dispersion_z(z)` — Z(z) via series/asymptotic
- `chi0_classical(k, omega, params)` — Vlasov susceptibility
- `epsilon_vlasov(k, omega, params)` — collisionless dielectric
- `epsilon_mermin(k, omega, nu, params)` — standard Mermin
- `f_sum_rule(k, nu, params)` — numerical integration
- `dynamic_structure_factor(k, omegas, nu, params)` — S(k,ω)

## GPU Promotion (March 6, 2026)

`physics/gpu_dielectric.rs` — batched Mermin ε(k,ω) on GPU. Each thread
evaluates one ω point through the full chain: Z(z) → W(z) → χ₀ → ε → Im[1/ε].

Uses barracuda's `math_f64.wgsl` polyfills for `exp_f64`/`sin_f64`/`cos_f64`
(native WGSL transcendentals are f32-only on NVK/NAK).

**Numerical stability**: For |z| ≥ 4, W(z) is computed directly via its
asymptotic expansion, avoiding catastrophic cancellation in 1 + z·Z(z).
This gives **better physics** than the CPU power series (100% DSF positivity
vs 98% on CPU; 100% passive-medium compliance).

| Shader | Purpose | Origin |
|--------|---------|--------|
| `dielectric_mermin_f64` | Batched ε(k,ω) + loss function | **NEW** |
| `complex_f64` | Complex arithmetic preamble | Reuse (HMC) |
| `math_f64` | f64 transcendental polyfills | barraCuda |

### GPU Physics Checks (12/12)

| Check | Weak (Γ=1) | Moderate (Γ=10) | Screened (κ=2) |
|-------|:---:|:---:|:---:|
| f-sum GPU vs CPU | 3.3% | 5.2% | 1.8% |
| DSF positivity | 100% | 100% | 100% |
| High-freq |loss| | 2.7e-7 | 2.7e-8 | 1.1e-7 |
| Passive medium | 100% | 100% | 100% |

## Completed Mermin Implementation (March 6, 2026)

The "completed" Mermin from Chuna & Murillo (2024, Eq. 26) adds momentum
conservation to the standard Mermin via a denominator correction:

```
Standard:  D = 1 + (iν/ω) × R
Completed: D = 1 + (iν/ω) × R × (1 - G_p)

where R   = W(z_ν)/W(0) = (ε₀(k,ω+iν)-1)/(ε₀(k,0)-1)
  and G_p = R × ω(ω+iν) / (k² v_th²)
```

### New Functions

- `epsilon_completed_mermin(k, omega, nu, params)` — CPU
- `dynamic_structure_factor_completed(k, omegas, nu, params)` — CPU
- `f_sum_rule_integral_completed(k, nu, params, omega_max)` — CPU
- `eps_completed_mermin(k, omega, nu)` — GPU (WGSL)
- GPU shader now accepts `use_completed` flag to switch modes

### Completed Mermin Test Results (7 new tests)

| Test | Result |
|------|--------|
| Recovers standard at ν→0 | ✅ diff < 0.01 |
| High-freq limit ε→1 | ✅ dev < 0.01 |
| Static limit matches standard | ✅ diff < 1e-10 |
| DSF ≥99% positive (Γ=1,10; κ=1,2) | ✅ all pass |
| Differs from standard at finite ν | ✅ diff > 1e-6 |
| Passive medium Im[ε] ≥ 0 | ✅ all ω/ωₚ |
| f-sum improvement or <10% error | ✅ |

### Python Control Updated

`epsilon_completed_mermin()` in `bgk_dielectric_control.py` now uses the
real momentum-conserving formula instead of the previous pass-through.

## Experiment 047: DSF vs MD Validation

**Status**: ✅ COMPLETE — 14/14 Rust checks passed

The analytical Mermin DSF has been compared against MD reference data from
the Dense Plasma Properties Database (MurilloGroupMSU):

| κ | Γ | q=0.54 | q=1.09 | q=1.99 |
|---|---|:------:|:------:|:------:|
| 2 | 31 | **Δ=0.008** (0.8%) ✓ | Δ=0.141 ✓ | Δ=0.490 · |
| 1 | 14 | Δ=0.312 ✓ | Δ=0.677 · | Δ=0.697 · |

**Best result**: κ=2, q=0.54 — completed Mermin peak position matches MD
within 0.8%. Mermin works best in the collective regime (small q, moderate κ).

The susceptibility χ(k,ω) comparison confirms the complementary picture:
Mermin overestimates χ amplitude but captures peak positions at small q.
See `experiments/047_DSF_VS_MD_VALIDATION.md` for full analysis.

## Remaining

1. **Multi-component extension**: Electron-ion plasmas
2. **Local field correction G(k)**: Would bring DSF and χ amplitudes
   into better agreement (STLS or QLCA self-consistent theory)

## Connection to Other Papers

- **Paper 1** (Sarkas MD): Produces S(k,ω) from simulation
- **Paper 5** (Stanton-Murillo): Transport coefficients from Green-Kubo
- **Paper 38** (NIF XRTS): S(k,ω) is THE measured quantity in XRTS
- **Paper 43** (Chuna gradient flow): Same author, different domain

## References

- Mermin, Phys. Rev. B 1, 2362 (1970)
- Chuna & Murillo, Phys. Rev. E 111, 035206 (2024), arXiv:2405.07871
- Stanton & Murillo, Phys. Rev. E 91, 033104 (2015)
