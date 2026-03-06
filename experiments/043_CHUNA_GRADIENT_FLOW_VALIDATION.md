# Experiment 043: Chuna Gradient Flow Validation

**Date**: March 6, 2026
**Paper**: Bazavov & Chuna, arXiv:2101.05320 (2021)
**Status**: ✅ GPU COMPLETE — Python → CPU → GPU (5 integrators, 14/14 CPU tests, 7/7 GPU checks, **38.5× GPU speedup**)
**Priority**: P2 — extends Papers 8-12 (lattice QCD)

---

## Objective

Reproduce the low-storage commutator-free Runge-Kutta (LSCFRK) integrators
for Wilson gradient flow on SU(3) gauge fields. Validate coefficient
derivation, integrator convergence, and scale-setting observables (t₀, w₀).

## Evolution Path

```
Python (gradient_flow_control.py) → BarraCuda CPU (gradient_flow.rs) → BarraCuda GPU → sovereign
```

## What Was Implemented

### Integrators (5 total, derived from first principles)

| Integrator | Order | Stages | Free Parameters | Source |
|-----------|:-----:|:------:|:---------------:|--------|
| Euler | 1 | 1 | — | Reference |
| RK2 | 2 | 2 | — | Standard |
| RK3 Lüscher (LSCFRK3W6) | 3 | 3 | c₂=1/4, c₃=2/3 | Lüscher JHEP 2010 |
| LSCFRK3W7 (Chuna) | 3 | 3 | c₂=1/3, c₃=3/4 | Bazavov & Chuna 2021 |
| LSCFRK4CK | 4 | 5 | Carpenter-Kennedy 1994 | Numerical roots |

### Key Result: Compile-Time Coefficient Derivation

`const fn derive_lscfrk3(c2, c3)` solves the four 3rd-order Taylor series
conditions at compile time. Only two free parameters (c₂, c₃) remain — the
rest is algebra. This IS the derivation, not a lookup table.

### Scale Observables

- **t₀**: t²⟨E(t)⟩ = 0.3 (Lüscher 2010)
- **w₀**: t d/dt[t²E(t)] = 0.3 (BMW, arXiv:1203.4469) — Chuna's primary observable

### Validation (14/14 tests)

| Test | What | Result |
|------|------|--------|
| cold_start_zero_energy | E(cold) = 0 | ✓ |
| euler_flow_smooths | E decreases under flow | ✓ |
| rk3_flow_smooths_faster_than_euler | Higher-order accuracy | ✓ |
| flow_preserves_unitarity | U†U = I after 50 steps | ✓ (< 1e-10) |
| t2_e_increases_monotonically | Dimensionless observable | ✓ |
| find_t0_on_hot_start | t₀ > 0 found | ✓ |
| find_w0_on_hot_start | w₀ > 0 found | ✓ |
| lscfrk3w7_agrees_with_luscher | W6 ≈ W7 at ε=0.01 | ✓ |
| lscfrk4ck_fourth_order | CK45 ≈ W7 at ε=0.01 | ✓ |
| derivation_produces_known_w6_coefficients | A,B match paper | ✓ |
| derivation_produces_known_w7_coefficients | A,B match paper | ✓ |
| order_conditions_satisfied_for_w7 | 4 conditions + 2 row sums | ✓ |
| luscher_and_w6_enum_produce_same_result | Bit-identical | ✓ |
| w_function_monotonic_increasing | W(t) increases | ✓ |

## What's Next

1. **GPU promotion**: Gradient flow kernels as WGSL shaders (reuses SU(3) GPU primitives)
2. **Production scale**: t₀/w₀ on 16⁴+ thermalized configs for physical scale setting
3. **Convergence benchmark**: Step-size refinement study (ε=0.02 → 0.001) comparing W6 vs W7

## Python Control

`control/gradient_flow/scripts/gradient_flow_control.py`:
- Algorithm-identical to Rust (same LCG PRNG, Cayley exponential, gauge force)
- Three integrators: Euler, RK3 Lüscher (W6), LSCFRK3W7 (Chuna)
- Validates coefficients, flow smoothing, t₀, w₀, monotonicity
- Output: `control/gradient_flow/results/gradient_flow_control.json`

## GPU Promotion (March 6, 2026)

`lattice/gpu_flow.rs` — only 1 new WGSL shader (`su3_flow_accumulate_f64`), all
other operations reuse HMC's GPU infrastructure:

| Shader | Purpose | Origin |
|--------|---------|--------|
| `su3_gauge_force_f64` | Flow force Z = -∂S/∂U | HMC (reuse) |
| `su3_flow_accumulate_f64` | K = Aᵢ K + Z | **NEW** |
| `su3_link_update_f64` | U = exp(ε Bᵢ K) U | HMC (reuse) |
| `wilson_plaquette_f64` | E(t) measurement | HMC (reuse) |

### GPU vs CPU Parity (4⁴ lattice)

| Integrator | Plaquette Δ | Status |
|------------|:-----------:|:------:|
| Euler | 1.27e-9 | ✅ |
| RK3 Lüscher (W6) | 1.29e-9 | ✅ |
| LSCFRK3W7 (Chuna) | 7.67e-10 | ✅ |
| LSCFRK4CK | 2.91e-10 | ✅ |

### Benchmark (8⁴ lattice, W7, t=0→1.0)

| Substrate | Time | Speedup |
|-----------|:----:|:-------:|
| CPU (single core) | 5.56s | 1.0× |
| GPU (RTX 3090) | 0.14s | **38.5×** |

Cross-spring evolution: gradient flow GPU reuses the same shader pipeline as
HMC (hotSpring) which was already adopted by neuralSpring for spectral analysis.

## References

- Lüscher, JHEP 08 (2010) 071 — Wilson flow definition and t₀ scale
- BMW Collaboration, arXiv:1203.4469 — w₀ scale
- Bazavov & Chuna, arXiv:2101.05320 — LSCFRK integrators
- Williamson, J. Comput. Phys. 35, 48 (1980) — 2N-storage RK
- Carpenter & Kennedy, NASA TM-109112 (1994) — 4th-order 5-stage coefficients
