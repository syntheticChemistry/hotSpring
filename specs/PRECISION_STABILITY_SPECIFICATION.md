# hotSpring Precision & Stability Specification

> **HISTORICAL** — Snapshot from March 6, 2026 (v0.6.19). Current state: v0.6.32, 870 tests.
> Precision strategy and cancellation audit remain valid; test/binary counts superseded.
> Retained as fossil record of precision architecture decisions.

**Date**: March 6, 2026
**Version**: hotSpring v0.6.19 — 724 tests, 12/12 GPU physics checks, 9/9 cancellation families audited
**Domain**: Numerical stability across f32, DF64, f64, and encrypted compute
**License**: AGPL-3.0-only

---

## Purpose

hotSpring is the precision physics spring. Our domain — plasma physics,
nuclear structure, lattice QCD, spectral theory — demands the highest
numerical fidelity in the ecoPrimals ecosystem. This specification documents
what we've proven about numerical stability and how other springs can rely
on our precision for their own domains.

**Precise reality**: if the math is stable, the physics is predictable.
If the physics is predictable, every spring that depends on it can trust
the numbers they get.

---

## The Stability Guarantee

### What We Proved (March 6, 2026)

The plasma dispersion function W(z) = 1 + z·Z(z) is the canonical
cancellation-prone special function in plasma physics. At z ≈ 5.5,
the naive computation subtracts two values that are 60× larger than
their difference.

| Precision | Naive Algorithm | Stable Algorithm | Digits Preserved |
|-----------|:---:|:---:|:---:|
| **f32** (7 digits) | -8.03e7 (GARBAGE) | -1.689e-2 (correct) | **7/7** |
| **DF64** (14 digits) | ~12 digits (cancellation eats 2) | 14/14 (predicted) | **14/14** |
| **f64** (15 digits) | wrong sign (cancellation eats all) | -1.689e-2 (correct) | **15/15** |
| **CKKS FHE** | infeasible (80 mults, 10⁹× noise) | feasible (10 mults, 1× noise) | scheme-dependent |

The stable algorithm (direct asymptotic expansion for |z| ≥ 4) preserves
**all available precision** at every floating-point tier. No cancellation
amplification. No FMA sensitivity. No GPU-vs-CPU divergence.

### Evidence

- 5 unit tests in `dielectric.rs` covering f32 and f64 stability
- `validate_gpu_dielectric` binary: 12/12 physics checks on RTX 3090
- 100% DSF positivity on GPU (vs 98% on CPU with naive algorithm)
- 100% passive-medium compliance (no unphysical gain)
- f32-vs-f64 parity: 1e-7 relative error (full f32 precision)

---

## hotSpring's Precision Contribution to the Ecosystem

### What Each Spring Specializes In

Every spring evolves different mathematics. hotSpring owns the hardest
precision problems because our physics has the most extreme dynamic ranges.

| Spring | Domain Math | Precision Challenge | Depends on hotSpring |
|--------|-----------|-------------------|--------------------|
| **hotSpring** | Plasma dispersion, lattice QCD, nuclear pairing, spectral theory | Catastrophic cancellation, f64 GPU parity, DF64 stability | — (we are the source) |
| **groundSpring** | Chi-squared, noise characterization, RMSE, error prediction | Tail probabilities (erfc), accumulated rounding in long series | erfc_f64 stability, chi-squared CDF precision |
| **wetSpring** | Shannon entropy, HMM, Bray-Curtis, ODE biosystems | log(p) near p≈0, exp overflow in forward/backward | log_f64 fix (absorbed), exp_f64 polyfill |
| **neuralSpring** | Softmax, matrix correlation, spectral diagnostics | exp(x)/Σexp(x) near equal logits, batch IPR | log-sum-exp trick, batch_ipr_f64 stability |
| **airSpring** | Van Genuchten, Richards PDE, FAO-56 ET₀ | Power-law near singularities, seasonal accumulation | Stable f64 ODE integration pattern |

### How groundSpring Benefits from Our Precision

groundSpring validates ALL springs — it is the noise/error characterization
layer. Its reliability depends on the precision of its statistical tests:

```
groundSpring chi-squared test
  → uses chi_squared_f64.wgsl (CDF + quantile)
    → CDF uses erfc (complementary error function)
      → erfc(x) = 1 - erf(x) for large x has CANCELLATION
        → same instability pattern as W(z) = 1 + z·Z(z)
```

If groundSpring's chi-squared CDF is unstable at the tails, its error
prediction system gives wrong p-values. Wrong p-values mean wrong
pass/fail decisions across ALL springs. hotSpring's stability work
directly enables reliable error prediction.

**Specific debt item**: groundSpring should adopt the same pattern
(direct asymptotic expansion for erfc at large x) to avoid the
1 - erf(x) cancellation. barraCuda ISSUE-006 tracks this.

---

## Precision Tiers

### Tier Definitions

| Tier | Representation | Digits | Hardware | Throughput (RTX 3090) |
|------|---------------|:---:|----------|:---:|
| F16 | IEEE half | 3-4 | Tensor cores | ~142 TFLOPS |
| F32 | IEEE single | 7 | CUDA/shader cores | ~35.6 TFLOPS |
| DF64 | f32 pair (Knuth/Dekker) | 14 | f32 cores (2-4 ops/elem) | ~8.9 TFLOPS |
| F64 | IEEE double | 15-16 | f64 ALUs (1:64 ratio) | ~0.56 TFLOPS |

### Precision Routing Policy

hotSpring physics code uses the MINIMUM precision that produces correct
physics. The stable algorithm determines which tier is sufficient:

| Computation | Required Digits | Tier | Rationale |
|-------------|:---:|:---:|-----------|
| MD force integration | 7 | F32 | Forces are O(1), no cancellation |
| Yukawa pair potential | 7 | F32 | exp(-κr)/r is monotonic |
| Plasma dispersion W(z) | 7 (with stable algo) | F32 | Asymptotic avoids cancellation |
| HMC gauge force | 14 | DF64 | SU(3) unitarity requires >7 digits |
| Lattice plaquette sum | 14 | DF64 | Global sum of small differences |
| CG solver residual | 15 | F64 | Convergence test needs full precision |
| Nuclear binding energy | 15 | F64 | 1e-3 MeV accuracy on O(1000 MeV) scale |

This is NOT about choosing the "best" precision — it's about choosing the
cheapest precision that the STABLE algorithm can use correctly. The stability
guarantee makes this choice safe.

---

## Stability Verification Checklist

For any new physics module promoted to GPU, verify:

### 1. Identify Cancellation Points

Scan for expressions of the form:
- `1 + (near -1)` — susceptibility kernels, response functions
- `A - B` where `|A| ≈ |B| ≫ |A-B|` — energy differences, phase factors
- `exp(-x²)` × (large polynomial) — Gaussian-weighted sums

### 2. Test at Critical z Values

For each special function, identify the cancellation region and test:
```rust
#[test]
fn w_stable_correct_at_cancellation_region() {
    for &z_re in &[4.0, 5.0, 5.57, 6.0, 8.0, 10.0] {
        let z = Complex::new(z_re, 0.197);
        let stable = plasma_dispersion_w_stable(z);
        assert!(stable.abs() < 0.1);
        assert!(stable.re < 0.0);
    }
}
```

### 3. Test f32-vs-f64 Parity

The stable algorithm should agree across precision tiers:
```rust
#[test]
fn w_stable_f32_vs_f64_parity() {
    for &z_re in &[5.0, 5.57, 6.0, 8.0, 10.0] {
        let w_f64 = plasma_dispersion_w_stable(Complex::new(z_re, 0.197));
        let w_f32 = w_stable_f32(C32::new(z_re as f32, 0.197));
        assert!((w_f32.re as f64 - w_f64.re).abs() / w_f64.re.abs() < 1e-4);
    }
}
```

### 4. Validate Physics Properties

Always validate PHYSICS, not CPU parity:
- Conservation laws (f-sum, energy, charge)
- Positivity (DSF, probabilities, densities)
- Asymptotic limits (high-freq → 1, low-temp → ground state)
- Causality (Kramers-Kronig, retarded response)

### 5. Document Precision Requirements

Every GPU-promoted module must state:
- Minimum precision tier (F32 / DF64 / F64)
- Cancellation regions identified (or "none found")
- Stable algorithm used (or "no cancellation — naive is safe")
- Physics validation checks (list)

---

## Full Cancellation Inventory (Experiment 046)

9 cancellation-prone computation families identified across hotSpring's
70+ WGSL shaders and 40+ validation binaries.

### Tier A: Severe (algorithm fails at f32)

| ID | Function | Fix | Status |
|----|----------|-----|:---:|
| C1 | Plasma W(z) = 1+z*Z(z) | Direct asymptotic for \|z\|≥4 | ✅ FIXED |
| C2 | BCS v² = 0.5*(1-eps/e_qp) | Stable: Δ²/(2·E_qp·(E_qp+\|ε\|)) | ✅ FIXED |
| C3 | Jacobi diff = aqq-app | Degenerate guard (t=±1 at \|diff\|<1e-14) | ✅ GUARDED |

### Tier B: Moderate (reduced accuracy)

| ID | Function | Mitigation | Status |
|----|----------|------------|:---:|
| C4 | Flow energy (1-plaq)*6 | Acceptable at f64 for t₀ physics | ✅ Documented |
| C5 | Mermin (eps-1) ratio | Already mitigated via chi0 | ✅ Mitigated |
| C6 | SU(3) Gram-Schmidt | Theoretical risk; CGS ok in practice | ✅ Documented |

### Tier C: Low (no fix needed)

| ID | Function | Status |
|----|----------|:---:|
| C7 | Yukawa (1+κr) | OK — κr > 0.1 in all physical configs |
| C8 | ESN (1-α)*state | OK — α ∈ [0.1, 0.5], no cancellation |
| C9 | Sigmoid overflow | Clamp exists |

---

## Stability Test Results (Current)

### Plasma Dispersion Function (Paper 44) — 5 tests

| Test | Status | What It Proves |
|------|:---:|----------------|
| `w_stable_avoids_cancellation_f32` | ✅ | f32 naive = GARBAGE, stable = correct |
| `w_stable_vs_naive_convergence_region_f32` | ✅ | f32 stable correct at all large z |
| `w_stable_correct_at_cancellation_region` | ✅ | f64 stable correct at z=4-10 |
| `w_stable_f32_vs_f64_parity` | ✅ | f32-f64 agreement to 1e-7 |
| `naive_has_cancellation_error_stable_does_not` | ✅ | Naive shows >1% error at z=5.5 |

### BCS Occupation v² (HFB Nuclear Pairing) — 5 tests

| Test | Status | What It Proves |
|------|:---:|----------------|
| `bcs_v2_stable_matches_naive_near_fermi_surface` | ✅ | Both formulas agree near ε≈0 |
| `bcs_v2_naive_loses_precision_at_tails` | ✅ | Naive loses ~4 digits at eps=100 |
| `bcs_v2_f32_naive_garbage_at_tails` | ✅ | f32 naive produces 0, stable correct |
| `bcs_v2_stable_preserves_symmetry` | ✅ | v²(+ε) + v²(-ε) = 1 to 1e-14 |
| `bcs_v2_stable_range_extreme` | ✅ | v² ∈ [0,1] at eps=±1e6 |

### GPU Physics Checks (Paper 44 Mermin Dielectric)

| Check | Weak (Γ=1) | Moderate (Γ=10) | Screened (κ=2) |
|-------|:---:|:---:|:---:|
| f-sum GPU vs CPU | 3.3% | 5.2% | 1.8% |
| DSF positivity | 100% | 100% | 100% |
| High-freq |loss| | 2.7e-7 | 2.7e-8 | 1.1e-7 |
| Passive medium | 100% | 100% | 100% |

### GPU Physics Checks (Paper 43 SU(3) Gradient Flow)

| Check | Result |
|-------|--------|
| GPU-CPU parity (7/7) | ✅ All pass, bit-level agreement |
| Speedup | 38.5× on RTX 3090 |

---

## CKKS FHE Depth Analysis

| Computation | Naive Depth | Stable Depth | Noise Amplification | FHE Feasible? |
|-------------|:---:|:---:|:---:|:---:|
| W(z) power series | ~80 | ~10 (asymptotic) | 10⁹× → 1× | No → **Yes** |
| BCS v² | ~3 (1 div, 1 sub) | ~3 (1 mul, 1 add, 1 div) | cancellation → none | Yes → **Yes (better)** |
| Jacobi rotation | ~5 | ~5 (guard = constant) | varies → bounded | Yes |
| Flow energy | ~1 | ~1 | 10⁴× (single op) | Yes |

---

## Modules with Verified Stability

| Module | Precision Tier | Cancellation | Algorithm | Status |
|--------|:---:|:---:|-----------|:---:|
| `physics/gpu_dielectric.rs` | F64 | Yes (z>4) | Asymptotic W(z) | ✅ |
| `physics/hfb_common.rs` | F64 | Yes (eps>>Δ) | Stable BCS v² | ✅ |
| `batched_hfb_density_f64.wgsl` | F64 | Yes (eps>>Δ) | Stable BCS v² | ✅ |
| `bcs_bisection_f64.wgsl` | F64 | Yes (eps>>Δ) | Stable BCS v² | ✅ |
| `deformed_density_energy_f64.wgsl` | F64 | Yes (eps>>Δ) | Stable BCS v² | ✅ |
| `batched_eigh_nak_optimized_f64.wgsl` | F64 | Yes (degenerate) | Guard: t=±1 | ✅ |
| `lattice/gpu_flow.rs` | F64 | Mild (1-plaq) | Acceptable at t₀ | ✅ |
| `lattice/hmc.rs` | DF64 | Minor (SU(3)) | Cayley-Hamilton | ✅ |
| `physics/dielectric.rs` (CPU) | F64 | Yes (z>4) | Updated to stable | ✅ |

### Modules Needing Stability Audit

| Module | Risk | Specific Concern |
|--------|------|------------------|
| `physics/kinetic_fluid.rs` | Medium | Moment integrals near zero |
| `lattice/dirac.rs` | Medium | CG near-zero modes |
| `spectral/anderson.rs` | Low | IPR is ratio, no cancellation |

---

## Cross-References

| Document | Location | Content |
|----------|----------|---------|
| GPU f64 Stability | `ecoPrimals/wateringHole/GPU_F64_NUMERICAL_STABILITY.md` | Full technical writeup |
| Evolution Plan | `ecoPrimals/wateringHole/NUMERICAL_STABILITY_EVOLUTION_PLAN.md` | f32 → DF64 → f64 → FHE roadmap |
| Cross-Spring Issues | `ecoPrimals/wateringHole/SPRING_EVOLUTION_ISSUES.md` ISSUE-006 | barraCuda/coralReef ownership |
| Experiment 046 | `experiments/046_PRECISION_STABILITY_ANALYSIS.md` | Full 9-family stability analysis |
| Experiment 044 | `experiments/044_CHUNA_BGK_DIELECTRIC.md` | Paper 44 GPU results |
| Experiment 043 | `experiments/043_CHUNA_GRADIENT_FLOW_VALIDATION.md` | Paper 43 GPU results |

---

*hotSpring v0.6.19 — AGPL-3.0-only*
