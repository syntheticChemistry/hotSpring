# Experiment 026: 4D Anderson & Wegner Block Proxy Pipeline

**Status:** PLANNED
**Date:** February 28, 2026
**Depends on:** Exp 025 (3D baseline proxy), Spec: ANDERSON_4D_WEGNER_PROXY
**License:** AGPL-3.0-only

---

## Motivation

Experiment 025 established the 3D scalar Anderson model as a CG cost
proxy, generating 171 training data points in 67 seconds. However, the
3D scalar model has two fidelity gaps relative to the 4D QCD Dirac
operator:

1. **Wrong dimensionality** — 3D (6 neighbors) vs 4D (8 neighbors).
   The critical disorder W_c differs by 2× (16.5 vs 34.5) and the
   critical exponent nu changes from 1.57 to 1.1.

2. **No internal structure** — scalar on-site potential vs SU(3) matrix
   links. The scalar model produces GOE level statistics; the Dirac
   operator has GUE statistics due to broken time-reversal symmetry.

This experiment runs a three-tier proxy comparison:

| Tier | Model | Matrix size (L=8) | Est. cost | Level stats class |
|------|-------|-------------------|-----------|-------------------|
| 1 | 3D scalar (Exp 025 baseline) | 512 × 512 | 200 ms | GOE |
| 2 | **4D scalar** | **4,096 × 4,096** | **2-5 s** | **GOE** |
| 3 | **4D block Wegner (N_c=3)** | **12,288 × 12,288** | **10-30 s** | **GUE** |

---

## Phase 1: 4D Scalar Anderson

### Parameters

| Parameter | Values |
|-----------|--------|
| Lattice sizes | L = 4, 6, 8 |
| Disorder W | 4, 8, 16, 20, 25, 30, 34.5, 36, 40, 50 |
| Seeds | 42, 137, 271 |
| Eigenvalues | Full (L=4,6), Lanczos k=300 (L=8) |
| Boundary conditions | Periodic in all 4 directions |

Total: 3 × 10 × 3 = 90 data points.

### Implementation: anderson_4d

Add to `barracuda::spectral::anderson`:

```rust
pub fn anderson_4d(
    lx: usize, ly: usize, lz: usize, lt: usize,
    disorder: f64, seed: u64,
) -> SpectralCsrMatrix
```

Neighbor connectivity: 8 per site (±x, ±y, ±z, ±t with periodic BCs).
Index mapping: `idx(ix,iy,iz,it) = ((ix*ly + iy)*lz + iz)*lt + it`.

### Diagnostics per point

| Diagnostic | Symbol | What it measures |
|------------|--------|-----------------|
| Level spacing ratio | ⟨r⟩ | GOE (0.53) vs Poisson (0.39) |
| Smallest eigenvalue | λ_min | Condition number proxy |
| Bandwidth | B | Spectral spread |
| IPR (from spacing variance) | IPR | Localization strength |
| Wall time | t_wall | Cost |
| CPU energy | E_cpu | Energy consumed (RAPL) |

### Success criteria

- [ ] ⟨r⟩ → 0.53 (GOE) at W << W_c and → 0.39 (Poisson) at W >> W_c
- [ ] Critical disorder W_c ~ 34.5 ± 2 visible in ⟨r⟩ crossover
- [ ] L=8 eigendecomposition < 10 seconds on Threadripper

---

## Phase 2: 4D Block Anderson (Wegner Orbital Model)

### Parameters

| Parameter | Values |
|-----------|--------|
| Lattice sizes | L = 4, 6, 8 |
| Block size (N_c) | 3 (matching SU(3)) |
| Disorder W | 4, 8, 16, 25, 34.5, 50, 70, 100 |
| Seeds | 42, 137, 271 |
| Eigenvalues | Lanczos k=300 |

Total: 3 × 8 × 3 = 72 data points.

### Implementation: anderson_4d_block

New data structures:

```rust
pub struct BlockCsrMatrix {
    pub n_sites: usize,
    pub block_size: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub blocks: Vec<Vec<f64>>,  // row-major block_size × block_size
}
```

On-site blocks: GUE(3) — random 3×3 Hermitian, scaled by disorder W.
Hopping blocks: Haar U(3) — random 3×3 unitary.

SpMV for Lanczos: block-sparse multiply. Each block-row has 8 neighbor
blocks plus diagonal. For L=8: 4,096 block-rows, each with ≤9 blocks
of 3×3. Total float ops per SpMV: ~4,096 × 9 × 9 × 2 ~ 663K flops.

### Diagnostics per point

Same as Phase 1, plus:

| Diagnostic | Symbol | What it measures |
|------------|--------|-----------------|
| Level spacing ratio | ⟨r⟩ | GUE (0.60) vs Poisson (0.39) |
| Color participation ratio | CPR | How many colors participate in each mode |
| Block eigenvalue spread | σ_block | Variance of on-site block eigenvalues |

### Success criteria

- [ ] ⟨r⟩ → 0.60 (GUE, not GOE) at low disorder
- [ ] ⟨r⟩ → 0.39 (Poisson) at high disorder
- [ ] GUE-to-Poisson transition visible at W_c
- [ ] L=8 block Lanczos < 60 seconds on Threadripper

---

## Phase 3: Cross-Tier Comparison

Using the same disorder → QCD β mapping from Exp 025, compare the
three proxy tiers' ability to predict actual CG iterations from the
Exp 024 production run.

### Mapping: disorder W → QCD β

The effective disorder in QCD comes from plaquette variance. From the
Exp 024 production data:

| β (QCD) | ⟨P⟩ | plaq_var | Effective W (est.) | Actual CG |
|---------|------|----------|-------------------|-----------|
| 4.44 | 0.329 | 2.2e-5 | ~40 (strong disorder) | 55,045 |
| 5.00 | 0.402 | — | ~25 | 54,399 |
| 5.50 | 0.515 | — | ~12 | 51,422 |
| 5.69 | 0.542 | — | ~8 | 50,562 |
| 6.00 | 0.571 | — | ~5 | 46,848 |
| 6.13 | 0.588 | — | ~4 (weak disorder) | 46,079 |

The W → CG mapping is what NPU Head 14 learns. This experiment
generates training data at multiple proxy fidelities to determine which
tier provides the best CG prediction.

### Success criteria

- [ ] Wegner (Tier 3) CG predictions have lower MAE than 3D scalar (Tier 1)
- [ ] 4D scalar (Tier 2) improves over 3D scalar (Tier 1)
- [ ] Combined multi-tier features outperform any single tier

---

## Phase 4: Computational Cost / Energy Comparison

Every proxy point logs wall time, CPU energy (RAPL where available),
and CPU temperature. This data feeds into the cost-benefit analysis:
is the Wegner model's improved prediction accuracy worth its 50-100×
higher cost compared to 3D scalar?

### Expected cost budget

| Tier | Points | Est. cost/point | Total est. | Energy est. |
|------|--------|----------------|------------|-------------|
| 1 (3D scalar, from Exp 025) | 126 | 200 ms | 25 s | ~5 J |
| 2 (4D scalar) | 90 | 3 s | 270 s | ~50 J |
| 3 (4D Wegner) | 72 | 20 s | 1,440 s | ~300 J |
| **Total** | **288** | — | **~29 min** | **~355 J** |

For context, a single 8⁴ dynamical trajectory costs ~40 seconds and
~14,800 J (370W × 40s). The entire three-tier proxy sweep costs less
than 2 trajectories in wall time and ~2.4% of one trajectory's energy.

---

## Hardware Assignment

| Component | Task |
|-----------|------|
| Threadripper (CPU) | All proxy eigendecompositions |
| Titan V | Available for GPU-accelerated Lanczos if CPU is too slow at L=8 Wegner |
| RTX 3090 | Reserved for primary HMC production — not touched by proxies |
| NPU (Akida) | Receives training data after proxy pipeline completes |

---

## Output Files

| File | Contents |
|------|----------|
| `results/exp026_anderson_4d_scalar.jsonl` | Tier 2 data (4D scalar) |
| `results/exp026_anderson_4d_wegner.jsonl` | Tier 3 data (4D block Wegner) |
| `results/exp026_cross_tier_comparison.jsonl` | Merged comparison data |
| `results/exp026_energy_log.jsonl` | Per-point energy measurements |

---

## Run Command

```bash
cargo run --release --bin gpu_physics_proxy -- \
  --mode=4d-sweep \
  --output=results/exp026_anderson_4d_scalar.jsonl \
  --output-wegner=results/exp026_anderson_4d_wegner.jsonl \
  --energy-log=results/exp026_energy_log.jsonl \
  2>&1 | tee results/exp026.log
```

---

## What This Contributes to the System

1. **Better CG prediction** — Tier 3 (Wegner) shares the same symmetry
   class (GUE) as the Dirac operator, so its spectral statistics
   directly predict CG behavior. Tier 1 (3D scalar) is in GOE, which
   only qualitatively tracks the Dirac spectrum.

2. **NPU training data quality** — Higher-fidelity proxy data should
   reduce the NPU's CG prediction error, leading to better adaptive
   steering decisions (better dt/n_md suggestions, better β ordering).

3. **Multi-fidelity fusion** — The NPU can receive features from all
   three tiers simultaneously: cheap-and-rough (3D scalar), moderate
   (4D scalar), and expensive-and-accurate (4D Wegner). The ESN
   learns the optimal weighting.

4. **Energy baseline** — First systematic energy measurement across
   proxy tiers, feeding into Exp 027's broader energy accounting.
