# Spec: 4D Anderson & Wegner Block Proxy System

**Status:** DRAFT
**Date:** February 28, 2026
**License:** AGPL-3.0-only
**Depends on:** Exp 025 (3D Anderson baseline), upstream `barracuda::spectral`

---

## Problem

The current Anderson proxy (Exp 025) uses a 3D scalar Anderson
Hamiltonian to predict CG solver difficulty for the 4D QCD Dirac
operator. This has two dimensional mismatches:

1. **Spatial dimension**: 3D proxy for a 4D lattice (6 neighbors vs 8)
2. **On-site structure**: scalar disorder vs SU(3) matrix links

Both mismatches reduce the proxy's predictive power. The level spacing
ratio from 3D scalar Anderson captures the qualitative
localization/delocalization transition but misses the exact critical
exponents (nu ~ 1.57 in 3D vs ~ 1.1 in 4D) and the matrix eigenvalue
repulsion from the color structure.

## Proposed Solution: Three-Tier Proxy Hierarchy

### Tier 1: 4D Scalar Anderson (anderson_4d)

Same as `anderson_3d` but on a 4-dimensional hypercubic lattice with
periodic boundary conditions. Each site has 8 nearest neighbors
(+/- in x, y, z, t directions) and a random scalar potential W_i.

**Matrix size:** L^4 × L^4. For L=8, that's 4,096 × 4,096 — matching
the QCD lattice site count exactly.

**Cost estimate:** Full eigendecomposition of 4096 × 4096 dense matrix
~2-5 seconds on CPU. Lanczos for partial spectrum ~0.5-1 second.

**What it captures:**
- Correct spatial connectivity (8 neighbors = 4D hypercubic)
- Correct volume scaling
- Correct critical disorder (W_c ~ 34.5 in 4D vs ~ 16.5 in 3D)
- Correct critical exponent (nu ~ 1.1 in 4D)

**What it misses:**
- Color structure (SU(3) matrix at each link)
- Spin structure (staggered fermion phases)

### Tier 2: 4D Block Anderson — Wegner Orbital Model (anderson_4d_block)

Extension of Tier 1 where each on-site potential is a random N_c × N_c
Hermitian matrix (drawn from GUE) and each hopping term is a random
N_c × N_c unitary matrix (drawn from Haar measure on U(N_c)).

For QCD: N_c = 3 (three colors).

**Matrix size:** (L^4 × N_c) × (L^4 × N_c). For L=8, N_c=3, that's
12,288 × 12,288.

**Cost estimate:** Lanczos for 200 eigenvalues of 12,288 × 12,288
sparse matrix ~10-30 seconds on CPU. Full eigendecomposition ~1-3
minutes.

**What it captures:**
- Everything from Tier 1
- Internal matrix structure (eigenvalue repulsion from color)
- Block-matrix level statistics that match the Dirac operator's
  symmetry class (GUE / chiral unitary)
- The scaling of the smallest eigenvalue with N_c

**What it misses:**
- Exact SU(3) gauge structure (uses random unitary, not links from a
  gauge ensemble)
- Staggered fermion sign structure

### Tier 3: Full QCD Dirac Operator (existing)

The actual CG solve on the staggered Dirac operator. This is what we
are predicting, not a proxy.

## Comparison of Proxy Fidelity

| Proxy | Dim | On-site | Hopping | Matrix size (L=8) | Cost | Level stats match |
|-------|-----|---------|---------|-------------------|------|-------------------|
| 3D scalar (current) | 3D | scalar | scalar -1 | 512 × 512 | 200 ms | Qualitative |
| **4D scalar** | **4D** | **scalar** | **scalar -1** | **4,096 × 4,096** | **2-5 s** | **Good** |
| **4D block (Wegner)** | **4D** | **3×3 GUE** | **3×3 U(3)** | **12,288 × 12,288** | **10-30 s** | **Quantitative** |
| QCD Dirac | 4D | staggered | SU(3) gauge link | 12,288 × 12,288 | 40 s (CG) | Exact |

The Wegner model sits in a cost sweet spot: ~75% of the information
content at ~50% of the cost of the actual CG solve. More importantly,
the Wegner level statistics are in the same universality class (GUE)
as the Dirac operator, whereas the scalar Anderson model is in the GOE
class. This class match means the eigenvalue correlation functions —
which directly determine CG convergence — will track the real system
more faithfully.

## Implementation Plan

### 1. anderson_4d (new function in barracuda::spectral)

```rust
pub fn anderson_4d(
    lx: usize, ly: usize, lz: usize, lt: usize,
    disorder: f64,
    seed: u64,
) -> SpectralCsrMatrix
```

Identical to `anderson_3d` but adds the t-dimension neighbors. 8
neighbors per site instead of 6. Periodic boundary conditions in all
four directions.

CSR storage: ~8 × L^4 nonzeros (hopping) + L^4 diagonal entries.
For L=8: ~36,864 nonzeros. Sparse — Lanczos efficient.

### 2. anderson_4d_block (new function)

```rust
pub struct BlockCsrMatrix {
    pub n_sites: usize,
    pub block_size: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub blocks: Vec<Vec<f64>>,  // each block is block_size^2 f64s
}

pub fn anderson_4d_block(
    lx: usize, ly: usize, lz: usize, lt: usize,
    disorder: f64,
    block_size: usize,  // 3 for SU(3)
    seed: u64,
) -> BlockCsrMatrix
```

On-site blocks: random Hermitian matrices from GUE(N_c).
Hopping blocks: random unitary matrices from Haar(U(N_c)).

The `BlockCsrMatrix` supports sparse matrix-vector multiply where the
"entries" are dense blocks. The Lanczos algorithm works unchanged —
it only needs SpMV and inner products.

### 3. block_lanczos (new function)

```rust
pub fn block_lanczos(
    h: &BlockCsrMatrix,
    k: usize,
    seed: u64,
) -> LanczosTridiag
```

Standard Lanczos on the expanded (n_sites × block_size) vector space.
The SpMV performs block-sparse operations.

### 4. NPU training integration

The proxy pipeline (`gpu_physics_proxy.rs`) adds two new phases:

- **Phase 3: 4D Scalar Anderson** — same parameter sweep as Phase 1 but
  in 4D. Outputs: level_spacing_ratio, lambda_min, bandwidth, ipr.
- **Phase 4: 4D Block Anderson (Wegner)** — sweep over disorder strength
  and block size. Outputs: block-level spacing ratio, block lambda_min,
  participation ratio in color space.

NPU Head 14 (Anderson CG) receives features from the best available
proxy tier. During production runs, the proxy pipeline runs on CPU or
Titan V concurrently with the main HMC on the RTX 3090.

## Random Matrix Generation

### GUE(N_c) for on-site blocks

Generate a random N_c × N_c matrix A with i.i.d. standard normal
entries, then form H = (A + A†) / 2. This is a sample from the
Gaussian Unitary Ensemble scaled by 1/sqrt(2).

### Haar(U(N_c)) for hopping blocks

Generate a random N_c × N_c complex matrix A with i.i.d. standard
normal entries, then compute its QR decomposition A = QR. The matrix Q
(after fixing the sign of the diagonal of R) is a Haar-distributed
random unitary matrix.

For N_c = 3, both operations are 3×3 matrix arithmetic — negligible
cost per site.

## Energy Measurement Requirements

All proxy runs must log:

| Field | Source | Unit |
|-------|--------|------|
| `wall_ms` | std::time::Instant | milliseconds |
| `cpu_energy_uj` | RAPL (if available) | microjoules |
| `gpu_power_mw` | hwmon / nvidia-smi | milliwatts |
| `gpu_temp_c` | hwmon | degrees C |
| `cpu_temp_c` | k10temp hwmon | degrees C |

See Experiment 027 for the full energy tracking specification.

## Validation

1. **4D scalar Anderson:** Level spacing ratio must converge to GOE
   (⟨r⟩ ~ 0.53) at low disorder and Poisson (⟨r⟩ ~ 0.39) at high
   disorder, with critical W_c ~ 34.5 (literature value).

2. **4D block Anderson (Wegner):** Level spacing ratio must converge
   to GUE (⟨r⟩ ~ 0.60) at low disorder — the GUE value, not GOE,
   because the block structure breaks time-reversal symmetry.

3. **CG prediction accuracy:** After NPU training, Head 14 CG
   predictions from Wegner data should have lower mean absolute error
   than predictions from 3D scalar data.

## Files

- `specs/ANDERSON_4D_WEGNER_PROXY.md` — this document
- `experiments/026_4D_ANDERSON_WEGNER_PROXY.md` — experiment protocol
- `barracuda/src/spectral/anderson.rs` — upstream implementation target
- `barracuda/src/bin/gpu_physics_proxy.rs` — proxy pipeline binary
