# hotSpring v0.6.9 → ToadStool S62 Sync Handoff

**Date:** February 24, 2026
**From:** hotSpring (biomeGate compute campaign, complete)
**To:** ToadStool / BarraCuda core team
**Sync:** hotSpring pulled toadStool S53→S62 (172 files, +15,847/-7,066 lines)
**Status:** hotSpring compiles cleanly against S62, 39/39 suites, zero warnings

---

## What Happened

hotSpring handed off the DF64 core streaming discovery (Experiment 012, Feb 24).
ToadStool absorbed it in Session 58 and, across Sessions 58–62, built the
**entire production DF64 HMC pipeline** — far beyond the handoff's scope.

hotSpring pulled, confirmed clean compilation, deleted its local `df64_core.wgsl`,
bumped to v0.6.9, and updated its absorption tracking.

---

## What ToadStool Built (S58–S62)

### DF64 Shaders (all NEW, built by toadStool from hotSpring's df64_core.wgsl)

| Shader | Purpose | Zone Pattern |
|--------|---------|--------------|
| `df64_core.wgsl` | Df64 struct + arithmetic | Foundation |
| `su3_df64.wgsl` | Cdf64, SU(3) matmul/adjoint/trace in DF64 | Compute zone |
| `su3_hmc_force_df64.wgsl` | Gauge force (18 matmuls on FP32 cores) | 40% of HMC |
| `wilson_plaquette_df64.wgsl` | Plaquette Re Tr(U)/3 in DF64 | 15% of HMC |
| `wilson_action_df64.wgsl` | Wilson action 1−ReTr/3 in DF64 | ~5% of HMC |
| `kinetic_energy_df64.wgsl` | Tr(P²) per link in DF64 | ~5% of HMC |
| `gemm_df64.wgsl` | Dense GEMM with DF64 dot products | General linalg |
| `lennard_jones_df64.wgsl` | LJ force in DF64 | MD forces |

### Production Wiring

Every lattice op now auto-selects:
```rust
let strategy = GpuDriverProfile::from_adapter(&adapter).fp64_strategy();
match strategy {
    Fp64Strategy::Native => /* use f64 shaders */,
    Fp64Strategy::Hybrid => /* use DF64 shaders */,
}
```

This is implemented in: `plaquette.rs`, `hmc_force_su3.rs`, `gpu_wilson_action.rs`,
`gpu_kinetic_energy.rs`. The 6.7× speedup on consumer GPUs is available now.

### Other S54–S62 Work

| Session | Headline |
|---------|----------|
| S54 | Cross-spring: graph_laplacian, hessian, pow_f64 fix, 5 WGSL shaders |
| S55 | Deep debt: MHA decomposition, refactors, unsafe audit |
| S56 | Final absorptions: belief_propagation, boltzmann, disordered_laplacian |
| S57 | Coverage: +47 tests, println→tracing, docs cleanup |
| S58 | **df64_core.wgsl, Fp64Strategy, ODE bio, NMF, pow_f64 patch** |
| S59 | Anderson correlated, ridge regression, validation harness |
| S60–61 | MHA decomposition, Conv2D GPU, NVK guard, SpMM, TransE |
| S62 | DF64 expansion into all lattice ops, deep debt reduction |

---

## What hotSpring Did (This Sync)

1. Pulled toadStool S53→S62
2. Confirmed clean compile (zero errors, zero warnings)
3. Deleted local `df64_core.wgsl` — now upstream at `barracuda::ops::lattice::su3::WGSL_DF64_CORE`
4. Updated ABSORPTION_MANIFEST with full S58–S62 tracking
5. Bumped to v0.6.9 with CHANGELOG entry
6. Archived the pre-sync handoff (superseded by toadStool's implementation)

---

## What hotSpring Still Has Locally

hotSpring's `lattice/` module still uses its own `include_str!` shader loading
and `GpuF64::create_pipeline_f64()` for pipeline creation. This works but does
not yet benefit from the upstream DF64 auto-selection.

### Current Local Shaders (20 files in `lattice/shaders/`)

| Category | Files | Status |
|----------|:-----:|--------|
| SU(3) + complex | `su3_f64.wgsl`, `complex_f64.wgsl` | Upstream has identical + DF64 variants |
| HMC core | `wilson_plaquette_f64`, `su3_gauge_force_f64`, `su3_link_update_f64`, `su3_momentum_update_f64`, `su3_kinetic_energy_f64` | Upstream has f64 + DF64 variants with auto-select |
| CG solver | `complex_dot_re_f64`, `axpy_f64`, `xpay_f64`, `cg_compute_alpha_f64`, `cg_compute_beta_f64`, `cg_update_xr_f64`, `cg_update_p_f64`, `sum_reduce_f64` | Absorbed (S31d) |
| Dynamical | `dirac_staggered_f64`, `staggered_fermion_force_f64`, `gaussian_fermion_f64` | Dirac absorbed (S31d) |
| PRNG | `su3_random_momenta_f64` | Local only |

### Rewire Path (Future)

To get the DF64 6.7× speedup in hotSpring's production pipeline:

1. **Option A (minimal)**: Import `su3_df64_preamble()` from upstream, add
   `Fp64Strategy` detection to `GpuHmcPipelines::new()`, create DF64 pipeline
   variants alongside f64 pipelines. ~200 lines changed in `gpu_hmc.rs`.

2. **Option B (full lean)**: Migrate `GpuHmcPipelines` to use upstream
   `barracuda::ops::lattice::*` ops directly (each op handles its own shader
   selection). This is a larger refactor but eliminates shader duplication.

Both options produce identical physics — the upstream DF64 shaders maintain f64
at load/store boundaries and only use DF64 for bulk matmuls.

---

## Production Results Available for Validation

hotSpring's biomeGate campaign completed Experiment 013:

| Run | Lattice | Result | Reference |
|-----|---------|--------|-----------|
| RTX 3090 32⁴ (f64) | 1M sites, 12 β points | χ=40.1 at β=5.69 | β_c=5.692 (3 sig figs) |
| Titan V 16⁴ (NVK) | 65K sites, 9 β points | χ~1.0 at β_c | First NVK QCD production |

These results serve as the **native f64 baseline**. The DF64 hybrid should
produce identical physics (within DF64's 14-digit precision) at ~6.7× speed.
Validation plan: rerun 3 β points with DF64 and compare observables.

---

## Remaining hotSpring → ToadStool Items

| Priority | Item | Status |
|:--------:|------|--------|
| 🟡 P1 | ESN reservoir shaders (2 WGSL) | Ready for absorption |
| 🟢 P2 | `su3_random_momenta_f64.wgsl` | Local, potential upstream utility |
| 🟢 P2 | `gaussian_fermion_f64.wgsl` | Local, dynamical fermion PRNG |
| 🟢 P2 | `staggered_fermion_force_f64.wgsl` | Local, dynamical fermion force |
| 🔵 P3 | forge substrate discovery bridge | Pattern reference for toadStool |

---

---

## Update: v0.6.10 — DF64 Core Streaming LIVE (Feb 25, 2026)

### What Was Done

hotSpring v0.6.10 rewired `GpuHmcPipelines` to auto-select the DF64 gauge
force shader on consumer GPUs. The implementation imports `WGSL_DF64_CORE` +
`WGSL_SU3_DF64` from upstream and writes a LOCAL `su3_gauge_force_df64.wgsl`
that uses hotSpring's neighbor-buffer indexing convention.

### Why Not Use Upstream Ops Directly?

**Site-indexing incompatibility.** toadStool's lattice ops compute neighbors
from site coordinates inline (t-major: `idx = t * NxNyNz + x * NyNz + y * Nz + z`).
hotSpring stores links using t-minor ordering: `idx = t + Nt * (x + Nx * (y + Ny * z))`.
Passing hotSpring's link buffers to upstream ops produces wrong neighbor lookups
and catastrophic HMC blowup (ΔH ~4800, non-physical).

**Recommendation for toadStool:** Consider standardizing site ordering, or
making the lattice ops accept a pre-computed neighbor buffer as an alternative.
This would allow cross-spring consumers to use upstream ops directly regardless
of their internal indexing convention.

### Benchmark Results (v0.6.10, RTX 3090, DF64 gauge force active)

| Lattice | Volume | CPU ms/traj | GPU ms/traj | GPU/CPU |
|---------|--------|------------|------------|---------|
| 4⁴ | 256 | 71.7 | 21.3 | 3.4× |
| 8⁴ | 4,096 | 1,146 | 31.4 | 36.5× |
| 8³×16 | 8,192 | 2,322 | 43.4 | 53.5× |
| 16⁴ | 65,536 | 18,424 | 259 | 71.1× |

All trajectories produce physical plaquette values and small ΔH, confirming
the DF64 precision is sufficient for HMC gauge force computation.

### Cross-Spring Evolution (complete trail)

1. **hotSpring Exp 012** (Feb 24) — `df64_core.wgsl` (Dekker f32-pair arithmetic)
2. **toadStool S58** — absorbs `df64_core.wgsl`, creates `su3_df64.wgsl`
3. **toadStool S58-S62** — full DF64 HMC pipeline + `Fp64Strategy` auto-select
4. **hotSpring v0.6.10** — imports upstream DF64 math, writes local DF64 force
   with neighbor-buffer indexing (discovered site-ordering incompatibility)
5. **wetSpring** — contributed `NvvmAdaF64Transcendentals` workaround for RTX 4070
6. **neuralSpring** — benefits from DF64 for ESN reservoir precision ops

*hotSpring v0.6.10 — DF64 core streaming live on consumer GPUs.
All validation checks pass. Deconfinement signal reproduced at β=5.69.*

---

## Update: v0.6.11 — Site-Indexing Standardization (Feb 25, 2026)

### What Was Done

hotSpring adopted toadStool's t-major site-indexing convention:

| Property | Before (v0.6.10) | After (v0.6.11) |
|----------|------------------|------------------|
| Formula | `x + Nx*(y + Ny*(z + Nz*t))` | `t*NxNyNz + x*NyNz + y*Nz + z` |
| Fastest varying | x | z |
| Changed | — | `Lattice::site_index()` + `site_coords()` in `wilson.rs` |

This is a two-function change. All downstream Rust code and all WGSL shaders
(which use pre-computed neighbor buffers) follow automatically.

### Validation (v0.6.11)
- 119/119 unit tests pass
- 3/3 pure GPU HMC checks (plaq=0.584339, 100% acceptance, DF64 active)
- 6/6 GPU beta scan checks (plaquette monotonic, cross-lattice ≤5% parity)
- 7/7 streaming HMC checks (dispatch/streaming parity exact to 1e-8)

### Why This Matters

With matching site indexing, hotSpring can now use toadStool's upstream lattice
ops directly — no buffer reordering, no local shader copies for data layout
reasons. The remaining local shaders exist for the neighbor-buffer pattern
(pre-computed rather than inline coordinate math).

**See:** `TOADSTOOL_SITE_INDEXING_NAK_SOLVER_HANDOFF_FEB25_2026.md` for the
full evolution task charging toadStool with neighbor-buffer support and the
Rust-native NAK solver.
