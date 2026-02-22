# hotSpring — BarraCUDA Requirements

**Last Updated**: February 22, 2026
**Purpose**: GPU kernel requirements, gap analysis, and evolution priorities

---

## Current Kernel Usage (Validated)

| Kernel / Primitive | WGSL Shader | Phase | Validation |
|-------------------|-------------|-------|------------|
| Yukawa pair force | `yukawa_force_f64.wgsl` | C-E | 9/9 PP cases, 0.000% drift |
| Velocity Verlet integration | `velocity_verlet_f64.wgsl` | C-E | 80k steps sustained |
| Cell-list neighbor search | `cell_list_f64.wgsl` | D-E | 4.1x faster than all-pairs |
| Berendsen thermostat | `berendsen_f64.wgsl` | C-E | Equilibration validated |
| Nose-Hoover thermostat | `nose_hoover_f64.wgsl` | C | NVT ensemble |
| GEMM (matrix multiply) | `gemm_f64.wgsl` | B, F | HFB Hamiltonian construction |
| Batched eigensolve | `BatchedEighGpu` | F | 791 HFB nuclei, 99.85% convergence |
| Fused map-reduce | `FusedMapReduceF64` | B | Observables (KE, PE, pressure) |
| PPPM Ewald | `PppmGpu` | E | Kappa=0 Coulomb validation |
| SSF (static structure factor) | `SsfGpu` | E | Wavenumber-space observables |
| RBF interpolation | CPU (SciPy) | A | Surrogate baseline |
| MLP surrogate | CPU (PyTorch) | A | Neural surrogate comparison |
| Scalar reduction | `ReduceScalarPipeline` | C-E | KE/PE sum, thermostat (v0.5.12) |
| Shader optimizer | `WgslOptimizer` | All | Loop unrolling, ILP reordering (v0.5.15 rewire) |
| Driver profiling | `GpuDriverProfile` | All | Hardware-accurate latency model (v0.5.15 rewire) |

---

## Requirements for Bazavov Extension (Lattice QCD)

### Status Update (Feb 19, 2026)

The lattice QCD infrastructure has been built and validated on CPU. All items
previously marked "Not in BarraCUDA" for Complex f64, SU(3), and HMC are now
**implemented in hotSpring** (`barracuda/src/lattice/`, ~2,800 lines) with WGSL
templates ready for GPU promotion.

### Remaining Gaps

| Need | Current Status | Priority | Effort |
|------|---------------|----------|--------|
| ~~**FFT (momentum-space transforms)**~~ | ✅ **Done** — toadstool `Fft1DF64` + `Fft3DF64` (commit `1ffe8b1a`, 14 GPU tests, roundtrip 1e-10) | — | — |
| ~~**Complex f64 arithmetic**~~ | ✅ **Done** — `lattice/complex_f64.rs` + GPU WGSL (`shaders/math/complex_f64.wgsl`, toadstool `8fb5d5a0`) | — | — |
| ~~**SU(3) matrix operations**~~ | ✅ **Done** — `lattice/su3.rs` + GPU WGSL (`shaders/math/su3.wgsl`, toadstool `8fb5d5a0`) | — | — |
| ~~**Hybrid Monte Carlo (HMC)**~~ | ✅ **Done** — `lattice/hmc.rs`, Cayley exponential, 96-100% acceptance | — | — |
| ~~**GPU SU(3) plaquette shader**~~ | ✅ **Done** — `wilson_plaquette_f64.wgsl` (toadstool `8fb5d5a0`, 4D, 6 orientations, periodic BC) | — | — |
| ~~**GPU HMC force shader**~~ | ✅ **Done** — `su3_hmc_force_f64.wgsl` (toadstool `8fb5d5a0`, staple sum + algebra projection) | — | — |
| ~~**GPU U(1) Abelian Higgs**~~ | ✅ **Done** — `higgs_u1_hmc_f64.wgsl` (toadstool `8fb5d5a0`, Wirtinger factor baked in) | — | — |
| ~~**GPU Dirac operator**~~ | ✅ **Done** — `WGSL_DIRAC_STAGGERED_F64` validated 8/8 (max error 4.44e-16) | — | — |
| ~~**GPU CG solver**~~ | ✅ **Done** — `WGSL_COMPLEX_DOT_RE_F64` + `WGSL_AXPY_F64` + `WGSL_XPAY_F64`, 9/9 checks | — | — |
| ~~**Pure GPU QCD workload**~~ | ✅ **Done** — `validate_pure_gpu_qcd` (3/3): HMC → GPU CG on thermalized configs | — | — |
| **Larger lattice sizes (8^4, 16^4)** | 4^4 + 6^4 + 8^3×4 validated on GPU; scaling to 16^4 next | **P2** | Low |

### Kernels That Transfer Directly (Confirmed)

| Lattice QCD Need | BarraCUDA Kernel | Status |
|-----------------|-----------------|--------|
| Gauge field update (MD) | Velocity Verlet | ✅ Adapted as HMC leapfrog in `hmc.rs` |
| Thermodynamic observables | FusedMapReduceF64 | ✅ Used for plaquette averages |
| Eigenvalue computation | BatchedEighGpu | ✅ Available for Dirac spectrum |
| Parameter scans | L1 Pareto framework | ✅ Same structure for coupling scans |
| Statistical analysis | Monte Carlo infrastructure | ✅ Jackknife/bootstrap available |

### Kachkovskiy Extension (Spectral Theory / Transport)

**Status Update (Feb 22, 2026)**: Full spectral theory stack **absorbed upstream**
into `barracuda::spectral` (ToadStool Rewire v4). hotSpring now re-exports from
the upstream crate — local implementations deleted (~41 KB), backward-compatible
`CsrMatrix` type alias provided. 1D/2D/3D Anderson localization, almost-Mathieu
Aubry-André transition, Herman's formula γ = ln|λ|, Sturm eigensolve, transfer
matrix Lyapunov, Poisson level statistics, CSR SpMV, Lanczos eigensolve, and
Hofstadter butterfly all live in the shared barracuda crate. `BatchIprGpu` (from
neuralSpring) now available for GPU localization diagnostics.

| Need | Current Status | Priority | Effort |
|------|---------------|----------|--------|
| ~~**Tridiagonal eigensolve**~~ | ✅ **Done** — Sturm bisection in `spectral/` | — | — |
| ~~**Transfer matrix Lyapunov**~~ | ✅ **Done** — iterative renormalization in `spectral/` | — | — |
| ~~**Level statistics**~~ | ✅ **Done** — spacing ratio ⟨r⟩ in `spectral/` | — | — |
| ~~**1D Anderson model**~~ | ✅ **Done** — `spectral::anderson_hamiltonian()`, 10/10 checks | — | — |
| ~~**Almost-Mathieu operator**~~ | ✅ **Done** — `spectral::almost_mathieu_hamiltonian()`, Herman validated | — | — |
| ~~**Lanczos eigensolve**~~ | ✅ **Done** — full reorthogonalization in `spectral/`, cross-validated vs Sturm to 4.4e-16 | — | — |
| ~~**Sparse matrix-vector product (SpMV)**~~ | ✅ **Done** — CSR format in `spectral/`, verified vs dense reference (0 error) | — | — |
| **Matrix exponentiation** | Cayley exponential validated for SU(3) in `lattice/hmc.rs` | **P2** | Medium — generalize beyond 3×3 anti-Hermitian |
| ~~**2D Anderson model**~~ | ✅ **Done** — `spectral::anderson_2d()`, GOE→Poisson transition validated, 11/11 checks | — | — |
| ~~**3D Anderson model**~~ | ✅ **Done** — `spectral::anderson_3d()`, mobility edge + dimensional hierarchy, 10/10 checks | — | — |
| ~~**GPU SpMV shader**~~ | ✅ **Done** — `WGSL_SPMV_CSR_F64` validated 8/8 (machine-epsilon parity) | — | — |
| ~~**GPU Lanczos**~~ | ✅ **Done** — GPU SpMV inner loop + CPU control, 6/6 checks | — | — |
| **Fully GPU-resident Lanczos** | GPU SpMV validated; add GPU dot/axpy/scale for N>100k systems | **P1** | Medium |

### Stretch Goals (Updated)

| Need | Why | Status |
|------|-----|--------|
| ~~Conjugate gradient solver~~ | Dirac operator inversion | ✅ **Done** — CPU (`lattice/cg.rs`) + GPU (3 WGSL shaders, 9/9 checks) |
| ~~Python CG baseline~~ | Prove Rust 200× faster | ✅ **Done** — `bench_lattice_cg` + `lattice_cg_control.py` |
| Multi-GPU communication | Lattice domain decomposition | Pending |
| Stochastic trace estimator | Disconnected diagrams for flavor-singlet physics | Pending |

---

## BarraCUDA Evolution Path for hotSpring

```
Phase A-F (DONE)              Bazavov Extension (DONE on CPU)     GPU Promotion (NEXT)
─────────────────             ──────────────────────────────      ───────────────────
Yukawa force       ────────→  Gauge plaquette force    ✅         → WGSL plaquette shader
Velocity Verlet    ────────→  HMC integrator           ✅         → WGSL Cayley exp shader
BatchedEighGpu     ────────→  Dirac eigenvalues        ✅ (CG)    → WGSL Dirac SpMV ✅ (8/8)
FusedMapReduce     ────────→  Lattice observables      ✅         → GPU reduction ✅
Real f64           ────────→  Complex f64              ✅         → GPU WGSL (toadstool 8fb5d5a0) ✅
N/A                ────────→  SU(3) matrix ops         ✅         → GPU WGSL (toadstool 8fb5d5a0) ✅
N/A                ────────→  FFT                      ✅         → GPU Fft1DF64/3D (toadstool 1ffe8b1a) ✅
N/A                ────────→  CG solver (D†D)          ✅ (CPU+GPU) → WGSL CG pipeline ✅ (9/9)
N/A                ────────→  Pure GPU workload        ✅ (HMC+CG) → 3/3 thermalized ✅
```

---

## ~~Next Science Target: Stanton & Murillo Transport (Paper 5)~~ ✅ DONE

All components implemented and validated (13/13 checks pass):

| Component | Depends On | Status |
|-----------|-----------|--------|
| Green-Kubo integrator (VACF → D*) | Existing VACF observable | ✅ `md/observables/` |
| Stress tensor observable (σ_αβ) | Yukawa pair force kernel | ✅ `md/observables/` |
| Heat current observable (J_Q) | Pair force + velocities | ✅ `md/observables/` |
| Daligault (2012) D* fit | Analytical model | ✅ `md/transport.rs` (Sarkas-calibrated) |
| Stanton-Murillo (2016) η*, λ* fits | Analytical models | ✅ `md/transport.rs` |
| Validation binary | ValidationHarness | ✅ `bin/validate_stanton_murillo.rs` |

---

## ToadStool Handoff Notes

**Active handoffs:**
- `wateringHole/handoffs/HOTSPRING_V064_TOADSTOOL_HANDOFF_FEB22_2026.md` — **comprehensive** v0.6.4 evolution handoff (absorption targets, upstream adoption candidates, cross-spring insights)
- `wateringHole/handoffs/HOTSPRING_TOADSTOOL_REWIRE_V4_FEB22_2026.md` — spectral lean details
- `wateringHole/handoffs/CROSS_SPRING_EVOLUTION_FEB22_2026.md` — full cross-spring shader evolution map
- `wateringHole/handoffs/HOTSPRING_V063_EVOLUTION_HANDOFF_FEB22_2026.md` — v0.6.3 WGSL extraction, coverage push

(22 prior handoffs archived to `wateringHole/handoffs/archive/`)

### Key Facts for ToadStool Team

- 22 papers reproduced, 637 unit + 24 integration tests, 33/33 validation suites, ~$0.20 total compute cost (spectral tests now upstream)
- RTX 4070 sustains f64 MD at 149-259 steps/s; Titan V (NVK) produces identical physics
- Energy drift 0.000% over 80k steps sets the precision bar for any new integrator
- `ReduceScalarPipeline` is the most-used upstream primitive after `WgpuDevice`
- All shader compilation routes through `ShaderTemplate::for_driver_profile()`
- `CellListGpu` **FIXED** (toadstool `8fb5d5a0`) — **migrated** (v0.6.2): local `GpuCellList` deleted, 3 shaders removed, -282 lines

### Evolution Timeline

| Version | ToadStool Absorptions | hotSpring Rewires |
|---------|----------------------|-------------------|
| v0.5.12 | `ReduceScalarPipeline` | Rewired both MD paths, deleted local `SHADER_SUM_REDUCE` |
| v0.5.15 | `WgslOptimizer`, `GpuDriverProfile`, `StatefulPipeline` | Rewired all shader compilation via `for_driver_profile()` |
| v0.5.16 | NAK eigensolve shader, `StatefulPipeline` impl, `CellListGpu` attempt, `scalar_buffer()`/`max_f64`/`min_f64` on ReduceScalar | Paper 13 (Abelian Higgs), doc audit |
| S18-25 | `CellListGpu` BGL **fix**, Complex64+SU(3) WGSL, Wilson plaquette+HMC+Higgs GPU, **GPU FFT f64** | **Rewire v3**: deprecate local GpuCellList, unblock Tier 3 lattice QCD |
| S25-31h | Full `spectral` module absorption, `BatchIprGpu`, `GenEighGpu`, `GemmCachedF64`, `NelderMeadGpu`, WGSL precision fixes | **Rewire v4**: spectral lean — deleted ~41 KB local code, re-export from upstream |

### ToadStool v0.5.16 Absorption Review (Feb 20, 2026)

**Absorbed successfully:**
- NAK-optimized eigensolve (`batched_eigh_nak_optimized_f64.wgsl`) — 5 workarounds, drop-in
- `StatefulPipeline` with `run_iterations()` / `run_until_converged()` — available
- `ReduceScalarPipeline` gains `scalar_buffer()` for zero-copy GPU chaining, `max_f64`, `min_f64`
- `WgslLoopUnroller` — `@unroll_hint N` annotation, up to 32× unroll
- NAK deficiencies documented in `contrib/mesa-nak/NAK_DEFICIENCIES.md`

**~~Still broken~~ FIXED — `CellListGpu` prefix-sum BGL (toadstool `8fb5d5a0`):**
ToadStool commit `8fb5d5a0` rebuilt the scan BGL to 4 bindings matching
`prefix_sum.wgsl` exactly, split single pipeline into `scan_pass_a` + `scan_pass_b`,
and added `n_groups` to scan params. hotSpring's local `GpuCellList` is now
**deprecated** — migration to upstream `CellListGpu` planned for next cycle.

### Open Items for ToadStool (Updated Feb 22, 2026 — Post Rewire v4)

1. ~~**Fix `CellListGpu` prefix-sum BGL**~~ — ✅ **DONE** (commit `8fb5d5a0`)
2. **Absorb NPU reservoir transport** — ESN shaders, weight export, Akida wiring (see NPU handoff)
3. ~~**FFT primitive**~~ — ✅ **DONE** (commit `1ffe8b1a`, `Fft1DF64` + `Fft3DF64`, 14 GPU tests)
4. ~~**Complex f64 WGSL shader**~~ — ✅ **DONE** (commit `8fb5d5a0`, `shaders/math/complex_f64.wgsl`)
5. ~~**SU(3) WGSL shader**~~ — ✅ **DONE** (commit `8fb5d5a0`, `shaders/math/su3.wgsl`)
6. ~~**Lattice plaquette + HMC WGSL shaders**~~ — ✅ **DONE** (commit `8fb5d5a0`, 3 GPU shaders)
7. **ESN `export_weights()` method on `esn_v2::ESN`** — needed for GPU-train → NPU-deploy path
8. ~~**GPU Dirac SpMV shader**~~ — ✅ **Done**: `WGSL_DIRAC_STAGGERED_F64` (8/8 checks, 4.44e-16)
9. ~~**GPU SpMV for spectral theory**~~ — ✅ **Done** and **absorbed**: `barracuda::spectral::WGSL_SPMV_CSR_F64`
10. ~~**GPU Lanczos**~~ — ✅ **Done** and **absorbed**: upstream `barracuda::spectral::lanczos`
11. ~~**GPU CG solver**~~ — ✅ **Done**: 3 WGSL shaders (9/9 checks, iterations match exactly)
12. ~~**Pure GPU workload**~~ — ✅ **Done**: HMC → GPU CG on thermalized configs (3/3, 4.10e-16)
13. **Fully GPU-resident Lanczos** — GPU dot + axpy + scale for N > 100k (next P1)
14. **Absorb Staggered Dirac shader** — `WGSL_DIRAC_STAGGERED_F64` ready for upstream (8/8 checks, Tier 1)
15. **Absorb CG solver shaders** — `WGSL_COMPLEX_DOT_RE_F64` + `WGSL_AXPY_F64` + `WGSL_XPAY_F64` (9/9 checks, Tier 1)
16. **Absorb HFB shader suite** — potentials + density + BCS bisection (14+GPU+6 checks, Tier 2)
17. **Absorb deformed HFB pipeline** — GPU-resident SCF loop, could use upstream `GemmCachedF64`
