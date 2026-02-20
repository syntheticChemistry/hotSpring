# hotSpring — BarraCUDA Requirements

**Last Updated**: February 20, 2026
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
| **FFT (momentum-space transforms)** | Not in BarraCUDA | **P0** | High — needed for full QCD with dynamical fermions |
| ~~**Complex f64 arithmetic**~~ | ✅ **Done** — `lattice/complex_f64.rs` + WGSL template | — | — |
| ~~**SU(3) matrix operations**~~ | ✅ **Done** — `lattice/su3.rs` + WGSL template | — | — |
| ~~**Hybrid Monte Carlo (HMC)**~~ | ✅ **Done** — `lattice/hmc.rs`, Cayley exponential, 96-100% acceptance | — | — |
| **GPU SU(3) plaquette shader** | WGSL template exists; needs compilation + validation | **P1** | Low |
| **GPU Dirac operator** | CPU implementation exists; needs WGSL port | **P1** | Medium |
| **Larger lattice sizes (8^4, 16^4)** | 4^4 validated; scaling untested | **P2** | Low |

### Kernels That Transfer Directly (Confirmed)

| Lattice QCD Need | BarraCUDA Kernel | Status |
|-----------------|-----------------|--------|
| Gauge field update (MD) | Velocity Verlet | ✅ Adapted as HMC leapfrog in `hmc.rs` |
| Thermodynamic observables | FusedMapReduceF64 | ✅ Used for plaquette averages |
| Eigenvalue computation | BatchedEighGpu | ✅ Available for Dirac spectrum |
| Parameter scans | L1 Pareto framework | ✅ Same structure for coupling scans |
| Statistical analysis | Monte Carlo infrastructure | ✅ Jackknife/bootstrap available |

### Kachkovskiy Extension (Spectral Theory / Transport)

| Need | Current Status | Priority | Effort |
|------|---------------|----------|--------|
| **Lanczos eigensolve** | `BatchedEighGpu` handles dense; need iterative for large sparse | **P1** | Medium — tridiagonalization + QR iteration on GPU |
| **Sparse matrix-vector product (SpMV)** | Not in BarraCUDA. CG solver exists in `lattice/cg.rs` (CPU) | **P1** | Medium — CSR format SpMV shader. Foundation for Lanczos |
| **Matrix exponentiation** | Cayley exponential validated for SU(3) in `lattice/hmc.rs` | **P2** | Medium — generalize beyond 3×3 anti-Hermitian |

### Stretch Goals (Updated)

| Need | Why | Status |
|------|-----|--------|
| ~~Conjugate gradient solver~~ | Dirac operator inversion | ✅ **Done** — `lattice/cg.rs` (CPU, 214 lines) |
| Multi-GPU communication | Lattice domain decomposition | Pending |
| Stochastic trace estimator | Disconnected diagrams for flavor-singlet physics | Pending |

---

## BarraCUDA Evolution Path for hotSpring

```
Phase A-F (DONE)              Bazavov Extension (DONE on CPU)     GPU Promotion (NEXT)
─────────────────             ──────────────────────────────      ───────────────────
Yukawa force       ────────→  Gauge plaquette force    ✅         → WGSL plaquette shader
Velocity Verlet    ────────→  HMC integrator           ✅         → WGSL Cayley exp shader
BatchedEighGpu     ────────→  Dirac eigenvalues        ✅ (CG)    → WGSL Dirac SpMV
FusedMapReduce     ────────→  Lattice observables      ✅         → GPU reduction
Real f64           ────────→  Complex f64              ✅         → WGSL template ready
N/A                ────────→  SU(3) matrix ops         ✅         → WGSL template ready
N/A                ────────→  FFT                      Pending    → Full QCD prerequisite
```

---

## ~~Next Science Target: Stanton & Murillo Transport (Paper 5)~~ ✅ DONE

All components implemented and validated (13/13 checks pass):

| Component | Depends On | Status |
|-----------|-----------|--------|
| Green-Kubo integrator (VACF → D*) | Existing VACF observable | ✅ `md/observables.rs` |
| Stress tensor observable (σ_αβ) | Yukawa pair force kernel | ✅ `md/observables.rs` |
| Heat current observable (J_Q) | Pair force + velocities | ✅ `md/observables.rs` |
| Daligault (2012) D* fit | Analytical model | ✅ `md/transport.rs` (Sarkas-calibrated) |
| Stanton-Murillo (2016) η*, λ* fits | Analytical models | ✅ `md/transport.rs` |
| Validation binary | ValidationHarness | ✅ `bin/validate_stanton_murillo.rs` |

---

## ToadStool Handoff Notes

**Active handoffs:**
- `wateringHole/handoffs/HOTSPRING_V0516_CONSOLIDATED_HANDOFF_FEB20_2026.md` — full primitive catalog + evolution lessons
- `wateringHole/handoffs/HOTSPRING_TOADSTOOL_REWIRE_FEB20_2026.md` — v0.5.16 absorption audit + CellListGpu bug + shader designs
- `wateringHole/handoffs/HOTSPRING_NPU_RESERVOIR_HANDOFF_FEB20_2026.md` — NPU reservoir transport (ESN → Akida absorption path)

(14 prior handoffs archived to `wateringHole/handoffs/archive/`)

### Key Facts for ToadStool Team

- 9 papers reproduced, 320 unit tests, 18/18 validation suites, ~$0.20 total compute cost
- RTX 4070 sustains f64 MD at 149-259 steps/s; Titan V (NVK) produces identical physics
- Energy drift 0.000% over 80k steps sets the precision bar for any new integrator
- `ReduceScalarPipeline` is the most-used upstream primitive after `WgpuDevice`
- All shader compilation routes through `ShaderTemplate::for_driver_profile()`
- `CellListGpu` has two open bugs (binding mismatch + `i32 %` truncation) — hotSpring uses local `GpuCellList`

### Evolution Timeline

| Version | ToadStool Absorptions | hotSpring Rewires |
|---------|----------------------|-------------------|
| v0.5.12 | `ReduceScalarPipeline` | Rewired both MD paths, deleted local `SHADER_SUM_REDUCE` |
| v0.5.15 | `WgslOptimizer`, `GpuDriverProfile`, `StatefulPipeline` | Rewired all shader compilation via `for_driver_profile()` |
| v0.5.16 | NAK eigensolve shader, `StatefulPipeline` impl, `CellListGpu` attempt, `scalar_buffer()`/`max_f64`/`min_f64` on ReduceScalar | Paper 13 (Abelian Higgs), doc audit |

### ToadStool v0.5.16 Absorption Review (Feb 20, 2026)

**Absorbed successfully:**
- NAK-optimized eigensolve (`batched_eigh_nak_optimized_f64.wgsl`) — 5 workarounds, drop-in
- `StatefulPipeline` with `run_iterations()` / `run_until_converged()` — available
- `ReduceScalarPipeline` gains `scalar_buffer()` for zero-copy GPU chaining, `max_f64`, `min_f64`
- `WgslLoopUnroller` — `@unroll_hint N` annotation, up to 32× unroll
- NAK deficiencies documented in `contrib/mesa-nak/NAK_DEFICIENCIES.md`

**Still broken — `CellListGpu` prefix-sum BGL mismatch:**
`cell_list_gpu.rs` creates a scan BGL with 3 bindings (input=0, output=1, params=2)
but `prefix_sum.wgsl` expects 4 bindings (params=0, input=1, output=2, scratch=3).
The binding order and count are incompatible. hotSpring keeps local `GpuCellList`
with its own `exclusive_prefix_sum.wgsl` (3-binding, matching layout).

### Open Items for ToadStool

1. **Fix `CellListGpu` prefix-sum BGL** — binding layout mismatch (3 vs 4 bindings, different order)
2. **Absorb NPU reservoir transport** — ESN shaders, weight export, Akida wiring (see NPU handoff)
3. **FFT primitive** — blocks full lattice QCD (Tier 3 papers 9-12)
4. **Complex f64 WGSL shader** — template exists in hotSpring `lattice/complex_f64.rs`, ready for promotion
5. **SU(3) WGSL shader** — template exists in hotSpring `lattice/su3.rs`, ready for promotion
6. **Lattice plaquette + HMC WGSL shaders** — CPU implementations validated, need GPU port
7. **ESN `export_weights()` method on `esn_v2::ESN`** — needed for GPU-train → NPU-deploy path
