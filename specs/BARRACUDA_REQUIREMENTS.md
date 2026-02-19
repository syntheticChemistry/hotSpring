# hotSpring — BarraCUDA Requirements

**Last Updated**: February 12, 2026
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

---

## Requirements for Bazavov Extension (Lattice QCD)

### Critical Gaps

| Need | Current Status | Priority | Effort |
|------|---------------|----------|--------|
| **FFT (momentum-space transforms)** | Not in BarraCUDA | **P0** | High — need both real and complex FFT for lattice gauge theory. Momentum-space operations are fundamental to all lattice QCD |
| **Complex f64 arithmetic** | Partial — real f64 only | **P0** | Medium — lattice fields are SU(3) matrices (complex 3x3). Need complex multiply, conjugate, trace |
| **SU(3) matrix operations** | Not in BarraCUDA | **P1** | Medium — link multiplication, staples, plaquette. All built on complex GEMM |
| **Hybrid Monte Carlo (HMC)** | Sarkas MD engine exists | **P1** | Medium — adapt Velocity Verlet for gauge field molecular dynamics. Same integration structure, different force law |

### Existing Kernels That Transfer Directly

| Lattice QCD Need | BarraCUDA Kernel | Adaptation |
|-----------------|-----------------|------------|
| Gauge field update (MD) | Velocity Verlet | Change force law from Yukawa to gauge plaquette |
| Thermodynamic observables | FusedMapReduceF64 | Same reduction pattern — sum over lattice sites |
| Eigenvalue computation | BatchedEighGpu | Dirac spectrum, correlation matrix eigenvalues |
| Parameter scans | L1 Pareto framework | Same structure — scan coupling constants |
| Statistical analysis | Monte Carlo infrastructure | Jackknife/bootstrap from existing MC |

### Stretch Goals

| Need | Why | Effort |
|------|-----|--------|
| Conjugate gradient solver | Dirac operator inversion — dominant cost in lattice QCD | High |
| Multi-GPU communication | Lattice domain decomposition | High |
| Stochastic trace estimator | Disconnected diagrams for flavor-singlet physics | Medium |

---

## BarraCUDA Evolution Path for hotSpring

```
Phase A-F (DONE)              Bazavov Extension (NEXT)
─────────────────             ──────────────────────
Yukawa force       ────────→  Gauge plaquette force
Velocity Verlet    ────────→  HMC integrator
BatchedEighGpu     ────────→  Dirac eigenvalues
FusedMapReduce     ────────→  Lattice observables
Real f64           ────────→  Complex f64 (NEW)
N/A                ────────→  FFT (NEW)
N/A                ────────→  SU(3) matrix ops (NEW)
```

---

## Next Science Target: Stanton & Murillo Transport (Paper 5)

Requires NO new BarraCUDA primitives. New hotSpring-local code only:

| Component | Depends On | Status |
|-----------|-----------|--------|
| Green-Kubo integrator (VACF → D) | Existing VACF observable | Not started |
| Stress tensor observable (σ_αβ) | Yukawa pair force kernel | Not started |
| Heat current observable (J_Q) | Pair force + velocities | Not started |
| Γ-κ parameter sweep | sarkas_gpu infrastructure | Not started |

These observables are computed from existing MD trajectories. No new GPU
shaders needed — the Green-Kubo integration runs on CPU over GPU-computed
time series. The stress tensor observable may benefit from a GPU shader
(pair force outer product reduction) but can start as a CPU post-process.

---

## ToadStool Handoff Notes

- hotSpring Phases C-F validated that RTX 4070 can sustain f64 MD at 149-259 steps/s
- The `log_f64` bug (found by wetSpring) affects any lattice observable using logarithms
- Cell-list O(N) scaling works for short-range forces; lattice QCD may need all-to-all for long-range
- Energy drift of 0.000% over 80k steps sets the precision bar for any new integrator
- GPU sovereignty handoff delivered Feb 18, 2026: NVK warp-packed eigensolve,
  149x NAK gap analysis, driver persistence — see `wateringHole/handoffs/`
