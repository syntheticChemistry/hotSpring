# Validation Methodology

**Purpose**: Document the two-phase validation approach used in the hotSpring study  
**See also**: [STUDY.md](STUDY.md) for complete results

---

## 1. Two-Phase Approach

Every published workload passes through two independent phases:

### Phase A: Python Control

Reproduce published computational physics results using the original Python implementations on our hardware. This establishes:
- **Correctness**: Our hardware produces correct physics
- **Performance baseline**: Traditional scientific Python stack (NumPy, SciPy, Numba, mystic)
- **Bug inventory**: Upstream issues found during reproduction

Phase A uses the original authors' code wherever possible. When code is inaccessible (Code Ocean gating), we reconstruct from the paper's description plus available data.

### Phase B: BarraCUDA Execution

Re-implement the same computations using only BarraCUDA native functions (Pure Rust + WGSL). Compare:
- **Accuracy**: chi2/datum on identical experimental datasets
- **Throughput**: evaluations per second
- **Evaluation efficiency**: function evaluations needed to reach solution quality
- **Dependencies**: external library count (BarraCUDA target: zero)
- **Reproducibility**: deterministic results from seed
- **Physical constraints**: Nuclear Matter Properties within published ranges

---

## 2. Workloads

### 2.1 Sarkas Molecular Dynamics (Phase A + C)

**Source**: Murillo Group, Michigan State University  
**Code**: [github.com/murillo-group/sarkas](https://github.com/murillo-group/sarkas)  
**Reference data**: Dense Plasma Properties Database

**Phase A**: Reproduce Dynamic Structure Factor S(q,omega) for Yukawa and Coulomb one-component plasmas. 12 cases spanning kappa = 0-3, Gamma = 10-1510. Five observables validated per case: DSF, SSF, VACF, g(r), total energy.

**Phase C (GPU MD)**: Re-execute the 9 PP Yukawa cases entirely on GPU using f64 WGSL shaders (`SHADER_F64`). Validate energy conservation, RDF, VACF, SSF, and diffusion coefficient against physical expectations. N=2000, Velocity-Verlet integrator, Berendsen thermostat, periodic boundary conditions.

**Acceptance criteria**:
- Phase A: Plasmon peak frequencies within 15% of reference spectra
- Phase C: Energy drift < 5%, RDF tail |g(r)-1| < 0.15, monotonic observable trends with coupling

### 2.2 Two-Temperature Model (Phase A only)

**Source**: Murillo Group / UCLA  
**Code**: [github.com/MurilloGroupMSU/Two-Temperature-Model](https://github.com/MurilloGroupMSU/Two-Temperature-Model)

Reproduce electron-ion temperature equilibration for noble gas plasmas (Ar, Xe, He). Both 0D (local) and 1D (hydrodynamic) formulations.

**Acceptance criterion**: Te-Ti equilibrium achieved within physically reasonable timescales.

### 2.3 Surrogate Learning (Phase A + B)

**Source**: Diaw et al. (2024), Nature Machine Intelligence  
**Code**: [Code Ocean DOI:10.24433/CO.1152070.v1](https://doi.org/10.24433/CO.1152070.v1) — **gated, registration denied**  
**Data**: [Zenodo DOI:10.5281/zenodo.10908462](https://doi.org/10.5281/zenodo.10908462) — open, CC-BY

Reproduce the iterative SparsitySampler + RBF surrogate workflow on 9 benchmark functions plus a physics-based EOS from validated Sarkas MD. The nuclear EOS headline result was reconstructed from first principles because the Code Ocean capsule is inaccessible.

**Acceptance criterion**: RBF surrogates converge (chi2 decreasing over rounds).

### 2.4 Nuclear EOS Parameter Optimization (Phase A + B)

**Problem**: Fit 10 Skyrme interaction parameters to minimize chi2 between calculated and experimental nuclear binding energies:

```
objective = chi2_BE/datum + lambda * chi2_NMP/datum
```

over 52 nuclei from the AME2020 mass evaluation (IAEA Nuclear Data Services).

#### Validation Levels

| Level | Physics | Fidelity | Cost per eval | Phase A | Phase B |
|-------|---------|----------|--------------|---------|---------|
| L1 | SEMF (analytic) | Low | ~microseconds | Python | BarraCUDA |
| L2 | Spherical HFB (SCF) | Medium | ~seconds | Python | BarraCUDA |
| L3 | Deformed HFB (SCF, 2D) | High | ~minutes | — | BarraCUDA (in progress) |

#### Optimization Methods

| Method | Algorithm | Library |
|--------|-----------|---------|
| DirectSampler | Round-based multi-start Nelder-Mead | BarraCUDA |
| SparsitySampler | RBF surrogate-guided + exploration | BarraCUDA / mystic |
| Python/mystic | SparsitySampler + NelderMeadSimplexSolver | SciPy ecosystem |

#### NMP Constraint Targets

From published literature (Chabanat 1998, Bender 2003):

| Property | Target | Sigma | Physical meaning |
|----------|--------|-------|-----------------|
| rho0 | 0.16 fm^-3 | 0.005 | Nuclear saturation density |
| E/A | -15.97 MeV | 0.5 | Binding energy per nucleon |
| K_inf | 230 MeV | 20 | Incompressibility |
| m*/m | 0.69 | 0.1 | Effective mass ratio |
| J | 32 MeV | 2 | Symmetry energy coefficient |

---

## 3. Comparison Protocol

1. Run Python reference with default settings, record chi2/datum and NMP
2. Run BarraCUDA with same experimental data, same parameter bounds, varied seeds
3. Compare on identical metrics: accuracy, NMP correctness, evaluation count, wall time
4. **Measure energy**: CPU via Intel RAPL, GPU via nvidia-smi polling (100ms interval), integrated to Joules
5. Report per-region accuracy breakdown (light A<56, medium 56-100, heavy 100-200, very heavy 200+)
6. Report bootstrap confidence intervals on chi2/datum
7. All results saved to JSON (structured benchmark reports with hardware inventory) for exact reproduction
8. Seed-deterministic: same seed produces identical results
9. **Three-substrate comparison**: Python, BarraCUDA CPU, BarraCUDA GPU — same physics, different hardware cost
10. **GPU MD validation** (Phase C): energy conservation, RDF physical trends, VACF diffusion, SSF compressibility — validated against known plasma physics expectations

---

## 4. Hardware

All experiments run on a single consumer workstation:

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (8P+8E, 24 threads) |
| RAM | 32 GB DDR5-4800 |
| OS | Pop!_OS 22.04 (Linux 6.17) |
| Rust | stable (rustc) |
| Python | 3.9 (Sarkas), 3.10 (surrogate, TTM) via micromamba |
| GPU (current) | RTX 4070 (SHADER_F64 confirmed, ~2x FP64:FP32 ratio via wgpu/Vulkan) |
| GPU (incoming) | Titan V (7.4 TFLOPS FP64) — planned for compute-bound L3 |

---

## 5. Inaccessible Systems

Three systems that could not be accessed during this study:

| System | Gate | Impact | Our Response |
|--------|------|--------|-------------|
| Code Ocean capsule (Diaw et al.) | Registration denied ("OS is denied") | Cannot run paper's nuclear EOS | Rebuilt from first principles |
| HFBTHO (ORNL Fortran solver) | Institutional access required | No axially-deformed reference | Built L3 from textbook equations |
| LANL nuclear simulation data | Restricted | No ground truth EOS | Used public AME2020 instead |

In each case, we replaced the gated system with an open equivalent built from public data and published equations. This is more work but produces a fully transparent, independently reproducible result.

---

## 6. Acceptance Criteria

| Level | Metric | Target |
|-------|--------|--------|
| L1 | chi2_BE/datum | < 10 (SEMF physics floor) |
| L2 | chi2_BE/datum | < 5 (spherical HFB physics floor) |
| L2 | NMP within 2sigma | 5/5 properties |
| L3 | chi2_BE/datum | < 0.5 (deformed HFB physics floor) |
| L3 | RMS (MeV) | < 1 MeV |

Current status: L1 target exceeded (**2.27** via DirectSampler, **1.52** via GPU extended run). L2 in progress (BarraCUDA best **16.11** via DirectSampler; Python SparsitySampler **1.93** — sampling strategy gap, not physics). GPU FP64 validated to sub-ULP precision (4.55e-13 MeV). L3 architecture built, energy functional debugging in progress.

### GPU MD (Phase C) — 80k production steps

| Observable | Criterion | Status (80k steps) |
|-----------|-----------|--------|
| Energy drift | < 5% | **All 9 cases <= 0.006%** |
| RDF tail | abs(g(r)-1) < 0.15 | **All 9 cases <= 0.0017** |
| RDF peak trend | Increases with Gamma | **Verified all 9** |
| Diffusion D* | Decreases with Gamma | **Verified all 9** |
| SSF S(k->0) | Physical compressibility range | **Verified all 9** |

Result: **9/9 PP Yukawa cases pass** at N=2000, 80k production steps on RTX 4070. 149-259 steps/s sustained. 3.7x GPU speedup vs CPU. 3.4x less energy per step. The 80k long run confirms energy conservation stability well beyond the 30k steps used in the original Sarkas study.

### GPU MD (Phase D) — N-Scaling + Native f64

- Native WGSL builtins (sqrt, exp on f64): 2-6× throughput improvement
- N-scaling: 500 → 20,000, 0.000% drift at all N
- Cell-list O(N) + branch-based wrapping (i32 % bug fixed)
- Paper parity: N=10,000 in **5.3 minutes**

### GPU MD (Phase E) — Paper-Parity Long Run

| Observable | Criterion | Status (N=10k, 80k steps) |
|-----------|-----------|--------|
| Energy drift | < 5% | **All 9 cases: 0.000-0.002%** |
| All-pairs vs cell-list | Physics-driven mode selection | **4.1× speedup for cell-list** |
| Total wall time | All 9 cases | **3.66 hours, $0.044** |

Result: **9/9 PP Yukawa cases pass** at N=10,000, 80k production steps — exact paper configuration. Cell-list (κ=2,3) runs at 118.5 steps/s avg; all-pairs (κ=1) at 28.8 steps/s avg. Toadstool GPU ops (BatchedEighGpu, SsfGpu, PppmGpu) wired into pipeline.

### Software Versions

| Component | Version | Notes |
|-----------|---------|-------|
| Rust | stable (1.77+) | `rustc --version` |
| wgpu | 0.19+ | Vulkan backend, SHADER_F64 |
| Python | 3.9 (Sarkas), 3.10 (surrogate, TTM) | via micromamba |
| Sarkas | v1.0.0 (pinned, fd908c41) | 3 patches applied |
| NumPy | 1.26+ | 2.x compat patches |
| PyTorch | 2.1+ (CUDA) | GPU RBF only |
| nalgebra | 0.32+ | CPU eigensolvers |
| rayon | 1.8+ | CPU parallel HFB |
| OS | Pop!_OS 22.04 (Linux 6.17) | |

### Grand Total: 160/160 Quantitative Checks Pass

| Phase | Checks | Description |
|-------|:------:|-------------|
| A (Python control) | 86 | 60 MD + 6 TTM + 15 surrogate + 5 EOS |
| C (GPU MD, N=2000) | 45 | 9 cases × 5 observables |
| D (N-scaling + builtins) | 16 | 5 N values + 6 cell-list diag + 5 native builtins |
| E (Paper-parity + rewire) | 13 | 9 long-run cases + 1 profiling + 3 GPU ops |
| **Total** | **160** | **All pass** |
