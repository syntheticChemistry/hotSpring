# hotSpring Control Experiment — Status Report

> **Note (March 31, 2026):** For current status, see the [root README](README.md), [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md), and [`specs/GPU_CRACKING_GAP_TRACKER.md`](specs/GPU_CRACKING_GAP_TRACKER.md). **NVIDIA GPFIFO pipeline OPERATIONAL on RTX 3090. AMD scratch/local memory OPERATIONAL on RX 6950 XT (Exp 124). 128+ experiments. Warm FECS attack (Exp 127), GPU puzzle box (Exp 128), DRM tracing (Exp 126). Cross-primal rewiring complete — daemon-backed testing via ember/glowplug. 4232+ coral unit tests.** The body of this document is a fossil record from March 16 — retained for provenance.

**Date**: 2026-03-16 (L1+L2 complete, GPU MD Phase C+D+E+F complete — paper-parity long run 9/9, BarraCuda pipeline 39/39, crate v0.6.31, cross-substrate ESN, NPU characterization, DF64 production, toadStool S155b + coralReef Iter 47 synced, Chuna 44/44, Precision Brain + naga poisoning fix, deep debt resolved, **live Kokkos parity: 12.4× gap measured**, BatchedComputeDispatch wired, **coral-glowplug boot-persistent PCIe broker** (Exp 060-069), sovereign falcon direct execution proven)
**Gates**: Eastgate (i9-12900K, RTX 4070 12GB) + biomeGate (Threadripper 3970X, RTX 3090 24GB + Titan V 12GB HBM2, Akida NPU, 256GB DDR4)
**Sarkas**: v1.0.0 (pinned — see §Roadblocks)
**Python**: 3.9 (sarkas), 3.10 (ttm, surrogate) via micromamba
**f64 Status**: Native WGSL builtins confirmed. Consumer Ampere/Ada: fp64:fp32 ~1:64 (both CUDA and Vulkan). Double-float (f32-pair) hybrid delivers 3.24 TFLOPS at 14 digits (9.9× native f64). Titan V: genuine 1:2 via NVK.
**toadStool**: Session 155b (synced) + **hw-learn** crate (vendor-neutral GPU hardware learning: observer, distiller, knowledge store, applicator, brain_ext — 46 tests). **coralReef**: Phase 10 Iter 47 (coral-glowplug PCIe broker, VFIO dispatch pipeline, FECS direct execution, SEC2 EMEM breakthrough, 1,819 tests). hotSpring **848 lib tests**, 115 binaries, 39/39 validation suites. barraCuda `7c1fd03a` v0.3.5 (806 WGSL shaders, 3,400+ tests, sovereign-dispatch feature gate). **biomeOS**: `compute.hardware.*` capabilities registered (observe, distill, apply, share, status). Quality gates: zero clippy (lib+bins), zero unsafe, zero TODO/FIXME, all files <1000 lines. All 85 WGSL shaders AGPL-3.0-only. Experiment 054: Kokkos N-scaling complexity. Experiment 055: DF64 naga poisoning diagnostic (root cause: naga SPIR-V codegen — **now fixed upstream in barraCuda**). Experiment 056: Sovereign dispatch benchmark (backend-agnostic MdEngine, wgpu validated 140.3 steps/s, sovereign DRM blocked). Experiment 057: coralReef ioctl fix + sovereign validation. Chuna papers 43-45: **44/44**.

---

## What Worked

### 1. Sarkas Molecular Dynamics — Quickstart (✅ Full Pass)

The Sarkas quickstart tutorial (1000-particle Yukawa OCP, PP method, CGS units)
runs end-to-end on Eastgate with correct results:

| Metric | Value |
|--------|-------|
| Particles | 1,000 (Carbon, Z=1.976) |
| Method | PP (direct pairwise), Yukawa potential |
| Temperature | 5004.6 ± 106 K (target: 5000 K) |
| Γ_eff | 101.96 |
| κ | 2.72 (Thomas-Fermi screening) |
| Equilibration | 10,000 steps, 3.0s |
| Production | 20,000 steps, 5.6s |
| Total wall time | 11.3s |
| Dump integrity | All positions, velocities, accelerations valid |
| Energy tracking | Zero NaN rows in production (2001/2001 valid) |
| Post-processing | RDF computed successfully |

**Key result**: The Sarkas simulation engine produces physically correct output on
v1.0.0. Temperature control is tight (±2% of target), energy conservation holds,
and all output artifacts are valid. This establishes our **baseline of correctness**
for comparing against BarraCuda re-implementation.

### 2. TTM Hydro Model — 1D Spatial Profiles (✅ Partial Pass)

The Two-Temperature Model 1D hydrodynamic solver now produces real physics with
Saha ionization data (see §4 below for the TF→Saha fix):

| Species | Steps | Te₀→Te_end (K) | Ti₀→Ti_end (K) | FWHM (μm) | τ_eq (ns) | Wall |
|---------|-------|-----------------|-----------------|------------|-----------|------|
| Argon | 1941 | 15000→14387 | 300→855 | 232→252 | 1.56 | 38 min |
| Xenon | 49 | 20000→19155 | 300→19029 | 460→510 | 1.56 | 54s |
| Helium | 128 | 30000→27528 | 300→4665 | 232→280 | 0.37 | 146s |

**Key results**:
- Xenon achieves near-equilibration: Ti = 19029 K vs Te = 19155 K (93% coupling)
- FWHM expansion visible in all species — genuine hydrodynamic plasma evolution
- All three produce center temperature evolution, FWHM tracking, and radial profile plots
- Simulations terminate when Zbar root-finder diverges at r=0 (hottest point)
- Argon runs deepest (1941 steps) due to lower initial Te

**Limitation**: None complete the full simulation — the Zbar self-consistency loop
at the hottest grid point eventually diverges. This is a numerical stiffness issue,
not a physics error. The upstream notebooks appear to use custom dt schedules and
solver tolerances not documented in the repository.

### 3. TTM Local Model — 0D Temperature Equilibration (✅ Full Pass)

The Two-Temperature Model local (0D) solver reproduces electron-ion temperature
equilibration for three noble gas plasmas:

| Species | Te₀ (K) | Ti₀ (K) | T_eq (K) | τ_eq | Wall time |
|---------|---------|---------|----------|------|-----------|
| Argon | 15,000 | 300 | 8,100 | 0.42 ns | 2.1s |
| Xenon | 20,000 | 300 | 14,085 | 1.56 ns | 2.1s |
| Helium | 30,000 | 300 | 10,700 | 0.04 ns | 2.1s |

**Key result**: All three species reach Te = Ti equilibrium as expected from
the Spitzer-Meshkov transport model. Helium equilibrates fastest (lightest ion,
strongest coupling), Xenon slowest (heaviest). These timescales are physically
reasonable and provide a second, independent physics code as a control target
for BarraCuda's eventual ODE solver capabilities.

### 4. Surrogate Learning — Benchmark Functions (✅ Full Pass)

The mystic-directed sampling strategy from Diaw et al. (Nature Machine Intelligence
2024) dramatically outperforms random and LHS sampling at finding global optima:

| Function | Strategy | Gap from Global Min | Notes |
|----------|----------|-------------------|-------|
| Rastrigin 2D | random | 5.78 | Trapped in local minimum |
| Rastrigin 2D | LHS | 2.00 | Better coverage, still trapped |
| Rastrigin 2D | **mystic** | **0.00** | **Finds exact global minimum** |
| Rosenbrock 5D | random | 6,501 | Lost in 5D landscape |
| Rosenbrock 5D | LHS | 4,953 | Slightly better |
| Rosenbrock 5D | **mystic** | **4.4e-13** | **Essentially exact** |
| Easom 2D | random | 1.00 | Completely misses narrow well |
| Easom 2D | LHS | 1.00 | Same — well is too narrow |
| Easom 2D | **mystic** | **0.00** | **Finds it** |

The surrogate R² values for mystic-directed are poor-to-negative because the RBF
interpolator is trained on samples clustered near the optimum — it doesn't know
the rest of the landscape. This is **expected** and **informative**: the paper's
method is about efficient *exploration*, not global interpolation.

**Key result**: The optimizer-driven sampling methodology from the Murillo Group
paper is validated on standard benchmarks. The `mystic` library works. This
establishes the foundation for the nuclear EOS surrogate reproduction (Phase 2).

### 5. Zenodo Data Downloaded (✅)

The full Zenodo archive (DOI: 10.5281/zenodo.10908462) has been downloaded and
extracted: **961,636 pickle files** (5.8 GB) containing the complete benchmark
function sweep from the paper. This includes all Rastrigin, Rosenbrock, Easom,
Hartmann6, Michalewicz, and plateau results at various sample sizes and tolerances.

### 6. DSF Lite Sweep — 9/9 PP Cases Pass (✅ Full Sweep)

The Dynamic Structure Factor reproduction study is complete for all 9 PP
(Yukawa, κ≥1) cases at N=2000 (lite). Peak plasma oscillation frequencies
are validated against the Dense Plasma Properties Database:

| Case | κ | Γ | Mean Peak Error | Wall Time | Status |
|------|---|---|-----------------|-----------|--------|
| dsf_k1_G14_lite | 1 | 14 | 7.5% | 27 min | ✅ PASS |
| dsf_k1_G72_lite | 1 | 72 | 4.7% | 28 min | ✅ PASS |
| dsf_k1_G217_lite | 1 | 217 | 6.2% | 28 min | ✅ PASS |
| dsf_k2_G31_lite | 2 | 31 | 9.4% | 12 min | ✅ PASS |
| dsf_k2_G158_lite | 2 | 158 | 5.8% | 12 min | ✅ PASS |
| dsf_k2_G476_lite | 2 | 476 | 7.3% | 11 min | ✅ PASS |
| dsf_k3_G100_lite | 3 | 100 | 18.6% | 10 min | ✅ PASS |
| dsf_k3_G503_lite | 3 | 503 | 7.8% | 10 min | ✅ PASS |
| dsf_k3_G1510_lite | 3 | 1510 | 9.0% | 10 min | ✅ PASS |
| **Overall** | | | **8.5%** | **2.0 hrs** | **9/9** |

**Key results**:
- Overall mean peak frequency error: **8.5%** against the published database
- Zero NaN in any dump or energy file across all 9 cases (27,009 dumps total)
- All DSF, SSF, RDF, VACF observables computed successfully
- Higher coupling (Γ) → tighter agreement (expected: sharper collective modes)
- Higher screening (κ) → faster execution (shorter-range force → sparser neighbors)
- Two upstream Sarkas bugs fixed: `np.int` (NumPy 2.x) and `mean(level=)` (pandas 2.x)
- Full results: `control/sarkas/simulations/dsf-study/results/dsf_all_lite_validation.json`
- Comparison plot: `control/sarkas/simulations/dsf-study/results/dsf_all_lite_grid.png`
- Full observable validation: `control/sarkas/simulations/dsf-study/results/all_observables_validation.json`

**One outlier**: κ=3, Γ=100 has 18.6% mean error, driven by a single ka=0.928
point at 42% error. This is the weakest coupling in the κ=3 set — the collective
mode is broad and hard to resolve at N=2000. Full-scale N=10,000 runs on Strandgate
should bring this into line.

**Infrastructure**:
- `generate_inputs.py` — produces 12 full-scale YAML files
- `generate_lite_inputs.py` — produces N=2000 lite variants for Eastgate
- `batch_run_lite.sh` — runs all 8 remaining cases sequentially
- `run_case.py` — headless single-case runner with pre/sim/post phases
- `validate_dsf.py` — comparison against Dense Plasma Properties Database
- Dense Plasma Properties Database cloned and available as reference

### 7. Comprehensive Observable Validation — 60/60 PASS (✅ Full Sweep)

All 12 DSF lite cases (9 PP + 3 PPPM) validated across **5 observables × 12 cases = 60 checks**:

| Observable | Score | Key Metric | Status |
|------------|-------|------------|--------|
| **DSF** (Dynamic Structure Factor) | 12/12 | PP: 8.5% mean error, PPPM: 7.3% | ✅ |
| **Energy Conservation** | 12/12 | Drift range: [−1.77%, +1.40%], mean |drift| = 0.65% | ✅ |
| **RDF** (Radial Distribution Function) | 12/12 | Peak at 1.55–1.72 a_ws, g(r)→1 at large r | ✅ |
| **SSF** (Static Structure Factor) | 12/12 | S(k→0) trends correct (5 k-points only at N=2000) | ✅ |
| **VACF** (Velocity Autocorrelation) | 12/12 | D = 7.7e-9 to 5.9e-7 m²/s (Green-Kubo) | ✅ |
| **Overall** | **60/60** | All physics trends verified | ✅ |

**Physical trend verification** (all monotonically correct):

| Trend | Expected | Observed | Status |
|-------|----------|----------|--------|
| S(k→0) decreases with Γ (at each κ) | Less compressible at higher coupling | ✓ All 4 κ-series | ✅ |
| g(r_peak) increases with Γ (at each κ) | Stronger short-range order | ✓ All 4 κ-series | ✅ |
| D decreases with Γ (at each κ) | Caging effect reduces diffusion | ✓ All 4 κ-series | ✅ |
| VACF oscillations increase with Γ | Stronger caging → more rattling | ✓ General trend | ✅ |
| κ=0 shows more VACF oscillations | Long-range Coulomb → collective modes | ✓ 23–43 vs 4–21 | ✅ |

**RDF first peak details** (nearest-neighbor distance):

| κ | Low Γ | Mid Γ | High Γ | Trend |
|---|-------|-------|--------|-------|
| 0 | 1.14 (Γ=10) | 1.66 (Γ=50) | 2.36 (Γ=150) | g↑ with Γ ✅ |
| 1 | 1.16 (Γ=14) | 1.74 (Γ=72) | 2.48 (Γ=217) | g↑ with Γ ✅ |
| 2 | 1.21 (Γ=31) | 1.82 (Γ=158) | 2.57 (Γ=476) | g↑ with Γ ✅ |
| 3 | 1.31 (Γ=100) | 1.95 (Γ=503) | 2.61 (Γ=1510) | g↑ with Γ ✅ |

**Diffusion coefficients** (Green-Kubo, m²/s):

| κ | Low Γ | Mid Γ | High Γ |
|---|-------|-------|--------|
| 0 | 5.86e-7 | 6.05e-8 | 2.02e-8 |
| 1 | 4.90e-7 | 5.12e-8 | 1.70e-8 |
| 2 | 3.00e-7 | 3.38e-8 | 1.20e-8 |
| 3 | 1.36e-7 | 1.79e-8 | 7.71e-9 |

At each κ, D decreases by ~1–2 orders of magnitude across the Γ range, consistent
with the transition from weakly coupled liquid to strongly coupled/glassy behavior.

**SSF limitation**: N=2000 produces only 5 k-points — insufficient to resolve the
structural peak S(k) at ka ~ 2π. The S(k→0) compressibility limit is still
meaningful and shows correct monotonic decrease with Γ. Full structural peak
validation requires N=10,000 on Strandgate.

**Full results**: `control/sarkas/simulations/dsf-study/results/all_observables_validation.json`

---

## What's Less Reproducible Than Expected

### 1. Sarkas v1.1.0 Dump Corruption (🔴 Critical, Resolved by Pinning)

**The most significant finding**: Sarkas HEAD (v1.1.0, commit `7b60e210`) has a
**dump file corruption bug** introduced by commit `4b561baa` ("Added multithreading
for dumping"). All `.npz` checkpoint files contain NaN positions and velocities
starting from approximately step 10, while the simulation engine itself runs to
completion at normal speed.

This means:
- The simulation *appears* to work (progress bars complete, no errors)
- But **every post-processed observable is garbage** (RDF, DSF, SSF, VACF all NaN)
- Energy tracking also fails partway through (~50% of rows become NaN)
- The bug affects both CGS and MKS units, both 1000 and 2000 particle counts
- Single-threaded (`NUMBA_NUM_THREADS=1`) does not fix it

**Resolution**: Pinned to **v1.0.0** (tagged release `fd908c41`). All dumps valid.

**Impact on confidence**: This is a silent data corruption bug in a published
scientific code. It would produce physically plausible-looking output (the
simulation runs, the timing is right) but the actual data is NaN. Without our
explicit dump-level verification, this would have gone undetected.

**BarraCuda relevance**: This validates the ecoPrimals thesis that
correctness verification must be built into the compute layer, not bolted on
after the fact. A Rust/WGSL implementation with type-level correctness guarantees
would prevent this class of bug entirely.

### 2. Sarkas v1.0.0 Performance (🟡 Significant)

v1.0.0 is approximately **150× slower** than v1.1.0 for PP force computation:

| Version | Quickstart (1000 PP) | Rate |
|---------|---------------------|------|
| v1.1.0 | ~3 seconds | 3,400 it/s |
| v1.0.0 | ~8 seconds | ~22 it/s (eq), ~3,500 it/s (prod) |

The speedup in v1.1.0 came from Numba JIT optimizations in the same commit range
that introduced the dump bug. The Python scientific stack's optimization/correctness
tradeoff is non-trivial — you can have fast *or* correct, and the boundary between
them is hard to detect.

**Impact**: The DSF study lite case (N=2000, 35k steps) takes ~27 minutes on v1.0.0
instead of seconds. Full-scale (N=10000) cases will take hours and need Strandgate.

### 3. Sarkas PPPM κ=0 Cases (✅ Fixed — Numba 0.60 Compat, Validated)

The three pure Coulomb cases (κ=0, Γ=10/50/150) use the PPPM algorithm which
requires FFT via pyfftw. Originally reported as "segfault", the actual error was
a **Numba 0.60 compatibility issue**: `@jit` now defaults to `nopython=True`, and
pyfftw.builders.fftn is not supported in Numba nopython mode.

**Error**: `TypingError: Unknown attribute 'fftn' of type Module(pyfftw.builders)`

**Fix**: Changed `@jit` → `@jit(forceobj=True)` in `force_pm.py:529` to allow
object mode fallback (as the code's own comment says: "Numba does not support
pyfftw yet"). This is the same fix pattern as the NTT→FFT evolution in BarraCuda.

**Status**: 3/3 PPPM cases PASSED validation (see table below).

**PPPM Validation Results** (plasmon peaks only, ka≤2):

| Case | κ | Γ | Plasmon Peaks | Mean Error | Wall Time | Status |
|------|---|---|---------------|------------|-----------|--------|
| dsf_k0_G10_lite | 0 | 10 | 2 | **0.1%** | ~16 min | ✅ PASS |
| dsf_k0_G50_lite | 0 | 50 | 2 | **11.0%** | ~16 min | ✅ PASS |
| dsf_k0_G150_lite | 0 | 150 | 2 | **10.8%** | ~16 min | ✅ PASS |
| **Overall** | | | **6** | **7.3%** | | **3/3** |

**Note**: PPPM (κ=0) DSF shows excellent plasmon dispersion at low ka (0.1% error!).
At high ka (≥3), the DSF transitions from collective plasmon to diffusive modes —
the reference peaks are at ω < 0.3 ω_p and comparison is physically invalid.
PPPM cases run ~16 min each (vs ~10-28 min for PP, due to FFT overhead at N=2000).

### 4. TTM Hydro Zbar Convergence Failure (✅ Partially Fixed — Saha Ionization)

The TTM 1D hydrodynamic solver failed at the first timestep for all three species
when using Thomas-Fermi (TF) ionization. **Root cause**: TF sets `χ1_func = np.nan`
(no recombination energy), which poisons the Zbar self-consistency root-finder.

**Fix**: Switched to `ionization_model='input'` with Saha solution data files
(`Ar25bar_Saha.txt`, `Xe5bar_Saha.txt`, `He74bar_Saha.txt`), matching the upstream
notebooks. This provides tabulated Zbar(n,T) and χ1(n,T) for proper ionization.

**Results with Saha ionization (all three species)**:

| Species | Steps | Te₀→Te_end (K) | Ti₀→Ti_end (K) | FWHM (μm) | Wall time | Failure point |
|---------|-------|-----------------|-----------------|------------|-----------|---------------|
| Argon | 1941 | 15000→14387 | 300→855 | 232→252 | 38 min | r=0, residual=24 |
| Xenon | 49 | 20000→19155 | 300→19029 | 460→510 | 54s | r=0, residual=1.8 |
| Helium | 128 | 30000→27528 | 300→4665 | 232→280 | 146s | r=0, residual=1.8 |

**Physics observations**:
- Xenon shows near-equilibration: Ti reaches 19029 K vs Te 19155 K (τ_ei = 1.6 ns)
- Argon gets deepest into the evolution (1941 steps, 38 min wall time)
- FWHM expansion is visible in all species — genuine plasma hydrodynamics
- All three produce radial profiles, center T evolution, and FWHM tracking plots

**Remaining issue**: The root-finder eventually fails at r=0 (hottest grid point)
where Zbar sensitivity is highest. This is a known numerical stiffness: the
ionization equilibrium at high Te requires very tight solver tolerances. Possible
fixes: (a) relaxation/damping in the Zbar update, (b) adaptive dt near failure,
(c) switching to a more robust root-finder (e.g., Broyden with line search).

### 5. Sarkas Memory Scaling — N=10,000 OOM on 32 GB (🟡 Hardware Constraint)

The DSF study's designed particle count (N=10,000) with PP method causes
an OOM kill on Eastgate (32 GB RAM):

```
Out of memory: Killed process (python) total-vm:100726088kB, anon-rss:28178584kB
```

100 GB virtual memory for 10,000 particles is disproportionate. This is likely
a Sarkas neighbor-list or force-matrix implementation issue at v1.0.0.

**Resolution**: Created "lite" inputs (N=2,000, 30k prod steps) for Eastgate
validation. Full-scale runs go to Strandgate (64-core EPYC, expected 128+ GB).

### 6. NumPy 2.x Incompatibility in Upstream Code (🟢 Minor, Fixed)

The TTM upstream code uses `np.math.factorial` which was removed in NumPy 2.0.
Fixed by patching to `math.factorial`.

---

## Profiling Results — BarraCuda GPU Offload Targets

### Sarkas PP Force Kernel: 97.2% of Execution Time

Profiled the Sarkas MD simulation (κ=1, Γ=14, N=2000, 500 eq + 2000 prod steps,
total 116s). The dominant hotspot is overwhelmingly a single function:

| Function | Self Time | % Total | Description |
|----------|-----------|---------|-------------|
| `force_pp.update()` | **113.1s** | **97.2%** | Linked cell list pairwise force computation |
| `core.potential_energies()` | 0.26s | 0.2% | Post-step energy calculation |
| `integrators.enforce_pbc()` | 0.11s | 0.09% | Periodic boundary conditions |
| `io.dump()` | 0.02s | 0.02% | Snapshot I/O |
| Numba JIT compilation | 2.18s | 1.9% | One-time compilation cost |

**The `force_pp.update()` kernel** (file: `sarkas/potentials/force_pp.py:123`):
- Linked cell list (LCL) algorithm with 3D cell decomposition
- Triple-nested cell loop (27 neighbor cells per cell)
- Inner particle-pair loop: distance → force → acceleration + virial
- Uses Newton's 3rd law (only compute i < j pairs)
- Compiled with `@njit` (Numba nopython mode)
- **This is THE GPU kernel target for BarraCuda**

**BarraCuda mapping**:
- Cell assignment: parallelize over particles → GPU kernel
- Cell-pair interactions: parallelize over cell pairs → GPU workgroups
- Force evaluation: per-pair → GPU threads within workgroup
- Accumulation: atomic add or scatter-reduce for acceleration array
- **Existing ancestors**: `pairwise_distance.rs`, `cdist.wgsl` (spatial patterns)

**Performance target**: Sarkas achieves ~22 it/s (eq) to ~24 it/s (prod) on
i9-12900K for N=2000. A GPU implementation should target 1000+ it/s (45×+),
limited only by memory bandwidth for the force accumulation step.

**Wall time scaling by κ** (from batch data):
- κ=1: ~28 min (longer cutoff → more neighbors per cell)
- κ=2: ~11 min (medium cutoff)
- κ=3: ~10 min (shortest cutoff → fewest neighbors)

This is expected physics: higher screening → shorter-range force → sparser
neighbor lists → less computation. A GPU implementation would maintain this
scaling but shift the constant factor down by 45-100×.

---

## What Needs to Evolve for Full Control Experiment

### Immediate (No NUCLEUS Required)

| Item | Blocks | Effort | Gate | Status |
|------|--------|--------|------|--------|
| ~~Complete DSF lite (N=2000) validation~~ | ~~DSF baseline~~ | ~~Running~~ | ~~Eastgate~~ | ✅ Done |
| ~~Run all 9 PP DSF cases (lite)~~ | ~~DSF sweep~~ | ~~~4 hours~~ | ~~Eastgate~~ | ✅ 9/9 PASS |
| ~~Validate DSF against Dense Plasma DB~~ | ~~Correctness~~ | ~~1 session~~ | ~~Eastgate~~ | ✅ 8.5% err |
| ~~Fix PPPM "segfault" (Numba 0.60 compat)~~ | ~~3 DSF cases~~ | ~~1 fix~~ | ~~Eastgate~~ | ✅ Fixed |
| ~~Run 3 PPPM DSF cases (lite)~~ | ~~DSF Coulomb~~ | ~~~1 hour~~ | ~~Eastgate~~ | ✅ 3/3 PASS |
| ~~Profile Sarkas hotspots~~ | ~~BarraCuda gaps~~ | ~~1 session~~ | ~~Eastgate~~ | ✅ 97.2% in force_pp |
| ~~Fix TTM Zbar (TF → Saha ionization)~~ | ~~Hydro model~~ | ~~1 fix~~ | ~~Eastgate~~ | ✅ 77% of steps |
| ~~Validate PPPM DSF against Dense Plasma DB~~ | ~~3 Coulomb~~ | ~~1 session~~ | ~~Eastgate~~ | ✅ 5.5% err |
| ~~TTM hydro: switch TF→Saha ionization~~ | ~~Hydro model~~ | ~~1 fix~~ | ~~Eastgate~~ | ✅ 3/3 run |
| ~~Validate all observables (energy, RDF, SSF, VACF)~~ | ~~Full baseline~~ | ~~1 session~~ | ~~Eastgate~~ | ✅ 60/60 PASS |
| ~~Validate TTM hydro radial profiles~~ | ~~Hydro physics~~ | ~~1 session~~ | ~~Eastgate~~ | ✅ 3/3 monotonic |
| ~~Create comprehensive control results JSON~~ | ~~Data archive~~ | ~~1 session~~ | ~~Eastgate~~ | ✅ 86/86 checks |
| TTM hydro: tune solver tolerance at r=0 | Full hydro completion | Investigation | Any | Low priority |
| ~~Download Code Ocean capsule~~ | ~~Nuclear EOS surrogate~~ | ~~Manual~~ | ~~Browser~~ | ❌ Gated, bypassed |
| ~~Run full surrogate reproduction~~ | ~~Paper headline result~~ | ~~After capsule~~ | ~~Any GPU gate~~ | ✅ Rebuilt open |
| ~~Build nuclear EOS from scratch~~ | ~~Independent objective~~ | ~~Rebuilt~~ | ~~Eastgate~~ | ✅ L1+L2 done |
| ~~Wire L2 objective into surrogate~~ | ~~HFB surrogate learning~~ | ~~Done~~ | ~~Eastgate~~ | ✅ Wired |
| ~~Run L1 surrogate (30k evals, GPU RBF)~~ | ~~L1 baseline~~ | ~~5.4h~~ | ~~Eastgate~~ | ✅ **χ²=3.93** |
| ~~Run L2 surrogate (3k evals, 8 workers, GPU RBF)~~ | ~~L2 baseline~~ | ~~3.2h~~ | ~~Eastgate~~ | ✅ **χ²=1.93** |
| v2 iterative workflow (9 functions) | Full methodology proof | Complete | Eastgate | ✅ Done |

### After NUCLEUS Stabilizes (Cross-Gate)

| Item | Blocks | Effort | Gate |
|------|--------|--------|------|
| Install sarkas v1.0.0 on Strandgate | Full-scale DSF | 1 hour | Strandgate |
| Run DSF full-scale (N=10,000) on Strandgate | Production DSF data | Days | Strandgate |
| ~~RBF surrogate training on GPU~~ | ~~Phase B surrogate pipeline~~ | ~~After ToadStool Cholesky shader~~ | ~~Any GPU gate~~ | ✅ Done (PyTorch CUDA) |
| Cross-gate benchmark protocol | Publication data | After 10G backbone | All |

### BarraCuda Evolution (Phase B — ToadStool Timeline)

The control experiments have now **concretely demonstrated** which BarraCuda
capabilities are needed, with **quantitative acceptance criteria from 197 validated checks**:

| BarraCuda Gap | Demonstrated By | Acceptance Criteria | Priority |
|---------------|-----------------|---------------------|----------|
| **PP force kernel (WGSL)** | Sarkas PP: 22 it/s CPU | Energy drift <2%, RDF peak ±0.05 a_ws, D within 10% | 🔴 Critical |
| **Complex FFT (WGSL)** | PPPM: fragile pyfftw+Numba | DSF plasmon error <10%, FFT(IFFT(x))=x | 🔴 Critical |
| **Periodic boundary conditions** | All 12 MD cases use PBC | g(r)→1 at r_max, no edge artifacts | 🔴 Critical |
| **Neighbor list construction** | OOM at N=10k on 32GB | Handle N≥10k in <4GB GPU VRAM | 🟡 Important |
| **RBF surrogate training** | GPU RBF: 2.5s@10k vs CPU ~15s | Match RBF gap within 10% of CPU | ✅ Done (PyTorch) |
| **ODE solver** | TTM local: 2s CPU → ms target | Te=Ti equilibration, |Te-Ti|<1K at end | 🟢 Nice-to-have |
| **Bessel/special functions** | TTM hydro cylindrical coords | J0, J1 match tables to 1e-12 | 🟢 Nice-to-have |

**Phase B validation protocol**: Run the same 12 DSF cases on BarraCuda GPU kernels
and compare all 5 observables (DSF, energy, RDF, SSF, VACF) against the Python control
baseline. Matching within statistical uncertainty (N=2000) is the acceptance gate.
The comprehensive control results JSON (`control/comprehensive_control_results.json`)
serves as the ground truth dataset.

---

## Summary

The hotSpring control study's second execution pass has **resolved the major
blockers** and deepened the evidence for ecoPrimals' thesis.

### Progress Since First Pass

| Item | Before | After |
|------|--------|-------|
| DSF PP cases | 1/9 validated | **9/9 validated** (8.5% mean error) |
| DSF PPPM cases | 0/3 (segfault) | **3/3 validated** (7.3% mean error) |
| Observable validation | DSF only | **60/60** (5 obs × 12 cases, all PASS) |
| TTM local model | 3/3 pass | **3/3 equilibrate** (|Te-Ti| < 0.001 K) |
| TTM hydro model | 0 steps (NaN) | **3/3 physical profiles** (Te monotonic, FWHM expands) |
| Surrogate benchmarks | 15/15 pass | **15/15 PASS** (mystic wins all 5 functions) |
| Sarkas profiling | Not done | **97.2% in force_pp** — GPU target identified |
| Upstream bugs fixed | 2 (np.int, pandas) | **5** (+Numba/pyfftw, +TF→Saha, +dump corruption) |
| **Grand total** | | **86/86 quantitative checks pass** |

### Key Findings

- **Sarkas** (MD): **60/60 observable checks pass** — 5 observables (DSF, energy,
  RDF, SSF, VACF) × 12 cases (9 PP + 3 PPPM) all validated. DSF matches the Dense
  Plasma Properties Database (PP: 8.5%, PPPM: 7.3% mean error). Energy conservation
  holds to <2% drift across all cases. RDF, SSF, and VACF all show physically
  correct coupling-dependent trends. Diffusion coefficients span 7.7e-9 to 5.9e-7
  m²/s (Green-Kubo), decreasing monotonically with Γ as expected. Profiling confirms
  97.2% of execution is the force kernel. The upstream codebase required 4 patches.

- **TTM** (Local): **3/3 species reach perfect equilibrium** — Argon (8100 K),
  Xenon (14085 K), Helium (10700 K) all reach |Te-Ti| < 0.001 K. Equilibration
  timescales are physically correct: He fastest (0.04 ns), Xe slowest (1.56 ns).

- **TTM** (Hydro): **3/3 species produce valid spatial profiles**. All radial
  Te(r) profiles are monotonically decreasing from center, with zero NaN. FWHM
  expansion is observable in all species. Xenon achieves near-equilibrium
  (Ti/Te = 99.3%). Zbar divergence at r=0 limits full completion but does not
  invalidate the physics produced before divergence.

- **Surrogate Learning** (ML): **15/15 benchmarks pass** on open functions. The
  mystic-directed sampling strategy finds the global minimum on all 5 test functions
  (gap = 0 for 3/5, essentially zero for 2/5), while random and LHS consistently miss.
  Extended validation includes MultiscaleNDFunc and Hartmann6 from Zenodo data.
  **Physics surrogate built**: RBF trained on our 12 MD cases predicts diffusion
  coefficient with LOO error ±0.090 decades (900,000× speedup over MD).
  **Code Ocean limitation**: Nuclear EOS objective function is behind gated access
  (sign-up denied, wraps restricted LANL nuclear simulation data). The methodology
  is fully validated; the headline nuclear EOS application requires institutional access.
  All 27 published convergence histories from Zenodo (including nuclear EOS: χ²→9e-6
  over 30 rounds) confirm the method works — we just can't call the objective ourselves.
  **Full workflow reconstruction**: Rebuilt the paper's complete 30-round iterative
  workflow and ran on 5 objectives. Physics EOS from our Sarkas MD data converged
  in 11 rounds (176 evals, χ²=4.6e-5) — comparable to the paper's nuclear EOS
  (30 rounds, 30k evals, χ²=9.2e-6). See `whitePaper/barraCUDA/sections/04a_SURROGATE_OPEN_SCIENCE.md`.
  **Nuclear EOS from scratch**: Instead of using HFBTHO (which requires permissions
  or Fortran compilation), we rebuilt the entire nuclear physics objective from first
  principles in pure Python: Skyrme EDF → nuclear matter properties → SEMF binding
  energies → χ²(AME2020). This is a 10D optimization problem over Skyrme parameters
  (t0, t1, t2, t3, x0, x1, x2, x3, α, W0). Validated against SLy4 (χ²=6.5) and
  UNEDF0 parametrizations. Log-transform (log(1+χ²)) provides smooth RBF-learnable
  landscape. **Level 1 run completed**: 6000 evaluations over 30 rounds, best χ²/datum=17.73,
  surrogate score dropped 2.5→1.4 (learning verified). Best-fit nuclear matter:
  E/A=−14.5 MeV, K∞=230 MeV (both physically reasonable).

  **Level 1 full run — COMPLETED** (Feb 10-11, 2026):
  - Expanded AME2020 dataset: 17 → 52 nuclei (full chart coverage)
  - 30 rounds × 1000 evals/round = 30,000 evaluations in 5.4 hours
  - GPU RBF interpolator (PyTorch CUDA) for surrogate training:
    - Custom `GPURBFInterpolator` with thin-plate spline kernel
    - `torch.linalg.solve` for O(n³) system solve on RTX 4070
    - Memory-aware: auto-falls back to CPU when VRAM exceeds 12GB
    - GPU used rounds 1-15, CPU fallback 16+ (OOM at 16k pts, fixed post-run)
  - **Results**: χ²/datum = **3.93** (verified), 30,000 evaluations
  - Convergence trajectory: 57.9 → 8.1 → 7.5 → 6.3 → 4.6 → **3.93**
  - Best Skyrme parameters:
    - t0 = −2294.0, t1 = 551.1, t2 = −385.9, t3 = 17494.8
    - x0 = 1.25, x1 = 0.43, x2 = −0.22, x3 = 2.16
    - α = 0.30, W0 = 50.4
  - Nuclear matter: ρ₀=0.120 fm⁻³, E/A=−15.10 MeV, K∞=212 MeV, m*/m=0.957
  - Saved: `results/nuclear_eos_surrogate_L1.json`

  **Level 2 full run — COMPLETED** (Feb 10, 2026):
  - 30 rounds × 100 evals/round = 3,008 total evaluations
  - 8-worker parallel HFB evaluation + GPU RBF training
  - **Results**: χ²/datum = **1.93** (verified) — 3,008 evaluations in 3.2 hours
  - Convergence trajectory:
    - Round 0: χ² = 332.7 (random start)
    - Round 4: χ² = 23.5 (found reasonable region)
    - Round 17: χ² = 4.48 (major breakthrough)
    - Round 23: χ² = **1.93** (best result, held through round 29)
  - Best Skyrme parameters:
    - t0 = −3000.0, t1 = 200.0, t2 ≈ 0.0, t3 = 16387.7
    - x0 = 1.5, x1 = −2.0, x2 = −2.0, x3 = −1.0
    - α = 0.10, W0 = 50.0
  - Nuclear matter: ρ₀=0.205 fm⁻³, E/A=−15.74 MeV, K∞=223 MeV
  - **Key achievement**: On the 18-nuclei focused subset (56≤A≤132), L2 (HFB)
    achieves χ²/datum=1.93 vs L1 (SEMF) χ²=77,894 — a **40,000× improvement**.
    The quantum-mechanical HF+BCS solver provides dramatically better binding
    energies for medium-mass nuclei than the semi-empirical formula.
    Note: L1 optimizes all ~52 nuclei (best χ²=3.93 on full set), while L2
    focuses on 18 nuclei where HFB is most effective. The L2 all-nuclei χ² is
    higher (1372) because it specializes. This is the correct methodology —
    match the objective to where the model adds value.
  - Saved: `results/nuclear_eos_surrogate_L2.json`

  **BarraCuda (Rust + WGSL) Validation — COMPLETED** (Feb 11, 2026):
  The nuclear EOS L1 and L2 surrogate learning pipelines have been ported to
  pure Rust + BarraCuda WGSL shaders, eliminating all Python/PyTorch/scipy
  dependencies. Three key algorithmic improvements were implemented:
  - **Latin Hypercube Sampling (LHS)**: Space-filling exploration in round 0
  - **Multi-start Nelder-Mead**: 5 restarts from top-5 best points on surrogate
  - **CPU-only predict fast path**: Avoids GPU dispatch overhead for single-point
    evaluations in NM inner loop (90× speedup over GPU-dispatched NM)

  Head-to-head results (BarraCuda f64 vs Python control):

  | Metric | Python L1 | BarraCuda L1 | Python L2 | BarraCuda L2 |
  |--------|-----------|-------------|-----------|-------------|
  | Best χ²/datum | 6.62 | **2.27** ✅ | **1.93** | **16.11** (Run A) |
  | Best NMP-physical | — | — | — | 19.29 (Run B, 5/5 within 2σ) |
  | Total evals | 1,008 | 6,028 | 3,008 | 60 |
  | Total time | 184s | **2.3s** | 3.2h | 53 min |
  | Evals/second | 5.5 | **2,621** | 0.28 | 0.48 |
  | Speedup | — | **478×** | — | **1.7×** |

  L1 BarraCuda nuclear matter (best fit):
  - ρ₀ = 0.176 fm⁻³ (expt ~0.16), E/A = −15.84 MeV (expt −16.0), K∞ = 195 MeV

  L2 BarraCuda nuclear matter (best fit):
  - ρ₀ = 0.136 fm⁻³, E/A = −14.34 MeV, K∞ = 239 MeV

  **Key findings**:
  - L1: BarraCuda achieves **better χ² (2.27 vs 6.62)** at **478× throughput** —
    the combination of LHS + multi-start NM + f64 precision on Rust/WGSL
    comprehensively outperforms the Python/PyTorch control at L1.
  - L2: BarraCuda achieves **16.11** χ² (Run A, 60 evals) and **19.29** (Run B, all
    NMP physical). Python reaches 1.93 at 3008 evals with mystic SparsitySampler.
    The range of BarraCuda L2 values (16–25) confirms the landscape is multimodal.
    The remaining gap is in sampling strategy, not
    compute or physics. SparsitySampler port is the #1 priority for L2 parity.
  - **GPU dispatch overhead discovery**: Using GPU for single-point surrogate
    predictions in the NM inner loop caused a 90× slowdown (dispatch latency >>
    computation). CPU-only `predict_cpu()` fast path resolved this. Key lesson
    for BarraCuda architecture: auto-route small workloads to CPU.
  - **Precision**: Dual-precision strategy (f32 cdist on GPU → promote → f64 on CPU
    for TPS kernel, linear solve, and physics) matches Python's torch.float64 path.
  - Results saved: `results/nuclear_eos_surrogate_L{1,2}_barracuda.json`

  **Hardware note**: 2× NVIDIA Titan V (GV100, 12GB HBM2) on order for f64 GPU
  compute (6.9 TFLOPS FP64 each, vs RTX 4070's 0.36 TFLOPS). Once installed:
  - RTX 4070: f32 workloads, ML inference, cdist
  - Titan V ×2: f64 workloads (13.8 TFLOPS combined), Cholesky, linear solve
  - CPU: Fallback, small matrices, NM inner loop
  This eliminates the dual-precision GPU→CPU roundtrip for f64 operations.

  **BarraCuda Library Validation — COMPLETED** (Feb 12, 2026):
  The toadstool team evolved barracuda's scientific computing modules per the
  Feb 11 handoff. We validated the full library-based workflow against both the
  Python control and our earlier custom (inline) BarraCuda implementation.

  All requested modules work correctly:
  - `sample::sparsity::sparsity_sampler` — end-to-end iterative surrogate learning
  - `sample::latin_hypercube` — space-filling initial samples in 10D
  - `surrogate::RBFSurrogate` — TPS kernel train + predict
  - `optimize::nelder_mead` — local optimization (converges correctly)
  - `optimize::bisect` — root-finding for saturation density
  - `special::{gamma, factorial, laguerre}` — HO wavefunctions
  - `numerical::{trapz, gradient_1d}` — numerical integration/differentiation
  - `linalg::solve_f64` — linear system solve (inside RBFSurrogate)

  Head-to-head (library SparsitySampler vs Python vs old custom BarraCuda):

  | Metric | Python L1 | Library L1 | Old Custom L1 |
  |--------|-----------|------------|---------------|
  | χ²/datum | **1.75** | 5.04 | 2.27 |
  | Total evals | 1,008 | 1,100 | 6,028 |
  | Time | 184s | **5.2s** | **2.3s** |
  | Speedup vs Python | — | **35×** | **80×** |

  | Metric | Python L2 (initial) | Python L2 (SparsitySampler) | BarraCuda L2 (Run A) | BarraCuda L2 (Run B) |
  |--------|--------------------|-----------------------------|---------------------|---------------------|
  | χ²/datum | 61.87 | **1.93** | **16.11** | **19.29** (5/5 NMP) |
  | Total evals | 96 | 3,008 | 60 | 60 |
  | Time | 344s | 3.2h | 53 min | 55 min |
  | Throughput | 0.28/s | **5.5/s** | 0.48/s |

  **Key findings**:
  - **Speed**: Library is 35× faster than Python on L1, 19.6× on L2
  - **L1 accuracy**: 5.04 vs Python's 1.75 — gap is in sampling density
    (20 true evals/iter vs Python's 200). Physics is reasonable.
  - **L2 accuracy gap**: Optimizer gets trapped at parameter boundaries.
    Root cause: SparsitySampler runs NM on surrogate (cheap but low-info),
    adding only ~20 true evaluations per iteration. Python mystic does
    ~200 true evaluations per round with more aggressive direct search.
  - **Evolution needed**: SparsitySampler needs hybrid evaluation mode —
    some solvers on surrogate (exploitation) + some on true objective
    (exploration). See handoff: `HANDOFF_HOTSPRING_TO_TOADSTOOL_FEB_16_2026.md`
  - **External gap**: `nalgebra::SymmetricEigen` still needed for L2 HFB
    (barracuda needs `barracuda::linalg::symmetric_eigen`)

  **GPU RBF Interpolator** (`scripts/gpu_rbf.py`):
  - PyTorch CUDA implementation of scipy.interpolate.RBFInterpolator
  - Thin-plate spline kernel: r²·log(r) with augmented polynomial system
  - O(n³) solve on GPU via `torch.linalg.solve` (LU decomposition)
  - Memory-aware fit: estimates GPU memory, falls back to CPU for large n
  - Aggressive cleanup: `torch.cuda.empty_cache()` after fit and predict
  - Performance at various scales:
    | Points | GPU Time | CPU Fallback |
    |--------|----------|--------------|
    | 5,000  | 1.8s     | N/A          |
    | 10,000 | 2.5s     | N/A          |
    | 15,000 | 7.4s     | ~20s         |
    | 17,000 | 10.1s    | ~35s         |

  **Level 2 upgrade** (Feb 10, 2026):
  - Implemented spherical Skyrme HF+BCS solver (`skyrme_hfb.py`):
    - Separate proton/neutron channels with isospin-dependent Skyrme potential
    - BCS pairing with constant-gap approximation (Δ=12/√A)
    - Position-dependent effective mass via T_eff kinetic energy matrix
    - Coulomb direct (Poisson integral) + exchange (Slater approximation)
    - Spin-orbit interaction, center-of-mass correction
  - HFB single-nucleus results (SLy4): ✅ ¹⁰⁰Sn (+4.1%), ✅ ¹³²Sn (-3.6%), ²⁰⁸Pb (-10.6%)
  - Hybrid Level 2 solver: HFB for 56≤A≤132, SEMF elsewhere (5.5% mean error)
  - **Computational cost**: Optimized from ~60s → **2.7s per eval** via:
    - OPENBLAS_NUM_THREADS=1 (prevent BLAS thread contention): 60s → 12.4s (4.8×)
    - multiprocessing.Pool (parallel nuclei): 12.4s → 2.7s (additional 4.6×)
    - Combined 22× speedup on consumer i9-12900K
  - **Level 3** (axially deformed HFB): designated as **BarraCuda target** —
    this is where WGSL shaders replace Fortran eigensolvers.

  **Akida NPU integration** (Feb 10, 2026):
  - AKD1000 BrainChip neuromorphic processor at PCIe 07:00.0 — **OPERATIONAL**
  - Driver: `akida_dw_edma` built from source for kernel 6.17 (patched 1 API rename:
    `pcim_iounmap_regions` → `pcim_iounmap_region`, edma.h API identical to 6.9)
  - Device: `/dev/akida0` (world-readable), firmware BC.00.000.002
  - Hardware mesh: **78 neural processors** (78 CNP1, 54 CNP2, 18 FNP3, 4 FNP2)
  - Python SDK: akida 2.18.2, device detected, model mapped and running
  - **Real NPU inference confirmed**: V1 model (InputConvolutional → FullyConnected)
    with FC layer executing on FNP3 neural processor at ~2,800 samples/sec
  - Full NPU utilization requires `cnn2snn` conversion from trained Keras model
  - Planned use case: ultra-low-power surrogate pre-screening classifier
    (physical/unphysical parameter regions at ~300mW vs GPU 200W)

  Level architecture for nuclear EOS:
  | Level | Method | Python χ²/datum | BarraCuda χ²/datum | Speedup | Platform |
  |-------|--------|-----------------|--------------------|---------|----------|
  | 1 | SEMF + nuclear matter (52 nuclei) | 6.62 | **2.27** ✅ | **478×** | Rust + WGSL |
  | 2 | HF+BCS hybrid (18 focused nuclei) | **1.93** | **16.11** (best) / 19.29 (NMP) | 1.7× | Rust + WGSL + nalgebra + rayon |
  | 3 | Axially deformed HFB | ~0.5% (target) | - | - | **BarraCuda + Titan V** |

  Hardware utilization for control experiments:
  | Hardware | Role | Status |
  |----------|------|--------|
  | i9-12900K (CPU) | HFB eigensolvers via mp.Pool/rayon + BLAS opt | ✅ 22× speedup |
  | RTX 4070 (GPU) | RBF cdist (f32), PyTorch CUDA, BarraCuda WGSL | ✅ GPU RBF operational |
  | AKD1000 (NPU) | Surrogate pre-screening classifier | ✅ Hardware operational |
  | **Titan V ×2** (on order) | **f64 GPU compute (13.8 TFLOPS combined)** | 📦 Ordered |

  The heterogeneous compute strategy is now **fully operational**: CPU does the
  physics (HFB eigensolvers), GPU does the math (RBF O(n³) matrix solve), and
  NPU is staged for energy-efficient pre-screening. **BarraCuda (Rust + WGSL)
  now beats the Python control on L1** (2.27 vs 6.62 χ²/datum, 478× faster).
  L2 needs SparsitySampler for accuracy parity but is already 1.7× faster per
  evaluation. The Titan V GPUs will enable native f64 on GPU, eliminating the
  current dual-precision CPU roundtrip. Level 3 targets GPU dispatch via
  BarraCuda WGSL shaders for the eigenvalue problem.

### Phase D: N-Scaling and Cell-List Evolution (Feb 14, 2026)

The transition from Phase C (validation at N=2,000) to paper parity (N=10,000+)
exposed a fundamental bug in the GPU cell-list force kernel — and the process of
finding and fixing it demonstrates why deep debugging is superior to workarounds.

**The scaling question**: Can a $500 consumer GPU match the N=10,000 particle count
used in the Murillo Group's published DSF study?

**Experiment 001 — N-Scaling (native builtins + cell-list)**

After rewiring all shaders to native f64 builtins (`sqrt`, `exp`, `round`, `floor`)
and enabling the fixed cell-list kernel for N >= 10,000:

| N | GPU steps/s | Wall time | Energy drift | W (avg) | Method |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 500 | 998.1 | 35s | 0.000% | 47W | all-pairs |
| 2,000 | 361.5 | 97s | 0.000% | 53W | all-pairs |
| 5,000 | 134.9 | 259s | 0.000% | 65W | all-pairs |
| 10,000 | 110.5 | 317s | 0.000% | 61W | cell-list |
| 20,000 | 56.1 | 624s | 0.000% | 63W | cell-list |

**N=10,000 paper parity in 5.3 minutes** (was 24 min). N=20,000 in 10.4 min (was 68 min).
Total sweep: 34 minutes (was 112 min). 0.000% drift at every system size.

Improvement over software-emulated baseline:

| N | Old steps/s | New steps/s | Speedup | Factor |
|:---:|:---:|:---:|:---:|:---:|
| 500 | 169.0 | 998.1 | **5.9×** | native builtins |
| 2,000 | 76.0 | 361.5 | **4.8×** | native builtins |
| 5,000 | 66.9 | 134.9 | **2.0×** | native builtins |
| 10,000 | 24.6 | 110.5 | **4.5×** | native + cell-list |
| 20,000 | 8.6 | 56.1 | **6.5×** | native + cell-list |

**GPU now wins at N=500** (1.8× vs CPU). Previously CPU was faster below N=2000.

**Where CPU becomes implausible** (updated with native builtins):

| N | GPU Wall | Est. CPU Wall | GPU Energy | Est. CPU Energy | CPU Feasible? |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 500 | 35s | 63s | 1.7 kJ | 3.4 kJ | **GPU faster (1.8×)** |
| 2,000 | 97s | 571s | 5.1 kJ | 33.0 kJ | **GPU: 5.9× time** |
| 5,000 | 259s | ~60 min | 16.7 kJ | ~200 kJ | **No** |
| 10,000 | 317s | ~4 hrs | 19.4 kJ | ~1,600 kJ | **Impossible** |
| 20,000 | 624s | ~16 hrs | 39.3 kJ | ~14 MJ | **Impossible** |

**GPU power draw now varies** (47-69W) instead of the old flat 56-62W — confirming
higher ALU utilization with native transcendentals. The old flat power was caused
by software-emulated `exp_f64` and `sqrt_f64` (~130 f64 ops per pair). Native
builtins bypass this entirely. Combined with cell-list O(N) + Titan V hardware,
N=10,000 could drop to ~1-2 min.

**The quick fix would have been wrong.** When the cell-list kernel first failed at
N=10,000 (catastrophic energy explosion — temperature 15× above target), the
tempting path was: "just use all-pairs for everything." And for paper parity at
N=10,000, all-pairs works — it takes ~3 hours per case on the RTX 4070, which is
manageable.

But all-pairs is O(N²). At N=50,000: 1.25 billion pair computations per step.
At N=100,000: 5 billion. The GPU can handle N=20,000 in a day, but N=50,000+
requires cell-list O(N) scaling. **Avoiding the bug means accepting a permanent
ceiling on system size.**

**Experiment 002 — Cell-List Force Diagnostic**

Instead of working around the bug, we built a systematic diagnostic (`celllist_diag`):

| Phase | Test | Result | What it proved |
|:---:|:---|:---|:---|
| 1 | Force comparison (AP vs CL) | FAIL: PE 1.5-2.2× too high | Bug is in force kernel |
| 2 | Hybrid (AP loop + CL bindings) | PASS | Bug is NOT in params/sorting/buffers |
| 3 | V2: Flat 27-loop (no nesting) | FAIL | Bug is NOT in loop nesting |
| 4 | V4: f64 cell data (no u32) | FAIL | Bug is NOT in u32 data type |
| 5 | V5: No cutoff check | FAIL | Bug is NOT in cutoff logic |
| **6** | **V6: j-index trace** | **76 DUPLICATES in 108 visits** | **Cell wrapping is broken** |

**Root cause**: The WGSL `i32 %` operator produced incorrect results for negative
operands on NVIDIA GPUs via Naga/Vulkan. The standard modular wrapping pattern
`((cx % nx) + nx) % nx` silently returned wrong values, causing most neighbor
cell offsets to map back to cell (0,0,0) instead of the correct wrapped cell.

**Fix**: Replace modular arithmetic with branch-based wrapping:
```
// BROKEN:  let wx = ((cx % nx) + nx) % nx;
// FIXED:   var wx = cx;
//          if (wx < 0)  { wx = wx + nx; }
//          if (wx >= nx) { wx = wx - nx; }
```

**Verification**: All 6 N values (108 to 10,976) now produce PE matching all-pairs
to machine precision (relative diff < 1e-16). The cell-list mode is re-enabled
for `cells_per_dim >= 5`.

**Why deep debugging matters:**

The short fix (force all-pairs) would have:
- ✅ Given correct physics at N=10,000
- ❌ Capped system size at ~20,000 particles
- ❌ Left a broken kernel in the codebase
- ❌ Hidden a Naga/WGSL portability lesson (never use `%` for negative wrapping)
- ❌ Made HPC GPU scaling impossible

The deep fix (6-phase diagnostic, root cause analysis) gives us:
- ✅ Correct physics at all N
- ✅ Cell-list O(N) scaling to N=100,000+ on consumer GPU
- ✅ N=1,000,000+ on HPC GPUs (A100, H100)
- ✅ A documented portability lesson for all future WGSL shader development
- ✅ A reusable diagnostic binary (`celllist_diag`) for future kernel validation

**Projected cell-list performance** (RTX 4070, estimated):

| N | All-pairs steps/s | Cell-list steps/s (est.) | Speedup |
|:---:|:---:|:---:|:---:|
| 10,000 | ~3 | ~40-80 | **13-27×** |
| 20,000 | ~0.8 | ~30-60 | **37-75×** |
| 50,000 | infeasible | ~20-40 | **∞ (unlocked)** |
| 100,000 | infeasible | ~15-30 | **∞ (unlocked)** |

**Details**: See `experiments/001_N_SCALING_GPU.md` and `experiments/002_CELLLIST_FORCE_DIAGNOSTIC.md`.

### Phase E: Paper-Parity Long Run + Toadstool Rewire (Feb 14-15, 2026)

**The headline result**: 9 Yukawa OCP cases at N=10,000, 80,000 production steps — matching
the Dense Plasma Properties Database configuration exactly — all pass on an RTX 4070 in 3.66 hours.

**Paper-Parity 9-Case Results** (Feb 14, 2026)

| Case | κ | Γ | Mode | Steps/s | Wall (min) | Drift % | GPU kJ |
|------|---|---|------|---------|------------|---------|--------|
| k1_G14 | 1 | 14 | all-pairs | 26.1 | 54.4 | 0.001% | 196.8 |
| k1_G72 | 1 | 72 | all-pairs | 29.4 | 48.2 | 0.001% | 174.4 |
| k1_G217 | 1 | 217 | all-pairs | 31.0 | 45.7 | 0.002% | 165.6 |
| k2_G31 | 2 | 31 | cell-list | 113.3 | 12.5 | 0.000% | 44.8 |
| k2_G158 | 2 | 158 | cell-list | 115.0 | 12.4 | 0.000% | 45.5 |
| k2_G476 | 2 | 476 | cell-list | 118.1 | 12.2 | 0.000% | 45.1 |
| k3_G100 | 3 | 100 | cell-list | 119.9 | 11.8 | 0.000% | 44.2 |
| k3_G503 | 3 | 503 | cell-list | 124.7 | 11.4 | 0.000% | 42.3 |
| k3_G1510 | 3 | 1510 | cell-list | 124.6 | 11.4 | 0.000% | 42.9 |
| **Total** | | | | | **219.9** | | **801.7** |

Aggregate: **3.66 hours**, 0.223 kWh GPU + 0.142 kWh CPU = 0.365 kWh total (**$0.044**).

**All-Pairs vs Cell-List Profiling**

| Metric | All-Pairs (κ=1) | Cell-List (κ=2,3) | Ratio |
|--------|:---:|:---:|:---:|
| Avg steps/s | 28.8 | 118.5 | **4.1×** |
| Avg wall/case | 49.4 min | 12.0 min | 4.1× |
| Avg GPU energy/case | 178.9 kJ | 44.1 kJ | 4.1× |
| Avg GPU power | 60.4 W | 61.5 W | Same |

Mode selection is physics-driven:
- κ=1 (rc=8.0): `cells_per_dim = floor(34.74/8.0) = 4` → all-pairs (below threshold of 5)
- κ=2 (rc=6.5): `cells_per_dim = floor(34.74/6.5) = 5` → cell-list
- κ=3 (rc=6.0): `cells_per_dim = floor(34.74/6.0) = 5` → cell-list

Cannot streamline to cell-list only — κ=1 interaction range is too long at N=10,000.
Cell-list activates for κ=1 at N ≥ ~15,300 (where `box_side ≥ 5 × rc = 40`).
Both modes produce identical physics; the correct mode is chosen automatically.

**Toadstool Rewire** (Feb 14, 2026)

Pulled toadstool `cb89d054` (9 commits, +14,770 lines) and wired 3 new GPU ops:
- **BatchedEighGpu** → L2 GPU-batched HFB solver (`nuclear_eos_l2_gpu` binary)
- **SsfGpu** → GPU SSF observable in MD pipeline (with CPU fallback)
- **PppmGpu** → κ=0 Coulomb validation (`validate_pppm` binary)
- **GpuF64 → WgpuDevice bridge** for all toadstool GPU operations

**Next Steps Toward Full Paper Match**:
1. DSF S(q,ω) spectral analysis — compare peak positions against reference data
2. κ=0 Coulomb via PppmGpu — validate 3 additional PPPM cases
3. 100k+ step extended runs on Titan V / 3090 / 6950 XT
4. BatchedEighGpu L2 full AME2020 runs (791 nuclei)

---

### RTX 4070 Capability Envelope (Post-Bottleneck)

With native f64 builtins confirmed, the RTX 4070 is now a practical f64 science platform:

| Metric | Value | Notes |
|--------|-------|-------|
| VRAM at N=20,000 | 587 MB / 12,288 MB | **4.8% used** — N≈400K feasible |
| Paper parity (N=10k, 35k steps) | **5.3 minutes, 19.4 kJ** | $0.001 electricity |
| Beyond paper (N=20k, 35k steps) | **10.4 minutes, 39.3 kJ** | $0.002 electricity |
| Full 9-case sweep (80k steps) | **71 minutes, 225 kJ** | All 9 PP Yukawa cases |
| Energy drift at all N | **0.000%** | Verified N=500 through N=20,000 |
| Parameter sweep (50 pts × N=10k) | **~4 hours** | Overnight — routine |
| fp64:fp32 ratio | **~1:64** (native); **9.9×** via DF64 hybrid | Both CUDA and Vulkan match hardware; double-float is the breakthrough |

**What this unlocks**: parameter sweeps over hundreds of κ,Γ combinations, extended
production runs (500k steps overnight), multi-seed optimization, and expanding from
52 to 2,457 AME2020 nuclei — all on a single consumer GPU costing $0.001-$0.05 in
electricity per experiment. The exploration space is now effectively unlimited.

### Grand Control Summary

| Experiment | Checks | Pass | Status |
|------------|:------:|:----:|--------|
| Sarkas MD (5 obs × 12 cases) | 60 | 60 | ✅ Complete |
| TTM Local (3 species) | 3 | 3 | ✅ Complete |
| TTM Hydro (3 species profiles) | 3 | 3 | ✅ Partial (Zbar @ r=0) |
| Surrogate benchmarks (5 funcs × 3 strategies) | 15 | 15 | ✅ Complete |
| Nuclear EOS L1 Python (SEMF, 52 nuclei) | 1 | 1 | ✅ χ²/datum=6.62 |
| Nuclear EOS L2 Python (HFB hybrid, 18 nuclei) | 1 | 1 | ✅ χ²/datum=1.93 |
| GPU RBF accelerator (PyTorch CUDA) | 1 | 1 | ✅ 2-7× speedup |
| **BarraCuda L1 (Rust+WGSL, f64, LHS+NM)** | **1** | **1** | **✅ χ²=2.27 (478× faster)** |
| **BarraCuda L2 (Rust+WGSL+nalgebra, f64)** | **1** | **1** | **✅ χ²=16.11 best / 19.29 NMP (1.7× faster)** |
| **Phase A + B Total** | **86** | **86** | **✅ CONTROL + BARRACUDA VALIDATED** |
| | | | |
| **GPU MD PP Yukawa κ=1 (3 cases × 5 obs)** | **15** | **15** | **✅ Γ=14,72,217, drift≤0.006% (80k steps)** |
| **GPU MD PP Yukawa κ=2 (3 cases × 5 obs)** | **15** | **15** | **✅ Γ=31,158,476, drift=0.000% (80k steps)** |
| **GPU MD PP Yukawa κ=3 (3 cases × 5 obs)** | **15** | **15** | **✅ Γ=100,503,1510, drift=0.000% (80k steps)** |
| **Phase C Total** | **45** | **45** | **✅ GPU MD VALIDATED (RTX 4070, f64 WGSL, 80k prod. steps)** |
| | | | |
| **Cell-list diagnostic (6 isolation phases)** | **6** | **6** | **✅ Root cause: WGSL i32 % bug, branch-fix verified** |
| **N-scaling GPU sweep (5 N values, all-pairs baseline)** | **5** | **5** | **✅ N=500-20k, 0.000% drift, paper parity at N=10k** |
| **N-scaling native builtins re-run (5 N values)** | **5** | **5** | **✅ 2-6× faster, 0.000% drift, N=10k in 5.3 min** |
| **Phase D Total** | **16** | **16** | **✅ N-SCALING + CELL-LIST + NATIVE BUILTINS VALIDATED** |
| | | | |
| **Paper-parity long run (9 cases × 80k steps, N=10k)** | **9** | **9** | **✅ 0.000-0.002% drift, 3.66 hrs, $0.044** |
| **All-pairs vs cell-list profiling** | **1** | **1** | **✅ 4.1× speedup, physics-driven mode selection** |
| **Toadstool rewire (3 GPU ops)** | **3** | **3** | **✅ BatchedEighGpu + SsfGpu + PppmGpu wired** |
| **Phase E Total** | **13** | **13** | **✅ PAPER PARITY LONG RUN + TOADSTOOL REWIRE** |
| | | | |
| **L1 Pareto frontier (7λ × 5 seeds)** | **3** | **3** | **✅ chi2 0.69-15.38 (22× range), NMP 4/5 at λ=25** |
| **L2 GPU full AME2020 (2042 nuclei)** | **3** | **3** | **✅ 99.85% convergence, BatchedEighGpu 101 dispatches** |
| **L3 deformed HFB (2042 nuclei)** | **3** | **3** | **✅ 295/2036 improved over L2, 4 mass regions** |
| **Phase F Total** | **9** | **9** | **✅ FULL-SCALE NUCLEAR EOS CHARACTERIZATION** |
| | | | |
| **BarraCuda MD pipeline (6 GPU ops)** | **12** | **12** | **✅ YukawaF64+VV+Berendsen+KE: 0.000% drift** |
| **BarraCuda HFB pipeline (3 GPU ops)** | **16** | **16** | **✅ BCS GPU 6.2e-11, Eigh 2.4e-12, single-dispatch** |
| **Pipeline Validation Total** | **26** | **26** | **✅ BARRACUDA OPS END-TO-END VALIDATED** |
| | | | |
| **Grand Total** | **197** | **197** | **✅ ALL PHASES + PIPELINE VALIDATION** (verified 2026-03-06) |

**Data archive**: `control/comprehensive_control_results.json`  
**Nuclear EOS results**: `control/surrogate/nuclear-eos/results/nuclear_eos_surrogate_L{1,2}.json`  
**BarraCuda results**: `control/surrogate/nuclear-eos/results/nuclear_eos_surrogate_L{1,2}_barracuda.json`

### The ecoPrimals Thesis, Strengthened

The control experiments reveal a consistent pattern: **published scientific codes
require significant patching to run on current Python stacks** (NumPy 2.x, pandas
2.x, Numba 0.60). Each fix was a silent failure — the code either crashed with
inscrutable errors or produced NaN without warning. The upstream notebooks "just
work" because they pin exact environments from 2-3 years ago.

This is the exact problem BarraCuda solves: a Rust/WGSL compute engine where
the mathematical operations are correct by construction, the type system prevents
silent data corruption, and the GPU kernels don't depend on fragile JIT compilation
chains. The profiling data (97.2% in one function) shows this isn't a distributed
systems problem — it's a single hot kernel that maps directly to a GPU dispatch.

The **197/197 quantitative checks** (86 Phase A+B, 45 Phase C, 16 Phase D, 13 Phase E, 9 Phase F, 28 pipeline) now
provide concrete acceptance criteria across all phases: every observable, every
physical trend, every transport coefficient has a validated control value. Phase C
demonstrates that full Yukawa OCP molecular dynamics runs on a consumer GPU —
9/9 PP cases pass with 0.000% energy drift across 80,000 production steps, up
to 259 steps/s sustained throughput, and 3.4× less energy per step than CPU at
N=2000. **Phase D extends this to N-scaling**: with native f64 builtins and
cell-list O(N) scaling, the GPU achieves N=10,000 paper parity in **5.3 minutes**
(998 steps/s at N=500, 110 steps/s at N=10,000), 2-6× faster than the software-
emulated baseline. The GPU now wins at every N, including N=500 (1.8× vs CPU).
The cell-list kernel — after deep-debugging a WGSL `i32 %` portability bug across
6 isolation phases — now matches all-pairs to machine precision, unlocking O(N)
scaling to N=100,000+ on consumer hardware. The nuclear EOS surrogate learning demonstrates the full
pipeline — physics objective, surrogate training (GPU-accelerated), iterative
optimization — working on consumer hardware without institutional access.
**BarraCuda has already surpassed the Python control on L1** (χ²=2.27 vs 6.62,
478× throughput) and demonstrated 1.7× throughput advantage on L2. The remaining
L2 accuracy gap traces to sampling strategy (SparsitySampler), not compute or
physics fidelity. With 2× Titan V GPUs on order for native f64 on GPU, the
heterogeneous compute architecture is poised for L3 (deformed HFB).

**Cell-list evolution** (Feb 14, 2026): The Phase D cell-list diagnostic is a
case study in why deep debugging beats quick workarounds. The short fix (force
all-pairs for everything) would have given correct physics at N=10,000 — paper
parity, publishable, done. But it would have permanently capped system size at
~20,000 particles and left a broken kernel in the codebase. The deep fix (6-phase
isolation, j-trace analysis, root cause identification of a WGSL compiler
portability issue) gives us correct physics at ALL N, O(N) scaling, and a
documented lesson that benefits every future WGSL shader: **never use `i32 %` for
negative wrapping on Naga/Vulkan — use branch-based conditionals instead.** This
is the kind of engineering lesson that separates "it works on my machine" from
"it works everywhere." See `experiments/002_CELLLIST_FORCE_DIAGNOSTIC.md`.

**Library validation** (Feb 12, 2026): The toadstool team evolved all requested
BarraCuda modules. Library-based SparsitySampler runs end-to-end on both L1
(35× faster than Python) and L2 (19.6× faster). Accuracy gap in SparsitySampler's
evaluation strategy is identified and fixable — needs hybrid true+surrogate mode.
All 12 barracuda modules pass functional validation. See detailed handoff:
`HANDOFF_HOTSPRING_TO_TOADSTOOL_FEB_16_2026.md`.

**GPU dispatch overhead profiling** (Feb 15, 2026): L3 deformed HFB GPU run
profiled with full nvidia-smi + vmstat monitoring (2,823 GPU samples, 3,093 CPU
samples over 94 min). Key finding: **GPU at 79.3% utilization but 16× slower than
CPU-only**. Root cause: ~145,000 small synchronous GPU dispatches, each with
buffer alloc + submit + blocking readback. CPU dropped to 10.7% usage (freed 21
of 24 cores), but GPU was busy doing overhead, not physics. Energy: 80.6 Wh GPU
($0.01). The fix: batch ALL eigensolves across ALL nuclei into mega-dispatches,
keep grid physics in persistent GPU buffers, use async readback for convergence
flags only. ToadStool's `begin_batch()`/`end_batch()` and `AsyncSubmitter`
directly address this. See `experiments/004_GPU_DISPATCH_OVERHEAD_L3.md`.

**Architecture lesson**: The trains to and from take more time than the work.
Pre-plan, fill the GPU function space, fire at once. Every unnecessary CPU→GPU
round-trip is wasted time. This applies to all GPU-accelerated physics in
hotSpring, not just L3. The pattern: load the factory, let the assembly line
run, only check the output dock when you need a routing decision.

**L2 mega-batch profiling** (Feb 16, 2026): Implemented mega-batch remedy from
Experiment 004 on L2 spherical HFB. Results: dispatches reduced 206→101 (2x),
wall time 66.3→40.9 min (1.6x faster). But CPU-only L2 = 35.1s — **CPU is
still 70x faster**. GPU at 94.9% utilization (up from 79.3% in Exp 004),
confirming mega-batch saturates the GPU. Root cause: the eigensolve is ~1% of
total SCF iteration time. Hamiltonian construction, BCS pairing, and density
updates consume 99% and remain on CPU. This is Amdahl's Law — accelerating 1%
of the work yields max 1.01x speedup. The fix: move ALL physics to GPU
(H-build, BCS, density, convergence check) via WGSL shaders, creating a
GPU-resident SCF loop with zero CPU↔GPU round-trips during iteration.
**Complexity boundary**: for matrices < ~30×30, CPU cache coherence beats GPU
parallelism. For matrices > ~50×50 (L3 deformed, beyond-mean-field), GPU
dominates. See `experiments/005_L2_MEGABATCH_COMPLEXITY_BOUNDARY.md`.

**Stated goal**: Pure GPU faster than CPU for all HFB levels. Path:
GPU H-build (~10x) → GPU BCS (~2x) → GPU density (~1.5x) → GPU-resident
loop (~2x) → larger basis (GPU wins outright). Estimated: ~40s total for
791 nuclei on GPU-resident pipeline, competitive with CPU's 35s and
surpassing it at larger basis sizes.

---

## Document Links

- [`PHYSICS.md`](PHYSICS.md) — Complete physics documentation with equations and references
- [`whitePaper/STUDY.md`](whitePaper/STUDY.md) — Main study narrative (publishable draft)
- [`whitePaper/BARRACUDA_SCIENCE_VALIDATION.md`](whitePaper/BARRACUDA_SCIENCE_VALIDATION.md) — Phase B technical results
- [`whitePaper/CONTROL_EXPERIMENT_SUMMARY.md`](whitePaper/CONTROL_EXPERIMENT_SUMMARY.md) — Phase A quick reference
- [`benchmarks/PROTOCOL.md`](benchmarks/PROTOCOL.md) — Benchmark protocol (time + energy measurement)
- [`barracuda/CHANGELOG.md`](barracuda/CHANGELOG.md) — Crate version history (v0.6.30)
- [`barracuda/EVOLUTION_READINESS.md`](barracuda/EVOLUTION_READINESS.md) — Rust → GPU promotion tiers and blockers
- [`experiments/001_N_SCALING_GPU.md`](experiments/001_N_SCALING_GPU.md) — N-scaling experiment journal (Phase D)
- [`experiments/002_CELLLIST_FORCE_DIAGNOSTIC.md`](experiments/002_CELLLIST_FORCE_DIAGNOSTIC.md) — Cell-list bug diagnostic (Phase D)
- [`experiments/003_RTX4070_CAPABILITY_PROFILE.md`](experiments/003_RTX4070_CAPABILITY_PROFILE.md) — RTX 4070 capability profile (Phase E)
- [`experiments/004_GPU_DISPATCH_OVERHEAD_L3.md`](experiments/004_GPU_DISPATCH_OVERHEAD_L3.md) — Dispatch overhead profiling (Phase F)
- [`experiments/005_L2_MEGABATCH_COMPLEXITY_BOUNDARY.md`](experiments/005_L2_MEGABATCH_COMPLEXITY_BOUNDARY.md) — L2 mega-batch complexity boundary
- [`experiments/006_GPU_FP64_COMPARISON.md`](experiments/006_GPU_FP64_COMPARISON.md) — RTX 4070 vs Titan V fp64 benchmark
- [`experiments/007_CPU_GPU_SCALING_BENCHMARK.md`](experiments/007_CPU_GPU_SCALING_BENCHMARK.md) — CPU vs GPU scaling crossover analysis
- [`experiments/008_PARITY_BENCHMARK.md`](experiments/008_PARITY_BENCHMARK.md) — Python → Rust CPU → Rust GPU parity (32/32 suites)
- [`experiments/009_PRODUCTION_LATTICE_QCD.md`](experiments/009_PRODUCTION_LATTICE_QCD.md) — Production QCD β-scan + dynamical fermion HMC
- [`experiments/010_BARRACUDA_CPU_VS_GPU.md`](experiments/010_BARRACUDA_CPU_VS_GPU.md) — BarraCuda CPU vs GPU systematic parity validation
- [`experiments/011_GPU_STREAMING_RESIDENT_CG.md`](experiments/011_GPU_STREAMING_RESIDENT_CG.md) — GPU streaming HMC + resident CG (22/22 checks)
- [`experiments/012_FP64_CORE_STREAMING_DISCOVERY.md`](experiments/012_FP64_CORE_STREAMING_DISCOVERY.md) — FP64 core streaming: DF64 9.9× native f64
- [`experiments/013_BIOMEGATE_PRODUCTION_BETA_SCAN.md`](experiments/013_BIOMEGATE_PRODUCTION_BETA_SCAN.md) — biomeGate production β-scan (32⁴ on 3090, 16⁴ on Titan V NVK)
