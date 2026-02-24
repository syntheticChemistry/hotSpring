# Cross-Spring Shader & Primitive Evolution

**Date:** 2026-02-22
**From:** hotSpring (computational physics biome)
**License:** AGPL-3.0-only

---

## How the Biome Model Produces Cross-Spring Evolution

Each Spring writes GPU shaders and systems for its domain. ToadStool (the
fungus) absorbs working code into barracuda, the shared compute crate. Once
absorbed, every Spring benefits — a precision fix from wetSpring improves
hotSpring's nuclear physics, and hotSpring's eigensolve optimizations
accelerate neuralSpring's ML training. No Spring imports another. They
evolve together through the shared substrate.

```
hotSpring ──→ barracuda ←── wetSpring
                  ↑
            neuralSpring
```

---

## 1. Precision Infrastructure (Multi-Spring)

### The `math_f64.wgsl` Preamble — Where All Springs Converge

The f64 software math library (`math_f64.wgsl`) is the most cross-pollinated
artifact in barracuda. Every Spring that does f64 GPU computation relies on it.

| Contribution | Spring | Impact | Date |
|-------------|--------|--------|------|
| `(zero + literal)` pattern for f64 constants | **wetSpring** | Fixed `f64(0.333...)` truncation through f32. Improved all transcendentals. | Feb 16, 2026 |
| `log_f64` precision: ~1e-3 → ~1e-15 | **wetSpring** | Coefficients 2/3→1/3 etc. were wrong. All Springs using log now correct. | Feb 16, 2026 |
| `exp_f64` constant precision | **wetSpring** | Same `(zero + literal)` pattern applied to exp coefficients. | Feb 16, 2026 |
| `i32 % negative` wrapping fix | **hotSpring** | Cell-list bug: WGSL `%` on negative integers wraps differently from CPU. Branch-based modulo added. | Feb 15, 2026 |
| `complex_f64.wgsl` (14 functions) | **hotSpring** | c64_new through c64_exp — used by lattice QCD across all Springs. | Feb 19, 2026 |
| `su3.wgsl` SU(3) color matrices | **hotSpring** | 3×3 complex matrix algebra — used by gauge theory in any Spring. | Feb 19, 2026 |

**Cross-Spring benefit**: wetSpring's precision fixes to `log_f64` directly improved
hotSpring's nuclear EOS optimizer convergence (BCS bisection uses log). hotSpring's
complex arithmetic is used by any Spring doing lattice field theory.

### `ShaderTemplate` — Driver-Aware Compilation

| Method | Contributing Spring | What it does |
|--------|-------------------|-------------|
| `for_driver_auto()` | **hotSpring** | NVK exp/log workaround: detects NVK/NAK driver, substitutes software exp_f64/log_f64 |
| `for_driver_profile()` | **hotSpring** | Same + ILP reordering via `GpuDriverProfile::latency_model()` |
| `substitute_fossil_f64()` | **hotSpring** | Replaces hand-written sqrt_f64/abs_f64 with native WGSL builtins |
| `with_math_f64_auto()` | **hotSpring** | Auto-injects only the math_f64 functions called by a shader |
| `patch_warp_size()` | **hotSpring** | Adapts eigensolve warp/wave size for NVK (32) vs RDNA2 (64) |
| `math_f64_preamble()` | **hotSpring** + **wetSpring** | Returns the full math_f64 lib (with wetSpring precision fixes) |

**Cross-Spring benefit**: All Springs compile shaders through `ShaderTemplate`.
hotSpring's NVK workarounds prevent silent numerical garbage on nouveau/Mesa
drivers. wetSpring's precision fixes are inherited automatically.

---

## 2. GPU Hardware Profiling (hotSpring → All Springs)

### `GpuDriverProfile` — Data-Driven Shader Specialization

| Finding | Spring | Hardware | Impact |
|---------|--------|----------|--------|
| NVK 4-7× slower on compute-bound (driver maturity) | **hotSpring** | Titan V (NVK) vs RTX 4070 (nvidia) | All Springs know NVK perf characteristics |
| Warp-packed eigensolve: 2.2× NVK speedup | **hotSpring** | Titan V, Feb 2026 | `optimal_eigensolve_strategy()` returns `WarpPacked(32)` for NAK |
| RTX 4070 f64 transcendentals fail (NvvmAda) | **wetSpring** | RTX 4070 (proprietary driver) | `needs_exp_f64_workaround()` returns true for NVVM+Ada |
| NAK eigensolve: 5 shader workarounds | **hotSpring** | Titan V (NVK/NAK compiler) | `batched_eigh_nak_optimized_f64.wgsl` with documented workarounds |

**Cross-Spring benefit**: neuralSpring's ML training shaders that use f64
automatically get the right workarounds on NVK. wetSpring's bio shaders
that use exp/log on Ada get the NVVM workaround.

---

## 3. Bio Primitives (wetSpring → barracuda)

wetSpring contributed the largest set of domain shaders to barracuda:

| Primitive | Type | Use in Other Springs |
|-----------|------|---------------------|
| Smith-Waterman banded GPU | Sequence alignment | Could align nuclear structure basis functions |
| Gillespie SSA | Stochastic simulation | Stochastic nuclear decay chains |
| Felsenstein pruning | Phylogenetic likelihood | Statistical model validation patterns |
| DADA2 E-step | Bioinformatics pipeline | — |
| SNP calling | Variant detection | — |
| HMM forward f64 | Hidden Markov models | **neuralSpring** sequence models |
| ANI batch | Average nucleotide identity | — |
| dN/dS batch | Selection pressure | — |
| Quality filter | Read QC | — |
| RF batch inference | Random forest | **neuralSpring** ensemble methods; **hotSpring** ESN transport prediction |
| Bray-Curtis f64 | Diversity metric | — |
| Hill kinetics f64 | Hill equation | Reaction kinetics in any domain |
| `GemmCached` | Cached GEMM | **hotSpring** repeated Hamiltonian builds; **neuralSpring** training loops |
| Batched RK4 ODE | ODE integration | **hotSpring** time-dependent nuclear structure |

**Cross-Spring benefit**: wetSpring's `GemmCached` (60× taxonomy speedup)
directly benefits hotSpring's HFB SCF loop where the same-dimension GEMM
is called thousands of times. The `HMM forward f64` shader is used by
neuralSpring for sequence modeling.

---

## 4. ML Session Primitives (neuralSpring → barracuda)

neuralSpring contributed the ML layer operations:

| Primitive | Type | Use in Other Springs |
|-----------|------|---------------------|
| TensorSession | Session API | Any Spring doing GPU tensor ops |
| matmul (4-tier router) | GEMM dispatch | **hotSpring** matrix operations; **wetSpring** taxonomy |
| relu, gelu, softmax | Activations | — |
| layer_norm | Normalization | — |
| attention | Self-attention | — |
| `matmul_gpu_evolved` | Optimized WGSL GEMM | All Springs with matrix workloads |
| `BatchIprGpu` | Inverse participation ratio | **hotSpring** Anderson localization GPU |
| `mean_reduce` | GPU mean reduction | All Springs |
| `rk4_parallel` | Parallel RK4 integration | **hotSpring** ODE systems; **wetSpring** kinetics |
| pairwise_hamming, pairwise_jaccard | Distance metrics | **wetSpring** diversity analysis |
| `batch_fitness`, `locus_variance` | Population genetics | **wetSpring** evolutionary dynamics |
| `spatial_payoff` | Game theory | Evolutionary modeling |

**Cross-Spring benefit**: neuralSpring's `BatchIprGpu` gives hotSpring a GPU-
accelerated path for computing inverse participation ratios in Anderson
localization studies. The `rk4_parallel` shader can replace hotSpring's CPU
RK45 for embarrassingly parallel ODE integration.

---

## 5. Spectral Theory (hotSpring → barracuda → All Springs)

hotSpring's Kachkovskiy spectral extension was absorbed in full:

| Primitive | Origin | Use in Other Springs |
|-----------|--------|---------------------|
| Anderson 1D/2D/3D | **hotSpring** | Condensed matter in any Spring |
| Lanczos eigensolve | **hotSpring** | Large sparse eigenproblems in any domain |
| CSR SpMV (CPU + GPU shader) | **hotSpring** | Any sparse matrix computation |
| Hofstadter butterfly | **hotSpring** | Quasiperiodic systems |
| Sturm tridiagonal | **hotSpring** | Tridiagonal eigenvalues in any context |
| Level statistics | **hotSpring** | Localization diagnostics |

**Cross-Spring benefit**: wetSpring could use Lanczos for large phylogenetic
covariance matrices. neuralSpring could use CSR SpMV for sparse neural
network operations.

---

## 6. Lattice QCD (hotSpring → barracuda)

| Primitive | Type | Cross-Spring Value |
|-----------|------|-------------------|
| Wilson plaquette GPU | Gauge measurement | Any gauge theory simulation |
| SU(3) HMC force GPU | Molecular dynamics | Lattice field theory |
| Abelian Higgs HMC GPU | U(1)+scalar HMC | Simplified gauge theory validation |
| Staggered Dirac GPU | Fermion operator | Lattice QCD propagators |
| CG solver GPU (3 shaders) | Linear system | Iterative sparse solve on GPU |
| FFT f64 (1D + 3D) | Fourier transform | **wetSpring** spectral analysis; PPPM electrostatics |

**Cross-Spring benefit**: The GPU CG solver is a general iterative linear
system solver usable by any Spring. The FFT primitives enable momentum-space
operations across all domains.

---

## 7. MD Infrastructure (hotSpring ↔ barracuda)

| Primitive | Direction | Cross-Spring Value |
|-----------|-----------|-------------------|
| CellListGpu (3-pass GPU build) | barracuda → hotSpring | Any N-body simulation |
| ReduceScalarPipeline | barracuda → hotSpring | Any GPU reduction |
| Yukawa force variants (3 shaders) | hotSpring → barracuda | Screened Coulomb simulations |
| Velocity-Verlet integrator | hotSpring → barracuda | MD time integration |
| Berendsen thermostat | hotSpring → barracuda | Temperature control |
| RDF histogram | hotSpring → barracuda | Structural analysis |
| VACF/SSF observables | barracuda → hotSpring | Transport properties |

---

## 8. Evolution Timeline

```
Feb 15 — hotSpring: cell-list i32 modulo bug → math_f64.wgsl fix
Feb 16 — wetSpring: f64 constant precision → math_f64.wgsl (all Springs benefit)
Feb 16 — wetSpring: log_f64 ~1e-3→~1e-15 → all Springs using log
Feb 17 — hotSpring: NVK eigensolve profiling → GpuDriverProfile
Feb 18 — hotSpring: warp-packed eigensolve → 2.2× NVK speedup
Feb 19 — hotSpring: complex_f64 + SU(3) + lattice shaders → barracuda absorption
Feb 19 — hotSpring: ReduceScalarPipeline rewire ← barracuda
Feb 19 — hotSpring: CellListGpu BGL fix feedback → barracuda
Feb 20 — wetSpring: bio ops (Smith-Waterman, Gillespie, Felsenstein) → barracuda
Feb 20 — neuralSpring: TensorSession + ML ops → barracuda
Feb 20 — hotSpring: CellListGpu fix confirmed, local deprecated
Feb 20 — hotSpring: FFT f64 unblocked (barracuda bug fixes)
Feb 21 — hotSpring: full spectral module → barracuda absorption
Feb 22 — hotSpring: spectral lean complete (local sources deleted)
Feb 22 — hotSpring: 10/10 spectral, 10/10 Hofstadter, 10/10 Anderson 3D,
          11/11 Lanczos — all running on upstream barracuda spectral
```

---

## 9. What This Means

The biome model works. Specific evidence:

1. **wetSpring's precision fix improves hotSpring's physics**: The `(zero + literal)`
   pattern for f64 constants in `math_f64.wgsl` was discovered by wetSpring doing
   bio computations. It fixed `log_f64` from ~1e-3 to ~1e-15 precision. hotSpring's
   BCS bisection uses `log()` internally — it now converges faster and more accurately.

2. **hotSpring's driver profiling protects all Springs**: The NVK `exp()`/`log()`
   workaround was found by hotSpring testing on Titan V. Any Spring compiling
   shaders on NVK/nouveau now automatically gets correct results.

3. **neuralSpring's BatchIprGpu enables hotSpring's Anderson localization on GPU**:
   Inverse participation ratio is a key localization diagnostic. hotSpring can now
   compute it on GPU rather than CPU for 3D Anderson models with 32³+ sites.

4. **wetSpring's GemmCached accelerates hotSpring's HFB solver**: The cached GEMM
   pattern (60× speedup for repeated same-dimension multiplications) directly maps
   to the HFB SCF loop where thousands of same-size H-builds occur.

5. **No cross-imports needed**: Each Spring writes to its domain, tests locally,
   and hands off to ToadStool. The fungus absorbs. Every Spring leans.

---

## 10. Update: Sessions 43–53 Cross-Spring Absorption (Feb 22–24, 2026)

ToadStool Sessions 43–53 completed the full cross-spring absorption plan:
**26 items** from all 4 Springs absorbed into barracuda. 448 files changed,
82k lines churned. The fungus now holds the complete compute substrate.

### hotSpring Contributions (S45–S52)

| Item | Impact | Who Benefits |
|------|--------|-------------|
| CG solver shaders (5 WGSL) | 15,360× readback reduction | All Springs doing iterative GPU solves |
| Lattice QCD GPU: su3, dirac, fermion (12 WGSL) | Full gauge theory on GPU | neuralSpring (field theory nets), wetSpring (lattice bio) |
| GPU-resident CG infrastructure (ReducePass, StreamObservables) | Zero CPU round-trips | All Springs with iterative GPU compute |
| solve_f64 CPU fallback | ESN-sized matrices (50–200 dim) | wetSpring (ESN training), neuralSpring (small systems) |
| Screened Coulomb eigensolve | Yukawa screening on GPU | wetSpring (electrostatics), neuralSpring (charged systems) |
| atanh.wgsl bind group fix | Correct spectral computation | All Springs using hyperbolic functions |
| Core streaming / DF64 discovery | 9.9× f64 throughput on consumer GPUs | **All Springs** — the biggest find |

### wetSpring Contributions (S45–S52)

| Item | Impact | Who Benefits |
|------|--------|-------------|
| BatchedOdeRK4Generic + ODE trait | One shader replaces 5 domain-specific ODEs | hotSpring (nuclear kinetics), neuralSpring (dynamical systems) |
| ESN NPU weight export (int8 quantization) | GPU→NPU deployment pipeline | neuralSpring (inference), metalForge (dispatch) |
| 5 bio ODE shaders (phage, bistable, cooperation, capacitor) | Domain-specific kinetics | neuralSpring (evolutionary dynamics) |
| batch_pair_reduce_f64 FMA fix | Correct f64 reductions on all drivers | **hotSpring** (BCS bisection uses batch reductions) |
| FlatTree (Newick + edge constructors) | Phylogenetic tree GPU compute | neuralSpring (evolutionary nets) |
| ESN ridge regression | Readout training on GPU | neuralSpring (reservoir computing), metalForge (NPU) |

### neuralSpring Contributions (S45–S52)

| Item | Impact | Who Benefits |
|------|--------|-------------|
| 38 GPU ops promoted to dispatch API | Unified tensor→GPU interface | **hotSpring** (eigensolve dispatch), wetSpring (bio tensor ops) |
| Conv2D/Pool GPU wiring | LeNet-5 and CNN on GPU | wetSpring (image-based bio), airSpring (sensor CNNs) |
| Swarm neuroevolution shaders | Population-based optimization on GPU | wetSpring (evolutionary search), airSpring (control) |
| xoshiro128ss.wgsl (GPU RNG) | Reproducible parallel randomness | hotSpring (Monte Carlo), wetSpring (stochastic bio) |
| Tolerance registry | Centralized precision management | **All Springs** — consistent error tolerances |
| FST variance decomposition | Population genetics on GPU | wetSpring (metagenomics) |

### airSpring Contributions (S49)

| Item | Impact | Who Benefits |
|------|--------|-------------|
| Richards 1D solver (Van Genuchten-Mualem) | Unsaturated flow in soil | wetSpring (environmental modeling), hotSpring (porous media) |

### The Key Cross-Pollination Events

1. **hotSpring's DF64 core-streaming → everyone**: The double-float hybrid
   strategy discovered on biomeGate applies to ALL consumer GPU compute
   across every Spring. Estimated 7× HMC speedup, similar gains for any
   f64-heavy workload.

2. **wetSpring's batch_pair_reduce fix → hotSpring's BCS convergence**:
   The FMA→multiply+add fix in the f64 batch reduction shader was found by
   wetSpring doing DADA2 E-step comparisons. hotSpring's BCS bisection uses
   the same reduction pattern — it now produces bit-identical results across
   NVK, RADV, and proprietary drivers.

3. **neuralSpring's xoshiro128ss → hotSpring's Monte Carlo**: The GPU-native
   PRNG shader from neuralSpring's neuroevolution work is exactly what
   hotSpring needs for GPU-resident HMC random momentum generation.

4. **hotSpring's CG shaders → neuralSpring's attention**: The conjugate
   gradient solver infrastructure (alpha/beta computation, tree reduction)
   maps directly to iterative refinement in attention mechanisms.

### Validation: 39/39 on biomeGate (toadStool S53)

hotSpring compiled against toadStool S53 with **zero errors, zero warnings**.
Full validation suite: **39/39 PASS in 6542.6s** on biomeGate (RTX 3090 +
Titan V + Akida NPU, Threadripper 3970X).

| Suite | Time | Spring Origin |
|-------|------|--------------|
| Special Functions | 0.6s | hotSpring + wetSpring |
| Linear Algebra | 0.5s | hotSpring |
| MD Forces | 2.1s | hotSpring |
| Nuclear EOS | 3.2s | hotSpring |
| HFB (SLy4) | 4.2s | hotSpring + wetSpring (GemmCached) |
| WGSL f64 Builtins | 1.7s | hotSpring + wetSpring (precision) |
| BarraCuda HFB | 1.7s | hotSpring + neuralSpring (TensorSession) |
| BarraCuda MD | 15.7s | hotSpring + wetSpring (bio forces) |
| PPPM Coulomb | 1.7s | hotSpring |
| CPU/GPU Parity | 8.8s | hotSpring + wetSpring (precision) |
| NAK Eigensolve | 2.7s | hotSpring |
| GPU Transport (Paper 5) | 1260.0s | hotSpring + wetSpring |
| Screened Coulomb | 0.6s | hotSpring |
| HotQCD EOS | 0.4s | hotSpring |
| Pure Gauge SU(3) | 28.9s | hotSpring |
| QCD β-Scan | 52.9s | hotSpring |
| Dynamical Fermion QCD | 65.8s | hotSpring |
| Abelian Higgs | 0.5s | hotSpring |
| NPU Pipeline | 0.7s | wetSpring + metalForge |
| Lattice QCD + NPU | 14.2s | hotSpring + wetSpring + metalForge |
| Spectral Theory | 6.2s | hotSpring + neuralSpring (BatchIpr) |
| Kachkovskiy (4 suites) | 32.5s | hotSpring + neuralSpring |
| BarraCuda Evolution | 6.1s | All Springs |
| GPU SpMV | 1.9s | hotSpring |
| GPU Lanczos | 3.3s | hotSpring |
| GPU Dirac/CG/QCD | 5.5s | hotSpring |
| GPU Streaming HMC | 441.1s | hotSpring |
| GPU Streaming Dynamical | 770.3s | hotSpring |
| Reservoir Transport (ESN) | 118.5s | wetSpring + neuralSpring |
| Stanton-Murillo (Paper 5) | 2108.7s | hotSpring |
| Transport Parity | 1577.2s | hotSpring + wetSpring |

---

*License: AGPL-3.0-only. All discoveries, code, and documentation are
sovereign community property. No proprietary dependency required.*
