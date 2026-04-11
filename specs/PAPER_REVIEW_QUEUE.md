# hotSpring — Paper Review Queue

> Current state: v0.6.32, 964 tests, 140 binaries, 128 WGSL shaders. Paper reproduction
> priorities and status are still authoritative.

**Last Updated**: April 11, 2026
**Purpose**: Track papers for reproduction/review, ordered by priority and feasibility
**Principle**: Reproduce, validate, then decrease cost. Each paper proves the
pipeline on harder physics — toadStool evolves the GPU acceleration in parallel.
**Crate**: hotspring-barracuda v0.6.32 — 964 tests, 140 binaries, 128 WGSL shaders
**Current Goal**: GPU RHMC (Nf=2, 2+1) → gradient flow on RHMC configs → Chuna validation meeting (late April)

**Evolution path per paper**: Python Control → BarraCuda CPU → BarraCuda GPU → metalForge

---

## Pipeline Status: Every Paper × Every Substrate

| # | Paper | Python Control | BarraCuda CPU | BarraCuda GPU | metalForge |
|---|-------|:---:|:---:|:---:|:---:|
| 1 | Sarkas Yukawa OCP MD | ✅ `sarkas-upstream/` (12 cases) | ✅ `validate_md` | ✅ `sarkas_gpu` (9/9, 0.000% drift) | — |
| 2 | TTM (laser-plasma) | ✅ `ttm/` (3 species) | ✅ `validate_ttm` (RK4 ODE, 3 species) | — (CPU-only ODE) | — |
| 3 | Diaw surrogate learning | ✅ `surrogate/` (9 functions) | ✅ `nuclear_eos_l1_ref` (surrogate path) | ✅ `nuclear_eos_gpu` (GPU RBF) | — |
| 4 | Nuclear EOS (SEMF→HFB) | ✅ `surrogate/nuclear-eos/` | ✅ `validate_nuclear_eos` (195/195) | ✅ `nuclear_eos_l2_gpu` + `l3_gpu` | — |
| 5 | Stanton-Murillo transport | ✅ `sarkas/../transport-study/` | ✅ `validate_stanton_murillo` (13/13) | ✅ `validate_transport` (CPU/GPU parity); `validate_transport_gpu_only` (~493s) | ✅ NPU: ESN transport prediction |
| 6 | Murillo-Weisheit screening | ✅ `screened_coulomb/` | ✅ `validate_screened_coulomb` (23/23) | — (CPU-only eigensolve) | — |
| 7 | HotQCD EOS tables | — (data only, no sim) | ✅ `validate_hotqcd_eos` | — (data validation) | — |
| 8 | Pure gauge SU(3) | ✅ `lattice_qcd/quenched_beta_scan.py` | ✅ `validate_pure_gauge` (12/12) | ✅ GPU plaquette + HMC force shaders | ✅ NPU: phase classification |
| 9 | Production QCD β-scan | ✅ `lattice_qcd/quenched_beta_scan.py` | ✅ `validate_production_qcd` (10/10) | ✅ GPU CG (9/9) + Dirac (8/8) | ✅ NPU: `validate_lattice_npu` (10/10) |
| 10 | Dynamical fermion QCD | ✅ `lattice_qcd/dynamical_fermion_control.py` | ✅ `validate_dynamical_qcd` (7/7) + Omelyan + Hasenbusch | ✅ `validate_pure_gpu_hmc` (8/8) + `validate_gpu_streaming` (9/9): streaming GPU-resident HMC | ✅ NPU: dyn phase classify (100%, 3.2e-7) |
| 11 | Hadronic vacuum polarization | ✅ `lattice_qcd/hvp_correlator_control.py` (8/8) | ✅ `validate_hvp_g2` (10/10) | ✅ GPU streaming HMC pipeline (6 shaders + GPU PRNG) | — |
| 12 | Freeze-out curvature | ✅ `lattice_qcd/freeze_out_control.py` (8/8) | ✅ `validate_freeze_out` (8/8) | ✅ GPU streaming HMC pipeline (susceptibility via GPU plaquette) | — |
| 13 | Abelian Higgs | ✅ `abelian_higgs/abelian_higgs_hmc.py` | ✅ `validate_abelian_higgs` (17/17) | ✅ GPU Higgs shader absorbed | — |
| 14 | Anderson 1D localization | ✅ `spectral_theory/spectral_control.py` | ✅ `validate_spectral` (10/10) | ✅ GPU SpMV (8/8) | — |
| 15 | Aubry-André transition | ✅ (in spectral_control.py) | ✅ (in validate_spectral) | ✅ (via GPU SpMV) | — |
| 16 | Jitomirskaya metal-insulator | ✅ (in spectral_control.py) | ✅ (in validate_spectral) | ✅ (via GPU SpMV) | — |
| 17 | Herman Lyapunov bound | ✅ (in spectral_control.py) | ✅ (in validate_spectral) | — (transfer matrix, CPU) | — |
| 18 | Lanczos eigensolve | ✅ (in spectral_control.py) | ✅ `validate_lanczos` (11/11) | ✅ `validate_gpu_lanczos` (6/6) | — |
| 19 | 2D Anderson scaling | ✅ (in spectral_control.py) | ✅ `validate_lanczos` (2D checks) | ✅ (via GPU Lanczos) | — |
| 20 | 3D Anderson mobility edge | ✅ (in spectral_control.py — 3D extension) | ✅ `validate_anderson_3d` (10/10) | — (large matrix, P2) | — |
| 21 | Hofstadter butterfly | ✅ (in spectral_control.py) | ✅ `validate_hofstadter` (10/10) | — (Sturm, CPU-natural) | — |
| 22 | Ten Martini (Cantor) | ✅ (in spectral_control.py) | ✅ (in validate_hofstadter) | — (Sturm, CPU-natural) | — |
| 23 | Sulfolobus meta-populations | — | — | — | — (wetSpring domain) |
| 24 | Anderson subseafloor review | — (reference only) | — | — | — |
| 43 | Chuna: SU(3) gradient flow integrators | ✅ `gradient_flow_control.py` | ✅ `gradient_flow` (5 integrators, t₀ + w₀ scale, 14/14) | ✅ `gpu_flow` (7/7, 38.5×) | ✅ `gradient_flow_production` (HMC+flow) |
| 44 | Chuna: Conservative dielectric functions (BGK) | ✅ `bgk_dielectric_control.py` | ✅ `dielectric` (25 tests, std+completed Mermin) | ✅ `dielectric_mermin_f64` (std+completed) | ✅ `validate_dsf_vs_md` (14/14, MD parity) |
| 45 | Chuna: Multi-species kinetic-fluid coupling | ✅ `kinetic_fluid_control.py` | ✅ `kinetic_fluid` (16 tests + 20/20) | ✅ `bgk_relaxation_f64` (GPU BGK) | 322× CPU |

### Totals

| Substrate | Papers with validation | Coverage |
|-----------|:---:|:---:|
| **Python Control** | **24/25** | Papers 1-6, 8-22, 43-45 (only 23/wetSpring missing) |
| **BarraCuda CPU** | **25/25** | All except 23 (wetSpring domain) — Papers 43-45 now validated |
| **BarraCuda GPU** | **20/25** | Papers 1, 3-5, 8-19 + pure GPU HMC + dynamical GPU + β-scan |
| **metalForge (GPU+NPU)** | **9/25** | Papers 5, 8-10, 12-16 (transport + QCD + Higgs + spectral) |

### Missing Controls (Action Items)

| Paper | What's Needed | Effort | Priority |
|-------|--------------|--------|----------|
| 7 (HotQCD EOS) | No control needed — uses published reference data | — | — |
| 11 (HVP g-2) | ✅ DONE — `hvp_correlator_control.py` (8/8): staggered CG + HVP kernel + mass ordering on 4⁴ | — | — |
| 12 (Freeze-out) | ✅ DONE — `freeze_out_control.py` (8/8): susceptibility β-scan, β_c=5.50 (3.3% of known 5.69) | — | — |
| 20 (3D Anderson) | ✅ DONE — 3D Anderson added to spectral_control.py (Feb 22, 2026) | — | — |
| 23 (Sulfolobus) | Bioinformatics pipeline (wetSpring domain) | Medium | P3 |
| 24 (Anderson subseafloor) | No control needed — reference only | — | — |
| 43 (Chuna gradient flow) | ✅ DONE — `gradient_flow_control.py`: 5 integrators, t₀ + w₀ scale, convergence analysis | — | — |
| 44 (Chuna dielectric) | ✅ DONE — `bgk_dielectric_control.py`: standard + completed Mermin, f-sum, DSF | ✅ `dielectric.rs` 25 tests | ✅ `dielectric_mermin_f64.wgsl` (std + completed) |
| 45 (Chuna kinetic-fluid) | ✅ DONE — `kinetic_fluid_control.py`: BGK relaxation, Sod shock, coupled | ✅ `kinetic_fluid.rs` 16 tests | ✅ `bgk_relaxation_f64.wgsl` |

### Cross-Substrate Parity Green Board (April 2, 2026)

Full paper-queue parity orchestrator (`run_all_parity.py`) validates all papers
with Python control results against each other.  **9/9 active papers ALL GREEN**,
74 checks across 10 registered papers.

| Paper | Checks | Status | Mode |
|-------|:------:|--------|------|
| 6 (screened Coulomb) | 34/34 | ✅ ALL PASS | Eigenvalue parity (15 cases × z/κ/l) |
| 8 (pure gauge SU(3)) | 9/9 | ✅ ALL PASS | Per-β plaquette parity (9 β values) |
| 9 (production QCD) | 10/10 | ✅ ALL PASS | Same + monotonicity check |
| 10 (dynamical fermion) | 2/2 | ✅ ALL PASS | Dynamical + quenched plaquette |
| 11 (HVP g-2) | 4/4 | ✅ ALL PASS | HVP positivity, C(t) shape, mass ordering |
| 12 (freeze-out) | 3/3 | ✅ ALL PASS | β_c location, monotonicity, transition |
| 13 (Abelian Higgs) | 12/12 | ✅ ALL PASS | Per-config plaquette + Higgs condensate |
| 43 (gradient flow) | — | ✅ | Nested integrator data (cross-substrate requires Rust JSON) |
| 44 (BGK dielectric) | — | ✅ | Nested quadrature data (cross-substrate requires Rust JSON) |
| 45 (kinetic-fluid) | — | SKIP | Python control exists, results not yet committed |

**Total science cost**: ~$0.30 for 25 papers, 500+ validation checks, 102 experiments.
Papers 6, 7, 13-22 add checks at negligible cost (CPU-only, <15 seconds each).
Papers 43-45 (Chuna) complete — ~$0.05 additional (gradient flow reuses SU(3) infrastructure).
Experiments 096-100 (silicon characterization): ~$0.10 (budget, saturation, composition, QCD profiling).
GPU RHMC production (Exp 101): ~$0.02 (Nf=2 + Nf=2+1, 640 trajectories).
Gradient flow at volume (Exp 102): ~$0.03 (convergence + 16^4 scale setting, in progress).

---

## BarraCuda Evolution: CPU → GPU → metalForge

The evolution path validates the same physics on progressively more capable
substrates. Each level proves correctness before promoting to the next.

### Level 1: BarraCuda CPU (Pure Rust Math)

All physics implemented in pure Rust, validated against Python controls.
No GPU required. This is the correctness foundation.

| Domain | Binary | Checks | Rust vs Python |
|--------|--------|:---:|:---:|
| MD forces + integrators | `validate_md` | pass | — |
| TTM 0D ODE (laser-plasma) | `validate_ttm` | 8/8 | — |
| Nuclear EOS (L1-L3) | `validate_nuclear_eos` | 195/195 | 478× faster |
| Transport coefficients | `validate_stanton_murillo` | 13/13 | — |
| Screened Coulomb | `validate_screened_coulomb` | 23/23 | 2274× faster |
| Pure gauge SU(3) HMC | `validate_pure_gauge` | 12/12 | 56× faster |
| Production QCD β-scan | `validate_production_qcd` | 10/10 | — |
| Production QCD v2 (Omelyan) | `validate_production_qcd_v2` | 10/10 | — |
| Dynamical fermion QCD | `validate_dynamical_qcd` | 7/7 | — |
| Abelian Higgs HMC | `validate_abelian_higgs` | 17/17 | 143× faster |
| HotQCD EOS tables | `validate_hotqcd_eos` | pass | — |
| Spectral theory (1D/2D/3D) | `validate_spectral` + 3 more | 41/41 | 8× faster |
| Hofstadter butterfly | `validate_hofstadter` | 10/10 | — |
| HVP g-2 (correlator + kernel) | `validate_hvp_g2` | 10/10 | — |
| Freeze-out (susceptibility β-scan) | `validate_freeze_out` | 8/8 | — |
| Lattice QCD CG solver | `validate_gpu_cg` (CPU path) | 9/9 | 200× faster |
| Special functions + linalg | `validate_special_functions` + `validate_linalg` | pass | — |

**Status**: 22/22 papers have BarraCuda CPU validation (**COMPLETE**). Rust consistently
50×–2000× faster than Python for identical algorithms.

### Level 2: BarraCuda GPU (WGSL Shaders via wgpu/Vulkan)

GPU acceleration for compute-bound operations. Same physics, dispatched
to consumer GPU (RTX 4070 or any Vulkan SHADER_F64 device).

| Domain | Shader / Binary | Checks | GPU vs CPU |
|--------|----------------|:---:|:---:|
| Yukawa MD (all-pairs + cell-list) | `sarkas_gpu` | 9/9 at N=10k | 4.1× (cell-list) |
| Nuclear EOS L2 (batched HFB) | `nuclear_eos_l2_gpu` | 791 nuclei | 1.7× |
| Nuclear EOS L3 (deformed HFB) | `nuclear_eos_l3_gpu` | 295 improved | — |
| CPU/GPU parity | `validate_cpu_gpu_parity` | 6/6 | parity 1e-15 |
| Staggered Dirac operator | `validate_gpu_dirac` | 8/8 | parity 4.44e-16 |
| CG solver (D†D) | `validate_gpu_cg` | 9/9 | **22.2× at 16⁴** |
| Pure GPU QCD workload | `validate_pure_gpu_qcd` | 3/3 | parity 4.10e-16 |
| **Pure GPU HMC** (all math on GPU) | `validate_pure_gpu_hmc` | **8/8** | plaq 0.0e0, force 1.8e-15, KE exact, Cayley 4.4e-16 |
| **GPU HMC scaling** | `bench_gpu_hmc` | 4 sizes | **5×(4⁴), 23.8×(8⁴), 27.5×(8³×16), 34.7×(16⁴)** |
| **Streaming GPU HMC** (GPU PRNG + encoder batch) | `validate_gpu_streaming` | **9/9** | Bit-identical parity, 2.4×–40× vs CPU, zero CPU→GPU |
| **GPU β-scan** (production QCD) | `validate_gpu_beta_scan` | **6/6** | 9 temps on 8⁴ + 8³×16 cross-check, 82s total |
| **GPU dynamical fermion HMC** | `validate_gpu_dynamical_hmc` | **6/6** | force 8.33e-17, CG 3.23e-12, 90% accept |
| GPU SpMV (spectral) | `validate_gpu_spmv` | 8/8 | parity 1.78e-15 |
| GPU Lanczos eigensolve | `validate_gpu_lanczos` | 6/6 | parity 1e-15 |
| Transport CPU/GPU | `validate_transport` | pass | — |
| NAK eigensolve | `validate_nak_eigensolve` | pass | — |
| PPPM Coulomb | `validate_pppm` | pass | — |
| HFB pipeline | `validate_barracuda_hfb` | 16/16 | Single-dispatch (v0.6.7) |
| MD pipeline | `validate_barracuda_pipeline` | 12/12 | — |

**Status**: 20/22 papers have GPU validation paths. **Full GPU QCD pipeline validated**:
- **Quenched HMC** (8/8): 5 WGSL shaders at machine-epsilon parity
- **GPU HMC scaling**: 5×(4⁴), 23.8×(8⁴), 27.5×(8³×16), **34.7×(16⁴)**
- **Streaming GPU HMC** (9/9): single-encoder dispatch + GPU PRNG, 2.4×–40× vs CPU, zero CPU→GPU
- **GPU β-scan** (6/6): 9 temperatures on 8⁴ + 8³×16 cross-check in 82s
- **Dynamical fermion HMC** (6/6): GPU CG + fermion force shader, force parity 8.33e-17, 90% accept
- Transfer: 0 bytes CPU→GPU (GPU PRNG), 16 bytes GPU→CPU per trajectory (ΔH + plaq)

### Level 3: metalForge (GPU + NPU + CPU Heterogeneous)

Mixed-substrate dispatch: GPU generates data, NPU classifies/predicts,
CPU orchestrates. $900 total hardware cost.

| Domain | Binary / Script | Checks | Key Result |
|--------|----------------|:---:|:---:|
| NPU beyond-SDK capabilities | `validate_npu_beyond_sdk` | 16/16 | 10 SDK assumptions overturned |
| NPU quantization cascade | `validate_npu_quantization` | 6/6 | f32/int8/int4 parity |
| NPU physics pipeline | `validate_npu_pipeline` | 10/10 | MD→ESN→NPU→D*,η*,λ* |
| Lattice QCD + NPU phase | `validate_lattice_npu` | 10/10 | β_c=5.715 (0.4% error) |
| Heterogeneous monitor | `validate_hetero_monitor` | 9/9 | 5 previously impossible capabilities |
| NPU HW pipeline | `npu_physics_pipeline.py` | 10/10 | 9,017× less energy than CPU |
| NPU HW beyond-SDK | `npu_beyond_sdk.py` | 13/13 | Hardware-validated |
| NPU HW quantization | `npu_quantization_parity.py` | 4/4 | Hardware-validated |
| NPU lattice phase (HW) | `npu_lattice_phase.py` | 9/9 | GPU HMC → NPU classify |

| **Mixed-substrate pipeline** | `validate_mixed_substrate` | **9/9** | 4 domains × GPU→ESN→NpuSim, 100% classify, max err 4.9e-7 |
| **Three-substrate streaming** | `validate_streaming_pipeline` | **13/13** | CPU baseline→GPU stream→NPU screen→CPU verify |
| **Three-substrate + real NPU** | `validate_streaming_pipeline --features npu-hw` | **16/16** | AKD1000 discovered (80 NPUs, 10 MB SRAM), HW 100% agreement |

**Status**: 9 physics domains have heterogeneous pipeline validation:
transport (5), pure gauge (8), production QCD (9), dynamical QCD (10),
freeze-out (12), Abelian Higgs (13), Anderson 1D (14), Aubry-André (15),
Jitomirskaya (16). NPU inference at 30mW. GPU+NPU+CPU streaming validated.
**Real Akida AKD1000 discovered and validated from pure Rust** (Feb 23, 2026).

**Three-Substrate Streaming Pipeline** (`validate_streaming_pipeline`, **16/16** with `--features npu-hw`, Feb 23 2026):
Full end-to-end validation of the CPU→GPU→NPU→CPU architecture:
1. **CPU baseline** (4⁴): HMC across 7 β values establishes ground truth (23.2s)
2. **GPU parity** (4⁴): streaming HMC matches CPU within 2.7% (statistical), 1.5× faster
3. **GPU scale** (8⁴): 16× volume, streaming HMC in 56.1s, physically correct plaquettes
4. **NPU screening**: ESN trained on GPU observables, 86% accuracy; NpuSimulator 100% agreement
4b. **Real NPU hardware**: AKD1000 @ PCIe 0000:08:00.0, 80 NPUs, 10 MB SRAM, 100% classification agreement, max error 2.3e-7
5. **CPU verification**: β_c = 5.246 (error 0.446, within tolerance), correct phase classification
Transfer: 0 bytes CPU→GPU (GPU PRNG) | 16B GPU→CPU/traj | 24B GPU→NPU/traj

**Real NPU Hardware Integration** (Feb 23, 2026):
`akida-driver` (pure Rust, toadStool) wired into hotSpring as optional `npu-hw` feature.
`NpuHardware` adapter in `barracuda/src/md/npu_hw.rs` discovers the AKD1000 via PCIe sysfs,
probes capabilities, and provides the same `predict()` interface as `NpuSimulator`.
Host-driven ESN reservoir math runs at f32; readout on CPU. Model deployment to hardware
(`.fbz` construction from ESN weights) requires the Python MetaTF toolchain — the Rust
`akida-models` crate parses but does not yet build `.fbz` files.

### Level 4: Sovereign Pipeline (all substrates, no proprietary deps)

| Milestone | Status |
|-----------|--------|
| NVK/nouveau on Titan V | ✅ 6/6 parity, 40/40 transport |
| Both GPUs produce identical physics | ✅ to 1e-15 |
| NPU on open driver (akida PCIe) | ✅ 34/35 HW checks |
| AGPL-3.0 on all source | ✅ 106 .rs + 34 .wgsl |
| Zero proprietary dependencies | ✅ (wgpu → Vulkan → open driver) |
| Zero external FFI/C bindings | ✅ all pure Rust |

---

### metalForge NPU Pipeline Validation (Feb 20, 2026)

The GPU→NPU physics pipeline has been validated end-to-end, proving that
transport prediction can be offloaded to a $300 neuromorphic processor:

| Experiment | Script | Checks | Status |
|------------|--------|:------:|--------|
| Hardware capabilities | `npu_beyond_sdk.py` | 13/13 | 10 SDK assumptions overturned |
| Quantization cascade | `npu_quantization_parity.py` | 4/4 | f32/int8/int4/act4 parity |
| Physics pipeline (HW) | `npu_physics_pipeline.py` | 10/10 | MD→ESN→NPU→D*,η*,λ* |
| Beyond-SDK math | `validate_npu_beyond_sdk` | 16/16 | Substrate-independent |
| Quantization math | `validate_npu_quantization` | 6/6 | Substrate-independent |
| Pipeline math | `validate_npu_pipeline` | 10/10 | Substrate-independent |

**Key results**:
- NPU inference: 0.32J for 800 predictions vs CPU Green-Kubo: 2,850J (**9,017× less energy**)
- Streaming: 2,531 inferences/s at batch=8 with 2,301× headroom over MD rate
- Multi-output: D*, η*, λ* in single dispatch, 12.7% overhead vs single output
- GPU never stops: NPU runs on independent PCIe device, zero GPU overhead
- Hardware cost: ~$300, amortized over ~190k GPU-equivalent transport predictions

### Heterogeneous Real-Time Monitor (Feb 20, 2026)

Five previously-impossible capabilities demonstrated on $900 consumer hardware:

| Experiment | Script | Checks | Status |
|------------|--------|:------:|--------|
| Hetero monitor (math) | `validate_hetero_monitor` | 9/9 | Live HMC + cross-substrate + steering |

**Key results**:
- Live HMC phase monitoring: 9 μs prediction per trajectory (0.09% overhead)
- Cross-substrate parity: f64 → f32 (error 5.1e-7) → int4 (error 0.13)
- Predictive steering: adaptive β scan, 62% compute savings, β_c error 0.013
- Multi-output transport: D*, η*, λ* predicted simultaneously
- Zero-overhead: ESN prediction never stalls HMC simulation

### Lattice QCD Heterogeneous Pipeline (Feb 20, 2026)

Phase classification for pure-gauge SU(3) via GPU HMC + NPU inference:

| Experiment | Script | Checks | Status |
|------------|--------|:------:|--------|
| Lattice phase (HW) | `npu_lattice_phase.py` | 9/9 | GPU HMC → NPU classify confined/deconfined |
| Lattice phase (math) | `validate_lattice_npu` | 10/10 | Real HMC observables + ESN + NpuSimulator |

**Key results**:
- β_c detected at 5.715 (known: 5.692, error: 0.023 — 0.4%)
- ESN classifier: 100% accuracy on test data
- NpuSimulator f32 parity: max error 2.8e-7 (essentially identical to f64)
- Plaquette monotonically increases from 0.337 (β=4.5) to 0.572 (β=6.5)
- **No FFT required** — phase structure from position-space observables only
- Hardware: GPU ($600) generates configs, NPU ($300) classifies — ~$900 total

---

## Review Queue — Reordered by Feasibility

### Tier 0 — Immediate: Zero New Primitives Required

These papers can be reproduced using only existing BarraCuda capabilities.
The goal is maximum science per dollar with no infrastructure investment.

| # | Paper | Journal | Year | Faculty | What We Need | What We Have | Status |
|---|-------|---------|------|---------|-------------|-------------|--------|
| 5 | Stanton & Murillo "Ionic transport in high-energy-density matter" | Phys Rev Lett 116, 075002 | 2016 | Murillo | MD across Yukawa phase diagram, VACF → transport coefficients, Green-Kubo integrals | Sarkas GPU MD (9/9 PP), VACF observable, FusedMapReduceF64 | **Done** — D* calibrated to Sarkas (12 points), 13/13 checks |
| 6 | Murillo & Weisheit "Dense plasmas, screened interactions, and atomic ionization" | Physics Reports | 1998 | Murillo | Eigensolve for effective potentials, screened Coulomb theory | Sturm bisection eigensolve, screening models | **Done** — 23/23 checks |

**Paper 5 status**: ✅ Complete. Green-Kubo transport pipeline validated.
13/13 checks pass. Four bugs fixed:
1. **V² normalization bug** in stress/heat ACF: `compute_stress_acf` and
   `compute_heat_acf` used `V/kT` prefactor when the total-stress convention
   requires `1/(VkT)`. Fixed — η* now O(10⁻¹), matching independent MD literature.
2. **Green-Kubo integral noise**: Plateau detection added to all three Green-Kubo
   integrals (D* VACF, η* stress ACF, λ* heat ACF).
3. **Insufficient equilibration**: Transport cases now use 50k equil steps (lite)
   / 100k (full), with final velocity rescale to target T*.
4. **Fit coefficient normalization**: Daligault (2012) strong-coupling coefficients
   (A, α) were ~70× too small due to reduced-unit convention mismatch between
   the original Python baseline (`v += dt × F × Γ`, effective mass = 1) and
   standard OCP units (m* = 3, ω_p time). Recalibrated using 12 Sarkas DSF study
   Green-Kubo D* values at N=2000 (physical units → D* = D/(a²ω_p)). Corrected
   coefficients: A(κ) = 0.808 + 0.423κ − 0.152κ², α(κ) = 1.049 + 0.044κ − 0.039κ².
   D* now matches Sarkas within 50% at N=500 (statistical noise at small system size).
   η*/λ* coefficients proportionally rescaled (not independently calibrated).

**Paper 6 status**: ✅ Complete. Screened Coulomb (Yukawa) bound-state solver
validated. Sturm bisection eigensolve for tridiagonal Hamiltonian — O(N) per
eigenvalue, scales to N=10,000+ grid points. 23/23 validation checks:
4 hydrogen eigenvalue vs exact, 7 Python-Rust parity (Δ ≈ 10⁻¹²), 3 critical
screening vs Lam & Varshni (1971), 6 physics trends, 3 screening models.
Key: same Yukawa exp(−κr)/r potential as MD Papers 1/5 but for atomic
binding (electron-ion) instead of ion-ion interaction.

### Tier 1 — Short-term: Public Data, No Simulation Required

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 7 | HotQCD lattice EOS tables (Bazavov et al.) | Nuclear Physics A 931, 867 | 2014 | Bazavov | Download public EOS tables, validate downstream thermodynamics, compare to Sarkas plasma EOS | **Done** — `lattice/eos_tables.rs` + `validate_hotqcd_eos` |

**Paper 7 status**: ✅ Complete. HotQCD EOS reference table (Bazavov et al. 2014)
hardcoded in `lattice/eos_tables.rs` with 14 temperature points. Validation checks:
pressure/energy monotonicity, trace anomaly peak near T_c, asymptotic freedom
(approach to Stefan-Boltzmann limit), thermodynamic consistency (s/T³ = ε/T⁴ + p/T⁴).
All checks pass.

### Tier 2 — Medium-term: Needs Complex f64 + SU(3) (No FFT)

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 8 | Pure gauge SU(3) Wilson action (subset of Bazavov) | — | — | Bazavov | Complex f64, SU(3) matrix ops, plaquette force, HMC/Metropolis. NO Dirac solver. NO FFT. | **Done** — `lattice/` module + `validate_pure_gauge` (12/12) |

**Paper 8 status**: ✅ Complete. Full CPU implementation of pure gauge SU(3)
lattice QCD validated on 4^4 lattice:
- `complex_f64.rs`: Complex f64 arithmetic with WGSL template
- `su3.rs`: SU(3) matrix algebra (multiply, adjoint, trace, det, reunitarize,
  random near-identity, Lie algebra momenta) with WGSL template
- `wilson.rs`: Wilson gauge action — plaquettes, staples, gauge force (dP/dt),
  Polyakov loop. 100% test coverage.
- `hmc.rs`: Hybrid Monte Carlo with Cayley matrix exponential (exactly unitary),
  leapfrog integrator, Metropolis accept/reject. 96-100% acceptance at β=5.5-6.0.
- `dirac.rs`: Staggered Dirac operator, fermion fields, D†D for CG
- `cg.rs`: Conjugate gradient solver for D†D x = b
- `multi_gpu.rs`: Temperature scan dispatcher (CPU-threaded, GPU-ready)
- All 12/12 validation checks pass. Plaquette values match strong-coupling
  expansion and known lattice results. HMC ΔH = O(0.01).

**Paper 13 status**: ✅ Complete. Abelian Higgs (1+1)D: U(1) gauge + complex
scalar Higgs field with HMC. Validates the full phase structure from
Bazavov et al. (2015):
- Cold start identities (plaquette=1, action=0, |φ|²=1, Polyakov=1)
- Weak coupling (β=6): plaquette 0.915, 84% acceptance
- Strong coupling (β=0.5): plaquette 0.236, 90% acceptance
- Higgs condensation (κ=2): ⟨|φ|²⟩ = 4.42
- Confined phase (β=1, κ=0.1): intermediate plaquette
- Large λ=10: ⟨|φ|²⟩ ≈ 1.01 (φ⁴ potential freezes modulus)
- Leapfrog reversibility: |ΔH| = 0.002 at dt=0.01
- **Rust 143× faster than Python** (12 ms vs 1750 ms)
Key: bridges SU(3) (Paper 8) to quantum simulation — same HMC framework,
U(1) gauge group, complex scalar matter field. Wirtinger-correct forces.
17/17 validation checks.

### Kachkovskiy Extension — Spectral Theory / Transport (No FFT Required)

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 14 | Anderson "Absence of diffusion in certain random lattices" | Phys. Rev. 109, 1492 | 1958 | Kachkovskiy | Tridiagonal eigensolve, transfer matrix, Lyapunov exponent, level statistics | **Done** — 10/10 checks |
| 15 | Aubry & André "Analyticity breaking and Anderson localization" | Ann. Israel Phys. Soc. 3, 133 | 1980 | Kachkovskiy | Almost-Mathieu operator, spectral transition at λ=1 | **Done** — validated via Lyapunov exponent |
| 16 | Jitomirskaya "Metal-insulator transition for the almost Mathieu operator" | Ann. Math. 150, 1159 | 1999 | Kachkovskiy | Extended/localized phase classification | **Done** — Aubry-André transition detected |
| 17 | Herman "Une méthode pour minorer les exposants de Lyapunov" | Comment. Math. Helv. 58, 453 | 1983 | Kachkovskiy | Lyapunov exponent lower bound for quasiperiodic | **Done** — γ = ln|λ| validated to 4 decimal places |

**Paper 14-17 status**: ✅ Complete. Spectral theory primitives validated in `spectral/`.
All implemented without FFT — pure position-space operator theory:
- **Tridiagonal eigensolve**: Sturm bisection, all eigenvalues to machine precision
- **Transfer matrix**: Lyapunov exponent via iterative renormalization
- **Anderson 1D**: γ(0) = W²/96 (Kappus-Wegner) verified at 7% accuracy
- **Almost-Mathieu**: Herman's formula γ = ln|λ| verified to error < 0.0001
- **Aubry-André transition**: clean metal-insulator separation at λ = 1
- **Poisson statistics**: ⟨r⟩ = 0.3858 (theory 0.3863, 0.1% error)
- **W² scaling**: disorder ratios 4.07 and 4.04 (theory 4.0)

**Papers 18-19**: ✅ Complete. Lanczos + SpMV + 2D Anderson validated in `spectral/`.
P1 primitives for GPU promotion now working on CPU:
- **CsrMatrix + SpMV**: CSR sparse matrix-vector product, verified vs dense (0 error)
- **Lanczos eigensolve**: Full reorthogonalization, cross-validated vs Sturm (Δ = 4.4e-16)
- **2D Anderson model**: `anderson_2d()` on square lattice, open BCs
- **GOE→Poisson transition**: ⟨r⟩ = 0.543 (weak, GOE) → 0.393 (strong, Poisson)
- **Clean 2D bandwidth**: 7.91 (theory 8.0, error 1.1%)
- **2D vs 1D**: 2D bandwidth 1.61× wider than 1D at same disorder

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 18 | Lanczos "An iteration method for the solution of the eigenvalue problem" | J. Res. Nat. Bur. Standards 45, 255 | 1950 | Kachkovskiy | CSR SpMV, Krylov tridiagonalization, reorthogonalization | **Done** — Lanczos vs Sturm parity to 4.4e-16, convergence validated |
| 19 | Abrahams, Anderson, Licciardello, Ramakrishnan "Scaling theory of localization" | Phys. Rev. Lett. 42, 673 | 1979 | Kachkovskiy | 2D Anderson model, GOE→Poisson level statistics transition | **Done** — GOE (⟨r⟩=0.543) and Poisson (⟨r⟩=0.393) regimes resolved on 16×16 lattice |
| 20 | Slevin & Ohtsuki "Critical exponent for the Anderson transition" | Phys. Rev. Lett. 82, 382 | 1999 | Kachkovskiy | 3D Anderson model, mobility edge, W_c ≈ 16.5 | **Done** — mobility edge detected (center ⟨r⟩=0.516 > edge ⟨r⟩=0.494), GOE→Poisson transition, dimensional hierarchy 1D<2D<3D |

**Papers 20 status**: ✅ Complete. 3D Anderson model validated, proving the
only dimensionality with a genuine metal-insulator transition:
- **Mobility edge**: band center GOE (⟨r⟩=0.516) coexists with band edge
  localization (⟨r⟩=0.494) at W=12 — IMPOSSIBLE in 1D or 2D
- **Dimensional hierarchy**: bandwidth 5.12 (1D) < 8.22 (2D) < 11.42 (3D)
- **Statistics hierarchy**: 1D always Poisson (⟨r⟩=0.384), 3D GOE (⟨r⟩=0.543)
- **Transition**: Δ⟨r⟩ = 0.14 from W=2 to W=35, crossing near W_c ≈ 16.5
- **Particle-hole symmetry**: |E_min + E_max| = 0 (exact on clean lattice)

| 21 | Hofstadter "Energy levels and wave functions of Bloch electrons in rational and irrational magnetic fields" | Phys. Rev. B 14, 2239 | 1976 | Kachkovskiy | Band counting at rational flux, Cantor set measure, spectral topology | **Done** — q=2,3,5 bands detected, Cantor measure convergence (1.66→0.003), α↔1-α symmetry exact |
| 22 | Avila & Jitomirskaya "The Ten Martini Problem" | Ann. Math. 170, 303 | 2009 | Kachkovskiy | Cantor spectrum for almost-Mathieu at λ=1 | **Done** — spectral measure decreasing with q confirms Cantor convergence |

**Paper 21-22 status**: ✅ Complete. Hofstadter butterfly validated:
- **Band counting**: α=1/2 → 2 bands, α=1/3 → 3 bands, α=1/5 → 5 bands (exact)
- **Cantor convergence**: total spectral measure 1.66 (q=2) → 0.003 (q=21) → 0
- **Symmetries**: α ↔ 1-α bandwidth identical (Δ < 10⁻¹⁵), E → -E within 0.28
- **Gap opening**: rational flux opens gaps, irrational (golden) fragments into Cantor dust
- **Localized phase**: λ=2 opens all q-1 gaps (Avila global theory)
- **277 flux values** computed in 8.3s via Sturm bisection (Q_max=30)

**Python control baselines**:
- **Spectral theory** (`control/spectral_theory/scripts/spectral_control.py`):
  Anderson 1D, Lyapunov exponent, level statistics, 2D Anderson, Hofstadter bands.
  Results match Rust. **Rust 8× faster than Python** (4.7s vs 38.3s).
- **Lattice QCD CG** (`control/lattice_qcd/scripts/lattice_cg_control.py`):
  Staggered Dirac + CG solver on 4⁴ lattice. Identical LCG PRNG, same algorithm.
  **Rust 200× faster than Python**: CG 5 iters in 0.33ms (Rust) vs 55.8ms (Python),
  CG 37 iters in 1.83ms (Rust) vs 369.8ms (Python), Dirac 0.023ms/apply (Rust)
  vs 4.59ms/apply (Python). Iteration counts match exactly. `bench_lattice_cg`.

**Remaining for Kachkovskiy (GPU promotion)**:
- ~~**GPU SpMV shader**~~ — ✅ **Done**: `WGSL_SPMV_CSR_F64` validated 8/8 (machine-epsilon parity, RTX 4070)
- ~~**GPU Lanczos**~~ — ✅ **Done**: GPU SpMV inner loop + CPU control, 6/6 checks (eigenvalues match to 1e-15)
- **Fully GPU-resident Lanczos** — GPU dot + axpy + scale for N > 100k systems (P2)
- **Generalized matrix exponentiation** — beyond 3×3 anti-Hermitian (P3)

### Tier 3b — Kokkos/LAMMPS Validation Baseline (Murillo → Chuna, March 2026)

**Priority**: P0 for hotSpring — entry point for Chuna review
**Source**: Murillo pointed us at Kokkos (Sandia). PhD student Thomas Chuna
(co-authored Bazavov SU(3), Murillo DSF) is the reviewer.

| # | Target | What | BarraCuda Equivalent | Status |
|---|--------|------|---------------------|--------|
| 43 | LAMMPS Yukawa OCP (Kokkos/CUDA) | 9 PP DSF cases via `pair_style yukawa` + `kspace_style pppm` | `sarkas_gpu` (9/9, 0.000% drift) | Queued |
| 44 | LAMMPS PPPM (cuFFT via Kokkos) | Ewald sum, B-spline charge assignment | WGSL 3D FFT + PPPM shader | Queued |
| 45 | Kokkos dispatch overhead | Compile-time CUDA template instantiation | Runtime WGSL JIT via wgpu | Queued |

**Evolution path**: Python (Sarkas) → Kokkos (LAMMPS) → Rust (hotSpring) → GPU (BarraCuda) → sovereign

**Chuna review package**: Sarkas reproduction + lattice QCD β-scan + PPPM +
DF64 precision + performance data. Lead with physics. Let him discover the
infrastructure. See `experiments/040_KOKKOS_LAMMPS_VALIDATION.md`.

**References**: Kokkos (Edwards et al. 2014, Trott et al. 2022),
Chuna & Bazavov (2021, arXiv:2101.05320), Chuna & Murillo (2024, arXiv:2405.07871)

### Tier 4 — Warm Dense Matter & Ignition (NIF/JLF Feb 2026, baseCamp Sub-thesis 07)

Murillo co-authored the "Roadmap for warm dense matter physics" (arXiv:2505.02494,
70+ authors, revised Feb 13 2026). He attended the NIF & JLF User Groups Meeting
(February 10-12, 2026, Livermore, CA) where the latest NIF ignition results were
discussed (6 successful shots, peak gain 2.3x at 5.2 MJ). These targets extend
hotSpring into the computational frontier of warm dense matter — Murillo's core domain.

#### Tier 4a — Immediate: Open Data / Code Reproduction

| # | Paper / Resource | Year | What | BarraCUDA Status |
|---|-----------------|------|------|------------------|
| 32 | Militzer FPEOS Database (Berkeley) | 2020+ | EOS tables for 11 elements, 10 compounds via PIMC + DFT-MD. Open C++/Python. 5,000 first-principles sims, 0.5-50 g/cc, 10⁴-10⁹ K | Reproduce lookup + interpolation in Rust; validate against published tables. GEMM + interpolation — ready |
| 33 | atoMEC average-atom code (SciPy Proceedings) | 2023 | Open Python average-atom model for WDM. Computationally cheap alternative to full MD. Perfect Phase 0 Python control | Ready for reproduction — ideal Python baseline |
| 34 | Vorberger, Graziani, Murillo et al. "Roadmap for warm dense matter physics" | arXiv:2505.02494 | 70+ author field survey. Maps every open computational challenge in WDM | Reference — identify specific reproduction targets from each section |

#### Tier 4b — Medium: GPU Transport Extension

| # | Paper / Resource | Year | What | BarraCUDA Status |
|---|-----------------|------|------|------------------|
| 35 | Transport coefficient comparison (Murillo's roadmap section) | 2025-26 | Viscosity, diffusion, thermal conductivity benchmark across codes. Direct extension of Paper 5 (Stanton-Murillo) | Green-Kubo validated (13/13). Extend to WDM conditions (higher T, partial ionization) |
| 36 | Dragon OF-DFT MD (multi-GPU) | 2023 | Orbital-free DFT for WDM. Institutional groups need 4-8 GPUs. Can BarraCUDA do it on 1 consumer GPU at smaller N? | GEMM + FFT (FFT validated ✅). New primitive: kinetic energy functional |
| 37 | Perturbo v3.0 GPU transport (arXiv:2511.03683) | 2025 | GPU electronic transport from electron-phonon. 40x speedup with OpenACC. Can WGSL match? | Boltzmann transport equation → potential new BarraCUDA primitive |

#### Tier 4c — Longer Horizon: NIF Diagnostics

| # | Paper / Resource | Year | What | BarraCUDA Status |
|---|-----------------|------|------|------------------|
| 38 | Dynamic structure factor S(q,ω) from MD | — | Key NIF diagnostic for XRTS. Bridge between MD simulation and experimental data | FFT validated ✅. Extend Sarkas MD to compute S(q,ω) via velocity autocorrelation → FFT |
| 39 | Colliding Planar Shocks (LLNL new NIF platform) | 2025 | Counter-propagating shocks for uniform WDM. Shock propagation modeling | Velocity Verlet + Hugoniot EOS (partially in place) |
| 40 | Dornheim et al. XRTS diagnostics — model-free temperature extraction | 2025 | Correlation function metrology for WDM (arXiv:2510.00493) | FFT + spectral analysis. Reference for validation against NIF data |
| 41 | Wavepacket MD for partially-ionized plasma (arXiv:2510.27446) | 2025 | Beyond classical point particles — quantum effects via wavepackets | New primitive: wavepacket evolution. Phase G target |

#### Tier 4d — Distributed WDM Compute

| # | Paper / Resource | Year | What | BarraCUDA Status |
|---|-----------------|------|------|------------------|
| 42 | Anderson & Fedak "The computational and storage potential of volunteer computing" | 2006 | Empirical measurement of volunteer compute capacity. Validates distributed approach | Already in Track 5 (#30). Cross-reference for WDM distribution |

**Connection to existing hotSpring work**: Tier 4 extends the Sarkas MD (Paper 1),
transport coefficients (Paper 5), and nuclear EOS (Paper 4) into WDM conditions.
The FFT gap is closed (toadStool `1ffe8b1a`). Transport is validated (13/13).
The progression: classical MD → WDM transport → OF-DFT → XRTS diagnostics →
distributed WDM computation on consumer GPU fleets.

**Connection to baseCamp Sub-thesis 07**: These reproduction targets provide the
validation backbone for the claim that WDM simulation doesn't require institutional
HPC. Each paper reproduced on consumer GPU strengthens the argument.

**Connection to NIF/JLF meeting (Feb 10-12, 2026)**: Murillo attended this meeting.
Papers 38-40 directly relate to NIF experimental diagnostics. Reproducing these
computational methods on consumer hardware provides the field with accessible tools.

### Chuna Extension — Murillo Group Integrators, Dielectric Theory & HED Coupling

Thomas Chuna (PhD student, Murillo Group, MSU Physics & CMSE) — referred by
Murillo (March 4, 2026). Published on lattice QCD integrators with Bazavov,
plasma dielectric functions with Murillo, and kinetic-fluid HED coupling with
Sagert/Haack/Murillo. Profile: `whitePaper/attsi/non-anon/contact/murillo/chuna_profile.md`

**Contact status (March 26, 2026)**: Chuna wants to meet. He's in Germany (UTC+1),
prefers mornings ET. Meeting deferred to late April — build RHMC data first.
Bazavov access is via Chuna (co-author on Paper 43) and Murillo (conduit).
See `whitePaper/attsi/non-anon/contact/chuna/`.

#### All Three Papers — COMPLETE

| # | Paper | Core Checks | GPU | Extension | Status |
|---|-------|:-----------:|:---:|:---------:|--------|
| 43 | Bazavov & Chuna — gradient flow integrators (arXiv:2101.05320) | 11/11 | ✅ 38.5× | Dynamical Nf=4 3/3 | **✅ COMPLETE** — 5 integrators, coefficients derived independently at compile time |
| 44 | Chuna & Murillo — conservative BGK dielectric (PRE 111, 035206) | 20/20 | ✅ GPU Mermin | DSF vs MD 14/14 | **✅ COMPLETE** — std + completed + multi-component Mermin, 322× faster |
| 45 | Haack et al. — kinetic-fluid coupling (JCP 2024) | 10/10 | ✅ GPU BGK | Coupled pipeline | **✅ COMPLETE** — Python 18/18, CPU 16+20/20, GPU BGK + Sod |

Total: **44/44 core checks + 3/3 dynamical extension** via `validate_chuna_overnight`.

#### Active — GPU RHMC (Nf=2, 2+1): Continuing Chuna's Physics at Consumer Scale

The next step beyond Paper 43 is running gradient flow on **Nf=2+1 RHMC configs** —
the same fermion content Chuna validates against in MILC. This requires the rooting
trick: `det(D†D)^{Nf/8}` via rational approximation.

**GPU RHMC infrastructure (Exp 099, March 26 2026): COMPLETE**

| Component | File | Status |
|-----------|------|--------|
| `RationalApproximation` (Remez + partial fractions) | `rhmc.rs` | ✅ CPU reference |
| `RhmcConfig::nf2()`, `RhmcConfig::nf2p1()` | `rhmc.rs` | ✅ Pre-configured |
| `GpuRhmcSectorBuffers` (per-flavor-sector GPU buffers) | `gpu_rhmc.rs` | ✅ |
| `gpu_multi_shift_cg_solve` (independent per-shift CG) | `gpu_rhmc.rs` | ✅ |
| `gpu_rhmc_heatbath_sector` (φ = r_hb(D†D) η) | `gpu_rhmc.rs` | ✅ |
| `gpu_rhmc_fermion_action_sector` (S_f = φ† r(D†D) φ) | `gpu_rhmc.rs` | ✅ |
| `gpu_rhmc_total_force_dispatch` (gauge + Σ fermion) | `gpu_rhmc.rs` | ✅ |
| `gpu_rhmc_trajectory` (full Omelyan MD) | `gpu_rhmc.rs` | ✅ |
| `production_rhmc_scan` (CLI production binary) | `bin/production_rhmc_scan.rs` | ✅ |
| `multi_shift_zeta_f64.wgsl` (zeta recurrence shader) | `shaders/` | ✅ |

**RHMC production (Exp 101, March 26): COMPLETE**
- Nf=2 validated at 4^4 (78% acc) and 8^4 (50% acc)
- Nf=2+1 validated at 4^4 (68-78% acc, correct ordering Q < 2+1 < 2)
- All 640 trajectories, ~$0.02 compute cost

**Gradient flow at volume (Exp 102, March 26-27): IN PROGRESS**
- CK4 stability confirmed: error 2.3e-6 at ε=0.1 (W6/W7 diverge at 1.6)
- Convergence orders: Euler 1.23✓, RK2 1.97✓, W6/W7/CK4 ~2.0-2.3 (8^4 finite-size suppressed)
- 16^4 quenched flow running (t₀/w₀ scale setting)

| Target | Lattice | Goal | Status |
|--------|---------|------|--------|
| Nf=2 validation (4^4) | 4^4 | Plaquette, acceptance, ΔH scaling | **✅ DONE** — Exp 101, ⟨P⟩=0.534/0.601, 74-78% acc |
| Nf=2 production (8^4) | 8^4 | β-scan across transition | **✅ DONE** — Exp 101, ⟨P⟩=0.498/0.542/0.600, 50% acc |
| Nf=2+1 validation (4^4) | 4^4 | 2-sector: light + strange | **✅ DONE** — Exp 101, ⟨P⟩=0.531/0.561/0.593, 68-78% acc |
| Flow convergence (8^4) | 8^4 | CK4 stability + order measurement | **✅ DONE** — Exp 102, CK4 error 2.3e-6 at ε=0.1 |
| Flow t₀/w₀ (16^4) | 16^4 | Clean scale setting (W7 integrator) | 🔄 **Running** — Exp 102 |
| Nf=2+1 production (8^4→16^4) | 8-16^4 | Scale 2-sector to volume | Next |
| Flow on Nf=2+1 configs | 16^4 | Chuna W7 integrator on RHMC configs | Exp 103 |
| Nf=2+1 production (32^4) | 32^4 | Weekend run — MILC-comparable physics | Near-term |
| Multi-GPU scaling | 32-48^4 | toadStool brain + HBM2 fleet | Medium-term |
| MILC-comparable | 64³×128 | Requires collaborator HPC allocation | Long-term |

**Why this matters for Chuna**: His Paper 43 integrators were validated on **MILC
Nf=2+1 configs**. Showing flow on Nf=2+1 RHMC configs generated on consumer GPU
means his integrators run on his physics, at home, for cents.

**Scaling to meet/exceed Chuna's MILC-scale work**: Detailed volume roadmap in
`whitePaper/baseCamp/silicon_characterization_at_scale.md`. The strategy is
threefold: (1) prove same physics at 100-1000x less hardware cost, (2) quantify
silicon waste in HPC codes via characterization pipeline, (3) provide portable
sovereign binary that runs identically on consumer and HPC hardware.

---

### R. Anderson Extension — Hot Spring Microbial Evolution (Taq Corollary)

Rika Anderson (Carleton College) studies microbial evolution in extreme environments,
including the **same Yellowstone hot springs** where *Thermus aquaticus* was discovered.
Her *Sulfolobus* paper is the direct empirical corollary to the Taq polymerase argument
in `gen3/CONSTRAINED_EVOLUTION_FORMAL.md` §1.1 — population genomics of extremophiles
under thermal constraint.

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 23 | Campbell, Anderson et al. "*Sulfolobus islandicus* meta-populations in Yellowstone National Park hot springs" | Env Microbiol 19:2392-2405 | 2017 | R. Anderson | Population genomics analysis, mobile genetic element susceptibility, meta-population structure. Public genomes — no wet lab required | Queued |
| 24 | Anderson (2021) "Tracking Microbial Evolution in the Subseafloor Biosphere" | mSystems 6:e00731-21 | 2021 | R. Anderson | Review/commentary — no reproduction needed, but the theoretical framework (stochastic vs deterministic evolution, Muller's ratchet, Lenski citation) directly informs CONSTRAINED_EVOLUTION_FORMAL.md | Reference |

**Paper 23 — Why it matters for hotSpring**: *Sulfolobus islandicus* lives at 65-85°C
in acidic (pH 2-4) hot springs — the same thermal environment that produced Taq polymerase
in *Thermus aquaticus*. Campbell & Anderson show that geographic isolation between hot
springs drives structured genomic variation, with different populations showing different
susceptibilities to viruses and mobile genetic elements. This is **constrained evolution
in action**: the hot spring environment constrains what survives, geographic isolation
creates independent evolutionary trajectories (like Lenski's 12 populations), and mobile
elements provide the raw material for innovation (like citrate metabolism in Ara-3).
The paper uses public *Sulfolobus* genomes — reproduction requires bioinformatics only,
no wet lab. wetSpring's sovereign 16S/metagenomics pipeline can handle the analysis.

**BarraCuda relevance**: Population genomics computation (pairwise SNP comparison → GEMM),
mobile element detection (sequence alignment → Smith-Waterman or BLAST-like), phylogenetic
tree construction (parallel likelihood evaluation), diversity metrics (Shannon, Simpson
→ FusedMapReduceF64). All primitives already validated in wetSpring.

---

### ToadStool Evolution Catch-Up (Feb 20, 2026 — Sessions 18-25)

ToadStool resolved ALL P0/P1/P2 blockers from the rewire document in 7 commits:

| Blocker | Commit | Status |
|---------|--------|--------|
| **CellListGpu BGL mismatch** (P0) | `8fb5d5a0` | **FIXED** — 4-binding layout, scan_pass_a/b split |
| **Complex f64 WGSL** (P1) | `8fb5d5a0` | **Absorbed** — `shaders/math/complex_f64.wgsl` (~70 lines) |
| **SU(3) WGSL** (P1) | `8fb5d5a0` | **Absorbed** — `shaders/math/su3.wgsl` (~110 lines, `@unroll_hint 3`) |
| **Wilson plaquette GPU** (P2) | `8fb5d5a0` | **Absorbed** — `shaders/lattice/wilson_plaquette_f64.wgsl` |
| **SU(3) HMC force GPU** (P2) | `8fb5d5a0` | **Absorbed** — `shaders/lattice/su3_hmc_force_f64.wgsl` |
| **U(1) Abelian Higgs GPU** (P2) | `8fb5d5a0` | **Absorbed** — `shaders/lattice/higgs_u1_hmc_f64.wgsl` |
| **GPU FFT f64** (P0 blocker) | `1ffe8b1a` | **DONE** — 3 bugs fixed, 14 GPU tests pass, 3D FFT available |

**Impact**: Papers 9-12 (full lattice QCD) are now unblocked. GPU FFT was THE
major prerequisite — toadstool has `Fft1DF64`, `Fft3DF64`, roundtrip-validated
to 1e-10 precision on RTX 3090. The full QCD Dirac solver can use GPU FFT for
momentum-space propagator computation. **Full pipeline COMPLETE**: Dirac 8/8 + CG 9/9. All GPU primitives validated.

### Tier 3 — Unblocked: Full Lattice QCD (FFT + GPU Lattice Now Available)

| # | Paper | Journal | Year | Faculty | What We Need | Status |
|---|-------|---------|------|---------|-------------|--------|
| 9 | Bazavov [HotQCD] (2014) "The QCD equation of state" | Nucl Phys A 931, 867 | 2014 | Bazavov | ~~FFT~~, ~~complex f64~~, ~~SU(3)~~, ~~HMC~~, ~~GPU Dirac~~, ~~GPU CG~~ | **Quenched β-scan 10/10** — 4^4+8^4 validated, production runs next |
| 10 | Bazavov et al. (2016) "Polyakov loop in 2+1 flavor QCD" | Phys Rev D 93, 114502 | 2016 | Bazavov | Same as #9 + Polyakov loop | **GPU RHMC** — Nf=2 (8^4) + Nf=2+1 (4^4) validated, Exp 101 |
| 11 | Bazavov et al. (2025) "Hadronic vacuum polarization for the muon g-2" | Phys Rev D 111, 094508 | 2025 | Bazavov | Same as #9 + subpercent precision | **CPU COMPLETE** — `validate_hvp_g2` (10/10), GPU pipeline ready |
| 12 | Bazavov et al. (2016) "Curvature of the freeze-out line" | Phys Rev D 93, 014512 | 2016 | Bazavov | Same as #9 + inverse problem | **CPU COMPLETE** — `validate_freeze_out` (8/8), β_c within 10% of known |

**Paper 9 status**: Quenched (pure gauge) β-scan validated on 4^4 and 8^4
lattices. `validate_production_qcd` binary: 10/10 checks pass.
- **4^4 scan**: 9 β values (4.0–6.5), plaquette monotonic, ⟨P⟩=0.588 at β=6.0
  (1.1% from Bali reference 0.594), susceptibility peaks at β=5.70 (=β_c).
- **8^4 scaling**: plaquette matches 4^4 within 2.8% at β=6.0.
- **Polyakov transition**: confined ⟨|L|⟩=0.285 < deconfined ⟨|L|⟩=0.334.
- **Determinism**: rerun-identical to machine epsilon.
- **Python control**: `control/lattice_qcd/scripts/quenched_beta_scan.py`
  (algorithm-identical HMC, same LCG seed).
- **Python control parity**: ✅ Complete. 9 β values on 4^4, all plaquettes
  agree within 2.3% (statistical noise on 30 trajectories). Python 446s vs
  Rust 8.0s — **Rust 56× faster**. Bug found and fixed in Python: uniform
  momentum distribution violated HMC detailed balance; corrected to Gaussian
  (Gell-Mann basis, matching Rust).
- **Dynamical fermions**: pseudofermion HMC module (`pseudofermion.rs`) complete.
  Heat bath, CG-based action, fermion force, combined leapfrog. 4/4 tests pass.
- **Dynamical QCD validation**: `validate_dynamical_qcd` (7/7 checks):
  ΔH O(dt²) scaling ✓, plaquette (0,1) ✓, S_F > 0 ✓, acceptance > 1% ✓,
  dynamical vs quenched shift bounded ✓, mass dependence ✓, phase ordering ✓.
  Heavy quarks (m=2.0) on 4^4 with quenched pre-thermalization.
  Python control: `control/lattice_qcd/scripts/dynamical_fermion_control.py`
  (algorithm-identical, S_F≈1500, ΔH≈1-18 — matches Rust exactly).
- **Omelyan integrator** (Feb 22, 2026): ✅ **DONE**. 2MN minimum-norm integrator
  (λ=0.1932) achieves O(dt⁴) shadow Hamiltonian errors at same cost as leapfrog.
  On 4^4 at β=6.0: mean |ΔH| drops from 0.68 (leapfrog) to 0.033 (Omelyan),
  acceptance 96.7% → 100%. On 8^4: acceptance > 60% at dt=0.04. This unblocks
  production-scale runs at 16^4+ where naive leapfrog gives <5% acceptance.
- **Hasenbusch mass preconditioning** (Feb 22, 2026): ✅ **DONE**. Two-level
  determinant split: heavy sector det(D†D(m_heavy)) + ratio sector
  det(D†D(m_light)/D†D(m_heavy)). Multiple time-scale leapfrog: gauge+heavy
  on outer steps, ratio on inner steps. Heavy sector CG converges faster
  (fewer iterations), reducing total CG cost. `validate_production_qcd_v2`
  validates both Omelyan and Hasenbusch with 10/10 checks.
- **Next**: (1) GPU promotion of HMC kernels,
  (2) 16^4/32^4 production runs with Omelyan + Hasenbusch.
See `experiments/009_PRODUCTION_LATTICE_QCD.md` for full results.

**ToadStool GPU primitives now available**: `Fft1DF64`, `Fft3DF64`,
`complex_f64.wgsl`, `su3.wgsl`, `wilson_plaquette_f64.wgsl`,
`su3_hmc_force_f64.wgsl`, `higgs_u1_hmc_f64.wgsl`, `CellListGpu` (fixed).
**Full GPU lattice QCD pipeline COMPLETE**: Dirac (`WGSL_DIRAC_STAGGERED_F64`, 8/8)
+ CG solver (`WGSL_COMPLEX_DOT_RE_F64` + `WGSL_AXPY_F64` + `WGSL_XPAY_F64`, 9/9).
All GPU primitives for 2+1 flavor QCD are validated on RTX 4070.

**Pure GPU Workload Validation** (`validate_pure_gpu_qcd`, 3/3):
CPU HMC thermalization (10 trajectories, 100% accepted, plaq=0.5323) → GPU CG
on thermalized configs (5 solves, all 32 iters matching CPU exactly, solution
parity 4.10e-16). CG iterations run entirely on GPU — only 24 bytes/iter
(α, β, ||r||²) transfer to CPU. Lattice upload: 160 KB once.
**Production-like workload: VALIDATED on thermalized gauge configurations.**

**GPU Scaling** (`bench_lattice_scaling`, RTX 4070):

| Lattice | Volume | GPU (ms) | CPU (ms) | Speedup |
|---------|-------:|---------:|---------:|--------:|
| 4⁴ | 256 | 9.3 | 1.7 | 0.2× |
| 8⁴ | 4,096 | 8.6 | 27.9 | **3.2×** |
| 8³×16 | 8,192 | 10.7 | 55.9 | **5.2×** |
| 16⁴ | 65,536 | 24.0 | 532.6 | **22.2×** |

Iterations identical (33) at every size. GPU crossover at V~2000.

**GPU HMC Scaling** (`bench_gpu_hmc`, RTX 4070, Omelyan n_md=10):

| Lattice | Volume | CPU ms/traj | GPU ms/traj | Speedup |
|---------|-------:|------------:|------------:|--------:|
| 4⁴ | 256 | 55.9 | 11.2 | **5.0×** |
| 8⁴ | 4,096 | 857.8 | 36.0 | **23.8×** |
| 8³×16 | 8,192 | 1,710.1 | 62.3 | **27.5×** |
| 16⁴ | 65,536 | 13,829.9 | 398.3 | **34.7×** |

Full HMC trajectories (gauge force + Cayley exp + momenta + Metropolis) run entirely
on GPU. At 16⁴: 13.8s→0.4s per trajectory. Transfer: only ΔH (8B) + plaquette (8B)
per trajectory. At 32⁴+ (production), GPU advantage exceeds 100×.

**Streaming GPU HMC + GPU PRNG** (`validate_gpu_streaming`, **9/9**, Feb 23 2026):
All dispatch overhead eliminated via single-encoder streaming. GPU-side PRNG generates
SU(3) algebra momenta directly on GPU (PCG hash + Box-Muller, zero CPU→GPU transfer).

| Lattice | Volume | CPU ms/traj | Dispatch | Streaming | Stream gain | vs CPU |
|---------|-------:|------------:|---------:|----------:|:-----------:|-------:|
| 4⁴ | 256 | 63.1 | 33.9 | 26.0 | 1.30× | **2.4×** |
| 8⁴ | 4,096 | 1,579.4 | 48.8 | 54.4 | 0.90× | **29.0×** |
| 8³×16 | 8,192 | 3,414.2 | 80.1 | 83.9 | 0.96× | **40.7×** |
| 16⁴ | 65,536 | 17,578.2 | 493.2 | 441.7 | 1.12× | **39.8×** |

Key results:
- **Bit-identical parity**: streaming vs dispatch ΔH error = 0.00, plaquette error = 0.00
- **GPU PRNG validated**: KE ratio 0.997 (expected: 4·V for SU(3) with 8 generators)
- **Full GPU-resident HMC**: plaquette 0.4988, 90% acceptance, zero CPU→GPU transfer
- **Small system rescue**: streaming 1.30× faster than dispatch at 4⁴ (dispatch overhead eliminated)
- **Large system efficiency**: streaming 1.12× at 16⁴ (encoder batching amortizes submission)
- **GPU dominates CPU at ALL sizes**: 2.4×(4⁴) to 40.7×(8³×16)
- **Transfer budget**: 0 bytes CPU→GPU (GPU PRNG), 16 bytes GPU→CPU per trajectory (ΔH + plaq)

**GPU β-Scan** (`validate_gpu_beta_scan`, 6/6): Full quenched QCD temperature
sweep on GPU — 9 β values on 8⁴ + 3-point 8³×16 cross-check. Plaquette monotonic,
β=6.0 plaq=0.561 (physical), 98.9% acceptance. Total GPU time: 82s. Transfer budget:
5.2 KB GPU→CPU for entire scan (147,456:1 upload:download ratio).

**Heterogeneous pipeline bypass** (still valid): Papers 9-10 phase structure is
also observable from position-space quantities (Polyakov loop, plaquette) without
FFT. GPU HMC → NPU phase classification → CPU validation against β_c ≈ 5.69.

---

## ~~Paper 5 — Detailed Reproduction Plan~~ ✅ COMPLETED

### Stanton & Murillo (2016): Ionic Transport in HED Matter

**Completed February 19, 2026.** All components implemented and validated:

| Component | Status | Location |
|-----------|--------|----------|
| Green-Kubo integrator (VACF → D*) | ✅ Done | `md/observables/` |
| Stress tensor observable (σ_αβ) | ✅ Done | `md/observables/` |
| Heat current observable (J_Q) | ✅ Done | `md/observables/` |
| Daligault (2012) D* analytical fit | ✅ Done, Sarkas-calibrated | `md/transport.rs` |
| Stanton-Murillo (2016) η*, λ* fits | ✅ Done | `md/transport.rs` |
| Validation binary | ✅ 13/13 pass | `bin/validate_stanton_murillo.rs` |

**Key findings**:
1. Daligault Table I coefficients were ~70× too small due to reduced-unit
   convention mismatch. Recalibrated against 12 Sarkas Green-Kubo D* values at N=2000.
2. **(v0.5.14)** Constant weak-coupling prefactor `C_w=5.3` had 44–63% error in the
   crossover regime (Γ ≈ Γ_x). Evolved to κ-dependent `C_w(κ) = exp(1.435 + 0.715κ + 0.401κ²)`.
   Errors now <10% across all 12 Sarkas calibration points. Transport grid expanded to 20
   (κ,Γ) configurations including 9 Sarkas-matched DSF points.

---

## Cost Actuals + Projection

| Paper | GPU Time | Cost | Status |
|-------|:---:|:---:|:---:|
| #5 Stanton-Murillo transport | ~2 hours | **$0.02** | ✅ Done |
| #6 Murillo-Weisheit screening | ~1 min | **$0.001** | ✅ Done |
| #7 HotQCD EOS tables | ~5 min | **$0.001** | ✅ Done |
| #8 Pure gauge SU(3) | ~2 hours | **$0.02** | ✅ Done |
| #13 Abelian Higgs (1+1)D | ~1 min | **$0.001** | ✅ Done |
| Total (Tier 0-2 + 13) | ~5 hours | **~$0.05** | **9/9 complete** |

**Cumulative science portfolio**: 22 papers reproduced, ~$0.20 total compute cost.
All Tier 0-2 targets complete + Paper 13 (Abelian Higgs) + Papers 14-22 (Kachkovskiy
spectral theory: 1D/2D/3D Anderson + almost-Mathieu + Lanczos + Hofstadter butterfly).
Python control baselines established: spectral theory Rust 8× faster, lattice QCD CG
**Rust 200× faster than Python**. Full GPU lattice QCD pipeline validated on
thermalized gauge configurations. Pure GPU workload: machine-epsilon parity (4.10e-16).
Next: production runs at scale + metalForge cross-system (GPU→NPU→CPU).

---

## Scaling Vision: From One GPU to Every Substrate

Our WGSL shaders run on any Vulkan-capable GPU — NVIDIA, AMD, Intel,
integrated, Steam Deck. The NPU offloads inference at <1W. The pipeline
is hardware-agnostic by design. Cost equations change with scale:

| Scale | Hardware | Feasibility | Cost/paper |
|-------|----------|-------------|------------|
| **Now** | 1× RTX 4070 ($600) + AKD1000 ($300) | Tier 0-1 + NPU inference | ~$0.01-0.10 |
| **Lab** | 4× mixed GPU + NPU | Tier 2 (SU(3)) + real-time transport | ~$0.02-0.20 |
| **Distributed** | 100s of idle GPUs (volunteer compute) | Tier 3 partial (HMC sweeps) | ~$0.001/GPU-hr |
| **Fleet** | Every idle GPU + NPU co-processor | Full lattice QCD + continuous monitoring | Approaches zero |

The architecture allows this evolution because:
- **WGSL/Vulkan**: runs on every GPU vendor, no CUDA lock-in
- **NPU offload**: inference at <1W, zero GPU overhead, 9,017× less energy than CPU
- **Independent dispatches**: Jacobi sweeps, HMC trajectories, parameter
  scans are embarrassingly parallel across GPUs
- **Streaming dispatch**: single-encoder batching eliminates per-operation
  submission overhead; GPU PRNG eliminates CPU→GPU data transfer entirely
- **GPU→NPU streaming**: GPU produces data, NPU consumes for inference,
  CPU orchestrates — three substrates, one physics pipeline
- **Sovereign stack**: AGPL-3.0, no proprietary dependencies, no vendor
  gatekeeping — any GPU that exposes Vulkan can contribute compute
- **Lattice decomposition**: each sublattice is an independent dispatch;
  communication is boundary exchange (small relative to compute)

Full lattice QCD remains the long-term north star. The path:
1. **Today**: transport coefficients, EOS tables — zero new GPU code ✅
2. **Near**: pure gauge SU(3) — complex f64 + plaquette force ✅
3. **Heterogeneous**: GPU HMC + NPU phase classification — lattice phase
   structure without FFT, using Polyakov loop + plaquette observables
4. **Medium**: ~~FFT~~ ✅ + ~~Dirac GPU~~ ✅ + ~~CG solver~~ ✅ — **FULL LATTICE QCD GPU PIPELINE COMPLETE**
5. **Pure GPU HMC**: ALL QCD math on GPU — gauge force, Cayley link update,
   momentum update, plaquette, kinetic energy as fp64 WGSL shaders. Full Omelyan
   trajectory: 100% acceptance, plaq=0.584, CPU parity 4.4e-16 ✅
6. **Streaming GPU HMC**: single-encoder dispatch eliminates per-operation submission
   overhead. GPU PRNG (PCG + Box-Muller) generates SU(3) momenta on-device — zero
   CPU→GPU transfer. 9/9 validation: bit-identical parity, 2.4×–40× vs CPU at all
   scales (4⁴ through 16⁴). Small systems now GPU-viable via dispatch elimination ✅
7. **Benchmark**: Rust 200× faster than Python (same algorithm, same seeds) ✅
8. **Scale**: distribute across any available GPU fleet

Each step builds on the last. We don't need the full stack to start —
we need the architecture to allow evolution. That architecture exists.

9. **NPU inference**: deploy trained models to NPU for continuous monitoring
   at <1W while GPU does the heavy lifting — GPU→NPU→CPU streaming
10. **Real-time heterogeneous**: live phase monitoring during HMC (0.09%
   overhead), predictive steering (62% compute savings), and cross-substrate
   parity (f64→f32→int4) — five previously-impossible capabilities validated
11. **Three-substrate streaming** (16/16 with `--features npu-hw`): CPU baseline → GPU streaming at scale →
   NPU screening in-flight → CPU final verification. Full pipeline validated
   end-to-end with real GPU HMC observables at 4⁴ and 8⁴ ✅
12. **Real NPU from Rust**: AKD1000 discovered via `akida-driver` (pure Rust,
   80 NPUs, 10 MB SRAM, PCIe Gen2 x1). Hardware probe + capability query +
   predict interface — all from Rust, no Python dependency ✅
13. **GPU dynamical fermion HMC** (6/6): Full QCD (gauge + staggered fermions) on
   GPU via 8 WGSL shaders. Force parity 8.33e-17, CG 3.23e-12, 90% accept ✅
14. **Streaming dynamical fermion HMC**: single-encoder dispatch + GPU PRNG for
   full dynamical QCD. Enables production runs at 32⁴+ (PENDING)
15. **RTX 4070 full capacity**: dynamical QCD at 40⁴ (2.56M sites, 8.2 GB VRAM),
   full EOS β-scan in ~39 days for ~$25 of electricity. Same lattice volume
   class as HotQCD 2014 that cost 100M institutional CPU-hours (PENDING)

### Long-Term Goal: Full Parity Slice of Reality

The objective is NOT to match institutional HPC throughput. It is to prove
that the full mathematical workflow — dynamical 2+1 flavor QCD with physical
observables — runs correctly on consumer hardware, at any scale the VRAM holds.

**RTX 4070** (12 GB, $600): max dynamical lattice 40⁴ (2.56M sites). Full EOS
in 39 days. 570× more physics per joule than Frontier.

**RTX 5090** (32 GB, $2000, in NUCLEUS mesh): max dynamical lattice 64⁴ (16.8M
sites). Near-frontier volume on a single consumer GPU.

**RTX 6000 Blackwell** (96 GB, acquirable): 128³×32 (67M sites) single-card.
Production lattice QCD on one desktop workstation.

The shaders are the mathematics. Compute time is a distribution problem.
One millionth of Frontier's compute at one hundred-millionth of its energy
is a 100× net efficiency win. Every idle GPU running these shaders contributes
to the same validated physics pipeline. The architecture scales; the math
is proven first, locally, on one GPU.

---

## Track 5 — Distributed Computing History ("Local Ruins" for NUCLEUS)

**Purpose**: Review foundational distributed computing papers to understand scheduling,
fault tolerance, and heterogeneous hardware management before deploying the basement
HPC as a covalent NUCLEUS mesh. These are the "local ruins" — systems that solved
distribution problems decades ago on volunteer hardware. Their lessons directly inform
NUCLEUS deployment.

**Connection to ecoPrimals**: ToadStool already has `HybridCloudScheduler` and
distributed GPU scheduling in `crates/distributed/`. BOINC papers inform how to evolve
this for the NUCLEUS mesh. Key difference: BOINC is server-client with anonymous
volunteers; NUCLEUS is peer-to-peer with covalent trust (family seed). The scheduling
algorithms are relevant; the trust model is not.

### Folding@home

| # | Paper | Journal | Year | Why | Status |
|---|-------|---------|------|-----|--------|
| 25 | Shirts & Pande "Screen Savers of the World, Unite!" | Science 290:1903-4 | 2000 | Founding vision — volunteer MD on idle CPUs. The original "latent gaming power" argument | Queue |
| 26 | Pande "Folding@home architecture" | Stanford CS | 2009 | 5 PetaFLOPS on volunteer hardware. Scheduling, fault tolerance, result validation at scale | Queue |
| 27 | Zimmerman et al. "Folding@home: Achievements from over twenty years of citizen science herald the exascale era" | Biophysical Journal 122(14):2852-2863 | 2023 | 20-year retrospective. Lessons learned, failure modes, what worked vs what didn't | Queue |

### SETI@home / BOINC

| # | Paper | Journal | Year | Why | Status |
|---|-------|---------|------|-----|--------|
| 28 | Anderson et al. "SETI@home: An Experiment in Public-Resource Computing" | CACM 45(11):56-61 | 2002 | First internet-scale volunteer computing. Task distribution, credit system, cheating resistance | Queue |
| 29 | Anderson "BOINC: A System for Public-Resource Computing and Storage" | 5th IEEE/ACM Grid Computing | 2004 | The framework that generalized SETI@home. Work unit model, redundant computation, scheduling hierarchy | Queue — **Priority: reproduce scheduling algorithm** |
| 30 | Anderson & Fedak "The computational and storage potential of volunteer computing" | IEEE/ACM CCGrid | 2006 | Empirical measurement of volunteer compute capacity. Validates the 200:1 citizen-to-cloud ratio from Latent Value Economy paper | Queue |
| 31 | Kondo et al. "Scheduling task parallel applications for rapid turnaround on desktop grids" | JPDC 67(11):1209-1227 | 2007 | BOINC scheduling algorithms for heterogeneous hardware. Directly comparable to ToadStool HybridCloudScheduler | Queue — **Priority: compare to HybridCloudScheduler** |

### What to Extract

| BOINC Concept | NUCLEUS Analog | Key Difference |
|---------------|----------------|----------------|
| Work units | NUCLEUS atomics (Tower/Node/Nest) | Fixed granularity vs adaptive composition |
| Redundant computation | Cryptographic verification (BearDog) | Quorum voting vs lineage trust |
| Anonymous volunteers | Covalent family (SoloKey FIDO2 seed) | Zero-trust vs family-trust |
| Server-client scheduling | biomeOS plasmodium (peer-to-peer) | Central coordinator vs distributed consensus |
| Heterogeneous CPU mix | DDR3→DDR5, RTX 2070→5090 mesh | CPUs only vs GPU/NPU/CPU mixed substrate |
| Credit system | sunCloud radiating attribution | Points vs cryptographic provenance |
| Result validation (quorum) | BearDog lineage hash | Statistical consensus vs deterministic verification |

---

## Notes

- **Bazavov full lattice**: ~~FFT~~ ✅ + ~~Dirac~~ ✅ + ~~CG~~ ✅ + ~~HMC~~ ✅ + ~~Streaming~~ ✅ — **ALL GPU PRIMITIVES COMPLETE**.
  FFT (toadstool `1ffe8b1a`), GPU Dirac (8/8), GPU CG (9/9), **GPU HMC (8/8)**, **Streaming HMC (9/9)** —
  6 fp64 WGSL shaders + GPU PRNG for fully GPU-resident Omelyan HMC with single-encoder dispatch.
  Zero CPU→GPU transfer. Production-ready for 16⁴+ volumes.
- **Paper 5 (Stanton-Murillo)** ✅ complete: Green-Kubo transport validated, 13/13 checks
- Paper 5 → Paper 6 forms a Murillo Group transport chain: MD → transport → screening
- Paper 7 bridges hotSpring ↔ lattice QCD without requiring lattice simulation
- Paper 8 (pure gauge) bridges MD ↔ lattice QCD with minimal new code
- Each paper reproduced at ~$0.01-0.10 proves the cost decrease thesis:
  **consumer GPU + Rust + open-source drivers → democratized computational science**
