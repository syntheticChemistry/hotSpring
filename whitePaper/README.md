# hotSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository  
**Purpose**: Document the replication of Murillo Group computational plasma physics on consumer hardware using BarraCuda  
**Date**: February 26, 2026 (v0.6.14 — cross-substrate ESN, NPU characterization, DF64 production)

---

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [STUDY.md](STUDY.md) | **Main study** — full writeup of the two-phase validation, data sources, results, and path to paper parity | Reviewers, collaborators |
| [BARRACUDA_SCIENCE_VALIDATION.md](BARRACUDA_SCIENCE_VALIDATION.md) | Phase B technical results — BarraCuda vs Python/SciPy numbers | Technical reference |
| [CONTROL_EXPERIMENT_SUMMARY.md](CONTROL_EXPERIMENT_SUMMARY.md) | Phase A summary — Python reproduction of published work | Quick reference |
| [METHODOLOGY.md](METHODOLOGY.md) | Two-phase validation protocol | Methodology review |
| [baseCamp/](baseCamp/) | Per-domain research briefings (Murillo plasma, lattice QCD, Kachkovskiy spectral, cross-spring, neuromorphic silicon) | Faculty, collaborators |

---

## What This Study Is

hotSpring is a computational physics biome within ecoPrimals. It replicates
published work from the Murillo Group (Michigan State University) on consumer
hardware, then re-executes the computations using BarraCuda — a pure Rust
scientific computing library. ToadStool/barracuda is the shared fungus: hotSpring
leans on it for GPU device management and shader dispatch, evolves new shaders and
physics systems locally, and hands them off for upstream absorption. Other springs
(neuralSpring, desertSpring) follow the same pattern independently — they don't
import hotSpring, but can review its code in `ecoPrimals/` and learn from it.

The study answers five questions:
1. **Can published computational science be independently reproduced?** (Answer: yes, but it required fixing 6 silent bugs and rebuilding physics that was behind a gated platform)
2. **Can Rust + WebGPU replace the Python scientific stack for real physics?** (Answer: yes — BarraCuda achieves 478× faster throughput and 44.8× less energy at L1, with GPU FP64 validated to 4.55e-13 MeV precision. Full Sarkas Yukawa MD runs on a $600 consumer GPU: 9/9 PP cases pass at N=10,000 with 80,000 production steps in 3.66 hours for $0.044. GPU-resident CG reduces readback by 15,360× and speeds dynamical fermion HMC by 30.7×. Bidirectional streaming pipeline dispatches 90%+ to GPU with async readback.)
3. **Can consumer GPUs do first-principles nuclear structure at scale?** (Answer: yes — the full AME2020 dataset (2,042 nuclei, 39x the published paper) runs on a single RTX 4070. L1 Pareto analysis, L2 GPU-batched HFB, and L3 deformed HFB all produce results. This is direct physics computation, not surrogate learning.)
4. **Does the Python → Rust → GPU evolution path extend beyond plasma physics?** (Answer: yes — lattice QCD (SU(3) pure gauge, HMC, staggered Dirac, dynamical fermion pseudofermion HMC), Abelian Higgs (U(1) gauge + Higgs field, 143× faster than Python), transport coefficients (Green-Kubo, Stanton-Murillo), screened Coulomb (Sturm eigensolve, 2274× faster than Python), and HotQCD EOS tables are all validated on CPU with WGSL templates ready for GPU promotion. 22 papers reproduced, 400+ validation checks, ~$0.20 total compute cost.)
5. **Can physics math be truly substrate-portable — CPU → GPU → NPU?** (Answer: yes — ESN reservoir math validated across f64 CPU, f32 NpuSimulator, int4 quantized, and real AKD1000 NPU hardware. 10 SDK assumptions overturned by probing beyond the SDK. The same WGSL shader math trains on GPU and deploys on NPU for inference at 30mW. See `metalForge/npu/akida/BEYOND_SDK.md`.)

---

## Key Results

### Phase A (Python Control): 86/86 checks pass

- Sarkas MD: 12 cases, 60 observable checks, 8.3% mean DSF peak error
- TTM: 6/6 equilibration checks pass
- Surrogate learning: 15/15 benchmark functions converge
- Nuclear EOS: Python L1 (chi2=6.62), L2 (chi2=1.93 via SparsitySampler)
- 5 silent upstream bugs found and fixed

### Phase B (BarraCuda): GPU-validated, energy-profiled

| Level | BarraCuda | Python/SciPy | Speedup | Energy Ratio |
|-------|-----------|-------------|---------|:------------:|
| L1 (SEMF baseline) | 4.99 chi2/datum | 4.99 | 28.8× (GPU) | **44.8× less** |
| L1 (DirectSampler) | **2.27** chi2/datum | 6.62 | **478×** | — |
| L2 (HFB) | 23.09 chi2/datum | **1.93** | 1.7× | — |

### Phase C (GPU MD): Sarkas on consumer GPU (N=2,000)

- **9/9 PP Yukawa cases pass** on RTX 4070 using f64 WGSL shaders
- Energy drift: **0.000%** across 80,000 production steps
- Sustained throughput: **149-259 steps/s** at N=2,000
- Full 9-case long sweep: **71 minutes**, ~225 kJ total GPU energy

### Phase D (Native f64 + N-scaling)

- Native WGSL builtins: 2-6× throughput improvement
- N=10,000 paper parity in **5.3 minutes**; N=20,000 in 10.4 minutes
- Cell-list O(N) scaling + WGSL `i32 %` bug deep-debugged

### Phase E (Paper-Parity Long Run + Toadstool Rewire)

- **9/9 PP Yukawa cases at N=10,000, 80k production steps** — exact paper config
- **3.66 hours total, $0.044 electricity**
- Cell-list **4.1× faster** than all-pairs for κ=2,3
- Energy drift: **0.000-0.002%** across all 9 cases
- Toadstool GPU ops wired: **BatchedEighGpu**, **SsfGpu**, **PppmGpu**

### Phase F (Full-Scale Nuclear EOS on Consumer GPU) — NEW

- **Full AME2020 dataset: 2,042 nuclei** (39x published paper's 52)
- L1 Pareto frontier: chi2_BE from **0.69** (pure BE) to **7.37** (NMP-balanced)
- L2 GPU-batched HFB: **791 HFB nuclei in 66 min**, 99.85% convergence, 206 GPU dispatches
- L3 deformed HFB: **295/2036 nuclei improved**, best-of-both chi2 = 13.92
- **Direct first-principles nuclear structure** — not surrogate learning
- Multi-GPU scaling path: each additional RTX 4070 ($600) doubles parameter throughput
- **197/197 quantitative checks pass** across all phases + pipeline validation

---

## Bazavov Extension: Lattice QCD on Consumer Hardware

hotSpring has extended from plasma physics to lattice gauge theory. The
Bazavov connection (CMSE & Physics, MSU) provides the bridge: both Murillo
and Bazavov study strongly coupled many-body systems with overlapping
computational methods (MD ↔ HMC, plasma EOS ↔ QCD EOS).

### Completed (February 22, 2026)

| Paper | Status | Implementation |
|-------|--------|----------------|
| Stanton & Murillo (2016) transport | **Done** | Green-Kubo D*/η*/λ*; Sarkas-calibrated fits + C_w(κ) evolution, 13/13 checks |
| Murillo & Weisheit (1998) screening | **Done** | Screened Coulomb eigenvalues; Sturm bisection, 23/23 checks, Rust 2274× Python |
| HotQCD EOS tables (Bazavov 2014) | **Done** | `lattice/eos_tables.rs` — thermodynamic validation passes |
| Pure gauge SU(3) Wilson action | **Done** | `lattice/` — 8 modules, 12/12 validation checks |
| Dynamical fermion QCD (Paper 10) | **Done** | `lattice/pseudofermion.rs` — pseudofermion HMC, 7/7 checks, Python control parity |
| Abelian Higgs (Bazavov 2015) | **Done** | `lattice/abelian_higgs.rs` — U(1)+Higgs HMC, 17/17 checks, Rust 143× Python |

### Lattice QCD Infrastructure Built

| Module | Lines | Purpose | GPU Status |
|--------|-------|---------|------------|
| `complex_f64.rs` | 316 | Complex f64 arithmetic | ✅ **Absorbed** — `complex_f64.wgsl` (toadstool `8fb5d5a0`) |
| `su3.rs` | 460 | SU(3) matrix algebra | ✅ **Absorbed** — `su3.wgsl` (toadstool `8fb5d5a0`) |
| `wilson.rs` | 338 | Wilson gauge action, plaquettes | ✅ **Absorbed** — `wilson_plaquette_f64.wgsl` |
| `hmc.rs` | 350 | HMC with Cayley exponential | ✅ **Absorbed** — `su3_hmc_force_f64.wgsl` |
| `abelian_higgs.rs` | ~500 | U(1)+Higgs (1+1)D HMC | ✅ **Absorbed** — `higgs_u1_hmc_f64.wgsl` |
| `dirac.rs` | 297 | Staggered Dirac operator | ✅ **Validated** — `WGSL_DIRAC_STAGGERED_F64` (8/8 checks, 4.44e-16) |
| `cg.rs` | 214 | Conjugate gradient for D†D | ✅ **Validated** — 3 WGSL shaders (9/9 checks, iterations match exactly) |
| `pseudofermion.rs` | ~1170 | Pseudofermion HMC (Paper 10) | ✅ **Validated** — pseudofermion HMC + GPU streaming dispatch (7/7 CPU, 13/13 streaming) |
| `eos_tables.rs` | 307 | HotQCD reference data | CPU-only (data) |
| `multi_gpu.rs` | 237 | Temperature scan dispatcher | CPU-threaded, GPU-ready |
| `gpu_hmc.rs` | ~2500 | GPU streaming HMC + resident CG + bidirectional stream | ✅ **Validated** — 9/9 streaming, 13/13 dyn streaming, 15,360× readback reduction |

### Remaining Gaps for Full Lattice QCD

| Gap | Needed For | Priority | Status |
|-----|-----------|----------|--------|
| ~~FFT (momentum-space)~~ | Full QCD with dynamical fermions | — | ✅ **Done** — toadstool `Fft1DF64`+`Fft3DF64` (14 GPU tests, 1e-10) |
| ~~GPU SU(3) plaquette shader~~ | GPU-accelerated HMC | — | ✅ **Done** — `wilson_plaquette_f64.wgsl` |
| ~~GPU HMC force + Abelian Higgs~~ | GPU-accelerated gauge evolution | — | ✅ **Done** — `su3_hmc_force_f64.wgsl` + `higgs_u1_hmc_f64.wgsl` |
| ~~GPU Dirac operator~~ | Fermion matrix-vector products | — | ✅ **Done** — `WGSL_DIRAC_STAGGERED_F64` validated 8/8 |
| ~~Dynamical fermion HMC~~ | Full QCD with sea quarks | — | ✅ **Done** — `pseudofermion.rs` validated 7/7, Python control parity |
| ~~Omelyan integrator~~ | ~~Production acceptance rates~~ | — | ✅ **Done** — Omelyan in gpu_hmc.rs, streaming dispatch achieves 50-90% acceptance |
| ~~Larger lattice sizes (8^4, 16^4)~~ | ~~Physical results~~ | — | ✅ **Done** — 4⁴-16⁴ streaming validated, 67× CPU at 16⁴. RTX 3090 enables 48⁴ |
| ~~Production beta-scan (32⁴)~~ | ~~Phase transition resolution~~ | — | ✅ **Done** — 12-point 32⁴ scan, 13.6h, χ=40.1 at β=5.69 matches β_c=5.692 |
| ~~DF64 hybrid core streaming~~ | ~~6.7× HMC speedup~~ | — | ✅ **Done** — v0.6.10-v0.6.12: gauge force, plaquette, KE all DF64. 60% of HMC in DF64, 2× speedup at 32⁴ |

### Production β-Scan Results (Feb 24, 2026)

The RTX 3090 completed a 12-point quenched SU(3) β-scan on a 32⁴ lattice (1M sites,
200 measurements/point, 3,000 HMC trajectories) in 13.6 hours for $0.58 of electricity.
The susceptibility peak χ=40.1 at β=5.69 matches the known critical coupling
β_c=5.692 to three significant figures — **deconfinement phase transition resolved**.
Finite-size scaling confirmed: 16⁴ (Titan V NVK, χ~1.0) vs 32⁴ (3090, χ=40-53).
This used only 1.6% of the 3090's chip (native f64). DF64 hybrid would reduce the
same run to ~2 hours. See `experiments/013_BIOMEGATE_PRODUCTION_BETA_SCAN.md`.

### Heterogeneous Hardware Pipeline: Lattice QCD Phase Structure

GPU FFT f64 is available (toadstool Session 25). The full GPU lattice QCD stack
is complete — Dirac SpMV (8/8), CG solver (9/9), and pseudofermion HMC (7/7)
are all validated. The **deconfinement phase transition** — the most
important observable in finite-temperature QCD — is now resolved on a 32⁴ lattice
using position-space quantities: the Polyakov loop ⟨|L|⟩ and plaquette ⟨P⟩.

**Pipeline**: GPU generates pure-gauge SU(3) configurations via HMC → NPU
classifies phases in real-time from (β, ⟨P⟩, ⟨|L|⟩) features → CPU validates
against known β_c ≈ 5.69. The NPU learns the mapping from gauge observables
to phase label (confined/deconfined) at negligible cost.

This heterogeneous approach makes lattice gauge theory phase structure accessible
on consumer hardware today — no FFT, no HPC, just GPU+NPU+CPU for ~$900 total.

**Validated**: `validate_lattice_npu` detects β_c = 5.715 (known 5.692, error 0.4%)
from real SU(3) HMC observables. ESN classifier achieves 100% test accuracy.
NpuSimulator f32 parity: max error 2.8e-7. 10/10 checks pass.

### What Was Previously Impossible — Now Demonstrated

`validate_hetero_monitor` (9/9 checks) proves five capabilities that were
literally impossible before the heterogeneous GPU+NPU+CPU pipeline:

| Capability | Before | Now | Measured |
|------------|--------|-----|----------|
| Live phase monitoring during HMC | Offline analysis only | ESN classifies each config as GPU generates it | 9.1 μs per prediction |
| Continuous transport prediction | Green-Kubo requires full trajectory | NPU predicts D*/η*/λ* from 10-frame window | Multi-output, all finite |
| Cross-substrate parity | Physics locked to one substrate | Same weights: CPU f64, f32 sim, int4 quantized | f32 error 5.1e-7, int4 error 0.13 |
| Zero-overhead monitoring | Monitoring paused simulation | Prediction takes 0.09% of HMC time | 9 μs vs 10.3 ms |
| Predictive steering | Uniform sampling wastes compute | ESN oracle focuses compute near phase boundary | 62% fewer evaluations, β_c error 0.013 |

### R. Anderson Extension: Hot Spring Microbial Evolution (Taq Corollary)

Rika Anderson (Carleton College) studies microbial evolution in extreme environments.
Her lab has published population genomics of *Sulfolobus islandicus* in the **same
Yellowstone hot springs** where *Thermus aquaticus* was discovered — the organism
whose Taq polymerase enabled PCR and anchors the constrained evolution thesis
(`gen3/CONSTRAINED_EVOLUTION_FORMAL.md` §1.1).

Campbell, Anderson et al. (2017) showed that geographic isolation between hot springs
drives structured genomic variation, with different populations showing different
susceptibilities to mobile genetic elements. This is constrained evolution in nature:
thermal constraint selects for survival, geographic isolation creates independent
evolutionary trajectories, and mobile elements provide material for innovation.

Anderson's 2021 mSystems paper explicitly cites Lenski's LTEE (§1.2 of the constrained
evolution thesis) and formalizes when stochastic forces dominate over deterministic
selection in extreme environments — introducing Muller's ratchet as a consequence
of extreme energy limitation.

**Papers queued**: See `specs/PAPER_REVIEW_QUEUE.md` — Papers 23-24 (Campbell 2017,
Anderson 2021). Reproduction of Paper 23 requires bioinformatics only (public
*Sulfolobus* genomes) — wetSpring's sovereign pipeline handles the computation.

### All Reproduced Papers (22 total)

Papers 14-22 are documented in `specs/PAPER_REVIEW_QUEUE.md`.

| # | Paper | Status | Highlights |
|---|-------|--------|------------|
| 1 | Sarkas Yukawa OCP MD | **Done** | 9/9 PP cases, GPU validated, 82× GPU speedup |
| 2 | Two-Temperature Model (TTM) | **Done** | 6/6 equilibration checks |
| 3 | Diaw et al. (2024) Surrogate Learning | **Done** | 15/15 benchmark convergence |
| 4 | Nuclear EOS (SEMF → HFB, AME2020) | **Done** | 2,042 nuclei, GPU-batched HFB, 478× speedup |
| 5 | Stanton & Murillo (2016) Transport | **Done** | 13/13 checks, Green-Kubo D*/η*/λ* |
| 6 | Murillo & Weisheit (1998) Screening | **Done** | 23/23 checks, Rust 2274× Python |
| 7 | HotQCD EOS tables (Bazavov 2014) | **Done** | Thermodynamic validation |
| 8 | Pure gauge SU(3) Wilson action | **Done** | 12/12 checks, HMC 96-100% acceptance |
| 10 | Dynamical fermion QCD (Bazavov 2016) | **Done** | 7/7 checks, pseudofermion HMC, Python control parity |
| 13 | Abelian Higgs (Bazavov 2015) | **Done** | 17/17 checks, U(1)+Higgs HMC, Rust 143× Python |

---

## metalForge: Hardware Beyond the SDK

hotSpring's metalForge initiative characterizes actual hardware behavior versus
vendor documentation. The same methodology that confirmed Titan V (GV100) native
f64 at 1:2 via NVK and measured consumer Ampere/Ada fp64 at ~1:64 (matching CUDA
hardware, correcting an earlier overestimate) was applied to the BrainChip AKD1000 NPU.

### NPU Beyond-SDK Discoveries (10 SDK assumptions overturned)

| Discovery | SDK Claim | Reality | Validation |
|-----------|-----------|---------|------------|
| Input channels | 1 or 3 only | Any count (tested 1-64) | 13/13 HW checks pass |
| FC layers | Independent | Merge into single HW pass (SkipDMA) | 6.7% overhead for 7 extra layers |
| Batch inference | Single sample | Batch=8 → 2.35× throughput | 427μs/sample at batch=8 |
| FC width | ~hundreds | Tested to 8192+ neurons | All map to hardware |
| Multi-output | Not documented | 10 outputs = 4.5% overhead vs 1 | Free cost for physics |
| Weight mutation | Not supported | set_variable() updates without reprogram | Exact linearity (error=0) |
| Power | "30mW" | Board floor 900mW, chip below noise | True inference unmeasurably small |
| Memory | 8MB SRAM | PCIe BAR1 exposes 16GB address space | Full NP mesh addressable |
| Program format | Opaque | FlatBuffer (program_info + program_data) | Weights via DMA, not in program |
| Engine | Simple inference | SkipDMA, on-chip learning, register access | C++ symbol analysis confirmed |

### NPU Validation Suite

| Suite | Checks | Status |
|-------|--------|--------|
| Python hardware (npu_beyond_sdk.py) | 13/13 | All pass on AKD1000 |
| Rust math (validate_npu_beyond_sdk) | 16/16 | All pass (pure math, substrate-independent) |
| Quantization cascade (npu_quantization_parity.py) | 4/4 | f32<0.01%, int8<5%, int4+act4<50% |
| Rust quantization (validate_npu_quantization) | 6/6 | All pass |
| Physics pipeline (npu_physics_pipeline.py) | 10/10 | MD→ESN→NPU→D*,η*,λ* end-to-end |
| Pipeline math (validate_npu_pipeline) | 10/10 | All pass (substrate-independent) |

### Cross-Substrate Math Parity

The same ESN reservoir math is validated on four substrates:

| Substrate | Precision | Error vs f64 | Validated |
|-----------|-----------|:---:|:---:|
| CPU (Rust) | f64 | reference | 16/16 checks |
| NpuSimulator (Rust) | f32 | <0.001% | 6/6 checks |
| Quantized (Rust) | int4 | <30% | 6/6 checks |
| AKD1000 hardware | int4 | measured | 13/13 checks |

This proves the physics is substrate-portable: train on GPU → deploy on NPU.

---

## Evolution Architecture: Biome → Fungus → Absorption

hotSpring is a biome. ToadStool/barracuda is the fungus — present in every
biome, evolved independently by each. ToadStool at Session 42+ with 612 shaders
(Jacobi fix, f64 compile fixes). The pattern:

```
hotSpring writes local → validates physics → hands off to toadstool → leans on upstream → deletes local
```

### What hotSpring evolved that toadstool absorbed

| Contribution | Type | Upstream Location | Session |
|-------------|------|-------------------|---------|
| Complex f64 arithmetic | WGSL shader | `shaders/math/complex_f64.wgsl` | S18 |
| SU(3) matrix algebra | WGSL shader | `shaders/math/su3.wgsl` | S18 |
| Wilson plaquette | WGSL shader | `shaders/lattice/wilson_plaquette_f64.wgsl` | S18 |
| SU(3) HMC force | WGSL shader | `shaders/lattice/su3_hmc_force_f64.wgsl` | S18 |
| Abelian Higgs HMC | WGSL shader | `shaders/lattice/higgs_u1_hmc_f64.wgsl` | S18 |
| CellList GPU fix | Rust op | `barracuda::ops::md::neighbor` | S18 |
| NAK eigensolve | WGSL shader | `batched_eigh_nak_optimized_f64.wgsl` | S4 |
| ReduceScalar feedback | Pipeline | `barracuda::pipeline` | S18 |
| Driver profiling | System | `barracuda::device::capabilities` | S7 |
| Staggered Dirac GPU | WGSL+Rust | `ops/lattice/dirac.rs` + `shaders/lattice/dirac_staggered_f64.wgsl` | S31d |
| CG solver (3 kernels) | WGSL+Rust | `ops/lattice/cg.rs` + `shaders/lattice/cg_kernels_f64.wgsl` | S31d |
| SubstrateCapability model | Rust enum | `device/substrate.rs` | S31d |
| 5 spherical HFB shaders | WGSL | `shaders/science/hfb/` (density, potentials, energy, hamiltonian) | S36-37 |
| 5 deformed HFB shaders | WGSL | `shaders/science/hfb_deformed/` (energy, potential, wavefunction, hamiltonian, bcs) | S36-37 |
| ESN export/import weights | Rust API | `esn_v2::ESN::export_weights()` + `import_weights()` | S36-37 |
| Spectral module (full) | Rust+WGSL | `barracuda::spectral::*` (Anderson, Lanczos, Hofstadter, Sturm) | S25-31h |

### What hotSpring is evolving now (ready for absorption)

| Module | Shader | Tests | Absorption priority |
|--------|--------|-------|---------------------|
| GPU Polyakov loop | `polyakov_loop_f64.wgsl` | 6/6 beta scan | Tier 1 — bidirectional evolution with toadStool |
| NVK allocation guard | Rust integration | Production tested | Tier 1 — protects nouveau users |
| `su3_math_f64.wgsl` | Naga-safe SU(3) pure math | 13/13 | Tier 1 — composition-safe for shader prepending |
| Screened Coulomb eigensolve | CPU (Sturm bisection) | 23/23 | Tier 2 |
| Tolerance/config module | Rust pattern (no WGSL) | 172 constants | Tier 2 (pattern) |

### Loop unroller u32 bug — FIXED

The loop unroller in `loop_unroller.rs` had a u32 literal suffix bug: `substitute_loop_var`
emitted bare ints (`"0"`) instead of `u32` literals (`"0u"`). This caused `BatchedEighGpu`
single-dispatch shader validation to panic. **Fixed** upstream; `catch_unwind` workaround removed.

### Cross-spring shader evolution

ToadStool's barracuda crate benefits from multi-Spring contributions.
See `wateringHole/handoffs/CROSS_SPRING_EVOLUTION_FEB22_2026.md` for the
full map. Key cross-pollination:

- **wetSpring → all**: `(zero + literal)` f64 constant precision fix in `math_f64.wgsl` improved `log_f64` from ~1e-3 to ~1e-15
- **hotSpring → all**: NVK driver workarounds via `ShaderTemplate`, spectral module, eigensolve profiling
- **neuralSpring → hotSpring**: `BatchIprGpu` for GPU Anderson localization diagnostics
- **wetSpring → hotSpring**: `GemmCached` (60× speedup for repeated GEMM) benefits HFB SCF loop

### metalForge Forge: Hardware Discovery

`metalForge/forge/` discovers local hardware via wgpu (same path barracuda uses)
plus local NPU/CPU probing. It finds: 2 GPUs (RTX 4070 + Titan V, both SHADER_F64),
1 NPU (AKD1000), 1 CPU (i9-12900K with AVX2). Dispatch routes workloads by
capability: f64 compute → GPU, quantized inference → NPU, validation → CPU.

biomeGate (Threadripper 3970X, RTX 3090 + Titan V, Akida NPU) is lab-deployable
for extended compute. Node profiles (`metalForge/nodes/biomegate.env`) configure
GPU selection via `HOTSPRING_GPU_ADAPTER`. NVK setup guide at
`metalForge/gpu/nvidia/NVK_SETUP.md` makes Titan V driver installation reproducible.

---

## Relation to Other Documents

- **`whitePaper/barraCuda/`** (main repo, gated): The BarraCuda evolution story — how scientific workloads drove the library's development. Sections 04 and 04a reference hotSpring data.
- **`whitePaper/gen3/`** (main repo, gated): The constrained evolution thesis — hotSpring provides quantitative evidence for convergent evolution between ML and physics math.
- **`wateringHole/handoffs/`** (internal): Detailed technical handoffs to the ToadStool/BarraCuda team with code locations, bug fixes, and GPU roadmap.
- **This directory** (`hotSpring/whitePaper/`): Public-facing study focused on the science replication itself.

---

## Reproduction

```bash
# Phase A (Python, ~12 hours total)
bash scripts/regenerate-all.sh

# Phase B (BarraCuda, ~2 hours total)
cd barracuda
cargo run --release --bin nuclear_eos_l1_ref          # L1: ~3 seconds
cargo run --release --bin nuclear_eos_l2_ref -- --seed=42 --lambda=0.1   # L2: ~55 min
```

```bash
# Phase C-E (GPU MD, requires SHADER_F64 GPU)
cd barracuda
cargo run --release --bin sarkas_gpu -- --full    # 9 PP Yukawa cases, N=2000, 30k steps (~60 min)
cargo run --release --bin sarkas_gpu -- --long    # 9 cases, N=2000, 80k steps (~71 min)
cargo run --release --bin sarkas_gpu -- --paper   # 9 cases, N=10k, 80k steps (~3.66 hrs, paper parity)
cargo run --release --bin sarkas_gpu -- --nscale  # N-scaling: N=500-20000
cargo run --release --bin nuclear_eos_l2_gpu      # GPU-batched L2 HFB (BatchedEighGpu)
cargo run --release --bin validate_pppm           # PppmGpu kappa=0 Coulomb validation
```

```bash
# Phase F (Full-scale nuclear EOS, 2,042 nuclei)
cd barracuda
cargo run --release --bin nuclear_eos_l1_ref -- --nuclei=full --pareto       # L1 Pareto (~11 min)
cargo run --release --bin nuclear_eos_l2_gpu -- --nuclei=full --phase1-only  # L2 GPU (~66 min)
cargo run --release --bin nuclear_eos_l3_ref -- --nuclei=full --params=best_l2_42  # L3 (~4.5 hrs)
```

No institutional access required. No Code Ocean account. No Fortran compiler. AGPL-3.0 licensed.

---

## Codebase Health (Feb 26, 2026)

| Metric | Value |
|--------|-------|
| Crate | v0.6.14 |
| Unit tests | ~**665** pass + 1 env-flaky, 6 GPU/heavy-ignored (spectral tests upstream in barracuda) |
| Integration tests | **31** pass (3 suites: physics, data, transport) |
| WGSL shaders | **27** lattice + 14 MD + 14 HFB + 8 diag = **62** |
| Rust files | **135+** |
| Coverage | 74.9% region / 83.8% function |
| Validation suites | **39/39** pass |
| metalForge forge tests | **19** pass |
| Experiments | **21** (001-021) |
| Python control scripts | **34** (Sarkas, surrogate, TTM, NPU, reservoir, lattice, spectral theory) |
| Rust validation binaries | **78** (physics, MD, lattice, NPU, transport, spectral, benchmarks, production, ESN cross-substrate) |
| `expect()`/`unwrap()` in library | **0** (crate-level deny) |
| Clippy warnings | **0** (pedantic + nursery, workspace-wide) |
| Doc warnings | **0** |
| Unsafe blocks | **0** |
| External FFI/C bindings | **0** (all pure Rust except wgpu GPU driver bridge) |
| Centralized tolerances | **172** constants (including 8 solver config, 6 dynamical QCD) |
| Hardcoded solver params | **0** (all centralized in `tolerances/`) |
| Files over 1000 LOC | **1** (`hfb_gpu_resident/mod.rs` — monolithic GPU pipeline) |
| Provenance records | All validation targets traced to Python origins or DOIs |
| AGPL-3.0 compliance | All `.rs` and `.wgsl` files |

---

## GPU FP64 Status (Feb 24, 2026)

Native FP64 GPU compute confirmed on RTX 4070, RTX 3090, and Titan V via `wgpu::Features::SHADER_F64` (Vulkan backend):
- **Precision**: True IEEE 754 double precision (0 ULP error vs CPU f64)
- **Performance**: Consumer Ampere/Ada fp64:fp32 ~1:64 (hardware limit, same as CUDA); Titan V (GV100) provides native 1:2 via 2,560 dedicated FP64 cores. Confirmed by `bench_fp64_ratio` FMA chain micro-benchmark (Feb 24 2026).
- **Implication**: Consumer GPUs provide exact fp64 for correctness; the Titan V (~$500 used) provides compute-class fp64 throughput via the open-source NVK driver
- **Multi-GPU**: RTX 3090/4070 (nvidia proprietary) and Titan V (NVK/nouveau open-source) both produce identical physics to 1e-15
- **Phase C validation**: Full Yukawa MD (9 cases, N=2000, 80k steps) runs at 149-259 steps/s sustained with 0.000% energy drift
- **Phase E validation**: Full paper-parity (9 cases, N=10,000, 80k steps) completes in 3.66 hours with 0.000-0.002% drift. Cell-list 4.1× faster than all-pairs.
- **GPU-only transport pipeline**: `validate_transport_gpu_only` runs full Green-Kubo D*/η*/λ* pipeline on GPU with zero readback (~493s).
- **Unidirectional pipeline**: GPU sum-reduction eliminates per-particle readback — 10,000× bandwidth reduction at N=10,000
- **GPU-resident CG solver**: α, β, rz scalars computed on GPU. 10-iteration batches with 8-byte convergence readback. 15,360× readback reduction (37 MB → 2.4 KB per trajectory).
- **Bidirectional streaming**: 90%+ data to GPU, async readback for CG convergence, NPU branch for phase screening.
- **biomeGate**: RTX 3090 (24GB) extends lattice capacity to 48⁴ dynamical fermion. Node profiles at `metalForge/nodes/`.
