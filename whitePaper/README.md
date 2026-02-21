# hotSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository  
**Purpose**: Document the replication of Murillo Group computational plasma physics on consumer hardware using BarraCUDA  
**Date**: February 20, 2026

---

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [STUDY.md](STUDY.md) | **Main study** — full writeup of the two-phase validation, data sources, results, and path to paper parity | Reviewers, collaborators |
| [BARRACUDA_SCIENCE_VALIDATION.md](BARRACUDA_SCIENCE_VALIDATION.md) | Phase B technical results — BarraCUDA vs Python/SciPy numbers | Technical reference |
| [CONTROL_EXPERIMENT_SUMMARY.md](CONTROL_EXPERIMENT_SUMMARY.md) | Phase A summary — Python reproduction of published work | Quick reference |
| [METHODOLOGY.md](METHODOLOGY.md) | Two-phase validation protocol | Methodology review |

---

## What This Study Is

hotSpring replicates published computational plasma physics from the Murillo Group (Michigan State University) on consumer hardware, then re-executes the computations using BarraCUDA — a Pure Rust scientific computing library with zero external dependencies.

The study answers five questions:
1. **Can published computational science be independently reproduced?** (Answer: yes, but it required fixing 6 silent bugs and rebuilding physics that was behind a gated platform)
2. **Can Rust + WebGPU replace the Python scientific stack for real physics?** (Answer: yes — BarraCUDA achieves 478× faster throughput and 44.8× less energy at L1, with GPU FP64 validated to 4.55e-13 MeV precision. Full Sarkas Yukawa MD runs on a $600 consumer GPU: 9/9 PP cases pass at N=10,000 with 80,000 production steps in 3.66 hours for $0.044.)
3. **Can consumer GPUs do first-principles nuclear structure at scale?** (Answer: yes — the full AME2020 dataset (2,042 nuclei, 39x the published paper) runs on a single RTX 4070. L1 Pareto analysis, L2 GPU-batched HFB, and L3 deformed HFB all produce results. This is direct physics computation, not surrogate learning.)
4. **Does the Python → Rust → GPU evolution path extend beyond plasma physics?** (Answer: yes — lattice QCD (SU(3) pure gauge, HMC, staggered Dirac), Abelian Higgs (U(1) gauge + Higgs field, 143× faster than Python), transport coefficients (Green-Kubo, Stanton-Murillo), screened Coulomb (Sturm eigensolve, 2274× faster than Python), and HotQCD EOS tables are all validated on CPU with WGSL templates ready for GPU promotion. 9 papers reproduced, 300+ validation checks, ~$0.20 total compute cost.)
5. **Can physics math be truly substrate-portable — CPU → GPU → NPU?** (Answer: yes — ESN reservoir math validated across f64 CPU, f32 NpuSimulator, int4 quantized, and real AKD1000 NPU hardware. 10 SDK assumptions overturned by probing beyond the SDK. The same WGSL shader math trains on GPU and deploys on NPU for inference at 30mW. See `metalForge/npu/akida/BEYOND_SDK.md`.)

---

## Key Results

### Phase A (Python Control): 86/86 checks pass

- Sarkas MD: 12 cases, 60 observable checks, 8.3% mean DSF peak error
- TTM: 6/6 equilibration checks pass
- Surrogate learning: 15/15 benchmark functions converge
- Nuclear EOS: Python L1 (chi2=6.62), L2 (chi2=1.93 via SparsitySampler)
- 5 silent upstream bugs found and fixed

### Phase B (BarraCUDA): GPU-validated, energy-profiled

| Level | BarraCUDA | Python/SciPy | Speedup | Energy Ratio |
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
- **195/195 quantitative checks pass** across all phases + pipeline validation

---

## Bazavov Extension: Lattice QCD on Consumer Hardware

hotSpring has extended from plasma physics to lattice gauge theory. The
Bazavov connection (CMSE & Physics, MSU) provides the bridge: both Murillo
and Bazavov study strongly coupled many-body systems with overlapping
computational methods (MD ↔ HMC, plasma EOS ↔ QCD EOS).

### Completed (February 20, 2026)

| Paper | Status | Implementation |
|-------|--------|----------------|
| Stanton & Murillo (2016) transport | **Done** | Green-Kubo D*/η*/λ*; Sarkas-calibrated fits + C_w(κ) evolution, 13/13 checks |
| Murillo & Weisheit (1998) screening | **Done** | Screened Coulomb eigenvalues; Sturm bisection, 23/23 checks, Rust 2274× Python |
| HotQCD EOS tables (Bazavov 2014) | **Done** | `lattice/eos_tables.rs` — thermodynamic validation passes |
| Pure gauge SU(3) Wilson action | **Done** | `lattice/` — 8 modules, 12/12 validation checks |
| Abelian Higgs (Bazavov 2015) | **Done** | `lattice/abelian_higgs.rs` — U(1)+Higgs HMC, 17/17 checks, Rust 143× Python |

### Lattice QCD Infrastructure Built

| Module | Lines | Purpose | GPU Status |
|--------|-------|---------|------------|
| `complex_f64.rs` | 316 | Complex f64 arithmetic | ✅ **Absorbed** — `complex_f64.wgsl` (toadstool `8fb5d5a0`) |
| `su3.rs` | 460 | SU(3) matrix algebra | ✅ **Absorbed** — `su3.wgsl` (toadstool `8fb5d5a0`) |
| `wilson.rs` | 338 | Wilson gauge action, plaquettes | ✅ **Absorbed** — `wilson_plaquette_f64.wgsl` |
| `hmc.rs` | 350 | HMC with Cayley exponential | ✅ **Absorbed** — `su3_hmc_force_f64.wgsl` |
| `abelian_higgs.rs` | ~500 | U(1)+Higgs (1+1)D HMC | ✅ **Absorbed** — `higgs_u1_hmc_f64.wgsl` |
| `dirac.rs` | 297 | Staggered Dirac operator | **P1** — GPU SpMV needed (last QCD blocker) |
| `cg.rs` | 214 | Conjugate gradient for D†D | **P2** — needs GPU SpMV + dot + axpy |
| `eos_tables.rs` | 307 | HotQCD reference data | CPU-only (data) |
| `multi_gpu.rs` | 237 | Temperature scan dispatcher | CPU-threaded, GPU-ready |

### Remaining Gaps for Full Lattice QCD

| Gap | Needed For | Priority | Status |
|-----|-----------|----------|--------|
| ~~FFT (momentum-space)~~ | Full QCD with dynamical fermions | — | ✅ **Done** — toadstool `Fft1DF64`+`Fft3DF64` (14 GPU tests, 1e-10) |
| ~~GPU SU(3) plaquette shader~~ | GPU-accelerated HMC | — | ✅ **Done** — `wilson_plaquette_f64.wgsl` |
| ~~GPU HMC force + Abelian Higgs~~ | GPU-accelerated gauge evolution | — | ✅ **Done** — `su3_hmc_force_f64.wgsl` + `higgs_u1_hmc_f64.wgsl` |
| GPU Dirac operator | Fermion matrix-vector products | **P1** | CPU validated; WGSL port needed |
| Larger lattice sizes (8^4, 16^4) | Physical results | **P2** | GPU lattice shaders now available |

### Heterogeneous Hardware Pipeline: Lattice QCD Phase Structure

GPU FFT f64 is now available (toadstool Session 25). The full lattice QCD stack
(Tier 3) only needs GPU Dirac SpMV for dynamical fermions. Meanwhile, the
**deconfinement phase transition** — the most important observable in finite-
temperature QCD — is visible in purely position-space quantities: the Polyakov
loop ⟨|L|⟩ and plaquette ⟨P⟩. No FFT needed for phase structure.

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

### All Reproduced Papers (18 total, 9 original + 9 Kachkovskiy spectral)

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
| 13 | Abelian Higgs (Bazavov 2015) | **Done** | 17/17 checks, U(1)+Higgs HMC, Rust 143× Python |

---

## metalForge: Hardware Beyond the SDK

hotSpring's metalForge initiative characterizes actual hardware behavior versus
vendor documentation. The same methodology that found native f64 at 1:2 on
consumer GPUs (vs CUDA's advertised 1:32) was applied to the BrainChip AKD1000 NPU.

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

## Relation to Other Documents

- **`whitePaper/barraCUDA/`** (main repo, gated): The BarraCUDA evolution story — how scientific workloads drove the library's development. Sections 04 and 04a reference hotSpring data.
- **`whitePaper/gen3/`** (main repo, gated): The constrained evolution thesis — hotSpring provides quantitative evidence for convergent evolution between ML and physics math.
- **`wateringHole/handoffs/`** (internal): Detailed technical handoffs to the ToadStool/BarraCUDA team with code locations, bug fixes, and GPU roadmap.
- **This directory** (`hotSpring/whitePaper/`): Public-facing study focused on the science replication itself.

---

## Reproduction

```bash
# Phase A (Python, ~12 hours total)
bash scripts/regenerate-all.sh

# Phase B (BarraCUDA, ~2 hours total)
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

## Codebase Health (Feb 20, 2026)

| Metric | Value |
|--------|-------|
| Unit tests | **441** pass, 5 GPU-ignored (446 total) |
| Validation suites | **33/33** pass |
| Python control scripts | **34** (Sarkas, surrogate, TTM, NPU, reservoir, lattice, spectral theory) |
| Rust validation binaries | **50** (physics, MD, lattice, NPU, transport, spectral 1D/2D/3D, Lanczos, Hofstadter) |
| Clippy warnings | **0** (default + pedantic on library code) |
| Doc warnings | **0** |
| Unsafe blocks | **0** |
| TODO/FIXME/HACK markers | **0** |
| Centralized tolerances | **122** constants in `tolerances.rs` |
| Provenance records | All validation targets traced to Python origins or DOIs |
| AGPL-3.0 compliance | All `.rs` and `.wgsl` files |

---

## GPU FP64 Status (Feb 20, 2026)

Native FP64 GPU compute confirmed on RTX 4070 and Titan V via `wgpu::Features::SHADER_F64` (Vulkan backend):
- **Precision**: True IEEE 754 double precision (0 ULP error vs CPU f64)
- **Performance**: ~2x FP64:FP32 ratio for bandwidth-limited operations (not the CUDA-reported 1:64)
- **Implication**: The RTX 4070 is usable for FP64 science compute today via BarraCUDA's wgpu shaders
- **Multi-GPU**: RTX 4070 (nvidia proprietary) and Titan V (NVK/nouveau open-source) both produce identical physics to 1e-15
- **Phase C validation**: Full Yukawa MD (9 cases, N=2000, 80k steps) runs at 149-259 steps/s sustained with 0.000% energy drift
- **Phase E validation**: Full paper-parity (9 cases, N=10,000, 80k steps) completes in 3.66 hours with 0.000-0.002% drift. Cell-list 4.1× faster than all-pairs.
- **Unidirectional pipeline**: GPU sum-reduction eliminates per-particle readback — 10,000× bandwidth reduction at N=10,000
