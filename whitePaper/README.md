# hotSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository  
**Purpose**: Document the replication of Murillo Group computational plasma physics on consumer hardware using BarraCUDA  
**Date**: February 2026

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

The study answers three questions:
1. **Can published computational science be independently reproduced?** (Answer: yes, but it required fixing 5 silent bugs and rebuilding physics that was behind a gated platform)
2. **Can Rust + WebGPU replace the Python scientific stack for real physics?** (Answer: yes — BarraCUDA achieves 478× faster throughput and 44.8× less energy at L1, with GPU FP64 validated to 4.55e-13 MeV precision. Full Sarkas Yukawa MD runs on a $600 consumer GPU: 9/9 PP cases pass at N=10,000 with 80,000 production steps in 3.66 hours for $0.044.)
3. **Can consumer GPUs do first-principles nuclear structure at scale?** (Answer: yes — the full AME2020 dataset (2,042 nuclei, 39x the published paper) runs on a single RTX 4070. L1 Pareto analysis, L2 GPU-batched HFB, and L3 deformed HFB all produce results. This is direct physics computation, not surrogate learning.)

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

### Completed (February 19, 2026)

| Paper | Status | Implementation |
|-------|--------|----------------|
| Stanton & Murillo (2016) transport | **Partial** | D*, η*, λ* fits + Green-Kubo; normalization bug in stress/heat ACF |
| HotQCD EOS tables (Bazavov 2014) | **Done** | `lattice/eos_tables.rs` — thermodynamic validation passes |
| Pure gauge SU(3) Wilson action | **Done** | `lattice/` — 8 modules, 12/12 validation checks |

### Lattice QCD Infrastructure Built

| Module | Lines | Purpose | GPU-Ready |
|--------|-------|---------|-----------|
| `complex_f64.rs` | 316 | Complex f64 with WGSL template | WGSL string included |
| `su3.rs` | 460 | SU(3) matrix algebra with WGSL template | WGSL string included |
| `wilson.rs` | 338 | Wilson gauge action, plaquettes, force | Needs WGSL shader |
| `hmc.rs` | 350 | HMC with Cayley exponential | Needs WGSL shader |
| `dirac.rs` | 297 | Staggered Dirac operator | Needs WGSL shader |
| `cg.rs` | 214 | Conjugate gradient for D†D | Needs WGSL shader |
| `eos_tables.rs` | 307 | HotQCD reference data | CPU-only (data) |
| `multi_gpu.rs` | 237 | Temperature scan dispatcher | CPU-threaded, GPU-ready |

### Remaining Gaps for Full Lattice QCD

| Gap | Needed For | Priority |
|-----|-----------|----------|
| FFT (momentum-space) | Full QCD with dynamical fermions | High |
| GPU SU(3) plaquette shader | GPU-accelerated HMC | High |
| GPU Dirac operator | Fermion matrix-vector products | High |
| Larger lattice sizes (8^4, 16^4) | Physical results | Medium |

### Existing Reproduction (All Papers)

| # | Paper | Status | Phase |
|---|-------|--------|-------|
| 1 | Sarkas Yukawa OCP MD | Done | A + C-E (GPU) |
| 2 | Two-Temperature Model (TTM) | Done | A |
| 3 | Diaw et al. (2024) Surrogate Learning | Done | A |
| 4 | Nuclear EOS (SEMF → HFB, AME2020) | Done | A + F |
| 5 | Stanton & Murillo (2016) Transport | Partial | Green-Kubo normalization bug |
| 7 | HotQCD EOS tables (Bazavov 2014) | Done | Validation passes |
| 8 | Pure gauge SU(3) Wilson action | Done | 12/12 checks, HMC working |

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

## GPU FP64 Status (Feb 15, 2026)

Native FP64 GPU compute confirmed on RTX 4070 via `wgpu::Features::SHADER_F64` (Vulkan backend):
- **Precision**: True IEEE 754 double precision (0 ULP error vs CPU f64)
- **Performance**: ~2x FP64:FP32 ratio for bandwidth-limited operations (not the CUDA-reported 1:64)
- **Implication**: The RTX 4070 is usable for FP64 science compute today via BarraCUDA's wgpu shaders
- **Phase C validation**: Full Yukawa MD (9 cases, N=2000, 80k steps) runs at 149-259 steps/s sustained with 0.000% energy drift
- **Phase E validation**: Full paper-parity (9 cases, N=10,000, 80k steps) completes in 3.66 hours with 0.000-0.002% drift. Cell-list 4.1× faster than all-pairs.
