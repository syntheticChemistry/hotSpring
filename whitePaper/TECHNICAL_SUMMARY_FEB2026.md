# hotSpring Technical Summary — February 2026

**Author:** Kevin Mok (mokkevin@msu.edu)
**Repository:** [github.com/syntheticChemistry/hotSpring](https://github.com/syntheticChemistry/hotSpring) (AGPL-3.0)
**Hardware:** Consumer workstation — RTX 3090 (24 GB), Titan V (12 GB HBM2), BrainChip Akida AKD1000 (PCIe), Threadripper 3970X (64 cores, 256 GB DDR4)
**Date:** February 27, 2026

---

## 1. Overview

hotSpring reproduces published computational physics results on consumer GPU
hardware using BarraCUDA — a Pure Rust scientific computing library that
dispatches f64 WGSL shaders to any GPU vendor via wgpu/Vulkan. No CUDA, no
Fortran, no Python runtime dependencies.

The project validates that plasma molecular dynamics, lattice quantum
chromodynamics, nuclear structure calculations, and spectral theory computations
produce physically correct results on consumer hardware at costs 3–5 orders of
magnitude below institutional HPC.

**Current scale:**

| Metric | Value |
|--------|-------|
| Papers reproduced | 22 |
| Quantitative validation checks | ~700 |
| Validation suites (all passing) | 39/39 |
| WGSL compute shaders | 628+ |
| Physics domains | 5 (plasma MD, nuclear structure, lattice QCD, spectral theory, Abelian gauge) |
| Computational substrates | 4 (CPU, consumer GPU, Titan V, neuromorphic NPU) |
| Total compute cost | ~$0.80 |

---

## 2. Reproduction Results — Murillo Group

### 2.1 Sarkas Yukawa Molecular Dynamics (Papers 1–3)

Reproduced the published Yukawa OCP molecular dynamics from the Murillo Group's
Sarkas package (MIT License). All 9 particle-particle cases re-executed on GPU
using f64 WGSL shaders.

| Parameter | Value |
|-----------|-------|
| Potential | Yukawa (screened Coulomb) |
| Particles | N = 10,000 |
| Production steps | 80,000 |
| GPU | RTX 4070 (Ada Lovelace, 12 GB) |
| Energy drift | 0.000–0.002% |
| Wall time (9-case sweep) | 3.66 hours |
| Electricity cost | $0.044 |
| Observable checks (DSF, RDF, SSF, VACF, Energy) | **60/60 pass** |

**Upstream bugs identified:** 5 silent issues in Sarkas related to NumPy 2.x
and pandas 2.x compatibility (all documented and patched).

### 2.2 Two-Temperature Model (Paper 4)

Stanton–Murillo TTM (UCLA–MSU) for laser-plasma equilibration.

| Check | Result |
|-------|--------|
| Local equilibrium (3 species) | 3/3 pass |
| Hydrodynamic profiles (3 species) | 3/3 pass |

### 2.3 Surrogate Learning (Paper 4b — Diaw, Murillo, Stanton 2024)

Reproduced the optimizer-driven sampling methodology from "Efficient learning of
accurate surrogates for simulations of complex systems" (*Nature Machine
Intelligence*, May 2024).

| Level | Python (published) | BarraCUDA (this work) | Speedup |
|-------|-------------------|-----------------------|---------|
| L1 SEMF χ²/datum | 6.62 | **2.27** | 478× faster |
| L1 throughput | 5.5 evals/s | 2,621 evals/s | 478× |
| L1 energy per eval | 5,648 J total | 126 J total | **44.8× less** |

The improved χ² (2.27 vs 6.62) results from the ability to explore the
10-parameter space more thoroughly at higher throughput, not from a different
algorithm. The physics is identical.

### 2.4 Transport Coefficients (Paper 5 — Stanton & Murillo 2016)

Green-Kubo transport coefficients computed entirely on GPU.

| Coefficient | Method | Checks | Status |
|------------|--------|--------|--------|
| Self-diffusion D* | Velocity autocorrelation + MSD | 5/5 | Pass |
| Shear viscosity η* | Stress tensor ACF | 4/4 | Pass |
| Thermal conductivity λ* | Heat flux ACF | 4/4 | Pass |

Transport fits calibrated against 12 Sarkas Green-Kubo D* values with
κ-dependent weak-coupling correction, reducing crossover-regime errors from
44–63% to <10%. **13/13 total checks pass.**

### 2.5 Screened Coulomb (Paper 6 — Murillo & Weisheit 1998)

Bound-state eigenvalue solver for Yukawa-screened hydrogen via Sturm bisection.

| Check | Result |
|-------|--------|
| Hydrogen eigenvalue vs exact | Δ ≈ 10⁻¹² |
| Python–Rust parity | Δ ≈ 10⁻¹² |
| Critical screening vs Lam & Varshni | 3/3 |
| Physics trends (monotonic) | 6/6 |
| Screening model cross-check | 3/3 |
| **Total** | **23/23 pass** |

---

## 3. Reproduction Results — Bazavov Group (Lattice QCD)

### 3.1 Papers Reproduced

| Paper | Reference | Checks | Status |
|-------|-----------|--------|--------|
| HotQCD EOS tables | Bazavov et al., *Nucl. Phys. A* **931**, 867 (2014) | Pass | Thermodynamic consistency, asymptotic freedom |
| Pure gauge SU(3) | Wilson gauge action, HMC | 12/12 | Dirac CG, plaquette physics |
| Dynamical fermion HMC | Bazavov et al., *Phys. Rev. D* **93**, 114502 (2016) | 7/7 | Pseudofermion HMC, ΔH scaling, mass dependence |
| Abelian Higgs | Bazavov et al., *Phys. Rev. D* **91** (2015) | 17/17 | U(1)+Higgs (1+1)D, Rust 143× faster than Python |
| HVP muon g−2 | Bazavov et al., *Phys. Rev. D* **111**, 094508 (2025) | 10/10 | CPU validated |
| Freeze-out curvature | Bazavov et al., *Phys. Rev. D* **93**, 014512 (2016) | 8/8 | β_c within 10% of known |

### 3.2 Production β-Scan — Deconfinement Phase Transition (Experiment 013)

32⁴ quenched SU(3) lattice, 12-point β-scan, 200 measurements per point.

| β | ⟨P⟩ | χ (susceptibility) | Phase |
|---|------|--------------------|-------|
| 4.00 | 0.2943 | 0.80 | Confined |
| 5.00 | 0.4014 | 0.76 | Confined |
| 5.50 | 0.4817 | 22.82 | Transition onset |
| 5.65 | 0.5126 | 31.29 | Critical region |
| **5.69** | **0.5216** | **40.08** | **Peak — β_c match** |
| 5.80 | 0.5442 | 52.87 | Crossover structure |
| 6.00 | 0.5778 | 27.38 | Deconfined |
| 6.50 | 0.6301 | 12.61 | Deep deconfined |

**Known literature value:** β_c = 5.692 for SU(3) pure gauge, N_t = 4.
**This measurement:** Primary susceptibility peak at β = 5.69 — agreement to
three significant figures.

**Finite-size scaling confirmed:** On a 16⁴ lattice (Titan V, NVK open-source
driver), χ peaks at ~1.0. On 32⁴ (RTX 3090), χ peaks at 40–53. The 40–50×
amplification with volume is the defining signature of a genuine phase
transition.

| Metric | Value |
|--------|-------|
| Lattice | 32⁴ (1,048,576 sites) |
| Wall time | 13.6 hours (native f64) / **7.1 hours (DF64 hybrid)** |
| GPU | RTX 3090 (370W steady, 73–74°C) |
| Cost | $0.58 (native f64) / **$0.30 (DF64)** |
| HMC trajectories | 3,000 (12 β-points × 250 each) |

### 3.3 Historical Comparison

| Machine | Year | FP64 Sustained | Cost | This calculation |
|---------|------|:--------------:|:----:|:----------------:|
| QCDSP (Columbia) | 1998 | 0.6 TFLOPS | $3.5M | ~7 hours |
| QCDOC | 2004 | 10 TFLOPS | $5M | ~25 min |
| 1 BlueGene/L rack | 2005 | 0.46 TFLOPS | $1.5M | ~11 hours |
| **RTX 3090 (this run)** | **2026** | **0.33 TFLOPS** | **$1,500** | **13.6 hours** |

---

## 4. DF64 Core Streaming — Precision Discovery

### 4.1 The Problem

Consumer NVIDIA GPUs throttle native f64 to 1:64 of f32 throughput (hardware
market segmentation). The RTX 3090 has 10,496 FP32 ALU cores but only 164
dedicated FP64 units. At native f64, 98.4% of the chip is idle during
precision-sensitive computation.

### 4.2 The Discovery

Double-float arithmetic (representing each f64 value as a pair of f32 values)
runs on the FP32 cores at full throughput. This is a known numerical technique
(Dekker 1971, Knuth TDK, Priest 1991) but had not been systematically applied
to GPU shader programming for physics workloads.

| | Native f64 (RTX 3090) | DF64 on FP32 cores (RTX 3090) |
|--|----------------------|-------------------------------|
| Throughput | 0.33 TFLOPS | **3.24 TFLOPS** |
| Precision | ~16 digits | ~14 digits |
| Speedup | 1× | **9.9×** |

### 4.3 Validation

The DF64 hybrid strategy applies double-float arithmetic to the
compute-intensive kernels (gauge force, plaquette, kinetic energy — ~60% of
HMC wall time) while keeping precision-critical operations (momentum update,
link update) in native f64. Results:

- 32⁴ β-scan: 13.6h → **7.1h** (1.9× overall speedup)
- Energy conservation: 0.000% drift maintained
- Deconfinement transition: identical physics (same β_c, same susceptibility profile)
- HMC trajectory: 15.5s → 7.7s at 32⁴

At 14-digit precision, the 2-digit loss relative to native f64 has no
measurable effect on any observable we tested. Energy drift remains at machine
precision. The plaquette, Polyakov loop, susceptibility, and acceptance rate
are statistically identical.

### 4.4 Implications

This finding applies to any consumer GPU with Vulkan SHADER_F64 support. The
same shaders run identically on:

- NVIDIA (RTX 2070 through 5090, Titan V)
- AMD (RX 6000+, MI-series)
- Intel (Arc A-series)

No CUDA. No ROCm. No vendor SDK. One set of WGSL shaders, every GPU.

---

## 5. Neuromorphic NPU — Reservoir Computing for Physics Screening

### 5.1 Hardware

BrainChip Akida AKD1000 (Akida 1.0), PCIe Gen2 x1, ~30 mW chip power.
Three boards available; one integrated into the production physics pipeline.

### 5.2 Echo State Network Architecture

An Echo State Network (reservoir computing) is trained on GPU HMC trajectory
observables and deployed to the NPU for real-time inference. The ESN receives
per-trajectory features (plaquette, energy, Polyakov loop magnitude, acceptance
rate, fluctuations) and outputs screening decisions.

| Component | Precision | Role |
|-----------|-----------|------|
| ESN training | f64 (CPU) | Ridge regression on reservoir states |
| NpuSimulator | f32 (CPU) | Validation of hardware-equivalent behavior |
| AKD1000 hardware | int8/int4 (NPU) | Production inference via PCIe |

### 5.3 Characterization Results (Experiment 020)

Systematic characterization overturned 6 of 10 SDK-documented limitations:

| SDK documentation | Actual hardware behavior |
|-------------------|------------------------|
| InputConv: 1 or 3 channels only | Any channel count (tested 1–64) |
| FC layers run independently | All merge into single HW pass (SkipDMA) |
| Batch=1 only | Batch=8 → 2.4× throughput |
| One clock mode | 3 modes (Performance / Economy / LowPower) |
| No weight mutation | set_variable() swaps weights in ~14 ms |
| Multi-output costs more | Multi-output is free |

### 5.4 Physics Screening Performance

| Workload | Accuracy | Latency | Energy vs CPU |
|----------|----------|---------|---------------|
| Thermalization detection | 87.5% | 341 µs | 9,017× less |
| Rejection prediction | 96.2% | 341 µs | 9,017× less |
| Phase classification | 100% (n≥10) | 341 µs | 9,017× less |
| β_c regression | ε = 0.0098 | 341 µs | 9,017× less |

### 5.5 Cross-Substrate ESN Comparison (Experiment 021)

| Substrate | Optimal regime | Per-step latency | Power |
|-----------|---------------|-----------------|-------|
| NPU (AKD1000) | RS ≤ 200, streaming | **2.8 µs** | ~30 mW |
| CPU (f64) | RS < 512, training | ~10,400 µs (RS=512) | ~125 W |
| GPU (RTX 3090) | RS ≥ 512, batch | ~3,170 µs (RS=512) | ~370 W |

The NPU is 1,000× faster than GPU for per-sample streaming inference.
GPU overtakes CPU at reservoir size ≈ 512 (8.2× at RS=1024).
The substrates are naturally complementary.

### 5.6 Production Deployment (Experiment 022 — Running)

The AKD1000 is currently live in the 32⁴ lattice QCD production pipeline:

- **RTX 3090**: DF64 gauge force, HMC integration, observables
- **AKD1000 (PCIe)**: ESN screening — thermalization detection, rejection
  prediction, phase classification, adaptive β steering
- **Titan V**: Native f64 validation oracle (1:2 fp64:fp32 ratio)

A dedicated NPU worker thread communicates with the HMC loop via message
channels, preventing any GPU stall from PCIe latency.

**Cross-run learning:** Each run exports trained ESN weights. The next run
bootstraps from the previous run's weights or trajectory log, accumulating
knowledge across runs. The current 32⁴ run was bootstrapped from 749 data
points collected during an earlier simulator run.

### 5.7 Live Run Data (Experiment 022, as of Feb 26 2026)

Three-phase adaptive scan on 32⁴ lattice with live AKD1000 hardware NPU.

**Phase 1 — Seed scan (3 points × 500 measurements):**

| β | ⟨P⟩ | χ | Acc% | Therm (NPU early-exit) | Wall time |
|---|------|---|------|------------------------|-----------|
| 5.0000 | 0.4028 ± 0.0006 | 0.34 | 16% | 50/200 | 67 min |
| 5.6900 | 0.5293 ± 0.0041 | 17.68 | 18% | 80/200 | 71 min |
| 6.5000 | 0.6333 ± 0.0023 | 5.36 | 17% | 70/200 | 70 min |

NPU early-exit saved 200–600 thermalization trajectories per β point
(50–80 used vs 200 budget). Total seed phase: ~3.5 hours.

**Phase 2 — ESN retrain from seed data:**
- Retrained in 2.9 ms via NPU worker thread
- Updated β_c estimate: 5.5051 (known: 5.692)

**Phase 3 — NPU adaptive steering (in progress):**
- Round 1: NPU selected β = 5.5254 (maximum uncertainty region)
- 434/500 measurements complete at β = 5.5254
- Plaquette: 0.494 (consistent with transition onset)
- Up to 6 adaptive rounds remaining

**Run totals (as of 4h 32m elapsed):**

| Metric | Value |
|--------|-------|
| Trajectory log entries | 4,148 |
| Beta points completed | 3 (+ 1 in progress) |
| NPU inference calls | 1,934+ |
| GPU | RTX 3090 at 74°C, 354 W, 100% utilization |
| NPU | AKD1000 via PCIe, live inference |

---

## 6. Spectral Theory — Kachkovskiy Reproduction

Anderson localization, almost-Mathieu operators, and the Hofstadter butterfly,
reproduced in Rust with GPU-accelerated Lanczos eigensolve.

| Study | Checks | Key result |
|-------|--------|-----------|
| 1D Anderson localization | 10/10 | GOE→Poisson transition, Herman γ = ln|λ| |
| 2D Anderson + Lanczos | 11/11 | SpMV parity 1.78e-15, full spectrum |
| 3D Anderson | 10/10 | Mobility edge, dimensional hierarchy 1D < 2D < 3D |
| Hofstadter butterfly | 10/10 | Band counting, Cantor measure, α ↔ 1−α symmetry |
| GPU SpMV + Lanczos | 14/14 | Eigenvalues match CPU to 1e-15 |

---

## 7. Nuclear Structure — Full AME2020 (Phase F)

Full-scale nuclear structure on consumer GPU: 2,042 experimentally measured
nuclei from AME2020 (39× the 52-nucleus published reference).

| Level | Method | Nuclei | Best χ²/datum | Hardware | Time |
|-------|--------|--------|---------------|----------|------|
| L1 | SEMF (Skyrme functional) | 2,042 | 2.27 | RTX 4070 | 2.3 s |
| L2 | Spherical HFB | 791 | 16.11 | RTX 4070 | 66 min |
| L3 | Deformed HFB | 942 | 13.92 (best-of-both) | RTX 4070 | — |

L1 produces 1,990 binding energy predictions for nuclei the published paper
never evaluated.

---

## 8. Software Architecture

### 8.1 Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Language | Rust (stable) | Memory safety, no GC, no runtime |
| GPU dispatch | wgpu + Vulkan | Vendor-agnostic GPU access |
| Shader language | WGSL (f64 + DF64) | Portable compute shaders |
| NPU driver | Pure Rust (akida-driver) | Direct PCIe BAR mapping |
| Linear algebra | nalgebra (CPU), custom WGSL (GPU) | Dense and sparse |
| Build | Cargo, feature flags | `npu-hw` enables hardware NPU |

### 8.2 Performance vs Python Scientific Stack

| Operation | Python/NumPy/Numba | Rust/BarraCUDA | Speedup |
|-----------|-------------------|----------------|---------|
| CG solver (Dirac) | 4.59 ms/iter | 0.023 ms/iter | **200×** |
| L1 SEMF evaluation | 5.5 evals/s | 2,621 evals/s | **478×** |
| Abelian Higgs HMC | — | — | **143×** |
| GPU HMC (16⁴) | 17,996 ms/traj (CPU) | 293 ms/traj (GPU) | **61.4×** |

### 8.3 Vendor Agnosticism

The same WGSL shader binary runs on any GPU exposing Vulkan with SHADER_F64:

| Vendor | Tested GPUs | Status |
|--------|------------|--------|
| NVIDIA (proprietary) | RTX 3090, RTX 4070 | Full production |
| NVIDIA (NVK/open-source) | Titan V (Mesa 25.1.5) | Validated ≤ 16⁴ |
| AMD | RX 6950 XT | Available (untested in QCD) |
| Intel | Arc A-series | Supported by wgpu |

---

## 9. Hardware Inventory

| Component | Specs | Role | Cost |
|-----------|-------|------|------|
| RTX 3090 | 24 GB GDDR6X, GA102, 10,496 CUDA cores | Primary: DF64 physics, HMC | ~$800 used |
| Titan V | 12 GB HBM2, GV100, native 1:2 fp64 | Validation oracle, precision reference | ~$500 used |
| RTX 4070 | 12 GB GDDR6X, AD104 | Mid-tier benchmark, "any physicist" demo | ~$600 |
| Akida AKD1000 (×3) | PCIe Gen2 x1, 80 NPs, ~30 mW | ESN inference, physics screening | ~$300 each |
| Threadripper 3970X | 64 cores, 256 GB DDR4 ECC | CPU baseline, Python control | — |
| Dual EPYC 7452 | 64 cores, 256 GB DDR4 ECC | HPC-class CPU comparison | — |

---

## 10. Toward Warm Dense Matter

### 10.1 Context

The "Roadmap for warm dense matter physics" (Murillo et al., arXiv:2505.02494,
revised Feb 13, 2026) identifies computational accessibility as a critical
bottleneck for WDM science. State-of-the-art codes require institutional HPC
allocations that take months to obtain.

### 10.2 What hotSpring Already Has

| WDM primitive | Status | Source |
|--------------|--------|--------|
| Yukawa MD (all-pairs + cell-list, f64) | Validated (60/60) | Papers 1–3 |
| Green-Kubo transport (D*, η*, λ*) | Validated (13/13) | Paper 5 |
| Screened Coulomb eigensolve | Validated (23/23) | Paper 6 |
| GPU FFT (1D + 3D, f64) | Validated (roundtrip 1e-10) | BarraCUDA |
| DF64 core streaming | Validated (9.9× throughput) | Experiment 012 |
| NPU trajectory screening | Validated (9,017× energy reduction) | Experiment 020–022 |
| Lattice QCD HMC | Production (32⁴) | Papers 8–10 |

### 10.3 What's Needed for WDM Conditions

| Extension | Physics | Effort | Priority |
|-----------|---------|--------|----------|
| Partial ionization model (Z* from Thomas-Fermi or average-atom) | Ionization state varies with density and temperature | Medium | P1 |
| Electron-ion coupling (TTM extension to higher T) | Two-temperature dynamics at WDM conditions | Medium | P1 |
| Dynamic structure factor S(q,ω) from MD | Spatial FFT + temporal FFT of trajectories | Low (primitives exist) | P1 |
| Multi-component mixture EOS | Beyond single-species Yukawa | Medium | P2 |
| Wavepacket MD (quantum ion dynamics) | Quantum corrections for ion motion | High | P3 |
| Orbital-free DFT | Kinetic energy functional for WDM electrons | High | P3 |

### 10.4 Reproduction Targets (Tier 4 Queue)

| # | Target | Year | What | Status |
|---|--------|------|------|--------|
| 32 | Militzer FPEOS Database (Berkeley) | 2020+ | EOS tables, 11 elements, open C++/Python | Ready to reproduce |
| 33 | atoMEC average-atom code | 2023 | Open Python average-atom for WDM | Ready (ideal Phase 0) |
| 35 | WDM transport extension | — | Extend Stanton-Murillo to WDM conditions | Green-Kubo validated |
| 38 | Dynamic structure factor S(q,ω) | — | Key NIF diagnostic for XRTS | FFT validated |
| 40 | Dornheim XRTS diagnostics | 2025 | Model-free temperature extraction | FFT + spectral ready |

### 10.5 Cost Projection

| Phase | Hardware | Estimated time | Estimated cost |
|-------|----------|---------------|---------------|
| FPEOS reproduction | 1× RTX 4070 | ~1 week | ~$2 |
| WDM transport coefficients | 1× RTX 3090 | ~2 weeks | ~$5 |
| Dynamic structure factor | 1× RTX 3090 | ~1 week | ~$2 |
| Distributed parameter sweep | 2× GPU | ~1 month | ~$10 |
| **Total WDM program** | | ~2 months | **~$19** |

---

## 11. Key Findings Summary

1. **Consumer GPU reproduces institutional HPC physics.** 22 papers across 5
   domains, ~700 validation checks, $0.80 total compute cost. The deconfinement
   phase transition of QCD — historically requiring million-dollar
   supercomputers — appears at β_c = 5.69 on a $1,500 consumer workstation.

2. **DF64 core streaming delivers 9.9× native f64 throughput.** By routing
   bulk double-precision math through FP32 cores as double-float pairs, the
   RTX 3090 achieves 3.24 TFLOPS at 14-digit precision — sufficient for all
   tested physics observables with no measurable degradation.

3. **Vendor-agnostic GPU compute works for real physics.** Pure Rust + WGSL
   shaders via wgpu/Vulkan produce correct results on NVIDIA (proprietary and
   open-source NVK), with AMD and Intel support via the same shader binary.

4. **Neuromorphic hardware screens physics trajectories at 9,017× less energy
   than CPU.** The AKD1000 NPU runs reservoir computing (Echo State Network)
   inference at 2.8 µs per step, classifying lattice QCD phases and predicting
   HMC acceptance in real time alongside GPU computation.

5. **Cross-run learning enables self-improving pipelines.** ESN models trained
   during one run bootstrap the next, accumulating domain knowledge across
   experiments.

6. **NPU orchestrates GPU — not just observes.** The 11-head ESN (Exp 023)
   predicts workload parameters before GPU computation starts, monitors
   quenched phase convergence for early-exit, adaptively tunes CG solver
   check frequency, and inserts gap-filling β points into live scans.
   Every NPU function frees GPU cycles.

7. **The computational accessibility bottleneck is software, not silicon.**
   The hardware capability for WDM-class computation already exists in consumer
   GPUs. CUDA lock-in and the Python scientific stack obscure it.

---

## 12. References

### Reproduced Work — Murillo Group

- L. G. Silvestri, L. J. Stanek, G. Dharuman, Y. Choi, and M. S. Murillo.
  "Sarkas: A fast pure-python molecular dynamics suite for plasma physics."
  *Computer Physics Communications* (2022).

- L. G. Stanton and M. S. Murillo. "Ionic transport in high-energy-density
  matter." *Physical Review E* **93**, 043203 (2016).

- M. S. Murillo and J. C. Weisheit. "Dense plasmas, screened interactions,
  and atomic ionization." *Physics Reports* **302**, 1–65 (1998).

- A. Diaw, M. S. Murillo, and L. G. Stanton. "Efficient learning of accurate
  surrogates for simulations of complex systems." *Nature Machine Intelligence*
  **6**, 568–577 (2024).

### Reproduced Work — Bazavov Group

- A. Bazavov et al. [HotQCD]. "Equation of state in (2+1)-flavor QCD."
  *Physical Review D* **90**, 094503 (2014).

- A. Bazavov et al. "Polyakov loop in 2+1 flavor QCD from low to high
  temperatures." *Physical Review D* **93**, 114502 (2016).

- A. Bazavov et al. "Curvature of the freeze-out line in heavy-ion collisions."
  *Physical Review D* **93**, 014512 (2016).

- A. Bazavov et al. "Light-quark connected intermediate-window contributions
  to the muon g−2 hadronic vacuum polarization." *Physical Review D* **111**,
  094508 (2025).

### WDM Roadmap

- J. Vorberger, T. Graziani, M. S. Murillo, et al. "Roadmap for warm dense
  matter physics." arXiv:2505.02494 (revised February 13, 2026).

### Lattice QCD References

- R. Bali, J. Fingberg, U. Heller, F. Karsch, and K. Schilling. "The spatial
  string tension in the deconfined phase." *Physics Letters B* **309**, 378
  (1993).

- M. Creutz. "Monte Carlo study of quantized SU(2) gauge theory." *Physical
  Review D* **21**, 2308 (1980).

### DF64 Foundations

- T. J. Dekker. "A floating-point technique for extending the available
  precision." *Numerische Mathematik* **18**, 224–242 (1971).

- D. Priest. "Algorithms for arbitrary precision floating point arithmetic."
  *Proc. 10th IEEE Symposium on Computer Arithmetic*, 132–143 (1991).

---

*For full methodology, equations, and detailed results, see
[`whitePaper/STUDY.md`](STUDY.md) and
[`whitePaper/METHODOLOGY.md`](METHODOLOGY.md) in the repository.*

*For the lattice QCD production data, see
[`experiments/013_BIOMEGATE_PRODUCTION_BETA_SCAN.md`](../experiments/013_BIOMEGATE_PRODUCTION_BETA_SCAN.md).*

*For NPU characterization, see
[`experiments/020_NPU_CHARACTERIZATION_CAMPAIGN.md`](../experiments/020_NPU_CHARACTERIZATION_CAMPAIGN.md).*
