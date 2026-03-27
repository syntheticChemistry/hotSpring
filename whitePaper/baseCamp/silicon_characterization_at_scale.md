# Silicon Characterization at Scale — From Consumer to CERN

**Domain:** GPU silicon utilization for computational physics
**Updated:** March 26, 2026
**Status:** Exp 097-100 complete — full 4-phase characterization on 2 consumer GPUs
**Hardware:** RTX 3090 (Ampere), RX 6950 XT (RDNA2); incoming: 2x Titan V, 2x MI50, Tesla P80
**Protocol:** `wateringHole/GPU_FIXED_FUNCTION_SCIENCE_REPURPOSING.md`

---

## Thesis

HPC lattice QCD codes (QUDA, MILC, openQCD) optimize for one dimension of
GPU silicon: FP64 ALU throughput and HBM bandwidth. On an NVIDIA A100, this
means ~5 TFLOPS sustained out of ~830 TFLOPS total capacity — 0.6% silicon
utilization. On an H100, it drops to 0.4%.

The silicon characterization pipeline (budget → saturation → composition →
QCD kernel profiling) reveals that multiple silicon units can operate in
parallel on different sub-problems. Consumer GPU measurements prove this:
NVIDIA achieves 2.71x ALU+TMU composition, AMD achieves 1.65x ALU+BW
overlap and 202% CG pipeline efficiency. These compound effects are
invisible to codes that only use one functional unit.

QCD is the ideal workload validator because it exercises every type of
compute: dense algebra (force), stencil access (Dirac), tree reduction
(CG dot product), transcendental math (PRNG), serial chains (Polyakov
loop), and bandwidth-limited streaming (momentum update). A single HMC
trajectory touches every silicon unit if the routing layer knows how to
map kernels to hardware.

---

## Pipeline

```
Phase 1: bench_silicon_budget
  Theoretical peak per silicon unit from vendor specs + measured benchmarks.
  Computes: FP32/DF64/FP64 TFLOPS, tensor TFLOPS, BW, TMU/ROP rates,
  QCD working-set vs cache hierarchy, precision tier throughput.

Phase 2: bench_silicon_saturation
  Micro-experiments measuring actual peak of each unit in isolation.
  FMA chain (ALU), sequential read (BW), working-set sweep (cache),
  textureLoad flood (TMU), workgroup tree reduce (LDS), atomicAdd (global).

Phase 3: bench_qcd_silicon
  14 QCD proxy kernels × FP32/DF64 × 5 lattice sizes (4^4 through 32^4).
  Produces GFLOP/s, GB/s, sites/s per kernel per GPU per volume.
  Silicon opportunity analysis maps each kernel to its bottleneck unit.
  HMC trajectory cost model estimates wall time at measured rates.

Phase 4: bench_silicon_composition
  Multi-unit parallel experiments measuring compound throughput:
  ALU+TMU, ALU+BW, ALU+Reduce (CG pattern). Quantifies the composition
  multiplier — how much extra throughput parallel silicon units provide.
```

---

## Key Findings (Consumer Fleet)

### Card Personalities

**RTX 3090** — The TMU compositor. Best ALU+TMU composition (2.71x), highest
PRNG throughput (5,853 GFLOP/s), superior FP32 raw TFLOPS (35.6 vs 23.65).
Wins on reduction-style kernels (shared memory bandwidth). Natural role:
observation kernels, PRNG heat bath, tensor core candidate (via coralReef SASS).

**RX 6950 XT** — The cache-and-precision champion. 128 MB Infinity Cache
eliminates bandwidth cliff through 16^4 working sets. 6x faster atomics
(93.6 vs 15.6 Gatom/s). 1.82x DF64 advantage. 202% CG pipeline efficiency.
Dominates 11/14 QCD kernels at 32^4. Natural role: Dirac-heavy CG,
force evaluation, DF64 precision-critical accumulations.

### HPC Silicon Waste

| Card | Total compute | QUDA-style sustained | Utilization |
|------|--------------|---------------------|-------------|
| A100 SXM | ~830 TFLOPS | ~5 TFLOPS (FP64) | 0.6% |
| H100 SXM | ~3,600 TFLOPS | ~15 TFLOPS (FP64) | 0.4% |
| MI250X | ~527 TFLOPS | ~25 TFLOPS (FP64) | 4.7% |
| RTX 3090 | ~284 TFLOPS | ~2 TFLOPS (DF64 QCD) | 0.7% |
| RX 6950 XT | ~47 TFLOPS | ~3.5 TFLOPS (DF64 QCD) | 7.4% |

AMD's higher utilization reflects: no tensor cores to waste, 1:16 FP64
providing more native double throughput, and Infinity Cache keeping data
near ALUs. NVIDIA's waste is dominated by 142-990 TFLOPS of tensor capacity
that WGSL/CUDA QCD cannot access without sovereign ISA emission.

### Unlockable Capacity on HPC Cards

| Technique | A100 impact | H100 impact | Requires |
|-----------|------------|------------|----------|
| FP64 Tensor Core (DMMA) | 2x force eval | 2x force eval | coralReef SASS/PTX |
| TF32 inner MD steps | 15+ TFLOPS | 30+ TFLOPS | Precision routing |
| TMU PRNG composition | ALU freed for physics | ALU freed for physics | textureLoad in WGSL |
| Multi-precision HMC routing | 4-6x compound | 4-6x compound | PrecisionBrain |
| Cache-aware kernel tiling | L2-resident stencils | L2-resident stencils | Cache probe data |

---

## Hardware Strategy: Decommissioned HBM2 Fleet

Modern data center GPU silicon — HBM2 bandwidth, full-rate FP64, tensor
cores — is available in decommissioned cards at 1/100th the cost of new
A100s:

| Card | FP64 | HBM2 BW | VRAM | Tensor | Street price |
|------|------|---------|------|--------|-------------|
| Titan V (GV100) | 7.45 TFLOPS (1:2) | 653 GB/s | 12 GB | 110 TFLOPS FP16 | ~$300 |
| MI50 (Vega 20) | 6.7 TFLOPS (1:2) | 1,024 GB/s | 16 GB | None | ~$150 |
| Tesla P80 (GP100) | 4.7 TFLOPS (1:2) | 732 GB/s | 16 GB | None | ~$200 |
| V100 (GV100) | 7.8 TFLOPS (1:2) | 900 GB/s | 16-32 GB | 125 TFLOPS FP16 | ~$250 |

Fleet of 6 HBM2 cards (2x Titan V + 2x MI50 + Tesla P80 + 1 more): total
~37 TFLOPS FP64 native, ~4.5 TB/s HBM2 bandwidth, ~88 GB VRAM. For
comparison: one A100 SXM = 9.7 TFLOPS FP64, 2 TB/s, 80 GB.

The decommissioned fleet has 3.8x the FP64 TFLOPS and 2.3x the bandwidth
of a single A100 — at 1/30th the cost. What it lacks is the 40 MB L2 and
tensor core capacity. The silicon characterization pipeline reveals exactly
which workloads benefit from which advantage.

---

## Science Path

### Prove locally: every piece of silicon contributes

The 4-phase pipeline on consumer GPUs is complete. Extending to the HBM2
fleet (Titan V with native 1:2 FP64, MI50 with 1 TB/s HBM2, Tesla P80)
will demonstrate that full-rate FP64 + HBM2 bandwidth changes the silicon
utilization picture. On Titan V, native FP64 at 7.45 TFLOPS eliminates the
need for DF64 entirely — the FP32 lanes become available for mixed-precision
inner MD steps.

### Publish methodology: silicon characterization as a tool

The pipeline itself (budget → saturation → composition → QCD kernel profile)
is generalizable. Any GPU, any workload. The bench_silicon_budget binary
already handles 10+ GPU models. Adding HPC reference specs (A100, H100,
MI250X) as comparison columns produces a paper-ready characterization.

### Collaborate: someone at CERN runs it

The sovereign stack means the same binary runs identically on any hardware.
A collaborator (Bazavov, Chuna, or HotQCD/MILC member) with A100/H100
access runs:

```
cargo run --release --bin bench_silicon_budget
cargo run --release --bin bench_silicon_saturation
cargo run --release --bin bench_qcd_silicon
cargo run --release --bin bench_silicon_composition
```

The output is a performance surface that toadStool consumes via
`compute.performance_surface.report`. The results quantify the compound
effect on HPC hardware and demonstrate how much science is being left on
the table by FP64-only codes.

---

## Scaling Path: Meeting & Exceeding Chuna's MILC-Scale Work

### Chuna's Reference Scale

Bazavov & Chuna validated LSCFRK integrators on MILC collaboration configs:

| Parameter | MILC/Chuna scale | Our current scale | Gap |
|-----------|-----------------|-------------------|-----|
| Lattice | 16³×48 to 64³×128 | 4⁴ to 16⁴ | 4-500x volume |
| Fermion content | Nf=2+1+1 HISQ | Nf=2+1 staggered | Action improvement |
| Quark masses | Physical (m_π ≈ 135 MeV) | m=0.05-0.5 (unphysical) | Tuning |
| Flow observables | t₀, w₀ (continuum extrapolation) | t₀, w₀ (single lattice spacing) | Multi-β |
| Hardware | USQCD clusters (100s of A100-hours) | RTX 3090 (minutes-hours) | 1000x raw FLOPS |

### How We Close the Gap

The gap is NOT primarily in algorithm quality — our integrators, RHMC,
CG solvers, and gradient flow are the same mathematics. The gap is in
volume and action sophistication. Our silicon utilization advantage
shrinks the effective hardware gap significantly:

| Strategy | Volume gain | When |
|----------|-----------|------|
| **GPU-accelerated HMC** (existing) | 16⁴ feasible now | Exp 102 (this week) |
| **Overnight 32⁴ runs** (existing infrastructure) | 1M sites, production quality | Next week |
| **Multi-GPU via toadStool brain** | 2-4x per card pair | HBM2 fleet arrival (weeks) |
| **HBM2 fleet (6 cards, ~37 TFLOPS FP64)** | 3.8x single A100 in FP64 | Fleet assembly (weeks) |
| **Full silicon routing** (TMU+Tensor+ALU) | 4-10x effective throughput | Ongoing evolution |
| **coralReef SASS emission** (FP64 tensor cores) | 2x force on Titan V | Medium-term |
| **Collaborator runs on HPC** (same binary) | 100x+ raw compute | Chuna meeting (late April) |

### Volume Roadmap

| Target | Sites | GPU-hours (estimated) | Physics deliverable |
|--------|:-----:|:---------------------:|---------------------|
| ✅ 8⁴ Nf=2+1 RHMC | 4K | 0.3h | Fermion content validation |
| ✅ 8⁴ flow convergence | 4K | 0.5h | CK4 stability proof |
| 🔄 16⁴ quenched flow | 65K | ~2h | t₀/w₀ scale setting |
| 16⁴ Nf=2+1 RHMC + flow | 65K | ~6h | Dynamical t₀/w₀ |
| 32⁴ quenched flow (multi-β) | 1M | ~24h | Continuum approach |
| 32⁴ Nf=2+1 RHMC | 1M | ~48h | Production physics |
| 32³×8 finite-T Nf=2+1 | 262K | ~12h | Crossover temperature |
| Multi-GPU 48⁴ | 5.3M | days (fleet) | Approaching MILC volumes |
| HPC run 64³×128 | 33M | collaborator allocation | MILC-equivalent physics |

### What "Exceeding" Chuna's Scale Means

We don't need to run bigger lattices than MILC to exceed the scientific
contribution. The leverage is:

1. **Same physics, 100-1000x less hardware cost**: Demonstrating that a
   $1,500 consumer GPU (or a $1,000 decommissioned HBM2 fleet) produces
   publication-quality flow scales changes who can do lattice QCD.

2. **Full silicon utilization measurement**: No HPC code has measured
   compound silicon composition. Our pipeline quantifies the gap between
   what hardware provides and what codes use. Running bench_silicon_budget
   on an A100/H100 produces data that doesn't exist in the literature.

3. **Novel precision routing**: Mixed-precision HMC (TF32 inner steps,
   DF64 accumulation, FP64 Metropolis) is not implemented in MILC or
   QUDA. Our PrecisionBrain architecture can route sub-problems to their
   optimal silicon unit — something HPC codes cannot do because they
   target one precision tier.

4. **Portable sovereign stack**: The same `cargo run` works on consumer
   GPUs, HBM2 fleet, and HPC nodes. No CUDA lock-in, no MPI
   configuration hell, no institutional software dependencies.

---

## Cross-References

- **Experiments:** 096-100
- **Binaries:** `bench_silicon_budget`, `bench_silicon_saturation`, `bench_qcd_silicon`, `bench_silicon_composition`
- **BaseCamp (prior):** `silicon_science.md` (Exp 096, TMU/DF64 initial characterization)
- **toadStool surface:** `compute.performance_surface.report`, `compute.route.multi_unit`
- **Sovereign goal:** `SOVEREIGN_VALIDATION_GOAL.md` (Home → Lab → Cluster → CERN)
- **Hardware inventory:** `metalForge/gpu/nvidia/HARDWARE.md`
- **RHMC infrastructure:** Exp 099, `barracuda/src/lattice/gpu_hmc/gpu_rhmc.rs`
