# hotSpring Handoff: CompChem Sovereign Evolution — Papers 50-58

**Date:** 2026-05-26
**From:** hotSpring (biomeGate)
**To:** primalSpring, toadStool, coralReef, barracuda upstream
**Context:** [Sovereign CompChem Evolution](63b41c45-5abe-4465-9529-b7820295f465)

---

## Summary

Formalized the sovereign all-atom MD roadmap as Papers 50-58 in the paper queue.
Decomposed full enzyme-scale (93K atom) MD into 7 independent GPU kernels, each
following the established Python Control → BarraCuda CPU → BarraCuda GPU → metalForge
validation path. Paper 50 (FES Gaussian summation) is GPU-PROVEN with 11-14x speedup
and machine-epsilon parity (RMSD 1e-14).

---

## What Was Done

### Specs Updated
- **NEW**: `specs/SOVEREIGN_COMPCHEM_EVOLUTION.md` — Full kernel decomposition (K1-K7),
  math baselines, parallelism profiles, existing infrastructure mapping, composition path,
  scale precedent, hardware dispatch plan, and priority-ordered next actions.
- **UPDATED**: `specs/PAPER_REVIEW_QUEUE.md` — Papers 50-58 added to pipeline table.
  Totals updated (28/34 CPU, 21/34 GPU). Level 1 and Level 2 tables include CompChem entries.
- **UPDATED**: `specs/README.md` — Scope extended with sovereign MD composition path,
  new spec registered in index, BIOMEGATE status corrected (DRAFT not Active).

### Root Docs Cleaned
- `README.md` — Quick Start section updated (fossilized scripts replaced with Rust
  entry points), experiment count → 222, shader count → 129, CompChem GPU parity
  added to domain status table, directory structure updated, "What gets regenerated"
  table modernized.

### Debris Cleaned
- 28 PLUMED backup files (`bck.*`) removed from `control/gromacs_fel/` and
  `control/plumed_nest/target_02_chignolin_opes/`
- Draft/status drift fixed in specs index

### Pipeline Status
- `nest-validate guidestone run` currently executing final system (`enzyme_bound_2d`,
  ~84% complete at time of handoff). Full pipeline: 4 GROMACS+PLUMED simulations →
  Rust FES reconstruction → Rust parity validation → GuideStone population.

---

## Kernel Decomposition (Papers 50-58)

| Paper | Kernel | Math | Status |
|-------|--------|------|--------|
| 50 | FES Gaussian summation | `F(x,y) = -Σ h·exp(-(x-x₀)²/2σ²)·exp(-(y-y₀)²/2σ²)` | **GPU PROVEN** (11-14×) |
| 51 | Bonded forces | harmonic bond + angle + periodic dihedral + improper + 1-4 | Next |
| 52 | LINCS constraints | `B·λ = (\|r'\| - d₀)`, iterative projection | Pending |
| 53 | PME electrostatics | B-spline spread + FFT3D + force gather | Pending (FFT exists) |
| 54 | Full integrator | Velocity Verlet + K1-K4 composition | Pending |
| 55 | Cremer-Pople CV | 6-atom puckering geometry | CPU validated |
| 56 | WTMetaD bias kernel | Adaptive Gaussian deposition | Pending |
| 57 | OPES reweighting | Histogram-based FES from biased trajectory | CPU validated |
| 58 | 93K-atom sovereign MD | Industry-parity with GROMACS | Long-term target |

---

## Upstream Gaps Identified

| Gap ID | Description | Owner | Priority |
|--------|-------------|-------|----------|
| GAP-HS-115 | Bonded force field WGSL shaders (Paper 51) — harmonic_bond, angle, dihedral, improper | barracuda | P1 |
| GAP-HS-116 | LINCS constraint solver WGSL shader (Paper 52) — parallel graph-colored projection | barracuda | P2 |
| GAP-HS-117 | PME B-spline charge spreading shader (Paper 53) — order-4 interpolation on mesh | barracuda + toadStool | P2 |
| GAP-HS-118 | V-rescale thermostat shader (Bussi-Donadio-Parrinello) — requires GPU PRNG | barracuda | P3 |
| GAP-HS-119 | LJ+Coulomb direct shader (Paper 1 ext) — swap Yukawa kernel to LJ 12-6 | barracuda | P1 |

---

## Hardware Dispatch Plan

Based on validated GPU profiling (bench_device_pair, validate_dual_gpu_qcd):

| Workload | Target Device | Rationale |
|----------|--------------|-----------|
| Non-bonded (K1) | AMD RX 6950 XT | Largest FLOP budget, AMD 4.5× faster |
| PME FFT (K4) | NVIDIA RTX 3090 | 24 GB VRAM, memory-bound |
| FES analysis (K7) | AMD RX 6950 XT | NVK zero-poison avoidance |
| Position sync | PCIe | 93K × 3 × f64 = 2.2 MB/step (~0.07 ms) |

---

## Scale Precedent

| Domain | Proven scale | Gap to 93K enzyme |
|--------|-------------|-------------------|
| Lattice QCD | 2,097,152 sites (64³×8) | 93K is 22× smaller |
| Yukawa MD | 20,000 particles | 93K is 4.5× larger |
| CompChem FES | 12,100 grid × 20K Gaussians | Analysis only (done) |

Gap is algorithmic (bonded + constraints + PME), not fundamental scale.

---

## For primalSpring Audit

1. **Paper queue evolution**: 34 papers tracked (was 25). CompChem domain fully integrated.
2. **Fossilized scripts**: 8 bash/Python scripts moved to `scripts/fossils/`. `FOSSILS.md` documents replacements.
3. **Known debris**: `scripts/archive/` contains 63 hardware experiment scripts (Python + bash) — parallel fossil layer, correctly archived but not registered in `FOSSILS.md` (intentional: hardware fossils vs CompChem fossils are separate lineages).
4. **Vendored PLUMED build**: `control/plumed_nest/build/plumed2-2.10.0/` is a vendored upstream build tree. Large but needed for PLUMED library compilation. Consider `.gitignore` if not already tracked.
5. **Single Rust TODO**: `control/plumed_nest/nest-validate/src/targets.rs:394` — placeholder for Rust-native ingestion (replaces `ingest.sh`). Known, not urgent.
6. **Draft status drift**: Fixed — `BIOMEGATE_BRAIN_ARCHITECTURE.md` corrected from "Active" to "Draft" in specs index.
