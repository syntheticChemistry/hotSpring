# Experiment 220 — CAZyme Conformational Energy Landscapes

**Date:** May 24, 2026
**Status:** Phase 0 (GROMACS control environment established)
**Hardware:** strandGate (RTX 3090), biomeGate (2× Titan V + RTX 5060)
**Collaborators:** Alistaire (domain expert — computational chemistry, CAZymes),
Mark (HPC access — Texas A&M ACES, A100s)

---

## Objective

Validate molecular docking as a computationally cheap proxy for conformational
energy landscapes (Free Energy Landscapes / FEL) in carbohydrate-active enzymes
(CAZymes). This is the hotSpring entry point for biomolecular MD — evolving
existing barraCuda primitives (LJ, Coulomb, PPPM, Verlet/cell lists, VV,
Nose-Hoover, Langevin) to support bonded force fields and metadynamics.

### Scientific Question

Do AutoDock Vina molecular docking results for monosaccharide ring conformations
in CAZyme active sites correlate with the conformational energy landscapes
obtained from more expensive QM/MM and metadynamics calculations?

### Reference

- Ardèvol & Rovira (2015) JACS 137(24):7528–47 (esp. Fig. 10)
  https://pubs.acs.org/doi/10.1021/jacs.5b01156
- Alonso-Gil (2019) thesis — Chapter 2.2–2.4 (equations of state, QM/MM)
- Wei-Tse Hsu — Enhanced sampling tutorials for GROMACS

---

## Architecture

### Spring Ownership

| Component | Owner | Role |
|-----------|-------|------|
| MD engine + bonded FF + metadynamics bias | **hotSpring** | Core compute |
| Visualization / UI | **ludoSpring** via **petalTongue** | FEL rendering, CV plots |
| GPU math primitives (WGSL shaders) | **barraCuda** | LJ, Coulomb, PPPM, bonded terms |
| Shader compilation (WGSL → native ISA) | **coralReef** | f64/DF64, SM35–SM120 |
| GPU dispatch (VFIO sovereign) | **toadStool** | Compute submission pipeline |

### Compute Tiers

| Tier | Hardware | Use |
|------|----------|-----|
| Local dev | strandGate RTX 3090 | GROMACS control, barraCuda dev |
| biomeGate | 2× Titan V + RTX 5060 | Production sovereign dispatch |
| HPC | Texas A&M ACES (A100) | Scale validation via Mark's NSF allocation |

---

## Phases

### Phase 0 — GROMACS Industry Control (current)

**Goal:** Get GROMACS + PLUMED running with a reference metadynamics FEL for
one GH family representative. This is the validation target.

**Environment:**
- `conda activate gromacs-fel`
- GROMACS 2026.0 (CUDA, mixed precision, Colvars, PLUMED built-in)
- NVIDIA CUDA 12.9

**Steps:**
1. ✅ GROMACS 2026.0 installed (`conda create -n gromacs-fel`)
2. ✅ GPU support confirmed (CUDA, SM86 RTX 3090)
3. ✅ PLUMED + Colvars built-in confirmed
4. Run Wei-Tse Hsu enhanced sampling tutorial (alanine dipeptide metadynamics)
5. Set up GH10 system: substrate in active site, GROMOS 45a4 or GLYCAM06 FF
6. Run metadynamics: Cremer-Pople θ,φ as collective variables
7. Generate reference FEL (free energy surface)

**Validation criteria:**
- Metadynamics converges (hills deposited, FEL fills)
- Cremer-Pople CV space sampled (chair/boat/skew-boat)
- Energy barriers between conformations match literature order of magnitude

### Phase 1 — barraCuda Bonded Force Field Shaders (~1 week)

**Gap:** barraCuda has full nonbonded MD (LJ, Coulomb, PPPM, cell/Verlet lists)
but lacks bonded force field terms needed for carbohydrate simulations.

**New WGSL shaders needed:**
1. `harmonic_bond.wgsl` — V(r) = ½k(r - r₀)²
2. `harmonic_angle.wgsl` — V(θ) = ½k(θ - θ₀)²
3. `dihedral_torsion.wgsl` — V(φ) = Σ kₙ(1 + cos(nφ - δₙ))
4. `improper_dihedral.wgsl` — V(ψ) = ½k(ψ - ψ₀)²

**Existing primitives (ready):**
- LJ 6-12 with cell list neighbor finding
- Coulomb with PPPM long-range
- Velocity Verlet integrator
- Nose-Hoover / Langevin / Berendsen thermostats
- PBC (cubic, orthorhombic)
- RDF, MSD, VACF, SSF observables
- f64 and DF64 precision paths

### Phase 2 — hotSpring MD Engine (~1–2 weeks)

**Build unified MD loop in hotSpring that composes barraCuda primitives:**
1. Topology reader (GROMOS 45a4 / GLYCAM06 force field parameter files)
2. Coordinate reader (GRO/PDB format)
3. MD loop: nonbonded + bonded forces → VV integration → thermostat → PBC
4. Trajectory output (XTC or simple binary for FEL analysis)
5. Cross-tier parity: GROMACS single-point energy vs hotSpring single-point

### Phase 3 — Metadynamics Bias Layer (~1 week)

**Implement Cremer-Pople collective variable + Gaussian hill biasing:**
1. Cremer-Pople CV computation (θ, φ from ring atom coordinates)
2. Gaussian hill deposition (height, width, deposition interval)
3. Bias force computation and addition to atomic forces
4. Well-tempered metadynamics variant (hill height scaling)
5. FEL reconstruction from deposited hills

### Phase 4 — Validation Against GROMACS

**Parity checks:**
1. Single-point energy comparison (same coordinates, same FF)
2. Short NVT trajectory energy conservation comparison
3. FEL topology comparison (minima locations, barrier heights)
4. Cremer-Pople CV sampling coverage comparison

---

## Primal Readiness Assessment

### barraCuda — Existing MD Primitives

| Primitive | Status | Notes |
|-----------|--------|-------|
| LJ 6-12 | ✅ Ready | `lj_force_f64.wgsl`, cell list |
| Coulomb | ✅ Ready | Direct + PPPM long-range |
| Velocity Verlet | ✅ Ready | `velocity_verlet_f64.wgsl` |
| Nose-Hoover | ✅ Ready | `nose_hoover_f64.wgsl` |
| Langevin | ✅ Ready | `langevin_f64.wgsl` |
| Berendsen | ✅ Ready | `berendsen_f64.wgsl` |
| Cell list | ✅ Ready | `cell_list_f64.wgsl` |
| Verlet list | ✅ Ready | Runtime-adaptive selection |
| PBC | ✅ Ready | Cubic, orthorhombic |
| RDF | ✅ Ready | Observable pipeline |
| VACF/MSD/SSF | ✅ Ready | Full observable suite |
| Harmonic bond | ❌ Missing | Phase 1 |
| Harmonic angle | ❌ Missing | Phase 1 |
| Dihedral torsion | ❌ Missing | Phase 1 |
| Improper dihedral | ❌ Missing | Phase 1 |
| Topology/FF reader | ❌ Missing | Phase 2 (hotSpring) |
| Metadynamics bias | ❌ Missing | Phase 3 (hotSpring) |

### coralReef — No Gaps

Full f64/DF64 compilation to SM35–SM120 (NVIDIA) and GCN5/RDNA (AMD). All
shader features needed (workgroup barriers, u32 atomics, subgroup ops).

### toadStool — Adequate for Prototyping

Per-RPC alloc/free overhead acceptable for iterative development. Sustained MD
production would benefit from persistent VRAM handles (toadStool evolution
request, not blocking).

---

## Relationship to Existing Work

This experiment extends hotSpring's validated MD stack (Experiments 001–219)
from plasma physics (Yukawa/Coulomb particle systems) into biomolecular
territory (covalent bonds, ring puckering, enzyme active sites). The sovereign
GPU pipeline (Exp 162–219) provides the dispatch infrastructure. The existing
219 experiments prove the math is sound — this adds the chemistry layer.

---

## Downstream: helixVision Feed

helixVision (gen4 sporeGarden product, planned) composes neuralSpring/coralForge
(AlphaFold2/3 structure prediction, 154 checks) + wetSpring (16S/WGS) +
petalTongue (viz) into a sovereign genomics discovery platform. This CAZyme
work builds the **validation layer helixVision needs to trust its predictions.**

### What flows downstream

| Artifact | Source | helixVision use |
|----------|--------|----------------|
| Bonded FF shaders | barraCuda (Phase 1) | Structure relaxation, energy minimization |
| Topology reader | hotSpring (Phase 2) | Biomolecular system setup for any FF |
| Metadynamics bias | hotSpring (Phase 3) | Conformational exploration, structure validation |
| Cremer-Pople CV | hotSpring (Phase 3) | Ring puckering quality metric |
| GROMACS parity proof | hotSpring (Phase 4) | Trusted MD = trusted structure validation |
| coralForge/IPA | neuralSpring | Predicts structures that MD then validates |
| FEL viz (Stoddart heatmaps, 3D) | petalTongue | Interactive rendering of energy landscapes |
| ESN surrogates | neuralSpring | ML FEL prediction from docking scores (future) |

### The predict→validate→confirm→visualize loop

```
helixVision predicts structure (coralForge/AlphaFold)
     ↓
hotSpring validates via MD (bonded FF + metadynamics on sovereign GPU)
     ↓
GROMACS confirms hotSpring MD is correct (industry parity anchor)
     ↓
petalTongue renders FEL + 3D structure (ludoSpring)
```

### neuralSpring reusable systems

- **coralForge** Evoformer + IPA + diffusion — structure prediction core
- **ESN reservoir** — candidate ML surrogate for FEL-from-docking
- **petalTongue push client** — data channel pattern for viz scenarios
- **Heatmap + Scatter3D channels** — Stoddart diagram + CV trajectory overlays
- **FASTA/FASTQ streaming** — sequence ingestion for wetSpring pipeline
- **Squirrel inference routing** — trained FEL surrogate model serving

### petalTongue molecular viz spec (designed, not built)

petalTongue grammar-of-graphics spec includes molecular visualization:
PDB/mmCIF atom coordinates → GeomSphere (atoms) + GeomCylinder (bonds) →
Perspective3D with orbit camera → optional electron density isosurfaces.
CAZyme FEL is the first concrete customer for this spec.

---

## Scientific Question (Detailed)

**Core hypothesis:** AutoDock Vina docking scores for monosaccharide ring
conformations in CAZyme active sites correlate with conformational energy
landscapes from metadynamics.

**Why it matters:** FEL takes 12–48 hours/system on GPU HPC. Docking takes
~5 minutes on a laptop. If Spearman ρ holds, docking becomes a practical
screening tool for catalytically competent sugar geometries.

**Test design:**
1. Generate all 38 Stoddart diagram conformations (chairs, boats, skew-boats,
   half-chairs, envelopes) via CP_reconstruct.
2. Dock each into GH active site → rank by Vina score.
3. Compute reference FEL via GROMACS+PLUMED (Cremer-Pople θ,φ CVs).
4. Compare: do catalytic itinerary conformations (e.g. retaining GH10:
   ¹S₃ → [⁴H₃]‡ → ⁴C₁) score highest AND sit at FEL minima?
5. Quantify with Spearman ρ between FEL depth and Vina score per conformation.

**Scoping:** MD/MM sufficient for pre/post-TS conformations. QM/MM only needed
for transition-state characterization (not in scope for this experiment).

**Pilot system:** GH10 β-xylanase (PDB 1E0X), then survey 5–8 GH families
(retaining vs inverting mechanisms).

**Reference:** Ardèvol & Rovira (2015) JACS 137(24):7528–47, esp. Fig. 10.

---

## Notes

- Alistaire is domain expert for CAZyme biochemistry, QM/MM, and metadynamics.
  GROMACS validation workflow follows his guidance.
- Mark has NSF HPC access (A100 GPUs) for scale-up when local validation
  completes.
- Visualization (FEL surfaces, CV trajectories, Stoddart diagram heatmaps,
  3D molecular rendering) → ludoSpring via petalTongue. petalTongue should
  evolve to manage FEL rendering and interactive CV exploration UI.
- GROMACS is the industry control/validation target, not a dependency. The goal
  is to prove barraCuda+hotSpring matches GROMACS results, then exceed on
  consumer hardware via sovereign GPU dispatch.
- GROMACS provides the full parity anchor — its own visualization/analysis
  tools (gmx energy, gmx rdf, gmx msd, PLUMED sum_hills) produce reference
  data that hotSpring must match before claiming parity.
