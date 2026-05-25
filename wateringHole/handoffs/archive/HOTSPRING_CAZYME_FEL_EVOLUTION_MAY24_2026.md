# hotSpring Handoff — CAZyme FEL Biomolecular MD Evolution

**Date:** May 24, 2026
**From:** hotSpring
**To:** primalSpring (audit), barraCuda (shader evolution), ludoSpring/petalTongue (visualization), upstream ecosystem
**Experiment:** 220
**Status:** Phase 0.6 — Tier 0/1/2 parity MATCH; pseudoSpore v0.6.0 shipped; lithoSpore promotion staged

---

## Summary

hotSpring is evolving into biomolecular MD for a CAZyme conformational energy
landscape (FEL) project. This extends the validated plasma MD stack (Exp 001–219,
LJ/Coulomb/PPPM + sovereign GPU dispatch) to support bonded force fields,
carbohydrate ring puckering, and metadynamics enhanced sampling.

**Collaborators:**
- Alistaire — domain expert (CAZyme biochemistry, QM/MM, metadynamics)
- Mark — NSF HPC access (Texas A&M ACES, A100 GPUs)

**Scientific goal:** Validate AutoDock Vina molecular docking as a cheap proxy
for conformational energy landscapes in carbohydrate-active enzymes.

---

## What Was Done

1. **GROMACS 2026.0** installed on strandGate via conda (`gromacs-fel` env)
   - CUDA 12.9, mixed precision, PLUMED + Colvars built-in
   - GPU-accelerated (RTX 3090 detected)
   - Industry control/validation target for barraCuda parity

2. **Primal profiling completed** (barraCuda, coralReef, toadStool)
   - barraCuda: 71 MD-adjacent shaders, full nonbonded stack ready
   - coralReef: Full f64/DF64 compilation, all GPU targets — no gaps
   - toadStool: Sovereign VFIO dispatch adequate for prototyping

3. **Experiment 220 created** with 4-phase plan
4. **GAP-HS-111** (bonded FF + topology) and **GAP-HS-112** (petalTongue FEL viz) registered

---

## What's Needed from Ecosystem

### barraCuda (primal evolution request)

4 new WGSL shaders for bonded force field terms:
- `harmonic_bond.wgsl` — V(r) = ½k(r - r₀)²
- `harmonic_angle.wgsl` — V(θ) = ½k(θ - θ₀)²
- `dihedral_torsion.wgsl` — V(φ) = Σ kₙ(1 + cos(nφ - δₙ))
- `improper_dihedral.wgsl` — V(ψ) = ½k(ψ - ψ₀)²

These follow established barraCuda patterns (f64 + DF64 paths, cell list
neighbor finding). The existing LJ/Coulomb/PPPM + Verlet/cell list + VV +
thermostat stack is ready.

### ludoSpring / petalTongue (visualization evolution)

FEL visualization shared with ludoSpring. petalTongue should evolve to handle:
- 2D heatmap / 3D surface rendering over Cremer-Pople θ,φ space
- CV trajectory overlay on FEL surface
- Interactive ring puckering visualization
- Convergence diagnostics (hill height vs time, FES evolution)

### toadStool (future — not blocking)

For sustained production MD, persistent VRAM handles and fused buffer submission
would reduce per-RPC overhead. Current alloc/free per dispatch is fine for
prototyping and validation.

---

## Compute Tier Strategy

| Tier | Hardware | Role |
|------|----------|------|
| Local dev | strandGate RTX 3090 | GROMACS control + barraCuda dev |
| biomeGate | 2× Titan V + RTX 5060 | Sovereign dispatch production |
| HPC | ACES A100 (NSF) | Scale validation with Alistaire |

---

## Phase Plan

| Phase | Scope | Status |
|-------|-------|--------|
| 0.4 | Alanine dipeptide WTMetaD (benchmark) | ✅ Complete |
| 0.5 | Free xylose puckering FEL | ✅ Complete |
| 0.6 | pseudoSpore handoff + Tier 1/2 parity | ✅ Complete (this update) |
| 0.7 | Enzyme-bound FEL (after Alistaire convention confirmation) | Pending |
| 1 | barraCuda bonded FF shaders | 4 WGSL shaders (GAP-HS-111) |
| 2 | hotSpring topology reader + MD loop | CHARMM36 |
| 3 | Metadynamics bias layer | Cremer-Pople CVs |
| 4 | Parity validation | barraCuda FEL ≈ GROMACS FEL |

---

## Sovereign FEL Parity Results (May 24, 2026)

Three-tier sovereign reconstruction of free energy landscapes from GROMACS+PLUMED HILLS files:

| Comparison | Module 1 (Ala dipeptide 2D φ/ψ) | Module 2 (Xylose 1D θ) | Tolerance |
|------------|--------------------------------|------------------------|-----------|
| Tier 0 (GROMACS) → Tier 1 (Python) | 0.52 kJ/mol | 0.83 kJ/mol | 2.0 kJ/mol |
| Tier 0 (GROMACS) → Tier 2 (Rust) | 0.52 kJ/mol | 0.75 kJ/mol | 2.0 kJ/mol |
| Tier 1 (Python) → Tier 2 (Rust) | 0.00 kJ/mol | 0.00 kJ/mol | — |
| **Verdict** | **MATCH** | **MATCH** | — |

**Implementation:**
- Tier 1: `notebooks/cazyme_fel/puckering_fel.py` — Python sum_hills with periodic CV handling
- Tier 2: `staging/cazyme-fel/src/lib.rs` — Rust sum_hills with linear interpolation for grid alignment

---

## pseudoSpore v0.6.0 Handoff

Packaged artifact (`~/Desktop/pseudoSpore_cazyme_fel_v0.6.0.tar.gz`) for ABG domain expert review:
- 3 modules: alanine dipeptide (benchmark) → free xylose (substrate) → enzyme-bound (IN_FLIGHT)
- Machine-readable `validation.json` with inline errata
- Provenance trio: live sweetGrass braid + pseudo rhizoCrypt DAG + pseudo loamSpine ledger
- Honest audit (`AUDIT.md`) with 4 findings, all flagged

**Blocking item for Alistaire:** Cremer-Pople ring atom ordering convention (Finding 3 — 4C1 vs 1C4 label).

---

## lithoSpore Promotion

FermentBraid format alignment completed (`provenance/braids/hotspring_cazyme_fel.json`). Full promotion plan: `docs/LITHOSPORE_PROMOTION.md`. The `staging/cazyme-fel/` crate is the first non-LTEE module ready for lithoSpore chassis integration.

---

## Key References

- Iglesias-Fernández et al. 2015 — GH10 xylanase conformational FEL (PDB 2D24)
- Ardèvol & Rovira (2015) JACS — CAZyme catalytic itinerary (Fig. 10)
- Alonso-Gil (2019) thesis — QM/MM equations (Ch. 2.2–2.4)
- Wei-Tse Hsu — GROMACS enhanced sampling tutorials
- CHARMM36 force field (xylose parameters)
- LAMMPS Colvars (Cremer-Pople CV implementation reference)

---

## New Gaps Registered

| Gap | Description | Severity |
|-----|-------------|----------|
| GAP-HS-111 | Bonded FF terms + topology reader + metadynamics (4 WGSL shaders + CHARMM36 reader + Cremer-Pople CV + bias engine) | Medium |
| GAP-HS-112 | petalTongue FEL visualization evolution (2D/3D surface over Cremer-Pople coords) | Low |
