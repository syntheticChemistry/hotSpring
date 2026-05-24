# hotSpring Handoff — CAZyme FEL Biomolecular MD Evolution

**Date:** May 24, 2026
**From:** hotSpring
**To:** primalSpring (audit), barraCuda (shader evolution), ludoSpring/petalTongue (visualization), upstream ecosystem
**Experiment:** 220
**Status:** Phase 0 — GROMACS industry control established

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

| Phase | Scope | Target |
|-------|-------|--------|
| 0 (now) | GROMACS tutorial + GH10 reference FEL | Industry control |
| 1 (~1w) | barraCuda bonded FF shaders | 4 WGSL shaders |
| 2 (~2w) | hotSpring topology reader + MD loop | GROMOS 45a4/GLYCAM06 |
| 3 (~1w) | Metadynamics bias layer | Cremer-Pople CVs |
| 4 | Parity validation | barraCuda FEL ≈ GROMACS FEL |

---

## Key References

- Ardèvol & Rovira (2015) JACS — CAZyme catalytic itinerary (Fig. 10)
- Alonso-Gil (2019) thesis — QM/MM equations (Ch. 2.2–2.4)
- Wei-Tse Hsu — GROMACS enhanced sampling tutorials
- GROMOS 45a4 / GLYCAM06 force fields
- LAMMPS Colvars (Cremer-Pople CV implementation reference)

---

## New Gaps Registered

| Gap | Description | Severity |
|-----|-------------|----------|
| GAP-HS-111 | Bonded FF terms + topology reader + metadynamics | Medium |
| GAP-HS-112 | petalTongue FEL visualization evolution | Low |
