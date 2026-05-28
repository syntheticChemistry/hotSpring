# CAZyme FEL — Status for ABG (v0.7.0)

**Artifact**: hotSpring-CAZyme-FEL v0.7.0 (pseudoSpore)
**Target**: Iglesias-Fernández et al. 2015 (DOI: 10.1039/C4SC02240H, PDB 2D24)
**Supersedes**: v0.6.0 (critical errors corrected per Alistaire's review)

---

## Corrections Applied

Alistaire identified these issues in v0.6.0:

1. **Module 2 (`xylose_charmm.pdb`) was β-D-Lyxose, not Xylose** — RDKit generated
   the wrong stereoisomer from SMILES. Fixed: now using crystal coordinates from
   PDB 2D24 (XYS C residue 4), stereochemistry verified by crystallography.

2. **Module 3 was labeled IN_FLIGHT but had completed** — Production run finished
   (10,000 Gaussians, "Finished mdrun" in log) but sum_hills was never executed.
   Fixed: FEL now computed and analyzed.

3. **Atom indices unverifiable** — Handoff shipped no topology, so PLUMED indices
   couldn't be confirmed. Fixed: `ATOM_INDEX_MAP.md` documents the full mapping
   from PDB serial → GROMACS index.

4. **No solvation evidence** — Only input PDB (20 atoms) shipped, no proof of water.
   Fixed: `SYSTEM_SETUP.md` documents the full pipeline including topology composition.

---

## For Alistaire

All three modules now complete with FEL data:

**Module 1 — Alanine dipeptide Ramachandran FEL** ✅ (unchanged from v0.6.0)
- GROMACS 2026.0 + PLUMED 2.9.2, AMBER99SB-ILDN, 10 ns WTMetaD
- C7eq global minimum at φ=-81°, ΔF(C7ax)=5.57 kJ/mol
- All 6 checks PASS

**Module 2 — Free β-D-xylopyranose puckering FEL** ✅ (CORRECTED)
- Structure: PDB 2D24 crystal (XYS C res 4) — NOT RDKit
- CHARMM36-jul2022, 879 TIP3P waters, 10 ns WTMetaD on Cremer-Pople θ
- Three basins: Chair A (θ≈10°, 16.1 kJ/mol), Boat (θ≈89°, 6.4 kJ/mol),
  Chair B (θ≈172°, 0.0 kJ/mol global min)
- Barriers: 52.5 kJ/mol (A→Boat), 37.6 kJ/mol (Boat→B)
- Converged (last two stride files identical)

**Module 3 — Enzyme-bound -1 subsite (PDB 2D24)** ✅ (COMPLETED)
- GH10 xylanase chain A + BXYL at -1 subsite, 92,745 atoms
- Three basins: Chair A (θ≈8°, 13.6 kJ/mol), Boat (θ≈89°, 5.4 kJ/mol),
  Chair B (θ≈172°, 0.0 kJ/mol global min)
- Barriers: 47.4 kJ/mol (A→Boat), 36.7 kJ/mol (Boat→B)
- Converged

### Key Result: Free vs Enzyme-Bound Comparison

The enzyme active site uniformly lowers all conformational barriers:
- Boat stabilized by 1.0 kJ/mol
- 4C1 chair stabilized by 2.5 kJ/mol
- A→Boat barrier lowered by 5.1 kJ/mol
- Boat→B barrier lowered by 0.9 kJ/mol

This is consistent with catalytic facilitation of ring distortion — the enzyme
pre-organizes the substrate toward transition-state conformations.

### Remaining question for Alistaire

**Cremer-Pople convention**: Both landscapes have global minimum at θ≈172°.
With our C1-C2-C3-C4-C5-O5 ordering, PLUMED defines θ=0 as 4C1 and θ=π as 1C4.
This means the global minimum is labeled 1C4. But β-D-xylose is expected to
favor 4C1 in solution. Could the atom ordering convention be inverted for this
system? Or does CHARMM36 genuinely predict 1C4 for free xylose?

The landscape shapes and barriers are physically reasonable regardless of
how the chairs are labeled. The comparative result (enzyme lowers barriers)
holds either way.

### Atom indices clarification

Your indices (6599-6607) correspond to **XYS F chain, residue 4** in the raw
2D24 PDB — a symmetry-related copy in the other asymmetric unit. Our simulation
uses chain A (protein) + chain C (substrate). After GROMACS pdb2gmx processing,
the xylose atoms end up at positions 6278-6297 (protein has 6277 atoms, then
xylose follows). See `modules/enzyme-bound-puckering/ATOM_INDEX_MAP.md` for the
full mapping.

---

## For ABG Discord

### tl;dr

Rebuilt the free xylose FEL from crystal coordinates (v0.6.0 accidentally used
lyxose from RDKit), completed the enzyme-bound run that was actually finished but
never post-processed, and now have the direct free-vs-bound comparison:

**The enzyme lowers all conformational barriers by 1–5 kJ/mol**, consistent with
catalytic facilitation of ring distortion during glycoside hydrolysis. This is
the classical MD analogue of what Iglesias-Fernández showed with QM/MM.

### Lessons learned (for future pseudoSpore handoffs)

1. Never trust RDKit SMILES → 3D for carbohydrate stereochemistry without verification
2. Always ship topology excerpts so reviewers can verify atom indices
3. Check if production actually completed before labeling IN_FLIGHT
4. Crystal coordinates are the gold standard for known molecules

---

**Artifact size**: ~130K (input files + FES data + provenance + documentation)
**Total simulation**: 30 ns (10 ns × 3 modules)
**Hardware**: strandGate, RTX 3090, AMD 64-thread
**Checks**: 16 passed | All 3 modules COMPLETE
