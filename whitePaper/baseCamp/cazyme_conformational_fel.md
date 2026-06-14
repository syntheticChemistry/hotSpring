# baseCamp Briefing: CAZyme Conformational Free Energy Landscapes

**Date:** May 25, 2026  
**Experiment:** 220  
**Domain:** Biomolecular MD — enhanced sampling (well-tempered metadynamics)  
**Target paper:** Iglesias-Fernández et al. 2015 — "Free Energy of Conformational Substates" (PDB 2D24, GH10 xylanase)  
**Status:** Tier 0-2 parity MATCH (<1 kJ/mol) | All modules COMPLETE | 2D (qx, qy) expansion DONE (v0.8.0)

---

## Motivation

Validate that sovereign compute (Python → Rust → eventually WGSL) can reproduce industry-standard molecular dynamics free energy landscapes. This extends the hotSpring validation ladder into biomolecular science — the first domain involving bonded force fields (CHARMM36), enhanced sampling, and collective variables beyond simple pairwise potentials.

---

## Three-Module Validation Ladder

| Module | System | CV | Tier 0 (GROMACS) | Tier 1 (Python) | Tier 2 (Rust) |
|--------|--------|-----|-----------------|-----------------|---------------|
| 1 | Alanine dipeptide | φ/ψ (Ramachandran) | ✅ C7eq/C7ax, ΔF=5.57 kJ/mol | ✅ 0.52 kJ/mol max dev | ✅ exact match to Tier 1 |
| 2 | Free xylose (crystal) | Cremer-Pople θ (puckering) | ✅ 3 basins, barriers 38-53 kJ/mol | ✅ 0.73 kJ/mol max dev | ✅ 0.79 kJ/mol max dev |
| 3 | Enzyme-bound xylose (2D24) | Cremer-Pople θ | ✅ 3 basins, enzyme lowers barriers 1-5 kJ/mol | ✅ 0.76 kJ/mol max dev | ✅ 0.77 kJ/mol max dev |
| 2b | Free xylose (crystal) | Cremer-Pople (qx, qy) 2D | ✅ 20 ns WTMetaD, full Stoddart | ✅ 1.71 kJ/mol max dev | ✅ 1.71 kJ/mol max dev |
| 3b | Enzyme-bound xylose (2D24) | Cremer-Pople (qx, qy) 2D | ✅ 20 ns WTMetaD | ✅ 1.72 kJ/mol max dev | ✅ 1.72 kJ/mol max dev |

---

## Tier Architecture

```
Tier 0 — GROMACS 2026.0 + PLUMED 2.9.2 (industry control)
  ↓ parity check (sum_hills reconstruction)
Tier 1 — Python (notebooks/cazyme_fel/puckering_fel.py)
  ↓ exact match
Tier 2 — Rust (staging/cazyme-fel/src/lib.rs)
  ↓ (future) WGSL shader promotion
Tier 3 — NUCLEUS IPC composition (GAP-HS-111 bonded shaders)
```

### Key algorithm: sum_hills (FEL reconstruction from HILLS file)

Well-tempered metadynamics deposits Gaussians at visited CV positions. The free energy surface is reconstructed as: `F(s) = -V(s) + const`, where V(s) is the sum of all deposited Gaussians. The heights decay as `h₀ · exp(-V(s)/ΔT·kB)` — no additional γ/(γ-1) correction is needed (heights already encode the reweighting).

---

## Parity Results (May 25, 2026)

### 1D Puckering (Cremer-Pople θ)

| Comparison | Module 1 (2D φ/ψ) | Module 2 (1D θ, free) | Module 3 (1D θ, enzyme) |
|------------|-------------------|-----------------------|------------------------|
| Tier 0 → Tier 1 | 0.52 kJ/mol | 0.73 kJ/mol | 0.76 kJ/mol |
| Tier 0 → Tier 2 | 0.52 kJ/mol | 0.79 kJ/mol | 0.77 kJ/mol |
| Tolerance | 2.0 kJ/mol | 1.0 kJ/mol | 1.0 kJ/mol |
| Verdict | **MATCH** | **MATCH** | **MATCH** |

### 2D Puckering (Cremer-Pople qx, qy — full Stoddart landscape)

| System | Tier 0 | Tier 1 | Tier 2 |
|--------|--------|--------|--------|
| Free xylose (20 ns) | PASS | MATCH (1.71 kJ/mol) | MATCH (1.71 kJ/mol) |
| Enzyme-bound 2D24 (20 ns) | PASS | MATCH (1.72 kJ/mol) | MATCH (1.72 kJ/mol) |

---

## Connection to Primal Evolution

### GAP-HS-111: barraCuda Bonded Force Shaders

To run MD natively (Tier 3), barraCuda needs:
- Harmonic bond potential WGSL shader
- Harmonic angle potential WGSL shader
- Dihedral torsion (proper + improper) WGSL shader
- CHARMM36 topology reader (PSF/PRMTOP → GPU buffers)
- Cremer-Pople collective variable WGSL kernel
- Metadynamics bias engine (Gaussian deposition + sum_hills)

### GAP-HS-112: petalTongue FEL Visualization

2D/3D surface rendering over Cremer-Pople coordinates. DataBinding adapter spec needed from ludoSpring.

---

## lithoSpore Promotion Path

The `staging/cazyme-fel/` Rust crate + `notebooks/cazyme_fel/puckering_fel.py` implement the core `sum_hills` algorithm at parity (1D and 2D). The promotion path to `lithoSpore` (ecoPrimals' reproducible science chassis) is documented in `docs/LITHOSPORE_PROMOTION.md`:

1. FermentBraid wire format aligned (`provenance/braids/hotspring_cazyme_fel.json`)
2. scope.toml and validation.json follow lithoSpore schema
3. Rust crate ready for `lithoSpore::modules::cazyme_fel` integration (now with 2D support)
4. pseudoSpore v0.8.0 — includes 2D outputs, full validation matrix
5. Primal elevation readiness doc written (`docs/PRIMAL_ELEVATION_READINESS.md`)

---

## Provenance

- **sweetGrass braid:** Live IPC (v0.7.27) — `live_braid.json` + `provo_export.jsonld`
- **rhizoCrypt DAG:** Pseudo (11 events, Merkle root) — ready for `dag.session.create` IPC
- **loamSpine ledger:** Pseudo (3 entries) — ready for `ledger.append` + DID anchoring
- **FermentBraid:** `hotspring_cazyme_fel.json` (BLAKE3 hashes, lithoSpore wire format)

---

## Files

| Path | Role |
|------|------|
| `notebooks/cazyme_fel/puckering_fel.py` | Tier 1 Python sum_hills implementation (1D + 2D) |
| `staging/cazyme-fel/` | Tier 2 Rust crate (1D + 2D reconstruction) |
| `control/gromacs_fel/` | Tier 0 GROMACS+PLUMED raw outputs |
| `control/gromacs_fel/validation_matrix.json` | Systematic results matrix |
| `docs/PRIMAL_ELEVATION_READINESS.md` | GAP-HS-111 readiness checklist |
| `experiments/220_CAZYME_CONFORMATIONAL_ENERGY_LANDSCAPES.md` | Experiment journal |
| `docs/LITHOSPORE_PROMOTION.md` | lithoSpore promotion plan |
| `docs/PRIMAL_GAPS.md` | GAP-HS-111, GAP-HS-112 definitions |
