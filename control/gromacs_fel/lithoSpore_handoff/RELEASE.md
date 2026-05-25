# pseudoSpore Release — v0.7.0

**Artifact**: hotSpring-CAZyme-FEL
**Status**: All 3 modules COMPLETE, corrections applied
**Date**: May 24, 2026
**Supersedes**: v0.6.0 (contained critical errors identified by Alistaire)

---

## What Changed from v0.6.0

| Issue | v0.6.0 | v0.7.0 |
|-------|--------|--------|
| Module 2 structure | RDKit-generated β-D-Lyxose (WRONG) | PDB 2D24 crystal β-D-Xylose |
| Module 3 status | IN_FLIGHT | COMPLETE (sum_hills run) |
| Atom index documentation | None | ATOM_INDEX_MAP.md |
| Solvation evidence | None (unverifiable) | SYSTEM_SETUP.md with full pipeline |
| FEL comparison | Not possible | Free vs enzyme-bound complete |

### Errors found by Alistaire (ABG domain expert)

1. `xylose_charmm.pdb` was β-D-Lyxose in 1C4 pucker (φ=107°, θ=9.9°, Q=0.529)
2. `xylose_m1.pdb` was raw crystal coordinates (not prepared system input)
3. Atom indices 6278-based not reconciled with raw PDB numbering
4. Module 3 appeared incomplete but production had actually finished
5. No topology files → reviewer couldn't verify water or atom identities

All issues addressed in this release.

---

## Structural Checklist (lithoSpore ingestion)

| Requirement | Status |
|---|---|
| `ferment_transcript.json` with `braid_id` + `dag_session_id` + `merkle_root` + `spine_id` | Present, well-formed |
| `provenance/braids/` directory (lithoSpore ingestion path) | Present, 3 files |
| `scope.toml` (machine-readable birth certificate) | Updated to v0.7.0 with `[corrections]` section |
| `validation.json` (structured checks) | Present, errata annotated inline |
| Live sweetGrass braid (actual IPC, not pseudo) | `live_braid.json` — real sweetGrass v0.7.27 touched this |
| W3C PROV-O JSON-LD export | `provo_export.jsonld` — valid `@context`, correct `@graph` |
| Human-readable handoff doc | `ABG_HANDOFF.md` — updated with corrections |
| Reproducibility (configs + data) | All MDP/PLUMED/PDB files + FES outputs per module |
| Atom index mapping | `ATOM_INDEX_MAP.md` in Module 3 |
| Solvation documentation | `SYSTEM_SETUP.md` in Module 2 |

---

## Key Scientific Result

The enzyme active site (GH10 xylanase, PDB 2D24) uniformly lowers conformational
energy barriers for the -1 subsite xylose ring:

| Observable | Free xylose | Enzyme-bound | Delta |
|-----------|-------------|-------------|-------|
| Boat basin | 6.4 kJ/mol | 5.4 kJ/mol | −1.0 |
| 4C1 chair | 16.1 kJ/mol | 13.6 kJ/mol | −2.5 |
| A→Boat barrier | 52.5 kJ/mol | 47.4 kJ/mol | −5.1 |
| Boat→B barrier | 37.6 kJ/mol | 36.7 kJ/mol | −0.9 |

This is consistent with catalytic facilitation of ring distortion during
glycoside hydrolysis (Iglesias-Fernández et al. 2015).

---

## Open Question: Cremer-Pople Convention

Both free and enzyme-bound show global minimum at θ≈172° (1C4 by PLUMED
convention for C1-C2-C3-C4-C5-O5 ordering). β-D-xylose is expected to favor
4C1 in solution. This could reflect:

1. A convention mapping issue (atom ordering inverts the label assignment)
2. A CHARMM36 force field characteristic
3. The crystal starting geometry biasing the landscape

Alistaire's domain expertise is needed to resolve this. The landscape shape
and barrier heights are physically reasonable regardless of label assignment.

---

## Hash Note

lithoSpore proper uses BLAKE3 for content hashes; the pseudo uses SHA-256.
For a data-only / braid-passing handoff this doesn't matter — the hashes are
informational provenance, not signature-verified. When this eventually promotes
to a real lithoSpore module, the hashes get re-computed as BLAKE3 during
`litho assemble`.
