# CAZyme FEL Exploration Targets

**Status:** Active — ready for Alistaire review and prioritization
**Last Updated:** 2026-05-26
**Prerequisite:** pseudoSpore v1.5.0 GuideStone validated (46/46 PASS, May 26 2026)
**Pipeline:** `nest-validate guidestone run` (Rust-orchestrated, fully automated)
**Reference paper:** Iglesias-Fernández, Raich, Ardèvol, Rovira (2015) DOI: 10.1039/C4SC02240H

## Purpose

With the GH10 xylose puckering FEL pipeline validated end-to-end, this spec
documents nearby scientific targets that can be explored using the same
infrastructure. Each target varies one axis: substrate stereochemistry (epimers),
binding context (subsites, covalent state), or enzyme identity (GH families).

The validated pipeline establishes: for any 6-membered pyranose ring with known
GROMACS atom indices, we can produce a Cremer-Pople FEL (1D theta + 2D Stoddart)
comparing free vs enzyme-bound conformational landscapes. Adding a new target
requires only: topology + plumed.dat + atom indices.

---

## Validated Baseline (Reference for All Targets)

| System | Atoms | Sim time | Wall time | Key result |
|--------|-------|----------|-----------|------------|
| Free β-D-xylose 1D | 2,657 | 10 ns | 12 min | 3 basins: 4C1/boat/1C4; barriers 38-53 kJ/mol |
| Free β-D-xylose 2D | 2,657 | 20 ns | 24 min | Full Stoddart diagram |
| GH10-bound xylose 1D | 92,745 | 10 ns | 79 min | All barriers lowered 1-5 kJ/mol |
| GH10-bound xylose 2D | 92,745 | 20 ns | 158 min | Conformational selection visible |

**CV:** PLUMED `PUCKERING ATOMS=C1,C2,C3,C4,C5,O5` → theta (1D) or qx,qy (2D)
**Method:** Well-tempered metadynamics (HEIGHT=1.5, SIGMA=0.1, BIASFACTOR=15, PACE=500)
**Tier parity:** <1 kJ/mol (1D), <2 kJ/mol (2D) between Rust and GROMACS sum_hills

---

## Section 1: Xylose Epimers and Subisomers

**Question:** How does ring stereochemistry at C2, C3, or C4 alter the puckering FEL?
Does a single hydroxyl flip open or close the boat/skew-boat pathway?

| Sugar | Relationship | Ring formula | CHARMM36 residue | Available |
|-------|-------------|--------------|------------------|-----------|
| β-D-xylose | Reference | C5H10O4 (pentose) | BXYL | ✅ validated |
| β-D-lyxose | C2 epimer of xylose | C5H10O4 | BLXYL | ✅ in FF |
| α-L-arabinose | Enantiomer of D-xylose at C2,C4 | C5H10O4 | AARA | ✅ in FF |
| β-D-ribose | C3 epimer of xylose | C5H10O4 | BRIB | ✅ in FF |
| β-D-glucose | Xylose + C5-CH₂OH (hexose) | C6H12O5 | BGLC | ✅ in FF |
| β-D-mannose | C2 epimer of glucose | C6H12O6 | BMAN | ✅ in FF |
| β-D-galactose | C4 epimer of glucose | C6H12O6 | BGAL | ✅ in FF |

### Pipeline requirements per epimer (free sugar only)

1. Generate topology: CHARMM-GUI → single residue in TIP3P box (~2,500-3,000 atoms)
2. Equilibration: em → nvt → npt (existing MDP files, swap residue name)
3. PLUMED: identical to `plumed_xylose_1d.dat` / `plumed_xylose_2d.dat`, update atom indices
4. Wall time estimate: 12 min (1D) + 24 min (2D) per epimer = **~36 min each**
5. Total for all 6 new epimers: **~3.6 hours** (trivially parallelizable)

### Scientific value

- Lyxose vs xylose (C2 flip): does equatorial→axial OH at C2 alter boat barrier?
- Glucose vs xylose (CH₂OH at C5): does the bulky exocyclic group lock 4C1?
- Mannose (C2 epimer of glucose): important for mannosidase catalytic itinerary
- Literature gap: no systematic free-sugar Cremer-Pople comparison across all C2/C3/C4 epimers with same method/FF

### Expected outcomes

- Glucose and galactose: expected deep 4C1 minimum (CH₂OH stabilizes chair)
- Lyxose: may have lower boat barrier (axial C2-OH reduces 1,3-diaxial strain in chair)
- Mannose: expected slight 1S5 population (relevant to mannosidase mechanism)

---

## Section 2: Other Subsites on PDB 2D24

**Question:** Does the GH10 enzyme pre-organize the entire xylan chain, or only the
−1 subsite? Is conformational distortion localized or propagated?

### Subsite nomenclature (GH10 xylanase)

```
  −3    −2    −1    +1    +2    +3
  Xyl — Xyl — Xyl — Xyl — Xyl — Xyl
                  ↑
            Cleavage point (Glu128/Glu236)
```

### Available in PDB 2D24

| Subsite | Residue | Chain | PDB serial range | Status |
|---------|---------|-------|------------------|--------|
| −1 | BXYL 437 | A | 6599-6607 | ✅ Validated (modules 05/06) |
| −2 | XYS (chain C, res 3) | C | ~6458-6468 | Needs index extraction |
| +1 | XYS (chain C, res 5) | C | ~6479-6489 | Needs index extraction |

### Pipeline requirements

- Same 92,745-atom system (no new topology needed!)
- New PLUMED files with different `PUCKERING ATOMS=...` for −2 and +1
- Identical metadynamics parameters
- Wall time: same as enzyme-bound (~79 min per 1D, ~158 min per 2D)
- Total for both subsites: **~8 hours** (1D + 2D for −2 and +1)

### Scientific value

- Tests whether catalytic distortion is localized to −1 or extends to neighbors
- If −2 stays 4C1: enzyme selection happens only at cleavage site
- If +1 shows distortion: product release pathway involves conformational relaxation
- Directly tests Iglesias-Fernández Figure 6 (TS geometry involves −1 distortion)
- **Not previously published** for this specific system with metadynamics

---

## Section 3: 1E0X Covalent Intermediate

**Question:** How does the FEL change between Michaelis complex (2D24) and
glycosyl-enzyme intermediate (1E0X)? Is the product trapped in a specific
conformation?

### System details

| Property | PDB 2D24 (current) | PDB 1E0X (target) |
|----------|--------------------|--------------------|
| Organism | *S. olivaceoviridis* E-86 | *S. lividans* |
| Resolution | 1.41 Å | 1.80 Å |
| Ligand state | ES Michaelis complex | Glycosyl-enzyme (covalent) |
| −1 residue | BXYL (non-covalent) | X2F (covalent to Glu236) |
| Atom count | 92,745 (solvated) | ~80,000 (estimate) |
| Nucleophile distance | 5.76 Å (pre-attack) | 1.4 Å (covalent bond) |

### Pipeline requirements

1. **New topology:** CHARMM-GUI with covalent ligand patch (X2F → Glu236 ester bond)
2. PLUMED: identify ring atoms of X2F in new GROMACS numbering
3. Metadynamics: same parameters
4. Wall time estimate: ~79 min (1D) + ~158 min (2D) ≈ **4 hours**
5. Also need free X2F analog for baseline (small system, ~30 min)

### Scientific value

- Direct comparison of FEL at two points along the reaction coordinate
- 2D24 (Michaelis): expect −1 xylose pre-distorted toward TS geometry
- 1E0X (covalent): expect relaxation toward 4C1 (product state)
- Tests the "conformational itinerary" hypothesis: 4C1 → 2,5B → 2SO (retaining GH)
- Validates that puckering CVs capture mechanistically relevant conformational changes
- **Unique:** no one has compared FEL of same enzyme at two catalytic states with metadynamics

### Open questions for Alistaire

- Is the X2F residue parameterized in CHARMM36? Or need CGenFF?
- Should we use the same BIASFACTOR or increase for the covalent system?
- Does xylobiosyl (XYP-X2F) need both rings sampled simultaneously?

---

## Section 4: Other GH Families

**Question:** Does enzyme mechanism type (retaining vs inverting) produce different
puckering FELs? Do different substrates in the same mechanism class show conserved
distortion patterns?

### Target GH families

| Family | Mechanism | Substrate | Model PDB | Resolution | Atoms (est.) | Key question |
|--------|-----------|-----------|-----------|------------|:---:|--------------|
| **GH11** | Inverting | Xylose | 1XYN (*Trichoderma*) | 1.80 Å | ~25,000 | Same substrate, different mechanism — does FEL differ? |
| **GH5** | Retaining | Glucose | 3QR3 (Cel5A, *T. maritima*) | 1.60 Å | ~45,000 | Same mechanism as GH10, hexose vs pentose |
| **GH7** | Inverting | Glucose | 8CEL (Cel7A, *T. reesei*) | 1.90 Å | ~60,000 | Industrial cellulase; glucose puckering in inverting enzyme |
| **GH26** | Retaining | Mannose | 2QHA (*C. japonicus*) | 1.80 Å | ~50,000 | C2 epimer of glucose in retaining enzyme |
| **GH38** | Retaining | Mannose | 1QWN (MII, *D. melanogaster*) | 1.35 Å | ~55,000 | Target 06 validation standard (Grothaus) |
| **GH43** | Inverting | Arabinose | 3C7E (*G. stearothermophilus*) | 1.90 Å | ~40,000 | L-sugar processing |

### Prioritization rationale

1. **GH11 (1XYN):** Highest priority — same substrate (xylose), different mechanism.
   Direct test: does the retaining vs inverting distinction appear in the static FEL?
   Inverting enzymes use a single-displacement mechanism (no covalent intermediate),
   so the −1 xylose should show a different preferred distortion.

2. **GH38 (1QWN):** High priority — external validation standard (Grothaus et al. 2025,
   Target 06). Reproduce their result with our pipeline for cross-validation.

3. **GH5 (3QR3):** Medium — same mechanism class, different substrate. Tests whether
   retaining GH distortion pattern is substrate-specific or enzyme-specific.

4. **GH7 (8CEL):** Medium — industrially relevant cellulase. Glucose FEL in inverting context.

5. **GH26 (2QHA):** Lower — mannose in retaining enzyme. Useful after mannose free-sugar baseline (Section 1).

6. **GH43 (3C7E):** Lower — L-arabinose processing. Needs L-sugar parameterization check.

### Pipeline requirements per GH family

1. Download PDB, identify −1 subsite sugar and ring atoms
2. Generate CHARMM36 topology (CHARMM-GUI protein+ligand)
3. Solvate, equilibrate (em → nvt → npt)
4. Extract GROMACS atom indices for ring PUCKERING CV
5. Run free-sugar baseline (if new substrate: ~36 min)
6. Run enzyme-bound 1D + 2D (~4 hours per enzyme)
7. Compare FEL: free vs enzyme, retaining vs inverting

**Total for all 6 families:** ~24 hours compute (parallelizable to ~4 hours on 6 cores)

---

## Section 5: PLUMED-NEST Glycan Targets (05 + 06)

**Question:** Can we reproduce published glycan puckering results (Grothaus et al.)
with our pipeline, and extend to systems they haven't studied?

### Target 05 — Free N-glycans (plumID 22.028)

**Paper:** Grothaus, Bussi, Colombi Ciacchi, JCIM 62:4992-5008 (2022)
**Method:** REST2-RECT (Replica Exchange with Collective Tempering)
**Temperature:** 310.15 K
**Replicas:** 12 (RECT scaling 1→14)

| Glycan | Monosaccharides | Rings to sample | CVs |
|--------|:---:|:---:|-----|
| M5 (high-mannose) | 7 | 7 | 14 glycosidic dihedrals + 7 theta |
| FM5 (fucosylated M5) | 8 | 8 | 16 dihedrals + 8 theta |
| M9 (high-mannose) | 11 | 11 | 20 dihedrals + 11 theta |
| A2G2 (complex-type) | 11 | 11 | 20 dihedrals + 11 theta |
| A2G2S2 (sialylated) | 13 | 13 | 24 dihedrals + 13 theta |

**Monosaccharides covered:** mannose, glucose, galactose, fucose, NAG, sialic acid
**CHARMM36 residue filter:** MAN, GLC, GAL, FUC, NAG, SIA, BMA

**Difference from our pipeline:** REST-RECT uses replica exchange (12-16 walkers)
rather than single-walker well-tempered metadynamics. Our pipeline would need:
- Multi-replica support in `nest-validate` (not yet implemented)
- OR: single-replica metadynamics on each monosaccharide ring independently (simpler,
  but loses coupling between rings)

### Target 06 — Enzyme-bound glycan (plumID 25.007)

**Paper:** Grothaus et al. (2025, in preparation) — "Shaping the glycan landscape"
**Method:** REST-RECT + steered MD
**Enzyme:** Mannosidase II (MII, *Drosophila melanogaster*, PDB 1QWN)
**Substrate:** M5G0 high-mannose glycan (7 monosaccharides)

| System | Atoms | Replicas | Purpose |
|--------|:---:|:---:|---------|
| M5G0 free | ~300 | 12 | Glycan baseline |
| M5G0 + MII | ~16,322 | 16 | Enzyme-bound landscape |
| M5G0 + MII (steer conformer) | ~16,322 | — | Targeted transitions |
| M5G0 + MII (steer theta) | ~16,322 | — | Puckering pathway sampling |

**Key insight from Grothaus:** Enzymes induce pucker distortions in substrate glycans,
shifting population from 4C1 toward conformations along the reaction pathway.
This is exactly what we observe in our GH10 system.

### Integration strategy

**Phase 1 (immediate):** Use our validated single-walker metadynamics on the M5G0
system (free + MII-bound). Compare theta distributions to Grothaus REST-RECT results.
If agreement: our simpler method captures the essential physics.

**Phase 2 (future):** Implement multi-replica dispatch in `nest-validate` for
full REST-RECT reproduction. This maps to `toadStool` multi-walker orchestration.

**Domain profile:** `profiles/carbohydrate_pucker.toml` already defines:
- Residue filter: MAN, GLC, GAL, FUC, NAG, SIA, BMA
- Puckering zones: 4C1 (θ < 0.35), 1C4 (θ > 2.79), boat (1.2 < θ < 1.9)
- Tolerance: peak position ±0.2 rad, barrier ±5 kJ/mol
- Population constraint: 4C1 fraction > 85% for healthy sugars

---

## Section 6: Pipeline Requirements Summary

### Per-target checklist

| Step | Input | Output | Tool | Time |
|------|-------|--------|------|------|
| 1. Topology | PDB + CHARMM36 params | .gro + .top | CHARMM-GUI | manual |
| 2. Equilibrate | .gro + .top + .mdp | npt.gro + npt.cpt | gmx grompp/mdrun | ~5 min |
| 3. Atom indices | npt.gro + ring definition | plumed.dat | manual/script | 2 min |
| 4. Register | system def | nest-validate config | Rust code update | 5 min |
| 5. Simulate 1D | npt.gro + plumed.dat | HILLS + COLVAR | nest-validate run | variable |
| 6. Simulate 2D | npt.gro + plumed.dat | HILLS_2d + COLVAR_2d | nest-validate run | variable |
| 7. Finalize | HILLS → FES | fes_*.dat | nest-validate finalize | <1 min |
| 8. Validate | FES + target.toml | PASS/FAIL | nest-validate validate | <1 min |

### Compute time estimates

| Target class | Atom count | 1D time | 2D time | Total |
|-------------|:---:|:---:|:---:|:---:|
| Free sugar (new epimer) | ~2,700 | 12 min | 24 min | 36 min |
| GH10 other subsite | 92,745 | 79 min | 158 min | 4 hr |
| 1E0X covalent | ~80,000 | 70 min | 140 min | 3.5 hr |
| GH11 (small enzyme) | ~25,000 | 25 min | 50 min | 1.2 hr |
| GH5/7/26/38 (medium) | ~50,000 | 45 min | 90 min | 2.2 hr |
| M5G0 free (Grothaus) | ~300 | 5 min | 10 min | 15 min |
| M5G0 + MII (Grothaus) | ~16,322 | 18 min | 36 min | 54 min |

### Force field availability (CHARMM36-jul2022)

| Residue | Name | Available | Notes |
|---------|------|:---------:|-------|
| β-D-xylose | BXYL | ✅ | Validated in production |
| β-D-lyxose | BLXYL | ✅ | C2 epimer, in standard FF |
| α-D-xylose | AXYL | ✅ | Anomer |
| α-L-lyxose | ALXYL | ✅ | L-form |
| β-D-glucose | BGLC | ✅ | Standard hexose |
| β-D-mannose | BMAN | ✅ | Standard hexose |
| β-D-galactose | BGAL | ✅ | Standard hexose |
| NAG (GlcNAc) | BGLCNA | ✅ | N-glycan core |
| Fucose | AFUC | ✅ | 6-deoxy sugar |
| Sialic acid | ANE5AC | ✅ | 9-carbon sugar |
| α-L-arabinose | — | ⚠️ | May need CGenFF or GLYCAM parameterization |
| Covalent ester (X2F) | — | ⚠️ | Needs custom patch (Glu-OE2—C1 bond) |

---

## Section 7: Prioritization for Alistaire

### Tier 1 — Immediate (validated pipeline, minimal setup)

| # | Target | Effort | Time | Why first |
|---|--------|--------|------|-----------|
| 1 | **Free lyxose** (C2 epimer) | Swap BXYL→BLXYL in topology | 36 min | Nearest neighbor; tests stereochemistry hypothesis |
| 2 | **Free glucose/mannose/galactose** | New CHARMM-GUI topologies | 3×36 min | Hexose baselines needed for GH5/7/26/38 |
| 3 | **2D24 −2 subsite** | New atom indices only | 4 hr | Same system, no new setup; tests distortion propagation |
| 4 | **2D24 +1 subsite** | New atom indices only | 4 hr | Same system; tests product-side relaxation |

**Total Tier 1 compute: ~10 hours** (parallelizable to ~4 hours)

### Tier 2 — Near-term (new topology, established methodology)

| # | Target | Effort | Time | Why |
|---|--------|--------|------|-----|
| 5 | **GH11 (1XYN)** — inverting xylanase | Full CHARMM-GUI setup | 1.2 hr | Same substrate, different mechanism (key comparison) |
| 6 | **M5G0 + MII** (Grothaus system) | PDB + CHARMM-GUI | 54 min | External validation standard |
| 7 | **1E0X** covalent intermediate | Custom covalent patch | 3.5 hr | Two-state comparison along catalytic cycle |
| 8 | **GH5 (3QR3)** — retaining cellulase | CHARMM-GUI + glucose | 2.2 hr | Glucose in same mechanism class |

**Total Tier 2 compute: ~8 hours**

### Tier 3 — Medium-term (broader survey)

| # | Target | Effort | Time | Why |
|---|--------|--------|------|-----|
| 9 | **GH7 (8CEL)** — inverting cellulase | CHARMM-GUI | 2.2 hr | Industrial enzyme + glucose |
| 10 | **GH26 (2QHA)** — retaining mannanase | CHARMM-GUI | 2.2 hr | Mannose mechanism |
| 11 | **GH43 (3C7E)** — inverting arabinofuranosidase | Parameterization needed | 1.5 hr | L-sugar; unique |
| 12 | **Free N-glycans (M5, FM5, M9)** | REST-RECT infra needed | TBD | Grothaus reproduction |
| 13 | **All 38 Stoddart conformations** (docking) | AutoDock Vina setup | — | FEL ↔ docking correlation |

### Decision matrix for Alistaire

| Factor | Weight | Epimers | Subsites | 1E0X | GH families | Glycans |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Scientific novelty | High | Medium | **High** | **High** | **High** | Low (reproduction) |
| Pipeline readiness | — | **Ready** | **Ready** | Moderate | Moderate | Complex (RECT) |
| Compute cost | — | **Low** | Medium | Medium | Medium | High (replicas) |
| ABG relevance | High | Medium | **High** | **High** | **High** | Medium |
| Publication potential | High | Medium | **High** | **High** | **High** | Medium |

**Recommended first batch (maximum novelty / minimum effort):**
1. Free lyxose + glucose + mannose (baselines, 2 hours)
2. 2D24 −2 and +1 subsites (unprecedented, 8 hours)
3. GH11 (1XYN) inverting xylanase (key mechanistic comparison, 1.2 hours)

---

## References

- Iglesias-Fernández et al. (2015) Chem. Sci. 6:1167. DOI: 10.1039/C4SC02240H
- Grothaus, Bussi, Colombi Ciacchi (2022) JCIM 62:4992. DOI: 10.1021/acs.jcim.2c01049
- Grothaus et al. (2025) "Shaping the glycan landscape" (plumID 25.007)
- Cremer & Pople (1975) JACS 97:1354. DOI: 10.1021/ja00839a011
- Biarnés et al. (2007) JACS 129:10686. DOI: 10.1021/ja068411o

## Cross-references

- [`pseudoSpore_hotSpring-CompChem-GuideStone_v1.5.0/`](../pseudoSpore_hotSpring-CompChem-GuideStone_v1.5.0/) — validated methodology
- [`specs/SOVEREIGN_COMPCHEM_EVOLUTION.md`](SOVEREIGN_COMPCHEM_EVOLUTION.md) — sovereign pipeline roadmap (Papers 50-58)
- [`control/plumed_nest/target_05_glycan_pucker/`](../control/plumed_nest/target_05_glycan_pucker/) — Grothaus 2022 profiles
- [`control/plumed_nest/target_06_cazyme_glycan/`](../control/plumed_nest/target_06_cazyme_glycan/) — Grothaus 2025 profiles
- [`control/plumed_nest/profiles/carbohydrate_pucker.toml`](../control/plumed_nest/profiles/carbohydrate_pucker.toml) — domain profile
- [`TRANSLATE.md`](../pseudoSpore_hotSpring-CompChem-GuideStone_v1.5.0/TRANSLATE.md) — CV conventions
