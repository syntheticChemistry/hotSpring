# CompChem GuideStone v1.6.1

**Self-verifying baseline for enhanced sampling molecular dynamics.**

## What This Is (Plain Language)

Enzymes that break down plant cell walls (xylanases) need their sugar substrate to change shape during catalysis. This artifact computationally proves *how* the enzyme forces that shape change: it lowers the energy barriers between ring conformations. The paper describes a two-fold catalytic itinerary — 2SO → [2,5B]‡ → 5S1 and 1S3 → [4H3]‡ → 4C1 — where the enzyme active site selectively stabilizes the conformational pathway required to reach the transition state.

We compute "free energy landscapes" — maps showing how much energy it costs for the sugar ring to adopt every possible shape — both floating free in water and locked inside the enzyme active site. The enzyme landscape shows lower barriers and a preferred pathway that matches experimental predictions.

**Why it matters:** If molecular docking (cheap, fast) can approximate these landscapes (expensive, slow), we can screen enzyme-substrate interactions across entire CAZyme families without running days of simulation for each.

## Scientific Summary

This pseudoSpore contains validated free energy landscapes from 6 independent molecular simulations, a PLUMED-NEST validation aggregate (8 targets), and an exploration roadmap (13 proposed targets). It spans three enhanced sampling methods (well-tempered metadynamics, OPES, OPES_EXPLORE) and two scientific domains (protein folding, carbohydrate conformational landscapes).

**Key finding:** The GH10 xylanase active site lowers conformational barriers for the -1 subsite xylose ring, consistent with the two-fold catalytic itinerary (2SO → [2,5B]‡ → 5S1 and 1S3 → [4H3]‡ → 4C1) described by Iglesias-Fernández et al. ([doi:10.1039/C4SC02240H](https://doi.org/10.1039/C4SC02240H)).

## Quick Start

| Want to... | Do this |
|-----------|---------|
| See the science results | Open `modules/*/fes_*.dat` or generate figures with `nest-validate guidestone finalize .` |
| Verify integrity | `./validate` (works with just `b3sum`, or `nest-validate` for full science checks) |
| Check data freshness | `./refresh` |
| Read the PLUMED-NEST status | `modules/07_plumed_nest_validation/summary.md` |
| See future directions | `modules/08_exploration_roadmap/CAZYME_FEL_EXPLORATION_TARGETS.md` |
| Understand atom indices | `TRANSLATE.md` and `index_map.toml` |
| Deploy to ecoPrimals | `DEPLOY.md` |

## Modules

| # | System | Method | Time | Key Result |
|---|--------|--------|------|------------|
| 01 | Alanine dipeptide (vacuum) | WTMetaD phi/psi | 10 ns | C7eq/C7ax basins, 20-45 kJ/mol barriers |
| 02 | Chignolin CLN025 (water) | OPES+OPES_E HLDA | 61.8 ns | 10 transitions, dG_fold=+19.5 kJ/mol (340K > Tm) |
| 03 | Free xylose 1D (water) | WTMetaD theta | 10 ns | 4C1/boat/1C4 basins, 38-53 kJ/mol |
| 04 | Free xylose 2D (water) | WTMetaD qx,qy | 20 ns | Full Stoddart diagram |
| 05 | Enzyme-bound GH10 1D | WTMetaD theta | 10 ns | Barrier lowering by enzyme |
| 06 | Enzyme-bound GH10 2D | WTMetaD qx,qy | 20 ns | Conformational selection |
| 07 | PLUMED-NEST validation | Multi-method | -- | 8 targets (2 PASS, 6 profiled) |
| 08 | Exploration roadmap | Proposed | -- | 13 targets across 3 priority tiers |

## Verification

```bash
# Integrity only (requires b3sum or nest-validate)
./validate

# Full pipeline re-execution (requires GROMACS + PLUMED + nest-validate)
./run
```

Exit codes: 0 = full pass, 1 = fail, 2 = integrity-only pass (no science checks).

## File Inventory

| File | Purpose |
|------|---------|
| `scope.toml` | Module manifest with `[artifact]` + `[guidestone]` dual keys |
| `validation.json` | Machine-readable module statuses (litho wire format) |
| `environment.toml` | Hardware/software snapshot at emit time |
| `domain_profile.toml` | compchem-enhanced-sampling v1.0.0 audit profile |
| `tolerances.toml` | Quantitative acceptance criteria with literature references |
| `data.toml` | BLAKE3 hashes for all files + `[external_braids]` upstream references |
| `receipts/checksums.blake3` | Traditional BLAKE3 receipt for litho audit compatibility |
| `index_map.toml` | PDB 2D24 domain serials to GROMACS computation indices |
| `TRANSLATE.md` | Human-readable cross-reference legend |
| `DEPLOY.md` | Deployment guide: local, primals.eco, cellMembrane VPS, NUCLEUS |
| `validate` | Self-verification (nest-validate with b3sum fallback) |
| `refresh` | Data freshness check against upstream sources |
| `run` | Full pipeline: simulate, finalize, validate |
| `modules/` | Per-module FES results, configs, target specs |
| `structures/` | PDB/GRO coordinate files — full solvated systems (2657-atom free xylose, 92745-atom enzyme complex) |
| `topologies/` | GROMACS topology files (.top/.itp) for direct `gmx grompp` reproducibility |
| `figures/` | Rendered FES landscapes (PNG, pipeline-generated deterministically by `guidestone finalize`) |
| `configs/` | GROMACS MDP + PLUMED input files (flat, cross-referenced) |
| `provenance/` | ferment_transcript.json + braids/ lineage chain |
| `liveSpore.json` | Deployment tracking and software provenance |

## Collaboration Context

This GuideStone supports the **ABG Conformational Energy Landscapes** project — validating molecular docking approaches against conformational FEL computation across CAZyme families.

**Provenance chain:** v0.6.0 (first prototype, May 24 — atom index and substrate errors caught by Alistaire) -> v1.5.0 (corrected, full data, all PASS) -> v1.6.0 (PLUMED-NEST validation + roadmap) -> v1.6.1 (full-data, lithoSpore ingested, agentic pipeline).

**Scope note:** This artifact delivers the FEL half of the FEL-vs-docking comparison (the project's core question: does AutoDock Vina approximate FEL results?). The docking half is future work — this pseudoSpore is the validated prerequisite. See Module 08 for the full roadmap including docking correlation targets.

**Next steps** (see `modules/08_exploration_roadmap/`):
- Tier 1: Free lyxose/glucose/mannose baselines + 2D24 -2/+1 subsites (~10 hr)
- Tier 2: GH11 inverting xylanase (same substrate, different mechanism) + 1E0X covalent intermediate
- Tier 3: AutoDock Vina docking ↔ FEL correlation analysis (all 38 Stoddart conformations)
- Future: QM/MM via CPMD, REST-RECT multi-replica, HPC scaling on TAMU ACES

**Key domain references** (from Alistaire):
- Nin-Hill 2020 (Rovira group thesis) — Ch.3: QM/MM + metadynamics FEL methodology
- Alonso-Gil 2019 (Rovira group thesis) — Ch.2.2-2.4: PBE DFT equations for QM/MM
- PLUMED-NEST carbohydrate hits + GROMACS-PLUMED Google Group for pyranose FEL

## For Agents

Machine-parseable entry points: `scope.toml` (TOML), `validation.json` (JSON), `data.toml` (BLAKE3 manifest), `liveSpore.json` (deployment log). Use `litho ingest-pseudospore` or `nest-validate guidestone validate` for programmatic verification.

## License

AGPL-3.0-or-later
