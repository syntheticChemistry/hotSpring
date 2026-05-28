# CompChem Explorer — sporeGarden Product Specification

**Version:** 0.1 (architectural draft)
**Date:** May 28, 2026
**Status:** Pre-alpha — composition contract defined, no product binary yet
**Pattern:** sporeGarden gen4 (BYOB, PrimalBridge, graceful degradation)
**Domain:** Computational chemistry — enzyme conformational landscapes

---

## Product Identity

| Field | Value |
|-------|-------|
| **Name** | CompChem Explorer |
| **Org** | sporeGarden (gen4 product tier) |
| **Spring** | hotSpring (science validation) |
| **Domain** | CAZyme conformational free energy landscapes |
| **Audience** | Computational chemists studying enzyme catalysis |
| **Analog** | esotericWebb is to ludoSpring as CompChem Explorer is to hotSpring |

### What It Does

An interactive 3D molecular research tool that lets a chemist:
1. View conformational energy landscapes (FELs) as navigable 3D surfaces
2. Manipulate ring puckering in real-time and see the energy cost
3. Compare free vs enzyme-bound conformational preferences
4. Launch new simulations and monitor them live
5. Share validated results as pseudoSpores via NUCLEUS

### Why It Exists

The pseudoSpore (v1.7.0, 190/190 checks) proves the science is correct.
The pipeline (`nest-validate`) proves the compute is reproducible.
What's missing: **a human can't interact with it.** They get static figures
and TOML files. CompChem Explorer makes the validated computation tangible —
rotate a sugar ring in 3D, feel the energy barriers, compare enzymes visually.

---

## Composition Contract

### Particle Profile

**Proton-heavy** — compute-dominated. The product's value comes from running
expensive GPU calculations (FES reconstruction, MD force evaluation) and
displaying results. Storage and security are infrastructure, not features.

### Mixed Atomic Pattern

**Node + Dedicated Tower** — GPU compute (toadStool + barraCuda) is the
core resource. Tower (bearDog + songbird) provides encrypted transport for
signed pseudoSpore artifacts and collaborative sharing.

### Bonding Pattern

| Bond | Between | Pattern |
|------|---------|---------|
| Ionic lease | User session ↔ compute resources | Session-scoped GPU allocation |
| Covalent | Experiment ↔ provenance trio | Permanent, sealed lineage |
| Metallic | NUCLEUS internal | Standard mesh |

---

## Graceful Degradation (Three Tiers)

### Tier 1 — Standalone (no NUCLEUS, no GPU)

Available with just the product binary + a pseudoSpore directory:

- Browse pseudoSpore modules (scope.toml, validation.json)
- View pre-rendered FEL figures (PNG from `figures/`)
- Display 1D energy profiles (text-mode plot from `fes_*.dat`)
- Static PDB structure display (petalTongue headless → SVG/PNG)
- Validate integrity (`b3sum` check against checksums.blake3)

**Exit behavior:** Fully functional for reviewing results. No compute.

### Tier 2 — Node Atomic (ToadStool + BarraCuda + petalTongue)

Available when GPU compute and visualization primals are live:

- **Live FES reconstruction:** Upload HILLS file → GPU Gaussian summation → 
  real-time surface rendering (Paper 50 shader: 11-14x, RMSD 1e-14)
- **Interactive Cremer-Pople sphere:** 3D sphere where theta/phi map to ring
  conformations, colored by energy. Orbit camera, click a point → see ring shape.
- **Parameter exploration:** Vary biasfactor, sigma, pace — see how the FEL
  changes. Understand convergence interactively.
- **Cross-landscape comparison:** Side-by-side or difference-map rendering of
  free vs enzyme-bound surfaces. Animated interpolation between states.
- **Real-time binding distance:** Stream d_bind from live COLVAR → annotate
  on the FEL where the substrate currently sits.

**PrimalBridge methods consumed:**

| Method | Provider | Purpose |
|--------|----------|---------|
| `compute.dispatch.submit` | toadStool | GPU shader execution |
| `math.tessellate.isosurface` | barraCuda | FEL surface mesh generation |
| `math.project.perspective` | barraCuda | 3D projection math |
| `visualization.render.scene` | petalTongue | Scene graph rendering |
| `visualization.render.stream` | petalTongue | Frame streaming (60 Hz) |
| `interaction.poll` | petalTongue | User input (orbit, click, select) |
| `interaction.subscribe` | petalTongue | Event stream for navigation |

### Tier 3 — Full NUCLEUS (+ NestGate + Provenance + External Tools)

Everything in Tier 2, plus:

- **Launch simulations:** Orchestrate GROMACS+PLUMED via nest-validate from
  the GUI. Pick substrate, enzyme, method → submit → monitor.
- **Live monitoring:** Streaming COLVAR visualization during production MD.
  Binding distance, puckering angle, hill deposition — all real-time.
- **Provenance sealing:** Each experiment session gets a DAG (rhizoCrypt),
  ledger entry (loamSpine), and attribution braid (sweetGrass). The session
  is cryptographically linked to the input structures and parameters.
- **Collaborative sharing:** Emit pseudoSpore → NUCLEUS ingest → colleague
  on another gate ingests → validates independently → results are comparable.
- **AI-assisted analysis:** Squirrel can summarize FEL features, suggest
  next simulations, compare against literature baselines.

**Additional PrimalBridge methods:**

| Method | Provider | Purpose |
|--------|----------|---------|
| `storage.store` | NestGate | Persist pseudoSpore artifacts |
| `storage.retrieve` | NestGate | Load colleague's shared data |
| `dag.session.create` | rhizoCrypt | Experiment lineage |
| `entry.append` | loamSpine | Permanent ledger |
| `braid.create` | sweetGrass | Attribution and provenance |
| `crypto.sign` | BearDog | Receipt signing |
| `ai.query` | Squirrel | Analysis suggestions |

---

## petalTongue Integration

CompChem Explorer is the **first consumer** of petalTongue Phase 4
(molecular visualization). This drives implementation priority.

### Required petalTongue Capabilities

| Capability | Phase | Status |
|-----------|-------|--------|
| `Perspective3DCoord` + orbit camera | Phase 4 | Specified, not implemented |
| `GeomSphere` (atom rendering) | Phase 4 | Specified |
| `GeomCylinder` (bond rendering) | Phase 4 | Specified |
| `Navigate3D` (orbit/fly) | Phase 5 | Specified |
| `visualization.render.scene` | Current | Implemented (game scenes) |
| `visualization.render.stream` | Current | Implemented |
| `interaction.poll` | Current | Implemented |
| `DataBinding::Scatter3D` | Current | Implemented |

### Data Channels

| Channel | Data | Rate |
|---------|------|------|
| `Scatter3D` | Atom positions (x, y, z, element, radius) | Per frame (static) |
| `FieldMap` | FEL surface (theta, phi, energy grid) | On recompute (~1 Hz) |
| `GameScene` | Full molecular scene graph | 60 Hz (interactive) |

### Molecular Visualization Pipeline

```
PDB/GRO file
  → parse atoms + bonds (cazyme-fel or MDAnalysis)
    → DataBinding::Scatter3D (positions, elements, radii)
      → petalTongue GeomSphere + GeomCylinder
        → Perspective3DCoord (orbit camera)
          → visualization.render.scene → frame buffer
```

### FEL Surface Pipeline

```
HILLS file (or fes_*.dat)
  → barraCuda FES Gaussian summation (GPU)
    → 2D grid (theta × phi × energy)
      → barraCuda math.tessellate.isosurface → mesh
        → petalTongue GeomMesh3D
          → Perspective3DCoord → visualization.render.scene
```

---

## Science Pipeline Integration

The product wraps `nest-validate` as its backend orchestrator:

| Product Action | nest-validate Command | What Happens |
|---------------|----------------------|--------------|
| "New Simulation" | `guidestone run` | GROMACS+PLUMED production MD |
| "Rebuild FEL" | `guidestone finalize` | FES reconstruction + figures |
| "Validate" | `guidestone validate` | 190-check validation suite |
| "Package Results" | `guidestone deploy` | Emit pseudoSpore |
| "Share" | `guidestone deploy --nucleus` | NUCLEUS ingest |

The product binary itself does NOT implement MD. It orchestrates existing
validated tools and presents results interactively.

---

## Deploy Graph Stack

Six tiered deploy graphs, each building on the previous:

| Graph | Fragments | Primals | Unlocks |
|-------|-----------|---------|---------|
| `compchem_tower.toml` | Tower | bearDog, songbird, skunkBat | Signed transport |
| `compchem_node.toml` | Tower + Node | + toadStool, barraCuda, coralReef | GPU compute |
| `compchem_nest.toml` | Tower + Node + Nest | + NestGate | Persistent storage |
| `compchem_viz.toml` | Tower + Node + petalTongue | + petalTongue (live mode) | 3D molecular viz |
| `compchem_provenance.toml` | Full Nest | + rhizoCrypt, loamSpine, sweetGrass | Sealed experiments |
| `compchem_full.toml` | Full NUCLEUS | + Squirrel | AI analysis |

**Launch:** `biomeos deploy --graph graphs/compchem_full.toml`

---

## Niche Definition

```yaml
name: compchem-explorer
version: "0.1.0"
description: "Interactive CompChem FEL explorer — 3D molecular visualization + GPU compute"
spring: hotSpring
domain: computational-chemistry

composition:
  minimum: [beardog, songbird]
  recommended: [beardog, songbird, toadstool, barracuda, petaltongue]
  full: [beardog, songbird, skunkbat, toadstool, barracuda, coralreef, petaltongue, nestgate, rhizocrypt, loamspine, sweetgrass, squirrel]

petaltongue:
  mode: live
  tick_hz: 60
  data_channels: [Scatter3D, FieldMap, GameScene]

degradation:
  tier1: "Static figure viewer + pseudoSpore browser (no primals)"
  tier2: "GPU compute + interactive viz (Node + petalTongue)"
  tier3: "Full NUCLEUS: live simulations, provenance, collaboration"

science:
  pipeline: nest-validate
  artifact: pseudoSpore_hotSpring-CompChem-GuideStone
  domain_profile: compchem-enhanced-sampling
  validation_standard: derivation-anchoring-v1.0
```

---

## Evolution Roadmap

### Phase 0 — Current State (May 2026)

What exists today, usable without any new code:

- pseudoSpore v1.7.0: 16 FEL landscapes, 190/190 checks
- nest-validate: full orchestration (run/finalize/validate/deploy)
- cazyme-fel: Rust FES reconstruction, cross-landscape analysis
- Paper 50 GPU shader: proven 11-14x acceleration
- petalTongue: scene rendering, interaction polling (game scenes)
- NUCLEUS: 13 primals deployed, IPC routing validated

### Phase 1 — FEL Viewer (target: v0.2)

Minimum viable product — view existing pseudoSpore data interactively:

- [ ] Parse `fes_*.dat` files into renderable grids
- [ ] 2D heatmap rendering via petalTongue `FieldMap`
- [ ] PDB structure display via `Scatter3D` (atom positions)
- [ ] Module browser (list modules from scope.toml, show status)
- [ ] Static — no live compute, just visualization of existing data

**Blocks on:** petalTongue `FieldMap` data channel rendering

### Phase 2 — Interactive Compute (target: v0.4)

Add live GPU computation:

- [ ] HILLS → FES pipeline via barraCuda GPU dispatch
- [ ] Interactive parameter sliders (biasfactor, sigma, grid resolution)
- [ ] Cross-landscape difference maps (free vs bound, animated)
- [ ] Cremer-Pople sphere navigation (3D energy surface)

**Blocks on:** petalTongue Phase 4 (Perspective3DCoord, GeomSphere)

### Phase 3 — Live Simulation (target: v0.6)

Full orchestration loop:

- [ ] Launch GROMACS+PLUMED from the GUI
- [ ] Stream COLVAR in real-time → annotate on FEL surface
- [ ] Binding distance live alarm (substrate dissociation detection)
- [ ] Emit pseudoSpore from completed simulation
- [ ] Provenance sealing per session

**Blocks on:** biomeOS `nucleus ingest` wiring, petalTongue Phase 5

### Phase 4 — Collaborative & Publication (target: v1.0)

Full product for multi-site collaboration:

- [ ] Share pseudoSpore via NUCLEUS mesh
- [ ] Ingest colleague's results → validate → compare
- [ ] Generate publication-ready figures from interactive sessions
- [ ] AutoDock Vina integration (FEL ↔ docking correlation)
- [ ] Multi-enzyme family comparison dashboard

**Blocks on:** Stadial gate, NC-5.live, cross-gate routing

---

## Upstream Absorption Targets

| Target | Owner | What We Need |
|--------|-------|-------------|
| petalTongue Phase 4 | petalTongue team | `Perspective3DCoord`, `GeomSphere`, `GeomCylinder`, `GeomMesh3D` |
| petalTongue Phase 5 | petalTongue team | `Navigate3D`, 3D orbit/fly interaction |
| barraCuda tessellation | barraCuda team | `math.tessellate.isosurface` for FEL surface meshes |
| biomeOS nucleus ingest | biomeOS team | CLI wiring for `biomeos nucleus ingest <dir>` |
| coralReef FEL shader | coralReef team | WGSL compilation of `fes_gaussian_sum_f64.wgsl` to native ISA |

---

## Relationship to Existing Work

| Existing | Role in Product |
|----------|----------------|
| pseudoSpore v1.7.0 | Reference data artifact (16 FELs, calibrated thresholds) |
| nest-validate | Backend orchestrator (run, finalize, validate, deploy) |
| cazyme-fel crate | Core science library (FES reconstruction, cross-landscape, KS test) |
| Paper 50 GPU shader | Tier 2 acceleration (Gaussian summation on GPU) |
| SOVEREIGN_COMPCHEM_EVOLUTION.md | Roadmap to sovereign all-atom MD (Papers 51-58) |
| CAZYME_FEL_EXPLORATION_TARGETS.md | Scientific target list for Tier 3+ |
| Derivation Anchoring Standard | Quality gate — all product thresholds inherit this |

---

## Decision: Product vs Library

CompChem Explorer is a **product** (sporeGarden tier), not a library or tool:

- It has a **user interface** (petalTongue live mode)
- It **degrades gracefully** (runs without NUCLEUS)
- It **consumes primals** (never imported as a crate)
- It produces **user-facing artifacts** (figures, pseudoSpores, sessions)
- It follows the **BYOB pattern** (binaries from plasmidBin)

The existing hotSpring crate (`barracuda/`) and `nest-validate` remain as
the science validation layer. The product composes them — it does not replace
or duplicate them.

---

## References

- esotericWebb pattern: `gardens/esotericWebb/graphs/`, `primalSpring/graphs/cells/esotericwebb_cell.toml`
- petalTongue molecular spec: `primals/petalTongue/specs/GRAMMAR_OF_GRAPHICS_ARCHITECTURE.md`
- Composition onramp: `infra/wateringHole/GARDEN_COMPOSITION_ONRAMP.md`
- Gen4 patterns: `infra/whitePaper/gen4/architecture/COMPOSITION_PATTERNS.md`
- Sovereign CompChem: `hotSpring/specs/SOVEREIGN_COMPCHEM_EVOLUTION.md`
- Exploration roadmap: `pseudoSpore v1.7.0/modules/08_exploration_roadmap/`
- Derivation Anchoring: `infra/wateringHole/DERIVATION_ANCHORING_STANDARD.md`
- NUCLEUS validation: `primalSpring/specs/NUCLEUS_VALIDATION_MATRIX.md` (columns U-X)
