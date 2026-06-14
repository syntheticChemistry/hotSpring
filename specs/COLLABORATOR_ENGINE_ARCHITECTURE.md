# Collaborator Engine Architecture

**Status**: Active
**Created**: March 31, 2026
**guideStone Standard**: [`infra/wateringHole/GUIDESTONE_STANDARD.md`](../../../infra/wateringHole/GUIDESTONE_STANDARD.md)

---

## Purpose

For each PI in our contact network, hotSpring evolves from paper reproduction
to a production tool tailored to the PI's expertise. The result is a
**Collaborator Engine** — a Composition guideStone that produces data in the
PI's ecosystem format, with full provenance, at the quality level required
for publication and data sharing.

This spec defines the repeatable pattern. The Chuna Engine is the first
completed instance.

---

## The Three-Phase Pattern

```
Phase 1: Reproduce     → Spring guideStone (paper checks, validation/run)
Phase 2: Pivot         → Strategic decision informed by meeting with the PI
Phase 3: Engine        → Composition guideStone (PI's production tool)
```

### Phase 1: Reproduce

Study the PI's published papers. Implement the physics in Rust + WGSL.
Validate against their results. This produces the Spring guideStone —
the `validation/run` artifact that proves our code reproduces known work.

Phase 1 establishes credibility. It answers: *can we do their physics?*

### Phase 2: Pivot

Meet with the PI. Learn what they actually need. The pivot is the strategic
decision to stop proving we can replicate their work and start building a
tool for their work. Key questions:

- What is the PI's core expertise? (Build the workbench for that)
- Who consumes their output? (Build the export for that audience)
- What data formats does their ecosystem use? (Implement native I/O)
- What computations do they run repeatedly? (Memoize and schedule those)

The pivot is informed by the PI, not assumed. Chuna told us he is the
optimization expert and Bazavov needs more MILC data receipts. That
conversation shaped the Chuna Engine.

### Phase 3: Engine

Build the Composition guideStone. This is the production artifact the PI
uses in their actual research workflow. It must satisfy all five guideStone
properties.

---

## guideStone Compliance

Each of the five guideStone properties maps to specific engine requirements:

| Property | Engine Requirement |
|----------|-------------------|
| **1. Deterministic** | CPU-only path produces identical mathematical results. Substrate detection at runtime. No "works on my machine." |
| **2. Reference-Traceable** | Every output number traces to the PI's papers, domain standards (ILDG, QCDml, FAO-56), physical constants (CODATA, PDG), or mathematical proofs. |
| **3. Self-Verifying** | ILDG CRC / CHECKSUMS on all data products. Tampered input detected. Integrity chain from generation through measurement. |
| **4. Environment-Agnostic** | ecoBin compliant. No sudo, no GPU required, no network, no external dependencies. Static musl binary, cross-arch. |
| **5. Tolerance-Documented** | Every threshold has a physical or mathematical derivation in the output metadata. No magic numbers. Domain experts can audit. |

---

## Collaborator Engine Roster

| PI | Domain | Expertise | Phase | Engine | guideStone | Key Deliverable |
|----|--------|-----------|-------|--------|------------|-----------------|
| **Chuna** | Lattice QCD | Flow optimization, integrator design | **3 (Complete)** | Chuna Engine (6 binaries) | Composition | ILDG configs, QCDml 2.0 metadata, flow optimization workbench |
| **Bazavov** | Lattice QCD / MILC | EOS, finite-temperature QCD | 2 (Pivot) | Downstream of Chuna Engine | — | Measurement receipts in MILC-compatible formats |
| **Kachkovskiy** | Spectral theory | Anderson localization, quasiperiodic operators | 1 (Reproduce complete) | Future | — | Spectral localization workbench (Anderson, Hofstadter, almost-Mathieu) |
| **Murillo** | Plasma physics | Dense plasma transport, MD simulations | 1 (Reproduce complete) | Future | — | Transport benchmark engine (Yukawa OCP, DSF, Green-Kubo D*/η*/λ*) |
| **Stanton** | Transport | Transport coefficients, effective potentials | 1 (Reproduce complete) | Future | — | Transport coefficient validation engine (Stanton-Murillo fits) |

Phase 1 completion means we have validated our implementations against
their papers. Phase 2 requires a meeting with the PI to understand their
actual workflow. Phase 3 is the engine build.

---

## Engine Anatomy

Every Collaborator Engine follows the same structural pattern, established
by the Chuna Engine:

### Data I/O in the PI's Ecosystem Format

The engine reads and writes data in formats native to the PI's field.
For lattice QCD, this is ILDG/LIME gauge configurations with QCDml 2.0
metadata. For plasma physics, it would be HDF5 with Sarkas-compatible
structure. The engine is bidirectional — it can load the PI's existing
data and produce output they can feed to their existing tools.

Implementation: [`barracuda/src/lattice/ildg.rs`](../barracuda/src/lattice/ildg.rs),
[`barracuda/src/lattice/lime.rs`](../barracuda/src/lattice/lime.rs),
[`barracuda/src/lattice/qcdml.rs`](../barracuda/src/lattice/qcdml.rs)

### Measurement Provenance

Every output carries implementation provenance: code name, version, machine
name, institution, machine type. This satisfies guideStone Property 2
(Reference-Traceable) and enables reproducible, publishable results.

Implementation: [`barracuda/src/lattice/measurement.rs`](../barracuda/src/lattice/measurement.rs)
(`ImplementationInfo`, `ScaleSetting`, `StatisticalAnalysis`)

### Process Catalog with Cost Model

Every computation the engine performs is a node in a dependency DAG with a
known cost model calibrated from benchmark data. This enables the PI to
plan parameter sweeps with accurate time/cost estimates before committing
HPC resources.

Implementation: [`barracuda/src/lattice/process_catalog.rs`](../barracuda/src/lattice/process_catalog.rs)
(`PhysicsProcess`, `CostModel`, `ProcessCatalog`)

### Task Matrix for Systematic Sweeps

A persistent grid of (ensemble_params x observable_set) cells with
priority scheduling. Supports both continuous sweeps (background
production) and on-demand targeted requests (urgent measurements from
collaborators). Serializes to JSON and resumes across sessions.

Implementation: [`barracuda/src/lattice/task_matrix.rs`](../barracuda/src/lattice/task_matrix.rs)
(`TaskMatrix`, `SweepParams`, dependency-aware `next_ready()`)

### Export for Downstream Consumers

The engine produces output that the PI's downstream consumers can use
directly. For the Chuna Engine, Bazavov is the downstream consumer, so
the export produces QCDml 2.0 XML packages. For a future Murillo Engine,
the downstream consumer might be the Dense Plasma Properties Database.

Implementation: [`barracuda/src/bin/chuna_matrix.rs`](../barracuda/src/bin/chuna_matrix.rs)
(`export` subcommand)

---

## Worked Example: Chuna Engine

The Chuna Engine is the reference implementation of this pattern.

### Phase 1: Reproduce (Papers 43-45)

- Gradient flow with LSCFRK integrators (Paper 43)
- BGK dielectric function (Paper 44)
- Kinetic-fluid coupling (Paper 45)
- 44/44 checks pass. Spring guideStone achieved.
- Artifact: `validation/run`

### Phase 2: Pivot (March 31, 2026)

Meeting with Chuna revealed:
- He is the optimization expert — build a flow workbench, not a paper demo
- Bazavov always needs more MILC data — formalize our measurement output
- Chuna heading to LANL — needs ILDG/LIME interop for HPC data exchange
- The artifact should serve his work, not validate ours

### Phase 3: Engine (6 binaries)

| Binary | Purpose | guideStone Property Served |
|--------|---------|---------------------------|
| `chuna_generate` | Thermalized ILDG gauge configs + QCDml XML | 1 (Deterministic), 2 (Traceable), 3 (Self-Verifying) |
| `chuna_flow` | Gradient flow on any ILDG config | 2 (Traceable — papers cited in output) |
| `chuna_measure` | Full observable suite with provenance | 2, 5 (Traceable, Tolerance-Documented) |
| `chuna_convert` | ILDG/LIME import/export, QCDml emission | 3 (Self-Verifying — CRC on all data) |
| `chuna_benchmark_flow` | Integrator efficiency workbench | Chuna's core expertise |
| `chuna_matrix` | Task orchestration for production sweeps | Cost model, dependency scheduling |

Entry point: [`validation/chuna-engine`](../validation/chuna-engine)

Composition guideStone name: `hotSpring-guideStone-v0.7.0`

---

## What a Collaborator Engine Is NOT

- **Not a paper demo** — the PI does not need us to replay their published results
- **Not our internal validation suite** — `validation/run` handles that; the engine is for the PI's use
- **Not a general-purpose library** — it is scoped to the PI's domain and workflow
- **Not a replacement for the PI's existing tools** — it complements them (different hardware, same formats)
- **Not required for every PI** — only build an engine when Phase 2 reveals a clear production use case

---

## Future Engine Candidates

### Kachkovskiy Engine (Spectral Theory)

Phase 1 is complete (45/45 checks: Anderson 1D/2D/3D, Hofstadter, GPU
Lanczos). Phase 2 requires a meeting to understand what Kachkovskiy
actually needs. Possible directions:

- Localization length computation workbench
- Eigenvalue statistics export for mathematical analysis
- Large-scale GPU Lanczos for quasiperiodic operators

### Murillo Engine (Plasma Transport)

Phase 1 is complete (60/60 dense plasma MD, 13/13 transport). Phase 2
requires understanding Murillo's current research needs. Possible
directions:

- Transport coefficient benchmark engine (D*, η*, λ* across parameter space)
- DSF comparison workbench (MD vs dielectric models)
- Dense plasma properties database integration

### Stanton Engine (Transport Coefficients)

Phase 1 is complete (13/13 Stanton-Murillo transport checks). Could be
folded into the Murillo Engine or stand alone depending on Phase 2 findings.

---

## Relationship to Other Standards

| Standard | Relationship |
|----------|-------------|
| **guideStone** | A Collaborator Engine is a Composition guideStone. The five properties are the quality bar. |
| **External Validation Artifact** | Phase 1 produces the validation artifact (`validation/run`). Phase 3 supersedes it with the engine. |
| **ecoBin** | Engines are ecoBin-compliant: static musl, cross-arch, no runtime deps. |
| **Provenance Trio** | When wired, engine output + trio = permanently anchored, attributed data. |
