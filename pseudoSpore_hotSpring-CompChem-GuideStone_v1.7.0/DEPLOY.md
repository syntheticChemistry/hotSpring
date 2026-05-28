# Deployment Guide — pseudoSpore v1.7.0

## Overview

This pseudoSpore is a self-verifying computational chemistry artifact containing:
- 8 baseline modules from v1.6.1 (free/enzyme-bound xylose, PLUMED-NEST, roadmap)
- 8 free sugar epimer FEL modules (lyxose/glucose/mannose/galactose, 1D+2D each)
- 4 GH10 subsite analysis modules (-2 and +1 subsites, 1D+2D each)
- 2 GH11 inverting xylanase modules (1XYN -1 subsite, 1D+2D)
- 185/187 validation checks PASS, ~420 ns total simulation time

## Deployment Paths

### 1. Local (Alistaire Handoff)

Untar and self-check:

```bash
tar -xzf pseudoSpore_hotSpring-CompChem-GuideStone_v1.7.0.tar.gz
cd pseudoSpore_hotSpring-CompChem-GuideStone_v1.7.0/

# Self-verify integrity (BLAKE3 check against data.toml)
./validate

# Regenerate FES from HILLS data (requires GROMACS + PLUMED + nest-validate)
./refresh

# Full pipeline re-execution (simulation → finalize → validate)
./run
```

**Requirements for full reproduction:**
- GROMACS 2026.0 (conda-forge or system)
- PLUMED 2.10 (patched into GROMACS)
- `nest-validate` binary (Rust, from `control/plumed_nest/nest-validate/`)
- `cazyme-fel` crate (validation library)
- ~8 GB RAM, ~50 GB disk for full simulation artifacts
- Estimated compute: 4-5 hours (6-core CPU) for full re-run

**Read-only inspection (no software required):**
- `scope.toml` — full module index with methods, CVs, and validation status
- `modules/*/target.toml` — per-module metadata
- `modules/07_plumed_nest_validation/summary.md` — human-readable PLUMED-NEST status
- `modules/08_exploration_roadmap/CAZYME_FEL_EXPLORATION_TARGETS.md` — future directions

### 2. primals.eco (Web Publication)

Push summary artifacts to `sporeprint/` for the CI auto-refresh pipeline:

```bash
# From hotSpring root
cp pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/liveSpore.json sporeprint/compchem/
cp pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/scope.toml sporeprint/compchem/
cp pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/modules/07_plumed_nest_validation/summary.md sporeprint/compchem/validation_summary.md
cp pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/modules/08_exploration_roadmap/CAZYME_FEL_EXPLORATION_TARGETS.md sporeprint/compchem/
```

The `primals.eco` CI pipeline will:
1. Detect new artifacts in `sporeprint/`
2. Render markdown to the website
3. Publish BLAKE3 integrity metadata for external verification
4. Generate a DOI-ready landing page (when ready for formal publication)

### 3. cellMembrane VPS (Geo-Delocalized Deployment)

For remote validation and data distribution via the cellMembrane infrastructure:

**Channel 3 TLS on `membrane.primals.eco`:**
```bash
# Upload pseudoSpore to cellMembrane staging
scp pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1.tar.gz \
  songbird@membrane.primals.eco:/opt/litho/staging/

# Remote validation (SSH or via Songbird TURN relay)
ssh songbird@membrane.primals.eco \
  "cd /opt/litho/staging && tar -xzf *.tar.gz && cd pseudoSpore_* && ./validate"
```

**Songbird TURN Relay (Tier 2 — Geo-Delocalized):**
- Enables validation by remote collaborators without direct SSH
- TURN relay handles NAT traversal for distributed verification
- Integrity chain preserved via BLAKE3 data.toml checksums

**Future: Full lithoSpore Deployment (Tier 3):**
- Promote to lithoSpore via `litho promote` (requires NUCLEUS)
- Embeds a minimal NUCLEUS runtime for sovereign execution
- USB/portable media deployment with self-contained validation
- Target: any machine with Rust toolchain can verify and reproduce

### 4. NUCLEUS Composition (lithoSpore Promotion)

If the recipient has a NUCLEUS instance:

```bash
# Promote pseudoSpore to full lithoSpore (self-contained, sovereign)
litho promote pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/ \
  --output lithoSpore_hotSpring-CompChem-GuideStone_v1.6.1.img

# Deploy on USB or transfer
litho audit lithoSpore_hotSpring-CompChem-GuideStone_v1.6.1.img
```

The `[artifact]` key in `scope.toml` ensures `litho promote` can consume this pseudoSpore directly (backward-compatible shim alongside `[guidestone]`).

### 5. NUCLEUS Nest Deployment (postPrimordial)

Full NUCLEUS nest deployment with provenance trio signing. This is the target path for pseudoSpore 2.0 and requires the biomeOS CLI, a live Nest Atomic, and the provenance trio services.

```bash
# Deploy with NUCLEUS ingest (postPrimordial path)
nest-validate guidestone deploy \
  pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/ \
  --nucleus

# The pipeline will:
#   1-6: Standard pipeline (hash → validate → emit → hash → verify → package)
#   7:   biomeos nucleus ingest → NestGate content-addressed store
#        Falls back to litho ingest if biomeos is unavailable
```

**NUCLEUS Ingest Flow:**

```
biomeos nucleus ingest <pseudoSpore-dir> --verify
  → NestGate content-addressed store (receives envelope + data)
  → rhizoCrypt: DAG merkle root verified against BLAKE3 chain
  → loamSpine: spine_id registered in ledger with braid linkage
  → sweetGrass: attribution braid linked to lineage chain + bibliography DOIs
  → receipts/nucleus_ingest.toml written with gate + timestamp
```

**Provenance Trio Signing Contract:**

| Primal | Role | Contract |
|--------|------|----------|
| rhizoCrypt | DAG merkle integrity | `dag_merkle_root` matches BLAKE3 chain; DAG session registered |
| loamSpine | Ledger registration | `spine_id` registered in loamSpine ledger with `braid_id` and parent linkage |
| sweetGrass | Attribution braid | Lineage chain + bibliography DOIs + PLUMED-NEST plum_ids linked |

**Prerequisites:**
- `biomeos` CLI binary (in PATH or `plasmidBin`)
- Live Nest Atomic with NestGate, rhizoCrypt, loamSpine, sweetGrass services
- cellMembrane network permeability to NUCLEUS VPS (if remote)
- All standard pipeline artifacts present (scope.toml, data.toml, liveSpore.json)

**Current Status:** Slots are wired and pending. The deploy pipeline detects biomeos availability and falls back gracefully. primalSpring exp115 will validate the live path.

## Wire Format Compatibility

| Consumer | Key Required | Status |
|----------|-------------|--------|
| nest-validate | `[guidestone]` | Present ✓ |
| litho promote | `[artifact]` | Present ✓ (v1.6.1 shim) |
| biomeos nucleus ingest | `liveSpore.json` + `data.toml` | Present ✓ |
| sporeprint CI | `liveSpore.json` | Present ✓ |
| BLAKE3 verify | `data.toml` | Present ✓ |

## Data Provenance — Three Eras

```
Era 1: Ad-Hoc (v1.0.0 — v1.6.0)
  Hand-authored scope.toml blindly copied to validation.json and liveSpore.json

  lithoSpore_handoff_v0.7.0
    → pseudoSpore v1.5.0 (GuideStone promotion, 46/46 PASS)
      → pseudoSpore v1.6.0 (Alistaire data drop + PLUMED-NEST aggregate + exploration roadmap)

Era 2: Pipeline-Derived (v1.6.1)
  Every metadata value extracted from authoritative data files and cross-checked

        → pseudoSpore v1.6.1 (pipeline-derived metadata + agentic deploy)

Era 3: NUCLEUS Nest Deploy (v2.0+ target)
  Provenance trio independently signs via biomeos nucleus ingest

          → pseudoSpore v2.0 (provenance trio signed, NUCLEUS-registered)
```

Parent Merkle: `cbf908fb4c36d036ab9da1ffdac775a97dff6a0f640fb5706f6f7505a3b9bbea`
