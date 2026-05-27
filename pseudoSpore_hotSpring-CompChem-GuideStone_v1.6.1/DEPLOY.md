# Deployment Guide — pseudoSpore v1.6.1

## Overview

This pseudoSpore is a self-verifying computational chemistry artifact containing:
- 6 validated CAZyme FEL modules (46/46 checks PASS)
- 1 PLUMED-NEST validation aggregate (8 targets, 2 validated, 6 profiled)
- 1 exploration roadmap (13 proposed targets across 3 priority tiers)

## Deployment Paths

### 1. Local (Alistaire Handoff)

Untar and self-check:

```bash
tar -xzf pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1.tar.gz
cd pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/

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

## Wire Format Compatibility

| Consumer | Key Required | Status |
|----------|-------------|--------|
| nest-validate | `[guidestone]` | Present ✓ |
| litho promote | `[artifact]` | Present ✓ (v1.6.1 shim) |
| sporeprint CI | `liveSpore.json` | Present ✓ |
| BLAKE3 verify | `data.toml` | Present ✓ |

## Data Provenance

```
lithoSpore_handoff_v0.7.0
  → pseudoSpore v1.5.0 (GuideStone promotion, 46/46 PASS)
    → pseudoSpore v1.6.0 (Alistaire data drop + PLUMED-NEST aggregate + exploration roadmap)
      → pseudoSpore v1.6.1 (full-data + litho ingest compatibility + agentic pipeline)
```

Parent Merkle: `cbf908fb4c36d036ab9da1ffdac775a97dff6a0f640fb5706f6f7505a3b9bbea`
