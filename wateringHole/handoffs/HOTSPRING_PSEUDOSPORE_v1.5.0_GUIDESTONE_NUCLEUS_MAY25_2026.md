# hotSpring Handoff: pseudoSpore v1.5.0 / lithoSpore v2.3.0 + NUCLEUS biomeGate

**Spring:** hotSpring  
**Date:** May 25, 2026  
**Wave:** 50 (post-primordial)  
**Supersedes:** `archive/HOTSPRING_CAZYME_FEL_EVOLUTION_MAY24_2026.md` (v1.1.1 / v2.1.0)

---

## Status

| Component | Version | Status |
|-----------|---------|--------|
| pseudoSpore | v1.5.0 | GuideStone-grade, 12/12 audit PASS |
| lithoSpore chassis | v2.3.0 | Domain Profiles, automated emit/audit/promote |
| NUCLEUS biomeGate | niche-hotspring | 9 primals launched, Songbird registry seeded |
| plasmidBin compliance | Wave 50 | All primals sourced exclusively from plasmidBin |
| `litho audit` | 12 checks | Core + domain packs, `--json` structured output |

## Evolution Since v1.1.1

The pseudoSpore artifact evolved through 8 versions since the May 24 handoff:

| Version | Key Addition |
|---------|-------------|
| v1.2.0 | Audit-driven fixes (H1-H3, M1-M3, L1-L4); pipeline automation |
| v1.2.1 | Automated detection/prevention of all v1.2.0 finding classes |
| v1.3.0 | Domain Profiles — agnostic deployment infrastructure |
| v1.4.0 | GuideStone data chassis: `data.toml`, `liveSpore.json`, `tolerances.toml`, `validate`/`refresh` |
| v1.4.1 | Auto-populated `scope.toml` + `environment.toml` from live system probes |
| v1.4.2 | Emit host vs simulation host clarity; project version reporting |
| v1.5.0 | Human-readable README rewrite (science summary, quick start, file inventory) |

## pseudoSpore v1.5.0 Contents

```
pseudoSpore_hotSpring-CAZyme-FEL_v1.5.0/
├── README.md              # Science summary + quick start for humans
├── scope.toml             # 5 modules auto-discovered, paper DOI, origin
├── environment.toml       # Emit host hardware/software snapshot
├── domain_profile.toml    # Declarative domain config (carbohydrate-FEL)
├── data.toml              # Data manifest (BLAKE3 hashes, sources, licenses)
├── tolerances.toml        # Named tolerances with scientific justification
├── liveSpore.json         # Deployment provenance trail
├── validate               # Root entry: airgapped BLAKE3 validation
├── refresh                # Root entry: data freshness check + re-pull
├── index_map.toml         # Domain↔computation index translation
├── TRANSLATE.md           # Cross-reference legend
├── data/                  # 5 modules: free_xylose_1d, enzyme_bound_1d, ...
├── figures/               # Auto-generated FEL heatmaps + convergence
├── configs/               # GROMACS MDP + PLUMED input files
└── provenance/            # FermentBraid chain (BLAKE3 receipts)
```

## NUCLEUS biomeGate Deployment

- **Family:** hotspring-biome  
- **Node:** pop-os  
- **Composition:** niche-hotspring (9 primals)
- **Source:** plasmidBin-only (post-primordial Wave 50 mandate)
- **Registry:** Songbird `ipc.discover` operational — resolves tensor/compute/shader/crypto

### Healthy primals: beardog, songbird, toadstool, coralreef, sweetgrass  
### Registered: beardog, toadstool, barracuda, coralreef, nestgate, rhizocrypt, loamspine, sweetgrass

### primalSpring Validation
- `exp091_primal_routing_matrix`: **PASS**
- `exp094_composition_parity`: **PASS**

## Gaps (Active)

| ID | Description | Severity | Owner |
|----|-------------|----------|-------|
| GAP-HS-111 | barraCuda bonded FF + topology + metadynamics | Medium | hotSpring |
| GAP-HS-112 | petalTongue FEL visualization integration | Low | petalTongue |

## Next Phase

- Covalent HPC: Songbird mesh for distributed GPU dispatch
- Alistaire data handoff: full lithoSpore airgapped conference deployment
- PLUMED-NEST alignment: reproducibility rubric from published carbohydrate FEL entries
- Nin-Hill 2020 (Rovira group): DFT-based galactose FEL as validation target

## For Upstream (primalSpring)

- Wave 50 mandate absorbed: `target/release` hardcodes eliminated from all active scripts
- `start_primal.sh` patched to discover binaries in `$primal/$primal` layout
- `validate_composition.sh` patched similarly
- coralReef freshly built and staged into plasmidBin
- Request: skunkBat binary for full NUCLEUS composition
