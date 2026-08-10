# pseudoSpore Science Pipeline — strandGate Integration Spec

**Date:** Aug 10, 2026  
**Status:** Active  
**Scope:** Defines strandGate's pseudoSpore pipeline and its convergence with ironGate (NFT/results), westGate (CAS), and sporePrint (public trust surface).

---

## Architecture

```
strandGate (compute)          ironGate (NFT/results)          westGate (CAS)
────────────────────          ─────────────────────          ──────────────────
arxiv_production_campaign     NFT registration endpoint       Content-addressable store
        │                              ▲                              ▲
        ▼                              │                              │
arxiv_analysis                         │                              │
        │                              │                              │
        ▼                              │                              │
pseudospore_manifest                   │                              │
        │                              │                              │
        ▼                              │                              │
pseudospore_bundle ────────────────────┼──────────────────────────────┤
        │                              │                              │
        ▼                              │                              │
bearDog sign ──────────────────────────┤                              │
        │                              │                              │
        ▼                              ▼                              ▼
ironGate register ◄─── NFT entry ──── CAS ingest ────► westGate store
        │
        ▼
sporePrint (public page) ◄─── /pseudospore/<hash> ───► download URL
```

---

## Pipeline Stages

### Stage 1: Compute (strandGate owns)

| Binary | Input | Output | Status |
|--------|-------|--------|--------|
| `arxiv_production_campaign` | GPU + parameters | `production_v2/*.json` + `*.lat` | ACTIVE (27/45) |
| `arxiv_analysis` | `production_v2/*.json` | markdown tables, JSON summary | READY |

### Stage 2: Manifest (strandGate owns)

| Binary | Input | Output | Status |
|--------|-------|--------|--------|
| `pseudospore_manifest` | `production_v2/*` | scope.toml, checksums.blake3, environment.toml, validation.json | SCAFFOLDED |
| `pseudospore_validate` | manifest dir | pass/fail (BLAKE3 verification) | SCAFFOLDED |
| `pseudospore_bundle` | manifest + data | `.tar.gz` pseudoSpore bundle | SCAFFOLDED |

### Stage 3: Sign (bearDog — strandGate calls, bearDog owns key)

| Step | Protocol | Status |
|------|----------|--------|
| Compute BLAKE3 root hash over checksums.blake3 | Local | Ready (blake3 crate) |
| Request Ed25519 signature from bearDog | IPC: `crypto.sign` | Needs bearDog endpoint |
| Embed signature in bundle as `provenance/signature.ed25519` | Local | Scaffolded |

### Stage 4: Register (ironGate owns, strandGate pushes)

| Step | Protocol | Status |
|------|----------|--------|
| Push bundle hash + metadata to ironGate NFT registry | IPC: `nft.register` | Needs ironGate endpoint |
| ironGate stores NFT entry (hash, scope, signature) | ironGate internal | Not started |
| ironGate exposes `/pseudospore/<hash>` verification page | HTTP | Not started |

### Stage 5: Store (westGate owns, strandGate pushes)

| Step | Protocol | Status |
|------|----------|--------|
| `content.ingest` tarball to westGate CAS | IPC → TCP :7800 | westGate READY (per pen test) |
| westGate indexes by content hash | Internal | Already working |
| `content.locate` returns CAS address | IPC query | Already working |

### Stage 6: Publish (sporePrint owns)

| Step | Protocol | Status |
|------|----------|--------|
| QCD Rung 1 page on sporePrint | Zola template | Partially |
| Download link resolves to westGate CAS or direct | HTTP redirect | Not started |
| Validation instructions on page | Static content | Not started |

---

## Convergence Points

These are the interfaces strandGate scaffolds **now**, so other gates can wire them:

### 1. bearDog IPC — `crypto.sign`

```toml
[request]
method = "crypto.sign"
payload_blake3 = "abc123..."   # BLAKE3 of the file to sign
signer = "strandGate"          # Requesting gate identity

[response]
signature = "base64-ed25519..."
public_key = "base64-ed25519-pub..."
timestamp = "2026-08-10T15:30:00Z"
```

**Pattern**: strandGate computes the hash locally, sends only the hash over IPC. bearDog never sees the data. This pattern abstracts to any gate needing a signature.

### 2. ironGate IPC — `nft.register`

```toml
[request]
method = "nft.register"
artifact_type = "pseudoSpore"
name = "hotspring-qcd-sun"
version = "1.0.0-rung1"
content_blake3 = "abc123..."
scope_blake3 = "def456..."
signature = "base64-ed25519..."
cas_address = "westgate://content/<hash>"

[response]
nft_id = "uuid"
verification_url = "https://nestgate.io/pseudospore/<hash>"
registered_at = "2026-08-10T15:30:00Z"
```

**Pattern**: The NFT is a pointer — it references the CAS address and carries the signature. This decouples storage (westGate) from identity (ironGate) from compute (strandGate).

### 3. westGate IPC — `content.ingest`

```toml
[request]
method = "content.ingest"
path = "/path/to/pseudospore_hotspring-qcd-sun_v1.0.0-rung1.tar.gz"
metadata.artifact_type = "pseudoSpore"
metadata.name = "hotspring-qcd-sun"
metadata.version = "1.0.0-rung1"

[response]
cas_hash = "blake3:<hash>"
size_bytes = 12345678
indexed = true
```

**Pattern**: westGate already handles `content.ingest` (verified in braid pen test 86/87). strandGate just needs to call it with the bundled tarball.

---

## What strandGate Builds (our side)

1. **`pseudospore_manifest`** — generates scope.toml, checksums, environment ✓
2. **`pseudospore_validate`** — verifies integrity (ships IN the bundle) ✓  
3. **`pseudospore_bundle`** — creates .tar.gz ✓
4. **`pseudospore_sign`** — calls bearDog for Ed25519 (NEXT)
5. **`pseudospore_register`** — pushes to ironGate + westGate (NEXT)

## What Other Gates Build (their side)

| Gate | Builds | Pattern |
|------|--------|---------|
| bearDog | `crypto.sign` IPC endpoint | Request/response over UDS |
| ironGate | `nft.register` endpoint + verification page | REST + IPC hybrid |
| westGate | Already done (`content.ingest`, `content.locate`) | IPC over TCP :7800 |
| sporePrint | QCD page template + download routing | Zola + Caddy |

---

## Abstraction for Other Gates

This pattern is **not QCD-specific**. Any spring that produces data follows the same pipeline:

1. Compute (spring binary) → raw data
2. Manifest (pseudospore_manifest) → provenance metadata
3. Bundle (pseudospore_bundle) → distributable artifact
4. Sign (bearDog) → cryptographic identity
5. Register (ironGate) → NFT/discoverability  
6. Store (westGate) → content-addressed persistence
7. Publish (sporePrint) → public trust surface

The binaries in steps 2-3 can be generalized into a `lithoSpore` CLI that takes a config file describing the spring's outputs. strandGate's hotSpring implementation is the **reference pattern** for this abstraction.

---

## Timeline

| Phase | When | What |
|-------|------|------|
| **Now** | Campaign running | 27/45 configs computing |
| **Campaign done** | ~5h | All 45 configs complete |
| **Manifest + Bundle** | Post-campaign | Run pseudospore_manifest → pseudospore_bundle |
| **Validate** | Same day | Run pseudospore_validate to confirm integrity |
| **Sign** | When bearDog endpoint ready | Call crypto.sign |
| **Register** | When ironGate endpoint ready | Call nft.register |
| **Publish** | When sporePrint page ready | Live at public URL |

---

*This spec is strandGate's commitment to the ecosystem convergence. The interfaces defined here become the contract that other gates build against.*
