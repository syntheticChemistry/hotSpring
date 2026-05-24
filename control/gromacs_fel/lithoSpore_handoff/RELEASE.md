# pseudoSpore Release — v0.6.0

**Artifact**: hotSpring-CAZyme-FEL
**Status**: Ready, with two documented caveats
**Date**: May 24, 2026
**Compression**: tar.gz (pre-compressed, 106K / 41 files)

---

## Structural Checklist (lithoSpore ingestion)

| Requirement | Status |
|---|---|
| `ferment_transcript.json` with `braid_id` + `dag_session_id` + `merkle_root` + `spine_id` | Present, well-formed |
| `provenance/braids/` directory (lithoSpore ingestion path) | Present, 3 files |
| `scope.toml` (machine-readable birth certificate) | Present, `[[module]]` entries match lithoSpore pattern |
| `validation.json` (structured checks) | Present, errata annotated inline |
| Live sweetGrass braid (actual IPC, not pseudo) | `live_braid.json` — real sweetGrass v0.7.27 touched this |
| W3C PROV-O JSON-LD export | `provo_export.jsonld` — valid `@context`, correct `@graph` |
| Human-readable handoff doc | `ABG_HANDOFF.md` — clear separation of "for Alistaire" vs "for ABG Discord" |
| Reproducibility (configs + data) | All MDP/PLUMED/PDB files present per module |

---

## Two Caveats

Both are already documented in the artifact itself.

### 1. Module 3 is IN_FLIGHT

The ferment transcript honestly reports `modules_complete: 2`,
`modules_in_flight: 1`. The acceptance test (free vs enzyme-bound
comparison) can't run until it finishes. This is fine for a v0.6.0
handoff — it's an intermediate checkpoint, not a final deliverable.

### 2. Atom-ordering convention question is open

Alistaire needs to confirm the C1-C2-C3-C4-C5-O5 mapping before the
Module 2 landscape (and eventually Module 3) can be interpreted against
Iglesias-Fernandez 2015. This is explicitly flagged in `ABG_HANDOFF.md`
as the action item for him.

---

## Hash Note

lithoSpore proper uses BLAKE3 for content hashes; the pseudo uses
SHA-256. For a data-only / braid-passing handoff this doesn't matter —
the hashes are informational provenance, not signature-verified. When
this eventually promotes to a real lithoSpore module, the hashes get
re-computed as BLAKE3 during `litho assemble`.

---

## Bottom Line

The tarball is deliverable as-is. The `live_braid.json` from sweetGrass
IPC is the strongest piece — it means this isn't purely offline
pseudo-provenance, an actual running primal witnessed and signed
(unsigned pending, but structurally real) the artifact. Alistaire gets
the FES data he needs to answer the convention question, and the ferment
transcript is in the right shape for lithoSpore to eventually ingest it
as an upstream braid from hotSpring.
