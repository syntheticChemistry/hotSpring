# wateringHole — hotSpring Lab Artifacts & Handoffs

Lab-facing artifacts, evolution handoff documents, and hardware trace captures
for the hotSpring validation spring. This is **not** the ecosystem-level
`infra/wateringHole/` — it is hotSpring's local working area for hardware
experimentation and cross-session knowledge transfer.

Earlier handoffs (Apr 16 – May 7) were migrated to `ecoPrimals/infra/wateringHole/handoffs/`.
Only the most recent local handoffs are retained here.

## Contents

### `handoffs/`

| Date | File | Topic |
|------|------|-------|
| 2026-05-07 | `HOTSPRING_EVOLUTION_PASS_DEBT_REFACTOR_HANDOFF_MAY07_2026.md` | Evolution pass: deploy graph, `niche.rs`, clippy, exp070 RAII |
| 2026-05-10 | `HOTSPRING_CORALREEF_SOVEREIGN_KEEPALIVE_HANDOFF_MAY10_2026.md` | Sovereign pipeline hardening: BDF validation, keepalive clamping, switch health |
| 2026-05-10 | `HOTSPRING_DEEP_DEBT_PHASE4_UPSTREAM_HANDOFF_MAY10_2026.md` | Deep Debt Phase 4 upstream handoff: Tier 4 IPC-first, typed errors, L6 cert, patterns for primals/springs |

### Scripts (deprecated)

| File | Status | Notes |
|------|--------|-------|
| `warm_handoff.sh` | **DEPRECATED** | Legacy ad-hoc lab script for direct `insmod`/unbind/rebind. Violates current policy: all GPU driver transitions must go through `coralctl` / ember / glowplug. Kept as fossil record. |
| `exp192_postboot_warm_bf_init.py` | **Lab reference** | Post-reboot Boot Falcon (NVDEC0) init via direct BAR0 mmap. References missing exp191. For reference only — production path uses ember RPC. |

## Relationship to ecosystem wateringHole

The ecosystem-wide guidance hub lives at `ecoPrimals/infra/wateringHole/`.
That is the authoritative source for primal taxonomy, composition patterns,
NUCLEUS definitions, and cross-primal contracts. This directory is
hotSpring's local lab notebook — discoveries here get distilled into
upstream handoffs when they mature.
