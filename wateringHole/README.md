# wateringHole — hotSpring Lab Artifacts & Handoffs

Lab-facing artifacts, evolution handoff documents, and hardware trace captures
for the hotSpring validation spring. This is **not** the ecosystem-level
`infra/wateringHole/` — it is hotSpring's local working area for hardware
experimentation and cross-session knowledge transfer.

## Contents

### `handoffs/`

Dated evolution handoff documents capturing state transitions, hardware
discoveries, and architectural decisions. Each handoff is a fossil record
entry for the session that produced it. Chronological by filename suffix.

| Date | File | Topic |
|------|------|-------|
| 2026-04-16 | `HOTSPRING_BLACKWELL_DISPATCH_LIVE_HANDOFF_APR16_2026.md` | RTX 5060 Blackwell sovereign dispatch — first desktop GPU |
| 2026-04-17 | `HOTSPRING_V0632_STADIAL_AUDIT_HANDOFF_APR17_2026.md` | Stadial audit: `deny.toml`, MSRV, `#[expect]`, validation gates |
| 2026-04-21 | `HOTSPRING_GPU_GENERATION_PROFILE_FRONTIER_HANDOFF_APR21_2026.md` | GPU generation profiles, vendor-agnostic frontier |
| 2026-04-27 | `HOTSPRING_V0632_DEEPDEBT_PHASE46_HANDOFF_APR27_2026.md` | Deep debt Phase 46: capability discovery, deprecation cleanup |
| 2026-04-29 | `HOTSPRING_CORALREEF_K80_PGOB_NVIDIA470_HANDOFF_APR29_2026.md` | K80 PGOB nvidia-470 binary analysis, PMU firmware requirement |
| 2026-04-30 | `HOTSPRING_CORALREEF_K80_FECS_PFIFO_HANDOFF_APR30_2026.md` | K80 warm FECS/PFIFO: nouveau → VFIO handoff, SCHED_ERROR fix |
| 2026-05-06 | `HOTSPRING_CORALREEF_SPRINT_C_HW_VALIDATION_HANDOFF_MAY06_2026.md` | Sprint C: three-GPU HW validation (RTX 5060, Titan V, K80) |
| 2026-05-06 | `HOTSPRING_CORALREEF_EMBER_GATE_K80_COLDBOOT_HANDOFF_MAY06_2026.md` | Ember exclusive device gate + K80 cold-boot without PLX death |
| 2026-05-06 | `HOTSPRING_CORALREEF_SOVEREIGN_PIPELINE_HARDENING_HANDOFF_MAY06B_2026.md` | Three-GPU hardening: SLM, K80 PLL, Volta pipeline, unsafe audit |
| 2026-05-07 | `HOTSPRING_CORALREEF_TITANV_WARM_DMATRF_HANDOFF_MAY07_2026.md` | Titan V DMATRF to FECS IMEM — ROM security gate identified |
| 2026-05-07 | `HOTSPRING_EVOLUTION_PASS_DEBT_REFACTOR_HANDOFF_MAY07_2026.md` | Evolution pass: deploy graph, `niche.rs`, clippy, exp070 RAII |

### `mmiotraces/`

Raw GPU MMIO trace captures (Linux `mmiotrace`) for reverse-engineering
driver initialization sequences. Used to extract register programming
tables for sovereign boot pipelines.

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
