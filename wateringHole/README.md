# wateringHole — hotSpring Lab Artifacts & Handoffs

Lab-facing artifacts, evolution handoff documents, and hardware trace captures
for the hotSpring validation spring. This is **not** the ecosystem-level
`infra/wateringHole/` — it is hotSpring's local working area for hardware
experimentation and cross-session knowledge transfer.

**Historical naming:** References to `coral-ember` in this directory (handoffs,
tables, scripts) are **historical**. As of May 2026 the diesel-engine lineage
(ember / glowplug / cylinder) has been absorbed into **toadStool**; read those
names as fossil-record context unless a note explicitly marks current wiring.

Earlier handoffs (Apr 16 – May 7) were migrated to `ecoPrimals/infra/wateringHole/handoffs/`.
Only the most recent local handoffs are retained here.

## Contents

### `handoffs/`

Dated evolution handoff documents capturing state transitions, hardware
discoveries, and architectural decisions. Each handoff is a fossil record
entry for the session that produced it. Chronological by filename suffix.

| Date | File | On Disk | Topic |
|------|------|---------|-------|
| 2026-04-16 | `HOTSPRING_BLACKWELL_DISPATCH_LIVE_HANDOFF_APR16_2026.md` | upstream | RTX 5060 Blackwell sovereign dispatch — first desktop GPU |
| 2026-04-17 | `HOTSPRING_V0632_STADIAL_AUDIT_HANDOFF_APR17_2026.md` | upstream | Stadial audit: `deny.toml`, MSRV, `#[expect]`, validation gates |
| 2026-04-21 | `HOTSPRING_GPU_GENERATION_PROFILE_FRONTIER_HANDOFF_APR21_2026.md` | upstream | GPU generation profiles, vendor-agnostic frontier |
| 2026-04-27 | `HOTSPRING_V0632_DEEPDEBT_PHASE46_HANDOFF_APR27_2026.md` | upstream | Deep debt Phase 46: capability discovery, deprecation cleanup |
| 2026-04-29 | `HOTSPRING_CORALREEF_K80_PGOB_NVIDIA470_HANDOFF_APR29_2026.md` | upstream | K80 PGOB nvidia-470 binary analysis, PMU firmware requirement |
| 2026-04-30 | `HOTSPRING_CORALREEF_K80_FECS_PFIFO_HANDOFF_APR30_2026.md` | upstream | K80 warm FECS/PFIFO: nouveau → VFIO handoff, SCHED_ERROR fix |
| 2026-05-06 | `HOTSPRING_CORALREEF_SPRINT_C_HW_VALIDATION_HANDOFF_MAY06_2026.md` | upstream | Sprint C: three-GPU HW validation (RTX 5060, Titan V, K80) |
| 2026-05-06 | `HOTSPRING_CORALREEF_EMBER_GATE_K80_COLDBOOT_HANDOFF_MAY06_2026.md` | upstream | Ember exclusive device gate + K80 cold-boot without PLX death |
| 2026-05-06 | `HOTSPRING_CORALREEF_SOVEREIGN_PIPELINE_HARDENING_HANDOFF_MAY06B_2026.md` | upstream | Three-GPU hardening: SLM, K80 PLL, Volta pipeline, unsafe audit |
| 2026-05-07 | `HOTSPRING_CORALREEF_TITANV_WARM_DMATRF_HANDOFF_MAY07_2026.md` | upstream | Titan V DMATRF to FECS IMEM — ROM security gate identified |
| 2026-05-07 | `HOTSPRING_EVOLUTION_PASS_DEBT_REFACTOR_HANDOFF_MAY07_2026.md` | archive | Evolution pass: deploy graph, `niche.rs`, clippy, exp070 RAII |
| 2026-05-10 | `HOTSPRING_CORALREEF_SOVEREIGN_KEEPALIVE_HANDOFF_MAY10_2026.md` | archive | PLX keepalive → coral-ember; glowplug diesel validation |
| 2026-05-10 | `HOTSPRING_DEEP_DEBT_PHASE4_UPSTREAM_HANDOFF_MAY10_2026.md` | archive | Deep Debt Phase 4: Tier 4 IPC-first, typed errors, L6 cert |
| 2026-05-11 | `HOTSPRING_CORALREEF_SOVEREIGN_BARRIERS_HANDOFF_MAY11_2026.md` | archive | Sovereign barrier resolution: Volta ACR skip, HBM2 warm-handoff, benchScale VM path, K80 PCIe diagnosis |
| 2026-05-11 | `INFRA_MATURITY_ECOSYSTEM_HANDOFF_MAY11_2026.md` | archive | benchScale + agentReagents maturity, composition patterns, NUCLEUS deployment |
| 2026-05-11 | `HOTSPRING_SOVEREIGN_RUST_EVOLUTION_HANDOFF_MAY11_2026.md` | archive | Warm-catch pipeline elevated to pure Rust. ALL 3 GPUs sovereign. Era-agnostic roadmap. |
| 2026-05-12 | `HOTSPRING_COMPUTE_TRIO_CAPABILITY_EVOLUTION_HANDOFF_MAY12_2026.md` | archive | Compute trio rewire, capability discovery evolution, scenario expansion, downstream audit |
| 2026-05-12 | `HOTSPRING_IPC_TRANSPORT_EVOLUTION_HANDOFF_MAY12_2026.md` | archive | GAP-HS-092: call_by_capability proliferation, wildcard audit, production mock audit |
| 2026-05-12 | `HOTSPRING_COMPUTE_TRIO_PIPELINE_HANDOFF_MAY12_2026.md` | archive | GAP-HS-094: GlowplugClient NUCLEUS evolution, compute trio + hotQCD scenarios |
| 2026-05-12 | `HOTSPRING_VFIO_SOVEREIGN_DISPATCH_HANDOFF_MAY12_2026.md` | archive | GAP-HS-095: VFIO feature enable, in-process dispatch wiring, upstream Phase C/D gaps |
| 2026-05-12 | `HOTSPRING_WARM_VFIO_DISPATCH_EVOLUTION_HANDOFF_MAY12_2026.md` | archive | GAP-HS-095 cont: warm API, cold/warm mismatch fix, ember blocker, kernel-module roadmap |
| 2026-05-12 | `HOTSPRING_EMBER_GLOWPLUG_OWNERSHIP_AUDIT_HANDOFF_MAY12_2026.md` | archive | GAP-HS-096: dual-existence audit, cylinder translation fix, toadStool parity gap, cutover path |
| 2026-05-12 | `HOTSPRING_PHASE_C_EXECUTION_PLAN_MAY12_2026.md` | archive | Phase C execution plan: 7 work items (C1-C7), coralReef soft-deprecation, hotSpring validation checklist |
| 2026-05-13 | `HOTSPRING_DEEP_DEBT_SPRINT_MAY13_2026.md` | ✅ | Deep debt resolution: println migration, BDF discovery, pure-Rust blake3, boot scripts, CI gate |
| 2026-05-13 | `HOTSPRING_FULL_MODERNIZATION_PLASMIDBIN_HANDOFF_MAY13_2026.md` | ✅ | Full modernization: plasmidBin ecoBin deployment, legacy coral_gpu excision, NUCLEUS discovery |
| 2026-05-13 | `HOTSPRING_TRIO_REWIRE_MAY13_2026.md` | ✅ | Compute trio rewire: toadStool S258-S261, coralReef Sprint 9, barraCuda Sprint 23 |
| 2026-05-14 | `HOTSPRING_LOCAL_DEBT_COMPOSITION_EVOLUTION_HANDOFF_MAY14_2026.md` | ✅ | Local debt sprint: compile-then-dispatch, circuit breaker, dispatch unification, TOML aliases, tiered validation |
| 2026-05-14 | `HOTSPRING_PLASMIDBIN_LOCAL_OWNERSHIP_HANDOFF_MAY14B_2026.md` | ✅ | plasmidBin debt: release cascade, symlink-aware doctor, generalized upgrade, 13/13 NUCLEUS deployed |
| 2026-05-16 | `HOTSPRING_DIESEL_ENGINE_DRIVER_SKETCH_HANDOFF_MAY16_2026.md` | ✅ | Diesel engine driver sketch: GrInitSequence, WarmStateCapture, DriverProbe, PmuBootstrap, shared `nv::pri` |
| 2026-05-16 | `HOTSPRING_PLX_KEEPALIVE_BOOT_CATCH_HANDOFF_MAY16_2026.md` | ✅ | PLX keepalive boot-catch: class code extraction fix, event-driven interval, activity-aware backpressure, boot validation |
| 2026-05-16 | `HOTSPRING_WAVE17_SIGNAL_ADOPTION_HANDOFF_MAY16_2026.md` | ✅ | Wave 17: primal.announce, node.compute, tower.publish signal adoption |
| 2026-05-16 | `HOTSPRING_DOC_EVOLUTION_UPSTREAM_HANDOFF_MAY16_2026.md` | ✅ | Doc evolution: count normalization, unibin naming, handoff archival, upstream patterns |
| 2026-05-16 | `HOTSPRING_DIESEL_ENGINE_CAPABILITY_ABSTRACTION_HANDOFF_MAY16B_2026.md` | ✅ | Diesel engine capability abstraction: 6 subsystems generalized (PCIe Bridge Health, GspBridge caps, Memory Training dispatch, Falcon Boot wiring, Engine Ungating, DriverLab Executor), 6,989 tests, deployed |

### `mmiotraces/`

Raw GPU MMIO trace captures (Linux `mmiotrace`) for reverse-engineering
driver initialization sequences. Used to extract register programming
tables for sovereign boot pipelines.

### Scripts and Lab Artifacts

| File | Status | Notes |
|------|--------|-------|
| `exp192_postboot_warm_bf_init.py` | **Lab reference** | Post-reboot Boot Falcon (NVDEC0) init via direct BAR0 mmap. For reference only — production path uses `coralctl warm-catch`. |
| `titanv_*.bin` | **Lab reference** | Binary firmware extracts from mmiotrace (SEC2, PMU, HS) — used for sovereign compute investigation. |
| `gk110/` | **Lab reference** | GK110/GK210 register reference data. |

## Relationship to ecosystem wateringHole

The ecosystem-wide guidance hub lives at `ecoPrimals/infra/wateringHole/`.
That is the authoritative source for primal taxonomy, composition patterns,
NUCLEUS definitions, and cross-primal contracts. This directory is
hotSpring's local lab notebook — discoveries here get distilled into
upstream handoffs when they mature.
