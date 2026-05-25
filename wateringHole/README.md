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
| 2026-05-13 | `HOTSPRING_DEEP_DEBT_SPRINT_MAY13_2026.md` | archive | Deep debt resolution: println migration, BDF discovery, pure-Rust blake3, boot scripts, CI gate |
| 2026-05-13 | `HOTSPRING_FULL_MODERNIZATION_PLASMIDBIN_HANDOFF_MAY13_2026.md` | archive | Full modernization: plasmidBin ecoBin deployment, legacy coral_gpu excision, NUCLEUS discovery |
| 2026-05-13 | `HOTSPRING_TRIO_REWIRE_MAY13_2026.md` | archive | Compute trio rewire: toadStool S258-S261, coralReef Sprint 9, barraCuda Sprint 23 |
| 2026-05-14 | `HOTSPRING_LOCAL_DEBT_COMPOSITION_EVOLUTION_HANDOFF_MAY14_2026.md` | archive | Local debt sprint: compile-then-dispatch, circuit breaker, dispatch unification, TOML aliases, tiered validation |
| 2026-05-14 | `HOTSPRING_PLASMIDBIN_LOCAL_OWNERSHIP_HANDOFF_MAY14B_2026.md` | archive | plasmidBin debt: release cascade, symlink-aware doctor, generalized upgrade, 13/13 NUCLEUS deployed |
| 2026-05-16 | `HOTSPRING_DIESEL_ENGINE_DRIVER_SKETCH_HANDOFF_MAY16_2026.md` | archive | Diesel engine driver sketch: GrInitSequence, WarmStateCapture, DriverProbe, PmuBootstrap, shared `nv::pri` |
| 2026-05-16 | `HOTSPRING_PLX_KEEPALIVE_BOOT_CATCH_HANDOFF_MAY16_2026.md` | archive | PLX keepalive boot-catch: class code extraction fix, event-driven interval, activity-aware backpressure, boot validation |
| 2026-05-16 | `HOTSPRING_WAVE17_SIGNAL_ADOPTION_HANDOFF_MAY16_2026.md` | archive | Wave 17: primal.announce, node.compute, tower.publish signal adoption |
| 2026-05-16 | `HOTSPRING_DOC_EVOLUTION_UPSTREAM_HANDOFF_MAY16_2026.md` | archive | Doc evolution: count normalization, unibin naming, handoff archival, upstream patterns |
| 2026-05-16 | `HOTSPRING_DIESEL_ENGINE_CAPABILITY_ABSTRACTION_HANDOFF_MAY16B_2026.md` | archive | Diesel engine capability abstraction: 6 subsystems generalized |
| 2026-05-16 | `HOTSPRING_BOOT_PIPELINE_VBIOS_HANDOFF_MAY16C_2026.md` | archive | Vendor-agnostic BootPipeline trait, VBIOS interpreter fixes |
| 2026-05-16 | `HOTSPRING_PRIMALS_SPRINGS_EVOLUTION_HANDOFF_MAY16D_2026.md` | archive | Cross-team handoff: primal use/evolution, NUCLEUS composition patterns |
| 2026-05-17 | `HOTSPRING_COMPUTE_PARITY_MIXED_HARDWARE_HANDOFF_MAY17_2026.md` | archive | Wave 20 compute evolution: CPU/GPU parity scenarios, toadStool dispatch validation, metalForge NUCLEUS atomics + PCIe direct + biomeOS graph, CPU fallback dispatch, parity greenboard ALL GREEN |
| 2026-05-17 | `HOTSPRING_DUAL_TITAN_V_HANDOFF_MAY17_2026.md` | archive | Dual Titan V twin study baseline: K80→Titan V #2, sovereign.init validated on both, VBIOS ROM identical, IOMMU isolated, twin study surface live |
| 2026-05-17 | `HOTSPRING_FALCON_ACR_DMA_HANDOFF_MAY17_2026.md` | archive | Falcon ACR DMA boot solved: DMA backend wired, FECS cpuctl=0x10, stale coral-ember killed, 02:00.0 bound to vfio-pci |
| 2026-05-18 | `HOTSPRING_SOVEREIGN_BOOT_ABSTRACTION_HANDOFF_MAY18_2026.md` | ✅ | Sovereign boot abstraction: SovereignBootState enum, WarmKeepalive facade, sovereign.profile RPC, twin-card cold profiling, hardware line codified |
| 2026-05-18 | `HOTSPRING_WARM_KEEPALIVE_183MS_HANDOFF_MAY18_2026.md` | ✅ | Warm keepalive PROVEN: 183ms warm pipeline, falcon preservation, fd store end-to-end, 76× faster than cold |
| 2026-05-18 | `HOTSPRING_COMPREHENSIVE_EVOLUTION_HANDOFF_MAY18_2026.md` | ✅ | Comprehensive evolution: 213 experiments reconciled, dual Titan V fleet, K80 retired, primal evolution handoff |
| 2026-05-19 | `HOTSPRING_GPC_BOUNDARY_CE_VALIDATE_HANDOFF_MAY19_2026.md` | archive | GPC boundary analysis: PTOP parser fix, CE runlist discovery, sovereignty tier model, all engine domains power-gated, paths to Tier 2 |
| 2026-05-20 | `HOTSPRING_SOVEREIGN_DRIVER_ROTATION_EXP211_COMPLETE_MAY20_2026.md` | archive | Exp 211 complete, sovereign driver rotation codified in diesel engine, docs reconciled |
| 2026-05-21 | `HOTSPRING_KERNEL_HEALTH_PREFLIGHT_EXP216_MAY21_2026.md` | archive | Kernel health preflight: 3-layer autoconf detection, Exp 216, post-fix audit clean |
| 2026-05-21 | `HOTSPRING_NVSOV_DUAL_LOAD_DRIVER_INFRA_EXP217_218_MAY21_2026.md` | archive | Exp 217 TPC wall closed, Exp 218 nvsov dual-load (4 blockers solved, module loads), driver infra evolution (SymbolResolver, snapshot/compare RPCs, generation dispatch) |
| 2026-05-22 | `HOTSPRING_CATALYST_DRIVER_PATTERN_EXP219_MAY22_2026.md` | archive | Exp 219 Catalyst Driver Pattern: proprietary driver as catalyst → BAR0 capture → golden state replay. 3 new RPCs, 3-layer preservation, selective un-NOP for RM capability tables |
| 2026-05-23 | `HOTSPRING_GATE_DEPLOYMENT_MAY23_2026.md` | ✅ | **Covalent gate deployment:** biomeGate assignment confirmed (sole tenant, 2× Titan V + RTX 5060), proto-nucleate wired (9 primals), deployment flow documented, 3 gaps (HS-108/109/110) |
| 2026-05-24 | `HOTSPRING_CAZYME_FEL_EVOLUTION_MAY24_2026.md` | ✅ | **CAZyme FEL evolution:** GROMACS 2026.0 installed (CUDA, PLUMED, Colvars), Exp 220 Phase 0, biomolecular MD track active, 2 new gaps (HS-111/112), barraCuda bonded FF + petalTongue viz evolution requests |
| 2026-05-24 | `HOTSPRING_CATALYST_HW_VALIDATED_EXP219_MAY24_2026.md` | ✅ | Exp 219 HW execution: 26s pipeline, domain-scoped capture (897ms, 83K alive regs), surgical NopCallAt patches, SBR bridge reset recovery, fire-and-poll unbind, Tier 1 confirmed on Titan V |
| 2026-05-25 | `HOTSPRING_UEFI_MODEL_PRI_RING_RECOVERY_EXP221_MAY25_2026.md` | ✅ | **Exp 221 UEFI Model GPU Sovereignty:** PRI ring recovery validated (PGRAPH re-enable + enumerate), falcon registers accessible post-recovery (Degraded health). Falcon HS fuse boundary mapped — IMEM wiped, fuse-enforced HS blocks host PIO. RetAtEntry eliminated. Architecture pivots to Runtime Services model for Tier 2. Three boundaries documented (PCI framework / Falcon HS fuses / ACR secure boot). Both cards validated. |

### `mmiotraces/`

Raw GPU MMIO trace captures (Linux `mmiotrace`) for reverse-engineering
driver initialization sequences. Used to extract register programming
tables for sovereign boot pipelines.

### Scripts and Lab Artifacts

| File | Status | Notes |
|------|--------|-------|
| `titanv_*.bin` | **Lab reference** | Binary firmware extracts from mmiotrace (SEC2, PMU, HS) — used for sovereign compute investigation. Gitignored. |
| `gk110/` | **Lab reference** | GK110/GK210 register reference data. Gitignored. |
| `vbios/` | **Lab reference** | VBIOS binary data for sovereign boot analysis. |
| ~~`exp192_postboot_warm_bf_init.py`~~ | **Archived** | Moved to `scripts/archive/`. Superseded by `toadstool device warm-catch`. |

## Relationship to ecosystem wateringHole

The ecosystem-wide guidance hub lives at `ecoPrimals/infra/wateringHole/`.
That is the authoritative source for primal taxonomy, composition patterns,
NUCLEUS definitions, and cross-primal contracts. This directory is
hotSpring's local lab notebook — discoveries here get distilled into
upstream handoffs when they mature.
