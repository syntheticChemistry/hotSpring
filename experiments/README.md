<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Experiments Index

Chronological log of hotSpring experiments. Each experiment has a unique
numeric prefix for ordering and an `_TOPIC_DESCRIPTOR` suffix.

## Naming Convention

```
NNN_DESCRIPTOR.{sh,md,json}
```

- `sh` — runnable experiment scripts
- `md` — analysis write-ups, architecture investigations
- `json` — benchmark results and structured data

## Science Experiments (Lattice QCD / Physics)

| # | Name | Type | Domain |
|---|------|------|--------|
| 008 | PARITY_BENCHMARK | script | Nuclear EOS parity check |
| 032 | overnight_stratified | script | Stratified overnight validation |
| 033 | reality_ladder_rung0 | script | Reality ladder rung 0 |
| 034 | reality_ladder_rung1 | script | Reality ladder rung 1 |
| 035 | dp_memoization_overnight | script | DP memoization benchmark |
| 036 | extended_forward_dp | script | Extended forward DP |
| 037 | reverse_dp | script | Reverse DP |
| 038 | long_duration_phase_transition | script | Phase transition long-run |
| 039 | gen2_full_stream | script | Gen2 full streaming pipeline |
| 096 | SILICON_SCIENCE_TMU_QCD_MAPPING | analysis | TMU → QCD shader mapping |
| 097 | SILICON_BUDGET_SATURATION_COMPOSITION | analysis | Silicon budget saturation |
| 098 | QCD_SILICON_BENCHMARK_V2 | analysis | QCD silicon benchmark v2 |
| 099 | GPU_RHMC_ALL_FLAVORS | analysis | RHMC all flavor configurations |
| 100 | SILICON_CHARACTERIZATION_AT_SCALE | analysis | Silicon characterization |
| 101 | GPU_RHMC_PRODUCTION | analysis | RHMC production runs |
| 102 | GRADIENT_FLOW_AT_VOLUME | analysis | Gradient flow volume scaling |
| 103a | RHMC_GRADIENT_FLOW | analysis | RHMC + gradient flow combined |
| 103b | SELF_TUNING_RHMC_CALIBRATOR | analysis | Self-tuning RHMC calibration |
| 105 | SILICON_ROUTED_QCD_REVALIDATION | analysis | Silicon-routed QCD revalidation |
| 107 | SILICON_SATURATION_PROFILING | analysis | Silicon saturation profiling |

## Sovereign Compute Experiments (GPU Boot / Driver)

### Archived (fossil record — `archive/`)

| # | Name | Type | Key Finding |
|---|------|------|-------------|
| 001–057 | Early physics + sovereign GPU | various | Sarkas MD, nuclear EOS, TTM, surrogate, spectral, sovereign GPU cracking |
| 058–069 | VFIO through GlowPlug | analysis | VFIO pipeline, GlowPlug daemon, Falcon boot sequences |
| 070–089 | Dual Titan through GPCCS | analysis | Multi-GPU MMU, PFIFO, Falcon boot, SEC2, GPCCS |
| 090–107 | Silicon science + HS mode | analysis | GPCCS running, TMU mapping, QCD, RHMC, silicon saturation |
| 110–122 | Consolidation through WPR2 | analysis | VRAM page tables, dual-phase boot, WPR2 preservation/resolution |
| 123–131 | K80 + AMD + livepatch + reset | analysis | K80 sovereign, AMD scratch, warm handoff, FECS dispatch, reset evolution |
| 132–143 | Dual GPU boot + ACR | analysis | Ember dispatch, Kepler compute, SEC2 DMA, D-state, ACR root cause |

### Active

| # | Name | Type | Status |
|---|------|------|--------|
| 144 | PMC_BIT5_ACR_PROGRESS | analysis | ✅ SEC2 resets properly, IMEM PIO verified, BOOTVEC ignored, **VRAM dead** root cause found |
| 150 | CRASH_VECTOR_HUNT | analysis | ✅ Crash vectors eliminated by Ember Survivability Hardening. 8 consecutive runs survive. |
| 151 | REVALIDATION_AND_NEXT_STAGES | synthesis | ✅ Fossil record review, validated state, 6-stage plan |
| 152 | COMPUTE_DISPATCH_PROVENANCE_VALIDATION | framework | ✅ ToadStool compute.dispatch + blake3 witness + trio provenance |
| 153 | EMBER_FLOOD_RESURRECTION_PROOF | validation | ✅ Ember flood/resurrection under continuous fault injection |
| 154 | SEC2_ACR_PMU_FIRST_PIPELINE | investigation | ✅ SEC2→PMU first-boot pipeline, ACR chain ordering |
| 155 | K80_WARM_FECS_DISPATCH | validation | ✅ K80 warm-state FECS dispatch (Kepler PIO path) |
| 156 | REAGENT_TRACE_COMPARISON | analysis | ✅ Cross-reagent register trace comparison |
| 157 | K80_DEVINIT_REPLAY | investigation | ⚠️ K80 direct DEVINIT replay — PLL reprogramming risk |
| 158 | SEC2_REAL_FIRMWARE | investigation | ✅ SEC2 ACR BL executes, stalls on DMA (HBM2 not trained) |
| 159 | TITANV_VM_POST_HBM2 | breakthrough | ✅ HBM2 trained via nvidia-535 VM. FLR kills it. nouveau warm-cycle + reset_method clear preserves HBM2. |
| 160 | TITANV_MMIOTRACE_CAPTURE | capture | ✅ MMIOTRACE register capture for GV100 nouveau init |
| 161 | TITANV_NVDEC_SOVEREIGN_ATTEMPT | investigation | ✅ NVDEC engine sovereign dispatch attempt |
| 162 | TITANV_SOVEREIGN_COMPUTE_PIPELINE | architecture | ✅ Full sovereign compute pipeline design |
| 163 | FIRMWARE_BOUNDARY | **breakthrough** | ✅ **Architectural pivot.** Driver/firmware/hardware delineation. NOP dispatch via DRM (C + Rust). PMU mailbox mapped. PmuInterface created. |
| 164 | SOVEREIGN_COMPUTE_DISPATCH_PROVEN | **breakthrough** | ✅ **5/5 E2E phases pass.** f32 write/arith, multi-workgroup, f64 write, f64 Lennard-Jones. WGSL→SM70 SASS→DRM dispatch. Newton's 3rd law verified. *(No standalone journal — findings documented in Exp 165)* |
| 165 | SOVEREIGN_INIT_PIPELINE | **breakthrough** | ✅ **8-stage SovereignInit pipeline replaces nouveau.** `open_sovereign(bdf)` entry point. GR init extracted. FECS method probe. GR context Stage 7. Firmware-as-ingredient. 429 tests. |
| 166 | SOVEREIGN_BOOT_WIRING | investigation | ✅ **AdaptiveLifecycle delegation bug** found and fixed. `skip_sysfs_unbind` forwarding, `reset_method` permission error, `vfio-pci.ids` kernel parameter handling. 3 critical bugs resolved. |
| 167 | WARM_HANDOFF | validation | ✅ **Full vfio→nouveau→vfio round-trip on Titan V.** No D-state. HBM2 training preserved across swap cycle. K80 deferred (EBUSY). |
| 168 | SOVEREIGN_PIPELINE_COMPLETE | **milestone** | ✅ **Sovereign pipeline COMPLETE.** Fork-isolated MMIO gateway (6 RPCs). 6-stage sovereign init. PMU DEVINIT + VBIOS PROM wired as ember RPCs. 908 tests across coral-driver + coral-ember. |
| 169 | WARM_HANDOFF_VALIDATED | validation | ✅ **Full warm handoff cycle on Titan V.** vfio→nouveau→vfio round-trip. HBM2 warm state persists (pmc_enable=0x5fecdff1). Stages 1-3 pass. Falcon boot = next frontier. |
| 170 | SOVEREIGN_BOOT_E2E | **milestone** | ✅ **End-to-end `coralctl sovereign-boot`.** Vendor-ingredient loop: cold detect → nouveau warm → vfio swap → sovereign init. Warm detection heuristic (PMC popcount + PRAMIN). golden_state_path file reference. |
| 171 | K80_SOVEREIGN_INIT | validation | ⚠️ **K80 (GK210) BAR0 probe + PMC enable OK.** GDDR5 training BLOCKED (cold memory, PRAMIN returns PCIe timeout). VBIOS readable from PROM. Kepler: no signed firmware required. DEVINIT interpreter needed. |
| 189 | LTEE_B2_ANDERSON_FITNESS | analysis | 🔄 **LTEE B2 — Anderson fitness landscape (GuideStone).** Wiser et al. 2013 disorder analogy; Tier 1 baseline in `notebooks/papers/13-ltee-anderson-fitness.ipynb`; feeds foundation Thread 7 (anderson). See `EXPERIMENT_INDEX.md`. |

### Ember Survivability Hardening (2026-04-07)

Not a numbered experiment — a systematic architectural evolution tracked via plan:
- **Phase 1**: 6 critical lockup vectors eliminated (C1-C6)
- **Phase 2**: 4 moderate debt items hardened (M1-M4)
- **Phase 3**: Glowplug resurrection evolved (warm cycle, FdVault, warm_cycle RPC)
- **Validation**: 8 consecutive exp145 crash probes — zero lockups, all faults contained

### Multi-Ember Fleet Architecture (2026-04-07)

Architectural evolution — ember becomes per-device, glowplug becomes fleet orchestrator:
- **Per-device ember**: `--bdf` CLI flag, per-BDF socket paths, systemd template units
- **Fleet orchestrator**: `EmberFleet` in glowplug manages N active + M standby instances
- **Hot-standby pool**: Pre-spawned embers with `ember.adopt_device` RPC for instant takeover
- **Fault-informed resurrection**: Strategy selected by fault history (HotAdopt / WarmThenRespawn / FullRecovery)
- **Discovery file**: `<temp_dir>/biomeos/coral-ember-fleet.json` for external client routing
- **Backward compatible**: `fleet_mode = false` preserves legacy single-ember behavior

## NUCLEUS Composition Validation (April 2026)

Not numbered experiments — systematic composition infrastructure:

| Name | Type | Status |
|------|------|--------|
| `hotspring_unibin` | binary | ✅ **Eukaryotic Tier 1 entry** — `certify` / `validate` / `status` / `version`; primary NUCLEUS validation CLI (successor to `hotspring_guidestone`) |
| `validate_nucleus_composition` | binary | ✅ Validates all four atomic tiers (Tower/Node/Nest/NUCLEUS) via IPC + science parity probes (SEMF, plaquette, HMC) |
| `validate_nucleus_tower` | binary | ✅ Tower atomic (BearDog + Songbird) validation |
| `validate_nucleus_node` | binary | ✅ Node atomic (Tower + toadStool + barraCuda + coralReef) validation + science parity probes |
| `validate_nucleus_nest` | binary | ✅ Nest atomic (Tower + NestGate + rhizoCrypt + loamSpine + sweetGrass) validation |
| `validate_squirrel_roundtrip` | binary | ✅ Squirrel inference end-to-end (models, complete, embed) |
| `validate_primal_proof` | binary | ✅ **Level 6 primal proof** — calls barraCuda/BearDog over IPC (`tensor.matmul`, `stats.mean`, `crypto.hash`, etc.), compares vs Python/Rust baselines |
| `hotspring_guidestone` | binary | ✅ **guideStone Level 6 CERTIFIED** (legacy unified binary) — Bare: validates 5 properties (Deterministic, Traceable, Self-Verifying, Env-Agnostic, Tolerance-Documented). NUCLEUS additive: IPC parity via `primalspring::composition` API (scalar, vector, SEMF, crypto, compute). Prefer **`hotspring_unibin`** for new workflows. |
| `validate_science_probes` | library fn | ✅ compute health + math capability + provenance trio via IPC |
| `graphs/hotspring_qcd_deploy.toml` | deploy graph | ✅ 10 primals, bonding policy, spawn order for biomeOS |
| Composition audit + remediation | session | ✅ Socket fix, registration wiring, DAG/crypto alignment, validation.rs split |
| Stadial audit (April 17) | session | ✅ deny.toml, `#[expect]` migration, dyn elimination, tolerance centralization, unsafe→OnceLock |
| Primal composition proof (April 17) | session | ✅ Science parity probes, downstream_manifest alignment, all 13 methods dispatched |
| Level 6 primal proof audit (April 17) | session | ✅ `validate_primal_proof` harness, IPC mapping doc, downstream manifest corrected to primal IPC methods, capability domain routing fixed, dyn dispatch eliminated |
| guideStone alignment (April 18) | session | ✅ `hotspring_guidestone` binary, `primalspring` dep, composition API adoption, downstream manifest guideStone metadata, 5/5 properties certified (UniBin successor documented May 2026) |
| v0.9.16 absorption + primal proof (April 20) | session | ✅ BLAKE3 P3, `is_protocol_error()`, `validate-primal-proof.sh` script, plasmidBin ecoBin verified |
| v0.9.17 absorption (April 20) | session | ✅ genomeBin v5.1, guideStone v1.2.0, env var auto-setup (BEARDOG_FAMILY_SEED, SONGBIRD_SECURITY_PROVIDER, NESTGATE_JWT_SECRET), backward-compatible API |
| Property 3 CHECKSUMS + 30/30 bare (April 17) | session | ✅ Generated BLAKE3 CHECKSUMS manifest (15 source files), fixed deny.toml lookup for dual CWD, script builds from barracuda/ runs from root. **30/30 bare checks pass**, 3 SKIP (expected NUCLEUS liveness only). *(Historical test count: 993 lib tests at time of session.)* |
| Phase 46 composition template (April 27) | session | ✅ Absorbed primalSpring Phase 46 composition library. `tools/hotspring_composition.sh`: event-driven QCD + async tick model + DAG memoization + ledger sealing + scientific provenance braids + compute dispatch. Bare mode verified. |
| Deep debt evolution (April 27) | session | ✅ Capability-based primal discovery — `composition.rs` derives requirements from `niche::DEPENDENCIES` (single source of truth). Named accessors deprecated → `by_domain()`. Data-driven `PRIMAL_ALIASES`. Smart refactoring: `rhmc.rs` → `rhmc/mod.rs` + `remez.rs`, `nuclear_eos_helpers.rs` → `mod.rs` + `objectives.rs`. Pre-existing `DiscoveredDevice` compile errors fixed. *(Historical test count: 993/993 lib tests at time of session.)* Zero compilation errors. |

| 172 | NO_ACR_WARM_HANDOFF | md | ✅ Warm HBM2 without HS lockout by removing ACR firmware |
| 173 | VM_REAGENT_WPR_CAPTURE | investigation | ✅ VM reagent WPR capture. GV100 closed driver does NOT configure WPR (Volta predates GSP). Architectural pivot for Volta sovereign boot. |
| 174 | K80_SOVEREIGN_BOOT | investigation | ⚠️ K80 Kepler sovereign boot progress. GDDR5 training path. |
| 175 | RTX5060_SHARED_COMPUTE | **milestone** | ✅ RTX 5060 shared display/compute. UVM GPFIFO NOP timeout resolved (GAP-HS-031 RESOLVED). QMD v5.0 implemented. Sovereign dispatch LIVE. |
| 176 | QCD_PARITY_BENCHMARK | **milestone** | ✅ **Full HMC pipeline → native SASS on 3 GPU generations.** SM35 10/10, SM70 10/10, SM120 10/10. coralReef f64 lowering fixed (was 4/10 on Kepler). Vendor wgpu dispatch validated on RTX 5060. validate_pure_gauge 16/16 ALL CHECKS PASSED. |
| 177 | BLACKWELL_DISPATCH_ABI_FIXES | **milestone** | ✅ **Blackwell sovereign dispatch live.** QMD v5.0 + UVM faulting VA space + semaphore fence + UVM_REGISTER_CHANNEL. f64 div fixed (MUFU.RCP64H→F2F+RCP fallback). num_workgroups fixed (S2R NCTAID→LDC CBUF 7). 1404 tests passing. |
| 178 | K80_PGOB_NVIDIA470_ANALYSIS | investigation | ⚠️ **nvidia-470 binary analysis: PGOB PSW-only sequence discovered.** Static disassembly of `nv-kernel.o_binary` reveals `_nv029216rm` (ungate) and `_nv029114rm` (gate) use only `0x10a78c` bits 0-1. PSW-only requires running PMU firmware. Root cause narrowed: PRI ring has 0 GPCs enrolled. Pivoted to warm-catch (Exp 179). |
| 179 | K80_WARM_FECS_DISPATCH_PIPELINE | **milestone** | ✅ **K80 warm-catch FECS/PFIFO pipeline.** Nouveau warm-catch → VFIO rebind. FECS boots (Falcon v3 PIO). PFIFO runlist completes. SCHED_ERROR code=32 root-caused (RAMFC 0x3C/0x44) and fixed. Cold-boot sovereign (udev PLX fix, d3cold_allowed=0). |
| 180 | THREE_GPU_HARDWARE_VALIDATION | **milestone** | ✅ RTX 5060 19/19 (CUDA+DRM+discovery), Titan V 20/20 standalone VFIO, K80 device open + runlist pass. PGOB GPC gating confirmed as K80 dispatch blocker. |
| 181 | SOVEREIGN_DISPATCH_PIPELINE_SWEEP | investigation | 🔧 RTX 5060 8/8 PROVEN (WGSL→SM120→dispatch→readback). Titan V blocked (no PMU fw, SEC2/ACR). K80 cold-boot sovereign, PGOB dispatch blocker. Ember Exclusive Device Gate live. |
| 182 | K80_FECS_PIO_BOOT | diagnostic | 🔧 K80 GK210 FECS PIO boot diagnostic. Direct BAR0 mmap (`low-level` feature). Falcon IMEM/DMEM PIO path. |
| 183 | K80_FECS_INT_BOOT | diagnostic | 🔧 K80 GK210 FECS interrupt-driven boot. Direct BAR0 mmap (`low-level` feature). |
| 184 | K80_GR_SOVEREIGN | active | ✅ K80 GK210 sovereign GR init via ember RPC. Modern ember-wired path. Kepler falcon boot + firmware + PLX keepalive + switch preflight. |
| 185 | K80_NOUVEAU_GK210_CHIPSET | complete | ✅ Root cause: upstream nouveau has NO `case 0x0f2:` — GK210 unrecognized → -ENODEV. One-line patch: `case 0x0f2: device->chip = &nvf1_chipset;` |
| 186 | PMU_FW_EXTRACTION_ANALYSIS | complete | ✅ Kepler PMU from VBIOS (BIT tables). Volta PMU NOT in linux-firmware — needs nvidia-470 extraction. Enhanced exp168 probe. |
| 187 | TITANV_NVIDIA580_MMIOTRACE_PREP | prepared | 🔧 Capture script for nvidia-580 mmiotrace on Titan V. Determines WPR usage, informs FalconBootSolver Volta branch. Awaiting execution window. |
| 188 | K80_WARM_CATCH_BREAKTHROUGH | breakthrough | ✅ Patched nouveau RECOGNIZED GK210. First-ever GR init: 12 GiB GDDR5, 5 GPCs, 6 TPC/GPC. Post-rebind GPCs power-gated. PLX D3cold on ember stop. |
| — | K80_QEMU_VM_REAGENT | investigation | ✅ QEMU VM with K80 VFIO passthrough + proprietary nvidia-470.256.02. Module probed K80 successfully. Reagent template + build recipe stored in `agentReagents/`. |
| 189 | LTEE_B2_ANDERSON_FITNESS | notebook | 🔧 Tier 1 Python baseline — Wiser et al. 2013 LTEE Anderson fitness analogy. Power-law fitness model, Anderson Hamiltonian, localization analysis. |
| 190 | THREE_GPU_SOVEREIGN_VALIDATION | **validation** | ✅ Post-power-cycle sovereign validation across 3 GPU generations. RTX 5060 12/12 sovereign roundtrip PASS, 154 steps/s MD. Titan V warm (HBM2 from BIOS POST), FECS blocked (HS mode). K80 PLX alive (rev ca), PMC enabled, GPCs gated. |

## Paper Baseline Notebooks

12 publishable Python notebooks covering all 22 reproduced papers — see `notebooks/papers/`.
Each notebook wraps the corresponding `control/` Python baselines into clean,
narrative-driven notebooks with live compute (small problems) and frozen JSON
(production runs). These serve as entry points for collaborators and hooks
for primal provenance.

See `notebooks/papers/PAPER_NOTEBOOK_GUIDE.md` for the pattern.

## Benchmark Data

| # | Name | Format |
|---|------|--------|
| 053 | benchmark_results | JSON |
| 054 | kokkos_complexity_results | JSON |

## Utilities

- `archive/run_chuna_overnight.sh` — Run the Chuna overnight validation suite (archived)
- `archive/` — Completed experiments moved to fossil record
- `data/` — Experiment-associated data files

## Eukaryotic Evolution (May 2026)

6 representative experiment binaries have been absorbed into
`barracuda/src/validation/scenarios/` as Tier 1 (Rust) scenarios with
`ScenarioMeta` provenance tracking. 8 hardware-specific GPU experiment
binaries are preserved in `fossilRecord/experiments_prokaryotic_may2026/`.
The `hotspring_unibin validate` command is the eukaryotic entry point
for running all absorbed scenarios.
