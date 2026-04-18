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
- **Discovery file**: `/tmp/biomeos/coral-ember-fleet.json` for external client routing
- **Backward compatible**: `fleet_mode = false` preserves legacy single-ember behavior

## NUCLEUS Composition Validation (April 2026)

Not numbered experiments — systematic composition infrastructure:

| Name | Type | Status |
|------|------|--------|
| `validate_nucleus_composition` | binary | ✅ Validates all four atomic tiers (Tower/Node/Nest/NUCLEUS) via IPC + science parity probes (SEMF, plaquette, HMC) |
| `validate_nucleus_tower` | binary | ✅ Tower atomic (BearDog + Songbird) validation |
| `validate_nucleus_node` | binary | ✅ Node atomic (Tower + toadStool + barraCuda + coralReef) validation + science parity probes |
| `validate_nucleus_nest` | binary | ✅ Nest atomic (Tower + NestGate + rhizoCrypt + loamSpine + sweetGrass) validation |
| `validate_squirrel_roundtrip` | binary | ✅ Squirrel inference end-to-end (models, complete, embed) |
| `validate_primal_proof` | binary | ✅ **Level 5 primal proof** — calls barraCuda/BearDog over IPC (`tensor.matmul`, `stats.mean`, `crypto.hash`, etc.), compares vs Python/Rust baselines |
| `hotspring_guidestone` | binary | ✅ **guideStone Level 5 CERTIFIED** — unified guideStone binary. Bare: validates 5 properties (Deterministic, Traceable, Self-Verifying, Env-Agnostic, Tolerance-Documented). NUCLEUS additive: IPC parity via `primalspring::composition` API (scalar, vector, SEMF, crypto, compute) |
| `validate_science_probes` | library fn | ✅ compute health + math capability + provenance trio via IPC |
| `graphs/hotspring_qcd_deploy.toml` | deploy graph | ✅ 10 primals, bonding policy, spawn order for biomeOS |
| Composition audit + remediation | session | ✅ Socket fix, registration wiring, DAG/crypto alignment, validation.rs split |
| Stadial audit (April 17) | session | ✅ deny.toml, `#[expect]` migration, dyn elimination, tolerance centralization, unsafe→OnceLock |
| Primal composition proof (April 17) | session | ✅ Science parity probes, downstream_manifest alignment, all 13 methods dispatched |
| Level 5 primal proof audit (April 17) | session | ✅ `validate_primal_proof` harness, IPC mapping doc, downstream manifest corrected to primal IPC methods, capability domain routing fixed, dyn dispatch eliminated |
| guideStone alignment (April 18) | session | ✅ `hotspring_guidestone` binary, `primalspring` dep, composition API adoption, downstream manifest guideStone metadata, 5/5 properties certified |

## Benchmark Data

| # | Name | Format |
|---|------|--------|
| 053 | benchmark_results | JSON |
| 054 | kokkos_complexity_results | JSON |

## Utilities

- `archive/run_chuna_overnight.sh` — Run the Chuna overnight validation suite (archived)
- `archive/` — Completed experiments moved to fossil record
- `data/` — Experiment-associated data files
