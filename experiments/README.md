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
| 103 | RHMC_GRADIENT_FLOW | analysis | RHMC + gradient flow combined |
| 103 | SELF_TUNING_RHMC_CALIBRATOR | analysis | Self-tuning RHMC calibration |
| 105 | SILICON_ROUTED_QCD_REVALIDATION | analysis | Silicon-routed QCD revalidation |
| 107 | SILICON_SATURATION_PROFILING | analysis | Silicon saturation profiling |

## Sovereign Compute Experiments (GPU Boot / Driver)

### Archived (fossil record — `archive/`)

| # | Name | Type | Key Finding |
|---|------|------|-------------|
| 058–069 | VFIO through GlowPlug | analysis | VFIO pipeline, GlowPlug daemon, Falcon boot sequences |
| 070–089 | Dual Titan through GPCCS | analysis | Multi-GPU MMU, PFIFO, Falcon boot, SEC2, GPCCS |
| 090–107 | Silicon science + HS mode | analysis | GPCCS running, TMU mapping, QCD, RHMC, silicon saturation |
| 110–122 | Consolidation through WPR2 | analysis | VRAM page tables, dual-phase boot, WPR2 preservation/resolution |
| 123–126 | K80 + AMD + livepatch | analysis | K80 sovereign design, AMD scratch breakthrough, warm handoff confirmed |
| 127–131 | Warm FECS + reset arch | analysis | FECS dispatch, GPU puzzle box matrix, no-FLR recovery, reset evolution |
| 132–136 | Dual GPU sovereign boot | analysis | Ember dispatch, Kepler compute, cold boot pipeline, FBHUB/FBPA discovery |
| 137–139 | SEC2 DMA + D-state | analysis | DMA reconstruction, D-state root cause, ACR lockdown discovery |
| 140 | PMC bit 5 + safety arch | analysis | **SEC2 at bit 5 (not 22)**, uncrashable GPU safety architecture validated |
| 141–143 | ACR root cause | analysis | VBIOS DEVINIT suspected then **contradicted** — ACR fails even on POSTed GPU |

### Active

| # | Name | Type | Status |
|---|------|------|--------|
| 144 | PMC_BIT5_ACR_PROGRESS | analysis | KEY FINDINGS: SEC2 resets properly, IMEM PIO verified, BOOTVEC ignored, **VRAM dead** root cause found |
| 150 | CRASH_VECTOR_HUNT | analysis | **RESOLVED** — Crash vectors eliminated by Ember Survivability Hardening (fork-isolated MMIO, zero-I/O recovery, abort-not-exit). 8 consecutive runs survive. |
| 151 | REVALIDATION_AND_NEXT_STAGES | synthesis | Fossil record review, validated state, open questions, 6-stage plan for next phases |
| 152 | COMPUTE_DISPATCH_PROVENANCE_VALIDATION | framework | ToadStool compute.dispatch + blake3 witness + trio provenance (primalSpring audit) |

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

## Benchmark Data

| # | Name | Format |
|---|------|--------|
| 053 | benchmark_results | JSON |
| 054 | kokkos_complexity_results | JSON |

## Utilities

- `run_chuna_overnight.sh` — Run the Chuna overnight validation suite
- `archive/` — Completed experiments moved to fossil record
- `data/` — Experiment-associated data files
