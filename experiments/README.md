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
| 144–189 | Sovereign pipeline → warm-catch arc | various | PMC bit5 through LTEE Anderson fitness; per-number journals live under `archive/` (see also `archive/FOSSIL_RECORD_144_189.md`) |

### Active

| # | Name | Type | Status |
|---|------|------|--------|
| 191 | TOADSTOOL_S258_PBDMA_VALIDATION | validation | ✅ Compute trio pipeline: toadStool PBDMA dispatch (S258-S261), compile-then-dispatch wiring, circuit-breaker discovery |
| 191B | SOVEREIGN_DISPATCH_VALIDATED | validation | ✅ First e2e sovereign VFIO dispatch on Titan V (S263). Warm handoff validated, CPUCTL_ALIAS breakthrough, DMA roundtrip + GR init. Frontier: FECS PENDING_CTX_RELOAD |
| 192 | HARDWARE_VALIDATION_SPRINT_COMPUTE_TRIO | validation | ✅ Compute trio hardware validation sprint |
| 193 | PLX_D3COLD_KEEPALIVE_K80 | analysis | ✅ PLX D3cold keepalive on K80 |
| 194 | COLD_WARM_BOOT_ARCHITECTURE | analysis | ✅ Cold/warm boot architecture |
| 195 | DRIVER_LAB_MESA_VS_VENDOR | analysis | ✅ Driver lab: Mesa vs vendor |
| 196 | WARM_SWAP_VALIDATION_PLX_KEEPALIVE | validation | ✅ Warm swap + PLX keepalive validation |
| 197 | SOVEREIGN_INIT_RPC_WARM_COLD | validation | ✅ `sovereign.init` JSON-RPC wired, Titan V warm 88ms, K80 cold PRAMIN dead |
| 198 | VENDOR_AGNOSTIC_BOOT_PIPELINE | validation | ✅ `BootPipeline` trait, VBIOS interpreter fixes, VegaInit AMD stub, 591→606 tests |

> **Note:** 198 experiments total (001–189 archived + 190 archived final coral-ember + 191–198 active).

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

## Paper Baseline Notebooks

13 publishable Python notebooks covering all 25 reproduced papers — see `notebooks/papers/`.
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

## Sovereign Rust Evolution (May 2026)

The warm-catch breakthrough (Exp 188-190) was initially proven via shell scripts
and Python ("jelly strings"). These have been elevated to pure Rust in
**toadStool** (which absorbed the diesel engine lineage from coralReef):

- **ELF binary patcher** — Pure Rust ELF patcher (replaces `patch_nouveau_teardown.py`). Uses `object` crate.
- **Warm probe** — Standalone `WarmStateSnapshot` (PMC, PRAMIN, FECS, GPC registers).
- **Warm-catch orchestrator** — `device.warm_catch` JSON-RPC via toadStool ember. Era-aware settle durations.
- **`toadstool device warm-catch <BDF>`** — CLI entry point replacing all shell scripts.

> **Historical note:** Pre-May 2026 references to `coral-ember`, `coral-driver`, `coralctl` in
> archived experiments and scripts are **fossil record**. The diesel engine (ember/glowplug/cylinder)
> has been absorbed into toadStool. Read those names as historical context.

Original scripts archived in `scripts/archive/` as fossil record.

## Compute Trio Rewire + Capability Discovery (May 12, 2026)

hotSpring's interfaces with the compute trio (toadStool, barraCuda, coralReef)
have been rewired for capability-based discovery (GAP-HS-087, GAP-HS-088):

- **PrecisionTier/PhysicsDomain** re-exported from upstream barraCuda (15-tier/15-variant)
- **`toadstool-dispatch`** feature flag with `ToadStoolDispatchClient` (Phase C migration)
- **`validate_compute_trio_pipeline`** binary: end-to-end Yukawa + Wilson plaquette validation
- **All IPC provenance clients** evolved from hardcoded socket paths to `by_domain()` NUCLEUS discovery
- **Barrier shader validation** for coralReef `membar.{cta,gl}` emitter (9 WGSL shaders)
- **606** (cylinder) / **595** (default barracuda) / **1,041** (barracuda-local) lib tests pass

## Eukaryotic Evolution (May 2026)

17 experiment binaries have been absorbed into
`barracuda/src/validation/scenarios/` as Tier 1 (Rust) scenarios with
`ScenarioMeta` provenance tracking (20 with `barracuda-local` feature:
adds gradient-flow, dielectric-mermin, spectral-lanczos). 8 hardware-specific
GPU experiment binaries are preserved in `fossilRecord/experiments_prokaryotic_may2026/`.
The `hotspring_unibin validate` command is the eukaryotic entry point
for running all absorbed scenarios.
