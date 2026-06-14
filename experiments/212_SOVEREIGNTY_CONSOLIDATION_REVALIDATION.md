# Experiment 212: Sovereignty Consolidation & Revalidation

**Date:** 2026-05-20
**Hardware:** 2x NVIDIA Titan V (GV100), vfio-pci
**Spring:** hotSpring
**Status:** ✅ Complete — 3 abstraction gaps closed, pipeline consolidated
**Depends on:** Exp 197-211 (full sovereign arc)

## Objective

Review experiments 197-211, consolidate duplicate abstractions, close the
warm_capture→engine_ungate replay wire, make classification and validation
generation-aware via GenerationProfile, and revalidate the consolidated
pipeline before proceeding with K80/firmware extraction work.

## Background

Experiments 197-211 built the sovereign compute stack from RPC wiring through
tier classification, but the rapid experiment pace left three structural gaps
in the agnostic→atheistic→deistic progression:

1. **Warm capture learns but doesn't replay** — `WarmStateCapture` captures
   what nouveau initializes, `GrInitSequence` is extracted, but
   `SovereignStrategy::engine_ungate_sequences()` returns `None` for all
   strategies. The silicon-deistic replay path was open-circuited.

2. **Two parallel init abstractions** — `sovereign_init`/`SovereignStrategy`
   (production, 10+ stages) and `InitPipeline`/`BootPipeline` (4-phase
   sketch). `VoltaInit` was the default fallback for all unknown families.
   They needed a clear hierarchy.

3. **Hardcoded NVIDIA offsets in classification/validation** —
   `classify_tier()` and `ce_validate()` used literal register offsets
   (`0x41A004`, `0x104000`, etc.) instead of pulling from
   `GenerationProfile`. Cross-generation sovereign compute requires these
   to be data-driven.

## Phase 1: Warm Capture → Engine Ungate Wire (CLOSED)

The silicon-deistic bridge: nouveau initializes engines, `WarmStateCapture`
learns the sequence, sovereign pipeline replays it — eliminating the vendor
driver from subsequent boots.

### Changes

- **`sovereign_strategy.rs`**: Both `NvKeplerStrategy` and `NvAcrStrategy`
  now carry an optional `golden_sequences` field populated via
  `with_golden_sequences()`. `engine_ungate_sequences()` returns these
  when present, `None` when empty.

- **`sovereign_types.rs`**: Added `engine_init_path: Option<String>` to
  `SovereignInitOptions` — file path to a `GrInitSequence` JSON file
  for golden-state replay.

- **`gr_init.rs`**: Added `ChipFamily::engine_label()` for default engine
  naming when loading golden sequences.

- **RPC handler (`sovereign.rs`)**: Loads `GrInitSequence` from
  `engine_init_path`, deserializes via `from_json()`, pushes into
  `engine_init_sequences`. The existing `sovereign_init` engine_ungate
  stage replays them automatically.

### Wire

```text
WarmStateCapture → GrInitSequence.to_json() → golden-state.json
                                                    ↓
sovereign.init(engine_init_path="golden-state.json")
                                                    ↓
RPC handler loads → opts.engine_init_sequences
                                                    ↓
sovereign_init stage 3c: engine_ungate replays
```

This closes the **agnostic→atheistic** bridge: learn once from vendor
driver, replay forever from silicon knowledge.

### Tests

- `kepler_golden_sequences_wired` — PASS
- `volta_golden_sequences_wired` — PASS
- `empty_golden_sequences_returns_none` — PASS

## Phase 2: Init Pipeline Hierarchy (DOCUMENTED)

Two parallel init abstractions resolved. `InitPipeline`/`BootPipeline`
become the cross-vendor probe surface; `sovereign_init` remains the
production NVIDIA orchestrator.

### Changes

- **`init_pipeline.rs`**: Module-level doc rewritten to establish role
  hierarchy. `pipeline_for_family()` now returns `Option<Box<dyn
  InitPipeline>>` — returns `None` for unknown families instead of
  silently defaulting to `VoltaInit`.

- **`init_volta.rs`**: `BootPipeline::probe()` warm heuristic fixed from
  `pmc.count_ones() > 8` to `pmc.count_ones() >= 8 && pramin_accessible`,
  matching `is_warm_gpu()` in `sovereign_stages.rs`. Added PRAMIN window
  read (`0x700000`) as lightweight VRAM accessibility check via
  `RegisterAccess`.

- **`init_kepler.rs`**: Same warm heuristic fix applied. `verify()` also
  updated to `>= 8`.

## Phase 3: Generation-Aware Classification (IMPLEMENTED)

Hardcoded register offsets pulled into `GenerationProfile` so
`classify_tier()` and `ce_validate()` work across all NVIDIA generations
without per-function SM switches.

### Changes

- **`generation.rs`**: Added 5 new fields to `GenerationProfile`:
  - `fecs_pc_offset` (0x409624 — same across Kepler–Blackwell)
  - `gpc_broadcast_offset` (0x41A004)
  - `ce0_base_offset` (0x104000)
  - `pgraph_status_offset` (0x400700)
  - `ce_class` (varies: 0xA0B5 Kepler → 0xC3B5 Volta → 0xC5B5 Turing → etc.)

  All 11 generation profiles updated with correct values.

- **`sovereign_tiers.rs`**: Added `classify_tier_for_profile(bar0, profile)`
  that reads offsets from the profile. Original `classify_tier(bar0)` kept
  as backward-compatible convenience.

- **`ce_validate.rs`**: Added `validate_ce_with_profile(bar0, dma, profile)`
  that pulls CE class from `GenerationProfile`. `validate_ce()` wraps it
  with `None` (falls back to `VOLTA_DMA_COPY_A`).

### Tests

- `tier_offsets_present_on_all_profiles` — all 11 generations: PASS
- `ce_class_varies_by_generation` — Kepler/Volta/Turing: PASS

## Sovereignty Evolution Status

```text
EVOLUTION LADDER (updated Exp 212):

  ┌─ Vendor Agnostic ─────────────────────────────────────────┐
  │  BootPipeline, RegisterAccess, multi-vendor dispatch      │
  │  InitPipeline hierarchy documented, roles codified        │
  │  STATUS: ✅ ACHIEVED                                       │
  └───────────────────────────────────────────────────────────┘
                            ↓
  ┌─ Vendor Atheistic Infrastructure ─────────────────────────┐
  │  Tier 1 validated, 183ms warm, fd store, generation-aware │
  │  classify_tier_for_profile() — data-driven classification │
  │  STATUS: ✅ ACHIEVED                                       │
  └───────────────────────────────────────────────────────────┘
                            ↓
  ┌─ Vendor Atheistic Compute ────────────────────────────────┐
  │  Titan V: Tier 2 BLOCKED (GPC power domain)               │
  │  RTX 5060: VFIO 8/8 (KmodPromote via nvidia driver)      │
  │  K80: Priority 1 (unsigned falcons, no HS lock)           │
  │  STATUS: ⏳ PARTIALLY PROVEN                               │
  └───────────────────────────────────────────────────────────┘
                            ↓
  ┌─ Silicon Deistic ─────────────────────────────────────────┐
  │  Warm capture → replay wire CLOSED (Exp 212)              │
  │  engine_init_path loads golden-state GrInitSequence       │
  │  Learn once from vendor driver, replay forever            │
  │  STATUS: ⏳ WIRE CLOSED, AWAITING TIER 2 HARDWARE         │
  └───────────────────────────────────────────────────────────┘
```

## Revalidation Checklist

| Check | Result |
|-------|--------|
| `cargo check -p toadstool-cylinder` | ✅ 0 errors, warnings only |
| `cargo check -p toadstool-server` | ✅ 0 errors |
| `cargo test -p toadstool-cylinder -- sovereign_strategy` | ✅ 5/5 pass |
| `cargo test -p toadstool-cylinder -- init_pipeline` | ✅ 6/6 pass |
| `cargo test -p toadstool-cylinder -- generation` | ✅ 26/26 pass |
| Golden sequences wire: Kepler | ✅ PASS |
| Golden sequences wire: Volta | ✅ PASS |
| `pipeline_for_family(unknown)` returns `None` | ✅ PASS |
| BootPipeline warm heuristic matches `is_warm_gpu()` | ✅ PASS |
| All profiles have tier offsets | ✅ 11/11 |
| CE class varies by generation | ✅ PASS |

## Files Changed

### toadstool-cylinder (core)
- `crates/core/cylinder/src/vfio/sovereign_strategy.rs` — golden sequences
- `crates/core/cylinder/src/vfio/sovereign_types.rs` — `engine_init_path`
- `crates/core/cylinder/src/vfio/sovereign_tiers.rs` — `classify_tier_for_profile()`
- `crates/core/cylinder/src/vfio/ce_validate.rs` — `validate_ce_with_profile()`
- `crates/core/cylinder/src/vfio/init_pipeline.rs` — doc hierarchy, `Option` return
- `crates/core/cylinder/src/vfio/init_volta.rs` — warm heuristic fix
- `crates/core/cylinder/src/vfio/init_kepler.rs` — warm heuristic fix
- `crates/core/cylinder/src/nv/generation.rs` — tier offsets, CE class
- `crates/core/cylinder/src/nv/gr_init.rs` — `engine_label()`

### toadstool-server (RPC)
- `crates/server/src/pure_jsonrpc/handler/sovereign.rs` — `engine_init_path` loading

## Phase 5: Pre-K80 Readiness Check

| Item | Status | Detail |
|------|--------|--------|
| SM35 `GenerationProfile` | ✅ Ready | Kepler profile has correct offsets: `fecs_pc=0x409624`, `gpc_broadcast=0x41A004`, `ce0_base=0x104000`, `pgraph_status=0x400700`, `ce_class=0xA0B5` (KEPLER_DMA_COPY_A) |
| `PatchSet::kepler_warm_handoff()` | ✅ Ready | 5 targets: `gf100_gr_fini`, `nvkm_pmu_fini`, `nvkm_mc_disable`, `nvkm_mc_reset`, `gk104_fifo_fini`. All use `RetAfterFtrace` strategy |
| `WarmInitPlan::nouveau_k80()` | ✅ Ready | Uses `ModuleSource::System` (stock nouveau — no patching needed, unsigned falcons). Bare-metal safe |
| PLX bridge keepalive | ✅ Ready | `BridgeGuardian` in `glowplug::plx` auto-detects PEX 8747 ancestry, 5s config read keepalive. `sovereign_handoff` pins bridge hierarchy + disables FLR during swap |
| `classify_tier_for_profile(bar0, &KEPLER)` | ✅ Ready | Uses profile offsets. CE class 0xA0B5 wired for `validate_ce_with_profile()` |
| `NvKeplerStrategy` golden sequences | ✅ Ready | `with_golden_sequences()` wired, `engine_ungate_sequences()` returns them |
| K80 dual-die topology | ✅ Ready | `DeviceInit::dual_die()` models PLX topology, each die runs independent `InitPipeline` |

### Blockers

- **K80 hardware not yet arrived** — replacement on order
- Cannot run hardware validation until physical K80 present

## Next Steps

1. **K80 hardware validation** — when K80 arrives, run classify_tier_for_profile
   with SM35 profile, validate CE dispatch on unsigned-falcon Kepler
2. **Golden-state capture on Titan V** — run WarmStateCapture, save JSON,
   replay via engine_init_path to test the full wire on Volta
3. **PMU firmware extraction from nvidia-470** — alternate Tier 2 path
4. **VBIOS interpreter opcode archaeology** — Tier 3 path
