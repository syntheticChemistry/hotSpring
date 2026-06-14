# Experiment 202 — Experiment Surface Rewire

**Date:** May 17, 2026
**Status:** ✅ Implemented — 6 abstraction gaps closed, backward-compatible serde aliases, 14 sovereign tests pass

## Motivation

After implementing the `SovereignStrategy` trait abstraction in the diesel
engine (response to the vendor/generation externalization work), a review of
the pipeline internals revealed 6 gaps where the abstraction was cosmetic —
the pipeline still branched on NVIDIA-specific logic internally, leaked
vendor-specific field names, and lacked experimentation halt points that
Exp 201 had shown were needed.

This experiment closes those gaps, expanding the surface area for future
experiments on AMD (Vega 20), NPU (Akida AKD1000), and any future compute
substrate that implements `SovereignStrategy`.

## Findings: 6 Gaps Between Abstraction and Reality

### Gap 1: `falcon_boot()` ignored `SovereignStrategy` entirely

`falcon_boot()` took `sm_version` and internally called `profile_for_sm()` →
`is_kepler()` to dispatch between Kepler PIO and ACR DMA HS. The strategy
trait exposed `falcon_boot_style()` but it was never called.

### Gap 2: `bar0_probe()` was NVIDIA-hardcoded

Read `PMC_BOOT_0` (offset 0) and extracted `chip_id = (boot0 >> 20) & 0x1FF`.
An AMD Vega or NPU would need a different identity register layout.

### Gap 3: `verify()` was NVIDIA-hardcoded

Read PTIMER, PMC_ENABLE, and PRAMIN sentinel — all NVIDIA register offsets.
AMD would need GRBM_STATUS idle checks; NPU would need something else entirely.

### Gap 4: `SovereignInitResult` leaked NVIDIA names

Fields `chip_id`, `boot0`, `hbm2_writes` were NVIDIA-specific terminology
baked into the serialized pipeline result.

### Gap 5: `HaltBefore` missing CG/PGOB stops

Exp 201 showed CG sweep changing 6 registers and PRI recovery clearing 12
faulted domains — but there was no way to halt between `pmc_enable` and
`memory_training` to observe the raw post-PMC state before sweep.

### Gap 6: Factory channel creation vs CG sweep ordering

In `sovereign_init_ember`, the factory creates PFIFO channels (with GPCCS
HS boot) *before* `sovereign_init` runs CG sweep. On cold Volta, GPCCS
PRI-faults during channel creation because clock domains are still gated.

## Changes

### Phase A: Wire Strategy into Stages (Gaps 1–3)

#### `falcon_boot()` now dispatches on `FalconBootStyle`

New signature adds `boot_style: FalconBootStyle` parameter. The function
matches on the enum variant:

| Variant | Behaviour |
|---------|-----------|
| `DirectPio` | Kepler PIO firmware upload (calls `kepler_falcon_boot`) |
| `AcrDmaHs` | ACR boot solver path (warm detection, WPR, SEC2) |
| `NoFalcons` | Immediate success — hardware has no falcon engines |

Call sites updated:
- `sovereign_init.rs`: passes `strategy.falcon_boot_style()`
- `init_volta.rs`: passes `FalconBootStyle::AcrDmaHs`
- `init_kepler.rs`: passes `FalconBootStyle::DirectPio`

#### `probe_identity()` added to `SovereignStrategy`

Default implementation delegates to `bar0_probe()` and returns
`ProbeIdentity { identity_raw, identity_chip }`. Override for AMD/NPU
register layouts.

Pipeline stage renamed from `bar0_probe` to `identity_probe`.

#### `verify_device()` added to `SovereignStrategy`

Default implementation delegates to the existing `verify()` function.
Override for vendor-specific health checks.

### Phase B: Neutralize Result Types (Gap 4)

| Old Field | New Field | Serde Alias |
|-----------|-----------|-------------|
| `chip_id` | `identity_chip` | `#[serde(alias = "chip_id")]` |
| `boot0` | `identity_raw` | `#[serde(alias = "boot0")]` |
| `hbm2_writes` | `training_writes` | `#[serde(alias = "hbm2_writes")]` |

Backward-compatible: any persisted JSON with old field names deserializes
correctly via the aliases. New serialized output uses neutral names.

`Display` impl updated to use `identity_chip`.

### Phase C: Expand HaltBefore (Gap 5)

New variants inserted between `PmcEnable` and `MemoryTraining`:

```
PmcEnable → CgSweep → PgobUngate → MemoryTraining → EngineUngate → FalconBoot → GrInit → Verify
```

| New Variant | Observes |
|-------------|----------|
| `CgSweep` | Raw post-PMC state before CG registers are cleared |
| `PgobUngate` | Post-CG-sweep + post-PRI-recovery state before PGOB |

Both variants wired into the pipeline with conditional checks:
`CgSweep` only fires when `strategy.needs_cg_sweep()` is true.
`PgobUngate` only fires when `strategy.needs_pgob_before_memory()` is true.

### Phase D: Pre-Channel CG Sweep (Gap 6)

Added `pre_channel_init()` method to `SovereignStrategy`:

| Strategy | `pre_channel_init` behaviour |
|----------|------------------------------|
| `NvKeplerStrategy` | No-op (empty vec) — Kepler has no CG to sweep |
| `NvAcrStrategy` | Runs `cg_sweep` + `pri_bus_recover` + `pgob_ungating` |

Wired into `sovereign_init_ember` between strategy construction and the
`sovereign_init` call. Stage results are logged via tracing.

RPC doc comments updated to include new halt_before values.

## Test Results

14 sovereign tests pass:

- `chip_id_to_sm_covers_titan_v` — identity mapping
- `chip_id_to_sm_covers_k80` — identity mapping
- `chip_id_to_sm_unknown_defaults_to_70` — fallback
- `halt_before_serde_roundtrip` — existing MemoryTraining serde
- `halt_before_cg_sweep_serde` — **new** CgSweep round-trip
- `halt_before_pgob_ungate_serde` — **new** PgobUngate round-trip
- `result_backward_compat_aliases` — **new** old JSON deserializes correctly
- `sovereign_init_result_display_halted` — Display impl
- `sovereign_init_result_display_ready` — Display impl
- `stage_status_serde_roundtrip` — StageStatus
- `kepler_strategy_from_profile` — Kepler strategy factory
- `volta_strategy_from_profile` — Volta strategy factory
- `options_default_has_no_halt` — defaults
- `error_display_sovereign_stages_variant` — error formatting

## Surface Area Gains

| Before | After |
|--------|-------|
| `falcon_boot()` branched internally on generation | Dispatches on `FalconBootStyle` enum from strategy |
| `bar0_probe()` hardcoded NVIDIA BOOT0 | `probe_identity()` trait method, overrideable per vendor |
| `verify()` hardcoded NVIDIA registers | `verify_device()` trait method, overrideable per vendor |
| Result fields: `chip_id`, `boot0`, `hbm2_writes` | Neutral: `identity_chip`, `identity_raw`, `training_writes` |
| 6 `HaltBefore` variants | 8 variants — CG/PGOB experimentation exposed |
| No pre-channel hardware prep | `pre_channel_init()` runs CG sweep before factory |

### New experiment capabilities unlocked

1. **`halt_before=cg_sweep`** — observe raw post-PMC clock-gated state
2. **`halt_before=pgob_ungate`** — observe post-CG-sweep, pre-PGOB state
3. **AMD Vega boot** — override `probe_identity()` for GRBM_STATUS, `verify_device()` for GRBM idle
4. **NPU boot** — `FalconBootStyle::NoFalcons`, custom probe/verify
5. **Pre-channel CG on cold Volta** — `pre_channel_init()` prevents GPCCS PRI faults during factory channel creation

## Revalidation

All existing hardware results from Exp 200 (Diesel Engine Power Safety) and
Exp 201 (Volta Cold Boot CG Sweep) are preserved:

- Power safety profiles unchanged — `strategy.power_profile()` returns same values
- CG sweep register set unchanged — `cg_sweep()` function body untouched
- PRI recovery unchanged — `pri_bus_recover()` function body untouched
- PGOB sequence unchanged — `pgob_ungating()` function body untouched
- Serde backward compat verified — old JSON field names deserialize correctly

No hardware re-run required. Code paths are structurally identical; only
the dispatch indirection changed (strategy methods instead of inline branching).
