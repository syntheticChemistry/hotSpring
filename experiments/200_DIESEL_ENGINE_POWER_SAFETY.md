# Experiment 200 — Diesel Engine Power Safety

**Date:** May 17, 2026
**Status:** ✅ Validated (builds clean, ready for hardware test on Titan V)

## Motivation

Experiment 199 ended with a Tesla K80 catching fire on reboot. Post-mortem
analysis identified the root cause: writing `0xFFFF_FFFF` to PMC_ENABLE on a
cold Kepler GPU with uninitialised GDDR5 instantly ungated all engine clock
domains. The resulting inrush current exceeded the aged K80's VRM capacity.

Key insight: pre-firmware generations (Kepler, Maxwell) have no hardware power
sequencing — VBIOS devinit IS the power sequencer. Writing full PMC_ENABLE
before devinit completes is inherently unsafe on these parts.

## Changes

### 1. `PowerSafetyProfile` (generation.rs)

New struct encoding per-generation power safety policy:

```
PowerSafetyProfile {
    initial_pmc_mask: u32,         // what to write in stage 2
    full_enable_after_devinit: bool, // safe to write 0xFFFFFFFF after devinit?
    rollback_on_devinit_failure: bool, // restore PMC_ENABLE if devinit fails?
}
```

Two const profiles:
- `PRE_FIRMWARE` — mask `0xC000_2030` (PPCI + PBUS + PTIMER + PFIFO only),
  rollback on failure, no full enable after devinit
- `FIRMWARE_MANAGED` — full `0xFFFF_FFFF`, no rollback needed

### 2. Generation Profile Annotations

All 10 generation profiles (`KEPLER` through `BLACKWELL_B`) now carry
`power_safety`:

| Generation | Profile |
|-----------|---------|
| Kepler | `PRE_FIRMWARE` |
| Maxwell | `PRE_FIRMWARE` |
| Pascal | `FIRMWARE_MANAGED` |
| Volta | `FIRMWARE_MANAGED` |
| Turing | `FIRMWARE_MANAGED` |
| Ampere A | `FIRMWARE_MANAGED` |
| Ampere B | `FIRMWARE_MANAGED` |
| Ada | `FIRMWARE_MANAGED` |
| Hopper | `FIRMWARE_MANAGED` |
| Blackwell A | `FIRMWARE_MANAGED` |
| Blackwell B | `FIRMWARE_MANAGED` |

### 3. Staged PMC_ENABLE (sovereign_stages.rs)

`pmc_enable()` now:
1. Accepts `&PowerSafetyProfile`
2. Writes `power.initial_pmc_mask` instead of `0xFFFF_FFFF`
3. Returns `PmcEnableResult { before, after, mask }` for rollback

New functions:
- `pmc_enable_rollback(bar0, restore_value)` — restores PMC_ENABLE on failure
- `pmc_enable_full(bar0)` — writes `0xFFFF_FFFF` post-devinit (firmware-managed only)

### 4. Pipeline Restructure (sovereign_init.rs)

- Generation profile resolved in stage 1 (was after stage 3)
- Stage 2 uses staged `pmc_enable(bar0, &profile.power_safety)`
- New stage 3b: `pmc_enable_full()` after successful devinit, only for
  `FIRMWARE_MANAGED` profiles
- Devinit failure on `PRE_FIRMWARE` GPUs triggers `pmc_enable_rollback()`

## Pipeline Flow (Before vs After)

### Before (Exp 199)
```
bar0_probe → pmc_enable(0xFFFFFFFF) → memory_training → falcon_boot → ...
                    ↑ K80 fire here
```

### After (Exp 200)
```
bar0_probe → resolve_gen → pmc_enable(conservative_mask) → memory_training
                                                               ↓
                                                    [if FAIL + rollback_on_failure]
                                                    → pmc_enable_rollback(before_value) → STOP
                                                               ↓
                                                    [if OK + full_enable_after_devinit]
                                                    → pmc_enable_full(0xFFFFFFFF) → falcon_boot → ...
```

## Build Validation

```
cargo check → ✅ (only pre-existing warning in pfifo.rs)
```

## What This Prevents

1. **Kepler/Maxwell fire risk**: PMC_ENABLE limited to essential buses.
   No PGRAPH, CE, NVDEC, or memory controller engines enabled before devinit.
2. **Partial-clock persistence**: If devinit fails, PMC_ENABLE is rolled back
   to its pre-pipeline value — no partially-clocked state survives power cycles.
3. **Future pre-firmware GPUs**: Any new generation profile just picks
   `PRE_FIRMWARE` or `FIRMWARE_MANAGED`.

## Next Steps

- [ ] Live validation on Titan V (GV100, `FIRMWARE_MANAGED` — should behave
  identically to Exp 199 since mask is 0xFFFFFFFF)
- [ ] When replacement K80 arrives, validate `PRE_FIRMWARE` path with
  conservative mask and rollback
- [ ] PGRAPH CG ungating for Volta cold path (still blocks FECS HS boot)
