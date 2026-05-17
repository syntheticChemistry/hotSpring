# Experiment 203 — Warm/Cold Boot Convergence & Firmware Bridge Freeze

**Date:** May 17, 2026
**Status:** ✅ Implemented — VBIOS PLL opcodes activated, falcon warm-state extracted to enum, PfifoInitConfig unified, NvGspBridge documented as frozen dependency

## Motivation

Analysis of the warm and cold boot paths revealed:

1. Shared stages (identity probe through PGOB ungating, plus verify) are genuinely idempotent — running on a warm GPU produces no-op results. The divergence is entirely in memory training (cold-only) and falcon/GR boot (warm skips).

2. The VBIOS interpreter had 6 PLL-related opcodes (0x79, 0x4B, 0x34, 0x4A, 0x59, 0x87) that were stub-handled — stream alignment was preserved but no BAR0 writes occurred. These opcodes program clock PLLs and RAM-restrict conditional register writes that are critical for cold HBM2 training on GV100.

3. Falcon warm-state detection was inlined as raw BAR0 register reads in `falcon_boot()`, making the 4-way thermal classification (preserved / running / inconsistent / cold) opaque to the strategy layer.

4. `PfifoInitConfig` selection was scattered across `if warm { warm_handoff() } else { default() }` branches without considering the nuance of FECS preservation state.

5. `NvGspBridge` had no documentation explaining why it is a stable, frozen dependency that evolves glacially.

## Changes

### Phase 1: VBIOS Interpreter PLL Opcode Activation

Upgraded 6 stub opcodes to perform actual BAR0 writes:

| Opcode | Name | Before | After |
|--------|------|--------|-------|
| **0x79** | INIT_PLL | Skip (stride only) | Writes pre-computed PLL coefficient word to register |
| **0x4B** | INIT_PLL_INDIRECT | Skip | Writes freq value from ROM to PLL register |
| **0x34** | INIT_RAM_RESTRICT_PLL | Skip | Selects PLL value by RAM strap index, writes to register |
| **0x4A** | INIT_RAM_RESTRICT_PLL (variant) | Skip | Same as 0x34 |
| **0x59** | INIT_PLL2 | Skip | Writes extended PLL value to register |
| **0x87** | INIT_RAM_RESTRICT_ZM_REG | Skip | Writes register value selected by RAM strap index |

Also activated 3 previously-stubbed register copy opcodes:

| Opcode | Name | Before | After |
|--------|------|--------|-------|
| **0x88** | INIT_RAM_RESTRICT_ZM_REG_GROUP | Skip | Writes N sequential registers with RAM-strap-selected values |
| **0x8F** | (variant of 0x88) | Skip | Same as 0x88 |
| **0x90** | INIT_COPY_ZM_REG | Skip | Copies value from src register to dst register |
| **0x5F** | INIT_COPY_NV_REG | Skip | Masked register-to-register copy with shift |

### Phase 2: FalconWarmState Enum Extraction

New `FalconWarmState` enum on `SovereignStrategy`:

- `Cold` — no prior driver session, full boot needed
- `WarmPreserved { cpuctl, mailbox0 }` — FECS frozen by livepatch, skip all boot
- `WarmRunning { cpuctl, pc, mailbox0 }` — FECS actively running, skip all boot
- `Inconsistent { cpuctl }` — teardown state (0x12), try PIO re-bootstrap

New `detect_falcon_warm_state()` default method on `SovereignStrategy` reads FECS CPUCTL/MAILBOX0/PC and returns the enum. `NvKeplerStrategy` overrides to always return `Cold` (PIO path).

`falcon_boot()` signature changed: `warm_detected: bool` → `warm_state: FalconWarmState`. The function now dispatches on the enum instead of inline BAR0 register reads.

### Phase 2: PfifoInitConfig Convergence

New `PfifoInitConfig::for_thermal_state(warm: bool, fecs_preserved: bool)` unifies the selection logic:
- Cold → `default()` (aggressive fault clearing, full glow plug)
- Warm + FECS preserved → `warm_fecs_alive()` (gentlest path)
- Warm + FECS not preserved → `warm_handoff()` (standard warm)

New `pfifo_config()` method on `SovereignStrategy` trait uses `FalconWarmState` to drive selection.

### Phase 3: NvGspBridge Frozen Dependency Documentation

Documented `NvGspBridge` and `GspBridge` trait with:
- Firmware blobs are pinned artifacts, do not change per-chip
- Upload mechanisms (PIO, DMA HS) are hardware-defined register sequences
- Rust code evolves glacially — changes only for new GPU generations
- Future bridges (AMD, NPU) follow the same frozen-blob + pure-Rust pattern

## Verification

- `cargo check` — cylinder + server: clean (1 pre-existing warning)
- `cargo test` — 611 tests pass, 0 failures
- Backward compatibility preserved via `FalconWarmState::Cold` at legacy call sites

## Files Changed

- `cylinder/src/vfio/channel/devinit/script/interpreter/opcodes.rs` — PLL and RAM-restrict opcodes activated
- `cylinder/src/vfio/sovereign_strategy.rs` — `FalconWarmState` enum, `detect_falcon_warm_state()`, `pfifo_config()`, Kepler override
- `cylinder/src/vfio/sovereign_stages.rs` — `falcon_boot()` signature updated to `FalconWarmState`
- `cylinder/src/vfio/sovereign_init.rs` — Uses `strategy.detect_falcon_warm_state()` before `falcon_boot()`
- `cylinder/src/vfio/init_kepler.rs` — Updated to `FalconWarmState::Cold`
- `cylinder/src/vfio/init_volta.rs` — Updated to `FalconWarmState::Cold`
- `cylinder/src/vfio/channel/pfifo.rs` — `PfifoInitConfig::for_thermal_state()`
- `cylinder/src/vfio/channel/mod.rs` — Uses `for_thermal_state()` in `create_for_profile()`
- `cylinder/src/nv/nv_gsp_bridge.rs` — Frozen dependency documentation
- `cylinder/src/nv/gsp_bridge.rs` — Frozen dependency pattern documentation on trait
