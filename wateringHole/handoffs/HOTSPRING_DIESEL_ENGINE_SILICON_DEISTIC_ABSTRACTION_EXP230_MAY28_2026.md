# hotSpring → primalSpring Handoff: Diesel Engine Silicon Deistic Abstraction

**Date:** May 28, 2026
**Experiments:** 230–231 (Diesel Abstraction Revalidation + K80 Cross-Gen Quench Probe)
**Sprint:** S280
**Status:** Code complete, awaiting hardware revalidation

## Summary

Abstracted the diesel engine's crash defense and catalyst handoff pipeline from
Volta-specific hardcoding toward generation-aware "silicon deistic" goals. The
pre-handoff half of the diesel engine (catalyst pipeline, interrupt quench,
watchdog, patch sets) was deeply tied to GV100 + nvidia-470.256.02 — BAR0
offsets, GPC counts, interrupt register semantics, BDF strings all baked in.
This work makes adding a new GPU generation a data-driven addition rather than
a code surgery.

## Key Results

- **15 files** modified across `toadstool-cylinder` + `toadstool-server`
- **Full workspace compiles clean** (zero new warnings, zero errors)
- **Zero behavioral change** for existing Titan V path — same registers, cleaner dispatch
- **Kepler path unlocked** — `InterruptProfile::PRE_VOLTA` writes to 0x140 directly

## What Changed

### New Abstractions (toadStool → cylinder)
| Abstraction | File | Purpose |
|------------|------|---------|
| `InterruptProfile` | `nv/registers/pmc.rs` | Per-gen interrupt register semantics (direct vs SET/CLEAR) |
| `HandoffCapabilityProfile` | `sovereign_handoff/types.rs` | GPC count, register topology, BAR0 domains, PMC threshold |
| `PatchSet::from_recipe_toml()` | `module_patch/patch_sets/mod.rs` | Load patches from TOML recipes without recompiling |
| `PatchSet::by_profile()` | `module_patch/patch_sets/mod.rs` | Structured dispatch by (ChipFamily, driver, strategy) |
| `PatchStrategy::from_str()` | `module_patch/types.rs` | Parse TOML string format for patch strategies |
| `execute_handoff_with_heartbeat()` | `sovereign_handoff/pipeline.rs` | Pipeline heartbeat callback for watchdog integration |

### Refactored Hardcoding
| Was Hardcoded | Now Profile-Driven |
|--------------|-------------------|
| `0x180` INTR_EN_CLEAR | `InterruptProfile.disable_offset()` |
| `0..6u32` GPC loop | `hw.gpc_count` |
| `0x50_4000 + gpc * 0x8000` | `hw.tpc_base + gpc * hw.tpc_gpc_stride` |
| `0x80_0004` PCCSR base | `hw.pccsr_base` |
| `0x40_9000` FECS base | `hw.fecs_base` |
| `"0000:49:00.0"` in rm_trigger | `--bdf` CLI arg |
| `popcount < 10` warm check | `hw.pmc_warm_threshold` |
| `VOLTA_BAR0_DOMAINS` | `hw.bar0_domains` |
| `"_imem_gv100.bin"` firmware name | `hw.chip_name` |
| 120s watchdog timeout | Configurable via `activate()` |

### Files Changed (primals/toadStool)
- `crates/core/cylinder/Cargo.toml` — added `toml = "0.8"`
- `crates/core/cylinder/src/nv/registers/pmc.rs` — InterruptProfile + quench_interrupts + intx_disable
- `crates/core/cylinder/src/nv/generation.rs` — interrupt_profile field on all 11 generations
- `crates/core/cylinder/src/vfio/module_patch/types.rs` — PatchStrategy::from_str()
- `crates/core/cylinder/src/vfio/module_patch/patch_sets/mod.rs` — from_recipe_toml + by_profile
- `crates/core/cylinder/src/vfio/module_patch/patch_sets/nvidia.rs` — (no change, existing)
- `crates/core/cylinder/src/vfio/sovereign_handoff/types.rs` — HandoffCapabilityProfile + sm_version
- `crates/core/cylinder/src/vfio/sovereign_handoff/config.rs` — sm_version in all presets
- `crates/core/cylinder/src/vfio/sovereign_handoff/pipeline.rs` — profile-driven + heartbeats
- `crates/core/cylinder/src/vfio/sovereign_handoff/rm_trigger.rs` — shared quench
- `crates/core/cylinder/src/vfio/sovereign_handoff/pri_recovery.rs` — chip_name param
- `crates/core/cylinder/src/vfio/sovereign_handoff/mod.rs` — exports
- `crates/core/cylinder/src/bin/rm_trigger.rs` — --bdf CLI arg
- `crates/server/src/background/catalyst_watchdog.rs` — InterruptProfile + timeout
- `crates/server/src/pure_jsonrpc/handler/dispatch/sovereign.rs` — wired heartbeat + profile

### Files Changed (springs/hotSpring)
- `CHANGELOG.md` — abstraction entry
- `EXPERIMENT_INDEX.md` — Exp 230/231 entries, count 231
- `docs/PRIMAL_GAPS.md` — audit timestamp
- `experiments/results/experiment_catalog.json` — updated
- `experiments/230_DIESEL_ABSTRACTION_REVALIDATION.md` — new
- `experiments/231_K80_CROSSGEN_QUENCH_PROBE.md` — new

## Remaining Work

1. **Revalidation (Exp 230)**: Run catalyst handoff through abstracted infra on Titan V
2. **Cross-gen probe (Exp 231)**: Test PRE_VOLTA quench on K80 (when available)
3. **TOML recipe loading**: Wire `from_recipe_toml()` into pipeline dispatch path (currently available but not called from main path)
4. **Settle heuristic**: `settle_heuristic_ms` not yet added to profile (still by strategy name)

## Upstream Gaps for Primals Teams

- **toadStool**: `InterruptProfile` covers Kepler→Blackwell. Hopper and GB200 dual-die GPUs may need additional interrupt topology (per-MIG-instance quench).
- **primalSpring**: `PatchSet::from_recipe_toml()` enables a recipe-driven patch pipeline. Recipes could be versioned and distributed via wateringHole.
