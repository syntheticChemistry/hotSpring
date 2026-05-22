# Experiment 219 — Catalyst Driver Pattern for TPC Sovereignty

**Date**: 2026-05-22
**Status**: INFRASTRUCTURE COMPLETE — awaiting hardware execution
**Hardware**: 2x NVIDIA Titan V (GV100), BDFs `0000:02:00.0`, `0000:49:00.0`
**Dependency**: Exp 218 (nvidia-470 dual-load co-existence proved), Exp 217 (TPC wall confirmed firmware-dependent)

## Objective

Treat the proprietary nvidia-470 driver as a chemical catalyst: load it to
fully initialize the GPU compute pipeline (SEC2→ACR→PMU→GPCCS→FECS→TPC),
capture the resulting register state as a "golden snapshot", remove the
catalyst (warm swap to vfio-pci), and replay the captured state on future
boots — achieving TPC sovereignty without the proprietary driver at runtime.

## Core Insight

A catalyst enables a reaction but is not consumed in the product. The
nvidia-470 driver initializes GPU state that we want (TPC PRI stations);
we capture that state, remove the driver, and replay it. The twin Titan Vs
provide the A/B surface: Card A stays cold as the control, Card B runs the
catalyst, and the delta between them IS the catalyst product.

## Changes from Exp 218

### Phase 1: Selective Un-NOP (`nvidia_catalyst_handoff`)

Exp 218 used `nvidia_warm_handoff` (17 targets), which NOPed `nv_cap_init`
and `nv_cap_drv_init`. This prevented RM capability table creation and
blocked engine initialization — result was Tier 0 (Cold).

The new `nvidia_catalyst_handoff` (15 targets) removes those two from the
NOP set, allowing RM to fully initialize while keeping all co-load isolation
NOPs to prevent host conflicts.

**Removed from NOP set:**
- `nv_cap_init` (was Ret1AtEntry)
- `nv_cap_drv_init` (was Ret1AtEntry)

**Kept (co-load isolation):**
All procfs, chardev, nvlink, nvswitch, and ACPI NOPs remain.

### Phase 2: Catalyst Capture Step

Inserted between `seeder_settle` and `prepare_warm_swap` in the handoff
pipeline. While the catalyst driver owns the GPU at peak initialization:

1. Opens BAR0 via `MappedBar::from_sysfs_rw()`
2. Captures `SovereignSnapshot` for tier-relevant registers
3. Captures `Bar0Snapshot::capture_full()` for all 16 MiB
4. Builds `GrInitSequence` via `to_catalyst_replay()`
5. Persists snapshot + replay sequence to `/tmp/toadstool-catalyst-*.json`
6. Logs tier evidence (TPC alive? FECS running?)

### Phase 3: Twin-Card Differential

New `sovereign.catalyst_diff` RPC captures full BAR0 from both cards,
computes `Bar0Diff`, and produces a minimal `GrInitSequence` containing
only the registers the catalyst changed. Persists cold baseline, warm
snapshot, delta, and replay sequence.

### Phase 4: Catalyst Preservation (3 Layers)

**Layer 1 — Recipe (mother culture):**
`infra/catalysts/recipes/gv100_nvidia470.toml` — versioned manifest of
DKMS source, NOP set, objcopy sections, module rename. Given the same
inputs, reproduces an identical catalyst `.ko`.

**Layer 2 — Frozen Binary (store-bought starter):**
`/var/lib/toadstool/catalysts/frozen/nvsov_gv100_470.256.02_k*.ko` —
archived on successful catalyst handoff. Skip the patch pipeline when
the kernel matches.

**Layer 3 — Product (golden state):**
`infra/catalysts/products/` — full BAR0 snapshot, cold→catalyst diff,
and replay sequence as JSON.

### Phase 5: Golden State Persistence & Replay

`engine_init_path` now wired in the ember handler for `sovereign.init`.
Loads `GrInitSequence` from JSON and feeds it into `engine_init_sequences`
for replay during Stage 3c (Engine Ungating).

### Phase 6: Catalyst-Free Boot

New `sovereign.catalyst_boot` RPC orchestrates:
1. `nouveau_titanv` warm handoff (183ms)
2. `sovereign.init` with `engine_init_path` pointing to golden state JSON
3. Returns combined result with tier classification

## New Code

### cylinder crate
- `gr_init.rs`: `InitSource::Catalyst { driver_version, bdf }` variant
- `pri.rs`: `VOLTA_BAR0_DOMAINS` constant (22 domain ranges)
- `module_patch.rs`: `nvidia_catalyst_handoff()` patch set (15 targets),
  `PatchSet::to_json()`, `by_name("nvidia_catalyst_handoff")`
- `warm_capture.rs`: `Bar0Snapshot::to_catalyst_replay()`,
  `Bar0Snapshot::to_json()`, `Bar0Diff::to_replay_sequence()`,
  `Bar0Diff::to_json()`
- `sovereign_handoff.rs`: `HandoffConfig::nvidia_catalyst_titanv()`,
  catalyst capture step, catalyst preservation step,
  `HandoffResult::{catalyst_snapshot_path, catalyst_alive_count, catalyst_tier}`

### server crate
- `dispatch/mod.rs`: `sovereign_catalyst_boot` RPC, `engine_init_path`
  loading in ember handler, `nvidia_catalyst_titanv` in strategy list
- `sovereign.rs`: `sovereign_catalyst_diff` RPC
- `mod.rs`: routing for `sovereign.catalyst_boot`, `sovereign.catalyst_diff`

### infra
- `infra/catalysts/recipes/gv100_nvidia470.toml` — catalyst recipe
- `infra/catalysts/frozen/` — archived catalyst `.ko` files
- `infra/catalysts/products/` — captured catalyst product JSONs
- `infra/golden_state/` — golden state replay sequences

## New RPCs

| RPC | Type | Purpose |
|-----|------|---------|
| `sovereign.warm_handoff` (strategy=`nvidia_catalyst_titanv`) | Blocking | Catalyst handoff with capture |
| `sovereign.catalyst_diff` | Stateless | Full BAR0 twin-card differential |
| `sovereign.catalyst_boot` | Ember | Catalyst-free boot (nouveau + replay) |

## Execution Plan

```
1. sovereign.snapshot bdf=0000:02:00.0                           # Card A cold baseline
2. sovereign.warm_handoff bdf=0000:49:00.0                      # Card B catalyst handoff
   strategy=nvidia_catalyst_titanv
3. sovereign.catalyst_diff bdf_cold=0000:02:00.0                # Twin-card delta
   bdf_warm=0000:49:00.0
   persist_path=infra/catalysts/products
4. cp /tmp/toadstool-catalyst-replay-*.json                     # Archive golden state
   infra/golden_state/gv100_catalyst.json
5. sovereign.catalyst_boot bdf=0000:49:00.0                     # Catalyst-free validation
   engine_init_path=infra/golden_state/gv100_catalyst.json
```

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| RM init still blocked by remaining NOPs | Exclusive session — unload all nvidia, load unpatched 470 |
| TPC state doesn't survive warm swap | Already mitigated by `disable_flr` + `pin_bridge_hierarchy` |
| BAR0 capture misses internal GPU state | Capture PRAMIN window; use `sw_nonctx.bin` + golden diff |
| Replayed state depends on firmware shadows | Iterate: capture → swap → classify → refine |
| Zombie modules from failed attempts | Dynamic `module_name` override or reboot between attempts |
