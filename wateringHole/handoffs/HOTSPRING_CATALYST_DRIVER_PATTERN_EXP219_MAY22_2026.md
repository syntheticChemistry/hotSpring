# Catalyst Driver Pattern for TPC Sovereignty — hotSpring Handoff

**Date:** May 22, 2026
**From:** hotSpring
**To:** primalSpring (audit), toadStool (upstream code landed)
**Status:** Infra complete — 6 phases implemented, 63 tests pass, awaiting HW execution
**Experiments:** 219 (builds on 217, 218)

## Summary

Exp 219 evolves the nvidia-470 dual-load injection (Exp 218) into a
**Catalyst Driver Pattern** — the proprietary driver is treated as a
chemical catalyst: used to initialize GPU state, then removed. The
initialized state (BAR0 registers) is captured as a "golden state"
product that can be replayed on future boots without the catalyst.

### The Chemistry Analogy

| Chemistry            | GPU Sovereignty              |
|----------------------|------------------------------|
| Catalyst             | nvidia-470 proprietary driver |
| Reaction             | TPC/GPC/FECS initialization   |
| Product              | BAR0 golden state (register snapshot) |
| Store-bought reagent | Frozen `.ko` binary (pre-patched) |
| Recipe               | Catalyst recipe TOML (NOP set + build params) |
| Reformulation        | Re-run with different driver version or NOP set |

## What Changed from Exp 218

Exp 218 achieved dual-load (nvsov alongside nvidia-580) but got `Cold`
tier because `nv_cap_init` / `nv_cap_drv_init` were NOP-patched, blocking
RM capability table creation and preventing full compute init.

Exp 219 fixes this with a **selective un-NOP**: the `nvidia_catalyst_handoff`
patch set removes `nv_cap_init` and `nv_cap_drv_init` from the NOP list,
allowing RM to build capability tables while maintaining co-load isolation.

## New Infrastructure (landed in toadStool)

### Patch Set
- **`PatchSet::nvidia_catalyst_handoff()`**: 13 NOP targets (vs 15 in
  `nvidia_warm_handoff`). `nv_cap_init` and `nv_cap_drv_init` restored.
- **`PatchSet::to_json()`**: Serializes patch set for recipe archival.

### Handoff Pipeline
- **`HandoffConfig::nvidia_catalyst_titanv()`**: Catalyst-specific config
  with 15s settle time and `nvidia_catalyst_handoff` patch set.
- **Catalyst capture step**: Between `seeder_settle` and `prepare_warm_swap`
  in `execute_handoff()` — captures `SovereignSnapshot`, full 16 MiB
  `Bar0Snapshot`, builds `GrInitSequence` via `to_catalyst_replay()`,
  persists all to `/tmp/toadstool-catalyst-*.json`.
- **Catalyst preservation step**: Before module cleanup — copies frozen
  `.ko` to `/var/lib/toadstool/catalysts/frozen/`, persists patch set
  JSON to `/var/lib/toadstool/catalysts/recipes/`.

### BAR0 Domain Mapping
- **`VOLTA_BAR0_DOMAINS`**: 22-entry constant mapping BAR0 offset ranges
  to human-readable domain names (GPC, TPC, PPC, FBPA, etc.). Used by
  catalyst capture for register labeling.

### Replay Infrastructure
- **`Bar0Snapshot::to_catalyst_replay()`**: Filters non-zero, non-PRI-fault
  values to create a minimal `GrInitSequence` for golden state replay.
- **`Bar0Diff::to_replay_sequence()`**: Converts changed registers from
  a twin-card diff into a minimal replay sequence.
- **`InitSource::Catalyst`**: New variant with `driver_version` and `bdf`
  metadata for provenance tracking.

### New RPCs (3)
| RPC | Purpose |
|-----|---------|
| `sovereign.catalyst_diff` | Capture full BAR0 from cold + warm GPUs, compute diff, produce minimal replay sequence |
| `sovereign.catalyst_boot` | Catalyst-free boot: nouveau warm handoff + golden state replay via `sovereign.init` |
| `sovereign.warm_handoff nvidia_catalyst_titanv` | Full catalyst pipeline: patch, load, settle, capture, preserve, swap |

### Engine Init Path
- **`sovereign.init_ember`**: Now accepts `engine_init_path` parameter —
  loads a `GrInitSequence` JSON from disk and appends to init sequences
  for golden state replay.

## 3-Layer Catalyst Preservation

| Layer | Artifact | Location | Purpose |
|-------|----------|----------|---------|
| 1. Recipe | `gv100_nvidia470.toml` | `infra/catalysts/recipes/` | Reproducible build params + NOP set |
| 2. Frozen binary | `nvsov_gv100_470.256.02_k*.ko` | `infra/catalysts/frozen/` | Pre-patched module for direct reuse |
| 3. Product | BAR0 snapshot + diff + replay JSONs | `infra/catalysts/products/` | Captured golden state for replay |

## Files Changed (toadStool)

| File | Changes |
|------|---------|
| `cylinder/src/vfio/module_patch.rs` | `nvidia_catalyst_handoff()` patch set, `to_json()` |
| `cylinder/src/vfio/sovereign_handoff.rs` | `nvidia_catalyst_titanv()` config, catalyst capture step, catalyst preservation step, new `HandoffResult` fields |
| `cylinder/src/vfio/warm_capture.rs` | `to_catalyst_replay()`, `to_replay_sequence()`, `to_json()` methods |
| `cylinder/src/nv/gr_init.rs` | `InitSource::Catalyst` variant |
| `cylinder/src/nv/pri.rs` | `VOLTA_BAR0_DOMAINS` constant |
| `server/handler/sovereign.rs` | `sovereign.catalyst_diff` RPC |
| `server/handler/dispatch/mod.rs` | `sovereign.catalyst_boot` RPC, `engine_init_path` wiring |
| `server/handler/mod.rs` | Routing for new RPCs |

## Validation Results

- 63 tests pass (all relevant cylinder + server tests)
- `cargo check` clean, `cargo test` clean
- `PatchSet::nvidia_catalyst_handoff()` unit test validates 13 targets
- `InitSource::Catalyst` serde roundtrip verified
- `Bar0Snapshot::to_catalyst_replay()` filters PRI faults correctly
- `Bar0Diff::to_replay_sequence()` produces minimal delta
- `HandoffConfig::from_strategy("nvidia_catalyst_titanv")` resolves

## Execution Plan (post-push)

```bash
# Phase 1: Catalyst handoff — load nvidia-470, capture golden state
curl -s http://localhost:7700/rpc -d '{
  "jsonrpc":"2.0","id":1,
  "method":"sovereign.warm_handoff",
  "params":{"bdf":"0000:41:00.0","strategy":"nvidia_catalyst_titanv"}
}'
# Expected: HandoffResult with catalyst_snapshot_path, catalyst_tier

# Phase 2: Twin-card differential
curl -s http://localhost:7700/rpc -d '{
  "jsonrpc":"2.0","id":2,
  "method":"sovereign.catalyst_diff",
  "params":{
    "bdf_cold":"0000:42:00.0",
    "bdf_warm":"0000:41:00.0",
    "persist_path":"/var/lib/toadstool/catalysts/products/"
  }
}'
# Expected: diff_count, replay_count, persisted JSONs

# Phase 3: Catalyst-free boot using golden state
curl -s http://localhost:7700/rpc -d '{
  "jsonrpc":"2.0","id":3,
  "method":"sovereign.catalyst_boot",
  "params":{
    "bdf":"0000:41:00.0",
    "engine_init_path":"/var/lib/toadstool/catalysts/products/replay.json"
  }
}'
# Expected: TierEvidence with tpc_alive = true
```

## Upstream Gaps for primalSpring

- **Catalyst safety model**: What happens if catalyst driver crashes during
  capture? Recovery path needs definition.
- **Cross-generation portability**: Golden state from GV100 cannot be
  replayed on GP102. Recipe system needs generation guards.
- **Catalyst version matrix**: Which nvidia driver versions work as
  catalysts for which GPU generations? Need systematic testing.
- **`objcopy` dependency**: Still shells out for ksymtab stripping.
  Pure-Rust ELF section removal would eliminate this.
- **Module naming policy**: `nvsov` naming convention needs ecosystem
  standardization (from Exp 218 gap, still open).

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Catalyst driver crash during capture | Fork-isolated MMIO + ember survivability hardening |
| Golden state replay produces PRI faults | `to_catalyst_replay()` filters `0xbadf*` values |
| nvidia-580 host driver interference | Co-load isolation NOPs (procfs, chardev, ACPI, nvlink) |
| Kernel version sensitivity | Recipe TOML pins kernel version + patch offsets |
| BAR0 snapshot too large for replay | `to_replay_sequence()` filters to non-zero changed registers only |
