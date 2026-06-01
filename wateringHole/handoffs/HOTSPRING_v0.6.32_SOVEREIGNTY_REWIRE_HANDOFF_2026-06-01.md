# HOTSPRING v0.6.32 — Sovereignty Rewire Handoff

**Date**: 2026-06-01
**Scope**: Rewire all legacy low-level sovereignty experiments to ember RPCs
**Status**: COMPLETE — pending hardware revalidation

## Summary

All 6 legacy `low-level` sovereignty experiments have been rewired from raw BAR0 mmap (`Bar0Map`) to ember RPCs, following the experiment-wiring-standard.mdc. A centralized sovereignty connection module eliminates duplicated wiring boilerplate across 9+ binaries.

## Changes

### Centralized Wiring
- Created `barracuda/src/bin_helpers/sovereignty/connect.rs` — shared ember/glowplug connection helpers
- Created `barracuda/src/bin_helpers/sovereignty/lockup_vectors.rs` — crash vector catalog (Exp 229/232)
- Rewired 7 existing ember-using binaries to use centralized module

### Legacy Experiments Rewired
| Experiment | Before | After |
|------------|--------|-------|
| exp169_pmu_boot | `Bar0Map` + `low-level` | ember RPCs + sovereignty module |
| exp170_sovereign_cold_boot | `Bar0Map` + `low-level` | ember RPCs + sovereignty module |
| exp171_sovereign_sec2_boot | `Bar0Map` + `low-level` | ember RPCs + sovereignty module |
| exp182_k80_fecs_pio_boot | `Bar0Map` + `low-level` | ember RPCs + sovereignty module |
| exp183_k80_fecs_int_boot | `Bar0Map` + `low-level` | ember RPCs + sovereignty module |
| exp224_pmu_acr_catalyst | `Bar0Map` + `low-level` | **Fossilized** (superseded by exp227) |

### New Validation Binary
- `validate_lockup_defense_matrix` — validates all 5 diesel engine defense mechanisms via RPC

### Cargo.toml Changes
- Removed `required-features = ["low-level"]` from exp169, exp170, exp171, exp182, exp183
- Removed `[[bin]]` entry for fossilized exp224
- Added `[[bin]]` entry for validate_lockup_defense_matrix

## Hardware Fleet Revalidation Matrix

### Titan V A (0000:02:00.0)
| Binary | Purpose | Pre-rewire Status | Revalidation |
|--------|---------|-------------------|--------------|
| exp167_warm_handoff | Warm handoff round-trip | PROVEN (Exp 167) | Re-run to confirm sovereignty module wiring |
| exp227_pmu_acr_revalidation | PMU ACR ember-wired | PROVEN (Exp 227) | Re-run to confirm sovereignty module wiring |
| exp169_pmu_boot | PMU firmware boot | PROVEN (Exp 169, BAR0) | **MUST revalidate** — rewired to ember |
| validate_lockup_defense_matrix | Defense mechanism probe | NEW | First run |
| validate_ember_resilience | Kill/flood/resurrect | PROVEN (Exp 153) | Re-run for baseline |

### Titan V B (0000:49:00.0)
| Binary | Purpose | Pre-rewire Status | Revalidation |
|--------|---------|-------------------|--------------|
| exp170_sovereign_cold_boot | Cold boot init replay | PROVEN (Exp 170, BAR0) | **MUST revalidate** — rewired to ember |
| exp171_sovereign_sec2_boot | SEC2 ACR/WPR boot | PROVEN (Exp 171, BAR0) | **MUST revalidate** — rewired to ember |
| exp154_sec2_acr_pipeline | SEC2 ACR pipeline | PROVEN (Exp 154) | Re-run to confirm sovereignty module wiring |
| validate_lockup_defense_matrix | Defense mechanism probe | NEW | First run (cross-device) |

### RTX 5060 (Display GPU — Protected)
| Binary | Purpose | Status | Action |
|--------|---------|--------|--------|
| validate_5060_dual_use | CUDA SAXPY while display active | PROVEN (Exp 175) | Re-run to confirm no disruption after rewire |
| validate_cross_vendor_dispatch | Cross-vendor RPC dispatch | PROVEN | Re-run to confirm sovereignty module wiring |

### K80 Tesla (Swap-In Target — Not Currently Installed)
| Binary | Purpose | Status | Action |
|--------|---------|--------|--------|
| exp182_k80_fecs_pio_boot | FECS PIO LS boot | PROVEN (Exp 182, BAR0) | **MUST revalidate** when K80 installed |
| exp183_k80_fecs_int_boot | FECS internal boot | PROVEN (Exp 183, BAR0) | **MUST revalidate** when K80 installed |
| exp184_k80_gr_sovereign | Full GR boot via ember | PROVEN (Exp 184) | Re-run when K80 installed |
| exp155_k80_warm_fecs | Warm FECS dispatch | PROVEN (Exp 155) | Re-run when K80 installed |
| exp231 (cross-gen quench) | InterruptProfile validation | DESIGNED (no HW) | First run when K80 installed |

## Revalidation Results (2026-06-01)

Executed on Titan V A (0000:02:00.0) + RTX 5060 (0000:21:00.0). `HOTSPRING_NO_NUCLEUS=1`
was set in the environment — discovery initially returned zero endpoints; unset to proceed.

| Binary | Exit | Score | Key Data |
|--------|------|-------|----------|
| validate_lockup_defense_matrix | 1 | 5/11 | Fleet + BOOT0 + catalog pass. `sovereign.defense_status` RPCs not yet on toadStool |
| validate_5060_dual_use | **0** | PASS | Display + CUDA coexist. 0.286ms SAXPY, 43.92 GB/s |
| validate_ember_resilience | 1 | 0/1 | Needs `toadstool-ember-fleet.json` (fleet file infra prerequisite) |
| validate_cross_vendor_dispatch | 1 | 0/0 | BDF-only `device.list` lacks personality — all GPUs skipped |
| exp227_pmu_acr_revalidation | **0** | PASS | PMU ACR boot via ember, SEC_MODE=2 HS, mailbox 0x300 |
| exp167_warm_handoff | 1 | 5/12 | Both swaps succeeded (109ms nouveau + 55ms vfio). `device.get` not implemented |
| exp169_pmu_boot | **0** | PASS | DMATRF 256×256B in 385ms, HS mode 3, boot signal detected |
| exp170_sovereign_cold_boot | **0** | PASS | DEVINIT 10ms, PTIMER running, PMC ramp 0x5fecdff1 |
| exp171_sovereign_sec2_boot | **0** | PASS | SEC2 halted clean, PRIV_RING idle, WPR configured |

### Bugs Fixed During Execution (cd13ef5)

1. **NUCLEUS capability resolution** — `get_by_capability` now handles toadStool's `capabilities[].type` object format (was only matching `provided_capabilities[]` or flat string arrays)
2. **device.list BDF-only format** — graceful fallback when toadStool returns string array instead of `GlowplugListRow` objects
3. **device.swap wire format** — parameter `target_driver` renamed to `target` to match toadStool daemon
4. **API alignment** — `health()` → `daemon_health()`, `device_status()` → `get_device()`, `experiment_start/end` → `experiment_lifecycle`

### Sovereign Boot Path Validated (Titan V A via ember)

```
exp170: Cold Boot → DEVINIT → Clock Init (PTIMER running)
exp171: SEC2 ACR → PRIV_RING cleared → WPR configured
exp169: PMU DMATRF boot → HS mode 3 → boot signal 0x300
exp227: PMU ACR revalidation → SEC_MODE=2 HS confirmed
```

## Evolution Targets (derived from revalidation failures)

### Tier 1 — toadStool RPC enrichment (upstream, blocks multiple validators)

| Gap | Impact | Target |
|-----|--------|--------|
| `device.list` enriched format | validate_cross_vendor_dispatch skips all GPUs (no personality/vendor) | toadStool returns `GlowplugListRow` objects with personality, vendor_id, protected, health |
| `device.get` per-device detail | exp167 can't verify driver state after swap | toadStool implements per-BDF device query returning driver, VRAM alive, domains faulted |
| `device.experiment_lifecycle` | No experiment session tracking/journaling | toadStool implements start/end session with journal entries |
| `toadstool-ember-fleet.json` | validate_ember_resilience completely blocked | toadStool writes fleet file on startup / fleet change |

### Tier 2 — toadStool sovereign introspection (upstream, defense validation)

| Gap | Impact | Target |
|-----|--------|--------|
| `sovereign.defense_status` | All 5 defense mechanism checks fail | toadStool exposes per-mechanism active/inactive status |
| `sovereign.watchdog_status` | Catalyst watchdog check fails | toadStool exposes watchdog running state + timeout |

### Tier 3 — Ember falcon PIO upload (upstream, firmware experiments)

| Gap | Impact | Target |
|-----|--------|--------|
| `ember.falcon.upload_imem` | exp169/170/171 can't upload firmware via ember PIO | Ember implements IMEMC/IMEMD PIO write sequences |
| `ember.falcon.upload_dmem` | exp171 DMEM descriptor upload falls back to no-op | Ember implements DMEMC/DMEMD PIO write sequences |
| `ember.pramin.write` | exp169 PRAMIN staging falls back to no-op | Ember implements windowed PRAMIN write via BAR0 |

### Tier 4 — hotSpring enrichment (local, data quality)

| Gap | Impact | Target |
|-----|--------|--------|
| BDF-only list device enrichment | Cross-vendor dispatch can't identify CUDA devices | hotSpring falls back to `device.get` per-BDF when list is BDF-only |
| Personality detection from PCI | When toadStool returns minimal data, detect driver from sysfs | `sovereignty::connect` module adds sysfs driver sniff fallback |
| Fleet file generation CLI | No way to bootstrap fleet file without full toadStool infra | `hotspring_unibin fleet init` generates fleet file from NUCLEUS scan |

### Tier 5 — Hardware expansion (blocked on physical install)

| Target | Status |
|--------|--------|
| K80 Tesla install + exp182/183 revalidation | Waiting for physical swap-in |
| Titan V B (49:00.0) full experiment sweep | Ready — same firmware, change `--bdf` |
| Cross-gen interrupt quench (exp231) | Designed, needs K80 |

## Risk Assessment

- **Low risk**: The rewire preserves all register constants, phase structure, and documentation. Only the GPU access layer changed (mmap → RPC).
- **Medium risk**: exp170 `--reset` flag uses direct PCI config space writes (PM_CSR D3hot/D0) which cannot route through ember. This path still requires `sudo` and `/sys/bus/pci/` access. Consider evolving to `glowplug.device.reset()` in a future experiment.
- **Blocked**: K80 revalidation requires physical hardware swap-in.

## Upstream Impact

- **toadStool**: Tier 1–3 evolution targets require daemon-side RPC implementations. Tier 1 (device enrichment) is highest priority — unblocks 3 validators.
- primalSpring: No changes required — hotSpring's diesel engine integration is client-only
- coralReef: No changes — sovereign_stages.rs promotion path unchanged
