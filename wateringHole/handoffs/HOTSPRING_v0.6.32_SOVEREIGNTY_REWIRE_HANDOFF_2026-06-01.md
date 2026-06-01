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

## Revalidation Procedure

1. Ensure toadStool daemon is running with both Titan V embers active
2. Run `validate_lockup_defense_matrix` first (safety baseline)
3. Run rewired experiments on Titan V A, then Titan V B (one at a time)
4. Run `validate_5060_dual_use` during Titan V experiments to confirm no display disruption
5. Run `validate_ember_resilience` after all experiments complete (fleet health)

## Risk Assessment

- **Low risk**: The rewire preserves all register constants, phase structure, and documentation. Only the GPU access layer changed (mmap → RPC).
- **Medium risk**: exp170 `--reset` flag uses direct PCI config space writes (PM_CSR D3hot/D0) which cannot route through ember. This path still requires `sudo` and `/sys/bus/pci/` access. Consider evolving to `glowplug.device.reset()` in a future experiment.
- **Blocked**: K80 revalidation requires physical hardware swap-in.

## Upstream Impact

- primalSpring: No changes required — hotSpring's diesel engine integration is client-only
- toadStool: No changes required — all defenses are server-side
- coralReef: No changes — sovereign_stages.rs promotion path unchanged
