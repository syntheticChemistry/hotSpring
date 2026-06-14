# Experiment 131: Reset Architecture Evolution

**Date:** 2026-04-01
**Hardware:** Tesla K80 (GK210), Titan V (GV100), RTX 5060 (GB206)
**Status:** VALIDATED — deployed and hardware-tested

## Objective

Evolve the reset, diagnostic, and privilege architecture so that:
1. Ember is the single authority for all device resets (FLR, PMC, SBR)
2. Non-FLR GPUs (K80, Titan V) get proper software reset routing
3. PRI fault values never produce false "FECS running" diagnostics
4. No sudo/pkexec required in production code paths

## Changes

### 1. PRI Fault False-Positive Elimination

Before: PRI fault values like `0xbadf1201` were parsed for HALTED/STOPPED bits.
Since bits 4-5 happen to be clear in these patterns, they were misreported as "running".

Fixed in 7 locations:
- `ember.fecs.state` — new `pri_fault` field, bits gated by PRI check
- glowplug `warm_handoff` polling — rejects PRI faults before concluding "running"
- `gr_context::fecs_is_alive()` — early return false on PRI/DEAD_DEAD
- `GrEngineStatus::fecs_halted()` — new `fecs_inaccessible()` PRI guard
- `FalconState::is_halted()` — new `is_inaccessible()` method
- `observer/vfio.rs` — trace analysis PRI-aware
- `enrich_fecs_via_ember()` — logs `pri_fault` field

### 2. Ember Reset Authority

Ember now handles all reset methods as the VFIO FD holder:
- `flr` — VFIO_DEVICE_RESET via held fd, with PCIe DevCap bit 28 capability check
- `pmc` — PMC_ENABLE toggle via sysfs BAR0 (works on ALL GPUs)
- `auto` — FLR if capable, else PMC soft-reset (no-FLR fallback)
- `sbr`, `bridge-sbr`, `remove-rescan` — existing sysfs paths

Glowplug's `device.reset` now routes all methods through ember RPC.
Default method changed from `"flr"` to `"auto"`.

### 3. FLR Capability Detection

New `device_has_flr()` function reads PCI config space via sysfs to check
PCIe Express Capability DevCap bit 28 (FLReset). Confirmed:
- Titan V (GV100): `FLReset-` (no FLR)
- K80 (GK210): `FLReset-` (no FLR)
- RTX 3090 (GA102): FLR supported (from spec)

### 4. sudo/pkexec Removal

- `coralctl warm_fecs.rs`: livepatch control routes through `ember.livepatch.enable/disable`
- `deploy-dev.sh`: GPU binding routes through `ember.swap` RPC
- All production paths: ember/glowplug run as root via systemd, no privilege escalation

## Hardware Validation

```
K80 die 0:  ember.device_reset method=auto → PMC soft-reset 40ms ✓
K80 die 0:  ember.device_reset method=pmc  → PMC soft-reset 40ms ✓
Titan V:    ember.device_reset method=auto → PMC soft-reset 40ms ✓
Titan V:    ember.device_reset method=flr  → "does not support PCIe FLR" ✓
Titan V:    ember.fecs.state → pri_fault=false, halted=true, cpuctl=0x10 ✓
K80 die 0:  ember.fecs.state → pri_fault=true, cpuctl=0xbadf1200 ✓
All:        glowplug device.reset routes through ember correctly ✓
```

## Test Results

- coral-driver: 362 passed, 15 ignored
- coral-glowplug: 273 passed
- coral-ember: 158 passed, 1 ignored (8 livepatch tests skipped — module loaded)
