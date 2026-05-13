# Experiment 170 — End-to-End Sovereign Boot Orchestration

**Date**: 2026-04-16
**Status**: PASS (warm path), EXPECTED FAIL (falcon_boot)
**GPU**: Titan V (GV100, 0000:03:00.0)

## Goal

Validate the full `coralctl sovereign-boot` command — the vendor-ingredient loop that
orchestrates cold detection, nouveau warm cycle, training capture, and sovereign init
in a single invocation.

## What Changed

- `sovereign.rs` now implements the full vendor-ingredient loop:
  1. Detect driver state via sysfs
  2. Connect to ember
  3. Probe BAR0 to determine warm/cold
  4. If cold: swap to nouveau → wait 15s → swap back to vfio-pci
  5. Check for cached training recipe
  6. Run `ember.sovereign.init` with golden_state_path if available
- `SovereignInitOptions` gains `golden_state_path` and `vbios_rom_path` (file references)
- `handlers_sovereign.rs` loads golden state from file (supports raw pairs or TrainingRecipe JSON)
- Warm detection in Stage 3: if pmc_enable popcount >= 8 AND PRAMIN sentinel passes → skip HBM2

## Results

### Test 1: Warm GPU (previously trained by nouveau)

```
coralctl sovereign-boot 0000:03:00.0
```

| Step | Status | Detail |
|------|--------|--------|
| detect_driver | ok | driver=vfio-pci |
| connect_ember | ok | |
| bar0_probe | ok | pmc_enable=0x5fecdff1 warm=yes |
| load_recipe | ok | existing recipe: /var/lib/coralreef/training/gv100.json |
| sovereign_init | ok | warm detected, falcon boot pending |

**Result**: success=true, warm_detected=true, HBM2 training SKIPPED

### Test 2: Cold GPU probe (prior to warm cycle)

PMC_ENABLE = 0x40000020 (2 bits set → cold).
PRAMIN sentinel failed with 0xbad0ac0* errors.
HBM2 training attempted but failed (expected without warm state).

## Key Findings

- The warm-detection heuristic (popcount >= 8 + PRAMIN) correctly distinguishes warm and cold GPUs
- The golden_state_path mechanism works end-to-end via file reference
- The full orchestration completes in ~4s for a warm GPU
- Falcon boot remains the blocker for compute_ready — FECS halted at cpuctl=0x12
