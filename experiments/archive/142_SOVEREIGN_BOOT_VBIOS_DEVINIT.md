# Experiment 142: Sovereign Boot with VBIOS DEVINIT

**Date**: 2026-04-03
**GPU**: Titan V (GV100, 0000:03:00.0)
**Parent**: Exp 141 (ACR HS Auth Root Cause)
**Status**: RAN — hypothesis partially invalidated (see Results)

## Hypothesis

The ACR HS authentication failure (Exp 141) is caused by missing VBIOS DEVINIT.
After SBR, the crypto engine used for HS signature verification is uninitialized.
Wiring `interpret_boot_scripts` into `sovereign_boot()` as Phase 0 should initialize
PLLs, clocks, crypto engine, and memory controllers — enabling ACR to pass.

## Code Change

`sovereign_boot()` in `crates/coral-driver/src/vfio/channel/diagnostic/sovereign_boot.rs`
now runs a Phase 0 before recipe replay:

```
Phase 0: VBIOS DEVINIT (DevinitStatus::probe → execute_devinit_with_diagnostics)
Phase 1+: Recipe Replay (clock, devinit-recipe, pgraph, extended)
Phase 2: ACR Boot (FalconBootSolver — Strategy 7c primary)
```

`execute_devinit_with_diagnostics` handles the full pipeline:
1. Probes `DevinitStatus` (register 0x0002240C, bit 1)
2. Reads VBIOS from PROM (BAR0+0x300000) or sysfs or file
3. If secure boot detected → host-side VBIOS interpreter (`interpret_boot_scripts`)
4. Otherwise → PMU FALCON upload + execute, with interpreter fallback
5. Checks VRAM liveness via PRAMIN sentinel

## Run Protocol

```bash
# Start ember (holds VFIO FDs for Titan V)
sudo systemctl start coral-ember.service

# Run sovereign boot with VBIOS DEVINIT enabled
coralctl sovereign-boot --bdf 0000:03:00.0

# Observe Phase 0 log output:
#   phase 0 (VBIOS DEVINIT): needs_post=true devinit_reg=0x...
#   VBIOS interpreter: writes=N, unknown_opcodes=[...], pri_faults=M
#   phase 0 (VBIOS DEVINIT): completed, vram_alive=true/false
```

## Expected Outcomes

### Success (ACR passes)
- VBIOS interpreter runs, reports writes_applied > 0
- Recipe replay proceeds normally
- ACR Strategy 7c: HS authentication completes (mailbox transitions beyond 0x2d78)
- FECS/GPCCS boot → L10 SOLVED

### Partial (DEVINIT runs, ACR still fails)
- VBIOS interpreter runs but reports unknown_opcodes for GV100-era scripts
- Interpreter audit needed (see Exp 143)
- Some writes may hit PRI faults in uninitialized domains

### Failure (DEVINIT itself fails)
- PROM inaccessible (0xAA55 signature mismatch)
- No VBIOS source available
- Fallback: use file-based VBIOS dump

## Key Registers to Log

| Register | Address | What to watch |
|----------|---------|---------------|
| DEVINIT_STATUS | 0x0002240C | bit 1: 0=needs_post, 1=done |
| PROM_ENABLE | 0x00001854 | bit 0 masks PROM reads |
| PROM_BASE | 0x00300000 | should read 0x...AA55 |
| PMC_ENABLE | 0x00000200 | engine enable bitmap |
| PTIMER | 0x00009400 | ticking = GPU clock alive |
| SEC2 SCTL | 0x00840E00 | security mode after DEVINIT |

## Results (April 3, 2026)

### Hardware Run 1: Bridge PM Reset + Sovereign Boot

**Protocol**: `echo 1 | sudo tee /sys/bus/pci/devices/0000:00:01.3/reset` → 3s delay → `coralctl sovereign-boot 0000:03:00.0`

**Key Findings**:
- `DEVINIT_STATUS (0x0002240C) = 0x00000002` → bit 1 SET → `needs_post=false`
- PM bridge reset did NOT cold-reset the GPU (DEVINIT still shows POSTed)
- Phase 0 VBIOS DEVINIT: **skipped** (already POSTed)
- Phase 2 ACR: **ALL 15 strategies FAILED**
- SEC2 falcon fault: `POST-START FAULT pc=0x0155 exci=0x091f0000`
- SEC2 PMC bit not found in PTOP (using fallback bit 22)
- SEC2 does not exit HRESET after PMC toggle
- FECS remains in HRESET throughout

### Cross-reference: Exp 143 (No-SBR confirmation)

Same system, no reset at all (fresh cold boot, BIOS-POSTed):
- `DEVINIT_STATUS = 0x00000002` → `needs_post=false`
- Phase 0 VBIOS DEVINIT: **skipped** (already POSTed)
- Phase 2 ACR: **ALL 15 strategies FAILED** — identical failure mode
- SEC2: same `POST-START FAULT` pattern

### Root Cause Revision

**VBIOS DEVINIT is NOT the sole root cause of ACR failure.** The GPU is properly
POSTed (DEVINIT_STATUS confirms it), but SEC2 falcon still faults on startup.
The common failure across both experiments is:

1. **SEC2 PMC bit not in PTOP** — We can't find SEC2's PMC enable bit via the
   PTOP topology registers, falling back to hardcoded bit 22. This may be wrong.
2. **SEC2 does not exit HRESET** — After toggling PMC_ENABLE bit 22, SEC2 stays
   in HRESET (cpuctl=0x00000010) for 3+ seconds.
3. **Falcon POST-START FAULT** — When we force-start the falcon, it faults
   immediately with exci=0x091f0000 (PC advances: 0x0155→0x0157).

The root cause is in the **SEC2 HAL startup sequence**, not VBIOS DEVINIT.
Investigation should focus on: correct SEC2 PMC enable bit, SEC2 falcon
reset/scrub procedure, firmware load verification, and comparison with
nouveau's `gv100_acr_*` SEC2 init path.
