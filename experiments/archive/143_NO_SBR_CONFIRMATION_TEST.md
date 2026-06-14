# Experiment 143: No-SBR Confirmation Test

**Date**: 2026-04-03
**GPU**: Titan V (GV100, 0000:03:00.0)
**Parent**: Exp 141, Exp 142
**Status**: RAN — hypothesis CONTRADICTED (ACR fails even on POSTed GPU)

## Hypothesis

If VBIOS DEVINIT is the root cause of ACR failure, then skipping the SBR reset
entirely and using the GPU in its BIOS-POSTed state (after system BIOS but before
any driver loads) should allow ACR to succeed — because VBIOS DEVINIT was already
run by the GPU's boot ROM during system POST.

## Protocol

1. Unbind Titan V from any driver (vfio-pci should already hold it via ember)
2. Do NOT issue SBR (no `echo 1 > /sys/bus/pci/devices/.../reset`)
3. Run `coralctl sovereign-boot --bdf 0000:03:00.0`
4. Observe: `DevinitStatus::probe` should report `needs_post=false`
5. Phase 0 should be skipped ("already POSTed")
6. Recipe replay + ACR boot should proceed

## Expected Outcomes

### Success (confirms hypothesis)
- `needs_post=false` → Phase 0 skipped
- Recipe replay succeeds
- ACR Strategy 7c passes HS authentication
- This definitively proves VBIOS DEVINIT is the missing piece after SBR

### Failure (contradicts hypothesis)
- If ACR still fails on a BIOS-POSTed GPU, the root cause is NOT (only) VBIOS DEVINIT
- Would indicate additional initialization done by the NVIDIA driver that we're missing

## Results (April 3, 2026)

### Outcome: **Failure — contradicts hypothesis**

**Run**: Fresh cold boot → ember acquires Titan V → `coralctl sovereign-boot 0000:03:00.0` (no SBR, no reset)

**Observations**:
- `BOOT0 = 0x140000a1` → valid GV100
- `DEVINIT_STATUS (0x0002240C) = 0x00000002` → bit 1 set → `needs_post=false` ✓
- Phase 0 VBIOS DEVINIT: **skipped** (correctly: GPU is already POSTed) ✓
- Recipe replay: 337 steps, all applied, GPU alive ✓
- PTIMER ticking, VRAM accessible via PRAMIN ✓
- **Phase 2 ACR: ALL 15 strategies FAILED** ✗

**SEC2 Failure Pattern** (identical across all strategies):
```
SEC2 PMC bit not found in PTOP, using fallback bit 22
SEC2 did not exit HRESET after PMC reset (3s)
falcon did NOT halt after scrub (500ms timeout)
falcon_start_cpu: POST-START FAULT base=0x87000 pc=0x00a6 exci=0x091f0000
```

**Conclusion**: The root cause of ACR failure is NOT VBIOS DEVINIT. The GPU is
properly POSTed, all hardware appears initialized, but the SEC2 falcon cannot
be started by our HAL. The `POST-START FAULT` with `exci=0x091f0000` occurs
even on a fully BIOS-POSTed, never-SBR'd GPU.

**New hypothesis**: The SEC2 HAL startup sequence has a bug — likely in the
PMC enable bit lookup (PTOP doesn't list SEC2, fallback bit 22 may be wrong),
the falcon reset procedure, or the firmware upload format. See Exp 142 for
detailed analysis.
