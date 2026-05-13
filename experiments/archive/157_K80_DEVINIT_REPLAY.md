# Experiment 157: K80 DEVINIT Replay

## Date
2026-04-07

## Hypothesis
Replaying the VBIOS DEVINIT sequence through ember MMIO on a cold Tesla K80 (GK210) 
should train GDDR5 and bring the GPU to a compute-ready state without needing a VM POST.

## Background
- Cold K80 bound to vfio-pci has no trained VRAM (PRAMIN returns 0xFFFFFFFF)
- DEVINIT recipe captured from nvidia-470 VM capture (exp124b): 305 ops, 2 scripts
- Op types: ZmReg (direct write), NvReg (read-modify-write), Time (delay), ZmMaskAdd (read-add-write)
- Recipe includes PLL reconfiguration (SPLL at 0x8C040), PFB init (135 ops), thermal calibration

## Key Findings

### DEVINIT Recipe Structure
- Script 0: 210 ops starting from `PMC_ENABLE = 0x00002020` (minimal engines)
- Script 1: 95 ops (secondary init)
- 16 timing delays (37μs to 10ms)
- 8 ZmMaskAdd ops for thermal calibration

### Execution Result
- **Ops 0-54 succeeded** (59 total including writes + delays)
- **Op 55 triggered circuit breaker**: GPU went non-responsive (BOOT0 = 0xFFFFFFFF)
- Failure point: ZmMaskAdd read at 0x2070C (thermal sensor calibration)
- Root cause: PLL reprogramming at ops 15-18 (SPLL 0x8C040 mode change) caused clock domain glitch

### GPU State After Failure
- K80 became permanently non-responsive
- PCI rescan failed to recover device
- Both K80 dies (4c:00.0 and 4d:00.0) lost from PCI bus
- **Requires physical power cycle to recover**

## Lessons Learned

1. **PLL reprogramming on cold VFIO devices is destructive** — the DEVINIT relies on BROM having 
   already established a stable base clock. On a cold VFIO device, changing the PLL loses the only 
   working clock source.

2. **DEVINIT must be replayed in a VM** (with proper BIOS POST infrastructure), not via raw MMIO 
   from the host. The agentReagents VM approach is the correct path.

3. **Circuit breaker recovery**: Added breaker reset + retry logic for transient faults, but PLL 
   bricking is not transient — the device needs hardware reset.

4. **ZmMaskAdd field names**: Recipe uses `inv_mask`/`add_val` (inverted mask), not `mask`/`add`.
   Operation: `(read & ~inv_mask) + add_val`.

## Files
- Binary: `barracuda/src/bin/exp157_k80_devinit_replay.rs`
- Recipe: `data/k80/gk210_devinit_recipe.json`
- Log: `/tmp/exp157_k80_devinit.log`

## Next Steps
- K80 requires power cycle before further experiments
- Use agentReagents VM with nvidia-470 for proper POST + VFIO reclaim
- Consider filtering PLL ops from DEVINIT for safer partial replay
