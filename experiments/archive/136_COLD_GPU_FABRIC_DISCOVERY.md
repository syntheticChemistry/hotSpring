# Experiment 136: Cold GPU Fabric Discovery

**Date**: 2026-04-04
**GPU**: Titan V (GV100) at 0000:03:00.0
**Status**: CRITICAL DISCOVERY — unlocks safe warmup path

## Discovery

BIOS POST only fully initializes the **primary display GPU** (RTX 5070).
Secondary GPUs like the Titan V receive minimal initialization.

### Cold GPU Register Map

| BAR0 Range | Component | Status | Route |
|------------|-----------|--------|-------|
| 0x000-0x002 | PMC/PBUS | **Alive** | Direct BAR0 |
| 0x087000 | SEC2 Falcon | **Alive** | Direct (low BAR0, not PRI) |
| 0x120000 | PRI Ring Master | **Dead** (0xbad00100) | PRI-routed (self-referential) |
| 0x100000 | FBHUB/PFB | **Dead** | PRI-routed |
| 0x409000 | FECS | **Dead** (0xbadf1201) | PRI-routed |
| 0x41A000 | GPCCS | **Dead** | PRI-routed |
| 0x700000 | PRAMIN/VRAM | **Dead** (0xbad0fbXX) | Needs memory controller |

**PMC_ENABLE** at cold boot: `0x40000121` (bits 30, 8, 5, 0)

### Key Register Values (Cold)

- `BOOT0 = 0x140000a1` (valid GV100 chip ID)
- `SEC2 CPUCTL = 0x10` (HALTED — ROM halt from minimal BIOS init)
- `SEC2 SCTL = 0x3000` (LS mode, fuse-enforced)
- `SEC2 PC = 0x0083` (ROM halt address)
- `BAR0_WINDOW = 0x00000000` (PRAMIN window present but VRAM dead)

### Fatal Experiment: Enabling Memory Engines

Writing memory subsystem PMC bits (PFB/FBHUB/LTC, bits 16-28) on untrained DRAM
**permanently poisons the PRI ring**:

1. Engines wake up and try to access uninitialized HBM2
2. Memory access faults propagate as PRI ring errors
3. ALL PRI-routed registers (including SEC2) become inaccessible
4. **No recovery without FLR or reboot**

### Why ITFEN bits [5:4] Don't Stick

The falcon v1 bind interface (`nvkm_falcon_v1_bind_context`) sets ITFEN
bits [5:4] for DMA TLB control. On a cold GPU, these bits refuse writes
because the underlying FBHUB/MMU infrastructure (PRI-routed) is dead.
The v1 bind REQUIRES a working PRI ring and memory subsystem.

## Root Cause: Nouveau Crash

nouveau's probe function executes VBIOS DEVINIT scripts, which include:
- HBM2 DRAM training (timing calibration)
- PRI ring enumeration and initialization
- Memory controller configuration

On GV100 secondary GPUs, these scripts may hang because the GPU's
clock/power domains aren't configured for cold-start DEVINIT execution.
When nouveau hangs, the kernel thread enters TASK_UNINTERRUPTIBLE (D-state),
making the system unresponsive.

## Solution: nvidia Driver Warmup

The nvidia proprietary driver (already loaded for RTX 5070) handles
secondary GPU initialization more robustly than nouveau:
- Proper HBM2 training with timeouts
- Robust VBIOS DEVINIT execution
- Better error recovery

### Warmup Sequence

```bash
# 1. Swap Titan V from vfio-pci to nvidia (ember handles fd release)
coralctl swap 0000:03:00.0 nvidia

# 2. Wait for GPU init (check for /dev/dri/cardN)
sleep 10

# 3. Verify: PMC_ENABLE should be ~0x5fecdff1 (warm)
coralctl mmio read 0000:03:00.0 0x200

# 4. Swap back to vfio-pci (ember reacquires fd)
coralctl swap 0000:03:00.0 vfio
```

## Implications for Sovereign Boot

Cold GPU boot without a vendor driver requires:
1. PRI ring initialization (VBIOS DEVINIT or custom)
2. HBM2 DRAM training (complex, vendor-specific)
3. Memory controller configuration
4. Only THEN: ACR boot, FECS/GPCCS load

This is the sovereign DEVINIT path — our long-term goal.

## Files Modified

- `exp146_cold_fabric_init.rs` — cold GPU diagnostic + recovery test
- `exp145_v1_acr_boot.rs` — ACR boot with corrected falcon v1 interface
