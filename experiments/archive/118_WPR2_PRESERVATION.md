# Exp 118: WPR2 Preservation via No-Reset Swap + SBR Trigger

**Date:** 2026-03-26
**Hardware:** Titan V (GV100), BDF 0000:03:00.0
**Depends on:** Exp 117

## Motivation

Exp 117 showed WPR2 is valid during nouveau but dies during the swap. Two
approaches tested: (A) disable PCI reset before swap, (B) trigger SBR/
remove-rescan after swap to re-run FWSEC.

## Results

### Phase A: No-Reset Swap

**FAILED** — `reset_method` sysfs write requires root (permission denied).
Ember already disables reset_method during the swap, so this isn't the problem.
The falcon death is caused by **nouveau's unbind handler writing register resets**,
not by PCI-level FLR/SBR.

Confirmation: WPR2 valid during nouveau (0x2FFE00000..0x2FFE20000), SEC2 SCTL=0x7021.
After swap: all dead, WPR2 invalid. Same as Exp 117.

### Phase B: Direct BOOTSTRAP on surviving SEC2

SKIPPED — SEC2 did not survive the swap.

### Phase C: GlowPlug-mediated Resets

- `remove-rescan`: RPC timed out (device removal disrupted GlowPlug state)
- `sbr`: GlowPlug connection failed (cascading from remove-rescan disruption)

**CAUTION**: `remove-rescan` is destructive — can change BDF assignment and
disrupt the GlowPlug/Ember device management.

### Phase D: WPR2 Content Capture

Captured 128 KiB (32768 words) from WPR2 VRAM via PRAMIN while nouveau running.
Content is entirely **`0xBAD0ACxx`** poison values — hardware write-protection
in action. "WPR" = **Write Protection Region** — reads are physically blocked.

```
0x2ffe00000: bad0ac05 bad0ac09 bad0ac0a bad0ac0b
0x2ffe00010: bad0ac0c bad0ac0d bad0ac0e bad0ac0f
```

## Key Discoveries

### 1. The Reset is Software, Not Hardware

Ember already disables `reset_method` (PCI reset) for NVIDIA GPUs to protect
HBM2 training. The falcon death and WPR2 clearing are caused by **nouveau's
kernel driver unbind handler**, which writes to falcon CPUCTL registers to
set HRESET.

This means disabling PCI resets CANNOT preserve the GPU state.

### 2. WPR2 is Hardware-Protected (Read AND Write)

PRAMIN reads of the WPR2 region return `0xBAD0AC` poison values. The hardware
physically blocks BOTH reads and writes to the write-protected region. We
cannot capture, inspect, or modify WPR2 content from the host.

### 3. FWSEC Must Be Re-Triggered

Since we can't:
- Prevent nouveau from killing the GPU state (driver-level reset)
- Read/write WPR2 content (hardware protection)
- Manually set WPR2 boundaries (write-protected registers)

We MUST re-trigger FWSEC after the swap. FWSEC runs from VBIOS ROM and is the
only entity that can set WPR2 boundaries.

## Root Cause Summary (Exp 114-118)

The complete causal chain for the WPR copy stall:

```
nouveau unbind → falcon HRESET → WPR2 boundaries cleared
→ vfio-pci bind (no FWSEC trigger) → WPR2 = INVALID
→ ACR boot → firmware checks WPR2 → INVALID → skip WPR processing
→ BOOTSTRAP_FALCON → no firmware loaded → FECS/GPCCS stay in HRESET
```

## Next Steps (Priority Order)

1. **DEVINIT/FWSEC research**: Understand how FWSEC is triggered on GV100.
   Can we trigger it via specific register writes after the swap?

2. **Direct vfio-pci boot**: Bind vfio-pci WITHOUT ever loading nouveau.
   FWSEC ran during system boot — WPR2 might still be valid if we never
   loaded nouveau to clear it.

3. **Nouveau module modification**: Build a custom nouveau that doesn't
   reset falcons on unbind. This preserves WPR2 and falcon state.

4. **Parasitic mode**: Use BAR0 sysfs while nouveau is bound. Build
   compute pipeline on top of nouveau's active GPU state.
