# hotSpring × coralReef — Sovereign Compute Barrier Resolution

**Date:** May 10-11, 2026
**Spring:** hotSpring v0.6.32+
**Primals:** coralReef (coral-driver, coral-ember, coral-glowplug), barraCuda
**Hardware:** RTX 5060 (Blackwell SM120), Titan V (Volta GV100), Tesla K80 (Kepler GK210)

---

## Summary

Systematic resolution of sovereign compute barriers across three NVIDIA GPU
generations. The diesel engine architecture (hierarchical glowplug/ember) is
fully validated. The RTX 5060 is compute-proven. The Titan V reaches HBM2
warm state but requires benchScale VM isolation for FECS boot. The K80's
PCIe link is dead and requires a physical power cycle.

---

## Breakthroughs

### 1. Volta ACR Solver Skip (sovereign_stages.rs)

The `FalconBootSolver` was hanging >5 minutes trying ACR strategies on GV100
because WPR is not used on pre-GSP Volta. Added early-exit detection:

```
AcrSec2 + !wpr_configured + SM 70-74 → skip ACR solver → PIO FECS bootstrap
```

Sovereign init now completes in **4 seconds** on Titan V (was infinite hang).
Stages 1-3 all pass: bar0_probe OK, pmc_enable OK (0x5fecdff1), hbm2_training
SKIPPED (warm detected via PRAMIN sentinel).

### 2. HBM2 Warm-Handoff via `reset_method=none`

Discovered that `echo "none" > /sys/bus/pci/devices/$BDF/reset_method` prevents
VFIO from resetting the GPU during rebind. This preserves nouveau-trained HBM2
across the driver swap:

```
nouveau bind → HBM2 trained (12GB) → unbind → reset_method=none → vfio-pci bind → PRAMIN live
```

PMC_ENABLE stays at 0x5fecdff1 (all engines on). PRAMIN returns real data
(not 0xbad0acXX). This is the first successful warm state preservation on
GV100 in our stack.

### 3. FECS Barrier Characterization

FECS on GV100 is a Falcon v5 with HS (High Security) mode. Three paths
attempted, all blocked:

- **PIO direct upload:** Rejected by secure mode (cpuctl stays at 0x12 HRESET)
- **ACR chain:** No WPR on GV100 → SEC2 can't authenticate FECS firmware
- **nouveau:** Skips GR entirely on GV100 (PMU firmware not in linux-firmware)

Only nvidia-470 boots GR/FECS on GV100 (embeds PMU firmware). But nvidia-470
cannot coexist with nvidia-580 (serving RTX 5060 display).

### 4. Production Path: benchScale VM Isolation

Instead of swapping kernel drivers (which can crash the host DRM), the
production path uses benchScale + agentReagents to run nvidia-470 inside
a VM with the Titan V passed through via VFIO:

1. benchScale spins a VM with Titan V VFIO passthrough
2. VM loads nvidia-470 → GR/FECS/HBM2 fully initialized
3. VM shuts down → host reclaims device with warm state (`reset_method=none`)
4. coral-ember opens the warm device → sovereign init detects warm → compute ready

Host DRM (nvidia-580 + RTX 5060 display) stays completely uninterrupted.
Physical card swaps also viable — user has 1-2 cards from most NVIDIA generations.

### 5. K80 PCIe Link Recovery

K80 PLX PEX 8747 switch config space returns 0xFF (DLActive=false). Attempted:
- SBR from root port (AMD 40:01.3)
- PCIe link retrain via LnkCtl
- Bus remove + rescan from root port
- Global PCI rescan

All failed. The PLX switch lost power during D3cold and ACPI won't restore it
without a BIOS-level re-initialization. Requires physical power cycle with
`d3cold_allowed=0` set at boot.

---

## Files Changed

| File | Change |
|------|--------|
| `coralReef/crates/coral-driver/src/vfio/sovereign_stages.rs` | Volta CpuRm early-exit: skip ACR solver when !wpr_configured + SM 70-74 |
| `scripts/lab/titanv_nvidia470_warm_handoff.sh` | nvidia-470 warm-handoff script (display-free session, pre-benchScale) |
| `docs/PRIMAL_GAPS.md` | Updated GAP-HS-030, GAP-HS-047 with current state |

## RTX 5060 Validation (confirmed compute-ready)

- `validate_pure_gpu_qcd`: 3/3 PASS (solution parity 3.997e-16)
- `validate_pure_gpu_hmc`: 3/3 PASS (100% acceptance, 10/10 trajectories)
- `validate_cpu_gpu_parity`: 6/6 PASS (energy conservation, D* transport)

---

## Remaining Barriers

| GPU | Barrier | Path Forward |
|-----|---------|--------------|
| RTX 5060 | **None** — fully compute-ready | WGPU/Vulkan via nvidia-580 DRM |
| Titan V | FECS secure boot (no WPR, no PMU) | benchScale VM + nvidia-470 warm-handoff |
| Tesla K80 | PCIe link dead (D3cold, DLActive-) | Physical power cycle + `d3cold_allowed=0` |

## Upstream Debt

- `GAP-HS-030`: Duplicate ID collision with "Ember Absorption" entry (line ~302). Renumber one.
- `wateringHole/README.md`: Handoff table lists 11 entries but some files missing from disk.
- `EXPERIMENT_INDEX.md`: Missing entries for experiments 182-184.
