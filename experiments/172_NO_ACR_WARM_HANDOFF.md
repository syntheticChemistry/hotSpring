<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Experiment 172 — No-ACR Nouveau Warm Handoff

**Date**: 2026-04-16
**GPU**: Titan V (GV100, 0000:03:00.0)
**Goal**: Achieve warm HBM2 + clean HRESET falcons (no HS lockout)

## Background

Previous experiments (169–171) showed that after nouveau boots and swaps
back to vfio-pci, FECS/GPCCS remain in Heavy Secure (HS) mode:

- SCTL = 0x3000 (sec_mode=3, hardware-latched)
- HRESET cleared, falcon CPU running HS firmware
- PIO IMEM/DMEM blocked, ENGCTL/PMC resets don't clear HS
- FECS method interface unresponsive (stalls on GR_ENABLE)

This prevents the ACR boot solver from re-booting FECS, creating a
deadlock: HBM2 needs nouveau to train, but nouveau's ACR boot puts
falcons into unrecoverable HS mode.

## Approach: Remove ACR Firmware Before Nouveau

Move `/lib/firmware/nvidia/gv100/acr/` aside before loading nouveau.
Nouveau's init sequence:

1. PCI config, BAR mapping
2. VBIOS DEVINIT → PMU runs memory controller training (HBM2)
3. SEC2/ACR boot → **SKIPPED** (firmware files missing)
4. GR engine → unavailable (no ACR = no FECS)
5. DRM init → succeeds without acceleration

SEC2 firmware (`/lib/firmware/nvidia/gv100/sec2/`) must remain — nouveau
fails probe entirely without it (`sec2 ctor failed: -2`).

## Execution

```bash
# Step 1: Move ACR firmware aside
sudo mkdir -p /var/lib/coralreef/fw-stash
sudo mv /lib/firmware/nvidia/gv100/acr /var/lib/coralreef/fw-stash/acr

# Step 2: Swap to nouveau via ember
ember.swap → nouveau   # 6.5s bind, HBM2 trains

# Step 3: Swap back to vfio-pci
ember.swap → vfio-pci  # 14.2s bind

# Step 4: Restore ACR firmware
sudo mv /var/lib/coralreef/fw-stash/acr /lib/firmware/nvidia/gv100/acr
```

## dmesg Output (no-ACR load)

```
nouveau 0000:03:00.0: NVIDIA GV100 (140000a1)
nouveau 0000:03:00.0: bios: version 88.00.41.00.18
nouveau 0000:03:00.0: acr: firmware unavailable     ← ACR skipped!
nouveau 0000:03:00.0: pmu: firmware unavailable
nouveau 0000:03:00.0: fb: 12288 MiB of unknown memory type
nouveau 0000:03:00.0: drm: VRAM: 12288 MiB          ← HBM2 trained!
nouveau 0000:03:00.0: drm: GART: 536870912 MiB
[drm] Initialized nouveau 1.4.0                     ← DRM works
```

## Results After Swap-Back

| Register | Value | Meaning |
|----------|-------|---------|
| PMC_ENABLE | 0x5fecdff1 | 23 engines active (WARM) |
| PTIMER | counting | GPU alive |
| FECS CPUCTL | 0x10 | HRESET (CPU stopped, awaiting firmware) |
| FECS SCTL | 0x3000 | HS fuse capability (NOT operational HS) |
| FECS MAILBOX0 | 0x00 | No handshake |
| FECS HWCFG | 0x20204080 | IMEM=32KB, DMEM=8KB |
| SEC2 CPUCTL | 0x10 | HRESET (CPU stopped) |
| SEC2 SCTL | 0x3000 | HS fuse capability |
| PRAMIN sentinel | OK | VRAM accessible |

**Key difference from normal warm swap:**
- Normal: CPUCTL=0x10, falcon in HS mode with firmware loaded, PIO blocked
- No-ACR: CPUCTL=0x10, falcon in HRESET, PIO works, IMEM writable

## ACR Boot Solver Test

Ran sovereign_init with ACR firmware restored and DMA backend available.
FECS IMEM upload verified: `FECS IMEM verify: match=true` (PIO works!).

10 ACR strategies attempted, all failed at DMA configuration:
- VRAM strategies: FBIF_TRANSCFG bit 0 (PHYS_VID) drops from 0x91→0x90
- System memory strategies: instance block bind_stat TIMEOUT
- SEC2 firmware runs but stalls at PC=0x129 (DMA fault EXCI=0x041f0000)
- Direct FECS boot blocked by HS fuse (STARTCPU doesn't release HRESET)

## Conclusions

1. **No-ACR warm is the correct base state** for sovereign boot
2. HBM2 training is independent of ACR/FECS — VBIOS DEVINIT handles it
3. SEC2 firmware is required for nouveau probe (not just ACR)
4. FECS PIO IMEM upload verified working — firmware CAN be loaded
5. The remaining blocker is SEC2 DMA configuration for the ACR chain
6. FECS STARTCPU is blocked by HS fuse — only ACR can release HRESET

## Next Steps

- Debug SEC2 DMA fault: capture nouveau's exact FBIF/DMAIDX/IOMMU setup
- Try matching nouveau's DMA configuration (virtual + instance block)
- Consider tracing nouveau's SEC2 boot via ftrace/perf for register writes
