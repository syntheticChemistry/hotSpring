# Experiment 205 — Dual Titan V Twin Study Baseline

**Date:** May 17, 2026
**Hardware:** 2× Titan V (GV100), VFIO-bound
**Objective:** Validate sovereign pipeline parity across identical GPUs,
establish twin study surface for parallel experimentation.

## Summary

Second Titan V installed in former K80 PCIe slot. Both cards validated
through the full sovereign.init pipeline with identical results. VBIOS ROM
fingerprints match byte-for-byte, confirming identical firmware. Register
state converges after sovereign.init. Twin study surface is live.

## Hardware Topology

| Card | BDF | IOMMU Group | Slot | Driver | Role |
|------|-----|-------------|------|--------|------|
| Titan V #1 | `0000:02:00.0` | 65 | Original | `vfio-pci` | oracle |
| Titan V #2 | `0000:49:00.0` | 32 | Former K80 | `vfio-pci` | compute |
| RTX 5060 | `0000:21:00.0` | — | — | `nvidia` | display |

Both Titan Vs negotiate PCIe Gen3 x8 (downgraded from x16 capability).
Each IOMMU group contains only the GPU function and its audio function
(`xx:00.0` + `xx:00.1`), confirming clean isolation for independent
VFIO access.

## Sovereign Pipeline Comparison

Both cards run warm (post-UEFI POST) after reboot.

| Stage | TV1 (02:00.0) | TV2 (49:00.0) | Match |
|-------|---------------|---------------|-------|
| bar0_probe | 12ms, boot0=0x140000a1 | 12ms, boot0=0x140000a1 | ✓ |
| pmc_enable | 74ms, 0x5fecdff1 | 74ms, 0x5fecdff1 | ✓ |
| cg_sweep | 0 changed, 12 faulted | 0 changed, 12 faulted | ✓ |
| pri_recovery | 9 alive, 4 faulted, recovered | 9 alive, 4 faulted, recovered | ✓ |
| pgob_ungating | 14 GPCs alive | 14 GPCs alive | ✓ |
| memory_training | skipped (warm) | skipped (warm) | ✓ |
| falcon_boot | failed (ACR no DMA) | failed (ACR no DMA) | ✓ |
| **total** | **3164ms** | **3163ms** | **✓** |

Note: Titan V #2's first sovereign.init after reboot showed 6 CG changes
(PTHERM master gate, CG1, CG2, LTC1, LTC3, LTC5 cleared) because it was
freshly VFIO-bound without prior BAR0 access. Subsequent runs converge to
the same 0-changed state as TV1.

## Register Twin Comparison (post sovereign.init)

| Register | TV1 | TV2 | Status |
|----------|-----|-----|--------|
| PMC_BOOT0 (0x000000) | 0x140000a1 | 0x140000a1 | MATCH |
| PMC_ENABLE (0x000200) | 0x5fecdff1 | 0x5fecdff1 | MATCH |
| PMC_ID (0x000a04) | 0xbad00200 | 0xbad00200 | MATCH |
| PTIMER (0x009400) | varies | varies | DIFFER (expected) |
| FUSE (0x021000) | 0xc0040000 | 0xc0040000 | MATCH |
| NV_PROM[0] (0x300000) | 0xeb72aa55 | 0xeb72aa55 | MATCH |
| PGRAPH_STATUS (0x400700) | 0x00000000 | 0x00000000 | MATCH |

Only PTIMER differs (monotonic timer, expected to diverge).

## VBIOS ROM Identity

```
TV1: PROM[0:64] sha256=af04a2c636da558f  header=55 aa 72 eb 4b 37 34 30 30 e9 4c 19 77 cc 56 49
TV2: PROM[0:64] sha256=af04a2c636da558f  header=55 aa 72 eb 4b 37 34 30 30 e9 4c 19 77 cc 56 49
```

Byte-identical VBIOS ROMs. Any opcode/stride fix validated on one card
applies directly to the other.

## Infrastructure Changes

- Updated `/etc/toadstool/glowplug.toml`: replaced K80 device entries
  with Titan V #2 at `0000:49:00.0` (name: `titan-v-2`, role: `compute`,
  health_policy: `active`)
- Restarted `toadstool-ember` daemon — all 3 devices enumerated

## Twin Study Value

Identical silicon + identical firmware = controlled variable elimination:

- **A/B register experiments**: write different init sequences to each card,
  compare outcomes without firmware variance
- **Warm vs cold divergence**: reboot one card (FLR or D3 cycle) while the
  other stays warm to study state drift
- **VBIOS interpreter validation**: any VBIOS script fix produces identical
  op counts on both cards (ground truth)
- **Falcon ACR debugging**: attempt different boot strategies on each card
  simultaneously
- **Timing studies**: compare register write latencies across PCIe domains
  (IOMMU group 65 vs 32)

## Current Blocker

Same as Experiment 204: falcon_boot fails on both cards because ACR
(Authenticated Code Runner) HS boot requires a DMA backend not yet
provided. GPCCS PIO boot also times out (`cpuctl=0x00000000`). This is the
next frontier — sovereign falcon microcode loading.

## Files Changed

- `/etc/toadstool/glowplug.toml` — K80 entries replaced with Titan V #2
