# SPDX-License-Identifier: AGPL-3.0-only

# hotSpring Handoff: Dual Titan V Twin Study Baseline

**Date:** 2026-05-17
**From:** hotSpring (Exp 205)
**To:** toadStool, coralReef, primalSpring, sibling springs
**Scope:** Dual-GPU hardware topology, twin study surface, sovereign pipeline validation

---

## What Was Done

### 1. Hardware Change — K80 Replaced with Second Titan V

The Tesla K80 (dual GK210B behind PLX PEX 8747) was removed after its
fire incident (Exp 199). A second identical Titan V (GV100, 12 GB HBM2)
was installed in the same PCIe slot.

| Card | BDF | IOMMU Group | Driver | Role |
|------|-----|-------------|--------|------|
| Titan V #1 | `0000:02:00.0` | 65 | `vfio-pci` | oracle |
| Titan V #2 | `0000:49:00.0` | 32 | `vfio-pci` | compute |
| RTX 5060 | `0000:21:00.0` | — | `nvidia` | display |

Both Titan Vs negotiate PCIe Gen3 x8. Each IOMMU group contains only
the GPU + audio function pair — clean isolation for independent VFIO.

### 2. Sovereign Pipeline Validation

Both cards pass `sovereign.init` with identical stage results:

- `bar0_probe`: boot0=0x140000a1, chip_id=0x140
- `pmc_enable`: 0x5fecdff1 (all engines enabled, warm state)
- `cg_sweep`: 0 changed, 12 faulted (converged)
- `pri_recovery`: 9 alive, 4 faulted, recovered=true
- `pgob_ungating`: 14 GPCs alive
- `memory_training`: skipped (warm)
- `falcon_boot`: failed (ACR no DMA — known frontier)

Total pipeline time: ~3164ms per card, within 1ms of each other.

### 3. Register & ROM Parity

All key registers match between cards (except PTIMER, which is expected
to diverge as a monotonic counter):

| Register | Value | Match |
|----------|-------|-------|
| PMC_BOOT0 | 0x140000a1 | MATCH |
| PMC_ENABLE | 0x5fecdff1 | MATCH |
| PMC_ID | 0xbad00200 | MATCH |
| FUSE | 0xc0040000 | MATCH |
| NV_PROM[0] | 0xeb72aa55 | MATCH |
| PGRAPH_STATUS | 0x00000000 | MATCH |

VBIOS ROM header (first 64 bytes) produces identical SHA-256:
`af04a2c636da558f`. Any opcode/stride fix applies to both cards.

### 4. Infrastructure Change

`/etc/toadstool/glowplug.toml` updated: K80 entries replaced with
`titan-v-2` at `0000:49:00.0`, health_policy=active.

---

## What This Enables (Twin Study Surface)

- **A/B register experiments**: Write different init sequences to each card
  without firmware variance as a confound
- **Warm vs cold divergence**: FLR or D3-cycle one card while the other
  stays warm to study state drift in isolation
- **VBIOS interpreter ground truth**: Identical ROMs mean any script fix
  produces identical op counts on both cards
- **Parallel falcon ACR debugging**: Try different boot strategies
  simultaneously
- **PCIe domain timing**: Compare register write latencies across IOMMU
  groups 65 vs 32

---

## Current Blocker

**Falcon ACR boot**: Both cards fail at `falcon_boot` because ACR HS boot
requires a DMA backend not yet provided. GPCCS PIO boot also times out
(`cpuctl=0x00000000`). This is the shared next frontier.

---

## Upstream Dependencies

- **toadStool**: `sovereign.init` works on both BDFs via compute.sock
- **ember daemon**: All 3 devices enumerated after config update + restart
- No code changes to toadStool or barracuda required for this experiment
