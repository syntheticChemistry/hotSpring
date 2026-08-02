# Experiment 235: biomeGate Fleet Bootstrap + BAR0 First Contact

**Date:** 2026-08-02
**Status:** COMPLETE
**Gate:** biomeGate
**Hardware:** RTX 5060 (SM100), Titan V GV100 (SM70), Tesla K80×2 GK210 (SM37)

## Objective

Bootstrap toadStool sovereign infrastructure on biomeGate for the first time.
Establish BAR0 MMIO access to all VFIO-bound GPUs. Identify silicon via
PMC_BOOT0. Map the generation landscape before any warm cycles.

This is the foundation experiment — everything else depends on MMIO access.

## Procedure

1. Build and start toadStool server (`toadstool server --socket`)
2. Verify VFIO device detection and PLX bridge pinning
3. Read PMC_BOOT0 (offset 0x0) on each VFIO GPU — chip identification
4. Read PMC_ENABLE (offset 0x200) — engine state (cold vs warm)
5. Read interrupt registers (0x140, 0x160, 0x180) — generation baseline

## Results

### Fleet Bootstrap

```
Socket:     /tmp/toadstool-biome.sock  (riboCipher handshake: [0xEC, 0x01])
TCP:        127.0.0.1:42217
Fleet file: /run/user/1000/biomeos/toadstool-ember-fleet.json
Mode:       standalone (development, no biomeOS)
```

Root required for BAR0 sysfs resource0 access (VFIO cdev path not wired
for mmap in current server build). Socket created at 0600 by root, chmod
666 for user-space experiment binaries.

### PLX Bridge Topology

```
3 VFIO GPUs, 3 bridge hierarchies pinned (10 total):
  0000:49:00.0  PLX upstream bridge    → K80 hierarchy
  0000:4a:08.0  PLX downstream fn0     → 0000:4b:00.0
  0000:4a:10.0  PLX downstream fn1     → 0000:4c:00.0
  0000:21:00.0  direct (no PLX)        → Titan V
```

### BAR0 First Contact

| BDF | PMC_BOOT0 | Chip | PMC_ENABLE | State |
|-----|-----------|------|------------|-------|
| 0000:21:00.0 | 0x140000A1 | GV100 (Titan V) | 0x40000121 (pop=4) | COLD |
| 0000:4b:00.0 | 0x0F22D0A1 | GK210 (K80 fn0) | 0xC0002020 (pop=4) | COLD |
| 0000:4c:00.0 | 0x0F22D0A1 | GK210 (K80 fn1) | 0xC0002020 (pop=4) | COLD |

PMC_ENABLE differs by generation even in cold state:
- GV100 cold: `0x40000121` (PBUS, PFIFO, PMC minimal)
- GK210 cold: `0xC0002020` (Kepler cold default, different engine mask)

### Interrupt Register Baseline

| BDF | 0x140 INTR_EN_0 | 0x160 INTR_EN_SET | 0x180 INTR_EN_CLR |
|-----|-----------------|-------------------|-------------------|
| 21:00.0 (GV100) | 0x00000000 | 0x00000000 | 0x00000000 |
| 4b:00.0 (GK210) | 0x00000000 | 0x00000001 | 0x00000000 |
| 4c:00.0 (GK210) | 0x00000000 | 0x00000001 | 0x00000000 |

K80's INTR_EN_SET@0x160 reads 0x1 while Titan V reads 0x0 — different
register semantics (on Kepler, 0x160 maps to a different hardware register
than on Volta where it's the SET interface). This is the generation
boundary visible in bare silicon.

### Infrastructure Issues Identified

1. `no_bus_reset` kmod fails: missing `flex`, kernel header `autoconf.h` mismatch
2. nouveau blacklisted: `/lib/modprobe.d/nvidia-graphics-drivers.conf` → `alias nouveau off`
3. nvidia-470 nvsov module not installed — catalyst pipeline blocked
4. rm_trigger binary not installed at `/usr/local/bin/`
5. `/dev/kmsg` sentinel requires CAP_SYSLOG

### K80 fn0 FLR Incident

During warm cycle attempt, `toadstool device swap 0000:4b:00.0 nouveau`
failed (nouveau blacklisted). The swap unbound vfio-pci but nouveau never
bound. Rebind to vfio-pci triggered VFIO FLR:

```
BEFORE: PMC_ENABLE = 0xC0002020 (pop=4, cold)
AFTER:  PMC_ENABLE = 0xFFFFFFFF (pop=32, all engines bulk-enabled)
```

Register interface locked. INTR_EN_0 reads 0xFFFFFFFF, writes are dropped.
This is the Exp 199 failure class — proves PowerSafetyProfile is not cosmetic.
Recovery requires PCIe SBR or power cycle.

## Success Criteria

- [x] toadStool server starts and binds socket
- [x] All 3 VFIO GPUs detected with correct VEN:DEV
- [x] PLX bridges pinned for K80 hierarchy
- [x] PMC_BOOT0 readable on all 3 GPUs (chip identification)
- [x] PMC_ENABLE readable (cold state confirmed)
- [x] Generation baseline registers captured
- [x] Fleet file written for client discovery
