# Experiment 130: No-FLR Recovery and PRI Ring Lessons

**Date:** 2026-04-01
**Hardware:** Tesla K80 (GK210 dual-die), Titan V (GV100)
**Status:** VALIDATED — all reset paths tested on live hardware post-reboot

## Key Findings

### 1. K80 and Titan V Lack FLR

Neither GPU supports PCI Function Level Reset:
- K80 (Kepler): `reset_method` is empty
- Titan V (Volta GV100): `VFIO_DEVICE_RESET` returns EINVAL

Recovery must be handled entirely by our driver stack.

### 2. PRI Ring Writes from Userspace Are Destructive

Attempting `gk104_privring_init`-style writes through VFIO BAR0 mmap:
```
0x12004C = 0x2  (PRIV_RING_COMMAND ACK)
0x122204 = 0x2  (GPC0 slave ring start)
0x122304 = 0x2  (FBPA slave ring start)
0x12004C = 0x4  (ack)
0x12004C = 0x2  (start)
```

**Result:** Corrupted the PRI ring fabric on both die 0 and die 1. FECS registers
that previously read 0x10 (HALTED) switched to 0xbad0011f (PRI fault). The corruption
is persistent across PMC_ENABLE toggles — only a full reboot recovers.

**Root cause:** nouveau's `gk104_privring_init` uses `nvkm_mask()` (read-modify-write),
not raw writes. Our raw writes overwrote entire register contents with wrong values,
corrupting the ring topology state machine.

**Lesson:** Never write to PRI ring control registers from userspace. The ring fabric
must be initialized by VBIOS devinit during POST, or by the kernel driver at modprobe.

### 3. PMC Soft-Reset Is the Universal Recovery Path

PMC_ENABLE toggle (write 0, wait, restore) successfully resets all engine domains:
```
glowplug: device.write_register offset=0x200 value=0 allow_dangerous=true
sleep 100ms
glowplug: device.write_register offset=0x200 value=0xe011316c allow_dangerous=true
```

After PMC reset, die 1 returned to baseline PRI-fault state (0xbadf1200 = standard
engine-uninitialized, same as boot). Die 0's ring corruption persisted through PMC reset,
indicating the PRI ring fabric state is separate from PMC-managed engines.

### 4. PFIFO Clock Domain Requires VBIOS DEVINIT

Even with the nvidia-470 recipe (775 register writes from cold→warm diff):
- PGRAPH domain becomes accessible (FECS CPUCTL readable, FECS PIO boot succeeds)
- PFIFO domain remains in PRI fault (PFIFO_ENABLE=0, PBDMA_MAP=0xbad0011f)

The PFIFO clock domain is gated by PLL/clock configuration that only VBIOS devinit
scripts set up. The nvidia-470 recipe covers PGRAPH clocks but not PFIFO clocks.

### 5. Ember mmio.read Uses sysfs BAR0 (Not VFIO)

`ember.mmio.read` opens `/sys/bus/pci/devices/{bdf}/resource0` via `Bar0Access`,
not the VFIO device fd. This means ember MMIO reads can differ from test MappedBar
reads (which use VFIO BAR0 mmap). Both paths map the same physical resource, but
access timing and PRI ring state can cause divergent results.

## Architecture Changes (Phase 1 — pre-reboot)

| Change | Location | Purpose |
|--------|----------|---------|
| Remove `init_pri_ring()` | `kepler.rs` | Destructive writes broke die state |
| PRI fault detection | `init.rs` | PRI values (0xbad0xxxx) no longer trigger false warm handoff |
| `pmc_soft_reset()` | `vfio_compute/mod.rs` | Universal no-FLR recovery via BAR0 |
| `device.reset method=pmc` | glowplug `device_ops.rs` | RPC for PMC reset |
| FLR through ember | `helpers.rs` | `try_vfio_flr()` uses ember's held VFIO fd |
| `open_from_fds_with_recipe()` | `vfio_compute/mod.rs` | Cold-boot recipe for Kepler |

## Architecture Changes (Phase 2 — post-reboot validation)

| Change | Location | Purpose |
|--------|----------|---------|
| PRI fault in `fecs.state` | ember `handlers_device.rs` | `pri_fault` field prevents false "running" |
| PRI guard in warm_handoff poll | glowplug `device_ops.rs` | PRI faults no longer mistaken for FECS running |
| PRI guard in `fecs_is_alive` | `gr_context.rs` | Rejects PRI values before checking HALTED/STOPPED bits |
| PRI guard in `GrEngineStatus` | `gr_status.rs` | `fecs_inaccessible()` + `fecs_halted()` PRI-aware |
| PRI guard in `FalconState` | `diagnostics.rs` | `is_inaccessible()` for PRI/DEAD_DEAD values |
| PRI guard in observer | `observer/vfio.rs` | Trace analysis rejects PRI faults as "running" |
| Ember PMC soft-reset | `helpers.rs` | `try_pmc_soft_reset()` via sysfs BAR0 for non-FLR GPUs |
| Ember FLR capability check | `helpers.rs` | `device_has_flr()` reads PCIe DevCap bit 28 |
| Ember auto-reset routing | `handlers_device.rs` | FLR→ember FD, no-FLR→PMC, graceful FLR rejection |
| Glowplug reset→ember | `device_ops.rs` | All resets route through ember (FD holder) |
| Remove sudo/pkexec | `warm_fecs.rs`, `deploy-dev.sh` | Livepatch via ember RPC, binds via ember swap |

## Post-Reboot Validation Results (2026-04-01)

```
ember.fecs.state (Titan V):  pri_fault=false halted=true  cpuctl=0x00000010
ember.fecs.state (K80 die 0): pri_fault=true  halted=false cpuctl=0xbadf1200
ember.device_reset auto (Titan V): PMC soft-reset, 40ms — SUCCESS
ember.device_reset flr  (Titan V): "does not support PCIe FLR" — CORRECT REJECTION
ember.device_reset pmc  (K80):     PMC soft-reset, 40ms — SUCCESS
glowplug device.reset auto (K80):  Routes through ember → PMC — SUCCESS
glowplug device.reset flr (Titan V): Routes through ember → CORRECT REJECTION
```

K80 exp128a2 (nvidia-470 recipe): FECS RUNNING (cpuctl=0x0), PFIFO dead (needs DEVINIT).
Titan V warm handoff (nouveau): BLOCKED — `pmu: firmware unavailable` on GV100.

## Next Steps

1. **K80:** Investigate VBIOS DEVINIT replay for PFIFO clock domain
2. **Titan V:** Find PMU firmware path (nouveau extraction or nvidia driver route)
3. **Both:** Validate through toadStool `shader.dispatch` once channels are available
