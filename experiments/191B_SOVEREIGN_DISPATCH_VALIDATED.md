# Experiment 191B: Sovereign Dispatch Validated on Titan V

**Date:** May 14, 2026
**Hardware:** Titan V (GV100 / SM70) @ 0000:02:00.0
**Stack:** toadStool S262 (locally evolved), coralReef Sprint 11, barraCuda Sprint 69
**Predecessor:** Exp 191 (toadStool S258 PBDMA validation — planned)

---

## Summary

First successful end-to-end sovereign VFIO dispatch on Titan V through
toadStool IPC. All phases validated:

1. **Warm handoff** — patched nouveau NOP'd teardown preserves live FECS
2. **Warm catch** — BAR0 PMC_ENABLE probe detects live-warm state
3. **VFIO open** — iommufd/cdev path, PFIFO channel created
4. **DMA roundtrip** — alloc → upload → dispatch → sync → readback
5. **GR context init** — SET_OBJECT with Volta compute class 0xC3C0

## Bugs Found and Fixed During Validation

### Bug 1: `warm_detect()` reads PCI config space instead of BAR0

**File:** `crates/server/src/glowplug_client.rs`

`device.warm_catch` RPC always returned `pmc_enable: 0x00000000` because it
called `read_pci_config_u32(bdf, 0x200)` — reading PCI config space at offset
0x200, not BAR0 register 0x200 (PMC_ENABLE).

**Fix:** Replaced with `nvpmu::bar0::Bar0Access::open(bdf)` which mmaps
`/sys/bus/pci/devices/{bdf}/resource0` and reads via volatile aligned MMIO.
Also added FECS CPUCTL readout (0x409100) to the warm-catch response.

### Bug 2: `probe_warm_fecs()` rejects live-warm FECS state

**File:** `crates/core/cylinder/src/nv/compute_device.rs`

Required `halted && mb0 != 0` — only recognized FECS that was stopped with
firmware resident. Our NOP'd-teardown patched nouveau leaves FECS **running**
(CPUCTL=0x10, not halted, MAILBOX0=0x0). This is a strictly better warm state.

**Fix:** Added `live_warm` detection path: FECS running + PMC popcount ≥ 16
is recognized as compute-ready alongside the existing preserved-warm path.

### Bug 3: Ember gate deadlocks on server-internal BAR0 probes

**File:** `crates/core/cylinder/src/vfio/ember_gate.rs`

The VFIO device factory in the server calls `probe_warm_fecs()`, which opens
`SysfsBar0`, which checks the ember gate. The gate connects to the ember socket
(same process) to ask `ember.list`, causing deadlock or self-blocking.

**Fix:** Added `EmberGateBypass` — thread-local RAII guard that disables the
gate. Used in `try_vfio_nvidia()` since the server IS ember.

### Bug 4: Legacy `coral-ember` holds VFIO FDs, blocking toadStool

Two old `coral-ember` processes (boot service + glowplug-spawned) held
`/dev/vfio/devices/vfio1` and `/dev/iommu`, preventing toadStool's
iommufd open from succeeding and causing EBUSY on the legacy group path.

**Fix:** Stopped and disabled `coral-ember.service` + `coral-glowplug.service`.
toadStool supersedes both.

### Bug 5: VFIO device drop triggers GPU reset (device-per-call pattern)

The factory created a new `NvVfioComputeDevice` per RPC call. When the device
was dropped after `device.vfio.open`, the iommufd FDs closed and the GPU
went cold (PMC_ENABLE dropped from 23 engines to 2).

**Fix:** Added persistent `cached_devices` map to `DispatchHandler`. Devices
are created once and held across calls. The factory pattern is preserved but
devices are cached after first creation.

## Hardware Data

### Warm State (after NOP'd nouveau teardown)

```
PMC_ENABLE     = 0x5fecdff1  (popcount = 23)
FECS_CPUCTL    = 0x00000010  (running, not halted)
FECS_MAILBOX0  = 0x00000000  (no pending message)
BOOT0          = GV100 / SM70
```

### VFIO Open

```
Backend        = iommufd/cdev (kernel 6.17)
IOMMU Group    = 69
CDEV           = vfio0
Regions        = 9
IRQs           = 5
BAR0 Size      = 16 MiB
```

### Channel Creation

```
Channel ID     = 0
GPFIFO IOVA    = 0x10000
USERD IOVA     = 0x11000
Instance IOVA  = 0x3000
PFIFO Live     = true
Doorbell       = Usermode
```

### DMA Roundtrip

```
Buffer         = 256 bytes (inout)
Dispatch Path  = local_cylinder
Status         = completed
Dispatch Time  = 1053 ms
Readback       = 256 bytes (zeroes — no-op shader binary)
```

### GR Context Init

```
Method Entries = 1 (SET_OBJECT 0xC3C0)
Status         = completed
Time           = < 1 ms
```

## CPUCTL_ALIAS Breakthrough (S263, May 15, 2026)

Critical discovery: Volta HS (Heavy Secure) falcons security-lock the `CPUCTL`
register at offset 0x100. Reading it always returns 0x10 (`HRESET`), regardless
of actual FECS state. This caused false "FECS dead" diagnoses throughout the
warm handoff pipeline.

The true FECS state is available via `CPUCTL_ALIAS` at offset 0x130, which
bypasses the HS security lock. Using CPUCTL_ALIAS, FECS was confirmed **alive
and running throughout** the warm handoff — nouveau's FECS context survives
the driver swap completely.

### Falcon Register Map (Volta HS Mode)

| Register | Offset | HS Behavior |
|----------|--------|-------------|
| `CPUCTL` | 0x100 | Security-locked, always reads 0x10 (HRESET) |
| `CPUCTL_ALIAS` | 0x130 | True state — use this for all probes |
| `MAILBOX0` | 0x040 | Readable |
| `PC` | 0x034 | Readable |

### Current Frontier: PENDING_CTX_RELOAD

With FECS confirmed alive, the dispatch pipeline completes e2e (DMA roundtrip
works, dispatch returns `completed`). However, PBDMA is not consuming pushbuffers
(`hw_get` not advancing). Channel status shows `PENDING_CTX_RELOAD` — FECS
has not loaded a GR context for our channel.

A 1MB GR context DMA buffer was allocated and its IOVA written into the channel
instance block. A scheduler cycle (`resubmit_runlist`: disable → preempt →
re-enable) was implemented to force FECS to re-evaluate. The channel still
shows PENDING_CTX_RELOAD.

**Hypothesis:** FECS (from nouveau) needs to copy a "golden context" from VRAM
into our system memory GR_CTX buffer. Our page table only maps system memory,
not VRAM, which may prevent this DMA transfer. Next step: identity-map a VRAM
region or extract the GR context initialization sequence from nouveau source.

## Next Steps

1. Solve PENDING_CTX_RELOAD — map VRAM golden context or extract nouveau GR init
2. Compile a real shader via coralReef (`shader.compile.wgsl`) and dispatch it
3. Verify GPU execution modifies buffer contents (not just DMA roundtrip)
4. Wire K80 PLX keepalive into toadStool native device management
5. Push local toadStool changes upstream with this validation data
