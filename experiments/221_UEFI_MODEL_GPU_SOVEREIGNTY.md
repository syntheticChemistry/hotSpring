# Experiment 221: UEFI Model GPU Sovereignty

**Date**: 2026-05-24
**Status**: Phase 3 complete (FirmwareInterface evolution). Phase 4 pending hardware validation.
**Upstream plan**: `uefi_model_gpu_sovereignty_5ead68c1.plan.md`

## Hypothesis

Treat the GPU's firmware boot chain (ACR→FECS→GPCCS) as a UEFI-like boot
service. Let the vendor driver perform full initialization, then transition
to sovereign control via an "ExitBootServices"-like handoff, preserving the
established PRI ring topology and TPC stations.

## Phase 1: Validate Boot Service Provider

**Goal**: Determine which driver can serve as GPU boot service on GV100 (Titan V).

### Nouveau (failed)
- nouveau lacks PMU firmware for GV100's SEC2-based ACR boot chain
- `dmesg` reports `pmu: firmware unavailable` and PRIVRING faults
- GR engine never initializes — no FECS/GPCCS/TPC station creation
- **Conclusion**: nouveau cannot be the boot service for Volta+

### NVIDIA 470.256.02 (confirmed as boot service)
- nvsov catalyst pattern (patched nvidia via `nvidia_catalyst_handoff`) loads successfully
- FECS firmware runs: `fecs_pc=0x0e24` (valid execution address)
- GPCCS firmware loaded: `gpccs_cpuctl=0x00000010` (HRESET set)
- PMU running: `pmu_cpuctl=0x00000040`
- BAR0 probing during nvidia ownership shows TPC registers return PRI faults,
  likely due to nvidia's runtime BAR0 MMIO locking (confirmed on RTX 5060 host)
- **Conclusion**: nvidia RM IS the boot service. It performs full
  ACR→FECS→GPCCS→TPC initialization.

**Settle time**: 60s is sufficient for complete nvidia RM initialization.

## Phase 2: Selective Teardown

**Goal**: Identify teardown functions that destroy PRI ring and NOP them.

### Diagnostic BAR0 Probe
Added inline BAR0 probe between nvidia unbind and vfio-pci rebind in
`sovereign_handoff.rs`. This definitively showed:

| Register | Before Unbind | After Unbind |
|----------|--------------|--------------|
| FECS PC  | `0x00000e9a` | `0xbadf5040` |
| GPCCS PC | valid        | PRI fault    |
| TPC0     | PRI fault    | PRI fault    |

The PRI ring is **destroyed during nvidia's `nv_pci_remove`** despite the
4 catalyst surgical NOPs at offsets `0x1fe`, `0x2a0`, `0x374`, `0x3a0`.

### nvidia_boot_services Patch Set (created)
Extended catalyst's 4 NOPs with additional targets:
- `nv_shutdown_adapter` (0x441) — runs hardware shutdown sequence

**Result**: Even a single additional NOP (`nv_shutdown_adapter`) causes
the driver's unbind to hang/deadlock. `nv_shutdown_adapter` is critical
for RM's internal state machine to complete removal.

### Finding: NOP Ceiling Reached
The 4 catalyst NOPs are the maximum that can be applied while still
allowing a clean nvidia unbind. Adding `nv_shutdown_adapter` NOP hangs
the unbind.

### CRITICAL FINDING: Full NOP Still Destroys PRI Ring

Experiment `nvidia_boot_services_titanv` (2026-05-24 19:49):
- Patch set: `nvidia_boot_services` — catalyst init (full RM compute
  init) + RetAtEntry on nv_pci_remove (zero teardown code runs)
- Result: **nv_pci_remove returned instantly (2s unbind via fire-and-poll),
  yet FECS pc shifted from running to 0xbadf5040 (PRI fault)**

This proves:
- **PRI ring destruction is NOT in nv_pci_remove**
- **It's in the kernel's PCI unbind infrastructure** (`pci_device_remove`,
  `pcibios_disable_device`, or IOMMU group operations)
- Possible causes:
  - PCI bus master disable (bit 2 of command register cleared during unbind)
  - `pci_disable_device()` power state transitions
  - IOMMU domain teardown/isolation
  - PCI config space write that triggers GPU-internal reset

### Path Forward

Since the PRI ring cannot survive any standard PCI driver unbind, the
options are:

1. **Patch the kernel PCI unbind path** — skip bus master disable and
   device disable for sovereign GPU devices
2. **No-unbind model** — keep nvsov loaded, use BAR0 through the
   existing driver's mapping. Skip vfio-pci entirely.
3. **PRI ring re-initialization from Cold** — accept PRI ring death.
   After vfio-pci binds, re-enumerate PRI ring stations using direct
   BAR0 writes to PRI_RINGMASTER_COMMAND (0x122000). The firmware's
   work isn't preserved, but we can replay the initialization sequence.
4. **SIGSTOP + memory transplant** — freeze the nvidia process and
   remap its BAR0 mapping to our process

## Phase 3: FirmwareInterface Evolution

**Goal**: Add boot service lifecycle to the trait system and create
PriRingAnchor in ember.

### Changes

1. **`FirmwareInterface` trait** (`glowplug/firmware.rs`):
   - Added `boot_services_complete() -> bool` — checks if firmware has
     finished init (FECS running + GPCCS loaded)
   - Added `exit_boot_services() -> Result<BootServiceEvidence>` — captures
     register state as evidence before handoff
   - Added `runtime_services_available() -> bool` — indicates post-handoff
     firmware capability
   - All have default implementations (false/Err) for non-boot-service devices

2. **`BootServiceEvidence`** (`ember/pri_ring_anchor.rs`):
   - HashMap of preserved hardware state snapshots
   - Engine name, description, timestamp
   - Builder methods `new()` and `record()`

3. **`PriRingAnchor`** (`ember/pri_ring_anchor.rs`):
   - Holds `BootServiceEvidence` across driver swaps
   - `PriRingHealth`: `Healthy | Degraded | Destroyed | Unknown`
   - Methods: `is_compute_ready()`, `needs_reboot()`, `update_health()`
   - Full serde roundtrip support

4. **`SwapOrchestrator`** (`glowplug/swap.rs`):
   - New optional `ExitBootServicesFn` callback slot
   - Inserted between Persist and Drop in the orchestration lifecycle
   - When configured, captures firmware evidence before the handle is released
   - Step logged as `exit_boot_services` (Ok or Skipped)

5. **`GpuFirmwareAccess`** (`runtime/gpu/glowplug/firmware.rs`):
   - `boot_services_complete()` checks FECS cpuctl+pc and GPCCS cpuctl
   - `exit_boot_services()` probes all Falcon engines + TPC status across
     6 GPCs, records register values as evidence

### Test Results
- 121 ember tests: all pass
- 109 glowplug tests: all pass (7-step lifecycle unchanged when
  ExitBootServices is not configured)
- 6 GPU firmware tests: all pass
- Binary deployed to `/usr/local/bin/toadstool`

## Phase 4: PRI Ring Recovery from Cold

### Critical Finding: PRI Ring Destruction is in Kernel PCI Framework

Tested `nvidia_boot_services_titanv` with RetAtEntry on `nv_pci_remove`:
- `nv_pci_remove` returned instantly (2s fire-and-poll)
- Zero nvidia teardown code executed
- **PRI ring still destroyed** (FECS pc: `0xbadf5040`)
- `request_mem_region` leaked → both cards locked after RetAtEntry runs

**Root cause**: The kernel's `pci_device_remove()` clears PMC_ENABLE,
disabling PGRAPH (bit 12). Without PGRAPH, PRI ring routing to
GPC/TPC/FECS/GPCCS returns PRI faults. This is NOT nvidia's code — it's
the PCI framework's standard driver unbind path.

### PRI Ring Recovery Experiment (2026-05-24 19:50)

After RetAtEntry run on Card B, direct BAR0 writes proved PRI ring is
**partially recoverable**:

| Step | Action | Result |
|------|--------|--------|
| 1 | PMC_ENABLE: set bit 12 (PGRAPH) | `0x40000121` → `0x40001121` |
| 2 | PRI ring master enumerate (0x12004c=0x4) | Status: `0x1` (active) |
| 3 | PRI ring master start (0x12004c=0x1) | Status: `0x1` |
| 4 | Read FECS cpuctl (0x409100) | `0x00000010` (halted, **accessible**) |
| 5 | Read FECS PC (0x40911c) | `0x00000000` (reset, not PRI fault) |
| 6 | Read FECS IMEM[0:16] | All zeros — **firmware wiped** |

**Result**: Top-level PRI ring recovers. Falcon registers accessible.
But IMEM is empty — firmware was wiped during unbind. GPC sub-ring status
`0xcf` (error/disconnected).

### Register Correction

Previously used `0x409624` for "FECS PC" — this is actually a CTXSW register
routed through the GPC sub-ring. Corrected to:
- `0x409100`: FECS CPUCTL (top-level PRI, reliable)
- `0x40911c`: FECS hardware PC (top-level PRI, reliable)
- `0x409624`: FECS CTXSW status (GPC sub-ring, broken after unbind)

### Architecture Pivot: Recover Rather Than Preserve

Since PRI ring cannot survive standard PCI unbind, the approach becomes:

1. **Boot Service Phase** (nvidia loaded): capture FECS/GPCCS IMEM/DMEM
   firmware before unbind
2. **Clean Unbind** (catalyst): standard `nv_pci_remove` runs cleanly,
   PRI ring dies as expected
3. **PRI Ring Recovery** (post-swap): re-enable PGRAPH, enumerate PRI
   ring stations, acknowledge interrupts
4. **Firmware Replay** (future): load captured firmware back into
   IMEM/DMEM, restart falcons for Tier 2

### Implementation (2026-05-24)

- `recover_pri_ring()` added to `sovereign_handoff.rs` — re-enables PGRAPH
  and re-enumerates PRI ring stations after unbind
- `nvidia_boot_services` patch set updated — no longer uses RetAtEntry
  (leaks iomem), now matches catalyst with post-swap recovery
- Diagnostic probe updated with correct falcon PC register (0x40911c)
  and PCI config space reads
- `exit_boot_services()` enhanced with PRI ring master status and
  PGRAPH status capture
- All tests pass (20 sovereign_handoff, 121 ember, 6 gpu firmware)

### Post-Reboot Validation (2026-05-24 20:40)

Ran `nvidia_boot_services_titanv` on both cards with PRI ring recovery:

| Card | PRI Recovery | FECS | GPCCS | IMEM | iomem Leak |
|------|-------------|------|-------|------|------------|
| A (02:00.0) | PMC→0x40001121 PGRAPH=ON | accessible | accessible | wiped | none |
| B (49:00.0) | PMC→0x40001121 PGRAPH=ON | accessible | accessible | wiped | none |

Both handoffs: success, 71s, clean unbind, clean rmmod, repeatable.

### IMEM Capture Attempts

1. **While nvidia loaded** (pre-swap): PIO writes to IMEMC gated by RM.
   IMEMD reads return zeros. nvidia RM blocks host PIO to FECS registers
   while active.

2. **After unbind + recovery** (post-swap): PIO works (IMEMC write
   accepted), but IMEM is genuinely empty. Firmware wiped during
   nvidia's `nv_pci_remove` teardown.

### CRITICAL: Firmware Replay is Not Viable on GV100

FECS/GPCCS are **fuse-enforced HS** (high-security) mode on GV100:
- `SCTL` bits[13:12] = 2 (HS), set by hardware fuses
- Direct IMEM PIO upload from host is **blocked** in HS mode
- Only ACR secure boot chain can load firmware
- ACR requires WPR (Write-Protected Region) in GPU memory
- Exp 173 proved: nvidia RM does NOT configure WPR on GV100 (pre-GSP era)
- nouveau lacks PMU firmware for GV100's ACR chain

**Conclusion**: Tier 2 (WarmCompute with FECS/GPCCS running) cannot be
achieved through unbind+recovery on GV100. The fuse-enforced falcon
security model is the ultimate barrier.

### Architecture Evolution: Runtime Services Model

Instead of ExitBootServices (firmware dies), use **Runtime Services**:

| UEFI Concept | GPU Equivalent | Status |
|--------------|---------------|--------|
| Boot Services | nvidia RM (ACR→FECS→GPCCS init) | Works |
| ExitBootServices | PCI unbind + PRI ring recovery | Works (Tier 1) |
| Runtime Services | nvidia as persistent compute backend | **New model** |

The path to Tier 2 on GV100 requires keeping nvidia loaded as a
"runtime service" — the UEFI model evolves from "exit and recover" to
"coexist and delegate". nvidia manages the falcon security boundary,
toadStool manages infrastructure (PFIFO, DMA, VRAM, PRI ring).

## Key Insights

1. **The firmware boundary is real**: GPU compute initialization REQUIRES
   vendor firmware (ACR→FECS→GPCCS). No open-source alternative for Volta+.

2. **The UEFI model is correct, but the transition is destructive**: Unlike
   real UEFI where `ExitBootServices` is clean, PCI unbind destroys the
   GPU's internal state. The Linux PCI framework clears PMC_ENABLE on
   driver unbind.

3. **PRI ring IS recoverable**: Re-enabling PGRAPH via BAR0 write to
   PMC_ENABLE (0x200, bit 12) + PRI ring master enumerate restores
   top-level PRI routing. Falcon hardware (FECS/GPCCS) becomes
   accessible again.

4. **Firmware is NOT recoverable**: IMEM is wiped during unbind. The
   path to Tier 2 requires capturing firmware during the boot service
   phase and replaying it after recovery.

5. **RetAtEntry is a dead end**: NOPing `nv_pci_remove` doesn't help
   (kernel PCI framework still kills PRI ring) AND leaks iomem regions,
   locking the card until reboot.

6. **Falcon HS fuses are the real boundary**: On Volta+, FECS/GPCCS
   are fuse-locked to HS (high-security) mode. No host IMEM access.
   Firmware can only be loaded through ACR secure boot, which requires
   WPR — not available on GV100 from vfio-pci. This is the fundamental
   reason Tier 2 requires nvidia RM.

7. **PRI ring recovery enables Tier 1+**: Even without Tier 2 compute,
   PRI ring recovery provides:
   - PGRAPH engine control (enable/disable/status)
   - Falcon register access (cpuctl, bootvec, mailbox)
   - PRI ring topology information
   - Infrastructure for future WPR configuration or ACR boot attempts
