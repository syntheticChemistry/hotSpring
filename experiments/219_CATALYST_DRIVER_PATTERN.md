# Experiment 219 — Catalyst Driver Pattern for TPC Sovereignty

**Date**: 2026-05-22 (infra) / 2026-05-24 (HW execution + teardown profiling)
**Status**: ✅ HARDWARE VALIDATED — catalyst pipeline completes in 26s, golden state captured
**Hardware**: 2x NVIDIA Titan V (GV100), BDFs `0000:02:00.0`, `0000:49:00.0`
**Dependency**: Exp 218 (nvidia-470 dual-load co-existence proved), Exp 217 (TPC wall confirmed firmware-dependent)

## Objective

Treat the proprietary nvidia-470 driver as a chemical catalyst: load it to
fully initialize the GPU compute pipeline (SEC2→ACR→PMU→GPCCS→FECS→TPC),
capture the resulting register state as a "golden snapshot", remove the
catalyst (warm swap to vfio-pci), and replay the captured state on future
boots — achieving TPC sovereignty without the proprietary driver at runtime.

## Core Insight

A catalyst enables a reaction but is not consumed in the product. The
nvidia-470 driver initializes GPU state that we want (TPC PRI stations);
we capture that state, remove the driver, and replay it. The twin Titan Vs
provide the A/B surface: Card A stays cold as the control, Card B runs the
catalyst, and the delta between them IS the catalyst product.

## Changes from Exp 218

### Phase 1: Selective Un-NOP (`nvidia_catalyst_handoff`)

Exp 218 used `nvidia_warm_handoff` (17 targets), which NOPed `nv_cap_init`
and `nv_cap_drv_init`. This prevented RM capability table creation and
blocked engine initialization — result was Tier 0 (Cold).

The new `nvidia_catalyst_handoff` (15 targets) removes those two from the
NOP set, allowing RM to fully initialize while keeping all co-load isolation
NOPs to prevent host conflicts.

**Removed from NOP set:**
- `nv_cap_init` (was Ret1AtEntry)
- `nv_cap_drv_init` (was Ret1AtEntry)

**Kept (co-load isolation):**
All procfs, chardev, nvlink, nvswitch, and ACPI NOPs remain.

### Phase 2: Catalyst Capture Step

Inserted between `seeder_settle` and `prepare_warm_swap` in the handoff
pipeline. While the catalyst driver owns the GPU at peak initialization:

1. Opens BAR0 via `MappedBar::from_sysfs_rw()`
2. Captures `SovereignSnapshot` for tier-relevant registers
3. Captures `Bar0Snapshot::capture_full()` for all 16 MiB
4. Builds `GrInitSequence` via `to_catalyst_replay()`
5. Persists snapshot + replay sequence to `/tmp/toadstool-catalyst-*.json`
6. Logs tier evidence (TPC alive? FECS running?)

### Phase 3: Twin-Card Differential

New `sovereign.catalyst_diff` RPC captures full BAR0 from both cards,
computes `Bar0Diff`, and produces a minimal `GrInitSequence` containing
only the registers the catalyst changed. Persists cold baseline, warm
snapshot, delta, and replay sequence.

### Phase 4: Catalyst Preservation (3 Layers)

**Layer 1 — Recipe (mother culture):**
`infra/catalysts/recipes/gv100_nvidia470.toml` — versioned manifest of
DKMS source, NOP set, objcopy sections, module rename. Given the same
inputs, reproduces an identical catalyst `.ko`.

**Layer 2 — Frozen Binary (store-bought starter):**
`/var/lib/toadstool/catalysts/frozen/nvsov_gv100_470.256.02_k*.ko` —
archived on successful catalyst handoff. Skip the patch pipeline when
the kernel matches.

**Layer 3 — Product (golden state):**
`infra/catalysts/products/` — full BAR0 snapshot, cold→catalyst diff,
and replay sequence as JSON.

### Phase 5: Golden State Persistence & Replay

`engine_init_path` now wired in the ember handler for `sovereign.init`.
Loads `GrInitSequence` from JSON and feeds it into `engine_init_sequences`
for replay during Stage 3c (Engine Ungating).

### Phase 6: Catalyst-Free Boot

New `sovereign.catalyst_boot` RPC orchestrates:
1. `nouveau_titanv` warm handoff (183ms)
2. `sovereign.init` with `engine_init_path` pointing to golden state JSON
3. Returns combined result with tier classification

## New Code

### cylinder crate
- `gr_init.rs`: `InitSource::Catalyst { driver_version, bdf }` variant
- `pri.rs`: `VOLTA_BAR0_DOMAINS` constant (22 domain ranges)
- `module_patch.rs`: `nvidia_catalyst_handoff()` patch set (15 targets),
  `PatchSet::to_json()`, `by_name("nvidia_catalyst_handoff")`
- `warm_capture.rs`: `Bar0Snapshot::to_catalyst_replay()`,
  `Bar0Snapshot::to_json()`, `Bar0Diff::to_replay_sequence()`,
  `Bar0Diff::to_json()`
- `sovereign_handoff.rs`: `HandoffConfig::nvidia_catalyst_titanv()`,
  catalyst capture step, catalyst preservation step,
  `HandoffResult::{catalyst_snapshot_path, catalyst_alive_count, catalyst_tier}`

### server crate
- `dispatch/mod.rs`: `sovereign_catalyst_boot` RPC, `engine_init_path`
  loading in ember handler, `nvidia_catalyst_titanv` in strategy list
- `sovereign.rs`: `sovereign_catalyst_diff` RPC
- `mod.rs`: routing for `sovereign.catalyst_boot`, `sovereign.catalyst_diff`

### infra
- `infra/catalysts/recipes/gv100_nvidia470.toml` — catalyst recipe
- `infra/catalysts/frozen/` — archived catalyst `.ko` files
- `infra/catalysts/products/` — captured catalyst product JSONs
- `infra/golden_state/` — golden state replay sequences

## New RPCs

| RPC | Type | Purpose |
|-----|------|---------|
| `sovereign.warm_handoff` (strategy=`nvidia_catalyst_titanv`) | Blocking | Catalyst handoff with capture |
| `sovereign.catalyst_diff` | Stateless | Full BAR0 twin-card differential |
| `sovereign.catalyst_boot` | Ember | Catalyst-free boot (nouveau + replay) |

## Execution Plan

```
1. sovereign.snapshot bdf=0000:02:00.0                           # Card A cold baseline
2. sovereign.warm_handoff bdf=0000:49:00.0                      # Card B catalyst handoff
   strategy=nvidia_catalyst_titanv
3. sovereign.catalyst_diff bdf_cold=0000:02:00.0                # Twin-card delta
   bdf_warm=0000:49:00.0
   persist_path=infra/catalysts/products
4. cp /tmp/toadstool-catalyst-replay-*.json                     # Archive golden state
   infra/golden_state/gv100_catalyst.json
5. sovereign.catalyst_boot bdf=0000:49:00.0                     # Catalyst-free validation
   engine_init_path=infra/golden_state/gv100_catalyst.json
```

## Hardware Execution Results (May 24, 2026)

### Teardown Profiling Campaign

Full profiling identified `capture_full` (16 MiB linear BAR0 scan) as the sole
bottleneck — 462 seconds (7.7 min) reading 4,194,304 registers, with PCIe
completion timeouts (~110μs each) on unmapped inter-domain gaps.

**Fix**: `Bar0Snapshot::capture_domains()` reads only the 22 known Volta BAR0
domains (641K registers) instead of the full 16 MiB. Live registers respond in
~1μs, so the scan finishes in under 1 second.

### Pipeline Timeline (optimized)

| Step | Duration | Notes |
|------|----------|-------|
| preflight | 2,090ms | module clean, IOMMU group free, kernel healthy |
| module_prep | 378ms | DKMS patched nvsov prepared (13/13 patches) |
| unbind_current | 100ms | vfio-pci unbound, sibling detached |
| deferred_insmod | 451ms | nvsov loaded + bound via driver_override |
| seeder_bind | 50ms | nvsov confirmed bound |
| seeder_settle | 15,000ms | RM initialization settle |
| catalyst_capture | 0ms | pre-swap tier=WarmInfrastructure (deferred) |
| prepare_warm_swap | 0ms | bridge pinned, FLR disabled |
| warm_swap | 7,101ms | nvsov→vfio-pci via fire-and-poll (7s RM teardown) |
| catalyst_full_capture | **897ms** | **83,623 alive regs** from domain-scoped scan |
| tier_classify | 0ms | skipped (pre-swap tier used) |
| catalyst_preserve | 40ms | frozen .ko + recipe JSON archived |
| module_cleanup | 102ms | guarded rmmod nvsov |
| **TOTAL** | **26,317ms** | Clean success, no timeouts |

### Before / After Comparison

| Metric | Before (full scan) | After (domain scan) |
|--------|-------------------|---------------------|
| capture_full | 462,094ms (7.7 min) | **897ms** |
| Total pipeline | 488s+ (RPC timeout) | **26s** |
| Pipeline result | Timeout failure | **Clean success** |
| Alive regs captured | 174,098 | 83,623 |

### Surgical NOP Patches for nv_pci_remove

Replaced blanket `RetAtEntry` for `nv_pci_remove` with four surgical
`NopCallAt` patches — allows PCI resource cleanup (`__release_region`,
`pci_disable_device`) to execute normally while NOP-ing GPU teardown calls:

| Offset | Target Function | Purpose |
|--------|----------------|---------|
| 0x374 | `nv_shutdown_adapter` | Skip full GPU shutdown |
| 0x3a0 | `rm_disable_gpu_state` | Skip RM state teardown |
| 0x1fe | `rm_cleanup_dynamic_power_mgmt` | Skip power management cleanup |
| 0x2a0 | `rm_free_private_state` | Skip RM memory deallocation |

### Key Discoveries

1. **Bridge-level SBR recovery**: `setpci -s BRIDGE BRIDGE_CONTROL.W` toggling
   bit 6 (Secondary Bus Reset) recovers GPUs from dirty catalyst states without
   a full power cycle. PCI remove/rescan alone is insufficient.
2. **Domain-scoped capture**: 515x speedup by reading only known register
   domains instead of linear 16 MiB scan.
3. **Fire-and-poll unbind**: Non-blocking unbind + sysfs polling prevents
   toadstool-ember from entering D-state during 7s NVIDIA RM teardown.
4. **Pipeline reordering**: BAR0 capture moved before sibling rebind to avoid
   PCI device lock contention with RM teardown.

### Captured Artifacts

| Artifact | Path | Size |
|----------|------|------|
| BAR0 snapshot (domain-scoped) | `/tmp/toadstool-catalyst-0000-02-00-0.json` | ~28 MB |
| Replay sequence | `/tmp/toadstool-catalyst-replay-0000-02-00-0.json` | 83,623 writes |
| Frozen .ko | `/var/lib/toadstool/catalysts/frozen/nvsov_gv100_470.256.02_k6.17.9-*.ko` | 41 MB |
| Recipe JSON | `/var/lib/toadstool/catalysts/recipes/gv100_nvidia470_patchset.json` | metadata |

### Tier Evidence (pre-swap, while nvsov owns GPU)

```
tier = WarmInfrastructure
pmc_enable = 0x5fecdff1
fecs_cpuctl = 0x00000010
fecs_pc = 0x0000009c
gpccs_cpuctl = 0x00000010
pmu_cpuctl = 0x00000010
pgraph_status = 0x00000000
tpc_alive = false
```

Tier 1 (WarmInfrastructure) confirmed — Falcon engines running (FECS, GPCCS,
PMU all at cpuctl=0x10), but TPC stations did not survive the warm swap.
This matches Exp 217/218 findings — TPC station creation requires GPCCS firmware
execution, which the catalyst preserves in register state but loses during the
unbind→rebind swap.

## Catalyst-Free Boot Replay (May 24, 2026)

### Execution

`sovereign.catalyst_boot` on `0000:02:00.0` with 83,623 register writes from
the captured golden state replay JSON.

### Pipeline

| Stage | Status | Duration | Detail |
|-------|--------|----------|--------|
| nouveau warm handoff | ✅ | 8,097ms | Tier 1 baseline established |
| identity_probe | ✅ | 14ms | GV100 chip=0x140 |
| pmc_enable | ✅ | 78ms | 23 engines, `0x5fecdff1` preserved |
| pgraph_reset | skipped | 0ms | Warm — preserving FECS/GPCCS |
| cg_sweep | ✅ | 0ms | 0 changed, 12 faulted |
| pri_recovery | ✅ | 85ms | 9 alive, 4 faulted, recovered |
| pgob_ungating | ✅ | 0ms | 0 GPCs alive |
| memory_training | skipped | 0ms | warm detected (PRAMIN sentinel ok) |
| **engine_ungate:GR_INIT** | ✅ | **26ms** | **83,623 writes applied** |
| falcon_boot | ✅ | 0ms | FECS warm-preserved, cpuctl=0x10, pc=0xa12 |
| gr_init | skipped | 0ms | FECS warm-preserved/running |
| verify | ✅ | 14ms | ptimer running, VRAM ok |
| **Total** | ✅ | **219ms** | catalyst_free=true |

### Post-Replay Tier Evidence

```
tier = warm_infrastructure (Tier 1)
pmc_enable = 0x5fecdff1 (23 engines)
fecs_pc = 0xbadf5040 (PRI fault)
gpc_enables = 0x00000000
gr_status = 0x00000081
tpc_alive = false
tpc_status = 0xbadf5040 (PRI fault)
```

### Analysis: Why Replay Doesn't Achieve Tier 2

The 83,623 register writes were applied successfully in 26ms. However, the
post-replay tier classification shows PRI faults (`0xbadf5040`) on FECS, TPC,
and CE status registers, and `gpc_enables = 0`. **The replay wrote register
values but did not re-establish the underlying hardware state.**

The fundamental issue: TPC PRI station creation is **firmware-mediated**.
The GPCCS firmware running on the Falcon microcontroller *creates* PRI ring
routing stations that map BAR0 addresses to physical TPC/GPC units. Writing
register values back only sets the *output* of that routing process — it
doesn't create the routing infrastructure itself. Without the PRI ring
stations, subsequent reads to TPC/GPC registers return PRI faults.

This is consistent with:
- **Exp 217**: BAR0-only Tier 2 path definitively closed. Full ungating +
  `sw_nonctx.bin` replay + PGRAPH reset all fail to create TPC PRI stations.
- **Exp 218**: TPC state did not survive warm swap (Tier 0).
- **Exp 219 catalyst capture**: TPC alive = false even while nvsov owns GPU
  (register state captured but PRI stations already lost during swap).

### Implications

1. **Register replay is necessary but not sufficient** for Tier 2. The golden
   state captures what firmware produces, but the production process itself
   (GPCCS firmware execution → PRI ring station creation) cannot be replayed
   through register writes alone.

2. **The Tier 2 wall is at the PRI ring level**, not the register level. The
   GPU's PRI (PRIvilege) ring is a hardware interconnect that routes MMIO
   accesses to specific engine instances. Firmware creates entries in this
   routing table; without those entries, engine registers are unreachable.

3. **Paths forward for Tier 2**:
   - **GPCCS firmware replay**: Load GPCCS firmware blob + trigger execution
     post-swap (requires IMEM/DMEM write + cpuctl start)
   - **PRI ring station injection**: Directly program the PRI hub/router
     registers that create station entries (requires reverse-engineering the
     PRI ring topology programming sequence)
   - **Selective swap**: Keep GPCCS running during the catalyst unbind by
     using a more granular driver removal strategy
   - **Pre-swap station snapshot**: Capture PRI ring state (hub, router, station
     registers) while the catalyst driver is active, replay those specifically

## Twin-Card Differential (May 24, 2026)

### Execution

Card A (`0000:02:00.0`) cold baseline, Card B (`0000:49:00.0`) catalyst-warmed.
`sovereign.catalyst_diff` with domain-scoped capture (fixed from full 16 MiB).

### Results

| Metric | Value |
|--------|-------|
| Total compared | 641,088 registers (22 domains) |
| Changed (cold→catalyst) | 371,940 |
| Unchanged | 269,148 |
| Cold alive | 3,913 |
| Warm alive | 83,382 |
| **Diff replay writes** | **80,976** |
| Capture time | **2s** (both cards) |

### Domain Breakdown of Catalyst Product

| Domain | Writes | Notes |
|--------|--------|-------|
| PRAMIN | 66,326 | Instance memory (page tables, firmware contexts) |
| GPC | 7,417 | 6,974 in TPC range (0x504000+) |
| PGRAPH | 2,880 | Engine state registers |
| PDISP | 1,309 | Display engine |
| PBDMA | 1,033 | DMA push buffer descriptors |
| PRIV_RING | 959 | PRI hub/router registers |
| LTC | 413 | L2 cache partitions |
| PFIFO | 190 | Channel scheduling |
| CE | 160 | Copy engines |
| PMU | 94 | Power management unit |
| PTHERM | 67 | Thermal management |
| PBUS | 53 | Bus interface |
| PCLOCK | 39 | Clock domain |
| SEC2 | 10 | Security engine |
| PFB | 19 | Framebuffer |
| PTIMER | 4 | Timer |
| PMC | 3 | Master control |

### Diff Replay Boot Result

`sovereign.catalyst_boot` with `gv100_catalyst_replay.json` (80,976 writes):
same as single-card replay — Tier 1 (WarmInfrastructure), TPC alive = false.
FECS/GPCCS cpuctl=0x10 (firmware loaded) but PC reads return `0xbadf5040`
(PRI fault). PRI ring interrupt `0x003f0100` persists after clear.

### PRI Fault Analysis

Post-replay probe on `0000:02:00.0`:

```
PRI_RING_INTR       = 0x003f0100 (persistent, not clearable)
FECS cpuctl          = 0x00000010 (OK — firmware loaded)
FECS pc              = 0xbadf5040 (PRI FAULT — can't read through PRI ring)
GPCCS cpuctl         = 0x00000010 (OK — firmware loaded)
GPCCS pc             = 0xbadf5040 (PRI FAULT — can't read through PRI ring)
GPC0-5 enable        = 0x00000002 (all enabled)
TPC status           = 0x00000000 (no fault, but not alive)
```

The Falcon engines *exist* (cpuctl readable) but PRI ring routing to PGRAPH
Falcon instances is broken. The 959 PRIV_RING register writes from the diff
set up hub configuration but don't create the per-engine routing stations
that the PRI ring master programs during firmware execution.

**Key distinction**: Writing PRI ring *configuration* registers is not the same
as triggering the PRI ring master to *enumerate and route* engine instances.
The routing table is built by firmware (GPCCS/FECS) during initialization, and
the register values we captured are the *result* of that enumeration, not the
*cause*.

### PRI Ring Master Re-enumeration Attempt

Post-replay, attempted to re-establish PRI routing through direct register
manipulation on `0000:02:00.0`:

1. **RINGMASTER_CMD = 0x4 (ENUMERATE)**: Command consumed (register cleared
   back to 0) but PRI ring interrupt `0x003f0100` persisted. No routing change.
2. **Ring interrupt clear (write 0x120058)**: Ineffective — interrupt is
   structural, not transient.
3. **RINGMASTER_START0 = 1**: Already set; re-write had no effect.
4. **FECS cpuctl start (bit 1)**: cpuctl changed from `0x10` to `0x12`
   (start bit accepted), but FECS PC still returns PRI fault `0xbadf5040`.
   The Falcon accepted the start command but can't execute because its
   PRI ring connection is broken.

**Conclusion**: PRI ring routing stations are destroyed by the vfio-pci
rebind and cannot be reconstructed via register writes or ring master commands
after the fact. The PGRAPH/FECS/GPCCS PRI stations are created during the
nvidia driver's `nv_gpu_ops_create_session()` sequence, which programs the
ring master to establish routing entries. These entries live in hardware
state that is not accessible through BAR0 MMIO — they are internal to the
PRI ring master's routing fabric.

### Tier 2 Frontier Assessment

The Tier 2 wall is at the **PRI ring routing fabric level**, which is below
BAR0-accessible registers. Possible approaches to break through:

1. **Pre-swap PRI ring capture**: Read ring master routing tables while the
   catalyst driver is still active (before unbind), looking for internal
   registers that define station routes.
2. **PGRAPH reset + FECS cold boot**: Instead of preserving warm state,
   reset PGRAPH and boot FECS from scratch using captured firmware blob.
   This would rebuild PRI stations fresh.
3. **PRAMIN firmware state**: The 66K PRAMIN writes include firmware contexts.
   If the GPCCS/FECS firmware state in PRAMIN can be restored *before* PRI
   ring enumeration, the firmware might create stations on restart.
4. **GPC power gate cycling**: Toggling GPC power gates (`PGRAPH_PGOB`)
   might trigger PRI station re-creation if the ring master firmware is
   properly configured.

### PRI Ring Reset Attempts

Exhaustive BAR0 manipulation on `0000:02:00.0` post-replay:

| Attempt | Registers Written | Result |
|---------|------------------|--------|
| RINGMASTER_CMD = 0x1 (RESET) | 0x12004c | CMD consumed, PRI_RING_INTR unchanged |
| RINGMASTER_CMD = 0x2 (START) | 0x12004c | PRI_RING_INTR cleared to 0x0, but FECS/GPCCS PCs still fault |
| RINGMASTER_CMD = 0x3 (RESET+START) | 0x12004c | Same — interrupt cleared, faults persist |
| PGRAPH bit toggle in PMC_ENABLE | 0x200 | PGRAPH re-enabled, no PRI change |
| Full PMC disable/enable cycle | 0x200 | PRI_RING_INTR cleared, GPCCS **degraded** from `0xbadf5040` to `0xbad00100` (PRI no-ack = engine deregistered) |
| FECS cpuctl start (bit 1) | 0x409100 | cpuctl accepted (0x10→0x12), PC still faulted |

**Definitive conclusion**: PRI ring routing stations are hardware-internal
state below the BAR0 MMIO layer. They are programmed by the PRI ring master
microcode during NVIDIA driver initialization and destroyed when the driver
unbinds. No combination of BAR0 register writes, ring master commands, or
PMC resets can re-establish them after the fact.

### Exp 219 Final Status

**Tier 1 (WarmInfrastructure): ✅ VALIDATED** — catalyst pipeline proven
(26s), 83K alive registers captured, domain-scoped scan operational,
3-layer preservation working, SBR bridge reset recovery codified.

**Tier 2 (WarmCompute): BLOCKED** — PRI ring routing wall confirmed at
three independent levels:
1. Exp 217: BAR0-only path closed (firmware-dependent TPC stations)
2. Exp 219 register replay: 83K writes applied, PRI stations absent
3. Exp 219 PRI ring master manipulation: hardware state below BAR0

The Tier 2 wall is **not** at the register level, the firmware level, or
the power gating level — it is at the **PRI ring routing fabric level**,
which is internal GPU interconnect hardware that BAR0 MMIO cannot access.

### FECS Cold Boot Attempt

Loaded `/lib/firmware/nvidia/gv100/gr/fecs_inst.bin` (25,632 bytes, 6408 words)
into FECS IMEM and `fecs_data.bin` (4,788 bytes) into DMEM via Falcon memory
ports (0x409180-0x4091c4). Set boot vector to 0, triggered start via cpuctl.

| Register | Before | After Start | Notes |
|----------|--------|-------------|-------|
| FECS cpuctl | 0x00000010 | 0x00000012 | Start bit accepted |
| FECS mailbox0 | 0x00000000 | 0x00000000 | Firmware didn't respond |
| FECS mailbox1 | 0x00000000 | 0x00000000 | No handshake |
| FECS pc | 0xbadf5040 | 0xbadf5040 | PRI fault persists |
| FECS os | 0x00000000 | 0x00000000 | No OS context |
| PRI_RING_INTR | 0x00000000 | 0x00000000 | Ring clean |

**Result**: cpuctl accepted start bit but Falcon core did not execute. This
is because **Volta Falcon engines are HS (Hardware Secured)** — FECS and GPCCS
require the ACR (Authentication Code Recovery) secure boot chain to
authenticate firmware before execution. Direct IMEM write + cpuctl start is
blocked by the hardware security model (Falcon SPE/WPR enforcement).

The secure boot chain on Volta:
1. **PMU/SEC2** loads ACR bootloader (`acr/ucode_load.bin`)
2. ACR authenticates and loads FECS firmware
3. FECS creates PRI ring stations during initialization
4. FECS loads and authenticates GPCCS firmware
5. GPCCS creates TPC PRI stations

Each step requires the previous step's hardware security context. This is
NVIDIA's deliberate security architecture to prevent unauthorized firmware
execution.

### Tier 2 Path Forward

All BAR0-level approaches exhausted. The Tier 2 wall is the ACR secure boot
chain. Remaining viable paths:

1. **ACR sovereign boot**: Implement the full ACR→FECS→GPCCS boot sequence
   through BAR0 MMIO, using the signed firmware blobs in
   `/lib/firmware/nvidia/gv100/`. This is what nouveau does in its `gr_init`
   path. (Exp 221: Sovereign ACR Boot)
2. **nouveau GR init injection**: Let nouveau complete its `gr_init` (which
   boots FECS/GPCCS via ACR), then swap to vfio-pci *without* destroying
   PRI stations. Requires patching nouveau's unbind to skip GR teardown.
   (Exp 222: Selective Nouveau Teardown)
3. **Catalyst with no unbind**: Keep nvsov loaded, use ioctl-based compute
   through the NVIDIA driver instead of raw MMIO. (Exp 223: Catalyst
   Compute via Driver)
4. **GPCCS firmware extraction**: Extract the authenticated GPCCS firmware
   from a running catalyst instance (capture IMEM/DMEM contents while
   Falcon is running, before ACR wipes them on halt). Use this for
   sovereign replay. (Exp 224: Firmware Extraction)

### Artifacts Persisted

| Artifact | Path |
|----------|------|
| Cold snapshot | `/var/lib/toadstool/catalysts/products/gv100_cold_bar0.json` |
| Warm snapshot | `/var/lib/toadstool/catalysts/products/gv100_catalyst_bar0.json` |
| Delta JSON | `/var/lib/toadstool/catalysts/products/gv100_catalyst_delta.json` |
| Diff replay | `/var/lib/toadstool/catalysts/products/gv100_catalyst_replay.json` |
| Golden state (repo) | `infra/golden_state/gv100_catalyst_replay.json` |

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| RM init still blocked by remaining NOPs | **Resolved**: selective un-NOP allows full RM init |
| TPC state doesn't survive warm swap | **Confirmed**: Tier 1 achieved; firmware-mediated PRI ring wall |
| BAR0 capture misses internal GPU state | 83,623 alive regs captured across 22 domains |
| Replayed state depends on firmware shadows | **Confirmed**: register replay alone insufficient for Tier 2 |
| Zombie modules from failed attempts | SBR bridge reset recovers without reboot |
| Linear BAR0 scan too slow | **Resolved**: domain-scoped capture (897ms vs 462s) |
| Register replay doesn't create PRI stations | **Identified**: GPCCS firmware execution required for TPC routing |
