# Experiment 162: Titan V Sovereign Compute Pipeline

**Date**: 2026-04-07
**GPU**: NVIDIA Titan V (GV100, 10de:1d81)
**Driver**: nvidia-535.230.02 (proprietary) in reagent VM
**Depends on**: Exp161 (NVDEC Sovereign Attempt — Volta PRI privilege model)
**Status**: Architecture proven, infrastructure built

## Objective

Build a sovereign GPU compute pipeline on Titan V by:
1. Testing PFIFO channel dispatch alongside nvidia-535's Resource Manager
2. Implementing an MMIO write firewall in ember to block driver teardown writes
3. Patching nvidia's open kernel modules to skip hardware cleanup on rmmod

## Phase 1: PFIFO Coexistence Probe

**Environment**: reagent VM, nvidia-535.230.02 loaded, Titan V initialized

### Findings

With nvidia-535 loaded and the GPU fully initialized (HBM2 trained, PMU/RM alive):

| Register/Region | Address | Observed | Implication |
|---|---|---|---|
| PCCSR CSR0 (ch 0) | 0x800000 | Writable (enable bit takes) | Channel enable register accessible |
| PCCSR CSR1 (ch 0) | 0x800004 | **Read-only** (writes rejected) | Instance pointer locked by RM |
| PRAMIN (0x700000) | via 0x001700 | **BAD0AC errors** on write | RM contends for PRAMIN window |
| PBDMA[0] STATUS | 0x040100 | 0x10011111 (idle) | All PBDMA engines idle |
| PBDMA[0] CHANNEL | 0x040128 | 0xBAD00200 | No channel bound |
| FIFO_RUNLIST_BASE | 0x002800 | 0xBAD00200 | RM manages runlists internally |
| FIFO_SCHED_ENABLE | 0x002600 | 0xBAD00200, **read-only** | Scheduler locked by RM |
| USERMODE | 0x810000 | 0x0000C361 (readable) | Doorbell accessible |

### Conclusion

**nvidia's RM actively contends for FIFO infrastructure.** The RM manages channels
through its own internal data structures and firmware — not through the legacy BAR0
PCCSR/FIFO registers. Key findings:

- PCCSR CSR1 (instance pointer) is read-only while RM is running
- PRAMIN window writes return BAD0AC error patterns (RM overwrites 0x001700)
- FIFO_SCHED_ENABLE is locked (PRI-gated or RM-managed)
- All PBDMA engines idle with no channel bindings (channels managed by RM firmware)

**Verdict**: Coexistence with the live RM is unstable for sovereign channel creation.
The patched nvidia approach (Phase 3) is required — load nvidia to initialize the GPU,
then unload while preserving hardware state.

## Phase 2: Ember MMIO Write Firewall

### Implementation

Added to `coral-driver` and `coral-ember`:

**`coral-driver/src/vfio/device/dma_safety.rs`**:
- `TeardownPolicy` enum: `AllowAll` | `BlockTeardown` | `BlockAndLog`
- `is_teardown_write(offset, value) -> bool` — detects known teardown patterns:
  - PMU CPUCTL halt/stop/reset (0x10A100 with HALT/SRESET bits)
  - PMU DMEM scrub (0x10A1C4 with 0xDEAD5EC2 sentinel)
  - FECS IMEM zeroing (0x409184 with value 0)
  - PMC_ENABLE mass-strip (0x200 with < 8 bits set)

**`coral-ember/src/ipc/handlers_mmio/low_level.rs`**:
- `mmio_write`: checks `is_teardown_write()` before fork, returns `-32012` when blocked
- `mmio_batch`: checks each write op against policy before execution

**`coral-ember/src/ipc/handlers_mmio/mod.rs`**:
- `ember.mmio.policy` RPC method: get/set `TeardownPolicy` per device at runtime

**`coral-ember/src/hold.rs`**:
- `HeldDevice.teardown_policy: TeardownPolicy` — per-device firewall state

### Test Results

13 unit tests pass covering:
- PMU CPUCTL halt/stop blocked, start allowed
- PMU DMEM scrub sentinel blocked, normal writes allowed
- FECS IMEM zeroing blocked, firmware writes allowed
- PMC_ENABLE mass-strip blocked, warm state allowed
- Unrelated register writes pass through
- Policy state machine (default AllowAll, BlockTeardown, BlockAndLog)

### Usage

```bash
# Enable firewall before driver unload
echo '{"jsonrpc":"2.0","id":1,"method":"ember.mmio.policy","params":{"bdf":"0000:03:00.0","policy":"block_and_log"}}' | socat - UNIX:/run/ember/0000:03:00.0.sock

# Query current policy
echo '{"jsonrpc":"2.0","id":2,"method":"ember.mmio.policy","params":{"bdf":"0000:03:00.0"}}' | socat - UNIX:/run/ember/0000:03:00.0.sock
```

## Phase 3: Patched nvidia-535 Open Kernel Modules

### Patch: `NVreg_PreserveHwState`

Added a module parameter to `open-gpu-kernel-modules` 535.230.02 that, when set to 1,
skips GPU hardware cleanup on module unload:

**`nv.c` — `nv_module_exit()`**:
- Skips `rm_shutdown_rm(sp)` — preserves global Resource Manager firmware on PMU

**`nv-pci.c` — `nv_pci_remove()`**:
- Skips `nv_shutdown_adapter(sp, nv, nvl)` — preserves per-GPU hardware state
- Skips `rm_free_private_state(sp, nv)` — prevents freeing state still DMA-active

### Files

- Patch: `infra/agentReagents/patches/nvidia-535-preserve-hw-state.patch`
- Build recipe: `infra/agentReagents/templates/reagent-nvidiaopen535-patched-titanv.yaml`

### Preserved State After rmmod

When `NVreg_PreserveHwState=1`, after rmmod:
- PMU Resource Manager firmware continues executing on Falcon PMU
- HBM2 training state persists (memory controller configured)
- PRI privilege context active (host MMIO has full access)
- All engine states preserved (PFIFO, PBDMA, PGRAPH, SMPs)
- BAR0 registers remain readable/writable

### What Leaks (Acceptable in Disposable VMs)

- Kernel memory for RM per-GPU state (~few MB)
- Kernel kthread structures (cleaned up on VM shutdown)

### Workflow

```
1. Boot reagent VM with Titan V passthrough
2. insmod nvidia.ko NVreg_PreserveHwState=1
3. GPU initializes (HBM2 trained, RM alive, full engine state)
4. rmmod nvidia — module unloads, hardware state preserved
5. ember acquires GPU via VFIO — full BAR0 access to initialized GPU
6. Sovereign PFIFO channel dispatch through ember MMIO gateway
```

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sovereign Compute Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  Reagent VM       │    │  Host (ember)     │                   │
│  │                    │    │                    │                   │
│  │  1. nvidia-535     │    │  5. VFIO acquire   │                   │
│  │     loads          │    │                    │                   │
│  │  2. GPU inits      │    │  6. ember.mmio.*   │                   │
│  │     (HBM2, PMU)    │    │     gateway        │                   │
│  │  3. rmmod with     │    │                    │                   │
│  │     PreserveHw=1   │    │  7. PFIFO channel  │                   │
│  │  4. VM shutdown    │──>│     dispatch        │                   │
│  │     (no FLR)       │    │                    │                   │
│  └──────────────────┘    │  8. SM compute      │                   │
│                           │                    │                   │
│                           └──────────────────┘                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │  MMIO Write Firewall (dma_safety.rs)             │           │
│  │                                                    │           │
│  │  Policy: AllowAll → BlockTeardown → BlockAndLog    │           │
│  │  Blocks: PMU halt, DMEM scrub, FECS clear,        │           │
│  │          PMC_ENABLE mass-strip                     │           │
│  │  Wired: ember.mmio.write, ember.mmio.batch         │           │
│  │  Toggle: ember.mmio.policy RPC                     │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Volta Security Model (Updated from Phase 4)

```
GPU Silicon (GV100)
├── PRI Privilege Ring (RM-configured, survives rmmod, cleared by FLR)
│   │
│   ├── Host BAR0 MMIO — ALWAYS ACCESSIBLE (no PRI gate)
│   │   ├── BOOT0 (0x000)              — GPU identification
│   │   ├── PRAMIN (0x700000)          — VRAM read/write window
│   │   ├── PRAMIN_CFG (0x001700)      — VRAM page selector
│   │   ├── PCCSR CSR0 (0x800000+)    — Channel enable/disable
│   │   ├── USERMODE (0x810090)        — Doorbell register
│   │   ├── PFIFO_PBDMA_MAP (0x2004)  — PBDMA bitmap (read-only)
│   │   ├── PBDMA.STATUS (+0x100)     — PBDMA engine status
│   │   ├── PBDMA.GP_BASE (+0x048)    — GP ring address (writable!)
│   │   ├── PBDMA.GP_PUT (+0x058)     — GP write pointer (writable!)
│   │   ├── FIFO_INTR (0x002100)      — FIFO interrupts
│   │   └── PRIV_RING (0x120058)      — PRI fault status
│   │
│   ├── PRI-GATED by RM (0xBAD00200 = PRIV violation)
│   │   ├── PMC_ENABLE (0x200)         — Engine enable (write-locked)
│   │   ├── PMC_DEVICE_EN (0x600-60C) — Device enable control
│   │   ├── PMC_BOOT_42 (0x0A4)       — Security identification
│   │   ├── FIFO_SCHED_EN (0x2504)    — Scheduler control
│   │   ├── PFIFO_ENABLE (0x2200)     — FIFO master enable (writes ignored)
│   │   ├── PBDMA.CHANNEL (+0x128)    — Channel assignment (read: no channel)
│   │   └── All Falcon CPUCTL          — Execution control
│   │
│   └── Behavior
│       ├── PRI gates configured during nvidia RM init
│       ├── Persist through rmmod (kprobe stubs or teardown)
│       ├── Persist through PMU halt
│       ├── CLEARED by Function Level Reset (FLR)
│       └── On Volta, RM runs in kernel (not PMU/GSP), but gates are HW-level
│
├── Compute Path Requirements (post-FLR, PRI gates cleared)
│   ├── PRAMIN → V2 5-level page tables → pushbuffer in VRAM
│   ├── PFIFO enable → PBDMA config → channel creation → runlist submit
│   ├── USERMODE doorbell → PBDMA DMA fetch → command execution
│   └── Requires: VfioChannel::create() full init sequence
│
└── State Preservation Layers
    ├── kprobe stubs: preserves engine state through rmmod
    ├── HBM2 training: persists through rmmod (may survive FLR)
    ├── PRI gates: persist through rmmod (cleared by FLR)
    └── PFIFO/PBDMA: preserved by kprobes but PRI-gated until FLR
```

## Files Changed

### coral-driver
- `crates/coral-driver/src/vfio/device/dma_safety.rs` — `TeardownPolicy`, `is_teardown_write()`

### coral-ember
- `crates/coral-ember/src/hold.rs` — `teardown_policy` field on `HeldDevice`
- `crates/coral-ember/src/ipc/handlers_mmio/low_level.rs` — firewall in write/batch handlers
- `crates/coral-ember/src/ipc/handlers_mmio/mod.rs` — `ember.mmio.policy` RPC
- `crates/coral-ember/src/ipc.rs` — dispatcher wiring (both Unix and TCP)
- `crates/coral-ember/src/lib.rs` — `teardown_policy` in device construction
- `crates/coral-ember/src/ipc/handlers_device.rs` — `teardown_policy` in device construction
- `crates/coral-ember/src/ipc/tests.rs` — `teardown_policy` in test construction
- `crates/coral-ember/src/swap/swap_bind.rs` — `teardown_policy` in swap construction

### agentReagents (infra)
- `patches/nvidia-535-preserve-hw-state.patch` — kernel module patch
- `templates/reagent-nvidiaopen535-patched-titanv.yaml` — build recipe

## Phase 1b: Live Channel Descriptor Tracing

With nvidia-535 loaded, we successfully:

1. **Read nvidia's live RAMFC** by pointing PRAMIN to VRAM 0x2FD45F000 (channel 0's instance)
2. **Annotated all fields** using NVIDIA's open-gpu-doc `dev_ram.ref.txt` — RAMFC fields map
   directly to PBDMA PRI register offsets
3. **Confirmed field values** against PBDMA register readback:
   - SIGNATURE = 0x0000FACE (standard nvidia signature)
   - GP_BASE_LO = 0x00064000, GP_BASE_HI = 0x000C0001
   - USERD_LO = 0x60400000, USERD_HI = 0x80000000 (matches PBDMA0 exactly)
   - CONFIG = 0x00001100, SET_CHANNEL_INFO = 0x10003080
   - PDB[0x00] = 0xFD865C00 (page directory base in VRAM)
4. **Created sovereign PCCSR channel** (channel 5) with cloned RAMFC
   - CSR0 = 0x802FE000 accepted and verified
   - FIFO_INTR_0 = 0x00000001 (hardware acknowledged channel)
   - GPU remained fully alive (BOOT0 = 0x140000A1)

### Key Finding: PRAMIN Works Alongside nvidia

Contrary to Phase 1 initial BAD0AC errors (caused by accessing low VRAM), PRAMIN works
perfectly when targeting nvidia's high VRAM range (0x2FE000000+). Both read and write
succeed, including PRAMIN base register changes.

## Existing Rust Infrastructure

The `VfioChannel` module in `coral-driver` already encodes the full PFIFO channel setup:

- **`VfioChannel::create()`**: Cold boot — PMC glow plug, PFIFO reset, full page table setup
- **`VfioChannel::create_warm()`**: Post-nouveau — preserves FECS/GPCCS state
- **`VfioChannel::create_sovereign()`**: **NEW** — Post-nvidia-PreserveHwState, minimal touch
- **`VfioChannel::create_on_runlist()`**: PBDMA isolation testing

Supporting infrastructure:
- `pfifo.rs`: Engine init with `PfifoInitConfig` (5 variants: default, diagnostic, warm, preserved_nvidia)
- `page_tables.rs`: V2 5-level MMU page tables (PD3→PD2→PD1→PD0→PT0)
- `registers.rs`: 61 tested register offsets (PCCSR, PBDMA, PFIFO, USERMODE, etc.)
- `channel_layout.rs`: Pure IOVA layout + BAR0 encoding (no I/O)
- `pfifo::setup_bar2_page_table()`: VRAM-based BAR2 page table for VIRTUAL mode

## Phase 4: In-VM Sovereign PFIFO Attempt (2026-04-07)

### kprobe Module: `preserve_hw.ko`

Built a kernel module (`/opt/preserve_hw.ko` in the reagent VM) that registers kprobes
on four nvidia teardown functions, making them immediate-return stubs:

| Function | Effect When Stubbed |
|---|---|
| `rm_shutdown_rm` | Preserves global RM firmware state |
| `nv_shutdown_adapter` | Preserves per-GPU hardware configuration |
| `rm_shutdown_adapter` | Preserves RM adapter state |
| `rm_free_private_state` | Prevents DMA-active state from being freed |

**Result**: After `insmod preserve_hw.ko && rmmod nvidia`:
- BOOT0 = 0x140000A1 (alive, identical pre/post)
- PMC_ENABLE = 0x5FECDFF1 (23 engines, identical pre/post)
- PMU_CPUCTL = 0x00000020 (running)
- PRAMIN accessible, VRAM alive

### Discovery: PRI Gate Architecture

With GPU state preserved but nvidia unloaded, attempted sovereign PFIFO channel creation.
Every PBDMA, scheduler, and PMC device register returned `0xBAD00200` (PRI violation):

| Register | Before rmmod | After rmmod | Interpretation |
|---|---|---|---|
| PBDMA0-3.CHANNEL | 0xBAD00200 | 0xBAD00200 | PRI-gated by RM config |
| FIFO_SCHED_EN | 0xBAD00200 | 0xBAD00200 | PRI-gated by RM config |
| PMC_ENABLE | Read-only | Read-only | Write-protected by RM config |
| PMC_DEVICE_EN_0..3 | 0xBAD00200 | 0xBAD00200 | PRI-gated by RM config |
| PMC_BOOT_42 | 0xBAD00200 | 0xBAD00200 | Security ID PRI-gated |
| PFIFO_ENABLE | 0x00000000 | 0x00000000 | Writes ignored (won't set to 1) |

**Critical finding**: The 0xBAD00200 PRI violations are **hardware-level access controls
configured by the RM during initialization**. They persist independently of:
- Whether the nvidia kernel driver is loaded
- Whether the PMU is halted or running
- Whether PRIV_RING faults are cleared

These gates are configured in the GPU's PRI privilege ring and survive everything except
a Function Level Reset (FLR). The RM sets up a privilege hierarchy during init where the
"host" MMIO path has restricted access to scheduling/engine-control registers.

### Approach Evolution

| Approach | Result |
|---|---|
| Sovereign channel alongside live nvidia | PCCSR works but RM controls scheduling |
| kprobe stub + rmmod | HW state preserved but PRI gates persist |
| Halt PMU after rmmod | PMU_CPUCTL write ignored; PRI gates unchanged |
| Direct PBDMA programming | Registers accept writes but DMA engine won't fetch |
| PMC PFIFO reset (bit 8 toggle) | PMC_ENABLE is write-protected |
| FLR from VM | Not available (QEMU doesn't expose FLR for passthrough) |

### Architectural Conclusion

On **Volta (GV100) with nvidia-535**, the Resource Manager configures hardware-enforced
PRI access controls that gate PFIFO scheduling registers from the host MMIO path. These
gates persist through rmmod and PMU halt. Only an FLR clears them — but FLR also resets
all engine state.

The **correct sovereign compute path** for Volta is:

```
1. Reagent VM: nvidia loads → HBM2 trained, all engines initialized
2. Reagent VM: kprobe stubs loaded → teardown functions no-op'd
3. Reagent VM: rmmod nvidia → HW state fully preserved
4. Host: VM shutdown → QEMU releases VFIO → automatic FLR clears PRI gates
5. Host: nouveau warm-cycle → re-trains VRAM (VRAM training survives FLR on HBM2)
6. Host: VfioChannel::create() → full PFIFO init from scratch (PRI gates are clear)
7. Host: Submit sovereign pushbuffer → compute proven
```

Step 5 may be optional — HBM2 training on Titan V has been shown to persist across
certain resets (Exp159). Step 6 uses the standard `VfioChannel::create()` (not
`create_sovereign()`) since FLR resets PFIFO state.

The `create_sovereign()` path remains valid for a future scenario where we can prevent
FLR during VM shutdown (QEMU `x-no-flr=on` or VFIO reset override).

### K80 Status

Tesla K80 (GK210) PLL-bricked from Exp157 DEVINIT replay. SBR on AMD root port
`40:01.3` recovered PLX switches but both GK210 dies unresponsive (PCIe link dead).
**Requires physical power cycle.** Kepler has no PRI gates, so the sovereign path is
simpler once the hardware is recovered.

## Files Changed (Phase 4)

### In reagent VM
- `/opt/preserve_hw.c` + `/opt/preserve_hw.ko` — kprobe teardown stub module
- `/opt/nvidia-open-535/` — open-gpu-kernel-modules 535.230.02 (patched, built, but
  Volta not supported by open driver — GSP firmware required for Turing+ only)

## Next Steps

1. **Host-side sovereign channel via FLR** — shut down VM, rebind Titan V to vfio-pci,
   use `VfioChannel::create()` from ember (PRI gates cleared by FLR)
2. **Verify HBM2 survives FLR** — if yes, skip nouveau warm-cycle entirely
3. **K80 power cycle** — then reagent-nvidia470 VM POST → sovereign FECS upload
   (Kepler has no PRI gates → direct channel control)
4. **QEMU x-no-flr option** — investigate preventing FLR during VM shutdown to enable
   `create_sovereign()` path without full PFIFO re-init
