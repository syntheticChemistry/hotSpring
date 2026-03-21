# hotSpring VFIO PFIFO Channel — GP_PUT Last Mile Handoff

**Source**: hotSpring on biomeGate (Titan V GV100, SM70, vfio-pci)
**Date**: March 9, 2026
**coralReef file**: `crates/coral-driver/src/vfio/channel.rs`
**Test**: `cargo test --test hw_nv_vfio --features vfio -- --ignored vfio_dispatch_nop_shader`
**Status**: 6/7 VFIO tests pass. Dispatch blocked on PBDMA not reading USERD GP_PUT.

---

## Executive Summary

The PFIFO channel initialization has been iterated from "GPU ignores everything"
to "PBDMA loads our channel context and reads our GPFIFO base address." The channel
is enabled, fault-free, and the PBDMA has fetched the correct GPFIFO pointer. The
remaining issue is that the PBDMA's internal GP_PUT register stays at 0 — it hasn't
read our GP_PUT=1 from the USERD buffer. This is a single-register encoding issue
in the runlist channel entry, not an architectural problem.

---

## What Was Fixed (7 bugs total, each confirmed via BAR0 diagnostics)

### 1. PBDMA_MAP Interpretation
- **Bug**: Assumed PBDMA0 existed; `PBDMA_MAP=0x0020000e` means PBDMAs 1,2,3,21
- **Fix**: Enumerate set bits correctly
- **Impact**: Stopped probing non-existent PBDMA0, started targeting correct engines

### 2. PBDMA_RUNL_MAP Sequential Indexing
- **Bug**: Used hardware PBDMA ID as register index; GV100 uses sequential indexing
- **Fix**: `0x2390 + seq_index * 4` where seq_index is 0..pbdma_count
- **Impact**: Correct PBDMA-to-runlist mapping (PBDMA1,2→RL1, PBDMA3→RL2, PBDMA21→RL4)

### 3. GK104 Fixed Runlist Submission Registers
- **Bug**: Used per-runlist registers; GV100 reuses GK104's single register pair
- **Fix**: `0x2270 = (target<<28)|(addr>>12)`, `0x2274 = (runlist_id<<20)|count`
- **Impact**: Runlist accepted by scheduler; channel transitions to enabled state

### 4. Empty Runlist Flush
- **Bug**: PFIFO_ENABLE toggle did not clear stale Nouveau PBDMA contexts
- **Fix**: Submit 0-entry runlists for all discovered runlists after PFIFO init
- **Source**: `gv100_fifo_init()` in nouveau
- **Impact**: Clean PCCSR state (0x00000000) before our channel creation

### 5. ENGN0_STATUS for GR Runlist Discovery
- **Bug**: TOP_INFO chained parsing unreliable for GV100's format
- **Fix**: Read `ENGN0_STATUS` (0x2640), bits [15:12] = GR engine's runlist ID
- **Impact**: Reliably discovers GR runlist = 1 on Titan V

### 6. PCCSR Fault Clearing Sequence
- **Bug**: PBDMA_FAULTED (bit 24) persisted from prior driver, not clearable while enabled
- **Fix**: Disable channel → W1C fault bits → re-enable. Call after bind, before enable.
- **Impact**: Channel starts fault-free

### 7. GP_BASE_HI Aperture Target (Latest Fix)
- **Bug**: RAMFC offset 0x04C (GP_BASE_HI → PBDMA GP_STATE) had bits [29:28]=0 (VRAM)
- **Fix**: Set bits [29:28]=1 (SYS_MEM_COHERENT, PBDMA encoding)
- **Code**: `(gpfifo_iova >> 32) as u32 | (limit2 << 16) | (PBDMA_TARGET_SYS_MEM_COHERENT << 28)`
- **Impact**: PBDMA now reads GPFIFO from system memory instead of VRAM

---

## Current Diagnostic State (Post All Fixes)

```
PCCSR_CHAN post-runlist: 0x00000003  PBDMA_FAULT=0    ← clean, enabled
GP_BASE_HI in instance: 0x10070000                    ← aperture=1 (COH), limit=7 (256 entries)
PBDMA2 GP_FETCH:        0x00001000                    ← our GPFIFO_IOVA
PBDMA2 GP_STATE:        0x10070000                    ← matches our GP_BASE_HI
PBDMA2 GP_PUT:          0x00000000                    ← PROBLEM: should be 1
PBDMA2 USERD:           0x00000000_0x00000000         ← PROBLEM: should show 0x2000
GP_GET from USERD:      0                             ← GPU never advanced
GP_PUT from USERD:      1                             ← We wrote this correctly
```

The PBDMA loaded our GPFIFO configuration but did **not** load our USERD pointer.
Without USERD, the PBDMA cannot read GP_PUT to know there's work to do.

---

## Root Cause Analysis: Why USERD Is Not Loaded

The PBDMA gets the USERD address from **two places**:
1. The RAMFC in the instance block (offset 0x008 = USERD_LO, 0x00C = USERD_HI)
2. The **runlist channel entry** DW0 (USERD address + target)

Looking at the runlist channel entry format (`gv100_runl_insert_chan`):

```
DW0: USERD_ADDR[31:9] | USERD_TARGET[3:2] | RUNQUEUE[1] | TYPE[0]=0
DW1: USERD_ADDR[63:32]
DW2: INST_ADDR[31:12] | INST_TARGET[5:4] | CHID[11:0]
DW3: INST_ADDR[63:32]
```

**Our current DW0**: `0x00002000` — USERD_ADDR=0x2000, TARGET=0 (VRAM), RUNQ=0

**The bug**: `USERD_TARGET[3:2]` is 0, which means VRAM. The USERD buffer is in
system memory. The PBDMA tries to read USERD from VRAM address 0x2000, finds nothing,
and therefore never sees our GP_PUT write.

### Fix Required

In `populate_runlist()`, DW0 should be:
```rust
// DW0: USERD_ADDR | USERD_TARGET(SYS_MEM_COH=2) | RUNQ | TYPE=0
write_u32_le(rl, 0x10, userd_iova as u32 | (TARGET_SYS_MEM_COHERENT << 2) | (runq << 1));
```

And DW2 should include INST_TARGET:
```rust
// DW2: INST_ADDR | INST_TARGET(SYS_MEM_NCOH=3) | CHID
write_u32_le(rl, 0x18, INSTANCE_IOVA as u32 | (3 << 4) | self.channel_id);
```

**Note**: The runlist entry uses **PCCSR-style** target encoding (2=COH, 3=NCOH),
NOT PBDMA-style (1=COH, 2=NCOH). This is the same encoding distinction that
confused GP_BASE_HI vs PCCSR_INST.

---

## Target Encoding Reference (Critical for Next Agent)

There are TWO different target encoding schemes in NVIDIA hardware:

### PCCSR/RAMIN/Runlist Encoding
| Value | Meaning |
|-------|---------|
| 0 | VID_MEM (VRAM) |
| 2 | SYS_MEM_COHERENT |
| 3 | SYS_MEM_NONCOHERENT |

Used by: `PCCSR_INST`, `RAMIN_PAGE_DIR_BASE`, runlist entry DW0/DW2, PDE/PTE

### PBDMA/RAMFC Encoding
| Value | Meaning |
|-------|---------|
| 0 | VID_MEM (VRAM) |
| 1 | SYS_MEM_COHERENT |
| 2 | SYS_MEM_NONCOHERENT |

Used by: `RAMFC_USERD_LO` bits [1:0], `RAMFC_GP_BASE_HI` bits [29:28]

The code already defines both:
```rust
const TARGET_SYS_MEM_COHERENT: u32 = 2;      // PCCSR/RAMIN/runlist
const PBDMA_TARGET_SYS_MEM_COHERENT: u32 = 1; // PBDMA/RAMFC
```

---

## Possible Additional Issue: Doorbell

After fixing the USERD target in the runlist, if GP_PUT still doesn't update:

The doorbell register (`NV_USERMODE_NOTIFY_CHANNEL_PENDING` at BAR0+0x810090)
needs to be written with the channel ID after updating GP_PUT in USERD. This
notifies the host scheduler to re-read USERD for the channel.

Current code in `vfio_compute.rs` does write the doorbell. But `DOORBELL_PROBE`
reads as `0x00000000` — this may mean the USERMODE block isn't enabled or mapped.
If doorbell isn't working, the PBDMA won't know to check for new GPFIFO entries.

---

## Files to Modify

1. **`coralReef/crates/coral-driver/src/vfio/channel.rs`**
   - `populate_runlist()` — add USERD_TARGET and INST_TARGET to runlist channel entry
   - Verify DW0/DW2 bit positions match `gv100_runl_insert_chan()` from nouveau

2. **`coralReef/crates/coral-driver/src/nv/vfio_compute.rs`**
   - Verify doorbell write happens AFTER GP_PUT write to USERD
   - Consider adding a memory fence between USERD write and doorbell

---

## How to Test

```bash
# Rebind: nouveau warm → vfio-pci
GPU=0000:4b:00.0; AUDIO=0000:4b:00.1
echo "" > /sys/bus/pci/devices/$GPU/reset_method
echo $GPU > /sys/bus/pci/drivers/nouveau/bind; sleep 3
echo $GPU > /sys/bus/pci/drivers/nouveau/unbind
echo "vfio-pci" > /sys/bus/pci/devices/$GPU/driver_override
echo $GPU > /sys/bus/pci/drivers/vfio-pci/bind
echo "" > /sys/bus/pci/devices/$GPU/reset_method

# Run the test (requires root for VFIO)
CORALREEF_VFIO_BDF=0000:4b:00.0 CORALREEF_VFIO_SM=70 \
  cargo test --test hw_nv_vfio --features vfio -- --ignored --test-threads=1 --nocapture vfio_dispatch_nop_shader
```

Key diagnostics to watch:
- `PBDMA2 USERD:` — should show `0x00002000` (not zero) after fix
- `PBDMA2 GP_PUT:` — should show 1 (not zero) after fix
- `GP_GET from USERD:` — should show ≥ 1 when dispatch succeeds

---

## Architecture Summary

```
hotSpring (physics) → barraCuda (WGSL shaders, DF64) → coralReef (WGSL→SASS, dispatch) → toadStool (GPU mgmt)
                                                             │
                                                    channel.rs ← THIS FILE
                                                             │
                                                    ┌────────┴────────┐
                                                    │  GPU PFIFO      │
                                                    │  ┌──────────┐   │
                                                    │  │ PBDMA 2  │──→│ GPFIFO (IOVA 0x1000)
                                                    │  │ runlist 1│   │ USERD  (IOVA 0x2000)
                                                    │  └──────────┘   │ INST   (IOVA 0x3000)
                                                    └─────────────────┘
```

**Confidence**: The fix (USERD_TARGET in runlist DW0) should resolve the GP_PUT
issue. After that, the PBDMA will read GP_PUT=1 from USERD, fetch one GPFIFO entry
from 0x1000, and process the push buffer. If the push buffer's NOP shader is
correctly formatted, the test passes — 7/7.

---

## Session-Wide sudo

For iterative testing, instead of per-command `pkexec`, run the test loop from
a root shell: `sudo -s` then execute the rebind+test commands. toadStool should
eventually implement `polkit` rules or a capability-based approach for VFIO
operations (CAP_SYS_RAWIO + VFIO group permissions) to avoid sudo entirely.

---

## Supersedes

- `CORALREEF_ITER43_PFIFO_HW_VALIDATION_RESULTS_MAR13_2026.md` (GP_BASE_HI found but not yet fixed)
- `HOTSPRING_CUDA_ORACLE_EVOLUTIONARY_DEBUGGING_HANDOFF_MAR13_2026.md` (oracle strategy complete)
- `SOVEREIGN_COMPUTE_TRIO_WIRING_GAPS_HANDOFF_MAR12_2026.md` (gaps 1-4 resolved)

*Validated on biomeGate, March 9, 2026.*
