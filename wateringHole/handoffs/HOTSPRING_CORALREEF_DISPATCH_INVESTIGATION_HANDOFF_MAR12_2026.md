# hotSpring × coralReef Dispatch Investigation — March 12, 2026

**From**: hotSpring (hardware testing on Titan V GV100 + RTX 3090 GA102)
**Status**: Four critical bugs FIXED in coralReef, one fundamental blocker IDENTIFIED
**Impact**: Once FECS context init is implemented, compute dispatch should work

---

## Executive Summary

hotSpring performed deep hardware debugging of coralReef's compute dispatch pipeline.
The dispatch test (`nouveau_full_dispatch_cycle`) returns 0 instead of expected 42.

**Root cause**: Multiple layered bugs, all now fixed except one fundamental blocker:

| Bug | Severity | Status | Component |
|-----|----------|--------|-----------|
| QMD field offsets wrong (every field) | Critical | **FIXED** | `coral-driver/src/nv/qmd.rs` |
| CBUF descriptor indirection missing | Critical | **FIXED** | `coral-driver/src/nv/mod.rs` |
| QMD version wrong for Volta (v2.1→v2.2) | High | **FIXED** | `coral-driver/src/nv/qmd.rs` |
| No sync on new UAPI (EXEC path) | High | **FIXED** | `coral-driver/src/nv/mod.rs` + `new_uapi.rs` |
| SM arch mismatch (test vs device) | Medium | **FIXED** | `tests/hw_nv_nouveau.rs` |
| Local mem window 32-bit on Volta | Medium | **FIXED** | `coral-driver/src/nv/mod.rs` |
| `uvm_compute` module not declared | Low | **FIXED** | `coral-driver/src/nv/mod.rs` |
| **FECS GR context not initialized** | **BLOCKER** | **IDENTIFIED** | Missing integration |

---

## Bug 1: QMD Field Offsets (Critical)

### Problem

`build_qmd_v21()` placed every QMD field at the wrong word offset. The NVIDIA
QMD header definitions use **bit positions** within the 256-byte (2048-bit) QMD,
but the code interpreted them as sequential word indices.

### Example — CTA_RASTER_WIDTH

- **Spec**: bits 224..256 → word 7, bits 0-31
- **Code had**: `q[1] = params.grid.x` (word 1 — WRONG)

Every field was wrong: grid dimensions, thread dimensions, register count,
shared memory, program address, CBUF bindings — all at incorrect offsets.

### Fix

Rewrote `build_qmd_v21()` using a `qmd_set_field()` helper that operates on
bit positions directly. All field placements now match the NVIDIA open headers
(`cl_c3c0qmd.h` as mirrored in `coral-reef-stubs/src/nvidia_headers.rs`).

Correct field map:

```
Field                    Bits         Word   Bits-in-word
QMD_MAJOR_VERSION        0..4         0      0-3
QMD_VERSION              4..8         0      4-7
CTA_RASTER_WIDTH         224..256     7      0-31
CTA_RASTER_HEIGHT        256..272     8      0-15
CTA_RASTER_DEPTH         272..288     8      16-31
CTA_THREAD_DIMENSION0    544..560     17     0-15
CTA_THREAD_DIMENSION1    560..576     17     16-31
CTA_THREAD_DIMENSION2    576..592     18     0-15
BARRIER_COUNT            592..597     18     16-20
REGISTER_COUNT           608..616     19     0-7
SHARED_MEMORY_SIZE       640..658     20     0-17
PROGRAM_ADDRESS_LOWER    832..864     26     0-31
PROGRAM_ADDRESS_UPPER    864..896     27     0-31
CBUF_ADDR_LOWER(i)       1536+i*64   48+i*2 0-31
CBUF_ADDR_UPPER(i)       1568+i*64   49+i*2 0-7
CBUF_SIZE_SHIFTED4(i)    1576+i*64   49+i*2 8-24
CBUF_VALID(i)            1593+i*64   49+i*2 bit 25
```

---

## Bug 2: CBUF Descriptor Indirection (Critical)

### Problem

The compiler (`naga_translate/expr.rs`) generates CBUF loads for storage buffers:

```
addr_lo = c[group][binding * 8]       // CBUF slot = group, offset = binding * 8
addr_hi = c[group][binding * 8 + 4]
size    = c[group][binding * 8 + 8]   // for arrayLength
```

The shader expects CBUF 0 to contain a **descriptor** with the storage buffer's
GPU virtual address. But the driver was setting the QMD CBUF address to the
storage buffer's VA directly. The GPU read "constant" data from the output
buffer (zeros), interpreted it as the base address, and stored 42 to VA 0.

### Fix

Allocate a descriptor buffer for each dispatch containing `[addr_lo, addr_hi, size]`
per binding. Set the QMD CBUF address to point to this descriptor buffer.

---

## Bug 3: QMD Version Selection

Volta uses QMD v2.2 (`MAJOR=2, VERSION=2`), not v2.1. Pascal uses v2.1.
Ampere uses v3.0. Added `build_qmd_v22()` and `build_qmd_for_sm()`.

---

## Bug 4: New UAPI Sync (DRM Syncobj)

On kernel 6.17 with the new UAPI, `gem_cpu_prep` does not provide sync for
EXEC-submitted work. Implemented DRM syncobj support:

- `syncobj_create()` / `syncobj_destroy()` / `syncobj_wait()`
- `exec_submit_with_signal()` — passes syncobj as signal in EXEC
- `NvDevice` creates a syncobj at open time (new UAPI only)
- `sync()` waits on the syncobj with 5-second timeout

---

## Bug 5: Other Fixes

- **SM arch mismatch**: `NvDevice::open()` finds RTX 3090 (renderD128) first,
  but test compiled for SM70. Added `sm_version` to `NvDevice`, diagnostic uses
  detected SM for compilation.
- **Local mem window**: Volta needs 64-bit `0xFF00_0000_0000_0000`, not 32-bit
  `0xFF00_0000`. Fixed based on SM version.
- **`uvm_compute` module**: File existed but wasn't declared in `mod.rs`. Added
  `#[cfg(feature = "nvidia-drm")] pub mod uvm_compute;`.

---

## The Remaining Blocker: FECS GR Context (Gap 3)

### Evidence

Kernel log shows on every dispatch attempt:

```
nouveau 0000:21:00.0: fifo:PBDMA0: CTXNOTVALID chid:4
nouveau 0000:21:00.0: fifo:c00000:0004:0004:[...] errored - disabling channel
nouveau 0000:21:00.0: [...]: channel 4 killed!
```

This occurs on **both GPUs** (Titan V on 4b:00.0 and RTX 3090 on 21:00.0).

### What CTXNOTVALID Means

The PBDMA (Push Buffer DMA engine) tried to load the channel's GR context
and found it invalid. Without a valid GR context, the compute engine cannot
execute any work. The pushbuf is processed (syncobj signals), but SEND_PCAS
fails because the compute class has no context.

### What's Needed

The GR context must be initialized before compute dispatch. This requires:

1. **Loading `sw_ctx.bin`** content into the GR context template — currently
   parsed for size only, content discarded in `firmware_parser.rs`.

2. **Submitting FECS init sequence** — `sw_bundle_init.bin` and
   `sw_method_init.bin` contain FECS method entries. `applicator.rs` already
   splits these from BAR0 writes, but there's no FECS channel submission path.

3. **BAR0 pre-init** — `apply_bar0()` exists but is never called in the real
   dispatch path. Needs `RegisterAccess` implementation (toadStool nvPmu
   `Bar0Access`).

4. **Integration into `NvDevice::open_from_drm()`** — Before channel creation
   or immediately after, apply the GR init sequence.

### What Already Exists in coralReef

| Component | Location | Status |
|-----------|----------|--------|
| Firmware parser (sw_ctx, sw_bundle_init, etc.) | `gsp/firmware_parser.rs` | Works — but discards sw_ctx content |
| GR init sequence builder | `gsp/gr_init.rs` | Works — builds from bundle+method init |
| BAR0/FECS splitter | `gsp/applicator.rs` | Works — splits sequence by address space |
| BAR0 applicator | `gsp/applicator.rs` | Works — needs RegisterAccess wiring |
| FECS channel submission | — | **MISSING** |
| sw_ctx.bin loading | — | **MISSING** |
| Integration with NvDevice | — | **MISSING** |

### Suggested Implementation Path

1. In `firmware_parser.rs`: store `sw_ctx.bin` content (not just size)
2. New function: `submit_fecs_init(channel, init_entries)` — push FECS methods
   via a dedicated channel or the compute channel before dispatch
3. In `NvDevice::open_from_drm()`: after channel creation, submit GR init
4. Consider whether BAR0 pre-init is needed (nouveau may handle this in kernel)
5. Test with the diagnostic test — buffer should show 42 once context is valid

---

## Validation Results

After all fixes, coralReef compiles cleanly:

- **155 library tests** pass (0 failures)
- **QMD unit tests** all pass with correct field extraction
- **Syncobj** creates/signals/waits correctly
- **Diagnostic test** confirms buffer unchanged (CTXNOTVALID), not a data issue

---

## Files Changed in coralReef

```
crates/coral-driver/src/nv/qmd.rs           — QMD field layout rewrite + qmd_set_field helper
crates/coral-driver/src/nv/mod.rs            — CBUF descriptor, SM tracking, syncobj, local_mem
crates/coral-driver/src/nv/ioctl/mod.rs      — syncobj exports, uvm_compute declaration
crates/coral-driver/src/nv/ioctl/new_uapi.rs — DRM syncobj create/destroy/wait + exec_with_signal
crates/coral-driver/tests/hw_nv_nouveau.rs   — SM-aware compilation, dispatch diagnostic
```

---

## For toadStool

The `RegisterAccess` bridge (Gap 4, closed in S147) will be needed when
coralReef integrates BAR0 pre-init. The `Bar0Access` → `RegisterAccess`
bridge should be ready for coralReef to consume.

## For barraCuda

Once coralReef resolves CTXNOTVALID, `CoralReefDevice::dispatch()` should
work end-to-end. The `dispatch_binary` path (Gap 1, closed) is correctly
wired — it will activate once the underlying driver can execute compute work.

---

*Generated by hotSpring dispatch investigation, March 12, 2026*
