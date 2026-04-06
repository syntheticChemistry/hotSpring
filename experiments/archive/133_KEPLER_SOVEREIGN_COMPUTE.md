# Experiment 133: Kepler Sovereign Compute Dispatch

**Date**: 2026-03-30
**Status**: Implementation complete, hardware validation pending
**Depends on**: Exp 128 (K80 cold boot + FECS), Exp 129 (K80 VFIO dispatch harness)

## Motivation

Exp 129 implemented a K80 VFIO dispatch test harness but used **Volta-shaped**
compute class methods and QMD layout. The push buffer emitted `SEND_PCAS_A` at
offset `0x0D00` and built QMD v2.1 (`clc3c0qmd.h`) — both wrong for Kepler
class `0xA1C0` (`KEPLER_COMPUTE_B`). Submitting this to hardware would send
methods to non-existent register offsets and produce a QMD the SKED cannot parse.

This experiment implements the correct Kepler compute dispatch path by adding:
1. A Kepler-specific QMD builder using the `cla1c0qmd.h` v1.7 field layout
2. Kepler-specific push buffer methods from `cla1c0.h`
3. Architecture-aware dispatch branching in the driver

## Key Findings from NVIDIA Open Headers

### Method Offset Differences (cla1c0.h vs clc3c0.h)

| Method | Volta (clc3c0) | Kepler (cla1c0) |
|--------|---------------|-----------------|
| INVALIDATE_SHADER_CACHES | 0x0088 | 0x021C |
| SET_SHADER_LOCAL_MEMORY_WINDOW | 0x077C/0x0780 (A/B pair) | 0x077C (single 32-bit) |
| SEND_PCAS_A | 0x0D00 (raw addr upper 32) | 0x02B4 (QMD addr >> 8) |
| SEND_SIGNALING_PCAS_B | 0x0D04 (raw addr lower 32) | 0x02BC (flags: SCHEDULE bit) |
| SET_PROGRAM_REGION_A/B | N/A (absolute addr in QMD) | 0x1608/0x160C |

Critical semantic difference: Volta `SEND_PCAS_A/B` takes the raw 64-bit QMD
address split into two 32-bit halves. Kepler `SEND_PCAS_A` takes the QMD address
**right-shifted by 8** (requiring 256-byte alignment), and `SEND_SIGNALING_PCAS_B`
takes flag bits (INVALIDATE, SCHEDULE) rather than an address.

### QMD Layout Differences (cla1c0qmd.h QMDV01_07 vs clc3c0qmd.h v2.1)

| Field | Volta v2.1 bits | Kepler v1.7 bits |
|-------|----------------|-----------------|
| QMD_VERSION | 4:7 | 576:579 |
| QMD_MAJOR_VERSION | 0:3 | 580:583 |
| CTA_RASTER_WIDTH | 224:255 | 384:415 |
| CTA_THREAD_DIM0 | 544:559 | 592:607 |
| PROGRAM addr | 832:895 (64-bit VA) | 256:287 (32-bit offset) |
| REGISTER_COUNT | 608:615 | 1496:1503 |
| BARRIER_COUNT | 592:596 | 1467:1471 |
| SHARED_MEM_SIZE | 640:657 | 544:561 |
| CBUF(i) base | bit 1536+i*64 | bit 928+i*64 |

Kepler QMD v1.7: `QMD_MAJOR_VERSION=1`, `QMD_VERSION=7`.
The `PROGRAM_OFFSET` field is a 32-bit offset relative to the program region base
set via `SET_PROGRAM_REGION_A/B` in the push buffer (not an absolute GPU VA).

## Changes

### 1. `coral-driver/src/nv/qmd.rs`

- Added `build_qmd_kepler(params, program_offset)` implementing QMDV01_07 layout
- Updated `build_qmd_for_sm`: SM 0-37 routes to `build_qmd_kepler`, SM 38-69
  to `build_qmd_v21` (Maxwell/Pascal), SM 70-79 to `build_qmd_v22`, SM 80+ to
  `build_qmd_v30`
- 15 new unit tests covering Kepler QMD field positions: version, grid dims,
  thread dims, program offset, register count, barrier count, shared memory,
  CBUF valid bits + addresses, cache invalidation, API call limit, SM routing

### 2. `coral-driver/src/nv/pushbuf.rs`

- Added `kepler_method` module with constants from `cla1c0.h`:
  `INVALIDATE_SHADER_CACHES` (0x021C), `SEND_PCAS_A` (0x02B4),
  `SEND_SIGNALING_PCAS_B` (0x02BC), `SET_PROGRAM_REGION_A/B` (0x1608/0x160C),
  `SET_SHADER_LOCAL_MEMORY_WINDOW` (0x077C)
- Added `PushBuf::compute_dispatch_kepler()` implementing the full Kepler launch
  sequence: SET_OBJECT, INVALIDATE, SET_PROGRAM_REGION, LOCAL_MEM_WINDOW,
  SEND_PCAS_A (shifted), SEND_SIGNALING_PCAS_B (SCHEDULE flag)
- 5 new unit tests verifying push buffer structure, method offsets, PCAS
  shifting, program region encoding, local memory window

### 3. `coral-driver/src/nv/ioctl/mod.rs`

- Added `NVIF_CLASS_KEPLER_COMPUTE_B = 0xA1C0` constant

### 4. `coral-driver/src/nv/pushbuf.rs` (class module)

- Re-exported `KEPLER_COMPUTE_B` alongside existing Volta/Turing/Ampere classes

### 5. `coral-driver/src/nv/vfio_compute/dispatch.rs`

- Extracted `build_dispatch_pushbuf()` helper to eliminate QMD/pushbuf
  duplication between `dispatch_inner` and `dispatch_inner_traced`
- SM <= 37 branch uses `PushBuf::compute_dispatch_kepler()` with shader IOVA
  as the program region base; SM >= 50 uses existing Volta path

### 6. Test Updates

- `exp129_kepler_vfio_dispatch.rs`: Added Phase 2b — compiler-driven dispatch
  using `coral_reef::compile_wgsl_full` with `NvArch::Sm35` target, validating
  the full pipeline from WGSL source through Kepler SASS codegen, QMD v1.7,
  and the new Kepler push buffer methods
- `dispatch.rs`: Updated all three SM match blocks to include `35..=37` routing
  to `NvArch::Sm35` for correct Kepler compilation

## Sequence Diagram

```
coral-reef                     qmd.rs                  pushbuf.rs              GPU (K80)
    |                            |                        |                      |
    | compile_wgsl(Sm35)         |                        |                      |
    |--------------------------->|                        |                      |
    |    Kepler SASS binary      |                        |                      |
    |<---------------------------|                        |                      |
    |                            |                        |                      |
    | build_qmd_kepler(v1.7)     |                        |                      |
    |   PROGRAM_OFFSET=0         |                        |                      |
    |   CTA dims @ bits 384+     |                        |                      |
    |   REG_COUNT @ bits 1496+   |                        |                      |
    |--------------------------->|                        |                      |
    |    256-byte QMD            |                        |                      |
    |<---------------------------|                        |                      |
    |                            |                        |                      |
    |                            | compute_dispatch_kepler|                      |
    |                            | SET_OBJECT(0xA1C0)     |                      |
    |                            | INVALIDATE(0x021C)     |                      |
    |                            | SET_PROGRAM_REGION_A/B |                      |
    |                            | LOCAL_MEM_WINDOW       |                      |
    |                            | SEND_PCAS_A(addr>>8)   |                      |
    |                            | SIGNALING_PCAS_B(SCHED)|                      |
    |                            |----------------------->|                      |
    |                            |                        | GPFIFO submit        |
    |                            |                        |--------------------->|
    |                            |                        |                      | SKED parses
    |                            |                        |                      | QMD v1.7
    |                            |                        |   GP_GET advance     | SM executes
    |                            |                        |<---------------------| Kepler SASS
```

## Test Matrix

| Test | GPU | Status |
|------|-----|--------|
| `exp129_phase1_kepler_open` | K80 | Pending HW |
| `exp129_phase2_kepler_nop_dispatch` | K80 | Pending HW (hand-crafted SASS) |
| `exp129_phase2b_kepler_compiled_dispatch` | K80 | Pending HW (coral-reef Sm35) |
| `exp129_phase3_kepler_data_compute` | K80 | Pending HW (DMA round-trip) |
| `vfio_dispatch_nop_shader` (with K80) | K80 | Pending HW |
| QMD v1.7 unit tests (13 tests) | N/A | **PASSING** |
| Kepler pushbuf unit tests (5 tests) | N/A | **PASSING** |
| Full coral-driver lib tests (381) | N/A | **PASSING** |

## Next Steps

1. **Hardware validation**: Run `exp129_phase2` and `exp129_phase2b` on physical
   K80 to validate that the Kepler SKED correctly parses QMD v1.7 and executes
   the dispatched shader.

2. **FECS prerequisite**: K80 dispatch requires FECS firmware booted via the
   cold boot path (`k80_cold_boot::cold_boot()`). If FECS is not running,
   GPFIFO submissions will fence-timeout.

3. **Data-producing compute**: Once NOP dispatch succeeds, Phase 3 can be
   extended with a real Kepler SASS shader that writes to a buffer, validating
   end-to-end sovereign compute on K80.

4. **Kepler cold boot integration**: Wire `k80_cold_boot::cold_boot()` into
   the `coralctl` workflow so FECS/GPCCS boot is automated before dispatch.
