# Experiment 154: SEC2/ACR Pipeline — PMU-First Hypothesis

**Status:** Active  
**Date:** 2026-04-07  
**Target:** Titan V (GV100) — BDF 0000:03:00.0  
**Prerequisite:** Warm GPU via nouveau cycle (exp 144/145 proven path)

## Background

Experiments 144–145 established that on a warm Titan V (nouveau-cycled VRAM alive),
the SEC2 bootloader (BL) executes via IMEM PIO upload and STARTCPU, but does not
reach HS (Heavy Secure) mode. The BL runs (TRACEPC shows instruction path), but
terminates with `SCTL=0x3000`, `HS=false`, `PC=0x03fb`, `MB0=0xcafebeef`.

Experiment 151 documented four hypotheses for the HS failure:

- **A — Missing crypto context:** PMU establishes WPR2 and the crypto keystore.
  Without PMU, BL signature verification may fail silently.
- **B — DMA descriptor mismatch:** BL descriptor uses virtual addresses but
  instance block page tables may not map the ACR firmware region correctly.
- **C — BROM state not initialized:** BROM registers return `0xbadf5040`;
  nouveau may set these before STARTCPU.
- **D — Missing DMEM initialization:** BL expects more DMEM state than we provide.

## Hypothesis Under Test

**A — PMU-first:** Boot the PMU falcon before SEC2 to establish WPR2 + crypto context.

## Resolution

This experiment routes the **full SEC2/ACR boot pipeline through ember IPC**
(first time), testing whether initializing the PMU falcon before SEC2 allows the
BL to reach HS mode.

## Pipeline (all through ember IPC)

1. `ember.prepare_dma(bdf, bus_master=true)` — quiesce + AER mask
2. `ember.sec2.prepare_physical(bdf)` — PMC reset, instance bind, PHYS_VID path
3. PMU-first boot:
   - `ember.falcon.upload_imem` (PMU base 0x10A000)
   - `ember.falcon.start_cpu` (PMU)
   - `ember.falcon.poll` for PMU ready
4. SEC2 ACR boot:
   - `ember.falcon.upload_imem` (SEC2 base 0x840000, secure=true)
   - `ember.falcon.start_cpu` (SEC2)
   - `ember.falcon.poll` — check for HS mode (`SCTL & 0x4000`)
5. DMEM forensics via `ember.mmio.batch` (hypothesis D readback)
6. BROM register comparison (pre/post boot)
7. `ember.cleanup_dma(bdf)` — decontaminate

## Key Registers

| Register | Offset | Role |
|----------|--------|------|
| SEC2_CPUCTL | 0x840100 | CPU control (start, halt) |
| SEC2_SCTL | 0x840240 | Security control (HS bit at 0x4000) |
| SEC2_PC | 0x840110 | Program counter |
| PMU_CPUCTL | 0x10A100 | PMU CPU control |
| PMU_PC | 0x10A110 | PMU program counter |
| BROM_MODSEL | 0x300200 | BROM module select (0xbadf5040 = uninitialized) |

## Expected Behavior

- **If hypothesis A correct:** PMU boot succeeds, SEC2 BL reaches HS mode
  (`SCTL & 0x4000 != 0`), BROM registers populated.
- **If hypothesis A insufficient:** PMU boot may still succeed, but SEC2 HS
  remains unreached. This narrows to hypotheses B/C/D.
- **Either way:** Full register traces through ember IPC establish the pipeline
  for subsequent hypothesis testing.

## Binary

```text
cargo run --release --bin exp154_sec2_acr_pipeline -- --bdf 0000:03:00.0
```

## Files

| File | Role |
|------|------|
| `barracuda/src/bin/exp154_sec2_acr_pipeline.rs` | Experiment binary |
| `barracuda/src/fleet_client.rs` | EmberClient (MMIO, falcon, SEC2, DMA) |
| `barracuda/src/ember_types.rs` | Typed response structs |
| `barracuda/src/register_maps/nv_gv100.rs` | GV100 register map |

## Results

*To be filled after live execution on Titan V.*
