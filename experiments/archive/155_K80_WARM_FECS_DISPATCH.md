# Experiment 155: K80 Warm-Cycle FECS Dispatch

**Status:** Active  
**Date:** 2026-04-07  
**Target:** Tesla K80 (GK210) — BDF 0000:41:00.0  
**Prerequisite:** nouveau driver available for warm cycle

## Background

Tesla K80 (Kepler GK210) has no ACR barrier — unlike Volta, Kepler does not
require signed firmware for FECS. The only blocker for sovereign compute on K80
is VRAM: when bound to vfio-pci from cold boot, VRAM is never trained (BIOS/driver
never ran), so PGRAPH and FECS regions fault.

Experiment 144 proved that on Titan V, a nouveau warm cycle trains VRAM and
makes PRAMIN accessible. Experiment 151 stated this is "expected for K80 but
needs test." This experiment validates that path.

## Hypothesis

Nouveau warm cycle on K80 trains VRAM, enabling FECS upload and sovereign
compute dispatch without any proprietary driver.

## Pipeline (all through ember IPC)

1. `glowplug.device_swap(k80_bdf, "nouveau", trace=true)` — warm cycle
2. Wait for nouveau to POST and train VRAM
3. `glowplug.device_swap(k80_bdf, "vfio-pci", trace=true)` — back to VFIO
4. `ember.mmio_read(k80_bdf, PRAMIN_WINDOW)` — verify VRAM alive
5. `ember.pramin_read` — bulk VRAM read to confirm accessibility
6. `ember.falcon.upload_imem` — FECS PIO firmware upload
7. `ember.falcon.start_cpu` — FECS start
8. `ember.falcon.poll` — FECS ready poll
9. `ember.fecs.state` — query FECS falcon status registers
10. `ember.cleanup_dma` — decontaminate

## Key Registers

| Register | Offset | Role |
|----------|--------|------|
| FECS_CPUCTL | 0x409100 | FECS CPU control |
| FECS_BOOTVEC | 0x409104 | Boot vector |
| FECS_PC | 0x409110 | Program counter |
| FECS_SCTL | 0x409240 | Security control |
| FECS_MAILBOX0 | 0x409040 | Firmware mailbox |
| PGRAPH_STATUS | 0x400700 | PGRAPH engine status |
| PRAMIN | 0x700000 | PRAMIN window (VRAM liveness indicator) |

## Expected Behavior

1. **Pre-warm:** PRAMIN reads return `0xBAD0AC0x` or `0xBADF3000` (VRAM dead)
2. **Post-warm:** PRAMIN reads return valid data (not dead patterns)
3. **FECS upload:** IMEM PIO succeeds (ok=true)
4. **FECS start:** PC advances from 0 (BL runs)
5. **FECS poll:** Falcon settles (halt or mailbox change)
6. If FECS runs: first sovereign compute on K80 without proprietary drivers

## VRAM Dead Patterns

| Pattern | Meaning |
|---------|---------|
| 0xBAD0AC00 | PRAMIN degraded (no VRAM training) |
| 0xBAD0AC01 | PRAMIN degraded variant |
| 0xBADF3000 | FBPA/LTC uninitialized |

## Binary

```text
cargo run --release --bin exp155_k80_warm_fecs -- --bdf 0000:41:00.0
```

## Reagent Comparison

Run `reagent-nouveau-k80.yaml` via agentReagents to capture what nouveau does
during warm boot, then compare with exp156_reagent_compare.

## Files

| File | Role |
|------|------|
| `barracuda/src/bin/exp155_k80_warm_fecs.rs` | Experiment binary |
| `barracuda/src/fleet_client.rs` | EmberClient (MMIO, falcon, warm_cycle) |
| `barracuda/src/glowplug_client.rs` | GlowplugClient (device.swap) |
| `coral-ember/src/ipc/handlers_mmio/fecs_state.rs` | ember.fecs.state handler |

## Results

*To be filled after live execution on Tesla K80.*
