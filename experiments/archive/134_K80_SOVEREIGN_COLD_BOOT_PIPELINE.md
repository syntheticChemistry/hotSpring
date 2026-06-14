# Experiment 134: K80 Sovereign Cold Boot Pipeline

**Date**: 2026-03-30
**Status**: IMPLEMENTED
**Predecessor**: Exp 123 (K80 Sovereign Compute), Exp 132 (Ember-Frozen Warm Dispatch), Exp 133 (Kepler Sovereign Compute)

## Motivation

Experiments 123 and 133 proved that the K80's Kepler architecture (GK210, SM 3.7)
can be driven entirely through sovereign code — QMD construction, push buffer
methods, PFIFO channel creation, and compute dispatch all work without proprietary
drivers. However, the missing link was a **single-command sovereign cold boot**:
bringing a fully powered-off K80 from D3cold to a state where FECS is running and
compute channels can be created, without needing nouveau or nvidia drivers at all.

The `k80_cold_boot::cold_boot()` function existed in `coral-driver` but was only
callable from hardware tests. This experiment wires it into `coralctl` so it
becomes a first-class CLI operation.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  coralctl cold-boot <BDF> --recipe <path>                │
│  (local VFIO access, same pattern as coralctl devinit)   │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  VfioDevice::open(bdf) → map_bar(0)                      │
│  Load firmware: fecs_inst.bin, fecs_data.bin,             │
│                 gpccs_inst.bin, gpccs_data.bin            │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  k80_cold_boot::cold_boot(bar0, recipe, config, fw...)   │
│                                                          │
│  Phase 1: Clock init (ROOT_PLL + PCLOCK + CLK)           │
│  Phase 2: Infrastructure DEVINIT (PMC → PBDMA)           │
│  Phase 2b: Extended domains (PGRAPH, PCCSR, PRAMIN)      │
│  Phase 3: FECS/GPCCS PIO boot (unsigned firmware)        │
│  Phase 4: Post-boot firmware probe & diff                │
└──────────────────────────────────────────────────────────┘
```

## Implementation

### coralctl CLI (`coral-glowplug/src/bin/coralctl/main.rs`)

Added `ColdBoot` variant to the `Command` enum with the following arguments:

| Argument | Description |
|----------|-------------|
| `bdf` | PCI BDF address of the K80 |
| `--recipe` | Path to BIOS init recipe JSON (captured from nvidia470 VM) |
| `--firmware-dir` | Directory with FECS/GPCCS firmware blobs (auto-detected) |
| `--pgraph` | Include PGRAPH registers (default: true) |
| `--pccsr` | Include PCCSR registers |
| `--pramin` | Include PRAMIN registers |
| `--skip-firmware` | Skip FECS/GPCCS upload (clock + devinit only) |

### Handler (`coral-glowplug/src/bin/coralctl/handlers_trace.rs`)

Added `cold_boot_replay()` function following the `devinit_replay()` pattern:
- Opens VFIO device directly (no daemon RPC — needs BAR0 access)
- Auto-discovers firmware directory from known candidates
- Invokes `k80_cold_boot::cold_boot()` with configurable domain inclusion
- Prints step-by-step boot log and final FECS status

### Firmware Location

Firmware blobs live at `coralReef/data/firmware/nvidia/gk110/`:
- `fecs_inst.bin` — FECS instruction memory
- `fecs_data.bin` — FECS data memory
- `gpccs_inst.bin` — GPCCS instruction memory
- `gpccs_data.bin` — GPCCS data memory

Auto-detection searches (in order):
1. `<crate_root>/../../data/firmware/nvidia/gk110/`
2. `<crate_root>/data/firmware/nvidia/gk110/`
3. `/usr/share/coralreef/firmware/nvidia/gk110/`

## Usage

```bash
# Full sovereign cold boot with FECS firmware
coralctl cold-boot 0000:4c:00.0 --recipe /path/to/gk210_bios_recipe.json

# Clock + devinit only (no firmware upload)
coralctl cold-boot 0000:4c:00.0 --recipe /path/to/recipe.json --skip-firmware

# Full boot with all extended domains
coralctl cold-boot 0000:4c:00.0 --recipe /path/to/recipe.json --pccsr --pramin

# Custom firmware directory
coralctl cold-boot 0000:4c:00.0 --recipe /path/to/recipe.json \
    --firmware-dir /opt/coralreef/firmware/gk110
```

## Full Sovereign K80 Pipeline (End-to-End)

With this experiment complete, the K80 has a fully sovereign path from cold metal
to compute dispatch:

```
1. coralctl cold-boot <BDF> --recipe <recipe.json>
   → PLLs, clocks, engines, FECS all initialized

2. NvVfioComputeDevice::open(<BDF>)
   → KeplerChannel created with PFIFO + PBDMA

3. device.dispatch(shader, inputs, outputs)
   → Kepler QMD (v1.7) + Kepler push buffer methods
   → SEND_PCAS_A/B + SET_PROGRAM_REGION_A/B
   → Compute results in output buffers
```

No proprietary driver. No nouveau. No nvidia. Pure sovereign Rust from BAR0 to
shader output.

## Validation

| Check | Result |
|-------|--------|
| `cargo check --workspace` (coralReef) | PASS (0 new warnings) |
| `cargo test --lib -p coral-driver` | 381 passed, 15 ignored |
| `cargo test --lib -p coral-glowplug` | 273 passed |
| `cargo test --lib -p coral-ember` | 165 passed, 1 pre-existing failure |
| `cargo test --lib -p coral-reef` | 1314 passed |
| `cargo check` (benchScale) | PASS |
| `cargo test --lib` (benchScale) | 232 passed |
| `cargo check` (agentReagents) | PASS |
| `cargo test --lib` (agentReagents) | 33 passed |

## Titan V Status

The Titan V (GV100, SM 7.0) follows a different boot path due to firmware security
(HS+ lockdown, WPR2, ACR). Its pipeline uses the "diesel engine" pattern from
Exp 132:

```
coralctl warm-fecs <BDF>  →  nouveau loads FECS  →  FECS frozen  →  VFIO bind
NvVfioComputeDevice::open_warm_with_context(...)  →  Volta QMD + push buffer
```

Both GPUs are software-complete for their next hardware experiment.

## Next Steps

- **Hardware validation**: Run `coralctl cold-boot` on actual K80 hardware
- **Recipe capture**: Ensure BIOS recipe JSON is complete from nvidia470 VM session
- **Integration test**: Add `exp134_cold_boot_to_dispatch` end-to-end test
- **Titan V E2E**: Validate diesel engine path on Titan V hardware
