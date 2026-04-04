# Experiment 059: coralReef GPU Power Management System Design

**Date**: March 14, 2026
**Hardware**: Titan V (GV100, SM70) — no PMU firmware, no GSP
**Software**: coralReef Phase 10, `coral-driver` crate
**Status**: Design — glow plug proven, full system sketched
**Chat**: [Glow plug to sovereign PMU](28732f32-750e-4053-a1ae-a8d39a738d7a)

---

## Context

Experiment 058 proved that the GPU actively manages its own power state:
- PCIe runtime PM puts the device in D3hot when no driver holds a reference
- The GPU's internal power controller gates clock domains within seconds
  of driver unbind, even when PCIe stays in D0
- Writing `PMC_ENABLE` from Rust re-enables all engines in microseconds
- Cold start → self-warm gives cleaner PFIFO state than inheriting nouveau

**Key fact**: Desktop Volta (GV100/Titan V) has **no PMU firmware**.
NVIDIA doesn't ship signed PMU firmware for this chip. Nouveau uses
`gm200_pmu_nofw()` — a stub that skips firmware entirely. All power
management is done via direct BAR0 register writes.

This means coralReef's power management is pure MMIO — no falcon
microcontroller to program, no firmware to load, no signing to deal with.

---

## Power State Model

Five states, from hottest to coldest:

| State | PMC_ENABLE | PFIFO | PCIe | Est. Power | Wake Latency |
|-------|------------|-------|------|------------|--------------|
| **Sovereign** | 0x5fecdff1 | Channels loaded | D0 | ~25W idle | < 1ms |
| **Warm** | 0x5fecdff1 | Enabled, no channels | D0 | ~20W | ~5ms |
| **Glow** | 0x40000020 | Gated (0xBAD0DA00) | D0 | ~10W | ~50ms |
| **Sleep** | 0x40000020 | Gated | D3hot | ~3W | ~100ms |
| **Off** | — | — | Powered off | 0W | Seconds |

### State transitions

```
Off ──boot──→ Sleep ──D0──→ Glow ──PMC──→ Warm ──channels──→ Sovereign
                ←──D3hot──   ←──gate──     ←──teardown──     ←──idle──
```

The current glow plug implements: `Glow → Warm` (PMC_ENABLE write) and
`Sleep → Glow` (D3hot → D0 via PCI config space).

### Standard driver comparison

| State | nvidia driver | nouveau | coralReef |
|-------|---------------|---------|-----------|
| Idle, display | Sovereign | Sovereign | N/A (headless) |
| Idle, no work | Warm + SLCG | Warm | Glow or Warm (configurable) |
| Suspended | Sleep | Sleep | Sleep (when allowed) |
| Compute burst | Sovereign | Sovereign | Sovereign |

Standard drivers sit at **Warm + SLCG** (sub-unit level clock gating)
at idle — engines are clocked but individual sub-units gate themselves
when idle, reducing power while keeping wake latency under 1ms.

coralReef can go lower: **Glow** state gates entire engine domains,
using ~10W less than Warm, with 50ms wake cost. Standard drivers never
go this low while the GPU is "in use" because display needs instant
response. A headless compute GPU has no such constraint.

---

## Register Map

### Tier 1: Engine Clock Domains (PMC)

| Register | Offset | Function |
|----------|--------|----------|
| NV_PMC_ENABLE | 0x000200 | Master clock domain enable — bit per engine |
| NV_PMC_DEVICE_ENABLE | 0x000204 | PBDMA enables |

PMC_ENABLE readback on GV100: `0x5fecdff1` (all supported engines).
Writing `0xFFFFFFFF` lets hardware mask to supported bits.

Key bits in PMC_ENABLE (GV100):
- Bit 0: PMC (always on)
- Bit 4: PTIMER
- Bit 8: PFIFO
- Bit 12: PGRAPH
- Bits 16+: Various engines (CE, NVDEC, etc.)

For fine-grained control, we can enable subsets:
- **Compute only**: PFIFO + PGRAPH + CE = bits 8, 12, 16+
- **Minimal**: Just PMC + PTIMER = bits 0, 4 (for register access)

### Tier 2: Sub-Unit Level Clock Gating (SLCG)

| Register | Offset | Function |
|----------|--------|----------|
| NV_PBUS_EXT_CG | 0x001C00 | Bus engine clock gating control |
| NV_PBUS_EXT_CG1 | 0x001C04 | Sub-unit level gating enables |

**NV_PBUS_EXT_CG (0x1C00)** fields:
- `IDLE_CG_DLY_CNT[5:0]`: Cycles before idle gating kicks in (0=immediate, 63=delayed)
- `IDLE_CG_EN[6]`: Enable idle clock gating
- `STALL_CG_EN[14]`: Enable stall clock gating
- `WAKEUP_DLY_CNT[19:16]`: Cycles to wake from gated state

**NV_PBUS_EXT_CG1 (0x1C04)** fields:
- `MONITOR_CG_EN[0]`: Monitor clock gating
- `SLCG_BL[1]` through `SLCG_PM[9]`: Individual sub-unit gating

These registers allow microsecond-granularity power management WITHIN
an engine domain that's already clocked via PMC_ENABLE.

### Tier 3: PCI Power State

| Register | Access | Function |
|----------|--------|----------|
| PCI PM CTRL | Config 0x64 | D0/D3hot power state |
| sysfs power/control | Filesystem | Runtime PM: "on" or "auto" |
| sysfs power_state | Filesystem | Current power state readback |

---

## Architecture: `coral-power`

```
coral-power (Rust module in coral-driver)
├── PowerProfile          Enum: Sovereign, Warm, Glow, Sleep, Off, Custom
├── PowerManager          Stateful manager, holds BAR0 reference
│   ├── set_profile()     Transition to named profile
│   ├── current_state()   Read PMC/PCI and determine state
│   ├── warm()            Glow → Warm (PMC_ENABLE write)
│   ├── sovereign()       Warm → Sovereign (channel setup)
│   ├── cool()            Warm → Glow (PMC gate selective engines)
│   ├── sleep()           Glow → Sleep (D3hot transition)
│   └── wake()            Sleep → Glow (D0 transition)
├── ClockGateConfig       Fine-grained SLCG configuration
│   ├── idle_delay         0..63 cycles before gating
│   ├── idle_gate_en       Auto-gate on idle
│   ├── stall_gate_en      Gate during stalls
│   ├── wakeup_delay       Cycles to wake
│   └── slcg_mask          Per-subunit gating enables
├── PowerPolicy           High-level policy enum
│   ├── OnDemand           Sleep when idle, warm on dispatch (edge/batch)
│   ├── AlwaysWarm         Keep engines clocked, SLCG for sub-units (cloud)
│   ├── AlwaysSovereign    Full power, channels loaded, instant dispatch
│   ├── Eco                Aggressive gating, minimum idle power
│   └── Custom(config)     User-defined register values
└── PowerMonitor          Telemetry (optional, for nvpmu integration)
    ├── power_draw_uw()    Estimated power from register heuristics
    ├── temperature()      GPU temp via 0x020460
    └── engine_activity()  Per-engine busy/idle from STATUS registers
```

### Integration with existing code

**`init_pfifo_engine()`** in `channel.rs` already does the Glow → Warm
transition. This becomes `PowerManager::warm()`.

**`GpuPowerController`** in toadStool's `nvpmu/src/power.rs` handles
D0/D3hot via sysfs. This becomes `PowerManager::wake()` / `sleep()`.

**The glow plug** in `diagnostic_matrix()` is the prototype for
`PowerManager::warm()` with `PowerPolicy::OnDemand`.

### Policy behavior

```
OnDemand:
  idle → cool() after 5s → sleep() after 60s
  dispatch request → wake() if sleeping → warm() → sovereign()
  dispatch complete → cool() after 5s

AlwaysWarm:
  startup → warm() + SLCG configured
  idle → SLCG auto-gates sub-units (hardware-managed, <1ms wake)
  dispatch → sovereign() (<1ms, just channel setup)
  never sleeps

AlwaysSovereign:
  startup → sovereign() (channels pre-loaded)
  idle → SLCG only (sub-unit level)
  dispatch → immediate (<1μs doorbell)
  never cools

Eco:
  idle → sleep() after 10s
  dispatch → full wake cycle (~150ms)
  SLCG aggressive (idle_delay=0)
  between dispatches → glow (PMC gated)
```

---

## Energy Comparison (Estimated)

For a Titan V at idle (no compute work):

| Configuration | Idle Power | Wake to Dispatch |
|---------------|-----------|-----------------|
| nvidia driver (display) | ~25W | Instant |
| nouveau (display) | ~25W | Instant |
| **coralReef AlwaysWarm** | ~15-20W | < 5ms |
| **coralReef OnDemand** | ~3-10W | 50-150ms |
| **coralReef Eco** | ~3W | 100-200ms |
| No driver (D3hot) | ~3W | Must bind driver |

The key insight: a headless compute GPU does not need instant response.
A 50ms wake is invisible for batch workloads. For latency-sensitive
inference serving, AlwaysWarm adds < 5ms and saves ~5-10W vs full power.

---

## Implementation Plan

### Phase 1: Stabilize glow plug (current — done)
- [x] D3hot prevention via `power/control=on`
- [x] Self-warming via `PMC_ENABLE` write in Rust
- [x] Cold start → warm verification in diagnostic matrix
- [x] Scripts updated for dual Titan V

### Phase 2: Power state abstraction
- [ ] Extract `PowerManager` struct from current inline code
- [ ] Implement `current_state()` — read PMC/PCI/PFIFO, classify
- [ ] Implement `warm()` and `cool()` transitions
- [ ] Add `ClockGateConfig` for SLCG registers (0x1C00/0x1C04)
- [ ] Integrate into `VfioChannel::new()` — auto-warm before channel setup

### Phase 3: Policy engine
- [ ] Implement `PowerPolicy` enum with idle timers
- [ ] Background thread for OnDemand policy (idle → cool → sleep)
- [ ] Telemetry: temperature, engine activity, estimated power
- [ ] Integration with toadStool `GpuPowerController` for D0/D3hot

### Phase 4: Multi-GPU orchestration
- [ ] Per-GPU power profiles in a multi-GPU rig
- [ ] Cross-GPU load balancing with power awareness
- [ ] Oracle GPU kept at AlwaysWarm for reference comparison
- [ ] Cloud mode: AlwaysSovereign with health monitoring

### Phase 5: Extended architectures
- [ ] Turing/Ampere: Has GSP, different power management path
- [ ] Blackwell (5060): Likely needs GSP interaction for power
- [ ] AMD (MI50): Separate register map, same policy abstraction

---

## Nouveau / Mesa Design Lessons

### What to learn from nouveau

1. **`nv50_mc_init()` / `gp100_mc_init()`**: Our glow plug already
   replicates this. Write `0xFFFFFFFF` to `PMC_ENABLE`.

2. **PFIFO init sequence**: `gk104_fifo_init()` → clear interrupts,
   configure PBDMAs, enable scheduler. We do this in `init_pfifo_engine()`.

3. **Power gating during idle**: nouveau's `nvkm_timer` and runpm
   (runtime power management) callbacks handle D0/D3hot transitions.
   We replicate this with sysfs `power/control`.

4. **What nouveau does NOT do on Volta**: Load PMU firmware. There is
   none. nouveau's power management on Volta is purely register-based,
   same as what we're building.

### What to learn from Mesa (NVK / gallium)

Mesa assumes the kernel driver handles power and provides a stable
GPU surface. Its relevant patterns:

1. **Lazy initialization**: Don't set up resources until first use.
   Our `OnDemand` policy follows this pattern.

2. **Fence-based completion**: Submit work, return fence, let GPU
   idle between submissions. Don't poll-wait.

3. **Memory placement hints**: Mesa tells the kernel where to place
   buffers based on access patterns. Our DMA allocator can use similar
   heuristics (VRAM for hot data, system for cold).

### What is novel in coralReef

Standard drivers are designed for display + compute on a "managed"
GPU. coralReef's scenario is fundamentally different:

1. **No display**: We can go lower than any display driver at idle
2. **VFIO isolation**: No kernel driver state to coordinate with
3. **On-demand warming**: Standard drivers never fully gate engines
   because display needs them. We can.
4. **Policy-driven**: Standard drivers have one power profile.
   We expose it as a configurable API.

---

## GV100-Specific Register Reference

### Proven working (Experiment 058)

| Register | Offset | Write | Effect |
|----------|--------|-------|--------|
| PMC_ENABLE | 0x000200 | 0xFFFFFFFF | Clock all engines (reads back 0x5fecdff1) |
| PFIFO_ENABLE | 0x002200 | 0 → 1 | Initialize PFIFO scheduler |
| PFIFO_INTR | 0x002100 | 0xFFFFFFFF | Clear all PFIFO interrupts |

### To explore (Phase 2)

| Register | Offset | Purpose |
|----------|--------|---------|
| NV_PBUS_EXT_CG | 0x001C00 | Idle/stall clock gating config |
| NV_PBUS_EXT_CG1 | 0x001C04 | Sub-unit level gating |
| GPU_TEMP | 0x020460 | Temperature sensor readback |
| PMC_DEVICE_ENABLE | 0x000204 | Per-PBDMA enable bits |
| THERM_STATUS | 0x020070 | Thermal status / throttle |

### Oracle Baseline (captured March 14, 2026)

Probed both GPUs simultaneously — oracle warm on nouveau, VFIO target cold:

| Register | Oracle (nouveau, warm) | VFIO (cold) | Notes |
|----------|----------------------|-------------|-------|
| PMC_ENABLE | 0x5fecdff1 | 0x40000020 | Engine domains |
| PMC_DEV_ENABLE | 0x00003fff | 0x00003fff | **Same** — persists across states |
| PBUS_EXT_CG | 0x00000000 | 0x00000000 | **Same** — nouveau doesn't use bus CG! |
| PBUS_EXT_CG1 | 0x000003fe | 0x000003fe | **Same** — all 9 SLCG active |
| PFIFO_ENABLE | 0x00000000 | 0xbad0da00 | Gated when PMC cold |
| GPU_TEMP | 0x20002eb8 (~46°C) | 0x20002608 (~38°C) | Cold GPU runs cooler |

**Key finding**: Nouveau does NOT enable bus-level idle clock gating
(`IDLE_CG_EN=0`, `STALL_CG_EN=0`). It relies entirely on SLCG (sub-unit
level) which is a hardware default — all 9 sub-units gate themselves
independently. This means:

1. **SLCG is free** — it's always on, requires no software management,
   and gates individual sub-units at microsecond granularity when idle.
2. **Bus-level CG is untapped** — we can go lower than nouveau by enabling
   `IDLE_CG_EN` with a short delay. Standard drivers don't because display
   needs zero-latency wake. Headless compute doesn't.
3. **PMC_ENABLE is the only coarse knob** — switching from 0x5fecdff1 to
   0x40000020 is what the hardware does autonomously. There's nothing in
   between that nouveau manages.

This simplifies our architecture: the real power savings come from
**choosing when to hold PMC_ENABLE high** (our policy engine), not
from complex register programming. The hardware already handles
fine-grained gating at the sub-unit level.

### Temperature decoding

GV100 thermal register at 0x020460: bits [15:8] give approximate °C.

| GPU | Raw | Decoded | State |
|-----|-----|---------|-------|
| Oracle | 0x20002EB8 | ~46°C | Warm, engines clocked, idle |
| VFIO | 0x20002608 | ~38°C | Cold, engines gated |

Delta: ~8°C between warm-idle and cold — this is the actual thermal
cost of keeping engines clocked with no work. Minimal.

---

*The glow plug was the first spark. The full system is a power plant.*
