# Experiment 027: Live Energy & Thermal Tracking

**Status:** PLANNED
**Date:** February 28, 2026
**Depends on:** All production runs (Exp 023, 024, 025, 026)
**License:** AGPL-3.0-only

---

## Motivation

The user observed that the 8⁴ dynamical run "feels cooler" than the
32⁴ quenched run. This is expected — an 8⁴ lattice uses 0.06% of the
RTX 3090's VRAM and ~20% of its SM cores, so the GPU draws far less
power than during the quenched 32⁴ run (which was sustained at 370W /
73°C for 14 hours).

But we have no quantitative record. Every production run should log
energy consumed and heat dissipated, for three reasons:

1. **Cost accounting** — electricity is a real expense. Knowing
   joules/trajectory lets us budget long runs accurately.
2. **Efficiency comparison** — energy/useful-measurement is a better
   metric than wall-time/measurement for comparing pipeline variants.
3. **Waste heat capture** — downstream studies may use compute waste
   heat for biological substrates (thermal mass for fermentation,
   mushroom cultivation, etc.). Quantifying heat output per run type
   enables those designs.

---

## Available Sensors (biomeGate)

### Confirmed available

| Sensor | hwmon | Readable | What it measures |
|--------|-------|----------|-----------------|
| k10temp | hwmon5 | Tctl, Tccd1, Tccd3, Tccd5, Tccd7 | CPU die temperatures |
| nouveau | hwmon7 | temp1 | Titan V GPU temperature |
| nvme × 3 | hwmon1-3 | Composite, Sensor 1 | SSD temperatures |
| acpitz | hwmon0 | temp1 | Ambient / board temp |
| RAPL | powercap | energy_uj (root only) | CPU package energy (µJ counter) |

### Conditionally available

| Sensor | Condition | What it measures |
|--------|-----------|-----------------|
| nvidia-smi | Proprietary driver loaded | RTX 3090 power draw, temp, utilization |
| nouveau power | — | Not exposed (nouveau doesn't report power draw) |

### Not available

| Sensor | Why | Workaround |
|--------|-----|------------|
| RTX 3090 power (via hwmon) | Proprietary driver doesn't expose hwmon | Use nvidia-smi or estimate from temperature delta |
| Wall outlet power | No smart plug instrumented | Future: add Shelly/Tasmota smart plug with MQTT |

---

## Measurement Strategy

### Tier A: Per-trajectory logging (in-binary)

Add to every production binary (production_dynamical_mixed,
production_dynamical_sweep, gpu_physics_proxy):

```rust
struct EnergySnapshot {
    cpu_temp_mc: Option<i64>,    // k10temp Tctl in milli-celsius
    gpu_temp_mc: Option<i64>,    // nvidia-smi or nouveau temp
    rapl_energy_uj: Option<u64>, // RAPL counter (if readable)
    wall_us: u64,                // already logged
    timestamp_unix_ms: u64,      // epoch milliseconds
}
```

Before and after each trajectory, read the sensors. Log the delta:

```json
{
  "beta": 5.69,
  "traj_idx": 42,
  "wall_us": 40123456,
  "cpu_temp_start_c": 49,
  "cpu_temp_end_c": 54,
  "gpu_temp_start_c": 65,
  "gpu_temp_end_c": 72,
  "rapl_delta_uj": 5800000,
  "estimated_gpu_energy_j": 14800
}
```

GPU energy estimate (when nvidia-smi unavailable): use the known TDP
curve. RTX 3090 at 73°C in a well-cooled case draws ~350-370W. At 42°C
idle, ~15-25W. Interpolate linearly between ambient and max temp.

### Tier B: Sidecar monitoring script

A lightweight Rust or shell script that samples all sensors every N
seconds and writes to a separate JSONL file. Runs alongside any
production binary without modifying it.

```bash
# Sample every 5 seconds
cargo run --release --bin energy_monitor -- \
  --interval-ms=5000 \
  --output=results/energy_monitor.jsonl
```

Or as a simple shell script reading sysfs:

```bash
while true; do
  ts=$(date +%s%3N)
  cpu=$(cat /sys/class/hwmon/hwmon5/temp1_input)
  gpu=$(cat /sys/class/hwmon/hwmon7/temp1_input)
  echo "{\"ts\":$ts,\"cpu_mc\":$cpu,\"gpu_mc\":$gpu}"
  sleep 5
done >> results/energy_sidecar.jsonl
```

### Tier C: Smart plug (future)

Add a WiFi smart plug (Shelly Plus Plug S or Tasmota-flashed device)
to the biomeGate power strip. Read wattage via HTTP API. This gives
true wall-outlet power including PSU efficiency losses, fans, NVMe,
RAM — the full system draw.

---

## What We Already Know

### Quenched 32⁴ (Exp 013, 022)

| Metric | Run 1 (native f64) | Run 2 (DF64 + NPU) |
|--------|:------------------:|:-------------------:|
| GPU temp | 73°C sustained | 74°C sustained |
| GPU power (est.) | ~370W | ~354W |
| Wall time | 13.6 h | 14.2 h |
| Electricity cost | $0.58 | $0.61 |
| Energy (est.) | 370W × 13.6h = **5.0 kWh** | 354W × 14.2h = **5.0 kWh** |
| Heat output | ~5.0 kWh = **18.0 MJ** | ~5.0 kWh = **18.0 MJ** |

### Dynamical 8⁴ (Exp 024)

| Metric | Value |
|--------|-------|
| GPU temp | **~42°C** (observed post-run) |
| GPU power (est.) | **~80-120W** (small lattice, low utilization) |
| Wall time | 10.6 h |
| Energy (est.) | ~100W × 10.6h = **1.1 kWh** |
| Heat output | ~1.1 kWh = **3.9 MJ** |

The 8⁴ dynamical run produces roughly **4.5× less heat** than the 32⁴
quenched run, despite similar wall times. The GPU is barely working.

### Why it feels cooler

The 3090 has 10,496 CUDA cores. The 8⁴ lattice (4,096 sites × 3 colors
= 12,288 elements in the Dirac vector) fits in L2 cache (6 MB). The
CG solver's SpMV never touches DRAM. Most shader cores sit idle.

At 32⁴, the lattice (1,048,576 sites × 3 colors = 3.1M elements) fills
~25 MB — well beyond L2. Every CG iteration streams from VRAM. All
10,496 cores are busy. The memory controller runs at full bandwidth.
That's where the power goes.

### Projected energy for scale-up

| Lattice | GPU util | Est. power | Wall/point | Energy/point | Heat/point |
|---------|----------|-----------|------------|-------------|------------|
| 8⁴ | 20% | 100W | 43 min | 0.07 kWh | 0.26 MJ |
| 16⁴ | 50% | 200W | 2-5 h | 0.6 kWh | 2.2 MJ |
| 32⁴ | 80% | 320W | 8-15 h | 3.8 kWh | 13.7 MJ |
| 48⁴ | 95% | 360W | 30-60 h | 16 kWh | 57.6 MJ |

---

## Energy Per Useful Measurement

The real metric isn't energy per trajectory — it's energy per
statistically independent measurement that contributes to the final
physics result. NPU adaptive steering improves this by:

1. Reducing thermalization waste (63% savings in Exp 022)
2. Concentrating measurements in the transition region
3. Predicting rejections (saving wasted trajectories)

| Configuration | Energy/trajectory | Useful fraction | Energy/useful meas |
|---------------|------------------|----------------|-------------------|
| Quenched 32⁴ (no NPU) | 18.6 kJ | 76% | 24.5 kJ |
| Quenched 32⁴ (NPU) | 10.2 kJ | 88% | 11.6 kJ |
| Dynamical 8⁴ (NPU) | 4.0 kJ | ~85% (est.) | 4.7 kJ |
| Dynamical 32⁴ (NPU, projected) | 172 kJ | ~85% | 202 kJ |

The NPU cuts energy-per-useful-measurement by ~53% in quenched mode.
We expect similar gains in dynamical mode once the NPU has enough
training data.

---

## Implementation Checklist

- [ ] Add `EnergySnapshot` struct to `barracuda::md` or `barracuda::util`
- [ ] Add sensor reading functions (sysfs for k10temp, nouveau, RAPL)
- [ ] Integrate before/after snapshot in `production_dynamical_mixed`
- [ ] Add `--energy-log` flag to all production binaries
- [ ] Write sidecar monitoring script (`energy_monitor` binary or shell)
- [ ] Validate RAPL readability (may need `chmod` or `setcap`)
- [ ] Retroactively estimate energy for Exp 013, 022, 024 from
      known temps and wall times (already done above)
- [ ] (Future) Instrument smart plug for true wall power measurement

---

## Waste Heat Capture Context

The biomeGate sits in a basement. For context on heat output:

| Source | Thermal output | Equivalent |
|--------|---------------|------------|
| 32⁴ quenched overnight | 18 MJ | ~1 hour of a 5 kW space heater |
| 32⁴ dynamical (est.) | 55 MJ | ~3 hours of a 5 kW space heater |
| Human body at rest | ~80W sustained | 6.9 MJ / day |

A sustained 32⁴ dynamical run produces the thermal equivalent of ~8
human bodies in a room. This is non-trivial for enclosed spaces and
potentially useful for:

- Maintaining fermentation chamber temperature (30-37°C)
- Substrate incubation for mycology (25-30°C)
- Winter heating offset for the workspace
- Thermal mass charging for overnight cultivation runs

Quantifying the actual heat output per experiment type is the first
step toward designing waste heat capture systems.

---

## Output Files

| File | Contents |
|------|----------|
| `results/exp027_energy_by_trajectory.jsonl` | Per-trajectory energy snapshots |
| `results/exp027_energy_sidecar.jsonl` | Continuous 5-second sensor samples |
| `results/exp027_energy_summary.md` | Aggregate energy analysis per experiment |
