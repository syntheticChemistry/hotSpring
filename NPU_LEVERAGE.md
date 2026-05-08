# hotSpring NPU Leverage

How hotSpring uses neuromorphic compute (Akida AKD1000) for physics simulation.

---

## NPU Subsystem Overview

hotSpring integrates NPU inference into lattice QCD, molecular dynamics, and
transport simulations. The NPU provides sub-millisecond classification and
regression between expensive GPU compute steps (HMC trajectories, MD
timesteps).

**Feature gate:** `npu-hw = ["dep:akida-driver", "dep:akida-models"]` in `Cargo.toml`.

**Key modules:**

| Module | Path | Purpose |
|--------|------|---------|
| NPU hardware adapter | `barracuda/src/md/npu_hw.rs` | Host-driven reservoir + NPU FC readout via `InferenceExecutor` |
| NPU tolerances | `barracuda/src/tolerances/npu.rs` | Quantization error bounds (f64→f32→int8→int4) |
| Reservoir tests | `barracuda/src/md/reservoir/tests.rs` | `NpuSimulator`, `MultiHeadNpu` validation |
| Discovery probe | `barracuda/src/discovery.rs` | `probe_npu_available()`, `/dev/akida*` detection |
| Prescreen tier | `barracuda/src/prescreen.rs` | Akida as tier-2 prescreen for expensive GPU compute |
| Multi-head steering | `barracuda/src/production/dynamical_mixed_pipeline/single_beta.rs` | `NpuSteering`, `MultiHeadNpu` for trajectory control |

---

## NPU Validation Binaries

### Core NPU pipeline

```bash
# Quantization parity: f64 → f32 → int8 → int4, verify round-trip bounds
cargo run --bin validate_npu_quantization

# Full pipeline: reservoir update → feature extract → NPU inference → interpret
cargo run --bin validate_npu_pipeline

# Beyond-SDK capabilities: multi-tenancy, online evolution, PUF
cargo run --bin validate_npu_beyond_sdk
```

### Physics-specific NPU

```bash
# Lattice QCD with NPU steering: HMC trajectory accept/reject via NPU
cargo run --bin validate_lattice_npu

# Multi-observable: phase + transport + quality on one chip simultaneously
cargo run --bin validate_multi_observable_npu

# Gen2 NPU: AKD1500 / next-gen silicon features
cargo run --bin validate_gen2_npu
```

### Campaign runner

```bash
# Automated experiment campaign across NPU configurations
cargo run --bin npu_experiment_campaign
```

### Hardware mode

For binaries that support hardware, add the feature flag:

```bash
cargo run --bin validate_npu_pipeline --features npu-hw
cargo run --bin validate_lattice_npu --features npu-hw
```

---

## Expected Output

### `validate_npu_quantization`

```
NPU Quantization Validation
============================
f64 → f32 max relative error: 5.96e-08
f32 → int8 max absolute error: 0.0039
f32 → int4 max absolute error: 0.0625
Round-trip f64 → int4 → f32: within tolerance
Per-channel vs per-layer: per-channel 2.3× tighter
PASS: all 15 precision tiers validated
```

### `validate_npu_pipeline`

```
NPU Pipeline Validation
========================
Reservoir: 50 neurons, spectral_radius=0.9
Readout: InputConv(50,1,1) → FC(128) → FC(1)
Feature extraction: 3.2 µs
NPU inference (software): 12.4 µs
Quality estimate: 0.847 (threshold: 0.85)
Steering decision: CONTINUE_THERMALIZATION
PASS: pipeline end-to-end
```

### `validate_multi_observable_npu`

```
Multi-Observable NPU
=====================
Head 0 (phase): confined (p=0.97)
Head 1 (transport): D*=1.23, η*=0.45, λ*=2.78
Head 2 (quality): 0.91
Total NPs used: 312 / 1000
Latency (3 heads): 89 µs
PASS: all heads produce valid output
```

---

## How It Works

### The Hybrid ESN Architecture

```
Lattice configuration (GPU: SU(3) links)
    │
    ▼
Observable extraction (GPU: plaquette, Polyakov, topQ)
    │
    ▼
ESN reservoir update (CPU or GPU: tanh(W_res·state + W_in·obs))
    │
    ▼
Readout inference (NPU: InputConv→FC→FC, int4, <100 µs)
    │
    ▼
Steering decision (CPU: accept/reject/continue)
    │
    ▼
Next HMC trajectory (GPU)
```

The reservoir runs in software because `tanh` is the only activation in the
reservoir — and AKD1000 implements bounded ReLU, not `tanh`. The readout
(linear FC layers) runs on NPU because it benefits from int4 throughput.

### Quantization Chain

hotSpring tracks precision through the full chain:

```
f64 (physics reference)
  → f32 (GPU compute)
    → int8 (conservative NPU)
      → int4 (production NPU, max throughput)
```

Each step has measured error bounds in `tolerances/npu.rs`. The precision
continuum is shared with barraCuda's 15-tier system.

---

## How to Extend for New Domains

### Adding a new physics readout

1. **Define the observable vector.** What measurements does the NPU classify?
   (e.g., for plasma: temperature, density, magnetic field components)

2. **Train or evolve the readout.** Use `NpuEvolver` for online evolution
   (136 gen/sec) or export trained weights from your framework.

3. **Convert to .fbz.** Use rustChip's pure Rust pipeline:
   ```bash
   cargo run -p akida-cli -- convert \
     --weights plasma_readout.npy \
     --arch "InputConv(8,1,1) FC(64) FC(3)" \
     --output plasma_readout.fbz --bits 4
   ```

4. **Add a validation binary.** Create `src/bin/validate_npu_plasma.rs`
   following the pattern in `validate_npu_pipeline.rs`.

5. **Register tolerances.** Add entries to `tolerances/npu.rs` for the
   new domain's acceptable quantization error.

### Connecting to rustChip

hotSpring's NPU code depends on `akida-driver` and `akida-models` when
the `npu-hw` feature is enabled. These are the same crates available in
rustChip — the API is identical.

For standalone NPU development (no hotSpring dependency), use rustChip
directly: [rustChip LEVERAGE.md](../../infra/rustChip/LEVERAGE.md).

---

## Further Reading

| Document | Path |
|----------|------|
| rustChip Nature Preserve: Physics | `infra/rustChip/baseCamp/preserve/physics.md` |
| rustChip Zoo Guide | `infra/rustChip/baseCamp/ZOO_GUIDE.md` |
| hotSpring metalForge NPU experiments | `metalForge/npu/akida/` |
| hotSpring neuromorphic silicon briefing | `whitePaper/baseCamp/neuromorphic_silicon.md` |
| Multi-head NPU conductor | `infra/rustChip/baseCamp/systems/npu_conductor.md` |
