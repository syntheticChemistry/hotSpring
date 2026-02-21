# hotSpring → ToadStool: NPU Reservoir Transport Handoff

**Date**: February 20, 2026
**From**: hotSpring (computational physics validation)
**To**: ToadStool/BarraCUDA (dispatch + hardware abstraction)
**Status**: Ready for absorption

---

## Executive Summary

hotSpring has validated reservoir computing for MD transport prediction
across three substrates: Python (NumPy), Rust CPU (pure f64), and
simulated NPU (f32). The ESN predicts self-diffusion coefficients (D*)
from short velocity trajectory segments with <22% test error across
the (κ, Γ) phase diagram.

ToadStool already has all infrastructure needed to absorb this:

| Component | ToadStool location | Status |
|---|---|---|
| `ESN` (GPU matmul) | `barracuda::esn_v2` | Ready — uses `matmul.wgsl` |
| `load_reservoir()` | `akida-driver::NpuBackend` | Ready — all 3 backends |
| Ridge readout | `akida-reservoir-research::ReadoutTrainer` | Ready |
| Dual-chip ensemble | `akida-reservoir-research::DualChipEnsemble` | Ready |
| `StatefulPipeline` | `barracuda::staging::stateful` | Ready — `run_iterations()` |
| `UnidirectionalPipeline` | `barracuda::staging::unidirectional` | Ready — streaming I/O |

**What's new from hotSpring**: validated physics application + WGSL shaders
for fused ESN operations + cross-substrate parity proof.

---

## What hotSpring Built

### 1. Python Control (`control/reservoir_transport/scripts/reservoir_vacf.py`)

- Self-contained: Yukawa MD (numba JIT) + ESN + Green-Kubo D*
- 6 (κ,Γ) cases: N=256, 4k production steps, 500-frame short segments
- ESN: reservoir_size=50, spectral_radius=0.95, leak_rate=0.3
- **Train error: 2.2%, Test error: 17.9%**
- Outputs: `control/reservoir_transport/results/reservoir_transport_baseline.json`
- Units: OCP E₀ convention (m*=3.0, force_prefactor=1.0, T*=1/Γ)

### 2. Rust CPU ESN (`barracuda/src/md/reservoir.rs`)

- Pure Rust ESN with xoshiro256++ PRNG, ridge regression via Gaussian elimination
- `velocity_features()` extracts [mean_v, speed, KE, v_rms, κ_scaled, Γ_scaled]
- `EchoStateNetwork::train()` / `predict()` — matches Python math
- `NpuSimulator` — f32 arithmetic mirroring Akida hardware path
- `ExportedWeights` — flattened f32 arrays matching `load_reservoir(w_in, w_res)` API
- 6 unit tests, all passing

### 3. Validation Binary (`bin/validate_reservoir_transport.rs`)

- 10/10 checks passing (ALL CHECKS PASSED)
- Runs CPU MD → VACF → ESN train → ESN predict → NPU sim → parity check
- **Train error: 12.4%, Test error: 21.9%**
- **CPU/NPU f64→f32 parity: 1.2×10⁻⁶** (essentially zero)
- Wall time: 38s (release mode)

### 4. WGSL Shaders (shader-originating math)

- `shaders/esn_reservoir_update.wgsl` — fused W_in·input + W_res·state → leaky tanh
- `shaders/esn_readout.wgsl` — W_out·state → prediction
- Both f32, workgroup_size(64), matching Akida's f32 arithmetic

---

## Benchmarks

| Metric | Python (NumPy) | Rust CPU | Ratio |
|--------|:---:|:---:|:---:|
| Full pipeline (MD + ESN, 6 cases) | 137s | 38s | **3.6×** |
| ESN only (train + 6 predict) | 16.2 ms | 10.9 ms | **1.5×** |
| D* test error | 17.9% | 21.9% | Comparable |
| CPU/NPU f32 parity | — | 1.2e-6 | ✓ |

---

## Absorption Path for ToadStool

### Step 1: Wire `esn_v2` to transport features

```rust
// ToadStool's ESN already does GPU matmul. Wire it to hotSpring's features:
let config = ESNConfig {
    input_size: 8,      // [mean_v(3), speed, KE, v_rms, κ, Γ]
    reservoir_size: 50,
    output_size: 1,     // D*
    spectral_radius: 0.95,
    connectivity: 0.2,
    leak_rate: 0.3,
    regularization: 1e-2,
    seed: 42,
};
let mut esn = ESN::new(config).await?;
esn.train(&velocity_features, &d_star_targets).await?;
```

### Step 2: Export to Akida

```rust
// Train on GPU, deploy to NPU
let (w_in, w_res) = esn.export_weights();  // need to add this method
let mut device = DeviceManager::discover()?.open_first()?;
device.backend().load_reservoir(
    bytemuck::cast_slice(&w_in),
    bytemuck::cast_slice(&w_res),
)?;
```

### Step 3: NPU inference

```rust
// Feed velocity features, get D* prediction
let input_bytes = bytemuck::cast_slice(&features_f32);
let result = executor.infer(input_bytes, &mut device)?;
let d_star: f32 = bytemuck::cast_slice(&result.output)[0];
```

### Step 4: Fused WGSL shader (optional optimization)

The `esn_reservoir_update.wgsl` shader fuses the matmul + tanh into a single
dispatch, avoiding intermediate buffer allocation. For small reservoirs (≤500),
this is faster than two separate matmul dispatches. ToadStool can compile this
via `ShaderTemplate::for_driver_profile()` and dispatch through `StatefulPipeline`.

### Step 5: Unidirectional streaming for real-time inference

```
MD simulation → UnidirectionalPipeline → velocity features → ESN reservoir → D* prediction
     GPU                                      GPU/NPU
```

The `UnidirectionalPipeline` ring buffer streams velocity frames from the MD
simulation directly to the ESN without CPU readback. The ESN readout (1 scalar)
is the only value that crosses the device boundary.

---

## Physics Context

The velocity autocorrelation function C(t) = ⟨v(0)·v(t)⟩ in a Yukawa OCP
has a characteristic temporal pattern:

- C(0) = 3T*/m* (thermal)
- Exponential decay with timescale ~ ω_p⁻¹
- Possible oscillations at strong coupling (caging)
- D* = (1/3) ∫₀^∞ C(t) dt (Green-Kubo relation)

This temporal pattern is exactly what echo state networks excel at. The
reservoir's fading memory naturally captures the exponential decay and
oscillation features. The readout maps final reservoir state → D*.

The key insight: **80k MD steps → 1k steps + ESN = 80× less compute for D***.
On NPU, inference is microwatts. On GPU, the full MD can stream directly to
the ESN via `UnidirectionalPipeline`.

---

## Unit Convention Note

hotSpring uses the OCP E₀ convention where m* = 3.0 (derived: 4πn·a_ws³).
The Python/Sarkas convention uses m*=1 with V*=Γ·exp(-κr)/r. Both are
standard; the Rust code documents the derivation in `md/config.rs`.

Evolution: consider migrating both conventions to m*=1 for ecosystem parity
with Python scientific computing tools (Sarkas, LAMMPS reduced units).

---

## Files to Absorb

| File | Purpose | Priority |
|---|---|---|
| `barracuda/src/md/reservoir.rs` | CPU ESN + NpuSimulator + weight export | **P0** |
| `barracuda/src/md/shaders/esn_reservoir_update.wgsl` | Fused GPU ESN kernel | **P0** |
| `barracuda/src/md/shaders/esn_readout.wgsl` | GPU readout kernel | **P1** |
| `barracuda/src/bin/validate_reservoir_transport.rs` | Validation binary | **P1** |
| `control/reservoir_transport/scripts/reservoir_vacf.py` | Python scaffold | Reference |
| `control/reservoir_transport/results/reservoir_transport_baseline.json` | Baseline data | Reference |

---

## Dual-Chip Ensemble Extension

With 2× AKD1000 available, the `DualChipEnsemble` from
`akida-reservoir-research` can split the phase diagram:

- **Chip 1**: reservoir trained on weak coupling (Γ < 50)
- **Chip 2**: reservoir trained on strong coupling (Γ ≥ 50)
- **Ensemble**: weighted average based on input Γ

This doubles the effective reservoir capacity and should improve
generalization across the full (κ, Γ) phase diagram.
