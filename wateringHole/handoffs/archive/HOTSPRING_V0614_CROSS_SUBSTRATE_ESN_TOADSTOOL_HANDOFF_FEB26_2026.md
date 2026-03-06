# toadStool / barracuda — Cross-Substrate ESN & NPU Evolution Handoff

**Date:** February 26, 2026
**From:** hotSpring (biomeGate)
**To:** toadStool / barracuda core team
**Covers:** GPU ESN dispatch, f32 buffer system, NPU characterization, cross-substrate comparison
**License:** AGPL-3.0-only

---

## Executive Summary

hotSpring achieved the first GPU ESN dispatch using the existing WGSL shaders
(`esn_reservoir_update.wgsl`, `esn_readout.wgsl`) that were designed for
toadStool absorption but never run on GPU from hotSpring until now. This
handoff documents:

1. **GPU ESN dispatch** — working GPU inference pipeline with timing data
2. **f32 buffer system** — new `GpuF64` methods for f32 shader support
3. **Cross-substrate scaling** — where CPU, GPU, and NPU each win
4. **NPU capability envelope** — every task the Akida CAN handle
5. **Engineering discoveries** — recurrent network GPU dispatch patterns
6. **What toadStool should absorb** — specific code + patterns

---

## Part 1: GPU ESN Dispatch (NEW — First Ever)

### What We Built

The ESN WGSL shaders existed in hotSpring since v0.6.13 but were never
dispatched to GPU. We built a complete GPU ESN inference pipeline:

```
ExportedWeights (f32) → GPU buffers → per-step reservoir update → readout → f32 readback
```

### Implementation

**Binary**: `barracuda/src/bin/cross_substrate_esn_benchmark.rs`

The `GpuEsn` struct wraps the complete GPU pipeline:
- Compiles both shaders via `GpuF64::create_pipeline()`
- Creates f32 storage buffers for W_in, W_res, W_out, state, input, output, params
- Per-timestep: uploads input → dispatches reservoir update → state updates in-place
- After all timesteps: dispatches readout → reads back f32 output

### Critical Discovery: Recurrent Network Dispatch Pattern

**Naive encoder batching breaks for recurrent networks.**

The ESN reservoir update at step `t` reads the state written by step `t-1`.
If you batch all timesteps into a single `CommandEncoder`, the
`queue.write_buffer()` for the input races with the encoded dispatches.
All dispatches see the LAST input, not their respective inputs.

**Correct pattern**: Per-step submit (one `dispatch()` call per timestep).

**toadStool action:** When absorbing ESN GPU dispatch into the compute
pipeline, implement either:
- Per-step submit (correct, simple, what hotSpring uses)
- Double-buffered ping-pong state management (advanced, for latency hiding)

Do NOT use `encode_pass()` batching for recurrent networks unless all
state buffers are independent per timestep.

---

## Part 2: f32 Buffer System

### New Methods on `GpuF64`

Added to `barracuda/src/gpu/buffers.rs`:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `create_f32_buffer` | `(&self, data: &[f32], label: &str) -> Buffer` | Read-only f32 storage |
| `create_f32_rw_buffer` | `(&self, data: &[f32], label: &str) -> Buffer` | Read-write f32 storage (state buffers) |
| `create_f32_output_buffer` | `(&self, count: usize, label: &str) -> Buffer` | Zero-initialized f32 output |
| `upload_f32` | `(&self, buffer: &Buffer, data: &[f32])` | Write f32 data to GPU buffer |
| `read_back_f32` | `(&self, buffer: &Buffer, count: usize) -> Result<Vec<f32>>` | Staging copy + map + read |

Also added `mapped_bytes_to_f32()` public function (parallel to existing `mapped_bytes_to_f64()`).

**toadStool action:** These methods are natural additions to the
`WgpuDevice` or `TensorContext` API. The f32 buffer system is required
for any f32 shader (ESN, NPU parity testing, mixed-precision pipelines).

---

## Part 3: Cross-Substrate Scaling Results

### Timing Matrix (μs, 50-step sequence, 8 features)

| RS   | CPU-f64   | CPU-f32   | GPU-f32   |
|------|-----------|-----------|-----------|
| 16   |      27   |      25   |   4,876   |
| 50   |     133   |     134   |   5,924   |
| 100  |     483   |     492   |   5,711   |
| 200  |   1,714   |   1,738   |   5,655   |
| 500  |  10,400   |  10,188   |   5,527   |
| 1024 |  16,481   |  —        |   3,665   |

### GPU Crossover Point

**RS ≈ 512** — below this, CPU wins; above, GPU parallelism dominates.

- RS=1024, SeqLen=200, OS=6: GPU is **8.2× faster** than CPU-f64
- RS=512: roughly tied (~1.0× ratio)
- RS=50: CPU is 42× faster than GPU

### GPU Dispatch Overhead

- Null dispatch: ~40μs
- Per-step with buffer upload: ~3.5ms (dominated by `queue.write_buffer` + submit cycle)

**toadStool action:** For ESN inference, consider:
1. For RS < 200: CPU inference is optimal, no GPU dispatch needed
2. For RS ≥ 512: GPU inference with per-step dispatch
3. For production: pre-allocate all buffers once, reuse across predictions

---

## Part 4: NPU Capability Envelope

### Confirmed Capabilities

| Capability              | Result | Notes |
|-------------------------|--------|-------|
| Threshold detection     | 100% accuracy | Binary classification via ESN readout |
| Streaming inference     | 2.8 μs/step | Matches CPU, 1000× faster than GPU |
| Multi-output (1–8)      | No latency penalty | max|Δ| < 3e-7 across all output counts |
| Weight mutation          | 141 μs/reload | Linear drift, predictable |
| QCD thermalization      | 100% (38/38) | NPU and GPU achieve identical accuracy |
| Multi-observable scoring | RMSE = 0.003 | 3-output anomaly scoring |
| Batch inference          | Linear scaling | No hardware pipelining in simulator |

### NPU vs GPU for QCD Screening

Both NPU and GPU achieve 100% thermalization detection accuracy and
identical multi-observable RMSE. The difference is latency:

| Metric | NPU-sim | GPU-f32 |
|--------|---------|---------|
| Per-inference latency | 2.8 μs | 3,170 μs |
| Streaming throughput | 357k inf/s | 317 inf/s |
| Power estimate | ~30mW | ~350W |

**The NPU owns the screening workload.** No other substrate can match
its latency for single-sample inference.

### NPU Characterization Campaign (Exp 020)

Beyond the cross-substrate comparison, the NPU characterization campaign
tested 6 pipeline placements:

| Placement | Description | Best Use |
|-----------|-------------|----------|
| A | Pre-thermalization screening | **Biggest win**: 3.15h savings projected |
| B | Mid-trajectory abort | Useful at large lattices |
| C | Post-trajectory classification | Baseline approach |
| D | Inter-beta steering | Needs more training data |
| E | Pre-run bootstrap | Warm-start from prior runs |
| F | All combined | 390 trajectories saved |

ESN models trained:
- Thermalization detector: 87.5% accuracy, 61.8% savings
- Rejection predictor: 96.2% accuracy
- 6-output multi-model: All outputs finite, multi-output free confirmed

**toadStool action:** The NPU placement framework (A–F) should be
captured as pipeline configuration options in metalForge's `Pipeline`
topology. The optimal placement (pre-thermalization, A) can be the
default for QCD production runs.

---

## Part 5: Substrate Assignment Guide

### Where Each Substrate Belongs

| Workload | Best Substrate | Why |
|----------|---------------|-----|
| ESN RS < 200 | **CPU-f64** | Sub-ms latency, full precision |
| ESN RS ≥ 512 | **GPU-f32** | Parallel matvec wins (8.2× at RS=1024) |
| Streaming 1-step inference | **NPU** | 2.8μs, no dispatch overhead |
| Multi-output screening | **NPU** | Free multi-head, constant latency |
| Weight mutation in flight | **NPU** | 141μs reload, supports online learning |
| Precision-critical readout | **CPU-f64** | f64 eliminates rounding |
| Physics simulation (HMC) | **GPU-f64** | DF64 on FP32 cores, massive parallelism |
| Large reservoir embedding | **GPU-f32** | For RS > 512 (future high-dim physics) |

### Recommended Production Pipeline

```
GPU₁ (HMC physics, DF64 on FP32 + native FP64)
  ↓ observable stream (plaquette, Polyakov, ΔH, CG iters, acceptance)
NPU (2.8μs pre-thermalization screening, anomaly scoring)
  ↓ flagged trajectories
CPU (f64 precision verification, readout arbitration, training update)
```

---

## Part 6: What toadStool Should Absorb

### Priority 1: f32 Buffer System

**Source**: `barracuda/src/gpu/buffers.rs` — `create_f32_*`, `upload_f32`, `read_back_f32`

These are natural `WgpuDevice` methods. Every f32 shader (ESN, NPU parity,
mixed-precision) needs them.

### Priority 2: GPU ESN Dispatch Pattern

**Source**: `GpuEsn` in `cross_substrate_esn_benchmark.rs`

The per-step dispatch pattern for recurrent networks should become a
first-class compute pipeline in toadStool:

```rust
pub struct GpuReservoirPipeline {
    reservoir_pipeline: ComputePipeline,
    readout_pipeline: ComputePipeline,
    // Pre-allocated buffers from ExportedWeights
    // Per-step submit with state buffer in-place update
}
```

### Priority 3: ExportedWeights → GPU Pipeline

`ExportedWeights` (f32 flat arrays) already maps exactly to the shader
binding layout. The path from CPU-trained ESN to GPU inference is:

```
EchoStateNetwork::train() → export_weights() → GpuReservoirPipeline::new()
```

### Priority 4: Cross-Substrate Parity Testing

The cross-substrate benchmark pattern (same workload on CPU/GPU/NPU, verify
parity within tolerance, measure timing) should become a reusable test
harness in barracuda for any compute pipeline.

### Lower Priority: NPU Pipeline Placement Framework

The placement framework (A–F) is currently captured in experiment code.
When metalForge evolves to production, these should become configurable
pipeline topologies in the `forge` crate.

---

## Part 7: Precision Notes

### f32 vs f64 ESN Accuracy

GPU (f32) vs CPU (f64) divergence depends on reservoir size:

| RS Range | Relative Error | Notes |
|----------|---------------|-------|
| ≤ 50     | < 1e-5        | Negligible |
| 100      | ~7%           | Precision "sweet spot" — errors compound without enough neurons to dilute |
| ≥ 200    | < 3%          | More neurons dilute individual precision effects |

For production: if ESN output is used for binary decisions (threshold/classification),
f32 is adequate at any RS. If used for quantitative predictions, RS=100 is the
danger zone — either use RS < 50 or RS ≥ 200.

---

## Part 8: Files Changed Since Feb 25 Handoff

| File | Change |
|------|--------|
| `barracuda/src/gpu/buffers.rs` | Added f32 buffer create/upload/readback methods |
| `barracuda/src/bin/cross_substrate_esn_benchmark.rs` | NEW: 6-experiment cross-substrate ESN benchmark |
| `barracuda/src/bin/npu_experiment_campaign.rs` | NEW: NPU characterization (Exp 020) |
| `barracuda/src/bin/production_beta_scan.rs` | Added `--trajectory-log` for per-trajectory JSONL |
| `experiments/020_NPU_CHARACTERIZATION_CAMPAIGN.md` | NEW: NPU characterization results |
| `experiments/021_CROSS_SUBSTRATE_ESN_COMPARISON.md` | NEW: Cross-substrate ESN results |
| `metalForge/README.md` | Updated with Exp 020 + 021 findings |
| `metalForge/npu/akida/EXPLORATION.md` | Updated with cross-substrate comparison |
| `whitePaper/baseCamp/neuromorphic_silicon.md` | NEW: Neuromorphic silicon exploration briefing |

### Validation Status

All new/modified code: **0 clippy warnings** (lib + bins), all existing tests pass.
- `cross_substrate_esn_benchmark`: 35/35 checks passed
- `npu_experiment_campaign`: 13/13 checks passed
- Library tests: 10/10 reservoir tests pass
