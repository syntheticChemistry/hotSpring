# hotSpring → ToadStool/BarraCUDA: metalForge NPU Discovery + Experiment Handoff

**Date:** 2026-02-20
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Builds on:** `HOTSPRING_V060_CONSOLIDATED_HANDOFF_FEB21_2026.md` (science portfolio)
and `HOTSPRING_NPU_RESERVOIR_HANDOFF_FEB20_2026.md` (ESN transport, archived)

---

## Executive Summary

hotSpring has completed comprehensive hardware probing of the BrainChip AKD1000
NPU, overturning 10 SDK assumptions and building a validated experiment suite
(29/29 checks across Python hardware and Rust math). Combined with the existing
science portfolio (18 papers, 33 validation suites, 454 tests), this establishes
the full train-on-GPU → deploy-on-NPU pipeline for physics workloads.

### What's New Since Last Handoff

| Change | Impact |
|--------|--------|
| 10 SDK assumptions overturned via hardware probing | NPU far more capable than documented |
| `npu_beyond_sdk.py` — 13/13 HW checks | Validated on actual AKD1000 |
| `validate_npu_beyond_sdk.rs` — 16/16 math checks | Substrate-independent math proof |
| `predict_return_state()` added to `NpuSimulator` | Enables readout weight manipulation |
| 4 new tolerances in `tolerances.rs` | FC depth, batch speedup, multi-output, mutation |
| 20/20 validation suites (was 18/18) | +2 NPU suites (quantization + beyond-SDK) |
| 72 centralized tolerances (was 68) | +4 NPU beyond-SDK hardware tolerances |
| metalForge docs updated | README, BEYOND_SDK, HARDWARE, EXPLORATION all current |

---

## Part 1: Hardware Discoveries ToadStool Should Know

These findings directly inform ToadStool's substrate dispatch strategy.

### 1.1 Arbitrary Input Dimensions (Discovery #1)

**SDK says**: InputConv accepts only 1 or 3 channels.
**Reality**: Any channel count works (tested 1-64 on hardware).

**Implication for ToadStool**: No need to reshape physics feature vectors.
The ESN's 50-dim input vectors and any future feature dimensions map directly.

### 1.2 FC Layer Merging via SkipDMA (Discovery #2)

**SDK says**: Each FC layer is an independent hardware operation.
**Reality**: All FC layers merge into a single hardware sequence. The intra-mesh
SkipDMA transfers activations between NPs without PCIe round-trips.

**Measured**: 7 FC layers (9 total layers) = 6.7% overhead vs 1 FC layer.

**Implication for ToadStool**: Deep networks are essentially free on NPU.
A 50→256→256→256→1 architecture runs in a single hardware dispatch.

### 1.3 Batch Inference Amortization (Discovery #3)

**SDK says**: No batch inference documentation.
**Reality**: Batch forward pass amortizes PCIe latency across samples.

| Batch Size | Per-Sample μs | Throughput | vs Single |
|:---:|:---:|:---:|:---:|
| 1 | 1001 | 999/s | baseline |
| 4 | 483 | 2072/s | 2.07× |
| 8 | 427 | 2344/s | **2.35×** |
| 16 | 588 | 1702/s | 1.70× |

**Sweet spot is batch=8**. Beyond 16, SRAM contention degrades throughput.

**Implication for ToadStool**: Buffer 8 inference requests before dispatching.
The `UnidirectionalPipeline` already supports this batching pattern.

### 1.4 Multi-Output Free Cost (Discovery #5 equivalent)

**Measured**: 10 outputs cost 4.5% more latency than 1 output.

**Implication for ToadStool**: Predict D*, viscosity, thermal conductivity
simultaneously from a single reservoir state. Multi-observable readout is free.

### 1.5 Weight Mutation Linearity (Discovery #6)

**Measured**: `set_variable("weights", w * k)` produces output × k with
error = 0.0000. Exact integer linearity.

**Implication for ToadStool**: Hot-swap readout weights without reprogramming
the model. Online learning or ensemble switching becomes ~14ms overhead.

### 1.6 Hardware Determinism (Discovery #7 equivalent)

**Measured**: 20 identical inputs → 1 unique output. Model save/load round-trip
produces bit-identical results.

**Implication for ToadStool**: NPU results are reproducible. No stochastic
noise from hardware — validation is straightforward.

---

## Part 2: Code to Absorb

### 2.1 Rust: `barracuda/src/md/reservoir.rs`

**What changed**: Added `predict_return_state()` to `NpuSimulator`.

```rust
pub fn predict_return_state(&mut self, input_sequence: &[Vec<f64>]) -> Vec<f32>
```

Returns the raw reservoir state (before readout) for external weight manipulation.
This enables the weight mutation validation pattern and multi-readout experiments.

ToadStool's `esn_v2` should absorb this — it allows readout weight testing
without rebuilding the reservoir.

### 2.2 Rust: `barracuda/src/bin/validate_npu_beyond_sdk.rs`

New validation binary. 16 checks across 6 test categories:

| Category | Checks | What it validates |
|----------|:------:|-------------------|
| Arbitrary input dims | 6 | ESN with input_size=2,5,8,16,50,64 produces finite predictions |
| Deep FC chain math | 2 | f64→f32 parity + int4 readout error bounded |
| Multi-output readout | 3 | 3 outputs: correct count, all finite, all distinct |
| Weight mutation | 2 | w×2 and w×(-3) linearity (error < 1%) |
| Wide FC quantization | 2 | reservoir_size=128,256 f64→f32 parity |
| Determinism | 1 | 10 identical runs produce identical results |

### 2.3 Rust: `barracuda/src/tolerances.rs`

4 new constants:

```rust
pub const NPU_FC_DEPTH_OVERHEAD: f64 = 0.30;        // depth=7 vs depth=1 latency
pub const NPU_BATCH_SPEEDUP_MIN: f64 = 1.5;         // batch=8 vs batch=1
pub const NPU_MULTI_OUTPUT_OVERHEAD: f64 = 0.30;     // 10 outputs vs 1 output
pub const NPU_WEIGHT_MUTATION_LINEARITY: f64 = 0.01; // w×k → output×k ratio error
```

### 2.4 Python: `control/metalforge_npu/scripts/npu_beyond_sdk.py`

Hardware validation script. 13 checks on actual AKD1000:

| Test | Checks | Key Measurement |
|------|:------:|-----------------|
| Arbitrary channels | 2 | 8/8 non-standard channels map to HW |
| FC merge (SkipDMA) | 2 | 5/5 depths merge, 6.7% overhead |
| Batch inference | 2 | 2.35× speedup, 427μs/sample at batch=8 |
| Wide FC scaling | 2 | 64-1024 neurons all map |
| Multi-output | 1 | 4.5% overhead for 10 outputs |
| Weight mutation | 2 | Exact 2× and -3× proportionality |
| Determinism | 1 | 20 runs → 1 unique output |
| Save/load parity | 1 | Bit-identical after round-trip |

Outputs `control/metalforge_npu/results/npu_beyond_sdk_baseline.json`.

### 2.5 Cargo.toml

New `[[bin]]` section:

```toml
[[bin]]
name = "validate_npu_beyond_sdk"
path = "src/bin/validate_npu_beyond_sdk.rs"
```

---

## Part 3: Barracuda Evolution Review

### What hotSpring Built in Barracuda (v0.6.0)

| Domain | Modules | Shaders | Validation |
|--------|---------|---------|------------|
| Nuclear EOS (L1-L3) | `physics/semf`, `hfb`, `hfb_gpu`, `hfb_deformed` | 14 WGSL | 9/9 |
| MD Yukawa OCP | `md/simulation`, `celllist`, `observables`, `transport` | 11 WGSL | 12/12 |
| Lattice QCD | `lattice/complex_f64`, `su3`, `wilson`, `hmc`, `dirac`, `cg` | WGSL templates ready | 12/12 |
| Abelian Higgs | `lattice/abelian_higgs` | — | 17/17 |
| Reservoir/ESN | `md/reservoir` + `esn_reservoir_update.wgsl`, `esn_readout.wgsl` | 2 WGSL | 22/22 |
| Transport | `md/transport` | — | 13/13 |
| Screened Coulomb | `physics/screened_coulomb` | — | 23/23 |
| HotQCD EOS | `lattice/eos_tables` | — | thermodynamic |
| Infrastructure | `validation`, `tolerances`, `provenance`, `data`, `gpu`, `error` | — | 9/9 tests |

### Known Issues for ToadStool

| Issue | Location | Workaround |
|-------|----------|------------|
| CellListGpu BGL mismatch | ToadStool `prefix_sum` (3 bindings vs hotSpring's 4) | hotSpring uses local `GpuCellList` |
| NPU device permissions | `/dev/akida0` needs `chmod 666` | `pkexec` on boot; needs Rust DeviceManager |
| Lattice WGSL templates | `complex_f64.rs`, `su3.rs` have WGSL strings but no GPU dispatch | Ready for GPU promotion via WgslOptimizer |
| Deformed HFB H-build on CPU | `hfb_deformed_gpu.rs` only offloads eigensolve | Full GPU H-build is next evolution |

### Primitives hotSpring Consumes from ToadStool

| Primitive | Import Path | Used By |
|-----------|-------------|---------|
| `ReduceScalarPipeline` | `barracuda::staging::reduce_scalar` | Energy, VACF, stress ACF |
| `StatefulPipeline` | `barracuda::staging::stateful` | Iterative MD, HFB SCF |
| `WgslOptimizer` | `barracuda::shaders` | All shader compilation |
| `GpuDriverProfile` | `barracuda::shaders` | ILP scheduling |
| `BatchedEighGpu` | `barracuda::linalg` | HFB eigensolve |
| `PppmGpu` | `barracuda::pppm` | κ=0 Coulomb |
| `SsfGpu` | `barracuda::linalg` | Structure factor |

---

## Part 4: Recommended Absorption Order

### Phase 1: Immediate (this week)

1. **Run validation**: `cargo run --bin validate_npu_beyond_sdk` — verify 16/16 pass
2. **Run HW validation**: `python control/metalforge_npu/scripts/npu_beyond_sdk.py` — verify 13/13 pass
3. **Absorb `predict_return_state()`** into ToadStool's `esn_v2`
4. **Absorb tolerances** — 4 new NPU constants

### Phase 2: NPU Integration (next sprint)

5. **Batch dispatch**: Wire `UnidirectionalPipeline` to buffer 8 samples before NPU forward
6. **Multi-output readout**: Deploy D* + viscosity + thermal conductivity from single state
7. **Economy clock mode**: Set as default for NPU inference (19% slower, 18% less power)

### Phase 3: Cross-Substrate Pipeline (month 1)

8. **Train on GPU**: ESN reservoir update via `esn_reservoir_update.wgsl` on RTX 4070
9. **Deploy on NPU**: Quantize weights → int4 → load via `set_variable()` → inference on AKD1000
10. **Rust NPU driver**: Open `/dev/akida0`, load FlatBuffer programs, bypass Python SDK

### Phase 4: Hardware-Native Design (month 2+)

11. **Native int4+ReLU reservoir**: Design for hardware's compute model, not float approximation
12. **Neuromorphic PDE elements**: Multi-pass FC chains for Poisson solver
13. **Direct C++ engine**: Rust FFI to Akida Engine for register-level control

---

## Part 5: What We Learned (Relevant to All ToadStool Evolution)

### The Vendor SDK Pattern

The NPU exploration confirms the same pattern as GPU f64:

| Substrate | SDK Claim | Reality | Method |
|-----------|-----------|---------|--------|
| **GPU** (CUDA) | f64 at 1:32 ratio | f64 at 1:2 via wgpu/Vulkan | Bypass CUDA driver |
| **NPU** (MetaTF) | InputConv: 1 or 3 channels | Any channel count works | Test every assumption |

**Lesson**: Vendor SDKs optimize for their target market (images for NPU, ML for
GPU). Physics workloads need different capabilities. Always test the silicon, not
the SDK.

### Quantization Budget

The ESN quantization cascade establishes the error budget for NPU deployment:

| Precision | Max Error vs f64 | Acceptable for Physics? |
|-----------|:---:|:---:|
| f32 | <0.001% | Yes — identical for all practical purposes |
| int8 | <5% | Yes — within MD statistical uncertainty |
| int4 | <30% | Marginal — use for screening, not final predictions |
| int4+act4 | <50% | No — use only for fast pre-filtering |

**Recommendation**: Use int8 for production NPU inference. Reserve int4 for
pre-screening classifiers where 30% error is acceptable.

### The Three-Substrate Pipeline

```
Train (GPU, f64):  ESN weights via ridge regression on GPU
                   → export f32 weights via ExportedWeights
                   → validate via NpuSimulator (16/16 pass)

Deploy (NPU, int4): Quantize f32 → int4 (30% error budget)
                    → load via set_variable()
                    → batch=8 inference at 427μs/sample

Validate (CPU, f64): Pure Rust math (no GPU/NPU dependency)
                     → ValidationHarness checks against tolerances
                     → 20/20 suites pass
```

This pipeline is the template for every future physics workload on NPU.

---

## File Manifest

| File | Type | Status |
|------|------|--------|
| `barracuda/src/md/reservoir.rs` | Rust lib | Modified — added `predict_return_state()` |
| `barracuda/src/bin/validate_npu_beyond_sdk.rs` | Rust bin | **New** — 16/16 checks |
| `barracuda/src/tolerances.rs` | Rust lib | Modified — +4 NPU tolerances |
| `barracuda/Cargo.toml` | Config | Modified — new `[[bin]]` |
| `control/metalforge_npu/scripts/npu_beyond_sdk.py` | Python | **New** — 13/13 HW checks |
| `control/metalforge_npu/results/npu_beyond_sdk_baseline.json` | Data | **New** — HW measurements |
| `metalForge/README.md` | Docs | Updated — discoveries table, control results |
| `metalForge/npu/akida/BEYOND_SDK.md` | Docs | Existing — 10 discovery analysis |
| `metalForge/npu/akida/HARDWARE.md` | Docs | Existing — architecture deep-dive |
| `metalForge/npu/akida/EXPLORATION.md` | Docs | Existing — novel applications |

---

*Train on GPU. Deploy on NPU. Validate on CPU. The math is portable.*
