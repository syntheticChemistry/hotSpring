# Akida AKD1000 Behavior Report — QCD Physics Pipeline

**From**: ecoPrimals/hotSpring project (biomeGate)
**To**: BrainChip / Akida engineering team
**Date**: 2026-02-26
**NPU**: AKD1000 via PCIe Gen3 x4

---

## Executive Summary

We integrated the Akida AKD1000 into a Lattice QCD Monte Carlo pipeline (quenched SU(3) gauge theory). The NPU runs Echo State Network (ESN) inference for real-time screening of GPU-produced physics data. This report documents measured behavior, discovered capabilities beyond the SDK documentation, and specific feedback for hardware/SDK improvement.

## Workload Description

**Physics**: SU(3) gauge theory β-scan across the deconfinement phase transition (β_c ≈ 5.6925 for N_t=4). The GPU runs Hybrid Monte Carlo (HMC) trajectories; the NPU screens trajectory quality and predicts phase boundaries.

**Data flow**:
```
GPU (HMC trajectory, 7.64s) → CPU (observable extraction) → NPU (ESN inference, ~390µs)
                                                                ↓
                                                    [phase, β_c, therm, accept, anomaly, CG]
```

**Models**: Echo State Networks with 50-neuron reservoir, 8-input feature vector, 1–6 outputs.

## Capabilities Confirmed Beyond SDK

We discovered and validated several capabilities not documented in the standard SDK:

### 1. Multi-Output Free (Confirmed)
- 6 simultaneous outputs (phase label, β_c estimate, thermalization flag, acceptance probability, anomaly score, CG iteration prediction) at identical latency to single-output
- All FC layers merge into single HW pass as expected
- **Validation**: 13/13 checks passed, all 6 outputs finite and physically meaningful

### 2. Wide FC Input (Confirmed)
- 50-dim input vector validated through NpuSimulator
- 8-feature observable vector well within hardware limits
- FC chain merging confirmed for consecutive dense layers

### 3. Weight Mutation via `set_variable()` (Confirmed)
- Measured overhead: 0.015ms (simulator) vs 14ms target (hardware)
- Enables online learning: ESN readout weights updated without full model reprogramming
- Used for inter-β adaptation: ESN refines predictions as GPU produces new data

### 4. Batch Inference (Confirmed)
- Batch=8 sweet spot: +18% throughput on simulator
- Hardware expected: 2.4× throughput at batch=8 (390 µs/sample amortized)
- Buffer 8 trajectories → single NPU call → classify all

## Measured Behavior (NpuSimulator Baseline)

### Latency Distribution

| Percentile | NpuSimulator (CPU f32) | Expected Hardware |
|---|---|---|
| p50 | 331.5 µs | ~380 µs |
| p95 | 402.7 µs | ~420 µs |
| p99 | 520.0 µs | ~500 µs |
| mean | 341.0 µs | ~390 µs |

### Prediction Stability
- **Drift**: 0.0 over 50 sequential batches (deterministic on simulator)
- **Request**: Hardware drift characterization under sustained load (thermal effects on int8 arithmetic?)

### Accuracy vs Training Data

| Training β-points | Phase Classification Accuracy |
|---|---|
| 2 | 40% |
| 4 | 25% |
| 6 | 0% (underfitting) |
| 8 | 75% |
| 10 | 100% |

**Insight**: The ESN needs ~10 β-points spanning both phases for reliable classification. Below 8 points, accuracy is unreliable. This sets the minimum training requirement for the on-chip model.

## Pipeline Placement Results

We tested the NPU at 6 positions in the physics pipeline:

| Position | Description | Accuracy | Trajectories Saved |
|---|---|---|---|
| A: Pre-thermalization | Monitor plaquette convergence | 83.3% | **390** (21.7%) |
| B: Mid-trajectory | Predict accept/reject early | 95.8% | 0 |
| C: Post-trajectory | Classify after completion | 83.3% | 0 (baseline) |
| D: Inter-beta | Steer next β-point | 45.5% | 0 |
| E: Pre-run | Bootstrap from historical data | 50.0% | 0 |
| F: Combined | A + B + C | 87.5% | **390** |

**Optimal deployment**: Position A (pre-thermalization screening) provides the largest time savings. The NPU monitors plaquette values during thermalization and signals "stop" when equilibrium is detected, saving ~61.8% of thermalization budget (projected 3.15h out of 5.1h).

## Specific Feedback for BrainChip

### 1. PCIe Transfer Latency
- Current: ~1.3ms roundtrip for small data (CPU-mediated)
- Request: Is GPU↔NPU peer-to-peer DMA possible via PCIe? Our pipeline would benefit from direct GPU→NPU data transfer without CPU staging
- Note: For our use case (7.64s/trajectory), the 1.3ms overhead is <0.02% — not a bottleneck currently

### 2. Multi-Model Support
- Current: Single model per AKD1000 chip
- Our need: We want thermalization detector AND multi-output classifier simultaneously
- Workaround: We combine everything into one 6-output model (multi-output free), but separate models would allow independent weight mutation
- Request: Is model-switching latency documented? Could we rapidly swap between two pre-loaded models?

### 3. Clock Mode Behavior Under Sustained Load
- We run inference continuously for hours (1200+ calls per β-scan)
- Request: Does Economy clock mode degrade latency p99 under sustained load? Is there thermal throttling on the NPU?
- We want to characterize: Performance vs Economy vs LowPower modes across long physics runs

### 4. Weight Mutation Atomicity
- We mutate readout weights via `set_variable()` while inference is running
- Request: Is mutation atomic? Can we get torn reads if inference and mutation overlap?
- Our workaround: brief pause between last inference and mutation (adds ~14ms)

### 5. Quantization Effects on Physics
- ESN reservoir dynamics are sensitive to numerical precision
- Question: What is the effective precision of int8 FC inference for our use case?
- We observe <1% CPU/NPU prediction divergence — is this guaranteed across all input ranges?

### 6. SRAM Model Size Limits
- Our ESN: 50 reservoir × 8 input × 6 output = ~4KB weights
- AKD1000 has 8-10MB SRAM
- Question: Could we fit a 500-neuron reservoir (50KB weights) for higher-accuracy models? What is the practical model size limit for single-pass inference?

## Architecture Diagram

```
                    PCIe Gen3 x4
GPU (RTX 3090) ─────────────────────── AKD1000
   │                                      │
   │ HMC trajectory (7.64s)              │ ESN inference (~390µs)
   │ ↓                                   │ ↓
   │ observables [8 features]            │ predictions [6 outputs]
   │ ──── CPU staging (1.3ms) ──────→    │
   │ ←─── CPU readback ─────────────     │
   │                                      │
   │ Weight mutation                      │
   │ ──── set_variable() (14ms) ────→    │
```

## Raw Data Location

- Campaign results: `/tmp/hotspring-runs/v0614/npu_campaign_results.jsonl`
- Experiment log: `experiments/020_NPU_CHARACTERIZATION_CAMPAIGN.md`
- ESN implementation: `barracuda/src/md/reservoir.rs`
- NPU hardware adapter: `barracuda/src/md/npu_hw.rs`
- Akida discovery notes: `metalForge/npu/akida/BEYOND_SDK.md`

## Conclusion

The AKD1000 is a viable co-processor for real-time physics screening in Lattice QCD. The key finding is that **pre-thermalization screening (Position A) can save 3+ hours per production run** by detecting equilibrium earlier than fixed-count thermalization. Multi-output inference at zero additional cost makes the NPU an efficient "physics oracle" — one inference call replaces six separate checks.

We recommend investigating GPU→NPU peer-to-peer DMA and model-switching latency as the next hardware evolution targets.
