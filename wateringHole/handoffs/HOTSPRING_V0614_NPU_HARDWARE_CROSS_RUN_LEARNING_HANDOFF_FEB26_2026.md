# hotSpring → toadStool / barracuda — NPU Hardware & Cross-Run Learning Handoff

**Date:** February 26, 2026
**From:** hotSpring (biomeGate)
**To:** toadStool / barracuda core team
**Covers:** Exp 022 NPU offload pipeline, live AKD1000 hardware integration, cross-run ESN learning, barracuda evolution review
**License:** AGPL-3.0-only

---

## Executive Summary

- **Live AKD1000 hardware NPU** is now integrated into the production 32⁴ lattice QCD
  mixed pipeline via PCIe. The `npu-hw` cargo feature links the `akida-driver` crate
  for direct BAR access. This is the first neuromorphic silicon in a lattice QCD pipeline.
- **Cross-run ESN learning** implemented: `--bootstrap-from` loads previous weights or
  trajectory logs; `--save-weights` exports trained models. Each run's ESN accumulates
  knowledge across experiments.
- **NPU worker thread** pattern (dedicated `std::thread` + `mpsc::channel`) prevents
  GPU stalls. Four placement points: thermalization detection, rejection prediction,
  phase classification, adaptive β steering. Overhead: 1.2ms per trajectory (0.016%
  of 7.6s GPU trajectory).
- **8⁴ validation**: 60% thermalization early-exit, 86% rejection accuracy, 5,947
  NPU calls. 32⁴ production run in progress with live hardware.

---

## Part 1: What Changed in barracuda (v0.6.14 additions)

### Modified Files

| File | Change | Impact |
|------|--------|--------|
| `src/bin/production_mixed_pipeline.rs` | NPU worker thread, 4 placements, cross-run bootstrap, trajectory logging | Major: full NPU offload architecture |
| `src/md/reservoir.rs` | `ExportedWeights` serde derives, `NpuSimulator` accessor methods | Enables JSON round-trip for cross-run learning |
| `src/lattice/gpu_hmc/observables.rs` | Action density bug fix: `plaq × 6` → `6(1 − plaq)` | Critical: was biasing ESN training data |

### New Patterns for Upstream

#### 1. NPU Worker Thread Pattern

```rust
fn spawn_npu_worker(
    npu: NpuSimulator,
    rx: mpsc::Receiver<NpuRequest>,
    tx: mpsc::Sender<NpuResponse>,
) -> std::thread::JoinHandle<()>
```

Typed request/response enums over `mpsc` channels. The NPU thread owns the ESN
instance exclusively — no shared mutable state, no locking. The GPU thread sends
observable feature vectors and receives screening decisions without blocking.

**toadStool action:** This pattern generalizes to any co-processor offload. Consider
adopting for GPU-NPU dispatch in `toadstool-core`: a typed channel protocol where
each substrate has a dedicated worker thread.

#### 2. Cross-Run Weight Serialization

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportedWeights {
    pub w_in: Vec<f32>,    // reservoir_size × input_size
    pub w_res: Vec<f32>,   // reservoir_size × reservoir_size
    pub w_out: Vec<f32>,   // output_size × reservoir_size
    pub input_size: usize,
    pub reservoir_size: usize,
    pub output_size: usize,
    pub leak_rate: f32,
}
```

JSON round-trip for ESN weights allows:
- `--save-weights=path.json` → exports final trained model
- `--bootstrap-from=path.json` → loads pre-trained model
- `--bootstrap-from=trajectories.jsonl` → trains from historical log

**toadStool action:** `ExportedWeights` should be a first-class barracuda type.
The pattern of training on one substrate and deploying on another (CPU f64 train
→ NPU int4 deploy) is the standard workflow. Making weights portable between
`EchoStateNetwork`, `NpuSimulator`, and `NpuHardware` is already implemented
in hotSpring; barracuda should own the serialization contract.

#### 3. Trajectory Logging

Per-trajectory JSONL output with fields: `traj_idx`, `beta`, `plaquette`,
`plaquette_var`, `polyakov_mag`, `polyakov_phase`, `acceptance_rate`,
`action_density`, `delta_h`, `accepted`, `is_therm`, `npu_reject_prediction`,
`npu_reject_confidence`.

This format enables:
- Post-hoc ESN training from any previous run
- Cross-run analysis (comparing simulator vs hardware NPU decisions)
- Data provenance (every trajectory logged with its NPU screening result)

---

## Part 2: Live AKD1000 Hardware Integration

### Kernel Module

The `akida-pcie.ko` module was built from source (`control/akida_dw_edma/`)
for kernel 6.17. One API rename was patched: `pcim_iounmap_regions` →
`pcim_iounmap_region` (6.17 API change). Device node: `/dev/akida0` with
world-readable permissions via udev rules.

### Cargo Feature Gate

```toml
[features]
npu-hw = ["akida-driver", "akida-models"]
```

When `npu-hw` is enabled, `production_mixed_pipeline` uses `NpuHardware`
(real Akida) instead of `NpuSimulator`. The same `ExportedWeights` flow
works for both — train on CPU f64, export f32, quantize to int8/int4 on
the AKD1000.

### Hardware Confirmation

```
NPU: AKD1000 (hardware) + worker thread
Bootstrap: loaded 749 data points, β_c estimate = 7.0000
```

The `akida-driver` crate (from `toadstool/crates/neuromorphic/akida-driver`)
successfully:
1. Discovers `/dev/akida0`
2. Maps PCIe BAR
3. Programs NP mesh with ESN weights
4. Runs inference and returns results

---

## Part 3: Barracuda Usage & Evolution Review

### What hotSpring Leans On (upstream barracuda)

| Module | Usage | Validation |
|--------|-------|------------|
| `ops::linalg::lu_solve` | ESN weight solving | 664 tests |
| `ops::linalg::BatchedEighGpu` | HFB nuclear structure | 16/16 |
| `ops::md::CellListGpu` | Yukawa MD cell-list | 9/9 PP |
| `ops::md::observables::SsfGpu` | Static structure factor | Transport pass |
| `spectral::*` | Anderson, Lanczos, Hofstadter | 45/45 |
| `device::WgpuDevice` | GPU adapter, shaders | All GPU bins |
| `pipeline::ReduceScalarPipeline` | GPU reduction | All GPU validation |
| `shaders::precision::ShaderTemplate` | f64 patching | All WGSL |

### What hotSpring Owns (local domain physics)

| Module | Reason | Absorb? |
|--------|--------|:-------:|
| `lattice/` (8 modules + gpu_hmc/) | QCD physics | Shaders: yes |
| `physics/` (HFB, SEMF) | Nuclear structure | Stays local |
| `md/reservoir.rs` | ESN for physics | Shaders: yes |
| `tolerances/`, `provenance/` | Validation infra | Stays local |

### Evolution Health

| Metric | Value | Target | Status |
|--------|-------|--------|:------:|
| clippy warnings | 0 | 0 | ✅ |
| TODOs in .rs | 0 | 0 | ✅ |
| unsafe blocks | 0 | 0 | ✅ |
| .unwrap() in lib | 0 | 0 | ✅ |
| Hardcoded device refs | 0 | 0 | ✅ |
| Test count | ~697 | growing | ✅ |
| Validation suites | 39/39 | all pass | ✅ |
| WGSL shaders | 62 | tracked | ✅ |
| Binaries | 78 | tracked | ✅ |

### WGSL Dedup Status

All inline WGSL duplicates (`abs_f64`, `cbrt_f64`) were eliminated in v0.6.14.
Shaders use `LazyLock<String>` composition with shared libraries. Zero
duplicated math across 62 shaders.

---

## Part 4: What to Absorb from Exp 022

### Immediate

| Item | File | Priority |
|------|------|:--------:|
| `ExportedWeights` serde contract | `md/reservoir.rs` | **P0** |
| NPU worker thread pattern | `bin/production_mixed_pipeline.rs` | P1 |
| Trajectory JSONL format spec | `bin/production_mixed_pipeline.rs` | P1 |
| Action density fix | `lattice/gpu_hmc/observables.rs` | **P0** (bug fix) |

### Medium Priority

| Item | Reason | Priority |
|------|--------|:--------:|
| Cross-run bootstrap protocol | Enables transfer learning across experiments | P1 |
| Typed `NpuRequest`/`NpuResponse` enums | Reusable co-processor dispatch pattern | P2 |
| Thermalization detection heuristics | Domain-specific but generalizable variance-ratio test | P2 |

---

## Part 5: Discoveries for Upstream Evolution

### 1. The NPU Worker Thread Eliminates GPU Stalls

By running ESN inference on a dedicated `std::thread`, the GPU never waits for
NPU decisions. The `mpsc` channel provides natural backpressure. This is the
correct architecture for heterogeneous pipelines: each substrate has its own
thread, communicating via typed channels.

### 2. Cross-Run Learning Closes the Feedback Loop

The `--bootstrap-from` / `--save-weights` pattern transforms the pipeline from
single-shot to iterative. Each run produces both physics results AND a trained
model. The n-th run's ESN starts where the (n-1)-th run's ESN finished.

This has direct implications for toadStool:
- `barracuda::esn::ExportedWeights` should be a stable serialization format
- Weight compatibility across `EchoStateNetwork`, `NpuSimulator`, `NpuHardware`
  must be maintained as a contract
- The trajectory log format should be standardized for cross-spring reuse

### 3. Action Density Bug Was Biasing Training Data

The pre-fix `action_density = plaq × 6` gave ~3.4 for typical plaquette ~0.56.
Correct: `6(1 − plaq)` gives ~2.6. Any ESN trained on the wrong feature vector
would learn a biased representation. All models trained in Exp 022 use the fix.

**toadStool action:** Add `action_density()` as a barracuda lattice primitive
with the correct formula. Springs should not compute this locally.

### 4. Live NPU Validates the Full Stack

The `akida-driver` → `akida-models` → `NpuHardware` → `ExportedWeights` →
`NpuSimulator` → `EchoStateNetwork` chain is now validated end-to-end on real
silicon. The weight export pipeline works: train in f64, quantize to int4,
deploy on AKD1000, get correct physics screening results.

---

## Part 6: Action Items

| # | Action | Owner | Priority |
|---|--------|-------|:--------:|
| 1 | Absorb `ExportedWeights` serde into barracuda core | toadStool | **P0** |
| 2 | Absorb action density fix into barracuda lattice | toadStool | **P0** |
| 3 | Add `action_density()` as barracuda lattice primitive | toadStool | P1 |
| 4 | Standardize trajectory JSONL format | toadStool + hotSpring | P1 |
| 5 | Evaluate NPU worker thread for `toadstool-core` dispatch | toadStool | P1 |
| 6 | Stabilize cross-run weight compatibility contract | toadStool | P1 |
| 7 | Absorb ESN reservoir shaders (Exp 021) | toadStool | P1 |
| 8 | Absorb `prng_pcg_f64.wgsl` (v0.6.14) | toadStool | P1 |

---

## Closing

Experiment 022 marks the transition from "NPU exploration" to "NPU production."
The AKD1000 is no longer a characterization target — it's a working physics
co-processor in a three-substrate lattice QCD pipeline. The cross-run learning
loop ensures the ESN improves with each experiment, accumulating knowledge that
transfers between runs and between simulator/hardware substrates.

The barracuda crate is at its cleanest state (0 clippy, 0 unsafe, 0 TODOs).
The 3 P0 items (ExportedWeights serde, action density fix, prng shader) are
ready for immediate absorption. The NPU worker thread pattern and trajectory
logging format are reusable across all springs running heterogeneous pipelines.
