SPDX-License-Identifier: AGPL-3.0-only

# hotSpring → toadStool: Deep Debt Audit + Absorption Handoff

**Date:** 2026-03-02
**From:** hotSpring v0.6.15
**To:** toadStool/barracuda team
**Covers:** Two-wave deep debt audit, code quality standards, capability-based GPU discovery, production pipeline decomposition, absorption recommendations
**License:** AGPL-3.0-only

---

## Executive Summary

- hotSpring v0.6.15 completed a comprehensive two-wave deep debt audit
- 711 tests, 0 clippy warnings, 0 files >1000 lines, 0 unsafe code
- Capability-based GPU discovery replaces hardcoded adapter names
- 6 oversized binaries decomposed into focused library modules
- All library structs documented, all errors typed, streaming I/O adopted
- Ready for ToadStool/BarraCuda team to absorb the mature patterns

---

## What hotSpring Evolved (for BarraCUDA absorption)

### 1. GPU Adapter Discovery (gpu/adapter.rs)

- `discover_best_adapter()` — enumerates wgpu adapters, selects by memory size and SHADER_F64
- `discover_primary_and_secondary_adapters()` — returns (primary, secondary) with env var override
- This pattern should be absorbed into barracuda::gpu or barracuda::device as a standard primitive
- Env vars HOTSPRING_GPU_PRIMARY/SECONDARY still work as overrides

### 2. Production Pipeline Architecture (production/ module)

- `production.rs` — MetaRow, BetaResult, AttentionState, shared helpers
- `production/npu_worker.rs` — 11-head NPU worker thread pattern (NpuRequest/NpuResponse over mpsc)
- `production/beta_scan.rs` — Quenched NPU types + run_beta_points_npu
- `production/titan_worker.rs` — Secondary GPU validation worker
- `production/cortex_worker.rs` — CPU cortex proxy worker
- `production/dynamical_bootstrap.rs` — GPU/NPU/Titan/cortex worker acquisition
- `production/dynamical_summary.rs` — Summary printing and JSON output
- `production/mixed_summary.rs` — Quenched mixed pipeline summary
- `production/titan_validation.rs` — Titan V validation

These are specific to hotSpring's lattice QCD workflow but the **patterns** (typed worker messages over channels, multi-substrate worker acquisition, structured result types) are universal.

### 3. NPU Experiment Infrastructure (npu_experiments/ module)

- TrajectoryRecord, PlacementResult, MultiOutputMetrics types
- generate_trajectory_data, build_*_dataset, evaluate_*
- 6 placement strategies (pre-therm, mid-traj, post-traj, inter-beta, pre-run, combined)
- characterize_npu_behavior returns Result<CharacterizationResults, HotSpringError>
- Reusable across any spring that needs NPU experiment campaigns

### 4. Complex Polyakov Average (lattice/wilson.rs)

- `Lattice::complex_polyakov_average() -> (f64, f64)` returning (Re, Im)
- Complements existing `average_polyakov_loop() -> f64` (magnitude only)
- Ready for upstream lattice module absorption

### 5. ESN Benchmark Infrastructure (bench/esn_benchmark.rs)

- GpuEsn — GPU ESN inference via WGSL (reservoir + readout shaders)
- SubstrateResult — per-substrate timing
- generate_test_sequence, generate_training_data, time_fn helpers
- For benchmarking CPU vs GPU vs NPU ESN inference

### 6. Nuclear EOS Helpers (nuclear_eos_helpers.rs)

- print_comparison_summary, print_reference_baselines, run_deep_residual_analysis
- l1_objective_nmp, make_l1_objective_nmp
- Reusable NMP optimization infrastructure

---

## Error Handling Evolution

- HotSpringError now has IoError(std::io::Error) and JsonError(serde_json::Error) variants
- All production functions return Result instead of empty Vec on error
- From<std::io::Error> and From<serde_json::Error> implemented for ? propagation

---

## Code Quality Standards Achieved

- `#![deny(clippy::expect_used, clippy::unwrap_used)]` enforced crate-wide in lib
- All test code uses `#[allow(clippy::unwrap_used)]` explicitly
- All f64 sorting uses total_cmp() (zero partial_cmp().unwrap())
- All JSON serialization uses unwrap_or_default() for graceful degradation
- All file I/O in production code uses BufReader streaming
- All `#[allow(missing_docs)]` removed from library code, replaced with real docs
- Zero hardcoded primal names in output strings
- All GPU adapter selection by capability, not name

---

## Sovereignty Compliance

- GPU discovery by memory/capability, not "titan" or "4070"
- No hardcoded "/tmp/" paths — uses std::env::temp_dir()
- No hardcoded primal names in output — generic descriptions ("GPU", "Validation GPU")
- Environment variable overrides preserved for manual control

---

## Absorption Recommendations for ToadStool

### Priority 1 (absorb directly)

| # | Target | Destination |
|---|--------|-------------|
| 1 | GPU adapter discovery pattern | barracuda::gpu::discover_best_adapter() |
| 2 | complex_polyakov_average | barracuda::ops::lattice::Lattice |
| 3 | Error type variants (IoError, JsonError) | barracuda::error |

### Priority 2 (adapt and absorb)

| # | Target | Destination |
|---|--------|-------------|
| 1 | Multi-substrate worker pattern (typed messages over mpsc) | barracuda::pipeline |
| 2 | ESN GPU inference (WGSL reservoir + readout) | barracuda::ops::esn or barracuda::nn |
| 3 | NPU experiment framework | barracuda::bench or barracuda::npu |

### Priority 3 (reference, don't absorb)

| # | Target | Reason |
|---|--------|--------|
| 1 | Production pipeline orchestration | hotSpring-specific HMC workflow |
| 2 | Nuclear EOS helpers | domain-specific |
| 3 | NpuSimulator/NpuHardware | already in hotSpring, NPU-specific |

---

## File Inventory

| Module | Lines | Purpose |
|--------|-------|---------|
| gpu/adapter.rs | ~160 | Capability-based GPU discovery |
| production.rs | ~580 | Shared production types and helpers |
| production/npu_worker.rs | 1000 | 11-head dynamical NPU worker |
| production/beta_scan.rs | ~600 | Quenched NPU β-scan |
| production/titan_worker.rs | ~150 | Secondary GPU worker |
| production/cortex_worker.rs | ~50 | CPU cortex worker |
| production/dynamical_bootstrap.rs | ~290 | Worker acquisition |
| production/dynamical_summary.rs | ~375 | Summary/JSON output |
| production/mixed_summary.rs | ~285 | Quenched summary |
| production/titan_validation.rs | ~130 | Titan V validation |
| npu_experiments/mod.rs | ~745 | NPU experiment types + helpers |
| npu_experiments/placements.rs | ~290 | Placement strategies |
| nuclear_eos_helpers.rs | ~780 | Nuclear EOS helpers |
| bench/esn_benchmark.rs | ~245 | ESN benchmark infrastructure |

---

## Validation

- 711 tests pass
- 0 clippy warnings (lib + bins)
- 0 files >1000 lines
- 0 unsafe code
- All library structs documented
- All errors typed with HotSpringError variants
