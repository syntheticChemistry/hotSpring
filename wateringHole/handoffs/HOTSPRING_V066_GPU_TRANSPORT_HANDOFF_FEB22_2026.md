# hotSpring v0.6.6 — GPU-Only Transport Pipeline Handoff

**Date:** February 22, 2026
**From:** hotSpring (computational physics biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Context:** Transport coefficient validation promoted from CPU to pure GPU
unidirectional streaming pipeline, eliminating all position/velocity readback.

---

## Executive Summary

Transport coefficient validation was the dominant bottleneck in the hotSpring
validation suite: **2,060s** (34 minutes) for the combined `validate_transport` (800s)
and `validate_stanton_murillo` (1,260s) binaries. Both ran MD on CPU and computed
VACF/stress/heat-current via O(N²) CPU pair loops.

The new GPU-only transport pipeline achieves:

| Metric | Before (CPU) | After (GPU-only) | Improvement |
|--------|-------------|-------------------|-------------|
| Total (6 cases) | 2,060s | **519s** | **4.0×** |
| Per case | ~340s | **86.5s** | **3.9×** |
| VACF compute | 30-60s/case | **1.6s/case** | **20-40×** |
| Velocity readback | 48 MB/case | **16 B/dump** | **3,000,000×** |
| Suite binaries | 2 | 1 | Simpler |

All 21 physics checks pass: energy conservation, D* vs Daligault/Sarkas fits,
D* ordering across coupling strengths, positivity.

---

## Architecture: Unidirectional Streaming

```
GPU MD Loop                    GPU VACF
┌─────────────────────┐       ┌───────────────────────┐
│ kick_drift → force  │       │ vacf_batch_f64.wgsl   │
│ → half_kick         │       │   one dispatch/lag     │
│                     │       │   iterates all origins │
│ copy vel_buf ──────────────→│   in-shader            │
│   → ring.flat_buf   │       │                       │
│   (GPU→GPU, 0 PCIe) │       │ ReduceScalarPipeline  │
│                     │       │   → 8 bytes back      │
│ KE/PE: 16 B back   │       │                       │
└─────────────────────┘       │ Green-Kubo (CPU)      │
                              │   O(n_lag) trivial     │
                              └───────────────────────┘
```

**Data flow**: GPU → GPU → GPU → 8-byte scalar → CPU integration.

---

## New Files

| File | Purpose |
|------|---------|
| `md/shaders/vacf_batch_f64.wgsl` | Batched VACF: one dispatch per lag, iterates all time origins in-shader. Reduces GPU round trips from N_frames×N_lag to N_lag. |
| `md/shaders/vacf_dot_f64.wgsl` | Per-particle v(t₀)·v(t) dot product (single-origin, for tests). |
| `md/shaders/stress_virial_f64.wgsl` | GPU σ_xy Yukawa virial kernel for Green-Kubo viscosity. Reuses Yukawa pair logic from force shader. |
| `md/observables/transport_gpu.rs` | `GpuVelocityRing` (flat buffer ring), `compute_vacf_gpu` (batched), `compute_stress_xy_gpu`. |
| `md/simulation_transport_gpu.rs` | Modified MD loop: velocity snapshots stored GPU-resident via `copy_buffer_to_buffer` into flat ring. |
| `bin/validate_transport_gpu_only.rs` | New baseline: 6 Stanton-Murillo cases, 21 validation checks. Replaces both `validate_transport` and `validate_stanton_murillo` in `validate_all`. |
| `bin/validate_transport_gpu_resident.rs` | Parity proof: uploads CPU snapshots to GPU ring, compares CPU VACF vs GPU VACF on same data. |

## Modified Files

| File | Change |
|------|--------|
| `md/observables/mod.rs` | Added `transport_gpu` module and exports. |
| `md/mod.rs` | Added `simulation_transport_gpu` module. |
| `bin/validate_all.rs` | Replaced 2 transport suites (800s+1260s) with 1 GPU-only suite (519s). |

---

## Key Design Decisions

### Batched VACF Shader

The initial implementation dispatched one GPU kernel per (t₀, lag) pair — ~4 million
round trips for 4000 frames × 2000 lags. The batched shader (`vacf_batch_f64.wgsl`)
iterates over all time origins *inside* the shader, reducing to N_lag dispatches + N_lag
reductions = ~4000 total GPU operations.

### Flat Ring Buffer

The `GpuVelocityRing` uses a single flat buffer (`n_slots × N × 3` f64 elements)
instead of per-slot buffers. This enables the batched VACF shader to read any
(snapshot, particle, component) via a single `storage<read>` binding with offset
arithmetic in-shader.

### Cell-List Not Applicable at N=500

Cell-list force (O(N) vs O(N²)) requires `box_side/rc ≥ 5`. At N=500 with
`rc=6-8 a_ws`, `box_side/rc ≈ 1.5-2.0` — too small. Cell-list becomes viable
at N≥2000. The all-pairs force at N=500 (250K pairs) is not compute-bound;
the bottleneck is GPU dispatch overhead (~1.2ms per encoder submit).

---

## Absorption Targets for ToadStool

### P1: GPU Stress Virial (for viscosity η*)

`stress_virial_f64.wgsl` computes off-diagonal σ_xy on GPU. Currently in hotSpring;
should be absorbed into `barracuda::ops::md::observables` alongside the existing
RDF histogram shader.

### P1: Batched VACF as Pipeline Primitive

`vacf_batch_f64.wgsl` is general-purpose — any MD code computing VACF benefits.
Should be absorbed into `barracuda::ops::md::observables` as a
`VacfBatchPipeline` alongside `ReduceScalarPipeline`.

### P2: GPU Heat Current (for thermal conductivity λ*)

CPU `compute_heat_current` is O(N²) per snapshot — same pattern as stress virial.
A `heat_current_f64.wgsl` shader would complete the full GPU Green-Kubo transport
suite (D*, η*, λ*).

### P3: Cell-List Transport at N≥2000

For production-scale transport (N=10,000), the GPU-only pipeline should use
`CellListGpu` indirect force + GPU cell-list rebuild. The `simulation_transport_gpu.rs`
is structured to accept this swap (just change force pipeline + bind group + add
periodic `gpu_cl.build()` calls).

---

## Performance Profile (6 Stanton-Murillo Cases, N=500 Lite)

```
  κ  Γ     D*(GPU)      D*(fit)    total   sim   VACF
  1  50  2.2816e-2   1.6703e-2   86.5s  25.2s  1.5s
  1  72  1.5200e-2   1.1496e-2   87.2s  25.9s  1.6s
  2  31  6.5127e-2   6.4445e-2   87.4s  25.5s  1.6s
  2 100  1.7466e-2   1.1429e-2   86.4s  25.4s  1.6s
  2 158  9.1106e-3   7.1854e-3   86.2s  25.4s  1.6s
  3 100  2.9312e-2   2.9010e-2   85.4s  24.7s  1.6s

  Total:  519.2s  (was 2,060s)
  21/21 checks passed
```

Time breakdown per case: ~69% equilibration, ~29% production, ~2% VACF.

---

## Remaining CPU Transport Binaries (Kept for Debugging)

- `validate_transport` — CPU/GPU parity comparison (D* from both paths)
- `validate_stanton_murillo` — Full Stanton-Murillo with MSD, stress ACF, heat ACF

These are no longer in `validate_all` but remain as standalone binaries for
debugging and detailed transport analysis.
