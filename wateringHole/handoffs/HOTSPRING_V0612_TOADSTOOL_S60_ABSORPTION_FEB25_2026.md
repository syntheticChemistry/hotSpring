# hotSpring v0.6.12 — toadStool S60 Absorption Handoff

**Date**: 2026-02-25
**From**: hotSpring (biomeGate)
**To**: toadStool team, all springs
**toadStool**: Session 60 (DF64 FMA + transcendentals + polyfill hardening)

---

## What Changed

### toadStool S60 Pulled and Absorbed

hotSpring pulled toadStool commit `93a61bb5` (S60) and absorbed the following
into the local HMC pipeline:

| Feature | Source | hotSpring Integration |
|---------|--------|----------------------|
| FMA-optimized `df64_core.wgsl` | toadStool S60 | Via `su3_df64_preamble()` |
| `df64_transcendentals.wgsl` | toadStool S60 | Via `su3_df64_preamble()` |
| DF64 kinetic energy shader | toadStool S58 | Local copy `su3_kinetic_energy_df64.wgsl` |
| DF64 Wilson plaquette shader | **NEW** (hotSpring) | Local `wilson_plaquette_df64.wgsl` |
| NVK allocation guard | toadStool S60 | Available via `GpuDriverProfile` |
| Ada f64 transcendental polyfills | toadStool S60 | Via `compile_shader_f64()` |

### DF64 Coverage Expanded (v0.6.12)

| HMC Kernel | % of Trajectory | DF64 Status | Source |
|------------|-----------------|-------------|--------|
| Gauge force | ~40% | DF64 since v0.6.10 | hotSpring local |
| Wilson plaquette | ~15% | **DF64 since v0.6.12** | hotSpring local (nbr-buffer) |
| Kinetic energy | ~5% | **DF64 since v0.6.12** | toadStool adaptation |
| Momentum update | ~5% | Native f64 | — |
| Link update | ~10% | Native f64 | — |
| CG/Dirac | ~20% | Native f64 | — |
| Random momenta | ~5% | N/A | — |

**Total DF64 coverage: ~60% of HMC** (up from 40% in v0.6.10).

### Benchmark Results (v0.6.12 vs v0.6.11)

| Lattice | v0.6.11 (force only) | v0.6.12 (force+plaq+KE) | Improvement |
|---------|---------------------|-------------------------|-------------|
| 8⁴ | 36.7 ms/traj | 32.2 ms/traj | **12% faster** |
| 16⁴ | 293 ms/traj | 269.9 ms/traj | **8% faster** |

All physics validated: 100% acceptance, correct plaquettes, no precision loss.

---

## What hotSpring Created (for toadStool to absorb)

### 1. `wilson_plaquette_df64.wgsl` (neighbor-buffer variant)

hotSpring's plaquette shader uses a neighbor-buffer binding layout (4 bindings)
rather than toadStool's coordinate-computation layout (3 bindings). hotSpring
created a DF64 variant that:
- Uses the same 4-binding layout: `params, links, nbr, out`
- Leverages `su3_mul_df64`, `su3_adjoint_df64`, `su3_re_trace_df64` from preamble
- Keeps hotSpring's neighbor-buffer indexing for site-ordering independence

**toadStool task**: Consider adding a neighbor-buffer variant of the plaquette
shader as an alternative binding mode. This allows any indexing convention.

### 2. Experiment 015: Mixed Pipeline Findings

- DF64 2x speedup confirmed at 32⁴ production scale (7.6s vs 15.5s per trajectory)
- Per-trajectory rate is β-independent and stable over thousands of trajectories
- DF64 draws 338W vs native f64's 374W (8% less power per trajectory)
- Adaptive NPU steering reduces required β points by ~40% vs uniform scan

### 3. `production_mixed_pipeline.rs` Binary

New production binary demonstrating three-substrate orchestration:
RTX 3090 (DF64) + NpuSimulator (ESN) + Titan V (NVK validation oracle).

---

## toadStool Items Still Pending (from earlier handoffs)

| Item | Status | Reference |
|------|--------|-----------|
| NAK compiler deficiency workarounds | In progress (W-003/W-004) | `NEXT_STEPS.md` |
| Flexible site-indexing (neighbor-buffer support) | Pending | `TOADSTOOL_SITE_INDEXING_NAK_SOLVER_HANDOFF_FEB25_2026.md` |
| NAK native Rust solver in toadStool | Pending | Same handoff |
| DF64 for momentum update | Not started | Expansion roadmap |
| DF64 for link update | Not started | Expansion roadmap |
| DF64 for CG/Dirac | Not started | Expansion roadmap |

---

## Cross-References

- Experiment 014: `experiments/014_DF64_UNLEASHED_BENCHMARK.md`
- Experiment 015: `experiments/015_MIXED_PIPELINE_BENCHMARK.md`
- toadStool spec: `specs/HYBRID_FP64_CORE_STREAMING.md`
- Previous handoff: `HOTSPRING_V0611_BARRACUDA_EVOLUTION_HANDOFF_FEB25_2026.md`
