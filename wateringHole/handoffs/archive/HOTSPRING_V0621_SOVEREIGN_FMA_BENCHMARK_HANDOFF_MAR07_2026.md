SPDX-License-Identifier: AGPL-3.0-only

# hotSpring v0.6.21 — Sovereign FMA Benchmark & Cross-Spring Evolution Handoff

**Date:** March 7, 2026
**From:** hotSpring v0.6.21 (731 tests, 99 binaries, 75 WGSL shaders)
**To:** toadStool (S128+), barraCuda (v0.3.3), coralReef (Phase 9)

---

## Summary

hotSpring v0.6.21 adds a sovereign compiler FMA fusion benchmark (`bench_sovereign_fma`)
that quantifies the optimization impact of barraCuda's sovereign compiler across all
hotSpring WGSL shaders. The cross-spring evolution document has been updated with
detailed provenance data for all five springs.

## FMA Fusion Benchmark Results

46 of 74 shaders parse standalone; 28 require `ShaderTemplate` preprocessing.

| Category | Shaders | FMA Fusions | Avg/shader |
|----------|---------|-------------|------------|
| lattice | 10 | 252 | 25.2 |
| nuclear | 10 | 96 | 9.6 |
| md | 16 | 129 | 8.1 |
| lattice-cg | 7 | 12 | 1.7 |
| md-esn | 2 | 7 | 3.5 |
| **TOTAL** | **46** | **498** | **10.8** |

Top FMA winners: `su3_link_update_f64` (73), `dirac_staggered_f64` (54),
`wilson_plaquette_f64` (45), `staggered_fermion_force_f64` (36).

Zero dead expressions — hotSpring's shaders are well-maintained.

## New Binary

- `bench_sovereign_fma` — feeds all WGSL shaders through `fma_fusion::fuse_multiply_add`
  and `dead_expr::eliminate`, reports per-shader and per-category stats.

## Debt Scan Results

| Check | Result |
|-------|--------|
| TODO/FIXME/HACK/XXX | **Zero** |
| `unwrap()` in lib code | **Zero** (all in bins/tests) |
| `unsafe` blocks | **Zero** |
| Stale feature flags | **None** |
| Files >1000 lines | **None** (largest: `dielectric.rs` at 990) |
| `#[allow(dead_code)]` | 8 items, all reserved for GPU evolution |

## Cross-Spring Evolution

Updated `specs/CROSS_SPRING_EVOLUTION.md` with:
- FMA fusion benchmark table
- Spring contribution summary (~100 hotSpring, ~80 wetSpring, ~34 neuralSpring, ~5 groundSpring, ~15 airSpring)
- Cross-pollination chains documenting shader lineage across all five springs

## Remaining P1 Adoption Opportunities

| API | Benefit |
|-----|---------|
| `GpuView<T>` + buffer-resident stats | 80×–600× for stats reductions (zero-copy MD) |
| `CorrelationF64::r_squared()` | Single-dispatch model validation |
| `AutocorrelationF64` for heat ACF | GPU path for `compute_heat_acf` |
| `BatchedOdeRK4` / `OdeSystem` | Replace local TTM RK4 with generic ODE |

## Validation

- `cargo fmt --check`: zero issues
- `cargo clippy --all-targets`: zero warnings
- `cargo test --lib`: 731 passed, 0 failed
- `cargo build --bins`: 99 binaries compiled
