# hotSpring ↔ toadStool — S68 Sync Status

**Date:** February 27, 2026
**From:** hotSpring
**To:** toadStool / barracuda core team
**toadStool HEAD:** `6b64b6e6` (S68+++)
**hotSpring HEAD:** `2a38a71` (v0.6.15)
**License:** AGPL-3.0-only

---

## Summary

Pulled toadStool through S68+++. Massive evolution: 8 sessions (S61–S68+++),
700 WGSL shaders (0 f32-only), universal precision architecture, sovereign
compiler with naga IR rewrite, chrono elimination, ~400 lines dead code removed.

**Bug found and fixed**: `gaussian_fermion_f64.wgsl` had f32 cos(theta) which
fails naga validation when math_f64 preamble rewrites cos → cos_f64. Fixed
in both toadStool and hotSpring copies. `su3_random_momenta_f64.wgsl` was
already fixed upstream in S68.

---

## Part 1: What toadStool S61–S68 Absorbed From hotSpring

### S64: Lattice QCD Shaders (8 files)

| Shader | Origin |
|--------|--------|
| `su3_math_f64.wgsl` | hotSpring lattice/su3.rs |
| `prng_pcg_f64.wgsl` | hotSpring lattice/constants.rs |
| `su3_f64.wgsl` | hotSpring lattice/su3.rs |
| `su3_gauge_force_df64.wgsl` | hotSpring DF64 evolution |
| `su3_kinetic_energy_df64.wgsl` | hotSpring DF64 evolution |
| `axpy_f64.wgsl` | hotSpring lattice/cg.rs |
| `complex_dot_re_f64.wgsl` | hotSpring lattice/cg.rs |
| `xpay_f64.wgsl` | hotSpring lattice/cg.rs |

### S66: Cross-Spring Patterns

- `NeighborMode` (from hotSpring P1 lattice patterns)
- `stats::regression`, `stats::hydrology`, `stats::moving_window_f64`

### S58: DF64 Lattice Shaders

- `wilson_plaquette_df64.wgsl`, `wilson_action_df64.wgsl`
- `kinetic_energy_df64.wgsl`, `su3_hmc_force_df64.wgsl`

---

## Part 2: What hotSpring Can Now Lean On

### Ready to Rewire (Next Session)

| hotSpring Local | toadStool Upstream | Benefit |
|-----------------|-------------------|---------|
| `lattice/shaders/*.wgsl` (22 files) | `barracuda::ops::lattice::absorbed_shaders` | Delete local copies, use upstream |
| `lattice/cg.rs` WGSL constants | `barracuda::ops::lattice::cg::*` | Use upstream CG shaders |
| `lattice/dirac.rs` | `barracuda::ops::lattice::dirac::DiracGpuLayout` | Use upstream Dirac |
| `md/reservoir.rs` ESN | `barracuda::esn_v2::ESN` | Use upstream ESN (but keep multi-head extension local) |
| Manual DF64 preamble | `compile_shader_df64()` | Use universal precision API |

### New Capabilities Available

| Capability | Impact |
|-----------|--------|
| `NeighborMode::PrecomputedBuffer` | Precompute 4D neighbor tables for repeated HMC |
| `compile_shader_universal(source, Precision)` | Single source for f32/f64/df64 |
| Sovereign compiler (FMA fusion, DCE) | Automatic shader optimization |
| `gpu_hmc_trajectory` orchestration | Full dynamical HMC trajectory upstream |
| `gpu_cg_solver` | CG solver orchestration upstream |
| ESN v2 with `to_npu_weights()` | Direct GPU-train → NPU-deploy |

---

## Part 3: Breaking Changes to Track

| Change | Status |
|--------|--------|
| chrono → std::time (S68+++) | hotSpring doesn't use toadstool time types directly — no impact |
| `Precision::Df64` enum variant | New; existing code unaffected |
| log → tracing | hotSpring uses own logging — no impact |
| 296 f32 WGSL files deleted | hotSpring uses local copies or f64 — no impact |

---

## Part 4: Rewiring Plan (Pinned for Next Session)

**Phase 1** (Low risk, high value):
1. Replace local lattice shaders with upstream `absorbed_shaders::*`
2. Switch to `compile_shader_df64()` for DF64 shader compilation
3. Adopt `NeighborMode::PrecomputedBuffer` for production HMC

**Phase 2** (Medium risk):
4. Migrate CG solver to upstream `gpu_cg_solver`
5. Migrate HMC trajectory to upstream `gpu_hmc_trajectory`
6. Adopt ESN v2 (keep multi-head local, use upstream reservoir)

**Phase 3** (Future):
7. Full lean — delete all local lattice code, lean entirely on upstream

Each phase validated by running the existing test suite + a short β-scan.
