# hotSpring v0.6.2 — Deep Debt Resolution & Pedantic Clean

**Date:** 2026-02-21
**From:** hotSpring (computational physics biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only

---

## Summary

v0.6.2 is a deep technical debt resolution release. No new physics, no new
validation suites — pure code quality, coverage, and evolution.

| Metric | v0.6.1 | v0.6.2 | Δ |
|--------|--------|--------|---|
| Clippy warnings (all targets) | ~1500 pedantic | **0** | −1500 |
| Unit tests | 505 | **638** | +133 |
| Region coverage | 65.7% | **72.4%** | +6.7% |
| Function coverage | 77.6% | **82.5%** | +4.9% |
| Local cell-list code | 282 lines + 3 shaders | **0** (upstream) | −282 lines, −3 files |
| Inline WGSL | 5 embedded in .rs | **0** (all in .wgsl) | +5 .wgsl files |
| Duplicate math | Gaussian elimination | **0** (uses solve_f64) | −60 lines |
| GPU energy pipeline | Stub | **Wired** (feature-gated) | ~200 lines of wiring |

---

## 1. Clippy: ~1500 → 0 warnings

All pedantic + nursery warnings resolved across **all targets** (lib, bins, tests).
The lint configuration is now centralized in `Cargo.toml` workspace lints:

```toml
[workspace.lints.clippy]
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
cast_precision_loss = "allow"   # usize→f64: indices ≤ 100k, within 2^52
similar_names = "allow"         # physics: κ, Γ, λ
# ... 20+ documented allows
```

`lib.rs` retains only `#![deny(clippy::expect_used, clippy::unwrap_used)]`.

### Categories resolved

| Category | Count | Method |
|----------|-------|--------|
| `mul_add` | 150+ | Manual IEEE 754 fused multiply-add |
| `doc_markdown` | 600+ | Backtick-wrapped identifiers |
| `must_use_candidate` | 186+ | Auto-fix + manual |
| `imprecise_flops` | 30+ | `cbrt`, `hypot`, `ln_1p` |
| `use_self` | 14 | `Self` keyword |
| `option_if_let_else` | 5 | `map_or_else` |
| `const_fn` | 4 | `lcg_step`, `next_u64`, etc. |
| `float_cmp` (test) | 14 | `#[allow]` on known-value checks |
| `single_char_pattern` | 5 | `'s'` not `"s"` |

---

## 2. CellListGpu Migration (P1 Evolution Target)

**Completed.** Local `GpuCellList` (282 lines + 3 WGSL shaders) replaced with
upstream `barracuda::ops::md::CellListGpu`.

### What changed

| Before | After |
|--------|-------|
| `GpuCellList::new(&gpu, n, box_side, cutoff)` | `CellListGpu::new(gpu.to_wgpu_device(), n, [box_side; 3], cutoff)?` |
| `gpu_cl.build(&gpu, &pos_buf)` | `gpu_cl.build(&pos_buf)?` |
| `gpu_cl.cell_start_buf` | `gpu_cl.cell_start()` |
| `gpu_cl.mx` / `gpu_cl.nc` | `gpu_cl.grid().0` / `gpu_cl.n_cells()` |

### Files deleted

- `src/md/shaders/cell_bin_f64.wgsl` — upstream `atomic_cell_bin.wgsl`
- `src/md/shaders/exclusive_prefix_sum.wgsl` — upstream `prefix_sum.wgsl`
- `src/md/shaders/cell_scatter.wgsl` — upstream `cell_list_scatter.wgsl`

### Compatibility

Force shader `yukawa_force_celllist_indirect_f64.wgsl` unchanged — bindings
4/5/6 (cell_start, cell_count, sorted_indices) match upstream buffer format.

---

## 3. Inline WGSL Extraction

5 inline shader strings extracted to dedicated `.wgsl` files:

| Module | Const | New file |
|--------|-------|----------|
| `lattice/cg.rs` | `WGSL_COMPLEX_DOT_RE_F64` | `lattice/shaders/complex_dot_re_f64.wgsl` |
| `lattice/cg.rs` | `WGSL_AXPY_F64` | `lattice/shaders/axpy_f64.wgsl` |
| `lattice/cg.rs` | `WGSL_XPAY_F64` | `lattice/shaders/xpay_f64.wgsl` |
| `lattice/dirac.rs` | `WGSL_DIRAC_STAGGERED_F64` | `lattice/shaders/dirac_staggered_f64.wgsl` |
| `spectral/csr.rs` | `WGSL_SPMV_CSR_F64` | `spectral/shaders/spmv_csr_f64.wgsl` |

All now loaded via `include_str!("shaders/filename.wgsl")`.

---

## 4. Duplicate Math Eliminated

`md/reservoir.rs` contained a hand-rolled Gaussian elimination (~60 lines).
Replaced with `barracuda::linalg::solve_f64` — one call per RHS column.
Zero local linear algebra code remains.

---

## 5. GPU Energy Pipeline Wired

`batched_hfb_energy_f64.wgsl` dispatch wired in `hfb_gpu_resident/mod.rs`
behind `#[cfg(feature = "gpu_energy")]`:

- `compute_energy_integrands` + `compute_pairing_energy` GPU passes
- Staging buffer readback with trapezoidal sum
- CPU fallback preserved when feature disabled

---

## 6. Test Coverage

133 new tests added across 15 modules:

| Module | New Tests | Focus |
|--------|-----------|-------|
| `hfb_deformed/potentials.rs` | 14 | Coulomb, mean-field, energy, Q20 |
| `hfb_deformed/basis.rs` | 10 | Quantum numbers, wavefunctions |
| `hfb_deformed/mod.rs` | 9 | Grid, density, BCS, result |
| `physics/hfb/mod.rs` | 11 | BCS, density, SEMF/HFB paths |
| `prescreen.rs` | 6 | Cascade filter edge cases |
| `hfb_gpu_types.rs` | 6 | Uniform construction, layout |
| `data.rs` | 8 | JSON parsing edge cases |
| `bench/hardware.rs` | 5 | RAPL, CPU/GPU probing |
| `md/observables/ssf.rs` | 4 | Single particle, zero-k |
| `screened_coulomb.rs` | 4 | Bound states, screening |
| Others | 56 | bench, error, discovery, etc. |

---

## 7. Refactoring

| File | Before | After | Method |
|------|--------|-------|--------|
| `bench.rs` (monolithic) | 1005 | 4 files: 193+218+354+246 | Module decomposition |
| `hfb_gpu_resident/mod.rs` | 1 function, 1100 lines | 7 helpers + orchestrator | Function extraction |
| `celllist_diag.rs` | 1156 | 951 | Shared helpers |
| `celllist.rs` | 947 | 665 | Upstream migration |
| `lib.rs` | 69 | 59 | Removed redundant allows |

---

## 8. Evolution Gaps Updated

| Gap | Status |
|-----|--------|
| ~~GPU energy integrands~~ | ✅ Wired (v0.6.2, feature-gated) |
| ~~Local GpuCellList~~ | ✅ Migrated to upstream (v0.6.2) |
| ~~Inline WGSL in lattice/spectral~~ | ✅ Extracted to .wgsl files |
| ~~Duplicate Gaussian elimination~~ | ✅ Replaced with `solve_f64` |
| Fully GPU-resident Lanczos | Pending (P2) — GPU dot+axpy+scale |
| Deformed HFB GPU H-build | Pending — shaders exist, unwired |

---

## 9. Codebase Health

| Check | Result |
|-------|--------|
| `cargo fmt --check` | Clean |
| `cargo clippy --all-targets` | **0 warnings** |
| `cargo doc --no-deps` | **0 warnings** |
| `cargo test` | **638 lib + 24 integration + 8 forge** |
| `unsafe` blocks | **0** |
| `expect()`/`unwrap()` in library | **0** |
| TODOs/FIXMEs | **0** |
| Version | **0.6.2** |
