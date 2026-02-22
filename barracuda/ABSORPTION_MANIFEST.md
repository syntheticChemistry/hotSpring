# hotSpring → BarraCUDA/ToadStool Absorption Manifest

**Date:** February 22, 2026
**Version:** v0.6.3
**License:** AGPL-3.0-only

---

## Biome Model

hotSpring is a biome. ToadStool/barracuda is the fungus — present in every
biome. hotSpring, neuralSpring, desertSpring each lean on toadstool separately,
evolve shaders and systems locally, and toadstool absorbs what works. Springs
don't import each other. They learn by reviewing code in `ecoPrimals/`.

## Absorption Pattern

hotSpring follows Write → Absorb → Lean:

1. **Write**: Implement physics on CPU with WGSL templates in Rust source
2. **Validate**: Test against Python baselines and known physics
3. **Hand off**: Document in `wateringHole/handoffs/` with code locations
4. **Absorb**: ToadStool absorbs as GPU shaders, BarraCUDA absorbs as ops
5. **Lean**: hotSpring rewires to upstream, deletes local code

---

## Already Absorbed (Lean Phase)

These were written by hotSpring and absorbed by toadstool/barracuda:

| Component | Session | Upstream Location | hotSpring Status |
|-----------|---------|-------------------|------------------|
| Complex f64 WGSL | S18-25 | `shaders/math/complex_f64.wgsl` | Leaning on upstream |
| SU(3) WGSL | S18-25 | `shaders/math/su3.wgsl` | Leaning on upstream |
| Wilson plaquette | S18-25 | `shaders/lattice/wilson_plaquette_f64.wgsl` | Leaning on upstream |
| SU(3) HMC force | S18-25 | `shaders/lattice/su3_hmc_force_f64.wgsl` | Leaning on upstream |
| Abelian Higgs HMC | S18-25 | `shaders/lattice/higgs_u1_hmc_f64.wgsl` | Leaning on upstream |
| CellListGpu fix | S25 | `barracuda::ops::md::neighbor` | Local deprecated |
| NAK eigensolve | S16 | `batched_eigh_nak_optimized_f64.wgsl` | Leaning on upstream |
| ReduceScalarPipeline | S12 | `barracuda::pipeline` | Leaning on upstream |
| GpuDriverProfile | S15 | `barracuda::device::capabilities` | Leaning on upstream |
| WgslOptimizer | S15 | `barracuda::shaders` | Leaning on upstream |

---

## Ready for Absorption (Category C)

These modules are self-contained, well-tested, documented, and follow
the absorption pattern. Each has WGSL templates (where applicable),
CPU reference implementations, and validation suites.

### Tier 1 — High Priority (GPU acceleration benefit)

| Module | Location | WGSL | Tests | What it does |
|--------|----------|------|-------|--------------|
| CSR SpMV | `spectral/csr.rs` | `WGSL_SPMV_CSR_F64` | 8/8 | Sparse matrix-vector product (f64) |
| Lanczos | `spectral/lanczos.rs` | Uses SpMV | 6/6 | Krylov eigensolve with reorthogonalization |
| Staggered Dirac | `lattice/dirac.rs` | `WGSL_DIRAC_STAGGERED_F64` | 8/8 | SU(3) staggered fermion operator |
| CG Solver | `lattice/cg.rs` | `WGSL_COMPLEX_DOT_RE_F64`, `WGSL_AXPY_F64`, `WGSL_XPAY_F64` | 9/9 | Conjugate gradient for D†D |
| ESN Reservoir | `md/reservoir.rs` | `esn_reservoir_update.wgsl`, `esn_readout.wgsl` | 16+ | Echo State Network for transport/phase prediction |

### Tier 2 — Medium Priority (CPU → upstream library)

| Module | Location | Tests | What it does |
|--------|----------|-------|--------------|
| Anderson 1D/2D/3D | `spectral/anderson.rs` | 31 | Anderson localization + Lyapunov exponent |
| Hofstadter butterfly | `spectral/hofstadter.rs` | 10 | Almost-Mathieu operator, band counting |
| Level statistics | `spectral/stats.rs` | Tests | GOE→Poisson spacing ratio |
| Sturm eigensolve | `spectral/tridiag.rs` | Tests | Bisection eigenvalues for tridiagonal |
| Screened Coulomb | `physics/screened_coulomb.rs` | 23/23 | Murillo-Weisheit Sturm eigensolve |
| Wilson action | `lattice/wilson.rs` | 12/12 | Plaquettes, staples, gauge force |
| HMC integrator | `lattice/hmc.rs` | Tests | Cayley matrix exp, leapfrog, Metropolis |
| Abelian Higgs | `lattice/abelian_higgs.rs` | 17/17 | U(1)+Higgs (1+1)D HMC |

### Tier 3 — Low Priority (hotSpring-specific, but reusable patterns)

| Module | Location | Notes |
|--------|----------|-------|
| BCS GPU | `physics/bcs_gpu.rs` | GPU BCS bisection (corrected shader) |
| Deformed HFB | `physics/hfb_deformed/` | Axially-deformed nuclear structure (refactored module dir) |
| HFB GPU resident | `physics/hfb_gpu_resident/` | Full GPU-resident SCF pipeline (refactored module dir) |
| Stanton-Murillo fits | `md/transport.rs` | Analytical transport coefficient models |
| Tolerance/config pattern | `tolerances/` | 154 centralized constants — reusable module pattern for upstream |

---

## Stays Local (Category A/B)

These are hotSpring-specific infrastructure:

| Module | Why it stays |
|--------|--------------|
| `validation.rs` | hotSpring pass/fail harness |
| `provenance.rs` | hotSpring baseline traceability |
| `tolerances/` | hotSpring validation thresholds |
| `discovery.rs` | hotSpring data path resolution |
| `data.rs` | AME2020 + Skyrme bounds |
| `prescreen.rs` | Nuclear EOS cascade filter |
| `bench/` | hotSpring benchmark harness (mod, hardware, power, report) |
| `error.rs` | HotSpringError (wraps BarracudaError) |

---

## WGSL Shader Inventory (for ToadStool)

All WGSL shaders in hotSpring, organized by absorption status:

### Already Absorbed

- `complex_f64.wgsl` → `shaders/math/complex_f64.wgsl`
- `su3.wgsl` → `shaders/math/su3.wgsl`
- `wilson_plaquette_f64.wgsl` → `shaders/lattice/`
- `su3_hmc_force_f64.wgsl` → `shaders/lattice/`
- `higgs_u1_hmc_f64.wgsl` → `shaders/lattice/`
- `batched_eigh_nak_optimized_f64.wgsl` → upstream shader

### Ready for Absorption

- `spmv_csr_f64.wgsl` (in `spectral/shaders/`)
- `dirac_staggered_f64.wgsl` (in `lattice/shaders/`)
- `complex_dot_re_f64.wgsl` (in `lattice/shaders/`)
- `axpy_f64.wgsl` (in `lattice/shaders/`)
- `xpay_f64.wgsl` (in `lattice/shaders/`)
- `esn_reservoir_update.wgsl` (in `md/shaders/`)
- `esn_readout.wgsl` (in `md/shaders/`)

### Local (physics-specific, not general)

- 14 physics shaders in `physics/shaders/` (HFB, BCS, deformed, SEMF, chi², spin-orbit)
- 11 MD production shaders in `md/shaders/` (Yukawa, cell-list, VV, RDF, ESN)

---

## NPU Substrate Discovery + Bridge (forge v0.2.0)

hotSpring's metalForge forge crate (v0.2.0, 19 tests) provides working
cross-substrate discovery, capability-based dispatch, physics workload
profiles, and a barracuda device bridge that toadstool doesn't have yet.

| Component | Location | What it does |
|-----------|----------|--------------|
| NPU probe | `forge/src/probe.rs::probe_npus()` | Discovers `/dev/akida*` + PCIe sysfs vendor scan + SRAM reporting |
| CPU probe | `forge/src/probe.rs::probe_cpu()` | `/proc/cpuinfo` + `/proc/meminfo` discovery |
| GPU probe | `forge/src/probe.rs::probe_gpus()` | wgpu adapters with VRAM from `adapter.limits()` |
| Capability model | `forge/src/substrate.rs` | 12-variant enum (F64Compute, QuantizedInference, BatchInference, ...) |
| Dispatch | `forge/src/dispatch.rs` | Routes workloads by capability (GPU > NPU > CPU) |
| **Profiles** | `forge/src/dispatch.rs::profiles` | Physics workloads: MD force, HFB eigensolve, lattice CG, ESN NPU/GPU, spectral SpMV, CPU validation, hetero phase classifier |
| **Bridge** | `forge/src/bridge.rs` | **Absorption seam**: forge substrate ↔ barracuda `WgpuDevice` |

The bridge module is the explicit absorption point:
- `create_device()` — forge substrate → barracuda `WgpuDevice` (via `from_adapter_index`)
- `best_f64_gpu()` — inventory scan → best f64-capable substrate
- `substrate_from_device()` — existing barracuda device → forge substrate

### Absorption targets for toadstool

| Forge Module | Toadstool Target | What to absorb |
|-------------|------------------|----------------|
| `substrate::Capability` | `device::unified::Capability` | Merge 12 forge variants into toadstool's 4 |
| `probe::probe_cpu()` | `substrate::Substrate::discover_all()` | Add CPU substrate to toadstool discovery |
| `probe::probe_npus()` | `device::akida` | PCIe sysfs vendor scan + `/dev` detection + SRAM reporting |
| `probe::probe_gpus()` | `substrate::Substrate::discover_all()` | VRAM via `adapter.limits()` + feature-to-capability mapping |
| `dispatch::route()` | `toadstool_integration::select_best_device()` | Capability-set routing complements `HardwareWorkload` |
| `dispatch::profiles` | — | Physics workload definitions for hotSpring domains |

See `wateringHole/handoffs/HOTSPRING_V061_FORGE_HANDOFF_FEB21_2026.md` for the
forge handoff with hardware measurements and validation results.

---

## Handoff Procedure

For each absorption candidate:

1. Open issue in toadstool describing the primitive
2. Create handoff doc in `wateringHole/handoffs/`
3. Include: Rust source, WGSL template, binding layout, dispatch geometry, test suite
4. After absorption: rewire hotSpring to `use barracuda::ops::*`, delete local code
5. Run `validate_all` to confirm 33/33 suites still pass
