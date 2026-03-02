# hotSpring → BarraCuda/ToadStool Absorption Manifest

**Date:** March 2, 2026
**Version:** v0.6.15 (synced to toadStool S78)
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
4. **Absorb**: ToadStool absorbs as GPU shaders, BarraCuda absorbs as ops
5. **Lean**: hotSpring rewires to upstream, deletes local code

---

## Already Absorbed (Lean Phase)

These were written by hotSpring and absorbed by toadstool/barracuda:

| Component | Session | Upstream Location | hotSpring Status |
|-----------|---------|-------------------|------------------|
| **Spectral module** | S25-31h | `barracuda::spectral` | **Fully leaning** — local sources deleted, re-exports from upstream |
| CSR SpMV + WGSL | S25-31h | `barracuda::spectral::SpectralCsrMatrix` | Leaning (alias `CsrMatrix` for compat) |
| Lanczos eigensolve | S25-31h | `barracuda::spectral::lanczos` | Leaning on upstream |
| Anderson 1D/2D/3D | S25-31h | `barracuda::spectral::anderson` | Leaning on upstream |
| Hofstadter butterfly | S25-31h | `barracuda::spectral::hofstadter` | Leaning on upstream |
| Sturm tridiagonal | S25-31h | `barracuda::spectral::tridiag` | Leaning on upstream |
| Level statistics | S25-31h | `barracuda::spectral::stats` | Leaning on upstream |
| BatchIprGpu | S25-31h | `barracuda::spectral::BatchIprGpu` | **NEW** — available via re-export |
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
| Staggered Dirac + CG | S31d | `ops/lattice/dirac.rs`, `ops/lattice/cg.rs` | **Fully absorbed upstream** — Dirac 8/8, CG 9/9 |

---

## Ready for Absorption (Category C)

These modules are self-contained, well-tested, documented, and follow
the absorption pattern. Each has WGSL templates (where applicable),
CPU reference implementations, and validation suites.

### Tier 1 — High Priority (GPU acceleration benefit)

| Module | Location | WGSL | Tests | What it does |
|--------|----------|------|-------|--------------|
| ~~Staggered Dirac~~ | ~~`lattice/dirac.rs`~~ | — | — | ✅ **Absorbed** (S31d) — `ops/lattice/dirac.rs` |
| ~~CG Solver~~ | ~~`lattice/cg.rs`~~ | — | — | ✅ **Absorbed** (S31d) — `ops/lattice/cg.rs` |
| ESN Reservoir | `md/reservoir/` | `esn_reservoir_update.wgsl`, `esn_readout.wgsl` | 16+ | Echo State Network for transport/phase prediction |
| GPU Polyakov loop | `lattice/shaders/polyakov_loop_f64.wgsl` | WGSL compute shader | 6/6 | GPU-resident Polyakov loop (v0.6.13, bidirectional with toadStool) |
| Naga-safe SU(3) math | `lattice/shaders/su3_math_f64.wgsl` | WGSL pure math | 13/13 | Composition-safe SU(3) without ptr I/O (v0.6.13) |

### Tier 2 — Medium Priority (CPU → upstream library)

| Module | Location | Tests | What it does |
|--------|----------|-------|--------------|
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
| Tolerance/config pattern | `tolerances/` | 172 centralized constants — reusable module pattern for upstream |

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
- ~~`dirac_staggered_f64.wgsl`~~ → **Absorbed** (S31d) — `shaders/lattice/dirac_staggered_f64.wgsl`
- ~~`complex_dot_re_f64.wgsl`, `axpy_f64.wgsl`, `xpay_f64.wgsl`~~ → **Absorbed** (S31d) — `shaders/lattice/cg_kernels_f64.wgsl`

### Absorbed in S58–S62 (Feb 24, 2026) — DF64 Core Streaming

hotSpring's DF64 discovery (Experiment 012) was absorbed and **extended** by toadStool:

- `df64_core.wgsl` → `shaders/math/df64_core.wgsl` — **local deleted** (was `lattice/shaders/df64_core.wgsl`)
- `su3_df64.wgsl` → `shaders/math/su3_df64.wgsl` — **NEW** (toadStool built DF64 SU(3) matrix algebra)
- `su3_hmc_force_df64.wgsl` → `shaders/lattice/` — **NEW** (toadStool built DF64 HMC force)
- `wilson_plaquette_df64.wgsl` → `shaders/lattice/` — **NEW** (toadStool built DF64 plaquette)
- `wilson_action_df64.wgsl` → `shaders/lattice/` — **NEW** (toadStool built DF64 Wilson action)
- `kinetic_energy_df64.wgsl` → `shaders/lattice/` — **NEW** (toadStool built DF64 kinetic energy)
- `gemm_df64.wgsl` → `shaders/linalg/` — **NEW** (toadStool built DF64 dense GEMM)
- `lennard_jones_df64.wgsl` → `ops/md/forces/` — **NEW** (toadStool built DF64 LJ force)
- `bench_fp64_ratio.rs` → `bin/bench_fp64_ratio.rs` — absorbed from hotSpring
- `Fp64Strategy` enum → `device/driver_profile.rs` — **NEW** (toadStool built auto-selection)
- `GpuDriverProfile::fp64_strategy()` → auto-selects Native vs Hybrid per-GPU

**Production HMC wiring**: All lattice ops (`plaquette.rs`, `hmc_force_su3.rs`,
`gpu_wilson_action.rs`, `gpu_kinetic_energy.rs`) now auto-select between f64
and DF64 shaders based on `Fp64Strategy`. The 6.7× speedup is available upstream.

**Key API**: `barracuda::ops::lattice::su3::su3_df64_preamble()` builds the
complete shader preamble (complex_f64 + su3 + df64_core + su3_df64).

### Also Absorbed (S54–S58)

- `patch_transcendentals_in_code` — NVK/NAK workaround for exp/log/pow (S58)
- `validation.rs` — validation harness pattern (from neuralSpring, originally hotSpring-inspired)
- `metropolis.wgsl` — GPU Metropolis-Hastings (from neuralSpring S54)
- `ode_bio/` — 5 biological ODE systems (from wetSpring S58)
- `nmf.rs` — Non-negative matrix factorization (from wetSpring S58)
- Anderson transport: `anderson_conductance()`, `localization_length()` (S52)

### Already Absorbed (local source deleted)

- `spmv_csr_f64.wgsl` → `barracuda::spectral::WGSL_SPMV_CSR_F64` (local dir deleted)
- `df64_core.wgsl` → `barracuda::ops::lattice::su3::WGSL_DF64_CORE` (local deleted, S58)

### Absorbed in v0.6.13 (Feb 25, 2026) — Cross-Spring Rewiring

- `polyakov_loop_f64.wgsl` — GPU-resident Polyakov loop (from toadStool → hotSpring, bidirectional)
- `su3_math_f64.wgsl` — naga-safe SU(3) pure math (hotSpring v0.6.13, **pending upstream absorption**)
- NVK allocation guard — `check_allocation_safe()` integration in `gpu_hmc.rs`
- PRNG type-safety fix in `su3_random_momenta_f64.wgsl` (f32→f64 cast removed)

### Ready for Absorption

- `su3_math_f64.wgsl` (in `lattice/shaders/`) — naga-safe SU(3) math for shader composition
- `polyakov_loop_f64.wgsl` (in `lattice/shaders/`) — GPU-resident temporal Wilson line
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

See `wateringHole/handoffs/archive/HOTSPRING_V061_FORGE_HANDOFF_FEB21_2026.md` for the
forge handoff with hardware measurements and validation results.

---

## Cross-Spring Evolution

ToadStool's barracuda crate benefits from multi-Spring contributions.
See `wateringHole/handoffs/CROSS_SPRING_EVOLUTION_FEB22_2026.md` for the
full cross-Spring evolution map.

Key cross-pollination:

| From | To | What | Impact |
|------|-----|------|--------|
| wetSpring | all Springs | `(zero + literal)` f64 constant precision in `math_f64.wgsl` | `log_f64` 1e-3 → 1e-15 precision |
| hotSpring | all Springs | NVK `exp()`/`log()` workaround via `ShaderTemplate` | Correct results on open-source drivers |
| hotSpring | all Springs | Spectral module (Anderson, Lanczos, CSR SpMV) | GPU-accelerated sparse eigensolve |
| wetSpring | hotSpring | `GemmCached` (60× speedup for repeated GEMM) | HFB SCF loop acceleration |
| neuralSpring | hotSpring | `BatchIprGpu` | GPU Anderson localization diagnostics |
| neuralSpring | all Springs | TensorSession (matmul, relu, softmax, attention) | GPU ML layer ops |
| hotSpring | all Springs | `complex_f64.wgsl` + `su3.wgsl` | Lattice field theory math |

---

## Handoff Procedure

For each absorption candidate:

1. Open issue in toadstool describing the primitive
2. Create handoff doc in `wateringHole/handoffs/`
3. Include: Rust source, WGSL template, binding layout, dispatch geometry, test suite
4. After absorption: rewire hotSpring to `use barracuda::ops::*`, delete local code
5. Run `validate_all` to confirm 39/39 suites still pass
