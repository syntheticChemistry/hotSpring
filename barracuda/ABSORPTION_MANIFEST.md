# hotSpring → BarraCuda/ToadStool Absorption Manifest

**Date:** March 11, 2026
**Version:** v0.6.31 (synced to barraCuda `7c1fd03a` v0.3.5, toadStool S155b, coralReef Iter 47)
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
| Brain B2/D1 | v0.6.18 | NautilusShell API | **Real implementations** (evolved from placeholder) |
| `force_anomaly` | v0.6.24 | `barracuda::nautilus::brain::force_anomaly` | **Delegated** — local wrapper calls upstream |
| `Fp64Strategy::Sovereign` | v0.6.24 | `barracuda::device::driver_profile` | **Wired** — routes like Native (coralReef path) |
| `PrecisionRoutingAdvice` (hw) | v0.6.24 | `barracuda::device::driver_profile` | **Integrated** — hotSpring routing queries hw-level advice |

---

## v0.6.28 Rewiring (March 10, 2026)

barraCuda pin updated (`59c8ec5` → `a012076`, v0.3.4 expanded):
- PrecisionBrain, HardwareCalibration, PrecisionTier now upstream in barraCuda (absorbed from hotSpring v0.6.25)
- toadStool S145 also absorbed PrecisionBrain with PrecisionHint routing
- 5 new PhysicsDomain variants added for upstream parity
- coralReef Iter 30 FMA lowering unlocks F64Precise via sovereign path

## v0.6.27 Rewiring (March 10, 2026)

barraCuda pin updated (`83aa08a` → `59c8ec5`, v0.3.3 → v0.3.4):
- Critical Fp64Strategy routing fix: DF64 reduce ops now correctly call `.df64()` on Hybrid devices
- PCIe topology via sysfs, VRAM quota enforcement, BglBuilder, sovereign validation via rayon
- toadStool S144: absorbed NVVM poisoning into `nvvm_safety.rs`, PCIe switch topology, `gpu_guards`, `compile_wgsl_multi`
- `DevicePair`/`WorkloadPlanner` refs updated S142 → S144
- `HardwareCalibration`/`PrecisionBrain` refs updated: note `nvvm_safety.rs` absorption, `gpu_guards`, Iter 28 → 29
- 847/847 tests pass, 0 clippy errors, sovereign compile 45/46

## v0.6.26 Rewiring (March 10, 2026)

5 barraCuda commits absorbed (`5c16458` → `83aa08a`):
- `tridiagonal_ql` eigensolver, LCG PRNG, activations API, Wright-Fisher popgen
- Batched f32 logsumexp shader, 5,658 LOC dead code cleaned upstream
- coralReef NVVM bypass: `sovereign_compile_available`, `tier_safe_with_sovereign()`
- toadStool S142 references in `DevicePair` and `WorkloadPlanner`
- 847/847 tests pass, 0 clippy errors, sovereign compile 45/46

## v0.6.24 Rewiring (March 9, 2026)

19 barraCuda commits absorbed (`cdd748d` → `27011af`):
- `Fp64Strategy::Sovereign` — new coralReef sovereign path, 10 match arms updated
- `compile_shader_universal()` removed — replaced with `compile_shader_df64()`
- `force_anomaly` — local 15-line function now delegates to upstream
- `PrecisionRoutingAdvice` — domain routing now queries barraCuda's hw-level advice
- 769/769 tests pass, 0 clippy errors, all benchmarks green

---

## coralReef Integration Status (March 10, 2026)

coralReef daemon running locally (Phase 10, Iter 30), discovered via
`$XDG_RUNTIME_DIR/ecoPrimals/coralreef-core.json`. barraCuda's `CoralCompiler`
IPC client connects via JSON-RPC on TCP. The `spawn_coral_compile()` fire-and-forget
path compiles assembled WGSL to native SM70 SASS via coralReef and caches binaries
in-memory.

### Discovery Fix (local barraCuda evolution)

barraCuda's `discovery.rs` was evolved to handle coralReef Phase 10 manifests:
- `read_capability_transport()` now checks both `"provides"` and `"capabilities"` arrays
- `read_jsonrpc_from_value()` now handles object-form `transports.jsonrpc` (extracts `tcp` field)
- `test_reset_allows_rediscovery` assertion accepts both `"coralReef"` and `"coralreef-core"` names

### Shader Compilation Coverage

**Standalone-parseable (46/74 shaders)**: These parse as valid WGSL without template
includes and successfully receive FMA fusion optimization (498 total FMA fusions).

**Template-dependent (28/74 shaders)**: These use modular WGSL includes (`Complex64`,
`Cdf64`, `c64_new`, `exp_f64`, `log_f64`, `cbrt_f64`, `round_f64`, `pow_f64`,
`abs_f64`, `su3_identity`, `prng_gaussian`, `params`) that are resolved at runtime
by `ShaderTemplate::for_driver_profile()`. They compile successfully through the
normal pipeline (post-template-merge) and are sent to coralReef via `spawn_coral_compile()`.

| Category | Standalone OK | Template-dependent | FMA fusions |
|----------|--------------|-------------------|-------------|
| nuclear | 10 | 4 | 96 |
| plasma | 0 | 2 | — |
| lattice | 10 | 10 | 252 |
| lattice-cg | 7 | 0 | 12 |
| lattice-df64 | 0 | 6 | — |
| lattice-util | 1 | 3 | 2 |
| md | 16 | 0 | 129 |
| md-esn | 2 | 0 | 7 |
| md-df64 | 0 | 3 | — |

### Benchmark Results (with coralReef live)

| Lattice | CPU ms/traj | GPU ms/traj | Speedup | ΔPlaq |
|---------|------------|-------------|---------|-------|
| 4^4 | 73.2 | 10.7 | 6.9× | 0.000991 |
| 8^4 | 1213.0 | 33.3 | 36.5× | 0.000681 |

Physics parity excellent (ΔP < 0.001). GPU path uses Hybrid Fp64Strategy
(DF64 on FP32 cores). coralReef compiles and caches native SM70 binaries
in background.

### In-Process Compilation Validation (sovereign-dispatch PoC)

With the `sovereign-dispatch` feature, barraCuda now has `coral-gpu` wired
as a path dependency. `CoralReefDevice` compiles hotSpring WGSL shaders to
native SM70/SM86 SASS binaries in-process (no daemon needed). The full
`GpuBackend` trait is implemented with `Mutex<GpuContext>` (unblocked by
`ComputeDevice: Send + Sync` in Iter 26).

**Iter 30 results (45/46 standalone shaders compiled per target)**:

`deformed_potentials_f64` SSARef truncation fixed in Iter 29 (was PANIC in Iter 26).
12/12 NVVM bypass patterns compile across SM70/SM75/SM80/SM86/SM89/RDNA2.
See `experiments/050_CORALREEF_ITER29_SOVEREIGN_VALIDATION.md`.

**Previous Iter 26 results (44/46)**:

| Shader | SM70 bytes | SM86 bytes | GPR | Instr | Status |
|--------|-----------|-----------|-----|-------|--------|
| su3_link_update_f64 | 26400 | 26400 | 62 | 1642 | OK |
| su3_gauge_force_f64 | 20512 | 20512 | 54 | 1274 | OK |
| dirac_staggered_f64 | 18800 | 18800 | 46 | 1167 | OK |
| yukawa_force_celllist_v2_f64 | 14496 | 14496 | 78 | 898 | OK |
| batched_hfb_energy_f64 | 6640 | 6640 | 38 | 407 | OK (fixed in Iter 26) |
| wilson_plaquette_f64 | 6256 | 6256 | 38 | 383 | OK |
| deformed_potentials_f64 | — | — | — | — | PANIC (f64 min/max SSARef truncation) |
| complex_f64 | — | — | — | — | FAIL (utility include, not standalone) |

Total native binary output: ~213 KB per target architecture.

### Iter 26 Blockers Resolved

1. **f64 Min/Max/Abs/Clamp** — Root cause of log2 panic. Fixed via `OpDSetP` +
   per-component `OpSel` for f64 pairs. `batched_hfb_energy_f64` now compiles.
2. **`ComputeDevice: Send + Sync`** — Trait bound added in coral-driver.
   `CoralReefDevice` now implements full `GpuBackend` (alloc/upload/dispatch/download)
   with `Mutex<GpuContext>`.
3. **Nouveau compute subchannel** — `create_channel()` now binds compute engine
   class based on SM version. Still EINVAL on Titan V (GV100) — kernel-side NVIF
   compute object instantiation likely needs deeper setup.

### Remaining coralReef Compilation Gaps

1. **`deformed_potentials_f64`** — panics at `func_math_helpers.rs:143` with
   "index out of bounds: len 1 but index 1". An f64 value flows through a
   math function that produces a 1-component SSARef where 2 is expected.
   Different code path from the Iter 26 fix.
2. **`complex_f64`** — utility include shader, not a standalone entry point.
   Compiles successfully through template merging path.

### CoralReefDevice Evolution

`CoralReefDevice` is now a full `GpuBackend` implementation:

- `new(target)` — compile-only context (no hardware needed)
- `with_auto_device()` — auto-detect DRM render node
- `from_descriptor(vendor, arch, driver)` — toadStool discovery integration
- `compile_wgsl(wgsl, target)` — standalone compilation
- Full `GpuBackend` trait: `alloc_buffer`, `upload`, `dispatch_compute`, `download`

Dispatch is gated on DRM backend maturity:

- **amdgpu**: E2E dispatch verified in coralReef Phase 10
- **nouveau**: compute subchannel bound (Iter 26), EINVAL on GV100 channel creation
- **nvidia-drm**: pending UVM integration for buffer allocation

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
| esn_baseline | library module | — | — | **Extracted to library** (v0.6.18) — ready for absorption |
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
