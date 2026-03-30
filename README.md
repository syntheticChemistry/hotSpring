# hotSpring

**Computational physics reproduction studies and control experiments.**

Named for the hot springs that gave us *Thermus aquaticus* and Taq polymerase — the origin story of the constrained evolution thesis. Professor Murillo's research domain is hot dense plasmas. A spring is a wellspring. This project draws from both.

---

## What This Is

hotSpring is where we reproduce published computational physics work from the Murillo Group (MSU) and benchmark it across consumer hardware. Every study has two phases:

- **Phase A (Control)**: Run the original Python code (Sarkas, mystic, TTM) on our hardware. Validate against reference data. Profile performance. Fix upstream bugs. **✅ Complete — 86/86 quantitative checks pass.**

- **Phase B (BarraCuda)**: Re-execute the same computation on ToadStool's BarraCuda engine — pure Rust, WGSL shaders, any GPU vendor. **✅ L1 validated (478× faster, better χ²). L2 validated (1.7× faster).**

- **Phase C (GPU MD)**: Run Sarkas Yukawa OCP molecular dynamics entirely on GPU using f64 WGSL shaders. **✅ 9/9 PP Yukawa DSF cases pass on RTX 4070. 0.000% energy drift at 80k production steps. Up to 259 steps/s sustained. 3.4× less energy per step than CPU at N=2000.**

- **Phase D (Native f64 Builtins + N-Scaling)**: Replaced software-emulated f64 transcendentals with hardware-native WGSL builtins. **✅ 2-6× throughput improvement. N=10,000 paper parity in 5.3 minutes. N=20,000 in 10.4 minutes. Full sweep (500→20k) in 34 minutes. 0.000% energy drift at all N. The f64 bottleneck is broken — double-float (f32-pair) on FP32 cores delivers 3.24 TFLOPS at 14-digit precision (9.9× native f64).**

- **Phase E (Paper-Parity Long Run + Toadstool Rewire)**: 9-case Yukawa OCP sweep at N=10,000, 80k production steps — matching the Dense Plasma Properties Database exactly. **✅ 9/9 cases pass, 0.000-0.002% energy drift, 3.66 hours total, $0.044 electricity. Cell-list 4.1× faster than all-pairs. Toadstool GPU ops (BatchedEighGpu, SsfGpu, PppmGpu) wired into hotSpring.**

- **Phase F (Kokkos-CUDA Parity + Verlet Neighbor List)**: Runtime-adaptive algorithm selection (AllPairs/CellList/VerletList) with DF64 precision on consumer GPUs. **✅ 9/9 cases pass, ≤0.004% drift. Verlet achieves 992 steps/s (κ=3) — gap vs Kokkos-CUDA closed from 27× to 3.7×. barraCuda v0.6.17.**

hotSpring answers: *"Does our hardware produce correct physics?"* and *"Can Rust+WGSL replace the Python scientific stack?"*

> **For the physics**: See [`PHYSICS.md`](PHYSICS.md) for complete equation documentation
> with numbered references — every formula, every constant, every approximation.
>
> **For the methodology**: See [`whitePaper/METHODOLOGY.md`](whitePaper/METHODOLOGY.md)
> for the two-phase validation protocol and acceptance criteria.

---

## Current Status (2026-03-30)

> **128 experiments** | **500+ quantitative checks** | **~$0.30 total science cost** | **870 lib tests, 139 binaries, 99 WGSL shaders** | **NVIDIA GPFIFO pipeline OPERATIONAL on RTX 3090** | **AMD scratch/local memory OPERATIONAL on RX 6950 XT** | **AMD sovereign compiler: 24/24 QCD shaders compile to native GFX ISA** | **Silicon saturation profiling: 7-tier routing, TMU PRNG, subgroup reduce, ROP atomics**
>
> **Sovereign GPU Pipeline (Exp 110-128, 2026-03-30):** FECS firmware survives the nouveau→vfio-pci driver swap via kernel livepatch (Exp 125). Warm handoff orchestration wired into `ember`/`glowplug` as first-class RPC operations — `ember.livepatch.*`, `ember.fecs.state`, `ember.mmio.read`, `device.warm_handoff` — replacing all shell-scripted GPU control with a programmable daemon interface (coralReef Iter 70d). K80 direct PIO boot validated (Exp 123). WPR2 root cause definitive (Exp 122). 10.5/11 sovereign layers on Volta. Puzzle Box Matrix (Exp 128) implements parallel solution tracks across K80 (Kepler, unsigned) and Titan V (Volta, HS+ signed). **Fleet:** 2x Titan V + RTX 5070 + K80.
>
> **NVIDIA Sovereign Compute Breakthrough (2026-03-30):** RTX 3090 GPFIFO command submission pipeline **fully operational** through coralReef's sovereign driver. Key fixes via `ioctl` interception of CUDA: `NV906F_CTRL_CMD_BIND`, TSG scheduling, `GET_WORK_SUBMIT_TOKEN` via Volta class (0xC36F), VRAM USERD, 48-byte RM_ALLOC on 580.x GSP-RM.
>
> **AMD Sovereign Compute — Local Memory Breakthrough (2026-03-30):** Three-layer fix unlocks per-thread scratch memory on RDNA2. Key discovery: amdgpu DRM Command Processor does NOT auto-initialize `FLAT_SCRATCH` for compute IB submissions. Fixed with `S_MOV_B32`+`S_SETREG_B32` shader prolog. **7/8 hardware parity tests pass** (1672 unit tests pass).
>
> **AMD Sovereign Compiler:** 24/24 QCD shaders compiled (WGSL → native GFX10.3 ISA) in 102ms. 38/39 dispatch tests pass. Remaining frontiers: EXEC masking, Wilson plaquette QCD dispatch.
>
> **AMD Sovereign Compiler:** 24/24 QCD shaders compiled (WGSL → native GFX10.3 ISA) in 102ms. 38/39 dispatch tests pass. Remaining frontier: EXEC masking for divergent wavefront control flow.
>
> **Science (Exp 096-103):** GPU RHMC production (Nf=2, Nf=2+1), gradient flow at volume (5 LSCFRK integrators), self-tuning RHMC (zero hand-tuned parameters). Silicon saturation profiling complete (Exp 105-106).
>
> See [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) for the full validation table and benchmark data.

| Domain | Status | Highlights |
|--------|--------|------------|
| **Dense Plasma MD** (Sarkas, 12 cases) | ✅ 60/60 | 9 PP Yukawa + 3 PPPM, paper-parity at N=10k |
| **Surrogate + Nuclear EOS** | ✅ 39/39 | BarraCuda 478× faster (χ²=2.27), HFB GPU, AME2020 |
| **Transport** (Stanton-Murillo) | ✅ 13/13 | GPU-only Green-Kubo D*/η*/λ* |
| **Lattice QCD** (quenched + dynamical) | ✅ 46/46 | HMC, Dirac CG, plaquette, SU(3) + U(1) Higgs |
| **GPU RHMC** (Nf=2, Nf=2+1) | ✅ Complete | True multi-shift CG, fermion force validated, ΔH=O(1), 8.5 GFLOP/s |
| **Gradient Flow** (Chuna 43) | ✅ Complete | 5 LSCFRK integrators, CK4 stability, t₀/w₀ |
| **Self-Tuning RHMC** | ✅ Complete | Zero hand-tuned parameters — spectral + acceptance-driven |
| **Spectral Theory** (Kachkovskiy) | ✅ 45/45 | Anderson 1D/2D/3D, Hofstadter, GPU Lanczos |
| **NPU** (AKD1000 hardware) | ✅ 34/35 | 10 SDK assumptions overturned, physics pipeline, phase detection |
| **Sovereign GPU** (coralReef) | ✅ GPFIFO + AMD scratch | RTX 3090 pipeline, AMD scratch/local f64 PASS, K80 direct boot, livepatch + warm handoff wired into ember/glowplug |
| **Silicon Characterization** | ✅ Complete | TMU, ROP, L2, shader cores — AMD vs NVIDIA personalities |
| **Silicon Saturation Profiling** | ✅ Complete | TMU PRNG, subgroup reduce, ROP atomics, capacity analysis |
| **Chuna Papers 43-45** | ✅ **44/44** | Gradient flow + BGK dielectric + kinetic-fluid coupling |

Full validation table (130+ rows) with per-experiment details: [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md)

### Science Ladder

Quenched SU(3) ✅ → Gradient Flow ✅ → LSCFRK Integrators ✅ → N_f=4 Infra ✅ → Chuna 44/44 ✅ → **N_f=2 ✅** → **N_f=2+1 ✅** → **Self-tuning ✅** → **True multi-shift CG ✅** → **Fermion force validated ✅** → **Silicon saturation profiling ✅** → **Sovereign NVIDIA GPFIFO ✅** → **AMD sovereign compiler 24/24 ✅** → **AMD scratch/local memory ✅** → **Livepatch warm handoff wired into daemons ✅** → K80/Titan V puzzle box matrix (next) → AMD EXEC masking → 16⁴+ dynamical production on sovereign pipeline. Cross-cutting sovereign validation matrix: [`specs/SOVEREIGN_VALIDATION_MATRIX.md`](specs/SOVEREIGN_VALIDATION_MATRIX.md).

## Evolution Architecture: Write → Absorb → Lean

hotSpring is a biome. ToadStool (barracuda) is the fungus — it lives in
every biome. hotSpring, neuralSpring, desertSpring each lean on toadstool
independently, evolve shaders and systems locally, and toadstool absorbs
what works. Springs don't reference each other — they learn from each other
by reviewing code in `ecoPrimals/`, not by importing.

```
hotSpring writes extension    → toadstool absorbs    → hotSpring leans on upstream
─────────────────────────       ──────────────────       ────────────────────────
Local GpuCellList (v0.5.13)  → CellListGpu fix (S25) → Deprecated local copy
Complex64 WGSL template      → complex_f64.wgsl      → First-class barracuda primitive
SU(3) WGSL template          → su3.wgsl              → First-class barracuda primitive
Wilson plaquette design       → plaquette_f64.wgsl    → GPU lattice shader
HMC force design             → su3_hmc_force.wgsl    → GPU lattice shader
Abelian Higgs design         → higgs_u1_hmc.wgsl     → GPU lattice shader
NAK eigensolve workarounds   → batched_eigh_nak.wgsl → Upstream shader
ReduceScalar feedback        → ReduceScalarPipeline  → Rewired in v0.5.12
Driver profiling feedback    → GpuDriverProfile      → Rewired in v0.5.15
```

**The cycle**: hotSpring implements physics on CPU with WGSL templates embedded
in the Rust source. Once validated, designs are handed to toadstool via
`ecoPrimals/wateringHole/handoffs/`. Toadstool absorbs them as GPU shaders. hotSpring
then rewires to use the upstream primitives and deletes local code. Each cycle
makes the upstream library richer and hotSpring leaner.

**What makes code absorbable**:
1. WGSL shaders in dedicated `.wgsl` files (loaded via `include_str!`)
2. Clear binding layout documentation (binding index, type, purpose)
3. Dispatch geometry documented (workgroup size, grid dimensions)
4. CPU reference implementation validated against known physics
5. Tolerance constants in `tolerances/` module tree (not inline magic numbers)
6. Handoff document with exact code locations and validation results

**Next absorption targets** (see `barracuda/ABSORPTION_MANIFEST.md`):
- Staggered Dirac shader — `lattice/dirac.rs` + `WGSL_DIRAC_STAGGERED_F64` (8/8 checks, Tier 1)
- CG solver shaders — `lattice/cg.rs` + 3 WGSL shaders (9/9 checks, Tier 1)
- Pseudofermion HMC — `lattice/pseudofermion/` (heat bath, force, combined leapfrog; 7/7 checks, Tier 1)
- ESN reservoir + readout — `md/reservoir/` (GPU+NPU validated, Tier 1)
- HFB shader suite — potentials + density + BCS bisection (14+GPU+6 checks, Tier 2)
- NPU substrate discovery — `metalForge/forge/src/probe.rs` (local evolution)

**Already leaning on upstream** (v0.6.32, synced to barraCuda v0.3.7 + toadStool S168 + coralReef Phase 10+, wgpu 28, pollster 0.3, bytemuck 1.25, tokio 1.50):

ToadStool **S168** adds `shader.dispatch` completing the orchestration layer for GPU shader pipelines. **barraCuda Sprint 23** landed the f64 precision fix (production numerical parity on mixed pipelines).

| Module | Upstream | Status |
|--------|----------|--------|
| `spectral/` | `barracuda::spectral::*` | **✅ Leaning** — 41 KB local deleted, re-exports + `CsrMatrix` alias |
| `md/celllist.rs` | `barracuda::ops::md::CellListGpu` | **✅ Leaning** — local `GpuCellList` deleted |

**Absorption-ready inventory** (v0.6.9):

| Module | Type | WGSL Shader | Status |
|--------|------|------------|--------|
| `lattice/dirac.rs` | Dirac SpMV | `WGSL_DIRAC_STAGGERED_F64` | (C) Ready — 8/8 checks |
| `lattice/cg.rs` | CG solver | `WGSL_COMPLEX_DOT_RE_F64` + 2 more | (C) Ready — 9/9 checks |
| `lattice/pseudofermion/` | Pseudofermion HMC | CPU (WGSL-ready pattern) | (C) Ready — 7/7 checks |
| `md/reservoir/` | ESN | `esn_reservoir_update.wgsl` + readout | (C) Ready — NPU validated |
| `physics/screened_coulomb.rs` | Sturm eigensolve | CPU only | (C) Ready — 23/23 checks |
| `physics/hfb_deformed_gpu/` | Deformed HFB | 5 WGSL shaders | (C) Ready — GPU-validated |

---

## BarraCuda Crate (v0.6.32)

The `barracuda/` directory is a standalone Rust crate providing the validation
environment, physics implementations, and GPU compute. Key architectural properties:

- **870 tests** (lib), **139 binaries**, **39 validation suites** (39/39 pass), **99 WGSL shaders** (all AGPL-3.0-only),
  **16 determinism tests** (rerun-identical for all stochastic algorithms). Includes
  lattice QCD (complex f64, SU(3), Wilson action, HMC, Dirac CG, pseudofermion HMC),
  Abelian Higgs (U(1) + Higgs, HMC), transport coefficients (Green-Kubo D*/η*/λ*,
  Sarkas-calibrated fits), HotQCD EOS tables, NPU quantization parity (f64→f32→int8→int4),
  and NPU beyond-SDK hardware capability validation. Test coverage: **74.9% region /
  83.8% function** (spectral tests upstream in barracuda; GPU modules require hardware
  for higher coverage). Measured with `cargo-llvm-cov`.
- **AGPL-3.0 only** — all `.rs` files and all 99 `.wgsl` shaders have
  `SPDX-License-Identifier: AGPL-3.0-only` on line 1.
- **Provenance** — centralized `BaselineProvenance` records trace hardcoded
  validation values to their Python origins (script path, git commit, date,
  exact command). `AnalyticalProvenance` references (DOIs, textbook citations)
  document mathematical ground truth for special functions, linear algebra,
  MD force laws, and GPU kernel correctness. All nuclear EOS binaries and
  library test modules source constants from `provenance::SLY4_PARAMS`,
  `NMP_TARGETS`, `L1_PYTHON_CHI2`, `MD_FORCE_REFS`, `GPU_KERNEL_REFS`, etc.
  DOIs for AME2020, Chabanat 1998, Kortelainen 2010, Bender 2003,
  Lattimer & Prakash 2016 are documented in `provenance.rs`.
- **Tolerances** — ~150 centralized constants in the `tolerances/` module tree with physical
  justification (machine precision, numerical method, model, literature).
  Includes 12 physics guard constants (`DENSITY_FLOOR`, `SPIN_ORBIT_R_MIN`,
  `COULOMB_R_MIN`, `BCS_DENSITY_SKIP`, `DEFORMED_COULOMB_R_MIN`, etc.),
  8 solver configuration constants (`HFB_MAX_ITER`, `BROYDEN_WARMUP`,
  `BROYDEN_HISTORY`, `CELLLIST_REBUILD_INTERVAL`, etc.),
  plus validation thresholds for transport, lattice QCD, Abelian Higgs,
  NAK eigensolve, PPPM, screened Coulomb, spectral theory, ESN heterogeneous
  pipeline, NPU quantization, and NPU beyond-SDK hardware capabilities.
  Zero inline magic numbers — all validation binaries and solver loops wired to `tolerances::*`.
- **ValidationHarness** — structured pass/fail tracking with exit code 0/1.
  55 of 115 binaries use it (validation targets). Remaining binaries are optimization
  explorers, benchmarks, and diagnostics.
- **Shared data loading** — `data::EosContext` and `data::load_eos_context()`
  eliminate duplicated path construction across all nuclear EOS binaries.
  `data::chi2_per_datum()` centralizes χ² computation with `tolerances::sigma_theo`.
- **Typed errors** — `HotSpringError` enum with full `Result` propagation
  across all GPU pipelines, HFB solvers, and ESN prediction. Variants:
  `NoAdapter`, `NoShaderF64`, `DeviceCreation`, `DataLoad`, `Barracuda`,
  `GpuCompute`, `InvalidOperation`, `IoError`, `JsonError`.   **Zero `.unwrap()` and zero `.expect()`
  in library code** — `#![deny(clippy::expect_used, clippy::unwrap_used)]` enforced crate-wide;
  all fallible operations use `?` propagation. Provably
  unreachable byte-slice conversions annotated with SAFETY comments.
- **Shared physics** — `hfb_common.rs` consolidates BCS v², Coulomb exchange
  (Slater), CM correction, Skyrme t₀, Hermite polynomials, and Mat type.
  Shared across spherical, deformed, and GPU HFB solvers.
- **GPU helpers centralized** — `GpuF64` provides `upload_f64`, `read_back_f64`,
  `dispatch`, `create_bind_group`, `create_u32_buffer` methods. All shader
  compilation routes through ToadStool's `WgslOptimizer` with `GpuDriverProfile`
  for hardware-accurate ILP scheduling (loop unrolling, instruction reordering).
  No duplicate GPU helpers across binaries.
- **Zero duplicate math** — all linear algebra, quadrature, optimization,
  sampling, special functions, statistics, and spin-orbit coupling use
  BarraCuda primitives (`SpinOrbitGpu`, `compute_ls_factor`).
- **Capability-based discovery** — runtime adapter enumeration by memory/capability
  (`discover_best_adapter`, `discover_primary_and_secondary_adapters`). Supports nvidia proprietary,
  NVK/nouveau, RADV, and any Vulkan driver. Buffer limits derived from
  `adapter.limits()`, not hardcoded. Data paths resolved via `HOTSPRING_DATA_ROOT`
  or directory discovery.
- **NaN-safe** — all float sorting uses `f64::total_cmp()`.
- **Zero external commands** — pure-Rust ISO 8601 timestamps (Hinnant algorithm),
  no `date` shell-out. `nvidia-smi` calls degrade gracefully.
- **No unsafe code** — zero `unsafe` blocks in the entire crate.
- **Quality gates**: Zero clippy warnings (lib), zero unsafe blocks, 8 scoped TODO(B2) markers (GPU-resident migration), all files <1000 lines, AGPL-3.0-only consistent.

```bash
cd barracuda
cargo test               # 870 tests (lib), 6 GPU/heavy-ignored (~700s; spectral tests upstream)
cargo clippy --all-targets  # Zero warnings (pedantic + nursery via Cargo.toml workspace lints)
cargo doc --no-deps      # Full API documentation — 0 warnings
cargo run --release --bin validate_all  # 39/39 suites pass
```

See [`barracuda/CHANGELOG.md`](barracuda/CHANGELOG.md) for version history.

---

## Quick Start

```bash
# Full regeneration — clones repos, downloads data, sets up envs, runs everything
# (~12 hours, ~30 GB disk space, GPU recommended)
bash scripts/regenerate-all.sh

# Or step by step:
bash scripts/regenerate-all.sh --deps-only   # Clone + download + env setup (~10 min)
bash scripts/regenerate-all.sh --sarkas      # Sarkas MD: 12 DSF cases (~3 hours)
bash scripts/regenerate-all.sh --surrogate   # Surrogate learning (~5.5 hours)
bash scripts/regenerate-all.sh --nuclear     # Nuclear EOS L1+L2 (~3.5 hours)
bash scripts/regenerate-all.sh --ttm         # TTM models (~1 hour)
bash scripts/regenerate-all.sh --dry-run     # See what would be done

# Or manually:
bash scripts/clone-repos.sh       # Clone + patch upstream repos
bash scripts/download-data.sh     # Download Zenodo archive (~6 GB)
bash scripts/setup-envs.sh        # Create Python environments
```

```bash
# Phase C: GPU Molecular Dynamics (requires SHADER_F64 GPU)
cd barracuda
cargo run --release --bin sarkas_gpu              # Quick: kappa=2, Gamma=158, N=500 (~30s)
cargo run --release --bin sarkas_gpu -- --full    # Full: 9 PP Yukawa cases, N=2000, 30k steps (~60 min)
cargo run --release --bin sarkas_gpu -- --long    # Long: 9 cases, N=2000, 80k steps (~71 min, recommended)
cargo run --release --bin sarkas_gpu -- --paper   # Paper parity: 9 cases, N=10k, 80k steps (~3.66 hrs)
cargo run --release --bin sarkas_gpu -- --scale   # GPU vs CPU scaling
```

### What gets regenerated

All large data (21+ GB) is gitignored but fully reproducible:

| Data | Size | Script | Time |
|------|------|--------|------|
| Upstream repos (Sarkas, TTM, Plasma DB) | ~500 MB | `clone-repos.sh` | 2 min |
| Zenodo archive (surrogate learning) | ~6 GB | `download-data.sh` | 5 min |
| Sarkas simulations (12 DSF cases) | ~15 GB | `regenerate-all.sh --sarkas` | 3 hr |
| TTM reproduction (3 species) | ~50 MB | `regenerate-all.sh --ttm` | 1 hr |
| **Total regeneratable** | **~22 GB** | `regenerate-all.sh` | **~12 hr** |

Upstream repos are pinned to specific versions and automatically patched:
- **Sarkas**: v1.0.0 + 3 patches (NumPy 2.x, pandas 2.x, Numba 0.60 compat)
- **TTM**: latest + 1 patch (NumPy 2.x `np.math.factorial` removal)

---

## Directory Structure

```
hotSpring/
├── README.md                           # This file
├── PHYSICS.md                          # Complete physics documentation (equations + references)
├── CONTROL_EXPERIMENT_STATUS.md        # Comprehensive status + results (197/197)
├── NUCLEAR_EOS_STRATEGY.md             # Nuclear EOS Phase A→B strategy
├── LICENSE                             # AGPL-3.0
├── .gitignore
│
├── whitePaper/                         # Public-facing study documents
│   ├── README.md                      # Document index
│   ├── STUDY.md                       # Main study — full writeup
│   ├── BARRACUDA_SCIENCE_VALIDATION.md # Phase B technical results
│   ├── CONTROL_EXPERIMENT_SUMMARY.md  # Phase A quick reference
│   ├── METHODOLOGY.md                # Two-phase validation protocol
│   └── baseCamp/                      # Per-domain research briefings
│       ├── murillo_plasma.md          # Murillo Group — dense plasma MD (Papers 1-6)
│       ├── murillo_lattice_qcd.md     # Lattice QCD — quenched & dynamical (Papers 7-12)
│       ├── kachkovskiy_spectral.md    # Spectral theory — Anderson, Hofstadter
│       ├── cross_spring_evolution.md  # Cross-spring shader ecosystem (164+ shaders)
│       └── neuromorphic_silicon.md    # AKD1000 NPU exploration — silicon behavior, cross-substrate ESN
│
├── barracuda/                          # BarraCuda Rust crate (870 tests, 139 binaries, 99 WGSL shaders)
│   ├── Cargo.toml                     # Dependencies (requires ecoPrimals/barraCuda)
│   ├── CHANGELOG.md                   # Version history
│   └── src/bin/                       # 129 binaries (validation, production, benchmarks)
│
├── experiments/                        # 128 experiment journals (fossil record); 001-057 archived under experiments/archive/
│   ├── archive/                        # experiments 001-057 (archived journals)
│   ├── 058-069: Precision, sovereign GPU cracking, GlowPlug, falcon boot
│   ├── 070-095: Backend matrix, MMU, WPR, sysmem HS mode breakthrough
│   ├── 096-103: Silicon characterization, GPU RHMC, gradient flow, self-tuning
│   └── 110-128: Consolidation, WPR2, K80 sovereign, VM capture, livepatch, warm handoff, puzzle box matrix
│
├── specs/                              # Specifications, requirements, gap trackers
├── control/                            # Python control scripts (by domain)
├── metalForge/                         # Hardware characterization (GPU, NPU, nodes)
├── scripts/                            # Regeneration, boot config, hw-test
├── benchmarks/                         # Kokkos/LAMMPS parity, protocol
└── data/                               # Reference data (gitignored large files)
```

---

## Key Documents

| Document | Purpose |
|----------|---------|
| [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) | Full validation table, benchmark data, studies, document index |
| [`PHYSICS.md`](PHYSICS.md) | Complete physics documentation — every equation, constant, approximation |
| [`specs/PAPER_REVIEW_QUEUE.md`](specs/PAPER_REVIEW_QUEUE.md) | Papers to review/reproduce, prioritized by tier |
| [`specs/SOVEREIGN_VALIDATION_MATRIX.md`](specs/SOVEREIGN_VALIDATION_MATRIX.md) | Sovereign validation ladder / cross-cutting matrix (DRM, drivers, hardware) |
| [`whitePaper/baseCamp/`](whitePaper/baseCamp/) | Per-domain research briefings (17 docs) |

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).
See [LICENSE](LICENSE) for the full text.

Sovereign science: all source code, data processing scripts, and validation results are
freely available for inspection, reproduction, and extension. If you use this work in
a network service, you must make your source available under the same terms.

---

*128 experiments, 870 tests, 139 binaries, 99 WGSL shaders, ~$0.30 total science cost.
Consumer GPUs reproduce HPC physics at paper parity. DF64 delivers 3.24 TFLOPS at
14-digit precision. GPU RHMC runs all-flavors dynamical QCD (Nf=2+1). Self-tuning
RHMC eliminates hand-tuned parameters. Chuna 44/44 checks pass. K80 validates the
sovereign compute stack without firmware barriers. 10.5/11 sovereign layers on Volta.
The full science ladder — quenched through dynamical fermions with gradient flow
scale setting — runs on consumer hardware. The scarcity was artificial.*
