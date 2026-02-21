# hotSpring Deprecation & Migration Tracker

**Status**: Active — tracking barracuda absorption and hotSpring minimization.
**Last updated**: Feb 21, 2026

## Principle

hotSpring is a **validation Spring** — its end-state is:
- Physics-specific shaders and math (nuclear EOS, HFB, SEMF, BCS)
- Validation binaries (pass/fail against documented baselines)
- A barracuda crate reference for all shared infrastructure

Systems that toadstool/barracuda have absorbed are deprecated here and
retained only as fossil record.

## Deprecated Systems (absorbed by barracuda)

### Archive Modules (fully deprecated)

| Module | Absorbed by | Status |
|--------|-------------|--------|
| `archive/surrogate.rs` | `barracuda::surrogate`, `barracuda::sample::sparsity` | Fossil record |
| `archive/stats.rs` | `barracuda::stats`, `barracuda::optimize::convergence_diagnostics` | Fossil record |
| `archive/nuclear_eos_l1.rs` | `bin/nuclear_eos_l1_ref.rs` | Fossil record |
| `archive/nuclear_eos_l2.rs` | `bin/nuclear_eos_l2_ref.rs`, `bin/nuclear_eos_l2_gpu.rs` | Fossil record |

### MD Shaders (deprecated — barracuda ops/md canonical)

| hotSpring shader | barracuda canonical | Notes |
|------------------|---------------------|-------|
| `md/shaders/yukawa_force_f64.wgsl` | `barracuda::ops::md::forces::YukawaForceF64` | Native builtins vs math_f64 |
| `md/shaders/yukawa_force_celllist_f64.wgsl` | `barracuda::ops::md::forces::YukawaCelllistF64` | cell_idx fix needs upstream |
| `md/shaders/yukawa_force_celllist_v2_f64.wgsl` | `barracuda::ops::md::forces::YukawaCelllistF64` | Flat-loop experiment |
| `md/shaders/vv_kick_drift_f64.wgsl` | `barracuda::ops::md::integrators::VelocityVerletKickDrift` | |
| `md/shaders/rdf_histogram_f64.wgsl` | `barracuda::ops::md::observables::Rdf` | |
| Inline KE shader | `barracuda::ops::md::observables::KineticEnergy` | |
| Inline Berendsen shader | `barracuda::ops::md::thermostats::BerendsenThermostat` | |
| Inline VV half-kick | `barracuda::ops::md::integrators::VelocityVerletHalfKick` | |

### MD Modules (deprecated)

| Module | Absorbed by | Status |
|--------|-------------|--------|
| `md/shaders.rs` | `barracuda::ops::md::*` | Retained for celllist_diag/f64_builtin_test |
| `md/cpu_reference.rs` | barracuda CPU force implementations | Fossil record |
| `md/shaders_toadstool_ref/` | Removed — barracuda repo is canonical | README retained as fossil |

### Snapshot Copies (removed)

| File | Action |
|------|--------|
| `md/shaders_toadstool_ref/yukawa_f64.wgsl` | Deleted — stale snapshot |
| `md/shaders_toadstool_ref/yukawa_celllist_f64.wgsl` | Deleted — stale snapshot |
| `md/shaders_toadstool_ref/velocity_verlet_split.wgsl` | Deleted — stale snapshot |
| `md/shaders_toadstool_ref/vv_half_kick_f64.wgsl` | Deleted — stale snapshot |

## Active Systems (hotSpring-specific, retained)

### Physics Shaders (no barracuda equivalent)

| Shader | Purpose | Pipeline stage |
|--------|---------|----------------|
| `batched_hfb_potentials_f64.wgsl` | Skyrme + Coulomb potentials | HFB SCF |
| `batched_hfb_hamiltonian_f64.wgsl` | H-matrix build (T_eff + V) | HFB SCF |
| `batched_hfb_density_f64.wgsl` | BCS v², density, mixing | HFB SCF |
| `batched_hfb_energy_f64.wgsl` | Energy functional (Phase 4 stub) | HFB SCF |
| `bcs_bisection_f64.wgsl` | BCS chemical potential | HFB SCF |
| `spin_orbit_pack_f64.wgsl` | SO diagonal + eigensolve packing | HFB SCF |
| `semf_batch_f64.wgsl` | SEMF mass formula | L1 |
| `semf_pure_gpu_f64.wgsl` | SEMF (GPU-resident powers) | L1 |
| `chi2_batch_f64.wgsl` | χ² evaluation | L1/L2 |
| `deformed_wavefunction_f64.wgsl` | Nilsson HO basis | L3 (Tier B) |
| `deformed_hamiltonian_f64.wgsl` | Deformed H-build | L3 (Tier B) |
| `deformed_density_energy_f64.wgsl` | Deformed density + energy | L3 (Tier B) |
| `deformed_gradient_f64.wgsl` | Density gradient | L3 (Tier B) |
| `deformed_potentials_f64.wgsl` | Deformed mean-field | L3 (Tier B) |

### Physics Modules (hotSpring-specific)

| Module | Purpose |
|--------|---------|
| `physics/semf.rs` | SEMF binding energy |
| `physics/nuclear_matter.rs` | NMP (ρ₀, E/A, K∞, m*/m, J) |
| `physics/constants.rs` | CODATA 2018 nuclear constants |
| `physics/hfb/` | Spherical HFB+BCS solver (refactored module dir) |
| `physics/hfb_common.rs` | Shared HFB helpers |
| `physics/hfb_gpu.rs` | Batched GPU HFB (uses barracuda BatchedEighGpu) |
| `physics/hfb_gpu_resident/` | GPU-resident HFB SCF loop (refactored module dir) |
| `physics/hfb_deformed/` | Deformed HFB, CPU (refactored module dir) |
| `physics/hfb_deformed_gpu/` | Deformed HFB, GPU Tier B (refactored module dir) |
| `physics/bcs_gpu.rs` | BCS bisection params + GPU types |

### Infrastructure Modules (hotSpring-specific)

| Module | Purpose |
|--------|---------|
| `data.rs` | AME2020 data loading, EOS context |
| `discovery.rs` | Capability-based data path resolution |
| `validation.rs` | ValidationHarness (pass/fail, exit 0/1) |
| `tolerances/` | 154 centralized tolerance + solver config constants |
| `provenance.rs` | Python baseline provenance records |
| `prescreen.rs` | NMP cascade filter |
| `gpu.rs` | GpuF64 wrapper (thin layer over barracuda WgpuDevice) |
| `bench.rs` | Benchmark harness (RAPL, nvidia-smi) |
| `error.rs` | Typed GPU/simulation errors |

### MD Modules (active but migrating)

| Module | Status |
|--------|--------|
| `md/config.rs` | Active — Sarkas-specific configuration |
| `md/observables/` | Active — energy validation, SSF via barracuda |
| `md/simulation.rs` | Active — should migrate to barracuda ops |

## Upstream Fixes Needed (toadstool/barracuda)

These fixes discovered in hotSpring should be upstreamed to barracuda:

### ~~1. BCS Bisection `target` Keyword~~ ✅ RESOLVED

Absorbed by toadstool commit `0c477306`. hotSpring rewired in v0.5.3.

### ~~2. Cell-List `cell_idx` Modulo Bug~~ ✅ RESOLVED

Absorbed by toadstool commit `8fb5d5a0`. hotSpring local `GpuCellList` deprecated.

### ~~3. Native f64 Builtins~~ ✅ RESOLVED

Absorbed as `ShaderTemplate::for_driver_auto()` and `GpuDriverProfile`.
All shader compilation now routes through driver-aware path (v0.5.15 rewire).

### 4. Eigensolve Encoder API (Enhancement)

**Issue**: `BatchedEighGpu::execute_*_buffers` creates its own encoder
and calls `queue.submit()` internally. This prevents merging eigensolve
into a single-submit SCF iteration.
**Action**: Add `execute_into_encoder` variant that records into an
existing command encoder without submitting.

### 5. Deformed HFB Integration (L3)

**Issue**: 5 deformed HFB shaders exist in hotSpring but aren't wired.
Requires `GenEighGpu` (Ax=λBx) for overlap eigensolve.
**Action**: Wire hotSpring deformed shaders with barracuda GenEighGpu.

## Cross-Spring Shader Evolution

Shaders developed in hotSpring may have cross-spring utility:

| Shader category | Current Spring | Potential use in |
|----------------|----------------|------------------|
| Yukawa force | hotSpring/wetSpring | ML force-field training |
| Eigensolve | hotSpring | Any batched eigenvalue problem |
| PPPM/Ewald | hotSpring (barracuda) | Charged-particle simulations |
| RDF/SSF | hotSpring | Material science, soft matter |

Physics shaders provide exact-math verification that ML shader libraries
cannot — physics allows verifying the math is correct. Rather than building
a less-specific ML shader library, the physics shaders serve as verified
building blocks that ML pipelines can compose.
