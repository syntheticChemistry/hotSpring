# hotSpring → ToadStool: v0.5.16 Rewire Review + Remaining Evolution

**Date:** 2026-02-20
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Context:** ToadStool commit `82f953c8` (Feb 19 absorption) reviewed against
hotSpring v0.5.16 (Feb 20 consolidated handoff)

---

## Executive Summary

ToadStool's Feb 19 absorption commit caught up to the Feb 19 consolidated
handoff. hotSpring has reviewed and audited the results. **The NAK eigensolve
shader, StatefulPipeline, and ReduceScalarPipeline enhancements were absorbed
cleanly.** However, the `CellListGpu` has a **critical binding layout mismatch**
that makes it non-functional — hotSpring retains its local `GpuCellList`.

This handoff documents the specific bug, provides concrete WGSL shader designs
for the next evolution targets (lattice QCD GPU promotion), and outlines the
remaining roadmap.

---

## Part 1: Absorption Audit Results

| Absorbed Item | Status | hotSpring Action |
|--------------|--------|------------------|
| `batched_eigh_nak_optimized_f64.wgsl` | Clean | Validated via `validate_nak_eigensolve` — identical math to baseline |
| `StatefulPipeline` (`run_iterations`, `run_until_converged`) | Clean | Available for future HFB/SCF convergence loops |
| `ReduceScalarPipeline` (`scalar_buffer()`, `max_f64`, `min_f64`) | Clean | `scalar_buffer()` enables GPU-side thermostat without readback |
| `WgslLoopUnroller` (`@unroll_hint N`) | Clean | Already wired via `ShaderTemplate::for_driver_profile()` |
| `contrib/mesa-nak/NAK_DEFICIENCIES.md` | Clean | Formal patch locations for Mesa NAK |
| `CellListGpu` (3-pass: bin + scan + scatter) | **BROKEN** | Prefix-sum BGL mismatch — see Part 2 |

**No local hotSpring code was deleted** in this round because nothing new
was superseded. The CellListGpu fix would allow deletion of hotSpring's
local `GpuCellList` (~200 lines + 3 WGSL shaders).

---

## Part 2: CellListGpu Bug — Specific Fix Required

### The Problem

`cell_list_gpu.rs` (lines ~136-145) creates a bind group layout for the
prefix-sum pass with **3 bindings**:

```
binding 0: input    (storage, read-only)
binding 1: output   (storage, read-write)
binding 2: params   (uniform)
```

But `prefix_sum.wgsl` (lines ~18-21) expects **4 bindings** in a different order:

```
@group(0) @binding(0) var<uniform>            params;      // uniform
@group(0) @binding(1) var<storage, read>      input_data;  // storage read
@group(0) @binding(2) var<storage, read_write> output_data; // storage read-write
@group(0) @binding(3) var<storage, read_write> scratch;     // scratch workspace
```

### The Fix

Option A (minimal): Change `cell_list_gpu.rs` to create a 4-binding BGL
matching `prefix_sum.wgsl`'s layout, and allocate the scratch buffer.

Option B (cleaner): Use the Blelloch exclusive prefix-sum pattern with a
2-binding layout matching the cell-list's natural flow. hotSpring's working
implementation uses this approach:

```wgsl
// hotSpring's exclusive_prefix_sum.wgsl (working, 3-pass)
@group(0) @binding(0) var<uniform>            params;   // { n: u32 }
@group(0) @binding(1) var<storage, read>      input;    // [N] u32
@group(0) @binding(2) var<storage, read_write> output;  // [N] u32
```

### hotSpring's Working 3-Pass Cell-List (Reference)

1. **`cell_bin_f64.wgsl`**: One thread per particle → compute cell index → `atomicAdd(&cell_counts[cell], 1)` → store `cell_ids[particle]`
2. **`exclusive_prefix_sum.wgsl`**: `cell_counts` → `cell_start` (exclusive prefix sum)
3. **`cell_scatter.wgsl`**: One thread per particle → `atomicAdd(&write_cursors[cell], 1)` → write `sorted_indices[cell_start[cell] + cursor] = particle`

Force kernel uses `sorted_indices` for neighbor lookup but keeps positions/velocities/forces in original particle order.

---

## Part 3: Shader Designs for Lattice QCD GPU Promotion

These WGSL templates are validated on CPU in hotSpring and ready for promotion
to ToadStool as first-class barracuda primitives.

### 3.1 Complex f64 WGSL (Ready — `lattice/complex_f64.rs`)

Complete f64 complex arithmetic library for GPU. Prepend to any shader
that needs complex math.

**Functions provided:**
`c64_new`, `c64_zero`, `c64_one`, `c64_add`, `c64_sub`, `c64_mul`,
`c64_conj`, `c64_scale`, `c64_abs_sq`, `c64_abs`, `c64_inv`, `c64_div`,
`c64_exp`

**Key implementation detail:** `c64_exp` uses `exp()` + `cos()` + `sin()`
builtins which require the NVK exp/log workaround on nouveau. Route through
`ShaderTemplate::for_driver_profile()` with exp/log flag.

**Lines:** ~60 WGSL (the Rust-side `WGSL_COMPLEX64` constant).

### 3.2 SU(3) Matrix WGSL (Ready — `lattice/su3.rs`)

3x3 complex matrix algebra for gauge fields. Depends on Complex64 WGSL.

**Functions provided:**
`su3_load`, `su3_store`, `su3_idx`, `su3_mul`, `su3_adjoint`, `su3_trace`,
`su3_re_trace`, `su3_add`, `su3_scale`

**Storage format:** Row-major, 18 f64 values per matrix (9 Complex64).

**Performance note:** `su3_mul` has a 3-deep nested loop (3×3×3 = 27 FMAs).
Mark inner loop with `@unroll_hint 3` for NAK optimization via `WgslLoopUnroller`.

**Lines:** ~70 WGSL.

### 3.3 Wilson Plaquette Shader (Design — needs implementation)

**Purpose:** Compute all plaquettes on a 4D lattice for SU(3) gauge action.

```wgsl
// Prepend: WGSL_COMPLEX64 + WGSL_SU3

@group(0) @binding(0) var<uniform> params;        // { nt, nx, ny, nz, beta }
@group(0) @binding(1) var<storage, read> links;   // [V × 4 × 18] f64
@group(0) @binding(2) var<storage, read_write> plaq_re_trace; // [V × 6] f64

@compute @workgroup_size(64)
fn plaquette(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= volume) { return; }
    // For each of 6 plane orientations (mu < nu):
    //   U_p = U_mu(x) * U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
    //   store Re Tr(U_p) / 3
}
```

**Dispatch:** `(ceil(V/64), 1, 1)` where V = nt × nx × ny × nz.

**Reduction:** Use `ReduceScalarPipeline::sum_f64()` to get average plaquette.

### 3.4 HMC Gauge Force Shader (Design — needs implementation)

**Purpose:** Compute the gauge force (derivative of Wilson action) for each link.

```wgsl
// For each link U_mu(x), the force is:
//   F_mu(x) = -beta/3 * Im Tr(staple * U_mu^dag)
// where staple = sum of 6 "staple" products around the link
```

**Note:** The staple computation accesses 12 neighboring links per link.
Memory access pattern is non-trivial on GPU — consider shared memory tiling
for lattice sites within a workgroup.

### 3.5 U(1) Abelian Higgs Force Shader (Design — needs implementation)

**Purpose:** GPU-accelerated HMC for U(1) gauge + complex Higgs.

```wgsl
// Prepend: WGSL_COMPLEX64

@group(0) @binding(0) var<uniform> params;     // { nt, ns, beta_pl, kappa, lambda, mu, dt }
@group(0) @binding(1) var<storage, read_write> link_angles; // [V × 2] f64
@group(0) @binding(2) var<storage, read_write> higgs;       // [V × 2] f64 (re, im)
@group(0) @binding(3) var<storage, read_write> pi_links;    // [V × 2] f64
@group(0) @binding(4) var<storage, read_write> pi_higgs;    // [V × 2] f64 (re, im)

// Leapfrog: half-kick momenta, full-step fields, half-kick momenta
// Wirtinger force: dp_higgs/dt = -2 * dS/dphi_conj (factor of 2 critical!)
// Gauge link force: F_theta = beta * Im(plaq) + 2*kappa * Im(hop)
```

**Critical physics:** The factor of 2 in the Wirtinger derivative must be
in the shader. Missing it causes |ΔH| >> 1 and 0% HMC acceptance.

---

## Part 4: StatefulPipeline Integration Path

ToadStool's `StatefulPipeline` maps directly to hotSpring's simulation patterns:

### MD Integration (future)

```rust
use barracuda::staging::{StatefulPipeline, StatefulConfig, KernelDispatch};

let sp = StatefulPipeline::new(device.clone(), StatefulConfig {
    convergence_scalars: 2,  // KE + PE
    label: Some("md_production".into()),
});

let chain = vec![
    KernelDispatch::new(half_kick_pipeline, half_kick_bg, wg),
    KernelDispatch::new(force_pipeline, force_bg, wg),
    KernelDispatch::new(second_kick_pipeline, second_kick_bg, wg),
    KernelDispatch::new(ke_pe_pipeline, ke_pe_bg, wg),
    KernelDispatch::new(reduce_pipeline, reduce_bg, reduce_wg),
];

// Run 100 VV steps in one GPU submission, readback 16 bytes
let energies = sp.run_iterations(&chain, &convergence_buf, 100)?;
```

### HFB/SCF Convergence (future)

```rust
let (n_iter, convergence) = sp.run_until_converged(
    &hfb_chain,
    &density_norm_buf,
    max_iterations: 200,
    readback_every: 5,
    tolerance: 1e-8,
)?;
```

This would replace the explicit `loop { ... poll() ... }` pattern in
`hfb_gpu_resident.rs` with a single call.

---

## Part 5: Remaining Evolution Roadmap

### P0 — CellListGpu Fix
Fix prefix-sum BGL mismatch. Once fixed, hotSpring can delete local
`GpuCellList` (~200 lines Rust + 3 WGSL shaders = ~400 lines total).

### P1 — Complex f64 + SU(3) WGSL Promotion
Absorb `WGSL_COMPLEX64` and `WGSL_SU3` from hotSpring's `lattice/` module
into barracuda as `shaders/math/complex_f64.wgsl` and `shaders/math/su3.wgsl`.
These are fully tested on CPU; need GPU validation pass.

### P1 — FFT Primitive
Blocks full lattice QCD (Tier 3 papers 9-12). Required for:
- Momentum-space fermion propagator
- PPPM long-range force improvements
- Spectral analysis of lattice configurations

### P2 — Wilson Plaquette + HMC GPU Shaders
Once Complex64 + SU(3) are promoted, implement plaquette and HMC force
shaders. CPU reference exists in hotSpring's `lattice/wilson.rs` and
`lattice/hmc.rs`.

### P2 — U(1) Abelian Higgs GPU Shader
Simpler than SU(3) (scalar angles instead of 3x3 matrices). Good test case
for the `StatefulPipeline` + `ReduceScalarPipeline` chain.

### P3 — GPU Dirac Operator + CG Solver
Sparse matrix-vector product for staggered Dirac operator. Foundation for
dynamical fermions. CPU reference in `lattice/dirac.rs` + `lattice/cg.rs`.

---

## Part 6: Current hotSpring Codebase State

| Metric | Value |
|--------|-------|
| Unit tests | **320** pass, 5 GPU-ignored |
| Validation suites | **18/18** pass (CPU) |
| Clippy warnings | **0** |
| Papers reproduced | **9** |
| Total compute cost | **~$0.20** |
| Local GPU cell-list | Active (upstream broken) |
| Shader compilation | All via `ShaderTemplate::for_driver_profile()` |
| ToadStool dependency | `path = "../../phase1/toadstool/crates/barracuda"` |

---

*License: AGPL-3.0-only. All discoveries, code, and documentation are
sovereign community property. No proprietary dependency required.*
