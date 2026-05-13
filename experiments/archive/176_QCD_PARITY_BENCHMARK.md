# Experiment 176: QCD Sovereign Compute Parity Benchmark

**Date:** 2026-04-16  
**GPU:** RTX 5060 (GB206, SM120, Blackwell B)  
**Driver:** NVIDIA 580.x proprietary  
**Objective:** Validate dual-path (sovereign coral-reef + vendor wgpu) QCD benchmark harness

## Summary

The parity benchmark harness (`bench_sovereign_parity.rs`) runs two QCD WGSL shaders
through both the sovereign (coral-reef SASS + coral-driver UVM) and vendor (wgpu/Vulkan)
paths, comparing correctness and performance.

## Results

### Sovereign Compile (WGSL → SM120 SASS) — PASS

| Kernel | SASS size | Compile time |
|--------|-----------|-------------|
| wilson_plaquette_f64 | 6272 bytes | ~470 ms |
| sum_reduce_f64 | 1360 bytes | ~97 ms |

The coral-reef compiler successfully compiles both QCD WGSL shaders to native SM120
(Blackwell) SASS binary. This validates the full compile pipeline: WGSL → Naga IR → NAK
IR → SASS.

### Vendor Dispatch (wgpu/Vulkan) — PASS

| Kernel | Volume | Dispatch time |
|--------|--------|--------------|
| wilson_plaquette_f64 | 4^4 (256) | 0.156 ms |
| wilson_plaquette_f64 | 8^4 (4096) | 0.167 ms |
| sum_reduce_f64 | 4^4 (256) | 0.057 ms |
| sum_reduce_f64 | 8^4 (4096) | 0.035 ms |

Both kernels dispatch and complete correctly via wgpu on the RTX 5060's Vulkan driver.

### Sovereign Dispatch — BLOCKED

The UVM compute backend fails to open: `NvUvmComputeDevice::open()` hits the NOP smoke
test GPFIFO timeout (GAP-HS-031). The coral-reef compile is complete, but we cannot yet
dispatch the SASS binary to the GPU via the sovereign path.

**Root cause:** The Blackwell GPFIFO/USERD doorbell mechanism has a known issue where
GP_GET stays at 0 after GP_PUT advances to 1. QMD v5.0 has been implemented (96-word
layout with shifted addresses/sizes), but the NOP test failure predates QMD and is likely
in the channel setup or doorbell mechanism.

## Phase 1 Fix Applied

QMD v5.0 (384-byte / 96-word) builder added to `coral-driver/src/nv/qmd.rs`:
- Grid dimensions at bits 0-63 (not 224-288 as in v3.0)
- Version at bits 64-71 (major=5, minor=0)
- Shared memory shifted by 7, program address shifted by 4
- CBUF descriptors at bit offset 2048 (not 1024/1536)
- `build_qmd_for_sm()` routes SM >= 100 to v5.0

40 QMD unit tests pass including 7 new v5.0 field-level tests.

## Phase 4: Full HMC Pipeline Compile Validation (10 shaders)

Expanded from 2 to 10 self-contained shaders covering every stage of the HMC pipeline:

| Stage | Shader | Description |
|-------|--------|-------------|
| Gauge action | wilson_plaquette_f64 | Wilson plaquette average |
| Reduction | sum_reduce_f64 | Parallel f64 sum |
| CG solver | cg_compute_alpha_f64 | Conjugate gradient alpha step |
| CG solver | cg_kernels_f64 | CG update/beta/xr kernels (3 entry points) |
| Gauge force | su3_gauge_force_f64 | SU(3) staple → force computation |
| Fermion force | staggered_fermion_force_f64 | dS_F/dU from CG solution |
| Fermion action | fermion_action_sum_f64 | RHMC sector action assembly |
| Accept/reject | metropolis_f64 | Metropolis-Hastings step |
| Dirac operator | dirac_staggered_f64 | Staggered Dirac D·ψ |
| Hamiltonian | hamiltonian_assembly_f64 | H = β(6V - plaq_sum) + T + S_f |

### Cross-Generation Results (all 10 shaders)

| Target | Pass/Total | Failed shaders |
|--------|-----------|----------------|
| SM 35 (Kepler/K80) | 4/10 | f64rcp (5), f64exp2 (1), IR assert (1) |
| SM 70 (Volta/Titan V) | **10/10** | — |
| SM 120 (Blackwell/5060) | **10/10** | — |

### SM 70 (Volta/Titan V) — Full HMC Pipeline SASS

| Shader | SASS size | Compile time |
|--------|-----------|-------------|
| wilson_plaquette_f64 | 6,096 bytes | 393 ms |
| sum_reduce_f64 | 1,328 bytes | 87 ms |
| cg_compute_alpha_f64 | 896 bytes | 22 ms |
| su3_gauge_force_f64 | 20,192 bytes | 1,660 ms |
| metropolis_f64 | 2,928 bytes | 55 ms |
| dirac_staggered_f64 | 19,376 bytes | 274 ms |
| staggered_fermion_force_f64 | 7,264 bytes | 357 ms |
| fermion_action_sum_f64 | 944 bytes | 46 ms |
| hamiltonian_assembly_f64 | 832 bytes | 10 ms |
| cg_kernels_f64 | 832 bytes | 23 ms |

### SM 120 (Blackwell/RTX 5060) — Full HMC Pipeline SASS

| Shader | SASS size | Compile time |
|--------|-----------|-------------|
| wilson_plaquette_f64 | 6,272 bytes | 397 ms |
| sum_reduce_f64 | 1,360 bytes | 88 ms |
| cg_compute_alpha_f64 | 896 bytes | 22 ms |
| su3_gauge_force_f64 | 20,384 bytes | 1,609 ms |
| metropolis_f64 | 2,928 bytes | 113 ms |
| dirac_staggered_f64 | 21,952 bytes | 779 ms |
| staggered_fermion_force_f64 | 7,472 bytes | 431 ms |
| fermion_action_sum_f64 | 976 bytes | 33 ms |
| hamiltonian_assembly_f64 | 832 bytes | 20 ms |
| cg_kernels_f64 | 912 bytes | 27 ms |

### validate_pure_gauge Sovereign Integration

`validate_pure_gauge --features sovereign-dispatch` now runs:
- All 12 CPU physics checks (PASS)
- Sovereign GPU compile validation for SM35/SM70/SM120 (10 shaders each)
- Auto-detect GPU sovereign compile (RTX 5060: 6,272 bytes SASS)
- Result: **16/16 checks pass** (ALL CHECKS PASSED — SM35 now at 100% after f64 lowering fix)

## coralReef f64 Lowering Fix (SM32 Kepler)

**Root cause:** `lower_f64_function()` in `codegen/lower_f64/mod.rs` had an early return
`if !is_amd && sm.sm() < 70 { return; }` that skipped f64 transcendental lowering for
all NVIDIA GPUs below SM70. This left `OpF64Rcp`, `OpF64Exp2`, etc. as unhandled
placeholder ops in the SM32 encoder.

**Fix (3 changes in coralReef):**

1. **Removed SM < 70 guard** in `lower_f64/mod.rs` — f64 transcendental lowering now
   runs for all NVIDIA GPUs (MUFU is f32-only on all generations).
2. **Added SM-aware integer add helper** `emit_iadd()` — emits `IAdd3` on SM70+ and
   `IAdd2` on SM32 (Kepler lacks the 3-operand integer add).
3. **Added SM-aware shift helper** `emit_shl_imm()` — emits `OpShf` on SM70+ and
   `OpShl` on SM32 (SM32's funnel shift doesn't support left-shift-low).
4. **Fixed `as_imm_not_i20/f20` assertion** — returns `None` instead of panicking when
   an immediate has a source modifier, forcing the register-encoding fallback.

All 1,314 coral-reef unit tests pass. exp2/trig polynomial lowering now correctly
targets SM32 ops.

### SM35 Compile Results (after fix)

| Shader | SASS size | Status |
|--------|-----------|--------|
| wilson_plaquette_f64 | 3,584 bytes | PASS (was PANIC) |
| sum_reduce_f64 | 832 bytes | PASS |
| cg_compute_alpha_f64 | 576 bytes | PASS (was PANIC) |
| su3_gauge_force_f64 | 11,584 bytes | PASS (was PANIC) |
| metropolis_f64 | 1,728 bytes | PASS (was PANIC) |
| dirac_staggered_f64 | 11,200 bytes | PASS |
| staggered_fermion_force_f64 | 4,160 bytes | PASS (was PANIC) |
| fermion_action_sum_f64 | 640 bytes | PASS (was FAIL) |
| hamiltonian_assembly_f64 | 576 bytes | PASS |
| cg_kernels_f64 | 576 bytes | PASS |

**validate_pure_gauge: 16/16 checks passed (ALL CHECKS PASSED)**

## Architecture Insight: Naga Bypass

coralReef's `compile_ir()` accepts a pre-built `Shader` IR and runs only the codegen
pipeline (opts → f64 lowering → legalize → RA → encode). The `Frontend` trait is
extensible — custom frontends can produce `Shader` without going through Naga for
WGSL/SPIR-V/GLSL. This enables future IR-to-IR compilation (e.g., from PTX or
programmatically constructed IR) without any Naga dependency.

## Next Steps

1. Debug the NOP GPFIFO timeout (GAP-HS-031): USERD allocation, error notifier, channel scheduling
2. Once sovereign dispatch works, the parity harness compares bitwise results (ULP parity)
3. Wire HMC dispatch (GPU buffers mirroring CPU lattice state) for full GPU physics validation
