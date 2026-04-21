# GPU Generation Profile Architecture — Frontier Gap Resolution Handoff

**From:** hotSpring (sovereign GPU pipeline evolution)
**To:** coralReef, barraCuda, primalSpring, biomeOS
**Date:** April 21, 2026
**coralReef iteration:** 85+
**barraCuda version:** 0.3.12
**License:** AGPL-3.0-or-later

---

## Summary

Six frontier gaps in the sovereign Rust GPU pipeline were identified and resolved.
These gaps emerged from the GPU Generation Profile Architecture work — abstracting
NVIDIA-specific hardware knowledge into vendor-agnostic traits and extending the
pipeline to AMD and Intel hardware families.

---

## Frontier Gaps Resolved

### 1. Multi-Binding `arrayLength()` — 16-Byte Descriptor Stride

**Problem:** `arrayLength()` on multi-binding shaders returned wrong values because
the 8-byte descriptor stride caused binding N's size field to alias binding N+1's
address field.

**Fix:** Target-dependent stride in the compiler (16 bytes for NVIDIA, 8 bytes for
AMD) synchronized across all NVIDIA dispatch paths.

| File | Change |
|------|--------|
| `coral-reef/src/codegen/naga_translate/expr.rs` | Stride dispatch: `if is_amd() { 8 } else { 16 }` |
| `coral-reef/src/codegen/naga_translate/func_ops.rs` | Same stride dispatch for `arrayLength` size offset |
| `coral-reef/src/codegen/ops/encoding_helpers.rs` | Updated stride documentation |
| `coral-driver/src/nv/vfio_compute/dispatch.rs` | 16-byte write stride + size field |
| `coral-driver/src/nv/mod.rs` | 16-byte write stride + size field (nouveau) |
| `coral-driver/src/nv/uvm_compute/compute_trait.rs` | 16-byte write stride + size field (UVM) |
| `coral-driver/src/nv/qmd.rs` | Updated CBUF documentation |

### 2. ShaderInfo Wire Protocol Propagation

**Problem:** `CompileResponse` from coralReef carried binary + size but not
compiler-derived metadata (GPR count, workgroup dimensions, shared memory, barriers).

**Fix:** Extended `CompileResponse` with nested `CompilationInfoResponse` field,
backward-compatible with legacy flat fields.

| File | Change |
|------|--------|
| `barraCuda/src/device/coral_compiler/types.rs` | `info: Option<CompilationInfoResponse>` + fallback |
| `barraCuda/src/device/coral_compiler/coral_compiler_tests.rs` | 3 deserialization tests |

### 3. GPU-Aware Target Selection

**Problem:** `select_target()` blindly picked the first compiler-reported architecture.
On multi-GPU systems or cross-compilation, this chose the wrong target.

**Fix:** Three-tier target selection: `BARRACUDA_TARGET_ARCH` env var → dispatch
primal's `compute.dispatch.capabilities` query → compiler-reported fallback.

| File | Change |
|------|--------|
| `barraCuda/src/device/sovereign_device.rs` | `query_dispatch_arch()` + `select_target()` rewrite |

### 4. Hardware-Gated E2E Integration Tests

**Problem:** No end-to-end test verified the full WGSL → compile → dispatch → readback
pipeline on real hardware with multi-binding shaders.

**Fix:** Five feature-gated integration tests covering single-binding, multi-binding,
`arrayLength`, multi-binding `arrayLength`, and `num_workgroups`.

| File | Change |
|------|--------|
| `coral-driver/Cargo.toml` | `hw-e2e = ["nouveau"]` feature |
| `coral-driver/tests/hw_sovereign_e2e.rs` | 5 E2E tests (feature-gated + `#[ignore]`) |

### 5. Intel i915/xe DRM Ioctl Scaffold

**Problem:** No Intel GPU support infrastructure existed in coral-driver.

**Fix:** Full ioctl definition module for both i915 (legacy) and xe (modern) Intel
DRM drivers: constants, `#[repr(C)]` structs, compute command encoding helpers,
and driver detection via `drm_version`.

| File | Change |
|------|--------|
| `coral-driver/src/intel/ioctl.rs` | i915 + xe ioctl definitions, GEM/VM/EXEC structs |
| `coral-driver/src/intel/mod.rs` | Module wiring |

### 6. AMD RDNA Binary Format Awareness

**Problem:** `AmdDevice::dispatch()` had no way to validate that a submitted binary
matched the device's GFX ISA version, and no metadata extraction from AMDGPU ELFs.

**Fix:** ELF detection, `EF_AMDGPU_MACH` → GFX version mapping, note section
metadata extraction (SGPR/VGPR counts, LDS, workgroup size), and ISA compatibility
validation integrated into `dispatch()`.

| File | Change |
|------|--------|
| `coral-driver/src/amd/shader_binary.rs` | ELF detection, metadata extraction, GFX validation |
| `coral-driver/src/amd/mod.rs` | `dispatch()` calls `validate_gfx_compat()` |

---

## Abstraction Evolution (from prior session)

These frontier fixes built on the GPU Generation Profile Architecture:

- **`GenerationProfile`** (NVIDIA): SM version, QMD version, completion strategy
- **`AmdGenerationProfile`**: GFX version, wave size, SDMA version
- **`IntelGenerationProfile`**: EU count, subslice topology, driver type
- **`HardwareCapabilities`**: Vendor-agnostic trait (wave size, f64 support, completion style)
- **`ComputeDevice`**: Vendor-agnostic dispatch trait implemented by NvDevice, AmdDevice, IntelDevice

---

## Composition Patterns

**Compiler ↔ Driver stride ABI:** The descriptor stride is now a compiler-driver
contract: NVIDIA targets get 16-byte slots `[va_lo, va_hi, size, pad]` enabling
`arrayLength()` on any binding. AMD targets keep 8-byte slots `[va_lo, va_hi]`
matching SGPR pair layout. The compiler selects stride based on target architecture;
the driver writes entries at the matching stride.

**Wire protocol versioning:** `CompileResponse` uses optional nested fields with
flat-field fallback. Newer servers send `info: { gpr_count, workgroup_size, ... }`;
older servers send flat `gpr_count`, `workgroup`. The client handles both transparently.

---

## For primalSpring / biomeOS

No action required. Changes are internal to coralReef's compiler/driver and
barraCuda's IPC client. The `shader.compile.wgsl` and `compute.dispatch` RPC
contracts are unchanged at the wire level. Intel and AMD support remains
scaffold-level — ready for hardware validation when devices are available.

---

## Remaining Pipeline Gaps

| ID | Description | Status |
|----|-------------|--------|
| amd-dispatch-exec | AMD PM4 DISPATCH_DIRECT end-to-end on real RDNA hardware | Scaffold only |
| intel-dispatch | Intel EU_GPGPU_WALKER end-to-end on real Arc/Xe hardware | Scaffold only |
| blackwell-sm-exception | SM Warp Exception on RTX 5060 (ESR 0x10) | Diagnosed, needs GPU_PROMOTE_CTX |
| cubin-assembly | Assemble CubinKernelInfo → complete ELF from SASS sections | Not started |
| shared-mem-dispatch | Pass ShaderInfo.shared_mem_bytes through QMD/PM4 | Wire ready, driver pending |
| multi-workgroup-grid | Non-trivial grid dimensions (>1 workgroup) E2E verified | Test exists, needs HW run |
