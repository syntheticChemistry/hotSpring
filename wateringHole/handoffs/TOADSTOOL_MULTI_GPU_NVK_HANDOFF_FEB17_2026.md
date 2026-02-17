# Handoff: Multi-GPU + NVK Open-Source Validation
> **SUPERSEDED** by `HOTSPRING_BARRACUDA_FULL_GPU_HANDOFF_FEB17_2026.md`

**Date:** February 17, 2026
**From:** hotSpring (computational physics validation)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only

---

## Summary

hotSpring now runs on two GPUs with two different Vulkan drivers,
producing identical physics results. This validates that our WGSL shaders
are driver-agnostic and the wgpu → Vulkan → GPU pipeline is truly portable.

| GPU | Architecture | Driver | `shaderFloat64` | HFB Checks |
|-----|-------------|--------|-----------------|------------|
| RTX 4070 | Ada (AD104) | nvidia proprietary 580.82 | true | 16/16 pass |
| Titan V | Volta (GV100) | NVK / nouveau (Mesa 25.1.5) | true | 16/16 pass |

**Numerical parity**: eigenvalue errors, orthogonality, BCS occupations, and
chemical potentials are identical to within 1e-15 across both GPUs and drivers.

---

## What We Did

1. **Installed Titan V** alongside RTX 4070 (PCIe slot 05:00.0)
2. **Discovered** that the NVIDIA open kernel module (`nvidia-dkms-580-open`)
   does not support Volta — requires GSP firmware (Turing+ only)
3. **Loaded nouveau** kernel module via `insmod` (bypassing `modprobe` alias
   set by nvidia-graphics-drivers package)
4. **Built NVK** from Mesa 25.1.5 source with `-Dvulkan-drivers=nouveau` —
   23 seconds on 24 cores, installs to `/opt/mesa-nvk/`
5. **Confirmed `shaderFloat64 = true`** on Titan V via NVK (unconditional in
   `nvk_physical_device.c:365`)
6. **Ran all validation binaries** on both GPUs with identical results
7. **Evolved `gpu.rs`** to support explicit adapter selection

---

## Issues for ToadStool

### 1. ShaderTemplate `zero` Redefinition Bug (HIGH)

**Symptom**: WGSL shader compilation fails with "redefinition of `zero`"
when `ShaderTemplate::with_math_f64_auto()` injects math preamble code
that defines `let zero = f64_const(x, 0.0);` and the user shader also has
`let zero = x - x;`.

**Affected**: All MD pipeline shaders (`yukawa_force`, `f64_builtin_test`
Phase 2), `nuclear_eos_gpu` SEMF shader. HFB shaders are NOT affected
(they use hand-written WGSL without `ShaderTemplate`).

**Impact**: Blocks `validate_barracuda_pipeline` and `sarkas_gpu` on both
GPUs. This is the same bug documented in v0.5.10 changelog.

**Fix suggestion**: `ShaderTemplate` should either:
- Use unique names for injected variables (`__math_zero` prefix)
- Check for existing definitions before injecting
- Provide a `skip_zero` option for shaders that define their own

### 2. Adapter Selection in BarraCUDA Core (MEDIUM)

hotSpring's `GpuF64::new()` now implements adapter selection:

```
HOTSPRING_GPU_ADAPTER=titan   → name substring match
HOTSPRING_GPU_ADAPTER=0       → index selection
HOTSPRING_GPU_ADAPTER=auto    → wgpu HighPerformance (legacy)
(unset)                       → first discrete GPU with SHADER_F64
```

**Recommendation for ToadStool**: The same pattern should be available in
`barracuda::device::WgpuDevice`. Currently `WgpuDevice::new()` uses
`request_adapter` with no targeting. Consider:
- `WgpuDevice::with_adapter_selector(selector: &str)`
- `WgpuDevice::enumerate_adapters() -> Vec<AdapterInfo>`
- Environment variable `BARRACUDA_GPU_ADAPTER` or `WGPU_ADAPTER`

The key insight: numeric-looking GPU model names (e.g. "4070") must fall
through to name matching if the parsed index exceeds adapter count.

### 3. NVK Compatibility Notes

- NVK (Mesa 25.1+) is Vulkan 1.4 conformant on Maxwell, Pascal, Volta,
  Turing, Ampere, Ada, Blackwell
- `shaderFloat64 = true` unconditionally (all NVIDIA GPUs have fp64 ALUs)
- Runtime power monitoring (`nvidia-smi`) is unavailable under NVK/nouveau;
  `query_gpu_power()` must degrade gracefully
- NVK requires the `nouveau` kernel module, which the nvidia driver package
  blacklists via `alias nouveau off` in `/lib/modprobe.d/`

### 4. Multi-GPU Coexistence Pattern

We successfully run **nvidia proprietary** on the RTX 4070 and **nouveau/NVK**
on the Titan V simultaneously:

```
RTX 4070 (01:00.0) → nvidia kernel module → nvidia proprietary Vulkan ICD
Titan V  (05:00.0) → nouveau kernel module → NVK Mesa Vulkan ICD
```

Both are visible to wgpu's `enumerate_adapters()`. Selection is by
`VK_ICD_FILENAMES` (colon-separated) and `HOTSPRING_GPU_ADAPTER`.

---

## Files Changed

| File | Change |
|------|--------|
| `barracuda/src/gpu.rs` | `enumerate_adapters()`, `select_adapter()`, `AdapterInfo` struct, `HOTSPRING_GPU_ADAPTER` env var, driver-agnostic power query |
| `README.md` | Hardware section updated with Titan V + NVK, adapter selection docs |
| `EVOLUTION_READINESS.md` | BCS+density gap marked resolved, promotion priorities updated |

---

## Remaining Work

1. **ShaderTemplate fix** → enables MD pipeline on all GPUs (ToadStool scope)
2. **Performance profiling** → Titan V vs RTX 4070 throughput comparison
   for HFB SCF, MD, eigensolve (needs working MD pipeline first)
3. **`WgpuDevice` adapter selection** → port hotSpring's pattern to barracuda
   core (ToadStool scope)
4. **hwmon/sysfs power monitoring** → for nouveau/NVK GPUs (low priority)
