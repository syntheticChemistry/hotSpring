# hotSpring v0.6.29 → Sovereign Dispatch Rewire

**Date:** March 9, 2026
**From:** hotSpring v0.6.29
**To:** barraCuda, toadStool, coralReef teams
**Upstream:** barraCuda `875e116`, coralReef Iter 33
**License:** AGPL-3.0-only

---

## Executive Summary

- **Backend-agnostic MD engine** (`sovereign_engine.rs`): `MdEngine<B: GpuBackend>` using `ComputeDispatch<B>` — same physics code runs on wgpu/Vulkan and sovereign/DRM.
- **wgpu path validated**: 140.3 steps/s on RTX 3090 (AllPairs N=2000), correct energies (KE=14.3381, PE=230.2525).
- **Sovereign path blocked**: `CoralReefDevice::with_auto_device()` succeeds but `dispatch_compute` fails at DRM ioctl (EINVAL). Compilation works, dispatch needs coral-driver kernel interface.
- **CPU-side energy sum** bypasses `ReduceScalarPipeline` zero bug — first correct energy readback from the generic engine.
- **Pinned to barraCuda `875e116`**: sovereign Device enum, kernel cache, ComputeDispatch<CoralReefDevice> all wired.

---

## What Changed in hotSpring

### New: `md/sovereign_engine.rs`

Generic Yukawa OCP simulation engine parameterized over `GpuBackend`:

```rust
pub fn run_simulation_generic<B: GpuBackend>(
    backend: &B,
    config: &MdConfig,
) -> Result<MdSimulation, String>
```

Uses `ComputeDispatch<B>` for all 5 shader dispatches:
- `yukawa_force_f64.wgsl` — force computation (.f64())
- `vv_kick_drift_f64.wgsl` — velocity Verlet kick-drift
- `vv_half_kick_f64.wgsl` — second half-kick
- `kinetic_energy_f64.wgsl` — per-particle KE
- `berendsen_f64.wgsl` — thermostat

Buffer lifecycle via `GpuBackend` trait: `alloc_buffer_f64_init`, `upload_f64`, `download_f64`.

### New: `bench_sovereign_dispatch` binary

Runs both backends head-to-head with identical config, reports:
- Wall time, steps/s, final energies for each backend
- Speedup ratio and energy agreement
- Cross-spring shader provenance table

### Modified: `Cargo.toml`

- Pinned `barracuda` rev `875e116` (from `8d63c77`)
- Added `bench_sovereign_dispatch` binary entry

---

## Upstream Dependencies

### barraCuda `875e116` (absorbed)

| Feature | Status |
|---------|--------|
| `Device::Sovereign` enum variant | Wired |
| `DeviceContext::Sovereign(CoralReefDevice)` | Wired |
| `Device::Auto` prefers sovereign | Wired |
| `sovereign_available()` probe | Available |
| Kernel cache in `dispatch_compute` | Wired |
| `ComputeDispatch<CoralReefDevice>` | Validated (compiles) |
| `DeviceSelection::is_sovereign()` | Available |

### coralReef Iter 33 (validated)

| Feature | Status |
|---------|--------|
| DF64 Yukawa shader compilation | SM70, SM86, RDNA2 validated |
| exp_df64 + sqrt_df64 transcendentals | Sovereign path correct |
| WGSL → SASS native binary | Working |
| DRM dispatch (nvidia-drm) | BLOCKED — ioctl EINVAL |
| DRM dispatch (amdgpu) | E2E ready (per coralReef docs) |

---

## Cross-Spring Shader Evolution

The backend-agnostic engine enables any spring's WGSL shaders to run through
sovereign dispatch. Provenance of key shaders:

| Shader | Origin Spring | Absorbed Into | Sovereign-Ready |
|--------|--------------|---------------|-----------------|
| df64_core.wgsl | hotSpring | barraCuda → all springs | Yes |
| df64_transcendentals.wgsl | hotSpring | barraCuda, coralReef | Yes (Iter 33 validated) |
| yukawa_force_f64.wgsl | hotSpring | barraCuda md/ | Yes |
| smith_waterman_f64.wgsl | wetSpring | barraCuda bio/ | Yes |
| hmm_viterbi_f64.wgsl | neuralSpring | barraCuda bio/ | Yes |
| matrix_correlation_f64.wgsl | neuralSpring | barraCuda stats/ | Yes |
| perlin_2d_f64.wgsl | ludoSpring | barraCuda procedural/ | Yes |

All 805 barraCuda WGSL shaders can route through `ComputeDispatch<CoralReefDevice>`
once DRM dispatch matures — the generic pattern is proven.

---

## Remaining Gaps (ordered by priority)

1. **coral-driver nvidia-drm dispatch** — blocks sovereign on NVIDIA GPUs
2. **ReduceScalarPipeline zero bug** — CPU-side sum works around it but GPU reduce is needed for production performance
3. **Per-dispatch overhead** — 1.8× vs batched encoding on wgpu. Future `BatchedComputeDispatch<B>` needed
4. **Verlet/CellList variants** — generic engine currently AllPairs only; Verlet needs `VerletListGpu` genericization
5. **AMD GPU testing** — coralReef reports amdgpu E2E ready; hotSpring needs RDNA2 hardware validation
