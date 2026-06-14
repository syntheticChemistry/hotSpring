SPDX-License-Identifier: AGPL-3.0-only

# Multi-Backend Dispatch Strategy

**Date:** March 10, 2026
**hotSpring:** v0.6.28
**barraCuda:** v0.3.4 (`a012076`)
**toadStool:** S145
**coralReef:** Phase 10, Iteration 30

---

## Discovery

On March 10, 2026, hardware diagnostics for coralReef Iteration 30 revealed that
Mesa's NVK Vulkan driver (25.1.5) provides full compute dispatch on the Titan V
(GV100, SM70) through the `nouveau` kernel module. This was previously assumed to
be limited — the Titan V had been sidelined in earlier production runs due to NVK
instability. NVK now delivers:

- Full 4-tier precision: F32, F64, F64Precise, DF64
- All 12 `PhysicsDomain` variants route correctly
- Dual-GPU cooperative dispatch with RTX 3090 (proprietary Vulkan)
- Native f64 at 1:2 throughput (Volta FP64 units)

This means hotSpring already has a working, production-quality dispatch path on
**both** GPUs through wgpu/Vulkan, independent of coralReef's sovereign
compilation pipeline.

---

## Three-Tier Dispatch Architecture

### Tier 1: wgpu/Vulkan (Production — Works Now)

**Path:** hotSpring → barraCuda → wgpu → Vulkan → GPU driver → hardware

| GPU | Driver | Vulkan Version | Status |
|-----|--------|---------------|--------|
| Titan V (GV100) | NVK/Mesa 25.1.5 via nouveau | 1.3.311 | Full 4-tier precision |
| RTX 3090 (GA102) | NVIDIA proprietary 580.119.02 | 1.4.312 | Full precision (NVVM risk on some shaders) |

**Strengths:**
- Works today on all 84 WGSL shaders
- Leverages mature driver stacks (Mesa, NVIDIA proprietary)
- Pipeline caching eliminates recompilation
- Standard Vulkan portability: AMD, Intel, Apple (via MoltenVK) all reachable
- barraCuda's `WgslOptimizer` + `GpuDriverProfile` handle driver quirks

**Limitations:**
- Bound by driver-imposed shader compilation decisions (FMA fusion, register allocation)
- NVVM poisoning risk on NVIDIA proprietary for some f64 transcendentals
- No control over instruction scheduling or memory layout
- NVK's NAK compiler produces invalid modules from sovereign SPIR-V (excluded)

**When to use:** All production runs. This is the default path.

### Tier 2: coralReef Sovereign (Long-Term Goal — Pure Rust)

**Path:** hotSpring → barraCuda → coralReef → WGSL → SASS/GFX binary → DRM ioctl → hardware

| GPU | Required Migration | Status |
|-----|-------------------|--------|
| Titan V (nouveau) | Legacy `DRM_NOUVEAU_CHANNEL_ALLOC` → new UAPI (`VM_INIT`/`VM_BIND`/`EXEC`) | Blocked |
| RTX 3090 (nvidia-drm) | UVM device allocation (`RM_ALLOC` status 0x1F) | Blocked |

**Strengths:**
- Pure Rust: no driver dependency, deploys on any hardware with a DRM node
- Full control: FMA policy per-shader (`FmaPolicy::Separate` for F64Precise)
- Instruction-level optimization: register allocation, scheduling, memory coalescing
- Bypasses NVVM poisoning entirely (sovereign SASS generation)
- Hardware-agnostic: same pipeline for NVIDIA, AMD, Intel once backends land

**Limitations:**
- DRM dispatch blocked on both GPU paths (documented in Experiment 051)
- Compilation-only parity today: 45/46 shaders compile, 0 dispatch
- Requires per-architecture backend (SM70, SM86, GFX9, GFX10, ...)

**When to use:** Precision-critical domains where FMA fusion must be controlled
(`F64Precise`), hardware with broken Vulkan drivers, sovereign deployment where
no driver installation is acceptable. Future: performance-critical paths once
dispatch overhead and scheduling quality reach Tier 1 parity.

**coralReef is not replacing wgpu** — it provides a sovereign bypass for scenarios
where the Vulkan driver stack introduces precision loss, compilation failures, or
deployment constraints. Both paths coexist.

### Tier 3: Kokkos/LAMMPS (Reference Target)

**Path:** External LAMMPS binary → Kokkos runtime → CUDA → NVIDIA hardware

| GPU | Required | Status |
|-----|----------|--------|
| RTX 3090 (CUDA) | LAMMPS compiled with `-DKokkos_ENABLE_CUDA=ON` | Not installed |

**Strengths:**
- Mature, production-proven HPC framework
- Compile-time CUDA optimization (ptxas, register allocation)
- cuFFT for PPPM (highly tuned)
- Established performance baselines in literature

**Limitations:**
- NVIDIA-only (CUDA backend)
- Proprietary driver required
- No runtime shader modification or precision policy
- No intelligence layer (no brain, no adaptive steering)

**When to use:** Never in production. Kokkos is the benchmark reference — the
performance ceiling that hotSpring's pure Rust pipeline aims to match or exceed.
Every Kokkos comparison reveals where barraCuda's WGSL shaders are slow (dispatch
overhead? arithmetic throughput? memory bandwidth?) and guides optimization.

---

## Performance Baselines

### Current Gap (RTX 3090, DF64, N=2000, March 2026)

| Case | barraCuda steps/s | Est. Kokkos-CUDA steps/s | Gap |
|------|-------------------|--------------------------|-----|
| k1_G14 | 181 (AllPairs) | ~720 | 4.0x |
| k2_G31 | 368 (Verlet) | ~1100 | 3.0x |
| k2_G158 | 846 (Verlet) | ~3050 | 3.6x |
| k3_G100 | 977 (Verlet) | ~3130 | 3.2x |
| k3_G1510 | 992 (Verlet) | ~3670 | 3.7x |

These estimates are derived from published Kokkos performance data scaled to
RTX 3090 hardware. Experiment 052 will replace estimates with measured values
once LAMMPS+Kokkos is installed.

### Where the Gap Lives

| Component | Estimated Contribution | Path to Close |
|-----------|----------------------|---------------|
| Dispatch overhead | ~15-20% | Batched encoder (already implemented), streaming dispatch |
| WGSL→SPIR-V compilation | ~5-10% | coralReef sovereign SASS (bypasses naga+driver) |
| Register pressure | ~10-15% | coralReef register allocation, workgroup size tuning |
| Memory access pattern | ~20-30% | Coalesced access optimization, shared memory usage |
| Arithmetic throughput | ~10-15% | FMA fusion (barraCuda sovereign compiler: 498 fusions) |
| cuFFT vs WGSL FFT | ~10-20% (PPPM only) | Continued WGSL 3D FFT optimization |

### Titan V (NVK) Baseline — New Data

| Case | NVK steps/s | Precision | Notes |
|------|-------------|-----------|-------|
| Full 4-tier | Validated | F32/F64/F64Precise/DF64 | All pass |
| HMC acceptance | 100% | Native f64 | P=0.584 (physically correct) |
| Dual-GPU | Working | Split + Redundant modes | PCIe 1.2 GB/s |

These numbers become the floor that coralReef's sovereign dispatch must meet
on the same hardware. If sovereign dispatch is slower than NVK/Vulkan, it has
no production value on this hardware.

---

## Evolution Strategy

### Phase 1: Quantify (Current)

1. **NVK baseline**: Measure wgpu/Vulkan performance on both GPUs across all 9
   Yukawa cases via `bench_md_parity`
2. **Kokkos baseline**: Install LAMMPS+Kokkos-CUDA, measure same 9 cases on
   RTX 3090
3. **Gap decomposition**: Profile dispatch overhead, memory bandwidth, arithmetic
   throughput to identify highest-impact optimization targets

### Phase 2: Close the Gap (Near-Term)

Target: reduce gap from 3-4x to 1.5-2x through wgpu/Vulkan optimizations:

1. **Workgroup size tuning**: `bench_wgsize_nvk` data for per-GPU optimal sizes
2. **Shared memory usage**: WGSL workgroup-shared buffers for force accumulation
3. **Batched dispatch**: Reduce per-dispatch overhead via mega-batched encoders
4. **Algorithm optimization**: Verlet list rebuild frequency, cell-list binning

### Phase 3: Sovereign Advantage (Long-Term)

Target: coralReef sovereign dispatch matches or beats Kokkos-CUDA:

1. **New UAPI migration**: `VM_INIT`/`VM_BIND`/`EXEC` for Titan V dispatch
2. **UVM device allocation fix**: Enable RTX 3090 sovereign dispatch
3. **Per-shader register allocation**: coralReef controls register count
4. **Instruction scheduling**: coralReef reorders for memory latency hiding
5. **FMA policy per-domain**: `F64Precise` where needed, fused FMA elsewhere

### Phase 4: Beyond Kokkos

Capabilities Kokkos cannot match:

- **Runtime precision routing**: PrecisionBrain selects F32/DF64/F64 per domain
- **Adaptive intelligence**: Nautilus brain learns from physics, steers parameters
- **Heterogeneous dispatch**: Titan V (precise) + RTX 3090 (throughput) cooperative
- **Vendor portability**: Same WGSL runs on AMD, Intel, Apple — zero code changes
- **Sovereign deployment**: No driver installation, no CUDA dependency

---

## Benchmark Infrastructure

### `MdBenchmarkBackend` Trait

```rust
pub trait MdBenchmarkBackend {
    fn name(&self) -> &str;
    fn kind(&self) -> BackendKind;
    fn available(&self) -> bool;
    fn run_yukawa_md(&self, spec: &MdBenchmarkSpec) -> Result<MdBenchmarkResult, String>;
}
```

Implementations:
- `BarraCudaMdBackend` — wgpu/Vulkan (Tier 1)
- `KokkosLammpsBackend` — external LAMMPS process (Tier 3)
- (Future) `CoralReefMdBackend` — sovereign dispatch (Tier 2)

### Binary: `bench_md_parity`

```bash
cargo run --release --bin bench_md_parity                    # all 9 cases, available backends
cargo run --release --bin bench_md_parity -- --quick         # single case (k2_G158)
cargo run --release --bin bench_md_parity -- --output=results.json
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_md_parity  # Titan V only
```

---

## References

- Edwards et al., "Kokkos: Enabling manycore performance portability", JPDC 74(12), 2014
- Trott et al., "Kokkos 3: Programming model extensions for the exascale era", IEEE CiSE 24(4), 2022
- LAMMPS Kokkos: https://docs.lammps.org/Speed_kokkos.html
- NVK: https://docs.mesa3d.org/drivers/nvk.html
- coralReef Iteration 30: FMA lowering + multi-device compile API
- hotSpring Experiment 051: Hardware data capture + root cause analysis
