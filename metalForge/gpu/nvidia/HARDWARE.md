# NVIDIA GPU Characterization — RTX 4070 + Titan V

**Purpose**: Document the GPU hardware that hotSpring's validated physics runs on,
establish baseline for cross-substrate comparison.

---

## Local Hardware

### RTX 4070 (Primary — all validated MD)

| Property | Value |
|----------|-------|
| Architecture | Ada Lovelace (AD104) |
| CUDA Cores | 5,888 |
| VRAM | 12 GB GDDR6X |
| Memory bandwidth | 504 GB/s |
| L2 Cache | 36 MB |
| TDP | 200W |
| PCIe Slot | `01:00.0` |
| Driver | 580.82.09 (proprietary) |
| f64 throughput | **1:2 via wgpu** (not CUDA's artificial 1:32) |

#### hotSpring Performance (validated)

| Workload | Performance | Notes |
|----------|-------------|-------|
| Yukawa MD (N=500, Γ=10) | 259 steps/s | GPU-resident, f64 |
| Yukawa MD (N=500, Γ=175) | 149 steps/s | Strong coupling, more force work |
| Energy drift (80k steps) | 0.000% | Sets precision bar |
| HFB eigensolve (791 nuclei) | 99.85% convergence | BatchedEighGpu |

### Titan V (Secondary — NVK validation)

| Property | Value |
|----------|-------|
| Architecture | Volta (GV100) |
| CUDA Cores | 5,120 |
| VRAM | 12 GB HBM2 |
| Memory bandwidth | 653 GB/s |
| L2 Cache | 4.5 MB |
| TDP | 250W |
| PCIe Slot | `05:00.0` |
| Driver | NVK (open-source Mesa) |
| f64 throughput | Native 1:2 (Volta has full-rate f64) |

#### NVK Build & Validation (Feb 21, 2026)

Pop!_OS 22.04 ships Mesa 25.1.5 without NVK compiled. We built NVK from
Mesa source (`-Dvulkan-drivers=nouveau`) and installed the ICD to
`~/.config/vulkan/icd.d/nouveau_icd.json`. The `hotspring-nouveau-titanv.conf`
modprobe config masks the NVIDIA proprietary driver's nouveau blacklist,
allowing nouveau to bind the Titan V while nvidia claims the RTX 4070.

| Check | Result |
|-------|--------|
| `vulkaninfo` | GPU1: NVIDIA TITAN V (NVK GV100), Vulkan 1.3.311 |
| SHADER_F64 | YES |
| TIMESTAMP_QUERY | YES |
| Driver / Compiler / Arch | NVK / NAK / Volta |
| `bench_gpu_fp64` | BCS bisection + eigensolve — PASS |
| `validate_cpu_gpu_parity` | **6/6 checks passed** |
| `validate_stanton_murillo` | **40/40 checks passed** (16.5 min) |
| Energy drift (5000 steps) | 0.0000% |
| D* CPU vs GPU | 17.9% relative diff (within 20% tolerance) |
| Mean energy parity | 6.0e-2 (within 8.0e-2 tolerance) |
| Stanton-Murillo transport | 6 (κ,Γ) cases, D*/η*/λ* all correct |

#### Key Finding

Titan V via NVK produces **identical physics** to RTX 4070 via proprietary
driver. This validates that BarraCUDA's wgpu path is driver-independent.

#### System Setup

```
# modprobe: /etc/modprobe.d/hotspring-nouveau-titanv.conf
#   Masks nvidia's "alias nouveau off" blacklist.
#   nouveau binds Titan V; nvidia binds RTX 4070.
# NVK ICD: ~/.config/vulkan/icd.d/nouveau_icd.json
#   Points to local Mesa NVK build at:
#   ~/Development/mesa-nvk-build/mesa-25.1.5/build/src/nouveau/vulkan/libvulkan_nouveau.so
# ToadStool barracuda detects NVK via adapter info strings and applies
#   Volta-specific driver profile (DriverKind::Nvk, CompilerKind::Nak, GpuArch::Volta).
```

---

## The f64 Discovery

BarraCUDA's defining contribution: consumer NVIDIA GPUs (GeForce) advertise
f64 at 1:32 throughput. This is a **CUDA driver limitation**, not a hardware
limitation. Via wgpu (Vulkan backend), the same silicon runs f64 at 1:2.

| GPU | CUDA f64:f32 | wgpu f64:f32 | Hardware Reality |
|-----|-------------|-------------|-----------------|
| RTX 4070 | 1:32 | **1:2** | 1:2 |
| Titan V | 1:2 (full) | **1:2** | 1:2 |
| RTX 3090 | 1:32 | **1:2** | 1:2 |

This is the precedent for metalForge: vendor SDKs present a limited view
of the hardware. Going lower finds capabilities they don't advertise.

---

## Future: AMD RDNA Cache Advantage

AMD RDNA3 GPUs (e.g., RX 7900 XTX) have up to **96 MB Infinity Cache** as
an L3, compared to NVIDIA's ~36 MB L2. For MD simulations where the
neighbor list and force arrays fit in cache, this could mean:

- Zero DRAM bandwidth pressure for moderate N (~10k particles)
- Cell-list data structures entirely cache-resident
- Potential 2-4× speedup on memory-bound force kernels

This is an unexplored frontier for computational physics on consumer GPUs.
BarraCUDA/ToadStool's wgpu backend already supports AMD via Vulkan — the
question is purely about characterization.

### Required Hardware (not yet available locally)

| Target | Price | Key Spec | Why |
|--------|-------|----------|-----|
| RX 7900 XTX | ~$700 | 96 MB Infinity Cache, 24 GB VRAM | Cache advantage study |
| RX 7800 XT | ~$450 | 64 MB Infinity Cache, 16 GB VRAM | Budget option |
| Arc A770 | ~$250 | 16 MB L2, XMX engines | Intel backend testing |

---

## ToadStool Absorption Lessons (Feb 20, 2026)

The f64 discovery was the prototype for metalForge's "probe beyond the SDK"
methodology. The same pattern has now driven 10+ GPU primitives from hotSpring
into toadstool/barracuda. Key lessons:

### What Made GPU Primitives Absorbable

1. **WGSL templates in Rust source**: Embedding WGSL shader strings directly
   in Rust modules (e.g., `WGSL_COMPLEX64` in `complex_f64.rs`) gives toadstool
   a complete, tested shader to lift without reimplementation.

2. **Binding layout documentation**: Every shader design in the handoff docs
   specifies exact `@group(0) @binding(N)` layouts, buffer types, and dispatch
   geometry. ToadStool can implement without reverse-engineering.

3. **CPU reference validates GPU correctness**: Each Rust module has unit tests
   that serve as the acceptance criteria for the GPU shader. If the GPU shader
   matches the CPU to 1e-10, it's correct.

4. **Driver-aware compilation**: All shaders route through
   `ShaderTemplate::for_driver_profile()` which handles NVK exp/log
   workarounds automatically. Shaders don't need per-driver variants.

### Absorption Timeline

| hotSpring Version | What Was Written | When Absorbed | Absorption Commit |
|------------------|-----------------|---------------|-------------------|
| v0.5.6 | SpinOrbitGpu integration | Already upstream | — |
| v0.5.8 | ShaderTemplate feedback | v0.5.15 rewire | — |
| v0.5.12 | ReduceScalar feedback | Same session | — |
| v0.5.13 | CellListGpu bug report + local fix | Session 25 | `8fb5d5a0` |
| v0.5.16 | Complex64, SU(3), plaquette, HMC, Higgs designs | Session 25 | `8fb5d5a0` |
| v0.5.16 | FFT need documented | Session 25 | `1ffe8b1a` |

### Impact: GPU Lattice QCD on Consumer Hardware

With toadstool Session 25, the full lattice QCD GPU stack exists:
- `complex_f64.wgsl` + `su3.wgsl` — algebraic foundation
- `wilson_plaquette_f64.wgsl` — gauge observable
- `su3_hmc_force_f64.wgsl` — gauge evolution
- `higgs_u1_hmc_f64.wgsl` — matter field coupling
- `Fft1DF64` / `Fft3DF64` — momentum-space transforms

A 16^4 SU(3) lattice (~37 MB) fits entirely in RTX 4070's 12 GB VRAM with
room for 300+ lattice copies. GPU HMC could run 100-1000x faster than CPU.
