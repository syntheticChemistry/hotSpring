# NVIDIA GPU Characterization — RTX 4070 + RTX 3090 + Titan V

**Purpose**: Document the GPU hardware that hotSpring's validated physics runs on,
establish baseline for cross-substrate comparison.

---

## Local Hardware

### RTX 4070 (Eastgate — all validated MD)

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
| f64 throughput | Hardware ~1:64 (AD104: limited FP64 units). DF64 (f32-pair) at ~3 TFLOPS. See `bench_fp64_ratio`. |

#### hotSpring Performance (validated)

| Workload | Performance | Notes |
|----------|-------------|-------|
| Yukawa MD (N=500, Γ=10) | 259 steps/s | GPU-resident, f64 |
| Yukawa MD (N=500, Γ=175) | 149 steps/s | Strong coupling, more force work |
| Energy drift (80k steps) | 0.000% | Sets precision bar |
| HFB eigensolve (791 nuclei) | 99.85% convergence | BatchedEighGpu |

### RTX 3090 (biomeGate — large-lattice compute)

| Property | Value |
|----------|-------|
| Architecture | Ampere (GA102) |
| CUDA Cores | 10,496 |
| VRAM | 24 GB GDDR6X |
| Memory bandwidth | 936 GB/s |
| L2 Cache | 6 MB |
| TDP | 350W |
| Driver | nvidia proprietary |
| f64 throughput | Hardware ~1:64 (GA102: 164 FP64 units, 0.33 TFLOPS). DF64 (f32-pair) at 3.24 TFLOPS. See `bench_fp64_ratio`. |

biomeGate's primary compute GPU. The 24 GB VRAM is 2× the RTX 4070, enabling
GPU-resident dynamical fermion lattices up to 24⁴ without sublattice decomposition.
The 936 GB/s memory bandwidth (1.86× the 4070) benefits memory-bound lattice
sweeps. Ampere GA102 uses the same nvidia proprietary Vulkan driver path as Ada
AD104 — all existing WGSL shaders run unmodified.

#### VRAM Capacity: Lattice Size vs GPU

SU(3) dynamical fermion HMC uses ~3.3 KB/site (gauge links, momenta, force,
9 fermion fields, neighbor table, phase table, reduction scratch).

| Lattice | Sites | VRAM | RTX 4070 (12 GB) | RTX 3090 (24 GB) | RTX 5090 (32 GB) |
|---------|------:|-----:|:-:|:-:|:-:|
| 8⁴ | 4,096 | 13 MB | Yes | Yes | Yes |
| 16⁴ | 65,536 | 209 MB | Yes | Yes | Yes |
| 24⁴ | 331,776 | 1.06 GB | Yes | Yes | Yes |
| 32⁴ | 1,048,576 | 3.3 GB | Yes | Yes | Yes |
| 40⁴ | 2,560,000 | 8.2 GB | Yes | Yes | Yes |
| 44⁴ | 3,748,096 | 11.9 GB | Tight | Yes | Yes |
| 48⁴ | 5,308,416 | 16.9 GB | No | Yes | Yes |
| 56⁴ | 9,834,496 | 31.3 GB | No | Tight (quenched) | Yes |
| 64⁴ | 16,777,216 | 53.4 GB | No | No | No (sublattice) |

The RTX 3090's 24 GB unlocks 48⁴ — a lattice volume 2× larger than the 4070's
practical maximum. For quenched QCD (no fermion buffers), up to ~56⁴ fits.

### Titan V (Eastgate + biomeGate — NVK validation)

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
driver. This validates that BarraCuda's wgpu path is driver-independent.

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

## The f64 Discovery (Corrected Feb 24, 2026)

**Original claim (incorrect):** Consumer GPUs run f64 at 1:2 via wgpu/Vulkan.

**Corrected via `bench_fp64_ratio` FMA chain micro-benchmark:**

| GPU | CUDA fp64 | Vulkan fp64 | Hardware | Strategy |
|-----|-----------|-------------|----------|----------|
| RTX 4070 (Ada) | ~1:64 | ~1:64 | ~1:64 (limited FP64 units) | DF64 hybrid |
| RTX 3090 (Ampere) | 0.29 TFLOPS | 0.33 TFLOPS | 1:64 (164 FP64 units) | DF64 hybrid |
| Titan V (Volta) | N/A (NVK only) | 0.59 TFLOPS | **1:2 (2560 FP64 cores)** | Native f64 |

Consumer Ampere/Ada fp64 is hardware ~1:64 — confirmed by both CUDA and Vulkan
giving the same fp64 throughput. The ratio IS silicon, not driver software.

**The real discovery:** Double-float (f32-pair) arithmetic on the FP32 cores
delivers **3.24 TFLOPS** at 14-digit precision — **9.9× faster than native
f64** on consumer GPUs. This "core streaming" strategy routes bulk math to
the massive FP32 array and reserves native f64 for precision-critical ops.

The Titan V has genuine 1:2 hardware (same GV100 die as Tesla V100),
accessible through the open-source NVK driver even after NVIDIA dropped
Volta from proprietary driver support.

This remains a metalForge precedent: understanding actual hardware behavior
(not just SDK documentation) reveals better strategies than either vendor path.

---

## Future: AMD RDNA Cache Advantage

AMD RDNA3 GPUs (e.g., RX 7900 XTX) have up to **96 MB Infinity Cache** as
an L3, compared to NVIDIA's ~36 MB L2. For MD simulations where the
neighbor list and force arrays fit in cache, this could mean:

- Zero DRAM bandwidth pressure for moderate N (~10k particles)
- Cell-list data structures entirely cache-resident
- Potential 2-4× speedup on memory-bound force kernels

This is an unexplored frontier for computational physics on consumer GPUs.
BarraCuda/ToadStool's wgpu backend already supports AMD via Vulkan — the
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
room for 300+ lattice copies. The RTX 3090's 24 GB fits 48⁴ dynamical fermion
lattices — 80× more sites than 16⁴. GPU HMC could run 100-1000x faster than CPU.
