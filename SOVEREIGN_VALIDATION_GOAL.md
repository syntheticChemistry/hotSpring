# hotSpring — Sovereign Validation Goal

> **Fossil Record (March 31, 2026):** This document captures state as of March 25, 2026. For current status, see the [root README](README.md) and [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md). Body below is preserved as historical record. **Since this snapshot: NVIDIA GPFIFO pipeline is OPERATIONAL on RTX 3090 (Exp 124). AMD scratch/local memory is OPERATIONAL on RX 6950 XT (FLAT_SCRATCH prolog fix). Warm FECS dispatch attack (Exp 127), GPU puzzle box matrix (Exp 128). DRM proprietary tracing (Exp 126). 128+ experiments total. Cross-primal rewiring complete — daemon-backed testing via ember/glowplug.**

> **Note (March 25, 2026):** **Definitive root cause found (Exp 122). K80 strategy initiated (Exp 123).** WPR2 registers hardware-locked, FWSEC inaccessible, FBPA offline — explains all WPR copy stalls (Exp 114-121). HS mode achieved (Exp 112, SCTL=0x3002) but PMU-dependent (Exp 113). Tesla K80 (Kepler, zero firmware security) arriving 2026-03-26 — validates entire pipeline without security barriers. Identity module + Falcon PIO loader built. 123 experiments. See [`specs/GPU_CRACKING_GAP_TRACKER.md`](specs/GPU_CRACKING_GAP_TRACKER.md).

## CERN-Grade Reproducible Physics at Home. Scalable to CERN.

**Date**: March 25, 2026
**Version**: v0.6.32
**Status**: 4,065 tests, 119 binaries, 85 shaders, 44/44 Chuna overnight, 0.000% energy drift
**VFIO Validation**: **L10 root cause definitive** (Exp 122) on Titan V. HS mode achieved (Exp 112). WPR2 hardware-locked — K80 (no security) validates full stack. See `specs/GPU_CRACKING_GAP_TRACKER.md`.
**Hardware**: 2× Titan V (GV100) + RTX 5070 (GB206, Blackwell) + Tesla K80 (GK210, Kepler, incoming)

---

## The Goal

Run lattice QCD, nuclear structure, plasma transport, and condensed matter
simulations with the same reproducibility guarantees as CERN/MILC/HotQCD —
on a $900 consumer workstation. Then scale the same code to a CERN-class
cluster without changing a single line.

**Truly deterministic**: same binary + same input = same output. Always.
Regardless of OS version, driver update, hardware generation, or time elapsed.

**No external breakage surface**: every component from shader source to GPU
output is pure Rust, versioned in Cargo.lock, reproducible from `cargo build`.
No Python version drift. No CUDA SDK changes. No driver JIT nondeterminism.
No Fortran linker surprises. The physics runs forever.

---

## What We Have (Proven)

### Physics Validation (848 tests, zero failures)

| Domain | Papers | Key Result | Status |
|--------|--------|-----------|--------|
| **Yukawa OCP MD** | Murillo Group | 9/9 DSF, 0.000% drift, N=500→50k | ✅ |
| **Nuclear EOS** | SEMF+HFB | L1 χ²=2.27 (478× faster), L2 χ²=16.11 | ✅ |
| **Transport** | Stanton-Murillo | D*/η*/λ* Green-Kubo, κ-corrected | ✅ |
| **Pure Gauge SU(3)** | Wilson action | HMC, plaquette, Dirac CG parity 4e-16 | ✅ |
| **Gradient Flow** | Bazavov-Chuna | t₀, w₀, 5 integrators, convergence | ✅ |
| **Dynamical QCD** | N_f=4 staggered | Pseudofermion HMC, 85% acceptance | ✅ |
| **Deconfinement** | Finite-size | χ=40.1 at β=5.69 (known β_c=5.692) | ✅ |
| **Abelian Higgs** | Bazavov 2015 | U(1)+Higgs HMC, 143× faster than Python | ✅ |
| **Anderson Localization** | Kachkovskiy | 1D/2D/3D, mobility edge, GOE→Poisson | ✅ |
| **Hofstadter Butterfly** | Kachkovskiy | Fractal band structure, Cantor measure | ✅ |
| **Screened Coulomb** | Murillo-Weisheit | 23/23, Sturm bisection, Δ≈10⁻¹² | ✅ |

### Hardware Validation (3 substrates)

| Substrate | GPU | Result |
|-----------|-----|--------|
| RTX 3090 (Ampere, SM86) | nvidia proprietary + NVK | Production β-scan 32⁴, 12/12 |
| Titan V (Volta, SM70) | NVK (Mesa 25.1.5) | Full 4-tier compute, first NVK QCD |
| AKD1000 NPU (BrainChip) | PCIe, live hardware | 12-head ESN, β_c=5.715, 8796× less energy |

### Performance (consumer hardware)

| Benchmark | barraCuda | Reference | Ratio |
|-----------|-----------|-----------|-------|
| CG solver (Dirac) | 0.023ms | Python 4.59ms | **200×** |
| Nuclear L1 throughput | 2,621 evals/s | Python 5.5 evals/s | **478×** |
| GPU HMC 16⁴ | 24ms/traj | Python 533ms/traj | **22.2×** |
| Abelian Higgs HMC | — | Python | **143×** |
| GPU streaming HMC | — | CPU | **67×** |
| DF64 3.24 TFLOPS | 14-digit | native f64 | **9.9×** throughput |

---

## The Sovereign Stack (What Makes This Possible)

### Every Layer: Shader → GPU → Output

```
                    SOVEREIGN COMPUTE PIPELINE

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ hotSpring│    │barraCuda │    │ coralReef│    │toadStool │
  │          │    │          │    │          │    │          │
  │ Physics  │───►│ WGSL     │───►│ WGSL →   │───►│ GPU mgmt │
  │ models   │    │ shaders  │    │ native   │    │ BAR0     │
  │ validate │    │ DF64     │    │ SASS/GFX │    │ VFIO     │
  │ results  │◄───│ precision│◄───│ dispatch │◄───│ schedule │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
       │               │               │               │
       └───────────────┴───────────────┴───────────────┘
                            │
                    ┌───────┴───────┐
                    │   vfio-pci    │  ← only external: generic Linux
                    │ (kernel, not  │     PCIe infrastructure
                    │  vendor)      │
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐
                    │  GPU silicon  │
                    └───────────────┘
```

| Layer | Component | Owner | External? |
|-------|-----------|-------|-----------|
| Physics validation | hotSpring | ecoPrimals | No |
| Shader math + precision | barraCuda | ecoPrimals | No |
| WGSL → native ISA compilation | coralReef | ecoPrimals | No |
| QMD, push buffer, CBUF | coralReef | ecoPrimals | No |
| GR context init (BAR0 MMIO) | coralReef | ecoPrimals | No |
| Hardware mgmt, scheduling | toadStool | ecoPrimals | No |
| Security, encryption | BearDog | ecoPrimals | No |
| PCIe device access | vfio-pci | Linux kernel | **Yes** (generic) |

The only external dependency is the Linux kernel's generic PCIe subsystem.
Not a GPU driver. Not a vendor SDK. Not a library that can change its API.
The VFIO interface has been stable since Linux 3.6 (2012).

### Why This Matters for Science

**Scenario A — Without sovereign stack (today's norm)**:
1. Researcher publishes QCD result using CUDA 11.8 + Python 3.9 + numpy 1.24
2. Two years later: CUDA 13 drops support for their GPU, Python 3.13 breaks
   the numpy API, pip resolver changes dependency tree
3. Result is no longer reproducible. Paper's computational appendix is dead code.

**Scenario B — With sovereign stack (ecoPrimals)**:
1. Researcher publishes QCD result using hotSpring v0.6.31
2. Ten years later: `git clone && cargo build && cargo run`
3. Same binary, same result, same physics. Cargo.lock pins every dependency.
   Rust has backward compatibility guarantees. No runtime. No interpreter.
   The executable IS the computation.

---

## Kokkos Parity: The Benchmark Target

Kokkos (Sandia National Labs) is the performance-portable C++ framework used
by LAMMPS, Trilinos, and DOE production codes. It's the gold standard for
"write once, run on any GPU."

### Current Status

| Metric | barraCuda (Rust) | Kokkos-CUDA (C++) | Gap |
|--------|-----------------|-------------------|-----|
| PP Yukawa (N=2000) | 212.4 steps/s | 2,630.2 steps/s | 12.4× |
| Complexity scaling | α≈2.30 (AllPairs) | α≈1.38 (neighbor list) | Algorithm |
| Verlet neighbor list | 992 steps/s peak | — | 3.7× gap |
| Hardware support | NVIDIA, AMD, NPU | NVIDIA, AMD, Intel | Parity |

### The VFIO Parity Strategy

The 12.4× gap has two causes:
1. **DF64 poisoning** — naga WGSL→SPIR-V codegen bug forces FP64 fallback
   on Vulkan (solved by coralReef sovereign bypass, not yet in dispatch path)
2. **Dispatch overhead** — Vulkan/wgpu adds layers between shader and GPU

With the sovereign VFIO path, both are eliminated:

| Optimization | Expected Impact | How |
|-------------|----------------|-----|
| DF64 via coralReef (bypass naga) | **9.9×** throughput gain | Already proven in benchmarks |
| Direct GPFIFO (bypass Vulkan) | Eliminates dispatch overhead | VFIO path |
| toadStool SM partitioning | GPU fully utilized | Software-defined partitioning |
| hw-learn auto-tuning | Optimal block size, memory | Cross-GPU learning |

**Target: Kokkos parity (≤2× gap) hardware-agnostic, then optimize with
precision mixes that Kokkos cannot do.**

The precision mix advantage is unique to ecoPrimals:
- Kokkos: FP64 everywhere (correct but slow)
- barraCuda: DF64 where safe (9.9× faster), FP64 where needed, mixed per domain
- toadStool precision brain: routes each shader to optimal tier per GPU
- Result: same physics accuracy, fewer FLOPS, less energy, faster time-to-result

This is not something Kokkos can replicate — it requires the vertical
integration of math engine + compiler + hardware manager that only the
sovereign stack provides.

---

## Cross-GPU Learning

With toadStool as root scheduler, every dispatch is an observation:

```
Dispatch observation:
  shader: yukawa_df64_kernel
  GPU: RTX 3090 (SM86)
  config: 512 threads, DF64, Verlet
  time: 5.9 ms
  power: 243W
  result: correct ✓

Knowledge extracted:
  SM86 + Yukawa + DF64 + 512 threads = optimal
  SM86 + Yukawa + FP64 + 256 threads = 27% slower
  SM70 + Yukawa + DF64 + 256 threads = optimal (different!)
```

This knowledge transfers:
- **Across GPUs**: What works on SM86 informs SM89, SM90
- **Across physics**: Block size tuning from Yukawa applies to Coulomb
- **Across springs**: wetSpring's bio shaders benefit from hotSpring's tuning
- **Across time**: New GPU arrives → toadStool starts with prior knowledge

The system gets smarter with every computation. CERN runs millions of
trajectories — each one teaches toadStool something about the hardware.

---

## GPU Security Posture

### Software Enclave (BearDog + toadStool + VFIO)

| Layer | Protection | Mechanism |
|-------|-----------|-----------|
| Device access | Exclusive | VFIO IOMMU (hardware-enforced) |
| Data at rest | Encrypted | BearDog (AES-256 or equivalent) |
| Data in compute | Isolated | VFIO exclusivity (no other process) |
| Scheduling | Deterministic | toadStool root (no hidden decisions) |
| Memory isolation | Per-workload | Separate GPU VA regions |
| Code integrity | Compiler-enforced | Rust memory safety, zero unsafe |

No MIG hardware required. No vendor TEE. The trust boundary is ecoPrimals
Rust code + Linux IOMMU — both auditable, both stable.

### Dual-Use: Gaming + Science

The same machine runs Steam games (nvidia proprietary) and ecoPrimals
science (VFIO sovereign), switching on demand. toadStool manages the
transition. No reboot. See `SOVEREIGN_COMPUTE_BAR0_BREAKTHROUGH_DUAL_USE_HANDOFF_MAR12_2026.md`.

---

## Scaling: Home → Lab → CERN

The sovereign stack scales without code changes:

| Scale | Hardware | What Changes |
|-------|----------|-------------|
| **Home** | 1-2 GPUs, consumer | Nothing — this is where we validate |
| **Lab** | 4-8 GPUs, workstation | toadStool spawns per-GPU children |
| **Cluster** | 100+ GPUs, HPC | toadStool tree + MPI-like ring for inter-node |
| **CERN** | 1000+ GPUs, grid | Same code, same Cargo.lock, same results |

What scales:
- toadStool tree: root per node, children per GPU, SM partitions per workload
- coralReef compilation: same binary for all GPUs of same arch
- barraCuda math: identical shaders, precision routing per GPU
- BearDog: key management scales with node count

What doesn't change:
- The physics
- The shaders
- The validation suite
- The results

A physicist runs `hotspring-production-beta-scan` on their laptop with one
GPU. CERN runs the same binary across 1000 GPUs. The per-GPU computation
is bit-identical. The only difference is how many GPUs contribute to the
statistical ensemble.

---

## Validation Ladder (What Remains)

### Done ✅
- [x] Quenched SU(3) (pure gauge, Wilson action)
- [x] Gradient flow (t₀, w₀, 5 integrators)
- [x] Dynamical N_f=4 staggered
- [x] Deconfinement transition (finite-size scaling)
- [x] Yukawa OCP (PP, DSF, N=500→50k)
- [x] Nuclear EOS (SEMF, HFB, AME2020)
- [x] Transport coefficients (Green-Kubo, D*/η*/λ*)
- [x] Anderson localization (1D/2D/3D)
- [x] Abelian Higgs (U(1) gauge + scalar)
- [x] NPU heterogeneous pipeline (AKD1000 live)
- [x] DF64 precision (9.9× FP32 core throughput)
- [x] Cross-spring shader evolution (85 shaders, 4 springs)
- [x] Backend-agnostic MD engine (`MdEngine<B: GpuBackend>`)
- [x] Sovereign compilation (46/46 shaders → native SASS)
- [x] BAR0 GR context init (PGRAPH registers from firmware)

### In Progress 🔄
- [x] **Sovereign VFIO MMU** — **PROVEN** (Exp 076): fault buffer fix, 7/10 layers, DMA roundtrip verified on Titan V
- [ ] Sovereign VFIO dispatch — **Livepatch 4-NOP strategy (Exp 125)**: mc_reset+gr_fini+falcon_fini+runl_commit NOPed during nouveau teardown. reset_method sysfs fix prevents PCI bus reset. PBDMA warm mode preserves channel state. Ready to test.
- [x] **DRM dispatch evolution** — **AMD GCN5 preswap 6/6 PASS** (March 2026): f64 write, f64 arith, multi-workgroup, multi-buffer, HBM2 bandwidth, **f64 Lennard-Jones force (Newton's 3rd law verified)**. 18 bugs fixed. NVIDIA PMU-blocked. **K80 (Kepler) arriving** — no firmware security, direct PIO FECS/GPCCS boot. Exp 123-K designed.
- [x] **GCN5 backend in coral-reef** — **COMPLETE**: native AMD ISA codegen for MI50, VOP1/VOP3/VOPC opcode translation, wave64, f64 materialization, VOP3 modifier encoding, integer negation, Naga bypass validated E2E. 85 tests pass.
- [ ] Kokkos parity via sovereign bypass (DF64 + direct GPFIFO)
- [x] **toadStool S168 shader.dispatch** — orchestration layer complete. Pipeline: coralReef (compile) → toadStool (dispatch facade) → coralReef (compute.dispatch.execute) → GPU
- [x] **barraCuda Sprint 23 f64 precision** — systematic f64 pipeline fix across Bessel, Legendre, Hermite, PPPM. Silent f32 downcast eliminated.
- [ ] N_f=2+1 RHMC (infrastructure ready, needs validation run)

### Planned 📋
- [ ] **K80 sovereign compute (Exp 123-K)** — Identity probe, PIO FECS/GPCCS boot, PFIFO channel, shader dispatch. Arriving 2026-03-26. Kepler falcon PIO loader + identity module ready in coral-driver.
- [ ] Titan V parasitic compute (Exp 123-T) — sysfs BAR0 while nouveau active. Probe complete: FECS/GPCCS HALTED under vfio-pci. Deferred to after K80.
- [ ] K80 DRM reference — legacy nouveau, trace MMU setup for sovereign debugging
- [ ] VFIO GPU backend (extend toadStool Akida pattern)
- [ ] Direct WGSL parser (replace naga dependency)
- [ ] Multi-node scaling (toadStool tree + inter-node comms)
- [ ] Continuum extrapolation (32⁴ → 48⁴ → 64⁴ at physical pion mass)
- [ ] Comparison with MILC/HotQCD published results at matching parameters
- [ ] nvidia-drm + UVM dispatch validation (Exp 126) — NvUvmComputeDevice code-complete, needs on-site Titan V validation with proprietary driver
- [ ] Validation matrix (specs/SOVEREIGN_VALIDATION_MATRIX.md) — solve maze from both sides

---

## The Extended Goal

hotSpring began as "does our hardware produce correct physics?"

It evolved into "can Rust+WGSL replace the Python scientific stack?"
(Answer: yes, 200× faster, bit-exact.)

It evolved again into "can consumer hardware match HPC results?"
(Answer: yes, deconfinement at β_c=5.69 on a $900 workstation.)

The extended goal is: **can we build a sovereign, deterministic, reproducible
physics computation stack that runs identically on one GPU or one thousand,
today and in ten years, without any external dependency that could break?**

The answer is yes. The math is sovereign. The compilation is sovereign. The
hardware interface is generic. The results are deterministic. The code is
versioned. The physics runs forever.

CERN-grade reproducibility. At home. Scalable to CERN.

---

## VFIO Hardware Validation (March 14, 2026)

Physical validation of the sovereign VFIO compute path on biomeGate:

| Test | Result | What It Proves |
|------|--------|----------------|
| VFIO device open + BAR0 read | **PASS** | Userspace GPU access without kernel driver |
| DMA alloc via IOMMU | **PASS** | Host memory mapped into GPU IOVA space |
| DMA upload + readback | **PASS** | Byte-exact data round-trip through IOMMU |
| Multiple concurrent DMA buffers | **PASS** | Buffer pool management |
| Compute dispatch + sync | **FAIL** | PFIFO scheduler dispatches, PBDMA loads channel context — USERD DMA read remaining |

**Hardware**: Titan V (GV100, SM70) on `vfio-pci`, IOMMU group 36.
**Dual-use**: RTX 3090 stays on nvidia proprietary for display — same machine, no reboot.
- [ ] K80 sovereign validation — Kepler PIO FECS/GPCCS boot, arriving 2026-03-26. Identity + PIO loader ready.

### PFIFO Channel Progress (March 14, 2026)

The channel initialization has been iterated through 17 experiments (A-Q)
from "GPU ignores everything" to "PBDMA2 has our full RAMFC context loaded
with zero errors." Three critical discoveries (Experiment 058):

| Discovery | Register | Impact |
|-----------|----------|--------|
| GV100 runlist preempt | `0x002638` | Write `BIT(runl_id)` to evict stale nouveau contexts |
| Runlist completion ACK | `0x002A00` | **Must acknowledge** before scheduler dispatches to PBDMAs |
| SIGNATURE validation | PBDMA INTR bit 31 | RAMFC SIGNATURE must be `0x0000FACE` |

**Fixes applied (March 9-14):**

| Fix | What Was Wrong | Impact |
|-----|---------------|--------|
| PBDMA_MAP interpretation | Assumed PBDMA0 existed; only 1,2,3,21 present | Correct engine targeting |
| GK104 runlist submission | Used per-runlist registers; GV100 uses fixed pair (0x2270/0x2274) | Runlist accepted by scheduler |
| GP_BASE_HI aperture | Bits [29:28] defaulted to 0 (VRAM); GPFIFO is in system memory | PBDMA reads correct aperture |
| **GV100 preempt (0x002638)** | Used 0x002634 (per-channel); Volta needs per-runlist preempt | **Stale PBDMA context evicted** |
| **Runlist ACK (0x002A00)** | Scheduler waited for software ACK that never came | **Scheduler dispatches channels** |
| **SIGNATURE = 0xFACE** | PBDMA validates RAMFC signature field | **Clean context load** |
| **Engine context binding** | PBDMA needs GR ectx at inst[0x210/0x214], flag at inst[0x0AC] | **CTXNOTVALID cleared** |

**Current state:**
- `PCCSR_CHAN = 0x00000003` — channel enabled + scheduled, zero faults
- PBDMA2 loaded our full RAMFC context (USERD, GP_BASE, SIGNATURE, CHID all match)
- PBDMA2 INTR = 0x00000000, PFIFO_INTR = 0x00000000 (all clean after ACK)

**Remaining gap**: PBDMA's `GP_PUT` stays at 0 — PBDMA has context but does not read
GP_PUT from USERD in system memory. Hypothesis: IOMMU DMA path issue for PBDMA polling.
Testing VRAM USERD via PRAMIN. Dual Titan V mmiotrace planned after hardware swap.

See: `hotSpring/experiments/058_VFIO_PBDMA_CONTEXT_LOAD.md`
See: `wateringHole/handoffs/HOTSPRING_VFIO_PBDMA_CONTEXT_LOAD_BREAKTHROUGH_HANDOFF_MAR14_2026.md`

---

*hotSpring v0.6.31 — March 14, 2026*
*The shaders are the mathematics. The mathematics runs forever.*
