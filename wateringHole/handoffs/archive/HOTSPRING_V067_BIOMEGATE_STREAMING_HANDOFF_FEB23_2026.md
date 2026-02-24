# hotSpring v0.6.7 — biomeGate Prep + Streaming CG + Debt Fix Handoff

**Date:** February 23, 2026
**From:** hotSpring (computational physics biome)
**To:** ToadStool / BarraCuda core team + biomeGate deployment
**License:** AGPL-3.0-only
**Purpose:** Document biomeGate node preparation, GPU-resident CG streaming pipeline,
upstream API debt fixes, and 155/155 validation pass prior to biomeGate handoff.

---

## Executive Summary

biomeGate (Threadripper 3970X, RTX 3090 + Titan V, Akida NPU, 256GB DDR4) is prepped
for deployment. The codebase compiles cleanly, all 155/155 validation checks pass across
15 suites, and node profiles make "git pull and run" trivial. GPU-resident CG achieves
15,360× readback reduction and 30.7× speedup for dynamical fermion HMC.

---

## Part 1: biomeGate Node Preparation

### Hardware Registered
- Added to whitePaper/gen3/about/HARDWARE.md (tower #10, 256GB DDR4, RTX 3090 + Titan V + Akida)
- RTX 3090 characterized in metalForge/gpu/nvidia/HARDWARE.md (24GB, Ampere GA102, VRAM capacity table)
- Aggregate capabilities updated: 10 towers, ~176GB GPU VRAM, ~1.2TB system RAM, 4× Akida (213 NPUs)

### Node Profile System
- metalForge/nodes/biomegate.env — HOTSPRING_GPU_ADAPTER=3090, HOTSPRING_GPU_PRIMARY=3090, HOTSPRING_GPU_SECONDARY=titan
- metalForge/nodes/eastgate.env — HOTSPRING_GPU_ADAPTER=4070 (formalized existing config)
- metalForge/nodes/README.md — usage guide, variable reference, "adding a new node" checklist

### Multi-GPU Made Node-Agnostic
- bench_multi_gpu.rs: hardcoded "4070"/"titan" replaced with HOTSPRING_GPU_PRIMARY/HOTSPRING_GPU_SECONDARY env vars
- Backward-compatible defaults (4070/titan) for Eastgate
- Phase 4 specialized routing now uses dynamic adapter names

### NVK Setup Guide
- metalForge/gpu/nvidia/NVK_SETUP.md: reproducible 6-step checklist for Titan V NVK on any node
- Covers Mesa build, ICD install, modprobe dual-driver config, verification, troubleshooting
- biomeGate-specific notes for Threadripper PCIe topology

### Scaling Estimates Updated
- whitePaper/STUDY.md Section 7.11: biomeGate capacity envelope (RTX 3090 VRAM table, campaign time estimates)
- 48⁴ dynamical fermion lattice fits in 24GB (16.9GB used)
- NUCLEUS cluster totals updated to include biomeGate

---

## Part 2: API Debt Fixes

### reservoir.rs (solve_f64 → CPU Gauss-Jordan)
- Upstream barracuda::linalg::solve_f64 now requires Arc<WgpuDevice> (GPU solver)
- ESN ridge regression uses small matrices (50–200 dim) that don't need GPU
- Replaced with local gauss_jordan_solve() — partial pivoting, f64, no external deps
- Validated: validate_reservoir_transport 10/10 pass, ESN train error 12.4%

### nuclear_eos binaries (sampler/surrogate device args)
- nuclear_eos_l1_ref.rs: direct_sampler and sparsity_sampler now take device as first arg
- nuclear_eos_gpu.rs: direct_sampler calls updated with device from GpuF64
- nuclear_eos_l2_hetero.rs: RBFSurrogate::train and sparsity_sampler updated
- nuclear_eos_l2_ref.rs: direct_sampler and sparsity_sampler updated

### Build Status
- cargo build --release: exit 0, warnings only (no errors)
- All 4 previously-broken binaries now compile

---

## Part 3: Validation Results (155/155)

| Suite | Checks | Time |
|-------|--------|------|
| validate_cpu_gpu_parity | 6/6 | 5.7s |
| validate_gpu_dirac | 8/8 | 1.0s |
| validate_gpu_cg | 9/9 | 1.6s |
| validate_pure_gauge | 12/12 | 48.8s |
| validate_dynamical_qcd | 7/7 | 91.8s |
| validate_gpu_streaming | 9/9 | 641.9s |
| validate_gpu_streaming_dyn | 13/13 | 796.4s |
| validate_abelian_higgs | 17/17 | 0.8s |
| validate_hotqcd_eos | 5/5 | 0.7s |
| validate_screened_coulomb | 23/23 | 1.4s |
| validate_gpu_spmv | 8/8 | 2.3s |
| validate_gpu_lanczos | 6/6 | 3.2s |
| validate_barracuda_evolution | 19/19 | 9.4s |
| validate_pure_gpu_qcd | 3/3 | 3.4s |
| validate_reservoir_transport | 10/10 | 167.1s |
| **Total** | **155/155** | **~1775s** |

---

## Part 4: What biomeGate Enables

### Immediate (git pull and run)
- `source metalForge/nodes/biomegate.env && cargo run --release --bin validate_gpu_streaming` — validates streaming on RTX 3090
- RTX 3090 24GB: 48⁴ dynamical fermion lattice GPU-resident (2× Eastgate's 40⁴ max)
- Threadripper 32c/64t: 32 parallel CPU β-values for temperature scans

### Extended Campaigns
| Campaign | Lattice | Est. time/traj | 1000 traj | 20-pt β-scan |
|----------|---------|---------------:|----------:|-------------:|
| Quenched β-scan | 48⁴ | ~25s | 7 hrs | 6 days |
| Dynamical QCD | 40⁴ | ~120s | 33 hrs | 28 days |
| Full dynamical | 48⁴ | ~300s | 83 hrs | 69 days |

### Dual-GPU Cooperative
- RTX 3090: production compute
- Titan V (NVK): verification runs (different driver, same physics)
- bench_multi_gpu validates cooperative dispatch automatically

---

## Part 5: ToadStool Absorption Targets

### New Shaders Ready for Absorption
| Shader | File | Purpose |
|--------|------|---------|
| sum_reduce_f64.wgsl | lattice/shaders/ | Tree reduction N→1 f64 |
| cg_compute_alpha_f64.wgsl | lattice/shaders/ | CG α = rz/pAp |
| cg_compute_beta_f64.wgsl | lattice/shaders/ | CG β = rz_new/rz_old |
| cg_update_xr_f64.wgsl | lattice/shaders/ | x += α*p, r -= α*ap |
| cg_update_p_f64.wgsl | lattice/shaders/ | p = r + β*p |

### Rust Patterns for Absorption
- GpuResidentCgPipelines: compiled pipeline cache for all CG sub-operations
- GpuResidentCgBuffers: GPU-resident scalar buffers + pre-built bind groups
- build_reduce_chain(): dynamic multi-pass reduction construction
- AsyncCgReadback: double-buffered staging for speculative batch overlap
- BidirectionalStream: std::sync::mpsc channels for CPU+NPU observation routing

### gauss_jordan_solve() for CPU contexts
- Clean CPU fallback when GPU device not available
- Pattern: upstream solve_f64 requires device; local code provides CPU path
- Suitable for toadstool's barracuda::linalg as an optional CPU fallback

---

## Part 6: Active Handoff Documents

| Document | Scope |
|----------|-------|
| **This document** (V067 biomeGate) | Node prep, streaming CG, API debt, 155/155 validation |
| `HOTSPRING_V067_BARRACUDA_EVOLUTION_HANDOFF_FEB22_2026.md` | Comprehensive BarraCuda evolution handoff |
| `HOTSPRING_V067_TOADSTOOL_SESSION42_HANDOFF_FEB22_2026.md` | S40-42 catch-up, loop_unroller fix |
| `HOTSPRING_V066_GPU_TRANSPORT_HANDOFF_FEB22_2026.md` | GPU-resident transport pipeline |
| `HOTSPRING_V064_DYNAMICAL_QCD_HANDOFF_FEB22_2026.md` | Pseudofermion HMC details |
| `CROSS_SPRING_EVOLUTION_FEB22_2026.md` | Cross-spring shader map |

All prior handoffs archived in `wateringHole/handoffs/archive/`.

---

*This handoff follows the unidirectional pattern: hotSpring → wateringHole → ToadStool.
No inter-Spring imports. All code is AGPL-3.0.*
