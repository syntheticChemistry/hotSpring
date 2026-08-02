# biomeGate Revalidation Specification — Diesel Engine + hotQCD

**Date**: Aug 2, 2026
**Gate**: biomeGate (GPU crankshaft)
**Status**: Active — revalidation in progress
**License**: AGPL-3.0-only

---

## Purpose

biomeGate returned from kernel failure with three GPU architectures spanning 10 years
of NVIDIA silicon (Kepler SM37 → Volta SM70 → Ada/Blackwell SM89+). This spec defines
the revalidation goals for two workstreams before new work begins:

1. **Diesel engine revalidation** — Sovereign GPU boot, warm handoff, cross-generation
   dispatch through the toadStool/coralReef pipeline. Validates silicon deism: all GPUs
   are the same to the compiler and dispatch engine.

2. **hotQCD revalidation** — Lattice QCD physics, PRNG polyfill correctness, GPU HMC
   streaming, silicon-tier routing across the new card fleet. Validates that science
   answers are correct on biomeGate hardware before advancing production.

## Hardware Profile

| GPU | Arch | SM | Compute | VRAM | VFIO BDF | Silicon Units |
|-----|------|----|---------|------|----------|---------------|
| RTX 5060 | Ada/Blackwell | 89+ | wgpu host | 8 GB | host | FP32, TMU, ROP, Tensor, RT |
| Titan V | Volta (GV100) | 70 | VFIO sovereign | 12 GB HBM2 | `21:00.0` | FP32, **FP64 1:2**, TMU, Tensor V1 |
| K80 die 0 | Kepler (GK210) | 37 | VFIO sovereign | 12 GB GDDR5 | `4b:00.0` | FP32, **FP64 1:3**, TMU |
| K80 die 1 | Kepler (GK210) | 37 | VFIO sovereign | 12 GB GDDR5 | `4c:00.0` | FP32, **FP64 1:3**, TMU |

**Notable**: Titan V has 1:2 FP64:FP32 ratio (HPC-class). K80 has 1:3 FP64:FP32 (also
HPC-class). Consumer cards (RTX 3090, RTX 5060) are 1:64. biomeGate is the only gate
with HPC-grade FP64 silicon for native f64 validation.

---

## Workstream 1: Diesel Engine Revalidation

### Goal

Prove that the toadStool diesel engine (ember/glowplug/cylinder) handles all three GPU
generations through a single code path. The `InterruptProfile`, `GenerationProfile`,
`PowerSafetyProfile`, and `SovereignStrategy` abstractions must dispatch correctly for
both Volta and Kepler — the generation boundary is where silicon deism either works or
breaks.

### Scope

| Experiment | SM Target | What It Validates | Success Criteria |
|------------|-----------|-------------------|------------------|
| 193 | K80 (37) | PLX D3cold keepalive | Hierarchy pin prevents switch death on new topology |
| 197 | Both | `sovereign.init` RPC | Cold start timing + register fingerprint |
| 199 | Both | Diesel engine sovereign boot | `bar0_source=ember` pipeline completes |
| 200 | K80 (37) | Power safety profiles | `PowerSafetyProfile::kepler()` stages PMC_ENABLE safely |
| 201 | Titan V (70) | Volta cold boot CG sweep | CG + PRI recovery + PGOB ungating |
| 204 | Titan V (70) | VBIOS interpreter | 422 ops, 231 BAR0 writes, stride fixes |
| 213 | Titan V (70) | Live warm handoff | `sovereign.classify_tier` returns correct tier |
| 219 | Titan V (70) | Catalyst driver pattern | nvidia-470 catalyst, BAR0 golden state capture |
| 227 | Titan V (70) | Tier 2 breakthrough | TPC probe fix at `0x50400c`, `tpc_alive=true` |
| 230 | Both | Diesel abstraction reval | `InterruptProfile` generation dispatch |
| **231** | **K80 (37)** | **Cross-gen quench probe** | **`PRE_VOLTA` writes 0x0 @ 0x140 (not 0xFF @ 0x180)** |
| 182 | K80 (37) | FECS PIO boot | K80 FECS via programmed I/O |
| 183 | K80 (37) | FECS interrupt boot | K80 FECS via interrupt-driven path |
| 184 | K80 (37) | GR sovereign init | Kepler GR initialization |
| 234 | Titan V (70) | Catalyst warm handoff | End-to-end VFIO sovereign dispatch |

### Key Abstractions Under Test

```
InterruptProfile::for_sm(sm) → PRE_VOLTA | VOLTA_PLUS
  PRE_VOLTA:   disable_offset=0x140, disable_value=0x00000000
  VOLTA_PLUS:  disable_offset=0x180, disable_value=0xFFFFFFFF

GenerationProfile::for_sm(sm) → name, ce_class, register offsets
  SM37 (Kepler): GPC_TPC offsets differ from Volta
  SM70 (Volta):  GPC broadcast at 0x41a004, FECS at 0x409624

PowerSafetyProfile → generation-aware PMC_ENABLE staging
  Kepler: staged writes to prevent GDDR5 thermal event (Exp 199/200)
  Volta:  full enable safe (HBM2 doesn't have the same risk)
```

### BDF Adaptation

All experiments from strandGate used BDFs `02:00.0` / `49:00.0`. biomeGate BDFs:

| strandGate BDF | biomeGate BDF | Card |
|----------------|---------------|------|
| `0000:02:00.0` | `0000:21:00.0` | Titan V |
| `0000:49:00.0` | (was 2nd Titan V) | N/A on biomeGate |
| N/A (K80 was dead) | `0000:4b:00.0` | K80 die 0 |
| N/A | `0000:4c:00.0` | K80 die 1 |

All experiment binaries accept `--bdf` CLI arg (Exp 230 abstraction). No code changes
needed — pass biomeGate BDFs at runtime.

---

## Workstream 2: hotQCD Revalidation

### Goal

Prove that lattice QCD science answers are correct on biomeGate hardware. The strandGate
PRNG root-cause AAR showed that GPU PRNG polyfills (`log_f64`/`sqrt_f64`/`cos_f64` in
Box-Muller) produced wrong physics (plaquette ⟨P⟩ 570σ off). The `cpu_mom` workaround
is in production. biomeGate validates the fix path: either coralReef-native compilation
with `lower_f64/` transcendental lowering, or corrected WGSL preambles.

### QCD Validation Matrix

| Domain | Binary / Test | What It Validates | Target GPU |
|--------|--------------|-------------------|------------|
| Bare scenarios | `hotspring_unibin validate` | 18 physics scenarios offline | CPU-only (no GPU needed) |
| Compute trio | `validate_compute_trio_pipeline` | Yukawa + Wilson plaquette via IPC | RTX 5060 (wgpu) |
| GPU HMC streaming | `gpu_hmc_trajectory_streaming_cpu_mom` | Correct physics with CPU momenta | RTX 5060 (wgpu) |
| PRNG parity | compare GPU vs CPU Box-Muller output | Polyfill correctness | RTX 5060 + Titan V |
| Silicon profiling | `bench_silicon_profile` | TMU, FP32, FP64, ROP, Subgroup | RTX 5060, Titan V, K80 |
| f64 transcendental | coralReef `lower_f64/` test suite | Newton-Raphson, polynomial accuracy | SM37, SM70, SM89 |
| DF64 Dekker | DF64 parity tests | f32-pair arithmetic vs native f64 | All three GPUs |
| RHMC | `production_dynamical_mixed` | Pseudofermion HMC on GPU | RTX 5060 (wgpu) |

### Silicon Tier Routing — New Profiles Needed

biomeGate introduces three cards not yet profiled:

| Card | Profiling Status | Key Silicon Feature |
|------|-----------------|---------------------|
| **RTX 5060** | NOT PROFILED | Unknown TMU count, FP64 ratio, tensor gen |
| **Titan V** | Partially profiled (strandGate, different driver) | **1:2 FP64** — native f64 king |
| **K80** | NOT PROFILED | **1:3 FP64**, no tensor, no RT, GDDR5 |

Run `bench_silicon_profile` on each to generate `profiles/silicon/*.json` before
production QCD dispatches. The silicon tier router uses these profiles to route kernels
to the cheapest available silicon unit.

### PRNG Fix Path — Two Options

**Option A: coralReef native compilation** (fast, avoids WGSL polyfills entirely)

Route momentum generation through coralReef's `lower_f64/` pipeline where
transcendentals are lowered in IR (Newton-Raphson + MUFU on NVIDIA, native on AMD).
Bypasses the broken WGSL polyfills. Requires sovereign dispatch (toadStool) to be
working on biomeGate.

**Option B: Corrected WGSL preambles** (complete, fixes root cause for all consumers)

Ship validated WGSL f64 transcendental preambles from coralReef:
- `log_f64_preamble.wgsl` — validated against `libm` log
- `sqrt_f64_preamble.wgsl` — validated against `libm` sqrt  
- `cos_f64_preamble.wgsl` / `sin_f64_preamble.wgsl` — validated trig
- `box_muller_f64_preamble.wgsl` — complete Box-Muller using validated transcendentals

biomeGate validates both options across SM37 (Kepler), SM70 (Volta), and SM89 (Ada) —
the three-generation spread proves vendor-agnosticism.

### Precision Validation

| Metric | Target | Method |
|--------|--------|--------|
| f64 sqrt relative error | < 1 ULP | Compare `sqrt_f64(x)` vs `libm::sqrt(x)` for 10K random inputs |
| f64 log relative error | < 2 ULP | Compare `log_f64(x)` vs `libm::log(x)` |
| f64 cos/sin relative error | < 4 ULP | Compare against `libm` trig |
| Box-Muller χ² | p > 0.01 | Chi-squared test of GPU-generated Gaussian distribution |
| Plaquette parity | |GPU - CPU| < 3σ | Wilson plaquette expectation value vs CPU reference |

---

## Cross-Cutting Concerns

### coralReef Compiler Coverage

coralReef already compiles for all three SM targets on biomeGate:

| Target | Backend | Status |
|--------|---------|--------|
| SM37 (Kepler) | SASS encode (`sm35/`) | Compile tests pass |
| SM70 (Volta) | SASS encode (`sm70/`) | Compile tests pass |
| SM89 (Ada) | SASS encode (`sm86/sm89/`) | Compile tests pass |
| SM100+ (Blackwell) | PTX emit | Compile tests pass |

The compiler can target all three GPUs regardless of whether they're physically present.
biomeGate's hardware validates that the compiled binaries actually execute correctly.

### toadStool Compute Dispatch

toadStool provides three dispatch paths relevant to biomeGate:

| Path | Code | GPU | Status |
|------|------|-----|--------|
| VFIO cold | `cylinder/vfio/` | Titan V, K80 | Ready (devices bound) |
| VFIO warm (catalyst) | `cylinder/vfio/sovereign_handoff.rs` | Titan V | Ready (Exp 227/234) |
| wgpu local | `runtime/gpu/` | RTX 5060 | Ready (host driver) |

### songBird Cross-Gate Dispatch

When both biomeGate and strandGate are on the mesh, `compute_dispatch/cross_gate.rs`
enables cross-gate GPU leasing via songBird IPC:
- strandGate (RTX 3090 + RX 6950 XT) handles high-VRAM QCD production
- biomeGate handles diesel engine validation and PRNG polyfill testing
- GAP-HS-005 (ionic GPU lease) prototype ready for live test

---

## Success Criteria

Revalidation is complete when:

1. **Diesel engine**: All Phase 1-5 experiments pass on biomeGate BDFs
2. **Exp 231**: K80 cross-gen quench probe validates `InterruptProfile::PRE_VOLTA`
3. **K80 sovereign**: At least one K80 experiment binary runs on live silicon
4. **Titan V Tier 2**: Exp 227 reproduces `tpc_alive=true` on biomeGate Titan V
5. **Silicon profiles**: RTX 5060 + Titan V + K80 profiles generated
6. **PRNG**: At least one fix path (A or B) produces correct Box-Muller on GPU
7. **Plaquette parity**: GPU HMC with fixed PRNG matches CPU within 3σ
8. **hotQCD bare**: `hotspring_unibin validate` — 18/18 scenarios pass

After these criteria are met, biomeGate advances to production diesel engine work
(coralReef polyfill shipping) and hotQCD continuous compute.

---

*biomeGate revalidation spec — silicon deism across 3 GPU generations.*
