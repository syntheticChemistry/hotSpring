# Experiment 050: coralReef Iter 29 Sovereign Pipeline Validation

**Date:** March 10, 2026
**hotSpring:** v0.6.26
**coralReef:** Phase 10, Iteration 29 (`2779c88`)
**barraCuda:** `83aa08a`
**toadStool:** S142

---

## Purpose

Validate the sovereign compilation pipeline against hotSpring's full
WGSL shader corpus using coralReef Iteration 29. Measure progress
since Iter 26 and document remaining gaps for upstream evolution.

---

## Results

### hotSpring Shader Compilation (validate_sovereign_compile)

**45/46** shaders compile to native SM70 + SM86 SASS. **0 panics.**

| Shader | SM70 | SM86 | Notes |
|--------|------|------|-------|
| 44 physics shaders | OK | OK | Nuclear, lattice, MD, ESN |
| `deformed_potentials_f64` | **OK** | **OK** | **Fixed since Iter 26** (was PANIC: SSARef truncation) |
| `complex_f64` | FAIL | FAIL | Utility include, not standalone entry point (expected) |

**Progress since Iter 26:** 44/46 → 45/46. `deformed_potentials_f64` unblocked.

**Largest compiled shaders** (most complex physics):
- `su3_link_update_f64`: 26,400B, 62 GPR, 1,642 instr
- `su3_gauge_force_f64`: 20,512B, 54 GPR, 1,274 instr
- `dirac_staggered_f64`: 18,800B, 46 GPR, 1,167 instr
- `yukawa_force_celllist_v2_f64`: 14,496B, 78 GPR, 898 instr

**Total native output:** ~220 KB per target architecture.

### NVVM Bypass Validation (coralReef nvvm_bypass tests)

**12/12** tests pass. All three NVVM-poisoning patterns compile through
coralReef's sovereign path across all target architectures.

| Pattern | SM70 | SM75 | SM80 | SM86 | SM89 | RDNA2 |
|---------|------|------|------|------|------|-------|
| f64 transcendentals (exp/log) | OK | OK | OK | OK | OK | OK |
| DF64 pipeline + transcendentals | OK | OK | OK | OK | OK | OK |
| F64Precise no-FMA (FmaPolicy::Separate) | OK | OK | OK | OK | OK | OK |

### Spring Absorption Wave 3 (coralReef corpus tests)

**9/14** pass, **5 ignored** with known gaps:

| Shader | SM70 | RDNA2 | Gap |
|--------|------|-------|-----|
| `deformed_potentials_f64` | OK | IGNORED | AMD `Discriminant` expression encoding |
| `diversity_f64` | OK | OK | -- |
| `euler_hll_f64` | IGNORED | OK | SM70 `vec3<f64>` encoding assertion |
| `hill_dose_response_f64` | IGNORED | IGNORED | f64 `log2` lowering edge case (SM70); AMD `Discriminant` (RDNA2) |
| `population_pk_f64` | OK | IGNORED | AMD `Discriminant` expression encoding |
| `verlet_build` | OK | OK | -- |
| `verlet_check_displacement` | OK | OK | -- |

### Spring Absorption Wave 1+2 (coralReef cross-spring tests)

**17/24** pass, **7 ignored** (all RDNA2 f64 literal constant materialization):

| Gap | Affected | Root Cause |
|-----|----------|------------|
| RDNA2 f64 literal constants | 7 shaders (anderson, mermin, dirac, su3 gauge, sum_reduce, yukawa, yukawa_verlet) | VOP3 limitation: f64 ops need literal constant materialization |
| All SM70 shaders | 17/17 OK | No NVIDIA gaps in wave 1+2 |

---

## Remaining Gaps for Upstream

### coralReef — Compilation Gaps

| Priority | Gap | Impact | Shaders |
|----------|-----|--------|---------|
| P1 | **DRM dispatch for NVIDIA** | Blocks sovereign dispatch on RTX 3090, Titan V | All 45 compiled shaders |
| P2 | RDNA2 f64 literal constant materialization (VOP3) | Blocks 7 f64 shaders on AMD | anderson, mermin, dirac, su3, sum_reduce, yukawa, yukawa_verlet |
| P2 | AMD `Discriminant` expression encoding | Blocks 3 wave-3 shaders on RDNA2 | deformed_potentials, hill_dose_response, population_pk |
| P3 | SM70 `vec3<f64>` encoding | Blocks `euler_hll_f64` on Volta | 1 shader |
| P3 | f64 `log2` lowering edge case | Blocks `hill_dose_response_f64` on SM70 | 1 shader |
| P4 | `complex_f64` utility include | Not standalone; expected failure | 1 non-shader |

### coralReef — Dispatch Gaps

| Target | Compile | Dispatch | Blocker |
|--------|---------|----------|---------|
| SM70 (Titan V) | 45/46 | Not operational | nouveau EINVAL on GV100 compute |
| SM86 (RTX 3090) | 45/46 | Not operational | nvidia-drm pending UVM |
| RDNA2 (AMD) | 38/46 | **E2E ready** | None — fully operational |

### toadStool — Orchestration Gaps

| Priority | Gap | Impact |
|----------|-----|--------|
| P2 | `PcieTransport` API stabilization | hotSpring can adopt measured bandwidth |
| P2 | `ResourceOrchestrator` multi-spring integration | Needed for biomeGate |
| P3 | `transport.open/stream/status` JSON-RPC client | Cross-primal GPU sharing |

### barraCuda — No Gaps

Pin update was clean. No blockers.

---

## Summary

The sovereign compilation pipeline is **near-complete for NVIDIA** (45/46
shaders, 12/12 NVVM bypass patterns). The single remaining compile
failure (`complex_f64`) is a non-standalone utility include.

The **dispatch gap is the critical path**: coralReef can compile the exact
shaders that poison NVVM, but DRM dispatch for NVIDIA is not yet
operational. AMD RDNA2 is E2E ready. NVIDIA Iteration 29 is advancing
the last-mile pipeline (ioctl + UVM).

When NVIDIA DRM dispatch lands, hotSpring's `PrecisionBrain` will
automatically route NVVM-blocked tiers through the sovereign path,
unlocking DF64 transcendentals (~3.24 TFLOPS) and F64Precise no-FMA
on the RTX 3090.
