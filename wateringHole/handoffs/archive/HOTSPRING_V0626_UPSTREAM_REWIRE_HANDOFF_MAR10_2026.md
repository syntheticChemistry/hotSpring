# hotSpring v0.6.26 — Upstream Primal Rewire Handoff

**Date:** March 10, 2026
**From:** hotSpring v0.6.26 (842 lib tests, 0 failures, 0 clippy warnings)
**To:** barraCuda / toadStool / coralReef teams
**License:** AGPL-3.0-only

## Executive Summary

Rewire of hotSpring to latest upstream primals:

- **barraCuda** `5c16458` → `83aa08a` (5 commits): eigensolver
  (`tridiagonal_ql`), LCG PRNG, canonical activations API, Wright-Fisher
  popgen, batched f32 logsumexp shader, 5,658 LOC dead code cleaned
- **coralReef** Phase 10 Iter 26 → Iter 29: NVVM poisoning bypass
  (Iter 28) integrated into hotSpring's `PrecisionBrain`. Sovereign-safe
  tiers now tracked in `HardwareCalibration`. NVIDIA last-mile pipeline
  foundation (Iter 29) unblocks future DRM dispatch.
- **toadStool** S138 → S142: `PcieTransport` (GPU-to-GPU topology) and
  `ResourceOrchestrator` (multi-tenant GPU allocation) referenced for
  future integration. Spring absorption + pedantic sweep absorbed.

**Math is universal. Precision is silicon. Sovereignty is compilation.**

---

## Part 1: barraCuda Pin Update (`5c16458` → `83aa08a`)

### What's New Upstream

| Commit | Change |
|--------|--------|
| `8ecc75a` | Eigensolver (`tridiagonal_ql`), LCG PRNG (`rng.rs`), canonical activations API, Wright-Fisher popgen |
| `a34f28c` | Batched f32 logsumexp shader, precision test refactor |
| `5c8ebc0` | `healthSpring` provenance domain added |
| `47734a5` | 1,116 LOC orphaned code removed (`cyclic_reduction_wgsl`, `max_abs_diff_f64`) |
| `83aa08a` | 4 drifted test dirs removed, `three_springs` cross-spring tests wired |

### Impact on hotSpring

No breaking API changes. Pin update compiled cleanly. New modules
available for future use:

- `barracuda::special::tridiagonal_ql` — CPU QL eigensolver (alternative
  to nalgebra for tridiagonal problems in HFB/Lanczos)
- `barracuda::activations` — canonical relu/sigmoid/gelu/swish/mish
  (hotSpring's ESN uses a custom `relu_tanh_approx`, not affected)
- `barracuda::rng::LcgRng` — deterministic PRNG for GPU seed generation
- `barracuda::ops::wright_fisher_f32` — population genetics (not used by hotSpring)

### No Duplicates

Checked: hotSpring has no local `logsumexp` or `wright_fisher` code.
No `SpringDomain` provenance references need updating (hotSpring
doesn't use `SpringDomain` directly).

---

## Part 2: coralReef NVVM Bypass Integration

### Background

hotSpring v0.6.25 discovered that NVIDIA's proprietary NVVM compiler
permanently poisons the wgpu device when compiling certain f64 shader
patterns (DF64 transcendentals, F64Precise no-FMA, any f64-containing
shader with transcendentals on F32 path).

coralReef Iteration 28 absorbed this finding and built a sovereign
bypass: WGSL → naga → codegen IR → native SASS, entirely bypassing
NVVM. Three test fixtures validate the exact poisoning patterns:

- `nvvm_poison_f64_transcendental.wgsl` — f64 exp/log
- `nvvm_poison_df64_pipeline.wgsl` — DF64 pipeline with transcendentals
- `nvvm_poison_f64precise_nofma.wgsl` — F64Precise with `FmaPolicy::Separate`

### Integration in hotSpring v0.6.26

**`HardwareCalibration`** (`hardware_calibration.rs`):

- New field: `sovereign_compile_available: bool`
- New method: `set_sovereign_available()` — marks coralReef as detected
- New method: `tier_safe_with_sovereign(tier)` — returns true if a tier
  is safe natively OR safe via coralReef bypass
- Display: `✓sov` for sovereign-upgraded tiers, `[coralReef bypass]` suffix

**`PrecisionBrain`** (`precision_brain.rs`):

- `new()` auto-detects coralReef via XDG manifest / socket / tmp paths
- When `nvvm_transcendental_risk` is true AND coralReef is detected,
  enables sovereign routing
- `build_route_table` uses `tier_safe_with_sovereign()` instead of
  `tier_safe()`, unlocking DF64 transcendentals and F64Precise on
  proprietary NVIDIA when coralReef is available

**Routing change example** (RTX 3090 with coralReef):

| Domain | Without coralReef | With coralReef |
|--------|-------------------|----------------|
| Dielectric | F64 (F64Precise blocked) | F64Precise ✓sov |
| Eigensolve | F64 (F64Precise blocked) | F64Precise ✓sov |
| MolecularDynamics | F64 (DF64 blocked, throttled) | DF64 ✓sov |
| LatticeQcd | F64 (DF64 blocked, throttled) | DF64 ✓sov |

### Current Limitation

Sovereign compilation validates shaders but DRM dispatch is not yet
operational for NVIDIA:

- AMD amdgpu: E2E ready
- NVIDIA nouveau: compute subchannel wired but EINVAL on GV100
- NVIDIA nvidia-drm: pending UVM integration (Iter 29 advancing)

When DRM dispatch matures, the sovereign routing in `PrecisionBrain`
will automatically take effect — no hotSpring code changes needed.

---

## Part 3: toadStool S142 References

### New Upstream Capabilities

**`PcieTransport`** (S142, 455 lines):
- GPU-to-GPU topology discovery via `/sys/bus/pci`
- PCIe link width/speed detection per device
- Transport layer for measured bandwidth (vs hotSpring's spec-sheet estimates)
- `transport.open/stream/status` JSON-RPC methods

**`ResourceOrchestrator`** (S142, 489 lines):
- Multi-tenant GPU allocation with reservation/release
- Resource limits per tenant
- Mock hardware backends for testing

### Integration in hotSpring v0.6.26

`DevicePair` and `WorkloadPlanner` module docs now reference these
toadStool capabilities for future integration:

- `device_pair.rs`: Notes `PcieTransport` for topology-aware routing
- `workload_planner.rs`: Notes `ResourceOrchestrator` for multi-spring allocation

### Other S139-S142 Changes

- S139: Spring absorption, pipeline DAG, discovery alignment
- S140: Deep debt + spring absorption sprint
- S141: Pedantic sweep (120+ clippy fixes, zero-copy GPU types)

---

## Part 4: Validation Results

### Build & Test Metrics

| Metric | v0.6.25 | v0.6.26 |
|--------|---------|---------|
| lib tests | 840 | 842 (+2 sovereign bypass tests) |
| binaries | 111+ | 111+ |
| WGSL shaders | 84 | 84 |
| clippy warnings | 0 | 0 |
| unsafe blocks | 0 | 0 |
| barraCuda pin | `5c16458` | `83aa08a` |
| toadStool sync | S138 | S142 |
| coralReef sync | Iter 26 | Iter 29 |

### coralReef Iter 29 Sovereign Compile Validation (live)

| Test Suite | Pass | Fail | Ignored | Notes |
|------------|------|------|---------|-------|
| hotSpring sovereign compile (SM70+SM86) | 45/46 | 1 | 0 | `complex_f64` expected fail (utility include) |
| NVVM bypass patterns | 12/12 | 0 | 0 | All 3 patterns × 6 targets |
| Spring absorption wave 3 | 9/14 | 0 | 5 | AMD discriminant + SM70 vec3/log2 |
| Spring absorption wave 1+2 | 17/24 | 0 | 7 | RDNA2 f64 literal constants |

**Iter 26 → Iter 29 progress:** `deformed_potentials_f64` panic resolved.
Total compile rate: 45/46 (97.8%). See `experiments/050_CORALREEF_ITER29_SOVEREIGN_VALIDATION.md`.

---

## Action Items

### For barraCuda team

1. No action required — pin update was clean, no hotSpring-side changes
   needed for the new modules
2. Consider: `tridiagonal_ql` could be promoted to the default
   eigensolver for small tridiagonal problems (hotSpring may adopt for
   Lanczos completion in future)

### For coralReef team

1. **DRM dispatch for NVIDIA** (P1): hotSpring's `PrecisionBrain` is
   ready to route through sovereign compilation. 45/46 shaders compile.
   12/12 NVVM bypass patterns pass. The missing piece is DRM dispatch
   for NVIDIA (nouveau compute or nvidia-drm UVM). When this lands,
   hotSpring's NVVM-blocked tiers automatically unlock.
2. **`deformed_potentials_f64`**: ~~Panicked in Iter 26~~ **Fixed in
   Iter 29** (SSARef truncation resolved). Now compiles 5,888B SM70.
3. **RDNA2 f64 literal constant materialization** (P2): 7 f64 shaders
   fail on AMD due to VOP3 limitation (anderson, mermin, dirac, su3
   gauge, sum_reduce, yukawa, yukawa_verlet).
4. **AMD `Discriminant` expression encoding** (P2): Blocks 3 wave-3
   shaders on RDNA2 (deformed_potentials, hill_dose_response,
   population_pk).
5. **SM70 `vec3<f64>` encoding** (P3): `euler_hll_f64` hits
   `reg.comps() <= 2` assertion on Volta encoder.
6. **f64 `log2` lowering edge case** (P3): `hill_dose_response_f64`
   `pow` pattern fails log2 lowering on SM70.

### For toadStool team

1. **`PcieTransport` API stabilization**: hotSpring's `DevicePair` can
   adopt measured bandwidth from `PcieTransport` when the API stabilizes.
   Current: estimated from PCIe gen/width spec. Target: runtime-measured.
2. **`ResourceOrchestrator` integration**: When hotSpring moves to
   multi-spring dispatch (biomeGate), `ResourceOrchestrator` should
   handle GPU reservation/release instead of hotSpring's manual device selection.

---

*hotSpring v0.6.26 — upstream primal rewire complete. barraCuda pin
advanced 5 commits. coralReef sovereign bypass integrated into precision
brain. toadStool S142 PcieTransport and ResourceOrchestrator referenced
for future orchestration integration. 842 tests, 0 warnings, 0 unsafe.*
