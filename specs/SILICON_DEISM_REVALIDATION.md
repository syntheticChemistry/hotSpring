# Silicon Deism Revalidation — biomeGate Methodology

**Author:** hotSpring team / biomeGate
**Date:** 2026-08-02
**Status:** ACTIVE — Phase 1 complete, Phases 2-4 pending

## Principle

> We make no assumptions. We retest, revalidate, and continue the work of the
> team before us with more modern abstractions.

Every result from the strandGate sovereign era (Exp 191–234) is **fossil
record** on biomeGate. Different hardware, different BDFs, different PCIe
topology, different kernel. The abstractions carry forward as code. The
register-level observations do not.

## What Is Silicon Deism

Silicon deism is the principle that **generation differences are data, not
code**. A single dispatch path treats all GPUs the same regardless of SM
generation (SM37/SM70/SM86/SM89/SM100), with generation-specific behavior
encoded in const profiles:

```
GenerationProfile::for_sm(sm) → one lookup, all register offsets + semantics
InterruptProfile::for_sm(sm)  → quench dispatch
PowerSafetyProfile            → staged PMC_ENABLE
SovereignStrategy trait       → boot sequence selection
```

The alternative — scattered `if sm >= 70` branches throughout the codebase —
is what silicon deism replaces.

## biomeGate Hardware Profile

| GPU | SM | Generation | BDF | Driver | Role |
|-----|-----|-----------|-----|--------|------|
| RTX 5060 | 100 | Blackwell | 0000:02:00.0 | nvidia (DRM) | wgpu production |
| Titan V GV100 | 70 | Volta | 0000:21:00.0 | vfio-pci | Sovereign dispatch |
| K80 GK210 fn0 | 37 | Kepler | 0000:4b:00.0 | vfio-pci | Cross-gen validation |
| K80 GK210 fn1 | 37 | Kepler | 0000:4c:00.0 | vfio-pci | Cross-gen validation |

Three generations spanning 10 years of GPU architecture. If the diesel engine
works identically on SM37, SM70, and SM100 — it works everywhere.

## Methodology

### Phase 1: Cold-State Register Semantics (COMPLETE)

**No warm cycles. No driver init. Pure MMIO on cold silicon.**

Read and write generation-critical registers to prove the hardware semantics
that the diesel engine abstractions encode:

| Register | SM37 (Kepler) | SM70 (Volta+) | Exp |
|----------|--------------|---------------|-----|
| INTR_EN_0@0x140 | R/W | R/O | 236 |
| INTR_EN_SET@0x160 | different HW | W/O (set) | 236 |
| INTR_EN_CLEAR@0x180 | different HW | W/O (clear) | 236 |
| PMC_ENABLE@0x200 | cold mask | cold mask | 235 |
| PMC_BOOT0@0x0 | chip ID | chip ID | 235 |

**Result:** InterruptProfile HARDWARE-PROVEN (6/7 PASS). The 1 failure was
infrastructure (FLR damage on fn0), not abstraction error.

### Phase 2: Warm-State Sovereign Boot (PENDING)

**Warm the GPUs, then validate diesel engine boot sequences.**

| Step | GPU | Method | Target |
|------|-----|--------|--------|
| 2a | K80 fn1 | nouveau insmod (bypass blacklist) | Warm PMC_ENABLE |
| 2b | K80 fn0 | PCIe SBR or power cycle | Recover from FLR |
| 2c | Titan V | catalyst (nvsov + rm_trigger) | Tier 1 WarmInfrastructure |
| 2d | Titan V | sovereign.classify_tier | Tier 2 WarmCompute |
| 2e | K80 | sovereign.init with PRE_VOLTA | SovereignStrategy validation |

Prerequisites:
- `sudo apt install flex` (fixes kmod compilation)
- Build + install nvidia-470 nvsov module
- `cargo build --release --bin rm_trigger` + install
- Override nouveau blacklist or force-insmod

### Phase 3: Cross-Gen Dispatch (PENDING)

**Run the same shader on different GPUs and compare results.**

| Experiment | Shader | SM37 Path | SM70 Path | SM100 Path |
|------------|--------|-----------|-----------|------------|
| Scalar reduce | sum_reduce_f64 | wgpu fallback | sovereign VFIO | wgpu subgroup |
| Plaquette | su3_plaquette_f64 | N/A (no wgpu) | sovereign | wgpu |
| PRNG (TMU) | su3_random_momenta_tmu | N/A | sovereign | wgpu |
| lower_f64 sqrt | Newton-Raphson | sovereign | sovereign | wgpu (polyfill) |

The goal is not identical throughput — it's identical correctness. Same
plaquette value, same reduction sum, same PRNG distribution.

### Phase 4: Production Parity (PENDING)

**Run the full HMC pipeline and compare physics observables.**

| Observable | Target | Tolerance |
|------------|--------|-----------|
| Plaquette ⟨P⟩ at β=6.0 | GPU ≈ CPU | |ΔP| < 3σ |
| Acceptance rate | > 50% | physically reasonable |
| Autocorrelation τ | < 20 | efficient sampling |
| Box-Muller χ² | p > 0.01 | correct PRNG distribution |

SM100 (RTX 5060) is the production physics path via wgpu. SM70 (Titan V)
via sovereign dispatch validates the compiler pipeline. SM37 (K80) is
the cross-gen proof — if Kepler gets the same physics, the abstractions
are complete.

## Fossil Record Policy

### What Gets Fossilized

- Register-level observations from other gates (different BDFs)
- Silicon profiles from hardware we don't have
- Experiment runs on hardware configurations we can't reproduce
- BAR0 captures with BDF-specific PRAMIN offsets

### What Carries Forward

- Code abstractions (GenerationProfile, InterruptProfile, etc.)
- Architecture decisions (warm boot > cold boot, catalyst pattern)
- Known-bad approaches (bulk PMC_ENABLE, WGSL f64 polyfills)
- Failure taxonomy (5 lockup vectors, FLR destruction)

### What Gets Retested

Everything. Every register read. Every quench. Every boot sequence. Every
shader result. On biomeGate silicon, at biomeGate BDFs, with biomeGate's
kernel and driver stack.

## Experiment Numbering

strandGate experiments: 191–234 (fossilized)
biomeGate experiments: 235+ (active)

| # | Title | Status |
|---|-------|--------|
| 235 | Fleet bootstrap + BAR0 first contact | COMPLETE |
| 236 | Cross-gen quench probe | COMPLETE (6/7) |
| 237 | RTX 5060 silicon profile | COMPLETE (27/28) |
| 238 | K80 warm cycle + full quench | PENDING (blocked: nouveau) |
| 239 | Titan V catalyst warm handoff | PENDING (blocked: nvsov) |
| 240 | Titan V Tier 2 revalidation | PENDING (blocked: Exp 239) |
| 241 | Cross-gen dispatch parity | PENDING (blocked: warm GPUs) |
| 242 | Production HMC cross-validation | PENDING (blocked: Exp 241) |

## Success Definition

Silicon deism is achieved when:

1. `InterruptProfile::for_sm(sm)` is hardware-proven on SM37 + SM70 ✓
2. `PowerSafetyProfile` prevents real damage (proven by fn0 FLR) ✓
3. `SovereignStrategy` boots both Kepler and Volta without SM-specific code
4. The same WGSL shader produces the same physics on SM37, SM70, SM100
5. coralReef `lower_f64/` generates correct transcendentals on all three SMs
6. Production HMC plaquette values match CPU reference across all GPUs

Items 1-2 are complete. Items 3-6 require warm GPUs.
