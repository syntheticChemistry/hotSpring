# NVK f64 Dispatch — Findings & Absorption Handoff

**Source**: hotSpring experiments 031–032 (RTX 3090 via NVK/NAK GA102)
**Target**: toadStool/barracuda sovereign compiler + dispatch layer
**Date**: 2026-03-02

## 1. Issues Found

### 1.1 Naga WGSL f64 Validation Failures

**Problem**: Naga's WGSL frontend rejects several valid f64 operations on NVK:

| Issue | WGSL Code | Error | Fix |
|-------|-----------|-------|-----|
| Modulo | `x % two_pi` (f64 operands) | Binary `%` not supported for f64 | Use `x - floor(x / two_pi) * two_pi` |
| Division | `x / two_pi` in floor arg | Naga rejects f64 division | Requires sovereign SPIR-V path (bypasses naga WGSL) |
| Literal typing | `let pi: f64 = 3.14159...` | Type mismatch: f64 expected, f32 got | Use `f64(3.14159...)` constructor syntax |
| `enable f64;` directive | `enable f64;` at top of shader | Not recognized as valid global item | Not needed — sovereign path sets capabilities in SPIR-V directly |

**Root cause**: Naga's WGSL validator doesn't fully support f64 extension types.
The `enable f64;` directive is spec-valid but Naga doesn't parse it.

**Resolution**: toadStool's sovereign compiler path (naga IR → SPIR-V emission with
`SPIRV_SHADER_PASSTHROUGH`) bypasses all these issues. The `ShaderTemplate::for_driver_auto`
correctly strips `enable f64;` because the sovereign path injects f64 capabilities
directly in the SPIR-V module.

### 1.2 FMA Fusion Changes CG Convergence

**Problem**: The sovereign compiler's FMA fusion (a*b+c → fma(a,b,c)) changes
rounding behavior in the CG solver's dot products and vector updates. This doesn't
break correctness but changes the convergence trajectory.

**Observation**: Both sovereign (FMA) and WGSL-text (no FMA) paths converge to
the same answer. The FMA path has slightly different iteration counts but
identical physics results.

**Decision**: hotSpring uses `create_pipeline_f64_precise` (WGSL-text path, no FMA)
for CG-critical kernels (Dirac, dot product, axpy, xpay) and `create_pipeline_f64`
(sovereign path with FMA) for everything else. This is a conservative choice —
toadStool may want to make this configurable or prove FMA-safety for CG.

### 1.3 GPU-to-CPU Readback Latency on NVK

**Problem**: `device.poll(Maintain::Wait)` + `map_async` readback takes 10–50ms
per call on NVK vs 0.1–1ms on proprietary drivers. For CG solvers that readback
a convergence scalar every N iterations, this overhead dominates.

**Impact on 4^4 lattice** (256 sites):
- 62 CG calls per Omelyan trajectory × ~200ms per CG call = ~12s per trajectory
- With per-dispatch overhead: ~42s per trajectory (vs ~2s on proprietary)

**Impact on 8^4 lattice** (4096 sites):
- Proportionally more compute per CG iteration offsets the readback overhead
- Expected ~20× improvement in readback-to-compute ratio vs 4^4

## 2. Solutions Built (for toadStool absorption)

### 2.1 Latency-Adaptive CG Check Interval

**File**: `barracuda/src/lattice/gpu_hmc/resident_cg_brain.rs`

After the first CG readback, measures readback latency. If > 5ms (NVK territory),
automatically scales `check_interval` to maintain at least 5× compute-to-readback
ratio:

```
readback = 40ms, 10 iters took 10ms compute
→ target: 40ms × 5 = 200ms compute
→ us_per_iter = 10ms/10 = 1ms
→ new check_interval = 200ms / 1ms = 200 iterations
→ clamp to [current, 200]
```

This reduces CG readback count from ~500 to ~3-5 per solve on NVK, while
preserving the same convergence guarantee.

**Absorption target**: `barracuda::gpu::CgSolverOptions` or
`barracuda::dispatch::AdaptiveCheckInterval`

### 2.2 Dispatch Coalescing

**File**: `barracuda/src/lattice/gpu_hmc/resident_cg_brain.rs`

Batches sequential GPU dispatches into single encoder submissions:

- **Pre-CG**: gauge force + momentum update → 1 submit (was 2)
- **Post-CG**: Dirac + fermion force + fermion momentum → 1 submit (was 3)
- **Link update**: encoder-based single pass (was separate dispatch)

Saves 3 vkQueueSubmit calls per force evaluation:
- 3 forces/step × 20 steps × 3 submits saved = 180 fewer submits/trajectory
- At ~2ms/submit on NVK: ~360ms saved per trajectory (~4% on 4^4)

**Absorption target**: toadStool's dispatch layer should support coalesced
encoder patterns natively, detecting NVK-class drivers and auto-batching.

### 2.3 Precise vs Sovereign Pipeline Selection

**File**: `barracuda/src/gpu/mod.rs`

Two compilation paths:
- `create_pipeline_f64()` — sovereign SPIR-V with FMA fusion (fast, for non-CG shaders)
- `create_pipeline_f64_precise()` — WGSL-text through naga (no FMA, for CG kernels)

**Absorption target**: `barracuda::device::CompilationStrategy::Precise | Sovereign`

## 3. NVK Performance Profile (RTX 3090 GA102)

| Metric | NVK | Proprietary (est.) | Ratio |
|--------|-----|--------------------|-------|
| Scalar readback | 10–50ms | 0.1–1ms | 50× |
| vkQueueSubmit overhead | ~2ms | ~0.1ms | 20× |
| CG convergence (4^4) | ~200ms/solve | ~10ms/solve | 20× |
| Full trajectory (4^4) | ~42s | ~2s | 21× |
| f64 DFMA throughput | ~1/32 FP32 | ~1/32 FP32 | 1× |

The readback and submission overhead, not compute throughput, is the primary
NVK performance gap for iterative solvers. The sovereign NAK effort should
prioritize command submission optimization alongside DFMA emulation.

## 4. Polyfill Fixes Applied to toadStool

**File**: `toadstool/crates/barracuda/src/shaders/precision/polyfill.rs`

- `sin_f64_safe`: replaced `x % two_pi` with `x - floor(x / two_pi) * two_pi`
- All f64 literals: changed `let val: f64 = N.N` to `let val = f64(N.N)` syntax
- These are naga-only issues — sovereign SPIR-V path handles them natively

## 5. Recommended toadStool NAK Priorities

1. **Command submission batching**: Auto-coalesce sequential dispatches on NVK
2. **Async readback pipeline**: Overlap compute with readback (speculative CG)
3. **Adaptive check interval**: First-readback latency probe → auto-scale
4. **f64 literal handling in naga fork**: Accept `let x: f64 = 3.14` and `%` on f64
5. **FMA safety analysis**: Prove or disprove that FMA fusion is safe for CG
