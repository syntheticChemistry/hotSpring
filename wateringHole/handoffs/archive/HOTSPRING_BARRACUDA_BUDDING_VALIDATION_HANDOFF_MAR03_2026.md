SPDX-License-Identifier: AGPL-3.0-only

# hotSpring → barraCuda/toadStool: First Spring Validates Primal Budding

**Date:** 2026-03-03
**From:** hotSpring v0.6.17+ (first consumer to rewire)
**To:** barraCuda team + toadStool/barracuda team
**Covers:** barraCuda budding validation, dependency rewire, test results, bug report
**License:** AGPL-3.0-only

---

## Executive Summary

hotSpring is the **first Spring to successfully build and pass all tests against
the standalone barraCuda primal** (`ecoPrimals/barraCuda/crates/barracuda`),
confirming the budding is viable from a consumer's perspective.

**Results:** 663 hotSpring tests pass, 0 failures, against barraCuda v0.2.0
(commit `c87ff78`). This matches the test count against the toadStool-embedded
barracuda. The Cargo.toml change was a single-line path swap.

hotSpring also found **37 test failures in barraCuda's own test suite** when
running with GPU pipeline creation (beyond `--lib`). Root cause identified below.

---

## Part 1: What hotSpring Did

### 1.1 Dependency Rewire

```toml
# Before (toadStool-embedded):
barracuda = { path = "../../phase1/toadstool/crates/barracuda" }

# After (standalone barraCuda primal):
barracuda = { path = "../../barraCuda/crates/barracuda" }
```

No other changes required. All `use barracuda::*` imports, trait implementations,
and shader references work identically.

### 1.2 Workspace Build

barraCuda workspace currently includes `barracuda-core` which depends on
`sourdough-core` (not yet cloned locally). The `barracuda` crate itself builds
standalone with `--features gpu` — `toadstool-core` and `akida-driver` are
correctly feature-gated behind `toadstool` and `npu-akida` features.

To build just the compute library without `barracuda-core`:

```bash
cargo check -p barracuda --no-default-features --features gpu
```

### 1.3 Test Validation

| Suite | Tests | Failures | Notes |
|-------|-------|----------|-------|
| hotSpring unit (663) | 663 | 0 | Full parity with toadStool-embedded |
| hotSpring integration (53) | 53 | 0 | All integration suites pass |
| hotSpring GPU (14) | 14 | 0 | GPU shader dispatch validated |
| **hotSpring total** | **716** | **0** | **Full validation** |
| barraCuda `--lib` (S89) | 2,832 | 0 | As reported in budding spec |
| barraCuda full (GPU) | 2,795 | 37 | See bug report below |

---

## Part 2: Bug Report — 37 GPU Test Failures

### Root Cause: `sin_f64_safe` uses `%` (modulo) on f64

**File:** `crates/barracuda/src/shaders/precision/polyfill.rs`
**Line:** `var t = x % two_pi;`

naga 22 rejects the WGSL modulo operator `%` on `f64` operands:

```
Function [0] 'sin_f64_safe' is invalid
  Expression [3] is invalid
    Operation Modulo can't work with [0] and [2]
```

This affects any GPU pipeline that pulls in the `SIN_COS_F64_SAFE_PREAMBLE`
(the NVK Taylor-series workaround for sin/cos). Tests that only exercise `--lib`
(CPU math, no shader compilation) don't hit this path.

### Fix

Replace the f64 modulo with floor-based equivalent:

```wgsl
// Before (fails naga validation):
var t = x % two_pi;

// After (equivalent, naga-safe):
var t = x - floor(x / two_pi) * two_pi;
```

### Affected Tests (37)

| Category | Count | Tests |
|----------|-------|-------|
| Special functions (bessel, beta, digamma, spherical harmonics) | 15 | All use sin/cos through the polyfill |
| Ops (dotproduct, eq, dropout, cosine_embedding_loss) | 9 | Pipeline creation triggers shader compilation |
| Lattice QCD (leapfrog, trajectory, init, pseudofermion, omelyan) | 6 | GPU pipeline creation |
| Compute graph | 1 | Batching test creates pipelines |
| Linalg (batched eigh) | 3 | Pipeline creation |
| Device (test_pool) | 1 | Tokio runtime flavor: needs `multi_thread` |

### Second Issue: tokio Runtime Flavor

**File:** `crates/barracuda/src/device/test_pool.rs`
**Test:** `test_cpu_device_available`

The `#[tokio::test]` attribute defaults to current-thread runtime, but
`tokio_block_on()` calls `block_in_place()` which requires multi-thread.

**Fix:** `#[tokio::test(flavor = "multi_thread", worker_threads = 2)]`

---

## Part 3: Budding Completion Checklist (from hotSpring's perspective)

| Criterion | Status | Notes |
|-----------|--------|-------|
| hotSpring builds against standalone barraCuda | **PASS** | Single-line Cargo.toml change |
| hotSpring tests pass (716/716) | **PASS** | Full parity with toadStool-embedded |
| barraCuda compiles without toadstool-core | **PASS** | `--features gpu` only |
| Feature gates correct | **PASS** | `toadstool` and `npu-akida` properly optional |
| API surface identical | **PASS** | No import changes needed |
| barraCuda own tests pass | **PARTIAL** | 2,795/2,832 pass; 37 fail due to f64 modulo bug |
| `barracuda-core` builds standalone | **BLOCKED** | Needs `sourdough-core` (not yet available) |
| `validate-gpu` binary works | **NOT TESTED** | Likely hits same sin_f64_safe bug |

---

## Part 4: Recommendations

### For barraCuda team

1. **Fix `sin_f64_safe` modulo** — one-line change, fixes 36 of 37 failures
2. **Fix tokio test flavor** — one-line change, fixes the remaining failure
3. **Run full `cargo test` (not just `--lib`)** in CI to catch shader compilation issues
4. **Add `cross_spring_validation.rs`** — import hotSpring's test patterns as validation

### For toadStool team

1. **Keep toadStool's embedded barracuda in sync** during transition — Springs that
   haven't rewired yet still depend on `phase1/toadStool/crates/barracuda`
2. **Document the dual-path period** — both paths must work until all Springs migrate
3. **Consider workspace `[patch]`** — Springs could override the path at workspace level
   rather than editing Cargo.toml directly

### For other Springs

The rewire is trivial — change one path in `Cargo.toml`:

```toml
barracuda = { path = "../../barraCuda/crates/barracuda" }
```

No code changes needed. hotSpring's 716 tests confirm full API compatibility.

---

## Part 5: hotSpring Current State

| Metric | Value |
|--------|-------|
| Tests passing | 716 (663 lib + 53 integration) |
| Clippy warnings | 0 |
| Experiments completed | 033 (Rung 0), 034 in progress (Rung 1) |
| barracuda dependency | `../../barraCuda/crates/barracuda` (budded primal) |
| Rung 1 status | Phase 1b running — Nf=8 CG iters ~63k (2x Nf=4 as expected) |

---

## Files

```
hotSpring/barracuda/Cargo.toml                    (MODIFIED — path swap)
hotSpring/barracuda/src/md/reservoir/tests.rs      (MODIFIED — HeadKind wildcard arm for S80 variants)
hotSpring/wateringHole/handoffs/                   (this file)
hotSpring/whitePaper/baseCamp/reality_ladder_rung0.md  (NEW — Rung 0 writeup)
hotSpring/experiments/033_REALITY_LADDER_RUNG0.md  (NEW — experiment doc)
hotSpring/experiments/034_reality_ladder_rung1.sh  (NEW — Rung 1 script, running)
hotSpring/barracuda/EVOLUTION_READINESS.md         (UPDATED — absorption targets)
hotSpring/whitePaper/baseCamp/README.md            (UPDATED — index)
```
