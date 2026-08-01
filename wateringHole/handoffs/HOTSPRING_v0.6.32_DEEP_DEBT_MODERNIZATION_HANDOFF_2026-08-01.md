# HOTSPRING v0.6.32 — Deep Debt Sprint + Modernization Handoff

**Date**: 2026-08-01
**Wave**: 155n (post-threshold)
**Gate**: strandGate
**Scope**: Codebase-wide deep debt resolution, idiomatic Rust modernization, file refactoring, dependency evolution
**Status**: COMPLETE

## Summary

Post-threshold deep-debt sprint resolving accumulated technical debt across the
hotSpring codebase. All changes maintain the 627-test baseline with 0 clippy
warnings on lib code. No files exceed 800 lines after refactoring.

## Changes

### Smart File Refactoring (>800L → domain-focused modules)

| Before | After | Strategy |
|--------|-------|----------|
| `serve.rs` (944L) | `serve/mod.rs` (310L) + `dispatch.rs` (434L) + `transport.rs` (181L) + `params.rs` (56L) | Domain split: types/server, routing, I/O, parsing |
| `cazyme-fel/lib.rs` (1303L) | 12 domain-focused modules, all <800L | Structural decomposition by concern |

### Error Type Evolution (manual → thiserror)

8 error types migrated from manual `Display`/`From` to `#[derive(thiserror::Error)]`:

- `error.rs` — `HotSpringError`
- `squirrel_client.rs` — `SquirrelError`
- `ttm.rs` — `TtmError`
- `base64_encode.rs` — `Base64Error`
- `compchem/topology/gromacs.rs` — `GromacsParseError`
- `bench/compute_backend.rs` — `BenchError`
- `low_level/bar0.rs` — `Bar0Error`, `PciInfoError`

### Lint Violations Fixed

6 `.expect()` calls in GPU CG modules (`resident_cg.rs`, `resident_cg_brain.rs`)
replaced with proper `Result` propagation. Functions now return
`Result<GpuDynHmcResult, HotSpringError>`.

`gpu_dot_re` in `fermion_bridge.rs` evolved from `f64::NAN` sentinel to
`Result<f64, HotSpringError>`.

### Dependency Evolution

| Change | Rationale |
|--------|-----------|
| Added `thiserror = "2"` | Replaces 200+ lines of manual error boilerplate |
| Removed `primal-proof` feature | Dead feature flag, never used |
| `naga` made optional | Shader introspection only needed for benchmarks |

### Production Stub Completion

| Stub | Resolution |
|------|------------|
| SPMV CPU fallback | Dense matrix-vector product implemented |
| BCS density pass | GPU dispatch for v² and density reconstruction |
| `gpu_dot_re` error path | Proper error propagation replaces NAN sentinel |

### Deprecated Code Cleanup

- `rhmc_shifted_cg.rs` — entire deprecated RHMC module removed
- `mapped_bytes_to_f32` — dead code in `gpu/buffers.rs` removed

### Hardcoded Value Evolution

- `/run` and `/etc/hostname` fallback paths centralized to `niche/mod.rs`
  constants: `FALLBACK_RUN_DIR`, `FALLBACK_HOSTNAME_PATH`
- Environment variable overrides added for both

### cazyme-fel Edition Upgrade

- Edition 2021 → 2024
- Added `rust-version = "1.87"`

## Verification

| Metric | Value |
|--------|-------|
| `cargo check` | 0 errors |
| `cargo clippy --lib` | 0 warnings (2 upstream in `primalspring` dep) |
| `cargo test --lib` | 627 tests, 626 pass, 1 pre-existing env failure |
| Files >800L | 0 |

The 1 test failure (`ipc::provenance::loamspine::tests::append_returns_none_when_not_running`)
is pre-existing — it expects loamSpine to be unavailable, but loamSpine runs as
part of the live NUCLEUS deployment on strandGate.

## Gaps Updated

`docs/PRIMAL_GAPS.md` updated with 9 new resolved items and GAP-HS-027
status change to PARTIALLY RESOLVED (IPC fused pipeline complete, local GPU
deferred to future sprint).

## Upstream Primal Team Notes

- **barraCuda team**: `thiserror` v2 is now a dependency. The `primal-proof`
  feature was dead and has been removed — if upstream ever referenced it,
  that reference should be cleaned.
- **primalSpring team**: 2 clippy warnings emit from `primalspring` crate
  (`aarch64_depot_path` and `chrono_lite_cutoff` never used). Non-blocking
  but should be addressed upstream.
- **cazyme-fel**: Now on edition 2024 / rust-version 1.87. Downstream
  consumers should ensure MSRV compatibility.
