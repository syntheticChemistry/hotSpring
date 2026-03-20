# hotSpring Comprehensive Audit — March 17, 2026

**Scope**: hotSpring v0.6.32 (barracuda crate) + metalForge forge, against wateringHole ecosystem standards, sibling springs, and audit criteria.

**Auditor**: Cursor AI (ecoPrimals audit protocol)  
**Date**: March 17, 2026

---

## Executive Summary

| Category | Status | Notes |
|----------|--------|-------|
| **Completion Status** | ✅ Strong | Zero TODOs/FIXMEs in production; provenance + tolerances centralized; experiments traceable |
| **Code Quality** | ⚠️ Partial | `#![forbid(unsafe_code)]` ✓; ~100+ `#[allow]` in production (physics-justified); clippy/fmt long-running |
| **Validation Fidelity** | ✅ Strong | Hardcoded expected values with provenance; exit 0/1; tolerances named and justified |
| **barraCuda Dependency** | ✅ Good | Pin `f82d60c6` (v0.3.5); delegates to upstream; 86 local WGSL shaders documented for absorption |
| **GPU Evolution Readiness** | ✅ Documented | `EVOLUTION_READINESS.md` maps Rust→WGSL→tier; Tier A/B/C; absorption targets listed |
| **Test Coverage** | ⚠️ Below Target | ~74.9% region / ~83.8% function (README); target 90%+ |
| **Ecosystem Standards** | ⚠️ Partial | AGPL-3.0-or-later ✓; ecoBin path; handoff naming convention followed |
| **Primal Coordination** | ⚠️ Partial | coralReef IPC (sovereign-dispatch); metalForge→barraCuda; no typed IPC clients for toadStool/Squirrel |

---

## 1. COMPLETION STATUS

### 1.1 TODOs, FIXMEs, Mocks, Debt

| Item | Status |
|------|--------|
| **TODO/FIXME/HACK in .rs** | ✅ **Zero** in production code (per handoff v0.6.32) |
| **Mocks** | ✅ **Zero** in production (no `mock`/`Mock` in .rs) |
| **Hardcoded paths** | ✅ Avoided via `discovery.rs` (`HOTSPRING_DATA_ROOT`, manifest parent, CWD) |
| **Hardcoded physics** | ✅ Intentional; provenance in `provenance.rs` (script, commit, date, command) |
| **Constants** | ✅ ~170 in `tolerances/` (core, md, physics, lattice, npu) — named, justified |

### 1.2 Provenance and Traceability

| Aspect | Status |
|--------|--------|
| **Python baseline provenance** | ✅ `provenance.rs` — script, commit, date, exact command, environment |
| **Data sources** | ✅ AME2020, Sarkas, Surrogate, Zenodo (10.5281/zenodo.10908462), DOIs documented |
| **Experiment traceability** | ✅ `experiments/` 001–069; `CONTROL_EXPERIMENT_STATUS.md`; numbered docs |
| **Validation targets** | ✅ Hardcoded expected values with `BaselineProvenance`; control JSON policy documented |

### 1.3 Gaps and Remaining Work

- **Handoff naming**: Follows `{SPRING}_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md` (e.g. `HOTSPRING_V0632_TRIO_REWIRE_HANDOFF_MAR13_2026.md`)
- **Python baseline rerun**: Not verified in this audit — recommend periodic `control/` rerun to confirm no drift
- **Exp 026 (4D Anderson-Wegner)**: Planned, not implemented
- **Exp 027 (Energy Thermal Tracking)**: Planned, not implemented

---

## 2. CODE QUALITY

### 2.1 Unsafe and Lint

| Item | Status |
|------|--------|
| **`#![forbid(unsafe_code)]`** | ✅ In `barracuda/src/lib.rs` |
| **`unsafe` blocks** | ✅ Zero in hotSpring application code |
| **`#[allow(...)]` in production** | ⚠️ ~100+ uses across 80+ files |

**`#[allow]` breakdown** (representative):
- `clippy::float_cmp` — exact known values in tests
- `clippy::expect_used`, `clippy::unwrap_used` — validation binaries (explicit pass/fail)
- `clippy::cast_possible_truncation` — physics bounds (e.g. n_levels ≤ 30)
- `clippy::similar_names`, `many_single_char_names` — physics notation (vx, vy, rho_p, etc.)
- `dead_code` — evolution/GPU paths not yet wired
- Workspace-level allows in `Cargo.toml` for physics: `cast_*`, `doc_markdown`, `suboptimal_flops`, `too_many_arguments`, etc.

**Recommendation**: Target zero `#[allow]` in production per ecosystem standard. Many are physics-justified; consider moving to workspace lints or documenting each in a central `LINT_JUSTIFICATIONS.md`.

### 2.2 File Size

| Threshold | Status |
|-----------|--------|
| **< 1000 LOC** | ✅ All files under 1000 LOC (largest: `nuclear_eos_helpers.rs` 978, `validate_chuna_overnight.rs` 969) |

### 2.3 Dependencies (ecoBin)

| Check | Status |
|------|--------|
| **Application C deps** | ✅ No openssl, ring, aws-lc-sys in hotSpring |
| **barracuda** | Git rev `f82d60c6`; local patch via `.cargo/config.toml` to `../../barraCuda/crates/barracuda` |
| **Transitive C** | wgpu → ash → libvulkan; tokio → libc (ecosystem, not hotSpring code) |
| **akida-driver / akida-models** | Optional `npu-hw` feature; path to toadStool neuromorphic crates |

### 2.4 Zero-Copy and I/O

- `discovery.rs` uses path resolution, not buffering
- barraCuda uses `bytes::Bytes` for zero-copy IPC (per absorption manifest)
- I/O parsers: not audited for stream vs buffer; recommend review of data-loading paths

---

## 3. VALIDATION FIDELITY

### 3.1 Pattern Compliance

| Requirement | Status |
|-------------|--------|
| **Hardcoded expected values** | ✅ With provenance |
| **Explicit pass/fail** | ✅ `ValidationHarness` |
| **Exit 0/1** | ✅ `validation.rs` |
| **Machine-readable summary** | ✅ stdout |
| **Tolerances named, centralized** | ✅ `tolerances/` modules |

### 3.2 Tolerance Justification

- `tolerances/mod.rs` documents categories: machine precision, numerical method, physical model, literature
- Each tolerance has physical/literature basis (e.g. NMP 2σ, energy drift 0.5%)
- Control JSON policy: JSON files authoritative; constants frozen at commit

### 3.3 Python Baseline Reproducibility

- Provenance records include exact command and environment
- **Recommendation**: Add CI or periodic script to rerun Python controls and assert no drift

---

## 4. BARRACUDA DEPENDENCY HEALTH

### 4.1 Version and Pin

| Item | Value |
|------|-------|
| **barraCuda pin** | `f82d60c6` (v0.3.5) |
| **Upstream barraCuda** | v0.3.5 (PRIMAL_REGISTRY) |
| **Local patch** | `.cargo/config.toml` patches to `../../barraCuda/crates/barracuda` |

### 4.2 Primitive Delegation

- **Spectral**: Fully leaning on `barracuda::spectral`
- **Lattice**: Dirac, CG, pseudofermion, HMC absorbed upstream
- **MD**: CellListGpu, SsfGpu, PppmGpu from barraCuda
- **HFB/BCS**: BatchedEighGpu, bcs_bisection_f64
- **ESN**: `esn_v2::MultiHeadEsn`, NautilusShell
- **ReduceScalarPipeline**, **GpuDriverProfile**, **WgslOptimizer** — all upstream

### 4.3 Local WGSL Shaders (Absorption Candidates)

- 86 `.wgsl` files in `barracuda/src/physics/shaders/`, `lattice/shaders/`, `md/shaders/`, etc.
- `EVOLUTION_READINESS.md` documents Write→Absorb→Lean path
- Active absorption targets: `lattice/rhmc.rs`, `gpu_hmc/hasenbusch.rs`, `production/npu_worker.rs`, etc.

---

## 5. GPU EVOLUTION READINESS

### 5.1 Tier Mapping

| Tier | Meaning | hotSpring Examples |
|------|---------|-------------------|
| **A** | Rewire — shader exists | semf, hfb, bcs_gpu, md/simulation, celllist |
| **B** | Adapt — shader exists, needs mod | hfb_deformed_gpu |
| **C** | New — no shader | hfb_deformed (CPU), nuclear_matter |
| **✅** | Absorbed | spectral, Dirac, CG, pseudofermion, ESN, Nautilus |

### 5.2 TensorSession

- **TensorSession** is a barraCuda primitive (neuralSpring absorption)
- hotSpring does not use TensorSession directly; uses domain-specific ops (lattice, MD, HFB)
- **Recommendation**: Evaluate TensorSession for fused multi-op pipelines where applicable (e.g. MD→ESN→observables)

### 5.3 Blocker Documentation

- `EVOLUTION_READINESS.md` lists blockers per module (e.g. BCS Brent on CPU for hfb_gpu_resident)
- Deformed HFB: H-build on CPU; deformed Hamiltonian shaders exist but not fully wired

---

## 6. TEST COVERAGE

### 6.1 Current State

| Metric | Value | Target |
|--------|-------|--------|
| **Line coverage** | ~74.9% (README) | 90%+ |
| **Function coverage** | ~83.8% | — |
| **Unit tests** | In-module `#[cfg(test)]` | — |
| **Integration tests** | 7 files (`integration_physics`, `integration_data`, etc.) | — |
| **Validation binaries** | 115+ | — |
| **Determinism** | Fixed seeds in tolerances (`DEFAULT_VELOCITY_SEED`, etc.) | — |

### 6.2 Gaps

- **Coverage below 90%**: Recommend `cargo llvm-cov` audit and targeted tests for uncovered paths
- **CI coverage gate**: No `--fail-under-lines` or equivalent in repo (per explore summary)
- **Stochastic algorithms**: Seeds fixed; tolerances justified in `tolerances/`

---

## 7. ECOSYSTEM STANDARDS (wateringHole)

### 7.1 License

| Standard | Requirement | Status |
|----------|-------------|--------|
| **scyBorg** | AGPL-3.0-or-later | ✅ `Cargo.toml`: `license = "AGPL-3.0-or-later"` |
| **LICENSE file** | AGPL-3.0 | ✅ Present |

### 7.2 Architecture

| Standard | Status |
|----------|--------|
| **ecoBin** | Pure Rust application code; transitive C via wgpu/tokio |
| **Files < 1000 LOC** | ✅ |
| **Single-responsibility** | ✅ Modular layout |
| **Data provenance** | ✅ Public repos, DOIs, Zenodo, SRA (where applicable) |
| **Sovereignty** | No vendor lock-in; wgpu/vulkan-portability |

### 7.3 Handoffs

| Convention | Example | Status |
|------------|---------|--------|
| `{SPRING}_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md` | `HOTSPRING_V0632_TRIO_REWIRE_HANDOFF_MAR13_2026.md` | ✅ Followed |
| Location | `wateringHole/handoffs/` (and `hotSpring/wateringHole/handoffs/`) | ✅ |

---

## 8. PRIMAL COORDINATION

### 8.1 Current Wiring

| Primal | Integration | Status |
|--------|--------------|--------|
| **barraCuda** | Direct dependency (git rev) | ✅ |
| **coralReef** | `CoralCompiler` IPC via `GLOBAL_CORAL.compile_wgsl_direct()`; `sovereign-dispatch` feature | ✅ |
| **toadStool** | metalForge `bridge.rs` — `Substrate` ↔ `WgpuDevice`; hardware discovery | ✅ |
| **metalForge** | `forge/` — probe, dispatch, bridge, inventory, pipeline, substrate | ✅ |

### 8.2 Gaps

| Item | Status |
|------|--------|
| **Typed IPC clients** | No dedicated toadStool/Squirrel/biomeOS JSON-RPC clients in hotSpring |
| **Capability registration** | Discovery via `control/` subdirs (`available_capabilities()`); no explicit primal registry entry from hotSpring |
| **MCP tool definitions** | Not present |
| **Provenance trio** | sweetGrass/rhizoCrypt/loamSpine — not wired (per handoff, awaiting coralReef/toadStool deliverables) |

### 8.3 Sibling Spring Patterns (airSpring V0.8.9)

- `PRIMAL_NAME` / `PRIMAL_DOMAIN` constants
- `OnceLock` GPU probe caching (toadStool S158)
- `cast` module for safe numeric casts
- `DispatchOutcome<T>` library type
- `discover_shader_compiler()` / `discover_inference_primal()` — three-tier resolution
- **Recommendation**: Adopt `PRIMAL_NAME`/`PRIMAL_DOMAIN` and `discover_shader_compiler()` pattern if hotSpring exposes capabilities to biomeOS

---

## 9. RECOMMENDATIONS (Prioritized)

### P0 — High Impact

1. **Coverage**: Raise line coverage to 90%+; add CI `--fail-under-lines 90` gate.
2. **`#[allow]` reduction**: Audit each allow; move justifiable ones to workspace or document in `LINT_JUSTIFICATIONS.md`; eliminate unwarranted allows.
3. **Python baseline CI**: Add periodic rerun of control scripts to detect baseline drift.

### P1 — Medium Impact

4. **TensorSession evaluation**: Identify MD/observable pipelines that could use fused TensorSession.
5. **Primal discovery**: Add `discover_shader_compiler()` / capability-based discovery if hotSpring registers with biomeOS.
6. **Provenance trio**: Wire into `provenance_pipeline` graph when coralReef/toadStool deliverables land (per handoff).

### P2 — Lower Priority

7. **Zero-copy I/O audit**: Review data-loading paths for stream vs buffer.
8. **Handoff sync**: Ensure `hotSpring/wateringHole/handoffs/` and `ecoPrimals/wateringHole/handoffs/` are aligned; avoid duplication.

---

## 10. FILES REFERENCE

| Document | Path |
|----------|------|
| Evolution readiness | `barracuda/EVOLUTION_READINESS.md` |
| Absorption manifest | `barracuda/ABSORPTION_MANIFEST.md` |
| Provenance | `barracuda/src/provenance.rs` |
| Tolerances | `barracuda/src/tolerances/` |
| Validation harness | `barracuda/src/validation.rs` |
| Discovery | `barracuda/src/discovery.rs` |
| Specs | `specs/` |
| Experiments | `experiments/` |

---

**Audit complete.** hotSpring is in strong compliance with ecosystem standards. Primary gaps: test coverage below 90%, `#[allow]` count in production, and primal coordination (typed IPC, capability registration). No unsafe code, no mocks in production, provenance and tolerances well-structured.
