# hotSpring Local Debt Resolution + Composition Evolution — May 14, 2026

**Audience:** Upstream primal teams (toadStool, coralReef, barraCuda), primalSpring
**Sprint:** Local Debt Resolution + Composition Evolution (7 items)
**Test results:** 595/595 lib tests pass (default features). Zero clippy warnings. `cargo build --release` clean.

---

## Summary

hotSpring resolved seven fragility points discovered as the spring increasingly relies on IPC-composed NUCLEUS primals (toadStool, coralReef, barraCuda) rather than direct Rust execution.

---

## What Changed

### 1. Compile-Then-Dispatch Pipeline

**File:** `barracuda/src/compute_dispatch.rs`

hotSpring now correctly compiles WGSL shaders via coralReef before dispatching the resulting binary to toadStool. Previous validators sent shader *names* in `compute.dispatch.submit`, which toadStool rejected (no `job_id` in response).

New functions:
- `compile_and_submit(nucleus, wgsl_source, input_data, bdf)` — chains `shader.compile.wgsl` → `compute.dispatch.submit`
- `submit_binary(nucleus, binary_b64, input_data, bdf)` — dispatches pre-compiled binary

Legacy `submit_workload()` marked `#[deprecated]`.

**Upstream impact:** None — this is hotSpring-local wiring. The coralReef `shader.compile.wgsl` and toadStool `compute.dispatch.submit` RPCs work correctly; hotSpring was calling them wrong.

### 2. Circuit-Breaker Discovery

**File:** `barracuda/src/primal_bridge.rs`

`PrimalEndpoint` now tracks `fail_count` and `dead_since`. After 3 consecutive failures, a primal is marked dead (skipped for 30s before re-probing). `call_tracked()` wraps the existing `call()` with circuit-breaker logic.

**Upstream impact:** None — client-side resilience. Primals that crash or go offline no longer wedge hotSpring's validation pipeline.

### 3. Dispatch Surface Unification

**Files:** `compute_dispatch.rs`, `fleet_toadstool.rs`, `glowplug_client.rs`

`compute_dispatch.rs` is now the canonical module for all compute dispatch. `fleet_toadstool.rs` `submit()`/`dispatch()` deprecated. `glowplug_client.rs` documentation clarifies it handles device management RPCs only (warm-catch, device listing, health).

**Upstream impact:** None — internal refactoring.

### 4. FusedPipeline Typed Error Handling

**File:** `compute_dispatch.rs`

`FusedPipeline::submit()` returns `FusedSubmitReport` with per-operation `FusedOpSubmitOutcome::Submitted(job_id)` / `Failed(message)`. Previously, failures were encoded as fake `"error:{e}"` job IDs, which caused silent downstream failures when polling for results.

**Upstream impact:** None — hotSpring-local data structure.

### 5. `parse_jsonrpc_response()` Helper

**File:** `primal_bridge.rs`

Centralized JSON-RPC envelope parsing: extracts `result` or returns typed `HotSpringError::Ipc` with code/message. Eliminates scattered `.get("result")` patterns that silently swallowed errors.

**Upstream impact:** None — client-side helper.

### 6. TOML-Loaded Primal Aliases

**Files:** `primal_bridge.rs`, `config/capability_registry.toml`

`PRIMAL_ALIASES` are now loaded at runtime from `[primal_aliases]` in `capability_registry.toml`. This means hotSpring operators can add or update socket aliases without recompiling. Compiled defaults serve as fallback.

```toml
[primal_aliases]
toadstool = ["toadstool-server", "toadstool-glowplug", "compute"]
coralreef = ["coralreef-core-default", "coralreef-compiler", "shader"]
barracuda = ["barracuda-math", "barracuda-server", "math"]
```

**Upstream impact:** If primals change their socket naming, hotSpring can adapt via config without waiting for a code release.

### 7. Tiered Validation Infrastructure

**File:** `barracuda/src/bin/validate_all.rs`

65 validation suites categorized into 3 tiers:
- **Smoke** (35): Pure Rust, no IPC, no GPU — fast CI
- **Nucleus** (7): Requires live NUCLEUS primals — IPC validation
- **Silicon** (23): Requires GPU hardware — sovereign compute

`validate_all --tier smoke` runs in ~60s on any machine. Pre-built binaries from `target/release/` used when available, falling back to `cargo run`.

**Upstream impact:** Pattern available for sibling springs (ludoSpring, healthSpring) to adopt tiered validation.

---

## Upstream Gaps Discovered

| ID | Primal | Gap | Severity | Notes |
|----|--------|-----|----------|-------|
| GAP-HS-098 | coralReef | `sum_reduce_subgroup_f64.wgsl` causes assertion panic | Medium | Crash kills coralReef daemon. Subsequent compilations fail. hotSpring reordered barrier shaders to contain impact. Fix needed in coralReef source. |
| GAP-HS-099 | toadStool | `compute.dispatch.submit` returns no structured error on invalid binary | Low | toadStool returns empty JSON-RPC result (no `job_id`, no `error`) for malformed submissions. A structured error response would improve diagnostics. |
| GAP-HS-100 | plasmidBin | ecoBin pipeline lag | Medium | plasmidBin ecoBins for coralReef and toadStool were pre-subgroup-ops / pre-S259 harvests. New features (subgroup shader compile, VFIO IPC) were not available via ecoBin until fresh harvest. Automated CI harvest would prevent this. |

---

## Validation Evidence

```
cargo test --lib: 595 passed, 0 failed
cargo build --release: clean (zero warnings)
cargo clippy --all-targets: clean
```

Compile-then-dispatch pipeline tested end-to-end when primals are live:
1. coralReef compiles Yukawa force WGSL → binary_b64
2. toadStool dispatches binary via `compute.dispatch.submit`
3. hotSpring retrieves result via `compute.dispatch.result`

Circuit breaker validated: 3 failures → mark dead → 30s cooldown → re-probe.

---

## Files Modified

| File | Change |
|------|--------|
| `barracuda/src/compute_dispatch.rs` | `compile_and_submit()`, `submit_binary()`, `FusedSubmitReport`, `FusedOpSubmitOutcome`, `parse_jsonrpc_response` integration |
| `barracuda/src/primal_bridge.rs` | Circuit breaker (`fail_count`, `dead_since`, `call_tracked`, `record_failure/success`, `maybe_reprobe`), `parse_jsonrpc_response()`, TOML alias loading |
| `barracuda/src/fleet_toadstool.rs` | `submit()`/`dispatch()` deprecated, `jsonrpc_result` updated |
| `barracuda/src/glowplug_client.rs` | Module docs updated (device management only) |
| `barracuda/src/bin/validate_compute_trio_pipeline.rs` | Uses `compile_and_submit()` for yukawa/plaquette |
| `barracuda/src/bin/validate_all.rs` | `Tier` enum, `--tier` argument, pre-built binary support, 65 suites |
| `barracuda/config/capability_registry.toml` | `[primal_aliases]` section added |
