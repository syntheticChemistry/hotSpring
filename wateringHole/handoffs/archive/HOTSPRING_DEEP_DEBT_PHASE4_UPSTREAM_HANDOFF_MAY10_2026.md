# hotSpring Deep Debt Phase 4 — Upstream Handoff

**Date:** May 10, 2026
**Spring:** hotSpring v0.6.32
**primalSpring:** v0.9.25
**guideStone:** Level 6 CERTIFIED (NUCLEUS Deployment Validation)
**Tests:** 1,019 lib (0 failed, 6 ignored) · 155 binaries · 128 WGSL · 7 deploy graphs

---

## Summary

hotSpring has completed Deep Debt Phase 4 — the post-interstadial evolution
covering Tier 4 IPC-first rewiring, typed error propagation, hostname
consolidation, smart file refactoring, and full documentation alignment.
All explicit targets from the primalSpring post-interstadial guidance
(May 10, 2026) have been implemented and verified.

---

## For primalSpring Team

### Completed from interstadial guidance

1. **Tier 4 IPC-first rewiring** — `barracuda` is `optional = true`. 25+
   modules gated behind `#[cfg(feature = "barracuda-local")]`. The
   `primal-proof` feature flag enables IPC-only builds without linking
   the local barraCuda library. All local compute has fallbacks or is gated.

2. **CI cross-sync** — Zero drift against canonical capability registry
   (413 methods). `tools/check_method_strings.sh` validates alignment.

3. **skunkBat audit logging** — Added to all 7 deploy graphs as optional
   `defense` node with `depends_on = ["beardog", "nestgate"]`.

4. **`composition.status` absorption** — `ipc/biome_status.rs` queries
   biomeOS v3.51 `{ active_users, primal_health, resource_pressure }`.
   Integrated into L6 certification deployment checks.

5. **`method.register` absorption** — `ipc/method_register.rs` dynamically
   registers 24 hotSpring physics/compute methods with biomeOS.

6. **Sovereignty** — BearDog TLS, Songbird NAT traversal, PetalTongue web
   serving referenced in deploy graphs. No remaining external service deps.

### Patterns evolved that may benefit ecosystem

- **`HotSpringError::Ipc(String)`** — Typed error variant for IPC failures.
  `send_jsonrpc` returns `Result<_, HotSpringError>` instead of
  `Result<_, String>`. Consider standardizing this pattern in primalSpring
  `composition/` module for all springs.

- **`impl From<HotSpringError> for String`** — Allows clean `?` at binary
  boundaries without wrapping every call. Useful when binaries need
  `Result<_, String>` but library code uses typed errors.

- **`niche::hostname()`** — Centralized hostname resolution with env var
  chain fallback. Consider adding to primalSpring `niche` module template
  so all springs get portable hostname resolution.

- **`#![forbid(unsafe_code)]` + `#[path]` for hardware modules** — Library
  maintains `forbid(unsafe_code)` while hardware-access modules are included
  in binaries via `#[path]` rather than `pub mod`. Clean separation.

- **Local fallbacks for IPC-only builds** — `Complex64`, `bisect`, `hermite`,
  `factorial`, `lu_solve`, `MD_WORKGROUP_SIZE` all have local implementations
  behind `#[cfg(not(feature = "barracuda-local"))]`. Enables genuine
  compilation without barraCuda.

### Open items for primalSpring

- **Capability registry**: hotSpring registers 24 methods. If any method
  naming convention has evolved since registry v413, update
  `config/capability_registry.toml` to match.

- **Deploy graph schema**: hotSpring has 7 deploy graphs, each with
  `skunkbat` as optional. If the graph schema evolves (e.g., required
  skunkBat), hotSpring graphs need updating.

---

## For barraCuda Team

### What hotSpring consumes from barraCuda

- `barracuda::optimize::bisect` — root-finding (local fallback exists)
- `barracuda::special::{hermite, factorial}` — special functions (local fallbacks exist)
- `barracuda::ops::linalg::lu_solve` — linear algebra (local fallback exists)
- `barracuda::ops::lattice::cpu_complex::Complex64` — complex arithmetic (local fallback exists)
- `barracuda::device::capabilities::WORKGROUP_SIZE_COMPACT` — GPU workgroup size (fallback: 64)
- `barracuda::error::BarracudaError` — error type (gated behind `barracuda-local`)

### Recommendation

All barraCuda dependencies now have local fallbacks. hotSpring can compile
and pass all 1,019 tests without barraCuda linked. The `primal-proof`
feature flag proves this. When barraCuda evolves its API, hotSpring will
absorb via the `barracuda-local` feature gate — no breaking changes.

---

## For coralReef Team

### Patterns validated by hotSpring

- **Ember typed errors**: `fleet_ember.rs` evolved from `Result<_, String>`
  to `Result<_, HotSpringError>` (24 pub fns). Consider evolving
  `coral-ember`'s client API to return typed errors rather than strings.

- **Glowplug integration**: `glowplug_client.rs` wraps `HotSpringError`
  into `GlowplugError::Transport(String)`. If glowplug evolves a typed
  error, hotSpring will absorb directly.

- **BDF validation**: hotSpring's sovereign pipeline (K80 experiments 154–184)
  exercises coral-ember's BDF validation, keepalive clamping, and switch
  health extensively. Any changes to these APIs need corresponding updates
  in hotSpring's 15+ sovereign experiment binaries.

---

## For toadStool Team

### NPU steering

- `lattice/pseudofermion/npu_steering.rs` and `adaptive.rs` are gated
  behind `barracuda-local`. When toadStool evolves its NPU dispatch
  contracts, hotSpring will absorb via the `npu-hw` feature gate
  (requires `akida-driver` + `akida-models`).

---

## For projectNUCLEUS Team

### Workload alignment issue

`workloads/hotspring/hotspring-md-validation.toml` references:
- Binary: `validate_sarkas_md` — **does not exist**. Correct name: `sarkas_gpu`
- Path: `/home/irongate/...` — **hardcoded**. Should use `$ECOPRIMALS_ROOT`
  or relative paths.

### Deploy graph alignment

hotSpring now has 7 deploy graphs covering all science domains:
- `hotspring_qcd_deploy.toml` — Lattice QCD
- `hotspring_md_deploy.toml` — Molecular dynamics
- `hotspring_nuclear_eos_deploy.toml` — Nuclear EOS
- `hotspring_plasma_deploy.toml` — Dense plasma
- `hotspring_plasma_md_deploy.toml` — Plasma + MD combined
- `hotspring_spectral_deploy.toml` — Spectral theory
- `hotspring_sovereign_gpu_deploy.toml` — Sovereign GPU compute

All use `tower_atomic + node_atomic + nest_atomic` composition fragments
with skunkBat audit node.

---

## For foundation Team

### Thread alignment

hotSpring is correctly mapped in foundation `THREAD_INDEX.toml` to:
- Thread 1: Whole-Cell Modeling (via nuclear EOS)
- Thread 2: Plasma Physics / Lattice QCD (primary domain)
- Thread 7: Anderson Spectral Theory

`BASECAMP_PAPER_MAP.toml` references hotSpring across 8 paper entries,
all status "validated" or better.

### Data targets

- `data/sources/thread02_plasma.toml` references hotSpring plasma controls
- `data/targets/thread01_wcm_targets.toml` references nuclear EOS validation

No foundation documentation drift detected.

---

## For river delta spring teams

### Composition patterns hotSpring has validated

1. **IPC-first compilation** — The `primal-proof` feature proves a spring
   can compile without its upstream library linked. Other springs with
   barraCuda dependencies can follow the same pattern.

2. **Deploy graph per science domain** — Rather than one monolithic graph,
   hotSpring uses domain-specific graphs. Each graph lists only the primals
   and capabilities needed for that domain.

3. **Typed IPC errors** — `send_jsonrpc` → `HotSpringError::Ipc` instead
   of `String`. Cleaner error propagation for NUCLEUS composition debugging.

4. **`niche::hostname()` centralization** — Single function for hostname
   resolution instead of scattered `/etc/hostname` reads.

5. **L6 certification** — Deploy graph coverage, biomeOS status, method
   registration, and skunkBat wiring. Other springs at L5 can follow
   hotSpring's `certification/deployment.rs` as a reference.

---

## Verification

```
cargo fmt --check      → zero drift
cargo clippy --lib     → zero new warnings (19 pre-existing dead_code)
cargo test --lib       → 1,019 passed, 0 failed, 6 ignored
cargo check            → all 155 binaries compile
primal-proof build     → compiles clean without barraCuda
```
