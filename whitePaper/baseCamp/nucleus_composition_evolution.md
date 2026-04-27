# SPDX-License-Identifier: AGPL-3.0-or-later

# NUCLEUS Composition Evolution — Primal Composition Tier

**Spring:** hotSpring (BarraCuda crate)  
**Updated:** April 27, 2026  
**Status:** guideStone Level 5 CERTIFIED (primalSpring v0.9.17, guideStone v1.2.0). Phase 46 composition template absorbed. Deep debt evolution complete — capability-based discovery from `niche::DEPENDENCIES`, deprecated named accessors, data-driven aliases. `hotspring_guidestone` binary validates 5 bare properties + NUCLEUS IPC parity. **Bare mode: 30/30 PASS** (3 SKIP = expected NUCLEUS liveness). Property 3 BLAKE3 CHECKSUMS manifest covers 15 validation-critical source files — verified via `primalspring::checksums::verify_manifest()`. `validate-primal-proof.sh` wraps the full primal proof workflow (builds from barracuda/, runs from root, auto-sets BEARDOG_FAMILY_SEED/SONGBIRD_SECURITY_PROVIDER/NESTGATE_JWT_SECRET). BLAKE3 checksums, protocol tolerance, family-aware discovery, genomeBin v5.1, deployment validation absorbed. 993/993 lib tests pass.

---

## Three-Tier Validation Arc

hotSpring’s validation story stacks three tiers that must agree before a science claim is trusted in production:

| Tier | Role | What “green” means |
|------|------|---------------------|
| **1 — Python** | Reference controls, published numerics, legacy tooling | Baseline observables and tolerances are defined |
| **2 — Rust** | Sovereign, reproducible physics in `hotspring_barracuda` | Same checks as tier 1 on CPU/GPU paths (`validate_*`, `cargo test --lib`) |
| **3 — NUCLEUS IPC composition** | Real primals behind JSON-RPC over UDS | Local Rust baselines match results obtained when the same capability is invoked **through biomeOS discovery and primal routing** |

Tier 3 is **Primal Composition**: it is not enough that `hotspring_primal` compiles; the niche must behave correctly when science is exercised **as a routed capability** (call-by-domain), the same way a deployed graph will call it.

---

## What Composition Validation Means for hotSpring

Science in deployment is not “a single binary.” It is:

- **Discovery** (Songbird) → **trust** (BearDog) → **compile** (coralReef) → **dispatch** (ToadStool) → **math/physics** (BarraCuda + hotSpring) → **storage + DAG + ledger + attribution** (NestGate + provenance trio), with optional **inference** (Squirrel).

Composition validation proves:

1. Required primals are **alive** and expose the expected capability surface.  
2. **Capability-based routing** (`call_by_capability(domain, method, params)`) resolves the same endpoints the deploy graph implies.  
3. **Science parity probes** compare trusted in-process Rust values to values returned when the equivalent JSON-RPC is routed through IPC (within documented tolerances).

Code paths:

- Composition model + science probes: `barracuda/src/composition.rs` (`validate_science_probes`, `AtomicType`, `composition_health`).  
- IPC context + routing: `barracuda/src/primal_bridge.rs` (`NucleusContext`, `call_by_capability`).  
- Full-stack parity (SEMF, plaquette, HMC): `barracuda/src/bin/validate_nucleus_composition.rs` (Phase 5).  
- Node-focused lattice/EOS parity: `barracuda/src/bin/validate_nucleus_node.rs`.

---

## Evolution: Standalone Rust → NUCLEUS → Parity → Server Dispatch

1. **Standalone Rust + WGSL** — Physics and tolerances live entirely in the crate; GPU paths validated per-domain (`validate_gpu_*`, lattice, MD, nuclear EOS, etc.).

2. **`validate_nucleus_*` binaries** — Slice the NUCLEUS atom into testable fragments:
   - `validate_nucleus_tower` — BearDog + Songbird (`barracuda/src/bin/validate_nucleus_tower.rs`)
   - `validate_nucleus_node` — compute/shader/math stack (`validate_nucleus_node.rs`)
   - `validate_nucleus_nest` — NestGate + provenance trio (`validate_nucleus_nest.rs`)
   - `validate_nucleus_composition` — full graph + parity probes (`validate_nucleus_composition.rs`)

3. **Science parity probes** — Rust baseline vs IPC for SEMF binding energy, Wilson plaquette on a small thermalized lattice, and `physics.hmc_trajectory` JSON shape + plaquette/acceptance fields (see Phase 5 in `validate_nucleus_composition.rs`).

4. **Full server dispatch** — `barracuda/src/bin/hotspring_primal.rs` serves all methods in `niche::LOCAL_CAPABILITIES` (13 physics/compute methods + composition/health/MCP). **GAP-HS-026** (April 17, 2026): every local method is wired; pending placeholders removed. See `docs/PRIMAL_GAPS.md`.

5. **`hotspring_guidestone` binary** — Unified guideStone deployable (primalSpring v0.9.17, guideStone v1.2.0):
   - **Bare mode**: Validates Properties 1-5 (Deterministic, Reference-Traceable, Self-Verifying [BLAKE3 CHECKSUMS — 15 source files], Environment-Agnostic, Tolerance-Documented) without any primals deployed. **30/30 checks pass**, 3 SKIPs (expected NUCLEUS liveness only). Property 3 verifies per-file BLAKE3 hashes + `deny.toml` present.
   - **NUCLEUS additive mode**: IPC parity via `primalspring::composition` API — scalar parity, vector parity, SEMF end-to-end, crypto provenance witness, compute dispatch against live primals.
   - **Protocol tolerance**: `is_protocol_error()` classifies HTTP-on-UDS (Songbird, petalTongue) as SKIP, matching v0.9.16+ liveness semantics.
   - **Family-aware discovery**: Inherited via `CompositionContext` — `{capability}-{FAMILY_ID}.sock` resolved before fallback.
   - **Env var auto-setup**: `validate-primal-proof.sh` auto-sets `BEARDOG_FAMILY_SEED`, `SONGBIRD_SECURITY_PROVIDER`, `NESTGATE_JWT_SECRET` when `FAMILY_ID` is provided.

6. **`validate-primal-proof.sh`** — End-to-end script. Bare mode (domain only) and `--full` mode (pre-flight `primalspring_guidestone` + domain `hotspring_guidestone`). Detects bare vs live NUCLEUS automatically.

7. **Phase 46 Composition Template (April 27, 2026)** — `tools/hotspring_composition.sh` implements hotSpring's event-driven QCD computation lane:
   - **Async tick model**: Convergence-based progression (not fixed-rate) — simulations run until physics converges, with `domain_on_tick()` as the entry point.
   - **DAG memoization**: Parameter sweeps as directed acyclic graphs — `VERTEX_STACK` tracks lattice configurations, `BRANCH_STACK` tracks coupling constants, visited vertices are memoized to avoid recomputation.
   - **Scientific provenance**: `sweetGrass` braids carry peer-review audit metadata (coupling constants, lattice dimensions, Monte Carlo sweeps, algorithm, convergence tolerance).
   - **Compute dispatch**: `toadStool`/`barraCuda` tensor workloads through the composition.
   - **Ledger sealing**: Each simulation run as a sealed `loamSpine` spine for reproducibility.
   - `tools/nucleus_composition_lib.sh` (41-function library from primalSpring) provides discovery, transport, DAG, ledger, braids, petalTongue, and sensor stream wiring.

8. **Deep Debt Evolution (April 27, 2026)** — Capability-based discovery refactored to single source of truth:
   - `composition.rs` derives all primal requirements from `niche::DEPENDENCIES` — eliminated hardcoded name→domain maps.
   - `primal_bridge.rs` named accessors (`toadstool()`, `beardog()`, `coralreef()`) deprecated with `#[deprecated]`; all 8 production call sites migrated to `by_domain()`.
   - Data-driven `PRIMAL_ALIASES` table replaces hardcoded alias fallback.
   - Smart refactoring: `rhmc.rs` (989L) → `rhmc/mod.rs` + `rhmc/remez.rs`; `nuclear_eos_helpers.rs` (978L) → `mod.rs` + `objectives.rs`.
   - Pre-existing compile errors fixed (`DiscoveredDevice` API in `nuclear_eos_l2_*` binaries).
   - 993/993 lib tests pass. Zero compilation errors.

**Forward evolution (tracked as gaps):** **TensorSession** fused pipelines (GAP-HS-027), **LIME/ILDG zero-copy** (GAP-HS-028), and expanded **cross-primal science parity** (more observables routed the same way production will call them).

---

## Deploy Graph vs `FullNucleus`

**Deploy graph (spring-local):** `graphs/hotspring_qcd_deploy.toml`

- **10 primals** appear as peer `[[graph.nodes]]` entries (BearDog, Songbird, coralReef, ToadStool, BarraCuda, NestGate, rhizoCrypt, loamSpine, sweetGrass, Squirrel).  
- **`hotspring_primal`** is the **spawning** application node (order 10) that depends on the core stack.  
- **Bonding:** `bond_type = "Metallic"`, `trust_model = "InternalNucleus"`, with **tiered encryption** (`encryption_tiers.tower/node/nest`) matching NUCLEUS boundary semantics.

**`AtomicType::FullNucleus` (composition code):** nine **required** primals for the atom — BearDog, Songbird, ToadStool, BarraCuda, coralReef, NestGate, rhizoCrypt, loamSpine, sweetGrass — defined in `composition.rs` (`required_primals()`). Squirrel remains **optional** meta-tier; hotSpring itself is the client under test, not a row in `required_primals()`.

---

## Capability-Based Routing

NUCLEUS routing is **by capability domain**, not by hard-coded process names:

- Discovery fills `NucleusContext` from live sockets / capability lists.  
- Callers use **`call_by_capability(domain, method, params)`** (see `primal_bridge.rs` and `composition::get_by_capability`).  
- `niche.rs` splits **`LOCAL_CAPABILITIES`** (served by `hotspring_primal`) from **`ROUTED_CAPABILITIES`** (canonical provider per method, e.g. ToadStool for `compute.dispatch.submit`).

This matches biomeOS / Neural API expectations: **call by capability**, let the graph decide which binary instance satisfies the domain.

---

## Science Parity Probes (Rust ↔ IPC)

| Probe | Local baseline | IPC exercise | Typical tolerance |
|-------|----------------|--------------|-------------------|
| **SEMF** | `physics::semf_binding_energy` (Pb-208) | `call_by_capability("compute", "physics.nuclear_eos", …)` | relative error ≤ `1e-10` (see harness in `validate_nucleus_composition.rs`) |
| **Plaquette** | `lattice::wilson::Lattice::hot_start` average plaquette | `physics.lattice_qcd` | absolute ≤ `1e-12` |
| **HMC trajectory** | same code path as server handler | `physics.hmc_trajectory` | response shape + finite plaquette / acceptance bit |

**Library-level probes** (`validate_science_probes` in `composition.rs`) additionally assert compute, math, and provenance trio reachability — with **honest skip** when `HOTSPRING_NO_NUCLEUS=1` or discovery is empty.

---

## Patterns Absorbed from primalSpring

Implemented in `barracuda/src/validation/composition.rs` and re-exported from `barracuda/src/validation/mod.rs`:

| Pattern | Purpose |
|---------|---------|
| **`ValidationSink` enum** | Stdout / Null / Ndjson output — **no `dyn` sink** (stadial hygiene) |
| **`CompositionResult`** | Named checks, pass/fail/skip accounting, NDJSON-friendly telemetry |
| **`OrExit`** | `.or_exit("msg")` on `Result` / `Option` for zero-panic binaries |
| **`check_skip` / `check_or_skip`** | CI-honest “not available” vs real failure |
| **`exit_code_skip_aware()`** | `0` = pass with ≥1 executed check, `1` = failure, `2` = all skipped |

Science binaries that must distinguish “optional primal missing” from “broken IPC” should use the skip-aware exit code (see `validate_squirrel_roundtrip.rs`).

---

## Validation Commands (composition-focused)

From the `barracuda/` crate root:

```bash
cargo test --lib
cargo run --release --bin validate_nucleus_tower
cargo run --release --bin validate_nucleus_node
cargo run --release --bin validate_nucleus_nest
cargo run --release --bin validate_nucleus_composition
cargo run --release --bin validate_compute_dispatch
cargo run --release --bin validate_squirrel_roundtrip   # exit 2 if Squirrel absent
cargo run --release --bin validate_all                  # includes all of the above suites
```

**guideStone primal proof (recommended):**

```bash
# Bare mode (no NUCLEUS required)
./scripts/validate-primal-proof.sh

# Full mode (pre-flight + domain against live NUCLEUS)
FAMILY_ID=hotspring-validation ./scripts/validate-primal-proof.sh --full
```

**Standalone / lab without NUCLEUS:**

```bash
export HOTSPRING_NO_NUCLEUS=1
cargo run --release --bin validate_nucleus_composition
```

Expect skip-pass behavior where discovery is empty; binaries are written to degrade honestly.

---

## Scaling to Other Springs

The same template applies to any spring niche:

1. **Declare** `LOCAL_CAPABILITIES` vs `ROUTED_CAPABILITIES` and register with biomeOS (`register_with_target()` → `lifecycle.register` + `capability.register`).  
2. **Pin** a deploy graph fragment set (`tower_atomic`, `node_atomic`, `nest_atomic`, optional `nucleus`, `provenance_trio`, `meta_tier`) in TOML under `graphs/`.  
3. **Validate** atom-by-atom binaries, then a **single composition + parity** binary comparing in-crate truth to IPC.  
4. **Serve** real JSON-RPC methods on the spring binary — no “registered but pending” surfaces.

Reference pattern doc for new niches: `primalSpring/graphs/downstream/NICHE_STARTER_PATTERNS.md`.

---

## References

- Deploy graph: `graphs/hotspring_qcd_deploy.toml`  
- Spring manifest entry (6th spring): `primalSpring/graphs/spring_deploy/spring_deploy_manifest.toml` (`spring_name = "hotspring"`)  
- guideStone standard: `primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md` (v1.1.0)
- plasmidBin depot: `primalSpring/wateringHole/PLASMINBIN_DEPOT_PATTERN.md`
- Gap ledger: `docs/PRIMAL_GAPS.md` (GAP-HS-026/032/033/035 resolved, GAP-HS-027/028 forward work)  
- Ecosystem handoffs: `infra/wateringHole/handoffs/HOTSPRING_V0632_*`
