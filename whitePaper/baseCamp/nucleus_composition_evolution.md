# SPDX-License-Identifier: AGPL-3.0-or-later

# NUCLEUS Composition Evolution — Primal Composition Tier

**Spring:** hotSpring (BarraCuda crate)  
**Updated:** April 17, 2026  
**Status:** Composition tier operational — Python/Rust baselines extended through IPC-composed primals; server dispatch wired (GAP-HS-026 resolved)

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

Forward evolution (tracked as gaps): **TensorSession** fused pipelines (GAP-HS-027), **LIME/ILDG zero-copy** (GAP-HS-028), and expanded **cross-primal science parity** (more observables routed the same way production will call them).

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
- Gap ledger: `docs/PRIMAL_GAPS.md` (GAP-HS-026 resolved, GAP-HS-027/028 forward work)  
- Ecosystem handoff: `infra/wateringHole/handoffs/HOTSPRING_V0632_PRIMAL_ABSORPTION_HANDOFF_APR17_2026.md`
