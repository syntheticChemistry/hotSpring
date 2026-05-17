# SPDX-License-Identifier: AGPL-3.0-or-later

# hotSpring IPC Degradation Behavior

**Spring:** hotSpring v0.6.32
**Date:** May 17, 2026
**Absorbed from:** primalSpring `CompositionContext` degradation table (lithoSpore R1)
**Pattern:** `has_capability()` before `call()`. Never gate science behind provenance.

---

## Invariant

**No IPC call panics on unreachable primals.** All RPC calls return `Result` —
callers decide whether to skip, retry, or abort. The `HOTSPRING_NO_NUCLEUS=1`
environment variable enables standalone mode where all IPC is skipped.

---

## Per-Capability Degradation

| Capability | Primal | Unreachable Behavior | Consumer Impact |
|------------|--------|----------------------|-----------------|
| `compute.*` (dispatch) | toadStool | `call()` → `Err(Ipc(...))` | No GPU dispatch; `dispatch_cpu_fallback()` available for `vector_add` and `semf_batch` workloads. Science validation continues via CPU path. |
| `shader.compile.*` | coralReef | `call()` → `Err(Ipc(...))` | Compile-then-dispatch pipeline unavailable; pre-compiled shaders via `include_str!` still work for wgpu path. |
| `crypto.*` | BearDog | `call()` → `Err(Ipc(...))` | Optional signing skipped; `dag_provenance` falls back to unsigned DAG sessions. |
| `discovery.*` | Songbird | `discover()` → empty context | Falls to lower discovery tier or standalone mode. Physics unaffected. |
| `orchestration` | biomeOS | `announce()` → `Err(...)` logged | Self-registration skipped; CLI and physics fully functional. |
| `dag.*` | rhizoCrypt | `call()` → `Err(Ipc(...))` | Tier 3 provenance unavailable; Tier 2 science still runs. DAG session not created. |
| `spine.*` / `entry.*` | loamSpine | `call()` → `Err(Ipc(...))` | No ledger entry; DAG session valid but unbacked. |
| `braid.*` | sweetGrass | `call()` → `Err(Ipc(...))` | No attribution braid; DAG + spine are partial provenance. |
| `storage.*` | NestGate | `call()` → `Err(Ipc(...))` | Artifact storage unavailable; computation results stay in memory. |
| `inference.*` | Squirrel | `call()` → `Err(Ipc(...))` | ML inference unavailable; science paths independent of inference. |
| `visualization.*` | petalTongue | `call()` → `Err(Ipc(...))` | No rendered figures; data still valid. |
| `defense.*` / `security.*` | SkunkBat | `call()` → `Err(Ipc(...))` | Audit logging unavailable; science unaffected. |
| `precision.route` | barraCuda | `call()` → `Err(Ipc(...))` | Precision advisory unavailable; local routing used. |
| `primal.list` | biomeOS | `call()` → `Err(Ipc(...))` | Primal enumeration unavailable; discovery via socket scan. |

---

## Discovery and Circuit Breaker

`NucleusContext` implements lifecycle-aware IPC via `call_tracked()`:

- **3 consecutive failures** → endpoint marked dead (`dead_since` set)
- **30s cooldown** → `maybe_reprobe()` re-probes the endpoint
- **Retry policy**: one retry after 2s for connection-class errors
  (`connect:`, `reset`, `broken pipe`, `refused`)
- **Transport**: `call()` returns `Err(HotSpringError::Ipc(...))` for
  unknown primals, dead endpoints, and transport failures

---

## Validation Degradation

### Composition Validators (`validate_nucleus_*`)

Missing or dead primals → **failed checks** (`passed: false`), not silent
success. This is **honest degradation** — the validator reports exactly
what couldn't be validated.

### Science Parity Probes (`validate_science_probes`)

- **Compute/math not discovered**: probes return `passed: true` with
  `skip_reason: "standalone mode"` — intentional skip-pass for bare
  environments.
- **Provenance trio absent**: same skip-pass pattern.
- **Partial trio** (e.g. DAG alive but spine/braid down): `passed: false`
  with fractional skip reason — partial provenance is reported, not hidden.

### guideStone (`hotspring_unibin certify`)

- **Bare mode (30/30 checks)**: Properties 1-5 validated without primals.
  3 SKIPs for expected NUCLEUS liveness probes.
- **NUCLEUS additive mode**: IPC parity via `primalspring::composition` API.
  Missing primals → additional SKIPs, not failures.

---

## Tier 2 Live Science API (`ipc/tier2.rs`)

Functions in the Tier 2 module (`workload_preflight`, `list_workloads`,
`precision_advisory`, `dispatch_capabilities`) use `Option` returns:

- RPC failure → `None` (graceful degradation)
- Callers pattern-match on `Some`/`None` to decide local vs remote path
- Module docs state: "degrades gracefully"

---

## Registration

`niche::register_with_target()` performs self-registration with biomeOS:

1. If `HOTSPRING_NO_NUCLEUS=1` → skip entirely (info log)
2. If biomeOS socket not discovered → defer (info log)
3. Try `primal.announce` → if failed, fall back to legacy
   `lifecycle.register` + `capability.register`
4. Legacy registration errors → logged, not fatal

**Physics must never depend on registration success.**

---

## Provenance Trio Transaction Semantics

Per `PROVENANCE_TRIO_INTEGRATION_GUIDE.md` and Wave 20 handoff:

- The trio commit flow is **not atomic**
- **DAG without braid** = valid partial provenance
- **Braid without spine** = attribution without permanence
- **No rollback** — DAG sessions are append-only
- **Partial state must be reported** (e.g. `primals_reached` list)
- **Domain logic must not fail on partial provenance** — provenance is
  enrichment, not a gate

`commit_provenance()` in `dag_provenance.rs` implements this:
`nest.commit` via `signal.dispatch` with fallback to
`ledger.record` + `attribution.braid`. If trio is unavailable, the
computation result is still valid.

---

## References

- `barracuda/src/primal_bridge.rs` — `NucleusContext`, `call()`, `call_tracked()`
- `barracuda/src/composition.rs` — atomic validation, science parity probes
- `barracuda/src/niche/mod.rs` — registration, standalone mode
- `barracuda/src/ipc/tier2.rs` — live science API degradation
- `barracuda/src/dag_provenance.rs` — `commit_provenance()` trio semantics
- `barracuda/src/compute_dispatch/mod.rs` — `dispatch_cpu_fallback()`
- `primalSpring/ecoPrimal/src/composition/context.rs` — upstream degradation table
- `infra/wateringHole/PROVENANCE_TRIO_INTEGRATION_GUIDE.md` — trio commit flow
