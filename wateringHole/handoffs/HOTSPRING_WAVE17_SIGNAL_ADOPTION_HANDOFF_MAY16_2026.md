# hotSpring Wave 17 Signal Adoption Handoff — May 16, 2026

**Sprint:** Wave 17 Signal Adoption
**Spring:** hotSpring v0.6.32
**Upstream:** primalSpring `SIGNAL_ADOPTION_STANDARD.md` + `PRIMAL_ANNOUNCE_PROTOCOL.md`
**Archetype:** Compute-heavy (Node atomic dominant)

---

## What was done

### 1. `primal.announce` registration (niche.rs)

`register_with_target()` refactored from a monolithic 114-line function into three
composable pieces:

- **`try_primal_announce()`** — sends a single `primal.announce` JSON-RPC call
  carrying all methods, capabilities, semantic mappings, and signal tiers. On
  success, registration is complete in one round-trip.
- **`legacy_register()`** — the original `lifecycle.register` + N × `capability.register`
  pattern, kept as automatic fallback for older biomeOS.
- **`discover_biomeos_socket()`** — socket probe logic extracted for reuse.

The `primal.announce` payload includes:
- `primal`, `socket`, `pid`, `version`
- `capabilities: ["physics", "composition"]`
- `methods`: all `LOCAL_CAPABILITIES`
- `routed_methods`: all `ROUTED_CAPABILITIES` with `canonical_provider`
- `semantic_mappings`: physics semantic map
- `signal_tiers: ["node", "nest"]`

### 2. `node.compute` signal dispatch (compute_dispatch.rs)

New `dispatch_node_compute()` dispatches GPU workloads via the `node.compute` signal:

```
signal.dispatch("node.compute", {
    wgsl_source, input, input_hash, spring, bdf?
})
```

biomeOS decomposes this into the compile → submit → execute graph over the
toadStool → coralReef → barraCuda pipeline. Falls back to
`compile_and_submit()` for older biomeOS.

### 3. `tower.publish` signed publication (compute_dispatch.rs)

New `publish_result()` dispatches via the `tower.publish` signal:

```
signal.dispatch("tower.publish", {
    content, content_hash, topic, spring
})
```

biomeOS decomposes into sign (bearDog) → announce (songBird) → audit (skunkBat).
Falls back to direct `crypto.sign_ed25519` + `discovery.announce`.

### 4. Capability registry signal annotations

`capability_registry.toml` extended with:
```toml
[signals]
adopted = ["node.compute", "tower.publish"]
candidates = ["nest.store", "nest.commit"]

[registration]
method = "primal.announce"
fallback = ["lifecycle.register", "capability.register", "method.register"]
```

---

## What stays as `ctx.call()`

Per the signal adoption standard, domain-specific physics operations are NOT
signal candidates. These stay as direct `ctx.call()`:

- `stats.mean`, `stats.variance`, `stats.autocorrelation`
- `linalg.eigensolve`, `linalg.cholesky`
- `tensor.matmul`, `tensor.fma`
- `physics.lattice_qcd`, `physics.hfb_solve`, etc.

Signals replace only orchestration sequences that biomeOS can graph-execute.

---

## Next signal candidates

| Signal | Use case | Priority |
|--------|----------|----------|
| `nest.store` | Physics result → provenance chain (content.put + dag + spine + braid) | Medium |
| `nest.commit` | Session finalization → dehydrate + certificate | Medium |
| `nest.retrieve` | Result retrieval from content-addressed storage | Low |

---

## Metrics

- 595/595 lib tests pass
- Zero clippy warnings (lib, `--features barracuda-local`)
- Zero format drift
- Backward-compatible: all signal paths have automatic legacy fallback

---

## Upstream asks

1. **Confirm `primal.announce` wire field names**: primalSpring `context.rs` uses
   `primal_id` + `transport`; the protocol doc specifies `primal` + `socket`. We
   send `primal` + `socket` per the protocol. biomeOS should accept both.

2. **`node.compute` graph validation**: The signal graph (`graphs/signals/node_compute.toml`)
   should match the toadStool → coralReef → barraCuda pipeline that hotSpring
   validates. If the graph nodes diverge, hotSpring's `dispatch_node_compute()`
   fallback will activate but the signal path won't exercise.

3. **13 methods pending primalSpring canonical registry**: Cross-registry sync
   test (`integration_registry_sync.rs`) identifies 13 hotSpring-specific methods
   not yet in primalSpring's 451-method registry. Advisory, not blocking.
