# hotSpring — IPC Transport Evolution Handoff (May 12, 2026)

**GAP-HS-092 — `call_by_capability` Proliferation**

## Summary

All hotSpring IPC client modules now use a unified 3-tier resolution pattern
for primal communication, eliminating the previous split between
`by_domain()` discovery and `send_jsonrpc()` transport.

## Pattern: Unified 3-Tier IPC Resolution

Every IPC client function now follows:

```
1. call_by_capability(domain, method, params)  — NUCLEUS routing + transport
2. send_jsonrpc(socket, method, params)        — direct socket (discovered or env-var)
3. socket_dirs() / env fallback                — CI/lab without NUCLEUS
```

**Why this matters for sibling springs:**

- Primal code never embeds socket paths beyond the fallback tier
- Discovery and transport are unified in one call — no socket leakage
- The pattern degrades gracefully: NUCLEUS → env var → filesystem scan
- All primal knowledge is self-knowledge; other primals discovered at runtime

## Modules Evolved

| Module | Domain | Method | Fallback |
|--------|--------|--------|----------|
| `ipc/biome_status.rs` | `composition` | `composition.status` | `BIOMEOS_SOCKET` env → `socket_dirs()` |
| `ipc/method_register.rs` | `composition` | `method.register` | `BIOMEOS_SOCKET` env → `socket_dirs()` |
| `ipc/skunkbat.rs` | `security` | `security.audit_log` | `by_domain("security")` direct socket |
| `ipc/provenance/sweetgrass.rs` | `attribution` | `attribution.braid` | `by_domain("attribution")` direct socket |
| `ipc/provenance/rhizocrypt.rs` | `dag` | `dag.submit_witness` | `by_domain("dag")` direct socket |
| `ipc/provenance/loamspine.rs` | `ledger` | `ledger.record` | `by_domain("ledger")` direct socket |
| `fleet_toadstool.rs` | `compute` | `compute.dispatch.*` | Cached socket from `discover()` |
| `fleet_client.rs` | `ember` | `ember.list` (BDF scan) | Filesystem diesel layout scan |

## Code Pattern (reference for sibling springs)

```rust
pub fn query_my_service() -> Option<MyResult> {
    let ctx = NucleusContext::detect();
    let params = serde_json::json!({ "key": "value" });

    // Tier 1: Unified capability-based routing + transport
    if let Ok(resp) = ctx.call_by_capability("my_domain", "my.method", params.clone()) {
        return serde_json::from_value(resp).ok();
    }

    // Tier 2: Env var fallback
    if let Ok(p) = std::env::var("MY_SERVICE_SOCKET") {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            let resp = send_jsonrpc(&path, "my.method", &params).ok()?;
            return serde_json::from_value(resp).ok();
        }
    }

    // Tier 3: Socket-dir filesystem scan (CI/lab)
    let socket = niche::socket_dirs()
        .into_iter()
        .map(|d| d.join("my_service/my_service.sock"))
        .find(|p| p.exists())?;
    let resp = send_jsonrpc(&socket, "my.method", &params).ok()?;
    serde_json::from_value(resp).ok()
}
```

## Hardware Calibration Refactoring

`TierCapability::failed(tier)` and `TierCapability::compiled_only(tier, compile_us)`
constructors replace ~50 lines of repeated struct initialization across the
GPU calibration probing pipeline. Pattern: when a struct has common "default
failure" states, provide named constructors rather than repeating field lists.

## Wildcard Match Audit Results

All `_ =>` match arms on `PrecisionTier` (15-variant upstream enum) were audited.
They are intentional fallbacks — unknown/new upstream tiers (Binary, Int2,
Quantized4, FP8, BF16, TF32, AF32, Mixed8, Mixed16, Stochastic, Custom)
correctly default to F32 pipeline compilation and `(false, false)` for f64
shader flags. No hidden bugs.

## Production Mock Audit

Zero production mocks found. All `mock`/`stub`/`fake` patterns are properly
isolated to `#[cfg(test)]` modules.

## Metrics

- **584** lib tests (default) / **1,036** (barracuda-local + toadstool-dispatch)
- **166** binaries | **64/64** validation suites | **128** WGSL shaders
- **9** validation scenarios (default) / **12** (barracuda-local)
- Zero clippy warnings | Zero lint errors

## Adoption Guidance for Sibling Springs

1. **Migrate `by_domain()` + `send_jsonrpc()` pairs** to single
   `call_by_capability()` calls in your IPC modules
2. **Keep fallback tiers** for CI/lab environments where NUCLEUS isn't running
3. **Use `NucleusContext::detect()`** — it's zero-cost when NUCLEUS isn't available
4. **Don't hardcode primal socket paths** — let capability discovery find them
5. **Test with `HOTSPRING_NO_NUCLEUS=1`** to verify fallback degradation

## Related Documents

- `docs/PRIMAL_GAPS.md` — GAP-HS-092 full detail
- `docs/PRIMAL_PROOF_IPC_MAPPING.md` — IPC Transport Evolution section
- `CHANGELOG.md` — "IPC Transport Evolution: call_by_capability Proliferation"

---

AGPL-3.0-or-later
