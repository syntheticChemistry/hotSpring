# Experiment 153: Ember Flood/Resurrection Proof

**Date:** 2026-04-07
**Status:** ACTIVE — validation binary implemented, ready for live fleet run
**Track:** Sovereign GPU compute reliability
**Driven by:** Iter 78 audit — exp145 kill probes passed, flood scenario caused system locks

---

## Background

### Experiment history

- **Exp 138–140**: Crash vector hunt across Titan V (GV100) and K80 (GK210).
  Identified D-state cascades triggered by non-probe BAR0 traffic and sysfs
  reads racing with VFIO teardown.

- **Exp 144**: PMC BIT5 ACR progress — SEC2 ACR boot runs but HS auth blocked
  by VBIOS DEVINIT contradiction.

- **Exp 145**: Individual ember kill attempts (SIGTERM, SIGKILL, crash injection).
  All passed — glowplug detected the death, waited out heartbeat timeout,
  and respawned ember. GPU state preserved via FdVault checkpointed VFIO fds.

- **Exp 150–151**: Revalidation and next stages. Confirmed fleet architecture
  (per-device embers, hot-standby pool, systemd template units) is stable for
  individual fault scenarios.

### The gap: flood scenario

When multiple concurrent clients hammered a single ember with rapid-fire RPC
requests (instead of one-shot kills), the system locked:

1. Ember's accept loop spawned handlers faster than it could drain them.
2. The backlog of open Unix connections held kernel resources.
3. When ember eventually died (OOM or watchdog), socket teardown cascaded.
4. Glowplug's heartbeat (3s interval) was too slow to detect the rapid failure.
5. The vault.restore_fds path was **unimplemented** in glowplug — the new ember
   couldn't reclaim VFIO fds from the vault, so the GPU went through PM reset.

## Hypothesis

The sacrificial ember chain handles single kills (exp145) but floods can
overwhelm the resurrection cycle because:

1. Glowplug's heartbeat interval (default 3s × 3 missed = 9s detection) is slow
   relative to flood-induced failure (<1s to overwhelm).
2. Concurrent RPC connections hold Unix sockets open, preventing clean teardown
   and delaying glowplug's detection of ember death.
3. The `vault.restore_fds` handler was missing — the new ember fell back to
   fresh VFIO open from sysfs, triggering a cold GPU cycle.

## Resolution (implemented)

### 1. vault.restore_fds wired (coral-glowplug)

- Added `FdVault::restore_for_bdfs()` — takes vaulted fds for requested BDFs,
  returns them + JSON manifest describing the fd layout.
- Added `send_with_fds()` — SCM_RIGHTS fd passing via `rustix::net::sendmsg`.
- Added `handle_vault_restore()` — synchronous handler for the `vault.restore_fds`
  JSON-RPC method, receiving the request and sending fds back.
- Added MSG_PEEK detection in the async socket server's Unix accept path to route
  vault requests to the synchronous handler (SCM_RIGHTS requires raw fd access).
- Wired vault `Arc<FdVault>` from `EmberLifecycle` through to the socket server.

### 2. Flood test infrastructure (hotSpring)

- Added `FloodTestConfig`, `FloodTestResult`, `flood_test()` to
  `barracuda/src/fleet_client.rs` — spawns N concurrent threads hammering
  `ember.status`, measuring latency, success rate, and timing.
- Added `verify_ember_alive()` and `extract_ember_pid()` helpers.

### 3. Validation binary

- `barracuda/src/bin/validate_ember_resilience.rs` — 6-phase validation:
  1. Fleet baseline (discovery + probe all embers + glowplug health)
  2. Checkpoint verification (vault has fd entries)
  3. Single-kill proof (SIGKILL ember, verify resurrection)
  4. RPC flood test (50 threads × 500 requests, measure degradation)
  5. Hot-standby adoption (resilient routing with fault hints)
  6. Post-resurrection dispatch (device.get proves GPU path live)

## Protocol

### Prerequisites

- Running coral-glowplug in fleet mode (`coral-ember@*.service` per device)
- At least one compute GPU bound to VFIO (Titan V or K80)
- Fleet discovery file at `$XDG_RUNTIME_DIR/biomeos/coral-ember-fleet.json`

### Execution

```bash
# Verify fleet is running
systemctl status coral-glowplug coral-ember@*

# Run the validation binary
cargo run --release --bin validate_ember_resilience

# Or via the orchestration script
./scripts/exp153_flood_test.sh
```

### Expected behavior

| Phase | Expected outcome |
|-------|-----------------|
| Baseline | All embers alive, glowplug healthy |
| Checkpoint | At least one device with vram_alive in vault |
| Single kill | Ember dies, glowplug resurrects within 30s |
| Flood | Ember degrades or dies; OTHER embers unaffected (isolation) |
| Post-flood | Glowplug detects death, resurrects within 45s |
| Dispatch | device.get returns valid device info post-resurrection |

### Targets

- **Titan V (GV100)** — `0000:03:00.0` — primary VFIO compute
- **Tesla K80 (GK210)** — `0000:4c:00.0` / `0000:4d:00.0` — secondary compute

## Results

*To be filled during live execution.*

### Phase 1: Fleet baseline
```
(pending)
```

### Phase 3: Single-kill timing
```
(pending)
```

### Phase 4: Flood metrics
```
concurrency:
total_requests:
success_count:
failure_count:
median_latency_ms:
p99_latency_ms:
ember_survived: (yes/no)
isolation_verified: (yes/no)
resurrection_time_s:
```

## Follow-up: Backpressure (conditional)

If the flood test reveals that ember dies uncleanly or the resurrection cycle
takes too long, the following defenses should be added to coral-ember:

1. **Connection limit** in accept loop — cap concurrent clients (e.g. 32)
2. **Request rate limiting** — drop connections exceeding N requests/second
3. **Graceful overload response** — JSON-RPC error `{"code":-32000,"message":"overloaded"}`
4. **Voluntary sacrifice threshold** — if backpressure exceeds limit for T seconds,
   trigger orderly shutdown (checkpoint fds → die) rather than uncontrolled death

## Files

| File | Role |
|------|------|
| `coralReef/crates/coral-glowplug/src/fd_vault.rs` | `restore_for_bdfs()`, `send_with_fds()`, `handle_vault_restore()` |
| `coralReef/crates/coral-glowplug/src/socket/mod.rs` | MSG_PEEK vault routing, `is_vault_restore_request()` |
| `coralReef/crates/coral-glowplug/src/main.rs` | Vault Arc threading to accept_loop |
| `hotSpring/barracuda/src/fleet_client.rs` | `flood_test()`, `FloodTestConfig`, `verify_ember_alive()` |
| `hotSpring/barracuda/src/bin/validate_ember_resilience.rs` | 6-phase validation binary |
| `hotSpring/scripts/exp153_flood_test.sh` | Orchestration: systemctl + binary + log collection |
