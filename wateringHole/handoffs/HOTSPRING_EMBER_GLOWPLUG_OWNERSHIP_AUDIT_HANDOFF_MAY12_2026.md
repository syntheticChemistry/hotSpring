# hotSpring -> Ember/Glowplug Ownership Audit Handoff

**Date:** May 12, 2026
**GAP:** GAP-HS-096
**Trigger:** Ember subprocess crash traced to stale coralReef binary; dual-existence discovered
**Status:** coralReef binaries rebuilt + bugs fixed; toadStool parity audit complete

---

## Discovery: Dual-Existence of Ember/Glowplug/Cylinder

Ember, glowplug, and cylinder were evolved in coralReef through hardware
experimentation with hotSpring. toadStool then absorbed them via Wave 8
Phase A (ember, S237) and Phase B (glowplug, S239). However, the absorption
was types and traits, not the daemon runtime. Both primals now contain
implementations at different maturity levels.

## Ownership Map

### coralReef (deployed, still required)

| Component | Path | What It Does |
|-----------|------|-------------|
| `coral-ember` binary | `crates/coral-ember/` | Immortal VFIO fd holder, `ember.swap`, `ember.warm_catch`, MMIO handlers, journal, kmod, sovereign boot |
| `coral-glowplug` binary | `crates/coral-glowplug/` | Diesel engine ECU, cylinder spawning, `device.*` handler dispatch |
| `coralctl` binary | `crates/coral-glowplug/src/bin/coralctl/` | 20+ CLI commands: warm-fecs, warm-catch, swap, status, health, dispatch, probe, snapshot, etc. |
| `coral-driver` | `crates/coral-driver/` | VFIO stack, DRM enum, AMD GEM/PM4, NV BAR0/QMD/pushbuf, vfio_compute |
| `coral-gpu` | `crates/coral-gpu/` | Unified GpuContext (compile + dispatch), warm API |
| systemd | `coral-glowplug.service` | Root ECU, spawns cylinders which spawn embers |

### toadStool (Phase A+B absorbed, not deployed as daemon)

| Component | Path | What It Does | Parity Gap |
|-----------|------|-------------|------------|
| `toadstool-ember` (lib) | `crates/core/ember/` | ResourceHandle, HeldResource, VfioResourceHandle | Metadata-only VfioResourceHandle (no real VFIO fds) |
| `toadstool-glowplug` (lib) | `crates/core/glowplug/` | SwapOrchestrator (7-step), SysfsSwapExecutor, DevicePersonality | Quiesce/persist/restore are stubs; warm_cycle always false |
| `toadstool-server` | `crates/server/` | GlowPlugClient (list/swap/reacquire Rust API), JSON-RPC: ember.list, ember.status | No device.swap, warm-catch, or cylinder RPC |
| `toadstool` CLI | `crates/cli/` | Daemon mode (workload manager) | No warm-fecs, swap, health CLI commands |
| GPU runtime | `crates/runtime/gpu/src/glowplug/` | GPU-specific personality/discovery/firmware | Firmware reads exist; no warm pipeline |

### Phase C PENDING (nobody owns yet)

| Component | Description |
|-----------|-------------|
| `toadstool-cylinder` crate | Per-device subprocess isolation (generalized from coral-glowplug cylinder) |
| coral-driver absorption | VFIO stack, DRM enum, AMD, NV hardware modules into toadStool |
| VFIO fd holding | End-to-end fd ownership through driver swaps |
| Warm pipeline | warm-fecs, warm-catch, livepatch, nouveau round-trip |
| coralctl parity | ~20 CLI subcommands in toadstool CLI |

## Bugs Fixed During Audit

### 1. Stale coral-ember binary (zombie crash)

**Symptom:** `/usr/local/bin/coral-ember` (pre-absorption build) crashed immediately
when spawned by cylinder, producing `[coral-ember] <defunct>` zombies.

**Fix:** Rebuilt all three binaries from current coralReef source:
```
cargo build --release -p coral-ember -p coral-glowplug
sudo cp target/release/{coral-ember,coral-glowplug,coralctl} /usr/local/bin/
```

### 2. Cylinder device.swap translation (diesel engine routing gap)

**Symptom:** `coralctl warm-fecs` sends `device.swap` -> ECU routes by BDF to
cylinder -> cylinder forwards raw to ember -> ember returns "method not found:
device.swap" (ember only knows `ember.swap`).

**Root cause:** In diesel engine mode, the ECU's `ecu_rpc.rs` routes `device.*`
methods to the cylinder by BDF. The cylinder's catch-all `_method` arm forwards
the raw request to ember without translating the method name.

**Fix:** Added translation layer in `cylinder.rs` catch-all:
```rust
method => {
    let translated = if let Some(suffix) = method.strip_prefix("device.") {
        let ember_method = format!("ember.{suffix}");
        let mut translated_req: serde_json::Value =
            serde_json::from_str(line).unwrap_or_default();
        translated_req["method"] = serde_json::Value::String(ember_method);
        serde_json::to_string(&translated_req).unwrap_or_else(|_| line.to_string())
    } else {
        line.to_string()
    };
    match ember.as_ref() {
        Some(e) => forward_to_ember(&e.socket_path, &translated, req.id),
        None => make_error(req.id, -32000, "ember not running"),
    }
}
```

### 3. Titan V Warm Dispatch Results (post-fix)

| Test | Result |
|------|--------|
| VFIO warm open (SM70) | PASS |
| WGSL compile (write_constant) | PASS (192 bytes, 22 GPRs) |
| WGSL compile (wilson_plaquette_f64) | PASS (6000 bytes, 38 GPRs) |
| WGSL compile (su3_gauge_force_f64) | PASS (20096 bytes, 54 GPRs) |
| Dispatch + readback | FAIL (0xDEADBEEF — FECS compute context not initialized) |

Compilation works through the full pipeline: WGSL -> coral-reef compiler ->
native Volta SASS. Dispatch doesn't execute because the FECS falcon isn't in
a compute-ready state after a single nouveau round-trip (the warm-fecs cycle
loads FECS firmware but doesn't bring up a compute context/channel).

## Cutover Path: coralReef -> toadStool

### Phase C Prerequisites (for toadStool to replace coral-glowplug)

1. **`toadstool-cylinder` crate** — per-device subprocess with VFIO fd holding
2. **coral-driver absorption** — VFIO, DRM, AMD, NV hardware modules
3. **`toadstool daemon` upgrades:**
   - Wire `ember.swap`, `device.swap`, `ember.warm_catch` into server JSON-RPC
   - SwapOrchestrator quiesce/persist/restore real implementations
   - Warm cycle pipeline (nouveau round-trip + livepatch + FECS preservation)
4. **`toadstool` CLI upgrades:**
   - `toadstool warm-fecs`, `toadstool swap`, `toadstool status`, `toadstool health`
   - Migrate coralctl's ~20 subcommands
5. **systemd:** `toadstool.service` replacing `coral-glowplug.service`
6. **Testing:** All warm-catch + dispatch validation must pass under toadStool daemon

### Transition Strategy

1. **Now:** coralReef binaries deployed, bugs fixed. Warm dispatch partially proven.
2. **Phase C:** toadStool absorbs coral-driver, creates cylinder crate, adds RPC.
3. **Parallel run:** Both daemons available; hotSpring validates against toadStool.
4. **Cutover:** toadStool daemon deployed, coralReef binaries soft-deprecated.
5. **Cleanup:** coralReef `crates/coral-ember` and `crates/coral-glowplug` archived.

## Evolution Pattern

hotSpring solved hardware lifecycle problems locally in coralReef.
coralReef evolved ember/glowplug/cylinder through iteration.
toadStool absorbed the abstractions (Phase A+B) but not the runtime.
The runtime must now follow the abstractions into toadStool (Phase C+D).
Pattern: solve locally -> hand upstream -> absorb -> resolve with new abstraction.
