# Sovereign Pipeline + PLX Keepalive Evolution Handoff

**Date:** 2026-05-10
**coralReef iteration:** 95+
**Status:** PLX keepalive hardened. Sovereign pipeline pre-flight integrated. Config validation enforced.

---

## Summary

The PLX PEX 8747 keepalive has been evolved from a fragile bash script
(`plx-keepalive.sh`) into a first-class `coral-ember` subsystem. The
sovereign boot pipeline now performs PCIe switch pre-flight checks before
attempting BAR0 probes, saving 10+ seconds on dead hardware. Configuration
validation prevents path-traversal via malformed BDFs and CPU-spinning
from zero-interval keepalive.

## Architecture Changes (coralReef upstream)

### coral-ember

| Component | Change | Why |
|-----------|--------|-----|
| `config.rs` | Added `validate_bdf()` — 12-char `DDDD:BB:DD.F` hex check | BDF strings go directly into sysfs paths; malformed strings = path traversal risk |
| `config.rs` | Added `validate_pcie_switches()` — interval clamped to ≥250ms | `keepalive_interval_ms = 0` caused CPU-spinning tight loop |
| `pcie_keepalive.rs` | `pci_config_probe()` reads vendor-ID (0x00) AND COMMAND register (0x04) | Matches the TLP traffic volume the old bash script generated |
| `pcie_keepalive.rs` | Endpoint device pings via `endpoint_bdfs` map | K80 GPUs behind the PLX switch now receive periodic config-space reads |
| `pcie_keepalive.rs` | `SwitchHealth` gains `endpoint_alive`, `last_error` fields | Distinguishes "dead" from "I/O error" and reports endpoint liveness |
| `ipc/handlers_switch.rs` | Lock-poison returns JSON-RPC error (-32603) | Previously returned empty array — indistinguishable from "no switches" |
| `runtime.rs` | Validates all BDFs at startup; builds endpoint map from raw TOML | Fails fast on bad config; endpoints derived from `upstream_switch` fields |

### coral-glowplug

| Component | Change | Why |
|-----------|--------|-----|
| `config.rs` | `Config::load()` calls `validate()` — BDF + cross-ref checks | Catches orphaned `upstream_switch` references at load time |
| `health.rs` | `query_switch_health()` wrapped in `spawn_blocking` + 2s timeout | Was blocking the Tokio async health loop for up to 10s on hung ember |
| `health.rs` | Switch health failures promoted from `debug` to `warn` level | Silent failures in production are invisible |
| `sovereign.rs` | `preflight_switch_check()` before BAR0 probe | Short-circuits with actionable error when PLX is D3cold-gated |
| `sovereign.rs` | `device_config_for_bdf()` replaces `kepler_fw_dir_for_bdf()` | Single config lookup function for all device fields |
| `error.rs` | Added `ConfigError::ValidationFailed` variant | Clean error path for semantic config validation |

### Systemd / boot wiring

| Change | Why |
|--------|-----|
| `k80-sovereign-wake.service` orders `After=coral-glowplug.service` | Was only after ember — race condition on glowplug socket |
| `coral-ember.service` gains `StartLimitIntervalSec=300 / Burst=3` | Prevents infinite crash-restart loops |
| `k80-wake-and-run.sh` uses socket-readiness poll loops (30s) | Replaced fixed `sleep 2` / `sleep 1` — eliminates race conditions |
| `k80-wake-and-run.sh` extracts BDFs from `glowplug.toml` via `tomllib` | Zero hardcoded BDFs — machine-agnostic |
| `install-boot-config.sh` disables deprecated `plx-keepalive.service` | Was still installing/enabling the deprecated bash keepalive |

## Upstream Debt for coralReef

### Patterns discovered in hotSpring that should inform coralReef evolution

1. **Config-as-contract**: The `[[pcie_switch]]` schema with `downstream_ports`
   and `upstream_switch` cross-references is a composition pattern — switches
   should be first-class topology entities in coralReef, not just glowplug config.

2. **Blocking sync RPC in async context**: `EmberClient::simple_rpc()` is
   synchronous (10s timeout). Every async caller must wrap it in `spawn_blocking`.
   Consider an async `EmberClient` variant or at minimum document the blocking
   contract on the type.

3. **Config validation gap**: `EmberConfig` and `Config` deserialize via
   `toml::from_str` with no post-validation. The `validate_bdf` and
   `validate_pcie_switches` functions are bolt-ons. A builder pattern or
   `TryFrom<RawConfig>` would be cleaner long-term.

4. **Switch health as topology primitive**: The keepalive thread, switch status
   RPC, and sovereign pre-flight all treat the PCIe switch as a first-class
   health entity. This pattern should generalize to any PCIe bridge/switch in
   the topology — not just PLX.

5. **Experiment wiring standard**: `.cursor/rules/experiment-wiring-standard.mdc`
   says all GPU access goes through ember, but `exp169`/`170`/`171`/`182`/`183`
   still use direct BAR0 mmap (`low-level` feature). Either the standard should
   acknowledge diagnostic-only exceptions, or those experiments should be
   refactored to use ember RPC.

## Composition Patterns for NUCLEUS Deployment

### neuralAPI / biomeOS integration points

- `coral-ember` and `coral-glowplug` use `CORALREEF_EMBER_SOCKET` and
  `CORALREEF_GLOWPLUG_CONFIG` env vars — these should be wired into the
  NUCLEUS composition graph when sovereign GPU compute is a node.
- The `niche.rs` pattern in hotSpring already declares GPU compute capability
  via `resolve_neural_api_socket` — coralReef's health surface (`device.list`,
  `device.health`, `ember.switch.status`) should be queryable via the same
  neuralAPI routing.

### Downstream systems that should absorb these patterns

- **agentReagents**: K80 QEMU VM reagent uses `nvidia-470` to POST GPUs.
  The pre-flight switch check pattern should be wired into reagent launch
  scripts — if the PLX is dead, reagent launch should fail fast instead of
  hanging on VFIO group open.
- **Other springs**: Any spring doing GPU compute on PLX-bridged hardware
  benefits from the `upstream_switch` → `switch_alive` health propagation.
- **primalSpring**: The guideStone certification should include a
  "sovereign-compute-ready" property for springs that do GPU work.

## Config File Reference

System-wide config: `/etc/coralreef/glowplug.toml`

```toml
[[pcie_switch]]
bdf = "0000:49:00.0"
name = "plx-pex-8747"
downstream_ports = ["0000:4a:08.0", "0000:4a:10.0"]
keepalive_interval_ms = 3000
disable_aspm = true

[[device]]
bdf = "0000:4b:00.0"
upstream_switch = "0000:49:00.0"   # links to [[pcie_switch]] entry
kepler_fw_dir = "/var/lib/coralreef/firmware/gk110"
```

## Verification

After power cycle:
```bash
# Switch keepalive alive:
echo '{"jsonrpc":"2.0","method":"ember.switch.status","params":{},"id":1}' \
  | socat - UNIX-CONNECT:/run/coralreef/ember.sock

# Expected: alive=true, vendor_id=0x8747, endpoint_alive populated, consecutive_dead=0
```

## Files Changed (hotSpring local)

- `scripts/boot/k80-wake-and-run.sh` — BDFs from config, poll loops, DRM rules from config
- `scripts/boot/install-boot-config.sh` — disables deprecated plx-keepalive
- `scripts/boot/post-boot-oracle-capture.sh` — fixed stale BDF (4a→4b)
- `scripts/boot/plx-keepalive.sh` — DEPRECATED header (kept as fossil record)
- `wateringHole/warm_handoff.sh` — DEPRECATED header
- `/etc/coralreef/glowplug.toml` — removed dead `fleet_mode`, `standby_pool_size`

## Files Changed (coralReef upstream)

- `coral-ember/src/config.rs` — `validate_bdf`, `validate_pcie_switches`
- `coral-ember/src/pcie_keepalive.rs` — endpoint pings, COMMAND reads, `last_error`
- `coral-ember/src/runtime.rs` — config validation, endpoint map
- `coral-ember/src/ipc/handlers_switch.rs` — lock-poison error
- `coral-ember/src/lib.rs` — export validation functions
- `coral-glowplug/src/config.rs` — `Config::validate()`
- `coral-glowplug/src/health.rs` — `spawn_blocking`, `warn` level
- `coral-glowplug/src/sovereign.rs` — `preflight_switch_check`, `device_config_for_bdf`
- `coral-glowplug/src/error.rs` — `ValidationFailed` variant
