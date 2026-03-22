# HOTSPRING → coralReef / toadStool / barraCuda: Ember Hardening Handoff

**Date:** March 22, 2026
**From:** hotSpring (Sovereign MMU Evolution Sprint, Exp 074 deep debt resolution)
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** coralReef commits up to `ce66de4` (ember BDF allowlist, preflight device checks, VRAM write-readback canary)

---

## Executive Summary

- **3 deep debts resolved**: VRAM health false positives, unmanaged BDF operations, D-state kernel hangs from blind sysfs writes
- **264 tests pass** (86 ember + 178 glowplug) — all run quickly with full test isolation from live daemons
- **Warm-swap operational**: Both Titan V cards (`0000:03:00.0`, `0000:4b:00.0`) round-trip nouveau ↔ vfio with HBM2 alive
- **Ember FD sharing validated**: `EmberSession` (`SCM_RIGHTS`) provides dup'd VFIO file descriptors to diagnostic binaries — direct BAR0 access without triggering VFIO group-level bus resets
- **Display GPU safety guard**: `coral-ember` refuses to unbind devices actively driving a display, preventing the kernel crashes discovered during this sprint

---

## Part 1: Deep Debt — VRAM Health Check False Positives

### Problem

`coral-glowplug`'s `vram_alive` health check read a PRAMIN offset and returned `true` if the value was nonzero. On a cold-booted GPU with untrained HBM2, PRAMIN reads return stale/undefined data that passes a nonzero check — a false positive.

### Resolution

Replaced the read-nonzero check with a **write-readback canary test**:

```rust
const CANARY: u32 = 0xC0A1_BEEF;
// Write canary → read back → restore original → compare
```

The `pramin_write_readback` function in `coral-glowplug/src/device/health.rs`:
1. Sets BAR0_WINDOW to page 0
2. Reads original value at PRAMIN_BASE + 0x100
3. Writes `0xC0A1_BEEF`
4. Reads back — if mismatch, VRAM is dead
5. Restores original value

### Files Changed

- `coral-glowplug/src/device/health.rs` — `pramin_write_readback()` function, `check_health()` updated

### Action Items

- **coralReef**: Adopted in `ce66de4`. No further action.
- **barraCuda**: The `RegisterMap` health-check patterns should adopt write-readback verification rather than simple register reads. The canary offset (PRAMIN_BASE + 0x100) avoids firmware-sensitive regions.
- **toadStool**: The `hw-learn` health feed should distinguish "VRAM responsive" (write-readback passes) from "VRAM initialized" (HBM2 trained by nouveau/nvidia).

---

## Part 2: Deep Debt — BDF Allowlist

### Problem

`coral-ember` accepted any BDF in RPC requests. A test suite accidentally sent `swap_device` RPCs for the display GPU (`0000:21:00.0`), unbinding the `nvidia` driver and causing a kernel oops in `nvidia_modeset`. This crashed the system twice.

### Resolution

`coral-ember`'s `run()` function now collects all configured device BDFs from `glowplug.toml` into an `Arc<HashSet<String>>`. Every BDF-bearing RPC (`ember.vfio_fds`, `ember.release`, `ember.reacquire`, `ember.swap`) validates the requested BDF against this allowlist before processing.

Unmanaged BDFs receive a JSON-RPC error:

```json
{"jsonrpc": "2.0", "error": {"code": -32001, "message": "BDF 0000:21:00.0 is not managed by ember"}, "id": 1}
```

### Files Changed

- `coral-ember/src/lib.rs` — `HashSet<String>` collection from config, passed to `ipc::handle_client`
- `coral-ember/src/ipc.rs` — `require_managed_bdf()` helper, integrated into 4 RPC handlers, 6 new unit tests

### Action Items

- **coralReef**: Adopted in `ce66de4`. No further action.
- **toadStool**: `GlowPlugClient` should handle the `-32001` error gracefully — the BDF may have been removed from config or the device may not be present.
- **barraCuda**: The config-driven device scope pattern (allowlist from TOML) is a model for any multi-device management layer.

---

## Part 3: Deep Debt — Pre-flight Device Checks

### Problem

`coral-ember`'s `handle_swap_device` performed sysfs unbind/bind operations without verifying the device was in a safe state. Attempting to unbind a device in D3cold or with config space returning `0xFFFF` (device not responding) caused kernel-level stalls, leaving processes in D-state (uninterruptible sleep) that required a reboot.

### Resolution

A `preflight_device_check` function in `coral-ember/src/swap.rs` runs before any unbind:

1. **Sysfs path existence**: If `/sys/bus/pci/devices/<BDF>` doesn't exist, the device is already effectively unbound. For `target="unbound"`, this returns immediate success.
2. **Power state**: Reads `power_state` from sysfs. If D3hot, attempts recovery via `D0` write. If D3cold or unreadable, returns error.
3. **Config space accessibility**: Reads first 2 bytes of PCI config space. If vendor ID is `0xFFFF`, the device is not responding — returns error.

### Files Changed

- `coral-ember/src/swap.rs` — `preflight_device_check()`, `is_active_display_gpu()`, "unbound" target short-circuit

### Action Items

- **coralReef**: Adopted in `ce66de4`. No further action.
- **toadStool**: `hw-learn` should observe and log power state transitions (D0 ↔ D3hot ↔ D3cold) as training signals for device reliability prediction.
- **barraCuda**: No direct impact, but the D3hot recovery pattern (write `D0` to `power_state`) is useful knowledge for any PCIe device management.

---

## Part 4: Display GPU Safety Guard

### Problem

`coral-ember` had no concept of "this GPU is driving the display." When tests or misconfigured RPCs targeted the display GPU, unbinding its driver caused immediate kernel crashes.

### Resolution

`is_active_display_gpu()` in `coral-ember/src/swap.rs` checks:
1. Does the device have active DRM connectors? (presence of `/sys/class/drm/card*/device` symlink pointing to the BDF)
2. Is the device bound to the `nvidia` proprietary driver with an associated DRM card?

If either check is true, `handle_swap_device` returns an error and refuses to proceed.

### Action Items

- **coralReef**: Adopted. The guard is conservative — it blocks swap on any device with DRM presence, even if the connectors are unused. A future refinement could check connector status (connected vs disconnected).
- **toadStool**: The display GPU detection logic should be shared — `hw-learn`/`sysmon` should mark display-attached GPUs as "do not disturb" in the hardware census.

---

## Part 5: Test Isolation

### Problem

`cargo test -p coral-glowplug` connected to a live `coral-ember` daemon during test execution. Tests that exercised `DeviceSlot::swap()` or `check_health()` sent real RPC commands to ember, which executed real sysfs operations — including the display GPU unbind that crashed the system.

### Resolution

Introduced a thread-local RAII guard in `coral-glowplug/src/ember.rs`:

```rust
#[cfg(test)]
thread_local! {
    static EMBER_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
```

`EmberClient::disable_for_test()` returns a guard that sets `EMBER_DISABLED` to `true` for the duration of the test. When `EMBER_DISABLED` is true, `EmberClient::connect()` returns an error without attempting a socket connection.

All 24 `coral-glowplug` tests that could trigger ember connections were updated to use this guard.

### Files Changed

- `coral-glowplug/src/ember.rs` — `EMBER_DISABLED` thread-local, `DisableForTestGuard`
- `coral-glowplug/src/device/coverage_tests.rs` — guard added to 8 tests
- `coral-glowplug/src/device/health_tests.rs` — guard added to 4 tests
- `coral-glowplug/src/device/tests.rs` — guard added to 12 tests

### Action Items

- **toadStool**: If `GlowPlugClient` is used in tests, adopt a similar test isolation pattern — never connect to live daemons from unit tests.
- **coralReef**: The `#[cfg(test)]` thread-local pattern is idiomatic Rust for test isolation without `unsafe` code. Consider documenting it as a crate convention.

---

## Part 6: Lessons Learned

### VFIO Group-Level Bus Reset

VFIO performs a **group-level bus reset** when the last file descriptor for an IOMMU group is closed. This reset:
- Is independent of `reset_method` (clearing `reset_method` only affects the per-device FLR/SBR path)
- Kills HBM2 training state
- Cannot be prevented except by keeping at least one FD open

**Implication**: `coral-ember` must hold the VFIO FD permanently to preserve HBM2. Diagnostic binaries must obtain access via `EmberSession` (FD sharing) rather than opening the device directly.

### D3hot Recovery

Writing `D0` to `/sys/bus/pci/devices/<BDF>/power_state` can recover a device from D3hot. This is a PCI spec-compliant operation but should only be attempted after verifying the device sysfs path exists and config space is readable.

### Config Space 0xFFFF

A PCI config space vendor ID of `0xFFFF` means the device is not responding on the bus. This can happen after a failed reset, D3cold entry, or hardware fault. **Never attempt sysfs unbind/bind on a device returning 0xFFFF** — the kernel will stall in D-state.

### Display Driver Unbind Crashes

Unbinding the `nvidia` proprietary driver from a GPU actively rendering a display causes a kernel oops in `nvidia_modeset`. The crash is immediate and unrecoverable without a power cycle. The `nouveau` driver may handle this more gracefully (untested), but the safe approach is to never unbind any display-attached GPU.

---

## Compute Trio Status

### Fleet Configuration (biomeGate, post-reboot validated)

| Device | BDF | Driver | Role | HBM2 | Power |
|--------|-----|--------|------|-------|-------|
| Titan V #1 | `0000:03:00.0` | `vfio-pci` | Sovereign compute (Oracle) | Alive (warm-swapped) | D0 |
| Titan V #2 | `0000:4b:00.0` | `vfio-pci` | Sovereign compute (Target) | Alive (warm-swapped) | D0 |
| RTX 5060 | `0000:21:00.0` | `nvidia` | Display + DRM oracle | N/A (GDDR7) | D0 |

### Operational Sequence (validated March 22, 2026)

1. Boot → both Titans on `vfio-pci` (ember holds FDs)
2. `coralctl swap 0000:03:00.0 nouveau` → HBM2 trained by nouveau
3. `coralctl swap 0000:03:00.0 vfio` → warm swap back, `vram_alive=true`
4. Repeat for `0000:4b:00.0`
5. `bench_vram_probe --ember 0000:03:00.0` → BAR0 via EmberSession, VRAM read/write verified
6. `bench_mmu_fault_diagnostic --ember 0000:03:00.0` → MMU fault diagnostic via EmberSession

### Test Counts

| Crate | Tests | Status |
|-------|-------|--------|
| `coral-ember` | 86 | All pass |
| `coral-glowplug` | 178 | All pass (isolated from live daemon) |
| `hotSpring/barracuda` | 848 | All pass |
| **Total** | **1,112** | **All pass** |

---

## Per-Primal Action Items

### coralReef

1. All 3 deep debts are delivered (`ce66de4`). No code changes needed.
2. **Future**: Refine `is_active_display_gpu()` to check DRM connector status (connected vs disconnected) for multi-monitor setups.
3. **Future**: Consider making the VRAM write-readback canary offset configurable per vendor (PRAMIN layout differs across GPU architectures).

### toadStool

1. Handle JSON-RPC `-32001` error from ember (unmanaged BDF) in `GlowPlugClient`.
2. Adopt `EmberClient::disable_for_test()` pattern if wiring `GlowPlugClient` into test suites.
3. Feed `vram_alive` (write-readback verified) and power state into `hw-learn` health models.
4. Mark display-attached GPUs as "do not disturb" in sysmon hardware census.

### barraCuda

1. Adopt write-readback canary pattern for `RegisterMap` health checks.
2. The config-driven BDF allowlist is a model for multi-device management scoping.
3. D3hot recovery and 0xFFFF config space detection are useful patterns for any PCIe device layer.
4. `coralctl probe`, `coralctl vram-probe`, `coralctl mmio read/write`, and `coralctl snapshot save/diff` are now available for GPU state inspection — use these for register map validation.

---

*This handoff documents the ember hardening sprint that resolved 3 deep debts,
prevented 2 classes of kernel crashes, and established the operational baseline
for the sovereign compute pipeline on biomeGate's dual Titan V + RTX 5060 fleet.*
