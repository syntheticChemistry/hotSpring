# Handoff: Compute Trio Evolution — AMD D3cold Resolution, BrainChip Akida NPU, Vendor-Agnostic Triangle Architecture

**Date:** March 20, 2026
**From:** hotSpring
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** AMD Vega 20 lifecycle resolution (4 strategies tested), BrainChip AKD1000 NPU integration, `amdgpu.runpm=0` kernel parameter, `stabilize_after_bind()` hook, `PmResetAndBind` strategy, zero-sudo `coralctl`, triangle architecture for trio evolution

---

## Executive Summary

Four boot cycles of empirical testing against the AMD Radeon VII (Vega 20 / GFX906) established a **hardware-level firmware limitation**: the Vega 20 SMU cannot survive more than one vfio→amdgpu round-trip per boot. This is not a software bug — it is a property of the AMD SMU mailbox firmware on Vega 20 silicon. The first round-trip works reliably; the second corrupts SMU state regardless of the reset strategy used.

Separately, the BrainChip AKD1000 Akida neuromorphic NPU was integrated into the GlowPlug lifecycle with zero issues — unlimited round-trips, simple PCIe semantics, no vendor quirks.

The trio (coralReef + toadStool + barraCuda) should now evolve toward a **fully vendor-agnostic triangle architecture** where:
- **coralReef** provides GlowPlug (lifecycle broker) + compiler to toadStool
- **toadStool** provides hardware resources + dispatch to barraCuda
- **barraCuda** does the math, routing through toadStool for compilation and onto hardware via coralReef

---

## Part 1: AMD Vega 20 — Definitive Analysis

### Strategies Tested (4 boot cycles)

| # | Strategy | Cycle 1 | Cycle 2 | Failure Mode |
|---|----------|---------|---------|--------------|
| 1 | Simple bind + stabilize_after_bind | ✅ pass | ❌ D3cold | amdgpu probe hangs on corrupted SMU |
| 2 | Simple bind + power/control=on | ✅ pass | ❌ D3cold | Runtime PM not the cause |
| 3 | PCI remove + rescan | ✅ pass | ❌ device lost | AMD internal bridges power off slot |
| 4 | PM power cycle (D3hot→D0) + bind | ✅ pass | ❌ D3cold | SMU state unrecoverable after 1st cycle |

### Root Cause

The Vega 20 SMU (System Management Unit) firmware has a **one-shot reinitialization** property:
- From BIOS/POST state → amdgpu probe → clean SMU init ✅
- From amdgpu shutdown → vfio-pci ownership → amdgpu reprobe → SMU init ✅ (first time)
- From 2nd amdgpu shutdown → vfio-pci → amdgpu reprobe → SMU mailbox corruption ❌

The `trn=2 ACK should not assert` kernel error is the SMU's training mailbox failing — the firmware cannot recover its internal state machine after the second transition.

### Mitigations Deployed (all remain in production)

| Mitigation | Layer | Purpose |
|-----------|-------|---------|
| `amdgpu.runpm=0` on kernel cmdline | Boot | Prevents runtime PM from entering D3 between swaps |
| `ExecStartPre` in coral-ember.service | Systemd | Clears `reset_method` + pins power before ember starts |
| `stabilize_after_bind()` trait method | coral-ember | Re-pins power/bridge after every driver bind |
| `reset_method=""` in `prepare_for_unbind()` | coral-ember | Prevents vfio-pci bus reset on fd close |
| `pin_bridge_power()` | coral-ember | Prevents upstream PCIe bridges from power-gating |
| `PmResetAndBind` strategy | coral-ember | PM power cycle before native driver rebind |
| `coralreef-amd-shutdown.service` | Systemd | Shutdown guard: unbinds AMD from amdgpu, parks on vfio-pci before kernel shutdown hooks |
| `/usr/local/bin/coralreef-amd-shutdown-guard` | Shutdown | Timeout-wrapped unbind + vfio-pci park for each AMD device |

### Shutdown Guard (Critical)

The amdgpu driver's `.shutdown()` callback tries to send SMU power-down messages. If
the SMU is in ANY degraded state (including after a round-trip, or even during normal
shutdown on some firmware versions), these messages loop forever — `trn=2 ACK should not
assert` repeated thousands of times, blocking kernel shutdown.

**Fix: `coralreef-amd-shutdown.service`** — a systemd oneshot that runs `Before=shutdown.target`.
On ExecStop, it:
1. Checks each AMD device's `power_state`
2. If D0 (healthy): unbinds from amdgpu (timeout 8s), parks on vfio-pci
3. If D3cold (dead): skips (nothing userspace can do)
4. Service has `TimeoutStopSec=15` — systemd proceeds regardless after 15s

vfio-pci's shutdown path is silent — no SMU communication, no hang.

### Practical Guidance

- **AMD Vega 20**: One vfio round-trip per boot is reliable. Plan workloads accordingly.
- **AMD RDNA** (untested): Uses same conservative strategy. May behave differently — needs empirical validation.
- **Clean shutdown**: Shutdown guard parks AMD on vfio-pci automatically. `amdgpu.runpm=0` prevents runtime PM drift.

---

## Part 2: BrainChip AKD1000 Akida NPU Integration

### Hardware Profile

| Property | Value |
|----------|-------|
| Vendor ID | `0x1e7c` |
| Device ID | `0xbca1` |
| PCI Class | Co-processor (0b40) |
| Driver | `akida-pcie` (kernel module `akida_pcie`) |
| BARs | 3 × 4MB (64-bit prefetchable) |
| IOMMU Group | Solo (group 33) |

### Changes Made

| Crate | File | Change |
|-------|------|--------|
| coral-glowplug | `pci_ids.rs` | `BRAINCHIP_VENDOR_ID = 0x1e7c`, `AKD1000_DEVICE_ID = 0xbca1` |
| coral-glowplug | `personality.rs` | `AkidaPersonality`, `Personality::Akida`, registry entry |
| coral-glowplug | `device.rs` | `akida-pcie` match arms in activate/swap/bind_driver |
| coral-ember | `vendor_lifecycle.rs` | `BrainChipLifecycle` (SimpleBind, 3s settle, basic health) |
| coral-ember | `swap.rs` | `akida-pcie` in bind_native match, DRM isolation skip for non-GPU |

### Test Results

- `akida-pcie → vfio-pci → akida-pcie`: clean round-trip, unlimited cycles
- No DRM isolation needed (co-processor, not GPU)
- SimpleBind strategy works perfectly

### Pattern for Future Non-GPU Accelerators

The Akida integration proves GlowPlug works for **any PCIe device**, not just GPUs:
1. Add vendor/device IDs to `pci_ids.rs`
2. Add personality struct (no DRM card path needed)
3. Add lifecycle (SimpleBind defaults are safe)
4. Skip DRM isolation check for non-DRM drivers

This pattern applies to: FPGAs, TPUs, SmartNICs, DSPs, crypto accelerators.

---

## Part 3: VendorLifecycle Trait — Final State

### Trait Definition (updated)

```rust
pub trait VendorLifecycle: Send + Sync + fmt::Debug {
    fn description(&self) -> &str;
    fn prepare_for_unbind(&self, bdf: &str, current_driver: &str) -> Result<(), String>;
    fn rebind_strategy(&self, target_driver: &str) -> RebindStrategy;
    fn settle_secs(&self, target_driver: &str) -> u64;
    fn stabilize_after_bind(&self, bdf: &str, target_driver: &str);  // NEW
    fn verify_health(&self, bdf: &str, target_driver: &str) -> Result<(), String>;
}

pub enum RebindStrategy {
    SimpleBind,
    SimpleWithRescanFallback,
    PciRescan,              // WARNING: does not work for AMD Vega 20
    PmResetAndBind,         // NEW: PM power cycle before bind
}
```

### Implementation Matrix

| Lifecycle | Vendor | vfio→native | native→vfio | Post-bind |
|-----------|--------|-------------|-------------|-----------|
| NvidiaLifecycle | 0x10de | SimpleBind | SimpleBind | pin_power |
| AmdVega20Lifecycle | 0x1002 (Vega 20) | PmResetAndBind | SimpleBind | pin_power + bridge + reset_method + autosuspend |
| AmdRdnaLifecycle | 0x1002 (other) | PmResetAndBind | SimpleBind | same as Vega 20 |
| IntelXeLifecycle | 0x8086 | SimpleBind | SimpleBind | pin_power |
| BrainChipLifecycle | 0x1e7c | SimpleBind | SimpleBind | pin_power |
| GenericLifecycle | other | SimpleWithRescanFallback | SimpleBind | pin_power |

---

## Part 4: Zero-Sudo coralctl

### Architecture

```
User (biomegate) ──coralctl──► /run/coralreef/glowplug.sock (root:coralreef 0660)
                                    │
                                    ▼
                              coral-glowplug (root, systemd)
                                    │
                                    ▼ ember.sock (root:root 0600)
                              coral-ember (root, systemd, VFIO fds)
```

- `glowplug.sock`: user-facing, group `coralreef`, mode 0660
- `ember.sock`: service-to-service only, root:root 0600
- User joins `coralreef` group → full `coralctl` access, zero privilege escalation
- Commands: `coralctl status`, `coralctl swap <bdf> <target>`, `coralctl health`

---

## Part 5: Triangle Architecture — Trio Evolution Path

### Current State

```
                    coralReef
                   (GlowPlug + Compiler)
                  /                      \
                 /                        \
        toadStool ─────────────────── barraCuda
     (HW Resources + Dispatch)      (Math + Shaders)
```

### What Each Primal Owns

| Primal | Responsibility | Key Interfaces |
|--------|---------------|----------------|
| **coralReef** | PCIe lifecycle (GlowPlug/Ember), WGSL→SPIR-V compilation, VFIO fd management | `glowplug.sock` JSON-RPC, `ember.sock` SCM_RIGHTS |
| **toadStool** | Hardware discovery, GPU dispatch, algorithm selection, hw-learn telemetry | GlowPlug socket client, GpuDriverProfile, WgslOptimizer |
| **barraCuda** | Precision math (DF64, f64), physics kernels, shader templates | ShaderTemplate → toadStool compilation → coralReef hardware |

### Evolution Priorities for Vendor-Agnostic Triangle

#### For coralReef (immediate)
1. **Ember per-device isolation**: Move sysfs operations to per-device threads so one D3cold device doesn't freeze all operations
2. **D3cold pre-check**: Before any sysfs write, read `power_state` — if D3cold, return error immediately instead of hanging
3. **VendorLifecycle → VendorProfile**: Merge with RegisterMap trait for unified vendor dispatch
4. **RDNA validation**: Test RX 5000/6000/7000 series

#### For toadStool (immediate)
1. **GlowPlug socket client**: Connect to `glowplug.sock`, use `device.list`/`device.swap`/`health.check` RPCs
2. **Lifecycle-aware dispatch**: Know that AMD round-trips are expensive (one per boot); prefer keeping AMD on one personality
3. **Hardware census via GlowPlug**: Replace manual BDF enumeration with RPC-based device discovery

#### For barraCuda (informational)
1. **RegisterMap + VendorLifecycle convergence**: Both dispatch from PCI IDs — consider unified `VendorProfile`
2. **Shader ISA metadata**: The lifecycle knows what hardware is present; barraCuda should query capabilities through toadStool

### Triangle Data Flow (Target State)

```
barraCuda: "I need DF64 matrix multiply on 16GB HBM2"
    │
    ▼
toadStool: "Radeon VII available (amdgpu), compiling shader..."
    │
    ├──► coralReef compiler: WGSL → SPIR-V (vendor-optimized)
    │
    ├──► coralReef GlowPlug: "swap 4d:00.0 to vfio" (if needed)
    │
    └──► toadStool dispatch: submit to hardware, collect results
    │
    ▼
barraCuda: receives results, validates physics
```

---

## Part 6: Hardware Configuration (biomeGate, March 20, 2026)

| Slot | Device | BDF | Role | IOMMU | Boot Driver |
|------|--------|-----|------|-------|-------------|
| 1 | RTX 5060 (GB206) | varies | Display head + simple CUDA | — | nvidia |
| 2 | Titan V (GV100) | 0000:03:00.0 | VFIO oracle, HBM2 compute | 73 (solo) | vfio-pci |
| 3 | Radeon VII (Vega 20) | 0000:4d:00.0 | AMD compute, 16GB HBM2 | 38 (solo) | amdgpu |
| 4 | BrainChip AKD1000 | 0000:45:00.0 | Neuromorphic NPU inference | 33 (solo) | akida-pcie |

**Kernel cmdline**: `nvidia-drm.modeset=1 vfio-pci.ids=10de:1d81 amdgpu.runpm=0`

---

## Part 7: Key Files Changed

| File | Change |
|------|--------|
| `coralReef/crates/coral-ember/src/vendor_lifecycle.rs` | `stabilize_after_bind()`, `PmResetAndBind`, `BrainChipLifecycle`, `BRAINCHIP_VENDOR` |
| `coralReef/crates/coral-ember/src/swap.rs` | `pci_remove_rescan()`, `PmResetAndBind` implementation, `is_drm_driver()` gate |
| `coralReef/crates/coral-ember/src/sysfs.rs` | `pm_power_cycle()` helper |
| `coralReef/crates/coral-glowplug/src/pci_ids.rs` | `BRAINCHIP_VENDOR_ID`, `AKD1000_DEVICE_ID` |
| `coralReef/crates/coral-glowplug/src/personality.rs` | `AkidaPersonality`, `Personality::Akida` |
| `coralReef/crates/coral-glowplug/src/device.rs` | `akida-pcie` match arms |
| `/etc/modprobe.d/coralreef-amd.conf` | `options amdgpu runpm=0` |
| `/etc/systemd/system/coralreef-amd-shutdown.service` | Shutdown guard: parks AMD on vfio-pci before kernel shutdown |
| `/usr/local/bin/coralreef-amd-shutdown-guard` | Timeout-wrapped AMD unbind + vfio-pci park script |
| `/boot/efi/loader/entries/Pop_OS-current.conf` | `amdgpu.runpm=0` on kernel cmdline |
| `/etc/coralreef/glowplug.toml` | 3 devices: titan-v, radeon-vii (4d:00.0), akida-npu |

---

## Part 8: Test Counts

- **coral-ember**: 6 unit tests (VendorLifecycle strategies, IDs, settle times)
- **coral-glowplug**: 88 unit tests (personality, config, device, sysfs, socket)
- **coral-glowplug integration**: 63 tests (JSON-RPC, chaos, fuzz, TCP)
- **Total**: 157 tests, all passing
- **Live validation**: Titan V unlimited round-trips, Radeon VII 1 round-trip, Akida unlimited round-trips

---

## Supersedes

Updates the **VendorLifecycle AMD D3cold handoff (Mar 19)** with:
- PCI remove/rescan: ~~recommended for AMD~~ → **does not work** (bridge powers off)
- PM power cycle: documented as strategy, does not solve multi-cycle limitation
- AMD Vega 20: **one round-trip per boot** is the definitive hardware limitation
- BrainChip Akida: **new** — fully integrated, unlimited round-trips
- `stabilize_after_bind()`: **new** trait method for post-bind power pinning
- Zero-sudo coralctl: **new** — documented socket permissions architecture

---

## License

AGPL-3.0-only. Part of the ecoPrimals sovereign compute fossil record.
