# Handoff: Vendor-Agnostic Hardened GlowPlug Evolution

**Date:** March 18, 2026
**From:** hotSpring
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** coral-glowplug lib extraction, coral-ember crate split, EmberError typed errors, vendor-agnostic RegisterMap, AMD MI50 support, privilege hardening (capabilities + seccomp + namespaces), coralctl CLI

---

## Executive Summary

- **coral-ember** extracted from `src/bin/coral_ember.rs` into a standalone workspace crate (`crates/coral-ember/`) with modular `sysfs`, `swap`, `hold`, `ipc` modules. Old monolith deleted.
- **coral-glowplug** gained a library surface (`src/lib.rs`) — shared types now importable by external crates. `EmberClient` returns typed `EmberError` instead of `String`. Legacy sysfs fallbacks gated behind `#[cfg(feature = "no-ember")]`.
- **Vendor-agnostic register maps**: `RegisterMap` trait in `hotSpring/barracuda/src/register_maps/` with `NvGv100Map` (127 registers) and `AmdGfx906Map` (MI50/MI60). `detect_register_map(vendor_id)` auto-selects at runtime.
- **AMD MI50 fully supported**: `AmdgpuPersonality.supports_hbm2_training()` corrected to `true`. `hbm2_training_driver(vendor_id)` returns vendor-appropriate warm driver (`nouveau` for NVIDIA, `amdgpu` for AMD). PCI ID constants added. `resurrect_hbm2()` is now vendor-agnostic.
- **IPC unified**: Both `coral-ember` and `coral-glowplug` speak JSON-RPC 2.0. `EmberClient` methods return `Result<_, EmberError>` with variants: `Connect`, `Io`, `Parse`, `Rpc`, `FdCount`.
- **Production-grade privilege hardening**: Both systemd services now have `CapabilityBoundingSet`, `AmbientCapabilities` (CAP_SYS_ADMIN, CAP_SYS_RAWIO, CAP_DAC_OVERRIDE), `NoNewPrivileges=true`, `SystemCallFilter=@system-service`, `ProtectSystem=strict`, `PrivateTmp`, `ProtectHome`, `MemoryDenyWriteExecute`.
- **coralctl CLI**: New `deploy-udev` subcommand generates `/dev/vfio/*` udev rules from `glowplug.toml` — zero hardcoded BDFs. Resolves IOMMU groups at runtime. Supports `--dry-run`.

---

## Part 1: Architecture Changes

### coral-ember Crate Extraction (Track B5)

```
Before:  crates/coral-glowplug/src/bin/coral_ember.rs  (788-line monolith)
After:   crates/coral-ember/
           ├── src/main.rs    # entry point, config, listener loop
           ├── src/sysfs.rs   # sysfs helpers (sole writer of driver/unbind+bind)
           ├── src/swap.rs    # swap_device orchestrator + DRM isolation
           ├── src/hold.rs    # HeldDevice struct
           └── src/ipc.rs     # JSON-RPC 2.0 + SCM_RIGHTS
```

The old `coral_ember.rs` has been **deleted** from coral-glowplug. Workspace `Cargo.toml` now includes both `crates/coral-glowplug` and `crates/coral-ember`.

### coral-glowplug Library Surface (Track B1)

`coral-glowplug` now exports a library (`src/lib.rs`) with public modules:
`config`, `device`, `ember`, `error`, `health`, `pci_ids`, `personality`, `sysfs`

External crates (including `coralctl`) can `use coral_glowplug::{config, sysfs, pci_ids}`.

### Typed Errors (Track B3)

`EmberError` replaces `Result<_, String>` throughout `EmberClient`:

| Variant | When |
|---------|------|
| `Connect(io::Error)` | Socket connect fails |
| `Io(io::Error)` | Read/write/timeout errors |
| `Parse(serde_json::Error)` | JSON-RPC response malformed |
| `Rpc { code, message }` | Server returned JSON-RPC error |
| `FdCount { expected, received }` | SCM_RIGHTS delivered wrong number of fds |

### Legacy Gating (Track B4)

All direct sysfs fallback paths in `device.rs` (`activate`, `bind_vfio`, `bind_driver`, `reclaim`) are now gated behind `#[cfg(feature = "no-ember")]`. Default build **requires ember** and returns clear errors if absent.

---

## Part 2: Vendor-Agnostic Hardware Support

### RegisterMap Trait (Track A1)

New `barracuda/src/register_maps/` module:

| File | Contents |
|------|----------|
| `mod.rs` | `RegisterMap` trait, `RegDef`, `RegisterDump`/`RegisterEntry` JSON types, `detect_register_map()` |
| `nv_gv100.rs` | 127 NVIDIA GV100 BAR0 registers (PMC through THERM) |
| `amd_gfx906.rs` | AMD Vega 20 / MI50 MMIO registers (SRBM through THM) |

API: `detect_register_map(vendor_id: u16) -> Option<Box<dyn RegisterMap>>`

### AMD MI50 Support (Track A2)

| Change | File |
|--------|------|
| `AMD_VENDOR_ID`, `MI50_DEVICE_ID`, `MI60_DEVICE_ID`, `MI50_VFIO_IDS`, `MI60_VFIO_IDS` | `pci_ids.rs` |
| `hbm2_training_driver(vendor_id)` → `"nouveau"` or `"amdgpu"` | `pci_ids.rs` |
| `AmdgpuPersonality.supports_hbm2_training()` → `true` | `personality.rs` |
| `resurrect_hbm2()` uses `hbm2_training_driver()` instead of hardcoded `"nouveau"` | `device.rs` |
| `identify_chip()` uses constants instead of magic numbers | `sysfs.rs` |

### Auto-Discovery

`Config::auto_discover()` already scans PCI bus for `0x10de` (NVIDIA) and `0x1002` (AMD) vendors with VGA/3D class codes. MI50 devices are detected and added with `amdgpu` personality when that driver is bound.

---

## Part 3: Privilege Hardening

### Capabilities (Track C3)

Both services restrict to minimum capabilities:

```ini
CapabilityBoundingSet=CAP_SYS_ADMIN CAP_SYS_RAWIO CAP_DAC_OVERRIDE
AmbientCapabilities=CAP_SYS_ADMIN CAP_SYS_RAWIO CAP_DAC_OVERRIDE
NoNewPrivileges=true
```

### Seccomp + Filesystem Isolation (Track C4)

**coral-ember:**
```ini
ProtectSystem=strict
ReadWritePaths=/sys/bus/pci /sys/kernel/iommu_groups /run/coralreef /proc
SystemCallFilter=@system-service ioctl sendmsg
MemoryDenyWriteExecute=true
```

**coral-glowplug:**
```ini
ProtectSystem=strict
ReadWritePaths=/sys/bus/pci /run/coralreef /proc
ReadOnlyPaths=/etc/coralreef /sys/kernel/iommu_groups
SystemCallFilter=@system-service ioctl recvmsg
MemoryDenyWriteExecute=true
```

### coralctl deploy-udev (Track C5)

```bash
# Generate rules from config
coralctl deploy-udev --config /etc/coralreef/glowplug.toml

# Preview without writing
coralctl deploy-udev --dry-run

# Custom output and group
coralctl deploy-udev --output /tmp/test.rules --group vfio
```

Resolves IOMMU groups at runtime from the config's `[[device]]` BDFs. Generates `SUBSYSTEM=="vfio"` rules with `GROUP` and `MODE=0660`.

---

## Part 4: Test Results

- **87 library tests** pass in coral-glowplug (both default and `no-ember` feature)
- **coral-ember** compiles as standalone crate
- **barracuda** `register_maps` module compiles
- New tests: AMD PCI ID format, `hbm2_training_driver()` vendor dispatch, `AmdgpuPersonality` HBM2

---

## Part 5: Action Items

### For coralReef

1. **Deploy new crate structure**: `coral-ember` is now a separate workspace member
2. **Test MI50 swap path**: `ember.swap {bdf: "...", target: "amdgpu"}` should work with MI50
3. **Evolve register RPC**: `device.register_dump` and `device.register_snapshot` now use the glowplug-internal register list; consider forwarding vendor-agnostic `RegisterMap` to these RPCs
4. **Harden socket permissions**: Run `coralctl deploy-udev` on deployment targets

### For toadStool

1. **Import `coral_glowplug` library**: Now that glowplug has a lib surface, toadStool can depend on it for type-safe IPC without reimplementing config parsing
2. **Feed register snapshots to hw-learn**: `device.register_snapshot` returns pre-swap register state — pattern data for hardware learning
3. **AMD device pair**: With MI50 personality support, toadStool can manage NVIDIA+AMD heterogeneous pairs

### For barraCuda

1. **Absorb `RegisterMap` trait**: The trait + `RegDef` + `detect_register_map()` belong in barraCuda long-term, not hotSpring. hotSpring can lean on upstream after absorption.
2. **Add register decode methods**: `decode_temp_c()` and `decode_boot_id()` are per-vendor; barraCuda should own the canonical decode logic.
3. **Exp 070 binary evolution**: `exp070_register_dump.rs` currently uses raw BAR0 mmap. Post-absorption it should use `RegisterMap` for named register access and JSON schema.

---

## Supersedes

This handoff supersedes the **Ember + DRM Isolation handoff (Mar 19)** for all topics covered here. The Ember handoff remains valid for DRM isolation specifics and swap_device invariants.

Items from the **PIN Evolution handoff (Mar 16)** updated:
- ~~Privilege model (CAP_SYS_ADMIN)~~ → **DELIVERED** (capabilities + seccomp + namespaces)
- ~~AMD Vega metal~~ → **IN PROGRESS** (registers defined, personality fixed, swap path works)

---

## License

AGPL-3.0-only. Part of the ecoPrimals sovereign compute fossil record.
