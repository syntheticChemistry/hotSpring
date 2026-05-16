# hotSpring -> Phase C Execution Plan: coralReef -> toadStool Cutover

**Date:** May 12, 2026
**Depends on:** GAP-HS-095, GAP-HS-096, PHASE_C_CORAL_DRIVER_SPLIT_PLAN.md
**Status:** Plan ready for upstream ingestion

---

## Context

The dual-existence audit (GAP-HS-096) mapped the exact boundaries between
coralReef's deployed daemon runtime and toadStool's absorbed library types.
This plan specifies the concrete work needed for toadStool to fully replace
the coralReef daemon stack.

## Phase C Work Items (for toadStool upstream)

### C1: Create `toadstool-cylinder` crate (~4k LOC)

Generalize coral-glowplug's `cylinder.rs` into a standalone crate:

| Source (coralReef) | Target (toadStool) |
|--------------------|-------------------|
| `coral-glowplug/src/cylinder.rs` (661 lines) | `crates/core/cylinder/src/lib.rs` |
| `CylinderSpec`, `derive_cylinders()` | `CylinderSpec` with DevicePersonality |
| `EmberChild::spawn()` | Generalized subprocess spawner |
| `run_cylinder_server()` | Per-device Unix socket server |
| `handle_rpc()` with translation | Method dispatch with `device.*` -> ember translation |
| `forward_to_ember()` | Forward-to-subprocess with timeout |
| `orchestrator.rs` (route_rpc_by_bdf) | BDF-based routing table |

Key adaptation: cylinder must work with `toadstool-ember` ResourceHandle
types, not raw coral-ember IPC.

### C2: Absorb coral-driver hardware modules (~15k LOC)

Per the existing PHASE_C_CORAL_DRIVER_SPLIT_PLAN.md:

| Module | LOC est | Notes |
|--------|---------|-------|
| `vfio/device/` | ~2k | VFIO device open, BAR mapping, config space |
| `vfio/dma.rs` | ~500 | DmaBuffer, page-aligned host memory |
| `vfio/pci_discovery/` | ~300 | sysfs PCI scanning under VFIO |
| `vfio/isolation.rs` | ~200 | Fork-isolated BAR0 MMIO |
| `vfio/channel/` | ~3k | GPFIFO, pushbuf, semaphore, glowplug subdir |
| `drm.rs` + `linux_paths.rs` | ~300 | DRM render node enumeration |
| `amd/` | ~2k | GEM, PM4, DRM ioctl, AmdDevice |
| `nv/bar0.rs` + `ioctl/` + `pushbuf.rs` | ~2k | NV hardware access |
| `nv/qmd/` | ~1k | Compute queue descriptor |
| `nv/vfio_compute/` | ~3k | VFIO compute dispatch (the critical path) |
| `hardware.rs` + `error.rs` | ~500 | Device abstraction |

These move into `toadstool-cylinder` or a new `toadstool-hw` crate,
wrapped by `hw-safe` containment.

### C3: Wire daemon RPC surface (~2k LOC)

Add to `toadstool-server` JSON-RPC handler:

| Method | Source | Priority |
|--------|--------|----------|
| `device.swap` | `GlowPlugClient::swap_device_orchestrated` | P0 |
| `device.warm_catch` | New: warm-catch pipeline | P0 |
| `device.health` | BAR0 probe + VRAM check | P1 |
| `device.reset` | SysfsSwapExecutor::release | P1 |
| `ember.swap` | Alias for device.swap | P1 |
| `ember.warm_catch` | Alias for device.warm_catch | P1 |
| `ember.reacquire` | GlowPlugClient::reacquire | P1 |
| `ember.vfio_fds` | SCM_RIGHTS fd passing | P2 |

### C4: VFIO fd holding (~1k LOC)

Evolve `VfioResourceHandle` from metadata-only to real fd owner:

1. `VfioResourceHandle::open(bdf)` -> VFIO group fd + device fd
2. Hold fds across SwapOrchestrator cycles (disable reset_method before swap)
3. Pass fds to dispatch via `ember.vfio_fds` / SCM_RIGHTS
4. `release()` closes fds (current implementation clears metadata only)

### C5: Warm pipeline (~2k LOC)

Port from coralReef:

1. `warm-fecs`: nouveau round-trip with livepatch freeze
2. `warm-catch`: binary-patch nouveau, train memory, preserve state
3. SwapOrchestrator `warm_cycle_performed` -> actually set to true
4. Livepatch sysfs control (enable/disable NOP patches)

### C6: CLI parity (~3k LOC)

Add to `toadstool` CLI:

| Command | Source | Notes |
|---------|--------|-------|
| `toadstool swap <BDF> <TARGET>` | coralctl swap | Direct SwapOrchestrator call |
| `toadstool warm-fecs <BDF>` | coralctl warm-fecs | Nouveau round-trip |
| `toadstool warm-catch <BDF>` | coralctl warm-catch | Full warm-catch pipeline |
| `toadstool status` | coralctl status | Device listing + health |
| `toadstool health <BDF>` | coralctl health | Per-device BAR0 probe |
| `toadstool probe <BDF>` | coralctl probe | Register dump |
| `toadstool mmio read/write` | coralctl mmio | Direct BAR0 access |

Lower priority: snapshot, oracle, journal, experiment, capture.

### C7: systemd service

Create `toadstool.service`:
- ExecStart=/usr/local/bin/toadstool daemon --config /etc/toadstool/devices.toml
- Same capabilities as coral-glowplug.service
- RuntimeDirectory=toadstool (socket at /run/toadstool/toadstool.sock)
- Parallel-run period: both services available, hotSpring validates

## coralReef Soft-Deprecation (post Phase C)

After toadStool daemon passes all hotSpring validation:

1. `coral-ember/Cargo.toml`: add `deprecated` note
2. `coral-glowplug/Cargo.toml`: add `deprecated` note
3. `coral-glowplug.service`: disable, replace with `toadstool.service`
4. `coralctl`: symlink to `toadstool` or keep as shim
5. `coral-driver`: hardware modules marked "moved to toadstool-cylinder"
6. `coral-gpu`: retains compiler (WGSL->SASS), dispatch routing uses toadStool IPC

## hotSpring Validation Checklist

For each Phase C deliverable, hotSpring runs:

- [ ] `validate_vfio_sovereign` — warm open + compile + dispatch + readback
- [ ] `s_vfio_dispatch` scenario — harness integration
- [ ] coralctl parity: `toadstool warm-fecs` produces same Titan V warm state
- [ ] K80 warm-catch: `toadstool warm-catch` with --memory-type gddr5
- [ ] RTX 5060: dispatch via toadStool IPC (Phase D)
- [ ] 590/590 lib tests pass throughout

## Execution Order

1. C1 (cylinder crate) + C2 (driver absorption) — toadStool upstream, parallel
2. C3 (RPC wiring) — depends on C1
3. C4 (VFIO fd holding) — depends on C2
4. C5 (warm pipeline) — depends on C2 + C4
5. C6 (CLI parity) — depends on C3
6. C7 (systemd) — depends on C3 + C5
7. hotSpring validation — continuous, each step
8. coralReef soft-deprecation — after all validation passes

Estimated total: ~27k LOC across toadStool upstream.
Pattern: hotSpring solves locally, hands patterns upstream, primals absorb.
