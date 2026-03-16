# hotSpring v0.6.32 — Compute Trio Rewire Handoff

**Date**: March 13, 2026
**From**: hotSpring
**To**: coralReef, toadStool, barraCuda (informational)
**License**: AGPL-3.0-only
**Covers**: hotSpring v0.6.31 → v0.6.32

---

## Executive Summary

- **Trio pins synced**: barraCuda `82ff983` → `b95e9c59` (v0.3.5), coralReef Iter 37 → Iter 47 (`ff54331`), toadStool S147 → S156 (`ebe6a7cc`)
- **Stale API purge**: `CoralReefDevice::from_descriptor()` removed upstream — all 3 call sites rewired to auto-discovery (`with_auto_device()`)
- **Sovereign compile binary rewritten**: `validate_sovereign_compile` now uses `CoralCompiler` IPC client instead of removed direct compilation API
- **Zero debt**: 848 lib tests pass, 0 clippy warnings (--all-features), 0 TODOs/FIXMEs in src
- **hotSpring remains pinned** — awaiting coralReef and toadStool deliverables before Exp 070

---

## What Changed

### Stale API Cleanup

barraCuda evolved `CoralReefDevice` between `82ff983` and `b95e9c59`:
- Removed `from_descriptor(vendor, arch, driver)` in favor of internal auto-discovery
- Compilation now goes through `CoralCompiler` IPC (JSON-RPC) rather than direct in-process calls

**Files fixed:**

| File | Old API | New API |
|------|---------|---------|
| `src/bench/md_backend.rs` | 4-strategy `from_descriptor` loop | Single `with_auto_device()` |
| `src/bin/bench_sovereign_dispatch.rs` | 5-strategy `from_descriptor` loop | Single `with_auto_device()` |
| `src/bin/validate_sovereign_compile.rs` | `CoralReefDevice::new(GpuTarget)` + `dev.compile_wgsl()` | `GLOBAL_CORAL.compile_wgsl_direct()` async IPC |

### Other Cleanups

- Removed unused `.cargo/config.toml` patches for `coral-gpu`, `coral-reef`, `coral-driver` (compilation is IPC-only now)
- Gated `NPU_DEVICE_DIRS`/`NPU_DEVICE_PREFIXES` behind `#[cfg(not(feature = "npu-hw"))]`
- Fixed redundant closure in `pipeline_eval.rs`
- `cargo fmt` applied across all bins

---

## Validation

| Target | Tests | Clippy | Fmt |
|--------|-------|--------|-----|
| `hotspring-barracuda` lib | 848 pass, 0 fail | 0 warnings (--all-features) | clean |
| `hotspring-barracuda` bins | all compile | clean | clean |
| `hotspring-forge` (metalForge) | 25 pass, 0 fail | 1 nursery note (example) | clean |

---

## Trio State at Pin

| Primal | Version | Commit | Key Capability |
|--------|---------|--------|----------------|
| barraCuda | v0.3.5 | `b95e9c59` | 806 WGSL shaders, zero-copy BytesMut, IPC-first CoralReefDevice, pedantic deny |
| coralReef | Iter 47 | `ff54331` | GlowPlug boot-persistent daemon, FECS direct execution, DRM fencing |
| toadStool | S156 | `ebe6a7cc` | Specialty resurrection, standards compliance, 20k+ tests |

---

## What hotSpring Needs Before Unpinning

### From coralReef (7 tasks)

1. **coral-glowplug JSON-RPC protocol**: `glowplug.device.list`, `.health`, `.swap`, `.resurrect`, `.status`, `.shutdown`
2. **SCM_RIGHTS fd passing**: VFIO container fds to toadStool
3. **Trait-based personality system**: No enum changes for new vendors
4. **AMD Vega metal** (`amd_metal.rs`): SMC, GRBM, UMC, GFX, registers, power-on for MI50/GFX906
5. **GP_PUT DMA read**: Complete Exp 058 USERD_TARGET fix
6. **DRM consumer fencing**: `lsof /dev/dri/renderD*` before nouveau bind
7. **Privilege model**: Replace `sudo tee` with `CAP_SYS_ADMIN` + polkit

### From toadStool (3 tasks)

1. **GlowPlug socket client**: `GlowPlugClient` crate — `connect`, `list_devices`, `health`, `swap`, `resurrect`
2. **VFIO device in sysmon**: Detect vfio-pci, IOMMU group, `sovereign: true`
3. **hw-learn GlowPlug health feed**: VRAM/power/domain into learning pipeline

### From barraCuda

- **None** — IPC-first design works cleanly. hotSpring's 848 tests confirm API stability.

---

## Superseded Handoffs (archived)

- `HOTSPRING_V0631_GAP_CLOSURE_REWIRE_HANDOFF_MAR12_2026.md` → `archive/`
