# Kernel Health Preflight + Exp 216 — hotSpring Handoff

**Date:** May 21, 2026
**From:** hotSpring
**To:** primalSpring (audit), toadStool (upstream code landed)
**Status:** Exp 216 complete, kernel health preflight abstracted, post-fix audit clean

## Summary

Exp 216 identified and abstracted a critical hidden bug: a corrupted `autoconf.h`
in the kernel headers directory (modified May 3 by an out-of-tree build) silently
shifted `struct module` field offsets by 24 bytes, causing every freshly compiled
kernel module to fail with a misleading `Invalid relocation target` error.

This class of bug is **invisible to static analysis** of `.ko` files and only
manifests at runtime during the kernel's in-memory relocation pass.

## What Was Built

### 1. `kernel_health.rs` — 3-Layer Detection (cylinder)

New module `cylinder/src/vfio/kernel_health.rs` with:

- **Layer 1**: `autoconf.h` freshness check (mtime vs kernel image)
- **Layer 2**: Struct layout probe (compile tiny module, read offsets; DKMS fallback)
- **Layer 3**: Reference module cross-check (parse `.gnu.linkonce.this_module` RELA)
- `full_kernel_health_check()` → `KernelHealthReport` (serializable)
- `repair_autoconf()` — package restore from `.deb` cache
- 12 unit tests, all passing. 700 total cylinder tests.

### 2. Integration Points

- **Sovereign handoff preflight** (step 0d): Blocks `Patched`/`DkmsPatched` handoffs if `layout_matches == false`
- **DKMS build guard**: `ensure_build_environment_healthy()` in `kmod.rs`
- **`sovereign.kernel_health` RPC**: Returns full health report JSON, optional repair
- **`toadstool kernel-health` CLI**: Text/JSON output, `--repair` flag

### 3. Post-Fix Audit

Scanned all 20 DKMS `.ko` files and 10 installed modules:

| Status | Count | Details |
|--------|-------|---------|
| Corrupted (exit=0x490) | 6 | `acpi_call` (installed!), `nvidia/470.256.02` (5 files) |
| Clean (exit=0x4a8) | 14 | Pre-corruption or post-fix builds |

All corrupted modules rebuilt. Final scan: **20/20 DKMS OK, 10/10 installed OK**.

### 4. Experiment Impact

**No experiment conclusions invalidated.** Experiments 210-215 used binary-patched
stock `nouveau.ko` (correct offsets from kernel build). The corruption only affected
DKMS-compiled modules (nvidia-470 load failures in Exp 211 timeframe).

## Upstream Gaps for primalSpring Audit

| Gap | Description | Owner |
|-----|-------------|-------|
| Multi-distro repair | `repair_autoconf()` only handles `.deb` packages; needs Fedora RPM, Arch pacman paths | toadStool |
| Immutable system detection | NixOS/Guix: autoconf corruption is structurally impossible; should short-circuit | toadStool |
| Container override | DKMS in containers may have intentional header mismatch; needs `--trust-headers` | toadStool |
| Glowplug HealthProbe | Kernel health not yet wired into `HealthProbe` trait for passive monitoring | toadStool |

## Files Changed

### hotSpring
- `experiments/216_KERNEL_AUTOCONF_MISMATCH_DETECTION.md` — new
- `experiments/README.md` — Exp 216 entry added
- `EXPERIMENT_INDEX.md` — count 215→216, Exp 216 summary
- `README.md` — test count 634→700, RPC 17→19

### toadStool
- `cylinder/src/vfio/kernel_health.rs` — new
- `cylinder/src/vfio/mod.rs` — register `kernel_health` module
- `cylinder/src/vfio/kmod.rs` — `BuildEnvironmentCorrupted` error, `ensure_build_environment_healthy()`
- `cylinder/src/vfio/sovereign_handoff.rs` — step 0d kernel health preflight gate
- `server/src/pure_jsonrpc/handler/sovereign.rs` — `sovereign.kernel_health` RPC
- `server/src/pure_jsonrpc/handler/mod.rs` — route registration
- `cli/src/commands/kernel_health.rs` — new
- `cli/src/commands/definitions.rs` — `KernelHealth` subcommand
- `cli/src/commands/dispatch/mod.rs` — dispatch wiring
- `cli/src/commands/mod.rs` — module registration
- `cli/Cargo.toml` — `toadstool-cylinder` dependency

## Archive Candidates

The two active handoffs in `wateringHole/handoffs/` from May 19-20 can now be
archived — their work is complete and documented in experiments 210-215:

- `HOTSPRING_GPC_BOUNDARY_CE_VALIDATE_HANDOFF_MAY19_2026.md` → archive
- `HOTSPRING_SOVEREIGN_DRIVER_ROTATION_EXP211_COMPLETE_MAY20_2026.md` → archive
