<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Sovereign Boot Abstraction + Profiling Handoff

**Date**: 2026-05-18
**Experiment**: 207
**Hardware**: Dual Titan V (GV100), RTX 5060
**toadStool**: uncommitted (cylinder boot_state + sovereign_profile, ember warm_keepalive)

## What Was Done

Four-phase implementation formalizing the warm/cold boot model and building
an experimentation framework:

### Phase 1 — Unified Boot State (`cylinder::vfio::boot_state`)

The `SovereignBootState` enum replaces scattered warm/cold signals with a
single source of truth. `probe_boot_state()` reads PMC_ENABLE, tests PRAMIN
sentinel, and optionally probes falcon state.

- `SovereignBootState::Warm { pmc_popcount, pramin_ok, falcon }`
- `SovereignBootState::Cold { reason, pmc_enable }`
- `ColdBootReason`: `PowerOnReset | BusReset | D3Cold | FdLost | Unknown`
- `BootCapability`: flags for what warm vs cold allows

### Phase 2 — WarmKeepalive Facade (`ember::warm_keepalive`)

`WarmKeepalive` wraps `VfioAnchor` with lifecycle semantics. The dispatch
handler uses `WarmKeepaliveRef` (non-owning) for clutch engagement.
`DmaSpec` bridges ember's `AnchorBackendRef` to cylinder's `DmaBackend`.

### Phase 3 — Profiling Framework (`cylinder::vfio::sovereign_profile`)

`sovereign.profile` JSON-RPC method returns `SovereignProfile`:
per-stage microsecond timings, register snapshots (BOOT0/PMC/PTIMER/FECS/GPCCS)
at pre- and post-pipeline checkpoints.

### Phase 4 — Twin-Card Experiments

Both Titan Vs profiled in cold state after service restart. Results confirm
the hardware line: HBM2 untrained despite engines enabled, only full power
cycle recovers. Card-to-card memory training time variance observed (5.4s
vs 10.5s) — likely thermal/HBM2 lot variability.

## Hardware Line (Codified)

Cold boot = power-on reset = board-level VR sequencing = boot ROM trains
HBM2 = same wall NVIDIA faces. Software cannot train HBM2.

**What software can do**: prevent transition to cold via VfioAnchor +
systemd FileDescriptorStore (fd persistence across daemon restarts).
Keepalive = never let go of VFIO fds.

## Key Types for Upstream Teams

| Type | Crate | Role |
|------|-------|------|
| `SovereignBootState` | cylinder | Warm/Cold enum with diagnostics |
| `ColdBootReason` | cylinder | Why the GPU is cold |
| `BootCapability` | cylinder | What the current state allows |
| `probe_boot_state()` | cylinder | Authoritative detection function |
| `SovereignProfile` | cylinder | Per-stage timing + register snapshots |
| `WarmKeepalive` | ember | Owns VfioAnchor, lifecycle facade |
| `WarmKeepaliveRef` | ember | Non-owning borrow for handlers |
| `DmaSpec` | ember | AnchorBackendRef → DmaBackend bridge |

## Deployment Notes

- The `toadstool` binary lives in `crates/cli` (`toadstool-cli`), not
  `crates/server`. Build with `cargo build --release -p toadstool-cli`.
- `sovereign.profile` requires the cylinder `vfio` feature.

## Gaps for Upstream

| Gap | Owner | Priority |
|-----|-------|----------|
| `WarmKeepalive` not yet wired to systemd fd store send/receive | toadStool | High |
| `ColdBootReason` heuristics for `BusReset` vs `D3Cold` need PCI link-state probing | toadStool | Medium |
| `SovereignProfile` stage names are string-based; could be an enum | toadStool | Low |
| AMD `probe_boot_state` equivalent not implemented | toadStool | Future |
| `sovereign.profile` not yet tested on Kepler (GK210) | hotSpring | Next HW expansion |
