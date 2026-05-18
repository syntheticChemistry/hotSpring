<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Experiment 207 — Sovereign Boot Abstraction + Twin-Card Profiling

**Date**: 2026-05-18
**Hardware**: Dual Titan V (GV100, 0000:02:00.0 + 0000:49:00.0)
**Status**: ✅ Complete

## Objective

Formalize the warm/cold boot model, build a profiling instrument, and run
twin-card experiments to characterize boot state transitions on identical
hardware.

## Context

After Exp 205 (twin-card baseline) and Exp 206 (Falcon ACR DMA), the
warm/cold detection was scattered across `is_warm_gpu()`, `FalconWarmState`,
and `warm_detected` booleans. This experiment replaces all of that with a
single unified abstraction and adds a profiling RPC for rapid experimentation.

## Architecture Changes (toadStool)

### Phase 1 — Boot State Abstraction (`cylinder::vfio::boot_state`)

- `SovereignBootState` enum: `Warm { pmc_popcount, pramin_ok, falcon }` /
  `Cold { reason, pmc_enable }`.
- `ColdBootReason`: `PowerOnReset | BusReset | D3Cold | FdLost | Unknown`.
- `BootCapability` flags: `skip_memory_training`, `acr_boot_available`.
- `probe_boot_state()`: reads PMC_ENABLE, tests PRAMIN sentinel,
  optionally probes falcon — returns `SovereignBootState`.
- **Hardware line documented**: Cold boot = boot ROM must train HBM2 on
  power-on reset. This is the same wall NVIDIA's own driver faces.

### Phase 2 — WarmKeepalive Facade (`ember::warm_keepalive`)

- `WarmKeepalive`: owns a `VfioAnchor`, exposes `device_fd()`,
  `backend_ref()`, `ioas_id()`, `leak()`.
- `WarmKeepaliveRef<'a>`: non-owning view for dispatch handlers.
- `DmaSpec`: bridges `AnchorBackendRef` → `DmaBackend`.
- `KeepaliveStore`: shared `HashMap<String, Arc<WarmKeepalive>>`.

### Phase 3 — Profiling Framework (`cylinder::vfio::sovereign_profile`)

- `SovereignProfile`: wraps `SovereignInitResult` + stage timings + snapshots.
- `StageTimingUs`: per-stage microsecond precision with fraction-of-total.
- `RegisterSnapshot`: BOOT0, PMC_ENABLE, PTIMER, FECS, GPCCS at
  pre-pipeline and post-pipeline checkpoints.
- `sovereign.profile` JSON-RPC method exposed in toadStool server.

### Phase 4 — Pipeline Integration

- `sovereign_init()` calls `probe_boot_state()` at pipeline start.
- `SovereignInitResult` gains `boot_state: Option<SovereignBootState>`.
- Dispatch handler uses `WarmKeepaliveRef` for clutch engagement.

## Twin-Card Cold Profiling Results

Both cards tested after `systemctl stop toadstool && systemctl start toadstool`
(service restart, no power cycle — GPUs entered cold state).

### Titan V #1 (0000:02:00.0)

| Stage | Duration |
|-------|----------|
| boot_state_probe | <1ms |
| devinit | 5419ms |
| engine_enable | 2ms |
| falcon_boot | 3697ms |
| gr_init | 86ms |
| total pipeline | 11.3s |

- Boot state: **Cold (Unknown)** — PMC_ENABLE=0x5fecdff1 (popcount>16
  but PRAMIN sentinel test failed → memory untrained).

### Titan V #2 (0000:49:00.0)

| Stage | Duration |
|-------|----------|
| boot_state_probe | <1ms |
| devinit | 10537ms |
| engine_enable | 2ms |
| falcon_boot | 224ms |
| gr_init | 43ms |
| total pipeline | 13.0s |

- Boot state: **Cold (Unknown)** — same PMC pattern, PRAMIN untrained.
- Card #2 memory training 2× slower (different thermal history / HBM2 lot
  variability on identical GPUs — a benefit of twin-study methodology).

## Key Finding: Hardware Line Confirmed

Even with engines enabled (high PMC_ENABLE popcount), untrained HBM2 means
the GPU is non-functional. The PRAMIN sentinel test correctly identifies
this. `SovereignBootState::Cold` accurately classifies the state.

**The hardware line is unbreachable by software**: only a full power cycle
(including board-level VR sequencing) trains HBM2. This is the same
limitation NVIDIA's proprietary driver faces. Our keepalive strategy
(VfioAnchor + systemd FileDescriptorStore) prevents transitions to cold
state between service restarts; it cannot recover from cold.

## Deployment Note

The `sovereign.profile` method initially returned "Method not found" despite
correct implementation. Root cause: `cargo build -p toadstool-server` only
builds the library. The main binary is `toadstool-cli` (`crates/cli`). After
`cargo build --release -p toadstool-cli` and redeployment, the method worked.
