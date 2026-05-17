# Diesel Engine Driver Sketch + PRI Refactor Handoff

**Date:** 2026-05-16  
**Session:** [Diesel engine driver sketch](379a2383-b2b1-4884-85b5-10c69458363b)  
**Scope:** toadStool cylinder — sovereign compute driver components  
**Status:** Implemented, reviewed, refactored, validated  

## What Happened

Four new modules in `toadstool-cylinder` sketching the Rust-side driver
replacement components. These capture, formalize, and replay what Linux
kernel GPU drivers do during `probe()` / `init()`. Followed by a review
pass that extracted shared abstractions and eliminated code duplication.

## New Components (cylinder)

### 1. `nv::pri` — PRI Fault Detection (shared)

Single source of truth for NVIDIA PRI (Private Register Interface) error
pattern classification. Replaces 4 independent copies across `gr_init`,
`driver_probe`, `pmu_init`, and `warm_capture`.

- `is_pri_fault(val)` — `0xFFFF_FFFF`, `0xBADFxxxx`, `0xBAD0xxxx`, `0xDEAD_DEAD`
- `is_error_or_zero(val)` — fault patterns + zero (for "alive" counting)
- `domain_for_offset(offset, domains)` — BAR0 offset → domain name lookup

**Upstream relevance:** Any code probing BAR0 registers should use `nv::pri`
instead of inline pattern matching.

### 2. `nv::gr_init` — GR Init Sequence Capture & Replay

`GrInitSequence` is an ordered list of BAR0 register writes that reproduces
a driver's GPU initialization. Built from cold/warm BAR0 snapshot diffs.

Key methods:
- `from_bar0_diff()` — constructs from snapshot pairs with domain labeling
- `apply(bar0)` — replays onto hardware (handles masked read-modify-write)
- `validate(bar0)` — readback verification, returns mismatches
- `merge()` — combines sequences from multiple driver sources
- `to_json()` / `from_json()` — persistent storage for cross-session use

`ChipFamily` enum now delegates `from_sm()` to `profile_for_sm()` for
consistency with `GenerationProfile` SM ranges.

### 3. `vfio::warm_capture` — Automated Warm State Pipeline

Orchestrates the cold→warm snapshot→diff→sequence extraction pipeline:

- `Bar0Snapshot` — point-in-time BAR0 register capture with alive counting
- `Bar0Diff` — structural diff with range filtering
- `WarmStateCapture` — complete pipeline output (cold + warm + diff + GrInitSequence)

Designed to receive snapshots from glowplug-orchestrated driver swaps.
Bridges `bar_cartography::BarMap` into the new snapshot-based format.

### 4. `nv::driver_probe` — Multi-Driver Comparison Lab

Formalizes the "Driver Lab" concept (Exp 195) into reusable types:

- `FalconState` enum — probes any falcon at arbitrary BAR0 base
  (NotStarted/Halted/Running/HsLocked/PriGated)
- `TrialResult` — captures PMC_ENABLE, PGRAPH liveness, falcon states,
  PFIFO status for a single driver trial
- `DriverProbe` — multi-trial comparison with analysis methods

### 5. `nv::pmu_init` — Kepler PMU Bootstrap

PMU falcon initialization for unsigned-firmware GPUs (Kepler/Maxwell):

- `PmuSnapshot` — captures PMU register state, derives `FalconState`
- `PmuBootstrap` — reset→IMEM→DMEM→start→mailbox handshake→PFIFO test
- Cross-references `vfio::channel::devinit::pmu` (devinit-era naming)

### 6. Sovereign Init Integration

`sovereign_init.rs` Stage 3b: Kepler PGRAPH ungating. If a `GrInitSequence`
is provided in `SovereignInitOptions`, the pipeline replays it to ungate
PGRAPH before falcon boot on NoAcr GPUs. Refactored from 35 inline lines
to 10-line delegation to `GrInitSequence::apply()`.

## Patterns for Upstream Teams

### For toadStool
- `nv::pri` should be the canonical PRI fault helper for all new BAR0 probing
- `GrInitSequence` JSON persistence enables cross-session capture→replay workflows
- `ChipFamily::from_profile()` is preferred over `from_sm()` for type safety

### For primalSpring / primals audit
- Zero TODO/FIXME/HACK in library code (confirmed)
- Zero barracuda Rust source TODOs (confirmed)
- 40 new tests with full coverage of edge cases (error patterns, domain boundaries, serde roundtrips)

### For coralReef
- `FalconState::probe(bar0, base)` is a reusable building block for any falcon
  status check — SEC2, FECS, GPCCS, PMU all use the same register layout
- `PmuBootstrap` captures the Kepler PMU init sequence that coralReef's
  `devinit/pmu.rs` implements procedurally — candidate for convergence

## Gaps for Upstream

| Gap | Owner | Priority |
|-----|-------|----------|
| `GrInitSequence::apply()` not yet tested on hardware | toadStool | P1 — next warm boot session |
| `PmuBootstrap::full_boot()` needs real Kepler firmware | toadStool | P1 — K80 warm capture |
| `FalconState` and `devinit::pmu` falcon state logic overlap | toadStool | P2 — convergence candidate |
| `pmu_init::pmu_reg` vs `devinit::pmu::pmu_reg` naming | toadStool | P3 — cosmetic unification |
| `DriverProbe` needs nvidia-470 VM trial data (Exp 195 Trial 3) | hotSpring | P1 — benchScale VM |

## Validation

- 40 module tests pass (pri, gr_init, driver_probe, pmu_init, warm_capture)
- Full workspace: 585+ tests pass
- Zero lint errors across all modified files
- `cargo check --lib` clean (1 pre-existing warning in unrelated `pfifo.rs`)

## Next Steps

1. Hardware validation: warm boot Titan V + K80, capture `GrInitSequence`, attempt `apply()` replay
2. K80: `PmuBootstrap::full_boot()` with firmware from nouveau warm capture
3. Driver Lab Trial 3: nvidia-470 in benchScale VM for Titan V comparison
4. Continue evolving warm→cold sovereign across the compute trio
