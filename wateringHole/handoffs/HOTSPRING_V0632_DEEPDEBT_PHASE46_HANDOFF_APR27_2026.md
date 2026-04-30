# hotSpring v0.6.32 Deep Debt + Phase 46 Handoff

**From:** hotSpring
**To:** primalSpring, barraCuda, biomeOS
**Date:** April 27, 2026

## Summary

Two complementary evolution passes completed on the same day.

### Deep Debt Evolution

- **Capability-based primal discovery**: `composition.rs` derives all primal
  requirements from `niche::DEPENDENCIES` (single source of truth). Hardcoded
  name→domain maps removed.
- **Named accessors deprecated**: `primal_bridge.rs` methods (`toadstool()`,
  `beardog()`, etc.) deprecated → `by_domain("compute")`, `by_domain("crypto")`.
- **Data-driven alias resolution**: `PRIMAL_ALIASES` table replaces hardcoded
  `if primal == "coralreef"` checks.
- **Smart refactoring**:
  - `lattice/rhmc.rs` (989L) → `rhmc/mod.rs` (802L) + `rhmc/remez.rs` (190L)
  - `nuclear_eos_helpers.rs` (978L) → `mod.rs` (824L) + `objectives.rs` (174L)
- **Pre-existing compile errors fixed**: `nuclear_eos_l2_ref.rs` and
  `nuclear_eos_l2_hetero.rs` updated for upstream `DiscoveredDevice` API change.

### Phase 46 Composition Template

- `tools/hotspring_composition.sh`: Event-driven QCD computation via NUCLEUS
  composition library. Async tick model, DAG memoization, ledger sealing,
  scientific provenance braids.
- `tools/nucleus_composition_lib.sh`: 41-function NUCLEUS wiring library from
  primalSpring.
- Bare mode verified: all functions degrade gracefully without NUCLEUS primals.

### EVOLUTION Markers for Upstream

| Marker | File | Description |
|--------|------|-------------|
| `EVOLUTION(B2)` | `resident_cg.rs`, `resident_cg_brain.rs`, `hasenbusch.rs`, `dynamical.rs` | GPU-resident Hamiltonian assembly — blocked on fused kernel |
| `EVOLUTION(GPU)` | `hfb_deformed_gpu/mod.rs` | Wire deformed WGSL shaders |

### Active PRIMAL_GAPS

- GAP-HS-001: Squirrel end-to-end (low, waiting neuralSpring)
- GAP-HS-005: IONIC cross-family GPU lease (medium, blocked upstream)
- GAP-HS-006: BTSP session crypto (medium, blocked upstream)

### Validation

993/993 lib tests pass. Zero compilation errors. `cargo clippy` clean.
