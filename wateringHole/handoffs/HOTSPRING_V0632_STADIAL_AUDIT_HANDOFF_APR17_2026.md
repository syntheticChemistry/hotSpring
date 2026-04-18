# hotSpring v0.6.32 Stadial Audit Handoff

**From:** hotSpring
**To:** primalSpring, barraCuda, biomeOS
**Date:** April 17, 2026
**Version:** 0.6.32
**Wire Standard:** L2 (JSON-RPC 2.0, UDS)
**License:** AGPL-3.0-or-later

---

## What Changed

### Stadial Gate Compliance

1. **Added `deny.toml`** — ecoBin C-dep bans with wrappers for blake3/cc, ring/rustls,
   pkg-config/khronos-egl. Bans async-trait as stadial invariant.
2. **Added `rust-version = "1.87"`** to Cargo.toml (ecosystem MSRV target).
3. **Migrated `#[allow]` → `#[expect(lint, reason)]`** across all production binary
   code (11 files, ~20 sites). Library-level allows in `#[cfg(test)]` retained as
   `#[allow]` per test-code convention.
4. **Updated `niche.rs` proto-nucleate reference** from stale standalone file to
   `downstream_manifest.toml` (v0.9.15 graph consolidation alignment).

### Validation Suite Completeness

5. **Expanded `validate_all.rs`** from 37 to 62 suites. Added: TTM, gradient flow
   (CPU + GPU), Chuna papers 43/44/45, BGK dielectric (CPU + GPU), kinetic-fluid,
   DSF-vs-MD, FPEOS, atoMEC, freeze-out, HVP g-2, production QCD v2, GPU β-scan,
   GPU dynamical HMC, sovereign round-trip, precision matrix, full CPU/GPU parity,
   all NUCLEUS composition validators, and Squirrel round-trip.

### Code Quality

6. **Eliminated `Box<dyn RegisterMap>`** in `register_maps/mod.rs` → concrete
   `GpuRegisterMap` enum dispatch.
7. **Eliminated `Arc<dyn ValidationSink>`** in `validation/composition.rs` → concrete
   `ValidationSink` enum (Stdout/Null/Ndjson variants).
8. **Centralized `validate_chuna.rs` inline tolerances** — replaced ~15 numeric
   literals with named constants from `tolerances::physics` and `tolerances::lattice`.
   Added 10 new named constants with documented rationale.

---

## Active Gaps Handed Back to primalSpring

| Gap ID | Primal | Description |
|--------|--------|-------------|
| GAP-HS-005 | BearDog/Songbird | IONIC cross-family GPU lease (crypto.sign_contract) |
| GAP-HS-006 | barraCuda/BearDog | BTSP session crypto (Phase 3) |
| GAP-HS-026 | hotSpring | **RESOLVED** — All 13 physics/compute methods wired in server |
| GAP-HS-027 | barraCuda | TensorSession not adopted (awaiting API stabilization) |
| GAP-HS-028 | hotSpring | LIME/ILDG zero-copy I/O |

---

## barraCuda Version

- **Pinned:** git rev `b95e9c59` (v0.3.11)
- **wgpu:** 28 (aligned)
- **pollster:** 0.4 (hotSpring) vs 0.3 (barraCuda) — dual version, no type conflict

---

## Composition Status

| Capability | Status |
|------------|--------|
| `health.liveness` | Served |
| `health.readiness` | Served |
| `capabilities.list` | Served |
| `mcp.tools.list` | Served (5 tools) |
| `composition.health` | Served |
| `physics.*` (9 methods) | Served (13 methods, catch_unwind safety) |
| `compute.*` (4 methods) | Served (13 methods, catch_unwind safety) |
| `inference.*` | Routed to Squirrel |
| `crypto.*` | Routed to BearDog |
| `storage.*` | Routed to NestGate |
| `dag.*` / `spine.*` / `braid.*` | Routed to provenance trio |

---

## Validation Commands

```bash
cd barracuda
cargo test --lib                        # unit tests
cargo run --release --bin validate_all  # full validation suite (62 suites)
cargo deny check                        # dependency hygiene
cargo clippy --all-targets              # lint check
cargo doc --lib --no-deps               # doc check
```
