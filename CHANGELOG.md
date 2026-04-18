# Changelog

All notable changes to hotSpring.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file covers the spring as a whole. For crate-level details see
`barracuda/CHANGELOG.md`.

## Unreleased — Stadial Audit + Primal Composition Proof (April 17, 2026)

### Added
- `deny.toml` for barracuda + metalForge/forge — ecoBin C-dep bans, async-trait ban, license allowlist
- `rust-version = "1.87"` in Cargo.toml (ecosystem MSRV target)
- Composition parity tolerances: `COMPOSITION_SEMF_PARITY_REL` (1e-10), `COMPOSITION_PLAQUETTE_PARITY_ABS` (1e-12) in `tolerances/physics.rs`
- Science parity probes in `validate_nucleus_node` and `validate_nucleus_composition` — local Rust values vs IPC-routed primal results within centralized tolerances
- `niche::set_family_id()` + `OnceLock` for thread-safe family ID resolution
- wateringHole handoffs: `HOTSPRING_V0632_STADIAL_AUDIT_HANDOFF_APR17_2026.md`, `HOTSPRING_V0632_PRIMAL_ABSORPTION_HANDOFF_APR17_2026.md`

### Fixed
- **GAP-HS-026 resolved**: All 13 physics/compute methods wired in `hotspring_primal.rs` server dispatch with `catch_unwind` safety — no more `-32001` pending stubs
- **Unsafe elimination**: Replaced `unsafe { std::env::set_var("FAMILY_ID", ...) }` with `niche::set_family_id()` using `OnceLock`
- **`dyn` dispatch eliminated**: `GpuRegisterMap` enum replaces `Box<dyn RegisterMap>`; `ValidationSink` enum replaces `Arc<dyn ValidationSink>`
- **`#[allow]` → `#[expect]` migration**: All production binary code (11 files, ~20 sites) migrated to `#[expect(lint, reason = "...")]`; library `#[allow]` in `#[cfg(test)]` retained per convention
- **Inline tolerances centralized**: ~15 numeric literals in `validate_chuna.rs` replaced with named constants from `tolerances::*`; 10 new constants added with documented rationale
- **Deploy graph capability fix**: `coralreef` `by_capability` corrected from `shader_compile` to `shader` in `hotspring_qcd_deploy.toml`
- **Proto-nucleate references**: Updated stale `hotspring_qcd_proto_nucleate.toml` → `downstream_manifest.toml` in niche.rs, deploy graph, validators
- **Downstream manifest aligned**: `validation_capabilities` updated to actual `physics.*`/`compute.*` methods; `nestgate` added to `depends_on`
- **Spring validation manifest**: Capabilities expanded from 3 to all 13 served methods

### Changed
- `validate_all.rs` expanded from 37 to 62 suites (three-tier: Python baselines → Rust validation → NUCLEUS IPC composition)
- `composition.rs` docstring clarified: `validate_science_probes()` checks liveness, not numeric parity
- `primal_bridge.rs` and `toadstool_report.rs` now use centralized `niche::family_id()`

### Documentation
- Root docs (README, EXPERIMENT_INDEX, whitePaper): dates, binary counts (164), suite counts (62/62), composition narrative updated
- `docs/PRIMAL_GAPS.md`: GAP-HS-026 marked RESOLVED; composition tolerance constants documented
- `infra/wateringHole/NUCLEUS_SPRING_ALIGNMENT.md`: test count corrected (985), proto-nucleate reference updated
- `primalSpring/graphs/spring_deploy/spring_deploy_manifest.toml`: hotSpring added as 6th science spring

## v0.6.32 — Composition Audit + Doc Cleanup (April 11, 2026)

### Added
- `graphs/` directory with deploy TOMLs aligned to proto-nucleate
- `validate_squirrel_roundtrip` binary for end-to-end inference validation
- Root-level CHANGELOG (this file)
- wateringHole handoff: `HOTSPRING_V0632_COMPOSITION_AUDIT_PRIMAL_EVOLUTION_HANDOFF_APR11_2026.md`

### Fixed
- **Socket naming mismatch**: `hotspring_primal` now uses `niche::resolve_server_socket()` for family-scoped socket names (`hotspring-physics-{family_id}.sock`) — previously hardcoded `hotspring-physics.sock`
- **biomeOS registration**: `register_with_target()` now called on server startup — previously existed but was never invoked
- **barraCuda pin reconciled**: `Cargo.toml` now matches CHANGELOG v0.6.32 pin (`b95e9c59`)
- **DAG method names aligned**: `dag_provenance.rs` uses canonical wire names (`dag.session.create`, `dag.event.append`, `dag.merkle.root`) matching `capability_registry.toml`
- **Crypto method name**: `receipt_signing.rs` uses `crypto.sign_ed25519` (was `crypto.sign`)
- **Capability validation binaries**: `validate_nucleus_composition` and `validate_nucleus_tower` use canonical capability names (`crypto.sign_ed25519`, `crypto.verify_ed25519`, `discovery.find_primals`, `shader.compile.wgsl`)
- **Registered-but-pending dispatch**: `hotspring_primal` returns structured `-32001` error for registered physics methods not yet dispatched (was generic `-32601 Method not found`)
- **discover_capabilities()**: Now delegates to `niche::all_capabilities()` as source of truth

### Changed
- `validation.rs` split into `validation/` module (harness, telemetry, composition, tests) — was 1392 lines, now all files under 520 LOC
- `--family-id` CLI argument now sets `FAMILY_ID` env var (was silently ignored)

### Documentation
- Root README: fixed baseCamp listing (5→17 docs), barraCuda pin ref (`fbad3c0a`→`b95e9c59`), added `graphs/` and `CHANGELOG.md` to directory tree, expanded composition narrative
- `whitePaper/README.md`: status promoted from "Working draft" to "Current", codebase health table updated (Feb 26→Apr 11, v0.6.14→v0.6.32)
- `whitePaper/baseCamp/` listing in README/whitePaper now reflects full 17-doc set
- `experiments/README.md`: added NUCLEUS composition section, noted Exp 164 journal gap, disambiguated Exp 103a/103b
- Archived superseded handoff `HOTSPRING_V0632_COMPOSITION_EVOLUTION_HANDOFF_APR11_2026`

## v0.6.32 — Composition Evolution (April 10-11, 2026)

See `barracuda/CHANGELOG.md` for full details.

## License

AGPL-3.0-or-later
