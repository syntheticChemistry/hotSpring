# Changelog

All notable changes to hotSpring.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file covers the spring as a whole. For crate-level details see
`barracuda/CHANGELOG.md`.

## Unreleased — Property 3 CHECKSUMS + Script Fix (April 17, 2026)

### Added
- **BLAKE3 CHECKSUMS manifest** (`validation/CHECKSUMS`): 15 validation-critical source files hashed with BLAKE3, following primalSpring's source-integrity pattern. Covers guideStone binary, physics modules, provenance, tolerances, composition, niche, lib, Cargo.toml, and validate-primal-proof.sh.
- Old binary-artifact CHECKSUMS preserved as `validation/CHECKSUMS.v0631-binaries`.

### Changed
- **Property 3 (Self-Verifying)**: `deny.toml` lookup now checks both `deny.toml` (barracuda/) and `barracuda/deny.toml` (repo root), so the guideStone passes from either CWD.
- **`validate-primal-proof.sh`**: Now builds from `barracuda/` then runs the binary from the repo root, so `validation/CHECKSUMS` resolves correctly. BLAKE3 checksums verify on every invocation.

### Verified
- Bare guideStone: **30/30 checks pass** (3 SKIP — expected NUCLEUS liveness only). Property 3 now fully green with all 15 file hashes verified.
- 990 unit tests pass (0 failures, 6 ignored)
- guideStone runs correctly from both barracuda/ and repo root

## Blackwell Dispatch Live (April 16, 2026)

### Fixed (coralReef Iter 85)
- **f64 division on Blackwell**: `MUFU.RCP64H` returns 0 on SM120 hardware — coralReef now uses F2F(f64→f32) + MUFU.RCP + F2F(f32→f64) seed on SM ≥ 100, 2 Newton-Raphson iterations for full precision. Same fix applied to f64 sqrt (RSQ64H → F2F+RSQ).
- **`@builtin(num_workgroups)` on Blackwell**: S2R NCTAID_X/Y/Z returns [0,0,0] on SM120 — coralReef now emits LDC c[7][0/4/8] from driver constants CBUF on SM ≥ 100. coral-driver populates CBUF 7 with grid dimensions at dispatch time.
- **Semaphore fence ordering**: `submit_fence_release` uses compute engine's SET_REPORT_SEMAPHORE (subchannel 1) instead of PBDMA — ensures fence completes only after compute work
- **UVM write access**: `map_external_allocation` now sets `gpu_mapping_type = 1` (ReadWriteAtomic)
- **QMD v5.0 completeness**: GRID_*_RESUME fields, SM_CONFIG_SHARED_MEM_SIZE, QMD_GROUP_ID = 0x1f

### Documentation
- wateringHole handoff: `HOTSPRING_BLACKWELL_DISPATCH_LIVE_HANDOFF_APR16_2026.md`

## primalSpring v0.9.17 Absorption (April 20, 2026)

### Changed
- **primalspring dependency**: v0.9.16 → v0.9.17 (genomeBin v5.1, 46 cross-arch binaries, deployment-validated end-to-end, guideStone standard v1.2.0)
- **Doc reference**: v0.9.16 → v0.9.17 in guideStone binary module doc
- **`validate-primal-proof.sh`**: Auto-sets required NUCLEUS env vars when FAMILY_ID is provided — `BEARDOG_FAMILY_SEED` (derived from FAMILY_ID), `SONGBIRD_SECURITY_PROVIDER=beardog`, `NESTGATE_JWT_SECRET` (random Base64). Header updated for v0.9.17 genomeBin depot workflow.

### Verified
- Bare guideStone: 14/14 checks pass against primalSpring v0.9.17 (backward-compatible API)
- All v0.9.17 known issues addressed in script: beardog seed, songbird security provider, nestgate JWT, coralReef `--rpc-bind`

### Documentation
- CHANGELOG, README, PRIMAL_GAPS: v0.9.17 absorption session
- wateringHole handoffs: comprehensive evolution patterns + v0.9.17 absorption

## primalSpring v0.9.16 Absorption (April 20, 2026)

### Added
- `scripts/validate-primal-proof.sh` — end-to-end primal proof validation script. Bare mode (domain only) and `--full` mode (pre-flight `primalspring_guidestone` + domain `hotspring_guidestone`). Detects bare-only vs live NUCLEUS automatically.
- plasmidBin deployment workflow documented in README Quick Start

### Changed
- **Property 3 (Self-Verifying)**: Upgraded from manual CHECKSUMS file-exists check to `primalspring::checksums::verify_manifest()` — BLAKE3 per-file hash verification with PASS/FAIL/SKIP semantics (v0.9.16 pattern)
- **Protocol tolerance**: Added `is_protocol_error()` arms to `validate_provenance_witness` and `validate_compute_dispatch` — Songbird/petalTongue HTTP-on-UDS classified as SKIP (reachable but incompatible), matching v0.9.16 liveness semantics
- **primalspring dependency**: Auto-updated v0.9.15 → v0.9.16 (BLAKE3 checksums module, family-aware discovery, protocol error classification)
- **Doc reference**: v0.9.15 → v0.9.16 in guideStone binary module doc

### Verified
- Bare guideStone: 14/14 checks pass (4 SKIP expected — no CHECKSUMS manifest, no NUCLEUS primals)
- Known v0.9.16 issues handled: `is_protocol_error()` → SKIP for HTTP-on-UDS, `is_connection_error()` → SKIP for BearDog BTSP reset
- plasmidBin ecoBin present: `hotspring_primal` (v0.6.32, x86_64 musl-static)

### Documentation
- CHANGELOG, README, PRIMAL_GAPS: v0.9.16 absorption session
- wateringHole handoff: `HOTSPRING_V0632_V0916_ABSORPTION_HANDOFF_APR20_2026.md`

## Sovereign Compute Parity + guideStone (April 18-19, 2026)

### Added
- `bench_sovereign_parity` binary — dual-path QCD benchmark (coral-gpu sovereign vs wgpu vendor)
- Experiments 173–176: VM reagent WPR capture, K80 sovereign boot, RTX 5060 shared compute, QCD parity benchmark
- Full HMC pipeline (10 shaders) compiles to native SASS on SM35/SM70/SM120 via coralReef
- `validate_pure_gauge` sovereign GPU compile integration (16/16 checks pass)
- QMD v5.0 support for Blackwell (SM120+) in coral-driver
- **`hotspring_guidestone` binary** — self-validating NUCLEUS deployable using `primalspring::composition` API
- `deny.toml` for barracuda + metalForge/forge — ecoBin C-dep bans
- `validate_primal_proof` binary — Level 5 primal proof harness (10 manifest capabilities)
- wateringHole handoffs: sovereign compile parity, Level 5 composition proof, Blackwell dispatch handoff

### Fixed
- coralReef `UvmPageableMemAccessParams` ABI bug: struct was 4 bytes, kernel expects 8 — `pageable_mem_access` field was missing, causing false "failure" reports (ioctl was always succeeding)
- coralReef coral-kmod VRAM alloc flags: 2MB huge page attrs on 4KB buffers caused FAULT_PDE; fixed to PAGE_SIZE_4KB
- coralReef f64 transcendental lowering: removed SM < 70 guard, SM32 now lowers via MUFU seed + Newton-Raphson
- coralReef SM32 encoder: `emit_iadd` (IAdd2 for Kepler), `emit_shl_imm` (OpShl for Kepler)
- coralReef `as_imm_not_i20`/`as_imm_not_f20`: graceful fallback when source modifiers on immediates
- Broken doc links: `CONTROL_EXPERIMENT_STATUS.md` and `NUCLEAR_EOS_STRATEGY.md` marked superseded
- GAP-HS-026 resolved: All 13 physics/compute methods wired in `hotspring_primal.rs` server dispatch
- Unsafe elimination: `niche::set_family_id()` with `OnceLock` replaces `set_var`
- `dyn` dispatch eliminated, `#[allow]` → `#[expect]` migration

### Changed
- guideStone Level 5 CERTIFIED (all 5 guideStone properties satisfied)
- `validate_all.rs` expanded from 37 to 64 suites
- Binary count: 140 → 166
- GAP-HS-031 root cause identified: GR context buffers not eagerly promoted on Blackwell (UVM VA space registration fails, GPU_PROMOTE_CTX requires kernel privilege)

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
