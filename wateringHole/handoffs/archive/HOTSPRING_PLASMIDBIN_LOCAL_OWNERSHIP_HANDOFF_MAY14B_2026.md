# hotSpring — plasmidBin Local Ownership + Debt Resolution Handoff

**Date**: 2026-05-14  
**Sprint**: plasmidBin Local Debt Resolution  
**Scope**: ecoBin harvest lag, deployment tooling evolution, full NUCLEUS deployment

---

## Problem Statement

plasmidBin ecoBin harvest lag caused incremental GitHub Releases to ship only
updated primals, leaving 10/13 primals unavailable from the latest release.
`fetch.sh` only checked the latest release tag, causing widespread FAIL results
on `--all` fetches. Additionally, `doctor.sh` could not correctly identify
`static-pie` ecoBins behind backward-compat symlinks.

## Implemented Fixes

### 1. Release Cascade in `fetch.sh`

Evolved `fetch.sh` to cascade through the 5 most recent releases when a binary
is not found in the latest. This solves the incremental-release problem without
requiring every release to carry all 13 primals.

- New `CASCADE_DEPTH=5` parameter
- `resolve_fallback_tags()` function queries recent releases
- Download loop: primary tag → cascade tags → mirror
- Summary reports `Cascaded: N (from older releases)`
- **Result**: `--force --all` now returns 13/13, 0 failures

### 2. Symlink-Aware Doctor (`doctor.sh`)

Fixed `doctor.sh` to follow symlinks when checking binary properties:
- `file` → `file -L` (follows symlinks to actual ELF binary)
- `du -h` → `du -hL` (reports real file size, not symlink size)
- Also fixed aarch64 and coordination primal checks

**Result**: `doctor.sh` now correctly reports 13/13 ecoBin (static-pie, stripped)

### 3. Generalized `upgrade-primal.sh`

Created `scripts/boot/upgrade-primal.sh` — a unified upgrade script replacing
the toadstool-only script. Supports:
- `--all` (all 13 NUCLEUS primals)
- `--trio` (compute trio: toadstool, barracuda, coralreef)
- `--status` (deployment status matrix)
- `--check` (dry-run update check)
- `--force` (force reinstall)
- Automatic rollback on service start failure
- Service map for systemd integration

### 4. User-Mode Systemd Services

Created `barracuda-user.service` and `coralreef-user.service` for user-mode
systemd deployment, fixing the template variable (`%I`/`%U`) issue that
prevented the system-mode service files from working in `--user` mode.

### 5. Full NUCLEUS Deployment

All 13 primals deployed to `/usr/local/bin/`:
- beardog, songbird, skunkbat (Tower atomic)
- toadstool, barracuda, coralreef (Node atomic compute)
- nestgate, rhizocrypt, loamspine, sweetgrass (Nest atomic)
- squirrel, petaltongue, biomeos (Meta-tier)

Compute trio verified live:
- toadstool: `health.version` → v0.1.0 (system service)
- barracuda: IPC live, `precision.route` responds (user service)
- coralreef: `health.version` → v0.1.0 (user service)

## Upstream Gaps Found

### GAP-PB-001: Incremental Release Missing Checksums

**Owner**: primalSpring (plasmidBin maintainer)  
**Issue**: `skunkbat` has no checksum entry in v2026.05.12's `checksums.toml`
for `x86_64-unknown-linux-musl`. All other 12 primals have checksums.  
**Impact**: Checksum verification skipped for skunkbat.

### GAP-PB-002: barracuda Missing `health.version` RPC

**Owner**: barraCuda team  
**Issue**: barracuda returns `-32601 Unknown method` for `health.version`.
toadstool and coralreef both implement it.  
**Impact**: Cannot verify barracuda version via standard health RPC.

## Files Changed in plasmidBin

- `fetch.sh`: Release cascade logic (non-breaking addition)
- `doctor.sh`: Symlink-aware `file -L` / `du -hL` (bug fix)

## Files Changed in hotSpring

- `scripts/boot/upgrade-primal.sh`: New generalized upgrade script
- `scripts/boot/barracuda-user.service`: New user-mode service
- `scripts/boot/coralreef-user.service`: New user-mode service

## Verification

```
plasmidBin doctor: 39 pass, 10 warn (aarch64 not built), 0 fail
hotSpring tests:   595/595 pass, 0 clippy warnings
Compute trio IPC:  3/3 live, 9 sockets
NUCLEUS deployed:  13/13 primals current
```
