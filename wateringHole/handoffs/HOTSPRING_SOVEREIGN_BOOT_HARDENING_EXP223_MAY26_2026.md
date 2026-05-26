# hotSpring Handoff — Sovereign Boot Infrastructure Hardening

**Date**: 2026-05-26
**Author**: Session 4 (ACR Sovereign Boot Catalyst → Infrastructure Hardening)
**Experiments**: 223 (ACR Sovereign Boot Catalyst), exp224 (PMU ACR Catalyst binary)
**Status**: EVOLVED — infrastructure hardened, unit tested, ready for next reboot cycle

## Summary

Extracted, abstracted, and hardened the sovereign boot infrastructure that was
developed across experiments 219–223. The flawed Python catalyst scripts are
superseded by idiomatic Rust with shared modules and safety guards.

## Key Outcomes

### 1. Shared Falcon Module (`barracuda/src/low_level/falcon.rs`)

Canonical Falcon v5 register map verified against toadStool and envytools:
- 24 register offsets with documentation (CPUCTL, BOOTVEC, CPUCTL_ALIAS, ENGCTL, etc.)
- 5 engine bases (PMU, FECS, GPCCS, SEC2, NVDEC)
- `FalconSnapshot` — read all diagnostic registers, decode SEC_MODE/cpu_state/IMEM/DMEM sizes
- PIO upload/verify helpers matching toadStool's `falcon_pio.rs`
- `Bar0Domain` presets for SafeBar0 domain validation

### 2. Hardened BAR0 (`barracuda/src/low_level/bar0.rs`)

- `Bar0Error` typed error enum (DeadLink, Unaligned, OutOfDomain, DenyListed)
- `r32_checked()` — dead-link sentinel detection (0xFFFF_FFFF = PCIe link down)
- `open_bdf()` — BDF-based open with `HOTSPRING_SYSFS_PCI` env support
- `SafeBar0::with_deny_list()` — ENGCTL deny-list prevents accidental falcon destruction
- Alignment checks on all register access (panics on misaligned offset)

### 3. Library Export

`pub mod low_level` exported from barracuda lib behind `low-level` feature gate.
Changed `#![forbid(unsafe_code)]` → `#![deny(unsafe_code)]` with
`#[allow(unsafe_code)]` on the low_level module only.

### 4. Unit Tests (16 total)

**bar0.rs (9 tests):** domain validation, boundary spanning, overflow, empty
domains, alignment rejection, deny-list enforcement, dead-link sentinel,
bar0_map_size fallback, error Display.

**falcon.rs (7 tests):** SEC_MODE decode (NS/LS/HS/??), CPUCTL bit values match
toadStool, cpu_state priority (HALT > HRESET > RUN), IMEM/DMEM size from HWCFG,
engine base + offset math, domain preset coverage, Display impl.

### 5. exp224 Rewired

Removed all inline register constants and duplicated structs. Now uses shared
`falcon::*` and `bar0::*` imports. Opens target GPU via `SafeBar0::open_with_deny_list`
with PMU+FECS ENGCTL deny entries.

## Architecture Discoveries (Session 3)

- PMU in HS mode 2 is a firmware fortress — host PIO blocked, CPUCTL_ALIAS unresponsive
- Correct boot path: Boot Falcon (NVDEC) → SEC2 → ACR → PMU (toadStool's `sovereign.init`)
- toadStool successfully brings both GPUs to `compute_ready: true` via this path
- WGSL compute dispatch validated post-init

## Fossil Record

| Artifact | Status | Location |
|----------|--------|----------|
| `acr_sovereign_boot.py` | SUPERSEDED (v1, flawed ENGCTL) | `infra/catalysts/reagents/` |
| `post_reboot_acr_boot.py` | SUPERSEDED (v2, correct approach) | `infra/catalysts/reagents/` |
| `gv100_acr_catalyst.json` | Reference data | `infra/catalysts/reagents/` |
| `exp224_pmu_acr_catalyst.rs` | ACTIVE (v3, Rust, shared modules) | `barracuda/src/bin/` |
| `falcon.rs` | ACTIVE (shared module) | `barracuda/src/low_level/` |

## Upstream Gaps for Review

- `experiment_catalog.json` still says 220 experiments (should be 223)
- `EXPERIMENT_INDEX.md` (root) may need 223 row added
- ecoPrimals `infra/wateringHole/` ecosystem README last updated May 18
- `infra/catalysts/frozen/` and `infra/catalysts/products/` referenced in docs but don't exist in repo

## Next Steps (Post-Reboot)

1. Investigate toadStool `sovereign.init` internals — map which Boot Falcon
   it uses, DMA buffer staging, PMU HS boot sequence
2. Extract working boot sequence into standalone Rust binary (no daemon dependency)
3. Validate production physics workloads post-sovereign-init
