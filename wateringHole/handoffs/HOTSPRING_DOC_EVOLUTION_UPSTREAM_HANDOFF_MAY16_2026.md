# hotSpring — Documentation Evolution + Upstream Handoff (May 16, 2026)

## Summary

Complete documentation alignment pass across hotSpring root docs,
whitePaper/baseCamp/, experiments/, wateringHole/, capability registry,
and notebook. Archival sweep of stale handoffs. Upstream patterns
documented for primalSpring and sibling spring absorption.

---

## Changes Applied

### Test/binary/suite count normalization (592 → 595)

All living documents updated to canonical counts:

| Metric | Value |
|--------|-------|
| Lib tests (default) | **595** |
| Lib tests (barracuda-local) | **1,041** |
| Validation suites | **65** |
| Binaries | **167** |
| WGSL shaders | **128** |
| Deploy graphs | **7** |
| Experiments | **196** |
| Scenarios | **17/20** |

Files updated: `README.md`, `whitePaper/README.md`, `whitePaper/baseCamp/README.md`,
`whitePaper/baseCamp/nucleus_composition_evolution.md`, `experiments/README.md`,
`notebooks/01-composition-validation.ipynb`.

Historical changelog entries (May 13, May 14) retain their era-accurate 592 counts
as fossil record.

### `hotspring_primal` → `hotspring_unibin` normalization

All living documentation now references `hotspring_unibin` as the canonical binary.
`hotspring_primal.rs` is referenced only as fossilized source where historically
accurate.

Files updated: `whitePaper/baseCamp/nucleus_composition_evolution.md` (2 references),
`barracuda/config/capability_registry.toml` (schema comment),
`barracuda/src/bin/validate_nucleus_node.rs` (code comment),
`notebooks/01-composition-validation.ipynb` (summary table).

Historical references in `PRIMAL_GAPS.md` resolved gap entries and `CHANGELOG.md`
historical entries left as fossil record.

### experiments/README.md — active experiment expansion

Experiments 192–196 added to Active table (were on disk but not listed):

| # | Name |
|---|------|
| 192 | HARDWARE_VALIDATION_SPRINT_COMPUTE_TRIO |
| 193 | PLX_D3COLD_KEEPALIVE_K80 |
| 194 | COLD_WARM_BOOT_ARCHITECTURE |
| 195 | DRIVER_LAB_MESA_VS_VENDOR |
| 196 | WARM_SWAP_VALIDATION_PLX_KEEPALIVE |

### EXPERIMENT_INDEX.md — typo fix

Stray `||` table delimiter on Exp 194 row corrected to `|`.

### scripts/README.md — path correction

Archived script paths corrected from `lab/` (nonexistent) to `archive/` (actual location).

### PRIMAL_GAPS.md — reference fixes

- Stale handoff filename `HOTSPRING_V0632_DEEP_DEBT_RESOLUTION_HANDOFF_MAY13_2026.md`
  corrected to `HOTSPRING_DEEP_DEBT_SPRINT_MAY13_2026.md`.
- Historical test count 592 → 595 in May 13 deep debt entry.

### wateringHole handoff archival

Seven May 12 handoffs moved from `handoffs/` to `handoffs/archive/`:
- COMPUTE_TRIO_CAPABILITY_EVOLUTION
- COMPUTE_TRIO_PIPELINE
- EMBER_GLOWPLUG_OWNERSHIP_AUDIT
- IPC_TRANSPORT_EVOLUTION
- PHASE_C_EXECUTION_PLAN
- VFIO_SOVEREIGN_DISPATCH
- WARM_VFIO_DISPATCH_EVOLUTION

Titan V DMATRF May 7 handoff status corrected from `✅` to `upstream`
(file was migrated to `infra/wateringHole/handoffs/`).

---

## Current hotSpring State (for upstream teams)

### Wave 17 Signal Adoption (complete)

hotSpring has adopted three primalSpring Wave 17 signals:

1. **`primal.announce`** — Single atomic registration replaces legacy
   `lifecycle.register` + N × `capability.register` + `method.register`.
   Falls back automatically for pre-Wave-17 biomeOS.

2. **`node.compute`** — GPU workload dispatch. biomeOS decomposes into
   compile → submit → execute graph. Falls back to `compile_and_submit()`.

3. **`tower.publish`** — Signed result publication. biomeOS decomposes
   into sign → announce → audit. Falls back to direct `crypto.sign_ed25519`
   + `discovery.announce`.

**Next candidates:** `nest.store`, `nest.commit` (awaiting upstream nest
evolution in nestGate).

### Composition Pattern Insights (for sibling springs)

1. **UniBin pattern**: Single binary with subcommands (`certify`, `validate`,
   `serve`, `status`, `version`). `harvest-ecobin.sh` builds musl-static
   for plasmidBin. Every spring should adopt this over monolithic server binaries.

2. **Signal fallback pattern**: Always implement legacy IPC path alongside
   new signal adoption. Test both paths. Signal APIs that biomeOS doesn't
   yet decompose will return errors; the fallback catches them.

3. **Capability registry TOML**: `config/capability_registry.toml` declares
   local/routed capabilities, signal tiers, and registration method.
   Bidirectional sync test validates TOML ↔ code consistency.

4. **Niche self-knowledge**: `niche.rs` contains ONLY self-knowledge
   (capabilities, dependencies, semantic mappings, cost estimates).
   Primal discovery is runtime-only via `by_domain()`. No hardcoded
   primal names in routing logic.

5. **Three-tier validation**: Python baselines → Rust validation → NUCLEUS
   composition validation. Same tolerance/exit-code methodology proves
   composition equivalence to direct execution.

6. **Deploy graph**: `graphs/hotspring_*_deploy.toml` — 7 graphs covering
   full QCD compute lane. Node binary is `hotspring_unibin`.

### Deep Debt Status

| Metric | Status |
|--------|--------|
| TODO/FIXME/HACK | **Zero** |
| `unsafe` sites | **10** (all necessary: BAR0 MMIO + CUDA FFI) |
| Mock leakage | **Zero** (NpuSimulator is intentional production sim) |
| Files >800L (lib) | **Zero** (after niche + compute_dispatch refactoring) |
| External deps | **14** (all pure Rust except cudarc + wgpu) |
| Clippy warnings | **Zero** |
| Format drift | **Zero** |

### Open Frontiers

| Area | Status | Owner |
|------|--------|-------|
| LTEE B2 Anderson → lithoSpore module 7 | maintenance | strandGate |
| Plasma/QCD Thread 2 → foundation | active | strandGate |
| `nest.store` / `nest.commit` signal adoption | awaiting nestGate | strandGate |
| FECS PENDING_CTX_RELOAD (Titan V sovereign) | blocked on driver lab | biomeGate |
| K80 PLX D3cold prevention (SwapGuard validated) | solved, needs integration | biomeGate |
| Papers remaining (~25 queued) | backlog | strandGate |
| Militzer FPEOS / atoMEC datasets | partial ingest | strandGate |

---

## Upstream Asks

### For primalSpring

1. **451-method registry sync**: hotSpring is at Wave 17. Confirm HEAD
   registry matches our `capability_registry.toml` signal declarations.
2. **`nest.store` / `nest.commit` signal spec**: When nestGate evolves,
   hotSpring is ready to adopt. Need signal decomposition spec.
3. **sourDough v0.3.0 timeline**: Shell scripts (`harvest-ecobin.sh`,
   `build-guidestone.sh`) are functional; sourDough internalization
   will absorb them.

### For sibling springs

1. **Adopt `primal.announce`**: One call replaces 3+ registration calls.
   See `hotSpring/barracuda/src/niche/mod.rs::try_primal_announce()` for
   reference implementation with fallback.
2. **Adopt UniBin pattern**: `hotspring_unibin.rs` + `clap` subcommands.
   guideStone certification works per-mode.
3. **Check your PRIMAL_GAPS.md**: If you have open barraCuda or coralReef
   IPC gaps, the compute trio code is evolving under hotSpring hands.

---

## Files Changed This Sprint

```
README.md                                        (test counts 592→595)
EXPERIMENT_INDEX.md                              (typo fix)
whitePaper/README.md                             (counts, suites)
whitePaper/baseCamp/README.md                    (counts)
whitePaper/baseCamp/nucleus_composition_evolution.md  (counts, unibin)
experiments/README.md                            (192-196, counts)
scripts/README.md                                (archive paths)
docs/PRIMAL_GAPS.md                              (handoff ref, count)
notebooks/01-composition-validation.ipynb        (counts, unibin)
barracuda/config/capability_registry.toml        (comment fix)
barracuda/src/bin/validate_nucleus_node.rs        (comment fix)
wateringHole/README.md                           (status alignment)
wateringHole/handoffs/ → archive/                (7 May 12 handoffs)
```
