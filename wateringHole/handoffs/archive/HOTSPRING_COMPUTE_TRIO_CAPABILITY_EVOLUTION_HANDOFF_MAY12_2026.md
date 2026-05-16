# hotSpring → Primal & Spring Teams: Compute Trio Rewire + Capability Discovery Evolution

**Date:** May 12, 2026  
**Spring:** hotSpring v0.6.32  
**Gap IDs:** GAP-HS-087 (Compute Trio Rewire), GAP-HS-088 (Deep Debt Capability Discovery)  
**For:** toadStool team, coralReef team, barraCuda team, primalSpring composition team, biomeOS neuralAPI team  

---

## Executive Summary

hotSpring has completed two interlocking sprints that fundamentally evolve how
it interacts with the compute trio (toadStool, barraCuda, coralReef) and how
it discovers peer primals at runtime. The core change: **hotSpring no longer
hardcodes knowledge of other primals' socket paths or internal names.** All IPC
discovery is now capability-based via `NucleusContext::by_domain()` and
`call_by_capability()`, with env-var overrides as CI/lab fallbacks only.

---

## 1. Compute Trio Rewire (GAP-HS-087)

### What Changed

| Area | Before | After |
|------|--------|-------|
| **PrecisionTier** | Local 4-variant enum (F32, DF64, F64, F64Precise) | Re-exported from `barracuda::device::precision_tier` — 15-tier canonical enum (Binary → DF128). hotSpring routing produces 4 physics tiers; full enum accepted for forward compatibility |
| **PhysicsDomain** | Local 12-variant enum | Re-exported from upstream — 15-variant (adds Inference, Training, Hashing) |
| **FmaPolicy** | Local definition | Re-exported from `barracuda::device::fma_policy` |
| **HardwareHint** | Not present | Added to `PrecisionRoute` metadata. Domain-based defaults: Inference/Training → TensorCore; physics domains → Compute |
| **toadStool dispatch** | ember-only IPC via `fleet_ember.rs` | New `fleet_toadstool.rs` (feature-gated `toadstool-dispatch`) with `ToadStoolDispatchClient` — parallel IPC path for Phase C cutover |
| **Barrier validation** | Not tested | 9 WGSL shaders using `workgroupBarrier()` cataloged; `validate_barrier_shaders()` tests coralReef's `membar.{cta,gl}` PTX emitter via `call_by_capability("shader", ...)` |
| **E2E validation** | Per-component | `validate_compute_trio_pipeline` binary: Yukawa force + Wilson plaquette through full barraCuda→coralReef→toadStool→hardware chain |

### What This Means for Upstream Teams

**toadStool team:**
- hotSpring is ready for Phase C cutover. `ToadStoolDispatchClient` mirrors
  the ember client's timeout structure. Methods: `compute.dispatch.capabilities`,
  `compute.dispatch.submit`, `health.liveness`.
- `HardwareHint` field on `PrecisionRoute` carries dispatch preferences that
  toadStool can use for multi-GPU, heterogeneous routing.
- `validate_compute_trio_pipeline` can be run against your test harness once
  trio daemons are available on target hardware.

**coralReef team:**
- Barrier shader validation exercises the `membar.{cta,gl}` emitter.
  `BARRIER_SHADERS` constant lists the 9 WGSL files; any regressions in
  `workgroupBarrier()` → PTX lowering will be caught.
- Provider label in `PRIMAL_PROOF_IPC_MAPPING.md` updated: glowplug marked
  "soft-deprecated, absorbed by toadStool Phase A+B."

**barraCuda team:**
- hotSpring now consumes your 15-tier/15-variant canonical enums via feature-gated
  re-exports. No more enum drift between springs. Sprint 56d alignment complete.

---

## 2. Capability Discovery Evolution (GAP-HS-088)

### Pattern: by_domain() as Primary Discovery

All IPC paths now follow this pattern:

```rust
let ctx = NucleusContext::detect();
let socket = ctx
    .by_domain("attribution")  // domain, not primal name
    .filter(|ep| ep.alive)
    .map(|ep| PathBuf::from(&ep.socket))?;
```

**Evolved modules:**

| Module | Domain | Old Pattern | New Pattern |
|--------|--------|-------------|-------------|
| `sweetgrass.rs` | `attribution` | `niche::socket_dirs()` → `biomeos/sweetgrass.sock` | `by_domain("attribution")` |
| `rhizocrypt.rs` | `dag` | `niche::socket_dirs()` → `biomeos/rhizocrypt.sock` | `by_domain("dag")` |
| `loamspine.rs` | `ledger` | `niche::socket_dirs()` → `biomeos/loamspine.sock` | `by_domain("ledger")` |
| `skunkbat.rs` | `security` | `niche::socket_dirs()` → `skunkbat/skunkbat.sock` | `by_domain("security")` |
| `precision_brain.rs` | `shader` | XDG_DATA_DIRS scan + env vars first | `by_domain("shader")` first |
| `toadstool_report.rs` | `compute` | `niche::socket_dirs()` + family-based name | `by_domain("compute")` first |
| `compute_dispatch.rs` | `shader` | Direct `send_jsonrpc` on socket | `call_by_capability("shader", ...)` |
| `deployment.rs` | N/A | `REQUIRED_PRIMALS` hardcoded list | `required_primals()` from `niche::DEPENDENCIES` |

### What This Means for Upstream Teams

**primalSpring composition team:**
- hotSpring is now the reference implementation for `by_domain()` / `call_by_capability()`
  IPC discovery. The pattern is documented in `docs/PRIMAL_PROOF_IPC_MAPPING.md`
  and can be absorbed into the composition guide for sibling springs.
- `niche::DEPENDENCIES` is the single source of truth for required primals,
  capability domains, and routing. Springs should derive deployment validation
  from this table, not hardcoded name lists.

**biomeOS neuralAPI team:**
- Deploy graphs updated: skunkBat `order` values deduplicated (was 9/9 with
  sweetgrass, now sweetgrass=9 / skunkbat=10) for deterministic boot sequencing.
- 7 deploy graphs validated for `by_capability` node metadata consistency.
- All graphs use `proto_nucleate` coordination. neuralAPI can rely on
  deterministic startup order across all hotSpring pipeline variants.

---

## 3. NUCLEUS Composition Patterns for neuralAPI Deployment

### Deploy Graph Architecture

hotSpring ships 7 TOML deploy graphs:

| Graph | Physics Domain | Compute Primals |
|-------|---------------|-----------------|
| `hotspring_qcd_deploy` | Lattice QCD | coralReef + toadStool + barraCuda |
| `hotspring_md_deploy` | Molecular Dynamics | coralReef + toadStool + barraCuda |
| `hotspring_nuclear_eos_deploy` | Nuclear EOS | coralReef + toadStool + barraCuda |
| `hotspring_plasma_deploy` | Plasma Transport | coralReef + toadStool + barraCuda |
| `hotspring_plasma_md_deploy` | Plasma + MD | toadStool + barraCuda (lightweight) |
| `hotspring_spectral_deploy` | Spectral Theory | barraCuda only (CPU-side) |
| `hotspring_sovereign_gpu_deploy` | Sovereign GPU Boot | full nest |

### Deployment via neuralAPI

```bash
biomeos deploy --graph graphs/hotspring_qcd_deploy.toml
```

The graph declares `proto_nucleate` coordination. biomeOS/neuralAPI:
1. Reads node list and `order` values for deterministic boot sequence
2. Spawns primals in order (beardog → songbird → coralReef → toadStool → ...)
3. Each node's `health_method` (`health.liveness`) is polled before proceeding
4. `by_capability` metadata on each node declares what capability domain it serves
5. hotSpring (`order` = last) discovers all primals via `NucleusContext::detect()`

### Capability Domain Registry

hotSpring's `niche.rs` declares these capability domains:

| Domain | Provider | Required |
|--------|----------|----------|
| `crypto` | beardog | yes |
| `discovery` | songbird | yes |
| `shader` | coralReef | yes |
| `compute` | toadStool | yes |
| `math` | barraCuda | yes |
| `storage` | nestGate | yes |
| `dag` | rhizoCrypt | no |
| `ledger` | loamSpine | no |
| `attribution` | sweetGrass | no |

---

## 4. Remaining Work (GAP-HS-087 Residuals)

| Item | Owner | Blocked On |
|------|-------|------------|
| Wire `TensorSession` into `gpu_hmc/mod.rs` | hotSpring + barraCuda | GAP-HS-027 (TensorSession adoption) |
| Default dispatch path: ember → toadStool | hotSpring + toadStool | Phase C cutover stabilization |
| Cross-generation validation | hotSpring | trio daemons on K80/Titan V/RTX 5060/MI50 |
| `fleet_client.rs` coralReef layout removal | hotSpring | Phase C completion (ember fully deprecated) |

---

## 5. Validation

- **1,031** library tests pass (`barracuda-local` + `toadstool-dispatch`)
- **Zero** clippy warnings
- **19** files changed across compute trio rewire + capability discovery
- All deploy graphs validate via `certification/deployment.rs`

---

## References

- `docs/PRIMAL_GAPS.md` — GAP-HS-087 (active), GAP-HS-088 (resolved)
- `barracuda/CHANGELOG.md` — Compute Trio Rewire + Deep Debt entry
- `docs/PRIMAL_PROOF_IPC_MAPPING.md` — Updated provider labels
- `barracuda/src/fleet_toadstool.rs` — toadStool dispatch client
- `barracuda/src/bin/validate_compute_trio_pipeline.rs` — E2E validation binary
