# SPDX-License-Identifier: AGPL-3.0-only

# Cross-Team Handoff: hotSpring → Primals & Springs Teams

**Date:** 2026-05-16  
**From:** hotSpring (Experiments 001–198)  
**To:** toadStool, coralReef, barraCuda, biomeOS, primalSpring, sibling springs  
**Scope:** Evolution insights, NUCLEUS composition patterns, neuralAPI deployment, sovereign boot, atomic instantiations

---

## Executive Summary

hotSpring has completed 198 experiments spanning Python→Rust validation,
NUCLEUS composition proof, and sovereign GPU compute across three hardware
generations (K80/GK210, Titan V/GV100, RTX 5060/GB206). The work has
produced patterns, abstractions, and lessons that are ready for upstream
absorption and cross-team evolution.

**Key artifacts for absorption:**
1. `BootPipeline` trait — vendor-agnostic boot abstraction (3 implementations)
2. VBIOS interpreter fixes — K80 cold boot path unblocked at software level
3. `DeviceTopology` — multi-die/multi-function PCIe vocabulary
4. NUCLEUS composition validation methodology — Python→Rust→Primal three-tier proof
5. neuralAPI signal adoption patterns — `primal.announce`, `node.compute`, `tower.publish`
6. Deep debt resolution patterns — zero TODO/FIXME/HACK, zero production mocks

---

## Part 1: Primal Use & Evolution Review

### Primal Dependency Map (hotSpring niche)

| Primal | Role in hotSpring | IPC Methods Used | Evolution Status |
|--------|------------------|------------------|-----------------|
| **toadStool** | GPU dispatch, device management, sovereign boot | `compute.dispatch.submit`, `device.*`, `sovereign.init` | Active — BootPipeline just added |
| **coralReef** | Shader compilation (WGSL→SASS/GCN) | `shader.compile.wgsl` | Stable — compile-then-dispatch pipeline wired |
| **barraCuda** | Physics computation engine | `tensor.*`, `physics.*` | Stable — all 13 methods wired |
| **BearDog** | Trust/crypto | `crypto.sign_ed25519`, `crypto.hash` | Stable — BTSP Phase 3 |
| **Songbird** | Discovery | `discovery.announce`, `discovery.resolve` | Stable — family-aware |
| **NestGate** | Storage/DAG | `store.*`, `dag.*` | Stable |
| **rhizoCrypt** | Provenance | `provenance.*` | Stable |
| **loamSpine** | Ledger | `ledger.*` | Stable |
| **sweetGrass** | Scientific provenance braids | `braid.*` | Stable |
| **Squirrel** | ML inference (optional) | `inference.*` | Optional meta-tier |

### Patterns That Worked

1. **Capability-based routing (`call_by_capability`)**: Replaced all hardcoded
   socket paths. `niche::DEPENDENCIES` as single source of truth. Named
   accessors deprecated. This pattern should be mandatory for all springs.

2. **Three-tier validation**: Python baseline → Rust proof → NUCLEUS IPC parity.
   Tolerance-driven, exit-code-aware (0=pass, 1=fail, 2=all-skipped).
   Composition validators degrade honestly when primals are absent.

3. **Compile-then-dispatch**: `compile_and_submit()` chains coralReef
   `shader.compile.wgsl` → toadStool `compute.dispatch.submit` with compiled
   `binary_b64`. This is the canonical pattern for GPU workload dispatch.

4. **Circuit-breaker discovery**: `PrimalEndpoint` with `fail_count`/`dead_since`,
   3-failures-to-dead, 30s cooldown. Prevents cascade failure when a primal
   is down. `call_tracked()` for lifecycle-aware IPC.

5. **TOML-driven capability aliases**: `config/capability_registry.toml` loaded
   at runtime via `OnceLock`. Compiled defaults as fallback. Enables runtime
   capability remapping without recompilation.

### Patterns to Evolve

1. **`BootPipeline` → neuralAPI**: The `sovereign.probe`/`sovereign.verify`
   JSON-RPC methods should be exposed via toadStool's server so biomeOS can
   orchestrate fleet-level warm/cold assessment. This is the missing link
   between hardware boot and NUCLEUS composition — currently hotSpring's
   `sovereign.init` works but isn't discoverable via capability routing.

2. **TensorSession adoption (GAP-HS-027)**: barraCuda's fused multi-op
   pipeline is ready (`sub`, `negate`, GEMM routing). hotSpring hasn't
   adopted it yet. First candidate: HMC trajectory as single fused session.

3. **Cross-family GPU lease (GAP-HS-005)**: BearDog ionic bonding for
   multi-family metallic fleet pooling is not yet implemented upstream.

---

## Part 2: NUCLEUS Composition Patterns

### The Composition Stack

```
biomeOS
  └─ neuralAPI dispatch
       └─ NUCLEUS composition (9 required primals)
            ├─ Tower: BearDog + Songbird (trust + discovery)
            ├─ Node: toadStool + barraCuda + coralReef (compute + compile + dispatch)
            └─ Nest: NestGate + rhizoCrypt + loamSpine + sweetGrass (storage + provenance)
```

### Atomic Instantiation Patterns

hotSpring validates composition atom-by-atom, proving each tier independently
before testing the full NUCLEUS:

| Atom | Binary | What It Proves |
|------|--------|---------------|
| `tower_atomic` | `validate_nucleus_tower` | BearDog + Songbird alive, capability surface correct |
| `node_atomic` | `validate_nucleus_node` | Compute/shader/math stack via IPC, science parity probes |
| `nest_atomic` | `validate_nucleus_nest` | NestGate + provenance trio reachable, DAG creation works |
| `full_nucleus` | `validate_nucleus_composition` | All 4 tiers + science parity (SEMF, plaquette, HMC) |

**Deploy graph**: `graphs/hotspring_qcd_deploy.toml` — 10 primals as
`[[graph.nodes]]`, bonding policy, spawn order, tiered encryption.

### Scaling to Other Springs

Template for any spring adopting NUCLEUS:

1. **Declare** `LOCAL_CAPABILITIES` vs `ROUTED_CAPABILITIES` in `niche.rs`
2. **Register** with biomeOS via `register_with_target()` (lifecycle + capability)
3. **Pin** deploy graph fragments in TOML under `graphs/`
4. **Validate** atom-by-atom, then single composition + parity binary
5. **Serve** real JSON-RPC methods on the spring binary — no pending placeholders

Reference: `primalSpring/graphs/downstream/NICHE_STARTER_PATTERNS.md`

### neuralAPI Signal Adoption (Wave 17)

hotSpring adopted three neuralAPI signals from biomeOS:

| Signal | Purpose | Fallback |
|--------|---------|----------|
| `primal.announce` | Single-call registration (replaces lifecycle+capability+method) | Legacy multi-call for older biomeOS |
| `node.compute` | GPU workload dispatch through biomeOS graph decomposition | `compile_and_submit()` direct |
| `tower.publish` | Signed result publication (sign→announce→audit) | Direct `crypto.sign_ed25519` + `discovery.announce` |

These are declared in `config/capability_registry.toml` with `tier = "adopted"`.
Candidates for next wave: `nest.store`, `nest.commit`.

---

## Part 3: Sovereign Boot — What We Learned

### The BootPipeline Abstraction

**Key insight**: Every PCIe compute device follows the same boot sequence
regardless of vendor:

```
probe → is_warm → devinit → engine_init → verify
```

The `BootPipeline` trait captures this with `&dyn RegisterAccess` as the
universal MMIO interface. Vendor-specific detail lives in associated types
(`ProbeResult`/`InitResult`). Summary types bridge to universal consumers.

**Design decision**: Cold boot paths return `DriverError::Unsupported` via
`BootPipeline` because they require vendor-specific machinery (VBIOS replay,
falcon firmware, memory training) that `RegisterAccess` alone cannot express.
The full cold path uses `InitPipeline` (NVIDIA-specific, `&MappedBar`).

This means `BootPipeline` is useful for:
- Warm detection and verification (fleet-level health monitoring)
- Boot topology discovery (DeviceTopology)
- Universal probe/verify JSON-RPC (neuralAPI integration)

But NOT for:
- Cold boot execution (requires vendor-specific `InitPipeline`)
- Firmware loading (requires fork-isolated MMIO)

### GPU-Specific Findings

| GPU | Warm Boot | Cold Boot | Key Blocker |
|-----|-----------|-----------|-------------|
| **RTX 5060** | Not needed (sovereign from power-on) | N/A | None — fully sovereign |
| **Titan V** | ✅ warm-catch preserves FECS | Requires SEC2/ACR chain | GspBridge stub (stage 4) |
| **K80** | ✅ warm-catch via patched nouveau | VBIOS DEVINIT replay | vfio-pci BAR access (iommufd) |

### VBIOS Interpreter State

Four bugs fixed in K80 VBIOS parsing. The interpreter now correctly handles
Script 1 of the K80 VBIOS. Remaining work:

1. Scripts 2+ may have additional unhandled opcodes
2. Hardware validation requires proper VFIO device open (not sysfs)
3. Memory training (`PRAMIN` writes) needs VRAM access, which requires
   devinit to have run — chicken-and-egg for cold K80

---

## Part 4: Upstream Asks & Gaps

### For toadStool
- [ ] Absorb `BootPipeline` + `DeviceTopology` into public API (`toadstool-core`)
- [ ] Wire `sovereign.probe` / `sovereign.verify` as JSON-RPC methods
- [ ] Fix K80 VFIO device open (iommufd path for BAR0 access when vfio-pci bound)
- [ ] Evolve `InitPipeline` → `BootPipeline` delegation for warm paths

### For coralReef
- [ ] Implement real `GspBridge` for Titan V (warm FECS state → dispatch)
- [ ] Validate VBIOS interpreter Scripts 2+ for K80

### For barraCuda
- [ ] TensorSession first adoption candidate: HMC trajectory fused pipeline
- [ ] Consider `BootProbeInfo` / `BootInitInfo` as upstream types

### For biomeOS
- [ ] Wire `sovereign.probe` / `sovereign.verify` into fleet health monitoring
- [ ] `primal.announce` integration test with hotSpring's capability surface
- [ ] `node.compute` end-to-end: biomeOS decomposes compile→submit→execute

### For primalSpring
- [ ] Document `BootPipeline` pattern in composition standards
- [ ] Add `sovereign.probe` / `sovereign.verify` to method registry
- [ ] Consider DeviceTopology as a NUCLEUS vocabulary type

### For sibling springs
- [ ] Adopt three-tier validation pattern (Python→Rust→NUCLEUS IPC)
- [ ] Adopt `call_by_capability` routing (no hardcoded socket paths)
- [ ] Adopt circuit-breaker discovery pattern
- [ ] Review `tools/nucleus_composition_lib.sh` for composition wiring

---

## Part 5: Archive Candidates & Debris

### Already Archived (fossil record in scripts/archive/)
- Shell/Python warm-catch scripts (replaced by pure Rust pipeline)
- Legacy oracle capture scripts
- Deprecated warm-handoff scripts
- Pre-eukaryotic experiment binaries (in fossilRecord/)

### Current Debris Assessment
- **Zero TODO/FIXME/HACK** in codebase (deep debt audit complete)
- **Zero stale scripts** in scripts/ (all either active or in archive/)
- **Experiments 001-143** archived, 190 archived as final coral-ember
- **wateringHole/handoffs/archive/**: 7 May 12 handoffs properly archived
- **Historical naming**: coral-ember, coral-driver, coralctl references in
  archived docs are fossil record (diesel engine absorbed into toadStool)

### Documentation Fossil Record Policy
All docs are kept as fossil record in `ecoPrimals/`. We do not delete
documentation — it traces the evolution of ideas. Stale content is
marked with "Historical note" or moved to `archive/` subdirectories
with date-stamped references.

---

## Part 6: Test Count Summary

| Crate | Tests | Notes |
|-------|-------|-------|
| toadstool-cylinder | **606** | +15 from BootPipeline + VegaInit (was 591) |
| barracuda (default) | **595** | IPC-first, `default = []` |
| barracuda (barracuda-local) | **1,041** | Full feature set |
| Validation suites | **65** | 35 smoke + 7 nucleus + 23 silicon |
| Binaries | **167** | Including UniBin, validate_*, experiment bins |
| WGSL shaders | **128** | Lattice, MD, HFB, spectral, transport |
| Deploy graphs | **7** | QCD, plasma, MD, nuclear-EOS, spectral, sovereign |

---

## Conclusion

hotSpring's 198-experiment arc has proven that:

1. **Python→Rust fidelity** is achievable with tolerance-driven validation (500+ quantitative checks, $0.30 total science cost)
2. **NUCLEUS composition** works — IPC-routed results match direct Rust execution under documented tolerances
3. **Sovereign GPU compute** is viable across three hardware generations in pure Rust
4. **Vendor-agnostic abstractions** (BootPipeline, RegisterAccess) can capture the universal boot sequence while preserving vendor-specific detail

The work is ready for upstream absorption. The patterns documented here
are battle-tested across real hardware and should inform the evolution
of toadStool's public API, biomeOS's fleet orchestration, and the
NUCLEUS composition standards for all springs.
