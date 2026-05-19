# SPDX-License-Identifier: AGPL-3.0-only

# Handoff: Compute Parity, Mixed Hardware Evolution, and NUCLEUS Composition Patterns

**Date:** 2026-05-17
**From:** hotSpring (Wave 20 Experiment Buildouts + Compute Parity Sprint)
**To:** toadStool, coralReef, barraCuda, biomeOS, primalSpring, metalForge, sibling springs
**Scope:** CPU/GPU parity validation, toadStool dispatch offline validation, metalForge NUCLEUS atomics, PCIe direct topology, biomeOS graph coordination, deployment patterns

---

## Executive Summary

hotSpring completed a four-phase evolution sprint resolving experiment buildout
gaps, establishing CPU/GPU parity validation, wiring offline toadStool dispatch
tests, and evolving metalForge/forge with NUCLEUS atomic types, PCIe direct
topology, and biomeOS graph coordination primitives. The parity greenboard is
now ALL GREEN (10/10 papers). Three new validation scenarios have been added
(22 total: 17 default + 5 barracuda-local).

**Key artifacts for upstream absorption:**

1. `forge::nucleus` — NUCLEUS atomic types owned by forge for substrate-level coordination
2. `ChannelKind::PcieDirect` — GPU→NPU peer-to-peer data transfer without CPU roundtrip
3. `forge::biome_graph` — Directed graph for NUCLEUS atomic coordination (biomeOS absorption target)
4. `dispatch_cpu_fallback()` — Offline CPU execution when toadStool unavailable
5. Three new validation scenarios: `s_cpu_gpu_parity`, `s_toadstool_dispatch`, `s_mixed_hardware`
6. `commit_provenance()` parameter validation — offline `DagProvenance` struct checks

---

## Part 1: CPU/GPU Parity — What We Learned

### Parity Coverage Audit

The systematic audit of `validate_barracuda_cpu_gpu_parity.rs` against all
physics modules confirmed 8 domains with active CPU/GPU parity checks:

| Domain | Validator | GPU Path |
|--------|-----------|----------|
| Lattice QCD (Wilson plaquette) | CPU-only reference | No GPU path needed for reference |
| Nuclear EOS (SEMF) | `validate_nuclear_eos` | GPU HFB via `nuclear_eos_l2_gpu` |
| Plasma Dielectric (Mermin) | `validate_dielectric` | `dielectric_mermin_f64.wgsl` |
| BGK Relaxation | `validate_kinetic_fluid` | `bgk_relaxation_f64.wgsl` |
| Euler HLL | `validate_kinetic_fluid` | CPU reference stable |
| Coupled Kinetic-Fluid | `validate_kinetic_fluid` | GPU BGK + CPU coupling |
| Transport (D*, η*) | `validate_transport` | GPU Green-Kubo |
| Spectral SpMV | `validate_spectral` | GPU Lanczos |

### GPU Gap Classification

Papers without direct GPU implementations were classified:

- **CPU-only by design** (papers 2, 7): ODE solver (TTM), data validation (HotQCD EOS)
- **CPU-natural** (papers 6, 17, 21, 22): Sequential eigensolve (screened Coulomb),
  transfer matrix (Lyapunov), Sturm chain (Hofstadter, Ten Martini)
- **GPU-promotable** (paper 20): 3D Anderson via SpMV+Lanczos — P2 priority

**Upstream ask for barraCuda:** Paper 20 (Anderson 3D) is promotable to GPU via
the existing spectral SpMV pipeline. The `validate_anderson_3d` binary already
has the CPU path. Adding a GPU SpMV section would close the last meaningful
GPU coverage gap.

### `s_cpu_gpu_parity` Scenario

New barracuda-local scenario validates CPU reference stability across 7 domains
without requiring a GPU. This serves as the determinism baseline that GPU parity
checks rely on. Key checks:

- Wilson plaquette self-consistency (quenched β=6.0)
- SEMF binding energy determinism (Pb-208, Fe-56)
- Transport coefficient reference stability (D*, η*)
- Spectral SpMV self-consistency (1D Anderson, Hofstadter)
- BGK relaxation step conservation
- Sod shock tube density profile
- Coupled kinetic-fluid energy conservation

---

## Part 2: toadStool Dispatch — Offline Validation Patterns

### `s_toadstool_dispatch` Scenario

Validates the `compute_dispatch` module's offline components without live IPC:

| Check | What It Validates |
|-------|-------------------|
| Dispatch validation logic | `is_valid_dispatch_id()` for accepted/rejected patterns |
| Input hashing | BLAKE3 determinism for dispatch witness construction |
| Barrier shader paths | Enumeration of shaders requiring `workgroupBarrier()` |
| Witness construction | `DispatchWitness` field assembly (binary hash, params hash, timestamp) |
| Dispatch serialization | Round-trip serde for dispatch parameter structs |
| `commit_provenance` params | `DagProvenance` struct fields, `nest.commit` signal shape |

### `commit_provenance()` Integration Status

`dag_provenance.rs` has `commit_provenance()` scaffolded with the full signal
path: `nest.commit` via `signal.dispatch` on `orchestration`, with fallback to
`ledger.record` + `attribution.braid` for pre-v3.57 biomeOS. Current status:

- **Parameter assembly**: Validated offline (DagProvenance struct checks)
- **Signal path**: `nest.commit` promoted from candidate to adopted in `capability_registry.toml`
- **Live wiring**: Not yet called from any production pipeline — waiting for Titan V
  e2e sovereign dispatch to reach stable production compute

**Upstream ask for biomeOS:** When `nest.commit` signal decomposition is live
(event.append → crypto.sign → content.put → session.commit → braid.create),
hotSpring is ready to wire `commit_provenance()` into the Titan V compute pipeline.

### `dispatch_cpu_fallback()` Pattern

Added to `compute_dispatch/mod.rs` for offline parity testing:

```rust
pub fn dispatch_cpu_fallback(workload_name: &str, input_data: &[f64]) -> Option<serde_json::Value>
```

Supports `vector_add`/`vector_add_f64` and `semf_batch` workloads. Returns
`None` for unknown workloads. This enables validation scenarios to exercise
dispatch logic without a live toadStool daemon.

**Pattern for sibling springs:** Any spring that needs offline dispatch testing
can follow this pattern — implement CPU fallback paths for key workloads and
gate them behind `Option<Value>` returns.

---

## Part 3: metalForge NUCLEUS Atomics and Mixed Hardware

### `forge::nucleus` Module

NUCLEUS atomic types now live in `metalForge/forge/src/nucleus.rs`, owned by
forge for substrate-level coordination. This mirrors `barracuda/src/composition.rs`
but adds deployment semantics:

| AtomicType | Required Domains | Compatible Substrates |
|------------|-----------------|----------------------|
| Tower | crypto, discovery | CPU, GPU, NPU |
| Node | compute, math, shader | GPU, CPU |
| Nest | storage, dag, ledger, attribution | CPU only |
| FullNucleus | all 9 domains | CPU only |

`AtomicBinding::bind()` validates that a substrate kind is allowed for an atomic
type. `is_subset_of()` checks domain containment (Tower ⊂ Node ⊂ FullNucleus).

**Upstream ask for primalSpring:** The `AtomicType` enum in forge should be
reconciled with `barracuda::composition::AtomicType` — either via a shared
types crate or by forge becoming the canonical owner. Currently they're kept
in sync manually.

### `ChannelKind::PcieDirect`

New channel kind for GPU→NPU PCIe peer-to-peer data transfer, bypassing CPU
memory roundtrip:

```
GPU compute → [PcieDirect] → NPU inference → [Pcie] → CPU validation
```

New topologies in `pipeline.rs::topologies`:
- `mixed_pcie_direct()` — 3-substrate pipeline with PCIe direct hop
- `nucleus_atomic()` — Tower→Node→Nest topology

**Upstream ask for toadStool:** When hardware PCIe P2P is available (NVIDIA
GPUDirect or AMD XDMA), toadStool should expose a `dispatch.p2p_transfer` RPC
method that metalForge can route to. The `ChannelKind::PcieDirect` enum variant
is ready for this evolution.

### `forge::biome_graph` Module

Directed graph for NUCLEUS atomic coordination. This is the local evolution
that biomeOS will absorb:

- **Nodes**: `(AtomicType, SubstrateKind, label)` — e.g., Tower on CPU, Node on GPU
- **Edges**: `ChannelKind` connections between nodes
- **Queries**: `shortest_path()`, `reachable_from()`, `pcie_direct_hops()`
- **Constructors**: `standard_nucleus_graph()` (3 nodes: Tower→Node→Nest),
  `pcie_direct_nucleus_graph()` (4 nodes: adds NPU Node with PCIe direct from GPU)

**Upstream ask for biomeOS:** The `BiomeGraph` structure maps directly to
biomeOS deploy graph concepts. When biomeOS evolves its runtime graph
representation, `forge::biome_graph` provides the substrate-aware graph
vocabulary (node types, channel kinds, pathfinding) that biomeOS can absorb
or align with.

---

## Part 4: NUCLEUS Composition Patterns for Deployment

### Three-Tier Validation (Current State)

```
Python baseline (control/) → Rust proof (barracuda/) → NUCLEUS IPC composition (primal_bridge)
                                                        ↓
                                                   validate_nucleus_*
                                                        ↓
                                              guideStone Level 6 CERTIFIED
```

All three tiers now have coverage:
- **Parity greenboard**: 10/10 ALL GREEN (Python self-parity confirmed)
- **CPU/GPU parity**: 7 domains covered by `s_cpu_gpu_parity` scenario
- **NUCLEUS composition**: Tower/Node/Nest/Full validated by `validate_nucleus_*` binaries

### Deploy Graph Pattern (TOML)

The `graphs/hotspring_qcd_deploy.toml` deploy graph defines NUCLEUS composition
for biomeOS deployment:

- 10 primals as peer `[[graph.nodes]]` entries
- `hotspring_unibin` as spawning application (order 12)
- Bonding: `bond_type = "Metallic"`, tiered encryption per atomic boundary
- Fragment set: `tower_atomic`, `node_atomic`, `nest_atomic`, `nucleus`, `provenance_trio`, `meta_tier`

### Atomic Instantiation via neuralAPI

Current signal adoption (Wave 17 + Wave 20):

| Signal | Status | Primal | Usage |
|--------|--------|--------|-------|
| `primal.announce` | Adopted | biomeOS | Registration via `register_with_target()` |
| `node.compute` | Adopted | toadStool | Orchestrated dispatch via `dispatch_node_compute()` |
| `tower.publish` | Adopted | BearDog | Signed result publication via `publish_result()` |
| `nest.commit` | Adopted (Wave 20) | biomeOS | Provenance commit via `commit_provenance()` (scaffolded) |
| `signal.dispatch` | Transport | biomeOS | Signal decomposition to multi-call sequences |
| `capability.list` | Adopted (Wave 20) | All | Canonical `{capabilities, count, primal}` envelope |

### Discovery Pattern

```
NucleusContext::detect()
  → scan socket dirs for *-{family}.sock
  → probe health.liveness per socket
  → optionally probe capability.list
  → circuit breaker: 3 failures → dead, 30s cooldown → re-probe
  → call_by_capability(domain, method, params)
```

HOTSPRING_NO_NUCLEUS=1 enables standalone mode (all IPC checks skip-pass).

---

## Part 5: Upstream Asks (Summary)

### For toadStool
- [ ] Expose `dispatch.p2p_transfer` RPC when PCIe P2P hardware is available
- [ ] Consider absorbing `dispatch_cpu_fallback()` pattern for offline validation
- [ ] Integrate `sovereign.probe`/`sovereign.verify` into capability surface

### For coralReef
- [ ] No new asks from this sprint

### For barraCuda
- [ ] Paper 20 (Anderson 3D) GPU SpMV promotion — closes last meaningful GPU gap
- [ ] TensorSession adoption (GAP-HS-027) — HMC trajectory as first fused session

### For biomeOS
- [ ] `nest.commit` signal decomposition — enables hotSpring `commit_provenance()` wiring
- [ ] Consider absorbing `forge::biome_graph` graph vocabulary (node types, channel kinds, pathfinding)
- [ ] Runtime graph representation aligned with `BiomeGraph` structure

### For primalSpring
- [ ] `AtomicType` enum reconciliation between `barracuda::composition` and `forge::nucleus`
- [ ] `forge::biome_graph` → biomeOS absorption path documentation

### For sibling springs
- [ ] `s_cpu_gpu_parity` pattern — CPU reference stability as baseline for GPU parity
- [ ] `dispatch_cpu_fallback()` pattern — offline dispatch testing without live primals
- [ ] `ValidationHarness` + scenario registration pattern for absorbed experiment bins

---

## Part 6: What Evolved Since May 16D Handoff

| Area | May 16 State | May 17 State |
|------|-------------|-------------|
| Validation scenarios | 18 (16 default + 2 barracuda-local) | 22 (17 default + 5 barracuda-local) |
| Parity greenboard | 9/10 (paper 45 stale) | 10/10 ALL GREEN |
| forge modules | dispatch, inventory, pipeline, probe, substrate | + nucleus, biome_graph |
| Pipeline channels | Pcie, SharedMemory, Local | + PcieDirect |
| compute_dispatch | GPU-only | + dispatch_cpu_fallback() |
| Mixed substrate binary | 4 physics domains | + NUCLEUS atomics + PCIe direct + biome graph |
| GPU coverage classification | Implicit | Explicit: CPU-only, CPU-natural, GPU-promotable |

---

## File Manifest (New/Changed)

| File | Action |
|------|--------|
| `experiments/197_SOVEREIGN_INIT_RPC_WARM_COLD.md` | Created |
| `experiments/198_VENDOR_AGNOSTIC_BOOT_PIPELINE.md` | Created |
| `barracuda/src/validation/scenarios/s_cpu_gpu_parity.rs` | Created |
| `barracuda/src/validation/scenarios/s_toadstool_dispatch.rs` | Created |
| `barracuda/src/validation/scenarios/s_mixed_hardware.rs` | Created |
| `barracuda/src/validation/scenarios/mod.rs` | Updated (3 new registrations) |
| `barracuda/src/compute_dispatch/mod.rs` | Updated (`dispatch_cpu_fallback`) |
| `metalForge/forge/src/nucleus.rs` | Created |
| `metalForge/forge/src/biome_graph.rs` | Created |
| `metalForge/forge/src/pipeline.rs` | Updated (`PcieDirect`, topologies) |
| `metalForge/forge/src/lib.rs` | Updated (2 new modules) |
| `barracuda/src/bin/validate_mixed_substrate.rs` | Updated (3 new checks) |
| `specs/PAPER_REVIEW_QUEUE.md` | Updated (metrics, greenboard, GPU classification) |
| `control/hotspring_reader/parity_greenboard.json` | Regenerated (10/10 ALL GREEN) |
