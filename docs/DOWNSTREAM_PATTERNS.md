# Downstream Pattern Integration — hotSpring

**Date:** May 12, 2026
**Repos audited:** `gardens/projectNUCLEUS`, `gardens/foundation`

---

## projectNUCLEUS

**Role:** Deploy/validate product — primals, curated deploy graphs, toadStool
workload TOMLs, gate manifests, operational scripts.

**hotSpring integration points:**

- `workloads/hotspring/` — **6 workload TOMLs** covering 3-tier compute ladder
  (CPU/GPU/sovereign): `hotspring-md-validation.toml`, `hotspring-lattice-qcd.toml`,
  `hotspring-ltee-anderson.toml`, `hotspring-composition-health.toml`,
  `hotspring-gpu-sovereign-dispatch.toml`, `hotspring-sovereign-roundtrip.toml`.
  Each documents `compute_tier`, `ladder`, and `spring = "hotSpring"` metadata
- `graphs/node_atomic.toml` — positions hotSpring as consumer of the Node
  atomic (compute) pattern alongside neuralSpring, wetSpring, ludoSpring
- `specs/LIVE_SCIENCE_API.md` — lists hotSpring in the "which springs are
  green" dashboard contract (currently `checks_passing: 0` — needs update
  after gate validation runs)
- `deploy/gate_manifest.toml` — maps gates to atomics and graphs

**Patterns absorbed:**

1. **UniBin validation surface** — `hotspring_unibin validate --scenario <name>`
   is the stable contract for plasmidBin and toadStool workloads. Registered
   scenarios: `semf-parity`, `lattice-plaquette`, `md-yukawa-ocp` (config
   smoke), `sarkas-yukawa-md` (foundation-grade), `composition-health`,
   `tolerance-ordering`, `ltee-anderson`. The `spectral-lanczos` scenario
   requires `barracuda-local` and is not available in `primal-proof` builds.
   8 scenarios registered (7 default + 1 barracuda-local)
2. **Workload TOML contract** — `[metadata]`, `[execution]`, `[resources]`,
   `[security]` sections; `[provenance]` pending toadStool Gap 5 resolution
3. **Path portability** — workload uses `$SPRINGS_ROOT` (verify expansion
   with live toadStool; may need wrapper script per Gap 8)
4. **Wire Standard L3** — `bonding_policy`, `tcp_fallback_port`, capability
   lists, `fallback = "skip"` for optional primals

---

## foundation

**Role:** Scientific "soil" — lineage maps, data source/target TOMLs,
thread indexes, validation graphs.

**hotSpring integration points:**

- `lineage/THREAD_INDEX.toml` — Thread 2 (Plasma Physics / Lattice QCD)
  lists `springs = ["hotSpring"]`, status upgraded to "active"
- `data/sources/thread02_plasma.toml` — 18 literature anchors with notes
  referencing hotSpring Tier 4 validation targets
- `data/targets/thread02_plasma_targets.toml` — **12 Sarkas Yukawa MD
  validation targets** (energy drift, RDF, D*, viscosity, Daligault fit)
  with `validated = true` and provenance
- `data/targets/thread01_wcm_targets.toml` — cross-thread lineage with
  `spring = "hotSpring"` for specific targets
- `graphs/foundation_validation.toml` — validation-oriented composition
  graph (Tower + Node + Nest + optional petalTongue/squirrel)
- `validation/plasma-20260511/` — first foundation validation run
  (12/12 PASS, provenance manifest + validation summary)

**Patterns absorbed:**

1. **Sediment layer lifecycle** — spring proves result, binary to plasmidBin,
   workload TOML, `foundation_validate.sh` wraps with provenance
2. **Thread expression** — Thread 2 expression doc created
   (`PLASMA_QCD_SOVEREIGN_GPU.md`), referenced from `thread02_plasma.toml`
   and `THREAD_INDEX.toml`
3. **Data anchor registration** — validation runs should register BLAKE3
   hashes in NestGate, feeding `validation/run-*` traceability

---

## Action Items for hotSpring

| Priority | Action | Status |
|----------|--------|--------|
| P0 | Build UniBin release binary for plasmidBin | **Done** (3.3M, v0.6.32) |
| P0 | Fix workload binary name + path portability | **Done** (UniBin + `$SPRINGS_ROOT`) |
| P0 | Seed foundation Thread 2 validation targets | **Done** (12 targets, 12/12 PASS) |
| P1 | Update Live Science API counts after gate validation runs | Pending |
| P1 | Create Thread 2 expression doc for foundation | **Done** (`PLASMA_QCD_SOVEREIGN_GPU.md`) |
| P1 | Create additional NUCLEUS workloads (nuclear EOS, spectral) | Future |
| P1 | Add foundation Thread 2 workload TOML | **Done** (`workloads/thread02_plasma/hs-sarkas-md.toml`) |
| P1 | Unblock lithoSpore module 7 with B2 expected JSON | **Ready** (artifacts shipped, litho needs integration) |
| P1 | Add scenario workloads for new registry entries (screened-coulomb, transport, gradient-flow, dielectric) | Pending |
| P1 | Upstream LTEE doc alignment (foundation STARTED→COMPLETE, litho QUEUED→COMPLETE) | Needs cross-repo PRs |
| P2 | Register validation BLAKE3 hashes in NestGate | Future |
| P2 | Populate `blake3`/`retrieved` in thread02_plasma.toml sources | Future |
| P2 | Verify `$SPRINGS_ROOT` expansion with live toadStool dispatch | Future |
| P2 | Foundation Phase 5 workload scan: consolidate `workloads/hotspring/` under `thread02_plasma/` | Pending |
| P2 | Foundation Phase 6 target schema alignment (`metric` vs `expected_value`) | Pending |
