# Downstream Pattern Integration — hotSpring

**Date:** May 8, 2026
**Repos audited:** `gardens/projectNUCLEUS`, `gardens/foundation`

---

## projectNUCLEUS

**Role:** Deploy/validate product — primals, curated deploy graphs, toadStool
workload TOMLs, gate manifests, operational scripts.

**hotSpring integration points:**

- `workloads/hotspring/hotspring-md-validation.toml` — toadStool workload
  pointing at `validate_sarkas_md` (binary name mismatch: should be
  `validate_md` or `sarkas_gpu` from barracuda crate)
- `graphs/node_atomic.toml` — positions hotSpring as consumer of the Node
  atomic (compute) pattern alongside neuralSpring, wetSpring, ludoSpring
- `specs/LIVE_SCIENCE_API.md` — lists hotSpring in the "which springs are
  green" dashboard contract (currently `checks_passing: 0`)
- `deploy/gate_manifest.toml` — maps gates to atomics and graphs

**Patterns to absorb:**

1. **Stable validation binary surface** — expose small, stable set of
   `validate_*` binaries (or `validate_all` umbrella) for plasmidBin
2. **Workload TOML contract** — `[metadata]`, `[execution]`, `[resources]`,
   `[security]` sections; prepare for `[provenance]`
3. **Path portability** — use `ECOPRIMALS_ROOT` or wrapper scripts instead
   of hardcoded `/home/irongate/...` paths
4. **Wire Standard L3** — `bonding_policy`, `tcp_fallback_port`, capability
   lists, `fallback = "skip"` for optional primals

---

## foundation

**Role:** Scientific "soil" — lineage maps, data source/target TOMLs,
thread indexes, validation graphs.

**hotSpring integration points:**

- `lineage/THREAD_INDEX.toml` — Thread 2 (Plasma Physics / Lattice QCD)
  lists `springs = ["hotSpring"]`
- `data/sources/thread02_plasma.toml` — literature anchors with notes
  referencing hotSpring Tier 4 validation targets
- `data/targets/thread01_wcm_targets.toml` — cross-thread lineage with
  `spring = "hotSpring"` for specific targets
- `graphs/foundation_validation.toml` — validation-oriented composition
  graph (Tower + Node + Nest + optional petalTongue/squirrel)

**Patterns to absorb:**

1. **Sediment layer lifecycle** — spring proves result, binary to plasmidBin,
   workload TOML, `foundation_validate.sh` wraps with provenance
2. **Thread expression** — Thread 2 expression is empty (`expression = ""`);
   hotSpring should contribute plasma/QCD expression material
3. **Data anchor registration** — validation runs should register BLAKE3
   hashes in NestGate, feeding `validation/run-*` traceability

---

## Action Items for hotSpring

| Priority | Action | Status |
|----------|--------|--------|
| P0 | Fix workload binary name (`validate_sarkas_md` → `validate_md`) | Pending (upstream projectNUCLEUS) |
| P0 | Update Live Science API counts after validation suite passes | Pending |
| P1 | Create additional workload TOMLs for nuclear_eos, lattice_qcd, spectral | Future |
| P1 | Write Thread 2 expression doc for foundation | Future |
| P2 | Register validation BLAKE3 hashes in NestGate | Future |
