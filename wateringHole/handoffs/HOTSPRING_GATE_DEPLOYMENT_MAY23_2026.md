# hotSpring — biomeGate Covalent Deployment Status

**Date:** May 23, 2026
**From:** hotSpring v0.6.32
**To:** primalSpring (coordination), cellMembrane (infra), projectNUCLEUS (pipeline)
**Status:** READY — sovereign GPU stack operational, plasmidBin deployment path validated
**Gate:** biomeGate (sole tenant, HBM2 dedicated)
**Experiments:** 219 total, 24 validation scenarios

---

## Gate Assignment Confirmation

| Field | Value |
|-------|-------|
| Gate | **biomeGate** |
| Hardware | TR 3970X, 2× Titan V (GV100, 12GB HBM2 each), RTX 5060 (display) |
| RAM | 256 GB DDR4 |
| Tenant | hotSpring (sole — HBM2 dedicated) |
| IOMMU | Group 69 (Titan V #1), isolated root complex |
| Sovereign GPU | VFIO at boot (both Titan Vs), ember daemon managed |

### Hardware Inventory Reconciliation

The upstream audit lists "Titan V + K80" for biomeGate. **Correction:**
K80 was retired (caught fire, Exp 199). Replaced by second Titan V.
Actual fleet per `scripts/boot/glowplug.toml`:

| Device | BDF | Role | Boot Personality |
|--------|-----|------|-----------------|
| RTX 5060 (GB206) | 0000:21:00.0 | display | nvidia (LOCKED) |
| Titan V #1 (GV100) | 0000:02:00.0 | oracle | vfio |
| Titan V #2 (GV100) | 0000:03:00.0 | compute | vfio |

---

## Proto-Nucleate Composition

Source: `primalSpring/graphs/downstream/downstream_manifest.toml` (spring_name = "hotspring")

| Atomic | Primals | Role |
|--------|---------|------|
| Tower | BearDog, Songbird | Security, discovery |
| Node | coralReef, toadStool, barraCuda | Shader compilation, GPU dispatch, tensor math |
| Nest | NestGate, rhizoCrypt, loamSpine, sweetGrass | Storage, DAG, ledger, attribution |

**Profile:** proton_heavy (Node-dominant)
**Fragments:** tower_atomic, node_atomic, nest_atomic
**Bonding:** Metallic / InternalNucleus
**Security:** BTSP enforced, UDS-only, 0 TCP ports

### Validation Capabilities Required

```
tensor.matmul, tensor.create, tensor.add, tensor.scale
stats.mean, stats.std_dev
compute.dispatch
crypto.hash
tolerances.get
validate.gpu_stack
```

---

## Deployment Flow

### Standard Path (plasmidBin)

```bash
# 1. Fetch primals (from primalSpring tools or infra/plasmidBin)
cd ../primalSpring/tools
./fetch_primals.sh --all    # pull v2026.05.23 binaries (13 primals)

# 2. Launch NUCLEUS (niche-hotspring = 9 required primals)
export FAMILY_ID="hotspring-biomeGate"
cd ../../infra/plasmidBin
./nucleus_launcher.sh --family-id "$FAMILY_ID" --composition niche-hotspring

# 3. Validate composition
./validate_composition.sh niche-hotspring --live    # exit 0 = PASS

# 4. Run primalspring certify
cd ../../springs/primalSpring/ecoPrimal
cargo run --release --bin primalspring_unibin -- certify

# 5. Run hotspring primal proof
cd ../../springs/hotSpring
FAMILY_ID="hotspring-biomeGate" ./scripts/validate-primal-proof.sh --full
```

### Sovereign GPU Stack (biomeGate-specific)

```bash
# Boot-time: VFIO binds Titan Vs, ember daemon manages fleet
sudo systemctl start toadstool-ember.service
sudo systemctl start toadstool-glowplug.service

# Sovereign dispatch validation
curl -s http://localhost:7700/rpc -d '{
  "jsonrpc":"2.0","id":1,
  "method":"sovereign.warm_handoff",
  "params":{"bdf":"0000:02:00.0","strategy":"nvidia_catalyst_titanv"}
}'
```

---

## Validation Status

### What PASSES Today

| Check | Status | Evidence |
|-------|--------|----------|
| Deploy graphs (7) comply with Dark Forest | PASS | `s_dark_forest_gate` scenario — all 5 pillars |
| Anderson spectral parity (Python↔Rust) | PASS | `s_anderson_parity` — 6/6 observables |
| CPU/GPU parity (f64 WGSL) | PASS | 9/9 Yukawa OCP cases, 0.000% drift |
| Sovereign GPU dispatch (VFIO) | PASS | Exp 217-219, toadStool ember + catalyst |
| Cross-architecture (x86_64 + aarch64) | PASS | Phase G — 59/59 × 5 substrates |
| guideStone certification (bare) | PASS | L5, 5/5 properties |
| 24 validation scenarios compile | PASS | `cargo build --release` clean |
| 993 lib tests | PASS | `cargo test --lib` |
| Trio provenance (`primals_reached`) | PASS | `dag_provenance.rs` reports trio contacts |

### What Requires Live NUCLEUS

| Check | Requires | Status |
|-------|----------|--------|
| `validate-primal-proof.sh --full` | 9 primals on UDS | PENDING — needs `nucleus_launcher.sh` run |
| `primalspring_unibin certify` | biomeOS + NUCLEUS | PENDING — composition pre-flight |
| `validate_composition.sh --live` | 9 primals health triad | PENDING — plasmidBin binaries on gate |
| Node atomic live wiring | toadStool + barraCuda + coralReef | PENDING — `compute.dispatch` E2E |
| Nest atomic live wiring | rhizoCrypt + loamSpine + sweetGrass | PENDING — trio commit E2E |

### Rewiring Status

| Metric | Value |
|--------|-------|
| Rewiring tier | 3 (IPC parity active, library + IPC dual-lane) |
| IPC coverage | ~15-25% (highest density in composition.rs path) |
| petalTongue | Not wired |
| sweetGrass | Not wired |

---

## Gaps Discovered

### GAP-HS-008: biomeGate Hardware Doc Drift

- **Severity:** Low (documentation only)
- **Description:** Upstream audit and `HARDWARE.md` still list K80 at biomeGate.
  Actual: K80 retired Exp 199, replaced by second Titan V.
- **Action:** Update upstream gate assignment table.

### GAP-HS-009: skunkBat in niche-hotspring vs Proto-Nucleate

- **Severity:** Low
- **Description:** `infra/plasmidBin/ports.env` includes `skunkbat` in
  `niche-hotspring`, but proto-nucleate `depends_on` does not list it.
  Deploy graph includes it as `required = false` (optional defense node).
- **Action:** Reconcile — either add skunkbat to proto-nucleate depends_on
  or remove from niche-hotspring in ports.env. Deploy graph has it correctly
  as optional, which is the intended semantics.

### GAP-HS-010: Sovereign GPU Stack Not in validate-primal-proof.sh

- **Severity:** Medium
- **Description:** The VFIO/ember/glowplug sovereign dispatch pipeline
  (Exp 110-219) is validated by toadStool experiments and systemd services,
  but not integrated into the standard `validate-primal-proof.sh` flow.
  A live NUCLEUS deployment on biomeGate needs both: plasmidBin primals
  (standard path) AND sovereign GPU stack (biomeGate-specific path).
- **Action:** Add optional `--sovereign` flag to validate-primal-proof.sh
  that probes `sovereign.warm_handoff` status when on biomeGate.

### GAP-HS-011: Dual-Gate Operational Clarity

- **Severity:** Low
- **Description:** hotSpring operates on both strandGate (dev/AMD+NV parity)
  and biomeGate (sovereign HBM2 dispatch). No single runbook merges both
  contexts. Operators need to know which gate they're on and what to run.
- **Action:** Document in `scripts/boot/README.md` or equivalent.

---

## Contention Assessment

biomeGate is **sole-tenant** (hotSpring only). No multi-domain contention
expected. This is intentional — HBM2 workloads (QCD lattice sweeps) are
resource-exclusive. Contention scenarios:

| Potential Issue | Assessment |
|----------------|-----------|
| Socket conflicts | None — sole tenant, full UDS namespace |
| Capability collisions | None — no competing spring |
| Resource exhaustion | Managed — toadStool ember daemon arbitrates GPU access |
| VFIO group conflicts | None — both Titan Vs in separate IOMMU groups |

---

## Next Steps

1. **Pull v2026.05.23 plasmidBin binaries** on biomeGate via `fetch_primals.sh --all`
2. **Start NUCLEUS** via `nucleus_launcher.sh --composition niche-hotspring`
3. **Run full validation chain** (primalspring certify + hotspring validate)
4. **Document any IPC failures** — hand back to primalSpring via `PRIMAL_GAPS.md`
5. **Confirm sovereign stack compose** — toadStool ember alongside NUCLEUS primals
6. **Post follow-up handoff** with live validation results

---

## References

- Proto-nucleate: `primalSpring/graphs/downstream/downstream_manifest.toml`
- Deploy graphs: `hotSpring/graphs/hotspring_*_deploy.toml` (7 files)
- Sovereign GPU config: `hotSpring/scripts/boot/glowplug.toml`
- Primal proof script: `hotSpring/scripts/validate-primal-proof.sh`
- Composition script: `hotSpring/tools/hotspring_composition.sh`
- Gap registry: `hotSpring/docs/PRIMAL_GAPS.md`
- Deployment standard: `infra/wateringHole/DEPLOYMENT_VALIDATION_STANDARD.md`
- NUCLEUS alignment: `infra/wateringHole/NUCLEUS_SPRING_ALIGNMENT.md`
