# SPDX-License-Identifier: AGPL-3.0-or-later

# hotSpring Cross-Tier Parity

**Spring:** hotSpring v0.6.32
**Date:** May 17, 2026
**Pattern:** primalSpring/docs/VALIDATION_TIERS.md § Cross-Tier Parity Pattern
**Reference:** ecoPrimals/infra/wateringHole/PROVENANCE_TRIO_INTEGRATION_GUIDE.md

---

## Three-Layer Proof

| Tier | Role | hotSpring Implementation |
|------|------|--------------------------|
| **Tier 1** | Python baseline confirms published science | `control/` scripts (10 domains) |
| **Tier 2** | Rust implementation matches Python | `validate_*` binaries + `s_anderson_parity` scenario |
| **Tier 3** | Provenance chain records who ran what | `dag_provenance.rs` → trio (rhizoCrypt/loamSpine/sweetGrass) |

---

## Parity Coverage

### Paper-Level Greenboard (Tier 1 ↔ Tier 2)

Python control scripts produce reference JSON; `run_all_parity.py --self-parity`
compares against Rust validation outputs. The greenboard covers 10 papers:

| Paper | Domain | Python Control | Rust Validator | Checks | Status |
|-------|--------|----------------|----------------|--------|--------|
| 6 | Screened Coulomb | `screened_coulomb/reference_eigenvalues.json` | `validate_screened_coulomb` | 34 | ALL PASS (0.0 delta) |
| 8 | Pure gauge SU(3) | `lattice_qcd/quenched_beta_scan.py` | `validate_pure_gauge` | 9 | ALL PASS |
| 9 | Production QCD | `lattice_qcd/quenched_beta_scan.py` | `validate_production_qcd` | 10 | ALL PASS |
| 10 | Dynamical fermion QCD | `lattice_qcd/dynamical_fermion_control.py` | `validate_dynamical_qcd` | 2 | ALL PASS |
| 11 | HVP g-2 correlator | `lattice_qcd/hvp_correlator_control.py` | `validate_hvp` | 4 | ALL PASS |
| 12 | Freeze-out susceptibility | `lattice_qcd/freeze_out_control.py` | `validate_freeze_out` | 3 | ALL PASS |
| 13 | Abelian Higgs U(1) | `abelian_higgs/abelian_higgs_reference.json` | `validate_abelian_higgs` | 12 | ALL PASS |
| 43 | SU(3) gradient flow | `gradient_flow/gradient_flow_control.py` | `validate_gradient_flow` | 0 (structural) | ALL PASS |
| 44 | Conservative dielectric | `bgk_dielectric/bgk_dielectric_control.py` | `validate_bgk_dielectric` | 0 (structural) | ALL PASS |
| 45 | Kinetic-fluid coupling | `kinetic_fluid/kinetic_fluid_control.py` | `validate_kinetic_fluid` | 0 (structural) | ALL PASS |

**Greenboard**: `control/hotspring_reader/parity_greenboard.json` — **10/10 ALL GREEN**

### Anderson / Spectral Theory Parity (lithoSpore Pattern)

Follows the lithoSpore `litho parity` format — structured JSON comparison
with documented tolerances per observable.

**Python side:** `spectral_parity.py` compares `spectral_control.json`
against Rust reference values.

| Check | Python Value | Rust Reference | Error | Tolerance | Status |
|-------|-------------|----------------|-------|-----------|--------|
| Herman/Lyapunov λ=2 | 0.6931710 | ln(2) = 0.6931472 | 3.4e-5 rel | 2e-2 | PASS |
| Level statistics (Poisson) | ⟨r⟩ = 0.38893 | 0.38629 | 2.6e-3 abs | 5e-2 | PASS |
| 3D bandwidth OBC | 11.276311449430906 | 11.276311449430901 | 4.7e-16 rel | 1e-8 | PASS |
| GOE→Poisson monotonic | true | true | — | boolean | PASS |
| Dimensional hierarchy | 1D < 2D < 3D | true | — | boolean | PASS |
| Mobility edge (center > edge) | 0.5104 > 0.4962 | true | — | boolean | PASS |

**Parity report:** `control/spectral_theory/results/spectral_parity.json` — **6/6 ALL PASS**

**Rust side:** `s_anderson_parity` validation scenario (23 scenarios, `barracuda-local`)
exercises the same algorithms in-process with 14 assertions across 5 domains.

### CPU/GPU Parity

`s_cpu_gpu_parity` scenario validates CPU reference stability across 7 physics
domains that have GPU counterparts:

1. Lattice QCD (plaquette, delta-H)
2. SEMF (Pb-208, Fe-56 binding energy)
3. Transport coefficients (Stanton-Murillo viscosity)
4. Spectral SpMV (Anderson 2D sparse matrix-vector)
5. BGK relaxation (Boltzmann transport)
6. Euler shock tube (Sod problem)
7. Coupled kinetic-fluid (energy conservation)

---

## Tolerance Documentation

Tolerances are justified per domain:

| Domain | Tolerance | Justification |
|--------|-----------|---------------|
| Screened Coulomb eigenvalues | 1e-10 abs | Sturm vs scipy on identical grid |
| QCD plaquettes | 5e-3 abs | Thermalization noise on 4^4 lattice |
| HVP correlator | 0.5 abs | Order-of-magnitude on same lattice |
| Anderson Lyapunov | 2e-2 rel | Transfer matrix statistical convergence |
| Anderson level stats | 5e-2 abs | Finite-size / realization count |
| 3D bandwidth | 1e-8 rel | Clean lattice, no disorder noise |

---

## How to Run

```bash
# Paper-level greenboard (Tier 1 ↔ Tier 2)
cd control && python3 run_all_parity.py --self-parity

# Anderson spectral parity (lithoSpore pattern)
cd control/spectral_theory && python3 scripts/spectral_parity.py

# Regenerate Python baseline first
cd control/spectral_theory && python3 scripts/spectral_parity.py --regen

# Rust-side anderson parity scenario
cd barracuda && cargo test --lib s_anderson_parity --features barracuda-local

# CPU/GPU parity scenario
cd barracuda && cargo test --lib s_cpu_gpu_parity --features barracuda-local
```

---

## Provenance Chain (Tier 3)

When the trio is available, `commit_provenance()` records the validation
result into the provenance chain:

1. **rhizoCrypt** — DAG session with per-phase events and merkle root
2. **loamSpine** — ledger entry with result hash and timestamp
3. **sweetGrass** — attribution braid linking validation to operator

The commit flow is **not atomic** (per trio transaction semantics):
- DAG without braid = valid partial provenance
- Partial state reported via `primals_reached` in return value
- Domain logic never gates on provenance availability

See `docs/DEGRADATION_BEHAVIOR.md` for what happens when trio primals
are unreachable.

---

## References

- `control/hotspring_reader/parity_greenboard.json` — paper-level greenboard
- `control/spectral_theory/results/spectral_parity.json` — Anderson parity report
- `barracuda/src/validation/scenarios/s_anderson_parity.rs` — Rust parity scenario
- `barracuda/src/validation/scenarios/s_cpu_gpu_parity.rs` — CPU/GPU parity scenario
- `barracuda/src/dag_provenance.rs` — trio commit with `primals_reached`
- `primalSpring/docs/VALIDATION_TIERS.md` — upstream tier definitions
- `ecoPrimals/infra/wateringHole/PROVENANCE_TRIO_INTEGRATION_GUIDE.md` — trio semantics
