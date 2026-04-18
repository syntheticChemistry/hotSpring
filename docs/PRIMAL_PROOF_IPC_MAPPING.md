# hotSpring — Primal Proof IPC Mapping

**Spring:** hotSpring v0.6.32
**Purpose:** Maps hotSpring domain science to barraCuda/primal JSON-RPC methods for Level 5 validation
**Date:** April 17, 2026
**License:** AGPL-3.0-or-later

---

## Overview

For the primal proof (Level 5), hotSpring must validate that peer-reviewed science
produces correct results when domain math is routed through NUCLEUS primals over IPC,
rather than called in-process via library imports. This document maps each domain
science path to the specific primal JSON-RPC methods it exercises.

The validation harness (`validate_primal_proof`) calls these methods against a running
NUCLEUS deployed from plasmidBin ecobins, compares results against Python baselines,
and reports PASS/FAIL/SKIP per capability.

---

## barraCuda IPC Methods (32 methods, UDS JSON-RPC 2.0)

The barraCuda ecobin primal exposes:

| Category | Methods |
|----------|---------|
| **Tensor** | `tensor.matmul`, `tensor.create`, `tensor.add`, `tensor.scale`, `tensor.clamp`, `tensor.reduce`, `tensor.sigmoid`, `tensor.batch.submit` |
| **Statistics** | `stats.mean`, `stats.std_dev`, `stats.weighted_mean` |
| **Compute** | `compute.dispatch` |
| **Noise** | `noise.perlin2d`, `noise.perlin3d` |
| **Math** | `math.sigmoid`, `math.log2`, `activation.fitts`, `activation.hick` |
| **FHE** | `fhe.ntt`, `fhe.pointwise_mul` |
| **Device** | `device.list`, `device.probe` |
| **Validation** | `tolerances.get`, `validate.gpu_stack` |
| **Health** | `health.liveness`, `health.readiness`, `health.check` |
| **Discovery** | `capabilities.list`, `identity.get`, `primal.info`, `primal.capabilities` |
| **RNG** | `rng.uniform` |

Other primals called:

| Primal | Methods |
|--------|---------|
| **BearDog** | `crypto.hash` (blake3 provenance witnesses) |
| **toadStool** | `compute.dispatch` (GPU orchestration) |

---

## Domain Science → IPC Method Mapping

### 1. Nuclear EOS (SEMF Binding Energy)

**Python baseline:** `control/nuclear_eos/scripts/semf_binding_energy.py`
**Local Rust:** `physics::semf_binding_energy(Z, N, params)` → `barracuda::optimize::bisect`

| Step | IPC Method | Params | Expected |
|------|------------|--------|----------|
| Create parameter tensor | `tensor.create` | `{ "shape": [10], "data": SLY4_PARAMS }` | tensor id |
| Scale energy terms | `tensor.scale` | `{ "tensor": id, "scalar": A }` | scaled tensor |
| Sum binding components | `stats.weighted_mean` | `{ "values": [vol, surf, coul, sym, pair], "weights": [1,1,1,1,1] }` | B/A in MeV |
| Compare vs baseline | — | — | `|ipc - python| / python < COMPOSITION_SEMF_PARITY_REL (1e-10)` |

**Python baseline value:** Pb-208 B/A = 7.867 MeV (SLY4 params, `control/nuclear_eos`)

### 2. Lattice QCD Plaquette

**Python baseline:** `control/lattice_qcd/scripts/wilson_plaquette.py`
**Local Rust:** `lattice::wilson::Lattice::hot_start(…).average_plaquette()`

| Step | IPC Method | Params | Expected |
|------|------------|--------|----------|
| Create gauge field tensor | `tensor.create` | `{ "shape": [V, 4, 3, 3], "dtype": "complex_f64" }` | tensor id |
| Matrix multiply (link product) | `tensor.matmul` | `{ "a": U_mu, "b": U_nu }` | product tensor |
| Trace and average | `stats.mean` | `{ "values": traces }` | average plaquette |
| Compare vs baseline | — | — | `|ipc - python| < COMPOSITION_PLAQUETTE_PARITY_ABS (1e-12)` |

**Python baseline value:** 4^4 hot-start plaquette ~ 0.333 (SU(3) random, seed-dependent)

### 3. HMC Trajectory

**Python baseline:** `control/lattice_qcd/scripts/hmc_trajectory.py`
**Local Rust:** `lattice::hmc::hmc_step()`

| Step | IPC Method | Params | Expected |
|------|------------|--------|----------|
| GPU shader dispatch | `compute.dispatch` | `{ "shader": "hmc_leapfrog_f64", "workgroups": [V/64, 1, 1] }` | dispatch result |
| Tensor operations | `tensor.matmul`, `tensor.add`, `tensor.scale` | leapfrog + force | trajectory data |
| Observable statistics | `stats.mean` | `{ "values": plaquettes }` | mean plaquette |
| Compare vs baseline | — | — | finite plaquette + acceptance ∈ {true, false} |

### 4. CG Solver (Conjugate Gradient)

**Python baseline:** `control/lattice_qcd/scripts/cg_solve.py`
**Local Rust:** `lattice::cg::cg_solve()`

| Step | IPC Method | Params | Expected |
|------|------------|--------|----------|
| Matrix-vector product | `tensor.matmul` | `{ "a": D_dagger_D, "b": x }` | Ax |
| Vector addition | `tensor.add` | `{ "a": r, "b": p, "alpha": alpha }` | updated residual |
| Dot product / norm | `stats.mean` | `{ "values": r_squared }` | convergence check |
| Compare vs baseline | — | — | iteration count matches, residual < tolerance |

### 5. Gradient Flow

**Python baseline:** `control/gradient_flow/scripts/flow_integrators.py`
**Local Rust:** `lattice::gradient_flow::integrate()`

| Step | IPC Method | Params | Expected |
|------|------------|--------|----------|
| Gauge field evolution | `compute.dispatch` | `{ "shader": "gradient_flow_f64" }` | evolved field |
| Scale setting (t₀, w₀) | `tensor.scale` + `stats.mean` | flow-time observables | t₀/w₀ values |
| Compare vs baseline | — | — | `|ipc_t0 - python_t0| < tolerance` |

### 6. Molecular Dynamics

**Python baseline:** `control/sarkas/scripts/yukawa_md.py`
**Local Rust:** `md::cpu_reference::run_simulation_cpu()`

| Step | IPC Method | Params | Expected |
|------|------------|--------|----------|
| Force evaluation | `compute.dispatch` | `{ "shader": "yukawa_force_f64" }` | forces tensor |
| Integration step | `tensor.add` + `tensor.scale` | velocity Verlet | updated positions |
| Observable averaging | `stats.mean` | `{ "values": kinetic_energies }` | temperature |
| Compare vs baseline | — | — | energy drift < 0.002%, RDF/VACF within tolerance |

### 7. Observable Statistics

All science domains use observable averaging:

| Observable | IPC Method | Tolerance |
|------------|------------|-----------|
| Plaquette ⟨P⟩ | `stats.mean` | COMPOSITION_PLAQUETTE_PARITY_ABS |
| Polyakov loop ⟨\|L\|⟩ | `stats.mean` | 1e-10 relative |
| Energy per particle | `stats.mean` + `stats.std_dev` | domain-specific |
| χ² per datum | `stats.weighted_mean` | 1e-10 relative |

### 8. Provenance Witness

| Step | IPC Method | Primal | Expected |
|------|------------|--------|----------|
| Hash result digest | `crypto.hash` | BearDog | blake3 hash matches local computation |

---

## Test Procedure

To run the primal proof against a live NUCLEUS:

```bash
# 1. Deploy primals from plasmidBin ecobins
biomeos deploy --graph graphs/hotspring_qcd_deploy.toml

# 2. Verify primals are alive
ls $XDG_RUNTIME_DIR/biomeos/*.sock

# 3. Set family ID (must match the deployed NUCLEUS)
export FAMILY_ID=default

# 4. Run the primal proof harness
cargo run --bin validate_primal_proof

# Exit 0 = all PASS, exit 1 = at least one FAIL, exit 2 = all SKIP (no NUCLEUS)
```

In standalone mode (no primals on UDS), the harness exits 2 (all SKIP) — verified.

---

## Validation Harness Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All exercised capabilities PASS |
| 1 | At least one capability FAIL |
| 2 | All capabilities SKIP (NUCLEUS not deployed) |

---

## References

- barraCuda IPC surface: `infra/plasmidBin/barracuda/`
- Python baselines: `control/` (per-domain subdirectories)
- Tolerances: `barracuda/src/tolerances/` (308 named constants)
- Provenance: `barracuda/src/provenance.rs` (BaselineProvenance records)
- Proto-nucleate: `primalSpring/graphs/downstream/downstream_manifest.toml`
