# hotSpring Deploy Graphs

Deploy graphs consumed by biomeOS for NUCLEUS composition deployment.

## Files

- `hotspring_qcd_deploy.toml` — Primary deploy graph for lattice QCD / HPC physics

## Deploy Graph vs Proto-Nucleate

hotSpring maintains **two distinct graph types** — they serve different purposes
and validate at different levels:

### Proto-Nucleate (downstream_manifest.toml)

    primalSpring/graphs/downstream/downstream_manifest.toml  (spring_name = "hotspring")

The proto-nucleate defines the **target primal composition** — which primals,
which IPC capabilities, which bonding model. It has **no spring binary as a node**
because the spring is external to the NUCLEUS. The spring validates *against* the
composition, not *inside* it.

Used for: Level 5 primal proof — the `validate_primal_proof` harness calls primal
methods listed in `validation_capabilities` over IPC and compares against baselines.

### Deploy Graph (hotspring_qcd_deploy.toml)

    graphs/hotspring_qcd_deploy.toml

The deploy graph defines **how to deploy** the spring's own integration server
(`hotspring_primal`, order = 10) alongside primals. It includes spawn order, health
probes, and fragment references. The spring binary appears as a node here because
this is the Level 2-3 integration test — the "Rust proof" where the spring server
dispatches domain science via in-process library calls.

Used for: `biomeos deploy --graph graphs/hotspring_qcd_deploy.toml`

### Summary

| Property | Proto-Nucleate | Deploy Graph |
|----------|---------------|--------------|
| Location | primalSpring/graphs/downstream/ | graphs/ (this directory) |
| Spring binary as node | No | Yes (order = 10) |
| Purpose | Level 5 primal proof | Level 2-3 integration |
| Consumed by | validate_primal_proof harness | biomeOS deploy |
| Primals | Pure NUCLEUS primals only | Primals + spring server |

## License

AGPL-3.0-or-later
