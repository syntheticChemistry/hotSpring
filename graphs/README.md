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

### guideStone Deployment (Level 5-6)

The guideStone is a self-validating deployable. In Level 6, it deploys as a node
in the proto-nucleate graph alongside primals. In Level 5 (current), it runs
externally and validates IPC parity. The `hotspring_guidestone` binary is the
unified Level 5 artifact — it validates all 5 bare properties (including
Property 3 BLAKE3 CHECKSUMS via `primalspring::checksums::verify_manifest()` —
15 validation-critical source files hashed) then probes NUCLEUS IPC when primals
are deployed. **Bare mode: 30/30 checks pass** (primalSpring v0.9.17).
`validation/CHECKSUMS` holds the BLAKE3 manifest. See
`primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md` for the standard.

The `primalspring_guidestone` binary provides base composition certification (6
layers: graph parsing, discovery, health, capability parity, cross-atomic
pipeline, bonding, crypto). hotSpring's domain guideStone inherits that and only
validates QCD physics on top.

### Summary

| Property | Proto-Nucleate | Deploy Graph | guideStone |
|----------|---------------|--------------|------------|
| Location | primalSpring/graphs/downstream/ | graphs/ (this directory) | validation/ + validate_primal_proof |
| Spring binary as node | No | Yes (order = 10) | No (validates externally) |
| Purpose | Level 5 primal proof target | Level 2-3 integration | Level 5-6 self-validation |
| Consumed by | validate_primal_proof harness | biomeOS deploy | biomeOS (Level 6) or standalone |
| Primals | Pure NUCLEUS primals only | Primals + spring server | Primals (additive) |

## License

AGPL-3.0-or-later
