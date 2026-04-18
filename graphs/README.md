# hotSpring Deploy Graphs

Deploy graphs consumed by biomeOS for NUCLEUS composition deployment.

## Files

- `hotspring_qcd_deploy.toml` — Primary deploy graph for lattice QCD / HPC physics

## Relationship to Proto-Nucleate

The proto-nucleate lives in primalSpring:

    primalSpring/graphs/downstream/downstream_manifest.toml  (spring_name = "hotspring")

Proto-nucleate defines WHAT primals compose (the target composition).
Deploy graphs define HOW to deploy them (spawn order, health checks, bonding).

## Usage

```bash
biomeos deploy --graph graphs/hotspring_qcd_deploy.toml
```

## License

AGPL-3.0-or-later
