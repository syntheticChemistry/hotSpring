# nest-validate — Rust-Native PLUMED-NEST Validation Suite

Replaces the bash/Python orchestration layer with a single native binary.
No zombie processes, no jelly-string conda wrappers, proper signal handling.

## What It Replaces

| Old (bash/Python)           | New (Rust)                        |
|-----------------------------|-----------------------------------|
| `validate_all.sh`           | `nest-validate validate`          |
| `ingest.sh`                 | `nest-validate ingest`            |
| `analysis/analyze.py`       | `nest-validate analyze <dir>`     |
| Manual `gmx mdrun` + `&`   | `nest-validate run <dir>`         |
| — (none)                    | `nest-validate report`            |

## Build

```bash
cd nest-validate
cargo build --release
```

Binary: `target/release/nest-validate`

## Usage

```bash
# Full validation suite
./target/release/nest-validate validate

# Single target
./target/release/nest-validate validate --target 01

# JSON output (for downstream tooling / litho audit)
./target/release/nest-validate validate --json

# Parity report (quantitative pass/fail against published values)
./target/release/nest-validate report

# Ingest PLUMED-NEST archives
./target/release/nest-validate ingest --target 02

# Launch GROMACS+PLUMED with proper process management
./target/release/nest-validate run target_02_chignolin_opes
```

## Architecture

```
src/
├── main.rs       CLI dispatch, target discovery, environment detection
├── fes.rs        FES reconstruction (1D/2D Gaussian kernel summation)
├── hills.rs      HILLS file parser (1D + 2D)
├── colvar.rs     COLVAR parser, reweighted FES, transition counting
├── targets.rs    Per-target analysis dispatch, GROMACS process management
├── ingest.rs     Archive download (ureq), tar extraction, PLUMED validation
├── parity.rs     Tolerance classes, quantitative pass/fail, NUCLEUS readiness
└── report.rs     Structured report types (JSON-serializable)
```

## What This Solves

1. **Zombie processes** — No more `kill` cleanup after interrupted runs
2. **Redundant conda wrappers** — Direct binary, no Python startup overhead
3. **Fragile shell pipes** — Structured error handling, not `set -e` hope
4. **FES analysis parity** — Native Gaussian kernel summation identical to PLUMED
5. **Structured output** — JSON reports consumable by litho audit / pseudoSpore

## Tolerance Registry

Industry-standard tolerance classes derived from PLUMED-NEST reproduction:

| Tolerance               | Value  | Unit      | Source       |
|-------------------------|--------|-----------|--------------|
| `fes_barrier_kjmol`    | ±5.0   | kJ/mol    | plumID:19.009|
| `basin_position_rad`   | ±0.5   | rad       | plumID:19.009|
| `folding_fe_kjmol`     | ±4.0   | kJ/mol    | plumID:24.029|
| `convergence_std_kjmol`| ±3.0   | kJ/mol    | plumID:19.009|
| `block_stderr_kjmol`   | ±5.0   | kJ/mol    | general      |
| `height_decay_ratio`   | ±0.15  | ratio     | plumID:19.009|
| `puckering_theta_deg`  | ±10.0  | degrees   | plumID:22.028|
| `binding_fe_kjmol`     | ±8.0   | kJ/mol    | plumID:23.004|
