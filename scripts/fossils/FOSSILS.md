# Fossilized Scripts

Scripts archived here represent **superseded patterns** — ad-hoc bash/Python glue
that has been fully absorbed into Rust-native tooling.

They are retained for provenance (showing what the Rust equivalents replaced) but
MUST NOT be invoked in any live pipeline.

## Fossil Registry

| Fossil | Replaced By | Date |
|--------|-------------|------|
| `finalize.sh` | `nest-validate guidestone finalize` | 2026-05-26 |
| `analyze_puckering.py` | `cazyme-fel` crate + `nest-validate cazyme` | 2026-05-26 |
| `analyze_FES.sh` | `nest-validate guidestone validate` (Phase 3) | 2026-05-26 |
| `validate-guidestone-multi.sh` | `nest-validate guidestone validate` (multi-substrate planned) | 2026-05-26 |
| `regenerate-all.sh` | `nest-validate guidestone run` (full pipeline) | 2026-05-26 |
| `download-data.sh` | `nest-validate guidestone run --fetch` (planned) | 2026-05-26 |
| `setup-envs.sh` | External tool detection in `nest-validate env` | 2026-05-26 |
| `clone-repos.sh` | `nest-validate ingest` + native git ops | 2026-05-26 |

## Active Scripts (Not Fossilized)

These remain because they are **thin launchers** or **artifact production** tools
that invoke Rust binaries. They contain no complex logic:

- `build-guidestone.sh` — artifact production (creates plasmidBin contents)
- `validate-primal-proof.sh` — primal proof chain (already uses plasmidBin resolution)
- `harvest-ecobin.sh` — musl-static binary harvesting
- `coverage.sh` / `ci-coverage-gate.sh` — thin cargo-llvm-cov wrappers
- `prepare-usb.sh` — hardware deployment (USB liveSpore)
- `build-container.sh` — Docker image build
