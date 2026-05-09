# Fossil Record: Experiment Binaries (Prokaryotic Era)

**Snapshot date**: May 9, 2026  
**Superseded by**: `hotspring_unibin validate` and `barracuda/src/validation/scenarios/`  
**Provenance**: hotSpring v0.6.32, pre-interstadial eukaryotic evolution

## What This Contains

Snapshot of 8 experiment binaries from `barracuda/src/bin/exp*.rs` that were
the prokaryotic pattern for hardware-specific experiments:

| Binary | Domain | Description |
|--------|--------|-------------|
| `exp070_register_dump.rs` | GPU RE | BAR0 register dump (JSON output) |
| `exp070_register_diff.rs` | GPU RE | Register diff analysis (cold vs warm) |
| `exp154_sec2_acr_pipeline.rs` | GPU RE | Titan V SEC2/ACR pipeline via ember IPC |
| `exp155_k80_warm_fecs.rs` | GPU RE | K80 warm-cycle FECS dispatch |
| `exp156_reagent_compare.rs` | GPU RE | Reagent trace comparison |
| `exp157_k80_devinit_replay.rs` | GPU RE | K80 DEVINIT replay experiment |
| `exp158_sec2_real_firmware.rs` | GPU RE | SEC2 real firmware experiment |
| `exp167_warm_handoff.rs` | GPU RE | nouveau HBM2 → vfio sovereign compute |

## Why Fossilized

These experiment binaries are **hardware-specific GPU reverse engineering**
experiments. Unlike physics validation (SEMF, lattice QCD, MD) which can be
absorbed into `validation/scenarios/` as Tier 1 Rust scenarios, these require
specific GPU hardware (Titan V, K80) and low-level access (BAR0 mmap, FECS,
SEC2, DMA). They remain as `[[bin]]` targets in `Cargo.toml` (some gated by
`low-level` or `cuda-validation` features) but their source is preserved here
as a dated snapshot.

## What Supersedes Them

Representative physics patterns from the broader experiment binary collection
have been absorbed into:

- `barracuda/src/validation/scenarios/` — `ScenarioMeta` + `ScenarioRegistry`
- `barracuda/src/certification/` — guideStone organelle (L0–L5)
- `hotspring_unibin validate` — unified CLI entry point

The hardware experiments remain active binaries but are not part of the
eukaryotic UniBin evolution — they are specialized tools, not validation
scenarios.

## License

AGPL-3.0-or-later
