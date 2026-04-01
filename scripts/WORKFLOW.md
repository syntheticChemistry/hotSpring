# Scripts — Unified GPU Cracking Workflow

> **FOSSIL RECORD** — This document describes a manual workflow that predates
> `coralctl` / `ember` / `glowplug`. The canonical script reference is
> [`scripts/README.md`](README.md). The scripts listed below have been moved
> to `scripts/archive/` or superseded by `coralctl` subcommands.

## Modern Equivalents

| Legacy Script | Modern Equivalent |
|---|---|
| `capture_multi_backend.sh` | `coralctl capture <BDF> --backend <driver>` |
| `compare_snapshots.py` | `coralctl snapshot diff <BDF>` |
| `distill_oracle_recipe.sh` | `coralctl oracle distill <trace>` |
| `cross_card_oracle.py` | `coralctl mmio read <BDF> <offset>` |
| `read_bar0_deep.py` / `read_bar0_regs.py` | `coralctl mmio read <BDF> <offset>` |
| `build_nvidia_oracle.sh` | Handled by `agentReagents` VM image build |
| `gpu-ctl` | `coralctl swap <BDF> <driver>` / `coralctl status <BDF>` |
| `deploy_glowplug.sh` / `deploy_ember_first_time.sh` | `scripts/boot/install-boot-config.sh` + systemd |
| `setup_dual_titanv.sh` | `coralctl swap <BDF> vfio` |
| `rebind_titanv_nvidia.sh` | `coralctl swap <BDF> nvidia` |
| `warm_and_test.sh` | `coralctl warm-fecs <BDF>` |
| `bar0_read.py` | `coralctl mmio read <BDF> <offset>` |
| `parse_mmiotrace.py` | `coralctl trace-parse <file>` |
| `apply_recipe.py` | `coralctl oracle apply <BDF> <recipe>` |
| `replay_devinit.py` | `coralctl devinit replay <BDF>` |
| `generate_titanv_recipe.py` | `coralctl trace-parse --recipe-json <file>` |
| `extract_devinit.py` | `coralctl devinit replay <BDF>` |
| `titan_timing_attack.sh` | `coralctl warm-fecs <BDF>` |

## Data Output Structure (unchanged)

```
data/082/
├── nouveau_0000_03_00.0_20260324_HHMMSS/
│   ├── bar0_cold_vfio.bin
│   ├── mmiotrace_raw.txt
│   ├── mmiotrace_writes.txt
│   ├── mmiotrace_falcon_init.txt
│   ├── mmiotrace_acr_dma.txt
│   ├── mmiotrace_demmio.txt
│   ├── bar0_warm_nouveau.bin
│   ├── bar0_residual_post_nouveau.bin
│   └── manifest.json
├── nvidia_0000_03_00.0_20260324_HHMMSS/
│   └── (same structure)
└── nvidia_oracle_0000_03_00.0_20260324_HHMMSS/
    └── (same structure)
```
