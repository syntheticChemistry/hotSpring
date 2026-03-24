# Scripts — Unified GPU Cracking Workflow

## Active Scripts

### Capture & Analysis

| Script | Purpose | Usage |
|--------|---------|-------|
| `capture_multi_backend.sh` | Swap-capture-return cycle for any driver backend | `sudo ./capture_multi_backend.sh <driver> [BDF]` |
| `compare_snapshots.py` | Cross-driver register snapshot comparison | `python3 compare_snapshots.py <baseline> <warm> [--filter falcon\|acr\|mmu\|pfifo] [--json]` |
| `distill_oracle_recipe.sh` | Feed mmiotrace through hw-learn distiller | `./distill_oracle_recipe.sh <trace_file> [output.json]` |
| `cross_card_oracle.py` | Live BAR0 register dump (nouveau-bound) | `sudo python3 cross_card_oracle.py [BDF]` |
| `read_bar0_deep.py` | Deep BAR0 dump (140+ registers) | `sudo python3 read_bar0_deep.py [BDF]` |
| `read_bar0_regs.py` | Quick BAR0 register check | `sudo python3 read_bar0_regs.py [BDF]` |

### Build & Deploy

| Script | Purpose | Usage |
|--------|---------|-------|
| `build_nvidia_oracle.sh` | Build renamed nvidia_oracle.ko module | `sudo ./build_nvidia_oracle.sh [VERSION]` |
| `gpu-ctl` | Passwordless sysfs bind/unbind via sudoers | `gpu-ctl status\|bind\|unbind\|d0 [BDF]` |
| `deploy_glowplug.sh` | Deploy GlowPlug daemon | See script |
| `deploy_ember_first_time.sh` | First-time Ember setup | See script |

### Lab Setup

| Script | Purpose |
|--------|---------|
| `setup_dual_titanv.sh` | Configure both Titan V GPUs for VFIO |
| `rebind_titanv_nvidia.sh` | Rebind Titan to nvidia (system module) |
| `warm_and_test.sh` | Warm GPU via nouveau and test accessibility |
| `boot/install-glowplug.sh` | Install GlowPlug systemd service |
| `boot/post-boot-oracle-capture.sh` | Post-boot register capture |

### Data Regeneration (Physics)

| Script | Purpose |
|--------|---------|
| `regenerate-all.sh` | Master regeneration: Sarkas, TTM, surrogate |
| `clone-repos.sh` | Clone/pin physics reference repos |
| `download-data.sh` | Zenodo data download |
| `setup-envs.sh` | Conda environment creation |

## Recommended Capture Campaign

```bash
# 1. Verify Titans are on VFIO
gpu-ctl status 0000:03:00.0
gpu-ctl status 0000:4a:00.0

# 2. Capture Titan #1 — all backends
sudo ./scripts/capture_multi_backend.sh nouveau 0000:03:00.0
sudo ./scripts/capture_multi_backend.sh nvidia  0000:03:00.0

# 3. Capture Titan #2 — all backends
sudo ./scripts/capture_multi_backend.sh nouveau 0000:4a:00.0
sudo ./scripts/capture_multi_backend.sh nvidia  0000:4a:00.0

# 4. Compare results
python3 scripts/compare_snapshots.py data/082/nouveau_*_03_*/manifest.json \
                                      data/082/nvidia_*_03_*/manifest.json

# 5. (After building nvidia_oracle)
sudo ./scripts/build_nvidia_oracle.sh 580.126.18
sudo ./scripts/capture_multi_backend.sh nvidia_oracle 0000:03:00.0

# 6. Distill recipes
./scripts/distill_oracle_recipe.sh data/082/nouveau_*/mmiotrace_raw.txt
./scripts/distill_oracle_recipe.sh data/082/nvidia_*/mmiotrace_raw.txt
```

## Archived Scripts

`scripts/archive/` contains superseded scripts:
- `capture_mmiotrace_oracle.sh` — replaced by `capture_multi_backend.sh`
- `capture_nouveau_mmiotrace.sh` — replaced by `capture_multi_backend.sh`
- `capture_nouveau_mmiotrace_v2.sh` — replaced by `capture_multi_backend.sh`
- `rebind_titanv_vfio.sh` etc. — replaced by `gpu-ctl`

## Data Output Structure

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
