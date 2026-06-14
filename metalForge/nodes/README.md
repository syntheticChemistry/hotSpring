# Node Environment Profiles

Per-gate environment files that configure hotSpring for the hardware on each machine. These formalize the `HOTSPRING_GPU_ADAPTER` selection that already works — no code changes, just env vars.

## Usage

Source the profile for your node before running jobs:

```bash
source metalForge/nodes/biomegate.env
cargo run --release --bin validate_gpu_streaming_dyn
```

Or inline without modifying the shell:

```bash
env $(cat metalForge/nodes/biomegate.env | grep -v '^#' | grep -v '^$' | xargs) \
  cargo run --release --bin bench_multi_gpu
```

## Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `HOTSPRING_GPU_ADAPTER` | Primary GPU for single-GPU jobs (name substring or index) | `3090`, `4070`, `titan` |
| `HOTSPRING_WGPU_BACKEND` | wgpu backend selection | `vulkan`, `metal`, `dx12` |
| `HOTSPRING_GPU_PRIMARY` | Primary GPU for multi-GPU benchmarks | `3090` |
| `HOTSPRING_GPU_SECONDARY` | Secondary GPU for multi-GPU benchmarks | `titan` |
| `HOTSPRING_SCRATCH` | Scratch directory for large-lattice checkpoints | `/mnt/scratch` |

## Available Profiles

| Profile | Node | Primary GPU | VRAM | CPU |
|---------|------|------------|------|-----|
| `biomegate.env` | biomeGate | RTX 3090 | 24 GB | Threadripper 3970X (32c/64t) |
| `eastgate.env` | Eastgate | RTX 4070 | 12 GB | i9-12900 (16c/24t) |

## Adding a New Node

1. Copy an existing `.env` file
2. Update `HOTSPRING_GPU_ADAPTER` to match your primary GPU's name (run `cargo run --release --bin bench_gpu_fp64` to see available adapters)
3. Set `HOTSPRING_GPU_PRIMARY` / `HOTSPRING_GPU_SECONDARY` if multi-GPU
4. Set `HOTSPRING_SCRATCH` if you have dedicated work drives
