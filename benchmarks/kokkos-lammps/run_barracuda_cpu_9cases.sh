#!/bin/bash
# Ratcheting Step 2: barraCuda CPU validation of 9 PP Yukawa DSF cases
# Uses sarkas_gpu --scale for CPU reference at N=500 and N=2000
# Then runs --full for all 9 GPU cases at N=2000

set -e

HOTSPRING_DIR="/home/biomegate/Development/ecoPrimals/hotSpring/barracuda"
RESULTS_DIR="$(dirname "$0")/results_barracuda_cpu"
mkdir -p "$RESULTS_DIR"

CARGO="/home/biomegate/.cargo/bin/cargo"
export RUSTUP_HOME="$HOME/.rustup"
export CARGO_HOME="$HOME/.cargo"
export PATH="/home/biomegate/.cargo/bin:/home/biomegate/.rustup/shims:/usr/bin:/bin:/usr/local/bin"
export HOME="$HOME"
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

echo "================================================================"
echo "  Ratcheting Step 2: barraCuda CPU Yukawa OCP Validation"
echo "  Scale test: N=500, N=2000 (GPU + CPU)"
echo "================================================================"

cd "$HOTSPRING_DIR"

# Run scaling test (GPU + CPU at N=500 and N=2000)
echo ""
echo "Running barraCuda --scale (GPU+CPU at N=500,2000)..."
$CARGO run --release --bin sarkas_gpu -- --scale 2>&1 | tee "$RESULTS_DIR/sarkas_gpu_scale.log"

echo ""
echo "================================================================"
echo "  barraCuda CPU validation complete"
echo "  Results in: $RESULTS_DIR"
echo "================================================================"
