#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Exp 086 — Cross-driver, cross-GPU falcon profiling matrix.
#
# Orchestrates: vfio-cold → nouveau-warm → vfio-post-nouveau → nvidia-warm → vfio-post-nvidia
# for both Titans, using GlowPlug swaps + Python BAR0 profiler.
#
# Usage: sudo bash exp086_run_matrix.sh
#
# Prerequisites:
#   - GlowPlug running with both Titans on vfio-pci
#   - coralctl in PATH
#   - coral-driver hw test binaries built

set -uo pipefail

TITAN1="0000:03:00.0"
TITAN2="0000:4a:00.0"
GPUS=("$TITAN1" "$TITAN2")
GPU_NAMES=("titan1" "titan2")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILER="$SCRIPT_DIR/exp086_falcon_profiler.py"
DATA_DIR="$SCRIPT_DIR/../data/086"
DRIVER_DIR="$SCRIPT_DIR/../../coralReef"

CORALCTL="coralctl"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$DATA_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }
separator() { echo ""; echo "========================================"; echo "$1"; echo "========================================"; echo ""; }

get_driver() {
    local bdf="$1"
    local link="/sys/bus/pci/devices/$bdf/driver"
    if [ -L "$link" ]; then
        basename "$(readlink "$link")"
    else
        echo "none"
    fi
}

wait_for_driver() {
    local bdf="$1"
    local expected="$2"
    local timeout="${3:-15}"
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        local actual
        actual="$(get_driver "$bdf")"
        if [ "$actual" = "$expected" ]; then
            log "  $bdf → $actual (${elapsed}s)"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    log "  WARN: $bdf still on $(get_driver "$bdf") after ${timeout}s (expected $expected)"
    return 1
}

run_profiler() {
    local bdf="$1"
    local output="$2"
    log "  Profiling $bdf → $output"
    python3 "$PROFILER" "$bdf" "$output" 2>&1 | while IFS= read -r line; do
        echo "    $line"
    done
}

run_acr_boot() {
    local bdf="$1"
    local output="$2"
    log "  ACR boot test on $bdf → $output"

    if ! command -v cargo &>/dev/null; then
        log "  SKIP: cargo not in PATH"
        echo "SKIPPED: cargo not available" > "$output"
        return
    fi

    local drv
    drv="$(get_driver "$bdf")"
    if [ "$drv" != "vfio-pci" ]; then
        log "  SKIP: $bdf on $drv, need vfio-pci for ACR test"
        echo "SKIPPED: driver=$drv, need vfio-pci" > "$output"
        return
    fi

    (
        cd "$DRIVER_DIR" 2>/dev/null || { echo "SKIP: coralReef dir not found" > "$output"; return; }
        CORALREEF_VFIO_BDF="$bdf" cargo test \
            --package coral-driver \
            --test hw_nv_vfio \
            -- vfio_falcon_boot_solver \
            --ignored --nocapture 2>&1
    ) > "$output" || true

    local lines
    lines=$(wc -l < "$output")
    log "  ACR boot: $lines lines captured"
}

swap_gpu() {
    local bdf="$1"
    local target="$2"
    log "Swapping $bdf → $target"
    $CORALCTL swap "$bdf" "$target" 2>&1 || true
    sleep 2
    wait_for_driver "$bdf" "$target"
}

# ══════════════════════════════════════════════════════════════════════
# Pre-flight checks
# ══════════════════════════════════════════════════════════════════════

separator "PRE-FLIGHT CHECKS"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Must run as root (sudo)" >&2
    exit 1
fi

if ! command -v "$CORALCTL" &>/dev/null; then
    echo "ERROR: coralctl not in PATH" >&2
    exit 1
fi

for i in "${!GPUS[@]}"; do
    local_drv="$(get_driver "${GPUS[$i]}")"
    log "${GPU_NAMES[$i]} (${GPUS[$i]}): driver=$local_drv"
done

if ! [ -f "$PROFILER" ]; then
    echo "ERROR: Profiler not found: $PROFILER" >&2
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════
# Phase A: vfio-cold baseline
# ══════════════════════════════════════════════════════════════════════

separator "PHASE A: VFIO-COLD BASELINE"

for i in "${!GPUS[@]}"; do
    bdf="${GPUS[$i]}"
    name="${GPU_NAMES[$i]}"
    drv="$(get_driver "$bdf")"

    if [ "$drv" != "vfio-pci" ]; then
        log "$name not on vfio-pci (on $drv), swapping..."
        swap_gpu "$bdf" "vfio"
    fi

    run_profiler "$bdf" "$DATA_DIR/${name}_vfio_cold.json"
done

# ══════════════════════════════════════════════════════════════════════
# Phase B: nouveau warm-up
# ══════════════════════════════════════════════════════════════════════

separator "PHASE B: NOUVEAU WARM-UP"

for i in "${!GPUS[@]}"; do
    bdf="${GPUS[$i]}"
    name="${GPU_NAMES[$i]}"

    log "--- $name: nouveau cycle ---"

    swap_gpu "$bdf" "nouveau"
    sleep 5
    run_profiler "$bdf" "$DATA_DIR/${name}_nouveau_warm.json"

    swap_gpu "$bdf" "vfio"
    sleep 3
    run_profiler "$bdf" "$DATA_DIR/${name}_vfio_post_nouveau.json"
    run_acr_boot "$bdf" "$DATA_DIR/${name}_acr_post_nouveau.txt"
done

# ══════════════════════════════════════════════════════════════════════
# Phase C: nvidia warm-up
# ══════════════════════════════════════════════════════════════════════

separator "PHASE C: NVIDIA WARM-UP"

for i in "${!GPUS[@]}"; do
    bdf="${GPUS[$i]}"
    name="${GPU_NAMES[$i]}"

    log "--- $name: nvidia cycle ---"

    swap_gpu "$bdf" "nvidia"
    sleep 5
    run_profiler "$bdf" "$DATA_DIR/${name}_nvidia_warm.json"

    swap_gpu "$bdf" "vfio"
    sleep 3
    run_profiler "$bdf" "$DATA_DIR/${name}_vfio_post_nvidia.json"
    run_acr_boot "$bdf" "$DATA_DIR/${name}_acr_post_nvidia.txt"
done

# ══════════════════════════════════════════════════════════════════════
# Phase D: cold ACR reference (re-baseline after all swaps)
# ══════════════════════════════════════════════════════════════════════

separator "PHASE D: COLD ACR REFERENCE"

for i in "${!GPUS[@]}"; do
    bdf="${GPUS[$i]}"
    name="${GPU_NAMES[$i]}"

    drv="$(get_driver "$bdf")"
    if [ "$drv" != "vfio-pci" ]; then
        swap_gpu "$bdf" "vfio"
        sleep 3
    fi

    run_profiler "$bdf" "$DATA_DIR/${name}_vfio_final.json"
    run_acr_boot "$bdf" "$DATA_DIR/${name}_acr_cold.txt"
done

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

separator "COMPLETE"

log "All captures saved to: $DATA_DIR/"
echo ""
ls -lh "$DATA_DIR/"
echo ""

log "Captured files:"
for f in "$DATA_DIR"/*.json; do
    [ -f "$f" ] && echo "  JSON: $(basename "$f")"
done
for f in "$DATA_DIR"/*.txt; do
    [ -f "$f" ] && echo "  ACR:  $(basename "$f")"
done

echo ""
log "Next: python3 $SCRIPT_DIR/exp086_analyze.py $DATA_DIR/"
