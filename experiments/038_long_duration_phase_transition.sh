#!/usr/bin/env bash
# Experiment 038: Long-Duration Single-Point Runs
#
# Current runs give the reject prediction head only 3-5 data points per beta.
# This experiment runs 50+ trajectories at 3 phase-transition betas to provide
# dense time-series data for the reject head and phase classifier.
#
# Design:
#   - 3 betas near the deconfinement transition: 5.5, 5.69, 6.0
#   - 4^4 lattice, mass=0.1 (well-understood regime)
#   - 50 trajectories each (10x more than exp035)
#   - Bootstrap from all exp035 data for rich NPU knowledge
#
# Key metrics:
#   - Reject prediction accuracy (should improve with dense data)
#   - Head confidence for REJECT_PREDICT and PHASE_CLASSIFY
#   - Time-series autocorrelation of NPU predictions
#   - Phase classification near the transition (beta=5.69 is critical)
#
# Hardware: RTX 3090 (Motor) + AKD1000 (Cerebellum)
# 2026-03-03

set -euo pipefail

BINARY="./barracuda/target/release/production_dynamical_mixed"
RESULTS="results"
ADAPTER="3090"
BASE_SEED=20260306

export PATH="/home/biomegate/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/local/bin:/usr/bin:/bin"
export HOTSPRING_GPU_ADAPTER="$ADAPTER"

mkdir -p "$RESULTS"

log() { echo "[$(date +%H:%M:%S)] $*"; }

WEIGHTS_DIR="$RESULTS/weights"
mkdir -p "$WEIGHTS_DIR"
LOGFILE="$RESULTS/exp038_long_duration.log"

# Rich bootstrap from all exp035 data
BOOTSTRAP="$WEIGHTS_DIR/phase3_8x8.bin"
for F in "$RESULTS"/exp035_*.jsonl; do
    [ -f "$F" ] && BOOTSTRAP="$BOOTSTRAP,$F"
done

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  Long-Duration Phase Transition Study" | tee -a "$LOGFILE"
log "  4^4, mass=0.1, 50 traj/beta, 3 critical betas" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"

for BETA in 5.5 5.69 6.0; do
    BSLUG=$(echo "$BETA" | tr '.' 'p')
    log "  β=$BETA — 50 trajectories (dense reject/phase data)" | tee -a "$LOGFILE"

    $BINARY \
        --lattice=4 \
        --betas=$BETA \
        --mass=0.1 \
        --n-fields=1 \
        --cg-max-iter=5000 \
        --therm=10 \
        --meas=50 \
        --max-adaptive=0 \
        --seed=$((BASE_SEED + 1000)) \
        --bootstrap-from="$BOOTSTRAP" \
        --save-weights="$WEIGHTS_DIR/exp038_long_b${BSLUG}.bin" \
        --output="$RESULTS/exp038_4x4_b${BSLUG}_long.jsonl" \
        2>&1 | tee -a "$LOGFILE"

    log "  β=$BETA complete" | tee -a "$LOGFILE"
done

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  EXPERIMENT 038 COMPLETE — Long Duration" | tee -a "$LOGFILE"
log "" | tee -a "$LOGFILE"
log "  Key comparisons:" | tee -a "$LOGFILE"
log "    β=5.50 (confined): reject accuracy, phase=confined" | tee -a "$LOGFILE"
log "    β=5.69 (critical): reject accuracy, phase=transition" | tee -a "$LOGFILE"
log "    β=6.00 (deconfined): reject accuracy, phase=deconfined" | tee -a "$LOGFILE"
log "  Focus: does dense data at critical beta improve head trust?" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
