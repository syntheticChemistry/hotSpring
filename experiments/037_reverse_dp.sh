#!/usr/bin/env bash
# Experiment 037: Reverse DP — Large to Small
#
# Tests whether the NPU can generalize downward: trained on 8^4 data only,
# predict 4^4 and 2^4. The DP memoization hypothesis predicts that forward
# (small→large) transfers better than reverse (large→small), because smaller
# subproblems are components of larger ones but not vice versa.
#
# Design:
#   Phase 1: Train ESN from exp035 8^4 results only (no 2^4 or 4^4)
#   Phase 2: NPU-steered 4^4 with 8^4-only knowledge
#   Phase 3: NPU-steered 2^4 with 8^4-only knowledge
#
# Key metrics to compare with exp035 (forward DP):
#   - H7 confidence at 4^4 and 2^4 (should be lower than forward DP)
#   - CG estimate accuracy (8^4 knows large CG; does it predict small?)
#   - Quenched-length prediction quality (smaller lattices need fewer)
#
# Hardware: RTX 3090 (Motor) + AKD1000 (Cerebellum)
# 2026-03-03

set -euo pipefail

BINARY="./barracuda/target/release/production_dynamical_mixed"
RESULTS="results"
ADAPTER="3090"
BASE_SEED=20260305

export PATH="/home/biomegate/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/local/bin:/usr/bin:/bin"
export HOTSPRING_GPU_ADAPTER="$ADAPTER"

mkdir -p "$RESULTS"

log() { echo "[$(date +%H:%M:%S)] $*"; }

BETAS_FULL="5.0,5.2,5.4,5.5,5.69,5.8,6.0"
WEIGHTS_DIR="$RESULTS/weights"
mkdir -p "$WEIGHTS_DIR"
LOGFILE="$RESULTS/exp037_reverse_dp.log"

# Bootstrap from 8^4 data ONLY — no 2^4 or 4^4
REVERSE_BOOTSTRAP="$RESULTS/exp035_8x8_m0p05.jsonl,$RESULTS/exp035_8x8_m0p1.jsonl"

# ═══════════════════════════════════════════════════════════════
# PHASE 1: 4^4 steered from 8^4-only knowledge
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  REVERSE DP — Phase 1: 4^4 steered from 8^4-only knowledge" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"

for MASS in 0.05 0.1 0.2; do
    MSLUG=$(echo "$MASS" | tr '.' 'p')
    log "  Phase 1 — 4^4, mass=$MASS (reverse DP from 8^4)" | tee -a "$LOGFILE"

    $BINARY \
        --lattice=4 \
        --betas=$BETAS_FULL \
        --mass=$MASS \
        --n-fields=1 \
        --cg-max-iter=5000 \
        --therm=5 \
        --meas=5 \
        --max-adaptive=2 \
        --seed=$((BASE_SEED + 1000)) \
        --bootstrap-from="$REVERSE_BOOTSTRAP" \
        --save-weights="$WEIGHTS_DIR/exp037_reverse_4x4.bin" \
        --output="$RESULTS/exp037_4x4_m${MSLUG}.jsonl" \
        2>&1 | tee -a "$LOGFILE"

    log "  Phase 1 — 4^4 mass=$MASS complete" | tee -a "$LOGFILE"
done

# ═══════════════════════════════════════════════════════════════
# PHASE 2: 2^4 steered from 8^4-only knowledge
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  REVERSE DP — Phase 2: 2^4 steered from 8^4-only knowledge" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"

for MASS in 0.05 0.1 0.2; do
    MSLUG=$(echo "$MASS" | tr '.' 'p')
    log "  Phase 2 — 2^4, mass=$MASS (reverse DP from 8^4)" | tee -a "$LOGFILE"

    $BINARY \
        --lattice=2 \
        --betas=$BETAS_FULL \
        --mass=$MASS \
        --n-fields=1 \
        --cg-max-iter=5000 \
        --therm=3 \
        --meas=5 \
        --max-adaptive=0 \
        --seed=$((BASE_SEED + 2000)) \
        --bootstrap-from="$REVERSE_BOOTSTRAP" \
        --save-weights="$WEIGHTS_DIR/exp037_reverse_2x2.bin" \
        --output="$RESULTS/exp037_2x2_m${MSLUG}.jsonl" \
        2>&1 | tee -a "$LOGFILE"

    log "  Phase 2 — 2^4 mass=$MASS complete" | tee -a "$LOGFILE"
done

# ═══════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  EXPERIMENT 037 COMPLETE — Reverse DP" | tee -a "$LOGFILE"
log "" | tee -a "$LOGFILE"
log "  Compare NPU confidence vs exp035 (forward DP):" | tee -a "$LOGFILE"
log "    exp035 4^4 H7: ~0.48-0.61 (forward from 2^4)" | tee -a "$LOGFILE"
log "    exp037 4^4 H7: ??? (reverse from 8^4)" | tee -a "$LOGFILE"
log "    exp035 2^4 H7: ~0.63 (no bootstrap, dense)" | tee -a "$LOGFILE"
log "    exp037 2^4 H7: ??? (reverse from 8^4)" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
