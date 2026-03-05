#!/usr/bin/env bash
# Experiment 039: Gen 2 Full-Stream NPU
#
# First run with the rewired NPU architecture:
#   - 21D input vector (15 trajectory + 6 proxy: Anderson + Potts)
#   - 4 specialized sub-models (trajectory predictor, phase oracle, CG cost, steering brain)
#   - All 3 phases (quenched, therm, measurement) stream real data (no zeros)
#   - Sub-model predictions feed into CG cap, phase classification, measurement early-term
#   - Regime-aware CG residual monitoring (thresholds scale near beta_c)
#
# Progressive volume ladder: 4^4 → 8^4, dense masses at each scale.
# Bootstrap from all prior exp035+exp036 data.
#
# Hardware: RTX 3090 (Motor) + AKD1000 (Cerebellum)
# 2026-03-01

set -euo pipefail

BINARY="./barracuda/target/release/production_dynamical_mixed"
RESULTS="results"
ADAPTER="3090"
BASE_SEED=20260301

export PATH="/home/biomegate/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/local/bin:/usr/bin:/bin"
export HOTSPRING_GPU_ADAPTER="$ADAPTER"

mkdir -p "$RESULTS"

log() { echo "[$(date +%H:%M:%S)] $*"; }

BETAS="5.0,5.2,5.4,5.5,5.69,5.8,6.0"
WEIGHTS_DIR="$RESULTS/weights"
mkdir -p "$WEIGHTS_DIR"
LOGFILE="$RESULTS/exp039_gen2_full_stream.log"

# Bootstrap from all available prior data
BOOTSTRAP=""
for W in "$WEIGHTS_DIR"/exp036_phase3_8x8.bin "$WEIGHTS_DIR"/phase3_8x8.bin; do
    [ -f "$W" ] && BOOTSTRAP="$W" && break
done
for F in "$RESULTS"/exp035_*.jsonl "$RESULTS"/exp036_*.jsonl "$RESULTS"/exp038_*.jsonl; do
    [ -f "$F" ] && BOOTSTRAP="$BOOTSTRAP,$F"
done

BOOTSTRAP_FLAG=""
[ -n "$BOOTSTRAP" ] && BOOTSTRAP_FLAG="--bootstrap-from=$BOOTSTRAP"

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  EXPERIMENT 039: Gen 2 Full-Stream NPU" | tee -a "$LOGFILE"
log "  21D input | 4 sub-models | Anderson+Potts proxy | all phases real" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"

# ═══════════════════════════════════════════════════════════════
# PHASE 1: 4^4 — 3 masses across 7 betas
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 1: 4^4 — masses 0.05, 0.1, 0.2 ═══" | tee -a "$LOGFILE"

for MASS in 0.05 0.1 0.2; do
    MSLUG=$(echo "$MASS" | tr '.' 'p')
    log "  4^4 mass=$MASS" | tee -a "$LOGFILE"

    $BINARY \
        --lattice=4 \
        --betas=$BETAS \
        --mass=$MASS \
        --n-fields=1 \
        --cg-max-iter=5000 \
        --therm=10 \
        --meas=20 \
        --quenched-pretherm=10 \
        --max-adaptive=2 \
        --seed=$((BASE_SEED + 1000)) \
        $BOOTSTRAP_FLAG \
        --save-weights="$WEIGHTS_DIR/exp039_phase1_4x4.bin" \
        --output="$RESULTS/exp039_4x4_m${MSLUG}.jsonl" \
        2>&1 | tee -a "$LOGFILE"

    BOOTSTRAP_FLAG="--bootstrap-from=$WEIGHTS_DIR/exp039_phase1_4x4.bin"
    for F in "$RESULTS"/exp039_4x4_*.jsonl; do
        [ -f "$F" ] && BOOTSTRAP_FLAG="$BOOTSTRAP_FLAG,$F"
    done

    log "  4^4 mass=$MASS complete" | tee -a "$LOGFILE"
done

# ═══════════════════════════════════════════════════════════════
# PHASE 2: 8^4 — same masses, bootstrapped from Phase 1
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 2: 8^4 — masses 0.05, 0.1, 0.2 ═══" | tee -a "$LOGFILE"

PHASE2_BOOTSTRAP="$WEIGHTS_DIR/exp039_phase1_4x4.bin"
for F in "$RESULTS"/exp039_4x4_*.jsonl "$RESULTS"/exp035_*.jsonl "$RESULTS"/exp036_*.jsonl; do
    [ -f "$F" ] && PHASE2_BOOTSTRAP="$PHASE2_BOOTSTRAP,$F"
done

BETAS_8="5.4,5.5,5.69,5.8,6.0"

for MASS in 0.05 0.1 0.2; do
    MSLUG=$(echo "$MASS" | tr '.' 'p')
    log "  8^4 mass=$MASS" | tee -a "$LOGFILE"

    $BINARY \
        --lattice=8 \
        --betas=$BETAS_8 \
        --mass=$MASS \
        --n-fields=1 \
        --cg-max-iter=10000 \
        --therm=15 \
        --meas=10 \
        --quenched-pretherm=10 \
        --max-adaptive=3 \
        --seed=$((BASE_SEED + 2000)) \
        --bootstrap-from="$PHASE2_BOOTSTRAP" \
        --save-weights="$WEIGHTS_DIR/exp039_phase2_8x8.bin" \
        --output="$RESULTS/exp039_8x8_m${MSLUG}.jsonl" \
        2>&1 | tee -a "$LOGFILE"

    PHASE2_BOOTSTRAP="$WEIGHTS_DIR/exp039_phase2_8x8.bin"
    for F in "$RESULTS"/exp039_8x8_*.jsonl; do
        [ -f "$F" ] && PHASE2_BOOTSTRAP="$PHASE2_BOOTSTRAP,$F"
    done

    log "  8^4 mass=$MASS complete" | tee -a "$LOGFILE"
done

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  EXPERIMENT 039 COMPLETE" | tee -a "$LOGFILE"
log "" | tee -a "$LOGFILE"
log "  Key metrics to watch:" | tee -a "$LOGFILE"
log "    - Sub-model trust emergence (any head trusted?)" | tee -a "$LOGFILE"
log "    - CG stall predictions from cg_cost_predictor" | tee -a "$LOGFILE"
log "    - Steering brain skip_decision activation" | tee -a "$LOGFILE"
log "    - Proxy features: Anderson ⟨r⟩ + Potts mag correlation" | tee -a "$LOGFILE"
log "    - Regime-aware CG anomaly threshold adaptation" | tee -a "$LOGFILE"
log "  Master log: $LOGFILE" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
