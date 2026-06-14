#!/usr/bin/env bash
# Experiment 036: Extended Forward DP
#
# Fills the mass/beta grid at each volume before scaling to 12^4.
# Builds on exp035 weights — progressive bootstrap with denser coverage.
#
# Phase 1: Dense 2^4 — add masses 0.01, 0.5 to complete 5-mass coverage
# Phase 2: Dense 4^4 — full 5 masses x 7 betas, bootstrapped from 2^4
# Phase 3: Dense 8^4 — full 5 masses x 7 betas, bootstrapped from 2^4+4^4
# Phase 4: Frontier 12^4 — first attempt, mass=0.1 only (NPU extrapolation test)
#
# Key questions:
#   - Does filling mass gaps improve H7 confidence at larger volumes?
#   - Can the NPU extrapolate to 12^4 from 2^4+4^4+8^4 knowledge?
#   - Do more heads cross the trust threshold with denser training data?
#
# Hardware: RTX 3090 (Motor) + AKD1000 (Cerebellum)
# Activation: ReluTanhApprox (piecewise-linear, AKD1000-native)
# Input: 6D (beta_norm, plaq, mass, chi, acceptance, volume_norm)
#
# 2026-03-03

set -euo pipefail

BINARY="./barracuda/target/release/production_dynamical_mixed"
RESULTS="results"
ADAPTER="3090"
BASE_SEED=20260304

export PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/local/bin:/usr/bin:/bin"
export HOTSPRING_GPU_ADAPTER="$ADAPTER"

mkdir -p "$RESULTS"

log() { echo "[$(date +%H:%M:%S)] $*"; }

BETAS_FULL="5.0,5.2,5.4,5.5,5.69,5.8,6.0"
BETAS_FRONTIER="5.4,5.5,5.69,5.8,6.0"
WEIGHTS_DIR="$RESULTS/weights"
mkdir -p "$WEIGHTS_DIR"
LOGFILE="$RESULTS/exp036_dp_extended.log"

# Bootstrap from exp035 weights (carry forward all previous learning)
EXP035_WEIGHTS="$WEIGHTS_DIR/phase3_8x8.bin"

# ═══════════════════════════════════════════════════════════════
# PHASE 1: 2^4 Gap Fill — masses 0.01, 0.5 (exp035 had 0.05, 0.1, 0.2)
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  PHASE 1: 2^4 Gap Fill — masses 0.01, 0.5" | tee -a "$LOGFILE"
log "  Bootstrapping from exp035 8^4 weights" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"

PHASE0_BOOTSTRAP="$EXP035_WEIGHTS"
# Include all exp035 data for rich bootstrap
for F in "$RESULTS"/exp035_*.jsonl; do
    [ -f "$F" ] && PHASE0_BOOTSTRAP="$PHASE0_BOOTSTRAP,$F"
done

for MASS in 0.01 0.5; do
    MSLUG=$(echo "$MASS" | tr '.' 'p')
    log "  Phase 1 — 2^4, mass=$MASS" | tee -a "$LOGFILE"

    $BINARY \
        --lattice=2 \
        --betas=$BETAS_FULL \
        --mass=$MASS \
        --n-fields=1 \
        --cg-max-iter=5000 \
        --therm=3 \
        --meas=5 \
        --max-adaptive=0 \
        --seed=$((BASE_SEED + 1000)) \
        --bootstrap-from="$PHASE0_BOOTSTRAP" \
        --save-weights="$WEIGHTS_DIR/exp036_phase1_2x2.bin" \
        --output="$RESULTS/exp036_2x2_m${MSLUG}.jsonl" \
        2>&1 | tee -a "$LOGFILE"

    log "  Phase 1 — 2^4 mass=$MASS complete" | tee -a "$LOGFILE"
done

log "═══ Phase 1 complete — 2^4 now has 5 masses ═══" | tee -a "$LOGFILE"

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Dense 4^4 — 5 masses x 7 betas
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  PHASE 2: Dense 4^4 — 5 masses x 7 betas" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"

PHASE1_BOOTSTRAP="$WEIGHTS_DIR/exp036_phase1_2x2.bin"
for F in "$RESULTS"/exp035_2x2_*.jsonl "$RESULTS"/exp036_2x2_*.jsonl; do
    [ -f "$F" ] && PHASE1_BOOTSTRAP="$PHASE1_BOOTSTRAP,$F"
done

for MASS in 0.01 0.05 0.1 0.2 0.5; do
    MSLUG=$(echo "$MASS" | tr '.' 'p')
    OUT="$RESULTS/exp036_4x4_m${MSLUG}.jsonl"
    # Skip if exp035 already has this mass
    if [ -f "$RESULTS/exp035_4x4_m${MSLUG}.jsonl" ] && [ "$MASS" != "0.01" ] && [ "$MASS" != "0.5" ]; then
        log "  Phase 2 — 4^4 mass=$MASS: reusing exp035 data" | tee -a "$LOGFILE"
        continue
    fi
    log "  Phase 2 — 4^4, mass=$MASS (NPU-steered from 2^4 knowledge)" | tee -a "$LOGFILE"

    $BINARY \
        --lattice=4 \
        --betas=$BETAS_FULL \
        --mass=$MASS \
        --n-fields=1 \
        --cg-max-iter=5000 \
        --therm=5 \
        --meas=5 \
        --max-adaptive=2 \
        --seed=$((BASE_SEED + 2000)) \
        --bootstrap-from="$PHASE1_BOOTSTRAP" \
        --save-weights="$WEIGHTS_DIR/exp036_phase2_4x4.bin" \
        --output="$OUT" \
        2>&1 | tee -a "$LOGFILE"

    log "  Phase 2 — 4^4 mass=$MASS complete" | tee -a "$LOGFILE"
done

log "═══ Phase 2 complete — 4^4 has 5 masses ═══" | tee -a "$LOGFILE"

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Dense 8^4 — 5 masses x 7 betas
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  PHASE 3: Dense 8^4 — 5 masses x 7 betas" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"

PHASE2_BOOTSTRAP="$WEIGHTS_DIR/exp036_phase2_4x4.bin"
for F in "$RESULTS"/exp035_*.jsonl "$RESULTS"/exp036_2x2_*.jsonl "$RESULTS"/exp036_4x4_*.jsonl; do
    [ -f "$F" ] && PHASE2_BOOTSTRAP="$PHASE2_BOOTSTRAP,$F"
done

for MASS in 0.01 0.05 0.1 0.2 0.5; do
    MSLUG=$(echo "$MASS" | tr '.' 'p')
    OUT="$RESULTS/exp036_8x8_m${MSLUG}.jsonl"
    if [ -f "$RESULTS/exp035_8x8_m${MSLUG}.jsonl" ] && [ "$MASS" != "0.01" ] && [ "$MASS" != "0.5" ]; then
        log "  Phase 3 — 8^4 mass=$MASS: reusing exp035 data" | tee -a "$LOGFILE"
        continue
    fi
    log "  Phase 3 — 8^4, mass=$MASS (NPU-steered from 2^4+4^4 knowledge)" | tee -a "$LOGFILE"

    $BINARY \
        --lattice=8 \
        --betas=$BETAS_FRONTIER \
        --mass=$MASS \
        --n-fields=1 \
        --cg-max-iter=10000 \
        --therm=8 \
        --meas=3 \
        --max-adaptive=3 \
        --seed=$((BASE_SEED + 3000)) \
        --bootstrap-from="$PHASE2_BOOTSTRAP" \
        --save-weights="$WEIGHTS_DIR/exp036_phase3_8x8.bin" \
        --output="$OUT" \
        2>&1 | tee -a "$LOGFILE"

    log "  Phase 3 — 8^4 mass=$MASS complete" | tee -a "$LOGFILE"
done

log "═══ Phase 3 complete — 8^4 has 5 masses ═══" | tee -a "$LOGFILE"

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Frontier 12^4 — NPU extrapolation test
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  PHASE 4: Frontier 12^4 — NPU extrapolation to unseen scale" | tee -a "$LOGFILE"
log "  mass=0.1 only, frontier betas, 2 measurements" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"

PHASE3_BOOTSTRAP="$WEIGHTS_DIR/exp036_phase3_8x8.bin"
for F in "$RESULTS"/exp035_*.jsonl "$RESULTS"/exp036_*.jsonl; do
    [ -f "$F" ] && PHASE3_BOOTSTRAP="$PHASE3_BOOTSTRAP,$F"
done

$BINARY \
    --lattice=12 \
    --betas=$BETAS_FRONTIER \
    --mass=0.1 \
    --n-fields=1 \
    --cg-max-iter=20000 \
    --therm=10 \
    --meas=2 \
    --max-adaptive=3 \
    --seed=$((BASE_SEED + 4000)) \
    --bootstrap-from="$PHASE3_BOOTSTRAP" \
    --save-weights="$WEIGHTS_DIR/exp036_phase4_12x12.bin" \
    --output="$RESULTS/exp036_12x12_m0p1.jsonl" \
    2>&1 | tee -a "$LOGFILE"

log "═══ Phase 4 complete ═══" | tee -a "$LOGFILE"

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
log "  EXPERIMENT 036 COMPLETE" | tee -a "$LOGFILE"
log "" | tee -a "$LOGFILE"
log "  Phase 1 (2^4 gap fill):   exp036_2x2_m0p01, m0p5" | tee -a "$LOGFILE"
log "  Phase 2 (4^4 dense):      exp036_4x4_m0p01, m0p5" | tee -a "$LOGFILE"
log "  Phase 3 (8^4 dense):      exp036_8x8_m0p01, m0p5" | tee -a "$LOGFILE"
log "  Phase 4 (12^4 frontier):  exp036_12x12_m0p1" | tee -a "$LOGFILE"
log "  Master log:               $LOGFILE" | tee -a "$LOGFILE"
log "" | tee -a "$LOGFILE"
log "  Key metrics:" | tee -a "$LOGFILE"
log "    - H7 confidence at 12^4 (extrapolation quality)" | tee -a "$LOGFILE"
log "    - New heads crossing trust threshold" | tee -a "$LOGFILE"
log "    - CG estimate accuracy improvement with denser data" | tee -a "$LOGFILE"
log "    - Quenched savings at 12^4" | tee -a "$LOGFILE"
log "═══════════════════════════════════════════════════════════════" | tee -a "$LOGFILE"
