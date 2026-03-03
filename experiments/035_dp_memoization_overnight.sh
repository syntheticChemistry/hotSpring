#!/usr/bin/env bash
# Experiment 035: DP Memoization Overnight
#
# Tests the core insight: the NPU is a memoization table for dynamic programming.
# Build the table bottom-up (2^4 dense → 4^4 NPU-steered → 8^4 NPU-steered).
#
# Phase 1: Dense 2^4 grid (cheap — seeds the memoization table)
#   - 7 betas × 3 masses × 5 traj each = 105 traj (~15 min total)
#   - NPU trains on this data after each mass sweep
#   - By end of Phase 1, NPU knows the full 2^4 parameter space
#
# Phase 2: NPU-steered 4^4 (medium — NPU predicts from 2^4 knowledge)
#   - Same β/mass grid but with NPU steering (dt, n_md, priority, therm)
#   - NPU uses HeadConfidence — trusted heads steer, untrusted fall back
#   - 7 betas × 3 masses × 5 traj = 105 traj (~2-3 hours)
#   - ReLU-approx-tanh activation (AKD1000-deployable reservoir dynamics)
#
# Phase 3: NPU-steered 8^4 (expensive — NPU predicts from 2^4+4^4 knowledge)
#   - Reduced grid: 5 betas × 2 masses (NPU picks the interesting subset)
#   - 5 betas × 2 masses × 3 traj = 30 traj (~4-6 hours)
#   - This is where we see if the DP memoization saves compute
#
# Key metrics to watch:
#   - NPU head confidence evolution across phases
#   - Therm early-exit rate (should increase as NPU learns)
#   - CG estimate accuracy (should improve phase-over-phase)
#   - Quenched savings percentage
#   - Wall time per trajectory (should decrease with better steering)
#
# Hardware: RTX 3090 (Motor) + AKD1000 (Cerebellum) + Titan V (Pre-motor)
# Activation: ReluTanhApprox (piecewise-linear, AKD1000-native)
# Input: 6D (beta_norm, plaq, mass, chi, acceptance, volume_norm)
#
# 2026-03-03

set -euo pipefail

BINARY="./barracuda/target/release/production_dynamical_mixed"
RESULTS="results"
ADAPTER="3090"
BASE_SEED=20260303

export PATH="/home/biomegate/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/local/bin:/usr/bin:/bin"
export HOTSPRING_GPU_ADAPTER="$ADAPTER"

mkdir -p "$RESULTS"

log() { echo "[$(date +%H:%M:%S)] $*"; }

BETAS_FULL="5.0,5.2,5.4,5.5,5.69,5.8,6.0"
BETAS_FRONTIER="5.4,5.5,5.69,5.8,6.0"
WEIGHTS_DIR="$RESULTS/weights"
mkdir -p "$WEIGHTS_DIR"

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Dense 2^4 Grid — Build the Memoization Table
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════"
log "  PHASE 1: Dense 2^4 Grid — Building DP Memoization Table"
log "  7 β × 3 masses × 5 traj, ReLU-approx-tanh, 6D input"
log "═══════════════════════════════════════════════════════════════"

for MASS in 0.05 0.1 0.2; do
    MSLUG=$(echo "$MASS" | tr '.' 'p')
    log "  Phase 1 — 2^4, mass=$MASS"

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
        --save-weights="$WEIGHTS_DIR/phase1_2x2.bin" \
        --output="$RESULTS/exp035_2x2_m${MSLUG}.jsonl" \
        2>&1 | tee -a "$RESULTS/exp035_dp_overnight.log"

    log "  Phase 1 — 2^4 mass=$MASS complete"
done

log "═══ Phase 1 complete — NPU has dense 2^4 memoization table ═══"
log ""

# ═══════════════════════════════════════════════════════════════
# PHASE 2: NPU-Steered 4^4 — Exploit 2^4 Knowledge
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════"
log "  PHASE 2: NPU-Steered 4^4 — Testing DP Transfer"
log "  Same grid, NPU steering from 2^4 experience"
log "═══════════════════════════════════════════════════════════════"

# Bootstrap from Phase 1 results + saved weights
PHASE1_BOOTSTRAP="$WEIGHTS_DIR/phase1_2x2.bin,$RESULTS/exp035_2x2_m0p05.jsonl,$RESULTS/exp035_2x2_m0p1.jsonl,$RESULTS/exp035_2x2_m0p2.jsonl"

for MASS in 0.05 0.1 0.2; do
    MSLUG=$(echo "$MASS" | tr '.' 'p')
    log "  Phase 2 — 4^4, mass=$MASS (NPU-steered from 2^4 knowledge)"

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
        --save-weights="$WEIGHTS_DIR/phase2_4x4.bin" \
        --output="$RESULTS/exp035_4x4_m${MSLUG}.jsonl" \
        2>&1 | tee -a "$RESULTS/exp035_dp_overnight.log"

    log "  Phase 2 — 4^4 mass=$MASS complete"
done

log "═══ Phase 2 complete — NPU has 2^4 + 4^4 memoization table ═══"
log ""

# ═══════════════════════════════════════════════════════════════
# PHASE 3: NPU-Steered 8^4 — The Expensive Frontier
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════"
log "  PHASE 3: NPU-Steered 8^4 — DP Transfer to New Scale"
log "  Reduced grid (frontier only), NPU has 2^4+4^4 knowledge"
log "═══════════════════════════════════════════════════════════════"

# Bootstrap from Phase 1+2 results + saved weights
PHASE2_BOOTSTRAP="$WEIGHTS_DIR/phase2_4x4.bin,$RESULTS/exp035_2x2_m0p05.jsonl,$RESULTS/exp035_2x2_m0p1.jsonl,$RESULTS/exp035_2x2_m0p2.jsonl,$RESULTS/exp035_4x4_m0p05.jsonl,$RESULTS/exp035_4x4_m0p1.jsonl,$RESULTS/exp035_4x4_m0p2.jsonl"

for MASS in 0.05 0.1; do
    MSLUG=$(echo "$MASS" | tr '.' 'p')
    log "  Phase 3 — 8^4, mass=$MASS (NPU-steered from 2^4+4^4 knowledge)"

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
        --save-weights="$WEIGHTS_DIR/phase3_8x8.bin" \
        --output="$RESULTS/exp035_8x8_m${MSLUG}.jsonl" \
        2>&1 | tee -a "$RESULTS/exp035_dp_overnight.log"

    log "  Phase 3 — 8^4 mass=$MASS complete"
done

log "═══ Phase 3 complete ═══"
log ""

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════════"
log "  EXPERIMENT 035 COMPLETE"
log ""
log "  Phase 1 (2^4 dense):   results/exp035_2x2_*.jsonl"
log "  Phase 2 (4^4 steered): results/exp035_4x4_*.jsonl"
log "  Phase 3 (8^4 steered): results/exp035_8x8_*.jsonl"
log "  Master log:            results/exp035_dp_overnight.log"
log ""
log "  Compare NPU stats across phases:"
log "    - therm_early_exits should increase (2^4 → 4^4 → 8^4)"
log "    - quenched_savings_pct should increase"
log "    - npu_reject_correct should improve"
log "    - HeadConfidence should report more trusted heads"
log "═══════════════════════════════════════════════════════════════"
