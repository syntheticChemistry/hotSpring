#!/usr/bin/env bash
# Experiment 034: Reality Ladder — Rung 1 (Multi-Field Nf Comparison)
# Compares Nf=4 (1 field), Nf=8 (2 fields), Nf=12 (3 fields) on 4^4.
# Fixed mass=0.1 (safe regime from Rung 0), varying beta across transition.
#
# The NPU already has Nf=4 data from Rung 0 and Exp 032.
# Here it sees how extra flavors shift the transition and CG cost.
#
# Phase 1a: Nf=4 baseline   — 5 betas × 5 traj = 25 traj (~20 min)
# Phase 1b: Nf=8            — 5 betas × 5 traj = 25 traj (~40 min, 2x CG)
# Phase 1c: Nf=12           — 5 betas × 5 traj = 25 traj (~60 min, 3x CG)
# Phase 1d: Nf=4 light mass — 3 betas × 5 traj = 15 traj (~25 min)
# Phase 1e: Nf=8 light mass — 3 betas × 5 traj = 15 traj (~50 min)
# Total: ~105 traj, ~3-4 hours
#
# Expected physics:
# - More flavors = stronger vacuum screening = transition shifts to lower beta
# - CG cost per trajectory scales linearly with n_fields
# - Plaquette should increase (more fluctuations) with more flavors
#
# 2026-03-03

set -euo pipefail

BINARY="./barracuda/target/release/production_dynamical_mixed"
RESULTS="results"
ADAPTER="3090"
BASE_SEED=20260303

export PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/local/bin:/usr/bin:/bin"
export HOTSPRING_GPU_ADAPTER="$ADAPTER"

mkdir -p "$RESULTS"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ═══════════════════════════════════════════════════════════════
# Phase 1a: Nf=4 baseline (1 field) — mass=0.1, 4^4
# Reference data for comparison. NPU has Rung 0 data at this point.
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 1a: Nf=4 (1 field) baseline, mass=0.1, 4^4 ═══"

$BINARY \
    --lattice=4 \
    --betas=5.0,5.3,5.5,5.69,6.0 \
    --mass=0.1 \
    --n-fields=1 \
    --cg-max-iter=5000 \
    --therm=2 \
    --meas=3 \
    --max-adaptive=0 \
    --seed=$BASE_SEED \
    --output="$RESULTS/exp034_4x4_nf4_m0p1.jsonl" \
    2>&1 | tee -a "$RESULTS/exp034_rung1_master.log"

log "  Phase 1a complete"

# ═══════════════════════════════════════════════════════════════
# Phase 1b: Nf=8 (2 fields) — mass=0.1, 4^4
# First multi-field run. CG cost should roughly double.
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 1b: Nf=8 (2 fields), mass=0.1, 4^4 ═══"

$BINARY \
    --lattice=4 \
    --betas=5.0,5.3,5.5,5.69,6.0 \
    --mass=0.1 \
    --n-fields=2 \
    --cg-max-iter=5000 \
    --therm=2 \
    --meas=3 \
    --max-adaptive=0 \
    --seed=$((BASE_SEED + 100)) \
    --output="$RESULTS/exp034_4x4_nf8_m0p1.jsonl" \
    2>&1 | tee -a "$RESULTS/exp034_rung1_master.log"

log "  Phase 1b complete"

# ═══════════════════════════════════════════════════════════════
# Phase 1c: Nf=12 (3 fields) — mass=0.1, 4^4
# Heavy flavor content. Transition should shift noticeably.
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 1c: Nf=12 (3 fields), mass=0.1, 4^4 ═══"

$BINARY \
    --lattice=4 \
    --betas=5.0,5.3,5.5,5.69,6.0 \
    --mass=0.1 \
    --n-fields=3 \
    --cg-max-iter=5000 \
    --therm=2 \
    --meas=3 \
    --max-adaptive=0 \
    --seed=$((BASE_SEED + 200)) \
    --output="$RESULTS/exp034_4x4_nf12_m0p1.jsonl" \
    2>&1 | tee -a "$RESULTS/exp034_rung1_master.log"

log "  Phase 1c complete"

# ═══════════════════════════════════════════════════════════════
# Phase 1d: Nf=4 at lighter mass=0.05, 4^4
# Cross-reference: does lighter mass at Nf=4 look like heavier mass at Nf=8?
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 1d: Nf=4 (1 field), mass=0.05, 4^4 ═══"

$BINARY \
    --lattice=4 \
    --betas=5.5,5.69,6.0 \
    --mass=0.05 \
    --n-fields=1 \
    --cg-max-iter=5000 \
    --therm=2 \
    --meas=3 \
    --max-adaptive=0 \
    --seed=$((BASE_SEED + 300)) \
    --output="$RESULTS/exp034_4x4_nf4_m0p05.jsonl" \
    2>&1 | tee -a "$RESULTS/exp034_rung1_master.log"

log "  Phase 1d complete"

# ═══════════════════════════════════════════════════════════════
# Phase 1e: Nf=8 at lighter mass=0.05, 4^4
# Heaviest fermion load at lighter mass — test the boundary.
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 1e: Nf=8 (2 fields), mass=0.05, 4^4 ═══"

$BINARY \
    --lattice=4 \
    --betas=5.5,5.69,6.0 \
    --mass=0.05 \
    --n-fields=2 \
    --cg-max-iter=5000 \
    --therm=2 \
    --meas=3 \
    --max-adaptive=0 \
    --seed=$((BASE_SEED + 400)) \
    --output="$RESULTS/exp034_4x4_nf8_m0p05.jsonl" \
    2>&1 | tee -a "$RESULTS/exp034_rung1_master.log"

log "  Phase 1e complete"

log "═══ Experiment 034 COMPLETE ═══"
log "Results in: $RESULTS/exp034_*.jsonl"
log "Master log: $RESULTS/exp034_rung1_master.log"
