#!/usr/bin/env bash
# Experiment 032: Stratified Overnight Run
# Feeds NPU full range of lattice sizes, betas, and dynamical configurations.
# 2026-03-02
#
# Phase 1: 4^4 wide beta scan     — ~40 min  (NPU control ON, validated)
# Phase 2: 6^4 key beta scan      — ~60 min  (NPU control OFF, dt calibrated)
# Phase 3: 8^4 production         — ~5.5 hr  (NPU control OFF, dt calibrated)
# Total: ~7.5 hours

set -euo pipefail

BINARY="./barracuda/target/release/production_dynamical_mixed"
RESULTS="results"
ADAPTER="3090"
BASE_SEED=20260302

export PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/local/bin:/usr/bin:/bin"
export HOTSPRING_GPU_ADAPTER="$ADAPTER"

mkdir -p "$RESULTS"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ═══════════════════════════════════════════════════════════════
# Phase 1: 4^4 Wide Beta Scan (NPU-steered, full range)
# 7 betas × (2 therm + 5 meas) = 49 trajectories × ~42s = ~34 min
# NPU control ON — 4^4 is validated, NPU learns from wide scan
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 1: 4^4 wide beta scan (NPU-steered) ═══"
$BINARY \
    --lattice=4 \
    --betas=5.0,5.3,5.5,5.6,5.69,5.8,6.0 \
    --mass=0.1 \
    --therm=3 \
    --meas=5 \
    --quenched-pretherm=3 \
    --seed=$((BASE_SEED + 1)) \
    --no-titan \
    --trajectory-log="$RESULTS/exp032_4x4_wide_scan.jsonl" \
    2>&1 | tee "$RESULTS/exp032_4x4_wide_scan.log"

log "Phase 1 complete. Trajectories:"
wc -l "$RESULTS/exp032_4x4_wide_scan.jsonl"

# ═══════════════════════════════════════════════════════════════
# Phase 2: 6^4 Key Beta Scan (manual params, volume scaling data)
# 3 betas × (3 therm + 5 meas) = 24 trajectories × ~95s = ~38 min
# dt=0.012 calibrated: 6^4 measured delta_H=0.55 at dt=0.015 → 0.22 at dt=0.012
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 2: 6^4 key beta scan (volume scaling) ═══"
$BINARY \
    --lattice=6 \
    --betas=5.5,5.69,6.0 \
    --mass=0.1 \
    --dt=0.012 \
    --n-md=42 \
    --therm=3 \
    --meas=5 \
    --quenched-pretherm=3 \
    --seed=$((BASE_SEED + 2)) \
    --no-titan \
    --no-npu-control \
    --trajectory-log="$RESULTS/exp032_6x6_key_scan.jsonl" \
    2>&1 | tee "$RESULTS/exp032_6x6_key_scan.log"

log "Phase 2 complete. Trajectories:"
wc -l "$RESULTS/exp032_6x6_key_scan.jsonl"

# ═══════════════════════════════════════════════════════════════
# Phase 3: 8^4 Production (long run, bulk of overnight)
# 3 betas × (10 therm + 30 meas) = 120 trajectories × ~168s = ~5.6 hr
# dt=0.008 from scaling: delta_H ≈ 0.31 → ~75% acceptance
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 3: 8^4 production run ═══"
$BINARY \
    --lattice=8 \
    --betas=5.5,5.69,6.0 \
    --mass=0.1 \
    --dt=0.008 \
    --n-md=63 \
    --therm=10 \
    --meas=30 \
    --quenched-pretherm=5 \
    --seed=$((BASE_SEED + 3)) \
    --no-titan \
    --no-npu-control \
    --trajectory-log="$RESULTS/exp032_8x8_production.jsonl" \
    2>&1 | tee "$RESULTS/exp032_8x8_production.log"

log "Phase 3 complete. Trajectories:"
wc -l "$RESULTS/exp032_8x8_production.jsonl"

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
log "═══ Overnight Complete ═══"
echo "  4^4: $(wc -l < "$RESULTS/exp032_4x4_wide_scan.jsonl") trajectories"
echo "  6^4: $(wc -l < "$RESULTS/exp032_6x6_key_scan.jsonl") trajectories"
echo "  8^4: $(wc -l < "$RESULTS/exp032_8x8_production.jsonl") trajectories"
echo "  Total JSONL files:"
ls -la "$RESULTS"/exp032_*.jsonl
