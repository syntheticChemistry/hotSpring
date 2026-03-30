#!/usr/bin/env bash
# Experiment 033: Reality Ladder — Rung 0 (Mass Scan at Nf=4)
# Mass x Beta x Volume training grid for NPU.
# No code changes required — varies --mass across validated lattice sizes.
#
# Phase 0a: 2^4 mass x beta grid     — ~15 min  (NPU-steered, fast interpolation data)
# Phase 0b: 4^4 mass x beta grid     — ~2.5 hr  (NPU-steered, mass-dependent steering)
# Phase 0c: 6^4 mass scan at key β   — ~2.5 hr  (manual params, volume scaling)
# Total: ~5.5 hours
#
# Masses: 0.5 (heavy), 0.2, 0.1 (baseline), 0.05, 0.01 (near-chiral)
# Lighter masses => higher CG cost, same delta_H at fixed volume/dt.
#
# 2026-03-02

set -euo pipefail

BINARY="./barracuda/target/release/production_dynamical_mixed"
RESULTS="results"
ADAPTER="3090"
BASE_SEED=20260303

export PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/local/bin:/usr/bin:/bin"
export HOTSPRING_GPU_ADAPTER="$ADAPTER"

mkdir -p "$RESULTS"

log() { echo "[$(date +%H:%M:%S)] $*"; }

MASSES_FULL="0.5,0.2,0.1,0.05,0.01"
MASSES_KEY="0.2,0.1,0.05"

# ═══════════════════════════════════════════════════════════════
# Phase 0a: 2^4 Mass x Beta Grid (fast NPU training data)
# 5 masses x 3 betas = 15 combos × (2 therm + 3 meas) = 75 traj
# ~5-10s per trajectory => ~10-15 min total
# NPU-steered: small lattice, let NPU learn mass patterns
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 0a: 2^4 Mass x Beta Grid ═══"

for mass in 0.5 0.2 0.1 0.05 0.01; do
    mass_tag=$(echo "$mass" | tr '.' 'p')
    log "  2^4 mass=$mass"
    $BINARY \
        --lattice=2 \
        --betas=5.0,5.5,6.0 \
        --mass="$mass" \
        --cg-max-iter=5000 \
        --therm=2 \
        --meas=3 \
        --quenched-pretherm=2 \
        --seed=$((BASE_SEED + 100)) \
        --no-titan \
        --max-adaptive=0 \
        --trajectory-log="$RESULTS/exp033_2x2_m${mass_tag}.jsonl" \
        2>&1 | tee -a "$RESULTS/exp033_2x2_mass_scan.log"
done

log "Phase 0a complete."
echo "  2^4 trajectories per mass:"
wc -l "$RESULTS"/exp033_2x2_m*.jsonl 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
# Phase 0b: 4^4 Mass x Beta Grid (NPU-steered mass learning)
# 5 masses x 5 betas = 25 combos × (3 therm + 5 meas) = 200 traj
# ~42s per trajectory => ~2.3 hr
# NPU-steered: validated on 4^4, NPU sees mass as new dimension
# cg-max-iter=5000 for safety at light masses
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 0b: 4^4 Mass x Beta Grid ═══"

for mass in 0.5 0.2 0.1 0.05 0.01; do
    mass_tag=$(echo "$mass" | tr '.' 'p')
    log "  4^4 mass=$mass"
    $BINARY \
        --lattice=4 \
        --betas=5.0,5.3,5.5,5.69,6.0 \
        --mass="$mass" \
        --cg-max-iter=5000 \
        --therm=3 \
        --meas=5 \
        --quenched-pretherm=3 \
        --seed=$((BASE_SEED + 200)) \
        --no-titan \
        --trajectory-log="$RESULTS/exp033_4x4_m${mass_tag}.jsonl" \
        2>&1 | tee -a "$RESULTS/exp033_4x4_mass_scan.log"
done

log "Phase 0b complete."
echo "  4^4 trajectories per mass:"
wc -l "$RESULTS"/exp033_4x4_m*.jsonl 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
# Phase 0c: 6^4 Mass Scan at Key Betas (volume scaling with mass)
# 3 masses x 3 betas = 9 combos × (3 therm + 5 meas) = 72 traj
# ~95s per trajectory => ~1.9 hr
# Manual params: dt=0.012, n_md=42 (calibrated from Exp 032)
# Skip mass=0.5 (trivially heavy) and mass=0.01 (CG very expensive at 6^4)
# ═══════════════════════════════════════════════════════════════

log "═══ Phase 0c: 6^4 Mass Scan at Key Betas ═══"

for mass in 0.2 0.1 0.05; do
    mass_tag=$(echo "$mass" | tr '.' 'p')
    log "  6^4 mass=$mass"
    $BINARY \
        --lattice=6 \
        --betas=5.5,5.69,6.0 \
        --mass="$mass" \
        --dt=0.012 \
        --n-md=42 \
        --cg-max-iter=5000 \
        --therm=3 \
        --meas=5 \
        --quenched-pretherm=3 \
        --seed=$((BASE_SEED + 300)) \
        --no-titan \
        --no-npu-control \
        --trajectory-log="$RESULTS/exp033_6x6_m${mass_tag}.jsonl" \
        2>&1 | tee -a "$RESULTS/exp033_6x6_mass_scan.log"
done

log "Phase 0c complete."
echo "  6^4 trajectories per mass:"
wc -l "$RESULTS"/exp033_6x6_m*.jsonl 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
log "═══ Rung 0 Complete ═══"
echo ""
echo "Result files:"
ls -la "$RESULTS"/exp033_*.jsonl 2>/dev/null || true
echo ""
echo "Total trajectories:"
cat "$RESULTS"/exp033_*.jsonl 2>/dev/null | wc -l
echo ""
echo "Mass scan grid:"
echo "  2^4: 5 masses x 3 betas (NPU-steered)"
echo "  4^4: 5 masses x 5 betas (NPU-steered)"
echo "  6^4: 3 masses x 3 betas (manual dt=0.012, n_md=42)"
echo ""
echo "Ready for Rung 1 (multi-field Nf=8+) analysis."
