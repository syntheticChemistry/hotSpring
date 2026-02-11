#!/usr/bin/env bash
# Batch run the 3 PPPM (kappa=0) DSF lite cases
# These require the force_pm.py forceobj fix (numba 0.60 compat)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="$STUDY_DIR/input_files"
LOG="$STUDY_DIR/results/batch_pppm_lite.log"

mkdir -p "$STUDY_DIR/results"

CASES=(
    dsf_k0_G10_mks_lite.yaml
    dsf_k0_G50_mks_lite.yaml
    dsf_k0_G150_mks_lite.yaml
)

TOTAL=${#CASES[@]}
PASSED=0
FAILED=0

echo "========================================" | tee "$LOG"
echo "PPPM Batch Start — $(date)" | tee -a "$LOG"
echo "Cases: $TOTAL" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for i in "${!CASES[@]}"; do
    CASE=${CASES[$i]}
    IDX=$((i + 1))
    echo "" | tee -a "$LOG"
    echo "[$IDX/$TOTAL] Running $CASE" | tee -a "$LOG"
    echo "  Start: $(date)" | tee -a "$LOG"
    
    if python3 "$SCRIPT_DIR/run_case.py" "$INPUT_DIR/$CASE" >> "$LOG" 2>&1; then
        echo "[$IDX/$TOTAL] $CASE — PASSED" | tee -a "$LOG"
        PASSED=$((PASSED + 1))
    else
        echo "[$IDX/$TOTAL] $CASE — FAILED" | tee -a "$LOG"
        FAILED=$((FAILED + 1))
    fi
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Batch Complete — $(date)" | tee -a "$LOG"
echo "  Total: $TOTAL" | tee -a "$LOG"
echo "  Passed: $PASSED" | tee -a "$LOG"
echo "  Failed: $FAILED" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

