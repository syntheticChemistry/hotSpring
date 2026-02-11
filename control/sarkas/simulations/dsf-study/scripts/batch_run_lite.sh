#!/usr/bin/env bash
# Batch runner for all lite DSF cases (PP method, kappa >= 1)
# Runs pre-processing + simulation + post-processing for each case sequentially.
# Logs output to results/batch_lite.log
#
# Usage: bash scripts/batch_run_lite.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(dirname "$SCRIPT_DIR")"
cd "$STUDY_DIR"

# Activate sarkas environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate sarkas

LOG="results/batch_lite.log"
mkdir -p results

echo "========================================" | tee "$LOG"
echo "DSF Lite Batch Run — $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# All lite PP cases (excluding k1_G14 which is already done)
CASES=(
    "dsf_k1_G72_mks_lite.yaml"
    "dsf_k1_G217_mks_lite.yaml"
    "dsf_k2_G31_mks_lite.yaml"
    "dsf_k2_G158_mks_lite.yaml"
    "dsf_k2_G476_mks_lite.yaml"
    "dsf_k3_G100_mks_lite.yaml"
    "dsf_k3_G503_mks_lite.yaml"
    "dsf_k3_G1510_mks_lite.yaml"
)

TOTAL=${#CASES[@]}
DONE=0
FAILED=0

for case in "${CASES[@]}"; do
    DONE=$((DONE + 1))
    echo "" | tee -a "$LOG"
    echo "[$DONE/$TOTAL] Running $case — $(date)" | tee -a "$LOG"
    echo "--------------------------------------------" | tee -a "$LOG"
    
    if python scripts/run_case.py "input_files/$case" 2>&1 | tee -a "$LOG"; then
        echo "[$DONE/$TOTAL] $case — PASSED" | tee -a "$LOG"
    else
        echo "[$DONE/$TOTAL] $case — FAILED (exit $?)" | tee -a "$LOG"
        FAILED=$((FAILED + 1))
    fi
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Batch Complete — $(date)" | tee -a "$LOG"
echo "  Total: $TOTAL" | tee -a "$LOG"
echo "  Passed: $((TOTAL - FAILED))" | tee -a "$LOG"
echo "  Failed: $FAILED" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

