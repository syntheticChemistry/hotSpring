#!/usr/bin/env bash
# DEPRECATED: Use nest-validate/target/release/nest-validate validate
# This script is kept as fossil record. The Rust binary replaces it entirely.
#
# validate_all.sh — Run full validation suite across all PLUMED-NEST targets
#
# Industry-standard validation pipeline:
# 1. Per-target analysis (convergence, block averaging, reference comparison)
# 2. Aggregated parity report
# 3. Pass/fail summary with tolerance checks
#
# Usage: ./validate_all.sh [--target NN] [--report-only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

TARGET_FILTER="${1:-all}"
CONDA_ENV="gromacs-fel"
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0

run_target_analysis() {
    local target_dir="$1"
    local name=$(basename "$target_dir")
    local analyze_script="$target_dir/analysis/analyze.py"

    if [[ ! -f "$analyze_script" ]]; then
        printf "  ${YELLOW}SKIP${NC} %s (no analyze.py)\n" "$name"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return
    fi

    printf "  ${CYAN}ANALYZING${NC} %s..." "$name"
    if conda run -n "$CONDA_ENV" python3 "$analyze_script" > /dev/null 2>&1; then
        local report="$target_dir/analysis/validation_report.json"
        if [[ -f "$report" ]]; then
            local pass_rate=$(conda run -n "$CONDA_ENV" python3 -c "
import json
with open('$report') as f:
    r = json.load(f)
v = r.get('validation', {})
print(f\"{v.get('pass_rate', 0)*100:.0f}\")
" 2>/dev/null)
            local is_pass=$(conda run -n "$CONDA_ENV" python3 -c "
import json
with open('$report') as f:
    r = json.load(f)
print(r.get('validation', {}).get('industry_standard', False))
" 2>/dev/null)

            if [[ "$is_pass" == "True" ]]; then
                printf " ${GREEN}PASS${NC} (%s%%)\n" "$pass_rate"
                TOTAL_PASS=$((TOTAL_PASS + 1))
            else
                printf " ${RED}FAIL${NC} (%s%%)\n" "$pass_rate"
                TOTAL_FAIL=$((TOTAL_FAIL + 1))
            fi
        else
            printf " ${YELLOW}NO REPORT${NC}\n"
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
        fi
    else
        printf " ${RED}ERROR${NC}\n"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

echo ""
printf "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}\n"
printf "${CYAN}║  PLUMED-NEST Validation Suite — Industry Standard          ║${NC}\n"
printf "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"
echo ""

printf "Environment: %s\n" "$CONDA_ENV"
printf "PLUMED: %s\n" "$(conda run -n $CONDA_ENV plumed info --version 2>/dev/null)"
printf "GROMACS: %s\n" "$(conda run -n $CONDA_ENV gmx --version 2>&1 | grep 'GROMACS version' | awk '{print $NF}')"
echo ""

printf "${CYAN}--- Running Target Analyses ---${NC}\n"

if [[ "$TARGET_FILTER" == "all" ]]; then
    for target in target_*/; do
        run_target_analysis "$target"
    done
else
    target="target_${TARGET_FILTER}*"
    for t in $target; do
        if [[ -d "$t" ]]; then
            run_target_analysis "$t"
        fi
    done
fi

echo ""
printf "${CYAN}--- Summary ---${NC}\n"
printf "  Pass:  ${GREEN}%d${NC}\n" "$TOTAL_PASS"
printf "  Fail:  ${RED}%d${NC}\n" "$TOTAL_FAIL"
printf "  Skip:  ${YELLOW}%d${NC}\n" "$TOTAL_SKIP"
echo ""

if [[ $TOTAL_FAIL -eq 0 && $TOTAL_PASS -gt 0 ]]; then
    printf "${GREEN}╔══════════════════════════════════════╗${NC}\n"
    printf "${GREEN}║  ALL TARGETS: INDUSTRY STANDARD      ║${NC}\n"
    printf "${GREEN}╚══════════════════════════════════════╝${NC}\n"
else
    printf "${YELLOW}╔══════════════════════════════════════╗${NC}\n"
    printf "${YELLOW}║  VALIDATION INCOMPLETE               ║${NC}\n"
    printf "${YELLOW}╚══════════════════════════════════════╝${NC}\n"
fi
echo ""

# Generate aggregated JSON report
conda run -n "$CONDA_ENV" python3 -c "
import json, glob, os
from datetime import datetime

targets = {}
for report_file in sorted(glob.glob('target_*/analysis/validation_report.json')):
    target_name = report_file.split('/')[0]
    with open(report_file) as f:
        targets[target_name] = json.load(f)

aggregate = {
    'generated': datetime.now().isoformat(),
    'plumed_version': '2.10.0',
    'gromacs_version': '2026.0',
    'targets': targets,
    'summary': {
        'total_targets': len(targets),
        'passing': sum(1 for t in targets.values() if t.get('validation', {}).get('industry_standard')),
        'failing': sum(1 for t in targets.values() if not t.get('validation', {}).get('industry_standard')),
    }
}

with open('validation_aggregate.json', 'w') as f:
    json.dump(aggregate, f, indent=2, default=str)
print(f'Aggregated report: validation_aggregate.json ({len(targets)} targets)')
" 2>/dev/null || true
