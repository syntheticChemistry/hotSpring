#!/usr/bin/env bash
# ============================================================
# hotSpring — Full data regeneration on a fresh clone
# ============================================================
# This script restores ALL upstream dependencies and regenerates
# ALL experimental data that is gitignored (>21 GB).
#
# Usage:
#   bash scripts/regenerate-all.sh              # Full regeneration
#   bash scripts/regenerate-all.sh --deps-only  # Just clone + env setup
#   bash scripts/regenerate-all.sh --sarkas     # Only Sarkas experiments
#   bash scripts/regenerate-all.sh --surrogate  # Only surrogate learning
#   bash scripts/regenerate-all.sh --ttm        # Only TTM experiments
#   bash scripts/regenerate-all.sh --nuclear    # Only nuclear EOS (L1+L2)
#   bash scripts/regenerate-all.sh --dry-run    # Show what would be done
#
# Prerequisites:
#   - micromamba or conda
#   - curl (for Zenodo download)
#   - ~30 GB free disk space
#   - GPU recommended for surrogate/nuclear-eos (RTX 4070 or better)
#
# Time estimates (Eastgate, i9-12900K + RTX 4070):
#   Dependencies:    ~10 min (clone + download + env setup)
#   Sarkas DSF:      ~3 hours (12 cases × ~15 min each)
#   Surrogate:       ~5.5 hours (benchmark + iterative workflow)
#   Nuclear EOS L1:  ~3 min (Python control)
#   Nuclear EOS L2:  ~3.2 hours (Python control, 8 workers)
#   TTM:             ~1 hour (local + hydro, all species)
#   TOTAL:           ~12 hours
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
MODE="${1:-all}"
DRY_RUN=false
if [[ "$MODE" == "--dry-run" ]]; then
    DRY_RUN=true
    MODE="all"
fi

# Detect package manager
if command -v micromamba &> /dev/null; then
    PM_RUN="micromamba run -n"
elif command -v conda &> /dev/null; then
    PM_RUN="conda run --no-banner -n"
else
    echo -e "${RED}ERROR: Neither micromamba nor conda found.${NC}"
    exit 1
fi

# ============================================================
header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  $1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
}

step() {
    echo -e "  ${GREEN}▸${NC} $1"
}

skip() {
    echo -e "  ${YELLOW}⊘${NC} $1 (skipping)"
}

run_or_dry() {
    if $DRY_RUN; then
        echo -e "  ${YELLOW}[dry-run]${NC} $*"
    else
        "$@"
    fi
}

# ============================================================
# Phase 0: Dependencies (always runs unless specific experiment selected)
# ============================================================
if [[ "$MODE" == "all" || "$MODE" == "--deps-only" ]]; then
    header "Phase 0: Dependencies"

    step "Cloning upstream repositories..."
    run_or_dry bash "$SCRIPT_DIR/clone-repos.sh"

    step "Downloading Zenodo data (~6 GB)..."
    run_or_dry bash "$SCRIPT_DIR/download-data.sh"

    step "Setting up Python environments..."
    run_or_dry bash "$SCRIPT_DIR/setup-envs.sh"

    if [[ "$MODE" == "--deps-only" ]]; then
        echo ""
        echo -e "${GREEN}Dependencies installed. Ready to run experiments.${NC}"
        exit 0
    fi
fi

# ============================================================
# Phase 1: Sarkas MD (12 DSF cases)
# ============================================================
if [[ "$MODE" == "all" || "$MODE" == "--sarkas" ]]; then
    header "Phase 1: Sarkas MD — DSF Study (12 cases)"
    step "Running 9 PP cases (κ=1,2,3 × 3 Γ values)..."
    step "Estimated time: ~2 hours"

    DSF_DIR="$PROJECT_DIR/control/sarkas/simulations/dsf-study"

    run_or_dry $PM_RUN sarkas bash "$DSF_DIR/scripts/batch_run_lite.sh"

    step "Running 3 PPPM cases (κ=0 × 3 Γ values)..."
    step "Estimated time: ~1 hour"

    run_or_dry $PM_RUN sarkas bash "$DSF_DIR/scripts/batch_run_pppm_lite.sh"

    step "Validating DSF against Dense Plasma Properties Database..."
    run_or_dry $PM_RUN sarkas python "$DSF_DIR/scripts/validate_dsf.py"
    run_or_dry $PM_RUN sarkas python "$DSF_DIR/scripts/validate_pppm_batch.py"

    step "Validating all observables (DSF, RDF, SSF, VACF, Energy)..."
    run_or_dry $PM_RUN sarkas python "$DSF_DIR/scripts/validate_all_observables.py"

    echo -e "  ${GREEN}✓ Sarkas complete — results in $DSF_DIR/results/${NC}"
fi

# ============================================================
# Phase 2: Surrogate Learning
# ============================================================
if [[ "$MODE" == "all" || "$MODE" == "--surrogate" ]]; then
    header "Phase 2: Surrogate Learning — Benchmark + Workflow"

    step "Running benchmark functions (quick validation)..."
    run_or_dry $PM_RUN surrogate python "$PROJECT_DIR/control/surrogate/scripts/run_benchmark_functions.py"

    step "Running full iterative workflow (9 objectives × 30 rounds)..."
    step "Estimated time: ~5 hours"
    run_or_dry $PM_RUN surrogate python "$PROJECT_DIR/control/surrogate/scripts/full_iterative_workflow.py"

    step "Verifying results..."
    run_or_dry $PM_RUN surrogate python "$PROJECT_DIR/control/surrogate/scripts/verify_results.py"

    echo -e "  ${GREEN}✓ Surrogate complete — results in control/surrogate/results/${NC}"
fi

# ============================================================
# Phase 3: Nuclear EOS (L1 + L2)
# ============================================================
if [[ "$MODE" == "all" || "$MODE" == "--nuclear" ]]; then
    header "Phase 3: Nuclear EOS — L1 (SEMF) + L2 (HF+BCS)"

    NEOS_DIR="$PROJECT_DIR/control/surrogate/nuclear-eos"

    step "Ensuring AME2020 experimental data exists..."
    if [ ! -f "$NEOS_DIR/exp_data/ame2020_selected.json" ]; then
        run_or_dry bash "$NEOS_DIR/exp_data/download_ame2020.sh"
    else
        echo "    ame2020_selected.json already exists — skipping download"
    fi

    step "Running L1 surrogate (SEMF, 52 nuclei, ~3 min)..."
    run_or_dry $PM_RUN surrogate python "$NEOS_DIR/scripts/run_surrogate.py" --level=1

    step "Running L2 surrogate (HF+BCS, 18 nuclei, ~3.2 hours)..."
    step "Estimated time: ~3.2 hours (8 parallel workers + GPU RBF)"
    run_or_dry $PM_RUN surrogate python "$NEOS_DIR/scripts/run_surrogate.py" --level=2

    echo -e "  ${GREEN}✓ Nuclear EOS complete — results in $NEOS_DIR/results/${NC}"
fi

# ============================================================
# Phase 4: TTM (Local + Hydro)
# ============================================================
if [[ "$MODE" == "all" || "$MODE" == "--ttm" ]]; then
    header "Phase 4: TTM — Local + Hydro Models"

    step "Running local model (ODE equilibration, 3 species)..."
    run_or_dry $PM_RUN ttm python "$PROJECT_DIR/control/ttm/scripts/run_local_model.py"

    step "Running hydro model (spatial profiles, 3 species)..."
    step "Estimated time: ~45 min"
    run_or_dry $PM_RUN ttm python "$PROJECT_DIR/control/ttm/scripts/run_hydro_model.py"

    echo -e "  ${GREEN}✓ TTM complete — results in control/ttm/reproduction/${NC}"
fi

# ============================================================
# Phase 5: Akida NPU driver (optional, needs root)
# ============================================================
if [[ "$MODE" == "all" ]]; then
    header "Phase 5: Akida NPU Driver (optional)"

    if [ -c "/dev/akida0" ]; then
        echo -e "  ${GREEN}✓ /dev/akida0 already present — driver loaded${NC}"
    else
        step "Akida driver needs kernel module build (requires root)"
        step "To build manually:"
        echo "    cd control/akida_dw_edma && make && sudo make install"
        echo "    sudo modprobe akida-pcie"
    fi
fi

# ============================================================
# Summary
# ============================================================
header "Regeneration Complete"
echo ""
echo "  Data locations:"
echo "    Sarkas simulations:    control/sarkas/simulations/dsf-study/Simulations/"
echo "    Sarkas results:        control/sarkas/simulations/dsf-study/results/"
echo "    Surrogate results:     control/surrogate/results/"
echo "    Nuclear EOS results:   control/surrogate/nuclear-eos/results/"
echo "    TTM reproduction:      control/ttm/reproduction/"
echo ""
echo "  Result JSONs (tracked in git):"
echo "    control/comprehensive_control_results.json"
echo "    control/sarkas/simulations/dsf-study/results/*.json"
echo "    control/surrogate/results/*.json"
echo "    control/surrogate/nuclear-eos/results/*.json"
echo ""
echo "  To verify everything:"
echo "    $PM_RUN surrogate python control/surrogate/scripts/verify_results.py"
echo ""
if $DRY_RUN; then
    echo -e "  ${YELLOW}This was a dry run — no commands were executed.${NC}"
    echo "  Remove --dry-run to execute."
fi

