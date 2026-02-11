#!/usr/bin/env bash
# hotSpring environment setup
# Run from hotSpring/ directory
#
# Supports both conda and micromamba (prefers micromamba).
# Non-interactive: safe for CI and automated runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_DIR/envs"

echo "=== hotSpring Environment Setup ==="
echo "Project directory: $PROJECT_DIR"
echo ""

# ---- Detect package manager ----
if command -v micromamba &> /dev/null; then
    PM="micromamba"
    PM_CREATE="micromamba create -y"
    PM_RUN="micromamba run -n"
    PM_LIST="micromamba env list"
elif command -v conda &> /dev/null; then
    PM="conda"
    PM_CREATE="conda env create -y"
    PM_RUN="conda run --no-banner -n"
    PM_LIST="conda env list"
else
    echo "ERROR: Neither micromamba nor conda found."
    echo "  Install micromamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
    echo "  Or conda:          https://docs.conda.io/projects/miniconda/en/latest/"
    exit 1
fi
echo "Package manager: $PM ($($PM --version 2>/dev/null || echo 'version unknown'))"
echo ""

# ---- Check system prerequisites ----
echo "--- Checking system prerequisites ---"

# FFTW3 (needed for Sarkas PPPM)
if pkg-config --exists fftw3 2>/dev/null; then
    echo "  FFTW3: $(pkg-config --modversion fftw3)"
elif ldconfig -p 2>/dev/null | grep -q libfftw3; then
    echo "  FFTW3: found (via ldconfig)"
else
    echo "  WARNING: FFTW3 not found. Sarkas PPPM method needs it."
    echo "    Install: sudo apt install libfftw3-dev"
fi

# gfortran (needed for FMM3D)
if command -v gfortran &> /dev/null; then
    echo "  gfortran: $(gfortran --version | head -1)"
else
    echo "  WARNING: gfortran not found. Needed for FMM3D compilation."
    echo "    Install: sudo apt install gfortran"
fi

echo ""

# ---- Create environments ----
create_env() {
    local yaml_file="$1"
    local env_name="$2"
    local req_file="${3:-}"  # optional requirements.txt for post-install

    if $PM_LIST 2>/dev/null | grep -q "$env_name"; then
        echo "--- Environment '$env_name' already exists. Skipping. ---"
        echo "    To recreate: $PM env remove -n $env_name && bash $0"
    else
        echo "--- Creating environment '$env_name' from $yaml_file ---"
        if [ "$PM" = "micromamba" ]; then
            micromamba create -y -n "$env_name" -f "$yaml_file"
        else
            conda env create -y -f "$yaml_file"
        fi
        echo "    Done."
    fi

    # Post-install from requirements.txt if provided
    if [ -n "$req_file" ] && [ -f "$req_file" ]; then
        echo "    Installing pinned deps from $req_file..."
        $PM_RUN "$env_name" pip install -r "$req_file" --quiet
    fi
    echo ""
}

create_env "$ENV_DIR/sarkas.yaml"    "sarkas"    "$PROJECT_DIR/control/sarkas/requirements.txt"
create_env "$ENV_DIR/surrogate.yaml" "surrogate" "$PROJECT_DIR/control/surrogate/requirements.txt"
create_env "$ENV_DIR/ttm.yaml"       "ttm"       ""

# ---- Install Sarkas from patched fork ----
echo "--- Installing Sarkas from patched upstream ---"
SARKAS_SRC="$PROJECT_DIR/control/sarkas/sarkas-upstream"
if [ -d "$SARKAS_SRC" ]; then
    $PM_RUN sarkas pip install -e "$SARKAS_SRC" --quiet
    echo "    Sarkas installed (editable mode)."
else
    echo "    WARNING: sarkas-upstream not found. Run scripts/clone-repos.sh first."
fi
echo ""

# ---- Validate environments ----
echo "--- Validating sarkas environment ---"
$PM_RUN sarkas python -c "import sarkas; print(f'  sarkas: {sarkas.__version__}')" 2>/dev/null || echo "  sarkas: FAILED TO IMPORT"
$PM_RUN sarkas python -c "import pyfftw; print(f'  pyfftw: OK')" 2>/dev/null || echo "  pyfftw: FAILED TO IMPORT"
$PM_RUN sarkas python -c "import numba; print(f'  numba: {numba.__version__}')" 2>/dev/null || echo "  numba: FAILED TO IMPORT"
$PM_RUN sarkas python -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>/dev/null || echo "  numpy: FAILED TO IMPORT"

echo ""
echo "--- Validating surrogate environment ---"
$PM_RUN surrogate python -c "from mystic.samplers import SparsitySampler; print('  mystic: OK (SparsitySampler available)')" 2>/dev/null || echo "  mystic: FAILED TO IMPORT"
$PM_RUN surrogate python -c "from scipy.interpolate import RBFInterpolator; print('  scipy RBF: OK')" 2>/dev/null || echo "  scipy RBF: FAILED TO IMPORT"

echo ""
echo "--- Validating ttm environment ---"
$PM_RUN ttm python -c "import scipy; print(f'  scipy: {scipy.__version__}')" 2>/dev/null || echo "  scipy: FAILED TO IMPORT"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick start:"
echo "  # Sarkas MD control"
echo "  $PM_RUN sarkas python control/sarkas/simulations/dsf-study/scripts/run_case.py <input.yaml>"
echo ""
echo "  # Surrogate learning reproduction"
echo "  $PM_RUN surrogate python control/surrogate/scripts/full_iterative_workflow.py --quick"
echo ""
echo "  # TTM control"
echo "  $PM_RUN ttm python control/ttm/scripts/run_local_model.py"
