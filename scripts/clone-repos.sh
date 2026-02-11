#!/usr/bin/env bash
# Clone and patch upstream repositories for hotSpring control experiments.
# Non-interactive. Idempotent (skips existing, applies patches if needed).
# Run from anywhere — resolves paths relative to this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

echo "╔══════════════════════════════════════════════╗"
echo "║     hotSpring — Clone Upstream Repos         ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ============================================================
# Helper: clone, pin to tag/commit, and apply patches
# ============================================================
clone_and_patch() {
    local url="$1"
    local dest="$2"
    local pin="${3:-}"        # tag or commit to checkout (optional)
    local patch_dir="${4:-}"  # directory containing .patch files (optional)

    if [ -d "$dest/.git" ]; then
        echo "  ✓ $dest already cloned."
        # Reset to ensure patches can be re-applied cleanly
        if [ -n "$pin" ]; then
            (cd "$dest" && git checkout -q "$pin" 2>/dev/null && git checkout -q . 2>/dev/null) || true
        fi
    else
        echo "  ↓ Cloning $url ..."
        git clone --quiet "$url" "$dest"
        if [ -n "$pin" ]; then
            echo "    Pinning to: $pin"
            (cd "$dest" && git checkout -q "$pin" 2>/dev/null) || true
        fi
    fi

    # Apply patches if directory exists and contains .patch files
    if [ -n "$patch_dir" ] && [ -d "$patch_dir" ]; then
        local patch_count
        patch_count=$(find "$patch_dir" -name "*.patch" 2>/dev/null | wc -l)
        if [ "$patch_count" -gt 0 ]; then
            echo "    Applying $patch_count patch(es) from $(basename "$patch_dir")/"
            for patch_file in "$patch_dir"/*.patch; do
                local patch_name
                patch_name=$(basename "$patch_file")
                if (cd "$dest" && git apply --check "$patch_file" 2>/dev/null); then
                    (cd "$dest" && git apply "$patch_file")
                    echo "      ✓ $patch_name"
                else
                    echo "      · $patch_name (already applied or N/A)"
                fi
            done
        fi
    fi
    echo ""
}

# ============================================================
# 1. Sarkas — Molecular Dynamics (pinned to v1.0.0)
# ============================================================
echo "── Sarkas (Murillo Group MD) ──────────────────"
echo "  v1.0.0 pinned — v1.1.0 has dump corruption bug"
clone_and_patch \
    "https://github.com/murillo-group/sarkas.git" \
    "$PROJECT_DIR/control/sarkas/sarkas-upstream" \
    "v1.0.0" \
    "$PROJECT_DIR/control/sarkas/patches"
# Patches fix:
#   - np.int → int (NumPy 2.x compat)
#   - .mean(level=) → .T.groupby(level=).mean().T (pandas 2.x compat)
#   - @jit → @jit(forceobj=True) for pyfftw PPPM (Numba 0.60 compat)

# ============================================================
# 2. Dense Plasma Properties Database
# ============================================================
echo "── Dense Plasma Properties Database ───────────"
clone_and_patch \
    "https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database.git" \
    "$DATA_DIR/plasma-properties-db/Dense-Plasma-Properties-Database"

# ============================================================
# 3. Two-Temperature Model
# ============================================================
echo "── Two-Temperature Model (UCLA-MSU) ──────────"
clone_and_patch \
    "https://github.com/MurilloGroupMSU/Two-Temperature-Model.git" \
    "$PROJECT_DIR/control/ttm/Two-Temperature-Model" \
    "" \
    "$PROJECT_DIR/control/ttm/patches"
# Patches fix:
#   - np.math.factorial → math.factorial (NumPy 2.x removed np.math)

# ============================================================
# 4. Optional: Teaching Content
# ============================================================
if [[ "${1:-}" == "--all" ]]; then
    echo "── Teaching Content (optional) ────────────────"
    clone_and_patch \
        "https://github.com/MurilloGroupMSU/Teaching_Content.git" \
        "$DATA_DIR/teaching-content"
fi

# ============================================================
# Summary
# ============================================================
echo "╔══════════════════════════════════════════════╗"
echo "║              Clone Summary                   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

check_repo() {
    local label="$1"
    local path="$2"
    if [ -d "$path/.git" ]; then
        local ver
        ver=$(cd "$path" && git describe --tags 2>/dev/null || git -C "$path" rev-parse --short HEAD 2>/dev/null || echo "unknown")
        echo "  ✓ $label ($ver)"
    else
        echo "  ✗ $label — NOT CLONED"
    fi
}

check_repo "Sarkas upstream       " "$PROJECT_DIR/control/sarkas/sarkas-upstream"
check_repo "Dense Plasma DB       " "$DATA_DIR/plasma-properties-db/Dense-Plasma-Properties-Database"
check_repo "Two-Temperature Model " "$PROJECT_DIR/control/ttm/Two-Temperature-Model"

echo ""
echo "Next steps:"
echo "  bash scripts/download-data.sh    # Download Zenodo archive (~6 GB)"
echo "  bash scripts/setup-envs.sh       # Create Python environments"
