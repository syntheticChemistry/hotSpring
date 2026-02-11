#!/usr/bin/env bash
# Clone Murillo Group repositories into hotSpring/
# Non-interactive: safe for CI and automated runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

echo "=== Cloning Murillo Group Repositories ==="
echo ""

clone_if_missing() {
    local url="$1"
    local dest="$2"

    if [ -d "$dest/.git" ]; then
        echo "--- $dest already cloned. Pulling latest. ---"
        (cd "$dest" && git pull --quiet)
    else
        echo "--- Cloning $url ---"
        git clone --quiet "$url" "$dest"
    fi
    echo ""
}

# Required repos
clone_if_missing \
    "https://github.com/murillo-group/sarkas.git" \
    "$PROJECT_DIR/control/sarkas/sarkas-upstream"

clone_if_missing \
    "https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database.git" \
    "$DATA_DIR/plasma-properties-db/Dense-Plasma-Properties-Database"

clone_if_missing \
    "https://github.com/MurilloGroupMSU/Two-Temperature-Model.git" \
    "$PROJECT_DIR/control/ttm/Two-Temperature-Model"

# Optional: Teaching content (only if --all flag passed)
if [[ "${1:-}" == "--all" ]]; then
echo "--- Optional: Teaching_Content repo ---"
    clone_if_missing \
        "https://github.com/MurilloGroupMSU/Teaching_Content.git" \
        "$DATA_DIR/teaching-content"
fi

echo "=== Clone Complete ==="
echo ""
echo "Repos cloned:"
echo "  Sarkas upstream:        $PROJECT_DIR/control/sarkas/sarkas-upstream/"
echo "  Plasma Properties DB:   $DATA_DIR/plasma-properties-db/Dense-Plasma-Properties-Database/"
echo "  Two-Temperature Model:  $PROJECT_DIR/control/ttm/Two-Temperature-Model/"
echo ""
echo "Next:"
echo "  bash $SCRIPT_DIR/download-data.sh    # Download Zenodo data"
echo "  bash $SCRIPT_DIR/setup-envs.sh       # Create Python environments"
