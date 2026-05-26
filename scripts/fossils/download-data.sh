#!/usr/bin/env bash
# Download external data for hotSpring control experiments.
# Non-interactive. Idempotent (skips existing data).
# Run from hotSpring/ directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

echo "=== hotSpring Data Download ==="
echo ""

# ============================================================
# 1. Zenodo - Surrogate Learning datasets (~6 GB)
# ============================================================
ZENODO_DIR="$DATA_DIR/zenodo-surrogate"
mkdir -p "$ZENODO_DIR"

echo "--- Surrogate Learning: Zenodo datasets ---"
echo "  DOI: 10.5281/zenodo.10908462"
echo "  Target: $ZENODO_DIR/"
echo ""

if [ -n "$(ls -A "$ZENODO_DIR" 2>/dev/null)" ]; then
    count=$(ls "$ZENODO_DIR" | wc -l)
    echo "  $count files already present in $ZENODO_DIR/. Skipping."
else
    echo "  Downloading from Zenodo (~6 GB)..."
    ZENODO_RECORD="10908462"
    ZENODO_API="https://zenodo.org/api/records/${ZENODO_RECORD}"

    if command -v curl &> /dev/null; then
        echo "  Fetching file list from Zenodo API..."
        FILE_URLS=$(curl -sL "$ZENODO_API" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for f in data.get('files', []):
    print(f['links']['self'])
" 2>/dev/null || true)

        if [ -n "$FILE_URLS" ]; then
            for url in $FILE_URLS; do
                filename=$(basename "$url")
                echo "  Downloading: $filename"
                curl -L --progress-bar -o "$ZENODO_DIR/$filename" "$url"
            done
            echo "  Done."

            # Unzip if there are zip files
            for zipfile in "$ZENODO_DIR"/*.zip; do
                if [ -f "$zipfile" ]; then
                    echo "  Extracting: $(basename "$zipfile")"
                    unzip -qo "$zipfile" -d "$ZENODO_DIR/"
                fi
            done
        else
            echo "  Could not parse Zenodo API. Download manually:"
            echo "    https://zenodo.org/records/${ZENODO_RECORD}"
            echo "    Save files to: $ZENODO_DIR/"
        fi
    else
        echo "  curl not found. Download manually:"
        echo "    https://zenodo.org/records/${ZENODO_RECORD}"
        echo "    Save files to: $ZENODO_DIR/"
    fi
fi
echo ""

# ============================================================
# 2. Code Ocean - Surrogate Learning capsule
# ============================================================
# NOTE: Code Ocean is a GATED platform. Sign-up is denied on some
# operating systems. We have fully reconstructed the paper's methodology
# without needing this capsule. See control/surrogate/REPRODUCE.md
# and whitePaper/barraCUDA/sections/04a_SURROGATE_OPEN_SCIENCE.md
# for details.
#
# The nuclear EOS objective function in this capsule wraps restricted
# LANL nuclear simulation data. We replaced it with a physics EOS
# derived from our own validated Sarkas MD simulations.
echo "--- Code Ocean capsule: SKIPPED ---"
echo "  Code Ocean denies sign-up on some operating systems."
echo "  The paper's methodology has been fully reconstructed without it."
echo "  See: control/surrogate/REPRODUCE.md"
echo ""

# ============================================================
# 3. Summary
# ============================================================
echo "=== Download Summary ==="
echo ""
echo "Zenodo data:      $ZENODO_DIR/"
if [ -n "$(ls -A "$ZENODO_DIR" 2>/dev/null)" ]; then
    du -sh "$ZENODO_DIR" | awk '{print "  Status: " $1 " present"}'
else
    echo "  Status: EMPTY — download needed"
fi
echo ""
echo "Dense Plasma DB:  $DATA_DIR/plasma-properties-db/"
if [ -d "$DATA_DIR/plasma-properties-db/Dense-Plasma-Properties-Database/.git" ]; then
    echo "  Status: Cloned (via clone-repos.sh)"
else
    echo "  Status: NOT CLONED — run scripts/clone-repos.sh first"
fi
echo ""
echo "Next: bash scripts/setup-envs.sh"
