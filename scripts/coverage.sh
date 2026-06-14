#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# coverage.sh — Generate llvm-cov coverage report for hotSpring.
#
# Usage:
#   ./scripts/coverage.sh              # summary only
#   ./scripts/coverage.sh html         # generate HTML report
#   ./scripts/coverage.sh --open       # generate + open HTML report
#
# Prerequisites:
#   cargo install cargo-llvm-cov
#   rustup component add llvm-tools-preview
#
# Target: 90%+ line coverage (currently ~45% lib-only, higher with bins)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BARRACUDA_DIR="$SCRIPT_DIR/../barracuda"

cd "$BARRACUDA_DIR"

COV_DIR="$BARRACUDA_DIR/target/llvm-cov"
mkdir -p "$COV_DIR"

case "${1:-summary}" in
    html)
        cargo llvm-cov --lib --html --output-dir "$COV_DIR/html"
        echo "Coverage report: $COV_DIR/html/index.html"
        ;;
    --open)
        cargo llvm-cov --lib --html --open --output-dir "$COV_DIR/html"
        ;;
    json)
        cargo llvm-cov --lib --json --output-path "$COV_DIR/coverage.json"
        echo "Coverage JSON: $COV_DIR/coverage.json"
        ;;
    summary|*)
        echo "═══════════════════════════════════════════"
        echo "  hotSpring Coverage Report (lib tests)"
        echo "═══════════════════════════════════════════"
        echo ""
        cargo llvm-cov --lib --summary-only 2>&1 | grep -E "^TOTAL|^---"
        echo ""
        echo "For full report: $0 html"
        echo "Target: 90%+ line coverage"
        ;;
esac
