#!/usr/bin/env bash
# CI coverage gate: runs llvm-cov and fails if line coverage < threshold.
# Designed for non-interactive CI pipelines (no HTML output, JSON only).
#
# Usage:
#   ./scripts/ci-coverage-gate.sh           # default: 90% threshold
#   COVERAGE_FAIL_UNDER_LINES=85 ./scripts/ci-coverage-gate.sh
#
# Exit codes:
#   0 — coverage meets threshold
#   1 — coverage below threshold or build failure
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/barracuda"

FAIL_UNDER_LINES="${COVERAGE_FAIL_UNDER_LINES:-90}"
IGNORE_REGEX="${COVERAGE_IGNORE_FILENAME_REGEX:-/tests/|/src/bin/|(^|/)tests\\.rs\$|.*_tests\\.rs\$|.*-tests\\.rs\$}"

echo "=== hotSpring CI Coverage Gate ==="
echo "  Threshold: ${FAIL_UNDER_LINES}% line coverage"
echo "  Workspace: $(pwd)"
echo ""

mkdir -p coverage

cargo llvm-cov --workspace \
  --json --output-path coverage/coverage.json \
  --ignore-filename-regex "$IGNORE_REGEX" \
  --fail-under-lines "$FAIL_UNDER_LINES" \
  "$@"

echo ""
echo "=== Coverage gate PASSED (>= ${FAIL_UNDER_LINES}%) ==="
