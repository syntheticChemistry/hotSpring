#!/usr/bin/env bash
# Run LLVM source-based coverage for hotspring-barracuda (see barracuda/.cargo/llvm-cov-config.toml).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/barracuda"

mkdir -p coverage/html

# Values mirror barracuda/.cargo/llvm-cov-config.toml
FAIL_UNDER_LINES="${COVERAGE_FAIL_UNDER_LINES:-90}"
IGNORE_REGEX="${COVERAGE_IGNORE_FILENAME_REGEX:-/tests/|/src/bin/|(^|/)tests\\.rs\$|.*_tests\\.rs\$|.*-tests\\.rs\$}"

EXTRA=()
if [[ -n "$FAIL_UNDER_LINES" && "$FAIL_UNDER_LINES" != "0" ]]; then
  EXTRA+=(--fail-under-lines "$FAIL_UNDER_LINES")
fi

exec cargo llvm-cov --workspace \
  --html --output-dir coverage/html \
  --json --output-path coverage/coverage.json \
  --ignore-filename-regex "$IGNORE_REGEX" \
  "${EXTRA[@]}" \
  "$@"
