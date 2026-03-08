#!/usr/bin/env bash
# Chuna Papers — Overnight Validation Suite
# SPDX-License-Identifier: AGPL-3.0-only
#
# Runs all Paper 43/44/45 validation systems in sequence.
# Log is tee'd to timestamped file for review.
#
# Usage:
#   ./run_chuna_overnight.sh          (all systems)
#   ./run_chuna_overnight.sh --quick  (convergence + CPU only, skip GPU)

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="overnight_logs"
mkdir -p "$LOG_DIR"

MAIN_LOG="${LOG_DIR}/chuna_overnight_${TIMESTAMP}.log"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Chuna Overnight Validation Suite                          ║"
echo "║  Papers 43 (Gradient Flow) / 44 (Dielectric) / 45 (BGK)   ║"
echo "║  Log: ${MAIN_LOG}                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd barracuda

echo "=== Phase 1: Compile in release mode ==="
cargo build --release --bin validate_chuna_overnight \
      --bin bench_flow_convergence \
      --bin gradient_flow_production \
      --bin validate_dsf_vs_md 2>&1 | tee -a "../${MAIN_LOG}"

echo ""
echo "=== Phase 2: Main overnight binary ==="
cargo run --release --bin validate_chuna_overnight 2>&1 | tee -a "../${MAIN_LOG}"

echo ""
echo "=== Phase 3: Convergence benchmark (all 5 integrators) ==="
cargo run --release --bin bench_flow_convergence 2>&1 | tee -a "../${MAIN_LOG}"

echo ""
echo "=== Phase 4: DSF vs MD validation ==="
cargo run --release --bin validate_dsf_vs_md 2>&1 | tee -a "../${MAIN_LOG}"

echo ""
echo "=== Phase 5: Production gradient flow (16⁴ — long run) ==="
cargo run --release --bin gradient_flow_production 2>&1 | tee -a "../${MAIN_LOG}"

echo ""
echo "=== Done ==="
echo "Full log: ${MAIN_LOG}"
echo "Timestamp: $(date)"
