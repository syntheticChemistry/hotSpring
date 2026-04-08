#!/usr/bin/env bash
# Experiment 153: Ember Flood/Resurrection Proof
# Orchestrates the fleet check, validation binary, and log collection.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/experiments/153_logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/exp153_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Experiment 153: Ember Flood/Resurrection Proof             ║"
echo "║  $(date)                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Phase 0: Verify fleet is running ──
echo "━━━ Pre-flight: Fleet status ━━━"
echo ""

if command -v systemctl &>/dev/null; then
    echo "coral-glowplug status:"
    systemctl status coral-glowplug --no-pager 2>&1 | head -10 || true
    echo ""

    echo "coral-ember instances:"
    systemctl list-units 'coral-ember@*' --no-pager 2>&1 | head -15 || true
    echo ""
else
    echo "  systemctl not available — checking sockets directly"
    if [[ -S "/run/coralreef/glowplug.sock" ]]; then
        echo "  glowplug socket: EXISTS"
    else
        echo "  glowplug socket: MISSING"
    fi
fi

# ── Build validation binary ──
echo "━━━ Building validate_ember_resilience ━━━"
echo ""

cd "$ROOT/barracuda"
cargo build --release --bin validate_ember_resilience 2>&1 | tail -5
BINARY="$ROOT/barracuda/target/release/validate_ember_resilience"

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found at $BINARY"
    exit 1
fi

# ── Run validation ──
echo ""
echo "━━━ Running validation binary ━━━"
echo ""
echo "  Log: $LOG_FILE"
echo ""

"$BINARY" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# ── Collect logs ──
echo ""
echo "━━━ Collecting system logs ━━━"
echo ""

JOURNAL_LOG="${LOG_DIR}/exp153_journal_${TIMESTAMP}.log"
if command -v journalctl &>/dev/null; then
    echo "  Collecting journalctl logs (last 5 minutes)..."
    journalctl -u coral-glowplug -u 'coral-ember@*' --since "5 minutes ago" --no-pager \
        > "$JOURNAL_LOG" 2>&1 || true
    echo "  Journal log: $JOURNAL_LOG"
    JOURNAL_LINES=$(wc -l < "$JOURNAL_LOG")
    echo "  Lines: $JOURNAL_LINES"
else
    echo "  journalctl not available — skipping journal collection"
fi

# ── Summary ──
echo ""
echo "━━━ Summary ━━━"
echo ""
echo "  Validation exit code: $EXIT_CODE"
echo "  Log: $LOG_FILE"
if [[ -f "$JOURNAL_LOG" ]]; then
    echo "  Journal: $JOURNAL_LOG"
fi
echo ""

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  RESULT: ALL CHECKS PASSED"
else
    echo "  RESULT: SOME CHECKS FAILED (exit $EXIT_CODE)"
fi

exit "$EXIT_CODE"
