#!/bin/sh
set -eu

# hotSpring Primal Proof Validation — genomeBin Depot Workflow (v0.9.17)
#
# Runs the full primal-proof chain:
#   1. Pre-flight: primalspring_guidestone (composition base check)
#   2. Domain: hotspring_guidestone (QCD physics + guideStone properties)
#
# Requires a running NUCLEUS (deployed via plasmidBin/nucleus_launcher.sh).
# See primalSpring/wateringHole/PLASMINBIN_DEPOT_PATTERN.md for depot setup.
#
# Auto-sets required env vars when FAMILY_ID is provided:
#   BEARDOG_FAMILY_SEED  — derived from FAMILY_ID if not set
#   SONGBIRD_SECURITY_PROVIDER — defaults to "beardog"
#   NESTGATE_JWT_SECRET  — random Base64 if not set
#
# Usage:
#   ./scripts/validate-primal-proof.sh                      # bare mode (no NUCLEUS)
#   FAMILY_ID=hs-val ./scripts/validate-primal-proof.sh     # against live NUCLEUS
#   FAMILY_ID=hs-val ./scripts/validate-primal-proof.sh --full  # pre-flight + domain

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BARRACUDA="$ROOT/barracuda"
PRIMALSPRING="$ROOT/../primalSpring/ecoPrimal"

FULL=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
    esac
done

FAMILY_ID="${FAMILY_ID:-}"
BEARDOG_FAMILY_SEED="${BEARDOG_FAMILY_SEED:-}"
SONGBIRD_SECURITY_PROVIDER="${SONGBIRD_SECURITY_PROVIDER:-beardog}"
NESTGATE_JWT_SECRET="${NESTGATE_JWT_SECRET:-}"

export SONGBIRD_SECURITY_PROVIDER
if [ -n "$FAMILY_ID" ] && [ -z "$BEARDOG_FAMILY_SEED" ]; then
    BEARDOG_FAMILY_SEED="$(echo "$FAMILY_ID" | sha256sum | cut -c1-64)"
    export BEARDOG_FAMILY_SEED
fi
if [ -n "$FAMILY_ID" ] && [ -z "$NESTGATE_JWT_SECRET" ]; then
    NESTGATE_JWT_SECRET="$(head -c 32 /dev/urandom | base64)"
    export NESTGATE_JWT_SECRET
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  hotSpring Primal Proof Validation (v0.9.17)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "  Family ID:  ${FAMILY_ID:-<none — bare mode>}"
echo "  Full mode:  $( [ "$FULL" = "1" ] && echo "yes (pre-flight + domain)" || echo "no (domain only)" )"
echo

OVERALL_EXIT=0

# ── Step 1: Pre-flight (optional, --full) ─────────────────────────
if [ "$FULL" = "1" ]; then
    echo "━━━ Step 1: Pre-flight — primalspring_guidestone ━━━"
    echo

    if [ -f "$PRIMALSPRING/Cargo.toml" ]; then
        PREFLIGHT_EXIT=0
        cd "$PRIMALSPRING"
        cargo run --release --bin primalspring_guidestone 2>&1 || PREFLIGHT_EXIT=$?
        cd "$ROOT"

        case $PREFLIGHT_EXIT in
            0) echo; echo "  >>> PRE-FLIGHT: ALL PASS" ;;
            1) echo; echo "  >>> PRE-FLIGHT: FAILURE — composition broken"
               OVERALL_EXIT=1 ;;
            2) echo; echo "  >>> PRE-FLIGHT: BARE ONLY — no primals discovered" ;;
            *) echo; echo "  >>> PRE-FLIGHT: UNEXPECTED EXIT $PREFLIGHT_EXIT"
               OVERALL_EXIT=1 ;;
        esac
    else
        echo "  SKIP: primalSpring not found at $PRIMALSPRING"
        echo "  (clone primalSpring as sibling to hotSpring)"
    fi
    echo
fi

# ── Step 2: Domain — hotspring_guidestone ─────────────────────────
echo "━━━ Step 2: Domain — hotspring_guidestone ━━━"
echo

DOMAIN_EXIT=0
DOMAIN_OUTPUT_FILE=$(mktemp)
cd "$BARRACUDA"
cargo run --release --bin hotspring_guidestone 2>&1 | tee "$DOMAIN_OUTPUT_FILE" || DOMAIN_EXIT=$?
cd "$ROOT"

BARE_ONLY=0
if grep -q "bare certification only" "$DOMAIN_OUTPUT_FILE" 2>/dev/null; then
    BARE_ONLY=1
fi
rm -f "$DOMAIN_OUTPUT_FILE"

case $DOMAIN_EXIT in
    0)
        echo
        if [ "$BARE_ONLY" = "1" ]; then
            echo "  >>> DOMAIN: BARE PASS — no NUCLEUS primals discovered"
            if [ -n "$FAMILY_ID" ]; then
                echo "  WARNING: FAMILY_ID is set but no primals found."
                echo "  Check that NUCLEUS is running: nucleus_launcher.sh status"
            fi
        else
            echo "  >>> DOMAIN: ALL PASS — primal proof CERTIFIED"
        fi
        ;;
    1)
        echo
        echo "  >>> DOMAIN: FAILURE — at least one check failed"
        OVERALL_EXIT=1
        ;;
    *)
        echo
        echo "  >>> DOMAIN: UNEXPECTED EXIT $DOMAIN_EXIT"
        OVERALL_EXIT=1
        ;;
esac

# ── Summary ───────────────────────────────────────────────────────
echo
echo "╔══════════════════════════════════════════════════════════════╗"
if [ "$OVERALL_EXIT" -eq 0 ]; then
    if [ "$BARE_ONLY" = "1" ]; then
        echo "║  RESULT: BARE CERTIFICATION ONLY                         ║"
        echo "║  Deploy NUCLEUS to complete the primal proof.             ║"
    else
        echo "║  RESULT: PRIMAL PROOF COMPLETE — ALL PASS                ║"
    fi
else
    echo "║  RESULT: FAILURES DETECTED                                ║"
fi
echo "╚══════════════════════════════════════════════════════════════╝"

exit $OVERALL_EXIT
