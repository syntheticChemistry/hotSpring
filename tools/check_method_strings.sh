#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Method string drift detector for hotSpring
# Adapted from primalSpring PG-65 (tools/check_method_strings.sh)
#
# Two checks:
#   1. LOCAL: All dotted method strings used in barracuda/src/ Rust source
#      must appear in barracuda/config/capability_registry.toml
#   2. CROSS: All methods declared in hotSpring's local registry must appear
#      in primalSpring's canonical registry (when primalSpring tree available)
#
# Usage: tools/check_method_strings.sh [--cross-only]

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LOCAL_REGISTRY="barracuda/config/capability_registry.toml"
PRIMALSPRING_REGISTRY="../primalSpring/config/capability_registry.toml"
SRC_DIR="barracuda/src"

if [[ ! -f "$LOCAL_REGISTRY" ]]; then
    echo "FAIL: $LOCAL_REGISTRY not found"
    exit 1
fi

CROSS_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --cross-only) CROSS_ONLY=true ;;
    esac
done

ERRORS=0

extract_local_registry_methods() {
    grep '^method\s*=' "$LOCAL_REGISTRY" \
        | grep -oP '"[^"]+"' \
        | tr -d '"' \
        | sort -u
}

extract_primalspring_registry_methods() {
    grep -oP '^\s*"[a-z][a-z0-9_]*\.[a-z0-9_.]+[a-z0-9]+"' "$PRIMALSPRING_REGISTRY" \
        | tr -d '"' \
        | sed 's/^[[:space:]]*//' \
        | sort -u
}

extract_source_method_strings() {
    rg -oN '"([a-z][a-z0-9_]*\.[a-z][a-z0-9_.]*[a-z0-9])"' "$SRC_DIR" \
        --no-filename -r '$1' 2>/dev/null \
        | grep -vE '\.(rs|toml|json|jsonl|md|txt|sh|yaml|yml|lock|wgsl|spv|lime|csv|log|dat)$' \
        | grep -vE '^(serde|derive|allow|cfg|test|bench|doc|feature|target|crate)\.' \
        | grep -vE '_telemetry\.' \
        | sort -u
}

# ── Local check: Rust source vs local registry ──────────────────────
if [[ "$CROSS_ONLY" == "false" ]]; then
    echo "── Local registry check ──"

    LOCAL_REGISTERED=$(extract_local_registry_methods)
    USED=$(extract_source_method_strings)

    LOCAL_ERRORS=0
    LOCAL_UNREGISTERED=()

    while IFS= read -r method; do
        [[ -z "$method" ]] && continue
        if ! echo "$LOCAL_REGISTERED" | grep -qxF "$method"; then
            LOCAL_UNREGISTERED+=("$method")
            ((LOCAL_ERRORS++)) || true
        fi
    done <<< "$USED"

    if [[ $LOCAL_ERRORS -eq 0 ]]; then
        TOTAL_REG=$(echo "$LOCAL_REGISTERED" | wc -l)
        TOTAL_USED=$(echo "$USED" | wc -l)
        echo "  OK: $TOTAL_USED method strings used, all in local registry ($TOTAL_REG registered)"
    else
        echo "  DRIFT: $LOCAL_ERRORS method(s) used in source but not in $LOCAL_REGISTRY:"
        for m in "${LOCAL_UNREGISTERED[@]}"; do
            echo "    $m"
        done
        ((ERRORS += LOCAL_ERRORS)) || true
    fi
    echo ""
fi

# ── Cross check: local registry vs primalSpring canonical ───────────
echo "── Cross-registry check (vs primalSpring canonical) ──"

if [[ ! -f "$PRIMALSPRING_REGISTRY" ]]; then
    echo "  SKIP: $PRIMALSPRING_REGISTRY not found (primalSpring not in sibling path)"
    echo "  To enable: clone primalSpring as a sibling spring in ecoPrimals/springs/"
else
    CANONICAL=$(extract_primalspring_registry_methods)
    LOCAL_METHODS=$(extract_local_registry_methods)

    CROSS_ERRORS=0
    CROSS_MISSING=()

    while IFS= read -r method; do
        [[ -z "$method" ]] && continue
        if ! echo "$CANONICAL" | grep -qxF "$method"; then
            CROSS_MISSING+=("$method")
            ((CROSS_ERRORS++)) || true
        fi
    done <<< "$LOCAL_METHODS"

    if [[ $CROSS_ERRORS -eq 0 ]]; then
        TOTAL_LOCAL=$(echo "$LOCAL_METHODS" | wc -l)
        TOTAL_CANONICAL=$(echo "$CANONICAL" | wc -l)
        echo "  OK: all $TOTAL_LOCAL local methods found in canonical registry ($TOTAL_CANONICAL methods)"
    else
        echo "  DRIFT: $CROSS_ERRORS method(s) in hotSpring registry but not in primalSpring canonical:"
        for m in "${CROSS_MISSING[@]}"; do
            echo "    $m"
        done
        echo ""
        echo "  These methods may need to be added to primalSpring's canonical registry,"
        echo "  or hotSpring is using method names that have been renamed upstream."
        ((ERRORS += CROSS_ERRORS)) || true
    fi
fi

echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo "PASS: All method string checks passed."
    exit 0
else
    echo "FAIL: $ERRORS total drift(s) detected."
    exit 1
fi
