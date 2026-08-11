#!/bin/sh
# validate.sh — BLAKE3 integrity + DAG consistency + Ed25519 signature verification
# Part of the hotspring-qcd-sun pseudoSpore bundle
# Requires: b3sum (https://github.com/BLAKE3-team/BLAKE3)
# Optional: minisign or openssl (for signature verification)
set -e

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="${BUNDLE_DIR}/MANIFEST.blake3"
DAG_FILE="${BUNDLE_DIR}/DAG.blake3"
SIGNATURE="${BUNDLE_DIR}/SIGNATURE.ed25519"
PUBKEY="${BUNDLE_DIR}/pubkey.ed25519"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { printf "${GREEN}  PASS${NC}: %s\n" "$1"; }
fail() { printf "${RED}  FAIL${NC}: %s\n" "$1"; }
warn() { printf "${YELLOW}  WARN${NC}: %s\n" "$1"; }

echo "============================================================"
echo "  hotspring-qcd-sun pseudoSpore validation"
echo "  Bundle: ${BUNDLE_DIR}"
echo "  Date:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo ""

ERRORS=0
CHECKED=0
SKIPPED=0

# --- Step 1: Check tools ---
echo "[0/3] Checking prerequisites..."
if ! command -v b3sum >/dev/null 2>&1; then
    echo ""
    fail "b3sum not found. Install via: cargo install b3sum"
    echo "     Or download from: https://github.com/BLAKE3-team/BLAKE3/releases"
    exit 2
fi
pass "b3sum $(b3sum --version 2>/dev/null | head -1)"

if [ ! -f "$MANIFEST" ]; then
    fail "MANIFEST.blake3 not found at ${MANIFEST}"
    exit 2
fi
pass "MANIFEST.blake3 found ($(wc -l < "$MANIFEST") entries)"
echo ""

# --- Step 2: BLAKE3 integrity ---
echo "[1/3] Verifying BLAKE3 content hashes..."
while IFS= read -r line; do
    # Skip comments and blank lines
    case "$line" in \#*|"") continue ;; esac

    # Format: <hash>  <relative_path>
    expected_hash=$(echo "$line" | awk '{print $1}')
    filepath=$(echo "$line" | awk '{$1=""; print substr($0,2)}')

    if [ -z "$expected_hash" ] || [ -z "$filepath" ]; then
        continue
    fi

    full_path="${BUNDLE_DIR}/${filepath}"

    if [ ! -f "$full_path" ]; then
        fail "${filepath} (MISSING)"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    actual_hash=$(b3sum --no-names "$full_path")

    if [ "$actual_hash" = "$expected_hash" ]; then
        CHECKED=$((CHECKED + 1))
    else
        fail "${filepath}"
        echo "       expected: ${expected_hash}"
        echo "       actual:   ${actual_hash}"
        ERRORS=$((ERRORS + 1))
    fi
done < "$MANIFEST"

if [ $ERRORS -eq 0 ]; then
    pass "All ${CHECKED} files verified"
else
    fail "${ERRORS} files failed, ${CHECKED} passed"
fi
echo ""

# --- Step 3: DAG consistency ---
echo "[2/3] Checking provenance DAG consistency..."
if [ ! -f "$DAG_FILE" ]; then
    warn "DAG.blake3 not found — skipping DAG verification"
    SKIPPED=$((SKIPPED + 1))
else
    # Collect all hashes from manifest
    manifest_hashes=$(awk '{print $1}' "$MANIFEST" | sort -u)
    dag_errors=0
    dag_entries=0

    while IFS= read -r line; do
        [ -z "$line" ] && continue
        # Skip comments
        case "$line" in \#*) continue ;; esac

        dag_entries=$((dag_entries + 1))
        # Each line: child_hash parent_hash_1 [parent_hash_2 ...]
        for hash in $line; do
            if ! echo "$manifest_hashes" | grep -q "^${hash}$"; then
                # Hash references something outside bundle — check if it's a known external ref
                case "$hash" in
                    "ROOT"|"GENESIS"|"EXTERNAL"*) ;;  # Sentinel values are OK
                    *)
                        fail "DAG references unknown hash: ${hash}"
                        dag_errors=$((dag_errors + 1))
                        ;;
                esac
            fi
        done
    done < "$DAG_FILE"

    if [ $dag_errors -eq 0 ]; then
        pass "DAG consistent (${dag_entries} entries, all references resolve)"
    else
        fail "DAG has ${dag_errors} unresolved references"
        ERRORS=$((ERRORS + dag_errors))
    fi
fi
echo ""

# --- Step 4: Ed25519 signature ---
echo "[3/3] Verifying Ed25519 signature..."
if [ ! -f "$SIGNATURE" ]; then
    warn "SIGNATURE.ed25519 not found — signature verification skipped"
    warn "Bundle integrity verified by BLAKE3 only (no authorship proof)"
    SKIPPED=$((SKIPPED + 1))
elif [ ! -f "$PUBKEY" ]; then
    warn "pubkey.ed25519 not found — cannot verify signature"
    SKIPPED=$((SKIPPED + 1))
else
    sig_verified=0
    if command -v minisign >/dev/null 2>&1; then
        if minisign -Vm "$MANIFEST" -p "$PUBKEY" -x "$SIGNATURE" 2>/dev/null; then
            sig_verified=1
        fi
    elif command -v signify-openbsd >/dev/null 2>&1; then
        if signify-openbsd -Vq -p "$PUBKEY" -x "$SIGNATURE" -m "$MANIFEST" 2>/dev/null; then
            sig_verified=1
        fi
    elif command -v openssl >/dev/null 2>&1; then
        if openssl pkeyutl -verify -pubin -inkey "$PUBKEY" \
            -sigfile "$SIGNATURE" -rawin -in "$MANIFEST" 2>/dev/null; then
            sig_verified=1
        fi
    fi

    if [ $sig_verified -eq 1 ]; then
        pass "Ed25519 signature VALID (authorship confirmed)"
    else
        if ! command -v minisign >/dev/null 2>&1 && \
           ! command -v signify-openbsd >/dev/null 2>&1 && \
           ! command -v openssl >/dev/null 2>&1; then
            warn "No Ed25519 verifier found (install minisign, signify, or openssl)"
            SKIPPED=$((SKIPPED + 1))
        else
            fail "Ed25519 signature INVALID or verification failed"
            ERRORS=$((ERRORS + 1))
        fi
    fi
fi

# --- Summary ---
echo ""
echo "============================================================"
if [ $ERRORS -eq 0 ]; then
    printf "  ${GREEN}VALIDATION PASSED${NC}\n"
    echo "  Files checked: ${CHECKED}"
    [ $SKIPPED -gt 0 ] && echo "  Checks skipped: ${SKIPPED}"
    echo ""
    echo "  All content hashes match. Data integrity confirmed."
    echo "  You may reproduce measurements with:"
    echo ""
    echo "    cd ${BUNDLE_DIR}/code"
    echo "    cargo run --release --bin arxiv_measure_battery -- \\"
    echo "        --config ${BUNDLE_DIR}/configs/"
    echo ""
    echo "  Or load configs into MILC:"
    echo "    ls ${BUNDLE_DIR}/configs/su3/*.milc"
    echo ""
else
    printf "  ${RED}VALIDATION FAILED${NC} (${ERRORS} errors)\n"
    echo "  The bundle may be corrupted or tampered with."
    echo "  Contact: ORCID 0009-0004-2141-0321"
fi
echo "============================================================"

exit $ERRORS
