#!/bin/bash
set -u

# Cross-substrate guideStone validation for hotSpring.
#
# Runs the validation artifact across 4 Docker substrates:
#   1. CPU-only (no GPU devices)
#   2. NVIDIA RTX 3090 (--gpus all)
#   3. AMD RX 6950 XT (/dev/dri passthrough)
#   4. Both GPUs (multi-GPU discovery + cross-GPU comparison)
#
# After all substrates complete, performs cross-substrate comparison of
# physics observables (plaquette, energy density) from JSON output to
# prove deterministic results across hardware.
#
# Prerequisites:
#   - Docker with nvidia-container-toolkit
#   - hotSpring validation/ artifact built (scripts/build-guidestone.sh)
#   - /dev/dri available for AMD passthrough
#
# Usage:
#   ./scripts/validate-guidestone-multi.sh [/path/to/validation]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="${1:-$REPO_ROOT/validation}"
ARTIFACT_DIR="$(cd "$ARTIFACT_DIR" && pwd)"

RUN_ID="$(date +%s)"
RESULTS_ROOT="/tmp/guidestone-multi-${RUN_ID}"
mkdir -p "$RESULTS_ROOT"

if [ ! -f "$ARTIFACT_DIR/run" ]; then
    echo "ERROR: $ARTIFACT_DIR/run not found — build artifact first"
    echo "       ./scripts/build-guidestone.sh"
    exit 1
fi
if [ ! -f "$ARTIFACT_DIR/bin/validate-x86_64" ]; then
    echo "ERROR: No x86_64 binary in $ARTIFACT_DIR/bin/"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  guideStone Cross-Substrate Validation                             ║"
echo "║  4 substrates × GPU parity checks = deterministic proof            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo
echo "  Artifact:  $ARTIFACT_DIR"
echo "  Results:   $RESULTS_ROOT"
echo "  Run ID:    $RUN_ID"
echo

PASS_COUNT=0
FAIL_COUNT=0
TOTAL=0

cleanup_all() {
    echo
    echo "  Cleaning up containers..."
    docker ps -a --filter "name=gs-val-" --format '{{.Names}}' | xargs -r docker rm -f 2>/dev/null || true
}
trap cleanup_all EXIT

run_substrate() {
    local NAME="$1"
    local LABEL="$2"
    local NUM="$3"
    shift 3
    local SUBSTRATE_DIR="$RESULTS_ROOT/$NAME"
    mkdir -p "$SUBSTRATE_DIR"
    TOTAL=$((TOTAL + 1))
    local CONTAINER="gs-val-${NAME}-${RUN_ID}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$NUM/4] Substrate: $LABEL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local START_S; START_S=$(date +%s)

    docker run -d \
        --name "$CONTAINER" \
        "$@" \
        --volume "$ARTIFACT_DIR:/opt/validation:ro" \
        --tmpfs /opt/validation/results:rw,size=50m \
        ubuntu:22.04 \
        sleep 7200 >/dev/null 2>&1

    local EXIT_CODE=0
    docker exec "$CONTAINER" /opt/validation/run 2>&1 | tee "$SUBSTRATE_DIR/stdout.txt" || EXIT_CODE=$?

    docker exec "$CONTAINER" cat /opt/validation/results/validate_chuna.json \
        > "$SUBSTRATE_DIR/validate_chuna.json" 2>/dev/null || true

    local END_S; END_S=$(date +%s)
    local ELAPSED=$((END_S - START_S))

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "  >>> PASS ($NAME) — ${ELAPSED}s"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  >>> FAIL ($NAME) — exit $EXIT_CODE, ${ELAPSED}s"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

# Substrate 1: CPU-only (no GPU devices)
run_substrate "cpu-only" "CPU-only Ubuntu 22.04" "1"

# Substrate 2: NVIDIA RTX 3090
run_substrate "nvidia-gpu" "NVIDIA RTX 3090 (--gpus all)" "2" \
    --gpus all

# Substrate 3: AMD RX 6950 XT
run_substrate "amd-gpu" "AMD RX 6950 XT (DRI passthrough)" "3" \
    --device=/dev/dri/renderD128 --device=/dev/dri/card0

# Substrate 4: Both GPUs
run_substrate "both-gpus" "Both GPUs (NVIDIA + AMD)" "4" \
    --gpus all --device=/dev/dri/renderD128 --device=/dev/dri/card0

# ─── Per-Substrate Summary ─────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Per-Substrate Results                                             ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
printf "║  Passed: %d / %d substrates                                        ║\n" "$PASS_COUNT" "$TOTAL"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo

for DIR in "$RESULTS_ROOT"/*/; do
    NAME=$(basename "$DIR")
    JSON="$DIR/validate_chuna.json"
    if [ -f "$JSON" ]; then
        TOTAL_CHECKS=$(python3 -c "import json; d=json.load(open('$JSON')); print(d['summary']['total'])" 2>/dev/null || echo "?")
        PASSED_CHECKS=$(python3 -c "import json; d=json.load(open('$JSON')); print(d['summary']['passed'])" 2>/dev/null || echo "?")
        DURATION=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['summary']['duration_ms']/1000:.1f}s\")" 2>/dev/null || echo "?")
        GPU_COUNT=$(python3 -c "
import json
d=json.load(open('$JSON'))
hp = d.get('hardware_profiles', [])
print(len(hp))
" 2>/dev/null || echo "0")
        echo "  $NAME: $PASSED_CHECKS/$TOTAL_CHECKS checks, $DURATION, GPUs=$GPU_COUNT"
    else
        echo "  $NAME: no JSON output"
    fi
done

# ─── Cross-Substrate Observable Comparison ─────────────────────────
echo
echo "━━━ Cross-Substrate Observable Comparison ━━━"
echo

COMPARISON_PASS=0
COMPARISON_FAIL=0

extract_observable() {
    local JSON="$1"
    local LABEL="$2"
    python3 -c "
import json, sys
d = json.load(open('$JSON'))
for c in d['checks']:
    if c['label'] == '$LABEL':
        print(c['observed'])
        sys.exit(0)
print('NaN')
" 2>/dev/null || echo "NaN"
}

compare_observable() {
    local OBS_NAME="$1"
    local LABEL="$2"
    local TOLERANCE="$3"
    local REF_SUBSTRATE="cpu-only"
    local REF_JSON="$RESULTS_ROOT/$REF_SUBSTRATE/validate_chuna.json"

    if [ ! -f "$REF_JSON" ]; then
        echo "  SKIP $OBS_NAME: no CPU-only reference"
        return
    fi

    local REF_VAL; REF_VAL=$(extract_observable "$REF_JSON" "$LABEL")
    if [ "$REF_VAL" = "NaN" ]; then
        echo "  SKIP $OBS_NAME: label '$LABEL' not found in CPU-only"
        return
    fi

    printf "  %-36s  CPU=%s\n" "$OBS_NAME" "$REF_VAL"

    for DIR in "$RESULTS_ROOT"/*/; do
        local NAME; NAME=$(basename "$DIR")
        [ "$NAME" = "$REF_SUBSTRATE" ] && continue
        local JSON="$DIR/validate_chuna.json"
        [ ! -f "$JSON" ] && continue

        local VAL; VAL=$(extract_observable "$JSON" "$LABEL")
        if [ "$VAL" = "NaN" ]; then
            printf "    %-30s  N/A (label not present)\n" "$NAME"
            continue
        fi

        local DIFF; DIFF=$(python3 -c "
ref = $REF_VAL
val = $VAL
diff = abs(ref - val)
print(f'{diff:.2e}')
" 2>/dev/null || echo "?")
        local OK; OK=$(python3 -c "
ref = $REF_VAL
val = $VAL
tol = $TOLERANCE
print('PASS' if abs(ref - val) < tol else 'FAIL')
" 2>/dev/null || echo "?")

        if [ "$OK" = "PASS" ]; then
            COMPARISON_PASS=$((COMPARISON_PASS + 1))
            printf "    %-30s  val=%s  |diff|=%s  PASS\n" "$NAME" "$VAL" "$DIFF"
        else
            COMPARISON_FAIL=$((COMPARISON_FAIL + 1))
            printf "    %-30s  val=%s  |diff|=%s  FAIL (tol=%s)\n" "$NAME" "$VAL" "$DIFF" "$TOLERANCE"
        fi
    done
}

# CPU-only observable comparisons: these should be bit-identical across
# CPU-only substrates (same binary, same config, same arithmetic)
compare_observable "gradient_flow_energy (8^4 RK3)" "gradient_flow_energy_smoothing" "0"
echo
compare_observable "dielectric_W0_real" "dielectric_W0_real" "1e-14"
echo
compare_observable "kf_mass_conservation_1" "kf_mass_conservation_1" "1e-12"
echo

# GPU-related cross-substrate comparisons use guideStone tolerances
compare_observable "gradient_flow_unitarity" "gradient_flow_unitarity" "1e-10"
echo
compare_observable "dielectric_drude_weak" "dielectric_drude_weak" "1e-12"
echo

echo
echo "━━━ Cross-Substrate Comparison Summary ━━━"
printf "  Comparisons: %d PASS, %d FAIL\n" "$COMPARISON_PASS" "$COMPARISON_FAIL"
echo

echo "  All results in: $RESULTS_ROOT/"
echo

TOTAL_FAIL=$((FAIL_COUNT + COMPARISON_FAIL))
if [ "$TOTAL_FAIL" -gt 0 ]; then
    echo "  RESULT: FAILURES DETECTED ($FAIL_COUNT substrates, $COMPARISON_FAIL comparisons)"
    exit 1
else
    echo "  RESULT: ALL SUBSTRATES AND COMPARISONS PASSED"
    exit 0
fi
