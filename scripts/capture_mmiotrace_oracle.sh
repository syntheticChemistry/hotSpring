#!/bin/bash
# capture_mmiotrace_oracle.sh — Capture nouveau boot sequence + annotate with demmio
#
# Captures the full MMIO register write sequence that nouveau performs when
# initializing a GV100 (Titan V). The trace is then annotated with demmio
# (envytools) to provide human-readable register names.
#
# This is Path A of the Oracle-Driven Sovereign Pipeline: extracting the
# "golden recipe" of register writes from a working driver.
#
# Prerequisites:
#   - envytools/demmio installed at /usr/local/bin/demmio
#   - Oracle card (03:00.0) on nouveau or unbound
#   - VFIO target (4a:00.0) already on vfio-pci (untouched)
#
# Usage: sudo ./scripts/capture_mmiotrace_oracle.sh [ORACLE_BDF]

set -euo pipefail

ORACLE="${1:-0000:03:00.0}"
ORACLE_AUD="${ORACLE%.*}.1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATADIR="$(cd "$(dirname "$0")/.." && pwd)/data"
TRACE_RAW="$DATADIR/mmiotrace_raw_${TIMESTAMP}.txt"
TRACE_ANNOTATED="$DATADIR/mmiotrace_demmio_${TIMESTAMP}.txt"
TRACE_WRITES="$DATADIR/mmiotrace_writes_${TIMESTAMP}.txt"
TRACE_PLL="$DATADIR/mmiotrace_pll_init_${TIMESTAMP}.txt"
TRACE_HBM2="$DATADIR/mmiotrace_hbm2_init_${TIMESTAMP}.txt"
KVER=$(uname -r)

mkdir -p "$DATADIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║ mmiotrace Capture + demmio Annotation Pipeline          ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Oracle: $ORACLE → nouveau                     ║"
echo "║ Output: $DATADIR/                              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Unbind oracle from any current driver
echo ">>> Step 1: Preparing oracle card..."
CURR_DRV=$(basename "$(readlink "/sys/bus/pci/devices/$ORACLE/driver" 2>/dev/null)" 2>/dev/null || echo "none")
if [ "$CURR_DRV" = "nouveau" ]; then
    echo "  Unbinding from nouveau..."
    echo "$ORACLE" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
    echo "$ORACLE_AUD" > /sys/bus/pci/drivers/snd_hda_intel/unbind 2>/dev/null || true
    sleep 2
elif [ "$CURR_DRV" = "vfio-pci" ]; then
    echo "  Unbinding from vfio-pci..."
    echo "$ORACLE" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
    echo "$ORACLE_AUD" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
    sleep 1
fi
echo "" > "/sys/bus/pci/devices/$ORACLE/driver_override"
echo "" > "/sys/bus/pci/devices/$ORACLE_AUD/driver_override" 2>/dev/null || true
echo "" > "/sys/bus/pci/devices/$ORACLE/reset_method" 2>/dev/null || true
echo "  Oracle unbound and ready"

# Step 2: Load nouveau dependencies
echo ">>> Step 2: Loading nouveau dependencies..."
modprobe drm 2>/dev/null || true
modprobe drm_kms_helper 2>/dev/null || true
modprobe ttm 2>/dev/null || true
echo "  Dependencies loaded"

# Step 3: Enable mmiotrace
echo ">>> Step 3: Enabling mmiotrace..."
if ! mountpoint -q /sys/kernel/debug; then
    mount -t debugfs debugfs /sys/kernel/debug
fi
echo mmiotrace > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on
echo "  mmiotrace enabled"

# Step 4: Bind nouveau — this is what we're capturing
echo ">>> Step 4: Loading nouveau → capturing MMIO writes..."
BEFORE_NS=$(date +%s%N)
echo "$ORACLE" > /sys/bus/pci/drivers/nouveau/bind 2>&1 || {
    echo "  Direct bind failed, trying modprobe..."
    modprobe nouveau 2>&1 || true
}

# Wait for full init
echo ">>> Step 5: Waiting 10s for full initialization..."
sleep 10
AFTER_NS=$(date +%s%N)
INIT_MS=$(( (AFTER_NS - BEFORE_NS) / 1000000 ))

# Step 6: Stop tracing
echo ">>> Step 6: Stopping mmiotrace..."
echo 0 > /sys/kernel/debug/tracing/tracing_on

# Step 7: Save raw trace
echo ">>> Step 7: Saving raw trace..."
cat /sys/kernel/debug/tracing/trace > "$TRACE_RAW"
echo nop > /sys/kernel/debug/tracing/current_tracer

RAW_LINES=$(wc -l < "$TRACE_RAW")
RAW_SIZE=$(du -h "$TRACE_RAW" | cut -f1)
echo "  Raw: $RAW_LINES lines ($RAW_SIZE) in ${INIT_MS}ms"

# Step 8: Run demmio annotation (GV100 = NV140)
echo ">>> Step 8: Annotating with demmio (GV100/NV140)..."
if command -v demmio >/dev/null 2>&1 || [ -x /usr/local/bin/demmio ]; then
    DEMMIO="${DEMMIO:-/usr/local/bin/demmio}"
    "$DEMMIO" -a GV100 -f "$TRACE_RAW" > "$TRACE_ANNOTATED" 2>/dev/null || {
        # Try without architecture flag
        "$DEMMIO" -f "$TRACE_RAW" > "$TRACE_ANNOTATED" 2>/dev/null || {
            echo "  ⚠ demmio annotation failed — raw trace still available"
            cp "$TRACE_RAW" "$TRACE_ANNOTATED"
        }
    }
    ANN_LINES=$(wc -l < "$TRACE_ANNOTATED")
    echo "  Annotated: $ANN_LINES lines → $TRACE_ANNOTATED"
else
    echo "  ⚠ demmio not found — skipping annotation"
    cp "$TRACE_RAW" "$TRACE_ANNOTATED"
fi

# Step 9: Extract write-only operations (these form the golden recipe)
echo ">>> Step 9: Extracting write operations..."
grep -E "^W " "$TRACE_RAW" > "$TRACE_WRITES" 2>/dev/null || true
WRITE_COUNT=$(wc -l < "$TRACE_WRITES")
echo "  Total writes: $WRITE_COUNT"

# Step 10: Extract PLL-specific writes (PCLOCK domain: 0x130000-0x13FFFF)
echo ">>> Step 10: Filtering PLL/clock init writes..."
grep -iE "0x0*1[3][0-9a-f]{4}" "$TRACE_WRITES" > "$TRACE_PLL" 2>/dev/null || true
PLL_COUNT=$(wc -l < "$TRACE_PLL" 2>/dev/null || echo 0)
echo "  PLL/clock writes: $PLL_COUNT → $TRACE_PLL"

# Step 11: Extract HBM2/FB-specific writes (FBPA: 0x9xxxxx, PFB: 0x10xxxx)
echo ">>> Step 11: Filtering HBM2/FB init writes..."
grep -iE "0x0*(9[0-9a-f]{5}|10[0-9a-f]{4})" "$TRACE_WRITES" > "$TRACE_HBM2" 2>/dev/null || true
HBM2_COUNT=$(wc -l < "$TRACE_HBM2" 2>/dev/null || echo 0)
echo "  HBM2/FB writes: $HBM2_COUNT → $TRACE_HBM2"

# Step 12: Quick BAR0 warm state capture while nouveau is live
echo ">>> Step 12: Capturing warm oracle BAR0 state..."
WARM_BAR0="$DATADIR/oracle_bar0_warm_${TIMESTAMP}.bin"
dd if="/sys/bus/pci/devices/${ORACLE}/resource0" of="$WARM_BAR0" bs=4096 count=4096 status=none 2>/dev/null || {
    echo "  ⚠ Could not capture warm BAR0 (resource0 read failed)"
}
if [ -f "$WARM_BAR0" ]; then
    WARM_SIZE=$(stat -c%s "$WARM_BAR0")
    echo "  Warm BAR0: $WARM_BAR0 ($WARM_SIZE bytes)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║ Capture Complete                                        ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Raw trace:      $TRACE_RAW ($RAW_LINES lines)"
echo "║ Annotated:      $TRACE_ANNOTATED"
echo "║ Write ops:      $TRACE_WRITES ($WRITE_COUNT writes)"
echo "║ PLL writes:     $TRACE_PLL ($PLL_COUNT)"
echo "║ HBM2 writes:    $TRACE_HBM2 ($HBM2_COUNT)"
echo "║ Warm BAR0:      ${WARM_BAR0:-none}"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Init time: ${INIT_MS}ms"
echo "║"
echo "║ Next: Feed into hw-learn distiller:"
echo "║   cargo run -p hw-learn -- distill $TRACE_ANNOTATED"
echo "╚══════════════════════════════════════════════════════════╝"
