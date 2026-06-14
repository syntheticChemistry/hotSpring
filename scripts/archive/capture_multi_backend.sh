#!/bin/bash
# capture_multi_backend.sh — Multi-backend mmiotrace + state capture for GV100 Titan V
#
# Captures the full MMIO register write sequence for any driver backend binding
# to a GV100. Supports nouveau, nvidia, and nvidia_oracle (renamed module).
#
# The driver is treated as a TOOL: bind, capture, unbind, return to vfio.
# Captured traces become permanent "recipes" for the ACR boot solver.
#
# Usage: sudo ./scripts/capture_multi_backend.sh <DRIVER> [BDF]
#   DRIVER: nouveau | nvidia | nvidia_oracle | nvidia_oracle_*
#   BDF:    PCI address (default: 0000:03:00.0)
#
# Output: data/082/<driver>_<bdf>_<timestamp>/
#
# Prerequisites:
#   - Target GPU on vfio-pci (GlowPlug default)
#   - envytools/demmio optional (for annotation)
#   - For nvidia: nvidia.ko already loaded (RTX 5060 keeps it pinned)
#   - For nvidia_oracle: nvidia_oracle.ko built and loadable

set -euo pipefail

DRIVER="${1:?Usage: $0 <nouveau|nvidia|nvidia_oracle> [BDF]}"
BDF="${2:-0000:03:00.0}"
AUD="${BDF%.*}.1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
DATADIR="$SCRIPTDIR/../data/082/${DRIVER}_${BDF//:/_}_${TIMESTAMP}"

KNOWN_DRIVERS="nouveau nvidia nvidia_oracle"
ORACLE_PREFIX="nvidia_oracle"

if [[ ! " $KNOWN_DRIVERS " =~ " $DRIVER " ]] && [[ "$DRIVER" != ${ORACLE_PREFIX}* ]]; then
    echo "ERROR: Unknown driver '$DRIVER'. Expected: $KNOWN_DRIVERS (or nvidia_oracle_*)"
    exit 1
fi

mkdir -p "$DATADIR"

echo "================================================================"
echo " Multi-Backend mmiotrace Capture (Exp 082)"
echo "================================================================"
echo " Driver:  $DRIVER"
echo " GPU:     $BDF"
echo " Output:  $DATADIR/"
echo "================================================================"
echo ""

CURR_DRV=$(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver" 2>/dev/null)" 2>/dev/null || echo "none")
echo ">>> Pre-capture: GPU currently on '$CURR_DRV'"

# Step 1: Capture cold VFIO baseline BAR0
echo ">>> Step 1: Cold baseline capture..."
if [ "$CURR_DRV" = "vfio-pci" ]; then
    COLD_BAR0="$DATADIR/bar0_cold_vfio.bin"
    dd if="/sys/bus/pci/devices/${BDF}/resource0" of="$COLD_BAR0" bs=4096 count=4096 status=none 2>/dev/null && {
        echo "  Cold BAR0: $(stat -c%s "$COLD_BAR0") bytes"
    } || echo "  Cold BAR0 capture failed (expected on cold VFIO)"
fi

# Step 2: Unbind from current driver
echo ">>> Step 2: Unbinding from $CURR_DRV..."
if [ "$CURR_DRV" != "none" ]; then
    echo "$BDF" > "/sys/bus/pci/drivers/$CURR_DRV/unbind" 2>/dev/null || true
    echo "$AUD" > "/sys/bus/pci/drivers/$CURR_DRV/unbind" 2>/dev/null || true
    sleep 1
fi
echo "" > "/sys/bus/pci/devices/$BDF/driver_override"
echo "" > "/sys/bus/pci/devices/$AUD/driver_override" 2>/dev/null || true
echo "" > "/sys/bus/pci/devices/$BDF/reset_method" 2>/dev/null || true
echo "  Unbound"

# Step 3: Load driver dependencies
echo ">>> Step 3: Loading dependencies for $DRIVER..."
case "$DRIVER" in
    nouveau)
        modprobe drm 2>/dev/null || true
        modprobe drm_kms_helper 2>/dev/null || true
        modprobe ttm 2>/dev/null || true
        ;;
    nvidia)
        # nvidia.ko already loaded by 5060 — just verify
        if ! grep -q "^nvidia " /proc/modules; then
            echo "  ERROR: nvidia module not loaded. Is the RTX 5060 active?"
            exit 1
        fi
        echo "  nvidia module loaded ($(cat /sys/module/nvidia/version))"
        ;;
    ${ORACLE_PREFIX}*)
        # Oracle module needs explicit load
        if ! grep -q "^${DRIVER} " /proc/modules; then
            echo "  Loading $DRIVER module..."
            insmod "/lib/modules/$(uname -r)/extra/${DRIVER}.ko" 2>/dev/null || {
                echo "  ERROR: Could not load ${DRIVER}.ko. Build it first."
                exit 1
            }
        fi
        echo "  $DRIVER module loaded"
        ;;
esac

# Step 4: Enable mmiotrace
echo ">>> Step 4: Enabling mmiotrace..."
if ! mountpoint -q /sys/kernel/debug; then
    mount -t debugfs debugfs /sys/kernel/debug
fi
echo mmiotrace > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on
echo "  mmiotrace active"

# Step 5: Bind target driver — THIS IS THE CAPTURE
echo ">>> Step 5: Binding $DRIVER → capturing MMIO writes..."
BEFORE_NS=$(date +%s%N)

echo "$BDF" > "/sys/bus/pci/drivers/$DRIVER/bind" 2>&1 || {
    echo "  Direct bind failed, trying driver_override..."
    echo "$DRIVER" > "/sys/bus/pci/devices/$BDF/driver_override"
    echo "$BDF" > "/sys/bus/pci/drivers_probe" 2>&1 || {
        echo "  ERROR: Could not bind $DRIVER to $BDF"
        echo 0 > /sys/kernel/debug/tracing/tracing_on
        echo nop > /sys/kernel/debug/tracing/current_tracer
        exit 1
    }
}

# Wait for full initialization
SETTLE=10
case "$DRIVER" in
    nouveau) SETTLE=12 ;;
    nvidia*) SETTLE=8 ;;
esac
echo ">>> Step 6: Waiting ${SETTLE}s for full initialization..."
sleep "$SETTLE"

AFTER_NS=$(date +%s%N)
INIT_MS=$(( (AFTER_NS - BEFORE_NS) / 1000000 ))

# Step 7: Stop tracing
echo ">>> Step 7: Stopping mmiotrace..."
echo 0 > /sys/kernel/debug/tracing/tracing_on

# Step 8: Save raw trace
echo ">>> Step 8: Saving raw trace..."
TRACE_RAW="$DATADIR/mmiotrace_raw.txt"
cat /sys/kernel/debug/tracing/trace > "$TRACE_RAW"
echo nop > /sys/kernel/debug/tracing/current_tracer
RAW_LINES=$(wc -l < "$TRACE_RAW")
RAW_SIZE=$(du -h "$TRACE_RAW" | cut -f1)
echo "  Raw: $RAW_LINES lines ($RAW_SIZE) in ${INIT_MS}ms"

# Step 9: Capture warm BAR0
echo ">>> Step 9: Capturing warm BAR0 state..."
WARM_BAR0="$DATADIR/bar0_warm_${DRIVER}.bin"
dd if="/sys/bus/pci/devices/${BDF}/resource0" of="$WARM_BAR0" bs=4096 count=4096 status=none 2>/dev/null && {
    echo "  Warm BAR0: $(stat -c%s "$WARM_BAR0") bytes"
} || echo "  Warm BAR0 capture failed"

# Step 10: Extract write operations
echo ">>> Step 10: Extracting write operations..."
TRACE_WRITES="$DATADIR/mmiotrace_writes.txt"
grep -E "^W " "$TRACE_RAW" > "$TRACE_WRITES" 2>/dev/null || true
WRITE_COUNT=$(wc -l < "$TRACE_WRITES")
echo "  Total writes: $WRITE_COUNT"

# Step 11: Filter falcon-specific writes (SEC2: 0x87xxx, FECS: 0x409xxx, GPCCS: 0x41Axxx, PMU: 0x10Axxx)
echo ">>> Step 11: Filtering falcon init writes..."
TRACE_FALCON="$DATADIR/mmiotrace_falcon_init.txt"
grep -iE "0x0*(87[0-9a-f]{3}|409[0-9a-f]{3}|41a[0-9a-f]{3}|10a[0-9a-f]{3})" "$TRACE_WRITES" > "$TRACE_FALCON" 2>/dev/null || true
FALCON_COUNT=$(wc -l < "$TRACE_FALCON" 2>/dev/null || echo 0)
echo "  Falcon writes: $FALCON_COUNT → $TRACE_FALCON"

# Step 12: Filter ACR/WPR-specific writes (instance block, DMA, bind)
echo ">>> Step 12: Filtering ACR/DMA writes..."
TRACE_ACR="$DATADIR/mmiotrace_acr_dma.txt"
grep -iE "0x0*(87[01][0-9a-f]{2}|200|624|668)" "$TRACE_WRITES" > "$TRACE_ACR" 2>/dev/null || true
ACR_COUNT=$(wc -l < "$TRACE_ACR" 2>/dev/null || echo 0)
echo "  ACR/DMA writes: $ACR_COUNT → $TRACE_ACR"

# Step 13: demmio annotation (optional)
echo ">>> Step 13: Annotating with demmio..."
TRACE_ANNOTATED="$DATADIR/mmiotrace_demmio.txt"
if command -v demmio >/dev/null 2>&1 || [ -x /usr/local/bin/demmio ]; then
    DEMMIO="${DEMMIO:-/usr/local/bin/demmio}"
    "$DEMMIO" -a GV100 -f "$TRACE_RAW" > "$TRACE_ANNOTATED" 2>/dev/null || {
        cp "$TRACE_RAW" "$TRACE_ANNOTATED"
        echo "  demmio failed — raw trace copied"
    }
    echo "  Annotated: $(wc -l < "$TRACE_ANNOTATED") lines"
else
    cp "$TRACE_RAW" "$TRACE_ANNOTATED"
    echo "  demmio not found — raw trace copied"
fi

# Step 14: Unbind and return to vfio-pci
echo ">>> Step 14: Returning GPU to vfio-pci..."
echo "$BDF" > "/sys/bus/pci/drivers/$DRIVER/unbind" 2>/dev/null || true
sleep 2
echo "vfio-pci" > "/sys/bus/pci/devices/$BDF/driver_override"
echo "vfio-pci" > "/sys/bus/pci/devices/$AUD/driver_override" 2>/dev/null || true
echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>&1 || true
echo "$AUD" > /sys/bus/pci/drivers/vfio-pci/bind 2>&1 || true
sleep 1

# Capture post-driver residual BAR0
RESIDUAL_BAR0="$DATADIR/bar0_residual_post_${DRIVER}.bin"
dd if="/sys/bus/pci/devices/${BDF}/resource0" of="$RESIDUAL_BAR0" bs=4096 count=4096 status=none 2>/dev/null && {
    echo "  Residual BAR0: $(stat -c%s "$RESIDUAL_BAR0") bytes"
} || echo "  Residual BAR0 capture failed"

# Write manifest
cat > "$DATADIR/manifest.json" <<MANIFEST
{
  "experiment": "082_multi_backend_oracle",
  "driver": "$DRIVER",
  "bdf": "$BDF",
  "timestamp": "$TIMESTAMP",
  "init_ms": $INIT_MS,
  "raw_lines": $RAW_LINES,
  "write_count": $WRITE_COUNT,
  "falcon_writes": $FALCON_COUNT,
  "acr_dma_writes": $ACR_COUNT,
  "kernel": "$(uname -r)",
  "nvidia_module_version": "$(cat /sys/module/nvidia/version 2>/dev/null || echo 'n/a')"
}
MANIFEST

echo ""
echo "================================================================"
echo " Capture Complete: $DRIVER on $BDF"
echo "================================================================"
echo " Init time:      ${INIT_MS}ms"
echo " Raw trace:      $TRACE_RAW ($RAW_LINES lines)"
echo " Write ops:      $TRACE_WRITES ($WRITE_COUNT)"
echo " Falcon writes:  $TRACE_FALCON ($FALCON_COUNT)"
echo " ACR/DMA writes: $TRACE_ACR ($ACR_COUNT)"
echo " Warm BAR0:      $WARM_BAR0"
echo " Residual BAR0:  $RESIDUAL_BAR0"
echo " Manifest:       $DATADIR/manifest.json"
echo "================================================================"
echo " GPU returned to vfio-pci. Ready for next backend or experiments."
echo ""
echo " Recommended next steps:"
echo "   # Diff falcon writes between drivers:"
echo "   diff data/082/nouveau_*/mmiotrace_falcon_init.txt \\"
echo "        data/082/nvidia_*/mmiotrace_falcon_init.txt"
echo "   # Feed ACR sequence to boot solver:"
echo "   # (analyze mmiotrace_acr_dma.txt for SEC2 bind_inst sequence)"
echo "================================================================"
