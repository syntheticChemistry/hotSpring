#!/bin/bash
# capture_nouveau_mmiotrace.sh — Capture nouveau's full PFIFO init sequence via mmiotrace
#
# Usage: sudo ./scripts/capture_nouveau_mmiotrace.sh [GPU_BDF]
# Default GPU_BDF: 0000:4b:00.0
#
# Output: scripts/nouveau_mmiotrace_$(date +%Y%m%d_%H%M%S).txt
#
# This captures EVERY MMIO write nouveau makes during GPU initialization,
# which we can then replay from Rust to properly warm the PFIFO scheduler.

set -euo pipefail

GPU="${1:-0000:4b:00.0}"
AUD="${GPU%.*}.1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$(dirname "$0")"
OUTFILE="$OUTDIR/nouveau_mmiotrace_${TIMESTAMP}.txt"
PFIFO_FILTERED="$OUTDIR/nouveau_pfifo_init_${TIMESTAMP}.txt"

echo "=== Nouveau mmiotrace capture ==="
echo "GPU: $GPU"
echo "Output: $OUTFILE"
echo ""

# Step 0: Ensure GPU is unbound from any driver
echo "[0] Unbinding GPU from current driver..."
echo "" > "/sys/bus/pci/devices/$GPU/reset_method" 2>/dev/null || true
echo "$GPU" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
echo "$AUD" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
echo "$GPU" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
echo "" > "/sys/bus/pci/devices/$GPU/driver_override"
sleep 1

# Step 1: Mount debugfs if needed
if ! mountpoint -q /sys/kernel/debug; then
    echo "[1] Mounting debugfs..."
    mount -t debugfs debugfs /sys/kernel/debug
fi

# Step 2: Enable mmiotrace
echo "[2] Enabling mmiotrace..."
echo mmiotrace > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on

# Step 3: Bind nouveau (this is what we're capturing)
echo "[3] Binding nouveau (capturing MMIO writes)..."
echo "$GPU" > /sys/bus/pci/drivers/nouveau/bind 2>&1
BIND_RC=$?
echo "    nouveau bind rc=$BIND_RC"

# Wait for init to complete
echo "[4] Waiting 8s for full initialization..."
sleep 8

# Step 4: Stop tracing
echo "[5] Stopping mmiotrace..."
echo 0 > /sys/kernel/debug/tracing/tracing_on

# Step 5: Save full trace
echo "[6] Saving trace to $OUTFILE..."
cat /sys/kernel/debug/tracing/trace > "$OUTFILE"

# Step 6: Reset tracer to nop
echo nop > /sys/kernel/debug/tracing/current_tracer

# Step 7: Extract PFIFO-relevant writes (0x002000-0x002FFF, 0x040000-0x04FFFF, 0x800000+)
echo "[7] Filtering PFIFO register writes..."
grep -E "^W " "$OUTFILE" | grep -iE " (0x0*2[0-9a-f]{3}|0x0*4[0-9a-f]{4}|0x0*8[0-9a-f]{5}|0x0*200) " > "$PFIFO_FILTERED" 2>/dev/null || true

TOTAL=$(wc -l < "$OUTFILE")
PFIFO_COUNT=$(wc -l < "$PFIFO_FILTERED" 2>/dev/null || echo 0)
echo ""
echo "=== Capture complete ==="
echo "Total MMIO ops: $TOTAL"
echo "PFIFO-related:  $PFIFO_COUNT"
echo "Full trace:     $OUTFILE"
echo "PFIFO filtered: $PFIFO_FILTERED"

# Step 8: Unbind nouveau, rebind vfio-pci
echo ""
echo "[8] Rebinding to vfio-pci..."
echo "$GPU" > /sys/bus/pci/drivers/nouveau/unbind 2>&1 || true
sleep 2
echo "vfio-pci" > "/sys/bus/pci/devices/$GPU/driver_override"
echo "vfio-pci" > "/sys/bus/pci/devices/$AUD/driver_override"
echo "$GPU" > /sys/bus/pci/drivers/vfio-pci/bind 2>&1
echo "$AUD" > /sys/bus/pci/drivers/vfio-pci/bind 2>&1
sleep 1
chmod 666 /dev/vfio/36 2>/dev/null || true

echo ""
echo "GPU on vfio-pci. Ready for testing."
echo "NOTE: GPU is WARM from nouveau init. Run diagnostic matrix now."
