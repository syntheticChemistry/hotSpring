#!/usr/bin/env bash
# Exp 084: Hardware validation of B1-B4 bind_inst fixes
# Run post-reboot: sudo ./scripts/exp084_b1b4_test.sh
#
# GlowPlug should auto-start on boot and bind both Titans to vfio-pci.
# This script verifies state, then runs the falcon boot solver on both.

CORAL_DIR="/home/biomegate/Development/ecoPrimals/coralReef"
HOTSPRING_DIR="/home/biomegate/Development/ecoPrimals/hotSpring"
TITAN1="0000:03:00.0"
TITAN2="0000:4a:00.0"
LOG_DIR="$HOTSPRING_DIR/data/084"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo " Exp 084: B1-B4 bind_inst Hardware Validation"
echo " $(date)"
echo "================================================================"

# ── Step 1: Verify GPU state ──
echo ""
echo ">>> Step 1: Checking GPU bindings..."
T1_DRV=$(lspci -ks "$TITAN1" | grep "Kernel driver in use" | awk '{print $NF}' || true)
T2_DRV=$(lspci -ks "$TITAN2" | grep "Kernel driver in use" | awk '{print $NF}' || true)
D_DRV=$(lspci -ks "21:00.0" | grep "Kernel driver in use" | awk '{print $NF}' || true)

echo "  Titan #1 ($TITAN1): ${T1_DRV:-unbound}"
echo "  Titan #2 ($TITAN2): ${T2_DRV:-unbound}"
echo "  RTX 5060 (21:00.0): ${D_DRV:-unbound}"

# ── Step 2: Bind any unbound Titans to vfio-pci ──
echo ""
echo ">>> Step 2: Ensuring Titans are on vfio-pci..."
echo "10de 1d81" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true

for BDF in "$TITAN1" "$TITAN2"; do
    CUR=$(lspci -ks "$BDF" | grep "Kernel driver in use" | awk '{print $NF}' || true)
    if [ "$CUR" = "vfio-pci" ]; then
        echo "  $BDF: vfio-pci OK"
    elif [ -n "$CUR" ]; then
        echo "  $BDF: unbinding from $CUR..."
        echo "$BDF" > "/sys/bus/pci/drivers/$CUR/unbind" 2>/dev/null || true
        sleep 1
        echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
        AUDIO="${BDF%.0}.1"
        echo "$AUDIO" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
        echo "  $BDF: bound to vfio-pci"
    else
        echo "  $BDF: binding to vfio-pci..."
        echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
        AUDIO="${BDF%.0}.1"
        echo "$AUDIO" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
        echo "  $BDF: bound to vfio-pci"
    fi
done

# ── Step 3: Check GlowPlug ──
echo ""
echo ">>> Step 3: GlowPlug status..."
if systemctl is-active --quiet coral-glowplug 2>/dev/null; then
    echo "  GlowPlug: active"
    /usr/local/bin/coralctl status 2>/dev/null || true
else
    echo "  GlowPlug: not running (tests work standalone)"
fi

# ── Step 4: Cargo env ──
export PATH="/home/biomegate/.cargo/bin:$PATH"
export CARGO_HOME="/home/biomegate/.cargo"
export RUSTUP_HOME="/home/biomegate/.rustup"
cd "$CORAL_DIR"

# ── Step 5: Test Titan #2 ──
echo ""
echo "================================================================"
echo " Falcon Boot Solver — Titan #2 ($TITAN2)"
echo "================================================================"
export CORALREEF_VFIO_BDF="$TITAN2"

echo ">>> falcon_boot_solver (all strategies)..."
cargo test --test hw_nv_vfio vfio_falcon_boot_solver \
    --features vfio -p coral-driver \
    -- --nocapture --ignored 2>&1 | tee "$LOG_DIR/titan2_boot_solver.txt"
echo ""
echo ">>> sysmem_acr_boot..."
cargo test --test hw_nv_vfio vfio_sysmem_acr_boot \
    --features vfio -p coral-driver \
    -- --nocapture --ignored 2>&1 | tee "$LOG_DIR/titan2_sysmem_acr.txt"

# ── Step 6: Test Titan #1 ──
echo ""
echo "================================================================"
echo " Falcon Boot Solver — Titan #1 ($TITAN1)"
echo "================================================================"
export CORALREEF_VFIO_BDF="$TITAN1"

T1_NOW=$(lspci -ks "$TITAN1" | grep "Kernel driver in use" | awk '{print $NF}' || true)
if [ "$T1_NOW" = "vfio-pci" ]; then
    echo ">>> falcon_boot_solver (all strategies)..."
    cargo test --test hw_nv_vfio vfio_falcon_boot_solver \
        --features vfio -p coral-driver \
        -- --nocapture --ignored 2>&1 | tee "$LOG_DIR/titan1_boot_solver.txt"
    echo ""
    echo ">>> sysmem_acr_boot..."
    cargo test --test hw_nv_vfio vfio_sysmem_acr_boot \
        --features vfio -p coral-driver \
        -- --nocapture --ignored 2>&1 | tee "$LOG_DIR/titan1_sysmem_acr.txt"
else
    echo ">>> Titan #1 not on vfio-pci (${T1_NOW:-unbound}) — skipping"
fi

# ── Step 7: Results ──
echo ""
echo "================================================================"
echo " RESULTS SUMMARY"
echo "================================================================"
for f in "$LOG_DIR"/titan*.txt; do
    [ -f "$f" ] || continue
    NAME=$(basename "$f" .txt)
    echo ""
    echo "--- $NAME ---"

    BIND5=$(grep -iE "stat.*=.*5|bind.*complete|bind_stat.*5" "$f" 2>/dev/null | head -3 || true)
    if [ -n "$BIND5" ]; then
        echo "  *** bind_stat REACHED 5! ***"
        echo "$BIND5" | while read -r line; do echo "    $line"; done
    else
        echo "  bind_stat did NOT reach 5"
    fi

    BIND_WRITES=$(grep -iE "bind_inst.*wrote|bind_inst:.*wrote" "$f" 2>/dev/null | head -3 || true)
    if [ -n "$BIND_WRITES" ]; then
        echo "  Bind writes:"
        echo "$BIND_WRITES" | while read -r line; do echo "    $line"; done
    fi

    RESULT=$(grep -E "test result:|SUCCEEDED|FAILED" "$f" 2>/dev/null | tail -1 || true)
    echo "  Outcome: ${RESULT:-check log}"
done

echo ""
echo ">>> Full logs: $LOG_DIR/"
echo ">>> Done at $(date)"
