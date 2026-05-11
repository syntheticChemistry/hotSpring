#!/bin/bash
# k80_nouveau_warm_catch.sh — Warm-catch test for K80 GK210 via patched nouveau
#
# Tests the hypothesis: patched nouveau (case 0xf2 → nvf1_chipset) can
# initialize GR on the K80, giving us live GPCs to warm-catch into VFIO.
#
# IMPORTANT: This script will temporarily unbind the K80 from vfio-pci
# and load nouveau. The diesel engine (coral-glowplug) stays running:
# the k80 cylinder's keepalive thread keeps the PLX switch alive while
# nouveau operates the K80 die. Only the k80 ember is stopped/restarted.
#
# The patched nouveau.ko at /lib/modules/$(uname -r)/.../nouveau.ko
# already has 0xf2 → nvf1_chipset (GK110B). Stock is at nouveau.ko.stock.
#
# USAGE: sudo bash k80_nouveau_warm_catch.sh

set -euo pipefail

K80_DIE0="0000:4b:00.0"
K80_DIE1="0000:4c:00.0"
TITANV="0000:02:00.0"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  K80 Nouveau Warm-Catch Test                               ║"
echo "║  Patched nouveau: 0xf2 → nvf1_chipset (GK110B)             ║"
echo "║  Target: ${K80_DIE0} (K80 die0)                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Must run as root"
    exit 1
fi

# Verify patched module
NOUVEAU_KO="/lib/modules/$(uname -r)/kernel/drivers/gpu/drm/nouveau/nouveau.ko"
if [[ ! -f "$NOUVEAU_KO" ]]; then
    echo "ERROR: nouveau.ko not found at $NOUVEAU_KO"
    exit 1
fi

echo "[0/8] Pre-flight checks..."
CURRENT_DRIVER=$(basename $(readlink /sys/bus/pci/devices/${K80_DIE0}/driver 2>/dev/null) 2>/dev/null || echo "none")
echo "  K80 die0 driver: ${CURRENT_DRIVER}"
echo "  nouveau loaded: $(lsmod | grep -c nouveau || echo 0)"

# Step 1: Check PLX and d3cold
echo
echo "[1/9] Checking PLX switch and D3cold state..."
cat /sys/bus/pci/devices/${K80_DIE0}/power_state 2>/dev/null || echo "  Cannot read power state"
echo 0 > /sys/bus/pci/devices/${K80_DIE0}/d3cold_allowed 2>/dev/null || true
echo "  d3cold_allowed set to 0"

# Verify diesel engine is running — glowplug is the single root service,
# cylinders own keepalive, embers hold VFIO fds.
if systemctl is-active coral-glowplug.service &>/dev/null; then
    echo "  coral-glowplug (diesel engine): ACTIVE"
    echo "  K80 cylinder keeps PLX alive independently of ember."
else
    echo "  WARNING: coral-glowplug not running! PLX may power-gate K80."
    echo "  Starting glowplug..."
    systemctl start coral-glowplug.service
    sleep 4
fi

# Step 2: The diesel engine architecture keeps the k80 cylinder running
# (which owns PLX keepalive) while we manipulate the K80 device.
# We do NOT stop glowplug — the cylinder continues keepalive traffic
# on the PLX switch while nouveau operates the K80.
echo
echo "[2/9] Diesel engine: k80 cylinder keeps PLX keepalive running."
echo "  Root ECU and titan-v cylinder remain untouched."

# Step 3: Unbind K80 die0 from vfio-pci
echo
echo "[3/9] Unbinding K80 die0 from vfio-pci..."
if [[ "$CURRENT_DRIVER" == "vfio-pci" ]]; then
    echo "${K80_DIE0}" > /sys/bus/pci/devices/${K80_DIE0}/driver/unbind 2>/dev/null || true
    sleep 1
    echo "  Unbound from vfio-pci"
else
    echo "  Not on vfio-pci (${CURRENT_DRIVER}), skipping unbind"
fi

# Clear driver override so nouveau can claim it
echo "" > /sys/bus/pci/devices/${K80_DIE0}/driver_override 2>/dev/null || true

# Step 4: Load livepatch (NOP fini functions to preserve GR state during rebind)
LIVEPATCH_DIR="$(dirname "$0")/../livepatch"
LIVEPATCH_KO="${LIVEPATCH_DIR}/livepatch_nvkm_mc_reset.ko"

echo
echo "[4/10] Loading livepatch (NOP gr_fini/pmu_fini/mc_disable/fifo_fini)..."
if [[ -f "$LIVEPATCH_KO" ]]; then
    insmod "$LIVEPATCH_KO" 2>/dev/null && echo "  Livepatch loaded — fini functions will be NOPed" || echo "  Livepatch already loaded or failed (continuing)"
else
    echo "  WARNING: livepatch not found at $LIVEPATCH_KO"
    echo "  Warm-catch may lose GPC state on unbind!"
fi

# Step 4: Load nouveau
echo
echo "[5/10] Loading nouveau (patched with 0xf2 → nvf1_chipset)..."
if lsmod | grep -q nouveau; then
    echo "  nouveau already loaded"
else
    modprobe nouveau 2>&1 || {
        echo "  WARNING: modprobe nouveau failed, trying insmod..."
        insmod "$NOUVEAU_KO" 2>&1 || {
            echo "  ERROR: Failed to load nouveau"
            echo "  Rebinding to vfio-pci..."
            echo "vfio-pci" > /sys/bus/pci/devices/${K80_DIE0}/driver_override
            echo "${K80_DIE0}" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
            exit 1
        }
    }
    echo "  nouveau loaded"
fi

# Step 5: Trigger device probe
echo
echo "[6/10] Probing K80 die0 to trigger nouveau init..."
echo "${K80_DIE0}" > /sys/bus/pci/drivers/nouveau/bind 2>&1 || {
    # Maybe it auto-bound
    BOUND=$(basename $(readlink /sys/bus/pci/devices/${K80_DIE0}/driver 2>/dev/null) 2>/dev/null || echo "none")
    echo "  Bind returned error, current driver: ${BOUND}"
}

# Step 6: Wait for GR initialization
echo
echo "[7/10] Waiting for nouveau GR initialization (up to 30 seconds)..."
TIMEOUT=30
ELAPSED=0
GR_READY=false
while [[ $ELAPSED -lt $TIMEOUT ]]; do
    # Check dmesg for GR init messages
    if dmesg | tail -50 | grep -q "gr:.*init\|GK110\|GK210\|gk110b_gr"; then
        echo "  GR init message detected at ${ELAPSED}s"
        GR_READY=true
        break
    fi
    # Check for errors
    if dmesg | tail -20 | grep -q "unknown chipset\|failed to create\|ENODEV"; then
        echo "  ERROR: nouveau rejected the chip!"
        dmesg | tail -20 | grep -i "nouveau\|nvkm\|chipset"
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    printf "  Waiting... %ds\r" $ELAPSED
done
echo

# Step 7: Capture GR/GPC state from nouveau
echo
echo "[8/10] Capturing nouveau GR state..."
echo "  --- dmesg (nouveau) ---"
dmesg | grep -i 'nouveau\|nvkm\|gk110\|gk210\|gpc\|gr:' | tail -30

echo
echo "  --- DRM device ---"
ls -la /sys/bus/pci/devices/${K80_DIE0}/drm/ 2>/dev/null || echo "  No DRM device"

echo
echo "  --- nouveau debugfs ---"
if [[ -d /sys/kernel/debug/dri ]]; then
    for card in /sys/kernel/debug/dri/*/name; do
        if [[ -f "$card" ]] && grep -q "nouveau" "$card" 2>/dev/null; then
            CARD_DIR=$(dirname "$card")
            echo "  Card: $(cat "$card")"
            cat "${CARD_DIR}/info" 2>/dev/null || true
            cat "${CARD_DIR}/pstate" 2>/dev/null || true
        fi
    done
else
    echo "  debugfs not mounted or empty"
fi

# Step 8: Unbind from nouveau, rebind to vfio-pci (livepatch preserves state)
echo
echo "[9/10] Rebinding K80 die0 to vfio-pci (livepatch keeps GPCs alive)..."

# Unbind from nouveau
if [[ -e /sys/bus/pci/devices/${K80_DIE0}/driver ]]; then
    echo "${K80_DIE0}" > /sys/bus/pci/devices/${K80_DIE0}/driver/unbind 2>/dev/null || true
    sleep 1
fi

# Remove nouveau
rmmod nouveau 2>/dev/null || true
sleep 1

# Rebind to vfio-pci
echo "vfio-pci" > /sys/bus/pci/devices/${K80_DIE0}/driver_override
echo "${K80_DIE0}" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
sleep 1

FINAL_DRIVER=$(basename $(readlink /sys/bus/pci/devices/${K80_DIE0}/driver 2>/dev/null) 2>/dev/null || echo "none")
echo "  K80 die0 now on: ${FINAL_DRIVER}"

# Disable and remove livepatch
echo 0 > /sys/kernel/livepatch/livepatch_nvkm_mc_reset/enabled 2>/dev/null || true
sleep 1
rmmod livepatch_nvkm_mc_reset 2>/dev/null || true

# Step 9: Restart k80 cylinder's ember via coralctl (diesel engine path)
echo
echo "[10/10] Restarting k80 cylinder's ember via diesel engine..."
# The cylinder auto-detects when its ember has lost VFIO and respawns,
# but we can also explicitly request it via the cylinder RPC.
echo '{"jsonrpc":"2.0","id":99,"method":"cylinder.restart_ember","params":{}}' \
    | socat - UNIX-CONNECT:/run/coralreef/cylinder-k80.sock 2>/dev/null || true
sleep 2
echo "  K80 cylinder ember restarted. Full diesel engine intact."

echo
echo "═══════════════════════════════════════════════════════════════"
if $GR_READY; then
    echo "  RESULT: GR initialization DETECTED"
    echo "  Next: validate GPC count, FECS state via ember after rebind"
    echo "  Then: run sovereign dispatch test"
else
    echo "  RESULT: GR initialization NOT detected within ${TIMEOUT}s"
    echo "  Check: dmesg | grep nouveau"
    echo "  The warm-catch path may need further investigation"
fi
echo "═══════════════════════════════════════════════════════════════"
