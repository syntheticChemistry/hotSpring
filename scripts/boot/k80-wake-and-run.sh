#!/bin/bash
# k80-wake-and-run.sh — Post-power-cycle K80 sovereign boot via coral stack.
#
# ALL GPU interaction routes through coral-ember + coral-glowplug.
# NEVER touch /sys/bus/pci/drivers/ directly while ember holds VFIO fds.
# Direct sysfs driver manipulation causes VFIO group kernel deadlocks.
#
# WHY warm-fecs (nouveau round-trip) is PERMANENTLY REMOVED:
#   Binding nouveau to the K80 causes it to assert PCIe L1/L2 power state
#   (nouveau's PM aggressively powers down GPUs with no display connected).
#   K80 deasserts CLKREQ# → PLX PEX 8747 removes PCIe clock from the slot
#   → PLX upstream port config space returns 0xFFFF → kernel logs
#   "bridge configuration invalid" → entire PLX subtree dies.
#   This cascade is hardware-level and cannot be prevented by ASPM disable
#   or power/control=on because nouveau overrides both during its PM init.
#   The PLX D3cold cascade requires a full system power cycle to recover.
#
# Sequence:
#   1. Verify PLX PEX 8747 + K80 config space accessible
#   2. Enable MSE + Bus Master on K80s (config space writes, safe)
#   3. Pin PLX subtree to D0 (power/control sysfs, safe)
#   4. Ensure coral-ember is running (holds VFIO fds + PCIe switch keepalive)
#   5. Ensure coral-glowplug is running (device lifecycle broker)
#   6. Verify ember's PCIe switch keepalive (replaces plx-keepalive.service)
#   7. coralctl sovereign-boot — full sovereign pipeline via coral stack
#      On cold K80: kepler_falcon_boot handles GR init from scratch
#      GDDR5 not trained → memory_training=Failed (non-fatal), GR_READY still set
#      compute_ready=true requires both GR_READY + VRAM; cold boot gets GR_READY

set -euo pipefail

GLOWPLUG_CONF="${CORALREEF_GLOWPLUG_CONFIG:-/etc/coralreef/glowplug.toml}"
BARRACUDA="${HOTSPRING_BARRACUDA:-/home/biomegate/Development/ecoPrimals/springs/hotSpring/barracuda}"

GLOWPLUG_SOCK=/run/coralreef/glowplug.sock
EMBER_SOCK=/run/coralreef/ember.sock

die() { echo "FATAL: $*" >&2; exit 1; }

# ── Extract BDFs from glowplug.toml (no hardcoded addresses) ─────────────────
[ -f "$GLOWPLUG_CONF" ] || die "Config not found: $GLOWPLUG_CONF"
eval "$(python3 -c "
import tomllib, sys, pathlib

cfg = tomllib.loads(pathlib.Path('$GLOWPLUG_CONF').read_text())

switches = cfg.get('pcie_switch', [])
if switches:
    sw = switches[0]
    print(f'PLX={sw[\"bdf\"]}')
    dps = sw.get('downstream_ports', [])
    for i, dp in enumerate(dps, 1):
        print(f'PLX_DP{i}={dp}')
    print(f'PLX_DP_COUNT={len(dps)}')
else:
    print('PLX=')
    print('PLX_DP_COUNT=0')

# K80s = devices with upstream_switch set (order by BDF for determinism)
k80s = sorted(
    [d for d in cfg.get('device', []) if d.get('upstream_switch')],
    key=lambda d: d['bdf']
)
for i, d in enumerate(k80s):
    print(f'K80_DIE{i}={d[\"bdf\"]}')
print(f'K80_COUNT={len(k80s)}')
")" || die "Failed to parse $GLOWPLUG_CONF"

[ -n "$PLX" ] || die "No [[pcie_switch]] found in $GLOWPLUG_CONF"
[ "$K80_COUNT" -gt 0 ] || die "No devices with upstream_switch found in $GLOWPLUG_CONF"

echo "========================================"
echo " K80 Sovereign Wake + Run"
echo "========================================"

# ── Step 1: Verify PLX ────────────────────────────────────────────────────────
echo ""
echo "[1] Verifying PLX PEX 8747..."
PLX_ID=$(setpci -s "$PLX" 0x00.L 2>/dev/null || true)
[ "$PLX_ID" = "ffffffff" ] || [ -z "$PLX_ID" ] && \
    die "PLX returns 0xffffffff — hardware power cycle required."
echo "    PLX VID:DID = 0x${PLX_ID}  ✓"

# ── Step 2: Verify K80 config space ──────────────────────────────────────────
echo ""
echo "[2] Verifying K80 config space..."
for i in $(seq 0 $((K80_COUNT - 1))); do
    varname="K80_DIE${i}"
    bdf="${!varname}"
    kid=$(setpci -s "$bdf" 0x00.L 2>/dev/null || true)
    echo "    K80 die${i} ($bdf) = 0x${kid}"
    [ "$kid" = "ffffffff" ] && die "K80 die${i} ($bdf) config unreadable — hardware power cycle required."
done

# ── Step 3: MSE + Bus Master ──────────────────────────────────────────────────
echo ""
echo "[3] Enabling MSE + Bus Master on K80s..."
for i in $(seq 0 $((K80_COUNT - 1))); do
    varname="K80_DIE${i}"
    bdf="${!varname}"
    setpci -s "$bdf" COMMAND=0x06 && echo "    K80 die${i} ($bdf) COMMAND=0x06  ✓" || true
done

# ── Step 4: Pin PLX subtree to D0 ─────────────────────────────────────────────
# Only write power/control (safe — does not touch driver binding).
# d3cold_allowed requires kernel config not present on this build; skip gracefully.
echo ""
echo "[4] Pinning PLX subtree to D0 (runtime PM)..."
ALL_PCIE_DEVS="$PLX"
for i in $(seq 1 "$PLX_DP_COUNT"); do
    varname="PLX_DP${i}"
    ALL_PCIE_DEVS="$ALL_PCIE_DEVS ${!varname}"
done
for i in $(seq 0 $((K80_COUNT - 1))); do
    varname="K80_DIE${i}"
    ALL_PCIE_DEVS="$ALL_PCIE_DEVS ${!varname}"
done
for dev in $ALL_PCIE_DEVS; do
    echo "on" > /sys/bus/pci/devices/${dev}/power/control 2>/dev/null || true
done
echo "    power/control=on for PLX subtree + K80 endpoints  ✓"

# ── Step 5: Ensure coral-ember is running ─────────────────────────────────────
echo ""
echo "[5] Checking coral-ember (immortal VFIO fd holder)..."
if ! systemctl is-active --quiet coral-ember 2>/dev/null; then
    echo "    Starting coral-ember..."
    systemctl start coral-ember || die "coral-ember failed to start"
fi
echo "    Waiting for ember socket..."
for i in $(seq 1 60); do
    [ -S "$EMBER_SOCK" ] && break
    sleep 0.5
done
[ -S "$EMBER_SOCK" ] || die "ember socket not ready after 30s: $EMBER_SOCK"
echo "    coral-ember active  ✓  ($EMBER_SOCK)"

# ── Step 5b: Fix DRM isolation rules (ember may overwrite with stale BDFs) ───
# coral-ember auto-generates 61-coralreef-drm-ignore.rules at startup.
# If it generates stale BDFs, warm-fecs blocks with "DRM isolation incomplete".
# Generate from glowplug.toml: all non-"shared" devices need DRM isolation.
DRM_RULES=/etc/udev/rules.d/61-coralreef-drm-ignore.rules
COMPUTE_BDFS=$(python3 -c "
import tomllib, pathlib
cfg = tomllib.loads(pathlib.Path('$GLOWPLUG_CONF').read_text())
for d in cfg.get('device', []):
    if d.get('role') != 'shared':
        print(d['bdf'])
" 2>/dev/null || true)

RULES_STALE=false
for cbdf in $COMPUTE_BDFS; do
    if ! grep -q "$cbdf" "$DRM_RULES" 2>/dev/null; then
        RULES_STALE=true
        break
    fi
done

if $RULES_STALE; then
    echo "    DRM isolation rules stale — regenerating from config..."
    {
        echo "# coralReef: DRM isolation for compute GPUs (auto-generated from $GLOWPLUG_CONF)"
        echo "# Prevents logind seat/uaccess tags — blocks driver swaps if active."
        echo ""
        for cbdf in $COMPUTE_BDFS; do
            echo "SUBSYSTEM==\"drm\", KERNELS==\"${cbdf}\", ENV{ID_SEAT}=\"\", ENV{ID_FOR_SEAT}=\"\", TAG-=\"seat\", TAG-=\"master-of-seat\", TAG-=\"uaccess\""
        done
    } > "$DRM_RULES"
    udevadm control --reload-rules
    udevadm trigger --subsystem-match=drm 2>/dev/null || true
    echo "    DRM isolation rules patched ✓"
else
    echo "    DRM isolation rules OK ✓"
fi

# ── Step 6: Ensure coral-glowplug is running ──────────────────────────────────
echo ""
echo "[6] Checking coral-glowplug (lifecycle broker)..."
if ! systemctl is-active --quiet coral-glowplug 2>/dev/null; then
    echo "    Starting coral-glowplug..."
    systemctl start coral-glowplug || die "coral-glowplug failed to start"
fi
echo "    Waiting for glowplug socket..."
for i in $(seq 1 60); do
    [ -S "$GLOWPLUG_SOCK" ] && break
    sleep 0.5
done
[ -S "$GLOWPLUG_SOCK" ] || die "glowplug socket not ready after 30s: $GLOWPLUG_SOCK"
echo "    coral-glowplug active  ✓  ($GLOWPLUG_SOCK)"

# ── Step 7: Verify ember PCIe switch keepalive ───────────────────────────────
# Ember's built-in keepalive thread (pcie_keepalive.rs) replaced plx-keepalive.sh.
# Query ember.switch.status to confirm the PLX switch is alive.
echo ""
echo "[7] Checking ember PCIe switch keepalive..."
SWITCH_JSON=$(echo '{"jsonrpc":"2.0","method":"ember.switch.status","params":{},"id":1}' \
    | socat - UNIX-CONNECT:"$EMBER_SOCK" 2>/dev/null || true)
if echo "$SWITCH_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)
switches = d.get('result', {}).get('switches', [])
if not switches:
    print('    No switches configured in ember'); sys.exit(0)
for sw in switches:
    alive = sw.get('alive', False)
    bdf = sw.get('bdf', '?')
    name = sw.get('name', '?')
    tag = '✓' if alive else '✗ DEAD'
    print(f'    {name} ({bdf}): {tag}')
    if not alive:
        sys.exit(1)
" 2>/dev/null; then
    echo "    ember switch keepalive OK  ✓"
else
    echo "    WARN: switch may be dead — check journalctl -u coral-ember"
fi

# ── Step 8: Sovereign boot via coral stack ────────────────────────────────────
# No warm-fecs: novel binding triggers PLX D3cold cascade on this hardware.
# See header comment for full explanation.
#
# kepler_falcon_boot boots FECS/GPCCS from cold state entirely in Falcon SRAM.
# GDDR5 training via DEVINIT is attempted first (non-fatal if it fails).
# Successful kepler_falcon_boot sets GR_READY even without trained GDDR5.
# compute_ready=true requires both GR_READY + VRAM sentinel — cold boot may
# achieve GR_READY only. VRAM training is a future sovereign_stage task.
echo ""
echo "[8] Running sovereign boot on K80 die0 via coralctl..."
echo "========================================"

SOVEREIGN_JSON=$(CORALREEF_EMBER_SOCKET="$EMBER_SOCK" coralctl sovereign-boot "$K80_DIE0" 2>&1) || true
echo "$SOVEREIGN_JSON"

# Extract per-stage outcomes for triage
COMPUTE_READY=$(echo "$SOVEREIGN_JSON" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(str(d.get('compute_ready','false')).lower())" \
    2>/dev/null || echo "false")

FALCON_BOOT_OK=$(echo "$SOVEREIGN_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for s in d.get('stages', []):
    if s.get('name') == 'falcon_boot' and s.get('status') == 'ok':
        print('true'); sys.exit(0)
print('false')
" 2>/dev/null || echo "false")

echo "========================================"
if [ "$COMPUTE_READY" = "true" ]; then
    echo " Sovereign boot complete — compute_ready=true  ✓"
elif [ "$FALCON_BOOT_OK" = "true" ]; then
    echo " PARTIAL: kepler_falcon_boot succeeded — GR_READY set  ✓"
    echo "   GDDR5 not trained (cold boot without warm-fecs) — VRAM verify failed."
    echo "   This is expected on first cold boot. GR_READY confirms FECS/GPCCS boot."
    echo "   Next step: implement GDDR5 training in sovereign_stages.rs."
else
    echo " WARN: sovereign pipeline did not complete."
    echo "   → stages output above for per-stage diagnosis"
    echo "   → Check: journalctl -u coral-ember -u coral-glowplug --since '5 min ago'"
    echo "   → Verify kepler_fw_dir in /etc/coralreef/glowplug.toml"
    echo "   → Run exp184 in diagnostic mode for register-level triage:"
    echo "       cd $BARRACUDA && ./target/release/exp184_k80_gr_sovereign --bdf $K80_DIE0 --dry-run"
    exit 1
fi

echo ""
echo "========================================"
echo " Done."
