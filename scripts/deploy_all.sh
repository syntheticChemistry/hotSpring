#!/bin/bash
# One-shot deployment: ember binaries + udev + sudoers + livepatch + services.
# After this, the Cursor agent can operate fully via `sudo -n` (no more pkexec).
#
# Run: pkexec bash /home/biomegate/Development/ecoPrimals/springs/hotSpring/scripts/deploy_all.sh
set -euo pipefail

HOTSPRING="/home/biomegate/Development/ecoPrimals/springs/hotSpring"
CORALREEF="/home/biomegate/Development/ecoPrimals/primals/coralReef"
TARGET="${CORALREEF}/target/release"
LP_SRC_DIR="${HOTSPRING}/scripts/livepatch"
KO_SRC="${LP_SRC_DIR}/livepatch_nvkm_mc_reset.ko"
KVER="$(uname -r)"
KO_DST="/lib/modules/${KVER}/extra/livepatch_nvkm_mc_reset.ko"
SYSFS_LP="/sys/kernel/livepatch/livepatch_nvkm_mc_reset"

echo "=== coralReef full deployment ==="
echo ""

# ── Phase 1: Sudoers ──
echo "▸ Phase 1: Deploying sudoers rules"
cp "${HOTSPRING}/scripts/boot/coralreef-sudoers" /etc/sudoers.d/coralreef
chmod 440 /etc/sudoers.d/coralreef
visudo -c -f /etc/sudoers.d/coralreef
echo "  sudoers: OK"

# ── Phase 2: Udev rules ──
echo "▸ Phase 2: Deploying udev rules"
cp "${HOTSPRING}/scripts/boot/99-coralreef-permissions.rules" /etc/udev/rules.d/
cp "${HOTSPRING}/scripts/boot/99-coralreef-vfio.rules" /etc/udev/rules.d/
udevadm control --reload-rules
echo "  udev: reloaded"

# ── Phase 3: Modprobe config ──
echo "▸ Phase 3: Deploying modprobe config"
cp "${HOTSPRING}/scripts/boot/coralreef-dual-titanv.conf" /etc/modprobe.d/
echo "  modprobe: OK"

# ── Phase 4: Livepatch module (build from source) ──
echo "▸ Phase 4: Building livepatch from ${LP_SRC_DIR}"
if [ -f "${LP_SRC_DIR}/Makefile" ]; then
    MAKE=/usr/bin/make /usr/bin/make -C "/lib/modules/${KVER}/build" M="${LP_SRC_DIR}" modules 2>&1 | tail -3
fi
if [ -f "$KO_SRC" ]; then
    echo "▸ Phase 4: Deploying livepatch module"
    if [ -d "$SYSFS_LP" ]; then
        echo "  disabling current livepatch..."
        echo 0 > "${SYSFS_LP}/enabled"
        sleep 2
        for i in $(seq 1 10); do
            [ "$(cat "${SYSFS_LP}/transition" 2>/dev/null)" = "0" ] 2>/dev/null && break
            sleep 1
        done
        rmmod livepatch_nvkm_mc_reset 2>/dev/null || true
        sleep 1
    fi
    mkdir -p "$(dirname "$KO_DST")"
    cp "$KO_SRC" "$KO_DST"
    depmod -a
    modprobe livepatch_nvkm_mc_reset
    if [ -d "$SYSFS_LP" ]; then
        echo "  livepatch: ACTIVE ($(ls "${SYSFS_LP}/nouveau/funcs/" 2>/dev/null | tr '\n' ' '))"
    else
        echo "  livepatch: FAILED to activate"
    fi
else
    echo "▸ Phase 4: Livepatch module not found at ${KO_SRC} — skipping"
fi

# ── Phase 5: Binaries + services ──
echo "▸ Phase 5: Deploying binaries"
for bin in coral-ember coral-glowplug; do
    if [ ! -f "${TARGET}/${bin}" ]; then
        echo "  SKIP ${bin}: not built"
        continue
    fi
    systemctl stop "${bin}" 2>/dev/null || true
done
sleep 1

for bin in coral-ember coral-glowplug; do
    if [ ! -f "${TARGET}/${bin}" ]; then continue; fi
    rm -f "/usr/local/bin/${bin}"
    cp "${TARGET}/${bin}" "/usr/local/bin/${bin}"
    chmod 755 "/usr/local/bin/${bin}"
    echo "  ${bin}: installed"
done

if [ -f "${TARGET}/coralctl" ]; then
    rm -f /usr/local/bin/coralctl
    cp "${TARGET}/coralctl" /usr/local/bin/coralctl
    chmod 755 /usr/local/bin/coralctl
    echo "  coralctl: installed"
fi

echo "▸ Phase 5b: Deploying systemd service files"
cp "${HOTSPRING}/scripts/boot/coral-ember.service" /etc/systemd/system/coral-ember.service
cp "${HOTSPRING}/scripts/boot/coral-glowplug.service" /etc/systemd/system/coral-glowplug.service
echo "  service files: installed"

echo "▸ Phase 6: Restarting services"
systemctl daemon-reload
systemctl start coral-ember
sleep 2
if systemctl is-active --quiet coral-ember; then
    echo "  coral-ember: ACTIVE"
else
    echo "  coral-ember: FAILED"
    journalctl -u coral-ember --no-pager -n 5
fi

systemctl start coral-glowplug
sleep 2
if systemctl is-active --quiet coral-glowplug; then
    echo "  coral-glowplug: ACTIVE"
else
    echo "  coral-glowplug: WARNING — not active (check journalctl -u coral-glowplug)"
fi

echo ""
echo "=== Deployment complete ==="
echo "  Agent can now use 'sudo -n' for all operations (no more pkexec)."
echo "  Ember has the reset_method race fix."
echo "  Livepatch NOPs mc_reset + gr_fini + falcon_fini + runl_commit."
