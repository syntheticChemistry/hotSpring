#!/bin/bash
# upgrade-toadstool.sh — Upgrade toadStool daemon via plasmidBin ecoBin
#
# Fetches the latest toadStool ecoBin from plasmidBin, installs it,
# and restarts the toadstool-ember systemd service with zero downtime.
#
# Usage:
#   sudo ./scripts/boot/upgrade-toadstool.sh           # upgrade and restart
#   sudo ./scripts/boot/upgrade-toadstool.sh --check    # check for updates only
#   sudo ./scripts/boot/upgrade-toadstool.sh --force    # force re-download

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOTSPRING_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ECOPRIMALS_ROOT="$(cd "$HOTSPRING_ROOT/../.." && pwd)"
PLASMIDBIN="${ECOPRIMALS_ROOT}/infra/plasmidBin"

CHECK_ONLY=false
FORCE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK_ONLY=true; shift ;;
        --force) FORCE="--force"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ ! -d "$PLASMIDBIN" ] || [ ! -x "$PLASMIDBIN/fetch.sh" ]; then
    echo "FATAL: plasmidBin not found at $PLASMIDBIN" >&2
    echo "  Clone: git clone https://github.com/ecoPrimals/plasmidBin $PLASMIDBIN" >&2
    exit 1
fi

CURRENT_VERSION=$(/usr/local/bin/toadstool --version 2>/dev/null || echo "not installed")
echo ">>> toadStool upgrade via plasmidBin"
echo "  Current: $CURRENT_VERSION"
echo "  Source:  $PLASMIDBIN"

cd "$PLASMIDBIN"

if $CHECK_ONLY; then
    echo ""
    echo "  Checking for updates..."
    git fetch --quiet origin 2>/dev/null || true
    LOCAL=$(git rev-parse HEAD 2>/dev/null || echo "?")
    REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "?")
    if [ "$LOCAL" = "$REMOTE" ]; then
        echo "  plasmidBin is up to date."
    else
        echo "  Update available (local=$LOCAL remote=$REMOTE)"
        echo "  Run: cd $PLASMIDBIN && git pull"
    fi
    exit 0
fi

echo ""
echo ">>> Pulling latest plasmidBin..."
git pull --quiet origin main 2>/dev/null || echo "  (git pull skipped — may be detached)"

echo ">>> Fetching toadstool ecoBin..."
./fetch.sh --primal toadstool $FORCE

TOADSTOOL_BIN="$PLASMIDBIN/primals/x86_64-unknown-linux-musl/toadstool"
if [ ! -f "$TOADSTOOL_BIN" ]; then
    TOADSTOOL_BIN="$PLASMIDBIN/primals/toadstool"
fi

if [ ! -f "$TOADSTOOL_BIN" ]; then
    echo "FATAL: ecoBin not found after fetch" >&2
    exit 1
fi

NEW_HASH=$(sha256sum "$TOADSTOOL_BIN" | cut -d' ' -f1)
OLD_HASH=$(sha256sum /usr/local/bin/toadstool 2>/dev/null | cut -d' ' -f1 || echo "none")

if [ "$NEW_HASH" = "$OLD_HASH" ] && [ -z "$FORCE" ]; then
    echo ""
    echo ">>> Already at latest version (hash match). Nothing to do."
    exit 0
fi

echo ">>> Installing..."
echo "  Old hash: ${OLD_HASH:0:16}..."
echo "  New hash: ${NEW_HASH:0:16}..."

sudo install -m 755 "$TOADSTOOL_BIN" /usr/local/bin/toadstool

NEW_VERSION=$(/usr/local/bin/toadstool --version 2>/dev/null || echo "unknown")
echo "  Installed: $NEW_VERSION"

echo ">>> Updating service file..."
sudo cp "$SCRIPT_DIR/toadstool-ember.service" /etc/systemd/system/toadstool-ember.service
sudo systemctl daemon-reload

echo ">>> Restarting toadstool-ember..."
sudo systemctl restart toadstool-ember
sleep 2

if systemctl is-active --quiet toadstool-ember; then
    echo ""
    echo ">>> Upgrade complete!"
    echo "  Version: $NEW_VERSION"
    echo "  Status:  $(systemctl is-active toadstool-ember)"
    echo ""
    echo "  Verify: echo '{\"jsonrpc\":\"2.0\",\"method\":\"device.list\",\"id\":1}' | socat - UNIX:/run/toadstool/biomeos/compute.sock"
else
    echo ""
    echo ">>> WARNING: toadstool-ember failed to start after upgrade!"
    echo "  Check: journalctl -u toadstool-ember -n 20"
    exit 1
fi
