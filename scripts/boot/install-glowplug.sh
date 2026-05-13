#!/bin/bash
# install-glowplug.sh — Install toadStool from plasmidBin ecoBins
#
# Fetches the latest toadStool ecoBin from plasmidBin (GitHub Releases)
# and installs the daemon binary, config, and systemd units.
#
# Falls back to cargo build from source if plasmidBin is not available.
#
# Usage: sudo ./scripts/boot/install-glowplug.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOTSPRING_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ECOPRIMALS_ROOT="$(cd "$HOTSPRING_ROOT/../.." && pwd)"
PLASMIDBIN="${ECOPRIMALS_ROOT}/infra/plasmidBin"

echo ">>> Installing toadStool diesel engine..."

if [ -d "$PLASMIDBIN" ] && [ -x "$PLASMIDBIN/fetch.sh" ]; then
    echo ">>> Using plasmidBin ecoBin deployment..."
    cd "$PLASMIDBIN"

    ./fetch.sh --primal toadstool
    TOADSTOOL_BIN="$PLASMIDBIN/primals/x86_64-unknown-linux-musl/toadstool"

    if [ ! -f "$TOADSTOOL_BIN" ]; then
        echo "FATAL: fetch succeeded but binary not found at $TOADSTOOL_BIN" >&2
        exit 1
    fi

    echo ">>> ecoBin: $(file "$TOADSTOOL_BIN" | cut -d: -f2)"
    sudo install -m 755 "$TOADSTOOL_BIN" /usr/local/bin/toadstool
else
    echo ">>> plasmidBin not found — building from source..."
    TOADSTOOL_ROOT="${ECOPRIMALS_ROOT}/primals/toadStool"
    if [ ! -d "$TOADSTOOL_ROOT" ]; then
        echo "FATAL: Neither plasmidBin nor toadStool source found" >&2
        exit 1
    fi
    cd "$TOADSTOOL_ROOT"
    cargo build --release -p toadstool-cli
    sudo install -m 755 target/release/toadstool /usr/local/bin/toadstool
fi

echo ">>> Installed: $(/usr/local/bin/toadstool --version)"

echo ">>> Installing config..."
sudo mkdir -p /etc/toadstool /var/lib/toadstool
sudo cp "$SCRIPT_DIR/glowplug.toml" /etc/toadstool/glowplug.toml

echo ">>> Installing systemd units..."
sudo cp "$SCRIPT_DIR/toadstool-ember.service" /etc/systemd/system/
if [ -f "$SCRIPT_DIR/toadstool-glowplug.service" ]; then
    sudo cp "$SCRIPT_DIR/toadstool-glowplug.service" /etc/systemd/system/
fi
sudo systemctl daemon-reload

echo ">>> Ensuring toadstool group..."
sudo getent group toadstool >/dev/null 2>&1 || sudo groupadd toadstool

echo ""
echo "Installation complete. To enable at boot:"
echo "  sudo systemctl enable toadstool-ember"
echo "  sudo systemctl start toadstool-ember"
echo ""
echo "Check status:"
echo "  systemctl status toadstool-ember"
echo "  toadstool device list"
echo "  echo '{\"jsonrpc\":\"2.0\",\"method\":\"device.list\",\"id\":1}' | sudo socat - UNIX:/run/toadstool/biomeos/compute.sock"
