#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORALREEF_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)/primals/coralReef"

echo ">>> Building coral-ember + coral-glowplug (release)..."
cd "$CORALREEF_ROOT"
cargo build -p coral-ember -p coral-glowplug --release

echo ">>> Installing binaries..."
sudo install -m 755 target/release/coral-ember /usr/local/bin/coral-ember
sudo install -m 755 target/release/coral-glowplug /usr/local/bin/coral-glowplug
sudo install -m 755 target/release/coralctl /usr/local/bin/coralctl

echo ">>> Installing config..."
sudo mkdir -p /etc/coralreef
sudo cp "$SCRIPT_DIR/glowplug.toml" /etc/coralreef/glowplug.toml

echo ">>> Installing systemd units..."
sudo cp "$CORALREEF_ROOT/crates/coral-glowplug/coral-ember.service" /etc/systemd/system/
sudo cp "$CORALREEF_ROOT/crates/coral-glowplug/coral-glowplug.service" /etc/systemd/system/
sudo systemctl daemon-reload

echo ">>> Generating udev rules from config..."
sudo coralctl deploy-udev --config /etc/coralreef/glowplug.toml
sudo udevadm control --reload-rules

echo ""
echo "Installation complete. To enable at boot:"
echo "  sudo systemctl enable coral-ember coral-glowplug"
echo "  sudo systemctl start coral-ember"
echo "  sudo systemctl start coral-glowplug"
echo ""
echo "Check status:"
echo "  systemctl status coral-ember coral-glowplug"
echo "  echo '{\"jsonrpc\":\"2.0\",\"method\":\"health.check\",\"id\":1}' | socat - UNIX:/run/coralreef/glowplug.sock"
