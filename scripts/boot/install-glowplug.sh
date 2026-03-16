#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORALREEF_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)/coralReef"

echo ">>> Building coral-glowplug (release)..."
cd "$CORALREEF_ROOT"
cargo build -p coral-glowplug --release

echo ">>> Installing config..."
sudo mkdir -p /etc/coralreef
sudo cp "$SCRIPT_DIR/glowplug.toml" /etc/coralreef/glowplug.toml

echo ">>> Installing systemd unit..."
sudo cp "$SCRIPT_DIR/coral-glowplug.service" /etc/systemd/system/
sudo systemctl daemon-reload

echo ">>> Installing udev rules + sudoers..."
sudo cp "$SCRIPT_DIR/99-coralreef-permissions.rules" /etc/udev/rules.d/
sudo cp "$SCRIPT_DIR/coralreef-sudoers" /etc/sudoers.d/coralreef
sudo chmod 440 /etc/sudoers.d/coralreef
sudo udevadm control --reload-rules

echo ""
echo "Installation complete. To enable at boot:"
echo "  sudo systemctl enable coral-glowplug"
echo "  sudo systemctl start coral-glowplug"
echo ""
echo "Manual start:"
echo "  coral-glowplug --bdf 0000:4a:00.0"
echo ""
echo "Check status:"
echo "  systemctl status coral-glowplug"
echo '  echo '"'"'{"Status":null}'"'"' | nc -U /run/coralreef/glowplug.sock'
