#!/bin/sh
# Install coralReef passwordless permissions and restart ember.
#
# Run via:  pkexec /path/to/install-coralreef-perms.sh
#   or:     sudo  /path/to/install-coralreef-perms.sh
#
# After first install, the script itself is in the sudoers rule,
# so subsequent runs need no pkexec.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REEF_ROOT="${REEF_ROOT:-$(cd "$SCRIPT_DIR/../../../primals/coralReef" 2>/dev/null && pwd || echo "/home/$USER/Development/ecoPrimals/primals/coralReef")}"

echo "=== Installing coralReef sudoers rule ==="
install -m 440 "$SCRIPT_DIR/99-coralreef-nopasswd" /etc/sudoers.d/99-coralreef-nopasswd
visudo -cf /etc/sudoers.d/99-coralreef-nopasswd
echo "  sudoers rule installed and validated"

echo "=== Installing service file ==="
cp "$SCRIPT_DIR/coral-ember.service" /etc/systemd/system/coral-ember.service
systemctl daemon-reload
echo "  service file updated"

echo "=== Deploying binaries ==="
if [ -f "$REEF_ROOT/target/release/coral-ember" ]; then
    systemctl stop coral-ember 2>/dev/null || true
    sleep 1
    install -m 755 "$REEF_ROOT/target/release/coral-ember" /usr/local/bin/coral-ember
    install -m 755 "$REEF_ROOT/target/release/coralctl" /usr/local/bin/coralctl
    echo "  binaries installed"
else
    echo "  skipped (no release build found)"
fi

echo "=== Starting ember ==="
systemctl start coral-ember
sleep 3
systemctl status coral-ember --no-pager || true
echo ""
echo "=== Done ==="
echo "Future operations need NO password:"
echo "  sudo systemctl restart coral-ember"
echo "  sudo install -m 755 target/release/coralctl /usr/local/bin/coralctl"
echo "  echo 0000:03:00.0 | sudo tee /sys/bus/pci/devices/0000:03:00.0/reset"
echo "  sudo $0   (re-run this script)"
