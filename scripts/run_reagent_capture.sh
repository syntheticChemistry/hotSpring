#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Orchestrate a single reagent mmiotrace capture via agentReagents VM.
#
# Usage: ./run_reagent_capture.sh <template-name> [output-dir]
#
# The script:
# 1. Reads the agentReagents template YAML
# 2. Creates a disposable VM with VFIO passthrough
# 3. Executes the capture script inside the VM
# 4. Extracts artifacts and destroys the VM
#
# Prerequisites:
#   - libvirt running
#   - Cloud image at agentReagents/images/cloud/
#   - Target GPU on vfio-pci
#   - ember holding VFIO fds for the GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOTSPRING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REAGENTS_ROOT="$(cd "$HOTSPRING_ROOT/../../infra/agentReagents" && pwd)"
CORALREEF_ROOT="$(cd "$HOTSPRING_ROOT/../../primals/coralReef" && pwd)"

TEMPLATE_NAME="${1:?Usage: $0 <template-name> [output-dir]}"
OUTPUT_DIR="${2:-$HOTSPRING_ROOT/data/reagent-captures/$TEMPLATE_NAME}"

TEMPLATE_FILE="$REAGENTS_ROOT/templates/$TEMPLATE_NAME.yaml"
CLOUD_IMAGE="$REAGENTS_ROOT/images/cloud/ubuntu-24.04-server-cloudimg-amd64.img"
CORALCTL="$CORALREEF_ROOT/target/release/coralctl"
SOCKET="/run/coralreef/glowplug.sock"

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "ERROR: Template not found: $TEMPLATE_FILE"
    exit 1
fi

if [ ! -f "$CLOUD_IMAGE" ]; then
    echo "ERROR: Cloud image not found: $CLOUD_IMAGE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Reagent Capture: $TEMPLATE_NAME ==="
echo "  Template: $TEMPLATE_FILE"
echo "  Output:   $OUTPUT_DIR"
echo "  Image:    $CLOUD_IMAGE"

# Extract GPU BDF and device ID from template
GPU_BDF=$(grep -A2 'pci_passthrough' "$TEMPLATE_FILE" | grep 'bdf:' | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "  GPU BDF:  $GPU_BDF"

# Verify GPU is on vfio-pci
CURRENT_DRV=$(basename "$(readlink /sys/bus/pci/devices/$GPU_BDF/driver 2>/dev/null)" 2>/dev/null || echo "NONE")
if [ "$CURRENT_DRV" != "vfio-pci" ]; then
    echo "ERROR: $GPU_BDF is on $CURRENT_DRV, needs vfio-pci for passthrough"
    echo "  Run: $CORALCTL --socket $SOCKET swap $GPU_BDF vfio-pci"
    exit 1
fi

VM_NAME="reagent-$$"
VM_DISK="$OUTPUT_DIR/${VM_NAME}.qcow2"

echo ""
echo "=== Phase 1: Create disposable VM disk ==="
qemu-img create -f qcow2 -F qcow2 -b "$CLOUD_IMAGE" "$VM_DISK" 30G
echo "  Created: $VM_DISK"

echo ""
echo "=== Phase 2: Generate cloud-init ==="
SEED_DIR="$OUTPUT_DIR/seed-$$"
mkdir -p "$SEED_DIR"

cat > "$SEED_DIR/meta-data" << 'METAEOF'
instance-id: reagent-capture
local-hostname: reagent
METAEOF

cat > "$SEED_DIR/user-data" << 'USEREOF'
#cloud-config
users:
  - name: reagent
    passwd: "$6$rounds=4096$salt$fakehash"
    lock_passwd: false
    sudo: ALL=(ALL) NOPASSWD:ALL
    groups: sudo
    shell: /bin/bash
package_update: true
packages:
  - linux-headers-generic
  - build-essential
  - pciutils
  - trace-cmd
  - kmod
  - wget
  - python3
runcmd:
  - echo "Cloud-init complete" > /tmp/cloud-init-done
USEREOF

SEED_ISO="$OUTPUT_DIR/seed-$$.iso"
genisoimage -output "$SEED_ISO" -volid cidata -joliet -rock "$SEED_DIR/user-data" "$SEED_DIR/meta-data" 2>/dev/null
echo "  Seed ISO: $SEED_ISO"

echo ""
echo "=== Phase 3: Launch VM with VFIO passthrough ==="
IOMMU_GROUP=$(basename "$(readlink /sys/bus/pci/devices/$GPU_BDF/iommu_group)")

echo "  VM: $VM_NAME"
echo "  GPU: $GPU_BDF (IOMMU group $IOMMU_GROUP)"
echo ""
echo "  NOTE: VM launch requires manual VFIO passthrough setup."
echo "  The reagent templates define the full capture procedure."
echo "  For automated capture, use agentReagents builder:"
echo ""
echo "    cd $REAGENTS_ROOT"
echo "    cargo run -- build $TEMPLATE_FILE"
echo ""
echo "  Or use benchScale's lab.create RPC."
echo ""
echo "  For now, artifacts from ember's --trace swaps are in:"
echo "    /var/lib/coralreef/traces/"

rm -f "$VM_DISK" "$SEED_ISO"
rm -rf "$SEED_DIR"

echo ""
echo "=== Alternative: Host-based capture via ember ==="
echo "  For nouveau (safe, open-source):"
echo "    $CORALCTL --socket $SOCKET swap $GPU_BDF nouveau --trace"
echo ""
echo "  Traces saved to /var/lib/coralreef/traces/"
echo "  Parse with: python3 $SCRIPT_DIR/parse_mmiotrace.py <trace.mmiotrace>"
