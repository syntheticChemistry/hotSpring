#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Orchestrate a single reagent mmiotrace capture via agentReagents VM.
#
# Usage: ./run_reagent_capture.sh <template-name> [output-dir]
#
# Prerequisites:
#   - libvirt running
#   - Cloud image at agentReagents/images/cloud/
#   - Target GPU on vfio-pci
#   - ember holding VFIO fds for the GPU
#   - agent-reagents binary built (cargo build --release in agentReagents)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOTSPRING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REAGENTS_ROOT="$(cd "$HOTSPRING_ROOT/../../infra/agentReagents" && pwd)"
CORALREEF_ROOT="$(cd "$HOTSPRING_ROOT/../../primals/coralReef" && pwd)"

TEMPLATE_NAME="${1:?Usage: $0 <template-name> [output-dir]}"
OUTPUT_DIR="${2:-$HOTSPRING_ROOT/data/reagent-captures/$TEMPLATE_NAME}"
TEMPLATE_FILE="$REAGENTS_ROOT/templates/$TEMPLATE_NAME.yaml"
CORALCTL="$CORALREEF_ROOT/target/release/coralctl"
SOCKET="/run/coralreef/glowplug.sock"
AGENT_REAGENTS="$REAGENTS_ROOT/target/release/agent-reagents"

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "ERROR: Template not found: $TEMPLATE_FILE" >&2
    echo "  Available templates:" >&2
    ls "$REAGENTS_ROOT/templates/"*.yaml 2>/dev/null | sed 's/.*\//    /' >&2
    exit 1
fi

# Build agent-reagents if binary is missing
if [ ! -f "$AGENT_REAGENTS" ]; then
    echo "Building agent-reagents (first run)..."
    (cd "$REAGENTS_ROOT" && cargo build --release)
fi

# Extract GPU BDF from template for pre-flight check
GPU_BDF=$(grep -A2 'pci_passthrough' "$TEMPLATE_FILE" | grep 'bdf:' | head -1 | sed 's/.*"\(.*\)".*/\1/')
if [ -n "$GPU_BDF" ]; then
    CURRENT_DRV=$(basename "$(readlink "/sys/bus/pci/devices/$GPU_BDF/driver" 2>/dev/null)" 2>/dev/null || echo "NONE")
    if [ "$CURRENT_DRV" != "vfio-pci" ]; then
        echo "ERROR: $GPU_BDF is on $CURRENT_DRV, needs vfio-pci for passthrough" >&2
        echo "  Run: $CORALCTL --socket $SOCKET swap $GPU_BDF vfio-pci" >&2
        exit 1
    fi
    echo "Pre-flight: $GPU_BDF on vfio-pci ✓"
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Reagent Capture: $TEMPLATE_NAME ==="
echo "  Template: $TEMPLATE_FILE"
echo "  Output:   $OUTPUT_DIR"
echo ""

# Delegate to agent-reagents builder which handles:
#   - Cloud image provisioning
#   - Cloud-init generation from template
#   - VM creation with VFIO passthrough
#   - In-guest capture script execution
#   - Artifact extraction
#   - VM teardown
exec "$AGENT_REAGENTS" build \
    --template "$TEMPLATE_FILE" \
    --output "$OUTPUT_DIR"
