#!/bin/bash
set -euo pipefail
# Deploy extended livepatch: NOP mc_reset + gf100_gr_fini + nvkm_falcon_fini
# Must be run as root (via pkexec from graphical terminal)

KO_SRC="/tmp/nouveau-build/livepatch/livepatch_nvkm_mc_reset.ko"
KO_DST="/lib/modules/$(uname -r)/extra/livepatch_nvkm_mc_reset.ko"
SYSFS="/sys/kernel/livepatch/livepatch_nvkm_mc_reset"

echo "=== Deploying extended livepatch (v2: mc_reset + gr_fini + falcon_fini) ==="

# Step 1: Disable current livepatch if active
if [ -d "$SYSFS" ]; then
    echo "step 1: disabling current livepatch..."
    echo 0 > "$SYSFS/enabled"
    sleep 2
    # Wait for transition to complete
    for i in $(seq 1 10); do
        if [ "$(cat "$SYSFS/transition" 2>/dev/null)" = "0" ] 2>/dev/null; then
            break
        fi
        sleep 1
    done
    echo "step 2: removing old module..."
    rmmod livepatch_nvkm_mc_reset 2>/dev/null || true
    sleep 1
else
    echo "step 1: no active livepatch found, skipping disable"
fi

# Step 2: Install new module
echo "step 3: installing new module to $KO_DST..."
mkdir -p "$(dirname "$KO_DST")"
cp "$KO_SRC" "$KO_DST"
depmod -a

# Step 3: Load new module
echo "step 4: loading extended livepatch..."
modprobe livepatch_nvkm_mc_reset

# Verify
echo "step 5: verifying..."
if [ -d "$SYSFS" ]; then
    echo "  livepatch ACTIVE"
    ls "$SYSFS/nouveau/"
else
    echo "  ERROR: livepatch not active!"
    exit 1
fi

echo "=== Extended livepatch deployed — gf100_gr_fini + nvkm_falcon_fini + nvkm_mc_reset all NOPed ==="
