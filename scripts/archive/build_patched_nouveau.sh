#!/bin/bash
# Build a patched nouveau.ko with NOP'd teardown functions.
#
# This compiles nouveau from mainline kernel source with four teardown
# functions gutted so that unbinding the driver preserves GPU state
# (GDDR5 training, PMC_ENABLE, GR falcon state). This is the source-level
# equivalent of the livepatch/kprobe/eBPF approaches that failed on
# kernel 6.17+ due to strict relocation checks.
#
# Target functions:
#   nvkm_mc_disable  — prevents PMC_ENABLE bit clearing
#   nvkm_mc_reset    — prevents engine reset cycles during subdev fini
#   nvkm_pmu_fini    — keeps PMU falcon running (controls DRAM refresh)
#   nvkm_fifo_fini   — prevents PFIFO runlist teardown
#   gf100_gr_fini    — prevents FECS/GPCCS falcon release
#
# Usage:
#   sudo ./build_patched_nouveau.sh
#
# Output:
#   /tmp/nouveau-patch/nouveau.ko  — patched module ready for insmod

set -euo pipefail

KVER=$(uname -r)
SRCDIR=/tmp/nouveau-patch/drivers/gpu/drm/nouveau
BUILDDIR=/lib/modules/$KVER/build
OUTMOD=/tmp/nouveau-patch/nouveau.ko

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ ! -d "$SRCDIR" ]; then
    log "Downloading kernel source (nouveau only)..."
    mkdir -p /tmp/nouveau-patch
    cd /tmp/nouveau-patch
    MAJOR=$(echo "$KVER" | cut -d. -f1)
    MINOR=$(echo "$KVER" | cut -d. -f2)
    PATCH=$(echo "$KVER" | cut -d. -f3 | cut -d- -f1)
    TARBALL="https://cdn.kernel.org/pub/linux/kernel/v${MAJOR}.x/linux-${MAJOR}.${MINOR}.${PATCH}.tar.xz"
    log "Fetching $TARBALL ..."
    curl -sL "$TARBALL" | xz -d | tar xf - --strip-components=1 \
        "linux-${MAJOR}.${MINOR}.${PATCH}/drivers/gpu/drm/nouveau/"
    log "Extracted $(find drivers/gpu/drm/nouveau -name '*.c' | wc -l) C files"
fi

cd /tmp/nouveau-patch

log "=== Patching teardown functions ==="

patch_function() {
    local file="$1" func="$2" rtype="$3"
    if ! grep -q "$func" "$file"; then
        log "WARN: $func not found in $file — skipping"
        return
    fi
    if grep -q "WARM_CATCH_NOP.*$func" "$file"; then
        log "  already patched: $func"
        return
    fi

    if [ "$rtype" = "void" ]; then
        # void function: insert return at top of body
        sed -i "/$func.*{$/,/^}/ {
            /^{/a\\
\\t/* WARM_CATCH_NOP: $func — preserve GPU state for VFIO handoff */\\
\\treturn;
        }" "$file"
    else
        # int function: insert return 0 at top of body
        sed -i "/$func.*{$/,/^}/ {
            /^{/a\\
\\t/* WARM_CATCH_NOP: $func — preserve GPU state for VFIO handoff */\\
\\treturn 0;
        }" "$file"
    fi
    log "  patched: $func (${rtype})"
}

# 1. nvkm_mc_disable — void
patch_function "$SRCDIR/nvkm/subdev/mc/base.c" "nvkm_mc_disable" "void"

# 2. nvkm_mc_reset — void
patch_function "$SRCDIR/nvkm/subdev/mc/base.c" "nvkm_mc_reset" "void"

# 3. nvkm_pmu_fini — int
patch_function "$SRCDIR/nvkm/subdev/pmu/base.c" "nvkm_pmu_fini" "int"

# 4. nvkm_fifo_fini — int
patch_function "$SRCDIR/nvkm/engine/fifo/base.c" "nvkm_fifo_fini" "int"

# 5. gf100_gr_fini — int
patch_function "$SRCDIR/nvkm/engine/gr/gf100.c" "gf100_gr_fini" "int"

log ""
log "=== Verifying patches ==="
grep -rn "WARM_CATCH_NOP" "$SRCDIR/" | while read -r line; do
    log "  $line"
done

log ""
log "=== Building patched nouveau.ko ==="
make -C "$BUILDDIR" M="$SRCDIR" CONFIG_DRM_NOUVEAU=m \
    CONFIG_DRM_NOUVEAU_BACKLIGHT=y \
    -j"$(nproc)" 2>&1 | tail -30

if [ -f "$SRCDIR/nouveau.ko" ]; then
    cp "$SRCDIR/nouveau.ko" "$OUTMOD"
    log ""
    log "=== BUILD SUCCESSFUL ==="
    log "Module: $OUTMOD"
    log "Size:   $(du -h "$OUTMOD" | cut -f1)"
    log ""
    log "Usage:"
    log "  # Unload stock nouveau if loaded"
    log "  sudo rmmod nouveau 2>/dev/null || true"
    log "  # Load patched module"
    log "  sudo insmod $OUTMOD"
    log "  # Bind K80 to nouveau"
    log "  echo 0000:4b:00.0 > /sys/bus/pci/drivers/nouveau/bind"
    log "  # Wait for GDDR5 training"
    log "  sleep 5"
    log "  # Unbind (teardown NOP'd — GDDR5 preserved)"
    log "  echo 0000:4b:00.0 > /sys/bus/pci/drivers/nouveau/unbind"
    log "  # Rebind to vfio-pci"
    log "  echo vfio-pci > /sys/bus/pci/devices/0000:4b:00.0/driver_override"
    log "  echo 0000:4b:00.0 > /sys/bus/pci/drivers/vfio-pci/bind"
else
    log "ERROR: Build failed — nouveau.ko not found"
    log "Check the build output above for errors"
    exit 1
fi
