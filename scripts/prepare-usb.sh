#!/bin/sh
set -eu

# Prepare a USB drive as a hotSpring guideStone liveSpore deployment.
#
# Supports two filesystem modes:
#   --ext4   (default) Native Linux — permissions preserved, direct execution
#   --exfat  Universal — readable on Windows/macOS/Linux; Linux uses tmpdir
#            fallback for execute bits; Windows/macOS use container or WSL2
#
# Usage:
#   sudo ./scripts/prepare-usb.sh /dev/sdX              # ext4 (default)
#   sudo ./scripts/prepare-usb.sh /dev/sdX --exfat      # universal
#   ./scripts/prepare-usb.sh --copy-only /mnt/usb        # skip format, copy only
#
# The script copies the full artifact including:
#   - validation/ (binaries, scripts, checksums, docs)
#   - container/hotspring-guidestone.tar (if built)
#   - hotspring.bat (Windows launcher)
#   - biomeOS spore markers
#   - liveSpore.json

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VALIDATION="$ROOT/validation"
ARTIFACT_NAME="hotSpring-guideStone-v0.7.0"

FS_TYPE="ext4"
COPY_ONLY=0
DEVICE=""
TARGET=""
LABEL="guideStone"

usage() {
    echo "Usage: $0 <device|--copy-only target_dir> [options]"
    echo ""
    echo "  <device>             Block device to format and populate (e.g. /dev/sdb)"
    echo "  --copy-only <dir>    Skip formatting, copy artifact to existing mount"
    echo ""
    echo "Options:"
    echo "  --ext4               Format as ext4 (default, Linux-native)"
    echo "  --exfat              Format as exFAT (universal, Windows/macOS/Linux)"
    echo "  --label=NAME         Volume label (default: guideStone)"
    echo ""
    echo "Examples:"
    echo "  sudo $0 /dev/sdb                  # ext4, auto-mount, copy"
    echo "  sudo $0 /dev/sdb --exfat          # exFAT, universal"
    echo "  $0 --copy-only /media/user/usb    # just copy to mounted drive"
    exit 1
}

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --ext4)      FS_TYPE="ext4" ;;
        --exfat)     FS_TYPE="exfat" ;;
        --label=*)   LABEL="${1#--label=}" ;;
        --copy-only)
            COPY_ONLY=1
            shift
            TARGET="${1:-}"
            [ -z "$TARGET" ] && usage
            ;;
        --help|-h)   usage ;;
        /dev/*)      DEVICE="$1" ;;
        *)
            if [ -d "$1" ] && [ "$COPY_ONLY" = "1" ]; then
                TARGET="$1"
            else
                echo "Unknown argument: $1"
                usage
            fi
            ;;
    esac
    shift
done

if [ "$COPY_ONLY" = "0" ] && [ -z "$DEVICE" ]; then
    usage
fi

echo "═══════════════════════════════════════════════════════════"
echo "  hotSpring liveSpore USB Preparation"
echo "  Artifact: $ARTIFACT_NAME"
if [ "$COPY_ONLY" = "1" ]; then
    echo "  Mode:     copy-only to $TARGET"
else
    echo "  Device:   $DEVICE"
    echo "  FS:       $FS_TYPE"
fi
echo "═══════════════════════════════════════════════════════════"
echo

# ── Format (unless copy-only) ────────────────────────────────────────

if [ "$COPY_ONLY" = "0" ]; then
    if [ "$(id -u)" != "0" ]; then
        echo "ERROR: formatting requires root. Run with sudo."
        exit 1
    fi

    if [ ! -b "$DEVICE" ]; then
        echo "ERROR: $DEVICE is not a block device"
        exit 1
    fi

    # Safety: refuse to format anything that looks like a system disk
    MOUNT_CHECK=$(mount | grep "^$DEVICE" || true)
    if echo "$MOUNT_CHECK" | grep -qE ' / | /boot | /home '; then
        echo "ERROR: $DEVICE appears to be a system disk. Refusing."
        exit 1
    fi

    echo "▸ Unmounting existing partitions on $DEVICE..."
    umount "${DEVICE}"* 2>/dev/null || true
    sleep 1

    echo "▸ Creating partition table..."
    parted -s "$DEVICE" mklabel gpt
    parted -s "$DEVICE" mkpart primary 1MiB 100%

    PART="${DEVICE}1"
    # Handle NVMe-style naming (e.g. /dev/nvme0n1p1)
    if [ ! -b "$PART" ]; then
        PART="${DEVICE}p1"
    fi
    sleep 1

    echo "▸ Formatting as $FS_TYPE..."
    case "$FS_TYPE" in
        ext4)
            mkfs.ext4 -L "$LABEL" -q "$PART"
            ;;
        exfat)
            if ! command -v mkfs.exfat >/dev/null 2>&1; then
                echo "ERROR: mkfs.exfat not found. Install exfatprogs:"
                echo "  apt install exfatprogs  # Debian/Ubuntu"
                echo "  dnf install exfatprogs  # Fedora"
                exit 1
            fi
            mkfs.exfat -L "$LABEL" "$PART"
            ;;
    esac

    echo "▸ Mounting..."
    TARGET=$(mktemp -d "/tmp/hotspring-usb.XXXXXX")
    mount "$PART" "$TARGET"
    echo "  Mounted at: $TARGET"
    echo
fi

if [ ! -d "$TARGET" ]; then
    echo "ERROR: Target directory does not exist: $TARGET"
    exit 1
fi

# ── Copy artifact ─────────────────────────────────────────────────────

DEST="$TARGET/$ARTIFACT_NAME"
echo "▸ Copying artifact to $DEST/..."
mkdir -p "$DEST"

# Core artifact files
for f in hotspring _lib.sh run run-matrix benchmark chuna-engine \
         deploy-nucleus run-overnight hotspring.bat \
         CHECKSUMS README GUIDESTONE.md LICENSE; do
    if [ -f "$VALIDATION/$f" ]; then
        cp "$VALIDATION/$f" "$DEST/"
    fi
done

# Binaries (preserving directory structure)
if [ -d "$VALIDATION/bin" ]; then
    cp -a "$VALIDATION/bin" "$DEST/"
fi

# Expected reference data
if [ -d "$VALIDATION/expected" ]; then
    cp -a "$VALIDATION/expected" "$DEST/"
fi

# Shaders
if [ -d "$VALIDATION/shaders" ]; then
    cp -a "$VALIDATION/shaders" "$DEST/"
fi

# Container tarball (if built)
if [ -d "$VALIDATION/container" ]; then
    cp -a "$VALIDATION/container" "$DEST/"
fi

# Create results directory
mkdir -p "$DEST/results"

# ── biomeOS ColdSpore markers ────────────────────────────────────────

echo "▸ Writing biomeOS ColdSpore markers..."

cat > "$TARGET/.biomeos-spore" <<SPORE_JSON
{
  "type": "ColdSpore",
  "artifact": "$ARTIFACT_NAME",
  "spring": "hotSpring",
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
SPORE_JSON

mkdir -p "$TARGET/biomeOS"
cat > "$TARGET/biomeOS/tower.toml" <<TOWER_TOML
[tower]
name = "hotSpring-guideStone-ColdSpore"
version = "0.7.0"
deployment_mode = "cold"

[capabilities]
validation = true
benchmark = true
generation = true
container = true

[required_primals]
TOWER_TOML

# ── liveSpore.json ────────────────────────────────────────────────────

echo "▸ Generating liveSpore.json..."
cat > "$TARGET/liveSpore.json" <<LIVESPORE_JSON
{
  "version": "3.0",
  "artifact": "$ARTIFACT_NAME",
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "filesystem": "$FS_TYPE",
  "architectures": {
    "x86_64": {"static": true, "gpu": true},
    "aarch64": {"static": true, "gpu": false}
  },
  "cross_platform": {
    "linux_native": true,
    "linux_exfat_tmpdir": $([ "$FS_TYPE" = "exfat" ] && echo "true" || echo "false"),
    "windows_wsl2": true,
    "windows_docker": true,
    "macos_docker": true,
    "container_image": "hotspring-guidestone:v0.7.0"
  },
  "deployment_strategy": {
    "linux_x86_64": "Direct execution — ./hotspring validate (auto GPU)",
    "linux_aarch64": "Direct execution — ./hotspring validate (CPU-only static)",
    "linux_exfat": "Tmpdir fallback for execute bits — transparent",
    "windows": "hotspring.bat → WSL2 or Docker",
    "macos": "./hotspring auto-dispatches to Docker",
    "container": "docker load < container/hotspring-guidestone.tar"
  },
  "validated_substrates": [
    "x86_64 Ubuntu 22.04 (CPU-only)",
    "x86_64 Ubuntu 22.04 (NVIDIA GPU passthrough)",
    "x86_64 Ubuntu 22.04 (AMD DRI passthrough)",
    "x86_64 Alpine 3.19 (musl-native)",
    "aarch64 Ubuntu 22.04 (qemu-user cross-arch)"
  ],
  "biomeos": {
    "spore_type": "ColdSpore",
    "marker": ".biomeos-spore",
    "tower": "biomeOS/tower.toml"
  },
  "runs": [],
  "systems_validated": []
}
LIVESPORE_JSON

# ── README at drive root ──────────────────────────────────────────────

echo "▸ Writing root README.txt..."
cat > "$TARGET/README.txt" <<ROOT_README
$ARTIFACT_NAME — ecoPrimals guideStone ColdSpore
=====================================================

Lattice QCD + Condensed Matter validation artifact.
Certified: deterministic, reference-traceable, self-verifying,
environment-agnostic, tolerance-documented.

Quick Start
-----------

  Linux x86_64 or aarch64 (ext4 USB or local copy):
    cd $ARTIFACT_NAME
    ./hotspring validate

  Linux (exFAT/FAT32/NFS — any filesystem, any arch):
    cd $ARTIFACT_NAME
    sh ./hotspring validate

  Windows:
    cd $ARTIFACT_NAME
    hotspring.bat validate
    (Uses WSL2 if available, otherwise Docker Desktop)

  macOS:
    cd $ARTIFACT_NAME
    sh ./hotspring validate
    (Auto-dispatches to Docker/Podman)

  Any OS with Docker:
    docker load < $ARTIFACT_NAME/container/hotspring-guidestone.tar
    docker run --rm -v ./results:/opt/validation/results \\
      hotspring-guidestone:v0.7.0 validate

For GPU acceleration:
    HOTSPRING_FORCE_GPU=1 ./hotspring validate      (Linux)
    docker run --rm --gpus all ...                   (Docker)

Contents
--------

  $ARTIFACT_NAME/      Full artifact (binaries, scripts, docs)
  .biomeos-spore          biomeOS ColdSpore marker
  biomeOS/                biomeOS tower configuration
  liveSpore.json          Machine-readable manifest + run history
  README.txt              This file

ROOT_README

# ── Set permissions (ext4 only — exfat ignores these) ─────────────────

echo "▸ Setting permissions..."
chmod +x "$DEST/hotspring" "$DEST/_lib.sh" \
         "$DEST/run" "$DEST/run-matrix" "$DEST/benchmark" \
         "$DEST/chuna-engine" "$DEST/deploy-nucleus" "$DEST/run-overnight" \
         2>/dev/null || true
find "$DEST/bin" -type f -exec chmod +x {} + 2>/dev/null || true
if [ -f "$DEST/container/docker-run.sh" ]; then
    chmod +x "$DEST/container/docker-run.sh" 2>/dev/null || true
fi

# ── Cleanup ───────────────────────────────────────────────────────────

if [ "$COPY_ONLY" = "0" ]; then
    echo "▸ Syncing..."
    sync
    echo "▸ Unmounting..."
    umount "$TARGET"
    rmdir "$TARGET"
fi

echo
echo "═══════════════════════════════════════════════════════════"
echo "  liveSpore USB ready"
echo "  Filesystem:    $FS_TYPE"
echo "  Artifact:      $ARTIFACT_NAME"
echo "  Cross-platform: Linux native + WSL2 + Docker"
if [ "$FS_TYPE" = "exfat" ]; then
    echo "  Readable on:   Windows, macOS, Linux (universal)"
else
    echo "  Readable on:   Linux (ext4, native permissions)"
fi
echo "═══════════════════════════════════════════════════════════"
