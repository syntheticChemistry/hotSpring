#!/bin/sh
set -eu

# Build the hotSpring guideStone OCI container image and export a
# portable tarball for offline deployment (USB, scp, air-gapped).
#
# Produces:
#   validation/container/hotspring-guidestone.tar   — docker load archive
#   validation/container/docker-run.sh              — convenience launcher
#
# Usage:
#   ./scripts/build-container.sh               # build + export
#   ./scripts/build-container.sh --no-export   # build only (no tarball)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VALIDATION="$ROOT/validation"

IMAGE_NAME="hotspring-guidestone"
IMAGE_TAG="v0.7.0"
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"

NO_EXPORT=0
for arg in "$@"; do
    case "$arg" in
        --no-export) NO_EXPORT=1 ;;
    esac
done

CTR=""
if command -v docker >/dev/null 2>&1; then
    CTR="docker"
elif command -v podman >/dev/null 2>&1; then
    CTR="podman"
else
    echo "ERROR: docker or podman required"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  hotSpring Container Builder"
echo "  Image:  $IMAGE_REF"
echo "  Engine: $CTR"
echo "═══════════════════════════════════════════════════════════"
echo

echo "▸ Building image..."
cd "$ROOT"
$CTR build -t "$IMAGE_REF" -t "${IMAGE_NAME}:latest" .
echo "  Image built: $IMAGE_REF"
echo

if [ "$NO_EXPORT" = "0" ]; then
    echo "▸ Exporting tarball..."
    mkdir -p "$VALIDATION/container"
    $CTR save "$IMAGE_REF" -o "$VALIDATION/container/hotspring-guidestone.tar"
    SIZE=$(du -h "$VALIDATION/container/hotspring-guidestone.tar" | cut -f1)
    echo "  Exported: container/hotspring-guidestone.tar ($SIZE)"
    echo

    echo "▸ Generating docker-run.sh..."
    cat > "$VALIDATION/container/docker-run.sh" <<'LAUNCHER'
#!/bin/sh
set -eu

# Convenience launcher for the hotSpring guideStone container.
# Run from the artifact directory (where this file lives).

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTIFACT_DIR="$(cd "$SELF_DIR/.." && pwd)"

IMAGE="hotspring-guidestone:v0.7.0"
TAR="$SELF_DIR/hotspring-guidestone.tar"

CTR=""
if command -v docker >/dev/null 2>&1; then
    CTR="docker"
elif command -v podman >/dev/null 2>&1; then
    CTR="podman"
else
    echo "ERROR: docker or podman required"
    echo "Install: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! "$CTR" image inspect "$IMAGE" >/dev/null 2>&1; then
    if [ -f "$TAR" ]; then
        echo "Loading image from $TAR..."
        "$CTR" load -i "$TAR"
    else
        echo "Image $IMAGE not found and tarball missing."
        exit 1
    fi
fi

GPU_FLAG=""
if [ "${HOTSPRING_FORCE_GPU:-0}" = "1" ]; then
    GPU_FLAG="--gpus all"
fi

CMD="${1:-validate}"

mkdir -p "$ARTIFACT_DIR/results" 2>/dev/null || true

exec "$CTR" run --rm \
    $GPU_FLAG \
    -v "$ARTIFACT_DIR/results:/opt/validation/results" \
    "$IMAGE" "$CMD" "$@"
LAUNCHER

    chmod +x "$VALIDATION/container/docker-run.sh"
    echo "  Generated: container/docker-run.sh"
fi

echo
echo "═══════════════════════════════════════════════════════════"
echo "  Container ready"
echo
echo "  Run:  docker run --rm -v ./results:/opt/validation/results $IMAGE_REF validate"
echo "  GPU:  docker run --rm --gpus all -v ./results:/opt/validation/results $IMAGE_REF validate"
echo "  Load: docker load < validation/container/hotspring-guidestone.tar"
echo "═══════════════════════════════════════════════════════════"
