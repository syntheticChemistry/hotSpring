#!/bin/sh
# _lib.sh — Shared functions for the hotSpring guideStone artifact.
# Sourced by ./hotspring and backward-compat entry points.
# POSIX sh — no bashisms.

# ── Integrity ────────────────────────────────────────────────────────

integrity_check() {
    if [ ! -f CHECKSUMS ]; then
        return 0
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        if sha256sum -c CHECKSUMS -s 2>/dev/null; then
            return 0
        elif sha256sum -c CHECKSUMS --quiet 2>/dev/null; then
            return 0
        else
            echo "INTEGRITY FAILED — files may be corrupted or tampered with"
            exit 1
        fi
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -c CHECKSUMS --quiet 2>/dev/null || {
            echo "INTEGRITY FAILED — files may be corrupted or tampered with"
            exit 1
        }
    else
        echo "WARNING: no sha256sum or shasum found — skipping integrity check"
    fi
}

# ── OS detection ─────────────────────────────────────────────────────

detect_os() {
    HOST_OS="$(uname -s 2>/dev/null || echo Unknown)"
    case "$HOST_OS" in
        Linux)          OS_TAG="linux" ;;
        Darwin)         OS_TAG="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) OS_TAG="windows-shell" ;;
        *)              OS_TAG="unknown" ;;
    esac
    export HOST_OS OS_TAG
}

# ── Architecture ─────────────────────────────────────────────────────

detect_arch() {
    _RAW_ARCH=$(uname -m)
    case "$_RAW_ARCH" in
        x86_64|amd64)   ARCH_TAG="x86_64"  ;;
        aarch64|arm64)   ARCH_TAG="aarch64" ;;
        *)
            echo "Unsupported architecture: $_RAW_ARCH"
            exit 1
            ;;
    esac
    export ARCH_TAG
}

# ── GPU capability probe ────────────────────────────────────────────

detect_gpu() {
    GPU_MODE=0

    # Only attempt GPU if glibc dynamic linker is present.
    # Static musl binaries cannot dlopen Vulkan.
    _HAS_GLIBC=0
    if [ -f /lib64/ld-linux-x86-64.so.2 ] 2>/dev/null; then
        _HAS_GLIBC=1
    elif [ -f /lib/ld-linux-aarch64.so.1 ] 2>/dev/null; then
        _HAS_GLIBC=1
    fi

    if [ "$_HAS_GLIBC" = "1" ]; then
        if ldconfig -p 2>/dev/null | grep -q libvulkan; then
            GPU_MODE=1
        elif [ -f /usr/lib/x86_64-linux-gnu/libvulkan.so.1 ]; then
            GPU_MODE=1
        elif [ -f /usr/lib64/libvulkan.so.1 ]; then
            GPU_MODE=1
        elif [ -f /usr/lib/libvulkan.so.1 ]; then
            GPU_MODE=1
        elif [ -f /usr/lib/aarch64-linux-gnu/libvulkan.so.1 ]; then
            GPU_MODE=1
        fi
    fi

    if [ "${HOTSPRING_NO_GPU:-0}" = "1" ]; then
        GPU_MODE=0
    fi
    if [ "${HOTSPRING_FORCE_GPU:-0}" = "1" ]; then
        GPU_MODE=1
    fi

    if [ "$GPU_MODE" = "1" ]; then
        MODE_LABEL="GPU (glibc + Vulkan)"
    else
        MODE_LABEL="CPU-only (static musl)"
    fi

    export GPU_MODE MODE_LABEL
}

# ── Binary resolution ───────────────────────────────────────────────
#
# Layout (arch-first):
#   bin/<arch>/gpu/<name>      — glibc, GPU-capable
#   bin/<arch>/static/<name>   — musl, CPU-only
#
# Legacy fallback:
#   bin/gpu/<name>-<arch>      — old gpu layout
#   bin/static/<name>-<arch>   — old static layout
#   bin/<name>-<arch>          — flat layout (symlinks)

resolve_binary() {
    _BIN_NAME="$1"

    # New arch-first paths
    _GPU_PATH="bin/${ARCH_TAG}/gpu/${_BIN_NAME}"
    _STATIC_PATH="bin/${ARCH_TAG}/static/${_BIN_NAME}"

    # Legacy paths (capability-first with arch suffix)
    _GPU_LEGACY="bin/gpu/${_BIN_NAME}-${ARCH_TAG}"
    _STATIC_LEGACY="bin/static/${_BIN_NAME}-${ARCH_TAG}"
    _FLAT_LEGACY="bin/${_BIN_NAME}-${ARCH_TAG}"

    if [ "$GPU_MODE" = "1" ]; then
        if [ -f "$_GPU_PATH" ]; then
            RESOLVED_BIN="$_GPU_PATH"
        elif [ -f "$_GPU_LEGACY" ]; then
            RESOLVED_BIN="$_GPU_LEGACY"
        elif [ -f "$_STATIC_PATH" ]; then
            RESOLVED_BIN="$_STATIC_PATH"
        elif [ -f "$_STATIC_LEGACY" ]; then
            RESOLVED_BIN="$_STATIC_LEGACY"
        elif [ -f "$_FLAT_LEGACY" ]; then
            RESOLVED_BIN="$_FLAT_LEGACY"
        else
            echo "Binary not found: $_BIN_NAME (arch=$ARCH_TAG, gpu=$GPU_MODE)"
            echo "Searched: $_GPU_PATH, $_GPU_LEGACY, $_STATIC_PATH, $_STATIC_LEGACY, $_FLAT_LEGACY"
            exit 1
        fi
    else
        if [ -f "$_STATIC_PATH" ]; then
            RESOLVED_BIN="$_STATIC_PATH"
        elif [ -f "$_STATIC_LEGACY" ]; then
            RESOLVED_BIN="$_STATIC_LEGACY"
        elif [ -f "$_FLAT_LEGACY" ]; then
            RESOLVED_BIN="$_FLAT_LEGACY"
        else
            echo "Binary not found: $_BIN_NAME (arch=$ARCH_TAG, static-only)"
            echo "Searched: $_STATIC_PATH, $_STATIC_LEGACY, $_FLAT_LEGACY"
            exit 1
        fi
    fi

    # Ensure the binary is executable. On ext4/native FS, chmod works
    # in-place. On exFAT/FAT32/NTFS/NFS-noexec, chmod is a no-op so we
    # copy to a tmpdir where permissions are enforced.
    if [ ! -x "$RESOLVED_BIN" ]; then
        chmod +x "$RESOLVED_BIN" 2>/dev/null || true
    fi
    if [ ! -x "$RESOLVED_BIN" ]; then
        _TMP_BIN="$(mktemp "${TMPDIR:-/tmp}/hotspring.XXXXXX")"
        cp "$RESOLVED_BIN" "$_TMP_BIN"
        chmod +x "$_TMP_BIN"
        RESOLVED_BIN="$_TMP_BIN"
    fi

    export RESOLVED_BIN
}

# ── Convenience: resolve and exec in one call ───────────────────────

resolve_and_exec() {
    _BIN_NAME="$1"
    shift
    resolve_binary "$_BIN_NAME"
    exec "$RESOLVED_BIN" "$@"
}

# ── Container dispatch (non-Linux hosts) ─────────────────────────────

CONTAINER_IMAGE="hotspring-guidestone:v0.7.0"
CONTAINER_TAR="container/hotspring-guidestone.tar"

container_available() {
    command -v docker >/dev/null 2>&1 || command -v podman >/dev/null 2>&1
}

container_cmd() {
    if command -v docker >/dev/null 2>&1; then
        echo "docker"
    elif command -v podman >/dev/null 2>&1; then
        echo "podman"
    fi
}

container_ensure_loaded() {
    _CTR="$(container_cmd)"
    [ -z "$_CTR" ] && return 1
    if ! "$_CTR" image inspect "$CONTAINER_IMAGE" >/dev/null 2>&1; then
        if [ -f "$CONTAINER_TAR" ]; then
            echo "  Loading container image from $CONTAINER_TAR..."
            "$_CTR" load -i "$CONTAINER_TAR"
        else
            echo "Container image not found and $CONTAINER_TAR missing."
            echo "Build it with: scripts/build-container.sh"
            return 1
        fi
    fi
}

container_exec() {
    _CTR="$(container_cmd)"
    _GPU_FLAG=""
    if [ "${HOTSPRING_FORCE_GPU:-0}" = "1" ]; then
        _GPU_FLAG="--gpus all"
    fi
    mkdir -p results 2>/dev/null || true
    exec "$_CTR" run --rm \
        $_GPU_FLAG \
        -v "$(pwd)/results:/opt/validation/results" \
        "$CONTAINER_IMAGE" "$@"
}

# ── Self-knowledge: append run to liveSpore.json ────────────────────

update_livespore() {
    _SUBCOMMAND="$1"
    _EXIT_CODE="$2"
    _OUTPUT_DIR="${3:-}"

    SPORE_ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)" || return 0
    LIVESPORE="$SPORE_ROOT/liveSpore.json"
    [ -f "$LIVESPORE" ] || return 0
    command -v python3 >/dev/null 2>&1 || return 0

    python3 -c "
import json, os, socket
from datetime import datetime, timezone
try:
    with open('$LIVESPORE') as f:
        m = json.load(f)
    entry = {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'hostname': socket.gethostname(),
        'command': '$_SUBCOMMAND',
        'mode': '$MODE_LABEL',
        'binary': '${RESOLVED_BIN:-unknown}',
        'exit_code': $_EXIT_CODE,
    }
    output_dir = '$_OUTPUT_DIR'
    if output_dir:
        for name in ['validate_chuna.json', 'benchmark_summary.json']:
            p = os.path.join(output_dir, name)
            if os.path.isfile(p):
                with open(p) as rf:
                    rd = json.load(rf)
                s = rd.get('summary', rd)
                entry['checks_passed'] = s.get('passed', 0)
                entry['checks_total'] = s.get('total', 0)
                entry['duration_ms'] = s.get('duration_ms', 0)
                break
    if 'runs' not in m:
        m['runs'] = []
    m['runs'].append(entry)
    hosts = set(m.get('systems_validated', []))
    hosts.add(socket.gethostname())
    m['systems_validated'] = sorted(hosts)
    with open('$LIVESPORE', 'w') as f:
        json.dump(m, f, indent=2)
except Exception:
    pass
" 2>/dev/null || true
}
