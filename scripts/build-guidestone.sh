#!/bin/sh
set -eu

# Build the hotSpring guideStone artifact — arch-first, dual-capability layout.
#
# Produces (per arch):
#   validation/bin/<arch>/static/<name>   — musl, CPU-only
#   validation/bin/<arch>/gpu/<name>      — glibc, GPU-capable (host arch only)
#
# Legacy compat (symlinks):
#   validation/bin/static/<name>-<arch>   → ../<arch>/static/<name>
#   validation/bin/gpu/<name>-<arch>      → ../<arch>/gpu/<name>
#   validation/bin/<name>-<arch>          → <arch>/static/<name>
#
# Requirements:
#   rustup target add x86_64-unknown-linux-musl
#   rustup target add aarch64-unknown-linux-musl  (for cross-compile)
#
# Usage:
#   ./build-guidestone.sh                  # x86_64 static + GPU + container
#   ./build-guidestone.sh --cross          # + aarch64 static cross-compile
#   ./build-guidestone.sh --static-only    # skip GPU build
#   ./build-guidestone.sh --no-container   # skip Docker image build

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BARRACUDA="$ROOT/barracuda"
VALIDATION="$ROOT/validation"
HOST_ARCH="x86_64"

CROSS=0
STATIC_ONLY=0
NO_CONTAINER=0
for arg in "$@"; do
    case "$arg" in
        --cross) CROSS=1 ;;
        --static-only) STATIC_ONLY=1 ;;
        --no-container) NO_CONTAINER=1 ;;
    esac
done

CORE_BINS="validate_chuna chuna_generate chuna_flow chuna_measure chuna_convert chuna_benchmark_flow chuna_matrix validation_matrix"
CORE_ARTIFACTS="validate chuna-generate chuna-flow chuna-measure chuna-convert chuna-benchmark-flow chuna-matrix validation-matrix"

BENCH_BINS="bench_gpu_fp64 bench_gpu_hmc bench_precision_tiers"
BENCH_ARTIFACTS="bench-gpu-fp64 bench-gpu-hmc bench-precision-tiers"

STATIC_TARGETS="x86_64-unknown-linux-musl"
if [ "$CROSS" = "1" ]; then
    STATIC_TARGETS="$STATIC_TARGETS aarch64-unknown-linux-musl"
fi

echo "═══════════════════════════════════════════════════════════"
echo "  hotSpring guideStone Artifact Builder"
echo "  Layout: bin/<arch>/{static,gpu}/<name>  (arch-first)"
echo "  Static: $STATIC_TARGETS"
if [ "$STATIC_ONLY" = "0" ]; then
    echo "  GPU:    host ($HOST_ARCH, glibc + Vulkan dlopen)"
fi
echo "═══════════════════════════════════════════════════════════"
echo

# ── Helper: install binaries into target directory ──

install_bins() {
    _SRC_DIR="$1"
    _DST_DIR="$2"
    _BIN_LIST="$3"
    _ARTIFACT_LIST="$4"

    mkdir -p "$_DST_DIR"

    set -- $_BIN_LIST
    for artifact_name in $_ARTIFACT_LIST; do
        cargo_bin="$1"
        shift
        src="$_SRC_DIR/$cargo_bin"
        dst="$_DST_DIR/$artifact_name"
        if [ ! -f "$src" ]; then
            echo "  WARNING: $src not found, skipping"
            continue
        fi
        cp "$src" "$dst"
        strip "$dst" 2>/dev/null || true
        chmod +x "$dst"
        size=$(du -h "$dst" | cut -f1)
        echo "  $(basename "$(dirname "$_DST_DIR")")/$(basename "$_DST_DIR")/$artifact_name  ($size)"
    done
}

# ── Phase 1: Static musl binaries (per arch) ──

STATIC_COUNT=0

for MUSL_TARGET in $STATIC_TARGETS; do
    case "$MUSL_TARGET" in
        x86_64*)  ARCH="x86_64" ;;
        aarch64*) ARCH="aarch64" ;;
    esac

    echo "▸ Phase 1: Static musl [$ARCH] ($MUSL_TARGET)"
    echo

    echo "  Building..."
    cd "$BARRACUDA"
    bin_list=""
    for bin in $CORE_BINS; do
        bin_list="$bin_list --bin $bin"
    done
    cargo build --release --target "$MUSL_TARGET" $bin_list
    N_CORE=$(echo $CORE_BINS | wc -w | tr -d ' ')
    echo "  $N_CORE static binaries compiled"

    echo "  Installing to bin/$ARCH/static/..."
    install_bins \
        "$BARRACUDA/target/$MUSL_TARGET/release" \
        "$VALIDATION/bin/$ARCH/static" \
        "$CORE_BINS" \
        "$CORE_ARTIFACTS"

    STATIC_COUNT=$((STATIC_COUNT + N_CORE))
    echo
done

# ── Phase 2: GPU-capable glibc binaries (host arch only) ──

GPU_COUNT=0

if [ "$STATIC_ONLY" = "0" ]; then
    echo "▸ Phase 2: GPU glibc [$HOST_ARCH] (host target)"
    echo

    echo "  Building core + bench..."
    cd "$BARRACUDA"
    gpu_bin_list=""
    for bin in $CORE_BINS $BENCH_BINS; do
        gpu_bin_list="$gpu_bin_list --bin $bin"
    done
    cargo build --release $gpu_bin_list
    N_GPU=$(echo $CORE_BINS $BENCH_BINS | wc -w | tr -d ' ')
    echo "  $N_GPU GPU binaries compiled"

    echo "  Installing core to bin/$HOST_ARCH/gpu/..."
    install_bins \
        "$BARRACUDA/target/release" \
        "$VALIDATION/bin/$HOST_ARCH/gpu" \
        "$CORE_BINS" \
        "$CORE_ARTIFACTS"

    echo "  Installing bench to bin/$HOST_ARCH/gpu/..."
    install_bins \
        "$BARRACUDA/target/release" \
        "$VALIDATION/bin/$HOST_ARCH/gpu" \
        "$BENCH_BINS" \
        "$BENCH_ARTIFACTS"

    GPU_COUNT=$N_GPU
    echo
fi

# ── Phase 3: Legacy symlinks ──

echo "▸ Phase 3: Legacy symlinks..."

# Legacy: bin/static/<name>-<arch> -> ../<arch>/static/<name>
mkdir -p "$VALIDATION/bin/static" "$VALIDATION/bin/gpu"

for MUSL_TARGET in $STATIC_TARGETS; do
    case "$MUSL_TARGET" in
        x86_64*)  ARCH="x86_64" ;;
        aarch64*) ARCH="aarch64" ;;
    esac

    for artifact_name in $CORE_ARTIFACTS; do
        # bin/static/<name>-<arch> -> ../<arch>/static/<name>
        ln -sf "../${ARCH}/static/${artifact_name}" \
            "$VALIDATION/bin/static/${artifact_name}-${ARCH}"
        # bin/<name>-<arch> -> <arch>/static/<name>  (flat compat)
        ln -sf "${ARCH}/static/${artifact_name}" \
            "$VALIDATION/bin/${artifact_name}-${ARCH}"
    done
done

if [ "$STATIC_ONLY" = "0" ]; then
    for artifact_name in $CORE_ARTIFACTS $BENCH_ARTIFACTS; do
        # bin/gpu/<name>-<arch> -> ../<arch>/gpu/<name>
        ln -sf "../${HOST_ARCH}/gpu/${artifact_name}" \
            "$VALIDATION/bin/gpu/${artifact_name}-${HOST_ARCH}"
    done
fi

echo "  Legacy symlinks created"

# ── Phase 4: Reference data ──

echo
echo "▸ Phase 4: Reference data..."
mkdir -p "$VALIDATION/expected"
if [ -f "$VALIDATION/results/validate_chuna.json" ]; then
    cp "$VALIDATION/results/validate_chuna.json" "$VALIDATION/expected/validate_chuna_reference.json"
    echo "  Copied validate_chuna.json -> expected/"
else
    echo "  No results/validate_chuna.json — run ./hotspring validate first"
fi

# ── Phase 5: Checksums ──

echo
echo "▸ Phase 5: Generating CHECKSUMS..."
cd "$VALIDATION"

: > CHECKSUMS.tmp

# Arch-first binaries
for MUSL_TARGET in $STATIC_TARGETS; do
    case "$MUSL_TARGET" in
        x86_64*)  ARCH="x86_64" ;;
        aarch64*) ARCH="aarch64" ;;
    esac
    for artifact_name in $CORE_ARTIFACTS; do
        f="bin/$ARCH/static/$artifact_name"
        [ -f "$f" ] && sha256sum "$f" >> CHECKSUMS.tmp
    done
done

if [ "$STATIC_ONLY" = "0" ]; then
    for artifact_name in $CORE_ARTIFACTS $BENCH_ARTIFACTS; do
        f="bin/$HOST_ARCH/gpu/$artifact_name"
        [ -f "$f" ] && sha256sum "$f" >> CHECKSUMS.tmp
    done
fi

# Scripts, cross-OS launchers, and docs
for f in \
    README LICENSE GUIDESTONE.md \
    hotspring _lib.sh hotspring.bat \
    run run-matrix benchmark chuna-engine deploy-nucleus run-overnight \
    expected/validate_chuna_reference.json \
    ; do
    [ -f "$f" ] && sha256sum "$f" >> CHECKSUMS.tmp
done

# Container tarball
if [ -f "container/hotspring-guidestone.tar" ]; then
    sha256sum "container/hotspring-guidestone.tar" >> CHECKSUMS.tmp
fi
if [ -f "container/docker-run.sh" ]; then
    sha256sum "container/docker-run.sh" >> CHECKSUMS.tmp
fi

mv CHECKSUMS.tmp CHECKSUMS
entries=$(wc -l < CHECKSUMS)
echo "  CHECKSUMS: $entries entries"

# ── Phase 6: Container image ──

if [ "$NO_CONTAINER" = "0" ]; then
    if command -v docker >/dev/null 2>&1 || command -v podman >/dev/null 2>&1; then
        echo
        echo "▸ Phase 6: Container image..."
        "$SCRIPT_DIR/build-container.sh"
        echo

        # Re-checksum the container artifacts
        cd "$VALIDATION"
        if [ -f "container/hotspring-guidestone.tar" ]; then
            grep -v "container/" CHECKSUMS > CHECKSUMS.tmp 2>/dev/null || true
            sha256sum "container/hotspring-guidestone.tar" >> CHECKSUMS.tmp
            [ -f "container/docker-run.sh" ] && sha256sum "container/docker-run.sh" >> CHECKSUMS.tmp
            mv CHECKSUMS.tmp CHECKSUMS
            echo "  CHECKSUMS updated with container artifacts"
        fi
    else
        echo
        echo "▸ Phase 6: Container image... SKIP (no docker/podman)"
    fi
else
    echo
    echo "▸ Phase 6: Container image... SKIP (--no-container)"
fi

# ── Phase 7: Cross-OS launchers ──

echo
echo "▸ Phase 7: Cross-OS launchers..."
cd "$VALIDATION"

# hotspring.bat is maintained in validation/ directly — verify it exists
if [ -f "hotspring.bat" ]; then
    echo "  hotspring.bat: present"
else
    echo "  WARNING: hotspring.bat not found in validation/"
fi

# ── Summary ──

echo
echo "═══════════════════════════════════════════════════════════"
echo "  Artifact ready: $VALIDATION/"
echo
echo "  Arch-first layout:"
for MUSL_TARGET in $STATIC_TARGETS; do
    case "$MUSL_TARGET" in
        x86_64*)  ARCH="x86_64" ;;
        aarch64*) ARCH="aarch64" ;;
    esac
    N=$(echo $CORE_ARTIFACTS | wc -w | tr -d ' ')
    echo "    bin/$ARCH/static/  $N musl binaries (CPU, universal)"
done
if [ "$STATIC_ONLY" = "0" ]; then
    N=$(echo $CORE_ARTIFACTS $BENCH_ARTIFACTS | wc -w | tr -d ' ')
    echo "    bin/$HOST_ARCH/gpu/    $N glibc binaries (GPU via Vulkan)"
fi
echo
echo "  Entry point: ./hotspring <command>"
echo "  Backward compat: ./run, ./benchmark, ./chuna-engine, ./deploy-nucleus"
echo "  Cross-OS: hotspring.bat (Windows), auto-Docker (macOS)"
if [ -f "$VALIDATION/container/hotspring-guidestone.tar" ]; then
    CSIZE=$(du -h "$VALIDATION/container/hotspring-guidestone.tar" | cut -f1)
    echo "  Container: hotspring-guidestone.tar ($CSIZE)"
fi
echo
echo "  Total: $STATIC_COUNT static + $GPU_COUNT GPU binaries"
echo "  Name:  hotSpring-guideStone-v0.7.0"
echo "═══════════════════════════════════════════════════════════"
