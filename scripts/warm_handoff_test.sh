#!/usr/bin/env bash
# warm_handoff_test.sh — Full warm handoff pipeline for Titan V VFIO dispatch.
#
# Orchestrates: livepatch → ember → nouveau cycle → VFIO warm open → dispatch test.
#
# Prerequisites:
#   - GPU bound to vfio-pci (or managed by glowplug)
#   - coral-ember built:  cargo build -p coral-ember  (in coralReef/)
#   - coral-glowplug built: cargo build -p coral-glowplug  (for coralctl)
#   - Kernel headers installed (for livepatch build)
#   - nouveau.ko available (modprobe nouveau must work)
#   - Run as root (or with sudo) for livepatch + driver binding
#
# Usage:
#   sudo ./warm_handoff_test.sh <BDF>
#   sudo ./warm_handoff_test.sh 0000:65:00.0
#   sudo CORALREEF_GLOWPLUG_SOCKET=/path/to/socket ./warm_handoff_test.sh 0000:65:00.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CORALREEF_ROOT="$(cd "$REPO_ROOT/../../primals/coralReef" && pwd)"
LIVEPATCH_DIR="$SCRIPT_DIR/livepatch"
LIVEPATCH_KO="$LIVEPATCH_DIR/livepatch_nvkm_mc_reset.ko"

BDF="${1:?Usage: $0 <PCI_BDF>  (e.g. 0000:65:00.0)}"
SETTLE_SECS="${WARM_SETTLE_SECS:-5}"
SOCKET="${CORALREEF_GLOWPLUG_SOCKET:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[warm-handoff]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# --- Phase 0: Preflight ---
log "Phase 0: Preflight checks"

if [[ $EUID -ne 0 ]]; then
    fail "Must run as root (livepatch + driver binding require privileges)"
fi

if [[ ! -d "/sys/bus/pci/devices/$BDF" ]]; then
    fail "PCI device $BDF not found in sysfs"
fi

BOOT0_PATH="/sys/bus/pci/devices/$BDF/resource0"
if [[ -f "$BOOT0_PATH" ]]; then
    ok "PCI device $BDF exists"
else
    warn "resource0 not accessible (device may need VFIO bind first)"
fi

# Build livepatch if not present
if [[ ! -f "$LIVEPATCH_KO" ]]; then
    log "Building livepatch kernel module..."
    MAKE=/usr/bin/make /usr/bin/make -C /lib/modules/"$(uname -r)"/build \
        M="$LIVEPATCH_DIR" modules || fail "Livepatch build failed"
fi
ok "Livepatch module: $LIVEPATCH_KO"

# Check for coralctl
CORALCTL="$CORALREEF_ROOT/target/debug/coralctl"
if [[ ! -x "$CORALCTL" ]]; then
    CORALCTL="$(command -v coralctl 2>/dev/null || true)"
fi
if [[ -z "$CORALCTL" || ! -x "$CORALCTL" ]]; then
    fail "coralctl not found. Build with: cargo build -p coral-glowplug (in coralReef/)"
fi
ok "coralctl: $CORALCTL"

# Determine socket
if [[ -z "$SOCKET" ]]; then
    NS="${BIOMEOS_ECOSYSTEM_NAMESPACE:-biomeos}"
    FAM="${BIOMEOS_FAMILY_ID:-default}"
    XDG="${XDG_RUNTIME_DIR:-/tmp}"
    SOCKET="$XDG/$NS/coral-glowplug-${FAM}.sock"
fi

# Check for ember binary
EMBER_BIN="$CORALREEF_ROOT/target/debug/coral-ember"
if [[ ! -x "$EMBER_BIN" ]]; then
    EMBER_BIN="$(command -v coral-ember 2>/dev/null || true)"
fi
if [[ -z "$EMBER_BIN" || ! -x "$EMBER_BIN" ]]; then
    warn "coral-ember not found — ember must already be running"
else
    ok "coral-ember: $EMBER_BIN"
fi

# --- Phase 1: Livepatch preparation ---
log "Phase 1: Livepatch preparation"

LP_SYSFS="/sys/kernel/livepatch/livepatch_nvkm_mc_reset/enabled"

# CRITICAL: Remove stale livepatch BEFORE loading nouveau.
# If livepatch is loaded targeting nouveau symbols while nouveau is being
# loaded, the kernel module subsystem deadlocks (spinlock in init_module).
if grep -q "livepatch_nvkm_mc_reset" /proc/modules 2>/dev/null; then
    LP_STATE=$(awk '/livepatch_nvkm_mc_reset/{print $5}' /proc/modules)
    if [[ "$LP_STATE" == "Live" ]]; then
        log "Removing stale livepatch from previous session..."
        [[ -f "$LP_SYSFS" ]] && echo 0 > "$LP_SYSFS" 2>/dev/null || true
        sleep 2
        rmmod livepatch_nvkm_mc_reset 2>/dev/null || fail "Cannot remove stale livepatch — reboot required"
        ok "Stale livepatch removed"
    else
        fail "livepatch_nvkm_mc_reset in '$LP_STATE' state — reboot required to clear module deadlock"
    fi
fi

# Livepatch will be loaded AFTER nouveau init (Phase 2 handles this).
# The coralctl warm-fecs flow: swap→nouveau → settle → insmod livepatch → swap→vfio.
if grep -q "^nouveau " /proc/modules 2>/dev/null; then
    NV_STATE=$(awk '/^nouveau /{print $5}' /proc/modules)
    if [[ "$NV_STATE" != "Live" ]]; then
        fail "nouveau module in '$NV_STATE' state — reboot required"
    fi
    log "nouveau already loaded — loading livepatch now"
    insmod "$LIVEPATCH_KO" || fail "insmod failed — nouveau symbols not resolvable"
    sleep 1
    if [[ -f "$LP_SYSFS" ]]; then
        echo 0 > "$LP_SYSFS"
        ok "Livepatch loaded and disabled (ready for warm-fecs cycle)"
    else
        fail "Livepatch loaded but sysfs control not found"
    fi
else
    log "nouveau not loaded — livepatch will be loaded after nouveau init"
    warn "If using coralctl warm-fecs, ensure it loads livepatch after nouveau settle"
fi

# --- Phase 2: Warm FECS cycle via coralctl ---
log "Phase 2: Running coralctl warm-fecs $BDF (settle=${SETTLE_SECS}s)"

SOCKET_ARGS=""
if [[ -S "$SOCKET" ]]; then
    SOCKET_ARGS="--socket $SOCKET"
fi

$CORALCTL $SOCKET_ARGS warm-fecs "$BDF" --settle "$SETTLE_SECS" 2>&1 | while IFS= read -r line; do
    echo "  $line"
done

ok "Warm FECS cycle complete"

# --- Phase 3: Verify falcon state ---
log "Phase 3: Verifying falcon state post-handoff"

# Quick BAR0 check via the VFIO test harness (read-only)
export CORALREEF_VFIO_BDF="$BDF"
export CORALREEF_VFIO_SM=0
export RUST_LOG="${RUST_LOG:-coral_driver=info}"

# --- Phase 4: Run warm dispatch test ---
log "Phase 4: Running vfio_dispatch_warm_handoff test"

cd "$CORALREEF_ROOT"
env -u RUSTUP_TOOLCHAIN cargo test -p coral-driver --features vfio --test hw_nv_vfio \
    vfio_dispatch_warm_handoff -- --ignored --nocapture --test-threads=1 2>&1 | while IFS= read -r line; do
    echo "  $line"
done
TEST_EXIT=${PIPESTATUS[0]}

if [[ $TEST_EXIT -eq 0 ]]; then
    echo ""
    ok "============================================"
    ok "  WARM HANDOFF DISPATCH TEST PASSED"
    ok "  Sovereign compute on Titan V via VFIO!"
    ok "============================================"
else
    echo ""
    warn "============================================"
    warn "  Warm handoff test exited with code $TEST_EXIT"
    warn "  Check FECS state diagnostics above."
    warn "============================================"
fi

# --- Phase 5: Cleanup ---
log "Phase 5: Cleanup"

if [[ -f "$LP_SYSFS" ]]; then
    echo 0 > "$LP_SYSFS" 2>/dev/null || true
    log "Livepatch disabled (can rmmod livepatch_nvkm_mc_reset if desired)"
fi

exit $TEST_EXIT
