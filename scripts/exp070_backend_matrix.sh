#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Experiment 070: Full Backend Matrix — Sovereign Reverse Engineering
#
# Drives Phases 1–7 using GlowPlug lend/reclaim pattern (ember-safe).
# NEVER stops the GlowPlug daemon — uses lend/reclaim for register dumps
# and swap for driver personality changes.
#
# Requires: socat, cargo (for building), both Titans on vfio at boot,
#           coral-ember running (for safe fd keepalive).
#
# Usage: sudo ./scripts/exp070_backend_matrix.sh [--phase N] [--quick]

set -euo pipefail

ORACLE="0000:03:00.0"
TARGET="0000:4a:00.0"
SOCK="/run/coralreef/glowplug.sock"
EMBER_SOCK="/run/coralreef/ember.sock"
DATA="data/070"
BIN_DIR="target/release"

PHASE="${1:-all}"
QUICK="${2:-}"

cd "$(dirname "$0")/.."
cd barracuda

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Experiment 070: Dual-Titan Full Backend Matrix                 ║"
echo "║  Oracle: $ORACLE    Target: $TARGET                             ║"
echo "║  Pattern: lend/reclaim (ember-safe)                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

mkdir -p "../$DATA"

# ─── Helpers ──────────────────────────────────────────────────────────
glowplug_rpc() {
    local method="$1" params="${2:-{}}"
    echo "{\"jsonrpc\":\"2.0\",\"method\":\"$method\",\"params\":$params,\"id\":1}" \
        | socat -t5 - UNIX-CONNECT:"$SOCK"
}

glowplug_swap() {
    local bdf="$1" target="$2"
    echo ">>> GlowPlug: swapping $bdf → $target"
    glowplug_rpc "device.swap" "{\"bdf\":\"$bdf\",\"target\":\"$target\"}"
    echo ""
}

glowplug_lend() {
    local bdf="$1"
    echo ">>> GlowPlug: lending $bdf"
    glowplug_rpc "device.lend" "{\"bdf\":\"$bdf\"}"
    echo ""
}

glowplug_reclaim() {
    local bdf="$1"
    echo ">>> GlowPlug: reclaiming $bdf"
    glowplug_rpc "device.reclaim" "{\"bdf\":\"$bdf\"}"
    echo ""
}

glowplug_list() {
    echo ">>> GlowPlug: listing devices"
    glowplug_rpc "device.list"
    echo ""
}

ember_status() {
    echo ">>> Ember: status"
    echo '{"request":"status"}' | socat -t5 - UNIX-CONNECT:"$EMBER_SOCK" 2>/dev/null || echo "  (ember not reachable)"
    echo ""
}

wait_for_driver() {
    local bdf="$1" driver="$2" secs="${3:-5}"
    echo ">>> Waiting ${secs}s for $driver init on $bdf..."
    sleep "$secs"
}

build_tools() {
    echo ">>> Building experiment tools..."
    cargo build --release --bin exp070_register_dump --bin exp070_register_diff 2>&1 | tail -5
}

dump_registers() {
    local bdf="$1" label="$2"
    local outfile="../$DATA/${label}.json"
    echo ">>> Dumping registers: $bdf → $outfile"
    "$BIN_DIR/exp070_register_dump" "$bdf" "$outfile"
}

diff_registers() {
    local baseline="$1" warm="$2" label="$3"
    local outfile="../$DATA/diff_${label}.json"
    echo ">>> Diffing: $baseline vs $warm → $outfile"
    "$BIN_DIR/exp070_register_diff" "../$DATA/${baseline}.json" "../$DATA/${warm}.json" "$outfile"
}

run_validation() {
    local label="$1"
    echo ">>> Running validation suite ($label)..."
    local logfile="../$DATA/validation_${label}.txt"
    {
        echo "=== Validation: $label ==="
        echo "=== validate_linalg ===" && "$BIN_DIR/validate_linalg" 2>&1 | tail -5
        echo "=== validate_special_functions ===" && "$BIN_DIR/validate_special_functions" 2>&1 | tail -5
        echo "=== validate_spectral ===" && "$BIN_DIR/validate_spectral" 2>&1 | tail -5
        echo "=== validate_barracuda_evolution ===" && "$BIN_DIR/validate_barracuda_evolution" 2>&1 | tail -5
    } | tee "$logfile"
    echo ">>> Validation log: $logfile"
}

# ─── Pre-flight ──────────────────────────────────────────────────────
build_tools
echo ""
ember_status
glowplug_list

# ─── Phase 1: Cold Baseline ─────────────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "1" ]; then
    echo ""
    echo "═══ Phase 1: Cold Baseline (Config A — both vfio) ═══"
    echo "--- Register dumps via sysfs resource0 (no lend/reclaim needed) ---"

    dump_registers "$ORACLE" "cold_oracle"
    dump_registers "$TARGET" "cold_target"

    echo "Phase 1 complete."
fi

# ─── Phase 2: nouveau Warm Oracle ───────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "2" ]; then
    echo ""
    echo "═══ Phase 2: nouveau Warm Oracle (Config B) ═══"
    glowplug_swap "$ORACLE" "nouveau"
    wait_for_driver "$ORACLE" "nouveau" 6

    # Dump registers while on nouveau (warm state)
    dump_registers "$ORACLE" "nouveau_warm_oracle"

    # Swap back to vfio and dump post-rebind state
    glowplug_swap "$ORACLE" "vfio"
    sleep 2
    dump_registers "$ORACLE" "nouveau_rebind_vfio_oracle"

    diff_registers "cold_oracle" "nouveau_warm_oracle" "nouveau_vs_cold_oracle"
    echo "Phase 2 complete."
fi

# ─── Phase 3: nvidia Warm Oracle ────────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "3" ]; then
    echo ""
    echo "═══ Phase 3: nvidia Warm Oracle (Config C) ═══"
    glowplug_swap "$ORACLE" "nvidia"
    wait_for_driver "$ORACLE" "nvidia" 4

    dump_registers "$ORACLE" "nvidia_warm_oracle"

    glowplug_swap "$ORACLE" "vfio"
    sleep 2
    dump_registers "$ORACLE" "nvidia_rebind_vfio_oracle"

    diff_registers "cold_oracle" "nvidia_warm_oracle" "nvidia_vs_cold_oracle"
    if [ -f "../$DATA/nouveau_warm_oracle.json" ]; then
        diff_registers "nouveau_warm_oracle" "nvidia_warm_oracle" "nvidia_vs_nouveau_oracle"
    fi
    echo "Phase 3 complete."
fi

# ─── Phase 4: Warm → Rebind → State Survival ────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "4" ]; then
    echo ""
    echo "═══ Phase 4: Warm → Rebind → Dispatch (KEY EXPERIMENT) ═══"

    echo "--- 4a: nouveau warm then rebind ---"
    glowplug_swap "$TARGET" "nouveau"
    wait_for_driver "$TARGET" "nouveau" 6

    dump_registers "$TARGET" "nouveau_warm_target"

    glowplug_swap "$TARGET" "vfio"
    sleep 2

    dump_registers "$TARGET" "nouveau_rebind_vfio_target"

    diff_registers "nouveau_warm_target" "nouveau_rebind_vfio_target" "nouveau_rebind_survival"
    diff_registers "cold_target" "nouveau_rebind_vfio_target" "nouveau_rebind_vs_cold"

    echo "--- 4b: nvidia warm then rebind ---"
    glowplug_swap "$TARGET" "nvidia"
    wait_for_driver "$TARGET" "nvidia" 4

    dump_registers "$TARGET" "nvidia_warm_target"

    glowplug_swap "$TARGET" "vfio"
    sleep 2

    dump_registers "$TARGET" "nvidia_rebind_vfio_target"

    diff_registers "nvidia_warm_target" "nvidia_rebind_vfio_target" "nvidia_rebind_survival"
    diff_registers "cold_target" "nvidia_rebind_vfio_target" "nvidia_rebind_vs_cold"

    echo "Phase 4 complete."
fi

# ─── Phase 5: NVK Compute Validation (Config D) ─────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "5" ]; then
    echo ""
    echo "═══ Phase 5: NVK Compute Validation (Config D — target on nouveau) ═══"
    glowplug_swap "$TARGET" "nouveau"
    wait_for_driver "$TARGET" "nouveau" 6
    run_validation "nvk_titan_v"
    glowplug_swap "$TARGET" "vfio"
    sleep 2
    echo "Phase 5 complete."
fi

# ─── Phase 6: nvidia Vulkan Compute Validation (Config E) ───────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "6" ]; then
    echo ""
    echo "═══ Phase 6: nvidia Vulkan Compute Validation (Config E) ═══"
    glowplug_swap "$TARGET" "nvidia"
    wait_for_driver "$TARGET" "nvidia" 4
    run_validation "nvidia_titan_v"
    glowplug_swap "$TARGET" "vfio"
    sleep 2
    echo "Phase 6 complete."
fi

# ─── Phase 7: Analysis ──────────────────────────────────────────────
if [ "$PHASE" = "all" ] || [ "$PHASE" = "7" ]; then
    echo ""
    echo "═══ Phase 7: Firmware Diff Analysis ═══"
    echo "Diff files generated in $DATA/:"
    ls -la "../$DATA"/diff_*.json 2>/dev/null || echo "  (no diffs yet — run phases 2-4 first)"
    echo ""
    echo "Key comparisons:"
    echo "  diff_nouveau_vs_cold_oracle.json     — what nouveau initializes"
    echo "  diff_nvidia_vs_cold_oracle.json      — what nvidia initializes"
    echo "  diff_nvidia_vs_nouveau_oracle.json   — driver-specific differences"
    echo "  diff_nouveau_rebind_survival.json    — what survives nouveau→vfio"
    echo "  diff_nvidia_rebind_survival.json     — what survives nvidia→vfio"
    echo "  diff_nouveau_rebind_vs_cold.json     — rebind state vs cold (residual warmth)"
    echo "  diff_nvidia_rebind_vs_cold.json      — rebind state vs cold (residual warmth)"
    echo ""
    echo "Next: analyze diffs to build FIRMWARE_INIT_MAP.md"
    echo "Phase 7 complete."
fi

echo ""
echo "═══ Experiment 070 session complete ═══"
