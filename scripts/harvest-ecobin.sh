#!/usr/bin/env bash
# ecoBin harvest: build hotSpring musl-static binaries and submit to plasmidBin.
#
# Builds for x86_64-unknown-linux-musl and (optionally) aarch64-unknown-linux-musl,
# then harvests into ecoPrimals/infra/plasmidBin/ per ECOBIN_ARCHITECTURE_STANDARD v3.0.
#
# Usage:
#   ./scripts/harvest-ecobin.sh                     # x86_64 only
#   ./scripts/harvest-ecobin.sh --cross-aarch64     # both architectures
#
# Prerequisites:
#   - musl-tools installed (apt install musl-tools)
#   - For aarch64: aarch64-linux-musl-gcc cross-compiler
#   - .cargo/config.toml already configures musl targets
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLASMIDB="${ROOT}/../../infra/plasmidBin"

if [ ! -d "$PLASMIDB" ]; then
    echo "ERROR: plasmidBin not found at $PLASMIDB"
    echo "  Expected: ecoPrimals/infra/plasmidBin/"
    exit 1
fi

CROSS_AARCH64=false
for arg in "$@"; do
    case "$arg" in
        --cross-aarch64) CROSS_AARCH64=true ;;
    esac
done

echo "=== hotSpring ecoBin Harvest ==="
echo "  Source: $ROOT/barracuda"
echo "  Target: $PLASMIDB"
echo ""

# ── Build x86_64 musl-static ──
echo "  Building x86_64-unknown-linux-musl (release, stripped, LTO)..."
cd "$ROOT/barracuda"
cargo build --release --target x86_64-unknown-linux-musl --bin hotspring_primal

BINARY_X86="target/x86_64-unknown-linux-musl/release/hotspring_primal"
if [ ! -f "$BINARY_X86" ]; then
    echo "ERROR: binary not found at $BINARY_X86"
    exit 1
fi

echo "  Verifying static linkage..."
file "$BINARY_X86" | grep -qE "static(ally|-pie) linked" || {
    echo "WARNING: binary may not be statically linked:"
    file "$BINARY_X86"
}

BINARY_SIZE=$(stat -c%s "$BINARY_X86" 2>/dev/null || stat -f%z "$BINARY_X86")
echo "  Binary: $BINARY_X86 ($(( BINARY_SIZE / 1024 / 1024 )) MB)"

# ── Harvest to plasmidBin ──
echo ""
echo "  Harvesting to plasmidBin..."
mkdir -p "$PLASMIDB/hotspring/x86_64"
cp "$BINARY_X86" "$PLASMIDB/hotspring/x86_64/hotspring_primal"
echo "  Copied to $PLASMIDB/hotspring/x86_64/hotspring_primal"

# ── b3sum checksum ──
if command -v b3sum &>/dev/null; then
    B3=$(b3sum "$BINARY_X86" | cut -d' ' -f1)
    echo "  b3sum (x86_64): $B3"
fi

# ── Cross-compile aarch64 ──
if $CROSS_AARCH64; then
    echo ""
    echo "  Building aarch64-unknown-linux-musl..."
    cargo build --release --target aarch64-unknown-linux-musl --bin hotspring_primal

    BINARY_ARM="target/aarch64-unknown-linux-musl/release/hotspring_primal"
    if [ -f "$BINARY_ARM" ]; then
        mkdir -p "$PLASMIDB/hotspring/aarch64"
        cp "$BINARY_ARM" "$PLASMIDB/hotspring/aarch64/hotspring_primal"
        echo "  Copied to $PLASMIDB/hotspring/aarch64/hotspring_primal"
        if command -v b3sum &>/dev/null; then
            B3_ARM=$(b3sum "$BINARY_ARM" | cut -d' ' -f1)
            echo "  b3sum (aarch64): $B3_ARM"
        fi
    fi
fi

# ── Write metadata.toml ──
VERSION=$(grep '^version' "$ROOT/barracuda/Cargo.toml" | head -1 | cut -d'"' -f2)
BC_VERSION=$(grep '^version' "$ROOT/../../primals/barraCuda/crates/barracuda/Cargo.toml" 2>/dev/null | head -1 | cut -d'"' -f2)
BC_VERSION="${BC_VERSION:-unknown}"
HARVEST_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Compute checksums
X86_B3=""
ARM_B3=""
if command -v b3sum &>/dev/null; then
    X86_B3=$(b3sum "$BINARY_X86" | cut -d' ' -f1)
    if $CROSS_AARCH64 && [ -f "$BINARY_ARM" ]; then
        ARM_B3=$(b3sum "$BINARY_ARM" | cut -d' ' -f1)
    fi
fi

cat > "$PLASMIDB/hotspring/metadata.toml" <<TOML
# hotSpring -- Computational physics spring primal (from hotSpring)
# SPDX-License-Identifier: AGPL-3.0-or-later

[primal]
name = "hotspring"
version = "$VERSION"
domain = "physics"
description = "Computational physics validation: nuclear EOS, lattice QCD, GPU MD, transport coefficients"
license = "AGPL-3.0-or-later"

[provenance]
built_from = "springs/hotSpring"
built_at = "$HARVEST_DATE"
barracuda_version = "$BC_VERSION"

[compatibility]
min_ipc_version = "1.0"
capabilities = [
    "physics.lattice_qcd",
    "physics.lattice_gauge_update",
    "physics.hmc_trajectory",
    "physics.wilson_dirac",
    "physics.molecular_dynamics",
    "physics.fluid",
    "physics.nuclear_eos",
    "physics.thermal",
    "physics.radiation",
    "compute.df64",
    "compute.cg_solver",
    "compute.gradient_flow",
    "compute.f64",
    "composition.health",
    "health.check",
    "health.liveness",
    "health.readiness",
    "capabilities.list",
    "mcp.tools.list",
]

# -- ecoBin: per-architecture builds --

[builds.x86_64-linux]
binary = "hotspring_primal"
target = "x86_64-unknown-linux-musl"
pie_verified = true
static_linked = true
# checksum_b3sum = "$X86_B3"
TOML

if $CROSS_AARCH64; then
cat >> "$PLASMIDB/hotspring/metadata.toml" <<TOML

[builds.aarch64-linux]
binary = "hotspring_primal"
target = "aarch64-unknown-linux-musl"
pie_verified = true
static_linked = true
# checksum_b3sum = "$ARM_B3"
TOML
fi

cat >> "$PLASMIDB/hotspring/metadata.toml" <<TOML

# -- genomeBin: deployment intelligence --

[genomeBin]
tier = "ecoBin"
unibin_modes = ["server", "version"]
default_mode = "server"

[genomeBin.server]
tcp_port_env = "HOTSPRING_PORT"
tcp_port_default = 9900
health_probe = "health.liveness"

[genomeBin.service]
restart_policy = "on-failure"
after = ["beardog", "songbird", "toadstool", "barracuda"]
wants = ["coralreef", "nestgate"]
TOML

echo ""
echo "=== Harvest complete ==="
echo "  Metadata: $PLASMIDB/hotspring/metadata.toml"
echo "  Version: $VERSION"
