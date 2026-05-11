#!/bin/bash
# DEPRECATED — replaced by coral-ember's built-in PCIe keepalive thread
# (crates/coral-ember/src/pcie_keepalive.rs) and [[pcie_switch]] config in
# glowplug.toml.  Kept in repo as reference documentation for the PLX
# keepalive protocol.  Do NOT enable plx-keepalive.service — use
# coral-ember.service instead.
#
# plx-keepalive.sh — Keep PLX PEX 8747 PCIe switch alive by generating periodic
# PCIe traffic so the BIOS/ACPI idle power-gating never triggers D3cold.
#
# Root cause: BIOS autonomously gates the PLX into D3cold after ~2h of idle
# (sometimes as short as ~11min), bypassing kernel d3cold_allowed=0. Generating
# both config-space and MMIO TLPs at 3-second intervals keeps the LTSSM in L0
# and prevents the BIOS idle timer from firing.
#
# Also forces ASPM L1 disabled on PLX and downstream ports at startup to prevent
# PCIe link state transitions that can cascade into BIOS power-gating.

PLX="0000:49:00.0"
PLX_DP1="0000:4a:08.0"
PLX_DP2="0000:4a:10.0"
K80_DIE0="0000:4b:00.0"
K80_DIE1="0000:4c:00.0"
PLX_RES="/sys/bus/pci/devices/${PLX}/resource0"
INTERVAL=3   # seconds between keepalive pings (was 8 — too long for short idle timers)

log() { logger -t plx-keepalive "$*"; }
warn() { log "WARN: $*"; }

# ── Disable ASPM L0s/L1 on PLX subtree ───────────────────────────────────────
# ASPM L1 transitions can trigger BIOS idle detection. Disable once at startup.
# LNKCTL bits[1:0]: 00=disabled, 01=L0s only, 10=L1 only, 11=both
disable_aspm() {
    local dev="$1"
    # Find PCIe capability offset
    local cap_off
    cap_off=$(setpci -s "$dev" CAP_PTR.B 2>/dev/null) || return
    # Walk capability list to find PCIe cap (id=0x10)
    local off=$((16#${cap_off}))
    for _i in $(seq 0 16); do
        local cap_id
        cap_id=$(setpci -s "$dev" "0x${cap_off}.B" 2>/dev/null) || break
        [ "$cap_id" = "10" ] && break
        cap_off=$(setpci -s "$dev" "$((off + 1)).B" 2>/dev/null) || break
        off=$((16#${cap_off}))
    done
    if [ "$cap_id" = "10" ]; then
        # LNKCTL is at PCIe cap + 0x10
        local lnkctl_off
        lnkctl_off=$(printf '%x' "$((off + 0x10))")
        local lnkctl
        lnkctl=$(setpci -s "$dev" "0x${lnkctl_off}.W" 2>/dev/null) || return
        local new_lnkctl
        new_lnkctl=$(printf '%04x' "$((16#${lnkctl} & 0xFFFC))")  # clear bits[1:0]
        setpci -s "$dev" "0x${lnkctl_off}.W=${new_lnkctl}" 2>/dev/null || true
        log "ASPM disabled on $dev: LNKCTL 0x${lnkctl} → 0x${new_lnkctl}"
    fi
}

# Wait up to 30s for PLX to enumerate and become accessible
for i in $(seq 1 30); do
    val=$(setpci -s "$PLX" 0x00.L 2>/dev/null)
    if [ -n "$val" ] && [ "$val" != "ffffffff" ]; then
        log "PLX alive at t=${i}s: VID:DID=0x${val}"
        break
    fi
    [ "$i" -eq 30 ] && { log "ERROR: PLX not accessible after 30s — link dead or wrong BDF"; exit 1; }
    sleep 1
done

# Disable ASPM on PLX subtree at startup
log "Disabling ASPM L0s/L1 on PLX subtree..."
for dev in "$PLX" "$PLX_DP1" "$PLX_DP2"; do
    disable_aspm "$dev" 2>/dev/null || true
done
log "ASPM disabled. Keepalive loop started (interval=${INTERVAL}s)"

# Main keepalive loop: generate upstream and downstream PCIe traffic
while true; do
    # Config reads to PLX and downstream ports
    plx_val=$(setpci -s "$PLX" 0x04.W 2>/dev/null)
    if [ -z "$plx_val" ] || [ "$plx_val" = "ffff" ]; then
        warn "PLX upstream config read returned ${plx_val:-empty} — link may be gated"
    fi
    setpci -s "$PLX_DP1" 0x04.W >/dev/null 2>&1 || true
    setpci -s "$PLX_DP2" 0x04.W >/dev/null 2>&1 || true

    # Config reads to K80 endpoints — generates TLPs through PLX fabric
    setpci -s "$K80_DIE0" 0x04.W >/dev/null 2>&1 || \
        warn "K80 die0 config read failed — may be gated"
    setpci -s "$K80_DIE1" 0x04.W >/dev/null 2>&1 || true

    # MMIO read to PLX BAR0 — MRd TLP across AMD→PLX link
    if [ -r "$PLX_RES" ]; then
        dd if="$PLX_RES" bs=4 count=1 iflag=skip_bytes skip=0 \
           of=/dev/null 2>/dev/null || true
    fi

    sleep "$INTERVAL"
done
