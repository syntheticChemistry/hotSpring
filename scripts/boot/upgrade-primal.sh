#!/bin/bash
# upgrade-primal.sh — Upgrade any NUCLEUS primal via plasmidBin ecoBin
#
# Fetches the latest ecoBin from plasmidBin (with release cascade),
# installs to /usr/local/bin, and restarts the associated systemd service
# if one exists.
#
# Usage:
#   sudo ./scripts/boot/upgrade-primal.sh toadstool
#   sudo ./scripts/boot/upgrade-primal.sh --all
#   sudo ./scripts/boot/upgrade-primal.sh barracuda --force
#   sudo ./scripts/boot/upgrade-primal.sh --check toadstool
#   sudo ./scripts/boot/upgrade-primal.sh --status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOTSPRING_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ECOPRIMALS_ROOT="$(cd "$HOTSPRING_ROOT/../.." && pwd)"
PLASMIDBIN="${ECOPRIMALS_ROOT}/infra/plasmidBin"

COMPUTE_TRIO=(toadstool barracuda coralreef)
ALL_PRIMALS=(beardog songbird skunkbat toadstool barracuda coralreef nestgate rhizocrypt loamspine sweetgrass squirrel petaltongue biomeos)

SERVICE_MAP=(
    "toadstool:toadstool-ember"
    "barracuda:barracuda"
    "coralreef:coralreef"
)

CHECK_ONLY=false
STATUS_ONLY=false
FORCE=""
DEPLOY_ALL=false
TARGETS=()

usage() {
    echo "Usage: $0 [OPTIONS] PRIMAL [PRIMAL...]"
    echo ""
    echo "Options:"
    echo "  --all        Upgrade all 13 NUCLEUS primals"
    echo "  --trio       Upgrade compute trio (toadstool, barracuda, coralreef)"
    echo "  --check      Check for updates only (no install)"
    echo "  --status     Show deployed vs ecoBin status for all primals"
    echo "  --force      Force re-download and reinstall"
    echo "  --help       Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)    DEPLOY_ALL=true; TARGETS=("${ALL_PRIMALS[@]}"); shift ;;
        --trio)   TARGETS=("${COMPUTE_TRIO[@]}"); shift ;;
        --check)  CHECK_ONLY=true; shift ;;
        --status) STATUS_ONLY=true; shift ;;
        --force)  FORCE="--force"; shift ;;
        --help)   usage; exit 0 ;;
        -*)       echo "Unknown option: $1"; usage; exit 1 ;;
        *)        TARGETS+=("$1"); shift ;;
    esac
done

if [[ ${#TARGETS[@]} -eq 0 ]] && ! $STATUS_ONLY; then
    echo "ERROR: Specify a primal name, --trio, or --all"
    usage
    exit 1
fi

if [ ! -d "$PLASMIDBIN" ] || [ ! -x "$PLASMIDBIN/fetch.sh" ]; then
    echo "FATAL: plasmidBin not found at $PLASMIDBIN" >&2
    echo "  Clone: git clone https://github.com/ecoPrimals/plasmidBin $PLASMIDBIN" >&2
    exit 1
fi

service_for() {
    local primal="$1"
    for entry in "${SERVICE_MAP[@]}"; do
        local key="${entry%%:*}"
        local val="${entry#*:}"
        if [[ "$key" == "$primal" ]]; then
            echo "$val"
            return
        fi
    done
    echo ""
}

ecobin_path() {
    local primal="$1"
    local arch_path="$PLASMIDBIN/primals/x86_64-unknown-linux-musl/$primal"
    local flat_path="$PLASMIDBIN/primals/$primal"
    if [[ -f "$arch_path" ]]; then
        echo "$arch_path"
    elif [[ -f "$flat_path" ]] && [[ ! -L "$flat_path" ]]; then
        echo "$flat_path"
    elif [[ -L "$flat_path" ]] && [[ -f "$flat_path" ]]; then
        echo "$flat_path"
    else
        echo ""
    fi
}

UPGRADED=0
SKIPPED=0
FAILED=0

if $STATUS_ONLY; then
    echo ">>> Primal deployment status — $(date -Iseconds)"
    printf "  %-15s %-10s %-10s %s\n" "PRIMAL" "DEPLOYED" "ECOBIN" "MATCH"
    printf "  %-15s %-10s %-10s %s\n" "------" "--------" "------" "-----"
    for p in "${ALL_PRIMALS[@]}"; do
        deployed="no"
        ecobin="no"
        match="-"
        if [[ -f "/usr/local/bin/$p" ]]; then deployed="yes"; fi
        eb=$(ecobin_path "$p")
        if [[ -n "$eb" ]]; then ecobin="yes"; fi
        if [[ "$deployed" == "yes" && "$ecobin" == "yes" ]]; then
            dh=$(b3sum --no-names "/usr/local/bin/$p" 2>/dev/null)
            eh=$(b3sum --no-names "$eb" 2>/dev/null)
            if [[ "$dh" == "$eh" ]]; then match="CURRENT"; else match="STALE"; fi
        fi
        svc=$(service_for "$p")
        svc_status=""
        if [[ -n "$svc" ]]; then
            local_state=$(systemctl is-active "$svc" 2>/dev/null) || \
                local_state=$(systemctl --user is-active "$svc" 2>/dev/null) || \
                local_state="n/a"
            svc_status=" [${local_state}]"
        fi
        printf "  %-15s %-10s %-10s %s%s\n" "$p" "$deployed" "$ecobin" "$match" "$svc_status"
    done
    exit 0
fi

echo ">>> Primal upgrade via plasmidBin — $(date -Iseconds)"
echo "  Source: $PLASMIDBIN"
echo "  Targets: ${TARGETS[*]}"
echo ""

echo ">>> Pulling latest plasmidBin..."
(cd "$PLASMIDBIN" && git pull --quiet origin main 2>/dev/null) || echo "  (git pull skipped)"

echo ">>> Fetching ecoBins..."
(cd "$PLASMIDBIN" && bash fetch.sh --all $FORCE 2>&1) || true
echo ""

for primal in "${TARGETS[@]}"; do
    echo "--- [$primal] ---"
    eb=$(ecobin_path "$primal")
    if [[ -z "$eb" ]]; then
        echo "  SKIP: no ecoBin available"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if $CHECK_ONLY; then
        if [[ -f "/usr/local/bin/$primal" ]]; then
            dh=$(b3sum --no-names "/usr/local/bin/$primal" 2>/dev/null)
            eh=$(b3sum --no-names "$eb" 2>/dev/null)
            if [[ "$dh" == "$eh" ]]; then
                echo "  CURRENT (hash match)"
            else
                echo "  UPDATE AVAILABLE"
            fi
        else
            echo "  NOT INSTALLED (ecoBin available)"
        fi
        continue
    fi

    if [[ -f "/usr/local/bin/$primal" ]]; then
        dh=$(b3sum --no-names "/usr/local/bin/$primal" 2>/dev/null)
        eh=$(b3sum --no-names "$eb" 2>/dev/null)
        if [[ "$dh" == "$eh" ]] && [[ -z "$FORCE" ]]; then
            echo "  CURRENT (hash match, skipping)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    svc=$(service_for "$primal")

    if [[ -f "/usr/local/bin/$primal" ]]; then
        sudo cp "/usr/local/bin/$primal" "/usr/local/bin/${primal}.prev"
        echo "  Backup: /usr/local/bin/${primal}.prev"
    fi

    if [[ -n "$svc" ]]; then
        echo '{"jsonrpc":"2.0","method":"health.drain","id":1}' \
          | socat -t5 - "UNIX:/run/toadstool/biomeos/compute.sock" >/dev/null 2>&1 || true
        sleep 1
        sudo systemctl stop "$svc" 2>/dev/null || true
        echo "  Stopped: $svc"
    fi

    sudo install -m 755 "$eb" "/usr/local/bin/$primal"
    echo "  Installed: /usr/local/bin/$primal"

    svc_file="$SCRIPT_DIR/${svc}.service"
    if [[ -n "$svc" ]] && [[ -f "$svc_file" ]]; then
        sudo cp "$svc_file" "/etc/systemd/system/${svc}.service"
        sudo systemctl daemon-reload
        sudo systemctl start "$svc"
        sleep 2
        if systemctl is-active --quiet "$svc"; then
            echo "  Service: $svc ACTIVE"
        else
            echo "  Service: $svc FAILED — rolling back"
            if [[ -f "/usr/local/bin/${primal}.prev" ]]; then
                sudo install -m 755 "/usr/local/bin/${primal}.prev" "/usr/local/bin/$primal"
                sudo systemctl start "$svc"
                sleep 2
                if systemctl is-active --quiet "$svc"; then
                    echo "  Rollback: OK"
                else
                    echo "  Rollback: FAILED — check journalctl -u $svc"
                fi
            fi
            FAILED=$((FAILED + 1))
            continue
        fi
    fi

    ver=$("/usr/local/bin/$primal" --version 2>/dev/null || echo "installed")
    echo "  Version: $ver"
    UPGRADED=$((UPGRADED + 1))
done

echo ""
echo "Summary:"
echo "  Upgraded: $UPGRADED"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   $FAILED"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
