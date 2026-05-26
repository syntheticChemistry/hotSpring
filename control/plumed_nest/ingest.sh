#!/usr/bin/env bash
# DEPRECATED: Use nest-validate/target/release/nest-validate ingest
# This script is kept as fossil record. The Rust binary replaces it entirely.
#
# ingest.sh — Download and validate PLUMED-NEST archives
#
# Downloads zip archives from PLUMED-NEST, extracts them, and validates
# that all PLUMED input files parse cleanly with the installed version.
#
# Usage:
#   ./ingest.sh --all              # Ingest all targets
#   ./ingest.sh --target 01        # Ingest target 01 only
#   ./ingest.sh --validate-only    # Just re-run parse validation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

declare -A TARGETS=(
    [01]="target_01_alanine_dipeptide|https://github.com/connorzzou/PLUMED-NEST/raw/main/PLUMED_GNN_SPIB.zip|24.020"
    [02]="target_02_chignolin_opes|https://github.com/dhimanray/OPES_and_OPES-EXPLORE/archive/main.zip|24.029"
    [03]="target_03_brd4_oneopes|https://zenodo.org/records/11126468/files/Supporting_Material.zip|24.017"
    [04]="target_04_muscarinic_funnel|https://github.com/riccardocapelli/papers_data/raw/master/muscarinic_m2_2019/input_data.zip|20.000"
    [05]="target_05_glycan_pucker|https://github.com/IsabellGrothaus/Data_repository/raw/main/Grothaus_et_al_2022_PLUMED.zip|22.028"
    [06]="target_06_cazyme_glycan|https://github.com/IsabellGrothaus/Data_repository/raw/main/Grothaus_et_al_2025_CAZymes_PLUMED.zip|25.007"
    [07]="target_07_amylase_qmmm|https://github.com/sudipdas789/Committor_Amylase/raw/main/Committor_Amylase_PLUMED_NEST.zip|25.012"
    [08]="target_08_urea_nucleation|https://github.com/zpengmei/Nucleation-OPES-GNN/archive/main.zip|22.039"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

TARGET_FILTER=""
VALIDATE_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)           TARGET_FILTER="all"; shift ;;
        --target)        TARGET_FILTER="$2"; shift 2 ;;
        --validate-only) VALIDATE_ONLY=true; shift ;;
        --help)
            echo "Usage: $0 [--all | --target NN | --validate-only]"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -z "$TARGET_FILTER" ]] && TARGET_FILTER="all"

ingest_target() {
    local id="$1"
    local entry="${TARGETS[$id]}"
    local dir="${entry%%|*}"
    local rest="${entry#*|}"
    local url="${rest%%|*}"
    local plumid="${rest##*|}"

    printf "${CYAN}=== Target %s (plumID:%s) ===${NC}\n" "$id" "$plumid"
    printf "  Directory: %s\n" "$dir"
    printf "  Archive:   %s\n" "$url"

    mkdir -p "$dir"/{archive,inputs,plumed,output,analysis,figures,reference}

    if ! $VALIDATE_ONLY; then
        local zipfile="$dir/archive/nest_${plumid}.zip"

        if [[ -f "$zipfile" ]]; then
            printf "  ${YELLOW}Archive exists, skipping download${NC}\n"
        else
            printf "  Downloading..."
            if curl -sfL -o "$zipfile" "$url"; then
                printf " ${GREEN}OK${NC} ($(du -h "$zipfile" | cut -f1))\n"
            else
                printf " ${RED}FAILED${NC}\n"
                return 1
            fi
        fi

        printf "  Extracting..."
        if unzip -qo "$zipfile" -d "$dir/archive/" 2>/dev/null; then
            printf " ${GREEN}OK${NC}\n"
        else
            printf " ${YELLOW}WARN${NC} (may be tar.gz or nested)\n"
        fi

        printf "  Copying PLUMED inputs..."
        find "$dir/archive" -name "plumed*.dat" -exec cp {} "$dir/plumed/" \; 2>/dev/null
        local count=$(find "$dir/plumed" -name "*.dat" 2>/dev/null | wc -l)
        printf " ${GREEN}%d .dat files${NC}\n" "$count"
    fi

    printf "  Validating PLUMED inputs...\n"
    local pass=0 fail=0
    while IFS= read -r datfile; do
        local fname=$(basename "$datfile")
        if plumed driver --natoms 100000 --parse-only --kt 2.49 --plumed "$datfile" >/dev/null 2>&1; then
            printf "    ${GREEN}PASS${NC} %s\n" "$fname"
            pass=$((pass + 1))
        else
            printf "    ${RED}FAIL${NC} %s\n" "$fname"
            fail=$((fail + 1))
        fi
    done < <(find "$dir/plumed" -name "*.dat" 2>/dev/null | sort)

    if [[ $pass -eq 0 && $fail -eq 0 ]]; then
        printf "    ${YELLOW}No .dat files found${NC}\n"
    else
        printf "  Result: ${GREEN}%d pass${NC}, ${RED}%d fail${NC}\n" "$pass" "$fail"
    fi
    echo ""
}

echo ""
printf "${CYAN}╔══════════════════════════════════════════════════╗${NC}\n"
printf "${CYAN}║  PLUMED-NEST Ingestion Pipeline                 ║${NC}\n"
printf "${CYAN}╚══════════════════════════════════════════════════╝${NC}\n"
echo ""

if [[ "$TARGET_FILTER" == "all" ]]; then
    for id in $(echo "${!TARGETS[@]}" | tr ' ' '\n' | sort); do
        ingest_target "$id"
    done
else
    if [[ -n "${TARGETS[$TARGET_FILTER]+x}" ]]; then
        ingest_target "$TARGET_FILTER"
    else
        echo "ERROR: Unknown target: $TARGET_FILTER"
        echo "Available: ${!TARGETS[*]}"
        exit 1
    fi
fi

printf "${CYAN}=== Ingestion Complete ===${NC}\n"
echo ""
echo "Next steps:"
echo "  1. Review extracted inputs in each target's plumed/ directory"
echo "  2. Prepare GROMACS topology/coordinates in inputs/"
echo "  3. Run: gmx mdrun -plumed plumed/plumed.dat -deffnm output/md"
echo "  4. Analyze: plumed sum_hills --hills output/HILLS --mintozero"
