#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Epimer FEL Pipeline: pdb2gmx → solvate → EM → NVT → NPT for 4 sugars
# Uses CHARMM36 FF, TIP3P water, same protocol as free xylose baseline
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

BASE="/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel"
GH10="$BASE/cazyme_gh10"
REFRESH="$BASE/guidestone_refresh"
FF="charmm36-jul2022"

export GMX_MAXBACKUP=-1

process_sugar() {
    local sugar=$1
    local pdb_resname=$2

    WORKDIR_1D="$REFRESH/free_${sugar}_1d"
    WORKDIR_2D="$REFRESH/free_${sugar}_2d"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  $sugar ($pdb_resname): pdb2gmx + solvate + equilibrate"
    echo "╚══════════════════════════════════════════════╝"

    cd "$WORKDIR_1D"
    ln -sf "$GH10/$FF.ff" .
    cp "$GH10/em.mdp" .
    cp "$GH10/nvt.mdp" .
    cp "$GH10/npt.mdp" .

    # pdb2gmx
    echo "  [1/6] pdb2gmx..."
    gmx pdb2gmx -f "$GH10/${sugar}_charmm.pdb" -o "${sugar}.gro" -p "${sugar}.top" \
        -water tip3p -ff "$FF" 2>&1 | tail -3

    # Box
    echo "  [2/6] editconf (cubic box, 1.2 nm buffer)..."
    gmx editconf -f "${sugar}.gro" -o "${sugar}_box.gro" \
        -c -d 1.2 -bt cubic 2>&1 | tail -2

    # Solvate
    echo "  [3/6] solvate..."
    gmx solvate -cp "${sugar}_box.gro" -cs spc216.gro \
        -o "${sugar}_solv.gro" -p "${sugar}.top" 2>&1 | tail -3

    # Energy minimization
    echo "  [4/6] energy minimization..."
    gmx grompp -f em.mdp -c "${sugar}_solv.gro" -p "${sugar}.top" -o em.tpr -maxwarn 1 2>&1 | tail -2
    gmx mdrun -deffnm em -ntmpi 1 -ntomp 4 2>&1 | tail -2
    PE=$(grep 'Potential Energy' em.log | tail -1)
    echo "  EM: $PE"

    # NVT
    echo "  [5/6] NVT equilibration (100 ps)..."
    gmx grompp -f nvt.mdp -c em.gro -r em.gro -p "${sugar}.top" -o nvt.tpr -maxwarn 1 2>&1 | tail -2
    gmx mdrun -deffnm nvt -ntmpi 1 -ntomp 4 -gpu_id 0 2>&1 | tail -2
    echo "  NVT complete."

    # NPT
    echo "  [6/6] NPT equilibration (100 ps)..."
    gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p "${sugar}.top" -o npt.tpr -maxwarn 1 2>&1 | tail -2
    gmx mdrun -deffnm npt -ntmpi 1 -ntomp 4 -gpu_id 0 2>&1 | tail -2
    echo "  NPT complete."

    # Copy equilibrated system to 2D directory
    cp npt.gro "${WORKDIR_2D}/"
    cp "${sugar}.top" "${WORKDIR_2D}/"
    cp posre.itp "${WORKDIR_2D}/" 2>/dev/null || true
    ln -sf "$GH10/$FF.ff" "${WORKDIR_2D}/$FF.ff"

    echo "  ✓ $sugar equilibration done → npt.gro copied to 2D dir"
}

# Run pipeline for each sugar
process_sugar "lyxose"    "BXYL"
process_sugar "glucose"   "BGLC"
process_sugar "mannose"   "BMAN"
process_sugar "galactose" "BGAL"

echo ""
echo "═══════════════════════════════════════════"
echo "  All 4 epimer equilibrations complete."
echo "═══════════════════════════════════════════"
