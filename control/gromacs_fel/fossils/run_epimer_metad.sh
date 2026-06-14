#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Metadynamics runs for all 4 free sugar epimers (1D theta + 2D qx,qy)
# Uses PLUMED patched GROMACS with GPU acceleration
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

BASE="/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel"
REFRESH="$BASE/guidestone_refresh"
GH10="$BASE/cazyme_gh10"
FF="charmm36-jul2022"

export GMX_MAXBACKUP=-1
export PLUMED_KERNEL="/home/strandgate/miniconda3/envs/gromacs-fel/lib/libplumedKernel.so"

run_metad() {
    local sugar=$1
    local dim=$2
    local workdir="$REFRESH/free_${sugar}_${dim}"
    local topname="${sugar}"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  $sugar $dim metadynamics"
    echo "╚══════════════════════════════════════════════╝"

    cd "$workdir"

    # Ensure topology and force field are available
    if [ ! -f "$FF.ff/forcefield.itp" ]; then
        ln -sf "$GH10/$FF.ff" .
    fi

    # For 1D: use npt.gro from equilibration as input
    # For 2D: use npt.gro copied from 1D dir
    local startgro="npt.gro"

    # grompp
    echo "  grompp..."
    gmx grompp -f md_meta.mdp -c "$startgro" -r "$startgro" \
        -p "${topname}.top" -o md_meta.tpr -maxwarn 1 2>&1 | tail -3

    # mdrun with PLUMED
    echo "  mdrun with PLUMED ($(grep nsteps md_meta.mdp | head -1 | tr -s ' ' | cut -d= -f2 | tr -d ' ') steps)..."
    gmx mdrun -deffnm md_meta -ntmpi 1 -ntomp 4 -gpu_id 0 \
        -plumed plumed.dat 2>&1 | tail -5

    echo "  ✓ $sugar $dim metadynamics complete."

    # Check output files exist
    if [ "$dim" = "1d" ]; then
        ls -la HILLS COLVAR 2>/dev/null | head -3
    else
        ls -la HILLS_2d COLVAR_2d 2>/dev/null | head -3
    fi
}

# Generate FES after each run
generate_fes() {
    local sugar=$1
    local dim=$2
    local workdir="$REFRESH/free_${sugar}_${dim}"

    cd "$workdir"
    echo "  Generating FES for $sugar $dim..."

    if [ "$dim" = "1d" ]; then
        plumed sum_hills --hills HILLS --outfile fes_theta.dat \
            --mintozero 2>&1 | tail -2
        echo "  FES: fes_theta.dat"
    else
        plumed sum_hills --hills HILLS_2d --outfile fes_2d.dat \
            --mintozero 2>&1 | tail -2
        echo "  FES: fes_2d.dat"
    fi
}

# Run all 1D first (shorter), then all 2D
for sugar in lyxose glucose mannose galactose; do
    run_metad "$sugar" "1d"
    generate_fes "$sugar" "1d"
done

for sugar in lyxose glucose mannose galactose; do
    run_metad "$sugar" "2d"
    generate_fes "$sugar" "2d"
done

echo ""
echo "═══════════════════════════════════════════"
echo "  All 8 epimer metadynamics runs complete."
echo "═══════════════════════════════════════════"
