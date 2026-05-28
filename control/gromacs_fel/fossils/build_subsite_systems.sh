#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Build enzyme+xylose systems for -2 and +1 subsites from 2D24 PDB
# Reuses protein topology from cazyme_2d24 build
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

BASE="/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel"
D24="$BASE/cazyme_2d24"
REFRESH="$BASE/guidestone_refresh"
FF="charmm36-jul2022"
STRUCTS="/home/strandgate/Development/ecoPrimals/springs/hotSpring/pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/structures"

export GMX_MAXBACKUP=-1

# Step 1: Generate xylose GRO at each subsite position using Python
python3 << 'PYEOF'
"""Extract xylose coordinates from 2D24 PDB for -2 and +1 subsites,
write as GRO files matching the pdb2gmx BXYL atom order."""
import os

PDB = "/home/strandgate/Development/ecoPrimals/springs/hotSpring/pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/structures/2D24.pdb"
D24 = "/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel/cazyme_2d24"

# Read existing xylose GRO to get atom name order (from pdb2gmx)
ref_gro = os.path.join(D24, "xylose_m1.gro")
with open(ref_gro) as f:
    ref_lines = f.readlines()
ref_atom_order = []
for line in ref_lines[2:22]:
    aname = line[10:15].strip()
    ref_atom_order.append(aname)
print(f"Reference atom order: {ref_atom_order}")

# Read 2D24 PDB
with open(PDB) as f:
    pdb_lines = f.readlines()

xys_coords = {}
for line in pdb_lines:
    if line.startswith("HETATM") and "XYS" in line[17:20]:
        chain = line[21:22]
        resnum = int(line[22:26].strip())
        aname = line[12:16].strip()
        x = float(line[30:38]) / 10.0  # A -> nm
        y = float(line[38:46]) / 10.0
        z = float(line[46:54]) / 10.0
        if chain == "C":
            xys_coords.setdefault(resnum, {})[aname] = (x, y, z)

# Map XYS heavy atom names to BXYL convention 
# XYS has: C1,C2,C3,C4,C5,O1(res1 only),O2,O3,O4,O5
# BXYL needs H atoms too, but we only have heavy atoms from PDB
# pdb2gmx will add H atoms when we process the complex PDB

# For the GRO approach, we need all 20 atoms including H
# Since XYS only has heavy atoms, we'll write a PDB for pdb2gmx
# Actually, the simplest approach: write the PDB complex (protein+xylose)
# and run pdb2gmx on it, just like the original

for tag, resnum in [("m2", 3), ("p1", 5)]:
    coords = xys_coords[resnum]
    print(f"\n{tag} (XYS C res {resnum}):")
    for aname in ["C1","C2","C3","C4","C5","O5"]:
        c = coords.get(aname, None)
        if c:
            print(f"  {aname}: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) nm")
    print(f"  Available heavy atoms: {list(coords.keys())}")
PYEOF

echo ""
echo "▶ Building enzyme systems for -2 and +1 subsites..."

for tag in m2 p1; do
    WORKDIR_1D="$REFRESH/enzyme_bound_${tag}_1d"
    WORKDIR_2D="$REFRESH/enzyme_bound_${tag}_2d"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Subsite $tag: pdb2gmx + solvate + equilibrate"
    echo "╚══════════════════════════════════════════════╝"

    cd "$WORKDIR_1D"
    ln -sf "$D24/$FF.ff" .

    # Copy the complex PDB
    cp "$STRUCTS/complex_A_${tag}.pdb" .

    # Rename HIS to HISE (epsilon-protonated, CHARMM36 convention) to avoid interactive
    sed -i 's/ HIS / HISE/g' "complex_A_${tag}.pdb"

    # Run pdb2gmx on the complex (protein + xylose)
    echo "  [1/7] pdb2gmx (protein + xylose)..."
    gmx pdb2gmx -f "complex_A_${tag}.pdb" -o complex.gro -p complex.top \
        -water tip3p -ff "$FF" -ignh 2>&1 | tail -5

    # Box
    echo "  [2/7] editconf..."
    gmx editconf -f complex.gro -o complex_box.gro \
        -c -d 1.2 -bt cubic 2>&1 | tail -2

    # Solvate
    echo "  [3/7] solvate..."
    gmx solvate -cp complex_box.gro -cs spc216.gro \
        -o complex_solv.gro -p complex.top 2>&1 | tail -3

    # Add ions
    echo "  [4/7] genion..."
    cp "$D24/ions.mdp" .
    gmx grompp -f ions.mdp -c complex_solv.gro -p complex.top -o ions.tpr -maxwarn 2 2>&1 | tail -2
    echo "SOL" | gmx genion -s ions.tpr -o complex_ions.gro -p complex.top \
        -pname NA -nname CL -neutral -conc 0.15 2>&1 | tail -3

    # EM
    echo "  [5/7] energy minimization..."
    cp "$D24/em.mdp" .
    gmx grompp -f em.mdp -c complex_ions.gro -p complex.top -o em.tpr -maxwarn 2 2>&1 | tail -2
    gmx mdrun -deffnm em -ntmpi 1 -ntomp 4 2>&1 | tail -3
    PE=$(grep 'Potential Energy' em.log | tail -1)
    echo "  EM: $PE"

    # NVT
    echo "  [6/7] NVT equilibration..."
    cp "$D24/nvt.mdp" .
    gmx grompp -f nvt.mdp -c em.gro -r em.gro -p complex.top -o nvt.tpr -maxwarn 2 2>&1 | tail -2
    gmx mdrun -deffnm nvt -ntmpi 1 -ntomp 4 -gpu_id 0 2>&1 | tail -3
    echo "  NVT complete."

    # NPT
    echo "  [7/7] NPT equilibration..."
    cp "$D24/npt.mdp" .
    gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p complex.top -o npt.tpr -maxwarn 2 2>&1 | tail -2
    gmx mdrun -deffnm npt -ntmpi 1 -ntomp 4 -gpu_id 0 2>&1 | tail -3
    echo "  NPT complete."

    # Copy to 2D directory
    cp npt.gro "$WORKDIR_2D/"
    cp complex.top "$WORKDIR_2D/"
    cp posre.itp "$WORKDIR_2D/" 2>/dev/null || true
    cp xylose.itp "$WORKDIR_2D/" 2>/dev/null || true
    ln -sf "$D24/$FF.ff" "$WORKDIR_2D/$FF.ff"

    echo "  ✓ $tag subsite system ready."
done

echo ""
echo "═══════════════════════════════════════════"
echo "  Both subsite systems built and equilibrated."
echo "═══════════════════════════════════════════"
