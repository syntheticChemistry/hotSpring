#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Build enzyme+xylose systems for -2 and +1 subsites
# Reuses protein topology from cazyme_2d24, places xylose at new positions
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

BASE="/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel"
D24="$BASE/cazyme_2d24"
REFRESH="$BASE/guidestone_refresh"
FF="charmm36-jul2022"
STRUCTS="/home/strandgate/Development/ecoPrimals/springs/hotSpring/pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/structures"

export GMX_MAXBACKUP=-1

# Step 1: Generate xylose GRO at each subsite from 2D24 PDB coordinates
python3 << 'PYEOF'
"""Create xylose GRO files at -2 and +1 subsite positions.
Uses the pdb2gmx-generated xylose from cazyme_2d24 as template for atom order,
and XYS coordinates from 2D24 PDB for the heavy atom positions.
pdb2gmx will be run separately on each xylose to add hydrogens.
"""
import os
import numpy as np

STRUCTS = "/home/strandgate/Development/ecoPrimals/springs/hotSpring/pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/structures"
D24 = "/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel/cazyme_2d24"
GH10 = "/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel/cazyme_gh10"

# Read 2D24 PDB XYS coordinates
pdb_path = f"{STRUCTS}/2D24.pdb"
with open(pdb_path) as f:
    lines = f.readlines()

xys_coords = {}
for line in lines:
    if line.startswith("HETATM") and "XYS" in line[17:20]:
        chain = line[21]
        resnum = int(line[22:26].strip())
        aname = line[12:16].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        if chain == "C":
            xys_coords.setdefault(resnum, {})[aname] = np.array([x, y, z])

# For each subsite, write a PDB of just the xylose (rename to BXYL, add O1 if missing)
for tag, resnum, desc in [("m2", 3, "-2 subsite"), ("p1", 5, "+1 subsite")]:
    coords = xys_coords[resnum]
    # XYS has C1,C2,C3,C4,C5,O2,O3,O4,O5 (9 atoms for non-terminal)
    # BXYL needs O1 too (anomeric hydroxyl)
    # If O1 is missing, add it at a reasonable position (C1 + ~1.4A along C1-O5 direction, rotated)
    if "O1" not in coords:
        c1 = coords["C1"]
        o5 = coords["O5"]
        c2 = coords["C2"]
        # O1 is on the opposite side of the ring from O5, approximately
        v_c1_o5 = o5 - c1
        v_c1_c2 = c2 - c1
        # O1 direction: approximately -v_c1_o5 projected perpendicular to c1_c2
        normal = np.cross(v_c1_o5, v_c1_c2)
        normal = normal / np.linalg.norm(normal)
        o1_dir = -v_c1_o5 + 0.5 * normal
        o1_dir = o1_dir / np.linalg.norm(o1_dir)
        coords["O1"] = c1 + 1.43 * o1_dir
    
    outpath = f"{GH10}/xylose_{tag}_charmm.pdb"
    with open(outpath, "w") as f:
        f.write(f"REMARK   {desc} xylose from 2D24, XYS C {resnum} -> BXYL\n")
        f.write(f"REMARK   O1 reconstructed if absent from non-terminal residue\n")
        serial = 1
        for aname in ["C1","C2","C3","C4","C5","O1","O2","O3","O4","O5"]:
            if aname in coords:
                pos = coords[aname]
                elem = "O" if aname.startswith("O") else "C"
                f.write(f"ATOM  {serial:5d} {aname:<4s} BXYLA   1    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {elem}\n")
                serial += 1
        f.write("TER\nEND\n")
    print(f"  {outpath}: {desc} ({serial-1} atoms)")

print("Done generating subsite xylose PDBs.")
PYEOF

echo ""
echo "▶ Building subsite systems..."

for tag in m2 p1; do
    WORKDIR_1D="$REFRESH/enzyme_bound_${tag}_1d"
    WORKDIR_2D="$REFRESH/enzyme_bound_${tag}_2d"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Subsite $tag: build + solvate + equilibrate"
    echo "╚══════════════════════════════════════════════╝"

    cd "$WORKDIR_1D"
    ln -sf "$D24/$FF.ff" .

    # Step 1: Run pdb2gmx on xylose ALONE to get GRO with hydrogens
    echo "  [1/8] pdb2gmx on xylose..."
    gmx pdb2gmx -f "$BASE/cazyme_gh10/xylose_${tag}_charmm.pdb" \
        -o "xylose_${tag}.gro" -p "xylose_${tag}.top" \
        -water tip3p -ff "$FF" 2>&1 | tail -3

    # Step 2: Copy existing protein GRO and topology from cazyme_2d24
    echo "  [2/8] Copying protein topology..."
    cp "$D24/protein_A.gro" .
    
    # Step 3: Combine protein + xylose GROs
    echo "  [3/8] Combining protein + xylose..."
    python3 << PYEOF2
# Combine protein and xylose GROs
prot_lines = open("protein_A.gro").readlines()
xyl_lines = open("xylose_${tag}.gro").readlines()

prot_natoms = int(prot_lines[1].strip())
xyl_natoms = int(xyl_lines[1].strip())
total = prot_natoms + xyl_natoms

with open("complex.gro", "w") as f:
    f.write("GH10_xylanase_xylose_ES_complex (${tag} subsite) in vacuum\n")
    f.write(f"{total:5d}\n")
    # Protein atoms (renumber)
    serial = 1
    for line in prot_lines[2:2+prot_natoms]:
        f.write(line[:15] + f"{serial:5d}" + line[20:])
        serial += 1
    # Xylose atoms
    for line in xyl_lines[2:2+xyl_natoms]:
        resnum = prot_natoms // 20 + 1  # approximate residue number after protein
        f.write(line[:15] + f"{serial:5d}" + line[20:])
        serial += 1
    # Box vector (use protein box or set large)
    f.write(prot_lines[-1])
print(f"  Combined: {total} atoms")
PYEOF2

    # Step 4: Create combined topology
    echo "  [4/8] Building combined topology..."
    cp "$D24/complex.top" .
    cp "$D24/xylose.itp" .
    cp "$D24/posre.itp" .

    # Step 5: Box
    echo "  [5/8] editconf..."
    gmx editconf -f complex.gro -o complex_box.gro \
        -c -d 1.2 -bt cubic 2>&1 | tail -2

    # Step 6: Solvate
    echo "  [6/8] solvate..."
    gmx solvate -cp complex_box.gro -cs spc216.gro \
        -o complex_solv.gro -p complex.top 2>&1 | tail -3

    # Step 7: Ions
    cp "$D24/ions.mdp" .
    gmx grompp -f ions.mdp -c complex_solv.gro -p complex.top -o ions.tpr -maxwarn 2 2>&1 | tail -2
    echo "SOL" | gmx genion -s ions.tpr -o complex_ions.gro -p complex.top \
        -pname NA -nname CL -neutral -conc 0.15 2>&1 | tail -3

    # Step 8: EM → NVT → NPT
    cp "$D24/em.mdp" .
    cp "$D24/nvt.mdp" .
    cp "$D24/npt.mdp" .

    echo "  [7/8] Energy minimization..."
    gmx grompp -f em.mdp -c complex_ions.gro -p complex.top -o em.tpr -maxwarn 2 2>&1 | tail -2
    gmx mdrun -deffnm em -ntmpi 1 -ntomp 4 2>&1 | tail -3
    echo "  EM: $(grep 'Potential Energy' em.log | tail -1)"

    echo "  [8a/8] NVT equilibration..."
    gmx grompp -f nvt.mdp -c em.gro -r em.gro -p complex.top -o nvt.tpr -maxwarn 2 2>&1 | tail -2
    gmx mdrun -deffnm nvt -ntmpi 1 -ntomp 4 -gpu_id 0 2>&1 | tail -3
    echo "  NVT done."

    echo "  [8b/8] NPT equilibration..."
    gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p complex.top -o npt.tpr -maxwarn 2 2>&1 | tail -2
    gmx mdrun -deffnm npt -ntmpi 1 -ntomp 4 -gpu_id 0 2>&1 | tail -3
    echo "  NPT done."

    # Copy to 2D directory
    cp npt.gro "$WORKDIR_2D/"
    cp complex.top "$WORKDIR_2D/"
    cp posre.itp "$WORKDIR_2D/"
    cp xylose.itp "$WORKDIR_2D/"
    ln -sf "$D24/$FF.ff" "$WORKDIR_2D/$FF.ff"

    echo "  ✓ $tag subsite system ready."
done

echo ""
echo "═══════════════════════════════════════════"
echo "  Both subsite systems built and equilibrated."
echo "═══════════════════════════════════════════"
