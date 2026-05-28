#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Build enzyme+xylose systems for -2 and +1 subsites
# Reuses protein topology from cazyme_2d24, fresh solvation
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

BASE="/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel"
D24="$BASE/cazyme_2d24"
REFRESH="$BASE/guidestone_refresh"
FF="charmm36-jul2022"
GH10="$BASE/cazyme_gh10"

export GMX_MAXBACKUP=-1

# Generate xylose PDBs at subsite positions
python3 << 'PYEOF'
import numpy as np

STRUCTS = "/home/strandgate/Development/ecoPrimals/springs/hotSpring/pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/structures"
GH10 = "/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel/cazyme_gh10"

with open(f"{STRUCTS}/2D24.pdb") as f:
    lines = f.readlines()

xys = {}
for line in lines:
    if line.startswith("HETATM") and "XYS" in line[17:20] and line[21] == "C":
        resnum = int(line[22:26].strip())
        aname = line[12:16].strip()
        xys.setdefault(resnum, {})[aname] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

for tag, resnum in [("m2", 3), ("p1", 5)]:
    coords = xys[resnum].copy()
    if "O1" not in coords:
        c1, o5, c2 = coords["C1"], coords["O5"], coords["C2"]
        normal = np.cross(o5 - c1, c2 - c1)
        normal = normal / np.linalg.norm(normal)
        o1_dir = -(o5 - c1) + 0.5 * normal
        o1_dir = o1_dir / np.linalg.norm(o1_dir)
        coords["O1"] = c1 + 1.43 * o1_dir
    
    with open(f"{GH10}/xylose_{tag}_charmm.pdb", "w") as f:
        f.write(f"REMARK   XYS C {resnum} -> BXYL for subsite {tag}\n")
        serial = 1
        for aname in ["C1","C2","C3","C4","C5","O1","O2","O3","O4","O5"]:
            if aname in coords:
                pos = coords[aname]
                elem = "O" if aname.startswith("O") else "C"
                f.write(f"ATOM  {serial:5d} {aname:<4s} BXYLA   1    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {elem}\n")
                serial += 1
        f.write("TER\nEND\n")
    print(f"  xylose_{tag}_charmm.pdb: {serial-1} atoms")
PYEOF

for tag in m2 p1; do
    WORKDIR_1D="$REFRESH/enzyme_bound_${tag}_1d"
    WORKDIR_2D="$REFRESH/enzyme_bound_${tag}_2d"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Subsite $tag: build + solvate + equilibrate"
    echo "╚══════════════════════════════════════════════╝"

    cd "$WORKDIR_1D"
    rm -f complex.* xylose_*.* protein_A.* em.* nvt.* npt.* posre.itp xylose.itp ions.* *.tpr *.cpt 2>/dev/null || true
    ln -sf "$D24/$FF.ff" .

    # pdb2gmx on xylose alone
    echo "  [1/7] pdb2gmx on xylose..."
    gmx pdb2gmx -f "$GH10/xylose_${tag}_charmm.pdb" \
        -o "xylose.gro" -p "xylose_tmp.top" \
        -water tip3p -ff "$FF" 2>&1 | tail -3

    # Copy protein GRO
    cp "$D24/protein_A.gro" .

    # Combine GROs
    echo "  [2/7] Combining protein + xylose GROs..."
    python3 -c "
prot = open('protein_A.gro').readlines()
xyl = open('xylose.gro').readlines()
pn = int(prot[1])
xn = int(xyl[1])
total = pn + xn
with open('complex.gro', 'w') as f:
    f.write('GH10_${tag}_subsite_complex\n')
    f.write(f'{total:5d}\n')
    for l in prot[2:2+pn]: f.write(l)
    for l in xyl[2:2+xn]: f.write(l)
    f.write(prot[-1])
print(f'  Combined: {total} atoms')
"

    # Build clean topology (protein + xylose only, no solvent/ions yet)
    echo "  [3/7] Building clean topology..."
    cp "$D24/xylose.itp" .
    cp "$D24/posre.itp" .

    # Create fresh complex.top from the original but strip SOL/NA/CL from molecules
    python3 -c "
lines = open('$D24/complex.top').readlines()
with open('complex.top', 'w') as f:
    in_molecules = False
    for line in lines:
        if '[ molecules ]' in line:
            in_molecules = True
        if in_molecules and ('SOL' in line or 'NA ' in line or 'CL ' in line):
            continue
        f.write(line)
print('  Clean topology written')
"

    # Box
    echo "  [4/7] editconf..."
    gmx editconf -f complex.gro -o complex_box.gro \
        -c -d 1.2 -bt cubic 2>&1 | tail -2

    # Solvate
    echo "  [5/7] solvate..."
    gmx solvate -cp complex_box.gro -cs spc216.gro \
        -o complex_solv.gro -p complex.top 2>&1 | tail -3

    # Ions
    echo "  [6/7] genion..."
    cp "$D24/ions.mdp" .
    gmx grompp -f ions.mdp -c complex_solv.gro -p complex.top -o ions.tpr -maxwarn 2 2>&1 | tail -2
    echo "SOL" | gmx genion -s ions.tpr -o complex_ions.gro -p complex.top \
        -pname NA -nname CL -neutral -conc 0.15 2>&1 | tail -3

    # Equilibration
    cp "$D24/em.mdp" .
    cp "$D24/nvt.mdp" .
    cp "$D24/npt.mdp" .

    echo "  [7a/7] Energy minimization..."
    gmx grompp -f em.mdp -c complex_ions.gro -p complex.top -o em.tpr -maxwarn 2 2>&1 | tail -2
    gmx mdrun -deffnm em -ntmpi 1 -ntomp 4 2>&1 | tail -3
    echo "  EM: $(grep 'Potential Energy' em.log | tail -1)"

    # Create index groups for thermostat coupling
    echo -e "1 | 12\nname 20 Protein_Other\n16 | 14 | 15\nname 21 Water_and_ions\nq" | \
        gmx make_ndx -f em.gro -o index.ndx 2>&1 | tail -2

    echo "  [7b/7] NVT equilibration..."
    gmx grompp -f nvt.mdp -c em.gro -r em.gro -p complex.top -n index.ndx -o nvt.tpr -maxwarn 2 2>&1 | tail -2
    gmx mdrun -deffnm nvt -ntmpi 1 -ntomp 4 -gpu_id 0 2>&1 | tail -3

    echo "  [7c/7] NPT equilibration..."
    gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p complex.top -n index.ndx -o npt.tpr -maxwarn 2 2>&1 | tail -2
    gmx mdrun -deffnm npt -ntmpi 1 -ntomp 4 -gpu_id 0 2>&1 | tail -3

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
