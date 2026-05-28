#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Build GH11 (1XYN) xylanase + xylose system
# Smaller system (~25k atoms vs ~92k for GH10)
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

BASE="/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel"
D24="$BASE/cazyme_2d24"
GH10="$BASE/cazyme_gh10"
REFRESH="$BASE/guidestone_refresh"
FF="charmm36-jul2022"
STRUCTS="/home/strandgate/Development/ecoPrimals/springs/hotSpring/pseudoSpore_hotSpring-CompChem-GuideStone_v1.6.1/structures"

export GMX_MAXBACKUP=-1

WORKDIR_1D="$REFRESH/gh11_bound_1d"
WORKDIR_2D="$REFRESH/gh11_bound_2d"

echo "╔══════════════════════════════════════════════╗"
echo "║  GH11 (1XYN): build + solvate + equilibrate"
echo "╚══════════════════════════════════════════════╝"

cd "$WORKDIR_1D"
ln -sf "$D24/$FF.ff" .

# Step 1: Extract protein chain A from 1XYN and rename HIS to HISE
echo "  [1/8] Preparing protein PDB..."
python3 -c "
with open('$STRUCTS/1XYN.pdb') as f:
    lines = f.readlines()
with open('protein_1xyn.pdb', 'w') as f:
    for line in lines:
        if line.startswith('ATOM') and line[21] == 'A':
            line = line.replace(' HIS ', ' HISE')
            f.write(line)
    f.write('TER\nEND\n')
print('  1XYN protein extracted')
"

# Step 2: pdb2gmx on protein
echo "  [2/8] pdb2gmx on protein..."
gmx pdb2gmx -f protein_1xyn.pdb -o protein.gro -p protein.top \
    -water tip3p -ff "$FF" -ignh 2>&1 | tail -3

# Step 3: pdb2gmx on xylose (placed at -1 subsite)
echo "  [3/8] pdb2gmx on xylose..."
gmx pdb2gmx -f "$STRUCTS/xylose_gh11_m1.pdb" -o xylose.gro -p xylose_tmp.top \
    -water tip3p -ff "$FF" 2>&1 | tail -3

# Step 4: Combine protein + xylose
echo "  [4/8] Combining protein + xylose..."
python3 -c "
prot = open('protein.gro').readlines()
xyl = open('xylose.gro').readlines()
pn = int(prot[1])
xn = int(xyl[1])
total = pn + xn
with open('complex.gro', 'w') as f:
    f.write('GH11_xylanase_xylose_ES_complex\n')
    f.write(f'{total:5d}\n')
    for l in prot[2:2+pn]: f.write(l)
    for l in xyl[2:2+xn]: f.write(l)
    f.write(prot[-1])
print(f'  Combined: {total} atoms')
"

# Step 5: Build combined topology
echo "  [5/8] Building topology..."
# Extract xylose ITP from the tmp topology
python3 -c "
with open('xylose_tmp.top') as f:
    content = f.read()
# Extract everything between [ moleculetype ] and [ system ]
import re
m = re.search(r'(\[ moleculetype \].*?)(?=; Include water|; Include position|\[ system \])', content, re.DOTALL)
if m:
    with open('xylose.itp', 'w') as f:
        f.write(m.group(1))
    print('  xylose.itp extracted')
"

# Create combined topology
cat > complex.top << 'TOPEOF'
; Combined topology: GH11 (1XYN) + xylose (-1 subsite)
#include "./charmm36-jul2022.ff/forcefield.itp"

TOPEOF

# Append protein moleculetype from protein.top (between first [ moleculetype ] and [ system ])
python3 -c "
with open('protein.top') as f:
    content = f.read()
import re
m = re.search(r'(\[ moleculetype \].*?)(?=; Include water|\[ system \])', content, re.DOTALL)
if m:
    with open('complex.top', 'a') as f:
        f.write(m.group(1))
        f.write('\n; Include position restraint file\n#ifdef POSRES\n#include \"posre.itp\"\n#endif\n\n')
        f.write('; Include xylose topology\n#include \"xylose.itp\"\n\n')
        f.write('#include \"./charmm36-jul2022.ff/tip3p.itp\"\n')
        f.write('#include \"./charmm36-jul2022.ff/ions.itp\"\n\n')
        f.write('[ system ]\nGH11_xylanase_xylose_ES_complex in water\n\n')
        f.write('[ molecules ]\nProtein_chain_A     1\nOther               1\n')
print('  complex.top assembled')
"

# Step 6: Box + Solvate + Ions
echo "  [6/8] Box + Solvate + Ions..."
gmx editconf -f complex.gro -o complex_box.gro -c -d 1.2 -bt cubic 2>&1 | tail -2

gmx solvate -cp complex_box.gro -cs spc216.gro \
    -o complex_solv.gro -p complex.top 2>&1 | tail -3

cp "$D24/ions.mdp" .
gmx grompp -f ions.mdp -c complex_solv.gro -p complex.top -o ions.tpr -maxwarn 2 2>&1 | tail -2
echo "SOL" | gmx genion -s ions.tpr -o complex_ions.gro -p complex.top \
    -pname NA -nname CL -neutral -conc 0.15 2>&1 | tail -3

# Step 7: EM
cp "$D24/em.mdp" .
echo "  [7/8] Energy minimization..."
gmx grompp -f em.mdp -c complex_ions.gro -p complex.top -o em.tpr -maxwarn 2 2>&1 | tail -2
gmx mdrun -deffnm em -ntmpi 1 -ntomp 4 2>&1 | tail -3
echo "  EM: $(grep 'Potential Energy' em.log | tail -1)"

# Create index groups
echo -e "1 | 12\nname 20 Protein_Other\n16 | 14 | 15\nname 21 Water_and_ions\nq" | \
    gmx make_ndx -f em.gro -o index.ndx 2>&1 | tail -2

# Step 8: NVT + NPT
cp "$D24/nvt.mdp" .
cp "$D24/npt.mdp" .

echo "  [8a/8] NVT equilibration..."
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p complex.top -n index.ndx -o nvt.tpr -maxwarn 2 2>&1 | tail -2
gmx mdrun -deffnm nvt -ntmpi 1 -ntomp 4 -gpu_id 0 2>&1 | tail -3

echo "  [8b/8] NPT equilibration..."
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p complex.top -n index.ndx -o npt.tpr -maxwarn 2 2>&1 | tail -2
gmx mdrun -deffnm npt -ntmpi 1 -ntomp 4 -gpu_id 0 2>&1 | tail -3

# Copy to 2D directory
cp npt.gro "$WORKDIR_2D/"
cp complex.top "$WORKDIR_2D/"
cp posre.itp "$WORKDIR_2D/" 2>/dev/null || true
cp xylose.itp "$WORKDIR_2D/"
cp index.ndx "$WORKDIR_2D/"
ln -sf "$D24/$FF.ff" "$WORKDIR_2D/$FF.ff"

echo "  ✓ GH11 system built and equilibrated."
echo ""
echo "═══════════════════════════════════════════"
echo "  GH11 system ready for metadynamics."
echo "═══════════════════════════════════════════"
