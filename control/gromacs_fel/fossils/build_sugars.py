#!/usr/bin/env python3
"""Build sugar PDB files with exact CHARMM36 .rtp atom naming.

Strategy:
1. Parse atom names from carb.rtp for each target residue
2. Generate 3D conformer via RDKit
3. Map RDKit atoms to CHARMM names using ring-walking algorithm
4. Write PDB with exact names from .rtp (order matches .rtp)
"""
import os
import re
from rdkit import Chem
from rdkit.Chem import AllChem

FF_DIR = "/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel/cazyme_gh10/charmm36-jul2022.ff"
OUTDIR = "/home/strandgate/Development/ecoPrimals/springs/hotSpring/control/gromacs_fel/cazyme_gh10"


def parse_rtp_atoms(rtp_path, resname):
    """Extract ordered atom names and their element from .rtp."""
    with open(rtp_path) as f:
        content = f.read()
    start = content.find(f"[ {resname} ]")
    next_block = content.find("\n[ ", start + 10)
    block = content[start:next_block]
    atoms_start = block.find("[ atoms ]")
    next_sec = block.find("[ ", atoms_start + 10)
    atoms_text = block[atoms_start:next_sec]
    atoms = []
    for line in atoms_text.split('\n')[1:]:
        line = line.strip()
        if line and not line.startswith(';') and not line.startswith('['):
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                elem = "H" if name.startswith("H") else ("O" if name.startswith("O") else "C")
                atoms.append((name, elem))
    return atoms


def walk_pyranose_ring(mol):
    """Identify ring atoms and walk C1→C2→C3→C4→C5, O5."""
    ring_info = mol.GetRingInfo()
    ring_atoms = None
    for ring in ring_info.AtomRings():
        if len(ring) == 6:
            ring_atoms = set(ring)
            break
    if not ring_atoms:
        raise ValueError("No pyranose ring found")

    ring_oxygen = None
    for idx in ring_atoms:
        if mol.GetAtomWithIdx(idx).GetSymbol() == "O":
            ring_oxygen = idx
            break

    o5_neighbors = [
        n.GetIdx() for n in mol.GetAtomWithIdx(ring_oxygen).GetNeighbors()
        if n.GetIdx() in ring_atoms
    ]

    c1_idx = c5_idx = None
    for n in o5_neighbors:
        nbrs = [x.GetIdx() for x in mol.GetAtomWithIdx(n).GetNeighbors()]
        exo_c = [x for x in nbrs if x not in ring_atoms and mol.GetAtomWithIdx(x).GetSymbol() == "C"]
        exo_o = [x for x in nbrs if x not in ring_atoms and mol.GetAtomWithIdx(x).GetSymbol() == "O"]
        if exo_c:
            c5_idx = n
        elif exo_o:
            if c1_idx is None:
                c1_idx = n
            else:
                c5_idx = n
        else:
            c5_idx = n

    if c1_idx is None:
        c1_idx = [n for n in o5_neighbors if n != c5_idx][0]
    if c5_idx is None:
        c5_idx = [n for n in o5_neighbors if n != c1_idx][0]

    ring_walk = [c1_idx]
    visited = {ring_oxygen, c1_idx}
    cur = c1_idx
    for _ in range(4):
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nb.GetIdx() in ring_atoms and nb.GetIdx() not in visited:
                ring_walk.append(nb.GetIdx())
                visited.add(nb.GetIdx())
                cur = nb.GetIdx()
                break

    name_map = {}
    for i, idx in enumerate(ring_walk):
        name_map[idx] = f"C{i+1}"
    name_map[ring_oxygen] = "O5"

    for i, idx in enumerate(ring_walk):
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nb.GetIdx()
            if nidx in ring_atoms or nidx in name_map:
                continue
            if nb.GetSymbol() == "O":
                name_map[nidx] = f"O{i+1}"
            elif nb.GetSymbol() == "C":
                name_map[nidx] = "C6"
                for nn in mol.GetAtomWithIdx(nidx).GetNeighbors():
                    if nn.GetSymbol() == "O" and nn.GetIdx() not in name_map:
                        name_map[nn.GetIdx()] = "O6"

    return name_map, ring_atoms


def assign_h_names_from_rtp(mol, name_map, rtp_atoms):
    """Assign hydrogen names by matching against .rtp atom list."""
    rtp_h_names = [a[0] for a in rtp_atoms if a[1] == "H"]

    parent_h_map = {}
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 1:
            continue
        parent_idx = atom.GetNeighbors()[0].GetIdx()
        parent_name = name_map.get(parent_idx, "?")
        parent_h_map.setdefault(parent_name, []).append(i)

    h_name_assignment = {}
    for parent_name, h_indices in parent_h_map.items():
        if parent_name.startswith("O"):
            rtp_name = f"H{parent_name}"
            if rtp_name in rtp_h_names:
                h_name_assignment[h_indices[0]] = rtp_name
        elif parent_name.startswith("C"):
            cnum = parent_name[1:]
            if len(h_indices) == 1:
                candidate = f"H{cnum}"
                if candidate in rtp_h_names:
                    h_name_assignment[h_indices[0]] = candidate
                else:
                    candidate = f"H{cnum}1"
                    if candidate in rtp_h_names:
                        h_name_assignment[h_indices[0]] = candidate
            else:
                for j, hidx in enumerate(h_indices):
                    candidate = f"H{cnum}{j+1}"
                    if candidate in rtp_h_names:
                        h_name_assignment[hidx] = candidate

    return h_name_assignment


def generate_sugar(name, smiles, rtp_resname, pdb_resname):
    """Generate a sugar PDB file."""
    rtp_atoms = parse_rtp_atoms(os.path.join(FF_DIR, "carb.rtp"), rtp_resname)

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    name_map, _ = walk_pyranose_ring(mol)
    h_names = assign_h_names_from_rtp(mol, name_map, rtp_atoms)

    all_names = {}
    all_names.update(name_map)
    all_names.update(h_names)

    rtp_name_order = [a[0] for a in rtp_atoms]
    rdkit_by_name = {}
    for rdkit_idx, aname in all_names.items():
        rdkit_by_name[aname] = rdkit_idx

    outpath = os.path.join(OUTDIR, f"{name}_charmm.pdb")
    conf = mol.GetConformer()

    with open(outpath, "w") as f:
        f.write(f"REMARK   Beta-D-{name}pyranose for CHARMM36 {rtp_resname}\n")
        f.write(f"REMARK   Generated by build_sugars.py — atom names match carb.rtp exactly\n")

        serial = 1
        for aname in rtp_name_order:
            if aname not in rdkit_by_name:
                print(f"  WARNING: {name}/{rtp_resname}: rtp atom {aname} not mapped!")
                continue
            rdkit_idx = rdkit_by_name[aname]
            pos = conf.GetAtomPosition(rdkit_idx)
            elem = "H" if aname.startswith("H") else ("O" if aname.startswith("O") else "C")
            f.write(
                f"ATOM  {serial:5d} {aname:<4s} {pdb_resname}A   1    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00           {elem}\n"
            )
            serial += 1

        f.write("TER\nEND\n")

    print(f"  {name}: {outpath} ({serial-1} atoms, rtp={rtp_resname}, pdb={pdb_resname})")
    return outpath


if __name__ == "__main__":
    print("Building sugar PDBs with exact CHARMM36 atom naming:\n")

    # Lyxose: uses BXYL topology (identical params), written as BXYL in PDB
    generate_sugar("lyxose",
                   "O[C@@H]1CO[C@H](O)[C@H](O)[C@H]1O",
                   rtp_resname="BXYL", pdb_resname="BXYL")

    # Hexoses: 4-char names fit PDB format
    generate_sugar("glucose",
                   "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
                   rtp_resname="BGLC", pdb_resname="BGLC")

    generate_sugar("mannose",
                   "OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O",
                   rtp_resname="BMAN", pdb_resname="BMAN")

    generate_sugar("galactose",
                   "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@H]1O",
                   rtp_resname="BGAL", pdb_resname="BGAL")

    print("\nDone. All PDBs ready for gmx pdb2gmx.")
