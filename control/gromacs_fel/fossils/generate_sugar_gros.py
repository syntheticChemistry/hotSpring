#!/usr/bin/env python3
"""Generate initial GRO structures for free sugar epimer FEL calculations.

Uses RDKit 3D embedding then writes GROMACS .gro format with CHARMM36-compatible
atom and residue names. GRO format handles 5-char residue names (e.g. BLYXP)
that overflow PDB's 4-char field.
"""
from rdkit import Chem
from rdkit.Chem import AllChem
import os


SUGARS = {
    "lyxose": {
        "smiles": "O[C@@H]1CO[C@H](O)[C@H](O)[C@H]1O",
        "resname": "BLYXP",
        "desc": "beta-D-Lyxopyranose",
        "is_hexose": False,
    },
    "glucose": {
        "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "resname": "BGLC",
        "desc": "beta-D-Glucopyranose",
        "is_hexose": True,
    },
    "mannose": {
        "smiles": "OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O",
        "resname": "BMAN",
        "desc": "beta-D-Mannopyranose",
        "is_hexose": True,
    },
    "galactose": {
        "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@H]1O",
        "resname": "BGAL",
        "desc": "beta-D-Galactopyranose",
        "is_hexose": True,
    },
}


def identify_ring_atoms(mol):
    """Walk the pyranose ring and return ordered dict: rdkit_idx -> CHARMM_name."""
    ring_info = mol.GetRingInfo()
    ring_atoms = None
    for ring in ring_info.AtomRings():
        if len(ring) == 6:
            ring_atoms = set(ring)
            break
    if ring_atoms is None:
        raise ValueError("No 6-membered ring found")

    ring_oxygen = None
    ring_carbons = []
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "O":
            ring_oxygen = idx
        else:
            ring_carbons.append(idx)

    o5_ring_neighbors = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(ring_oxygen).GetNeighbors()
        if n.GetIdx() in ring_atoms
    ]

    c1_idx = c5_idx = None
    for n in o5_ring_neighbors:
        neighbors = [x.GetIdx() for x in mol.GetAtomWithIdx(n).GetNeighbors()]
        exo_carbons = [
            x for x in neighbors
            if x not in ring_atoms and mol.GetAtomWithIdx(x).GetSymbol() == "C"
        ]
        exo_oxygens = [
            x for x in neighbors
            if x not in ring_atoms and mol.GetAtomWithIdx(x).GetSymbol() == "O"
        ]
        if exo_carbons:
            c5_idx = n
        elif exo_oxygens and c1_idx is None:
            c1_idx = n
        else:
            if c1_idx is not None:
                c5_idx = n
            else:
                c1_idx = n

    if c1_idx is None:
        c1_idx = [n for n in o5_ring_neighbors if n != c5_idx][0]
    if c5_idx is None:
        c5_idx = [n for n in o5_ring_neighbors if n != c1_idx][0]

    ring_order = [c1_idx]
    visited = {ring_oxygen, c1_idx}
    current = c1_idx
    for _ in range(4):
        for n in mol.GetAtomWithIdx(current).GetNeighbors():
            if n.GetIdx() in ring_atoms and n.GetIdx() not in visited:
                ring_order.append(n.GetIdx())
                visited.add(n.GetIdx())
                current = n.GetIdx()
                break

    name_map = {}
    for i, idx in enumerate(ring_order):
        name_map[idx] = f"C{i+1}"
    name_map[ring_oxygen] = "O5"

    for i, idx in enumerate(ring_order):
        for n in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = n.GetIdx()
            if nidx in ring_atoms or nidx in name_map:
                continue
            if n.GetSymbol() == "O":
                name_map[nidx] = f"O{i+1}"
            elif n.GetSymbol() == "C":
                name_map[nidx] = "C6"
                for nn in mol.GetAtomWithIdx(nidx).GetNeighbors():
                    if nn.GetSymbol() == "O" and nn.GetIdx() not in name_map:
                        name_map[nn.GetIdx()] = "O6"

    return name_map, ring_order


def assign_hydrogen_names(mol, name_map):
    """Assign CHARMM hydrogen names based on parent atom."""
    h_count = {}
    h_names = {}
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 1:
            continue
        parent = atom.GetNeighbors()[0]
        parent_name = name_map.get(parent.GetIdx(), "X")

        if parent_name.startswith("O"):
            h_names[i] = f"H{parent_name}"
        elif parent_name.startswith("C"):
            cnum = parent_name[1:]
            key = parent_name
            h_count[key] = h_count.get(key, 0) + 1
            cnt = h_count[key]
            if cnt == 1:
                h_names[i] = f"H{cnum}"
            else:
                h_names[i] = f"H{cnum}{cnt}"
    return h_names


def write_gro(mol, name_map, h_names, resname, desc, filepath):
    """Write GROMACS .gro format structure file."""
    conf = mol.GetConformer()
    natoms = mol.GetNumAtoms()

    all_names = {}
    all_names.update(name_map)
    all_names.update(h_names)

    # GRO format: heavy atoms then hydrogens, each with CHARMM name
    # Reorder: ring C1,C2,C3,C4,C5,O5 then exocyclic, then H
    heavy_order = []
    h_order = []
    for i in range(natoms):
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
            h_order.append(i)
        else:
            heavy_order.append(i)
    atom_order = heavy_order + h_order

    with open(filepath, "w") as f:
        f.write(f"{desc}\n")
        f.write(f"{natoms:5d}\n")
        for gro_idx, rdkit_idx in enumerate(atom_order, 1):
            pos = conf.GetAtomPosition(rdkit_idx)
            aname = all_names.get(rdkit_idx, f"X{rdkit_idx}")
            # GRO positions in nm (RDKit gives Angstroms)
            x, y, z = pos.x / 10.0, pos.y / 10.0, pos.z / 10.0
            f.write(f"{1:5d}{resname:<5s}{aname:>5s}{gro_idx:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"   3.00000   3.00000   3.00000\n")

    print(f"  {filepath}: {natoms} atoms, resname={resname}")


def main():
    outdir = "control/gromacs_fel/cazyme_gh10"
    os.makedirs(outdir, exist_ok=True)

    print("Generating sugar GRO files for CHARMM36 pdb2gmx:")
    for name, info in SUGARS.items():
        mol = Chem.MolFromSmiles(info["smiles"])
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        name_map, ring_order = identify_ring_atoms(mol)
        h_names = assign_hydrogen_names(mol, name_map)

        filepath = os.path.join(outdir, f"{name}_charmm.gro")
        write_gro(mol, name_map, h_names, info["resname"], info["desc"], filepath)

    print("Done.")


if __name__ == "__main__":
    main()
