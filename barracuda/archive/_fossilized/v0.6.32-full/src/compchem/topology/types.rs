// SPDX-License-Identifier: AGPL-3.0-or-later
//! Core types for molecular topology bonded interactions.
//!
//! These types are independent of any force field or file format.
//! They carry resolved parameters ready for GPU dispatch.

/// Atom entry from `[ atoms ]` section.
#[derive(Debug, Clone)]
pub struct TopologyAtom {
    /// 1-based atom number (GROMACS `nr`)
    pub nr: u32,
    /// Atom type string (e.g. "CC3162", "HCA1")
    pub atom_type: String,
    /// Residue number
    pub resnr: u32,
    /// Residue name
    pub residue: String,
    /// Atom name
    pub atom_name: String,
    /// Charge group number
    pub cgnr: u32,
    /// Partial charge (e)
    pub charge: f64,
    /// Mass (amu)
    pub mass: f64,
}

/// Bond entry from `[ bonds ]` — connectivity only, funct type preserved.
#[derive(Debug, Clone, Copy)]
pub struct TopologyBond {
    /// 0-based atom indices (converted from 1-based GROMACS)
    pub i: u32,
    pub j: u32,
    /// GROMACS function type (1 = harmonic, 6 = Morse, etc.)
    pub funct: u32,
    /// Optional explicit parameters from the `.top` line.
    /// For funct=1: `[r₀ (nm), k (kJ/mol/nm²)]`
    pub params: [f64; 4],
    /// Number of explicit params provided
    pub n_params: u32,
}

/// Angle entry from `[ angles ]`.
#[derive(Debug, Clone, Copy)]
pub struct TopologyAngle {
    /// 0-based atom indices
    pub i: u32,
    pub j: u32,
    pub k: u32,
    /// GROMACS function type (1 = harmonic, 5 = Urey-Bradley)
    pub funct: u32,
    pub params: [f64; 6],
    pub n_params: u32,
}

/// Dihedral entry from `[ dihedrals ]`.
#[derive(Debug, Clone, Copy)]
pub struct TopologyDihedral {
    /// 0-based atom indices
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub l: u32,
    /// GROMACS function type (1 = proper, 2 = improper, 9 = proper periodic multiple)
    pub funct: u32,
    pub params: [f64; 6],
    pub n_params: u32,
}

/// Molecule type definition.
#[derive(Debug, Clone)]
pub struct MoleculeType {
    pub name: String,
    pub nrexcl: u32,
    pub atoms: Vec<TopologyAtom>,
    pub bonds: Vec<TopologyBond>,
    pub angles: Vec<TopologyAngle>,
    pub dihedrals: Vec<TopologyDihedral>,
}

/// System composition: which molecules and how many.
#[derive(Debug, Clone)]
pub struct MoleculeCount {
    pub name: String,
    pub count: u32,
}

/// Full parsed topology.
#[derive(Debug, Clone)]
pub struct SystemTopology {
    pub molecule_types: Vec<MoleculeType>,
    pub system_name: String,
    pub molecules: Vec<MoleculeCount>,
}

impl SystemTopology {
    /// Total number of atoms in the system.
    pub fn total_atoms(&self) -> usize {
        let mut total = 0usize;
        for mc in &self.molecules {
            if let Some(mt) = self.molecule_types.iter().find(|m| m.name == mc.name) {
                total += mt.atoms.len() * mc.count as usize;
            }
        }
        total
    }

    /// Expand bonded terms across all molecule copies, returning global 0-based indices.
    pub fn expand_bonds(&self) -> Vec<TopologyBond> {
        let mut result = Vec::new();
        let mut atom_offset = 0u32;
        for mc in &self.molecules {
            if let Some(mt) = self.molecule_types.iter().find(|m| m.name == mc.name) {
                let n_atoms = mt.atoms.len() as u32;
                for copy in 0..mc.count {
                    let offset = atom_offset + copy * n_atoms;
                    for b in &mt.bonds {
                        result.push(TopologyBond {
                            i: b.i + offset,
                            j: b.j + offset,
                            ..*b
                        });
                    }
                }
                atom_offset += mc.count * n_atoms;
            }
        }
        result
    }

    /// Expand angle terms across all molecule copies.
    pub fn expand_angles(&self) -> Vec<TopologyAngle> {
        let mut result = Vec::new();
        let mut atom_offset = 0u32;
        for mc in &self.molecules {
            if let Some(mt) = self.molecule_types.iter().find(|m| m.name == mc.name) {
                let n_atoms = mt.atoms.len() as u32;
                for copy in 0..mc.count {
                    let offset = atom_offset + copy * n_atoms;
                    for a in &mt.angles {
                        result.push(TopologyAngle {
                            i: a.i + offset,
                            j: a.j + offset,
                            k: a.k + offset,
                            ..*a
                        });
                    }
                }
                atom_offset += mc.count * n_atoms;
            }
        }
        result
    }

    /// Expand dihedral terms across all molecule copies.
    pub fn expand_dihedrals(&self) -> Vec<TopologyDihedral> {
        let mut result = Vec::new();
        let mut atom_offset = 0u32;
        for mc in &self.molecules {
            if let Some(mt) = self.molecule_types.iter().find(|m| m.name == mc.name) {
                let n_atoms = mt.atoms.len() as u32;
                for copy in 0..mc.count {
                    let offset = atom_offset + copy * n_atoms;
                    for d in &mt.dihedrals {
                        result.push(TopologyDihedral {
                            i: d.i + offset,
                            j: d.j + offset,
                            k: d.k + offset,
                            l: d.l + offset,
                            ..*d
                        });
                    }
                }
                atom_offset += mc.count * n_atoms;
            }
        }
        result
    }
}
