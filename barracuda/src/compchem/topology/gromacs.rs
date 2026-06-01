// SPDX-License-Identifier: AGPL-3.0-or-later
//! GROMACS `.top` / `.itp` file parser.
//!
//! Parses the molecule-level topology (atoms, bonds, angles, dihedrals)
//! from standalone `.top` files. Does not resolve `#include` chains —
//! bonded parameters are either explicit in the `.top` (GROMACS
//! `pdb2gmx -missing` output) or resolved later from force field tables.
//!
//! # Supported sections
//! - `[ moleculetype ]`
//! - `[ atoms ]`
//! - `[ bonds ]` (funct 1 = harmonic)
//! - `[ pairs ]` (parsed but stored separately)
//! - `[ angles ]` (funct 1 = harmonic, 5 = Urey-Bradley)
//! - `[ dihedrals ]` (funct 1/9 = proper periodic, 2 = improper harmonic)
//! - `[ system ]`
//! - `[ molecules ]`

use super::types::*;

/// Errors from parsing GROMACS topology files.
#[derive(Debug)]
pub enum TopologyParseError {
    /// Underlying file I/O failure.
    Io(std::io::Error),
    /// Malformed or unsupported topology content.
    Format(String),
}

impl std::fmt::Display for TopologyParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "topology I/O error: {e}"),
            Self::Format(msg) => write!(f, "topology parse error: {msg}"),
        }
    }
}

impl std::error::Error for TopologyParseError {}

impl From<std::io::Error> for TopologyParseError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Parser for GROMACS topology files.
pub struct GmxTopology;

#[derive(Debug, PartialEq, Eq)]
enum Section {
    None,
    MoleculeType,
    Atoms,
    Bonds,
    Pairs,
    Angles,
    Dihedrals,
    System,
    Molecules,
    PositionRestraints,
    Other(String),
}

impl GmxTopology {
    /// Parse a GROMACS `.top` file from its text content.
    ///
    /// Returns a [`SystemTopology`] with all molecule types, bonded terms,
    /// and system composition. Atom indices are converted to 0-based.
    ///
    /// Skips `#include`, `#ifdef`, `#endif` directives — expects either a
    /// standalone topology or a pre-merged file.
    pub fn parse(content: &str) -> Result<SystemTopology, TopologyParseError> {
        let mut molecule_types: Vec<MoleculeType> = Vec::new();
        let mut system_name = String::new();
        let mut molecules: Vec<MoleculeCount> = Vec::new();
        let mut section = Section::None;

        let mut current_mol: Option<MoleculeType> = None;

        for raw_line in content.lines() {
            let line = strip_comment(raw_line);
            let trimmed = line.trim();

            if trimmed.is_empty()
                || trimmed.starts_with('#')
                || trimmed.starts_with(';')
            {
                continue;
            }

            if let Some(sec) = parse_section_header(trimmed) {
                if sec == Section::MoleculeType {
                    if let Some(mol) = current_mol.take() {
                        molecule_types.push(mol);
                    }
                }
                section = sec;
                continue;
            }

            match section {
                Section::MoleculeType => {
                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                    if parts.len() >= 2 {
                        current_mol = Some(MoleculeType {
                            name: parts[0].to_string(),
                            nrexcl: parts[1].parse().unwrap_or(3),
                            atoms: Vec::new(),
                            bonds: Vec::new(),
                            angles: Vec::new(),
                            dihedrals: Vec::new(),
                        });
                    }
                }
                Section::Atoms => {
                    if let Some(ref mut mol) = current_mol {
                        if let Some(atom) = parse_atom_line(trimmed) {
                            mol.atoms.push(atom);
                        }
                    }
                }
                Section::Bonds => {
                    if let Some(ref mut mol) = current_mol {
                        if let Some(bond) = parse_bond_line(trimmed) {
                            mol.bonds.push(bond);
                        }
                    }
                }
                Section::Angles => {
                    if let Some(ref mut mol) = current_mol {
                        if let Some(angle) = parse_angle_line(trimmed) {
                            mol.angles.push(angle);
                        }
                    }
                }
                Section::Dihedrals => {
                    if let Some(ref mut mol) = current_mol {
                        if let Some(dih) = parse_dihedral_line(trimmed) {
                            mol.dihedrals.push(dih);
                        }
                    }
                }
                Section::System => {
                    if system_name.is_empty() {
                        system_name = trimmed.to_string();
                    }
                }
                Section::Molecules => {
                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                    if parts.len() >= 2 {
                        molecules.push(MoleculeCount {
                            name: parts[0].to_string(),
                            count: parts[1].parse().unwrap_or(1),
                        });
                    }
                }
                _ => {}
            }
        }

        if let Some(mol) = current_mol {
            molecule_types.push(mol);
        }

        Ok(SystemTopology {
            molecule_types,
            system_name,
            molecules,
        })
    }
}

fn strip_comment(line: &str) -> &str {
    match line.find(';') {
        Some(pos) => &line[..pos],
        None => line,
    }
}

fn parse_section_header(line: &str) -> Option<Section> {
    if !line.starts_with('[') || !line.contains(']') {
        return None;
    }
    let inner = line
        .trim_start_matches('[')
        .split(']')
        .next()?
        .trim();
    Some(match inner {
        "moleculetype" => Section::MoleculeType,
        "atoms" => Section::Atoms,
        "bonds" => Section::Bonds,
        "pairs" => Section::Pairs,
        "angles" => Section::Angles,
        "dihedrals" => Section::Dihedrals,
        "system" => Section::System,
        "molecules" => Section::Molecules,
        "position_restraints" => Section::PositionRestraints,
        other => Section::Other(other.to_string()),
    })
}

fn parse_atom_line(line: &str) -> Option<TopologyAtom> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 8 {
        return None;
    }
    Some(TopologyAtom {
        nr: parts[0].parse().ok()?,
        atom_type: parts[1].to_string(),
        resnr: parts[2].parse().ok()?,
        residue: parts[3].to_string(),
        atom_name: parts[4].to_string(),
        cgnr: parts[5].parse().ok()?,
        charge: parts[6].parse().ok()?,
        mass: parts[7].parse().ok()?,
    })
}

fn parse_bond_line(line: &str) -> Option<TopologyBond> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let ai: u32 = parts[0].parse().ok()?;
    let aj: u32 = parts[1].parse().ok()?;
    let funct: u32 = parts[2].parse().ok()?;

    let mut params = [0.0f64; 4];
    let mut n_params = 0u32;
    for (idx, p) in parts[3..].iter().enumerate() {
        if idx >= 4 {
            break;
        }
        if let Ok(v) = p.parse::<f64>() {
            params[idx] = v;
            n_params += 1;
        }
    }

    Some(TopologyBond {
        i: ai - 1,
        j: aj - 1,
        funct,
        params,
        n_params,
    })
}

fn parse_angle_line(line: &str) -> Option<TopologyAngle> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 4 {
        return None;
    }
    let ai: u32 = parts[0].parse().ok()?;
    let aj: u32 = parts[1].parse().ok()?;
    let ak: u32 = parts[2].parse().ok()?;
    let funct: u32 = parts[3].parse().ok()?;

    let mut params = [0.0f64; 6];
    let mut n_params = 0u32;
    for (idx, p) in parts[4..].iter().enumerate() {
        if idx >= 6 {
            break;
        }
        if let Ok(v) = p.parse::<f64>() {
            params[idx] = v;
            n_params += 1;
        }
    }

    Some(TopologyAngle {
        i: ai - 1,
        j: aj - 1,
        k: ak - 1,
        funct,
        params,
        n_params,
    })
}

fn parse_dihedral_line(line: &str) -> Option<TopologyDihedral> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 5 {
        return None;
    }
    let ai: u32 = parts[0].parse().ok()?;
    let aj: u32 = parts[1].parse().ok()?;
    let ak: u32 = parts[2].parse().ok()?;
    let al: u32 = parts[3].parse().ok()?;
    let funct: u32 = parts[4].parse().ok()?;

    let mut params = [0.0f64; 6];
    let mut n_params = 0u32;
    for (idx, p) in parts[5..].iter().enumerate() {
        if idx >= 6 {
            break;
        }
        if let Ok(v) = p.parse::<f64>() {
            params[idx] = v;
            n_params += 1;
        }
    }

    Some(TopologyDihedral {
        i: ai - 1,
        j: aj - 1,
        k: ak - 1,
        l: al - 1,
        funct,
        params,
        n_params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const XYLOSE_TOP_EXCERPT: &str = r#"
[ moleculetype ]
; Name            nrexcl
Other               3

[ atoms ]
;   nr       type  resnr residue  atom   cgnr     charge       mass
     1     CC3162      1   BXYL     C1      1       0.34     12.011
     2       HCA1      1   BXYL     H1      2       0.09      1.008
     3      OC311      1   BXYL     O1      3      -0.65    15.9994
     4       HCP1      1   BXYL    HO1      4       0.42      1.008
     5     CC3263      1   BXYL     C5      5       0.02     12.011

[ bonds ]
;  ai    aj funct
    1     2     1
    1     3     1
    1     5     1

[ angles ]
;  ai    aj    ak funct
    2     1     3     5
    2     1     5     5
    3     1     5     5

[ dihedrals ]
;  ai    aj    ak    al funct
    2     1     3     4     9
    5     1     3     4     9

[ system ]
Xylose in water

[ molecules ]
Other               1
SOL               879
"#;

    #[test]
    fn test_parse_xylose_excerpt() {
        let topo = GmxTopology::parse(XYLOSE_TOP_EXCERPT).unwrap();

        assert_eq!(topo.molecule_types.len(), 1);
        let mol = &topo.molecule_types[0];
        assert_eq!(mol.name, "Other");
        assert_eq!(mol.nrexcl, 3);
        assert_eq!(mol.atoms.len(), 5);
        assert_eq!(mol.bonds.len(), 3);
        assert_eq!(mol.angles.len(), 3);
        assert_eq!(mol.dihedrals.len(), 2);

        assert_eq!(mol.atoms[0].atom_type, "CC3162");
        assert!((mol.atoms[0].charge - 0.34).abs() < 1e-10);
        assert!((mol.atoms[0].mass - 12.011).abs() < 1e-10);

        // Bond 1-2 should be 0-based: (0, 1)
        assert_eq!(mol.bonds[0].i, 0);
        assert_eq!(mol.bonds[0].j, 1);
        assert_eq!(mol.bonds[0].funct, 1);

        // Angle 2-1-3 → (1, 0, 2)
        assert_eq!(mol.angles[0].i, 1);
        assert_eq!(mol.angles[0].j, 0);
        assert_eq!(mol.angles[0].k, 2);
        assert_eq!(mol.angles[0].funct, 5);

        // Dihedral 2-1-3-4 → (1, 0, 2, 3)
        assert_eq!(mol.dihedrals[0].i, 1);
        assert_eq!(mol.dihedrals[0].j, 0);
        assert_eq!(mol.dihedrals[0].k, 2);
        assert_eq!(mol.dihedrals[0].l, 3);
        assert_eq!(mol.dihedrals[0].funct, 9);

        assert_eq!(topo.system_name, "Xylose in water");
        assert_eq!(topo.molecules.len(), 2);
        assert_eq!(topo.molecules[0].name, "Other");
        assert_eq!(topo.molecules[0].count, 1);
        assert_eq!(topo.molecules[1].name, "SOL");
        assert_eq!(topo.molecules[1].count, 879);
    }

    #[test]
    fn test_total_atoms() {
        let topo = GmxTopology::parse(XYLOSE_TOP_EXCERPT).unwrap();
        // Only "Other" has atoms defined (5 atoms × 1 copy). SOL not defined → 0.
        assert_eq!(topo.total_atoms(), 5);
    }

    #[test]
    fn test_expand_bonds_multiple_copies() {
        let top_str = r#"
[ moleculetype ]
Water   2

[ atoms ]
     1     OW      1   SOL     OW      1      -0.834     16.0
     2     HW      1   SOL    HW1      2       0.417      1.008
     3     HW      1   SOL    HW2      3       0.417      1.008

[ bonds ]
    1     2     1
    1     3     1

[ system ]
Water

[ molecules ]
Water   3
"#;
        let topo = GmxTopology::parse(top_str).unwrap();
        assert_eq!(topo.total_atoms(), 9);

        let bonds = topo.expand_bonds();
        assert_eq!(bonds.len(), 6);
        // Copy 0: (0,1), (0,2)
        assert_eq!((bonds[0].i, bonds[0].j), (0, 1));
        assert_eq!((bonds[1].i, bonds[1].j), (0, 2));
        // Copy 1: (3,4), (3,5)
        assert_eq!((bonds[2].i, bonds[2].j), (3, 4));
        assert_eq!((bonds[3].i, bonds[3].j), (3, 5));
        // Copy 2: (6,7), (6,8)
        assert_eq!((bonds[4].i, bonds[4].j), (6, 7));
        assert_eq!((bonds[5].i, bonds[5].j), (6, 8));
    }

    #[test]
    fn test_parse_real_xylose_top() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../control/gromacs_fel/cazyme_gh10_v2/xylose.top"
        );
        let Ok(content) = std::fs::read_to_string(path) else {
            eprintln!("Skipping: {path} not found");
            return;
        };

        let topo = GmxTopology::parse(&content).unwrap();
        assert_eq!(topo.molecule_types.len(), 1, "should have 1 molecule type (Other)");

        let mol = &topo.molecule_types[0];
        assert_eq!(mol.name, "Other");
        assert_eq!(mol.atoms.len(), 20, "xylose BXYL has 20 atoms");
        assert_eq!(mol.bonds.len(), 20, "xylose has 20 bonds");
        assert_eq!(mol.angles.len(), 35, "xylose has 35 angles");
        assert!(mol.dihedrals.len() > 40, "xylose has >40 dihedrals");

        // All bonds should be funct=1 (harmonic) with no explicit params
        for b in &mol.bonds {
            assert_eq!(b.funct, 1);
            assert_eq!(b.n_params, 0);
        }

        // All angles should be funct=5 (Urey-Bradley CHARMM)
        for a in &mol.angles {
            assert_eq!(a.funct, 5);
        }

        // All dihedrals should be funct=9 (proper periodic multiple)
        for d in &mol.dihedrals {
            assert_eq!(d.funct, 9);
        }

        assert_eq!(topo.molecules.len(), 2);
        assert_eq!(topo.molecules[0].count, 1);
        assert_eq!(topo.molecules[1].name, "SOL");
        assert_eq!(topo.molecules[1].count, 879);

        // Total atoms: 20 (xylose) + 879 * 0 (SOL not defined here) = 20
        assert_eq!(topo.total_atoms(), 20);
    }

    #[test]
    fn test_parse_with_comments_and_includes() {
        let content = r#"
; This is a comment
#include "forcefield.itp"

[ moleculetype ]
; Name  nrexcl
Test    3

[ atoms ]
; nr type resnr res atom cgnr charge mass
  1  CT    1    ALA  CA   1   0.0   12.0
  2  HC    1    ALA  HA   2   0.0    1.0

[ bonds ]
  1   2   1   0.109  284512.0  ; explicit params

#ifdef POSRES
[ position_restraints ]
  1  1  1000  1000  1000
#endif

[ system ]
Test system

[ molecules ]
Test  1
"#;
        let topo = GmxTopology::parse(content).unwrap();
        let mol = &topo.molecule_types[0];
        assert_eq!(mol.atoms.len(), 2);
        assert_eq!(mol.bonds.len(), 1);
        assert_eq!(mol.bonds[0].n_params, 2);
        assert!((mol.bonds[0].params[0] - 0.109).abs() < 1e-10);
        assert!((mol.bonds[0].params[1] - 284512.0).abs() < 1e-5);
    }
}
