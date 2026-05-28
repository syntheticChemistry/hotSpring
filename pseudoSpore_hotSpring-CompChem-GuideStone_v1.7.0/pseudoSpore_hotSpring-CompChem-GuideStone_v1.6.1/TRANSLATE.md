# Cross-Reference Legend

## Domain ↔ Computation Translation

This GuideStone bridges biochemistry domain concepts with computational implementation.

### Coordinate Systems

| Domain Concept | Computation | Units |
|---------------|-------------|-------|
| Backbone torsion φ | PLUMED TORSION C-N-CA-C | radians |
| Backbone torsion ψ | PLUMED TORSION N-CA-C-N | radians |
| Ring pucker θ (Cremer-Pople) | PLUMED PUCKERING (theta) | radians [0, π] |
| Ring pucker φ (Cremer-Pople) | PLUMED PUCKERING (phi) | radians [0, 2π] |
| Stoddart coords (qx, qy) | PLUMED PUCKERING (qx, qy) | nm |
| HLDA fold discriminant | PLUMED COMBINE of CONTACTMAPs | dimensionless |

### Puckering Convention

| θ Value | Conformation | Biochemistry Name |
|---------|--------------|-------------------|
| 0 | North pole | 4C1 chair (canonical) |
| π/2 | Equator | Boat / skew-boat |
| π | South pole | 1C4 chair (inverted) |

### Atom Index Translation (see index_map.toml)

Free xylose ring: PDB 2D24 serials 6469-6478 → GROMACS 1,5,8,9,13,17
Enzyme-bound ring: PDB 2D24 serials 6599-6607 → GROMACS 6278,6282,6285,6286,6290,6294

### File Format Reference

| Extension | Format | Producer |
|-----------|--------|----------|
| HILLS | PLUMED metadynamics Gaussian history | PLUMED METAD action |
| COLVAR | PLUMED collective variable trajectory | PLUMED PRINT action |
| fes_*.dat | Free energy surface on grid | plumed sum_hills / cazyme-fel |
| Kernels.data | OPES kernel compression file | PLUMED OPES_METAD action |
| *.tpr | GROMACS portable binary run input | gmx grompp |

### Validation Tool Mapping

| Check | Tool | Module |
|-------|------|--------|
| FES basin/barrier detection | nest-validate (Rust native) | 01, 03, 05 |
| OPES reweighting + transition counting | nest-validate (Rust native) | 02 |
| 1D FES parity RMSD | cazyme-fel | 03, 05 |
| 2D FES parity RMSD | cazyme-fel --2d | 04, 06 |
| Statistical convergence | nest-validate stats module | 01, 02 |
| BLAKE3 integrity | nest-validate guidestone verify | all |
