# Nuclear EOS: HFBTHO → From-Scratch Evolution

## What is HFBTHO?

**Hartree-Fock-Bogoliubov Transformation to Harmonic Oscillator** basis.
A Fortran 90 nuclear structure code from ORNL/LLNL that solves the nuclear
many-body problem using the Skyrme energy density functional.

Published in Computer Physics Communications:
- v1.66 (2005): Stoitsov et al., CPC 167, 43
- v2.00 (2013): Stoitsov et al., CPC 184, 1592
- v3.00 (2017): Perez, Schunck et al., CPC 220, 363

### What it computes

Given Skyrme parameters (t0, t1, t2, t3, x0, x1, x2, x3, α, W0):
1. Constructs the Skyrme energy density functional
2. Solves the HFB equations in an axially-deformed harmonic oscillator basis
3. Finds self-consistent nuclear density distributions
4. Computes binding energies, deformations, radii, and other observables
5. For ~72 nuclei per χ² evaluation (matching UNEDF1 calibration set)

### Why we don't use it

- Source code requires CPC program library access (gated)
- Requires Fortran compiler + LAPACK installation
- Wrapper code from Code Ocean capsule is also gated
- Philosophy: **we do science, not permissions**

## Our From-Scratch Solution

### Level Architecture

| Level | What | Method | Status |
|-------|------|--------|--------|
| 1 | `skyrme_hf.py` | SEMF + nuclear matter properties | ✅ Complete (3.4% mean error) |
| 2 | `skyrme_hfb.py` | Spherical HF+BCS with separate p/n | ✅ Complete (5.5% hybrid, Feb 10 2026) |
| 3 | BarraCUDA target | Axially deformed HFB in WGSL | 🎯 Future |

### Level 2 Details (`skyrme_hfb.py`)

**Physics implemented**:
- Separate proton/neutron single-particle Hamiltonians
- Isospin-dependent Skyrme potential (t0, t3 terms with x0, x3)
- Position-dependent effective mass via T_eff kinetic energy matrix
- BCS pairing with constant-gap approximation (Δ = 12/√A MeV)
- Coulomb direct (numerical Poisson integral) + exchange (Slater approximation)
- Spin-orbit interaction (∝ W0)
- Center-of-mass correction

**Results** (SLy4 parametrization):
- ¹⁰⁰Sn: 858.9 MeV (exp: 824.8, +4.1%) ✅
- ¹³²Sn: 1063.4 MeV (exp: 1102.9, -3.6%) ✅
- ²⁰⁸Pb: 1461.4 MeV (exp: 1636.4, -10.7%)

**What it captures that SEMF doesn't**:
- Shell effects (magic numbers N,Z = 28, 50, 82)
- Isospin dependence (separate p/n potentials)
- Self-consistent density profiles

**What it doesn't capture (Level 3)**:
- Axial deformation (β₂) — needs 2D HO basis
- Full self-consistent pairing — needs HFB gap equation
- Large-basis convergence — needs N_shells ≥ 20
- **This is the BarraCUDA target**: eigensolvers in WGSL

### Level 2 → Level 3 Evolution Path

The Level 2 code demonstrates that the physics WORKS in Python.
The Level 3 evolution replaces the inner loops with BarraCUDA shaders:

| Component | Level 2 (Python) | Level 3 (BarraCUDA) |
|-----------|-----------------|---------------------|
| Basis functions | `numpy` HO radial | `ho_basis.wgsl` |
| Eigenvalue solver | `numpy.linalg.eigh` | `eigensolver.wgsl` (Jacobi/Lanczos) |
| Density construction | `numpy` dot products | `density.wgsl` (GPU parallel) |
| Coulomb integral | `numpy.trapezoid` | `poisson_sphere.wgsl` |
| BCS occupation | `scipy.optimize.brentq` | `bcs_lambda.wgsl` |
| Self-consistency loop | Python iteration | Rust orchestration |

**Key insight**: Level 2 proves the physics is correct. Level 3 proves
BarraCUDA can do the SAME physics faster on ANY hardware.
This is constrained evolution: same math, different substrate.

## Files

- `../wrapper/skyrme_hf.py` — Level 1: nuclear matter + SEMF
- `../wrapper/skyrme_hfb.py` — Level 2: spherical HF+BCS
- `../wrapper/objective.py` — Objective function (χ² against AME2020)
- `../exp_data/ame2020_selected.json` — 52 nuclei experimental data
- `../scripts/run_surrogate.py` — Surrogate learning workflow
