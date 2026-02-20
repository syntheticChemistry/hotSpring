# Physics Model Documentation

**hotSpring — Complete Equation Reference**

This document describes every physics model, equation, and approximation used in the
hotSpring validation suite. It is intended for physics researchers and students
who wish to independently verify our results.

Sections 1-9 cover nuclear EOS physics (Skyrme EDF, HFB, SEMF). Section 10 covers
Yukawa OCP molecular dynamics (Phase C GPU MD). Section 11 covers transport
coefficients (Green-Kubo, Daligault/Stanton-Murillo analytical fits). Section 12
covers lattice QCD (SU(3) pure gauge, Wilson action, HMC, staggered Dirac, HotQCD EOS).
Section 13 covers screened Coulomb bound states (Murillo & Weisheit 1998).
Section 14 covers the Abelian Higgs model (Bazavov et al. 2015).

Nuclear EOS implementations exist in both Python (`control/surrogate/nuclear-eos/wrapper/`) and
Rust (`barracuda/src/physics/`). GPU MD implementations are in Rust (`barracuda/src/md/`)
with f64 WGSL shaders. See `whitePaper/BARRACUDA_SCIENCE_VALIDATION.md` for precision
comparisons.

---

## Table of Contents

1. [Physical Constants](#1-physical-constants)
2. [Skyrme Energy Density Functional](#2-skyrme-energy-density-functional)
3. [Nuclear Matter Properties](#3-nuclear-matter-properties-level-1)
4. [Semi-Empirical Mass Formula](#4-semi-empirical-mass-formula-level-1)
5. [Spherical Hartree-Fock + BCS](#5-spherical-hartree-fock--bcs-level-2)
6. [Objective Function and χ²](#6-objective-function-and-χ²)
7. [Experimental Data](#7-experimental-data)
8. [Approximations and Limitations](#8-approximations-and-limitations)
9. [References](#9-references)
10. [Yukawa OCP Molecular Dynamics](#10-yukawa-ocp-molecular-dynamics-phase-c)
11. [Transport Coefficients](#11-transport-coefficients)
12. [Lattice QCD](#12-lattice-qcd)
13. [Screened Coulomb Bound States](#13-screened-coulomb-bound-states)
14. [Abelian Higgs Model](#14-abelian-higgs-model)

---

## 1. Physical Constants

All constants follow CODATA 2018 [1]. Source: `barracuda/src/physics/constants.rs`,
`control/surrogate/nuclear-eos/wrapper/skyrme_hf.py`.

| Symbol | Value | Unit | Description |
|--------|-------|------|-------------|
| ℏc | 197.3269804 | MeV·fm | Reduced Planck constant × speed of light |
| m_N | 938.918 | MeV/c² | Average nucleon mass (m_p + m_n)/2 |
| m_p | 938.272046 | MeV/c² | Proton mass |
| m_n | 939.565378 | MeV/c² | Neutron mass |
| e² | 1.4399764 | MeV·fm | Coulomb constant e²/(4πε₀) |
| ℏ²/2m_N | 20.735 | MeV·fm² | Derived: (ℏc)²/(2 m_N) |

---

## 2. Skyrme Energy Density Functional

The Skyrme effective interaction is a zero-range density-dependent nuclear force
parametrized by 10 coupling constants. This is the foundation for all three levels.

**Reference**: Chabanat et al., "A Skyrme parametrization from subnuclear to neutron star
densities," Nucl. Phys. A 627, 710 (1997) [2]; Bender, Heenen, and Reinhard,
"Self-consistent mean-field models for nuclear structure," Rev. Mod. Phys. 75, 121 (2003) [3].

### 2.1 Parameter Space

The 10 Skyrme parameters optimized in this work:

| Parameter | Typical Range | Unit | Role |
|-----------|--------------|------|------|
| t₀ | [-3500, -1500] | MeV·fm³ | Central (s-wave) strength |
| t₁ | [200, 600] | MeV·fm⁵ | Effective mass / gradient (p-wave) |
| t₂ | [-700, 0] | MeV·fm⁵ | Tensor / effective mass correction |
| t₃ | [10000, 18000] | MeV·fm^(3+3α) | Density-dependent (3-body) |
| x₀ | [-1, 1] | — | Isospin mixing for t₀ |
| x₁ | [-2, 1] | — | Isospin mixing for t₁ |
| x₂ | [-2, 0] | — | Isospin mixing for t₂ |
| x₃ | [0, 2] | — | Isospin mixing for t₃ |
| α | [0.05, 0.5] | — | Density exponent |
| W₀ | [50, 200] | MeV·fm⁵ | Spin-orbit strength |

Well-known parametrizations used as baselines:
- **SLy4** [2]: t₀ = -2488.91, t₁ = 486.82, t₂ = -546.39, t₃ = 13777.0,
  x₀ = 0.834, x₁ = -0.344, x₂ = -1.0, x₃ = 1.354, α = 1/6, W₀ = 123.0
- **UNEDF0** [4]: t₀ = -1883.69, t₁ = 277.50, t₂ = -189.08, t₃ = 14603.6,
  x₀ = 0.0047, x₁ = -1.116, x₂ = -1.635, x₃ = 0.390, α = 0.3222, W₀ = 78.66

---

## 3. Nuclear Matter Properties (Level 1)

Infinite symmetric nuclear matter (SNM) properties are computed analytically from
Skyrme parameters. These serve two purposes: (a) SEMF coefficient derivation, and
(b) physical constraint filtering.

**Reference**: Bender, Heenen, Reinhard, Rev. Mod. Phys. 75, 121 (2003), §III.A [3].

### 3.1 Energy per Nucleon in Symmetric Nuclear Matter

```
E/A(ρ) = E_kin + E_t0 + E_t3 + E_t1t2
```

where ρ is the total nucleon density (fm⁻³) and:

**Fermi momentum:**
```
k_F = (3π²ρ/2)^(1/3)
```

**Kinetic energy density:**
```
τ = (3/5) k_F² ρ      [fm⁻⁵]
```

**Contributions to E/A:**
```
E_kin  = (ℏ²/2m)(3/5) k_F²                    Free Fermi gas kinetic energy
E_t0   = (3/8) t₀ ρ                            Contact (s-wave) interaction
E_t3   = (1/16) t₃ ρ^(α+1)                     Density-dependent term
E_t1t2 = (1/16) Θ τ                            Momentum-dependent term
```

where `Θ = 3t₁ + t₂(5 + 4x₂)` is the effective-mass combination.

### 3.2 Saturation Density

The saturation density ρ₀ is found by solving:

```
dE/dρ|_{ρ₀} = 0
```

Numerically solved via Brent's method (bisection) in [0.05, 0.30] fm⁻³.
Empirical value: ρ₀ ≈ 0.16 fm⁻³ [3].

### 3.3 Incompressibility

```
K_∞ = 9 ρ₀² (d²(E/A)/dρ²)|_{ρ₀}
```

Computed via centered finite differences. Empirical: K_∞ ≈ 230 ± 20 MeV [5].

### 3.4 Effective Mass

```
m*/m = 1 / (1 + (m_N / 4ℏ²c²) Θ ρ₀)
```

where `Θ = 3t₁ + t₂(5 + 4x₂)`. Empirical: m*/m ≈ 0.7–0.8 [3].

### 3.5 Symmetry Energy

The nuclear symmetry energy J is the cost of converting protons to neutrons:

```
J = J_kin + J_t0 + J_t3 + J_t1t2
```

where:
```
J_kin  = (ℏ²/2m)(k_F₀²)/(3 m*/m)
J_t0   = -(t₀/4)(2x₀ + 1) ρ₀
J_t3   = -(t₃/24)(2x₃ + 1) ρ₀^(α+1)
J_t1t2 = -(1/24) Θ_s τ₀
```

with `Θ_s = t₂(4 + 5x₂) - 3t₁x₁` and `τ₀ = (3/5)k_F₀²ρ₀`.
Empirical: J ≈ 32 ± 2 MeV [6].

### 3.6 Empirical Constraints (NMP Targets)

Used for constraint filtering and multi-objective optimization:

| Property | Symbol | Target | σ | Source |
|----------|--------|--------|---|--------|
| Saturation density | ρ₀ | 0.16 fm⁻³ | 0.005 | Bender et al. [3] |
| Energy/nucleon | E/A | -15.97 MeV | 0.5 | SLy4 baseline [2] |
| Incompressibility | K_∞ | 230 MeV | 20 | Blaizot [5] |
| Effective mass | m*/m | 0.69 | 0.1 | Chabanat et al. [2] |
| Symmetry energy | J | 32 MeV | 2 | Lattimer & Prakash [6] |

---

## 4. Semi-Empirical Mass Formula (Level 1)

The Bethe–Weizsäcker SEMF gives the total binding energy of a nucleus (Z, N)
with A = Z + N.

**Reference**: Evans, *The Atomic Nucleus* (1955) [7]; von Weizsäcker, Z. Phys. 96,
431 (1935).

### 4.1 Binding Energy Formula

```
B(Z,N) = a_v A - a_s A^(2/3) - a_c Z(Z-1)/A^(1/3) - a_a (N-Z)²/A + δ(Z,N)
```

### 4.2 Coefficient Derivation from Nuclear Matter

Unlike textbook SEMF with fitted constants, our coefficients are **derived** from
Skyrme nuclear matter properties (Section 3):

```
a_v = |E/A(ρ₀)|                   Volume term (binding per nucleon)
a_s = 1.1 × a_v                   Surface correction (empirical ratio)
a_c = 3e²/(5r₀)                   Coulomb, with r₀ = (3/(4πρ₀))^(1/3)
a_a = J                            Asymmetry (= symmetry energy)
a_p = 12/√A                        Pairing amplitude
```

The surface-to-volume ratio `a_s/a_v = 1.1` is a standard approximation. More
sophisticated treatments (e.g., Thomas–Fermi) give 1.0–1.2 depending on the
Skyrme parametrization. This could be evolved by deriving a_s from a
semi-infinite nuclear matter calculation.

### 4.3 Pairing Energy

```
δ(Z,N) = +a_p    if Z even and N even   (even-even)
        = 0       if A odd               (odd mass)
        = -a_p    if Z odd and N odd     (odd-odd)
```

The pairing gap amplitude `Δ = 12/√A` MeV is a phenomenological approximation to
the odd-even mass staggering. See Ring and Schuck, *The Nuclear Many-Body Problem*
(2004), §6.2 [8].

---

## 5. Spherical Hartree-Fock + BCS (Level 2)

The Level 2 solver computes binding energies using a self-consistent mean-field
approach with BCS pairing. It is applied to medium-mass nuclei (56 ≤ A ≤ 132)
where the SEMF is insufficient. Outside this range, the SEMF (Level 1) is used.

**Reference**: Ring and Schuck, *The Nuclear Many-Body Problem* (2004) [8];
Bender, Heenen, Reinhard, Rev. Mod. Phys. 75, 121 (2003) [3];
Vautherin and Brink, Phys. Rev. C 5, 626 (1972) [9].

### 5.1 Harmonic Oscillator Basis

Single-particle states are expanded in a spherical harmonic oscillator (HO) basis.

**HO frequency:**
```
ℏω = 41 A^(-1/3) MeV
```

This is the standard parametrization from Bohr and Mottelson, *Nuclear Structure*
Vol. I (1969), §2-4a [10]. It yields reasonable shell structure for medium-mass nuclei.

**Oscillator length:**
```
b = ℏc / √(m_N ℏω)    [fm]
```

**Radial wavefunctions:**
```
R_nl(r) = N_nl (r/b)^l exp(-r²/(2b²)) L_n^(l+1/2)(r²/b²)
```

where L_n^α is the generalized Laguerre polynomial, and N_nl is the normalization
constant ensuring ∫|R_nl|² r² dr = 1.

**Kinetic matrix elements (diagonal):**
```
T_nn = (ℏω/2)(2n + l + 3/2)
```

### 5.2 Mass Range for HFB: 56 ≤ A ≤ 132

The spherical HFB solver is applied only to nuclei with mass number A in the range
[56, 132]. This window is chosen because:

- **Below A = 56**: Nuclei are well-described by the SEMF; HFB with a spherical basis
  has limited added value and may struggle with light deformed nuclei.
- **Above A = 132**: Heavy nuclei (especially rare earths and actinides) are strongly
  deformed and require the axially-deformed solver (Level 3).
- **Within [56, 132]**: This range includes key doubly-magic nuclei (⁵⁶Ni, ⁷⁸Ni,
  ⁹⁰Zr, ¹⁰⁰Sn, ¹³²Sn) and the tin isotopic chain — ideal benchmarks for a
  spherical solver.

### 5.3 Self-Consistent Field (SCF) Iteration

The HF+BCS problem is solved iteratively:

1. Start from pure HO Hamiltonian (kinetic + HO potential)
2. Diagonalize to get single-particle energies and wavefunctions
3. Compute BCS occupation probabilities v²
4. Compute proton and neutron densities ρ_q(r) and kinetic densities τ_q(r)
5. Construct Skyrme mean-field potential U_q(r)
6. Add Coulomb potential (protons only)
7. Build full Hamiltonian H = T_eff + U_Skyrme + U_Coulomb
8. Diagonalize and return to step 3

Convergence criterion: |ΔE| < 10⁻⁶ MeV between iterations.

### 5.4 Densities

**Particle density (isospin q = proton or neutron):**
```
ρ_q(r) = Σ_i (2j_i+1) v²_i |R_i(r)|² / (4π r²)
```

**Kinetic density:**
```
τ_q(r) = Σ_i (2j_i+1) v²_i [(dR_i/dr)² + l_i(l_i+1)/r² R_i²] / (4π)
```

### 5.5 Skyrme Mean-Field Potential

The central Skyrme potential (t₀ + t₃ terms, with isospin structure):

```
U_q(r) = U_t0 + U_t3
```

where (for isospin q with opposite isospin q'):
```
U_t0 = t₀ [(1 + x₀/2)ρ - (1/2 + x₀)ρ_q]

U_t3 = (t₃/12) [(1 + x₃/2)(α+2)ρ^(α+1)
       - (1/2 + x₃)(α ρ^(α-1)(ρ_p² + ρ_n²) + 2ρ^α ρ_q)]
```

with ρ = ρ_p + ρ_n the total density.

### 5.6 Coulomb Potential

**Direct Coulomb** (Poisson integral, protons only):
```
V_C^dir(r) = 4πe² ∫ ρ_p(r') [r_</(r_>) ] r'² dr'
```

where r_< = min(r,r'), r_> = max(r,r'). Computed by radial integration.

**Exchange Coulomb** (Slater approximation):
```
V_C^ex(r) = -e² (3/π)^(1/3) ρ_p(r)^(1/3)
```

**Reference**: Slater, Phys. Rev. 81, 385 (1951) [11].

### 5.7 BCS Pairing (Constant-Gap Approximation)

**Pairing gap:**
```
Δ = 12/√A   MeV
```

This phenomenological formula reproduces the empirical odd-even mass staggering
(Ring and Schuck [8], §6.2). Applied identically to protons and neutrons.

**BCS occupation probabilities:**
```
v²_k = (1/2)(1 - (ε_k - λ)/E_k)
```

where:
```
E_k = √((ε_k - λ)² + Δ²)
```

is the quasiparticle energy and λ is the chemical potential determined by the
particle number constraint:
```
Σ_k (2j_k + 1) v²_k = N_q    (number of protons or neutrons)
```

The chemical potential λ is found by Brent's method (root-finding).

### 5.8 Spin-Orbit Coupling

The spin-orbit contribution to single-particle energies uses the standard form:
```
⟨ls⟩ = [j(j+1) - l(l+1) - 3/4] / 2
```

The spin-orbit potential strength W₀ enters the Hamiltonian through the radial
density derivative (∇ρ · ∇) contracted with the angular momentum operator.

### 5.9 Binding Energy

The total binding energy is:

```
E_HF = Σ_i (2j_i+1) v²_i ε_i         Single-particle energy sum
     - 1/2 ∫ [U_Skyrme ρ + V_C ρ_p] dV  Double-counting correction
     + E_pair                             Pairing correlation energy
     + E_CM                               Center-of-mass correction
```

**Center-of-mass correction** (harmonic oscillator estimate):
```
E_CM = -3/4 ℏω = -(3/4)(41 A^(-1/3)) MeV
```

This removes the spurious center-of-mass kinetic energy from the total. The
coefficient 3/4 is the standard HO approximation (Bohr and Mottelson [10],
§4-2).

### 5.10 Focused Nuclei

For efficiency, L2 evaluates a subset of 18 nuclei chosen to span the physics:

**Light (A < 56, SEMF only):** ⁴He, ¹⁶O, ⁴⁰Ca, ⁴⁸Ca

**Medium (56 ≤ A ≤ 132, HFB):** ⁵⁶Ni*, ⁵⁸Ni, ⁶²Ni, ⁷⁸Ni*, ⁹⁰Zr, ¹⁰⁰Sn*, ¹¹²Sn, ¹²⁰Sn, ¹²⁴Sn, ¹³²Sn*

**Heavy (A > 132, SEMF only):** ¹⁴⁰Ce, ¹⁵²Sm, ²⁰⁸Pb, ²³⁸U

*Asterisks denote doubly-magic nuclei (closed proton and neutron shells).*

---

## 6. Objective Function and χ²

### 6.1 Chi-Squared Per Datum

The primary objective for Skyrme parameter optimization:

```
χ²/datum = (1/N) Σᵢ [(B_calc(Zᵢ,Nᵢ) - B_exp(Zᵢ,Nᵢ)) / σ_theo]²
```

where:
- B_calc is the calculated binding energy (SEMF or HFB)
- B_exp is the AME2020 experimental value [12]
- N is the number of nuclei evaluated
- σ_theo is the theoretical uncertainty (see below)

### 6.2 Theoretical Uncertainty

We use a theoretical uncertainty rather than the experimental one:

```
σ_theo = max(0.01 × B_exp, 2.0 MeV)
```

**Rationale:** Experimental uncertainties for known nuclei are σ_exp ~ 0.001 MeV (keV
precision), but the SEMF is inherently a ~1% model. Using σ_exp would produce
χ² ~ 10⁸, making the landscape numerically unnavigable. The theoretical uncertainty
reflects the intrinsic accuracy of the model class, giving χ² ~ 1–10 for physical
parametrizations.

The 2 MeV floor prevents light nuclei (small B) from dominating the fit.

### 6.3 Physical Constraint Penalties

Soft penalty terms enforce physical nuclear matter properties:

```
penalty += 50 × (0.08 - ρ₀)/0.08    if ρ₀ < 0.08 fm⁻³
penalty += 50 × (ρ₀ - 0.25)/0.25    if ρ₀ > 0.25 fm⁻³
penalty += 20 × max(0, E/A + 5)     if E/A > -5 MeV (barely bound)
```

These are gradient-friendly (proportional, not hard walls) for optimizer convergence.

### 6.4 Log-Transformed Objective

For surrogate learning, we use:

```
f(x) = log(1 + χ²/datum)
```

This compresses the dynamic range (~4.6:1 instead of ~1700:1), improving RBF
surrogate accuracy in the 10-dimensional parameter space.

---

## 7. Experimental Data

### 7.1 Source

**AME2020** — Atomic Mass Evaluation 2020 [12]:
- Wang, M. et al., "The AME 2020 atomic mass evaluation (II)," Chinese Physics C 45, 030003 (2021)
- Huang, W.J. et al., "The AME 2020 atomic mass evaluation (I)," Chinese Physics C 45, 030002 (2021)
- Source: https://www-nds.iaea.org/amdc/ame2020/
- Downloaded and parsed by `control/surrogate/nuclear-eos/exp_data/download_ame2020.sh`

### 7.2 Selection Criteria

The full AME2020 database contains ~3400 masses. We select 52 nuclei spanning:
- Z from 2 (He) to 92 (U)
- N from 2 to 146
- Both stable and key radioactive species
- All doubly-magic nuclei
- The complete Sn (Z=50) isotopic chain from N=50 to N=82

Selection criteria:
- Experimental uncertainty < 0.1 MeV
- Coverage of magic numbers (2, 8, 20, 28, 50, 82, 126)
- Isotopic diversity (light, medium, heavy regions)

### 7.3 Data Format

Stored as `exp_data/ame2020_selected.json`:

```json
{
  "source": "AME2020",
  "reference": "Wang et al., Chinese Physics C 45, 030003 (2021)",
  "nuclei": [
    {"Z": 2, "N": 2, "A": 4, "symbol": "He",
     "binding_energy_MeV": 28.296, "uncertainty_MeV": 0.000}
  ]
}
```

All energies in MeV. Binding energies are positive (bound states).

---

## 8. Approximations and Limitations

### 8.1 Known Approximations

| Approximation | Where | Impact | Path to improvement |
|---------------|-------|--------|-------------------|
| Constant-gap BCS (Δ = 12/√A) | L2, L3 | Misses shell effects in pairing | State-dependent pairing (HFB) |
| Spherical symmetry | L2 | Cannot describe deformed nuclei | Axially-deformed HFB (L3) |
| Surface ratio a_s/a_v = 1.1 | L1 SEMF | ~5% error in surface term | Semi-infinite NM calculation |
| CM correction = -3/4 ℏω | L2 | ~1 MeV for A~100 | Projection method |
| Coulomb Slater exchange | L2 | ~0.5% of Coulomb energy | Exact exchange (expensive) |
| t₁/t₂ only in E/A, not in U | L2 | Missing gradient terms | Full Skyrme functional |
| HO basis truncation | L2 | Sensitive to n_shells choice | Larger bases / THO |
| No tensor force | All | Missing spin-isospin effects | Tensor Skyrme terms |

### 8.2 Accuracy Benchmarks

**L1 (SEMF)**:

| Method | χ²/datum | Evals | Runtime | Notes |
|--------|---------|-------|---------|-------|
| L1 SLy4 baseline (Python=CPU=GPU) | 4.99 | 100k | 114s / 7.3s / 4.0s | Implementation parity: identical on all substrates |
| Python L1 (mystic optimizer) | 6.62 | 1,008 | 184s | scipy LHS + Nelder-Mead |
| BarraCUDA L1 (DirectSampler) | **2.27** | 6,028 | **2.3s** | **478× faster**, better χ² |
| BarraCUDA L1 GPU (extended) | **1.52** | 48 | 32.4s | GPU-accelerated objective eval |

**L2 (HFB) — Evolution trajectory**:

| Stage | χ²/datum | Evals | Config | Validates |
|-------|---------|-------|--------|-----------|
| Initial (missing physics) | 28,450 | — | — | Baseline without Coulomb/BCS/CM |
| +5 physics features | ~92 | — | — | Physics implementation |
| +gradient_1d boundary fix | ~25 | — | — | 2nd-order stencils critical in SCF |
| +brent root-finding | ~18 | — | — | Root-finder precision ×10⁹ |
| **Run A** (best accuracy) | **16.11** | 60 | seed=42, λ=0.1 | Best BarraCUDA L2 achieved |
| **Run B** (best NMP) | **19.29** | 60 | seed=123, λ=1.0 | All 5 NMP within 2σ |
| GPU benchmark DirectSampler | 23.09 | 12 | energy-profiled | GPU cost: 32,500 J at 135W |
| Extended ref run | 25.43 | 1,009 | different seed/λ | Landscape is multimodal |
| **Python SparsitySampler** | **1.93** | 3,008 | mystic, 3.2h | Best L2 overall (sampling advantage) |

The 1,764× improvement from 28,450 to 16.11 came through four validation-driven
cycles. The range of L2 values (16–25) across configurations confirms the 10D
Skyrme landscape is multimodal — more evaluations do not guarantee better χ² without
the right sampling strategy. Python's SparsitySampler (1.93) outperforms BarraCUDA's
DirectSampler because it explores the landscape more efficiently, not because the
physics implementation differs. Porting SparsitySampler is the #1 L2 priority.

**GPU FP64 validation**: All substrates produce identical physics at the SLy4 reference
point (chi2 = 4.99). GPU precision: max |B_cpu - B_gpu| = 4.55e-13 MeV (sub-ULP).
GPU uses 44.8× less energy than Python for the same computation.

---

## 9. References

[1] Tiesinga, E. et al., "CODATA recommended values of the fundamental physical
    constants: 2018," Rev. Mod. Phys. 93, 025010 (2021).

[2] Chabanat, E. et al., "A Skyrme parametrization from subnuclear to neutron star
    densities," Nucl. Phys. A 627, 710 (1997) [Part I: SLy4 definition];
    Nucl. Phys. A 635, 231 (1998) [Part II: nuclei far from stabilities].

[3] Bender, M., Heenen, P.-H., and Reinhard, P.-G., "Self-consistent mean-field models
    for nuclear structure," Rev. Mod. Phys. 75, 121 (2003).

[4] Kortelainen, M. et al., "Nuclear energy density optimization," Phys. Rev. C 82,
    024313 (2010).

[5] Blaizot, J.-P., "Nuclear compressibilities," Phys. Rep. 64, 171 (1980).

[6] Lattimer, J. M. and Prakash, M., "The equation of state of hot, dense matter and
    neutron stars," Phys. Rep. 621, 127 (2016).

[7] von Weizsäcker, C. F., "Zur Theorie der Kernmassen," Z. Phys. 96, 431 (1935);
    Bethe, H. A. and Bacher, R. F., "Nuclear Physics A. Stationary States of Nuclei,"
    Rev. Mod. Phys. 8, 82 (1936).

[8] Ring, P. and Schuck, P., *The Nuclear Many-Body Problem*, Springer (2004).

[9] Vautherin, D. and Brink, D. M., "Hartree-Fock calculations with Skyrme's
    interaction. I. Spherical nuclei," Phys. Rev. C 5, 626 (1972).

[10] Bohr, A. and Mottelson, B. R., *Nuclear Structure*, Vol. I, Benjamin (1969).

[11] Slater, J. C., "A simplification of the Hartree-Fock method," Phys. Rev. 81,
     385 (1951).

[12] Wang, M. et al., "The AME 2020 atomic mass evaluation (II). Tables, graphs and
     references," Chinese Physics C 45, 030003 (2021).

[13] Diaw, A. et al., "Efficient learning of accurate surrogates for simulations of
     complex systems," Nature Machine Intelligence 6, 568 (2024).

---

## 10. Yukawa OCP Molecular Dynamics (Phase C)

### 10.1 Yukawa Potential

The screened Coulomb (Yukawa) interaction between two like charges in a one-component plasma:

```
V(r) = (Gamma / r) * exp(-kappa * r)
```

where all quantities are in OCP reduced units:
- Length scaled by Wigner-Seitz radius: a_ws = (3 / 4*pi*n)^(1/3)
- Time scaled by inverse plasma frequency: omega_p = sqrt(4*pi*n*q^2 / m)
- Gamma = q^2 / (a_ws * k_B * T) is the Coulomb coupling parameter
- kappa = a_ws / lambda_D is the dimensionless screening parameter

### 10.2 Force Computation

The pairwise force on particle i from particle j:

```
F_ij(r) = -(Gamma / r^2) * (1 + kappa * r) * exp(-kappa * r) * r_hat
```

Negative sign indicates repulsion for like charges. Total force on particle i is the sum over all j != i (all-pairs, O(N^2)) or over neighbors within cutoff r_c (cell-list, O(N)).

### 10.3 Periodic Boundary Conditions (Minimum Image Convention)

For a cubic box of side L:

```
dx = x_j - x_i
dx = dx - L * round(dx / L)
```

Applied to each Cartesian component. Only the nearest image of each particle contributes to the force.

### 10.4 Velocity-Verlet Integrator (Split Form)

The symplectic Velocity-Verlet algorithm, split into three GPU dispatches:

**Half-kick + drift**:
```
v(t + dt/2) = v(t) + (F(t) / m) * dt/2
x(t + dt)   = x(t + dt/2) + v(t + dt/2) * dt
x(t + dt)   = x(t + dt) mod L              [PBC wrap]
```

**Force computation**: Compute F(t + dt) from updated positions.

**Second half-kick**:
```
v(t + dt) = v(t + dt/2) + (F(t + dt) / m) * dt/2
```

Reduced mass m* = 3.0 in OCP units. Timestep dt = 0.001 (in units of omega_p^-1).

### 10.5 Berendsen Thermostat

During equilibration, velocities are rescaled to approach target temperature:

```
lambda = sqrt(1 + (dt / tau) * (T_target / T_current - 1))
v_i = lambda * v_i
```

where tau is the coupling constant (tau = 10*dt). Applied only during equilibration; removed for production (NVE) runs.

### 10.6 Temperature

Kinetic temperature from the equipartition theorem:

```
T = (2 / 3N) * sum_i (m * v_i^2 / 2)
```

In OCP reduced units with m* = 3.0, the target temperature T* = 1/Gamma.

### 10.7 Observables

**Radial Distribution Function g(r)**:
```
g(r) = (V / N^2) * <sum_{i<j} delta(r - r_ij)> / (4*pi*r^2*dr)
```
Computed from GPU-generated pair-distance histograms. Physical expectations: g(r) -> 1 at large r; peak height increases with Gamma (stronger coupling = more local order).

**Velocity Auto-Correlation Function (VACF)**:
```
Z(t) = <v(0) . v(t)> / <v(0) . v(0)>
```
Computed from stored velocity snapshots. Self-diffusion coefficient: D* = (1/3) * integral_0^inf Z(t) dt.

**Static Structure Factor S(k)**:
```
S(k) = (1/N) * |sum_j exp(i*k.r_j)|^2
```
Computed for discrete k-vectors compatible with periodic boundaries: k = 2*pi*n/L.

**Energy Conservation**:
```
E_total = KE + PE = sum_i (m*v_i^2/2) + (1/2)*sum_{i!=j} V(r_ij)
```
In NVE production, E_total is conserved. Energy drift = |E_final - E_initial| / |E_initial|.

### 10.8 References (Yukawa MD)

[14] Murillo, M. S. and Dharma-wardana, M. W. C., "Dense plasmas, screened interactions,
     and quantum effects," Phys. Rev. E 98, 023202 (2018).

[15] Stanton, L. G. and Murillo, M. S., "Unified description of linear screening in
     dense plasmas," Phys. Rev. E 91, 033104 (2015).

[16] Hamaguchi, S., Farouki, R. T., and Dubin, D. H. E., "Triple point of Yukawa
     systems," Phys. Rev. E 56, 4671 (1997).

[17] Allen, M. P. and Tildesley, D. J., *Computer Simulation of Liquids*, 2nd ed.,
     Oxford University Press (2017).

---

## 11. Transport Coefficients

Transport coefficients quantify how momentum, energy, and particles diffuse through
a plasma. hotSpring implements both molecular-dynamics (Green-Kubo) extraction and
analytical fit models from the literature.

**References**:
- Daligault, J., "Practical model for the self-diffusion coefficient in Yukawa
  one-component plasmas," Phys. Rev. E **86**, 047401 (2012) [18].
- Stanton, L. G. and Murillo, M. S., "Ionic transport in high-energy-density
  matter," Phys. Rev. E **93**, 043203 (2016) [19].
- Ohta, H. and Hamaguchi, S., "Molecular dynamics evaluation of self-diffusion in
  Yukawa systems," Phys. Plasmas **7**, 4506 (2000) [20].

### 11.1 Green-Kubo Relations

Transport coefficients are extracted from equilibrium MD via time-correlation functions.

**Self-diffusion coefficient** (from velocity autocorrelation):

```
D* = (1/3) integral_0^inf <v(0) . v(t)> dt
```

Equivalently from mean-squared displacement (Einstein relation):

```
D* = lim_{t->inf} (1/6t) <|r(t) - r(0)|^2>
```

Both methods are implemented. MSD serves as a cross-check on the VACF integral.

**Shear viscosity** (from off-diagonal stress autocorrelation):

```
η* = (V / k_B T) integral_0^inf <σ_xy(0) σ_xy(t)> dt
```

where σ_xy is the off-diagonal element of the microscopic stress tensor:

```
σ_αβ = (1/V) [ sum_i m v_iα v_iβ + (1/2) sum_{i≠j} r_ijα F_ijβ ]
```

**Thermal conductivity** (from heat current autocorrelation):

```
λ* = (V / k_B T²) integral_0^inf <J_q(0) . J_q(t)> dt
```

where the heat current is:

```
J_q = sum_i [e_i v_i + (1/2) sum_{j≠i} (F_ij . v_i) r_ij]
```

All Green-Kubo integrals use plateau detection to stop accumulation when the running
integral peaks, preventing noise from corrupting the result at long lag times.

### 11.2 Daligault (2012) Self-Diffusion Model

The practical model [18] interpolates between two asymptotic regimes.

**Weak coupling** (Landau-Spitzer):

```
D*_w = C_w(κ) × (3√π / 4) × Γ^(-5/2) / ln(Λ)
```

**Strong coupling** (Einstein frequency):

```
D*_s = A(κ) × Γ^(-α(κ))
```

**Crossover**:

```
D* = f × D*_w + (1 - f) × D*_s
f(Γ) = 1 / (1 + (Γ / Γ_x)^2)
Γ_x(κ) = 10 × exp(κ/2)
```

**Coulomb logarithm**:

```
ln(Λ) = ln(1/Γ_eff)   for Γ_eff < 0.1   (clamped ≥ 1.0)
ln(Λ) = ln(1 + 1/Γ_eff)   for Γ_eff ≥ 0.1   (clamped ≥ 0.1)
Γ_eff = Γ × exp(-κ)
```

**Calibrated coefficients** (recalibrated Feb 2026 using 12 Sarkas Green-Kubo
D* values at N=2000; original Daligault Table I coefficients were ~70× too small
due to reduced-unit normalization mismatch between the Python baseline convention
and standard OCP units):

```
C_w(κ) = exp(1.435 + 0.715κ + 0.401κ²)

A(κ)     = 0.808 + 0.423κ − 0.152κ²
α(κ)     = 1.049 + 0.044κ − 0.039κ²
```

The κ-dependent weak-coupling correction `C_w(κ)` replaces the earlier constant
`C_w=5.3` (v0.5.13). Yukawa screening suppresses the effective Coulomb logarithm
faster than the classical formula captures; the correction grows exponentially
with κ (4.2× at κ=0, 13× at κ=1, 87× at κ=2, 1325× at κ=3). With `C_w(κ)`,
crossover-regime errors drop from 44–63% to <10% across all 12 Sarkas calibration
points. Fitted from `calibrate_daligault_fit.py` weak-coupling correction analysis.

Source: `barracuda/src/md/transport.rs`, `control/sarkas/simulations/transport-study/scripts/calibrate_daligault_fit.py`.

### 11.3 Stanton & Murillo (2016) Viscosity and Thermal Conductivity

The practical transport models [19] use the same crossover structure as D*.

**Shear viscosity**:

```
η*_w = C_w(κ) × (5√π / 16) × Γ^(-5/2) / ln(Λ)
η*_s = A_η(κ) × Γ^(-α_η(κ))

A_η(κ)   = 0.44 + 0.23κ − 0.083κ²
α_η(κ)   = 0.76 + 0.040κ − 0.036κ²
```

**Thermal conductivity**:

```
λ*_w = C_w(κ) × (75√π / 64) × Γ^(-5/2) / ln(Λ)
λ*_s = A_λ(κ) × Γ^(-α_λ(κ))

A_λ(κ)   = 1.04 + 0.54κ − 0.19κ²
α_λ(κ)   = 0.90 + 0.042κ − 0.035κ²
```

The η* and λ* strong-coupling coefficients are proportionally rescaled from the D*
recalibration (same reduced-unit normalization fix). They have not been independently
calibrated against stress ACF or heat ACF data at N≥2000.

### 11.4 Reduced Units for Transport

All transport quantities use standard OCP reduced units:

```
D*  = D / (a_ws² ω_p)
η*  = η / (n m a_ws² ω_p)
λ*  = λ / (n k_B a_ws² ω_p)
```

where a_ws is the Wigner-Seitz radius, ω_p is the plasma frequency, n is the
number density, and m is the ion mass. The plasma frequency convention is:

```
ω_p² = n q² / (ε₀ m) = 3 q² / (4π ε₀ m a_ws³)
```

### 11.5 Validation Strategy

The transport pipeline is validated against:

1. **Internal consistency**: MSD-based D* agrees with VACF-based D* within 50%
2. **Energy conservation**: NVE drift < 5%
3. **Temperature stability**: T* fluctuations < 30% of target
4. **Analytical fits**: D* MSD agrees with Sarkas-calibrated Daligault fit within 80%
   (80% tolerance accounts for N=500 statistical noise in the validation binary)
5. **Physical ordering**: D*(κ=2) < D*(κ=1) at matched Γ (stronger screening
   increases effective coupling, reducing diffusion)
6. **Positivity**: All D*, η*, λ* must be positive and finite

Source: `barracuda/src/bin/validate_stanton_murillo.rs` (13/13 checks pass).

### 11.6 References (Transport)

[18] Daligault, J., "Practical model for the self-diffusion coefficient in Yukawa
     one-component plasmas," Phys. Rev. E 86, 047401 (2012).

[19] Stanton, L. G. and Murillo, M. S., "Ionic transport in high-energy-density
     matter," Phys. Rev. E 93, 043203 (2016).

[20] Ohta, H. and Hamaguchi, S., "Molecular dynamics evaluation of self-diffusion in
     Yukawa systems," Phys. Plasmas 7, 4506 (2000).

---

## 12. Lattice QCD

hotSpring implements SU(3) pure gauge lattice field theory on CPU, validated
and ready for GPU promotion. This section covers the mathematical foundations
implemented in `barracuda/src/lattice/`.

**References**:
- Wilson, K. G., "Confinement of quarks," Phys. Rev. D **10**, 2445 (1974) [21].
- Bazavov, A. et al., "Equation of state in (2+1)-flavor QCD," Phys. Rev. D
  **90**, 094503 (2014) [22].
- Creutz, M., *Quarks, Gluons and Lattices*, Cambridge University Press (1983) [23].
- Gattringer, C. and Lang, C. B., *Quantum Chromodynamics on the Lattice*,
  Springer (2010) [24].

### 12.1 SU(3) Gauge Theory on the Lattice

The fundamental degrees of freedom are SU(3) link variables U_μ(x) — 3×3
unitary matrices with det = 1 — living on the links of a 4-dimensional
hypercubic lattice. Each link connects site x to site x + μ̂.

**SU(3) matrix properties:**
```
U† U = 1     (unitarity)
det(U) = 1   (special)
```

The link variables are elements of the Lie group SU(3), which has 8 generators
(Gell-Mann matrices λ_a, a=1..8). The corresponding Lie algebra su(3) consists
of traceless anti-Hermitian 3×3 matrices: H† = -H, Tr(H) = 0.

### 12.2 Wilson Gauge Action

The simplest gauge-invariant lattice action uses the plaquette — the smallest
closed loop on the lattice:

```
U_P(x,μ,ν) = U_μ(x) U_ν(x+μ̂) U_μ†(x+ν̂) U_ν†(x)
```

The Wilson action is:

```
S_W = β Σ_{x,μ<ν} [1 - (1/N_c) Re Tr U_P(x,μ,ν)]
```

where β = 2 N_c / g² is the inverse coupling (N_c = 3 for QCD) and g is the
bare gauge coupling constant.

**Plaquette expectation value**: In the weak-coupling (β → ∞) limit,
⟨P⟩ = ⟨(1/N_c) Re Tr U_P⟩ → 1. At finite coupling, the perturbative
prediction is ⟨P⟩ ≈ 1 - d(N_c²-1)/(4N_c β) + O(1/β²), where d=4
is the spacetime dimension. For β=6.0 (standard benchmark), this gives
⟨P⟩ ≈ 0.593.

### 12.3 Staples and Force

The derivative of the Wilson action with respect to U_μ(x) involves the
sum over all plaquettes containing that link. The **staple** Σ_μ(x) is:

```
Σ_μ(x) = Σ_{ν≠μ} [U_ν(x+μ̂) U_μ†(x+ν̂) U_ν†(x) + U_ν†(x+μ̂-ν̂) U_μ†(x-ν̂) U_ν(x-ν̂)]
```

The gauge force (for HMC molecular dynamics) is:

```
F_μ(x) = -(β/2N_c) × ta(U_μ(x) Σ_μ(x))
```

where ta(M) extracts the traceless anti-Hermitian part: ta(M) = (M - M†)/2 - Tr(M - M†)/(2N_c).

### 12.4 Hybrid Monte Carlo (HMC)

HMC generates gauge configurations by augmenting the system with conjugate
momenta and performing Hamiltonian dynamics:

1. **Momentum refresh**: Draw P_μ(x) from Gaussian distribution (su(3)-valued)
2. **Leapfrog integration**: Evolve (U, P) using Hamilton's equations
3. **Metropolis accept/reject**: Accept with probability min(1, exp(-ΔH))

**Hamiltonian:**
```
H[U, P] = (1/2) Σ_{x,μ} Tr(P_μ(x)²) + S_W[U]
```

**Leapfrog steps:**
```
P(τ + ε/2)   = P(τ) - (ε/2) F[U(τ)]
U(τ + ε)     = exp(ε P(τ+ε/2)) U(τ)
P(τ + ε)     = P(τ + ε/2) - (ε/2) F[U(τ+ε)]
```

**Cayley approximation for group exponential:**
```
exp(εP) ≈ (1 + εP/2)(1 - εP/2)^{-1}
```

This preserves unitarity exactly (the Cayley transform maps anti-Hermitian
matrices to unitary matrices). Implementation in `hmc.rs` uses explicit
3×3 matrix inverse via the adjugate formula.

### 12.5 Staggered Dirac Operator

The staggered (Kogut-Susskind) fermion formulation replaces spinor degrees
of freedom with staggered phases:

```
(D_st χ)(x) = (m/2) χ(x) + (1/2) Σ_μ η_μ(x) [U_μ(x) χ(x+μ̂) - U_μ†(x-μ̂) χ(x-μ̂)]
```

where η_μ(x) = (-1)^{x_0 + ... + x_{μ-1}} are the staggered sign factors
and m is the bare quark mass. Each component of χ is a color 3-vector.

### 12.6 Conjugate Gradient Solver

The Dirac equation D†D χ = b is solved by the conjugate gradient method,
since D†D is Hermitian positive-definite:

```
r₀ = b - D†D χ₀
p₀ = r₀
α_k = ⟨r_k, r_k⟩ / ⟨p_k, D†D p_k⟩
χ_{k+1} = χ_k + α_k p_k
r_{k+1} = r_k - α_k D†D p_k
β_k = ⟨r_{k+1}, r_{k+1}⟩ / ⟨r_k, r_k⟩
p_{k+1} = r_{k+1} + β_k p_k
```

Convergence criterion: |r_k|² / |b|² < tolerance (typically 1e-10).

### 12.7 Deterministic PRNG for Lattice

All random number generation uses a Linear Congruential Generator (Knuth MMIX
variant) for bitwise determinism across runs:

```
seed_{n+1} = (a × seed_n + c) mod 2^64
a = 6364136223846793005     (Knuth MMIX multiplier)
c = 1442695040888963407     (Knuth MMIX increment)
```

**Uniform [0, 1):** u = (seed >> 11) / 2^53 (53-bit precision).

**Gaussian (Box-Muller):** z = sqrt(-2 ln u₁) cos(2π u₂), where u₁, u₂
are independent uniform deviates. The ln argument is clamped to 1e-30 to
avoid ln(0).

All lattice modules share these constants and helpers via `lattice/constants.rs`.

### 12.8 HotQCD Equation of State

The HotQCD collaboration's EOS tables (Bazavov et al. 2014 [22]) provide
thermodynamic quantities as functions of temperature. hotSpring validates:

- **Trace anomaly** (I/T⁴): peaks near the crossover temperature T_c ≈ 155 MeV
- **Pressure** (p/T⁴): monotonically increasing, approaching Stefan-Boltzmann
  limit at high T
- **Entropy** (s/T³): related to p/T⁴ by thermodynamic identity
- **Speed of sound** (c_s²): approaches 1/3 (conformal limit) at high T

### 12.9 References (Lattice QCD)

[21] Wilson, K. G., "Confinement of quarks," Phys. Rev. D 10, 2445 (1974).

[22] Bazavov, A. et al. (HotQCD Collaboration), "Equation of state in (2+1)-flavor
     QCD," Phys. Rev. D 90, 094503 (2014). DOI: 10.1103/PhysRevD.90.094503.

[23] Creutz, M., *Quarks, Gluons and Lattices*, Cambridge University Press (1983).

[24] Gattringer, C. and Lang, C. B., *Quantum Chromodynamics on the Lattice: An
     Introductory Presentation*, Springer (2010).

---

## 13. Screened Coulomb Bound States

Paper 6: Murillo & Weisheit, Physics Reports 302, 1-65 (1998).

Source: `barracuda/src/physics/screened_coulomb.rs`.

### 13.1 Radial Schrödinger Equation

The screened Coulomb (Yukawa) potential for an electron bound to an ion:

```
V(r) = -Z exp(-κr) / r
```

where Z is the nuclear charge and κ is the screening parameter. The radial
Schrödinger equation for angular momentum quantum number l:

```
[-½ d²/dr² + l(l+1)/(2r²) + V(r)] R(r) = E R(r)
```

### 13.2 Finite-Difference Discretization

Uniform grid r_i = (i+1)h, h = r_max/(N+1), i = 0, ..., N-1.

Tridiagonal Hamiltonian:
```
H_{ii}    = 1/h² + l(l+1)/(2r_i²) - Z exp(-κr_i)/r_i
H_{i,i±1} = -1/(2h²)
```

### 13.3 Sturm Bisection Eigensolve

Count eigenvalues below energy E via Sturm sequence:
```
d_0 = H_{00} - E
d_i = (H_{ii} - E) - H_{i,i-1}² / d_{i-1}
```

The number of sign changes in {d_0, d_1, ..., d_{N-1}} equals the number of
eigenvalues below E. Binary bisection on E yields individual eigenvalues to
machine precision. Complexity: O(N) per eigenvalue.

### 13.4 Critical Screening

At critical screening κ_c, the last bound state unbinds (E → 0⁻).
Reference values from Lam & Varshni (1971):

```
κ_c(1s) = 1.1906   (Z=1)
κ_c(2s) = 0.3189
κ_c(2p) = 0.2202
```

### 13.5 Screening Models

Three models implemented in `screened_coulomb.rs`:
- **Debye**: κ_D = sqrt(4π n_e e² / k_B T_e)
- **Thomas-Fermi**: κ_TF = sqrt(4π n_e e² / (2/3 E_F))
- **Ion-sphere**: κ_is = Z^(1/3) / a_ws (Wigner-Seitz)

### 13.6 References (Screened Coulomb)

[25] Murillo, M. S. and Weisheit, J. C., "Dense plasmas, screened interactions,
     and atomic ionization," Physics Reports 302, 1-65 (1998).

[26] Lam, C. S. and Varshni, Y. P., "Energies of s eigenstates in a static
     screened Coulomb potential," Phys. Rev. A 4, 1875 (1971).

---

## 14. Abelian Higgs Model

Paper 13: Bazavov et al., Phys. Rev. D 92, 076003 (2015).

Source: `barracuda/src/lattice/abelian_higgs.rs`.

### 14.1 Model Definition

U(1) gauge field coupled to a complex scalar Higgs field on a (1+1)D lattice
with N_t temporal and N_s spatial sites and periodic boundaries.

**Link variables:** U_μ(x) = exp(iθ_μ(x)), θ_μ ∈ [-π, π)

**Higgs field:** φ(x) ∈ ℂ

### 14.2 Action

```
S = S_gauge + S_higgs

S_gauge = β_pl Σ_{plaq} (1 - Re U_p)

S_higgs = Σ_x [-2κ Σ_μ Re(φ*(x) U_μ(x) φ(x+μ̂)) + |φ(x)|² + λ(|φ(x)|² - 1)²]
```

where β_pl = 1/g² is the inverse gauge coupling, κ is the hopping parameter,
λ is the scalar self-coupling, and U_p is the plaquette:

```
U_p(x) = U_0(x) U_1(x+0̂) U_0†(x+1̂) U_1†(x)
```

### 14.3 HMC for Complex Fields

Momenta: real scalars π_μ(x) for link angles, complex p(x) for Higgs.

Kinetic energy: T = ½ Σ π² + ½ Σ |p|²

**Wirtinger derivative for complex fields:**

The equation of motion for the Higgs momentum requires careful treatment
of the complex field via Wirtinger calculus:

```
dp/dt = -2 ∂S/∂φ*
```

The factor of 2 arises because dp/dt = -(∂S/∂φ_R + i ∂S/∂φ_I) = -2 ∂S/∂φ*
when using the Wirtinger derivative ∂/∂φ* = ½(∂/∂φ_R + i ∂/∂φ_I).

**Gauge link force:**
```
d/dθ Re[φ*(x) e^{iθ} φ(x+μ̂)] = -Im[φ*(x) e^{iθ} φ(x+μ̂)]
```

So the force is F_θ = -dS/dθ = 2κ Im(hop) from the Higgs term, plus
the Wilson gauge force from plaquettes containing the link.

### 14.4 Phase Structure

In (1+1)D, the model has no true phase transition but smooth crossovers:

- **Confined** (small β, small κ): disordered links, small ⟨|φ|²⟩
- **Higgs** (large κ): Higgs condensation, ⟨|φ|²⟩ >> 1
- **Coulomb** (large β, small κ): ordered links, ⟨Re U_p⟩ → 1
- **Large λ limit**: |φ| frozen to 1, maps to compact XY model

### 14.5 Chemical Potential

Temporal hopping acquires a chemical potential weight:

```
S_kin = -κ Σ_x [e^μ φ*(x) U_0(x) φ(x+0̂) + e^{-μ} φ*(x+0̂) U_0†(x) φ(x) + spatial terms]
```

This allows study of finite-density phases.

### 14.6 References (Abelian Higgs)

[27] Bazavov, A. et al., "Gauge-invariant implementation of the Abelian-Higgs
     model on optical lattices," Phys. Rev. D 92, 076003 (2015).

---

*This document is part of the hotSpring validation suite. License: AGPL-3.0.*
*Last updated: 2026-02-20.*
