# Physics Model Documentation

**hotSpring Nuclear EOS — Complete Equation Reference**

This document describes every physics model, equation, and approximation used in the
hotSpring validation suite. It is intended for nuclear physics researchers and students
who wish to independently verify our results.

All implementations exist in both Python (`control/surrogate/nuclear-eos/wrapper/`) and
Rust (`barracuda/src/physics/`). Both produce identical results by construction — see
`whitePaper/BARRACUDA_SCIENCE_VALIDATION.md` for precision comparisons.

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
    densities. Part II. Nuclei far from stabilities," Nucl. Phys. A 635, 231 (1998).

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

*This document is part of the hotSpring validation suite. License: AGPL-3.0.*
*Last updated: 2026-02-13.*
