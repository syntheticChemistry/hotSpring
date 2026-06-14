#!/usr/bin/env python3
"""
Screened Coulomb (Yukawa) bound-state eigenvalue reference generator.

Python control for Paper 6: Murillo & Weisheit (1998).

Uses scipy.linalg.eigh_tridiagonal (LAPACK dstevd) to compute
eigenvalues of the discretized radial Schrödinger equation with
a Yukawa potential V(r) = -Z exp(-κr)/r.

Same grid and discretization as barracuda/src/physics/screened_coulomb.rs:
  - Uniform grid r_i = (i+1)h, h = r_max/(N+1)
  - H_{ii} = 1/h² + l(l+1)/(2r_i²) - Z exp(-κr_i)/r_i
  - H_{i,i±1} = -1/(2h²)

Output: reference eigenvalues for Rust validation parity check.

References:
  - Murillo & Weisheit, Physics Reports 302, 1-65 (1998)
  - Lam & Varshni, Phys. Rev. A 4, 1875 (1971)
"""

import json
import numpy as np
from scipy.linalg import eigh_tridiagonal

N_GRID = 2000
R_MAX = 100.0

# Literature critical screening (Lam & Varshni 1971)
CRITICAL_SCREENING_LIT = {
    (1, 0): 1.19061,  # 1s
    (2, 0): 0.31750,  # 2s
    (2, 1): 0.21954,  # 2p
    (3, 0): 0.14459,  # 3s
    (3, 1): 0.10789,  # 3p
    (3, 2): 0.09025,  # 3d
}


def eigenvalues(z, kappa, l, n_grid=N_GRID, r_max=R_MAX):
    """Bound-state eigenvalues of screened Coulomb potential."""
    h = r_max / (n_grid + 1)
    inv_h2 = 1.0 / (h * h)
    l_f = float(l)
    centrifugal = l_f * (l_f + 1.0) / 2.0

    r = np.arange(1, n_grid + 1) * h
    diag = inv_h2 + centrifugal / (r * r) - z * np.exp(-kappa * r) / r
    off_diag = np.full(n_grid - 1, -0.5 * inv_h2)

    evals = eigh_tridiagonal(diag, off_diag, eigvals_only=True)
    return evals[evals < 0.0]


def critical_screening(z, n_state, l, n_grid=N_GRID, r_max=R_MAX):
    """Find κ_c via bisection on bound-state count."""
    target = n_state - l

    def has_state(kappa):
        return len(eigenvalues(z, kappa, l, n_grid, r_max)) >= target

    hi = z * 2.0
    while has_state(hi):
        hi *= 2.0
    lo = 0.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if has_state(mid):
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def main():
    print("=" * 65)
    print("  Screened Coulomb Reference Generator (Paper 6)")
    print(f"  Grid: N={N_GRID}, r_max={R_MAX}")
    print("=" * 65)

    # Hydrogen eigenvalues at κ=0 (reference: exact -Z²/(2n²))
    reference_data = {"grid": {"n_grid": N_GRID, "r_max": R_MAX}, "eigenvalues": [], "critical_screening": []}

    print("\n── Hydrogen eigenvalues at κ=0 ──")
    for l in [0, 1, 2]:
        evals = eigenvalues(1.0, 0.0, l)
        print(f"  l={l}: {len(evals)} bound states")
        for i, e in enumerate(evals[:5]):
            n_eff = l + i + 1
            exact = -0.5 / (n_eff * n_eff)
            rel_err = abs((e - exact) / exact)
            print(f"    E_{n_eff}{['s','p','d'][l]} = {e:.8f}  (exact {exact:.8f}, err {rel_err:.2e})")

    # Reference eigenvalues at several κ values
    print("\n── Eigenvalues vs screening (l=0) ──")
    kappa_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.1]
    for kappa in kappa_values:
        evals = eigenvalues(1.0, kappa, 0)
        entry = {"z": 1.0, "kappa": kappa, "l": 0, "eigenvalues": evals.tolist()}
        reference_data["eigenvalues"].append(entry)
        n_bound = len(evals)
        e_str = ", ".join(f"{e:.8f}" for e in evals[:3])
        print(f"  κ={kappa:.2f}: {n_bound} bound, E = [{e_str}{'...' if n_bound > 3 else ''}]")

    # l=1 at a few κ values
    for kappa in [0.0, 0.05, 0.1, 0.15, 0.2]:
        evals = eigenvalues(1.0, kappa, 1)
        entry = {"z": 1.0, "kappa": kappa, "l": 1, "eigenvalues": evals.tolist()}
        reference_data["eigenvalues"].append(entry)

    # Critical screening parameters
    print("\n── Critical screening parameters ──")
    for (n_state, l), lit in sorted(CRITICAL_SCREENING_LIT.items()):
        kc = critical_screening(1.0, n_state, l)
        rel_err = abs((kc - lit) / lit)
        label = f"{n_state}{'spd'[l]}"
        reference_data["critical_screening"].append(
            {"n": n_state, "l": l, "kappa_c_computed": kc, "kappa_c_literature": lit}
        )
        print(f"  κ_c({label}) = {kc:.6f}  (lit {lit:.5f}, err {rel_err:.2e})")

    # He+ (Z=2) for scaling check
    print("\n── He+ (Z=2) eigenvalues at κ=0 ──")
    evals_he = eigenvalues(2.0, 0.0, 0)
    exact_he = -2.0
    rel_he = abs((evals_he[0] - exact_he) / exact_he)
    print(f"  E(1s) = {evals_he[0]:.8f}  (exact {exact_he:.8f}, err {rel_he:.2e})")
    reference_data["eigenvalues"].append({"z": 2.0, "kappa": 0.0, "l": 0, "eigenvalues": evals_he[:5].tolist()})

    # Write reference JSON
    outpath = "control/screened_coulomb/reference_eigenvalues.json"
    with open(outpath, "w") as f:
        json.dump(reference_data, f, indent=2)
    print(f"\n  Reference data → {outpath}")
    print(f"  {len(reference_data['eigenvalues'])} eigenvalue sets, "
          f"{len(reference_data['critical_screening'])} critical screening values")


if __name__ == "__main__":
    main()
