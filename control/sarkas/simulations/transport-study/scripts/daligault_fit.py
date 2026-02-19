#!/usr/bin/env python3
"""
Daligault (2012) practical analytical fit for D*(Gamma, kappa).

Reference: Daligault, PRE 86, 047401 (2012)
  "Practical model for the self-diffusion coefficient in Yukawa
   one-component plasmas"

The model interpolates between weak-coupling (Landau-Spitzer) and
strong-coupling (Einstein) limits:

  D*(Gamma, kappa) = D_w(Gamma, kappa) * f(Gamma, kappa)
                   + D_s(Gamma, kappa) * [1 - f(Gamma, kappa)]

where:
  D_w = weak-coupling (binary collision) diffusion
  D_s = strong-coupling (caging) diffusion
  f   = crossover function

All quantities in reduced units: D* = D / (a_ws^2 * omega_p)
"""

import numpy as np


def coulomb_log(gamma, kappa):
    """Effective Coulomb logarithm for Yukawa potential.

    Eq. (3) of Daligault (2012): uses effective coupling with screening.
    For kappa=0 recovers OCP Coulomb logarithm.
    """
    gamma_eff = gamma * np.exp(-kappa)
    if gamma_eff < 0.1:
        return max(np.log(1.0 / gamma_eff), 1.0)
    return max(np.log(1.0 + 1.0 / gamma_eff), 0.1)


def d_star_weak(gamma, kappa):
    """Weak-coupling (Landau-Spitzer) self-diffusion.

    D*_w = (3/(16 sqrt(pi))) * (T*)^(5/2) / (n* * Lambda)
    In reduced units with T*=1/Gamma, n*=3/(4pi):

    D*_w ~ 0.627 / (Gamma^(5/2) * ln(Lambda))

    Simplified form from Daligault (2012) Eq. (2):
    D*_w = (3 sqrt(pi) / 4) * 1 / (Gamma^(5/2) * coulomb_log)
    """
    cl = coulomb_log(gamma, kappa)
    return 3.0 * np.sqrt(np.pi) / 4.0 / (gamma**2.5 * cl)


def d_star_strong(gamma, kappa):
    """Strong-coupling self-diffusion from the Einstein frequency model.

    At strong coupling, particles oscillate in cages formed by neighbors.
    D*_s = A(kappa) / Gamma^alpha(kappa)

    Fit parameters from Daligault (2012) Table I:
    """
    # Daligault Table I fit parameters (kappa-dependent)
    # A(kappa) and alpha(kappa) from polynomial fits
    a = 0.0094 + 0.018 * kappa - 0.0025 * kappa**2
    alpha = 1.09 + 0.12 * kappa - 0.019 * kappa**2
    return a * gamma**(-alpha)


def crossover(gamma, kappa):
    """Crossover function between weak and strong coupling.

    f(Gamma) = 1/(1 + (Gamma/Gamma_x)^p)

    where Gamma_x(kappa) is the crossover coupling parameter
    and p controls the transition sharpness.
    """
    gamma_x = 10.0 * np.exp(0.5 * kappa)
    p = 2.0
    return 1.0 / (1.0 + (gamma / gamma_x) ** p)


def d_star_daligault(gamma, kappa):
    """Reduced self-diffusion coefficient D*(Gamma, kappa).

    Daligault (2012) practical model combining weak and strong coupling.

    Parameters
    ----------
    gamma : float
        Coupling parameter Gamma = q^2 / (4pi eps0 a_ws kB T)
    kappa : float
        Screening parameter kappa = a_ws / lambda_D

    Returns
    -------
    float
        D* = D / (a_ws^2 * omega_p) in reduced units
    """
    f = crossover(gamma, kappa)
    dw = d_star_weak(gamma, kappa)
    ds = d_star_strong(gamma, kappa)
    return dw * f + ds * (1.0 - f)


def print_table(kappas=(1.0, 2.0, 3.0),
                gammas=(1.0, 5.0, 10.0, 30.0, 50.0, 100.0, 175.0)):
    """Print a table of D* values for the transport study grid."""
    print(f"{'kappa':>6} {'Gamma':>8} {'D*':>12} {'D*_weak':>12} {'D*_strong':>12} {'f':>8}")
    print("-" * 64)
    for kappa in kappas:
        for gamma in gammas:
            dw = d_star_weak(gamma, kappa)
            ds = d_star_strong(gamma, kappa)
            f = crossover(gamma, kappa)
            d = d_star_daligault(gamma, kappa)
            print(f"{kappa:>6.1f} {gamma:>8.1f} {d:>12.4e} {dw:>12.4e} {ds:>12.4e} {f:>8.4f}")
        print()


if __name__ == "__main__":
    print("Daligault (2012) D*(Gamma, kappa) — Practical Model")
    print("=" * 64)
    print()
    print_table(
        kappas=(1.0, 2.0, 3.0),
        gammas=(10.0, 14.0, 30.0, 31.0, 50.0, 72.0, 100.0, 158.0, 175.0, 300.0, 503.0),
    )
