#!/usr/bin/env python3
"""
Calibrate Daligault (2012) D* fit coefficients using Sarkas-validated data.

The Sarkas DSF study provides D_MKS (Green-Kubo from VACF) at 12 (Gamma, kappa)
points. These are converted to reduced D* = D/(a_ws^2 * omega_p) using the
physical parameters from the transport-study input generation script.

The original fit coefficients (as transcribed from the paper) give D* values
that are 6-500x too small. This script fits corrected A(kappa) and alpha(kappa)
for the strong-coupling formula using the validated Sarkas data.
"""

import numpy as np
from scipy.optimize import curve_fit

# Physical constants (same as generate_transport_inputs.py)
e = 1.602176634e-19
eps0 = 8.8541878128e-12
kB = 1.380649e-23
m_p = 1.67262192369e-27

Z = 1.0
n = 1.62e30
mass = m_p

a_ws = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)
omega_p = np.sqrt(n * (Z * e) ** 2 / (eps0 * mass))
a2_omega_p = a_ws ** 2 * omega_p

print(f"a_ws = {a_ws:.6e} m")
print(f"omega_p = {omega_p:.6e} rad/s")
print(f"a_ws^2 * omega_p = {a2_omega_p:.6e} m^2/s")
print()

# Sarkas DSF study D_MKS values (from all_observables_validation.json)
sarkas_data = [
    # (kappa, gamma, D_MKS [m^2/s])
    (0, 10, 5.856310235929256e-07),
    (0, 50, 6.053635489140258e-08),
    (0, 150, 2.0166172842531932e-08),
    (1, 14, 4.903065815923215e-07),
    (1, 72, 5.118468699472465e-08),
    (1, 217, 1.6971683693893374e-08),
    (2, 31, 3.0008979818574066e-07),
    (2, 158, 3.3843850462772176e-08),
    (2, 476, 1.2036150453795876e-08),
    (3, 100, 1.3560766046086524e-07),
    (3, 503, 1.7872184539095038e-08),
    (3, 1510, 7.712262677617913e-09),
]

# Convert to reduced D*
data = []
print(f"{'kappa':>5} {'Gamma':>6} {'D_MKS':>12} {'D*':>12}")
print("-" * 40)
for kappa, gamma, d_mks in sarkas_data:
    d_star = d_mks / a2_omega_p
    data.append((kappa, gamma, d_star))
    print(f"{kappa:>5} {gamma:>6} {d_mks:>12.4e} {d_star:>12.6f}")

print()

# Original Daligault fit (for comparison)
def coulomb_log(gamma, kappa):
    gamma_eff = gamma * np.exp(-kappa)
    if gamma_eff < 0.1:
        return max(np.log(1.0 / gamma_eff), 1.0)
    else:
        return max(np.log(1.0 + 1.0 / gamma_eff), 0.1)


def d_star_daligault_original(gamma, kappa):
    cl = coulomb_log(gamma, kappa)
    d_weak = 3.0 * np.sqrt(np.pi) / 4.0 / (gamma ** 2.5 * cl)
    a = 0.0094 + 0.018 * kappa - 0.0025 * kappa ** 2
    alpha = 1.09 + 0.12 * kappa - 0.019 * kappa ** 2
    d_strong = a * gamma ** (-alpha)
    gamma_x = 10.0 * np.exp(0.5 * kappa)
    f = 1.0 / (1.0 + (gamma / gamma_x) ** 2)
    return d_weak * f + d_strong * (1.0 - f)


# Fit new strong-coupling coefficients per kappa
# D*_s = A(kappa) * Gamma^(-alpha(kappa))
print("=" * 60)
print("Fitting strong-coupling coefficients from Sarkas data")
print("=" * 60)

kappa_values = sorted(set(k for k, _, _ in data))
fit_results = {}

for kappa in kappa_values:
    points = [(g, d) for k, g, d in data if k == kappa]
    gammas = np.array([g for g, _ in points])
    d_stars = np.array([d for _, d in points])

    # Only use points where strong coupling dominates (high Gamma)
    # Use log-linear fit: ln(D*) = ln(A) - alpha*ln(Gamma)
    log_g = np.log(gammas)
    log_d = np.log(d_stars)

    # Weighted linear fit (weight higher-Gamma points more, they're deeper in SC)
    weights = gammas / gammas.min()
    coeffs = np.polyfit(log_g, log_d, 1, w=weights)
    alpha = -coeffs[0]
    A = np.exp(coeffs[1])

    print(f"\nkappa = {kappa}:")
    print(f"  Points: {list(zip(gammas.tolist(), d_stars.tolist()))}")
    print(f"  A = {A:.4f}, alpha = {alpha:.4f}")
    for g, d in points:
        d_fit = A * g ** (-alpha)
        err = abs(d_fit - d) / d * 100
        print(f"    Gamma={g:>6}: D*={d:.6f}, fit={d_fit:.6f}, err={err:.1f}%")

    fit_results[kappa] = (A, alpha)

print()
print("=" * 60)
print("Corrected fit coefficients")
print("=" * 60)

kappas = np.array(kappa_values, dtype=float)
As = np.array([fit_results[k][0] for k in kappa_values])
alphas = np.array([fit_results[k][1] for k in kappa_values])

# Fit quadratic: A(kappa) = a0 + a1*kappa + a2*kappa^2
A_coeffs = np.polyfit(kappas, As, 2)
alpha_coeffs = np.polyfit(kappas, alphas, 2)

print(f"\nA(kappa) = {A_coeffs[2]:.4f} + {A_coeffs[1]:.4f}*kappa + {A_coeffs[0]:.4f}*kappa^2")
print(f"alpha(kappa) = {alpha_coeffs[2]:.4f} + {alpha_coeffs[1]:.4f}*kappa + {alpha_coeffs[0]:.4f}*kappa^2")

print("\nPer-kappa values:")
for k in kappa_values:
    A_quad = A_coeffs[2] + A_coeffs[1] * k + A_coeffs[0] * k ** 2
    a_quad = alpha_coeffs[2] + alpha_coeffs[1] * k + alpha_coeffs[0] * k ** 2
    print(f"  kappa={k}: A={fit_results[k][0]:.4f} (quad={A_quad:.4f}), alpha={fit_results[k][1]:.4f} (quad={a_quad:.4f})")

# Also fit weak-coupling correction factor
print()
print("=" * 60)
print("Weak-coupling correction analysis")
print("=" * 60)

for kappa in kappa_values:
    points = [(g, d) for k, g, d in data if k == kappa]
    for gamma, d_star in points:
        A, alpha = fit_results[kappa]
        d_strong = A * gamma ** (-alpha)
        gamma_x = 10.0 * np.exp(0.5 * kappa)
        f = 1.0 / (1.0 + (gamma / gamma_x) ** 2)

        if f > 0.1:
            d_weak_needed = (d_star - d_strong * (1 - f)) / f
            cl = coulomb_log(gamma, kappa)
            d_weak_old = 3.0 * np.sqrt(np.pi) / 4.0 / (gamma ** 2.5 * cl)
            correction = d_weak_needed / d_weak_old if d_weak_old > 0 else float("inf")
            print(
                f"  kappa={kappa}, Gamma={gamma}: f={f:.3f}, "
                f"D*_w_needed={d_weak_needed:.4e}, D*_w_old={d_weak_old:.4e}, "
                f"correction={correction:.2f}x"
            )

# Final summary: full model validation
print()
print("=" * 60)
print("Full corrected model validation")
print("=" * 60)

# Use corrected A, alpha from per-kappa fit; and weak-coupling with multiplier
# We'll determine the best weak-coupling multiplier by least-squares
def d_star_corrected(gamma, kappa, weak_mult):
    cl = coulomb_log(gamma, kappa)
    d_weak = weak_mult * 3.0 * np.sqrt(np.pi) / 4.0 / (gamma ** 2.5 * cl)
    A, alpha = fit_results.get(kappa, fit_results[min(fit_results.keys(), key=lambda k: abs(k - kappa))])
    d_strong = A * gamma ** (-alpha)
    gamma_x = 10.0 * np.exp(0.5 * kappa)
    f = 1.0 / (1.0 + (gamma / gamma_x) ** 2)
    return d_weak * f + d_strong * (1.0 - f)


# Optimize weak_mult
from scipy.optimize import minimize_scalar

def residual(weak_mult):
    total_err = 0
    for kappa, gamma, d_star in data:
        d_fit = d_star_corrected(gamma, kappa, weak_mult)
        total_err += ((d_fit - d_star) / d_star) ** 2
    return total_err

result = minimize_scalar(residual, bounds=(1.0, 50.0), method="bounded")
best_mult = result.x
print(f"Best weak-coupling multiplier: {best_mult:.2f}")

print(f"\n{'kappa':>5} {'Gamma':>6} {'D*_Sarkas':>12} {'D*_corrected':>14} {'D*_original':>14} {'err_new':>10} {'err_old':>10}")
print("-" * 80)
for kappa, gamma, d_star in data:
    d_new = d_star_corrected(gamma, kappa, best_mult)
    d_old = d_star_daligault_original(gamma, kappa)
    err_new = abs(d_new - d_star) / d_star * 100
    err_old = abs(d_old - d_star) / d_star * 100
    print(f"{kappa:>5} {gamma:>6} {d_star:>12.6f} {d_new:>14.6f} {d_old:>14.6e} {err_new:>9.1f}% {err_old:>9.0f}%")

# Output the Rust-ready coefficients
print()
print("=" * 60)
print("RUST-READY COEFFICIENTS")
print("=" * 60)
print(f"const WEAK_COUPLING_MULTIPLIER: f64 = {best_mult:.2f};")
print()
print("// Strong-coupling: D*_s = A(kappa) * Gamma^(-alpha(kappa))")
print(f"// A(kappa)     = {A_coeffs[2]:.4f} + {A_coeffs[1]:.4f}*kappa + {A_coeffs[0]:.6f}*kappa^2")
print(f"// alpha(kappa) = {alpha_coeffs[2]:.4f} + {alpha_coeffs[1]:.4f}*kappa + {alpha_coeffs[0]:.6f}*kappa^2")
print()
for k in kappa_values:
    print(f"// kappa={k}: A={fit_results[k][0]:.6f}, alpha={fit_results[k][1]:.6f}")
