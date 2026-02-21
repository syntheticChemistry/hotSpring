#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Spectral Theory Python Control — Kachkovskiy Extension Baseline

Computes the same spectral theory results as barracuda/src/spectral.rs
using NumPy. Establishes the Python parity baseline for:

  1. Anderson 1D eigenvalues (tridiagonal, numpy.linalg.eigh)
  2. Lyapunov exponent (transfer matrix product)
  3. Level spacing ratio (localization diagnostic)
  4. Anderson 2D eigenvalues (sparse, numpy.linalg.eigh on dense)
  5. Almost-Mathieu / Hofstadter bands
  6. Timing comparison: Python vs Rust

Provenance:
  Anderson (1958) Phys. Rev. 109, 1492
  Aubry & André (1980) Ann. Israel Phys. Soc. 3, 133
  Herman (1983) Comment. Math. Helv. 58, 453
  Hofstadter (1976) Phys. Rev. B 14, 2239

Usage:
  python3 spectral_control.py
  python3 spectral_control.py --json  # Output results as JSON

Dependencies: numpy only (no scipy required)
"""

import json
import sys
import time

import numpy as np

GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0
POISSON_R = 2.0 * np.log(2.0) - 1.0  # ≈ 0.3863


# ── Anderson 1D ──────────────────────────────────────────────────

def anderson_1d_hamiltonian(n, disorder, seed=42):
    """Tridiagonal Anderson Hamiltonian: H = -Δ + V."""
    rng = np.random.default_rng(seed)
    diagonal = disorder * (rng.random(n) - 0.5)
    off_diag = -np.ones(n - 1)
    H = np.diag(diagonal) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    return H


def anderson_1d_eigenvalues(n, disorder, seed=42):
    """All eigenvalues via dense diagonalization."""
    H = anderson_1d_hamiltonian(n, disorder, seed)
    return np.sort(np.linalg.eigh(H)[0])


# ── Lyapunov exponent ────────────────────────────────────────────

def lyapunov_exponent(potential, energy):
    """Transfer matrix Lyapunov exponent."""
    n = len(potential)
    if n == 0:
        return 0.0

    v_prev, v_curr = 0.0, 1.0
    log_growth = 0.0

    for v_i in potential:
        v_next = (energy - v_i) * v_curr - v_prev
        v_prev = v_curr
        v_curr = v_next
        norm = np.hypot(v_curr, v_prev)
        if norm > 0:
            log_growth += np.log(norm)
            v_curr /= norm
            v_prev /= norm

    return log_growth / n


def anderson_potential(n, disorder, seed=42):
    """Random potential V_i ~ Uniform[-W/2, W/2]."""
    rng = np.random.default_rng(seed)
    return disorder * (rng.random(n) - 0.5)


# ── Level spacing ratio ─────────────────────────────────────────

def level_spacing_ratio(eigenvalues):
    """Mean level spacing ratio ⟨r⟩ (Oganesyan-Huse 2007)."""
    spacings = np.diff(eigenvalues)
    spacings = spacings[spacings > 0]
    if len(spacings) < 2:
        return 0.0
    r_values = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return np.mean(r_values)


# ── Anderson 2D ──────────────────────────────────────────────────

def anderson_2d_hamiltonian(lx, ly, disorder, seed=42):
    """Dense 2D Anderson Hamiltonian on Lx × Ly square lattice."""
    n = lx * ly
    rng = np.random.default_rng(seed)
    H = np.zeros((n, n))

    for ix in range(lx):
        for iy in range(ly):
            i = ix * ly + iy
            H[i, i] = disorder * (rng.random() - 0.5)
            if ix > 0:
                j = (ix - 1) * ly + iy
                H[i, j] = H[j, i] = -1.0
            if iy > 0:
                j = ix * ly + (iy - 1)
                H[i, j] = H[j, i] = -1.0

    return H


# ── Almost-Mathieu / Hofstadter ──────────────────────────────────

def almost_mathieu_hamiltonian(n, lam, alpha, theta=0.0):
    """Almost-Mathieu operator (tridiagonal)."""
    diagonal = np.array([2.0 * lam * np.cos(2.0 * np.pi * alpha * i + theta) for i in range(n)])
    off_diag = -np.ones(n - 1)
    H = np.diag(diagonal) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    return H


# ── Validation checks ───────────────────────────────────────────

def run_all():
    results = {}
    json_mode = "--json" in sys.argv

    print("=" * 60)
    print("  Spectral Theory — Python Control Baseline")
    print("  NumPy", np.__version__)
    print("=" * 60)
    print()

    # ── 1. Anderson 1D eigenvalues ───────────────────────────────
    print("[1] Anderson 1D — Eigenvalues")
    n = 1000
    w = 4.0
    t0 = time.perf_counter()
    evals = anderson_1d_eigenvalues(n, w, seed=42)
    t_1d = time.perf_counter() - t0
    print(f"  N={n}, W={w}")
    print(f"  Spectrum: [{evals[0]:.4f}, {evals[-1]:.4f}]")
    print(f"  Time: {t_1d*1000:.1f} ms")
    results["anderson_1d"] = {
        "n": n, "w": w, "e_min": float(evals[0]), "e_max": float(evals[-1]),
        "time_ms": t_1d * 1000,
    }
    print()

    # ── 2. Lyapunov exponent (Herman's formula) ─────────────────
    print("[2] Almost-Mathieu — Herman's Formula γ = ln|λ|")
    n_lyap = 100_000
    lambdas = [1.5, 2.0, 3.0, 5.0]
    lyap_results = []
    t0 = time.perf_counter()
    for lam in lambdas:
        pot = np.array([2.0 * lam * np.cos(2.0 * np.pi * GOLDEN_RATIO * i) for i in range(n_lyap)])
        gamma = lyapunov_exponent(pot, 0.0)
        theory = np.log(lam)
        error = abs(gamma - theory)
        lyap_results.append({"lambda": lam, "gamma": gamma, "theory": theory, "error": error})
        print(f"  λ={lam:.1f}: γ={gamma:.4f}, ln(λ)={theory:.4f}, Δ={error:.4f}")
    t_lyap = time.perf_counter() - t0
    print(f"  Time: {t_lyap*1000:.1f} ms")
    results["herman_formula"] = {"n": n_lyap, "results": lyap_results, "time_ms": t_lyap * 1000}
    print()

    # ── 3. Anderson 1D level statistics ──────────────────────────
    print("[3] Anderson 1D — Poisson Level Statistics")
    n_stats = 1000
    w_stats = 8.0
    n_real = 10
    r_vals = []
    t0 = time.perf_counter()
    for seed in range(n_real):
        ev = anderson_1d_eigenvalues(n_stats, w_stats, seed=seed * 100 + 42)
        r_vals.append(level_spacing_ratio(ev))
    r_mean = np.mean(r_vals)
    t_stats = time.perf_counter() - t0
    print(f"  N={n_stats}, W={w_stats}, realizations={n_real}")
    print(f"  ⟨r⟩ = {r_mean:.4f} (Poisson = {POISSON_R:.4f})")
    print(f"  Time: {t_stats*1000:.1f} ms")
    results["level_stats_1d"] = {
        "n": n_stats, "w": w_stats, "r_mean": float(r_mean),
        "poisson_r": POISSON_R, "time_ms": t_stats * 1000,
    }
    print()

    # ── 4. Anderson 2D eigenvalues ───────────────────────────────
    print("[4] Anderson 2D — Eigenvalues")
    l_2d = 16
    w_2d = 5.0
    t0 = time.perf_counter()
    H_2d = anderson_2d_hamiltonian(l_2d, l_2d, w_2d, seed=42)
    evals_2d = np.sort(np.linalg.eigh(H_2d)[0])
    t_2d = time.perf_counter() - t0
    print(f"  L={l_2d}, N={l_2d**2}, W={w_2d}")
    print(f"  Spectrum: [{evals_2d[0]:.4f}, {evals_2d[-1]:.4f}]")
    print(f"  Time: {t_2d*1000:.1f} ms")
    results["anderson_2d"] = {
        "l": l_2d, "w": w_2d, "e_min": float(evals_2d[0]), "e_max": float(evals_2d[-1]),
        "time_ms": t_2d * 1000,
    }
    print()

    # ── 5. Hofstadter bands ──────────────────────────────────────
    print("[5] Hofstadter Butterfly — Band Counting")
    n_hof = 500
    for q, alpha in [(2, 0.5), (3, 1/3), (5, 0.2)]:
        t0 = time.perf_counter()
        H = almost_mathieu_hamiltonian(n_hof, 1.0, alpha)
        ev = np.sort(np.linalg.eigh(H)[0])
        t_hof = time.perf_counter() - t0

        spacings = np.diff(ev)
        median_spacing = np.median(spacings)
        gaps = np.where(spacings > 10 * median_spacing)[0]
        n_bands_raw = len(gaps) + 1

        band_widths = []
        prev = 0
        for g in gaps:
            band_widths.append(ev[g] - ev[prev])
            prev = g + 1
        band_widths.append(ev[-1] - ev[prev])
        n_wide = sum(1 for bw in band_widths if bw > 0.01)

        print(f"  α=1/{q}: {n_bands_raw} raw bands, {n_wide} wide (expect {q}), {t_hof*1000:.1f} ms")
    results["hofstadter"] = {"n": n_hof}
    print()

    # ── 6. Timing summary ────────────────────────────────────────
    print("[6] Timing Summary — Python Baseline")
    print(f"  Anderson 1D (N=1000):     {results['anderson_1d']['time_ms']:.1f} ms")
    print(f"  Herman Lyapunov (N=100k): {results['herman_formula']['time_ms']:.1f} ms")
    print(f"  Level stats (10 real):    {results['level_stats_1d']['time_ms']:.1f} ms")
    print(f"  Anderson 2D (16×16):      {results['anderson_2d']['time_ms']:.1f} ms")
    total_ms = sum(results[k]["time_ms"] for k in [
        "anderson_1d", "herman_formula", "level_stats_1d", "anderson_2d"
    ])
    print(f"  Total:                    {total_ms:.1f} ms")
    results["total_time_ms"] = total_ms
    print()

    # ── Output ───────────────────────────────────────────────────
    if json_mode:
        print(json.dumps(results, indent=2))

    print("=" * 60)
    print("  Python control complete — all results above are the")
    print("  baseline for Rust parity comparison.")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
