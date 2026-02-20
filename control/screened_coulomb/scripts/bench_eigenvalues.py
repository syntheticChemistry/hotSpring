#!/usr/bin/env python3
"""
Performance benchmark: Python/scipy screened Coulomb eigenvalues.

Measures wall-clock time for the same computation that barracuda performs
in pure Rust, enabling direct language-level comparison.
"""

import time
import numpy as np
from scipy.linalg import eigh_tridiagonal


def eigenvalues(z, kappa, l, n_grid, r_max):
    """Same discretization as barracuda screened_coulomb.rs"""
    h = r_max / (n_grid + 1)
    inv_h2 = 1.0 / (h * h)
    centrifugal = l * (l + 1.0) / 2.0

    r = np.arange(1, n_grid + 1) * h
    diag = inv_h2 + centrifugal / (r * r) - z * np.exp(-kappa * r) / r
    off_diag = np.full(n_grid - 1, -0.5 * inv_h2)

    evals = eigh_tridiagonal(diag, off_diag, eigvals_only=True)
    return evals[evals < 0.0]


def bench_single(z, kappa, l, n_grid, r_max, label, n_iter=10):
    """Run eigenvalue computation n_iter times and report timing."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        eigenvalues(z, kappa, l, n_grid, r_max)
        times.append(time.perf_counter() - t0)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  {label:30s}  {mean_ms:8.2f} ± {std_ms:5.2f} ms  (N={n_grid})")
    return mean_ms


def main():
    print("=" * 70)
    print("  Python/scipy Screened Coulomb Benchmark")
    print("  scipy.linalg.eigh_tridiagonal (LAPACK dstevd)")
    print("=" * 70)

    # Paper 6 validation grid
    print("\n── Standard validation grid (N=2000, r_max=100) ──")
    bench_single(1.0, 0.0, 0, 2000, 100.0, "H l=0 κ=0")
    bench_single(1.0, 0.5, 0, 2000, 100.0, "H l=0 κ=0.5")
    bench_single(1.0, 1.0, 0, 2000, 100.0, "H l=0 κ=1.0")
    bench_single(2.0, 0.0, 0, 2000, 100.0, "He+ l=0 κ=0")

    # Critical screening (80 bisection iterations)
    print("\n── Critical screening (80 bisections × eigensolve) ──")

    def bench_critical(z, n, l, n_grid, r_max, label, n_iter=3):
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()

            target = n - l

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

            times.append(time.perf_counter() - t0)

        mean_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        kc = 0.5 * (lo + hi)
        print(f"  {label:30s}  {mean_ms:8.0f} ± {std_ms:5.0f} ms  κ_c={kc:.5f}")
        return mean_ms

    bench_critical(1.0, 1, 0, 2000, 100.0, "κ_c(1s)")
    bench_critical(1.0, 2, 0, 2000, 100.0, "κ_c(2s)")

    # Scaling with grid size
    print("\n── Grid-size scaling (H l=0 κ=0) ──")
    for n_grid in [500, 1000, 2000, 5000, 10000]:
        bench_single(1.0, 0.0, 0, n_grid, 100.0, f"N={n_grid}", n_iter=5)

    # Full validation binary equivalent (23 checks)
    print("\n── Full validation equivalent (all 23 checks) ──")
    t0 = time.perf_counter()

    eigenvalues(1.0, 0.0, 0, 2000, 100.0)
    eigenvalues(1.0, 0.0, 1, 2000, 100.0)
    eigenvalues(1.0, 0.1, 0, 2000, 100.0)
    eigenvalues(1.0, 0.5, 0, 2000, 100.0)
    eigenvalues(1.0, 1.0, 0, 2000, 100.0)
    eigenvalues(2.0, 0.0, 0, 2000, 100.0)
    eigenvalues(1.0, 0.1, 1, 2000, 100.0)

    # Critical screening (3 states × 80 iterations)
    for (n, l) in [(1, 0), (2, 0), (2, 1)]:
        target = n - l
        hi = 2.0
        while len(eigenvalues(1.0, hi, l, 2000, 100.0)) >= target:
            hi *= 2.0
        lo = 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if len(eigenvalues(1.0, mid, l, 2000, 100.0)) >= target:
                lo = mid
            else:
                hi = mid

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  Full validation equivalent:     {total_ms:8.0f} ms total")


if __name__ == "__main__":
    main()
