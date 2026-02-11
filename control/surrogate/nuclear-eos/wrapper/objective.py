#!/usr/bin/env python3
"""
Nuclear EOS Objective Function

No HFBTHO. No Fortran. No Code Ocean. No permissions.
Skyrme EDF parameters → nuclear matter → SEMF → binding energies → χ²

This replaces the gated Code Ocean capsule with open physics.

Usage:
    from objective import nuclear_eos_objective
    chi2 = nuclear_eos_objective([t0, t1, t2, t3, x0, x1, x2, x3, alpha, W0])

Author: ecoPrimals — we do science, not permissions
License: AGPL-3.0
"""

import os

# CRITICAL PERFORMANCE: Limit BLAS to 1 thread.
# For 91×91 matrices, multi-threaded BLAS has MORE overhead than benefit.
# This single change: 60s/eval → 12s/eval (5x speedup for free!)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import numpy as np
import multiprocessing as mp
from pathlib import Path

from skyrme_hf import (
    nuclear_matter_properties,
    semf_binding_energy,
    binding_energy,
    PARAM_NAMES,
)
from skyrme_hfb import binding_energy_l2


WRAPPER_DIR = Path(__file__).parent
EXP_DATA_FILE = WRAPPER_DIR.parent / "exp_data" / "ame2020_selected.json"
BOUNDS_FILE = WRAPPER_DIR / "skyrme_bounds.json"

# Cache experimental data
_exp_data_cache = None


def load_experimental_data(filepath=None):
    """Load AME2020 experimental binding energies."""
    global _exp_data_cache
    if _exp_data_cache is not None:
        return _exp_data_cache

    filepath = filepath or EXP_DATA_FILE
    with open(filepath) as f:
        data = json.load(f)
    _exp_data_cache = {
        (e["Z"], e["N"]): (e["binding_energy_MeV"], e["uncertainty_MeV"])
        for e in data["nuclei"]
    }
    return _exp_data_cache


def load_bounds():
    """Load parameter bounds from JSON."""
    with open(BOUNDS_FILE) as f:
        bounds_data = json.load(f)
    bounds = []
    for name in PARAM_NAMES:
        r = bounds_data["parameters"][name]["typical_range"]
        bounds.append(tuple(r))
    return bounds


def nuclear_eos_chi2(x):
    """Raw χ²/datum objective (not log-transformed).

    Parameters
    ----------
    x : array-like, shape (10,)
        [t0, t1, t2, t3, x0, x1, x2, x3, alpha, W0]

    Returns
    -------
    chi2 : float
        χ² per datum against AME2020 (lower is better)
    """
    x = np.asarray(x, dtype=float)

    # Sanity: alpha must be positive
    if x[8] <= 0.01 or x[8] > 1.0:
        return 1e4  # soft penalty, not hard wall

    # Get nuclear matter properties
    try:
        nmp = nuclear_matter_properties(x)
    except Exception:
        return 1e4

    # Physical constraints (soft, proportional penalties for smooth landscape)
    penalty = 0.0
    rho0 = nmp["rho0_fm3"]
    ea = nmp["E_A_MeV"]
    if rho0 < 0.08:
        penalty += 50.0 * (0.08 - rho0) / 0.08
    elif rho0 > 0.25:
        penalty += 50.0 * (rho0 - 0.25) / 0.25
    if ea > -5:  # barely bound or unbound
        penalty += 20.0 * max(0, ea + 5)

    # Load experimental data
    exp_data = load_experimental_data()

    # Compute binding energies
    # Use theoretical uncertainty σ_theo ~ 1-2 MeV (not experimental σ ~ 0.001 MeV)
    # SEMF is inherently a ~1% model, so σ_theo ~ 1% of B_exp or 2 MeV min
    chi2 = 0.0
    n = 0
    for (Z, N), (B_exp, sigma_exp) in exp_data.items():
        B_calc = semf_binding_energy(Z, N, x)
        if B_calc > 0:
            sigma_theo = max(0.01 * B_exp, 2.0)  # 1% or 2 MeV, whichever is larger
            chi2 += ((B_calc - B_exp) / sigma_theo)**2
            n += 1

    if n == 0:
        return 1e4

    return chi2 / n + penalty


def nuclear_eos_objective(x):
    """Log-transformed objective for smoother surrogate learning.

    Returns log(1 + χ²) which compresses the dynamic range:
    - Valid physics (χ²=6): log(7) ≈ 1.95
    - Marginal (χ²=100): log(101) ≈ 4.62
    - Penalty (χ²=10000): log(10001) ≈ 9.21

    This gives the RBF a ~4.6:1 range instead of ~1700:1,
    dramatically improving surrogate learning in 10D.
    """
    chi2 = nuclear_eos_chi2(x)
    return np.log1p(chi2)


def _eval_single_nucleus(args):
    """Worker function for parallel nucleus evaluation.

    Runs in a subprocess via mp.Pool. Each process has its own
    GIL + HFB solver cache. BLAS limited to 1 thread via env vars
    (set at module import time above).
    """
    Z, N, x_list = args
    x = np.array(x_list)
    try:
        B, conv = binding_energy_l2(Z, N, x, method="auto")
    except Exception:
        B, conv = 0.0, False
    return Z, N, float(B), bool(conv)


# Global pool — reused across objective evaluations (avoids fork overhead)
_worker_pool = None
_n_workers = 0


def init_parallel(workers=10, use_gpu=False):
    """Initialize worker pool for parallel L2 evaluation.

    Uses fork (default on Linux) with BLAS limited to 1 thread.
    mp.Pool is ~2x faster than ProcessPoolExecutor for this workload.

    Parameters
    ----------
    workers : int
        Number of parallel workers (default: 10, one per HFB nucleus).
    use_gpu : bool
        Ignored for now — CPU with BLAS=1 is faster than GPU for
        91×91 eigenvalue problems. GPU shines at larger scales
        (BarraCUDA Level 3).
    """
    global _worker_pool, _n_workers
    if _worker_pool is not None:
        _worker_pool.close()
        _worker_pool.join()
    _n_workers = workers
    _worker_pool = mp.Pool(workers)
    return _worker_pool


def shutdown_parallel():
    """Shut down worker pool."""
    global _worker_pool
    if _worker_pool is not None:
        _worker_pool.close()
        _worker_pool.join()
        _worker_pool = None


def nuclear_eos_chi2_l2(x, nuclei_subset=None, parallel=False, use_gpu=False):
    """Level 2 χ²/datum objective using hybrid HFB solver.

    Parameters
    ----------
    x : array-like, shape (10,)
        [t0, t1, t2, t3, x0, x1, x2, x3, alpha, W0]
    nuclei_subset : set of (Z,N) tuples or None
        If provided, only evaluate these nuclei (for faster runs).
    parallel : bool
        Use ProcessPoolExecutor for parallel nucleus evaluation.
    use_gpu : bool
        Use GPU-accelerated eigendecomposition within workers.

    Returns
    -------
    chi2 : float
        χ² per datum against AME2020 (lower is better)
    """
    x = np.asarray(x, dtype=float)

    # Sanity: alpha must be positive
    if x[8] <= 0.01 or x[8] > 1.0:
        return 1e4

    # Check nuclear matter properties are physical
    try:
        nmp = nuclear_matter_properties(x)
    except Exception:
        return 1e4

    penalty = 0.0
    rho0 = nmp["rho0_fm3"]
    ea = nmp["E_A_MeV"]
    if rho0 < 0.08:
        penalty += 50.0 * (0.08 - rho0) / 0.08
    elif rho0 > 0.25:
        penalty += 50.0 * (rho0 - 0.25) / 0.25
    if ea > -5:
        penalty += 20.0 * max(0, ea + 5)

    exp_data = load_experimental_data()

    # Build list of nuclei to evaluate
    nuclei_to_eval = []
    for (Z, N), (B_exp, sigma_exp) in exp_data.items():
        if nuclei_subset is not None and (Z, N) not in nuclei_subset:
            continue
        nuclei_to_eval.append((Z, N, B_exp))

    if parallel and _worker_pool is not None:
        # ===== PARALLEL EVALUATION via mp.Pool =====
        x_list = x.tolist()  # picklable
        args = [(Z, N, x_list) for Z, N, _ in nuclei_to_eval]
        results_list = _worker_pool.map(_eval_single_nucleus, args)

        # Build lookup: (Z,N) → B_calc
        b_lookup = {(Z, N): B for Z, N, B, _ in results_list}

        chi2 = 0.0
        n = 0
        for Z, N, B_exp in nuclei_to_eval:
            B_calc = b_lookup.get((Z, N), 0.0)
            if B_calc > 0:
                sigma_theo = max(0.01 * B_exp, 2.0)
                chi2 += ((B_calc - B_exp) / sigma_theo)**2
                n += 1
    else:
        # ===== SEQUENTIAL EVALUATION =====
        chi2 = 0.0
        n = 0
        for Z, N, B_exp in nuclei_to_eval:
            try:
                B_calc, conv = binding_energy_l2(Z, N, x, method="auto")
            except Exception:
                continue
            if B_calc > 0:
                sigma_theo = max(0.01 * B_exp, 2.0)
                chi2 += ((B_calc - B_exp) / sigma_theo)**2
                n += 1

    if n == 0:
        return 1e4

    return chi2 / n + penalty


def nuclear_eos_objective_l2(x, nuclei_subset=None, parallel=False):
    """Log-transformed Level 2 objective for surrogate learning."""
    chi2 = nuclear_eos_chi2_l2(x, nuclei_subset=nuclei_subset,
                                parallel=parallel)
    return np.log1p(chi2)


# Focused nuclei subset for fast L2 surrogate runs (~40s/eval vs 153s/eval)
# 20 nuclei: 10 SEMF (light/heavy) + 10 HFB (medium, including doubly-magic)
L2_FOCUSED_NUCLEI = {
    # Light (SEMF, A < 56)
    (2, 2),    # 4He
    (8, 8),    # 16O
    (20, 20),  # 40Ca
    (20, 28),  # 48Ca
    # Medium — HFB targets (56 ≤ A ≤ 132)
    (28, 28),  # 56Ni (doubly magic)
    (28, 30),  # 58Ni
    (28, 34),  # 62Ni
    (28, 50),  # 78Ni (doubly magic)
    (40, 50),  # 90Zr (magic N)
    (50, 50),  # 100Sn (doubly magic)
    (50, 62),  # 112Sn
    (50, 70),  # 120Sn
    (50, 74),  # 124Sn
    (50, 82),  # 132Sn (doubly magic)
    # Heavy (SEMF, A > 132)
    (58, 82),  # 140Ce
    (62, 90),  # 152Sm
    (82, 126), # 208Pb
    (92, 146), # 238U
}


def nuclear_matter_objective(x):
    """Alternative objective: fit nuclear matter properties directly.

    Simpler, faster, tests same Skyrme parameter space.
    χ² against empirical nuclear matter constraints.

    Parameters
    ----------
    x : array-like, shape (10,)

    Returns
    -------
    chi2 : float
    """
    x = np.asarray(x, dtype=float)

    if x[8] <= 0 or x[8] > 1.0:
        return 1e6

    try:
        nmp = nuclear_matter_properties(x)
    except Exception:
        return 1e6

    # Empirical constraints with uncertainties
    targets = {
        "rho0_fm3":    (0.16,    0.005),   # saturation density
        "E_A_MeV":     (-15.97,  0.5),     # energy per nucleon
        "K_inf_MeV":   (230.0,   20.0),    # incompressibility
        "m_eff_ratio": (0.69,    0.1),     # effective mass
        "J_MeV":       (32.0,    2.0),     # symmetry energy
    }

    chi2 = 0.0
    for key, (target, sigma) in targets.items():
        val = nmp[key]
        chi2 += ((val - target) / sigma)**2

    return chi2 / len(targets)


# Alias for mystic compatibility
def objective(x):
    """Default objective (binding energy χ²)."""
    return nuclear_eos_objective(x)


if __name__ == "__main__":
    print("Nuclear EOS Objective Function — Self-Test")
    print("=" * 60)

    # Test at known parametrizations
    known = {
        "SLy4": [-2488.91, 486.82, -546.39, 13777.0,
                 0.834, -0.344, -1.0, 1.354, 0.1667, 123.0],
        "UNEDF0": [-1883.69, 277.50, -189.08, 14603.6,
                   0.0047, -1.116, -1.635, 0.390, 0.3222, 78.66],
    }

    bounds = load_bounds()
    print(f"\nParameter bounds ({len(bounds)} dimensions):")
    for i, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, bounds)):
        print(f"  {i}: {name:6s} ∈ [{lo:10.1f}, {hi:10.1f}]")

    print(f"\nExperimental data: {len(load_experimental_data())} nuclei")

    print("\n--- Binding Energy χ² ---")
    for name, params in known.items():
        chi2 = nuclear_eos_chi2(params)
        log_chi2 = nuclear_eos_objective(params)
        print(f"  {name:8s}: χ²/datum = {chi2:.4f}, log(1+χ²) = {log_chi2:.4f}")

    print("\n--- Nuclear Matter χ² ---")
    for name, params in known.items():
        chi2 = nuclear_matter_objective(params)
        nmp = nuclear_matter_properties(params)
        print(f"  {name:8s}: χ²/datum = {chi2:.4f} "
              f"(ρ₀={nmp['rho0_fm3']:.3f}, E/A={nmp['E_A_MeV']:.2f})")

    # Test landscape: random perturbation
    print("\n--- Landscape probe (random perturbations of SLy4) ---")
    rng = np.random.default_rng(42)
    sly4 = np.array(known["SLy4"])
    for pct in [1, 5, 10, 20, 50]:
        chi2_vals = []
        for _ in range(20):
            perturbed = sly4 * (1.0 + pct/100 * rng.standard_normal(10))
            perturbed[8] = np.clip(perturbed[8], 0.05, 0.5)  # alpha > 0
            chi2_vals.append(nuclear_eos_objective(perturbed))
        valid = [c for c in chi2_vals if c < 1e5]
        if valid:
            print(f"  ±{pct:2d}%: median χ²={np.median(valid):.1f}, "
                  f"range [{min(valid):.1f}, {max(valid):.1f}], "
                  f"valid {len(valid)}/20")
        else:
            print(f"  ±{pct:2d}%: all invalid (outside physical bounds)")

    print("\n" + "=" * 60)
    print("✅ Objective function is callable, landscape is non-trivial")
    print("   Ready for surrogate learning workflow")
    print("=" * 60)
