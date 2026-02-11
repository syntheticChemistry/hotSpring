#!/usr/bin/env python3
"""
Run iterative surrogate learning on the nuclear EOS objective.

No HFBTHO. No Code Ocean. No permissions.
Skyrme EDF parameters → nuclear matter → SEMF → binding energies → χ²

This reproduces the Diaw et al. (2024) methodology on a real nuclear
physics objective function that we built from first principles.

Usage:
    python run_surrogate.py [--quick]

Author: ecoPrimals
License: AGPL-3.0
"""

import sys
import os

# CRITICAL PERFORMANCE: Set before numpy import.
# For 91×91 eigenvalue problems, single-threaded BLAS is 5x faster
# than multi-threaded (eliminates thread synchronization overhead).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import time
import numpy as np

# Add wrapper and parent scripts to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WRAPPER_DIR = os.path.join(SCRIPT_DIR, "..", "wrapper")
PARENT_SCRIPTS = os.path.join(SCRIPT_DIR, "..", "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)  # For gpu_rbf.py
sys.path.insert(0, WRAPPER_DIR)
sys.path.insert(0, PARENT_SCRIPTS)

from objective import (
    nuclear_eos_objective, nuclear_eos_chi2,
    nuclear_eos_objective_l2, nuclear_eos_chi2_l2,
    nuclear_matter_objective, load_bounds, PARAM_NAMES,
    L2_FOCUSED_NUCLEI,
    init_parallel, shutdown_parallel,
)


# ============================================================================
# CachedObjective — captures every evaluation for RBF training
# ============================================================================

class CachedObjective:
    """
    Wraps an objective function to capture every evaluation.

    The paper trains its RBF surrogate on ALL evaluations made during
    SparsitySampler.sample_until(), not just the best solutions. This
    is the critical methodological detail that makes the surrogate
    converge much faster — N diverse evaluations per round provide
    far more information than 16 best-so-far points.
    """
    def __init__(self, func, name="unnamed"):
        self.func = func
        self.name = name
        self.cache_x = []
        self.cache_y = []

    def __call__(self, x):
        y = self.func(x)
        self.cache_x.append(np.array(x, dtype=float, copy=True))
        self.cache_y.append(float(y))
        return y

    @property
    def n_evals(self):
        return len(self.cache_x)

    def get_data(self):
        """Return all cached (X, y) as numpy arrays."""
        return np.array(self.cache_x), np.array(self.cache_y)


def run_nuclear_eos_surrogate(evals_per_round=100, max_rounds=30, tol=0.01,
                              level=1, workers=0):
    """Run iterative surrogate on nuclear EOS.

    Uses mystic SparsitySampler + RBF surrogates, exactly matching
    the Diaw et al. (2024) methodology.

    Parameters
    ----------
    level : int
        1 = SEMF only (fast, ~0.004s/eval)
        2 = Hybrid HFB/SEMF on focused subset (~12s/eval sequential,
            ~2.7s/eval with 10 parallel workers)
    workers : int
        Number of parallel workers for L2 HFB evaluation.
        0 = sequential (default). >0 = use mp.Pool.
        Recommended: 10 (one per HFB nucleus in focused set).
    """
    from mystic.samplers import SparsitySampler
    from mystic.solvers import NelderMeadSimplexSolver as solver
    from scipy.interpolate import RBFInterpolator as ScipyRBFInterpolator

    # GPU-accelerated RBF when available (6× faster at n≥5000)
    try:
        from gpu_rbf import GPURBFInterpolator, HAS_CUDA
        use_gpu_rbf = HAS_CUDA
    except ImportError:
        use_gpu_rbf = False

    if use_gpu_rbf:
        print("  🚀 GPU RBF: ENABLED (PyTorch CUDA)")
    else:
        print("  ⚠ GPU RBF: disabled (falling back to scipy CPU)")

    # Adaptive strategy: CPU for small n (overhead not worth it), GPU for large n
    GPU_RBF_THRESHOLD = 2000  # Switch to GPU when training data exceeds this

    bounds = load_bounds()
    dim = len(bounds)

    # Initialize parallel workers if requested
    parallel = workers > 0
    if parallel and level >= 2:
        init_parallel(workers=workers)
        print(f"  Parallel: {workers} workers (BLAS=1 per worker)")
    elif parallel and level == 1:
        print("  Note: L1 SEMF is fast enough, parallel not needed")
        parallel = False

    # Select objective based on level
    if level >= 2:
        obj_func = lambda x: nuclear_eos_objective_l2(
            x, nuclei_subset=L2_FOCUSED_NUCLEI, parallel=parallel)
        chi2_func = lambda x: nuclear_eos_chi2_l2(
            x, nuclei_subset=L2_FOCUSED_NUCLEI, parallel=parallel)
        model_desc = "Skyrme → HF+BCS (medium A) + SEMF (light/heavy) → χ²(AME2020)"
        n_nuclei = len(L2_FOCUSED_NUCLEI)
    else:
        obj_func = nuclear_eos_objective
        chi2_func = nuclear_eos_chi2
        model_desc = "Skyrme → nuclear matter → SEMF → χ²(AME2020)"
        n_nuclei = "all"

    accel_str = f"{workers} workers" if parallel else "CPU sequential"
    if use_gpu_rbf:
        accel_str += " + GPU RBF"

    print("=" * 70)
    print(f"NUCLEAR EOS SURROGATE LEARNING — Level {level}")
    print("No HFBTHO. No Code Ocean. No permissions. Just physics.")
    print("=" * 70)
    print(f"  Objective: {model_desc}")
    print(f"  Nuclei: {n_nuclei}")
    print(f"  Parameters: {dim} ({', '.join(PARAM_NAMES)})")
    print(f"  Evals/round: {evals_per_round}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Tolerance: {tol}")
    print(f"  Acceleration: {accel_str}")
    print(f"  BLAS threads: {os.environ.get('OPENBLAS_NUM_THREADS', '?')}")
    print("=" * 70)

    # Wrap objective with cache (captures ALL evaluations)
    cached = CachedObjective(obj_func, name=f"nuclear_eos_L{level}")

    # Initialize sampler (same pattern as full_iterative_workflow.py)
    npts = 16
    sampler = SparsitySampler(bounds, cached, npts=npts, solver=solver)

    # Generate independent test set for surrogate quality scoring
    n_test = 200 if level == 1 else 20  # Fewer test points for slow L2 (~60s/eval)
    test_x = np.array([np.random.uniform(*zip(*bounds)) for _ in range(n_test)])
    test_y = np.array([obj_func(xi) for xi in test_x])
    print(f"  Test set: {n_test} points, mean={np.mean(test_y):.2f}, "
          f"min={np.min(test_y):.4f}, max={np.max(test_y):.2f}")

    history = []
    t_start_total = time.time()

    for round_i in range(max_rounds):
        t0 = time.time()

        # Sample objective function
        target_evals = evals_per_round * (round_i + 1)
        sampler.sample_until(iters=target_evals)

        # Get ALL cached evaluations from our wrapper
        X, y = cached.get_data()
        n_evals = len(X)

        # Filter out penalty values (invalid physics)
        mask = y < 1e3
        n_clean = int(mask.sum())
        X_clean = X[mask] if n_clean > dim + 1 else X
        y_clean = y[mask] if n_clean > dim + 1 else y

        # Train RBF surrogate on clean data
        # Use GPU when data is large enough to benefit
        try:
            if len(X_clean) > dim + 1:
                rbf_t0 = time.time()
                if use_gpu_rbf and len(X_clean) >= GPU_RBF_THRESHOLD:
                    surr = GPURBFInterpolator(X_clean, y_clean,
                                              kernel='thin_plate_spline')
                    rbf_backend = "GPU"
                else:
                    surr = ScipyRBFInterpolator(X_clean, y_clean,
                                                kernel='thin_plate_spline')
                    rbf_backend = "CPU"
                rbf_dt = time.time() - rbf_t0

                # Score on independent test set
                test_pred = surr(test_x)
                score = float(np.mean(np.abs(test_y - test_pred)))
            else:
                score = np.inf
                rbf_dt = 0
                rbf_backend = "skip"
        except Exception as e:
            print(f"    RBF fit failed ({e}), retrying on CPU...")
            try:
                surr = ScipyRBFInterpolator(X_clean, y_clean,
                                            kernel='thin_plate_spline')
                test_pred = surr(test_x)
                score = float(np.mean(np.abs(test_y - test_pred)))
                rbf_backend = "CPU(fallback)"
                rbf_dt = 0
            except Exception as e2:
                print(f"    RBF fit failed on CPU too: {e2}")
                score = np.inf
                rbf_dt = 0
                rbf_backend = "fail"

        best_y = float(np.min(y_clean)) if len(y_clean) > 0 else np.inf
        # Convert best log(1+χ²) back to actual χ²
        best_chi2 = float(np.expm1(best_y))
        dt = time.time() - t0

        history.append({
            "round": round_i,
            "n_evals": n_evals,
            "n_clean": n_clean,
            "score": score,
            "best_log": best_y,
            "best_chi2": best_chi2,
            "time_s": round(dt, 1),
        })

        print(f"  Round {round_i:3d}: {n_evals:6d} evals ({n_clean:d} clean), "
              f"score={score:.4e}, best_χ²={best_chi2:.2f}, "
              f"RBF={rbf_dt:.1f}s[{rbf_backend}], {dt:.1f}s")

        if score < tol and round_i > 2:
            print(f"\n  ✅ CONVERGED at round {round_i}")
            break

    t_total = time.time() - t_start_total
    converged = score < tol

    # Get best parameters
    best_idx = np.argmin(y)
    best_params = X[best_idx] if len(X) > 0 else None
    best_log = float(np.min(y))
    best_chi2_actual = float(np.expm1(best_log))

    # Get actual χ² for best params (verify)
    if best_params is not None:
        best_chi2_verify = chi2_func(best_params)
    else:
        best_chi2_verify = np.inf

    # Results
    results = {
        "experiment": f"nuclear_eos_skyrme_L{level}",
        "date": time.strftime("%Y-%m-%d"),
        "level": level,
        "description": model_desc,
        "no_hfbtho": True,
        "no_code_ocean": True,
        "no_permissions": True,
        "methodology": "Diaw et al. (2024) SparsitySampler iterative workflow",
        "objective_transform": "log(1 + chi2) for smooth RBF learning",
        "n_params": dim,
        "param_names": PARAM_NAMES,
        "evals_per_round": evals_per_round,
        "max_rounds": max_rounds,
        "tolerance": tol,
        "rounds_used": len(history),
        "total_evaluations": n_evals,
        "final_score": score,
        "best_log1p_chi2": best_log,
        "best_chi2_per_datum": best_chi2_actual,
        "best_chi2_verify": best_chi2_verify,
        "best_params": best_params.tolist() if best_params is not None else None,
        "converged": converged,
        "total_time_s": round(t_total, 1),
        "history": history,
    }

    # Compare to paper (note: different objective functions!)
    model_names = {
        1: "SEMF with Skyrme nuclear matter coefficients",
        2: "Hybrid HFB/SEMF (HFB for 56≤A≤132, SEMF elsewhere)",
    }
    results["comparison"] = {
        "paper_nuclear_eos": {
            "rounds": 30,
            "evals": 30000,
            "final_chi2": 9.2e-6,
            "model": "HFBTHO (full self-consistent DFT)",
            "note": "Code Ocean (gated), Fortran wrapper"
        },
        "our_nuclear_eos": {
            "rounds": len(history),
            "evals": n_evals,
            "final_chi2": best_chi2_actual,
            "model": model_names.get(level, f"Level {level}"),
            "note": "Open, from scratch, no permissions"
        },
        "note": "χ² values not directly comparable — different fidelity models. "
                f"Paper uses full DFT. We use Level {level}. Both demonstrate "
                "surrogate learning methodology works on nuclear physics."
    }

    # Nuclear matter properties of best fit
    if best_params is not None:
        from skyrme_hf import nuclear_matter_properties
        nmp = nuclear_matter_properties(best_params)
        results["best_nuclear_matter"] = nmp

    # Save
    results_dir = os.path.join(SCRIPT_DIR, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    outfile = os.path.join(results_dir, f"nuclear_eos_surrogate_L{level}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULT: {'CONVERGED' if converged else 'MAX ROUNDS'}")
    print(f"  Best χ²/datum: {best_chi2_actual:.4f}  (log: {best_log:.4f})")
    print(f"  Verified χ²: {best_chi2_verify:.4f}")
    print(f"  Surrogate score: {score:.4e}")
    print(f"  Total evals: {n_evals}")
    print(f"  Time: {t_total:.1f}s")
    if best_params is not None:
        print(f"  Best params: {dict(zip(PARAM_NAMES, np.round(best_params, 2)))}")
        print(f"  Nuclear matter: ρ₀={nmp['rho0_fm3']:.3f} fm⁻³, "
              f"E/A={nmp['E_A_MeV']:.2f} MeV, K∞={nmp['K_inf_MeV']:.0f} MeV")
    print(f"  Saved: {outfile}")
    print(f"{'='*70}")

    # Cleanup parallel pool
    if parallel:
        shutdown_parallel()

    return results


if __name__ == "__main__":
    quick = "--quick" in sys.argv

    # Parse --level flag
    level = 1
    for arg in sys.argv:
        if arg.startswith("--level"):
            if "=" in arg:
                level = int(arg.split("=")[1])
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    level = int(sys.argv[idx + 1])

    # Parse --evals flag
    evals_per_round = None
    for arg in sys.argv:
        if arg.startswith("--evals"):
            if "=" in arg:
                evals_per_round = int(arg.split("=")[1])
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    evals_per_round = int(sys.argv[idx + 1])

    # Parse --rounds flag
    max_rounds = 30
    for arg in sys.argv:
        if arg.startswith("--rounds"):
            if "=" in arg:
                max_rounds = int(arg.split("=")[1])
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    max_rounds = int(sys.argv[idx + 1])

    # Parse --workers flag
    workers = 0
    for arg in sys.argv:
        if arg.startswith("--workers"):
            if "=" in arg:
                workers = int(arg.split("=")[1])
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    workers = int(sys.argv[idx + 1])

    # Defaults per level
    if evals_per_round is None:
        if quick:
            evals_per_round = 50
        elif level >= 2:
            if workers > 0:
                evals_per_round = 100  # Parallel makes more evals/round tractable
            else:
                evals_per_round = 30  # HFB is ~40s/eval, keep rounds manageable
        else:
            evals_per_round = 200

    # tol is on the surrogate MAE on test set (in log space)
    # A tol of 0.5 means we can predict log(1+χ²) within 0.5
    # That's ~exp(0.5) ≈ 1.65x multiplicative accuracy on χ²
    run_nuclear_eos_surrogate(
        evals_per_round=evals_per_round,
        max_rounds=max_rounds,
        tol=0.5,
        level=level,
        workers=workers,
    )
