#!/usr/bin/env python
"""
hotSpring - Surrogate Learning Control: Benchmark Functions

Reproduce the optimizer-driven sampling method from:
  "Efficient learning of accurate surrogates for simulations of complex systems"
  Diaw, McKerns, Sagert, Stanton, Murillo - Nature Machine Intelligence, May 2024

Phase 1: Run on standard benchmark functions (Rastrigin, Rosenbrock, Easom).
These are fast and validate that the mystic-based directed sampling approach
works correctly on our hardware before moving to the expensive nuclear EOS runs.

Usage:
    python run_benchmark_functions.py [--output-dir DIR] [--n-samples N]

Env: conda activate surrogate
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCRIPT_DIR.parent
HOTSPRING_DIR = CONTROL_DIR.parents[1]
RESULTS_DIR = CONTROL_DIR / "results"


# ============================================================
# Benchmark test functions
# ============================================================

def rastrigin(x):
    """Rastrigin function: many local minima, global min at origin."""
    A = 10
    n = len(x)
    return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)


def rosenbrock(x):
    """Rosenbrock function: narrow curved valley, global min at (1,1,...,1)."""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def easom(x):
    """Easom function (2D only): flat with sharp peak at (pi, pi)."""
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))


BENCHMARKS = {
    "rastrigin_2d": {
        "func": rastrigin,
        "ndim": 2,
        "bounds": [(-5.12, 5.12)] * 2,
        "global_min": 0.0,
        "global_min_x": [0.0, 0.0],
    },
    "rastrigin_5d": {
        "func": rastrigin,
        "ndim": 5,
        "bounds": [(-5.12, 5.12)] * 5,
        "global_min": 0.0,
        "global_min_x": [0.0] * 5,
    },
    "rosenbrock_2d": {
        "func": rosenbrock,
        "ndim": 2,
        "bounds": [(-5.0, 10.0)] * 2,
        "global_min": 0.0,
        "global_min_x": [1.0, 1.0],
    },
    "rosenbrock_5d": {
        "func": rosenbrock,
        "ndim": 5,
        "bounds": [(-5.0, 10.0)] * 5,
        "global_min": 0.0,
        "global_min_x": [1.0] * 5,
    },
    "easom_2d": {
        "func": easom,
        "ndim": 2,
        "bounds": [(-100.0, 100.0)] * 2,
        "global_min": -1.0,
        "global_min_x": [np.pi, np.pi],
    },
}


# ============================================================
# Sampling strategies
# ============================================================

def random_sampling(func, bounds, n_samples):
    """Baseline: uniform random sampling."""
    ndim = len(bounds)
    samples = np.array([
        np.random.uniform(lo, hi, n_samples) for lo, hi in bounds
    ]).T
    values = np.array([func(s) for s in samples])
    return samples, values


def latin_hypercube_sampling(func, bounds, n_samples):
    """Latin Hypercube Sampling — better space coverage than pure random."""
    ndim = len(bounds)
    # Simple LHS: divide each dimension into n_samples intervals, sample one per interval
    samples = np.zeros((n_samples, ndim))
    for d in range(ndim):
        lo, hi = bounds[d]
        perm = np.random.permutation(n_samples)
        intervals = np.linspace(lo, hi, n_samples + 1)
        for i in range(n_samples):
            samples[perm[i], d] = np.random.uniform(intervals[i], intervals[i+1])
    values = np.array([func(s) for s in samples])
    return samples, values


def mystic_directed_sampling(func, bounds, n_samples, n_rounds=5):
    """
    Optimizer-driven directed sampling using mystic.
    This is the key innovation from the paper: use constrained optimization
    to direct sampling toward informative regions of the function space.

    Each round:
      1. Run mystic optimizer to find promising regions
      2. Sample around those regions
      3. Accumulate samples
    """
    try:
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.monitors import VerboseMonitor
        from mystic.termination import ChangeOverGeneration
    except ImportError:
        print("ERROR: mystic not installed. Run: pip install mystic")
        sys.exit(1)

    ndim = len(bounds)
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]

    all_samples = []
    all_values = []
    samples_per_round = n_samples // n_rounds

    for rnd in range(n_rounds):
        # Run differential evolution to find optimizer trajectory
        solver = DifferentialEvolutionSolver2(ndim, samples_per_round)
        solver.SetRandomInitialPoints(lo, hi)
        solver.SetStrictRanges(lo, hi)
        solver.SetTermination(ChangeOverGeneration(tolerance=1e-8, generations=50))
        solver.SetGenerationMonitor(VerboseMonitor(0))  # silent

        solver.Solve(func)

        # Collect the population from this round (the optimizer's evaluated points)
        population = np.array(solver.population)
        energies = np.array([func(p) for p in population])

        all_samples.append(population)
        all_values.append(energies)

        # Also add perturbation samples around the best point
        best = solver.bestSolution
        n_perturb = max(1, samples_per_round // 4)
        for _ in range(n_perturb):
            perturbed = best + np.random.normal(0, 0.1, ndim) * np.array([hi_i - lo_i for lo_i, hi_i in bounds])
            perturbed = np.clip(perturbed, lo, hi)
            all_samples.append(perturbed.reshape(1, -1))
            all_values.append(np.array([func(perturbed)]))

    samples = np.vstack(all_samples)
    values = np.concatenate(all_values)

    return samples, values


def train_surrogate(samples, values):
    """
    Train a simple RBF surrogate model on the collected samples.
    Returns the trained model and training metrics.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # Try RBF interpolation first (what the paper uses conceptually)
    try:
        from scipy.interpolate import RBFInterpolator
        X_train, X_test, y_train, y_test = train_test_split(
            samples, values, test_size=0.2, random_state=42
        )
        rbf = RBFInterpolator(X_train, y_train, kernel='thin_plate_spline')
        y_pred = rbf(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"model": "RBF_TPS", "mse": mse, "r2": r2,
                "n_train": len(X_train), "n_test": len(X_test)}
    except ImportError:
        # Fallback to sklearn's KernelRidge
        from sklearn.kernel_ridge import KernelRidge
        X_train, X_test, y_train, y_test = train_test_split(
            samples, values, test_size=0.2, random_state=42
        )
        model = KernelRidge(kernel='rbf', alpha=1e-3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"model": "KernelRidge_RBF", "mse": mse, "r2": r2,
                "n_train": len(X_train), "n_test": len(X_test)}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Surrogate learning benchmark functions")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: control/surrogate/results/)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples per strategy (default: 500)")
    parser.add_argument("--functions", default=None,
                        help="Comma-separated benchmark names (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    func_names = list(BENCHMARKS.keys())
    if args.functions:
        func_names = [f.strip() for f in args.functions.split(",")]

    strategies = {
        "random": random_sampling,
        "lhs": latin_hypercube_sampling,
        "mystic_directed": mystic_directed_sampling,
    }

    print("=" * 70)
    print("hotSpring - Surrogate Learning: Benchmark Functions")
    print("=" * 70)
    print(f"Functions: {func_names}")
    print(f"Samples per strategy: {args.n_samples}")
    print(f"Strategies: {list(strategies.keys())}")
    print(f"Output: {output_dir}")
    print()

    all_results = []

    for func_name in func_names:
        if func_name not in BENCHMARKS:
            print(f"WARNING: Unknown benchmark '{func_name}', skipping.")
            continue

        bench = BENCHMARKS[func_name]
        print(f"\n{'='*50}")
        print(f"Benchmark: {func_name} (ndim={bench['ndim']})")
        print(f"{'='*50}")

        for strat_name, strat_func in strategies.items():
            print(f"\n  Strategy: {strat_name}")
            t0 = time.time()

            try:
                np.random.seed(42)  # Reproducibility
                samples, values = strat_func(bench["func"], bench["bounds"], args.n_samples)
                sampling_time = time.time() - t0
                print(f"    Sampling: {sampling_time:.2f}s, {len(samples)} samples")

                # Train surrogate
                t1 = time.time()
                metrics = train_surrogate(samples, values)
                training_time = time.time() - t1
                print(f"    Training: {training_time:.2f}s")
                print(f"    Model: {metrics['model']}")
                print(f"    MSE: {metrics['mse']:.6e}")
                print(f"    R²:  {metrics['r2']:.6f}")

                # How close did the best sample get to global min?
                best_idx = np.argmin(values)
                best_val = values[best_idx]
                best_x = samples[best_idx]
                gap = abs(best_val - bench["global_min"])
                print(f"    Best sample value: {best_val:.6f} (gap from global min: {gap:.6e})")

                result = {
                    "function": func_name,
                    "strategy": strat_name,
                    "n_samples": len(samples),
                    "sampling_time_s": round(sampling_time, 3),
                    "training_time_s": round(training_time, 3),
                    "model": metrics["model"],
                    "mse": metrics["mse"],
                    "r2": metrics["r2"],
                    "best_value": float(best_val),
                    "global_min": bench["global_min"],
                    "gap": float(gap),
                    "status": "PASS",
                }
            except Exception as ex:
                print(f"    FAILED: {ex}")
                import traceback
                traceback.print_exc()
                result = {
                    "function": func_name,
                    "strategy": strat_name,
                    "status": "FAIL",
                    "error": str(ex),
                }

            all_results.append(result)

    # Save results
    results_file = output_dir / "benchmark_functions_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Function':<20} {'Strategy':<18} {'MSE':>12} {'R²':>10} {'Gap':>12} {'Time':>8}")
    print("-" * 80)
    for r in all_results:
        if r.get("status") == "PASS":
            total_t = r.get("sampling_time_s", 0) + r.get("training_time_s", 0)
            print(f"{r['function']:<20} {r['strategy']:<18} "
                  f"{r['mse']:>12.2e} {r['r2']:>10.4f} "
                  f"{r['gap']:>12.2e} {total_t:>7.1f}s")
        else:
            print(f"{r['function']:<20} {r['strategy']:<18} {'FAIL':>12}")

    print(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    main()

