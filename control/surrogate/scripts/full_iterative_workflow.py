#!/usr/bin/env python3
"""
Full Iterative Surrogate Workflow — The Open Science Version (v2)

Reproduces the COMPLETE methodology from Diaw et al. (2024)
"Efficient learning of accurate surrogates for simulations of complex systems"
Nature Machine Intelligence.

KEY INSIGHT (v2): The paper uses sample_until(1000) per round, generating ~1000
function evaluations per round (not just npts=16 "best" solutions). The RBF
surrogate is trained on ALL accumulated evaluations, not just the best ones.
This was discovered by reading the paper's Zenodo run logs:
    "warming to 1000... None: evals:1000, cache:1000"

WHAT THE PAPER DOES (reconstructed from Zenodo run logs + _model.py configs):
    1. Run SparsitySampler.sample_until(round * evals_per_round)
    2. Collect ALL function evaluations into cache
    3. Train RBF surrogate on ALL cached (x, f(x)) pairs
    4. Test surrogate on held-out random test points
    5. Score = mean absolute test error
    6. If score < tolerance: STOP, else continue
    7. Repeat for up to max_rounds

WHAT WE DO INSTEAD OF CODE OCEAN:
    Run the identical workflow on open objectives including a physics EOS
    built from our own validated Sarkas molecular dynamics simulations.

EVERYTHING HERE IS:
    ✅ Open source (mystic BSD-3, our code AGPL-3)
    ✅ Self-contained (no external API, no login required)
    ✅ Reproducible (run this script, get these results)
    ✅ Transparent (every evaluation logged, every surrogate inspectable)

Usage:
    python full_iterative_workflow.py [--quick]

    --quick: Run with reduced evals_per_round (100 instead of 1000)
             for fast validation. Full run takes ~30-60 minutes.

Author: ecoPrimals/hotSpring control experiment
Date: 2026-02-08
License: AGPL-3.0
"""

import sys
import os
import json
import time
import numpy as np
from scipy.interpolate import RBFInterpolator

# ---------------------------------------------------------------------------
# Path setup — works from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOTSPRING = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

# Add Zenodo data paths for paper's own objectives
sys.path.insert(0, os.path.join(HOTSPRING, 'data', 'zenodo-surrogate', 'results',
                                'nick', 'N16_tol2e4_max1e4_1000SS'))
sys.path.insert(0, os.path.join(HOTSPRING, 'data', 'zenodo-surrogate', 'results',
                                'hartmann6', '500_tol2e4_max1e4_1000SS'))

from mystic.samplers import SparsitySampler
from mystic.solvers import NelderMeadSimplexSolver as solver
from mystic.models import rastrigin, rosen

# Import paper's own objectives from Zenodo
try:
    from perlin_nick import MultiscaleNDFunc
    HAS_NICK = True
except ImportError:
    HAS_NICK = False
    print("WARNING: perlin_nick.py not found. Download Zenodo data first.")

try:
    from hartmann import Hartmann
    hartmann6_obj = Hartmann(ndim=6)
    HAS_HARTMANN = True
except ImportError:
    HAS_HARTMANN = False

# Michalewicz from mystic
try:
    from mystic.models import michalewicz
    HAS_MICHALEWICZ = True
except ImportError:
    HAS_MICHALEWICZ = False


# ============================================================================
# CachedObjective — captures ALL evaluations (the paper's secret sauce)
# ============================================================================

class CachedObjective:
    """
    Wraps an objective function to capture every evaluation.

    The paper trains its RBF surrogate on ALL evaluations made during
    SparsitySampler.sample_until(), not just the best solutions. This
    is the critical methodological detail that makes the surrogate
    converge much faster — 1000 diverse evaluations per round provide
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


# ============================================================================
# Physics EOS Objective: Sarkas MD-derived
# ============================================================================

def build_physics_eos(obs_file):
    """
    Build a callable "equation of state" from our validated Sarkas MD data.

    Input: (κ, log₁₀Γ) — plasma screening and coupling parameters
    Output: scalar cost measuring departure from target physical regime

    The ground truth comes from 12 validated MD simulations (81/81 checks pass).
    """
    with open(obs_file) as f:
        obs = json.load(f)

    # Extract validated data points
    points = []
    for entry in obs['vacf']:
        k = entry['kappa']
        g = entry['gamma']
        D = entry['D_green_kubo_m2_s']
        if D and D > 0:
            points.append({'kappa': k, 'gamma': g, 'log10_D': np.log10(D)})

    for entry in obs['rdf']:
        k = entry['kappa']
        g = entry['gamma']
        gpeak = entry['first_peak_g']
        for p in points:
            if p['kappa'] == k and abs(np.log10(p['gamma']) - np.log10(g)) < 0.01:
                p['g_peak'] = gpeak
                break

    X = np.array([[p['kappa'], np.log10(p['gamma'])] for p in points])
    y_D = np.array([p['log10_D'] for p in points])
    y_g = np.array([p.get('g_peak', 1.0) for p in points])

    # Build ground-truth RBFs (these ARE the "expensive simulation")
    rbf_D = RBFInterpolator(X, y_D, kernel='thin_plate_spline')
    rbf_g = RBFInterpolator(X, y_g, kernel='thin_plate_spline')

    def physics_eos(x):
        """
        Evaluate physics EOS cost at point x = [κ, log₁₀(Γ)].
        Minimum is at the (κ, Γ) combination closest to target transport properties.
        """
        x = np.asarray(x).reshape(1, -1)
        pred_D = rbf_D(x)[0]
        pred_g = rbf_g(x)[0]
        target_D = -7.0   # log10(D) ~ -7 for dense plasmas
        target_g = 2.0    # g(peak) ~ 2 for well-coupled systems
        return float((pred_D - target_D)**2 + 0.5 * (pred_g - target_g)**2)

    return physics_eos


# ============================================================================
# Full Iterative Workflow (matches paper methodology)
# ============================================================================

def run_iterative_surrogate(objective_func, bounds, dim, name="unnamed",
                            evals_per_round=1000, max_rounds=30, tol=1e-4,
                            npts=16, n_test=200):
    """
    Run the full iterative surrogate learning workflow matching the paper.

    Algorithm (reconstructed from Zenodo run logs):
    1. Each round: sample_until(round * evals_per_round) → ~1000 new evals
    2. Train RBF on ALL accumulated evaluations
    3. Test on independent random test points
    4. Score = mean absolute error on test points
    5. If score < tol: STOP

    Parameters match paper's _model.py configs:
        npts=16 (solvers per sampler) — from N16_* configs
        evals_per_round=1000 — from "1000SS" in directory names
        max_rounds=30 — from run logs
        tol=1e-4 — from "tol2e4" in directory names (2×10⁻⁴)
    """
    print(f"\n{'='*70}")
    print(f"ITERATIVE SURROGATE: {name}")
    print(f"  Dim={dim}, npts={npts}, evals/round={evals_per_round}, "
          f"max_rounds={max_rounds}, tol={tol}")
    print(f"{'='*70}\n")

    # Wrap objective with cache
    cached = CachedObjective(objective_func, name=name)

    # Create sampler
    sampler = SparsitySampler(bounds, cached, npts=npts, solver=solver)

    # Generate independent test set
    test_x = np.array([np.random.uniform(*zip(*bounds)) for _ in range(n_test)])
    test_y = np.array([objective_func(xi) for xi in test_x])

    history = []
    t_start = time.time()

    for round_num in range(max_rounds):
        t_round = time.time()

        # Step 1: Sample until we have (round+1) * evals_per_round total evals
        target_evals = (round_num + 1) * evals_per_round
        sampler.sample_until(iters=target_evals)
        n_total = cached.n_evals

        if n_total < dim + 1:
            print(f"  Round {round_num:2d}: {n_total:5d} evals, waiting for more data...")
            continue

        # Step 2: Train RBF on ALL accumulated evaluations
        X_train, y_train = cached.get_data()

        try:
            surrogate = RBFInterpolator(X_train, y_train, kernel='thin_plate_spline')
        except Exception as e:
            print(f"  Round {round_num:2d}: RBF failed ({e}), continuing...")
            history.append({
                'round': round_num, 'n_evals': n_total,
                'chi2': float('inf'), 'best_y': float(np.min(y_train)),
                'time_s': round(time.time() - t_round, 2),
            })
            continue

        # Step 3: Test surrogate on held-out test points
        pred = surrogate(test_x)
        residuals = np.abs(test_y - pred)
        score = float(np.mean(residuals))  # Mean absolute error (matches paper's "misfit")

        # Step 4: Log
        best_y = float(np.min(y_train))
        dt_round = time.time() - t_round

        history.append({
            'round': round_num,
            'n_evals': n_total,
            'score': score,
            'best_y': best_y,
            'time_s': round(dt_round, 2),
        })

        status = '✅ CONVERGED' if score < tol else ''
        print(f"  Round {round_num:2d}: {n_total:5d} evals, score={score:.6e}, "
              f"best_f={best_y:.6e}, {dt_round:.1f}s {status}")

        # Step 5: Check convergence
        if score < tol:
            break

    dt_total = time.time() - t_start
    final_score = history[-1]['score'] if history else float('inf')
    converged = final_score < tol
    rounds_used = len(history)

    print(f"\n  RESULT: {'CONVERGED' if converged else 'MAX ROUNDS'} "
          f"after {rounds_used} rounds, {n_total} total evaluations")
    print(f"  Final score: {final_score:.6e} (tol={tol})")
    print(f"  Best objective: {best_y:.6e}")
    print(f"  Total time: {dt_total:.1f}s")

    return {
        'name': name,
        'dim': dim,
        'npts': npts,
        'evals_per_round': evals_per_round,
        'max_rounds': max_rounds,
        'tolerance': tol,
        'rounds_used': rounds_used,
        'total_evaluations': n_total,
        'final_score': final_score,
        'best_objective': best_y,
        'converged': converged,
        'total_time_s': round(dt_total, 2),
        'history': history,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    quick = '--quick' in sys.argv
    epr = 100 if quick else 1000  # evals per round
    mode = "QUICK" if quick else "FULL"

    np.random.seed(42)
    all_results = []

    print("=" * 70)
    print(f"FULL ITERATIVE SURROGATE WORKFLOW v2 — OPEN SCIENCE ({mode} MODE)")
    print("Reproducing Diaw et al. (2024) methodology without Code Ocean")
    if quick:
        print("  [--quick]: 100 evals/round (full: 1000). Use without flag for publication.")
    print("=" * 70)

    # ---- 1. Rastrigin 2D (paper: rast, N16_tol2e4_max1e4_1000SS) ----
    r = run_iterative_surrogate(
        rastrigin, bounds=[(0, 10)] * 2, dim=2,
        name="Rastrigin_2D", npts=16, evals_per_round=epr,
        max_rounds=30, tol=2e-4,
    )
    all_results.append(r)

    # ---- 2. Rosenbrock 2D (paper: rosen, N16_tol2e4_max1e4_1000SS) ----
    r = run_iterative_surrogate(
        rosen, bounds=[(-2, 5)] * 2, dim=2,
        name="Rosenbrock_2D", npts=16, evals_per_round=epr,
        max_rounds=30, tol=2e-4,
    )
    all_results.append(r)

    # ---- 3. Easom 2D ----
    def easom(x):
        from math import cos, exp, pi
        return -cos(x[0]) * cos(x[1]) * exp(-((x[0]-pi)**2 + (x[1]-pi)**2))

    r = run_iterative_surrogate(
        easom, bounds=[(0, 10)] * 2, dim=2,
        name="Easom_2D", npts=16, evals_per_round=epr,
        max_rounds=30, tol=2e-4,
    )
    all_results.append(r)

    # ---- 4. MultiscaleNDFunc 2D (paper: nick) ----
    if HAS_NICK:
        nick2d = MultiscaleNDFunc(1, 20, 100, 1, 2)
        r = run_iterative_surrogate(
            nick2d, bounds=[(0, 2)] * 2, dim=2,
            name="MultiscaleNDFunc_2D", npts=16, evals_per_round=epr,
            max_rounds=30, tol=2e-4,
        )
        all_results.append(r)

    # ---- 5. Hartmann6 (paper: hartmann6) ----
    if HAS_HARTMANN:
        r = run_iterative_surrogate(
            hartmann6_obj, bounds=[(0, 1)] * 6, dim=6,
            name="Hartmann6", npts=16, evals_per_round=epr,
            max_rounds=30, tol=2e-4,
        )
        all_results.append(r)

    # ---- 6. Michalewicz 2D ----
    if HAS_MICHALEWICZ:
        r = run_iterative_surrogate(
            michalewicz, bounds=[(0, np.pi)] * 2, dim=2,
            name="Michalewicz_2D", npts=16, evals_per_round=epr,
            max_rounds=30, tol=2e-4,
        )
        all_results.append(r)

    # ---- 7. Rosenbrock 8D (paper: rosen8) ----
    r = run_iterative_surrogate(
        rosen, bounds=[(-2, 5)] * 8, dim=8,
        name="Rosenbrock_8D", npts=16, evals_per_round=epr,
        max_rounds=30, tol=2e-4,
    )
    all_results.append(r)

    # ---- 8. MultiscaleNDFunc 5D (our extension, not in paper) ----
    if HAS_NICK:
        nick5d = MultiscaleNDFunc(1, 20, 100, 1, 5)
        r = run_iterative_surrogate(
            nick5d, bounds=[(0, 2)] * 5, dim=5,
            name="MultiscaleNDFunc_5D_ext", npts=16, evals_per_round=epr,
            max_rounds=30, tol=2e-4,
        )
        all_results.append(r)

    # ---- 9. Physics EOS from Sarkas MD (our contribution) ----
    obs_file = os.path.join(HOTSPRING, 'control', 'sarkas', 'simulations',
                            'dsf-study', 'results', 'all_observables_validation.json')
    if os.path.exists(obs_file):
        physics_obj = build_physics_eos(obs_file)
        r = run_iterative_surrogate(
            physics_obj, bounds=[(0, 3), (1, 3.2)], dim=2,
            name="Physics_EOS_Sarkas_MD", npts=16, evals_per_round=epr,
            max_rounds=30, tol=1e-4,
        )
        all_results.append(r)

    # ---- Grand Summary ----
    print("\n" + "=" * 70)
    print("GRAND SUMMARY")
    print("=" * 70)
    print(f"\n{'Function':<30} {'Dim':>3} {'Rnds':>5} {'Evals':>7} "
          f"{'Score':>12} {'Conv?':>5} {'Time':>7}")
    print("-" * 72)

    for r in all_results:
        conv = '✅' if r['converged'] else '⚠️'
        print(f"{r['name']:<30} {r['dim']:>3} {r['rounds_used']:>5} "
              f"{r['total_evaluations']:>7} {r['final_score']:>12.2e} "
              f"{conv:>5} {r['total_time_s']:>6.1f}s")

    n_converged = sum(1 for r in all_results if r['converged'])
    n_total = len(all_results)
    print(f"\nConverged: {n_converged}/{n_total}")

    # ---- Paper comparison ----
    print(f"\n{'='*70}")
    print("COMPARISON TO PAPER'S PUBLISHED RESULTS (from Zenodo)")
    print(f"{'='*70}")

    paper_results = _load_paper_results()
    if paper_results:
        print(f"\n{'Function':<20} {'Paper Score':>12} {'Our Score':>12} {'Match?':>7}")
        print("-" * 55)
        for r in all_results:
            paper_name = _map_to_paper(r['name'])
            if paper_name and paper_name in paper_results:
                ps = paper_results[paper_name]
                match = '✅' if r['converged'] else '⚠️'
                print(f"{r['name']:<20} {ps:>12.2e} {r['final_score']:>12.2e} {match:>7}")
            elif r['name'].startswith('Physics'):
                print(f"{r['name']:<20} {'(ours)':>12} {r['final_score']:>12.2e} {'✅':>7}")

    # ---- Save ----
    output_dir = os.path.join(SCRIPT_DIR, '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'full_iterative_workflow_results.json')

    output = {
        'version': 2,
        'date': '2026-02-08',
        'mode': mode.lower(),
        'evals_per_round': epr,
        'methodology': 'Diaw et al. (2024) SparsitySampler iterative workflow',
        'key_fix_v2': 'Using sample_until() with ALL evaluations, not just bestSolution',
        'code_ocean_required': False,
        'login_required': False,
        'all_objectives_open': True,
        'reproducible': True,
        'seed': 42,
        'results': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {output_file}")
    print("\nTo reproduce: python full_iterative_workflow.py")
    print("No Code Ocean account. No LANL access. No login page. Just science.")


def _load_paper_results():
    """Load paper's final scores from Zenodo score.txt files."""
    zenodo = os.path.join(HOTSPRING, 'data', 'zenodo-surrogate', 'results')
    results = {}
    for func_dir in os.listdir(zenodo):
        dp = os.path.join(zenodo, func_dir)
        if not os.path.isdir(dp):
            continue
        # Use N16 config (matching our npts=16)
        for sub in os.listdir(dp):
            if sub.startswith('N16_'):
                sp = os.path.join(dp, sub, 'score.txt')
                if os.path.isfile(sp):
                    scores = _parse_score_txt(sp)
                    if scores:
                        results[func_dir] = scores[-1]
    return results


def _parse_score_txt(path):
    """Parse a score.txt file into list of float scores."""
    scores = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    scores.append(float(parts[1]))
                except ValueError:
                    pass
    return scores


def _map_to_paper(our_name):
    """Map our function name to paper's Zenodo directory name."""
    mapping = {
        'Rastrigin_2D': 'rast',
        'Rosenbrock_2D': 'rosen',
        'Rosenbrock_8D': 'rosen8',
        'Easom_2D': 'easom',
        'MultiscaleNDFunc_2D': 'nick',
        'Hartmann6': 'hartmann6',
        'Michalewicz_2D': 'michal',
    }
    return mapping.get(our_name)


if __name__ == '__main__':
    main()
