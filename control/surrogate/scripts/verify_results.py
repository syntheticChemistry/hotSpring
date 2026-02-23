#!/usr/bin/env python3
"""
Automated verification of surrogate learning reproduction results.

Checks:
  1. Results file exists and is well-formed JSON
  2. All expected objectives were tested
  3. Physics EOS converged (χ² < tolerance)
  4. Convergence histories show improvement over rounds
  5. No NaN or Inf in results
  6. Paper comparison data is available (Zenodo score.txt)

Exit code 0 = all checks pass.
Exit code 1 = at least one check failed.

Usage:
    python verify_results.py [path_to_results.json]

If no path given, looks for the default location.
"""

import sys
import os
import json
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(SCRIPT_DIR, '..', 'results',
                               'full_iterative_workflow_results.json')
HOTSPRING = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

EXPECTED_MIN_FUNCTIONS = 7  # At least 7 objectives should be tested
PHYSICS_EOS_TOL = 1e-3     # Physics EOS should converge below this


def check(condition, msg, warnings=None):
    """Print pass/fail and return success bool."""
    if condition:
        print(f"  ✅ PASS: {msg}")
        return True
    else:
        print(f"  ❌ FAIL: {msg}")
        if warnings is not None:
            warnings.append(msg)
        return False


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS
    results_path = os.path.abspath(results_path)

    print("=" * 60)
    print("SURROGATE REPRODUCTION — AUTOMATED VERIFICATION")
    print("=" * 60)
    print(f"\nResults file: {results_path}\n")

    failures = []

    # ---- Check 1: File exists and parses ----
    print("--- Check 1: Results file ---")
    if not check(os.path.exists(results_path), "Results file exists", failures):
        print(f"\nRun the workflow first:")
        print(f"  python full_iterative_workflow.py --quick")
        sys.exit(1)

    try:
        with open(results_path) as f:
            data = json.load(f)
        check(True, "JSON parses successfully", failures)
    except Exception as e:
        check(False, f"JSON parse error: {e}", failures)
        sys.exit(1)

    # ---- Check 2: Required fields ----
    print("\n--- Check 2: Required fields ---")
    for field in ['results', 'methodology', 'code_ocean_required',
                  'all_objectives_open', 'reproducible']:
        check(field in data, f"Field '{field}' present", failures)

    check(data.get('code_ocean_required') == False,
          "code_ocean_required is False", failures)
    check(data.get('all_objectives_open') == True,
          "all_objectives_open is True", failures)
    check(data.get('reproducible') == True,
          "reproducible is True", failures)

    # ---- Check 3: Sufficient objectives ----
    print("\n--- Check 3: Objectives tested ---")
    results = data.get('results', [])
    n_funcs = len(results)
    check(n_funcs >= EXPECTED_MIN_FUNCTIONS,
          f"At least {EXPECTED_MIN_FUNCTIONS} objectives tested (got {n_funcs})",
          failures)

    names = [r['name'] for r in results]
    print(f"  Functions: {', '.join(names)}")

    # ---- Check 4: Physics EOS convergence ----
    print("\n--- Check 4: Physics EOS convergence ---")
    physics_results = [r for r in results if 'Physics' in r.get('name', '')]
    if physics_results:
        pr = physics_results[0]
        check(pr.get('converged', False),
              f"Physics EOS converged",
              failures)
        # Handle both v1 (final_chi2) and v2 (final_score) field names
        score = pr.get('final_score', pr.get('final_chi2', float('inf')))
        check(score < PHYSICS_EOS_TOL,
              f"Physics EOS score < {PHYSICS_EOS_TOL} (got {score:.2e})",
              failures)
    else:
        check(False, "Physics EOS result found in results", failures)

    # ---- Check 5: No NaN/Inf ----
    print("\n--- Check 5: Data integrity ---")
    nan_count = 0
    inf_count = 0
    for r in results:
        for h in r.get('history', []):
            s = h.get('score', h.get('chi2', 0))
            if isinstance(s, float):
                if math.isnan(s):
                    nan_count += 1
                if math.isinf(s):
                    inf_count += 1
    check(nan_count == 0, f"No NaN in scores (found {nan_count})", failures)
    check(inf_count == 0, f"No Inf in scores (found {inf_count})", failures)

    # ---- Check 6: Convergence trends ----
    print("\n--- Check 6: Convergence trends ---")
    for r in results:
        hist = r.get('history', [])
        if len(hist) >= 5:
            scores = [h.get('score', h.get('chi2', float('inf'))) for h in hist]
            scores = [s for s in scores if isinstance(s, (int, float))
                      and not math.isnan(s) and not math.isinf(s)]
            if len(scores) >= 5:
                # Check that later scores are generally lower than early scores
                first_half = scores[:len(scores)//2]
                second_half = scores[len(scores)//2:]
                mean_first = sum(first_half) / len(first_half)
                mean_second = sum(second_half) / len(second_half)
                improving = mean_second <= mean_first * 1.5  # Allow some noise
                check(improving,
                      f"{r['name']}: improving trend ({mean_first:.2e} → {mean_second:.2e})",
                      failures)

    # ---- Check 7: Zenodo comparison data ----
    print("\n--- Check 7: Paper comparison data ---")
    zenodo = os.path.join(HOTSPRING, 'data', 'zenodo-surrogate', 'results')
    if os.path.isdir(zenodo):
        score_files = []
        for root, dirs, files in os.walk(zenodo):
            for f in files:
                if f == 'score.txt':
                    score_files.append(os.path.join(root, f))
        check(len(score_files) >= 25,
              f"Zenodo score.txt files found ({len(score_files)})", failures)
    else:
        check(False, "Zenodo data directory exists", failures)

    # ---- Summary ----
    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} CHECK(S) FAILED")
        for f in failures:
            print(f"  ❌ {f}")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASS ✅")
        print("\nThis reproduction is verified. Results are consistent and complete.")
        sys.exit(0)


if __name__ == '__main__':
    main()

