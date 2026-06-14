#!/usr/bin/env python3
"""
Run all transport study cases and extract transport coefficients.

Workflow per case:
  1. PreProcess  — validate parameters
  2. Simulation  — run equilibration + production MD
  3. PostProcess — compute VACF observable
  4. Extract D* from VACF via Green-Kubo integration

Output: results/transport_baseline.json with full provenance.

Usage:
    python3 run_transport.py [--case transport_k1_G10] [--skip-existing]
"""

import argparse
import json
import os
import sys
import time
import types
import datetime
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
study_dir = os.path.dirname(script_dir)
sarkas_root = os.path.abspath(
    os.path.join(study_dir, "..", "..", "sarkas-upstream")
)

# Stub fdint (only needed for electron gas, not Yukawa)
fdint = types.ModuleType("fdint")
fdint.fdk = lambda k, phi: 0.0
fdint.ifd1h = lambda x: 0.0
sys.modules["fdint"] = fdint

sys.path.insert(0, sarkas_root)
os.chdir(study_dir)

import matplotlib
matplotlib.use("Agg")
import numpy as np


def get_sarkas_commit():
    """Get the sarkas upstream git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=sarkas_root
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_single_case(yaml_path, skip_existing=False):
    """Run a single transport case and return D* result."""
    from sarkas.processes import PreProcess, Simulation, PostProcess
    from sarkas.tools.observables import VelocityAutoCorrelationFunction

    yaml_path = os.path.abspath(yaml_path)
    basename = os.path.splitext(os.path.basename(yaml_path))[0]

    print(f"\n{'=' * 60}")
    print(f"  Case: {basename}")
    print(f"{'=' * 60}")

    # Check if simulation output already exists
    job_dir = os.path.join(study_dir, "Simulations", basename)
    if skip_existing and os.path.isdir(job_dir):
        print(f"  [SKIP] Output exists at {job_dir}")
        # Still try post-processing
    else:
        # PreProcess
        t0 = time.time()
        preproc = PreProcess(yaml_path)
        preproc.setup(read_yaml=True)
        preproc.run(timing=True, remove=True)
        print(f"  PreProcess: {time.time() - t0:.1f}s")

        # Simulation
        t0 = time.time()
        sim = Simulation(yaml_path)
        sim.setup(read_yaml=True)
        sim.run()
        sim_time = time.time() - t0
        print(f"  Simulation: {sim_time:.1f}s")

    # PostProcess — compute VACF
    t0 = time.time()
    postproc = PostProcess(yaml_path)
    postproc.setup(read_yaml=True)

    # VACF
    vacf = VelocityAutoCorrelationFunction()
    vacf.setup(postproc.parameters)
    vacf.compute()
    print(f"  PostProcess (VACF): {time.time() - t0:.1f}s")

    # Extract D* via Green-Kubo from Sarkas VACF dataframe
    # Sarkas stores multi-index: ('Time', nan, nan), ('H VACF', 'Total', 'Mean')
    params = postproc.parameters
    omega_p = params.total_plasma_frequency
    a_ws = params.a_ws

    acf_df = vacf.dataframe_acf
    # Extract time and total VACF columns via positional access
    t_arr = acf_df.iloc[:, 0].values  # Time column
    c_arr = acf_df.iloc[:, 7].values  # ('H VACF', 'Total', 'Mean')

    c0 = c_arr[0] if abs(c_arr[0]) > 1e-30 else 1.0
    c_norm = c_arr / c0

    # Green-Kubo: D = (1/3) integral <v(0).v(t)> dt
    # The factor 1/3 is already accounted for since 'Total' is sum of X+Y+Z
    # and D = integral_0^inf (1/3)<v(0).v(t)> dt
    d_mks = np.trapz(c_arr, t_arr) / 3.0

    # Convert to reduced units: D* = D / (a_ws^2 * omega_p)
    d_star_reduced = d_mks / (a_ws**2 * omega_p)

    # Parse kappa and gamma from filename
    parts = basename.split("_")
    kappa = float(parts[1][1:])
    gamma = float(parts[2][1:])

    print(f"  D* (reduced) = {d_star_reduced:.6e}")
    print(f"  C(0) = {c0:.6e}, C(t_max)/C(0) = {c_norm[-1]:.4f}")

    return {
        "case": basename,
        "kappa": kappa,
        "gamma": gamma,
        "D_star_reduced": float(d_star_reduced),
        "D_mks": float(d_mks),
        "C0": float(c0),
        "C_tail_ratio": float(c_norm[-1]),
        "n_vacf_points": len(t_arr),
        "t_max": float(t_arr[-1]),
        "sim_wall_time_s": sim_time if "sim_time" in dir() else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Run transport study cases")
    parser.add_argument("--case", type=str, default=None,
                        help="Run a single case (e.g. transport_k1_G10)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cases whose simulation output already exists")
    parser.add_argument("--lite", action="store_true",
                        help="Run lite (reduced N/steps) cases only")
    args = parser.parse_args()

    input_dir = os.path.join(study_dir, "input_files")
    results_dir = os.path.join(study_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.case:
        yaml_files = [os.path.join(input_dir, f"{args.case}.yaml")]
    else:
        manifest_name = "MANIFEST_LITE.txt" if args.lite else "MANIFEST.txt"
        manifest = os.path.join(input_dir, manifest_name)
        with open(manifest) as f:
            yaml_files = [
                os.path.join(input_dir, line.strip())
                for line in f if line.strip()
            ]

    results = []
    failed = []
    t_total = time.time()

    for yaml_path in yaml_files:
        if not os.path.exists(yaml_path):
            print(f"WARNING: {yaml_path} not found, skipping")
            continue
        try:
            result = run_single_case(yaml_path, args.skip_existing)
            results.append(result)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            failed.append((os.path.basename(yaml_path), str(ex)))

    total_time = time.time() - t_total

    provenance = {
        "generator": "hotSpring/control/sarkas/simulations/transport-study/scripts/run_transport.py",
        "sarkas_commit": get_sarkas_commit(),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "total_wall_time_s": total_time,
        "n_cases": len(results),
        "n_failed": len(failed),
    }

    output = {
        "provenance": provenance,
        "results": results,
        "failed": failed,
    }

    suffix = "_lite" if args.lite else ""
    output_path = os.path.join(results_dir, f"transport_baseline{suffix}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"TRANSPORT STUDY COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Cases run:    {len(results)}")
    print(f"  Failures:     {len(failed)}")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"  Output:       {output_path}")

    if failed:
        print(f"\n  FAILED CASES:")
        for name, err in failed:
            print(f"    {name}: {err}")

    for r in results:
        print(f"  k={r['kappa']:.0f} G={r['gamma']:<5.0f}  D*={r['D_star_reduced']:.4e}")


if __name__ == "__main__":
    main()
