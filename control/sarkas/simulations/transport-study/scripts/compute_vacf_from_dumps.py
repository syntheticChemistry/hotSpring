#!/usr/bin/env python3
"""
Standalone VACF computation from Sarkas dump files.

Sarkas stores initial-state velocities in checkpoint NPZ files rather than
dynamically-updated velocities. This script computes velocities from finite
differences of positions (central difference), then computes VACF and D*
via Green-Kubo integration.

This gives exact algorithmic parity with the Rust VACF implementation in
barracuda/src/md/observables.rs.

Reference: Green-Kubo relation for self-diffusion:
    D = (1/3) integral_0^inf <v(0).v(t)> dt

Usage:
    python compute_vacf_from_dumps.py [--lite] [--case CASE_NAME]
"""

import argparse
import json
import os
import sys
import datetime
import numpy as np
from pathlib import Path


def load_positions_sorted(dump_dir: Path, max_snapshots: int = None):
    """Load position snapshots sorted by step number."""
    files = sorted(dump_dir.glob("checkpoint_*.npz"),
                   key=lambda f: int(f.stem.split("_")[1]))
    if max_snapshots:
        files = files[:max_snapshots]

    positions = []
    times = []
    for f in files:
        d = np.load(f)
        positions.append(d["pos"])
        times.append(float(d["time"]))
    return np.array(positions), np.array(times)


def unwrap_pbc(positions, box_side):
    """Unwrap positions across periodic boundaries for continuous trajectories."""
    n_frames, n_particles, _ = positions.shape
    unwrapped = np.copy(positions)
    for i in range(1, n_frames):
        dr = unwrapped[i] - unwrapped[i - 1]
        dr -= box_side * np.round(dr / box_side)
        unwrapped[i] = unwrapped[i - 1] + dr
    return unwrapped


def compute_velocities_fd(positions, dt):
    """Compute velocities from central finite differences of positions."""
    n_frames = positions.shape[0]
    velocities = np.zeros_like(positions)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    return velocities


def compute_vacf(velocities, max_lag):
    """Compute velocity autocorrelation function averaged over particles and time origins."""
    n_frames, n_particles, _ = velocities.shape
    if max_lag > n_frames // 2:
        max_lag = n_frames // 2

    vacf = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    for tau in range(max_lag):
        n_origins = n_frames - tau
        dot_products = np.sum(
            velocities[:n_origins] * velocities[tau:tau + n_origins], axis=2
        )
        vacf[tau] = np.mean(dot_products)
        counts[tau] = n_origins * n_particles

    return vacf


def green_kubo_diffusion(vacf, dt):
    """Compute D from VACF via Green-Kubo: D = (1/3) integral C(t) dt."""
    return np.trapz(vacf, dx=dt) / 3.0


def compute_d_star(d_mks, a_ws, omega_p):
    """Convert MKS diffusion coefficient to reduced units."""
    return d_mks / (a_ws ** 2 * omega_p)


def parse_yaml_params(yaml_path):
    """Extract physical parameters from Sarkas YAML config header comments."""
    params = {}
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if "a_ws" in line and "=" in line:
                    params["a_ws"] = float(line.split("=")[1].strip().split()[0])
                elif "omega_p" in line and "=" in line:
                    params["omega_p"] = float(line.split("=")[1].strip().split()[0])
                elif "Gamma_target" in line:
                    params["gamma"] = float(line.split("=")[1].strip())
                elif "kappa_target" in line:
                    params["kappa"] = int(float(line.split("=")[1].strip()))
            elif line.startswith("Particles"):
                break
    with open(yaml_path) as f:
        content = f.read()
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("dt:"):
            params["dt"] = float(stripped.split(":")[1].strip().split()[0])
        elif stripped.startswith("prod_dump_step:"):
            params["prod_dump"] = int(stripped.split(":")[1].strip())
        elif stripped.startswith("num:"):
            params["n_particles"] = int(stripped.split(":")[1].strip())
        elif stripped.startswith("number_density:"):
            params["n_density"] = float(stripped.split(":")[1].strip().split()[0])
    return params


def process_case(case_name, sim_root, input_dir, lite=False):
    """Process a single transport case: load dumps, compute VACF, extract D*."""
    yaml_path = input_dir / f"{case_name}.yaml"
    dump_dir = sim_root / case_name / "Simulation" / "Production" / "dumps"

    if not dump_dir.exists():
        return None, f"No dumps at {dump_dir}"

    params = parse_yaml_params(yaml_path)
    dt_dump = params["dt"] * params["prod_dump"]
    a_ws = params["a_ws"]
    omega_p = params["omega_p"]
    n_particles = params["n_particles"]
    n_density = params["n_density"]
    box_side = (n_particles / n_density) ** (1.0 / 3.0)

    print(f"  Loading positions from {dump_dir}...")
    positions, times = load_positions_sorted(dump_dir)
    n_frames = positions.shape[0]
    print(f"  {n_frames} frames, {n_particles} particles, dt_dump = {dt_dump:.4e} s")

    print(f"  Unwrapping PBC (box_side = {box_side:.4e} m)...")
    unwrapped = unwrap_pbc(positions, box_side)

    print(f"  Computing velocities from finite differences...")
    velocities = compute_velocities_fd(unwrapped, dt_dump)

    max_lag = min(n_frames // 2, 2000)
    print(f"  Computing VACF (max_lag = {max_lag})...")
    vacf = compute_vacf(velocities, max_lag)

    c0 = vacf[0]
    c_tail = vacf[-1]
    tail_ratio = c_tail / c0 if abs(c0) > 1e-30 else 0.0

    d_mks = green_kubo_diffusion(vacf, dt_dump)
    d_star = compute_d_star(d_mks, a_ws, omega_p)

    print(f"  C(0) = {c0:.4e}, C(t_max)/C(0) = {tail_ratio:.6f}")
    print(f"  D_mks = {d_mks:.4e} m^2/s")
    print(f"  D* = {d_star:.4e}")

    result = {
        "case": case_name,
        "kappa": params["kappa"],
        "gamma": params["gamma"],
        "D_star_reduced": d_star,
        "D_mks": d_mks,
        "C0": c0,
        "C_tail_ratio": tail_ratio,
        "n_vacf_points": max_lag,
        "n_frames": n_frames,
        "n_particles": n_particles,
        "method": "finite_difference_positions",
    }
    return result, None


def main():
    parser = argparse.ArgumentParser(description="Compute VACF from Sarkas dump files")
    parser.add_argument("--lite", action="store_true", help="Use lite manifest")
    parser.add_argument("--case", type=str, help="Run single case")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    study_dir = script_dir.parent
    input_dir = study_dir / "input_files"
    sim_root = study_dir / "Simulations"
    results_dir = study_dir / "results"
    results_dir.mkdir(exist_ok=True)

    if args.case:
        cases = [args.case]
    elif args.lite:
        manifest = input_dir / "MANIFEST_LITE.txt"
        cases = [line.strip().replace(".yaml", "") for line in manifest.read_text().strip().split("\n")]
    else:
        manifest = input_dir / "MANIFEST.txt"
        cases = [line.strip().replace(".yaml", "") for line in manifest.read_text().strip().split("\n")]

    print("=" * 60)
    print("VACF from Position Finite Differences")
    print("=" * 60)

    results = []
    failed = []

    for case in cases:
        print(f"\n── {case} ──")
        result, err = process_case(case, sim_root, input_dir, args.lite)
        if err:
            print(f"  FAILED: {err}")
            failed.append({"case": case, "error": err})
        else:
            results.append(result)

    output = {
        "provenance": {
            "generator": "hotSpring/control/sarkas/simulations/transport-study/scripts/compute_vacf_from_dumps.py",
            "method": "central_finite_difference_of_unwrapped_positions",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "n_cases": len(results),
            "n_failed": len(failed),
        },
        "results": results,
        "failed": failed,
    }

    suffix = "_lite" if args.lite else ""
    out_path = results_dir / f"transport_baseline_fd{suffix}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\n{'=' * 60}")
    print(f"Output: {out_path}")
    for r in results:
        print(f"  k={r['kappa']} G={r['gamma']:<6} D*={r['D_star_reduced']:.4e}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
