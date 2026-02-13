#!/usr/bin/env python3
"""
Python L1 SEMF benchmark — apples-to-apples comparison with BarraCUDA.

Runs the same SLy4 evaluation and LHS sweep as nuclear_eos_gpu.rs,
using the Python physics code, with full energy monitoring.

Usage:
    python3 bench_python_l1.py

License: AGPL-3.0
"""
import os
import sys
import time
import json
import numpy as np

# Single-threaded BLAS for fair comparison
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WRAPPER_DIR = os.path.join(SCRIPT_DIR, "..", "wrapper")
sys.path.insert(0, WRAPPER_DIR)

from bench_wrapper import (
    BenchPhase, HardwareInventory, save_report, print_summary, PhaseResult,
)
from skyrme_hf import (
    nuclear_matter_properties as skyrme_nuclear_matter,
    semf_binding_energy as binding_energy_semf,
)
from objective import load_bounds, nuclear_eos_chi2

# ═══════════════════════════════════════════════════════════════════
#  Experimental data
# ═══════════════════════════════════════════════════════════════════

EXP_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "exp_data", "ame2020_selected.json")

def load_exp_data():
    with open(EXP_DATA_PATH) as f:
        data = json.load(f)
    nuclei = []
    for entry in data["nuclei"]:
        z = entry["Z"]
        n = entry["N"]
        b_exp = entry["binding_energy_MeV"]
        unc = entry.get("uncertainty_MeV", 0.0)
        nuclei.append(((z, n), (b_exp, unc)))
    nuclei.sort(key=lambda x: (x[0][0], x[0][1]))
    return nuclei

# SLy4 parameter set (same as Rust binary)
SLY4 = [
    -2488.91, 486.82, -546.39, 13777.0,
    0.834, -0.344, -1.0, 1.354, 0.1667, 123.0,
]

# ═══════════════════════════════════════════════════════════════════
#  Main benchmark
# ═══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  hotSpring Python L1 Benchmark                              ║")
    print("║  Same physics as BarraCUDA — measuring Python compute cost  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    hw = HardwareInventory.detect("Eastgate")
    print(f"  CPU: {hw.cpu_model}")
    print(f"  GPU: {hw.gpu_name} (not used by Python)")
    print(f"  Python: {hw.python_version}")
    print()

    nuclei = load_exp_data()
    n_nuclei = len(nuclei)
    print(f"  Nuclei: {n_nuclei} (AME2020 selected)")

    bounds = load_bounds()
    print(f"  Parameters: {len(bounds)} dimensions (Skyrme)")
    print()

    phases = []

    # ── L1 SEMF: SLy4 single-evaluation (100k iterations) ──────────
    print("══════════════════════════════════════════════════════════════")
    print("  L1 SEMF: SLy4 (Python, 100k iterations)")
    print("══════════════════════════════════════════════════════════════")
    print()

    n_iters = 100_000
    with BenchPhase("L1 SEMF", substrate="Python") as bp:
        energies = [0.0] * n_nuclei
        for _ in range(n_iters):
            for i, ((z, n_val), _) in enumerate(nuclei):
                energies[i] = binding_energy_semf(z, n_val, SLY4)

    # Compute chi2
    chi2 = 0.0
    for i, ((z, n_val), (b_exp, _)) in enumerate(nuclei):
        sigma = max(0.01 * b_exp, 2.0)
        chi2 += ((energies[i] - b_exp) / sigma) ** 2
    chi2 /= n_nuclei

    bp.set_physics(chi2=chi2, n_evals=n_iters,
                   notes=f"{n_nuclei} nuclei x {n_iters} iterations")
    result = bp.result()
    phases.append(result)

    print(f"  chi2/datum = {chi2:.4f}")
    print(f"  {result.per_eval_us:.1f} us/eval ({n_nuclei} nuclei, {n_iters} iterations)")
    print(f"  Wall time: {result.wall_time_s:.2f}s")
    print(f"  CPU energy: {result.energy.get('cpu_joules', 0):.1f} J")
    print(f"  CPU power:  {result.energy.get('cpu_joules', 0) / max(result.wall_time_s, 0.001):.0f} W avg")
    print()

    # ── L1 sweep: 512-point LHS ────────────────────────────────────
    print("══════════════════════════════════════════════════════════════")
    print("  L1 Sweep: 512-point LHS (Python)")
    print("══════════════════════════════════════════════════════════════")
    print()

    n_sweep = 512
    np.random.seed(42)
    # Latin hypercube sampling
    samples = []
    for _ in range(n_sweep):
        sample = [np.random.uniform(lo, hi) for lo, hi in bounds]
        samples.append(sample)

    with BenchPhase("L1 sweep", substrate="Python") as bp:
        chi2s = []
        for params in samples:
            # Same logic as l1_chi2_cpu in Rust
            nmp = skyrme_nuclear_matter(params)
            if nmp is None or nmp['rho0_fm3'] < 0.05 or nmp['rho0_fm3'] > 0.30 or nmp['E_A_MeV'] > 0.0:
                chi2s.append(1e10)
                continue
            c2 = 0.0
            count = 0
            for (z, n_val), (b_exp, _) in nuclei:
                b_calc = binding_energy_semf(z, n_val, params)
                if b_calc > 0:
                    sigma = max(0.01 * b_exp, 2.0)
                    c2 += ((b_calc - b_exp) / sigma) ** 2
                    count += 1
            if count > 0:
                chi2s.append(c2 / count)
            else:
                chi2s.append(1e10)

    best_idx = int(np.argmin(chi2s))
    best_chi2 = chi2s[best_idx]

    bp.set_physics(chi2=best_chi2, n_evals=n_sweep,
                   notes=f"{n_sweep}-point LHS sweep")
    result = bp.result()
    phases.append(result)

    print(f"  Best chi2/datum = {best_chi2:.4f} (idx {best_idx})")
    print(f"  {result.per_eval_us:.1f} us/eval")
    print(f"  Wall time: {result.wall_time_s:.3f}s")
    print(f"  CPU energy: {result.energy.get('cpu_joules', 0):.1f} J")
    print()

    # ── Summary ─────────────────────────────────────────────────────
    print_summary(hw, phases)

    # Save report
    results_dir = os.path.join(SCRIPT_DIR, "..", "..", "..", "..",
                                "benchmarks", "nuclear-eos", "results")
    try:
        path = save_report(hw, phases, results_dir)
        print(f"  Benchmark report: {path}")
    except Exception as e:
        print(f"  Warning: {e}")


if __name__ == "__main__":
    main()
