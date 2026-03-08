#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Compare analytical Mermin DSF against MD-derived S(k,ω) from the
Dense Plasma Properties Database (MurilloGroupMSU).

Data source: https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database
Citation: Choi, Dharuman, Murillo, Phys. Rev. E (under review)

The database contains S(q,ω) for Yukawa OCP (N=10000, 80k steps)
at various (κ, Γ) in reduced units (ω/ωₚ, q·a_ws).

This script:
  1. Loads reference MD DSF
  2. Computes analytical DSF from standard & completed Mermin
  3. Compares at matched wavevectors
  4. Outputs validation metrics to JSON
"""

import numpy as np
import json
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HOTSPRING_ROOT = SCRIPT_DIR.parents[2]
DB_DIR = HOTSPRING_ROOT / "data" / "plasma-properties-db" / "Dense-Plasma-Properties-Database" / "database"
DSF_DB_DIR = DB_DIR / "Yukawa_Dynamic_Structure_Factors"

sys.path.insert(0, str(SCRIPT_DIR))
from bgk_dielectric_control import (
    plasma_params, epsilon_mermin, epsilon_completed_mermin, plasma_dispersion_W,
)


def load_reference_dsf(kappa, gamma):
    """Load reference S(q,ω) from the Dense Plasma Properties Database."""
    filename = f"sqw_k{kappa}G{gamma}.npy"
    filepath = DSF_DB_DIR / filename
    if not filepath.exists():
        available = sorted([f.name for f in DSF_DB_DIR.glob("sqw_*.npy")])
        print(f"ERROR: {filename} not found. Available: {available}")
        sys.exit(1)

    data = np.load(filepath)
    w = data[0, 1:]
    q = data[1:, 0]
    sqw = data[1:, 1:]
    return w, q, sqw


def analytical_dsf(k, omegas, nu, params, completed=False):
    """Compute S(k,ω) from the (completed) Mermin dielectric function.

    Returns DSF in same reduced units as MD data (time unit = ωₚ⁻¹).
    """
    T = params["T"]
    n = params["n"]
    omega_p = params["omega_p"]
    k_D = params["k_D"]

    if k_D < 1e-10:
        return np.zeros(len(omegas))

    S = np.zeros(len(omegas))
    for i, omega in enumerate(omegas):
        if abs(omega) < 1e-12:
            continue
        try:
            if completed:
                eps = epsilon_completed_mermin(k, omega, nu, params)
            else:
                eps = epsilon_mermin(k, omega, nu, params)
            if not np.isfinite(eps):
                continue
            inv_eps = 1.0 / eps
            if not np.isfinite(inv_eps):
                continue
            loss = -np.imag(inv_eps)
            S[i] = (k**2 / (np.pi * n * omega)) * T * loss
        except (ZeroDivisionError, RuntimeWarning):
            continue
    return S


def find_peak(w, s, w_min=0.05):
    """Find the first significant local maximum above w_min."""
    mask = w > w_min
    if not np.any(mask):
        return 0.0, 0.0
    w_masked = w[mask]
    s_masked = s[mask]
    if len(s_masked) < 3:
        return 0.0, 0.0
    peak_idx = np.argmax(s_masked)
    return float(w_masked[peak_idx]), float(s_masked[peak_idx])


def compare_at_wavevector(q_val, w_ref, s_ref, params, nu):
    """Compare analytical vs MD DSF at a single wavevector."""
    a = params["a"]
    omega_p = params["omega_p"]

    k = q_val / a
    # Only use positive frequencies
    pos_mask = w_ref > 0.01
    w_pos = w_ref[pos_mask]
    s_ref_pos = s_ref[pos_mask]
    omegas = w_pos * omega_p

    s_std = analytical_dsf(k, omegas, nu, params, completed=False)
    s_cm = analytical_dsf(k, omegas, nu, params, completed=True)

    # Convert analytical DSF to reduced units: S_reduced = S_absolute × ωₚ
    s_std_reduced = s_std * omega_p
    s_cm_reduced = s_cm * omega_p

    s_ref_max = np.max(s_ref_pos) if len(s_ref_pos) > 0 else 0
    if s_ref_max < 1e-10:
        return None

    # Peak detection
    ref_peak_w, ref_peak_s = find_peak(w_pos, s_ref_pos)
    std_peak_w, std_peak_s = find_peak(w_pos, s_std_reduced)
    cm_peak_w, cm_peak_s = find_peak(w_pos, s_cm_reduced)

    # Integrated spectral weight (positive freq only)
    weight_ref = np.trapezoid(s_ref_pos, w_pos) if len(w_pos) > 1 else 0
    weight_std = np.trapezoid(s_std_reduced, w_pos) if len(w_pos) > 1 else 0
    weight_cm = np.trapezoid(s_cm_reduced, w_pos) if len(w_pos) > 1 else 0

    # L2 relative error
    mask = s_ref_pos > 0.01 * s_ref_max
    if np.sum(mask) > 0 and np.sum(s_ref_pos[mask]**2) > 1e-30:
        l2_std = np.sqrt(np.sum((s_std_reduced[mask] - s_ref_pos[mask])**2) /
                         np.sum(s_ref_pos[mask]**2))
        l2_cm = np.sqrt(np.sum((s_cm_reduced[mask] - s_ref_pos[mask])**2) /
                        np.sum(s_ref_pos[mask]**2))
    else:
        l2_std = l2_cm = float('nan')

    # Amplitude ratio (how well does Mermin capture the overall scale)
    amp_ratio_std = weight_std / weight_ref if weight_ref > 1e-30 else 0
    amp_ratio_cm = weight_cm / weight_ref if weight_ref > 1e-30 else 0

    return {
        "q": float(q_val),
        "ref_peak_w": float(ref_peak_w),
        "ref_peak_s": float(ref_peak_s),
        "std_peak_w": float(std_peak_w),
        "std_peak_s": float(std_peak_s),
        "cm_peak_w": float(cm_peak_w),
        "cm_peak_s": float(cm_peak_s),
        "peak_shift_std": float(abs(std_peak_w - ref_peak_w)),
        "peak_shift_cm": float(abs(cm_peak_w - ref_peak_w)),
        "weight_ref": float(weight_ref),
        "weight_std": float(weight_std),
        "weight_cm": float(weight_cm),
        "amp_ratio_std": float(amp_ratio_std),
        "amp_ratio_cm": float(amp_ratio_cm),
        "l2_rel_std": float(l2_std),
        "l2_rel_cm": float(l2_cm),
    }


def main():
    # Available cases from the database
    cases = [
        (1, 14,   "weak screening, near melting"),
        (1, 72,   "moderate coupling"),
        (1, 217,  "strong coupling"),
        (2, 31,   "stronger screening, weak coupling"),
        (0, 10,   "OCP, weak coupling"),
        (0, 50,   "OCP, moderate coupling"),
    ]

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DSF Comparison: Analytical Mermin vs MD (Murillo Group)   ║")
    print("║  Dense Plasma Properties Database                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    all_results = {}
    checks_passed = 0
    checks_total = 0

    for kappa, gamma, label in cases:
        filename = f"sqw_k{kappa}G{gamma}.npy"
        filepath = DSF_DB_DIR / filename
        if not filepath.exists():
            print(f"\n  SKIP: {filename} not found")
            continue

        print(f"\n═══ κ={kappa}, Γ={gamma} ({label}) ═══")

        w_ref, q_ref, sqw_ref = load_reference_dsf(kappa, gamma)
        params = plasma_params(float(gamma), float(kappa))
        nu = 0.1 * params["omega_p"]

        # Compare at selected wavevectors (low-q collective regime)
        q_targets = [0.5, 1.0, 2.0, 3.0, 5.0]
        case_results = []

        for q_target in q_targets:
            qi = np.argmin(np.abs(q_ref - q_target))
            q_val = q_ref[qi]

            result = compare_at_wavevector(q_val, w_ref, sqw_ref[qi, :], params, nu)
            if result is None:
                continue

            case_results.append(result)

            checks_total += 1
            peak_ok = result["peak_shift_std"] < 0.3 or result["peak_shift_cm"] < 0.3
            if peak_ok:
                checks_passed += 1

            peak_marker = "✓" if peak_ok else "·"
            print(f"  q={q_val:.2f}: "
                  f"peak MD={result['ref_peak_w']:.3f} "
                  f"std={result['std_peak_w']:.3f} "
                  f"cm={result['cm_peak_w']:.3f} | "
                  f"amp std={result['amp_ratio_std']:.3f} "
                  f"cm={result['amp_ratio_cm']:.3f} {peak_marker}")

        all_results[f"k{kappa}_G{gamma}"] = {
            "kappa": kappa,
            "gamma": gamma,
            "label": label,
            "comparisons": case_results,
        }

    print(f"\n{'═' * 64}")
    print(f"  {checks_passed}/{checks_total} peak position checks passed")
    print(f"  (tolerance: Δω < 0.3 ωₚ)")

    output = {
        "source": "MurilloGroupMSU/Dense-Plasma-Properties-Database",
        "url": "https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database",
        "method": "Yukawa OCP MD via Sarkas (N=10000, 80k steps)",
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "cases": all_results,
    }

    out_path = SCRIPT_DIR.parent / "results" / "dsf_vs_md_comparison.json"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
