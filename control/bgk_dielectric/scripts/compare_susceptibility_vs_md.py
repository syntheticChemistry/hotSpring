#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Compare analytical Mermin susceptibility χ(k,ω) against MD-derived χ(q,ω)
from the Dense Plasma Properties Database.

The susceptibility is a more direct comparison than the DSF since it removes
the thermal prefactor and directly probes the dielectric response.

Data source: https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database
"""

import numpy as np
import json
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HOTSPRING_ROOT = SCRIPT_DIR.parents[2]
DB_DIR = HOTSPRING_ROOT / "data" / "plasma-properties-db" / "Dense-Plasma-Properties-Database" / "database"
CHI_DB_DIR = DB_DIR / "Yukawa_Susceptibilities"

sys.path.insert(0, str(SCRIPT_DIR))
from bgk_dielectric_control import (
    plasma_params, epsilon_mermin, epsilon_completed_mermin,
)


def load_reference_chi(kappa, gamma):
    """Load reference χ(q,ω) from the database."""
    filename = f"chi_qw_k{kappa}G{gamma}.npy"
    filepath = CHI_DB_DIR / filename
    if not filepath.exists():
        available = sorted([f.name for f in CHI_DB_DIR.glob("chi_*.npy")])
        print(f"  SKIP: {filename} not found (available: {available[:3]}...)")
        return None, None, None

    data = np.load(filepath)
    w = data[0, 1:]
    q = data[1:, 0]
    chi = data[1:, 1:]
    return w, q, chi


def analytical_susceptibility(k, omega, nu, params, completed=False):
    """Compute χ(k,ω) from Mermin dielectric: χ = (ε - 1) / V(k)."""
    if abs(omega) < 1e-15:
        return 0.0 + 0j
    try:
        if completed:
            eps = epsilon_completed_mermin(k, omega, nu, params)
        else:
            eps = epsilon_mermin(k, omega, nu, params)
        if not np.isfinite(eps):
            return 0.0 + 0j
        v_k = 4 * np.pi / (k**2 + params["k_D"]**2) if params["k_D"] > 0 else 4 * np.pi / k**2
        chi = (eps - 1.0) / v_k
        return chi
    except (ZeroDivisionError, RuntimeWarning):
        return 0.0 + 0j


def compare_case(kappa, gamma, label):
    """Compare susceptibility for one (κ, Γ) case."""
    result = load_reference_chi(kappa, gamma)
    if result[0] is None:
        return None

    w_ref, q_ref, chi_ref = result
    params = plasma_params(float(gamma), float(kappa))
    nu = 0.1 * params["omega_p"]
    a = params["a"]
    omega_p = params["omega_p"]

    if params["k_D"] < 1e-10:
        print(f"  κ=0: Mermin susceptibility requires screening, skipping")
        return None

    print(f"\n═══ κ={kappa}, Γ={gamma} ({label}) ═══")

    q_targets = [0.5, 1.0, 2.0, 3.0]
    comparisons = []

    for q_target in q_targets:
        qi = np.argmin(np.abs(q_ref - q_target))
        q_val = q_ref[qi]
        k = q_val / a
        chi_ref_slice = chi_ref[qi, :]

        pos_mask = w_ref > 0.05
        w_pos = w_ref[pos_mask]
        chi_ref_pos = chi_ref_slice[pos_mask]

        # Compute analytical susceptibility (imaginary part is what matters)
        chi_std_im = np.zeros(len(w_pos))
        chi_cm_im = np.zeros(len(w_pos))
        for i, w in enumerate(w_pos):
            omega = w * omega_p
            chi_s = analytical_susceptibility(k, omega, nu, params, completed=False)
            chi_c = analytical_susceptibility(k, omega, nu, params, completed=True)
            chi_std_im[i] = np.imag(chi_s) * omega_p
            chi_cm_im[i] = np.imag(chi_c) * omega_p

        # Compare imaginary part magnitudes
        ref_max = np.max(np.abs(chi_ref_pos))
        if ref_max < 1e-12:
            continue

        std_max = np.max(np.abs(chi_std_im))
        cm_max = np.max(np.abs(chi_cm_im))

        # Relative amplitude
        amp_ratio_std = std_max / ref_max if ref_max > 0 else 0
        amp_ratio_cm = cm_max / ref_max if ref_max > 0 else 0

        # Peak position of imaginary part
        ref_peak_idx = np.argmax(np.abs(chi_ref_pos))
        std_peak_idx = np.argmax(np.abs(chi_std_im))
        cm_peak_idx = np.argmax(np.abs(chi_cm_im))

        ref_peak_w = w_pos[ref_peak_idx]
        std_peak_w = w_pos[std_peak_idx]
        cm_peak_w = w_pos[cm_peak_idx]

        delta_std = abs(std_peak_w - ref_peak_w)
        delta_cm = abs(cm_peak_w - ref_peak_w)
        best = min(delta_std, delta_cm)

        marker = "✓" if best < 0.3 else "·"
        print(f"  q={q_val:.2f}: "
              f"MD peak={ref_peak_w:.3f} "
              f"std={std_peak_w:.3f} cm={cm_peak_w:.3f} | "
              f"amp std={amp_ratio_std:.3f} cm={amp_ratio_cm:.3f} {marker}")

        comparisons.append({
            "q": float(q_val),
            "ref_peak_w": float(ref_peak_w),
            "std_peak_w": float(std_peak_w),
            "cm_peak_w": float(cm_peak_w),
            "delta_best": float(best),
            "amp_ratio_std": float(amp_ratio_std),
            "amp_ratio_cm": float(amp_ratio_cm),
        })

    return {
        "kappa": kappa,
        "gamma": gamma,
        "label": label,
        "comparisons": comparisons,
    }


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Susceptibility χ(k,ω): Analytical Mermin vs MD           ║")
    print("║  Dense Plasma Properties Database (MurilloGroupMSU)        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    cases = [
        (2, 31, "strong screening, moderate coupling"),
        (1, 14, "weak screening, near melting"),
        (1, 72, "moderate coupling"),
        (2, 158, "strong screening, strong coupling"),
        (3, 100, "very strong screening"),
    ]

    all_results = {}
    for kappa, gamma, label in cases:
        result = compare_case(kappa, gamma, label)
        if result is not None:
            all_results[f"k{kappa}_G{gamma}"] = result

    output = {
        "source": "MurilloGroupMSU/Dense-Plasma-Properties-Database",
        "observable": "susceptibility chi(q,omega)",
        "cases": all_results,
    }

    out_path = SCRIPT_DIR.parent / "results" / "susceptibility_vs_md_comparison.json"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
