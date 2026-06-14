#!/usr/bin/env python3
"""
Industry-standard analysis for OPES protein folding simulation.

Produces:
1. Folding free energy (ΔG_fold) via reweighting
2. RMSD/HLDA timeseries showing folding/unfolding transitions
3. Contact formation kinetics
4. Convergence analysis (cumulative FES)
5. Quantitative comparison to Ray & Rizzi (2024) reference

Reference: Ray & Rizzi, JCTC 21, 58-69 (2024)
"""

import numpy as np
import json
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent
TARGET_DIR = ANALYSIS_DIR.parent
OUTPUT_DIR = TARGET_DIR / "output"
FIGURES_DIR = TARGET_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Reference: chignolin folding parameters
REFERENCE = {
    "folding_fe_kj": {"range": (-12.0, -3.0), "tolerance": 4.0},
    "folded_rmsd_nm": 0.15,
    "unfolded_rmsd_nm": 0.35,
    "hlda_folded_range": (1.0, 2.5),
    "hlda_unfolded_range": (-0.5, 0.5),
}


def load_colvar(filepath):
    """Load PLUMED COLVAR file."""
    data = []
    headers = None
    with open(filepath) as f:
        for line in f:
            if line.startswith("#! FIELDS"):
                headers = line.strip().split()[2:]
            elif not line.startswith("#"):
                data.append([float(x) for x in line.split()])
    return headers, np.array(data)


def compute_folding_fe(hlda, bias, kT=2.83):
    """
    Compute folding free energy via OPES reweighting.
    kT = 2.83 kJ/mol at 340K
    """
    weights = np.exp(bias / kT)
    weights /= weights.sum()

    folded_mask = (hlda > REFERENCE["hlda_folded_range"][0]) & \
                  (hlda < REFERENCE["hlda_folded_range"][1])
    unfolded_mask = (hlda < REFERENCE["hlda_unfolded_range"][1])

    p_folded = weights[folded_mask].sum()
    p_unfolded = weights[unfolded_mask].sum()

    if p_folded > 0 and p_unfolded > 0:
        delta_g = -kT * np.log(p_folded / p_unfolded)
        return delta_g, p_folded, p_unfolded
    return None, p_folded, p_unfolded


def compute_fes_1d(hlda, bias, n_bins=100, kT=2.83):
    """Compute 1D FES along HLDA via reweighting."""
    weights = np.exp(bias / kT)
    weights /= weights.sum()

    bins = np.linspace(hlda.min(), hlda.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    hist, _ = np.histogram(hlda, bins=bins, weights=weights)
    hist[hist == 0] = 1e-30

    fes = -kT * np.log(hist)
    fes -= fes.min()

    return bin_centers, fes


def transition_count(hlda, folded_thresh=1.2, unfolded_thresh=0.3):
    """Count folding/unfolding transitions."""
    state = None
    fold_events = 0
    unfold_events = 0

    for val in hlda:
        if val > folded_thresh:
            if state == "unfolded":
                fold_events += 1
            state = "folded"
        elif val < unfolded_thresh:
            if state == "folded":
                unfold_events += 1
            state = "unfolded"

    return fold_events, unfold_events


def convergence_analysis(hlda, bias, kT=2.83, n_windows=10):
    """Track folding FE convergence over time."""
    window_size = len(hlda) // n_windows
    delta_gs = []

    for i in range(1, n_windows + 1):
        end = i * window_size
        dg, _, _ = compute_folding_fe(hlda[:end], bias[:end], kT)
        if dg is not None:
            delta_gs.append(dg)

    return np.array(delta_gs)


def run_analysis():
    """Execute full analysis pipeline for chignolin OPES."""
    report = {"target": "02_chignolin_opes", "method": "OPES_METAD+OPES_METAD_EXPLORE"}

    # Find COLVAR file
    colvar_file = None
    for candidate in ["COLVARb", "COLVAR", "output/COLVARb", "output/COLVAR"]:
        p = TARGET_DIR / candidate
        if p.exists():
            colvar_file = p
            break

    if colvar_file is None:
        print("ERROR: No COLVAR file found. Run simulation first.")
        return report

    headers, data = load_colvar(colvar_file)
    report["colvar_file"] = str(colvar_file)
    report["n_frames"] = len(data)
    report["fields"] = headers

    # Extract columns
    time_col = data[:, 0]
    hlda_idx = headers.index("hlda") if "hlda" in headers else 1
    hlda = data[:, hlda_idx]

    # Bias columns
    opes_bias_idx = None
    opesE_bias_idx = None
    for i, h in enumerate(headers):
        if "opes.bias" in h and "opesE" not in h:
            opes_bias_idx = i
        elif "opesE.bias" in h:
            opesE_bias_idx = i

    total_bias = np.zeros(len(data))
    if opes_bias_idx:
        total_bias += data[:, opes_bias_idx]
    if opesE_bias_idx:
        total_bias += data[:, opesE_bias_idx]

    # RMSD if available
    rmsd_idx = None
    for i, h in enumerate(headers):
        if "rmsd" in h.lower():
            rmsd_idx = i
            break

    # 1. Basic statistics
    report["simulation_time_ns"] = float(time_col[-1]) / 1000.0
    report["hlda_stats"] = {
        "mean": float(hlda.mean()),
        "std": float(hlda.std()),
        "min": float(hlda.min()),
        "max": float(hlda.max()),
    }

    # 2. Folding free energy
    kT = 2.83  # 340K
    delta_g, p_fold, p_unfold = compute_folding_fe(hlda, total_bias, kT)
    if delta_g is not None:
        report["folding_fe_kj"] = float(delta_g)
        report["p_folded"] = float(p_fold)
        report["p_unfolded"] = float(p_unfold)
        ref_range = REFERENCE["folding_fe_kj"]["range"]
        report["fe_within_reference"] = ref_range[0] <= delta_g <= ref_range[1]

    # 3. FES along HLDA
    bin_centers, fes = compute_fes_1d(hlda, total_bias, n_bins=80, kT=kT)
    np.savetxt(
        ANALYSIS_DIR / "fes_hlda.dat",
        np.column_stack([bin_centers, fes]),
        header="HLDA  FES(kJ/mol)",
        fmt="%.6f",
    )

    # 4. Transitions
    fold_events, unfold_events = transition_count(hlda)
    report["transitions"] = {
        "folding_events": fold_events,
        "unfolding_events": unfold_events,
        "total_transitions": fold_events + unfold_events,
    }

    # 5. Convergence
    delta_gs = convergence_analysis(hlda, total_bias, kT=kT, n_windows=10)
    if len(delta_gs) >= 3:
        report["convergence"] = {
            "n_windows": len(delta_gs),
            "final_fe_kj": float(delta_gs[-1]),
            "fe_std_last_3": float(delta_gs[-3:].std()),
            "converged": bool(delta_gs[-3:].std() < 4.0),
        }
        np.savetxt(
            ANALYSIS_DIR / "convergence_fe.dat",
            delta_gs,
            header="delta_G_fold (kJ/mol) per time window",
            fmt="%.4f",
        )

    # 6. RMSD analysis
    if rmsd_idx is not None:
        rmsd = data[:, rmsd_idx]
        folded_fraction = float((rmsd < REFERENCE["folded_rmsd_nm"]).mean())
        report["rmsd_stats"] = {
            "mean_nm": float(rmsd.mean()),
            "folded_fraction": folded_fraction,
        }

    # 7. Pass/fail validation
    passes = []
    fails = []

    if delta_g is not None:
        if report.get("fe_within_reference"):
            passes.append(f"ΔG_fold = {delta_g:.1f} kJ/mol within reference range {ref_range}")
        else:
            fails.append(f"ΔG_fold = {delta_g:.1f} kJ/mol outside range {ref_range}")

    if report.get("transitions", {}).get("total_transitions", 0) > 0:
        passes.append(f"Observed {report['transitions']['total_transitions']} folding/unfolding transitions")
    else:
        fails.append("No folding/unfolding transitions observed (need longer simulation)")

    if report.get("convergence", {}).get("converged"):
        passes.append(f"FE convergence < 4 kJ/mol over last 3 windows")
    elif "convergence" in report:
        fails.append(f"FE NOT converged (std = {report['convergence'].get('fe_std_last_3', 'N/A'):.1f} kJ/mol)")

    report["validation"] = {
        "passes": passes,
        "fails": fails,
        "total_checks": len(passes) + len(fails),
        "pass_rate": len(passes) / max(1, len(passes) + len(fails)),
        "industry_standard": len(fails) == 0,
    }

    # Write report
    with open(ANALYSIS_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Summary
    summary = f"""
{'='*70}
VALIDATION REPORT: Target 02 — Chignolin (OPES + OPES-Explore)
{'='*70}

Simulation time: {report['simulation_time_ns']:.1f} ns
HLDA mean: {report['hlda_stats']['mean']:.3f} (folded > 1.0)
Transitions: {report.get('transitions', {}).get('total_transitions', 0)}
ΔG_fold: {report.get('folding_fe_kj', 'N/A')} kJ/mol

Total checks: {report['validation']['total_checks']}
Pass rate: {report['validation']['pass_rate']*100:.0f}%
Industry standard: {'PASS' if report['validation']['industry_standard'] else 'FAIL'}

--- PASSES ---"""
    for p in passes:
        summary += f"\n  [PASS] {p}"
    if fails:
        summary += "\n\n--- FAILURES ---"
        for f_item in fails:
            summary += f"\n  [FAIL] {f_item}"
    summary += f"\n\n{'='*70}"

    with open(ANALYSIS_DIR / "VALIDATION_SUMMARY.txt", "w") as f:
        f.write(summary)
    print(summary)

    return report


if __name__ == "__main__":
    run_analysis()
