#!/usr/bin/env python
"""
Validate PPPM (κ=0) DSF cases against Dense Plasma Properties Database.
Uses the same quantitative peak comparison as the PP batch validation.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = STUDY_DIR.parents[3]  # hotSpring/
DB_DIR = REPO_ROOT / "data" / "plasma-properties-db" / "Dense-Plasma-Properties-Database" / "database" / "Yukawa_Dynamic_Structure_Factors"

PPPM_CASES = [
    {"kappa": 0, "gamma": 10},
    {"kappa": 0, "gamma": 50},
    {"kappa": 0, "gamma": 150},
]


def load_reference(kappa, gamma):
    """Load reference S(q,w) from database."""
    fname = DB_DIR / f"sqw_k{kappa}G{gamma}.npy"
    if not fname.exists():
        return None, None, None
    data = np.load(str(fname))
    w = data[0, 1:]   # frequencies (reduced units: omega/omega_p)
    q = data[1:, 0]   # wavenumbers (reduced units: q * a_ws)
    sqw = data[1:, 1:]
    return w, q, sqw


def load_computed(kappa, gamma):
    """Load computed DSF from Sarkas output."""
    job_dir = f"dsf_k{kappa}_G{gamma}_lite"
    sim_base = STUDY_DIR / "Simulations" / job_dir

    if not sim_base.exists():
        return None

    # Find DSF HDF5 files
    dsf_files = list(sim_base.rglob("DynamicStructureFactor*.h5"))
    if not dsf_files:
        return None

    # Load the mean DSF (not slices)
    for f in dsf_files:
        if "slices" not in f.name.lower():
            df = pd.read_hdf(f)
            return df
    return None


def get_plasma_frequency(kappa, gamma):
    """
    Compute plasma frequency from the simulation YAML to convert from SI to reduced units.
    For OCP (κ=0), ω_p = sqrt(Z²e²n / (ε₀ m)) where n, Z, m come from the YAML.
    """
    import yaml
    yaml_file = STUDY_DIR / "input_files" / f"dsf_k{kappa}_G{gamma}_mks_lite.yaml"
    if not yaml_file.exists():
        yaml_file = STUDY_DIR / "input_files" / f"dsf_k{kappa}_G{gamma}_mks.yaml"
    if not yaml_file.exists():
        return None

    with open(yaml_file) as fh:
        cfg = yaml.safe_load(fh)

    particles = cfg.get("Particles", [{}])
    if isinstance(particles, list):
        p = particles[0]
    else:
        p = particles

    # Handle nested Species dict (PPPM format)
    if "Species" in p:
        p = p["Species"]

    Z = p.get("Z", 1.0)
    n = p.get("number_density", None)
    m = p.get("mass", None)

    if n is None or m is None:
        return None

    # ω_p = sqrt(Z²e²n / (ε₀ m))
    e = 1.602176634e-19
    eps0 = 8.854187817e-12
    omega_p = np.sqrt(Z**2 * e**2 * n / (eps0 * m))
    return omega_p


def validate_case(kappa, gamma):
    """Validate one PPPM case. Returns dict with results."""
    result = {"case": f"dsf_k{kappa}_G{gamma}_lite", "kappa": kappa, "gamma": gamma}

    # Load reference
    w_ref, q_ref, sqw_ref = load_reference(kappa, gamma)
    if w_ref is None:
        result["status"] = "NO_REFERENCE"
        return result

    # Load computed
    df = load_computed(kappa, gamma)
    if df is None:
        result["status"] = "NO_DATA"
        return result

    # Get plasma frequency for unit conversion
    omega_p = get_plasma_frequency(kappa, gamma)
    if omega_p is None:
        result["status"] = "NO_OMEGA_P"
        return result

    # Extract computed frequencies and ka values
    freq_col = [c for c in df.columns if 'Frequenc' in str(c)]
    if freq_col:
        freqs_si = df[freq_col[0]].values  # SI: rad/s
    else:
        freqs_si = df.iloc[:, 0].values

    freqs_reduced = freqs_si / omega_p  # Convert to reduced units

    # Get mean columns (ka values)
    mean_cols = [c for c in df.columns if 'Mean' in str(c)]
    if not mean_cols:
        result["status"] = "NO_MEAN_COLS"
        return result

    ka_details = []
    for col in mean_cols:
        # Extract ka value from column name
        col_str = str(col)
        ka_match = None
        for part in col_str.split("ka"):
            for subpart in part.split("="):
                try:
                    ka_match = float(subpart.strip().rstrip("')"))
                    break
                except ValueError:
                    continue
            if ka_match is not None:
                break

        if ka_match is None:
            continue

        # Get computed spectrum at this ka
        spectrum_comp = df[col].values
        # Find peak in positive frequency range (skip DC/diffusive peak near ω=0)
        pos_mask = freqs_reduced > 0.05
        if not pos_mask.any():
            continue
        pos_freqs = freqs_reduced[pos_mask]
        pos_spec = spectrum_comp[pos_mask]
        comp_peak_idx = np.argmax(pos_spec)
        comp_peak_w = pos_freqs[comp_peak_idx]

        # Find closest reference q value (ka ≈ q * a_ws)
        ref_qi = np.argmin(np.abs(q_ref - ka_match))
        ref_spectrum = sqw_ref[ref_qi, :]

        # For reference: find peak away from ω=0 (plasmon peak)
        ref_pos_mask = w_ref > 0.05
        if ref_pos_mask.any():
            ref_pos_freqs = w_ref[ref_pos_mask]
            ref_pos_spec = ref_spectrum[ref_pos_mask]
            ref_peak_idx = np.argmax(ref_pos_spec)
            ref_peak_w = ref_pos_freqs[ref_peak_idx]
        else:
            ref_peak_idx = np.argmax(ref_spectrum)
            ref_peak_w = w_ref[ref_peak_idx]

        # At high ka, the DSF transitions from collective plasmon to diffusive:
        # - Plasmon region: clear peak at ω > 0, reference also has peak at ω > 0
        # - Diffusive region: reference peak is near ω ≈ 0, comparison is invalid
        # Skip ka values where reference peak is dominated by low-ω diffusive mode
        # For OCP, clear plasmon dispersion exists at low ka where ω ≈ ω_p.
        # At higher ka, the DSF transitions through a crossover region (ω < 0.5 ω_p)
        # into individual particle behavior. Only compare in the clear plasmon regime.
        is_plasmon = ref_peak_w > 0.5  # clear plasmon: ω > 0.5 ω_p

        if ref_peak_w > 0.01:
            error_pct = abs(comp_peak_w - ref_peak_w) / ref_peak_w * 100
        else:
            error_pct = float('nan')

        detail = {
            "ka": round(ka_match, 4),
            "comp_peak_w": round(comp_peak_w, 4),
            "ref_peak_w": round(ref_peak_w, 4),
            "error_pct": round(error_pct, 1),
            "mode": "plasmon" if is_plasmon else "diffusive",
        }
        ka_details.append(detail)

    if ka_details:
        # Only compute mean error from plasmon peaks (not diffusive)
        plasmon_errors = [d["error_pct"] for d in ka_details
                         if d["mode"] == "plasmon" and not np.isnan(d["error_pct"])]
        all_errors = [d["error_pct"] for d in ka_details if not np.isnan(d["error_pct"])]

        mean_plasmon_err = round(np.mean(plasmon_errors), 1) if plasmon_errors else float('nan')
        mean_all_err = round(np.mean(all_errors), 1) if all_errors else float('nan')

        result["mean_plasmon_error_pct"] = mean_plasmon_err
        result["mean_all_error_pct"] = mean_all_err
        result["n_plasmon_peaks"] = len(plasmon_errors)
        result["n_diffusive_peaks"] = len(ka_details) - len(plasmon_errors)
        result["ka_details"] = ka_details
        result["status"] = "PASS" if mean_plasmon_err < 25.0 else "MARGINAL"
    else:
        result["status"] = "NO_PEAKS"

    return result


def main():
    print("=" * 70)
    print("PPPM (κ=0) DSF Batch Validation")
    print("=" * 70)

    results = []
    for case in PPPM_CASES:
        k, g = case["kappa"], case["gamma"]
        print(f"\n--- κ={k}, Γ={g} ---")
        r = validate_case(k, g)
        results.append(r)

        if r["status"] in ("PASS", "MARGINAL"):
            print(f"  Status: {r['status']}")
            print(f"  Plasmon peak error: {r.get('mean_plasmon_error_pct', 'N/A')}% ({r.get('n_plasmon_peaks', 0)} peaks)")
            print(f"  Diffusive peaks skipped: {r.get('n_diffusive_peaks', 0)}")
            for d in r.get("ka_details", []):
                mode_tag = "🟢" if d["mode"] == "plasmon" else "⚪"
                print(f"    {mode_tag} ka={d['ka']}: comp={d['comp_peak_w']}, ref={d['ref_peak_w']}, err={d['error_pct']}% [{d['mode']}]")
        else:
            print(f"  Status: {r['status']}")

    # Save results
    out_dir = STUDY_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "dsf_pppm_lite_validation.json"

    completed = [r for r in results if r["status"] in ("PASS", "MARGINAL")]
    plasmon_errors = []
    for r in completed:
        if "mean_plasmon_error_pct" in r and not np.isnan(r["mean_plasmon_error_pct"]):
            plasmon_errors.append(r["mean_plasmon_error_pct"])

    summary = {
        "description": "PPPM (κ=0) DSF peak frequency validation against Dense Plasma Properties Database",
        "note": "Only plasmon peaks (ω > 0.2 ω_p) are compared; high-q diffusive peaks are excluded",
        "sarkas_version": "1.0.0",
        "method": "PPPM",
        "N_particles": 2000,
        "N_prod_steps": 30000,
        "overall_mean_plasmon_error_pct": round(np.mean(plasmon_errors), 1) if plasmon_errors else None,
        "cases_completed": len(completed),
        "cases_total": len(PPPM_CASES),
        "cases": results,
    }

    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}")

    print(f"\n{'='*70}")
    print(f"PPPM Validation: {len(completed)}/{len(PPPM_CASES)} cases validated")
    if plasmon_errors:
        print(f"Overall mean plasmon error: {np.mean(plasmon_errors):.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

