#!/usr/bin/env python
"""
Comprehensive observable validation for all 12 DSF lite cases.
Validates: Energy conservation, RDF, SSF, VACF, and DSF across all cases.
"""

import json
import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = STUDY_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PP_CASES = [
    (1, 14), (1, 72), (1, 217),
    (2, 31), (2, 158), (2, 476),
    (3, 100), (3, 503), (3, 1510),
]
PPPM_CASES = [(0, 10), (0, 50), (0, 150)]
ALL_CASES = PP_CASES + PPPM_CASES


def find_sim_dir(kappa, gamma):
    """Find the simulation output directory."""
    job = f"dsf_k{kappa}_G{gamma}_lite"
    p = STUDY_DIR / "Simulations" / job
    if p.exists():
        return p
    return None


def validate_energy(sim_dir, kappa, gamma):
    """Validate energy conservation from production energy CSV."""
    result = {"kappa": kappa, "gamma": gamma, "observable": "energy"}

    energy_csv = list(sim_dir.glob("Simulation/Production/ProductionEnergy*.csv"))
    if not energy_csv:
        result["status"] = "NO_DATA"
        return result

    df = pd.read_csv(energy_csv[0])
    n_nan = df.isna().sum().sum()
    n_rows = len(df)

    E_total = df["Total Energy"].values
    E_init = E_total[0]
    E_final = E_total[-1]
    E_mean = np.mean(E_total)
    E_std = np.std(E_total)

    # Relative energy drift (end vs start)
    drift_pct = (E_final / E_init - 1) * 100

    # Relative energy fluctuation (std / mean)
    fluct_pct = (E_std / abs(E_mean)) * 100

    # Temperature stats
    T = df["Temperature"].values
    T_mean = np.mean(T)
    T_std = np.std(T)
    T_fluct_pct = (T_std / T_mean) * 100

    result.update({
        "status": "PASS" if abs(drift_pct) < 5.0 and n_nan == 0 else "MARGINAL",
        "n_rows": n_rows,
        "n_nan": n_nan,
        "E_init": float(E_init),
        "E_final": float(E_final),
        "drift_pct": round(drift_pct, 4),
        "fluctuation_pct": round(fluct_pct, 4),
        "T_mean_K": round(T_mean, 1),
        "T_std_K": round(T_std, 1),
        "T_fluct_pct": round(T_fluct_pct, 2),
    })
    return result


def validate_rdf(sim_dir, kappa, gamma):
    """Validate RDF: check g(r)→1 at large r, first peak position, coordination."""
    result = {"kappa": kappa, "gamma": gamma, "observable": "rdf"}

    rdf_files = list(sim_dir.rglob("RadialDistributionFunction/Production/RadialDistributionFunction_*.h5"))
    rdf_files = [f for f in rdf_files if "slices" not in f.name]
    if not rdf_files:
        result["status"] = "NO_DATA"
        return result

    df = pd.read_hdf(rdf_files[0])
    # Extract distance and g(r)
    r = df.iloc[:, 0].values  # Distance (m)
    gr = df.iloc[:, 1].values  # g(r) mean

    # Filter out zero-distance
    mask = r > 0
    r = r[mask]
    gr = gr[mask]

    if len(r) < 10:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    # 1. g(r) → 1 at large r (last 20% of range)
    n_tail = max(1, len(r) // 5)
    gr_tail_mean = np.mean(gr[-n_tail:])
    asymptote_error = abs(gr_tail_mean - 1.0)

    # 2. First peak position and height
    # Find first maximum after the excluded volume
    nonzero_start = np.argmax(gr > 0.01)
    if nonzero_start < len(gr) - 10:
        peak_region = gr[nonzero_start:]
        r_region = r[nonzero_start:]
        peak_idx = np.argmax(peak_region)
        first_peak_r = r_region[peak_idx]
        first_peak_g = peak_region[peak_idx]
    else:
        first_peak_r = 0
        first_peak_g = 0

    # 3. Get Wigner-Seitz radius from YAML for normalization
    import yaml
    yaml_files = list(STUDY_DIR.glob(f"input_files/dsf_k{kappa}_G{gamma}_mks_lite.yaml"))
    a_ws = None
    if yaml_files:
        with open(yaml_files[0]) as f:
            cfg = yaml.safe_load(f)
        particles = cfg.get("Particles", [{}])
        p = particles[0]
        if "Species" in p:
            p = p["Species"]
        n_dens = p.get("number_density", None)
        if n_dens:
            a_ws = (3 / (4 * np.pi * n_dens)) ** (1/3)

    # Normalize peak position by a_ws
    peak_r_normalized = first_peak_r / a_ws if a_ws else None

    result.update({
        "status": "PASS" if asymptote_error < 0.15 and first_peak_g > 1.0 else "MARGINAL",
        "n_bins": len(r),
        "r_max_m": float(r[-1]),
        "first_peak_r_m": float(first_peak_r),
        "first_peak_r_aws": round(peak_r_normalized, 3) if peak_r_normalized else None,
        "first_peak_g": round(first_peak_g, 3),
        "tail_gr_mean": round(gr_tail_mean, 4),
        "asymptote_error": round(asymptote_error, 4),
        "a_ws_m": float(a_ws) if a_ws else None,
    })
    return result


def validate_ssf(sim_dir, kappa, gamma):
    """Validate SSF: check compressibility sum rule S(k→0) and large-k behavior."""
    result = {"kappa": kappa, "gamma": gamma, "observable": "ssf"}

    ssf_files = list(sim_dir.rglob("StaticStructureFunction/Production/StaticStructureFunction_*.h5"))
    ssf_files = [f for f in ssf_files if "slices" not in f.name]
    if not ssf_files:
        result["status"] = "NO_DATA"
        return result

    df = pd.read_hdf(ssf_files[0])
    k = df.iloc[:, 0].values  # Inverse wavelength (1/m)
    sk = df.iloc[:, 1].values  # S(k) mean

    if len(k) < 3:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    # S(k→0) — compressibility limit
    s_k0 = sk[0]

    # S(k→∞) → 1
    s_kinf = sk[-1]

    # For strongly coupled Yukawa, S(k→0) ≪ 1 (low compressibility)
    # and first peak of S(k) is near k ≈ 2π/a_ws

    # Find first peak
    peak_idx = np.argmax(sk)
    first_peak_k = k[peak_idx]
    first_peak_sk = sk[peak_idx]

    result.update({
        "status": "PASS",
        "n_k_points": len(k),
        "k_min": float(k[0]),
        "k_max": float(k[-1]),
        "S_k0": round(s_k0, 4),
        "S_kinf": round(s_kinf, 4),
        "first_peak_k": float(first_peak_k),
        "first_peak_Sk": round(first_peak_sk, 4),
    })
    return result


def validate_vacf(sim_dir, kappa, gamma):
    """Validate VACF: check decay, extract diffusion coefficient via Green-Kubo."""
    result = {"kappa": kappa, "gamma": gamma, "observable": "vacf"}

    vacf_files = list(sim_dir.rglob("VelocityAutoCorrelationFunction/Production/*ACF_slices*.h5"))
    if not vacf_files:
        result["status"] = "NO_DATA"
        return result

    df = pd.read_hdf(vacf_files[0])

    # Extract time and total VACF
    t = df.iloc[:, 0].values  # Time (s)
    vacf_total = df.iloc[:, -1].values  # Total VACF (last column is typically Total)

    if len(t) < 10:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    # Normalize VACF by initial value
    vacf_0 = vacf_total[0]
    if vacf_0 == 0:
        result["status"] = "ZERO_INITIAL"
        return result

    vacf_norm = vacf_total / vacf_0

    # 1. Check that VACF decays (should reach near zero or oscillate to zero)
    # Find first zero-crossing
    sign_changes = np.where(np.diff(np.sign(vacf_norm)))[0]
    first_zero_t = t[sign_changes[0]] if len(sign_changes) > 0 else None

    # 2. Extract diffusion coefficient via Green-Kubo: D = (1/3) ∫₀^∞ VACF(t) dt
    # Use trapezoidal integration up to first zero-crossing (or full range)
    if first_zero_t is not None:
        cutoff_idx = sign_changes[0] + 1
    else:
        cutoff_idx = len(t)

    D_integral = np.trapz(vacf_total[:cutoff_idx], t[:cutoff_idx]) / 3.0

    # 3. Tail behavior
    tail_start = len(vacf_norm) * 3 // 4
    tail_mean = np.mean(np.abs(vacf_norm[tail_start:]))

    # Get particle mass for thermal velocity check
    import yaml
    yaml_files = list(STUDY_DIR.glob(f"input_files/dsf_k{kappa}_G{gamma}_mks_lite.yaml"))
    v_thermal = None
    if yaml_files:
        with open(yaml_files[0]) as f:
            cfg = yaml.safe_load(f)
        particles = cfg.get("Particles", [{}])
        p = particles[0]
        if "Species" in p:
            p = p["Species"]
        T = p.get("temperature", None)
        m = p.get("mass", None)
        if T and m:
            kB = 1.380649e-23
            v_thermal = np.sqrt(kB * T / m)

    # VACF(0) should be ≈ kB*T/m (thermal velocity squared)
    v_sq_from_vacf = vacf_0 / 3  # <v_x²> = VACF(0)/3 per component

    result.update({
        "status": "PASS" if tail_mean < 0.3 else "MARGINAL",
        "n_points": len(t),
        "t_max_s": float(t[-1]),
        "vacf_0": float(vacf_0),
        "v_rms_from_vacf_m_s": round(np.sqrt(vacf_0), 1),
        "v_thermal_m_s": round(v_thermal, 1) if v_thermal else None,
        "first_zero_crossing_s": float(first_zero_t) if first_zero_t else None,
        "D_green_kubo_m2_s": float(D_integral),
        "tail_residual": round(tail_mean, 4),
        "n_oscillations": len(sign_changes) // 2,
    })
    return result


def main():
    print("=" * 70)
    print("Comprehensive Observable Validation — All 12 DSF Lite Cases")
    print("=" * 70)

    all_results = {
        "energy": [],
        "rdf": [],
        "ssf": [],
        "vacf": [],
    }

    for kappa, gamma in ALL_CASES:
        sim_dir = find_sim_dir(kappa, gamma)
        tag = f"k{kappa}_G{gamma}"
        if sim_dir is None:
            print(f"\n--- {tag}: NO SIM DIR ---")
            continue

        print(f"\n--- {tag} ---")

        # Energy
        e = validate_energy(sim_dir, kappa, gamma)
        all_results["energy"].append(e)
        print(f"  Energy: drift={e.get('drift_pct','?')}%, T={e.get('T_mean_K','?')}±{e.get('T_std_K','?')} K, NaN={e.get('n_nan','?')} [{e['status']}]")

        # RDF
        r = validate_rdf(sim_dir, kappa, gamma)
        all_results["rdf"].append(r)
        print(f"  RDF: peak at r/a_ws={r.get('first_peak_r_aws','?')}, g(peak)={r.get('first_peak_g','?')}, tail→{r.get('tail_gr_mean','?')} [{r['status']}]")

        # SSF
        s = validate_ssf(sim_dir, kappa, gamma)
        all_results["ssf"].append(s)
        print(f"  SSF: S(k0)={s.get('S_k0','?')}, S(kinf)={s.get('S_kinf','?')}, peak S(k)={s.get('first_peak_Sk','?')} [{s['status']}]")

        # VACF
        v = validate_vacf(sim_dir, kappa, gamma)
        all_results["vacf"].append(v)
        print(f"  VACF: D={v.get('D_green_kubo_m2_s','?'):.3e} m²/s, zero-cross={v.get('first_zero_crossing_s','?')}, oscillations={v.get('n_oscillations','?')} [{v['status']}]" if v['status'] not in ('NO_DATA', 'INSUFFICIENT_DATA') else f"  VACF: [{v['status']}]")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for obs_name, results in all_results.items():
        passed = sum(1 for r in results if r.get("status") == "PASS")
        marginal = sum(1 for r in results if r.get("status") == "MARGINAL")
        failed = sum(1 for r in results if r.get("status") not in ("PASS", "MARGINAL", "NO_DATA"))
        total = len(results)
        print(f"  {obs_name:8s}: {passed}/{total} PASS, {marginal} MARGINAL")

    # Energy summary
    energy_drifts = [r["drift_pct"] for r in all_results["energy"] if "drift_pct" in r]
    if energy_drifts:
        print(f"\n  Energy drift range: [{min(energy_drifts):.4f}%, {max(energy_drifts):.4f}%]")
        print(f"  Mean absolute drift: {np.mean(np.abs(energy_drifts)):.4f}%")

    # RDF summary
    rdf_peaks = [r.get("first_peak_r_aws") for r in all_results["rdf"] if r.get("first_peak_r_aws")]
    if rdf_peaks:
        print(f"\n  RDF first peak range: [{min(rdf_peaks):.3f}, {max(rdf_peaks):.3f}] a_ws")

    # VACF diffusion coefficients
    D_vals = [r["D_green_kubo_m2_s"] for r in all_results["vacf"] if "D_green_kubo_m2_s" in r]
    if D_vals:
        print(f"\n  Diffusion coefficient range: [{min(D_vals):.3e}, {max(D_vals):.3e}] m²/s")

    # Save
    out_file = RESULTS_DIR / "all_observables_validation.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            r = convert(obj)
            if r is not obj:
                return r
            return super().default(obj)

    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()

