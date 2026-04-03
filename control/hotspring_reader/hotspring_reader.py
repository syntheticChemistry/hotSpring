#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
hotspring_reader — load hotSpring JSON output into NumPy arrays.

Zero-dependency beyond NumPy. Reads ensemble manifests, per-config
measurement JSONs, and chuna_analyze output. Exports TSV for papers.

Usage:
    import hotspring_reader as hs

    # Load an analysis summary
    ana = hs.load_analysis("results/analysis.json")
    print(ana["plaquette"]["mean"], "±", ana["plaquette"]["error"])

    # Load all per-config measurements from a directory
    meas = hs.load_measurements("data/b6.0_L8/measurements/")
    print(meas["plaquette"])       # NumPy array of floats
    print(meas["flow_t0"])         # NumPy array (NaN where absent)
    print(meas["correlator"])      # 2D array [n_configs, n_t] or None

    # Load an ensemble manifest
    ens = hs.load_ensemble("data/b6.0_L8/ensemble.json")
    print(ens["beta"], ens["dims"], len(ens["configs"]))

    # Export analysis to TSV for paper tables
    hs.to_tsv(ana, "results/analysis.tsv")

    # Export measurement timeseries to TSV
    hs.timeseries_to_tsv(meas, "results/timeseries.tsv")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


# ── Ensemble manifest (ensemble.json from chuna_generate) ──────────────

def load_ensemble(path: Union[str, Path]) -> Dict[str, Any]:
    """Load an ensemble manifest JSON into a dict with typed fields.

    Returns dict with keys: schema_version, ensemble_id, created,
    gauge_action, fermion_action, beta, mass, nf, dims (np.array of 4 ints),
    provenance (dict), algorithm (dict), configs (list of dicts with
    trajectory, filename, plaquette, checksum_*), and optionally run (dict).
    """
    with open(path) as f:
        raw = json.load(f)

    result = dict(raw)
    result["dims"] = np.array(raw.get("dims", [0, 0, 0, 0]), dtype=np.int64)

    configs = raw.get("configs", [])
    result["configs"] = configs
    result["config_plaquettes"] = np.array(
        [c.get("plaquette", np.nan) for c in configs], dtype=np.float64
    )
    result["config_trajectories"] = np.array(
        [c.get("trajectory", 0) for c in configs], dtype=np.int64
    )
    return result


# ── Per-config measurements (meas_*.json from chuna_measure) ──────────

def load_measurements(dirpath: Union[str, Path]) -> Dict[str, Any]:
    """Load all measurement JSONs from a directory into NumPy arrays.

    Returns dict with keys:
        trajectory:     int array [N]
        plaquette:      float array [N]
        polyakov_abs:   float array [N]
        polyakov_re:    float array [N]
        polyakov_im:    float array [N]
        action_density: float array [N]
        flow_t0:        float array [N] (NaN where absent)
        flow_w0:        float array [N] (NaN where absent)
        topo_charge:    float array [N] (NaN where absent)
        condensate:     float array [N] (NaN where absent)
        wilson_loops:   list of dicts [N] or None
        correlator:     float array [N, Nt] or None (from HVP)
        hvp_integral:   float array [N] (NaN where absent)
        cg_iterations:  int array [N] (-1 where absent, from HVP)
        cg_residual:    float array [N] (NaN where absent, from HVP)
        wall_seconds:   float array [N]
        raw:            list of original dicts
        n_configs:      int
    """
    dirpath = Path(dirpath)
    files = sorted(dirpath.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {dirpath}")

    measurements: List[Dict] = []
    for fp in files:
        try:
            with open(fp) as f:
                m = json.load(f)
            if "gauge" not in m:
                continue
            measurements.append(m)
        except (json.JSONDecodeError, KeyError):
            continue

    if not measurements:
        raise ValueError(f"No valid measurement JSONs found in {dirpath}")

    measurements.sort(key=lambda m: m.get("trajectory", 0))

    n = len(measurements)
    result: Dict[str, Any] = {
        "n_configs": n,
        "raw": measurements,
    }

    result["trajectory"] = np.array(
        [m.get("trajectory", i) for i, m in enumerate(measurements)], dtype=np.int64
    )
    result["plaquette"] = np.array(
        [m["gauge"]["plaquette"] for m in measurements], dtype=np.float64
    )
    result["polyakov_abs"] = np.array(
        [m["gauge"]["polyakov_abs"] for m in measurements], dtype=np.float64
    )
    result["polyakov_re"] = np.array(
        [m["gauge"]["polyakov_re"] for m in measurements], dtype=np.float64
    )
    result["polyakov_im"] = np.array(
        [m["gauge"]["polyakov_im"] for m in measurements], dtype=np.float64
    )
    result["action_density"] = np.array(
        [m["gauge"]["action_density"] for m in measurements], dtype=np.float64
    )

    result["flow_t0"] = np.array(
        [_get_nested(m, "flow", "t0") for m in measurements], dtype=np.float64
    )
    result["flow_w0"] = np.array(
        [_get_nested(m, "flow", "w0") for m in measurements], dtype=np.float64
    )
    result["topo_charge"] = np.array(
        [_get_nested(m, "topology", "charge") for m in measurements], dtype=np.float64
    )
    result["condensate"] = np.array(
        [_get_nested(m, "fermion", "chiral_condensate") for m in measurements],
        dtype=np.float64,
    )

    # HVP correlator data
    hvp_present = [m.get("hvp") is not None for m in measurements]
    if any(hvp_present):
        result["hvp_integral"] = np.array(
            [_get_nested(m, "hvp", "hvp_integral") for m in measurements],
            dtype=np.float64,
        )
        result["cg_iterations"] = np.array(
            [
                m["hvp"]["cg_iterations"] if m.get("hvp") else -1
                for m in measurements
            ],
            dtype=np.int64,
        )
        result["cg_residual"] = np.array(
            [_get_nested(m, "hvp", "cg_residual") for m in measurements],
            dtype=np.float64,
        )
        corr_lengths = [
            len(m["hvp"]["correlator"])
            for m in measurements
            if m.get("hvp") and m["hvp"].get("correlator")
        ]
        if corr_lengths:
            nt = max(corr_lengths)
            corr = np.full((n, nt), np.nan, dtype=np.float64)
            for i, m in enumerate(measurements):
                if m.get("hvp") and m["hvp"].get("correlator"):
                    c = m["hvp"]["correlator"]
                    corr[i, : len(c)] = c
            result["correlator"] = corr
        else:
            result["correlator"] = None
    else:
        result["hvp_integral"] = np.full(n, np.nan, dtype=np.float64)
        result["cg_iterations"] = np.full(n, -1, dtype=np.int64)
        result["cg_residual"] = np.full(n, np.nan, dtype=np.float64)
        result["correlator"] = None

    result["wall_seconds"] = np.array(
        [m.get("wall_seconds", np.nan) for m in measurements], dtype=np.float64
    )

    # Wilson loops: keep as list (ragged, varies by config)
    wl = [m.get("wilson_loops") for m in measurements]
    result["wilson_loops"] = wl if any(w is not None for w in wl) else None

    return result


# ── Analysis output (from chuna_analyze) ───────────────────────────────

def load_analysis(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a chuna_analyze JSON output.

    Returns dict preserving all fields. ObservableStats sub-dicts
    have keys: mean, error, tau_int, tau_int_error, n_values.
    """
    with open(path) as f:
        return json.load(f)


# ── TSV export ─────────────────────────────────────────────────────────

_ANALYSIS_OBSERVABLES = [
    "plaquette",
    "polyakov_abs",
    "topological_charge",
    "t0",
    "w0",
    "condensate",
]

_ANALYSIS_SCALARS = [
    "topo_susceptibility",
    "plaquette_susceptibility",
    "polyakov_susceptibility",
]


def to_tsv(analysis: Dict[str, Any], path: Union[str, Path]) -> None:
    """Export analysis dict to a TSV file suitable for numpy.loadtxt().

    Columns: observable, mean, error, tau_int, tau_int_error, n_values
    Susceptibilities are appended with error=0 and tau_int=0.
    """
    lines = ["observable\tmean\terror\ttau_int\ttau_int_error\tn_values"]

    for obs in _ANALYSIS_OBSERVABLES:
        if obs in analysis and analysis[obs] is not None:
            d = analysis[obs]
            lines.append(
                f"{obs}\t{d['mean']:.12e}\t{d['error']:.6e}"
                f"\t{d['tau_int']:.4f}\t{d['tau_int_error']:.4f}"
                f"\t{d['n_values']}"
            )

    for sc in _ANALYSIS_SCALARS:
        if sc in analysis and analysis[sc] is not None:
            lines.append(f"{sc}\t{analysis[sc]:.12e}\t0\t0\t0\t0")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  TSV → {path} ({len(lines) - 1} observables)")


def timeseries_to_tsv(
    meas: Dict[str, Any], path: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
) -> None:
    """Export measurement timeseries arrays to TSV.

    Default columns: trajectory, plaquette, polyakov_abs, flow_t0, flow_w0,
    topo_charge, condensate, wall_seconds.
    """
    if columns is None:
        columns = [
            "trajectory", "plaquette", "polyakov_abs",
            "flow_t0", "flow_w0", "topo_charge",
            "condensate", "wall_seconds",
        ]

    available = [c for c in columns if c in meas and isinstance(meas[c], np.ndarray)]
    if not available:
        raise ValueError("No columns available for TSV export")

    header = "\t".join(available)
    n = meas["n_configs"]
    lines = [header]
    for i in range(n):
        row = []
        for c in available:
            v = meas[c][i]
            if np.isnan(v) if np.issubdtype(type(v), np.floating) else False:
                row.append("NaN")
            else:
                row.append(f"{v}" if isinstance(v, (int, np.integer)) else f"{v:.12e}")
        lines.append("\t".join(row))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  TSV → {path} ({n} rows × {len(available)} cols)")


# ── Flow curve extraction ──────────────────────────────────────────────

def extract_flow_curves(meas: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Extract flow curves from measurement data.

    Returns dict with:
        t:              float array [N_t] (flow times from first config)
        energy_density: float array [n_configs, N_t]
        t2_e:           float array [n_configs, N_t]
    or None if no flow data is present.
    """
    raw = meas.get("raw", [])
    flow_configs = [m for m in raw if m.get("flow") and m["flow"].get("flow_curve")]
    if not flow_configs:
        return None

    ref_curve = flow_configs[0]["flow"]["flow_curve"]
    nt = len(ref_curve)
    t = np.array([p["t"] for p in ref_curve], dtype=np.float64)

    n = len(flow_configs)
    ed = np.full((n, nt), np.nan, dtype=np.float64)
    t2e = np.full((n, nt), np.nan, dtype=np.float64)

    for i, m in enumerate(flow_configs):
        curve = m["flow"]["flow_curve"]
        for j, p in enumerate(curve[:nt]):
            ed[i, j] = p["energy_density"]
            t2e[i, j] = p["t2_e"]

    return {"t": t, "energy_density": ed, "t2_e": t2e}


# ── Utilities ──────────────────────────────────────────────────────────

def _get_nested(d: Dict, key1: str, key2: str, default: float = np.nan) -> float:
    """Safely extract nested optional field, returning NaN on absence."""
    sub = d.get(key1)
    if sub is None:
        return default
    v = sub.get(key2)
    return float(v) if v is not None else default


def summary(analysis: Dict[str, Any]) -> str:
    """One-line summary of an analysis dict for quick inspection."""
    parts = [f"ensemble={analysis.get('ensemble_id', '?')}"]
    parts.append(f"n={analysis.get('n_configs', '?')}")
    if "plaquette" in analysis:
        p = analysis["plaquette"]
        parts.append(f"<P>={p['mean']:.6f}±{p['error']:.1e}")
    if "t0" in analysis and analysis["t0"]:
        t = analysis["t0"]
        parts.append(f"t0={t['mean']:.4f}±{t['error']:.1e}")
    if "w0" in analysis and analysis["w0"]:
        w = analysis["w0"]
        parts.append(f"w0={w['mean']:.4f}±{w['error']:.1e}")
    return "  ".join(parts)


def list_observables(analysis: Dict[str, Any]) -> List[str]:
    """Return names of all observables present in an analysis dict."""
    obs = []
    for key in _ANALYSIS_OBSERVABLES:
        if key in analysis and analysis[key] is not None:
            obs.append(key)
    for key in _ANALYSIS_SCALARS:
        if key in analysis and analysis[key] is not None:
            obs.append(key)
    return obs


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hotspring_reader.py <path-to-json-or-dir>")
        print("  If path is a file: load as analysis and print summary + TSV")
        print("  If path is a dir:  load measurements and print timeseries TSV")
        sys.exit(0)

    target = Path(sys.argv[1])
    if target.is_dir():
        m = load_measurements(target)
        print(f"Loaded {m['n_configs']} configs")
        print(f"  <P> range: [{m['plaquette'].min():.6f}, {m['plaquette'].max():.6f}]")
        tsv_path = target / "timeseries.tsv"
        timeseries_to_tsv(m, tsv_path)
    elif target.is_file():
        a = load_analysis(target)
        print(summary(a))
        print(f"  Observables: {', '.join(list_observables(a))}")
        tsv_path = target.with_suffix(".tsv")
        to_tsv(a, tsv_path)
    else:
        print(f"Not found: {target}")
        sys.exit(1)
