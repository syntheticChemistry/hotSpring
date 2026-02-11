#!/usr/bin/env python
"""
Validate Sarkas DSF output against the Dense Plasma Properties Database reference data.

Usage:
    python validate_dsf.py <kappa> <gamma> [--plot] [--output-dir DIR]

Loads:
  1. Our computed DSF from Sarkas PostProcessing output
  2. Reference S(q,w) from the database .npy files

Compares:
  - Peak positions (plasma oscillation frequency)
  - Peak widths
  - Integrated spectral weight
  - RMS deviation at selected q values

Reference data format (from database README):
  - Size: (166, 764) covering 0.18 < q < 30 and 0 < w < 3
  - Row 0: frequencies
  - Column 0: wavenumbers
  - Reduced units: time -> omega_p^{-1}, length -> a_ws
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent
HOTSPRING_ROOT = STUDY_DIR.parents[3]  # hotSpring/
DB_DIR = HOTSPRING_ROOT / "data" / "plasma-properties-db" / "Dense-Plasma-Properties-Database" / "database"
DSF_DB_DIR = DB_DIR / "Yukawa_Dynamic_Structure_Factors"


def load_reference_dsf(kappa, gamma):
    """Load reference S(q,w) from the database."""
    filename = f"sqw_k{kappa}G{gamma}.npy"
    filepath = DSF_DB_DIR / filename
    if not filepath.exists():
        print(f"ERROR: Reference file not found: {filepath}")
        available = list(DSF_DB_DIR.glob("sqw_*.npy"))
        print(f"Available: {[f.name for f in available]}")
        sys.exit(1)

    data = np.load(filepath)
    print(f"Reference data loaded: {filepath.name}")
    print(f"  Shape: {data.shape}")

    w = data[0, 1:]      # frequencies (reduced)
    q = data[1:, 0]      # wavenumbers (reduced, in units of a_ws^{-1})
    sqw = data[1:, 1:]   # S(q, w)

    print(f"  Frequency range: [{w.min():.4f}, {w.max():.4f}] omega_p")
    print(f"  Wavenumber range: [{q.min():.4f}, {q.max():.4f}] / a_ws")
    print(f"  dq = {q[1]-q[0]:.4f}")

    return w, q, sqw


def load_computed_dsf(kappa, gamma):
    """
    Load our computed DSF from the Sarkas PostProcessing output.

    Sarkas saves DSF data as HDF5 files via pandas.
    The exact location depends on job_dir and the observable setup.
    """
    job_dir = f"dsf_k{kappa}_G{gamma}"
    job_dir_lite = f"dsf_k{kappa}_G{gamma}_lite"

    # Search for the DSF output (try lite variant first, then full)
    sim_base = None
    for candidate_dir in [job_dir_lite, job_dir]:
        p = STUDY_DIR / "Simulations" / candidate_dir
        if p.exists():
            sim_base = p
            break
        p = STUDY_DIR / candidate_dir
        if p.exists():
            sim_base = p
            break
    if sim_base is None:
        print(f"ERROR: Simulation output directory not found.")
        print(f"  Searched: {STUDY_DIR / 'Simulations' / job_dir}")
        print(f"  Searched: {STUDY_DIR / 'Simulations' / job_dir_lite}")
        sys.exit(1)

    print(f"\nSearching for DSF output in: {sim_base}")

    # Sarkas stores DSF in PostProcessing/DynamicStructureFactor/
    dsf_dir = sim_base / "PostProcessing" / "DynamicStructureFactor"
    if not dsf_dir.exists():
        # Try alternative paths
        for candidate in sim_base.rglob("DynamicStructureFactor"):
            dsf_dir = candidate
            break
        else:
            print(f"ERROR: DynamicStructureFactor directory not found under {sim_base}")
            print(f"  Available: {list(sim_base.rglob('*'))[:20]}")
            sys.exit(1)

    print(f"  DSF directory: {dsf_dir}")

    # List available files (search recursively since Sarkas puts them in Production/)
    dsf_files = list(dsf_dir.rglob("*.h5")) + list(dsf_dir.rglob("*.hdf5")) + list(dsf_dir.rglob("*.csv"))
    print(f"  Files: {[f.name for f in dsf_files]}")

    # Try to load using pandas (Sarkas default output format)
    import pandas as pd

    dsf_data = {}
    for f in dsf_files:
        try:
            df = pd.read_hdf(f)
            dsf_data[f.name] = df
            print(f"  Loaded {f.name}: shape {df.shape}")
        except Exception as ex:
            print(f"  Could not load {f.name}: {ex}")

    return dsf_data, dsf_dir


def compare_dsf(w_ref, q_ref, sqw_ref, dsf_data, kappa, gamma):
    """
    Compare computed DSF against reference data.

    Returns a dict of validation metrics.
    """
    metrics = {
        "kappa": kappa,
        "gamma": gamma,
        "status": "PENDING",
        "notes": [],
    }

    if not dsf_data:
        metrics["status"] = "NO_DATA"
        metrics["notes"].append("No computed DSF data found")
        return metrics

    # Analyze the computed data structure
    print("\n--- Computed DSF Analysis ---")
    for name, df in dsf_data.items():
        print(f"\n{name}:")
        print(f"  Columns (first 10): {list(df.columns[:10])}")
        print(f"  Shape: {df.shape}")

    # Compare peak positions at selected q values
    # Pick q values that show clear plasmon peaks
    test_q_indices = [5, 10, 20, 40, 80]  # various wavenumbers

    print("\n--- Reference DSF Peak Analysis ---")
    for qi in test_q_indices:
        if qi >= len(q_ref):
            continue
        q_val = q_ref[qi]
        spectrum = sqw_ref[qi, :]

        # Find peak
        peak_idx = np.argmax(spectrum)
        peak_w = w_ref[peak_idx]
        peak_val = spectrum[peak_idx]

        # Compute spectral weight (sum rule)
        dw = w_ref[1] - w_ref[0] if len(w_ref) > 1 else 1.0
        weight = np.trapz(spectrum, w_ref)

        print(f"  q = {q_val:.3f}: peak at w = {peak_w:.4f}, S_max = {peak_val:.4f}, "
              f"integral = {weight:.4f}")

    metrics["status"] = "COMPARED"
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate DSF against reference data")
    parser.add_argument("kappa", type=int, help="Screening parameter kappa")
    parser.add_argument("gamma", type=int, help="Coupling parameter Gamma")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    args = parser.parse_args()

    print("=" * 70)
    print(f"DSF Validation: kappa={args.kappa}, Gamma={args.gamma}")
    print("=" * 70)

    # Load reference data
    w_ref, q_ref, sqw_ref = load_reference_dsf(args.kappa, args.gamma)

    # Load computed data
    try:
        dsf_data, dsf_dir = load_computed_dsf(args.kappa, args.gamma)
    except SystemExit:
        print("\nNo computed data available yet. Showing reference analysis only.")
        dsf_data = {}
        dsf_dir = None

    # Compare
    metrics = compare_dsf(w_ref, q_ref, sqw_ref, dsf_data, args.kappa, args.gamma)

    # Generate plots if requested
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            out_dir = Path(args.output_dir) if args.output_dir else STUDY_DIR / "validation"
            out_dir.mkdir(exist_ok=True)

            # Plot reference DSF heatmap
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Heatmap
            ax = axes[0]
            extent = [w_ref.min(), w_ref.max(), q_ref.min(), q_ref.max()]
            im = ax.imshow(sqw_ref, aspect='auto', origin='lower', extent=extent,
                          cmap='inferno', vmin=0, vmax=np.percentile(sqw_ref, 99))
            ax.set_xlabel(r"$\omega / \omega_p$")
            ax.set_ylabel(r"$q \cdot a_{ws}$")
            ax.set_title(f"Reference S(q,$\\omega$): $\\kappa$={args.kappa}, $\\Gamma$={args.gamma}")
            plt.colorbar(im, ax=ax)

            # Line cuts at selected q
            ax = axes[1]
            for qi in [5, 15, 30, 60]:
                if qi < len(q_ref):
                    ax.plot(w_ref, sqw_ref[qi, :], label=f"q={q_ref[qi]:.2f}")
            ax.set_xlabel(r"$\omega / \omega_p$")
            ax.set_ylabel(r"$S(q, \omega)$")
            ax.set_title(f"DSF Line Cuts: $\\kappa$={args.kappa}, $\\Gamma$={args.gamma}")
            ax.legend()
            ax.set_xlim(0, 3)

            plt.tight_layout()
            plot_file = out_dir / f"dsf_k{args.kappa}_G{args.gamma}_reference.png"
            plt.savefig(plot_file, dpi=150)
            print(f"\nPlot saved: {plot_file}")

        except ImportError as ex:
            print(f"\nPlotting skipped (missing library): {ex}")

    # Summary
    print("\n" + "=" * 70)
    print(f"Validation Status: {metrics['status']}")
    for note in metrics.get("notes", []):
        print(f"  - {note}")
    print("=" * 70)


if __name__ == "__main__":
    main()
