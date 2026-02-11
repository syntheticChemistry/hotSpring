#!/usr/bin/env python
"""
hotSpring - TTM Control: Local Model Reproduction

Run the Two-Temperature Model local (ODE-based) solver headlessly for
multiple gas species. Reproduces the results from the Jupyter notebooks
in Two-Temperature-Model/jupyter/Local_model_solver/.

The local model evolves electron and ion temperatures at a single spatial
point (no radial diffusion) — it captures the equilibration physics.

Usage:
    python run_local_model.py [--species argon,xenon,helium] [--output-dir DIR]
                              [--model SMT|JT_GMS] [--dt DT] [--tmax TMAX]

Env: conda activate ttm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add TTM core to path
SCRIPT_DIR = Path(__file__).resolve().parent
TTM_DIR = SCRIPT_DIR.parent / "Two-Temperature-Model"
sys.path.insert(0, str(TTM_DIR))

RESULTS_DIR = SCRIPT_DIR.parent / "reproduction" / "local-model"


# ============================================================
# Gas species parameters (from the TTM notebooks and data)
# ============================================================

SPECIES = {
    "argon": {
        "Z": 18,          # Atomic number
        "A": 39.948,       # Atomic mass (amu)
        "n0": 6.2e26,     # Number density (1/m^3) — ~25 bar
        "Te_init": 15000,  # Initial electron temperature (K) — ~1.3 eV
        "Ti_init": 300,    # Initial ion temperature (K) — room temp
        "description": "Argon at ~25 bar, laser-heated electrons",
    },
    "xenon": {
        "Z": 54,
        "A": 131.293,
        "n0": 1.2e26,     # Number density (1/m^3) — ~5 bar
        "Te_init": 20000,  # K — ~1.7 eV
        "Ti_init": 300,
        "description": "Xenon at ~5 bar, laser-heated electrons",
    },
    "helium": {
        "Z": 2,
        "A": 4.0026,
        "n0": 1.8e27,     # Number density (1/m^3) — ~74 bar
        "Te_init": 30000,  # K — ~2.6 eV
        "Ti_init": 300,
        "description": "Helium at ~74 bar, laser-heated electrons",
    },
}


def run_local_model(species_name, model_name="SMT", dt=None, tmax=None):
    """
    Run the TTM local model for a given species.
    Returns a dict with time series and diagnostics.
    """
    from core.local_ODE_solver import LocalModel
    from core.physics import Physical_Parameters as pp
    from core.constants import m_p, k_B

    spec = SPECIES[species_name]
    m_i = spec["A"] * m_p

    print(f"\n--- {species_name.capitalize()} Local Model ({model_name}) ---")
    print(f"  Z = {spec['Z']}, A = {spec['A']}")
    print(f"  n₀ = {spec['n0']:.2e} /m³")
    print(f"  Te₀ = {spec['Te_init']} K ({spec['Te_init'] * k_B / 1.602e-19:.3f} eV)")
    print(f"  Ti₀ = {spec['Ti_init']} K")

    # Estimate Zbar from Thomas-Fermi
    Zbar_init = pp.Thomas_Fermi_Zbar(spec["Z"], spec["n0"], spec["Te_init"])
    n_e_init = Zbar_init * spec["n0"]
    print(f"  Z̄(TF) = {Zbar_init:.3f}")
    print(f"  n_e = {n_e_init:.2e} /m³")

    # Estimate equilibration time for dt/tmax
    Gamma_ii = pp.Gamma(spec["n0"], spec["Ti_init"], Z=Zbar_init)
    print(f"  Γ_ii = {Gamma_ii:.2f}")

    # Create solver
    model = LocalModel(
        Z=spec["Z"],
        m_i=m_i,
        Ti_init=spec["Ti_init"],
        Te_init=spec["Te_init"],
        ni_init=spec["n0"],
        transport_model=model_name,
    )

    # Set timestep and max time
    if dt is None:
        # Use 1% of initial equilibration time
        from core.physics import SMT as smt_mod, JT_GMS as jt_mod
        params = smt_mod if model_name == "SMT" else jt_mod
        tau_ei, _ = params.ei_relaxation_times(
            n_e_init, spec["n0"], m_i, Zbar_init, spec["Te_init"], spec["Ti_init"]
        )
        dt = max(tau_ei * 0.01, 1e-15)  # At least 1 fs
        print(f"  τ_ei = {tau_ei:.3e} s")

    if tmax is None:
        tmax = dt * 10000  # 10,000 steps default

    print(f"  dt = {dt:.3e} s ({dt*1e12:.3f} ps)")
    print(f"  tmax = {tmax:.3e} s ({tmax*1e9:.3f} ns)")
    print(f"  steps = {int(tmax/dt)}")

    # Run
    t0 = time.time()
    model.solve_ode(dt=dt, tmax=tmax)
    wall_time = time.time() - t0
    print(f"  Wall time: {wall_time:.2f}s")

    # Extract results
    t_array = np.array(model.t_saved_list)
    Te_array = np.array(model.Te_list)
    Ti_array = np.array(model.Ti_list)
    ne_array = np.array(model.n_e_list)
    G_array = np.array(model.G_list)

    # Find equilibration time (when |Te - Ti| < 10% of initial difference)
    dT_init = abs(spec["Te_init"] - spec["Ti_init"])
    equil_idx = None
    for idx in range(len(t_array)):
        if abs(Te_array[idx] - Ti_array[idx]) < 0.1 * dT_init:
            equil_idx = idx
            break

    t_equil = t_array[equil_idx] if equil_idx is not None else None

    print(f"  Final Te = {Te_array[-1]:.1f} K")
    print(f"  Final Ti = {Ti_array[-1]:.1f} K")
    print(f"  Final ΔT = {abs(Te_array[-1] - Ti_array[-1]):.1f} K")
    if t_equil is not None:
        print(f"  Equilibration (10%): {t_equil:.3e} s ({t_equil*1e9:.3f} ns)")
    else:
        print(f"  Equilibration: NOT REACHED in {tmax:.3e} s")

    return {
        "species": species_name,
        "model": model_name,
        "Z": spec["Z"],
        "A": spec["A"],
        "n0": spec["n0"],
        "Te_init": spec["Te_init"],
        "Ti_init": spec["Ti_init"],
        "dt": dt,
        "tmax": tmax,
        "n_steps": len(t_array),
        "wall_time_s": round(wall_time, 3),
        "Te_final": float(Te_array[-1]),
        "Ti_final": float(Ti_array[-1]),
        "t_equil_10pct": float(t_equil) if t_equil else None,
        "t_ns": t_array.tolist(),
        "Te_K": Te_array.tolist(),
        "Ti_K": Ti_array.tolist(),
    }


def save_results(result, output_dir):
    """Save time series data and summary."""
    species = result["species"]
    model = result["model"]

    # Save time series as CSV
    csv_file = output_dir / f"{species}_{model}_timeseries.csv"
    t = np.array(result["t_ns"])
    Te = np.array(result["Te_K"])
    Ti = np.array(result["Ti_K"])
    header = "t_s,Te_K,Ti_K"
    np.savetxt(csv_file, np.column_stack([t, Te, Ti]), delimiter=",",
               header=header, comments="")
    print(f"  Saved: {csv_file}")

    # Save plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Temperature evolution
        ax1.plot(t * 1e9, Te, 'r-', label=r'$T_e$', linewidth=2)
        ax1.plot(t * 1e9, Ti, 'b-', label=r'$T_i$', linewidth=2)
        ax1.set_xlabel('Time (ns)', fontsize=14)
        ax1.set_ylabel('Temperature (K)', fontsize=14)
        ax1.set_title(f'{species.capitalize()} TTM ({model})', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Log scale for early dynamics
        ax2.semilogy(t * 1e9, Te, 'r-', label=r'$T_e$', linewidth=2)
        ax2.semilogy(t * 1e9, Ti, 'b-', label=r'$T_i$', linewidth=2)
        ax2.set_xlabel('Time (ns)', fontsize=14)
        ax2.set_ylabel('Temperature (K)', fontsize=14)
        ax2.set_title(f'{species.capitalize()} TTM (log scale)', fontsize=16)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / f"{species}_{model}_evolution.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved: {plot_file}")
    except ImportError:
        print("  (matplotlib not available, skipping plot)")


def main():
    parser = argparse.ArgumentParser(description="TTM local model reproduction")
    parser.add_argument("--species", default="argon,xenon,helium",
                        help="Comma-separated species (default: argon,xenon,helium)")
    parser.add_argument("--model", default="SMT", choices=["SMT", "JT_GMS"],
                        help="Transport model (default: SMT)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Timestep in seconds (default: auto from τ_ei)")
    parser.add_argument("--tmax", type=float, default=None,
                        help="Max time in seconds (default: auto)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory")
    args = parser.parse_args()

    species_list = [s.strip() for s in args.species.split(",")]
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("hotSpring - TTM Local Model Reproduction")
    print("=" * 70)
    print(f"Species: {species_list}")
    print(f"Transport model: {args.model}")
    print(f"Output: {output_dir}")

    # Change to TTM directory so imports work
    os.chdir(TTM_DIR)

    all_results = []

    for species in species_list:
        if species not in SPECIES:
            print(f"\nWARNING: Unknown species '{species}'. Available: {list(SPECIES.keys())}")
            continue

        try:
            result = run_local_model(species, model_name=args.model,
                                     dt=args.dt, tmax=args.tmax)
            save_results(result, output_dir)

            # Don't save full time series in summary JSON (too large)
            summary = {k: v for k, v in result.items()
                       if k not in ("t_ns", "Te_K", "Ti_K")}
            all_results.append(summary)

        except Exception as ex:
            print(f"\nERROR running {species}: {ex}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "species": species,
                "status": "FAIL",
                "error": str(ex),
            })

    # Save summary
    summary_file = output_dir / "local_model_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print(f"\n\n{'='*70}")
    print("TTM LOCAL MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Species':<10} {'Model':<8} {'Te₀ (K)':>10} {'Ti₀ (K)':>10} "
          f"{'Te_f (K)':>10} {'Ti_f (K)':>10} {'t_eq (ns)':>10} {'Wall (s)':>10}")
    print("-" * 80)
    for r in all_results:
        if r.get("Te_final") is not None:
            t_eq = f"{r['t_equil_10pct']*1e9:.3f}" if r.get("t_equil_10pct") else "—"
            print(f"{r['species']:<10} {r['model']:<8} {r['Te_init']:>10.0f} {r['Ti_init']:>10.0f} "
                  f"{r['Te_final']:>10.1f} {r['Ti_final']:>10.1f} "
                  f"{t_eq:>10} {r['wall_time_s']:>10.3f}")
        else:
            print(f"{r['species']:<10} {'FAIL':<8}")
    print(f"\nSummary: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

