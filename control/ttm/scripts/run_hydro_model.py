#!/usr/bin/env python
"""
hotSpring - TTM Control: Hydro (Spatial) Model Reproduction

Run the full Two-Temperature Model with radial diffusion in cylindrical
coordinates. Reproduces the results from the Jupyter notebooks in
Two-Temperature-Model/jupyter/Hydro_Runner/.

The hydro model evolves Te(r,t) and Ti(r,t) on a cylindrical grid,
including electron-ion coupling, thermal conduction, hydrodynamic
expansion, and ionization dynamics.

Usage:
    python run_hydro_model.py [--species argon] [--output-dir DIR]
                              [--model SMT|JT_GMS] [--N-grid 101]
                              [--ionization TF|input] [--ionization-file FILE]

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

RESULTS_DIR = SCRIPT_DIR.parent / "reproduction" / "hydro-model"
DATA_DIR = TTM_DIR / "data"


# ============================================================
# Species configurations (from the Hydro_Runner notebooks)
# ============================================================

SPECIES_CONFIGS = {
    "argon": {
        "Z": 18,
        "A": 39.948,
        "n0": 6.2e26,              # /m^3 (~25 bar)
        "Te_init": 15000,           # K (peak electron temperature)
        "Ti_init": 300,             # K (room temperature ions)
        "Te_FWHM": 60e-6,          # m (laser FWHM)
        "r_max": 200e-6,           # m (grid extent)
        "N_grid": 101,
        "model": "SMT",
        "electron_temperature_model": "lorentz",
        "ion_temperature_model": "uniform",
        # Use Saha solution data — TF model lacks χ1 (recombination energy),
        # causing NaN in the Zbar self-consistency loop. Saha data provides
        # proper Zbar(n,T) and χ1(n,T) tables. This matches upstream notebooks.
        "ionization_model": "input",
        "ionization_file": "data/Ar25bar_Saha.txt",
        "description": "Argon 25 bar, Lorentz Te profile, Saha ionization",
    },
    "xenon": {
        "Z": 54,
        "A": 131.293,
        "n0": 1.2e26,              # /m^3 (~5 bar)
        "Te_init": 20000,           # K
        "Ti_init": 300,             # K
        "Te_FWHM": 120e-6,         # m
        "r_max": 500e-6,           # m
        "N_grid": 101,
        "model": "SMT",
        "electron_temperature_model": "lorentz",
        "ion_temperature_model": "uniform",
        "ionization_model": "input",
        "ionization_file": "data/Xe5bar_Saha.txt",
        "description": "Xenon 5 bar, Lorentz Te profile, Saha ionization",
    },
    "helium": {
        "Z": 2,
        "A": 4.0026,
        "n0": 1.8e27,              # /m^3 (~74 bar)
        "Te_init": 30000,           # K
        "Ti_init": 300,             # K
        "Te_FWHM": 60e-6,          # m
        "r_max": 200e-6,           # m
        "N_grid": 101,
        "model": "SMT",
        "electron_temperature_model": "lorentz",
        "ion_temperature_model": "uniform",
        "ionization_model": "input",
        "ionization_file": "data/He74bar_Saha.txt",
        "description": "Helium 74 bar, Lorentz Te profile, Saha ionization",
    },
}


def run_hydro(species_name, config_overrides=None):
    """
    Run the TTM hydro model for a given species.
    """
    from core.Hydro_solver import HydroModel
    from core.exp_setup import Cylindrical_Grid, Experiment
    from core.constants import m_p, k_B

    config = SPECIES_CONFIGS[species_name].copy()
    if config_overrides:
        config.update(config_overrides)

    print(f"\n{'='*60}")
    print(f"TTM Hydro: {species_name.capitalize()}")
    print(f"{'='*60}")
    print(f"  {config['description']}")
    print(f"  Z={config['Z']}, A={config['A']}, n₀={config['n0']:.2e} /m³")
    print(f"  Te₀={config['Te_init']} K, FWHM={config['Te_FWHM']*1e6:.0f} μm")
    print(f"  Grid: {config['N_grid']} points, r_max={config['r_max']*1e6:.0f} μm")
    print(f"  Model: {config['model']}, Te profile: {config['electron_temperature_model']}")

    # Build grid
    grid = Cylindrical_Grid(config["r_max"], N=config["N_grid"])

    # Build experiment
    exp_kwargs = dict(
        grid=grid,
        n0=config["n0"],
        Z=config["Z"],
        A=config["A"],
        Te_experiment_initial=config["Te_init"],
        Te_full_width_at_half_maximum=config["Te_FWHM"],
        ionization_model=config["ionization_model"],
        model=config["model"],
        electron_temperature_model=config["electron_temperature_model"],
        ion_temperature_model=config.get("ion_temperature_model", "uniform"),
        Ti_experimental_initial=config.get("Ti_init", 300),
        Te_experiment_is_peak=True,
    )

    if config.get("ionization_file"):
        exp_kwargs["ionization_file"] = config["ionization_file"]

    experiment = Experiment(**exp_kwargs)

    # Print timescales
    print(f"\n  Timescales:")
    print(f"    τ_ei (equil): {experiment.τei_Equilibration:.3e} s "
          f"({experiment.τei_Equilibration*1e12:.2f} ps)")
    print(f"    τ_diff_e (r_max): {experiment.τDiff_e_rmax:.3e} s "
          f"({experiment.τDiff_e_rmax*1e9:.2f} ns)")
    print(f"    τ_diff_e (dr): {experiment.τDiff_e_dr:.3e} s")

    # Build hydro solver
    hydro = HydroModel(experiment, model=config["model"])
    hydro.make_times()  # Auto-compute dt and tmax from timescales

    print(f"\n  Solver:")
    print(f"    dt = {hydro.dt:.3e} s ({hydro.dt*1e12:.3f} ps)")
    print(f"    tmax = {hydro.tmax:.3e} s ({hydro.tmax*1e9:.3f} ns)")
    print(f"    steps = {len(hydro.t_list)}")

    # Run
    print(f"\n  Running hydro solver...")
    t0 = time.time()
    hydro.solve_hydro()
    wall_time = time.time() - t0
    print(f"  Wall time: {wall_time:.2f}s")

    # Extract results
    r = grid.r
    t_saved = np.array(hydro.t_saved_list)
    Te_profiles = np.array(hydro.Te_list)
    Ti_profiles = np.array(hydro.Ti_list)
    ne_profiles = np.array(hydro.n_e_list)
    ni_profiles = np.array(hydro.n_i_list)
    v_profiles = np.array(hydro.v_list)
    FWHM_list = np.array(hydro.FWHM_list)

    # Center diagnostics
    Te_center = Te_profiles[:, 0]
    Ti_center = Ti_profiles[:, 0]

    print(f"\n  Results:")
    print(f"    Te(r=0, t=0) = {Te_center[0]:.1f} K")
    print(f"    Te(r=0, t=end) = {Te_center[-1]:.1f} K")
    print(f"    Ti(r=0, t=0) = {Ti_center[0]:.1f} K")
    print(f"    Ti(r=0, t=end) = {Ti_center[-1]:.1f} K")
    print(f"    FWHM(t=0) = {FWHM_list[0]*1e6:.1f} μm")
    print(f"    FWHM(t=end) = {FWHM_list[-1]*1e6:.1f} μm")

    return {
        "species": species_name,
        "config": {k: v for k, v in config.items() if k != "description"},
        "wall_time_s": round(wall_time, 3),
        "n_steps": len(t_saved),
        "dt": hydro.dt,
        "tmax": hydro.tmax,
        "tau_ei": experiment.τei_Equilibration,
        "tau_diff_e_rmax": experiment.τDiff_e_rmax,
        "Te_center_init": float(Te_center[0]),
        "Te_center_final": float(Te_center[-1]),
        "Ti_center_init": float(Ti_center[0]),
        "Ti_center_final": float(Ti_center[-1]),
        "FWHM_init_um": float(FWHM_list[0] * 1e6),
        "FWHM_final_um": float(FWHM_list[-1] * 1e6),
        # Data arrays for saving
        "_r": r,
        "_t": t_saved,
        "_Te": Te_profiles,
        "_Ti": Ti_profiles,
        "_FWHM": FWHM_list,
    }


def save_results(result, output_dir):
    """Save hydro model results."""
    species = result["species"]

    # Save center temperature time series
    t = np.atleast_1d(result["_t"])
    Te_all = np.atleast_2d(result["_Te"])
    Ti_all = np.atleast_2d(result["_Ti"])

    # Handle partial results (e.g. Zbar convergence failure)
    n_steps = min(len(t), Te_all.shape[0], Ti_all.shape[0])
    if n_steps == 0:
        print(f"  WARNING: No timesteps completed for {species}, skipping save.")
        return
    t = t[:n_steps]
    Te = Te_all[:n_steps, 0]
    Ti = Ti_all[:n_steps, 0]
    csv_file = output_dir / f"{species}_hydro_center_T.csv"
    np.savetxt(csv_file, np.column_stack([t, Te, Ti]),
               delimiter=",", header="t_s,Te_center_K,Ti_center_K", comments="")
    print(f"  Saved: {csv_file} ({n_steps} timesteps)")

    # Save FWHM evolution
    fwhm_file = output_dir / f"{species}_hydro_FWHM.csv"
    fwhm = np.atleast_1d(result["_FWHM"])
    n = min(len(t), len(fwhm))
    np.savetxt(fwhm_file, np.column_stack([t[:n], fwhm[:n]]),
               delimiter=",", header="t_s,FWHM_m", comments="")
    print(f"  Saved: {fwhm_file}")

    # Save radial profiles at select times
    n_times = len(t)
    snapshot_indices = [0, n_times // 4, n_times // 2, 3 * n_times // 4, n_times - 1]
    r = result["_r"]
    for idx in snapshot_indices:
        if idx >= len(result["_Te"]):
            continue
        snap_file = output_dir / f"{species}_hydro_profile_t{idx}.csv"
        np.savetxt(snap_file,
                   np.column_stack([r, result["_Te"][idx], result["_Ti"][idx]]),
                   delimiter=",", header="r_m,Te_K,Ti_K", comments="")

    # Generate plots (use trimmed arrays for consistency)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if n_steps < 2:
            print(f"  WARNING: Only {n_steps} timestep(s), skipping plots.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Center temperature evolution
        ax = axes[0, 0]
        ax.plot(t * 1e9, Te, 'r-', label=r'$T_e(r=0)$', linewidth=2)
        ax.plot(t * 1e9, Ti, 'b-', label=r'$T_i(r=0)$', linewidth=2)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title(f'{species.capitalize()} — Center Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Radial Te profiles at different times
        ax = axes[0, 1]
        cmap = plt.cm.coolwarm
        for i, idx in enumerate(snapshot_indices):
            if idx >= n_steps:
                continue
            color = cmap(i / len(snapshot_indices))
            label = f"t={t[idx]*1e9:.2f} ns"
            ax.plot(r * 1e6, Te_all[idx], color=color, label=label)
        ax.set_xlabel('r (μm)')
        ax.set_ylabel(r'$T_e$ (K)')
        ax.set_title(f'{species.capitalize()} — Electron Temperature Profiles')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Radial Ti profiles at different times
        ax = axes[1, 0]
        for i, idx in enumerate(snapshot_indices):
            if idx >= n_steps:
                continue
            color = cmap(i / len(snapshot_indices))
            label = f"t={t[idx]*1e9:.2f} ns"
            ax.plot(r * 1e6, Ti_all[idx], color=color, label=label)
        ax.set_xlabel('r (μm)')
        ax.set_ylabel(r'$T_i$ (K)')
        ax.set_title(f'{species.capitalize()} — Ion Temperature Profiles')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. FWHM evolution
        ax = axes[1, 1]
        ax.plot(t[:n] * 1e9, np.array(result["_FWHM"][:n]) * 1e6, 'k-', linewidth=2)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('FWHM (μm)')
        ax.set_title(f'{species.capitalize()} — Plasma Width Evolution')
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'TTM Hydro: {species.capitalize()}', fontsize=16, y=1.01)
        plt.tight_layout()
        plot_file = output_dir / f"{species}_hydro_evolution.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_file}")
    except ImportError:
        print("  (matplotlib not available, skipping plots)")


def main():
    parser = argparse.ArgumentParser(description="TTM hydro model reproduction")
    parser.add_argument("--species", default="argon",
                        help="Comma-separated species (default: argon)")
    parser.add_argument("--model", default="SMT", choices=["SMT", "JT_GMS"])
    parser.add_argument("--N-grid", type=int, default=101)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    species_list = [s.strip() for s in args.species.split(",")]
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("hotSpring - TTM Hydro Model Reproduction")
    print("=" * 70)
    print(f"Species: {species_list}")
    print(f"Model: {args.model}")
    print(f"Grid points: {args.N_grid}")
    print(f"Output: {output_dir}")

    os.chdir(TTM_DIR)

    all_results = []

    for species in species_list:
        if species not in SPECIES_CONFIGS:
            print(f"\nWARNING: Unknown species '{species}'. "
                  f"Available: {list(SPECIES_CONFIGS.keys())}")
            continue

        try:
            overrides = {"model": args.model, "N_grid": args.N_grid}
            result = run_hydro(species, config_overrides=overrides)
            save_results(result, output_dir)

            # Summary (without large arrays)
            summary = {k: v for k, v in result.items() if not k.startswith("_")}
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
    summary_file = output_dir / "hydro_model_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("TTM HYDRO SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        if r.get("Te_center_final") is not None:
            print(f"  {r['species']:<10}: Te {r['Te_center_init']:.0f}→{r['Te_center_final']:.0f} K, "
                  f"Ti {r['Ti_center_init']:.0f}→{r['Ti_center_final']:.0f} K, "
                  f"FWHM {r['FWHM_init_um']:.0f}→{r['FWHM_final_um']:.0f} μm, "
                  f"{r['wall_time_s']:.1f}s")
        else:
            print(f"  {r['species']:<10}: FAIL — {r.get('error', 'unknown')}")
    print(f"\nSummary: {summary_file}")


if __name__ == "__main__":
    main()

