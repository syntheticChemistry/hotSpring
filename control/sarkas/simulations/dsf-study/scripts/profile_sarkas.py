#!/usr/bin/env python
"""
Profile a Sarkas simulation to identify compute hotspots.

Usage:
    python profile_sarkas.py <yaml_file> [--top N] [--output-dir DIR]

Runs cProfile on the simulation phase and reports:
  - Top N functions by cumulative time
  - Breakdown: force calc vs FFT vs neighbor list vs integration vs I/O
  - Saves full profile to .prof file for further analysis with snakeviz/pstats

This data directly informs BarraCUDA Phase B shader prioritization.
"""

import argparse
import cProfile
import os
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent


def run_profiled_simulation(input_file):
    """Run Sarkas simulation under cProfile and return stats."""
    from sarkas.processes import PreProcess, Simulation

    import matplotlib
    matplotlib.use('Agg')

    # Preprocessing (not profiled — we want the simulation hotspots)
    print("--- PreProcessing (not profiled) ---")
    t0 = time.time()
    preproc = PreProcess(input_file)
    preproc.setup(read_yaml=True)
    preproc.run()
    print(f"Preprocessing: {time.time() - t0:.2f}s\n")

    # Profile the simulation
    print("--- Simulation (PROFILED) ---")
    profiler = cProfile.Profile()
    profiler.enable()

    sim = Simulation(input_file)
    sim.setup(read_yaml=True)
    sim.run()

    profiler.disable()

    return profiler


def categorize_functions(stats):
    """
    Categorize profiled functions into MD pipeline stages.
    Returns dict mapping category -> cumulative seconds.
    """
    categories = {
        "force_calculation": [],
        "fft_pppm": [],
        "neighbor_list": [],
        "integration": [],
        "io_diagnostics": [],
        "numba_jit": [],
        "other": [],
    }

    # Keywords for categorization
    force_kw = ["force", "potential", "coulomb", "yukawa", "ewald", "pppm_force",
                "short_range", "long_range", "acc_", "update_accel"]
    fft_kw = ["fft", "pyfftw", "pppm", "mesh_charge", "green_function",
              "charge_assignment", "force_interpolation"]
    neighbor_kw = ["neighbor", "cell_list", "verlet_list", "linked_cell",
                   "distance", "pair"]
    integration_kw = ["verlet", "integrat", "velocity_update", "position_update",
                      "thermostat", "berendsen", "rescale"]
    io_kw = ["dump", "save", "write", "print", "log", "csv", "h5", "hdf",
             "pickle", "io_", "diagnostic", "observable"]
    numba_kw = ["numba", "jit", "compile", "dispatcher", "_Compiler"]

    # Get all function stats
    # stats is a pstats.Stats object
    for key, value in stats.stats.items():
        filename, lineno, funcname = key
        cc, nc, tt, ct, callers = value
        full_name = f"{filename}:{funcname}"
        lower = full_name.lower()

        categorized = False
        for kw in fft_kw:
            if kw in lower:
                categories["fft_pppm"].append((full_name, ct))
                categorized = True
                break
        if not categorized:
            for kw in force_kw:
                if kw in lower:
                    categories["force_calculation"].append((full_name, ct))
                    categorized = True
                    break
        if not categorized:
            for kw in neighbor_kw:
                if kw in lower:
                    categories["neighbor_list"].append((full_name, ct))
                    categorized = True
                    break
        if not categorized:
            for kw in integration_kw:
                if kw in lower:
                    categories["integration"].append((full_name, ct))
                    categorized = True
                    break
        if not categorized:
            for kw in io_kw:
                if kw in lower:
                    categories["io_diagnostics"].append((full_name, ct))
                    categorized = True
                    break
        if not categorized:
            for kw in numba_kw:
                if kw in lower:
                    categories["numba_jit"].append((full_name, ct))
                    categorized = True
                    break
        if not categorized:
            categories["other"].append((full_name, ct))

    return categories


def main():
    parser = argparse.ArgumentParser(description="Profile Sarkas simulation")
    parser.add_argument("yaml_file", help="Sarkas YAML input file")
    parser.add_argument("--top", type=int, default=30,
                        help="Show top N functions (default: 30)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for profile output (default: study profiling/)")
    args = parser.parse_args()

    # Resolve paths
    input_file = os.path.abspath(args.yaml_file)
    if not os.path.exists(input_file):
        alt = str(STUDY_DIR / "input_files" / args.yaml_file)
        if os.path.exists(alt):
            input_file = alt
        else:
            print(f"ERROR: Input file not found: {args.yaml_file}")
            sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else STUDY_DIR / "profiling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract case name for file naming
    case_name = Path(input_file).stem  # e.g. dsf_k1_G14_mks

    os.chdir(STUDY_DIR)

    print("=" * 70)
    print("hotSpring - Sarkas Profiling")
    print("=" * 70)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print()

    # Run profiled simulation
    profiler = run_profiled_simulation(input_file)

    # Save raw profile
    prof_file = output_dir / f"{case_name}.prof"
    profiler.dump_stats(str(prof_file))
    print(f"\nProfile saved: {prof_file}")
    print(f"  Analyze with: python -m snakeviz {prof_file}")
    print(f"  Or: python -m pstats {prof_file}")

    # Print top functions
    print(f"\n{'='*70}")
    print(f"TOP {args.top} FUNCTIONS (by cumulative time)")
    print(f"{'='*70}\n")

    stats = pstats.Stats(profiler, stream=sys.stdout)
    stats.sort_stats("cumulative")
    stats.print_stats(args.top)

    # Categorize into MD pipeline stages
    print(f"\n{'='*70}")
    print("MD PIPELINE BREAKDOWN")
    print(f"{'='*70}\n")

    categories = categorize_functions(stats)

    # Get total simulation time from profile
    total_time = sum(v[4] for v in stats.stats.values() if v[4] > 0)
    # Better: use the root cumulative time
    stats.sort_stats("cumulative")

    # Sum top-level time per category (use max to avoid double-counting)
    cat_times = {}
    for cat, funcs in categories.items():
        if funcs:
            # Sum of cumulative times (upper bound, may double-count)
            cat_times[cat] = sum(ct for _, ct in funcs)
        else:
            cat_times[cat] = 0.0

    # Report
    print(f"{'Category':<25} {'Time (s)':>10} {'Top Function'}")
    print("-" * 80)
    for cat in ["force_calculation", "fft_pppm", "neighbor_list",
                "integration", "io_diagnostics", "numba_jit", "other"]:
        t = cat_times.get(cat, 0)
        funcs = categories.get(cat, [])
        top_func = ""
        if funcs:
            funcs_sorted = sorted(funcs, key=lambda x: x[1], reverse=True)
            top_func = funcs_sorted[0][0].split("/")[-1][:40]
        print(f"  {cat:<23} {t:>10.2f}s  {top_func}")

    # Save text summary
    summary_file = output_dir / f"{case_name}_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Sarkas Profile Summary: {case_name}\n")
        f.write(f"{'='*70}\n\n")

        stream = StringIO()
        stats_copy = pstats.Stats(profiler, stream=stream)
        stats_copy.sort_stats("cumulative")
        stats_copy.print_stats(args.top)
        f.write(stream.getvalue())

        f.write(f"\n\nMD Pipeline Breakdown:\n")
        f.write(f"{'-'*70}\n")
        for cat in ["force_calculation", "fft_pppm", "neighbor_list",
                    "integration", "io_diagnostics", "numba_jit", "other"]:
            t = cat_times.get(cat, 0)
            f.write(f"  {cat:<25} {t:.2f}s\n")

    print(f"\nSummary saved: {summary_file}")

    # BarraCUDA prioritization hint
    print(f"\n{'='*70}")
    print("BARRACUDA PRIORITIZATION (from this profile)")
    print(f"{'='*70}")
    ranked = sorted(
        [(cat, cat_times[cat]) for cat in ["force_calculation", "fft_pppm",
         "neighbor_list", "integration", "io_diagnostics"]],
        key=lambda x: x[1], reverse=True
    )
    for i, (cat, t) in enumerate(ranked, 1):
        if t > 0:
            print(f"  {i}. {cat}: {t:.2f}s — port this shader first")
    print()


if __name__ == "__main__":
    main()

