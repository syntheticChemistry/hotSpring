#!/usr/bin/env python
"""
Run all Sarkas DSF study cases in batch.

Usage:
    python run_batch.py [--dry-run] [--cases k0_G10,k1_G14,...] [--pppm-estimate]

Reads MANIFEST.txt for the full case list. Runs each case sequentially via run_case.py logic.
Produces a summary CSV at the end with timing for every case.
"""

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent
INPUT_DIR = STUDY_DIR / "input_files"
MANIFEST = INPUT_DIR / "MANIFEST.txt"


def load_manifest():
    """Load YAML filenames from MANIFEST.txt."""
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found. Run generate_inputs.py first.")
        sys.exit(1)
    cases = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]
    return cases


def parse_case_key(filename):
    """Extract 'k0_G10' from 'dsf_k0_G10_mks.yaml'."""
    return filename.replace("dsf_", "").replace("_mks.yaml", "")


def run_single_case(yaml_path, pppm_estimate=False):
    """
    Run a single DSF case (preprocess + simulation + postprocess).
    Returns dict with timing and status.
    """
    import importlib.util

    # Import run_case module dynamically to reuse its functions
    spec = importlib.util.spec_from_file_location("run_case", SCRIPT_DIR / "run_case.py")
    run_case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_case)

    result = {
        "yaml": yaml_path.name,
        "case": parse_case_key(yaml_path.name),
        "preprocess_s": None,
        "simulation_s": None,
        "postprocess_s": None,
        "total_s": None,
        "status": "PENDING",
        "error": None,
    }

    os.chdir(STUDY_DIR)
    total_t0 = time.time()

    try:
        result["preprocess_s"] = run_case.run_preprocess(
            str(yaml_path), pppm_estimate=pppm_estimate
        )
        result["simulation_s"] = run_case.run_simulation(str(yaml_path))
        result["postprocess_s"] = run_case.run_postprocess(str(yaml_path))
        result["status"] = "PASS"
    except Exception as ex:
        result["status"] = "FAIL"
        result["error"] = str(ex)
        traceback.print_exc()

    result["total_s"] = time.time() - total_t0
    return result


def main():
    parser = argparse.ArgumentParser(description="Run all DSF study cases")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print cases without running")
    parser.add_argument("--cases", default=None,
                        help="Comma-separated case keys to run (e.g. k1_G14,k2_G31). Default: all.")
    parser.add_argument("--pppm-estimate", action="store_true",
                        help="Run PPPM force error estimation during preprocessing")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip cases that already have simulation output")
    args = parser.parse_args()

    all_cases = load_manifest()

    # Filter to requested cases
    if args.cases:
        requested = set(args.cases.split(","))
        all_cases = [c for c in all_cases if parse_case_key(c) in requested]
        if not all_cases:
            print(f"ERROR: No matching cases found for: {args.cases}")
            sys.exit(1)

    print("=" * 70)
    print("hotSpring - Sarkas DSF Batch Runner")
    print("=" * 70)
    print(f"Cases to run: {len(all_cases)}")
    for c in all_cases:
        print(f"  {c}")
    print()

    if args.dry_run:
        print("[--dry-run] Exiting without running.")
        return

    results = []
    batch_t0 = time.time()

    for i, yaml_name in enumerate(all_cases, 1):
        yaml_path = INPUT_DIR / yaml_name
        case_key = parse_case_key(yaml_name)

        # Check if already completed
        if args.skip_completed:
            sim_dir = STUDY_DIR / "Simulations" / f"dsf_{case_key}"
            prod_dir = sim_dir / "Simulation" / "Production"
            if prod_dir.exists() and any(prod_dir.glob("*.npz")):
                print(f"\n[{i}/{len(all_cases)}] SKIP {case_key} (output exists)")
                results.append({
                    "yaml": yaml_name, "case": case_key,
                    "preprocess_s": 0, "simulation_s": 0, "postprocess_s": 0,
                    "total_s": 0, "status": "SKIPPED", "error": None,
                })
                continue

        print(f"\n{'='*70}")
        print(f"[{i}/{len(all_cases)}] Running: {case_key}")
        print(f"{'='*70}")

        result = run_single_case(yaml_path, pppm_estimate=args.pppm_estimate)
        results.append(result)

        # Print progress
        elapsed = time.time() - batch_t0
        print(f"\n[{i}/{len(all_cases)}] {case_key}: {result['status']} "
              f"({result['total_s']:.1f}s, batch elapsed: {elapsed:.0f}s)")

    # Write summary CSV
    summary_file = STUDY_DIR / "results" / "batch_timing.csv"
    summary_file.parent.mkdir(exist_ok=True)

    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case", "yaml", "status",
            "preprocess_s", "simulation_s", "postprocess_s", "total_s",
            "error",
        ])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    batch_total = time.time() - batch_t0
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"{'Case':<15} {'Status':<8} {'PreProc':>8} {'Sim':>10} {'Post':>8} {'Total':>10}")
    print("-" * 70)
    for r in results:
        def fmt(v):
            return f"{v:.1f}s" if v is not None else "—"
        print(f"{r['case']:<15} {r['status']:<8} {fmt(r['preprocess_s']):>8} "
              f"{fmt(r['simulation_s']):>10} {fmt(r['postprocess_s']):>8} {fmt(r['total_s']):>10}")
    print("-" * 70)
    print(f"Batch total: {batch_total:.0f}s ({batch_total/60:.1f} min)")
    print(f"Summary CSV: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

