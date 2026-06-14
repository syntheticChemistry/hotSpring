#!/usr/bin/env python
"""
hotSpring - Surrogate Learning Control: Full Reproduction

Reproduce the headline results from:
  "Efficient learning of accurate surrogates for simulations of complex systems"
  Diaw, McKerns, Sagert, Stanton, Murillo - Nature Machine Intelligence, May 2024

This script orchestrates the full reproduction workflow:
  1. Load Zenodo datasets (directed sampling data for nuclear EOS models)
  2. Load Code Ocean capsule code (or our reimplementation)
  3. Train surrogates using optimizer-driven sampling
  4. Compare against random/grid sampling baselines
  5. Record accuracy metrics, training curves, and timing

Prerequisites:
  - Zenodo data downloaded to hotSpring/data/zenodo-surrogate/
  - Code Ocean capsule downloaded to control/surrogate/code-ocean-capsule/
  - conda activate surrogate

Usage:
    python run_reproduction.py [--data-dir DIR] [--output-dir DIR] [--capsule-dir DIR]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCRIPT_DIR.parent
HOTSPRING_DIR = CONTROL_DIR.parents[1]
DATA_DIR = HOTSPRING_DIR / "data" / "zenodo-surrogate"
CAPSULE_DIR = CONTROL_DIR / "code-ocean-capsule"
RESULTS_DIR = CONTROL_DIR / "results"


def check_prerequisites():
    """Verify data and code are available."""
    issues = []

    # Check Zenodo data
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        issues.append(f"Zenodo data not found at {DATA_DIR}")
        issues.append("  Run: bash hotSpring/scripts/download-data.sh")

    # Check Code Ocean capsule
    if not CAPSULE_DIR.exists() or not any(CAPSULE_DIR.iterdir()):
        issues.append(f"Code Ocean capsule not found at {CAPSULE_DIR}")
        issues.append("  Download from: https://doi.org/10.24433/CO.1152070.v1")

    # Check mystic
    try:
        import mystic
    except ImportError:
        issues.append("mystic not installed: pip install mystic")

    return issues


def inventory_zenodo_data():
    """List and describe available Zenodo datasets."""
    if not DATA_DIR.exists():
        return {}

    inventory = {}
    for f in sorted(DATA_DIR.iterdir()):
        stat = f.stat()
        inventory[f.name] = {
            "path": str(f),
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "extension": f.suffix,
        }
    return inventory


def inventory_capsule():
    """Describe the Code Ocean capsule structure."""
    if not CAPSULE_DIR.exists():
        return {}

    inventory = {}
    for root, dirs, files in os.walk(CAPSULE_DIR):
        rel = os.path.relpath(root, CAPSULE_DIR)
        for f in files:
            full = os.path.join(root, f)
            stat = os.stat(full)
            key = os.path.join(rel, f) if rel != "." else f
            inventory[key] = {
                "size_kb": round(stat.st_size / 1024, 1),
                "extension": os.path.splitext(f)[1],
            }
    return inventory


def run_capsule_reproduction(capsule_dir, output_dir):
    """
    Attempt to run the Code Ocean capsule's main script.
    This is a best-effort approach — capsule structure varies.
    """
    # Common Code Ocean capsule layouts:
    # capsule/code/run.sh
    # capsule/code/run
    # capsule/code/main.py

    code_dir = capsule_dir / "code"
    if not code_dir.exists():
        code_dir = capsule_dir  # Maybe flat layout

    # Look for entry points
    candidates = [
        code_dir / "run.sh",
        code_dir / "run",
        code_dir / "main.py",
        code_dir / "run.py",
    ]

    entry = None
    for c in candidates:
        if c.exists():
            entry = c
            break

    if entry is None:
        # List what's available
        py_files = list(code_dir.glob("*.py"))
        sh_files = list(code_dir.glob("*.sh"))
        all_files = py_files + sh_files
        return {
            "status": "NO_ENTRY_POINT",
            "available_scripts": [str(f.name) for f in all_files],
            "message": "Could not find standard Code Ocean entry point. Manual review needed.",
        }

    return {
        "status": "ENTRY_FOUND",
        "entry_point": str(entry),
        "message": f"Found entry: {entry.name}. Run manually to reproduce.",
    }


def main():
    parser = argparse.ArgumentParser(description="Surrogate learning full reproduction")
    parser.add_argument("--data-dir", default=None, help="Zenodo data directory")
    parser.add_argument("--capsule-dir", default=None, help="Code Ocean capsule directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check prerequisites, don't run")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    capsule_dir = Path(args.capsule_dir) if args.capsule_dir else CAPSULE_DIR
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("hotSpring - Surrogate Learning: Full Reproduction")
    print("=" * 70)
    print(f"Zenodo data:    {data_dir}")
    print(f"Code Ocean:     {capsule_dir}")
    print(f"Output:         {output_dir}")
    print()

    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        print("PREREQUISITES:")
        for issue in issues:
            print(f"  ⚠  {issue}")
        print()

    # Inventory Zenodo data
    print("--- Zenodo Data Inventory ---")
    zenodo_inv = inventory_zenodo_data()
    if zenodo_inv:
        for name, info in zenodo_inv.items():
            print(f"  {name}: {info['size_mb']} MB")
    else:
        print("  (empty — download needed)")
    print()

    # Inventory Code Ocean capsule
    print("--- Code Ocean Capsule Inventory ---")
    capsule_inv = inventory_capsule()
    if capsule_inv:
        py_count = sum(1 for v in capsule_inv.values() if v["extension"] == ".py")
        total_size = sum(v["size_kb"] for v in capsule_inv.values())
        print(f"  {len(capsule_inv)} files ({py_count} Python), {total_size:.0f} KB total")
        # Show Python files
        for name, info in capsule_inv.items():
            if info["extension"] == ".py":
                print(f"    {name}: {info['size_kb']} KB")
    else:
        print("  (empty — download needed)")
    print()

    if args.check_only:
        print("[--check-only] Stopping after prerequisite check.")
        return

    if issues:
        print("Cannot proceed with reproduction — resolve prerequisites above.")
        print("Run benchmark functions first (no external data needed):")
        print("  python scripts/run_benchmark_functions.py")
        return

    # Try to run capsule
    print("--- Attempting Capsule Reproduction ---")
    capsule_result = run_capsule_reproduction(capsule_dir, output_dir)
    print(f"  Status: {capsule_result['status']}")
    print(f"  {capsule_result['message']}")

    if capsule_result.get("available_scripts"):
        print(f"  Available scripts: {capsule_result['available_scripts']}")

    # Save inventory report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "zenodo_inventory": zenodo_inv,
        "capsule_inventory": capsule_inv,
        "capsule_result": capsule_result,
        "prerequisites_issues": issues,
    }
    report_file = output_dir / "reproduction_inventory.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nInventory report saved: {report_file}")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("1. Review the Code Ocean capsule code manually")
    print("2. Identify exact dependencies — update envs/surrogate.yaml")
    print("3. Run the capsule's entry script")
    print("4. Compare output to paper's figures")
    print("5. Record training time and accuracy metrics")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

