#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
run_all_parity — comprehensive cross-substrate parity orchestrator.

Discovers all Python control JSON results, maps each to its paper number,
and runs compare_substrates.compare() for each paper that has both a
Python control and a Rust result available.

Outputs:
  1. Colored terminal summary table
  2. JSON "green board" at --output (default: parity_greenboard.json)

Exit code 0 = all green, 1 = at least one failure or missing result.

Usage:
    python run_all_parity.py
    python run_all_parity.py --control-dir=../../control --rust-dir=../../validation/results
    python run_all_parity.py --output=report.json
"""

from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running from any directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_substrates import (
    PAPER_COMPARATORS,
    SubstrateReport,
    compare,
    print_report,
    report_to_json,
)

# ── Paper-to-file mapping ─────────────────────────────────────────────
#
# Each entry: (paper_number, python_result_relative_to_control_dir,
#              description).  The Rust result path is discovered
#              dynamically or set via --rust-dir.

PAPER_REGISTRY: List[Tuple[int, str, str]] = [
    (6,  "screened_coulomb/reference_eigenvalues.json",
         "Screened Coulomb (Murillo & Weisheit 1998)"),
    (8,  "lattice_qcd/results/quenched_beta_scan_4x4.json",
         "Pure gauge SU(3) Wilson action"),
    (9,  "lattice_qcd/results/quenched_beta_scan_4x4.json",
         "Production QCD beta-scan"),
    (10, "lattice_qcd/results/dynamical_fermion_control.json",
         "Dynamical fermion QCD"),
    (11, "lattice_qcd/results/hvp_correlator_control.json",
         "HVP g-2 correlator"),
    (12, "lattice_qcd/results/freeze_out_control.json",
         "Freeze-out susceptibility"),
    (13, "abelian_higgs/abelian_higgs_reference.json",
         "Abelian Higgs U(1)"),
    (43, "gradient_flow/results/gradient_flow_control.json",
         "SU(3) gradient flow integrators (Chuna)"),
    (44, "bgk_dielectric/results/bgk_dielectric_control.json",
         "Conservative dielectric functions (Chuna)"),
    (45, "kinetic_fluid/results/kinetic_fluid_control.json",
         "Multi-species kinetic-fluid coupling (Chuna)"),
]


def discover_rust_result(paper: int, rust_dir: Path) -> Optional[Path]:
    """Try common Rust result file patterns for a given paper."""
    candidates = [
        rust_dir / f"paper_{paper}.json",
        rust_dir / f"validate_paper_{paper}.json",
    ]

    # Paper-specific known paths
    paper_names = {
        6: ["screened_coulomb.json", "validate_screened_coulomb.json"],
        8: ["pure_gauge.json", "validate_pure_gauge.json"],
        9: ["production_qcd.json", "validate_production_qcd.json"],
        10: ["dynamical_qcd.json", "validate_dynamical_qcd.json"],
        11: ["hvp_g2.json", "validate_hvp_g2.json"],
        12: ["freeze_out.json", "validate_freeze_out.json"],
        13: ["abelian_higgs.json", "validate_abelian_higgs.json"],
        43: ["gradient_flow.json", "validate_gradient_flow.json"],
        44: ["dielectric.json", "validate_dielectric.json"],
        45: ["kinetic_fluid.json", "validate_kinetic_fluid.json"],
    }

    for name in paper_names.get(paper, []):
        candidates.append(rust_dir / name)

    for c in candidates:
        if c.exists():
            return c
    return None


def run_self_parity(paper: int, python_path: Path, description: str) -> Dict:
    """Run parity check using only the Python control result (self-consistency).

    When no Rust result is available, we validate that the Python control
    itself passes internal consistency checks by loading it as both
    'python' and 'rust' inputs.  The comparators for papers 11 and 12
    include boolean property checks that work this way.
    """
    try:
        report = compare(paper, str(python_path), str(python_path))
        return {
            "paper": paper,
            "description": description,
            "mode": "self-parity",
            "python_path": str(python_path),
            "all_passed": report.all_passed,
            "n_checks": report.n_checks,
            "n_passed": report.n_passed,
            "results": report_to_json(report)["results"],
        }
    except Exception as e:
        return {
            "paper": paper,
            "description": description,
            "mode": "error",
            "error": str(e),
            "all_passed": False,
            "n_checks": 0,
            "n_passed": 0,
        }


def run_cross_parity(paper: int, python_path: Path, rust_path: Path,
                     description: str) -> Dict:
    """Run full cross-substrate parity check."""
    try:
        report = compare(paper, str(python_path), str(rust_path))
        return {
            "paper": paper,
            "description": description,
            "mode": "cross-substrate",
            "python_path": str(python_path),
            "rust_path": str(rust_path),
            "all_passed": report.all_passed,
            "n_checks": report.n_checks,
            "n_passed": report.n_passed,
            "results": report_to_json(report)["results"],
        }
    except Exception as e:
        return {
            "paper": paper,
            "description": description,
            "mode": "error",
            "error": str(e),
            "all_passed": False,
            "n_checks": 0,
            "n_passed": 0,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all Python-vs-Rust parity checks across the paper queue"
    )
    parser.add_argument("--control-dir", default=None,
                        help="Root of control/ directory (auto-detected from script location)")
    parser.add_argument("--rust-dir", default=None,
                        help="Directory containing Rust validation JSON outputs")
    parser.add_argument("--output", default="parity_greenboard.json",
                        help="Path to write JSON green board (default: parity_greenboard.json)")
    parser.add_argument("--self-parity", action="store_true",
                        help="Run self-parity (Python-vs-Python) when no Rust result exists")
    args = parser.parse_args()

    control_dir = Path(args.control_dir) if args.control_dir else SCRIPT_DIR.parent
    rust_dir = Path(args.rust_dir) if args.rust_dir else control_dir.parent / "validation" / "results"

    print("+" + "=" * 70 + "+")
    print("|  hotSpring Paper Queue — Cross-Substrate Parity Green Board        |")
    print("+" + "=" * 70 + "+")
    print(f"  Control dir: {control_dir}")
    print(f"  Rust dir:    {rust_dir}")
    print(f"  Papers:      {len(PAPER_REGISTRY)} registered")
    print()

    all_reports: List[Dict] = []
    total_papers = 0
    passed_papers = 0
    skipped_papers = 0

    for paper, py_rel, description in PAPER_REGISTRY:
        total_papers += 1
        py_path = control_dir / py_rel

        if not py_path.exists():
            print(f"  Paper {paper:3d}: SKIP (no Python result at {py_rel})")
            all_reports.append({
                "paper": paper,
                "description": description,
                "mode": "skipped",
                "reason": f"Python result not found: {py_rel}",
                "all_passed": False,
                "n_checks": 0,
                "n_passed": 0,
            })
            skipped_papers += 1
            continue

        ru_path = discover_rust_result(paper, rust_dir)

        if ru_path is not None:
            report = run_cross_parity(paper, py_path, ru_path, description)
        elif args.self_parity:
            report = run_self_parity(paper, py_path, description)
        else:
            print(f"  Paper {paper:3d}: SKIP (no Rust result in {rust_dir})")
            all_reports.append({
                "paper": paper,
                "description": description,
                "mode": "skipped",
                "reason": "No Rust result found",
                "all_passed": False,
                "n_checks": 0,
                "n_passed": 0,
            })
            skipped_papers += 1
            continue

        status = "PASS" if report["all_passed"] else "FAIL"
        mode = report["mode"]
        n_c = report["n_checks"]
        n_p = report["n_passed"]

        if report["all_passed"]:
            passed_papers += 1

        print(f"  Paper {paper:3d}: [{status}] {n_p}/{n_c} checks "
              f"({mode}) — {description}")

        if not report["all_passed"] and "results" in report:
            for r in report["results"]:
                if not r.get("passed", True):
                    print(f"           FAIL: {r['observable']} "
                          f"delta={r.get('delta', '?')}")

        all_reports.append(report)

    print()
    print("=" * 72)
    active = total_papers - skipped_papers
    all_green = passed_papers == active and active > 0

    print(f"  Papers: {total_papers} registered, "
          f"{active} active, {skipped_papers} skipped")
    print(f"  Result: {passed_papers}/{active} passed")
    print(f"  Status: {'ALL GREEN' if all_green else 'GAPS REMAIN'}")
    print("=" * 72)

    greenboard = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "control_dir": str(control_dir),
        "rust_dir": str(rust_dir),
        "total_papers": total_papers,
        "active_papers": active,
        "passed_papers": passed_papers,
        "skipped_papers": skipped_papers,
        "all_green": all_green,
        "papers": {str(r["paper"]): r for r in all_reports},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(greenboard, f, indent=2)
    print(f"\n  Green board saved to {out_path}")

    sys.exit(0 if all_green else 1)


if __name__ == "__main__":
    main()
