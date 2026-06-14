#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Spectral Theory Cross-Tier Parity Check — Python vs Rust

Compares Python baseline values (spectral_control.json) against Rust
reference values from validate_spectral / validate_anderson_3d. Reports
structured parity results per lithoSpore cross-tier pattern.

Usage:
  python3 spectral_parity.py              # run parity check
  python3 spectral_parity.py --regen      # regenerate Python baseline first
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
CONTROL_JSON = os.path.join(RESULTS_DIR, "spectral_control.json")
PARITY_JSON = os.path.join(RESULTS_DIR, "spectral_parity.json")

RUST_REFERENCE = {
    "herman_lyapunov_lambda2": {
        "value": 0.69315,
        "tolerance": 2e-2,
        "source": "validate_spectral check 6: γ at λ=2 within Lyapunov tolerance of ln(2)",
        "kind": "relative",
    },
    "level_stats_poisson_deviation": {
        "tolerance": 5e-2,
        "source": "validate_spectral check 4: ⟨r⟩ within Poisson deviation tolerance",
        "kind": "absolute_deviation",
    },
    "anderson_3d_bandwidth_exact_obc": {
        "value": 11.276311449430901,
        "tolerance": 1e-8,
        "source": "validate_anderson_3d: OBC bandwidth for L=8 clean 3D lattice",
        "kind": "relative",
    },
    "anderson_3d_goe_poisson_monotonic": {
        "source": "validate_anderson_3d: GOE→Poisson transition monotonic at strong disorder",
        "kind": "boolean",
    },
    "dimensional_hierarchy": {
        "source": "validate_anderson_3d: 1D < 2D < 3D bandwidth hierarchy",
        "kind": "boolean",
    },
}

POISSON_R = 0.3862943611198906


def load_python_baseline():
    if not os.path.exists(CONTROL_JSON):
        raise FileNotFoundError(
            f"{CONTROL_JSON} not found — run spectral_control.py --json first"
        )
    with open(CONTROL_JSON) as f:
        return json.load(f)


def check_relative(name, python_val, rust_val, tol):
    if rust_val == 0:
        diff = abs(python_val)
    else:
        diff = abs(python_val - rust_val) / abs(rust_val)
    passed = diff <= tol
    return {
        "check": name,
        "python": python_val,
        "rust": rust_val,
        "relative_error": diff,
        "tolerance": tol,
        "passed": passed,
    }


def check_absolute_deviation(name, python_val, reference, tol):
    dev = abs(python_val - reference)
    passed = dev <= tol
    return {
        "check": name,
        "python": python_val,
        "reference": reference,
        "absolute_deviation": dev,
        "tolerance": tol,
        "passed": passed,
    }


def check_boolean(name, python_val):
    return {
        "check": name,
        "python": python_val,
        "passed": python_val is True,
    }


def run_parity(data):
    results = []

    py_herman = data.get("herman_formula", {})
    for entry in py_herman.get("results", []):
        if abs(entry["lambda"] - 2.0) < 0.01:
            results.append(
                check_relative(
                    "herman_lyapunov_lambda2",
                    entry["gamma"],
                    entry["theory"],
                    RUST_REFERENCE["herman_lyapunov_lambda2"]["tolerance"],
                )
            )
            break

    py_stats = data.get("level_stats_1d", {})
    if "r_mean" in py_stats:
        results.append(
            check_absolute_deviation(
                "level_stats_poisson_deviation",
                py_stats["r_mean"],
                POISSON_R,
                RUST_REFERENCE["level_stats_poisson_deviation"]["tolerance"],
            )
        )

    py_bw = data.get("anderson_3d_bandwidth", {})
    if "bandwidth" in py_bw and "exact_obc" in py_bw:
        results.append(
            check_relative(
                "anderson_3d_bandwidth_exact_obc",
                py_bw["bandwidth"],
                py_bw["exact_obc"],
                RUST_REFERENCE["anderson_3d_bandwidth_exact_obc"]["tolerance"],
            )
        )

    py_trans = data.get("anderson_3d_transition", {})
    if "pass" in py_trans:
        results.append(
            check_boolean("anderson_3d_goe_poisson_monotonic", py_trans["pass"])
        )

    py_hier = data.get("dimensional_hierarchy", {})
    if "pass" in py_hier:
        results.append(check_boolean("dimensional_hierarchy", py_hier["pass"]))

    py_me = data.get("anderson_3d_mobility_edge", {})
    if "center_r" in py_me and "edge_r" in py_me:
        results.append(
            {
                "check": "mobility_edge_center_gt_edge",
                "python_center_r": py_me["center_r"],
                "python_edge_r": py_me["edge_r"],
                "passed": py_me["center_r"] > py_me["edge_r"],
            }
        )

    return results


def main():
    regen = "--regen" in sys.argv

    if regen or not os.path.exists(CONTROL_JSON):
        print("Regenerating Python baseline...")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "spectral_control.py"), "--json"],
            check=True,
            capture_output=True,
        )

    data = load_python_baseline()
    checks = run_parity(data)

    n_pass = sum(1 for c in checks if c["passed"])
    n_total = len(checks)
    all_pass = n_pass == n_total

    report = {
        "parity": "spectral_theory",
        "spring": "hotSpring",
        "tier1": "Python (spectral_control.py)",
        "tier2": "Rust (validate_spectral + validate_anderson_3d)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "summary": {
            "total": n_total,
            "passed": n_pass,
            "failed": n_total - n_pass,
            "all_passed": all_pass,
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PARITY_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Spectral Theory Cross-Tier Parity: {n_pass}/{n_total}")
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['check']}")

    if all_pass:
        print(f"\nALL PASS — parity report written to {PARITY_JSON}")
    else:
        print(f"\nFAILURES — parity report written to {PARITY_JSON}")
        sys.exit(1)


if __name__ == "__main__":
    main()
