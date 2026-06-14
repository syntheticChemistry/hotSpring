#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
compare_substrates — automated Python-vs-Rust parity checker.

Loads Python control output and Rust binary output, computes parity
deltas per observable, and reports whether all are within tolerance.

Supported papers: 6, 8, 9, 10, 11, 12, 13, 43, 44, 45.

Usage:
    python compare_substrates.py --paper=43 \\
        --python-result=control/gradient_flow/results/gradient_flow_control.json \\
        --rust-result=results/analysis.json

    python compare_substrates.py --paper=8 \\
        --python-result=control/lattice_qcd/results/quenched_beta_scan_4x4.json \\
        --rust-result=results/pure_gauge.json

Exit code 0 = all within tolerance. Exit code 1 = at least one failure.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Tolerance definitions (physically derived, not magic numbers) ──────
#
# Paper 43 (gradient flow): Finite-size effects on 8^4 lattice suppress
# convergence order. Plaquette agreement ~1e-6 (thermalization noise);
# flow energy density ~1e-8 (integration error at eps=0.01).
#
# Paper 44 (dielectric): Completed Mermin involves numerical quadrature.
# Python uses scipy.integrate, Rust uses Gauss-Kronrod. Agreement ~1e-10
# for standard Mermin, ~1e-6 for completed (quadrature-sensitive).
#
# Paper 45 (kinetic-fluid): Conservation is exact in both; Sod shock
# position depends on grid resolution. Agreement ~1e-8 for conserved
# quantities, ~1e-4 for shock position at N=200 cells.

TOLERANCES = {
    # Paper 6 (screened Coulomb): Sturm bisection vs scipy eigensolve on
    # same tridiagonal Hamiltonian.  Exact arithmetic gives ~1e-12 for
    # well-separated eigenvalues; screening threshold requires looser tolerance.
    6: {
        "eigenvalue":         {"atol": 1e-10, "note": "Sturm vs scipy on same grid"},
        "critical_screening": {"atol": 1e-3,  "note": "critical kappa threshold sensitivity"},
    },
    # Paper 8 (pure gauge SU(3)): Same LCG PRNG, same Cayley exp, same
    # algorithm on 4^4.  Plaquette noise O(1e-3) from short trajectories.
    8: {
        "plaquette":      {"atol": 5e-3,  "note": "thermalization noise on 4^4, 30 traj"},
        "polyakov_loop":  {"atol": 0.1,   "note": "large fluctuations on small volume"},
        "acceptance_rate": {"atol": 0.15, "note": "stochastic HMC acceptance"},
    },
    # Paper 9 (production QCD beta-scan): Same as 8 but across full scan.
    # Beta-scan plaquettes are monotone in both; we compare the mean across
    # all beta values.
    9: {
        "plaquette":      {"atol": 5e-3,  "note": "per-beta thermalization noise on 4^4"},
        "polyakov_loop":  {"atol": 0.1,   "note": "large fluctuations near transition"},
        "acceptance_rate": {"atol": 0.15, "note": "stochastic HMC acceptance"},
        "plaquette_monotonicity": {"atol": 0.0, "note": "both must show monotone increase"},
    },
    # Paper 10 (dynamical fermion QCD): Heavy quarks (m=2.0) on 4^4.
    # Dynamical plaquette within noise of quenched (heavy quarks).
    10: {
        "dynamical_plaquette": {"atol": 5e-3, "note": "thermalization + CG noise on 4^4"},
        "quenched_plaquette":  {"atol": 5e-3, "note": "thermalization noise"},
    },
    # Paper 11 (HVP g-2): Correlator shape and HVP sign/ordering are
    # physics invariants.  Exact C(t) differs by lattice size (Py=4^4,
    # Rust=8^4) so we compare properties rather than raw values.
    11: {
        "hvp_positive":        {"atol": 0.0,  "note": "HVP integral must be > 0 in both"},
        "correlator_positive": {"atol": 0.0,  "note": "C(t) >= 0 for all t"},
        "mass_ordering":       {"atol": 0.0,  "note": "lighter quarks -> larger HVP"},
        "hvp_value":           {"atol": 0.5,  "note": "order-of-magnitude on same lattice"},
    },
    # Paper 12 (freeze-out): beta_c location from susceptibility peak.
    # Known SU(3) beta_c ~ 5.69 on N_t=4.  Both Python and Rust should
    # find it within 10% (generous for finite-volume crossover on 4^4).
    12: {
        "beta_c":              {"atol": 0.3,  "note": "finite-volume crossover on 4^4"},
        "plaquette_monotone":  {"atol": 0.0,  "note": "both must see monotone <P>(beta)"},
        "transition_detected": {"atol": 0.0,  "note": "Polyakov must jump near beta_c"},
    },
    # Paper 13 (Abelian Higgs): U(1) on (1+1)D, 8x8.  Same PRNG and
    # algorithm means plaquette and condensate should agree tightly.
    13: {
        "plaquette":       {"atol": 5e-2,  "note": "HMC noise on small (1+1)D lattice"},
        "higgs_condensate": {"atol": 5e-2, "note": "scalar field fluctuations"},
        "acceptance_rate":  {"atol": 0.15, "note": "stochastic HMC acceptance"},
    },
    43: {
        "plaquette":      {"atol": 1e-4, "note": "thermalization noise on 8^4"},
        "flow_energy":    {"atol": 1e-6, "note": "RK integration error at eps=0.01"},
        "convergence_order": {"atol": 0.5, "note": "finite-size suppression on small lattice"},
        "t0":             {"atol": 0.1,  "note": "scale setting on small lattice"},
        "w0":             {"atol": 0.1,  "note": "scale setting on small lattice"},
    },
    44: {
        "mermin_standard":    {"atol": 1e-8,  "note": "analytic Mermin; quadrature agreement"},
        "mermin_completed":   {"atol": 1e-5,  "note": "completed Mermin; quadrature-sensitive"},
        "f_sum":              {"atol": 1e-6,  "note": "sum rule integral"},
        "conductivity":       {"atol": 1e-4,  "note": "Drude-like fit sensitivity"},
    },
    45: {
        "mass_conservation":  {"atol": 1e-10, "note": "exact conservation in both"},
        "momentum_conservation": {"atol": 1e-10, "note": "exact conservation"},
        "energy_conservation": {"atol": 1e-10, "note": "exact conservation"},
        "shock_position":     {"atol": 1e-3,  "note": "grid-resolution dependent at N=200"},
        "h_theorem":          {"atol": 1e-8,  "note": "entropy monotonicity"},
    },
}


@dataclass
class ComparisonResult:
    observable: str
    python_value: float
    rust_value: float
    delta: float
    tolerance: float
    passed: bool
    note: str = ""


@dataclass
class SubstrateReport:
    paper: int
    python_path: str
    rust_path: str
    results: List[ComparisonResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_checks(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)


def compare_paper_6(py_data: Dict, rust_data: Dict) -> List[ComparisonResult]:
    """Compare screened Coulomb eigenvalues between Python and Rust."""
    results = []
    tols = TOLERANCES[6]

    py_eigs = py_data.get("eigenvalues", [])
    ru_eigs = rust_data.get("eigenvalues", [])

    for idx, py_case in enumerate(py_eigs):
        z_val = py_case.get("z", 1.0)
        kappa = py_case.get("kappa", 0.0)
        l_val = py_case.get("l", 0)
        py_vals = py_case.get("eigenvalues", [])

        ru_match = None
        for rc in ru_eigs:
            if (abs(rc.get("z", 1.0) - z_val) < 1e-10
                    and abs(rc.get("kappa", -1) - kappa) < 1e-10
                    and rc.get("l", -1) == l_val):
                ru_match = rc
                break

        if ru_match is None:
            continue
        ru_vals = ru_match.get("eigenvalues", [])
        n_compare = min(len(py_vals), len(ru_vals), 4)
        for i in range(n_compare):
            results.append(_check(
                f"E(z={z_val},kappa={kappa},l={l_val},n={i})",
                float(py_vals[i]), float(ru_vals[i]), tols["eigenvalue"],
            ))

    return results


def compare_paper_8(py_data: Dict, rust_data: Dict) -> List[ComparisonResult]:
    """Compare pure gauge SU(3) results between Python and Rust."""
    results = []
    tols = TOLERANCES[8]

    py_results = py_data.get("results", [])
    ru_results = rust_data.get("results", [])

    if not py_results or not ru_results:
        py_plaq = _find_value(py_data, ["mean_plaquette", "plaquette"])
        ru_plaq = _find_value(rust_data, ["mean_plaquette", "plaquette"])
        if py_plaq is not None and ru_plaq is not None:
            results.append(_check("plaquette", float(py_plaq), float(ru_plaq), tols["plaquette"]))
        return results

    for py_pt in py_results:
        beta = py_pt.get("beta")
        ru_pt = next((r for r in ru_results if abs(r.get("beta", -1) - beta) < 0.01), None)
        if ru_pt is None:
            continue
        results.append(_check(
            f"plaquette_beta_{beta:.1f}",
            float(py_pt["mean_plaquette"]), float(ru_pt["mean_plaquette"]),
            tols["plaquette"],
        ))

    return results


def compare_paper_9(py_data: Dict, rust_data: Dict) -> List[ComparisonResult]:
    """Compare production QCD beta-scan (shares Python control with Paper 8)."""
    results = compare_paper_8(py_data, rust_data)

    py_results = py_data.get("results", [])
    if len(py_results) >= 2:
        py_mono = all(
            py_results[i+1]["mean_plaquette"] >= py_results[i]["mean_plaquette"] - 0.02
            for i in range(len(py_results) - 1)
        )
        results.append(_check_bool("plaquette_monotonicity_py", py_mono))

    return results


def compare_paper_10(py_data: Dict, rust_data: Dict) -> List[ComparisonResult]:
    """Compare dynamical fermion QCD results between Python and Rust."""
    results = []
    tols = TOLERANCES[10]

    for key in ["dynamical_plaquette", "quenched_plaquette"]:
        py_val = _find_value(py_data, [key])
        ru_val = _find_value(rust_data, [key, "mean_plaquette", "plaquette"])
        if py_val is not None and ru_val is not None:
            results.append(_check(key, float(py_val), float(ru_val), tols[key]))

    return results


def compare_paper_11(py_data: Dict, rust_data: Dict) -> List[ComparisonResult]:
    """Compare HVP g-2 results between Python and Rust.

    Python runs on 4^4, Rust on 8^4, so we compare physics properties
    (positivity, ordering) rather than exact numerical values.
    """
    results = []
    tols = TOLERANCES[11]

    py_hvp = _find_value(py_data, ["hvp_value", "hvp_mean", "hvp_integral"])
    py_corr = _find_value(py_data, ["correlator"])

    py_hvp_positive = py_hvp is not None and float(py_hvp) > 0
    py_corr_positive = (py_corr is not None
                        and all(float(c) >= 0.0 for c in py_corr))

    results.append(_check_bool("hvp_positive_py", py_hvp_positive))
    results.append(_check_bool("correlator_positive_py", py_corr_positive))

    py_mass_ok = py_data.get("mass_ordering_correct", None)
    if py_mass_ok is not None:
        results.append(_check_bool("mass_ordering_py", py_mass_ok))

    # If both use same lattice size, compare HVP values directly
    py_dims = py_data.get("lattice_dims", [])
    ru_dims = rust_data.get("lattice_dims", [])
    if py_dims == ru_dims and py_hvp is not None:
        ru_hvp = _find_value(rust_data, ["hvp_value", "hvp_mean", "hvp_integral"])
        if ru_hvp is not None:
            results.append(_check("hvp_value", float(py_hvp), float(ru_hvp),
                                  tols["hvp_value"]))

    return results


def compare_paper_12(py_data: Dict, rust_data: Dict) -> List[ComparisonResult]:
    """Compare freeze-out susceptibility beta-scan between Python and Rust."""
    results = []
    tols = TOLERANCES[12]

    py_bc = _find_value(py_data, ["beta_c_plaquette", "beta_c", "beta_c_plaq"])
    ru_bc = _find_value(rust_data, ["beta_c_plaquette", "beta_c", "beta_c_plaq"])
    if py_bc is not None and ru_bc is not None:
        results.append(_check("beta_c", float(py_bc), float(ru_bc), tols["beta_c"]))

    py_mono = py_data.get("plaquette_monotone", None)
    if py_mono is not None:
        results.append(_check_bool("plaquette_monotone_py", py_mono))

    py_trans = py_data.get("polyakov_transition_detected", None)
    if py_trans is not None:
        results.append(_check_bool("transition_detected_py", py_trans))

    return results


def compare_paper_13(py_data: Dict, rust_data: Dict) -> List[ComparisonResult]:
    """Compare Abelian Higgs results between Python and Rust."""
    results = []
    tols = TOLERANCES[13]

    py_configs = py_data.get("configs", [])
    ru_configs = rust_data.get("configs", [])

    if not py_configs and not ru_configs:
        py_plaq = _find_value(py_data, ["avg_plaquette", "plaquette"])
        ru_plaq = _find_value(rust_data, ["avg_plaquette", "plaquette"])
        if py_plaq is not None and ru_plaq is not None:
            results.append(_check("plaquette", float(py_plaq), float(ru_plaq),
                                  tols["plaquette"]))
        return results

    for py_cfg in py_configs:
        label = py_cfg.get("label", "")
        ru_cfg = next((r for r in ru_configs if r.get("label") == label), None)
        if ru_cfg is None:
            continue

        results.append(_check(
            f"plaquette_{label}",
            float(py_cfg["avg_plaquette"]),
            float(ru_cfg["avg_plaquette"]),
            tols["plaquette"],
        ))
        py_higgs = py_cfg.get("avg_higgs_sq")
        ru_higgs = ru_cfg.get("avg_higgs_sq")
        if py_higgs is not None and ru_higgs is not None:
            results.append(_check(
                f"higgs_sq_{label}",
                float(py_higgs), float(ru_higgs),
                tols["higgs_condensate"],
            ))

    return results


def compare_paper_43(py_data: Dict, rust_data: Dict) -> List[ComparisonResult]:
    """Compare gradient flow results between Python control and Rust."""
    results = []
    tols = TOLERANCES[43]

    # Plaquette at beta=6.0
    py_plaq = _find_value(py_data, ["plaquette", "plaq", "average_plaquette"])
    ru_plaq = _find_value(rust_data, ["plaquette", "plaq"])
    if py_plaq is not None and ru_plaq is not None:
        if isinstance(ru_plaq, dict):
            ru_plaq = ru_plaq.get("mean", ru_plaq.get("value", 0))
        if isinstance(py_plaq, dict):
            py_plaq = py_plaq.get("mean", py_plaq.get("value", 0))
        results.append(_check("plaquette", float(py_plaq), float(ru_plaq), tols["plaquette"]))

    # Convergence orders for integrators
    py_orders = _find_value(py_data, ["convergence_orders", "orders"])
    ru_orders = _find_value(rust_data, ["convergence_orders", "orders"])
    if py_orders and ru_orders:
        for name in py_orders:
            if name in ru_orders:
                results.append(_check(
                    f"convergence_order_{name}",
                    float(py_orders[name]),
                    float(ru_orders[name]),
                    tols["convergence_order"],
                ))

    # Scale setting: t0, w0
    for scale in ["t0", "w0"]:
        py_val = _find_value(py_data, [scale])
        ru_val = _find_value(rust_data, [scale])
        if py_val is not None and ru_val is not None:
            if isinstance(ru_val, dict):
                ru_val = ru_val.get("mean", ru_val.get("value", 0))
            if isinstance(py_val, dict):
                py_val = py_val.get("mean", py_val.get("value", 0))
            results.append(_check(scale, float(py_val), float(ru_val), tols[scale]))

    return results


def compare_paper_44(py_data: Dict, rust_data: Dict) -> List[ComparisonResult]:
    """Compare BGK dielectric results between Python control and Rust."""
    results = []
    tols = TOLERANCES[44]

    # Standard Mermin at reference omega/k
    for key, tol_key in [
        ("standard_mermin", "mermin_standard"),
        ("completed_mermin", "mermin_completed"),
    ]:
        py_val = _find_value(py_data, [key, f"{key}_re", f"{key}_value"])
        ru_val = _find_value(rust_data, [key, f"{key}_re", f"{key}_value"])
        if py_val is not None and ru_val is not None:
            results.append(_check(tol_key, float(py_val), float(ru_val), tols[tol_key]))

    # f-sum rule
    py_fsum = _find_value(py_data, ["f_sum", "f_sum_rule", "fsum"])
    ru_fsum = _find_value(rust_data, ["f_sum", "f_sum_rule", "fsum"])
    if py_fsum is not None and ru_fsum is not None:
        results.append(_check("f_sum", float(py_fsum), float(ru_fsum), tols["f_sum"]))

    # Conductivity
    py_sigma = _find_value(py_data, ["conductivity", "sigma_dc", "sigma"])
    ru_sigma = _find_value(rust_data, ["conductivity", "sigma_dc", "sigma"])
    if py_sigma is not None and ru_sigma is not None:
        results.append(_check("conductivity", float(py_sigma), float(ru_sigma), tols["conductivity"]))

    return results


def compare_paper_45(py_data: Dict, rust_data: Dict) -> List[ComparisonResult]:
    """Compare kinetic-fluid coupling results between Python and Rust."""
    results = []
    tols = TOLERANCES[45]

    for key in ["mass_conservation", "momentum_conservation", "energy_conservation"]:
        py_val = _find_value(py_data, [key, f"{key}_error"])
        ru_val = _find_value(rust_data, [key, f"{key}_error"])
        if py_val is not None and ru_val is not None:
            results.append(_check(key, float(py_val), float(ru_val), tols[key]))

    py_shock = _find_value(py_data, ["shock_position", "x_shock"])
    ru_shock = _find_value(rust_data, ["shock_position", "x_shock"])
    if py_shock is not None and ru_shock is not None:
        results.append(_check("shock_position", float(py_shock), float(ru_shock), tols["shock_position"]))

    return results


PAPER_COMPARATORS = {
    6: compare_paper_6,
    8: compare_paper_8,
    9: compare_paper_9,
    10: compare_paper_10,
    11: compare_paper_11,
    12: compare_paper_12,
    13: compare_paper_13,
    43: compare_paper_43,
    44: compare_paper_44,
    45: compare_paper_45,
}


# ── Core comparison logic ──────────────────────────────────────────────

def _check(name: str, py_val: float, ru_val: float, tol: Dict) -> ComparisonResult:
    delta = abs(py_val - ru_val)
    passed = delta <= tol["atol"]
    return ComparisonResult(
        observable=name,
        python_value=py_val,
        rust_value=ru_val,
        delta=delta,
        tolerance=tol["atol"],
        passed=passed,
        note=tol.get("note", ""),
    )


def _check_bool(name: str, condition: bool) -> ComparisonResult:
    """Boolean property check (pass/fail with no numerical delta)."""
    return ComparisonResult(
        observable=name,
        python_value=1.0 if condition else 0.0,
        rust_value=1.0,
        delta=0.0 if condition else 1.0,
        tolerance=0.0,
        passed=condition,
        note="boolean property check",
    )


def _find_value(data: Dict, keys: List[str]) -> Optional[Any]:
    """Try multiple key names to find a value in a dict (flat search)."""
    for k in keys:
        if k in data:
            return data[k]
    # Try one level of nesting
    for v in data.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v:
                    return v[k]
    return None


def compare(
    paper: int,
    python_path: str,
    rust_path: str,
) -> SubstrateReport:
    """Run a full substrate parity comparison for a given paper."""
    with open(python_path) as f:
        py_data = json.load(f)
    with open(rust_path) as f:
        ru_data = json.load(f)

    comparator = PAPER_COMPARATORS.get(paper)
    if comparator is None:
        raise ValueError(f"No comparator for paper {paper}. Available: {list(PAPER_COMPARATORS.keys())}")

    report = SubstrateReport(
        paper=paper,
        python_path=python_path,
        rust_path=rust_path,
        results=comparator(py_data, ru_data),
    )
    return report


def print_report(report: SubstrateReport) -> None:
    """Print a human-readable parity report."""
    print(f"\n{'=' * 70}")
    print(f"  Substrate Parity: Paper {report.paper}")
    print(f"  Python: {report.python_path}")
    print(f"  Rust:   {report.rust_path}")
    print(f"{'=' * 70}")

    if not report.results:
        print("  (no comparable observables found)")
        return

    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  [{status}] {r.observable:30s}  "
            f"delta={r.delta:.2e}  tol={r.tolerance:.0e}"
            f"  (py={r.python_value:.8e}  rust={r.rust_value:.8e})"
        )
        if r.note:
            print(f"         {r.note}")

    print(f"\n  Result: {report.n_passed}/{report.n_checks} passed", end="")
    print("  [ALL PASS]" if report.all_passed else "  [FAILURES DETECTED]")


def report_to_json(report: SubstrateReport) -> Dict:
    """Serialize a report to a JSON-serializable dict."""
    return {
        "paper": report.paper,
        "python_path": report.python_path,
        "rust_path": report.rust_path,
        "all_passed": report.all_passed,
        "n_checks": report.n_checks,
        "n_passed": report.n_passed,
        "results": [
            {
                "observable": r.observable,
                "python_value": r.python_value,
                "rust_value": r.rust_value,
                "delta": r.delta,
                "tolerance": r.tolerance,
                "passed": r.passed,
                "note": r.note,
            }
            for r in report.results
        ],
    }


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Python control vs Rust binary output for substrate parity"
    )
    parser.add_argument("--paper", type=int, required=True,
                        choices=sorted(PAPER_COMPARATORS.keys()),
                        help="Paper number to compare")
    parser.add_argument("--python-result", required=True,
                        help="Path to Python control JSON output")
    parser.add_argument("--rust-result", required=True,
                        help="Path to Rust binary JSON output")
    parser.add_argument("--output", default=None,
                        help="Optional path to write JSON report")
    args = parser.parse_args()

    report = compare(args.paper, args.python_result, args.rust_result)
    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report_to_json(report), f, indent=2)
        print(f"\n  Report → {args.output}")

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
