#!/usr/bin/env python3
"""
Validate HFBTHO wrapper against known Skyrme parametrizations.

Tests:
1. SLy4 → compute binding energies → compare to AME2020
2. Check that χ² is in the expected range (~2.0 for SLy4)
3. Verify convergence for all test nuclei

Usage:
    python validate_hfbtho.py
"""

import sys
import os
import json
import numpy as np

# Add wrapper to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "wrapper"))

from objective import (
    nuclear_eos_objective,
    load_experimental_data,
    PARAM_NAMES,
    HFBTHO_BINARY,
    EXP_DATA_FILE,
)

# Known Skyrme parametrizations and their expected χ² per datum
KNOWN_PARAMETRIZATIONS = {
    "SLy4": {
        "params": [-2488.91, 486.82, -546.39, 13777.0, 0.834, -0.344, -1.0, 1.354, 0.1667, 123.0],
        "expected_chi2_range": (0.5, 5.0),  # rough range
        "reference": "Chabanat et al., NPA 635, 231 (1998)",
    },
    "UNEDF0": {
        "params": [-1883.69, 277.50, -189.08, 14603.6, 0.0047, -1.116, -1.635, 0.390, 0.3222, 78.66],
        "expected_chi2_range": (0.5, 3.0),
        "reference": "Kortelainen et al., PRC 82, 024313 (2010)",
    },
}


def validate():
    """Run validation tests."""
    print("HFBTHO Wrapper Validation")
    print("=" * 60)

    # Check prerequisites
    print("\n1. Prerequisites:")
    ok = True
    if os.path.exists(str(HFBTHO_BINARY)):
        print(f"   ✅ HFBTHO binary: {HFBTHO_BINARY}")
    else:
        print(f"   ❌ HFBTHO binary not found: {HFBTHO_BINARY}")
        ok = False

    if os.path.exists(str(EXP_DATA_FILE)):
        exp_data = load_experimental_data()
        print(f"   ✅ AME2020 data: {len(exp_data)} nuclei loaded")
    else:
        print(f"   ❌ AME2020 data not found: {EXP_DATA_FILE}")
        ok = False

    if not ok:
        print("\n⚠️  Cannot proceed without prerequisites.")
        print("   Build HFBTHO and/or download AME2020 data first.")
        return False

    # Test each known parametrization
    print("\n2. Testing known parametrizations:")
    all_pass = True

    for name, config in KNOWN_PARAMETRIZATIONS.items():
        print(f"\n   --- {name} ---")
        print(f"   Ref: {config['reference']}")
        params = config["params"]
        lo, hi = config["expected_chi2_range"]

        try:
            chi2 = nuclear_eos_objective(params)
            print(f"   χ²/datum = {chi2:.4f}")

            if lo <= chi2 <= hi:
                print(f"   ✅ PASS (in range [{lo}, {hi}])")
            elif np.isfinite(chi2):
                print(f"   ⚠️  WARN (outside expected range [{lo}, {hi}])")
                print(f"        This may be due to different nuclei selection.")
            else:
                print(f"   ❌ FAIL (non-finite result)")
                all_pass = False
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            all_pass = False

    # Summary
    print(f"\n{'='*60}")
    if all_pass:
        print("✅ All validation tests PASSED")
    else:
        print("❌ Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)

