#!/usr/bin/env python3
"""
Generate ame2020_full.json from the AME2020 mass table (mass_1.mas20).

Parses the IAEA/AMDC AME2020 formatted text file and extracts ALL
experimentally measured nuclei (excludes entries marked with '#' which
are extrapolated/estimated values).

Reference:
  Wang et al., Chinese Physics C 45, 030003 (2021)
  https://amdc.impcas.ac.cn/masstables/Ame2020/mass_1.mas20

Output schema (matches ame2020_selected.json):
  {
    "source": "...",
    "reference": "...",
    "url": "...",
    "n_nuclei": N,
    "nuclei": [
      {"Z": int, "N": int, "A": int, "element": str,
       "binding_energy_MeV": float, "uncertainty_MeV": float},
      ...
    ]
  }
"""

import json
import os
import sys

ELEMENT_NAMES = {
    0: "n", 1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C",
    7: "N", 8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
    28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb",
    42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd",
    49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs",
    56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm",
    63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm",
    70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os",
    77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi",
    84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk",
    98: "Cf", 99: "Es", 100: "Fm", 101: "Md", 102: "No", 103: "Lr",
    104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt",
    110: "Ds", 111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc",
    116: "Lv", 117: "Ts", 118: "Og",
}


def parse_ame2020(filename):
    """Parse mass_1.mas20 and return list of experimentally measured nuclei."""
    nuclei = []
    skipped_estimated = 0
    skipped_other = 0

    with open(filename) as f:
        lines = f.readlines()

    # The data starts after the header. In AME2020, the header is 36 lines,
    # then there are column header lines. Data lines have specific structure.
    # We look for lines that have numeric data in the right positions.
    #
    # Format (from header):
    #   a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,f13.5,1x,f10.5,1x,a2,f13.5,f11.5,1x,i3,1x,f13.6,f12.6
    #   cc NZ  N  Z  A  el  o  mass_excess  unc  binding/A  unc  ..  beta  unc  ..  atomic_mass  unc
    #
    # Columns (0-indexed):
    #   [0]       : Fortran control character
    #   [1:4]     : N-Z
    #   [4:9]     : N
    #   [9:14]    : Z
    #   [14:19]   : A
    #   [20:23]   : Element symbol
    #   [23:27]   : Origin/flag
    #   [28:42]   : Mass excess (keV)       — '#' means estimated
    #   [42:54]   : Mass excess uncertainty
    #   [54:68]   : Binding energy/A (keV)  — '#' means estimated
    #   [68:79]   : Binding energy/A uncertainty

    for line in lines:
        if len(line) < 79:
            continue

        # Try to parse Z, N, A
        try:
            n_val = int(line[4:9].strip())
            z_val = int(line[9:14].strip())
            a_val = int(line[14:19].strip())
        except (ValueError, IndexError):
            continue

        # Skip neutron (Z=0) and hydrogen-1 (A=1, trivial)
        if z_val == 0 and a_val <= 1:
            continue
        if a_val < 1:
            continue

        # Element
        element = line[20:23].strip()
        if not element:
            element = ELEMENT_NAMES.get(z_val, f"Z{z_val}")

        # Binding energy per nucleon (keV) — columns 54:68
        be_str = line[54:68].strip()

        # Check if this is an estimated value (contains '#')
        if '#' in be_str:
            skipped_estimated += 1
            continue

        # Check for '*' (not calculable)
        if '*' in be_str or not be_str:
            skipped_other += 1
            continue

        try:
            be_per_a_kev = float(be_str)
        except ValueError:
            skipped_other += 1
            continue

        # Uncertainty — columns 68:79
        unc_str = line[68:79].strip()
        if '#' in unc_str or '*' in unc_str or not unc_str:
            unc_kev = 1.0  # default 1 keV if missing
        else:
            try:
                unc_kev = float(unc_str)
            except ValueError:
                unc_kev = 1.0

        # Skip entries with essentially zero binding (unbound systems)
        if be_per_a_kev <= 0:
            skipped_other += 1
            continue

        # Convert to total binding energy in MeV
        binding_energy_mev = be_per_a_kev * a_val / 1000.0
        uncertainty_mev = unc_kev * a_val / 1000.0

        # The origin flag in col 23:27 can indicate special states.
        # Unbound markers: -n, -p, -2n, -2p, -nn, -pp, -a, -np, etc.
        # A lone '-' is a data-source flag (previous evaluation), keep those.
        origin = line[23:27].strip()
        unbound_markers = ('-n', '-p', '-2n', '-2p', '-nn', '-pp', '-a', '-np',
                          '-2', '-3', '-4', '+p', '+n', '+a')
        if origin and any(origin.startswith(m) or origin.endswith(m) for m in unbound_markers):
            skipped_other += 1
            continue

        nuclei.append({
            "Z": z_val,
            "N": n_val,
            "A": a_val,
            "element": element,
            "binding_energy_MeV": round(binding_energy_mev, 6),
            "uncertainty_MeV": round(uncertainty_mev, 6),
        })

    # Deduplicate by (Z, N) — keep entry with smallest uncertainty
    seen = {}
    for nuc in nuclei:
        key = (nuc["Z"], nuc["N"])
        if key not in seen or nuc["uncertainty_MeV"] < seen[key]["uncertainty_MeV"]:
            seen[key] = nuc

    unique_nuclei = sorted(seen.values(), key=lambda x: (x["Z"], x["N"]))

    print(f"  Parsed {len(lines)} lines")
    print(f"  Skipped {skipped_estimated} estimated (# marked)")
    print(f"  Skipped {skipped_other} other (unbound, *, missing)")
    print(f"  Raw entries: {len(nuclei)}")
    print(f"  Unique (Z,N): {len(unique_nuclei)}")

    return unique_nuclei


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mass_file = os.path.join(script_dir, "mass_1.mas20.txt")

    if not os.path.exists(mass_file):
        print(f"ERROR: {mass_file} not found.")
        print("Download from: https://amdc.impcas.ac.cn/masstables/Ame2020/mass_1.mas20")
        sys.exit(1)

    print("=== Generating ame2020_full.json ===")
    print(f"  Source: {mass_file}")
    print()

    nuclei = parse_ame2020(mass_file)

    # Statistics
    z_values = sorted(set(n["Z"] for n in nuclei))
    a_range = (min(n["A"] for n in nuclei), max(n["A"] for n in nuclei))
    hfb_range = [n for n in nuclei if 56 <= n["A"] <= 132]
    deformed_range = [n for n in nuclei if n["A"] > 132]

    print()
    print(f"  Z range: {z_values[0]} ({nuclei[0]['element']}) to {z_values[-1]} ({nuclei[-1]['element']})")
    print(f"  A range: {a_range[0]} to {a_range[1]}")
    print(f"  Nuclei with 56 <= A <= 132 (L2 HFB range): {len(hfb_range)}")
    print(f"  Nuclei with A > 132 (deformed region):      {len(deformed_range)}")
    print()

    output = {
        "source": "AME2020 — Full experimentally measured nuclear masses",
        "reference": "Wang et al., Chinese Physics C 45, 030003 (2021)",
        "url": "https://amdc.impcas.ac.cn/masstables/Ame2020/mass_1.mas20",
        "note": "Experimentally measured only (# estimated values excluded). "
                "Generated by generate_ame2020_full.py from mass_1.mas20.txt",
        "n_nuclei": len(nuclei),
        "statistics": {
            "z_range": [z_values[0], z_values[-1]],
            "a_range": list(a_range),
            "n_hfb_range_56_132": len(hfb_range),
            "n_deformed_gt_132": len(deformed_range),
        },
        "nuclei": nuclei,
    }

    out_path = os.path.join(script_dir, "ame2020_full.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(nuclei)} nuclei to {out_path}")
    print()

    # Verify our 52-nucleus selected set is a subset
    selected_path = os.path.join(script_dir, "ame2020_selected.json")
    if os.path.exists(selected_path):
        with open(selected_path) as f:
            selected = json.load(f)
        selected_keys = {(n["Z"], n["N"]) for n in selected["nuclei"]}
        full_keys = {(n["Z"], n["N"]) for n in nuclei}
        overlap = selected_keys & full_keys
        missing = selected_keys - full_keys
        print(f"  Selected set overlap: {len(overlap)}/{len(selected_keys)}")
        if missing:
            print(f"  WARNING: {len(missing)} selected nuclei not in full set: {missing}")
        else:
            print(f"  All 52 selected nuclei found in full set.")

    print()
    print("=== Done ===")


if __name__ == "__main__":
    main()
