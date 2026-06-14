#!/usr/bin/env bash
# Download AME2020 Atomic Mass Evaluation data from IAEA
# Public data, no login required
# Reference: Wang et al., Chinese Physics C 45, 030003 (2021)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Downloading AME2020 Nuclear Mass Data ==="
echo "Source: IAEA Nuclear Data Services"
echo ""

# AME2020 mass table (mass excess and binding energies)
# The IAEA publishes this as a formatted text file
AME_URL="https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt"
AME_RCE_URL="https://www-nds.iaea.org/amdc/ame2020/rct1_1.rct20.txt"

echo "Downloading mass table..."
if curl -fsSL -o mass_1.mas20.txt "$AME_URL" 2>/dev/null; then
    echo "  ✅ mass_1.mas20.txt downloaded"
elif wget -q -O mass_1.mas20.txt "$AME_URL" 2>/dev/null; then
    echo "  ✅ mass_1.mas20.txt downloaded (wget)"
else
    echo "  ⚠️  Could not download from IAEA. Trying alternate URL..."
    # Alternate: the AMDC mirror
    ALT_URL="https://amdc.in2p3.fr/web/masseval.html"
    echo "  Visit: $ALT_URL"
    echo "  Manual download may be required."
fi

# Also try the nubase evaluation for ground state properties
NUBASE_URL="https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt"
echo "Downloading nubase table..."
curl -fsSL -o nubase_4.mas20.txt "$NUBASE_URL" 2>/dev/null && \
    echo "  ✅ nubase_4.mas20.txt downloaded" || \
    echo "  ⚠️  nubase download failed (not critical)"

echo ""
echo "=== Parsing AME2020 to JSON ==="

# Parse the mass table into a JSON file suitable for our wrapper
python3 << 'PYEOF'
import json
import os

# AME2020 mass table format (fixed-width columns):
# See header of mass_1.mas20.txt for column definitions
# Key columns: N, Z, A, Element, Mass Excess (keV), Binding Energy/A (keV)

nuclei = []
filename = "mass_1.mas20.txt"

if not os.path.exists(filename):
    print(f"  ⚠️  {filename} not found. Creating placeholder with known values.")
    # Fallback: use well-known binding energies from textbooks
    known = [
        {"Z": 2,  "N": 2,   "A": 4,   "element": "He", "binding_energy_MeV": 28.296,   "uncertainty_MeV": 0.001},
        {"Z": 6,  "N": 6,   "A": 12,  "element": "C",  "binding_energy_MeV": 92.162,   "uncertainty_MeV": 0.001},
        {"Z": 8,  "N": 8,   "A": 16,  "element": "O",  "binding_energy_MeV": 127.619,  "uncertainty_MeV": 0.001},
        {"Z": 20, "N": 20,  "A": 40,  "element": "Ca", "binding_energy_MeV": 342.052,  "uncertainty_MeV": 0.004},
        {"Z": 20, "N": 28,  "A": 48,  "element": "Ca", "binding_energy_MeV": 415.991,  "uncertainty_MeV": 0.020},
        {"Z": 28, "N": 28,  "A": 56,  "element": "Ni", "binding_energy_MeV": 483.988,  "uncertainty_MeV": 0.004},
        {"Z": 50, "N": 82,  "A": 132, "element": "Sn", "binding_energy_MeV": 1102.851, "uncertainty_MeV": 0.011},
        {"Z": 62, "N": 90,  "A": 152, "element": "Sm", "binding_energy_MeV": 1253.104, "uncertainty_MeV": 0.003},
        {"Z": 66, "N": 96,  "A": 162, "element": "Dy", "binding_energy_MeV": 1333.780, "uncertainty_MeV": 0.003},
        {"Z": 82, "N": 126, "A": 208, "element": "Pb", "binding_energy_MeV": 1636.430, "uncertainty_MeV": 0.001},
        {"Z": 92, "N": 146, "A": 238, "element": "U",  "binding_energy_MeV": 1801.695, "uncertainty_MeV": 0.002},
    ]
    output = {
        "source": "AME2020 (textbook values — full table download failed)",
        "reference": "Wang et al., Chinese Physics C 45, 030003 (2021)",
        "n_nuclei": len(known),
        "nuclei": known
    }
else:
    print(f"  Parsing {filename}...")
    with open(filename) as f:
        lines = f.readlines()

    # Skip header lines (first 39 lines in AME2020 format)
    data_lines = lines[39:]

    for line in data_lines:
        if len(line) < 80:
            continue
        try:
            # AME2020 fixed-width format (approximate column positions):
            # cols 1-5: N-Z, cols 6-10: N, cols 11-14: Z, cols 15-19: A
            # cols 20-22: Element, cols 29-41: Mass excess (keV)
            # cols 54-66: Binding energy/A (keV)
            N_Z = line[1:5].strip()
            N = int(line[5:10].strip())
            Z = int(line[10:14].strip())
            A = int(line[14:19].strip())
            element = line[20:22].strip()

            # Binding energy per nucleon (keV) — may have '#' for estimated
            be_str = line[54:66].strip().replace('#', '')
            if not be_str:
                continue
            be_per_a_kev = float(be_str)

            # Uncertainty (keV)
            unc_str = line[66:78].strip().replace('#', '')
            unc_kev = float(unc_str) if unc_str else 1.0

            # Convert to total binding energy in MeV
            binding_energy_MeV = be_per_a_kev * A / 1000.0
            uncertainty_MeV = unc_kev * A / 1000.0

            nuclei.append({
                "Z": Z,
                "N": N,
                "A": A,
                "element": element,
                "binding_energy_MeV": round(binding_energy_MeV, 3),
                "uncertainty_MeV": round(uncertainty_MeV, 3)
            })
        except (ValueError, IndexError):
            continue

    output = {
        "source": "AME2020 — IAEA Nuclear Data Services",
        "reference": "Wang et al., Chinese Physics C 45, 030003 (2021)",
        "url": "https://www-nds.iaea.org/amdc/ame2020/",
        "n_nuclei": len(nuclei),
        "nuclei": nuclei
    }

with open("ame2020_selected.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"  ✅ Saved {output['n_nuclei']} nuclei to ame2020_selected.json")
PYEOF

echo ""
echo "=== Done ==="
echo "Files:"
ls -la *.json *.txt 2>/dev/null

