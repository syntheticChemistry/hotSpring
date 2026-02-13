# Experimental Data — AME2020 Nuclear Binding Energies

## Source

**Atomic Mass Evaluation 2020 (AME2020)**

- Wang, M. et al., "The AME 2020 atomic mass evaluation (II). Tables, graphs and
  references," Chinese Physics C 45, 030003 (2021).
- Huang, W.J. et al., "The AME 2020 atomic mass evaluation (I). Evaluation of
  input data and adjustment procedures," Chinese Physics C 45, 030002 (2021).
- Data URL: https://www-nds.iaea.org/amdc/ame2020/
- This is public data from the IAEA Nuclear Data Services. No login required.

## How to Obtain

Run the download and selection script:

```bash
bash download_ame2020.sh
```

This will:
1. Download the raw AME2020 mass table (`mass_1.mas20.txt`) from IAEA
2. Parse the fixed-width format (columns documented in the script)
3. Select nuclei matching our validation criteria
4. Write `ame2020_selected.json`

## Selection Criteria

From the ~3400 nuclei in AME2020, we select **52 nuclei** spanning:

- **Z range**: 2 (He) to 92 (U)
- **Mass regions**: light (A < 56), medium (56-132), heavy (A > 132)
- **Shell closures**: all doubly-magic nuclei (⁴He, ¹⁶O, ⁴⁰Ca, ⁴⁸Ca, ⁵⁶Ni,
  ⁷⁸Ni, ⁹⁰Zr, ¹⁰⁰Sn, ¹³²Sn, ²⁰⁸Pb)
- **Isotopic chains**: Sn (Z=50) from N=50 to N=82
- **Uncertainty**: only nuclei with σ_exp < 0.1 MeV (well-measured masses)

This selection tests both the SEMF (light/heavy) and spherical HFB solver
(medium mass) across the chart of nuclides.

## Output Format

`ame2020_selected.json`:

```json
{
  "source": "AME2020",
  "reference": "Wang et al., Chinese Physics C 45, 030003 (2021)",
  "nuclei": [
    {
      "Z": 2,
      "N": 2,
      "A": 4,
      "symbol": "He",
      "binding_energy_MeV": 28.296,
      "uncertainty_MeV": 0.000
    }
  ]
}
```

- **binding_energy_MeV**: Total binding energy (positive = bound). This is
  `B/A × A` from the AME mass excess table.
- **uncertainty_MeV**: Experimental uncertainty propagated from the mass excess.
- All energies in MeV.

## Consumed By

- Python: `wrapper/objective.py::load_experimental_data()`
- Rust: `barracuda/src/data.rs::load_experimental_data()`

Both load the same JSON and produce identical (Z,N) → (B_exp, σ_exp) mappings.

## License

The AME2020 data is published by IAEA and freely available for scientific use.
Our selection script and JSON wrapper are AGPL-3.0.
