// SPDX-License-Identifier: AGPL-3.0-or-later

use super::types::BaselineProvenance;

/// AME2020 mass table DOI (Wang et al., Chinese Physics C 2021).
///
/// Used for experimental binding energies in [`HFB_TEST_NUCLEI`].
/// IAEA AMDC: <https://www-nds.iaea.org/amdc/>
pub const AME2020_DOI: &str = "10.1088/1674-1137/abddaf";

/// Machine-readable provenance for HFB test nuclei Python baselines.
///
/// `B_python` values from L2 spherical HF+BCS solver (`skyrme_hf.py`).
/// The Rust L2 solver may produce slightly different values due to
/// numerical method differences (bisection vs Brent, density mixing).
/// The 12% relative error tolerance accounts for this.
pub const HFB_TEST_NUCLEI_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "HFB test nuclei binding energies (SLy4, 6 nuclei)",
    script: "surrogate/nuclear-eos/wrapper/skyrme_hf.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.skyrme_hf --param=SLy4 --nuclei=Ni56,Zr90,Sn132,Pb208,Sn112,Zr94",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11)",
    value: 0.0, // aggregate: see HFB_TEST_NUCLEI for per-nucleus values
    unit: "MeV (per-nucleus binding energies in HFB_TEST_NUCLEI)",
};

/// HFB test nuclei with experimental and Python-computed binding energies.
///
/// Provenance: see [`HFB_TEST_NUCLEI_PROVENANCE`] for machine-readable record.
///
/// `B_exp` values from AME2020 (Wang et al., "The AME 2020 atomic mass evaluation (II)",
/// Chinese Physics C 2021). DOI: [`AME2020_DOI`], IAEA AMDC: <https://www-nds.iaea.org/amdc/>
pub const HFB_TEST_NUCLEI: &[(usize, usize, &str, f64, f64)] = &[
    // (Z, N, name, B_exp [MeV], B_python [MeV])
    (28, 28, "Ni-56", 483.99, 476.51),
    (40, 50, "Zr-90", 783.89, 782.44),
    (50, 82, "Sn-132", 1102.85, 1095.41),
    (82, 126, "Pb-208", 1636.43, 1631.58),
    (50, 62, "Sn-112", 953.53, 948.82),
    (40, 54, "Zr-94", 814.68, 811.33),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ame2020_doi_non_empty() {
        assert!(!AME2020_DOI.is_empty());
        assert!(AME2020_DOI.contains("10."));
    }

    #[test]
    fn hfb_test_nuclei_have_positive_energies() {
        for &(z, n, name, b_exp, b_python) in HFB_TEST_NUCLEI {
            assert!(z > 0, "{name} Z must be positive");
            assert!(n > 0, "{name} N must be positive");
            assert!(b_exp > 0.0, "{name} B_exp must be positive");
            assert!(b_python > 0.0, "{name} B_python must be positive");
        }
    }
}
