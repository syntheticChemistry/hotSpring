// SPDX-License-Identifier: AGPL-3.0-or-later

use super::types::BaselineProvenance;

/// Python baseline for quenched SU(3) β-scan (Paper 9).
///
/// Independent HMC at 9 β values on a 4^4 lattice, collecting plaquette,
/// Polyakov loop, and acceptance rate. Algorithm-identical to Rust:
/// same LCG PRNG, Cayley matrix exponential, leapfrog integrator.
/// Python control run pending; Rust value used as reference until then.
pub const QUENCHED_BETA_SCAN_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Quenched SU(3) β-scan on 4^4 (Paper 9)",
    script: "lattice_qcd/scripts/quenched_beta_scan.py",
    commit: "e047444 (hotSpring v0.6.4)",
    date: "2026-02-22",
    command: "python3 -u quenched_beta_scan.py",
    environment: "Python 3.10, NumPy 2.2",
    value: 0.588,
    unit: "<P> at β=6.0 on 4^4 (Rust reference; Python control pending)",
};

/// Python baseline for Abelian Higgs (1+1)D HMC timing.
///
/// Reference runtime from `abelian_higgs_hmc.py` on 8×8 lattice, β=6, κ=0.3, λ=1.0,
/// 50 thermalization + 100 trajectory HMC. Used to verify Rust speedup.
pub const ABELIAN_HIGGS_PYTHON_TIMING_MS: BaselineProvenance = BaselineProvenance {
    label: "Abelian Higgs (1+1)D Python HMC timing (8×8, 50 therm + 100 traj)",
    script: "abelian_higgs/scripts/abelian_higgs_hmc.py",
    commit: "3f0d36d (hotSpring main)",
    date: "2026-02-22",
    command: "python3 abelian_higgs_hmc.py",
    environment: "Python 3.10, NumPy 2.2",
    value: 1750.0,
    unit: "ms",
};

/// Publication: Bazavov et al. (2014), `HotQCD` continuum EOS.
pub const HOTQCD_DOI: &str = "10.1103/PhysRevD.90.094503";

/// `HotQCD` EOS provenance — published lattice QCD data, not Python runs.
///
/// Data points from Bazavov et al., PRD 90, 094503 (2014), Table I.
/// Used in `lattice/eos_tables.rs` for thermodynamic consistency validation.
pub const HOTQCD_EOS_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "HotQCD continuum EOS (Nf=2+1, physical pion mass)",
    script: "N/A (published data table, not from a Python script)",
    commit: "N/A (literature reference)",
    date: "2014-11-01",
    command: "N/A (extracted from PRD 90, 094503, Table I)",
    environment: "N/A (lattice QCD simulation, not reproducible from Python)",
    value: 0.0,
    unit: "p/T^4 (dimensionless; see eos_tables.rs for all data points)",
};

/// Reference source for pure gauge SU(3) validation.
///
/// - Cold plaquette = 1.0: definition (unit links)
/// - Strong-coupling: Creutz, "Quarks, Gluons and Lattices" (1983), Ch. 9
/// - `β_c` ≈ 5.69 for SU(3) on 4^4: Wilson (1974), Creutz (1980)
/// - Plaquette at β=6.0 on 8^4 ≈ 0.594: Bali et al. (1993)
pub const PURE_GAUGE_REFS: &str = "Creutz (1983), Wilson (1974), Bali et al. (1993)";

/// Known critical coupling `β_c` for SU(3) deconfinement on `N_t` = 4.
///
/// `β_c` ≈ 5.6925 (Wilson plaquette action) from:
/// - Bali et al., Phys. Rev. D 47, 3676 (1993)
/// - Engels et al., Nucl. Phys. B 332, 737 (1990)
/// - Creutz, Phys. Rev. D 21, 2308 (1980) (original SU(3) MC)
///
/// On finite 4^4 lattices, the crossover is broad; effective `β_c` measured
/// from susceptibility peaks falls in 5.65–5.72 depending on observable.
/// 5.6925 is the infinite-volume extrapolation for `N_t` = 4.
pub const KNOWN_BETA_C_SU3_NT4: f64 = 5.6925;

/// Python baseline for screened Coulomb bound-state eigenvalues.
///
/// 8 reference eigenvalues (1s, 2s, 2p, 3s for κ=0 and κ=0.5) computed via
/// `yukawa_eigenvalues.py` using a finite-difference Schrödinger solver.
/// Reference: Lam & Varshni (1971), Murillo & Weisheit (1998).
pub const SCREENED_COULOMB_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Screened Coulomb bound-state eigenvalues (8 states)",
    script: "screened_coulomb/scripts/yukawa_eigenvalues.py",
    commit: "3f0d36d (hotSpring main)",
    date: "2026-02-19",
    command: "python3 yukawa_eigenvalues.py",
    environment: "Python 3.10, NumPy 2.2, SciPy 1.14",
    value: 0.0,
    unit: "E_n (atomic units; per-state values in PYTHON_SCREENED_COULOMB_EIGENVALUES)",
};

/// Python reference eigenvalues for screened Coulomb (Yukawa) bound states.
///
/// Provenance: [`SCREENED_COULOMB_PROVENANCE`].
/// Order: 1s κ=0, 2s κ=0, 2p κ=0, 1s κ=0.1, 1s κ=0.5, 1s κ=1.0, He⁺ 1s κ=0.
/// Computed via `scipy.linalg.eigh_tridiagonal` in `yukawa_eigenvalues.py` with
/// N=2000 grid, `r_max=100`.
pub const PYTHON_SCREENED_COULOMB_EIGENVALUES: [f64; 7] = [
    -0.499_688_201_506_501_2,  // 1s κ=0 (H)
    -0.124_980_494_356_236_7,  // 2s κ=0 (H)
    -0.125_006_506_393_321_7,  // 2p κ=0 (H)
    -0.406_749_026_963_666_44, // 1s κ=0.1 (H)
    -0.147_862_177_236_561_37, // 1s κ=0.5 (H)
    -0.010_192_508_268_288_38, // 1s κ=1.0 (H)
    -1.995_029_791_615_593_2,  // 1s κ=0 (He⁺)
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotqcd_doi_is_valid() {
        assert!(HOTQCD_DOI.starts_with("10."));
        assert!(!HOTQCD_DOI.is_empty());
    }

    #[test]
    fn pure_gauge_refs_non_empty() {
        assert!(!PURE_GAUGE_REFS.is_empty());
    }
}
