// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance metadata for all Python baseline values.
//!
//! Every hardcoded expected value in validation binaries traces back to a
//! specific Python control run. This module centralizes that metadata so
//! validation binaries carry machine-readable provenance.
//!
//! # Provenance chain
//!
//! ```text
//! Python script → commit → environment → command → output → Rust constant
//! ```
//!
//! See `METHODOLOGY.md` and `whitePaper/BARRACUDA_SCIENCE_VALIDATION.md`
//! for the full validation protocol.
//!
//! ## Data Sources
//!
//! | Dataset / Publication | DOI / Accession | Notes |
//! |----------------------|-----------------|-------|
//! | AME2020 mass table | [10.1088/1674-1137/abddaf](https://doi.org/10.1088/1674-1137/abddaf) | Experimental binding energies; IAEA AMDC |
//! | `SLy4` (Chabanat et al.) | [10.1016/S0375-9474(98)00180-8](https://doi.org/10.1016/S0375-9474(98)00180-8) | Nucl. Phys. A 635, 231-256 (1998) |
//! | UNEDF0 (Kortelainen et al.) | [10.1103/PhysRevC.82.024313](https://doi.org/10.1103/PhysRevC.82.024313) | Phys. Rev. C 82, 024313 (2010) |
//! | Bender, Heenen, Reinhard | [10.1103/RevModPhys.75.121](https://doi.org/10.1103/RevModPhys.75.121) | E/A NMP target |
//! | Blaizot, Gogny, Grammaticos | [10.1016/0375-9474(76)90292-1](https://doi.org/10.1016/0375-9474(76)90292-1) | K∞ NMP target; Nucl. Phys. A 265, 315 (1976) |
//! | Lattimer & Prakash | [10.1016/j.physrep.2015.12.005](https://doi.org/10.1016/j.physrep.2015.12.005) | J (symmetry energy) NMP target |
//! | Sarkas MD software | [10.1016/j.cpc.2021.108245](https://doi.org/10.1016/j.cpc.2021.108245) | Silvestri et al., CPC (2022); control runs from `control/sarkas/` |
//! | Surrogate learning archive | [10.5281/zenodo.10908462](https://doi.org/10.5281/zenodo.10908462) | Zenodo CC-BY, nuclear EOS convergence |
//!
//! ## Commit Verification
//!
//! To verify baseline commits against the hotSpring repo:
//! ```text
//! git log --oneline fd908c41 -1   # Nuclear EOS control (pinned)
//! git log --oneline 0a6405f -1    # Transport study (Paper 5)
//! git log --oneline 381fdb64 -1   # MD baseline (standalone)
//! git log --oneline 3f0d36d -1    # Screened Coulomb eigenvalues
//! ```

mod analytical;
mod eos;
mod hfb;
mod lattice;
mod nmp;
mod transport;
mod types;

pub use analytical::*;
pub use eos::*;
pub use hfb::*;
pub use lattice::*;
pub use nmp::*;
pub use transport::*;
pub use types::BaselineProvenance;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_provenance_records_non_empty_fields() {
        for p in [
            &L1_PYTHON_CHI2,
            &L1_PYTHON_CANDIDATES,
            &L2_PYTHON_CHI2,
            &L2_PYTHON_CANDIDATES,
            &L2_PYTHON_TOTAL_CHI2,
            &HOTQCD_EOS_PROVENANCE,
            &QUENCHED_BETA_SCAN_PROVENANCE,
            &SCREENED_COULOMB_PROVENANCE,
            &DALIGAULT_CALIBRATION_PROVENANCE,
            &DALIGAULT_FIT_PROVENANCE,
            &TRANSPORT_MD_BASELINE_PROVENANCE,
            &HFB_TEST_NUCLEI_PROVENANCE,
            &ABELIAN_HIGGS_PYTHON_TIMING_MS,
            &TTM_ARGON_EQUILIBRIUM_K,
            &TTM_XENON_EQUILIBRIUM_K,
            &TTM_HELIUM_EQUILIBRIUM_K,
        ] {
            assert!(!p.label.is_empty(), "label empty: {}", p.label);
            assert!(!p.script.is_empty());
            assert!(!p.commit.is_empty());
            assert!(!p.date.is_empty());
            assert!(!p.command.is_empty());
            assert!(!p.environment.is_empty());
            assert!(!p.unit.is_empty());
        }
    }
}
