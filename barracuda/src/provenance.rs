// SPDX-License-Identifier: AGPL-3.0-only

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
//! | SLy4 (Chabanat et al.) | [10.1016/S0375-9474(98)00180-8](https://doi.org/10.1016/S0375-9474(98)00180-8) | Nucl. Phys. A 635, 231-256 (1998) |
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

/// A single provenance record tying a Rust reference value to its Python origin.
#[derive(Debug, Clone)]
pub struct BaselineProvenance {
    /// Human-readable label (e.g. "L1 best chi2/datum")
    pub label: &'static str,
    /// Python script that produced the value (relative to control/)
    pub script: &'static str,
    /// Git commit hash of the control repo at time of run
    pub commit: &'static str,
    /// Date of the control run (ISO 8601)
    pub date: &'static str,
    /// Exact command used to produce the baseline
    pub command: &'static str,
    /// Python environment spec (conda env name or requirements file)
    pub environment: &'static str,
    /// The reference value itself
    pub value: f64,
    /// Unit or description of the value
    pub unit: &'static str,
}

// ═══════════════════════════════════════════════════════════════════
// Nuclear EOS baselines — from control/surrogate/nuclear-eos/
// ═══════════════════════════════════════════════════════════════════

/// Python L1 (SEMF) best χ²/datum — surrogate/nuclear-eos/wrapper/objective.py
pub const L1_PYTHON_CHI2: BaselineProvenance = BaselineProvenance {
    label: "L1 Python best chi2/datum (52 nuclei)",
    script: "surrogate/nuclear-eos/wrapper/objective.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.objective --level=L1 --nuclei=selected --samples=1000 --seed=42",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11, mystic 0.4.2)",
    value: 6.62,
    unit: "chi2/datum",
};

/// Python L1 best candidate count
pub const L1_PYTHON_CANDIDATES: BaselineProvenance = BaselineProvenance {
    label: "L1 Python candidate count",
    script: "surrogate/nuclear-eos/wrapper/objective.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.objective --level=L1 --nuclei=selected --samples=1000 --seed=42",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11, mystic 0.4.2)",
    value: 1008.0,
    unit: "candidates evaluated",
};

/// Python L2 (HFB) best χ²/datum — surrogate/nuclear-eos/wrapper/objective.py
pub const L2_PYTHON_CHI2: BaselineProvenance = BaselineProvenance {
    label: "L2 Python best chi2/datum (52 nuclei)",
    script: "surrogate/nuclear-eos/wrapper/objective.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.objective --level=L2 --nuclei=selected --samples=100 --seed=42",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11, mystic 0.4.2)",
    value: 61.87,
    unit: "chi2/datum",
};

/// Python L2 candidate count
pub const L2_PYTHON_CANDIDATES: BaselineProvenance = BaselineProvenance {
    label: "L2 Python candidate count",
    script: "surrogate/nuclear-eos/wrapper/objective.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.objective --level=L2 --nuclei=selected --samples=100 --seed=42",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11, mystic 0.4.2)",
    value: 96.0,
    unit: "candidates evaluated",
};

/// Python L2 total χ² (un-normalized)
pub const L2_PYTHON_TOTAL_CHI2: BaselineProvenance = BaselineProvenance {
    label: "L2 Python total chi2 (52 nuclei, unnormalized)",
    script: "surrogate/nuclear-eos/wrapper/objective.py",
    commit: "fd908c41 (hotSpring control, pinned)",
    date: "2026-01-15",
    command: "python -m wrapper.objective --level=L2 --nuclei=selected --samples=100 --seed=42",
    environment: "envs/surrogate.yaml (Python 3.10, NumPy 1.24, SciPy 1.11, mystic 0.4.2)",
    value: 28_450.0,
    unit: "chi2_total",
};

// ═══════════════════════════════════════════════════════════════════
// NMP targets — from literature (not Python runs)
// ═══════════════════════════════════════════════════════════════════

/// Nuclear matter property targets and uncertainties.
///
/// Sources:
///   - ρ₀, K∞: Chabanat et al., Nucl. Phys. A 635, 231 (1998)
///   - E/A:     Bender, Heenen, Reinhard, Rev. Mod. Phys. 75, 121 (2003).
///     DOI: [10.1103/RevModPhys.75.121](https://doi.org/10.1103/RevModPhys.75.121)
///   - m*/m:    Chabanat 1998, Bender 2003
///   - J:       Lattimer & Prakash, Phys. Rep. 621, 127 (2016).
///     DOI: [10.1016/j.physrep.2015.12.005](https://doi.org/10.1016/j.physrep.2015.12.005)
pub const NMP_TARGETS: NmpTargets = NmpTargets {
    rho0: NmpTarget {
        value: 0.16,
        sigma: 0.005,
        unit: "fm^-3",
        source: "Chabanat 1998",
    },
    e_a: NmpTarget {
        value: -15.97,
        sigma: 0.5,
        unit: "MeV",
        source: "Bender 2003",
    },
    k_inf: NmpTarget {
        value: 230.0,
        sigma: 20.0,
        unit: "MeV",
        source: "Blaizot 1980, Chabanat 1998",
    },
    m_eff: NmpTarget {
        value: 0.69,
        sigma: 0.1,
        unit: "m*/m",
        source: "Chabanat 1998",
    },
    j: NmpTarget {
        value: 32.0,
        sigma: 2.0,
        unit: "MeV",
        source: "Lattimer & Prakash 2016",
    },
};

/// A single NMP target with uncertainty and source
#[derive(Debug, Clone, Copy)]
pub struct NmpTarget {
    pub value: f64,
    pub sigma: f64,
    pub unit: &'static str,
    pub source: &'static str,
}

/// All five NMP targets
#[derive(Debug, Clone, Copy)]
pub struct NmpTargets {
    pub rho0: NmpTarget,
    pub e_a: NmpTarget,
    pub k_inf: NmpTarget,
    pub m_eff: NmpTarget,
    pub j: NmpTarget,
}

impl NmpTargets {
    /// Values as array [ρ₀, E/A, K∞, m*/m, J] for compact comparison
    #[must_use]
    pub const fn values(&self) -> [f64; 5] {
        [
            self.rho0.value,
            self.e_a.value,
            self.k_inf.value,
            self.m_eff.value,
            self.j.value,
        ]
    }

    /// Sigmas as array
    #[must_use]
    pub const fn sigmas(&self) -> [f64; 5] {
        [
            self.rho0.sigma,
            self.e_a.sigma,
            self.k_inf.sigma,
            self.m_eff.sigma,
            self.j.sigma,
        ]
    }

    /// Check if NMP values are within n_sigma of targets
    #[must_use]
    pub fn within_sigma(&self, values: &[f64; 5], n_sigma: f64) -> [bool; 5] {
        let targets = self.values();
        let sigmas = self.sigmas();
        std::array::from_fn(|i| (values[i] - targets[i]).abs() < n_sigma * sigmas[i])
    }
}

// ═══════════════════════════════════════════════════════════════════
// Reference Skyrme parametrizations — from published literature
// ═══════════════════════════════════════════════════════════════════

/// SLy4 Skyrme parameters (Chabanat et al., Nucl. Phys. A 635, 231-256, 1998, Table I).
///
/// DOI: [10.1016/S0375-9474(98)00180-8](https://doi.org/10.1016/S0375-9474(98)00180-8)
///
/// Order: \[t₀, t₁, t₂, t₃, x₀, x₁, x₂, x₃, α, W₀\] in MeV·fm units.
/// This is the canonical reference parametrization used throughout hotSpring.
pub const SLY4_PARAMS: [f64; 10] = [
    -2488.913, // t₀
    486.818,   // t₁
    -546.395,  // t₂
    13777.0,   // t₃
    0.834,     // x₀
    -0.344,    // x₁
    -1.0,      // x₂
    1.354,     // x₃
    1.0 / 6.0, // α
    123.0,     // W₀
];

/// UNEDF0 Skyrme parameters (Kortelainen et al., Phys. Rev. C 82, 024313, 2010).
///
/// DOI: [10.1103/PhysRevC.82.024313](https://doi.org/10.1103/PhysRevC.82.024313)
///
/// Order: same as [`SLY4_PARAMS`].
pub const UNEDF0_PARAMS: [f64; 10] = [
    -1883.68,  // t₀
    277.50,    // t₁
    -207.20,   // t₂
    14263.6,   // t₃
    0.0085,    // x₀
    -1.532,    // x₁
    -1.0,      // x₂
    0.397,     // x₃
    1.0 / 6.0, // α
    79.53,     // W₀
];

/// Standard 10 Skyrme parameter names, in canonical order.
pub const PARAM_NAMES: [&str; 10] = [
    "t0", "t1", "t2", "t3", "x0", "x1", "x2", "x3", "alpha", "W0",
];

// ═══════════════════════════════════════════════════════════════════
// NMP chi2 evaluation — shared across all nuclear EOS binaries
// ═══════════════════════════════════════════════════════════════════

/// Compute NMP χ² from nuclear matter property values.
///
/// χ²_NMP = Σ ((value_i - target_i) / sigma_i)²
///
/// Uses [`NMP_TARGETS`] as the reference. Returns the sum of squared pulls
/// for the five nuclear matter observables.
pub fn nmp_chi2(values: &[f64; 5]) -> f64 {
    let targets = NMP_TARGETS.values();
    let sigmas = NMP_TARGETS.sigmas();
    targets
        .iter()
        .zip(sigmas.iter())
        .zip(values.iter())
        .map(|((&t, &s), &v)| ((v - t) / s).powi(2))
        .sum()
}

/// Compute NMP χ² from a [`NuclearMatterProps`](crate::physics::NuclearMatterProps).
///
/// Convenience wrapper that extracts the five NMP values and calls [`nmp_chi2`].
pub fn nmp_chi2_from_props(nmp: &crate::physics::NuclearMatterProps) -> f64 {
    let values = [
        nmp.rho0_fm3,
        nmp.e_a_mev,
        nmp.k_inf_mev,
        nmp.m_eff_ratio,
        nmp.j_mev,
    ];
    nmp_chi2(&values)
}

/// Standard NMP observable names.
pub const NMP_NAMES: [&str; 5] = ["rho0", "E/A", "K_inf", "m*/m", "J"];

/// Standard NMP observable units.
pub const NMP_UNITS: [&str; 5] = ["fm^-3", "MeV", "MeV", "", "MeV"];

/// Print a formatted NMP analysis table for a Skyrme parametrization.
///
/// Shows each NMP observable, its computed value, the target, sigma,
/// pull (deviation in sigma units), and PASS/FAIL status.
pub fn print_nmp_analysis(nmp: &crate::physics::NuclearMatterProps) {
    let values = [
        nmp.rho0_fm3,
        nmp.e_a_mev,
        nmp.k_inf_mev,
        nmp.m_eff_ratio,
        nmp.j_mev,
    ];
    let targets = NMP_TARGETS.values();
    let sigmas = NMP_TARGETS.sigmas();

    println!("  NMP Analysis:");
    println!(
        "    {:>8}  {:>10}  {:>8}  {:>10}  {:>8}  {:>6}  Status",
        "Prop", "Value", "Unit", "Target", "Sigma", "Pull"
    );
    for i in 0..5 {
        let pull = (values[i] - targets[i]).abs() / sigmas[i];
        let status = if pull < 2.0 { "PASS" } else { "FAIL" };
        println!(
            "    {:>8}  {:>10.4}  {:>8}  {:>10.4}  {:>8.4}  {:>6.2}σ  {status}",
            NMP_NAMES[i], values[i], NMP_UNITS[i], targets[i], sigmas[i], pull,
        );
    }
    println!("  NMP χ²/datum = {:.4}", nmp_chi2_from_props(nmp) / 5.0);
}

// ═══════════════════════════════════════════════════════════════════
// Sarkas MD baselines — from control/sarkas/
// ═══════════════════════════════════════════════════════════════════

/// Sarkas software version used for control runs
pub const SARKAS_VERSION: &str = "1.0.0";
/// Sarkas pinned commit
pub const SARKAS_COMMIT: &str = "fd908c41";
/// Publication reference for DSF study
pub const SARKAS_PAPER: &str = "Choi, Dharuman, Murillo, Phys. Rev. E 100, 013206 (2019)";

// ═══════════════════════════════════════════════════════════════════
// Transport coefficient baselines — validate_transport.rs
// ═══════════════════════════════════════════════════════════════════

/// Daligault (2012) analytical fit reference.
///
/// Test values in `md/transport.rs::d_star_matches_all_12_sarkas_calibration_points`
/// are computed from the Daligault (2012) practical model, NOT from MD simulation.
/// They are analytical values that both Rust and Python must reproduce.
///
/// Script distinction: `calibrate_daligault_fit.py` fits C_w(κ) from Sarkas MD;
/// `daligault_fit.py` is the validation script using the fitted model.
pub const DALIGAULT_FIT_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Daligault D*(Gamma,kappa) analytical fit validation",
    script: "sarkas/simulations/transport-study/scripts/daligault_fit.py",
    commit: "0a6405f (hotSpring main, Paper 5 transport commit)",
    date: "2026-02-19",
    command: "python3 daligault_fit.py",
    environment: "Python 3.10, NumPy 2.2",
    value: 2.8651e-4,
    unit: "D* reduced (k=1 G=50 reference value)",
};

/// Calibration script for Daligault weak-coupling correction C_w(κ).
///
/// `calibrate_daligault_fit.py` runs a grid search over (κ, Γ) points from
/// Sarkas MD and fits C_w(κ) = exp(1.435 + 0.715κ + 0.401κ²).
/// `daligault_fit.py` is the validation script using the fitted model.
pub const DALIGAULT_CALIBRATION_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Daligault C_w(kappa) calibration from 12 Sarkas points",
    script: "sarkas/simulations/transport-study/scripts/calibrate_daligault_fit.py",
    commit: "0a6405f (hotSpring main, Paper 5 transport commit)",
    date: "2026-02-19",
    command: "python3 calibrate_daligault_fit.py",
    environment: "Python 3.10, NumPy 2.2",
    value: 0.0,
    unit: "C_w fit coefficients (see md/transport.rs for usage)",
};

/// Standalone Python MD baseline for transport coefficients.
///
/// Uses `yukawa_md_baseline.py`: velocity-Verlet in reduced units,
/// FCC lattice init, Berendsen equilibration, NVE production with VACF.
/// Bypasses Sarkas (which has dump-file velocity storage issues).
///
/// Reference: Stanton & Murillo, PRE 93, 043203 (2016)
pub const TRANSPORT_MD_BASELINE_PROVENANCE: BaselineProvenance = BaselineProvenance {
    label: "Standalone Yukawa MD transport baselines (lite: N=500)",
    script: "sarkas/simulations/transport-study/scripts/yukawa_md_baseline.py",
    commit: "381fdb64",
    date: "2026-02-19",
    command: "python3 yukawa_md_baseline.py --lite",
    environment: "Python 3.10, NumPy 2.2, numba 0.60",
    value: 0.0, // aggregate: individual D* values in results JSON
    unit: "D* reduced (per-case values in transport_baseline_standalone_lite.json)",
};

/// Publication: Stanton & Murillo (2016) ionic transport.
pub const STANTON_MURILLO_DOI: &str = "10.1103/PhysRevE.93.043203";

/// Publication: Daligault (2012) practical D* model.
pub const DALIGAULT_DOI: &str = "10.1103/PhysRevE.86.047401";

// ═══════════════════════════════════════════════════════════════════
// HFB validation baselines — verify_hfb.rs
// ═══════════════════════════════════════════════════════════════════

/// AME2020 mass table DOI (Wang et al., Chinese Physics C 2021).
///
/// Used for experimental binding energies in [`HFB_TEST_NUCLEI`].
/// IAEA AMDC: <https://www-nds.iaea.org/amdc/>
pub const AME2020_DOI: &str = "10.1088/1674-1137/abddaf";

/// Machine-readable provenance for HFB test nuclei Python baselines.
///
/// B_python values from L2 spherical HF+BCS solver (`skyrme_hf.py`).
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
/// B_exp values from AME2020 (Wang et al., "The AME 2020 atomic mass evaluation (II)",
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

// ═══════════════════════════════════════════════════════════════════
// Analytical validation references
//
// These validation binaries compare against mathematical ground truth,
// not Python computational runs. BaselineProvenance is NOT applicable
// here because the expected values are exact analytical results or
// published reference tables, not outputs of specific Python scripts.
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// HotQCD EOS — from published lattice QCD data
// ═══════════════════════════════════════════════════════════════════

/// Publication: Bazavov et al. (2014), HotQCD continuum EOS.
pub const HOTQCD_DOI: &str = "10.1103/PhysRevD.90.094503";

/// HotQCD EOS provenance — published lattice QCD data, not Python runs.
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

// ═══════════════════════════════════════════════════════════════════
// Pure gauge SU(3) — from analytical/textbook results
// ═══════════════════════════════════════════════════════════════════

/// Reference source for pure gauge SU(3) validation.
///
/// - Cold plaquette = 1.0: definition (unit links)
/// - Strong-coupling: Creutz, "Quarks, Gluons and Lattices" (1983), Ch. 9
/// - β_c ≈ 5.69 for SU(3) on 4^4: Wilson (1974), Creutz (1980)
/// - Plaquette at β=6.0 on 8^4 ≈ 0.594: Bali et al. (1993)
pub const PURE_GAUGE_REFS: &str = "Creutz (1983), Wilson (1974), Bali et al. (1993)";

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
    unit: "E_n (atomic units; per-state values in validate_screened_coulomb.rs)",
};

// ═══════════════════════════════════════════════════════════════════
// Analytical validation references
//
// These validation binaries compare against mathematical ground truth,
// not Python computational runs. BaselineProvenance is NOT applicable
// here because the expected values are exact analytical results or
// published reference tables, not outputs of specific Python scripts.
// ═══════════════════════════════════════════════════════════════════

/// Reference source for special function validation.
///
/// Expected values are exact mathematical identities (e.g. Γ(n) = (n-1)!,
/// J₀(0) = 1) or published high-precision tables:
///   - Abramowitz & Stegun, "Handbook of Mathematical Functions" (1964)
///   - NIST Digital Library of Mathematical Functions, <https://dlmf.nist.gov> (2023)
///   - SciPy 1.11 `scipy.special` used for cross-validation, not as primary source
pub const SPECIAL_FUNCTION_REFS: &str =
    "Abramowitz & Stegun (1964), NIST DLMF (2023), scipy.special 1.11";

/// Reference source for linear algebra validation.
///
/// Expected values are mathematical identities: ‖A − LU‖ < ε, QᵀQ = I,
/// Ax = λx for eigendecomposition. Cross-validated against NumPy 1.26 / SciPy 1.11.
pub const LINALG_REFS: &str = "NumPy 1.26 / SciPy 1.11 linear algebra";

/// Reference source for optimizer validation.
///
/// Expected values are analytical optima of standard test functions:
/// Rosenbrock minimum at (1,1), sphere minimum at origin, etc.
/// ODE integrators validated via known solutions (exponential decay, harmonic).
/// Cross-validated against SciPy 1.11 `optimize` and `integrate`.
pub const OPTIMIZER_REFS: &str = "scipy.optimize 1.11, scipy.integrate 1.11";

/// Reference source for MD validation.
///
/// Expected values are analytical force laws: Lennard-Jones 12-6,
/// Coulomb 1/r², Morse exponential. Velocity-Verlet integration checked
/// via energy conservation and symplectic invariants. No Python baselines:
/// the physics is exact analytical math.
pub const MD_FORCE_REFS: &str = "Analytical LJ/Coulomb/Morse force laws, Allen & Tildesley (1987)";

/// Reference source for GPU kernel validation (Yukawa, PPPM).
///
/// GPU f32 kernels compared against CPU f64 reference with relative
/// tolerance accounting for f32 precision (~7 significant digits).
/// PPPM validated against direct Coulomb summation (exact but O(N²)).
pub const GPU_KERNEL_REFS: &str = "CPU f64 direct summation reference, f32 precision tolerance";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::NuclearMatterProps;

    #[test]
    fn nmp_targets_are_physical() {
        let t = NMP_TARGETS;
        assert!(t.rho0.value > 0.0, "saturation density must be positive");
        assert!(t.e_a.value < 0.0, "binding energy must be negative");
        assert!(t.k_inf.value > 0.0, "incompressibility must be positive");
        assert!(
            t.m_eff.value > 0.0 && t.m_eff.value < 1.0,
            "effective mass ratio must be in (0, 1)"
        );
        assert!(t.j.value > 0.0, "symmetry energy must be positive");
    }

    #[test]
    fn nmp_within_sigma_sly4() {
        // SLy4 values (Chabanat 1998): ρ₀=0.1595, E/A=-15.97, K∞=230, m*/m=0.69, J=32
        let sly4 = [0.1595, -15.97, 230.0, 0.69, 32.0];
        let within = NMP_TARGETS.within_sigma(&sly4, 2.0);
        assert!(
            within.iter().all(|&b| b),
            "SLy4 should be within 2σ of all targets"
        );
    }

    #[test]
    fn nmp_targets_values_returns_correct_array() {
        let vals = NMP_TARGETS.values();
        assert_eq!(vals.len(), 5);
        assert!((vals[0] - 0.16).abs() < 1e-10, "ρ₀");
        assert!((vals[1] - (-15.97)).abs() < 1e-10, "E/A");
        assert!((vals[2] - 230.0).abs() < 1e-10, "K∞");
        assert!((vals[3] - 0.69).abs() < 1e-10, "m*/m");
        assert!((vals[4] - 32.0).abs() < 1e-10, "J");
    }

    #[test]
    fn nmp_targets_sigmas_returns_correct_array() {
        let sigmas = NMP_TARGETS.sigmas();
        assert_eq!(sigmas.len(), 5);
        assert!((sigmas[0] - 0.005).abs() < 1e-10);
        assert!((sigmas[1] - 0.5).abs() < 1e-10);
        assert!((sigmas[2] - 20.0).abs() < 1e-10);
        assert!((sigmas[3] - 0.1).abs() < 1e-10);
        assert!((sigmas[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn nmp_within_sigma_known_values() {
        // Exact match: all within
        let exact = NMP_TARGETS.values();
        let within = NMP_TARGETS.within_sigma(&exact, 0.1);
        assert!(within.iter().all(|&b| b));

        // Far off: none within 0.1σ
        let far = [0.0, 0.0, 0.0, 0.0, 0.0];
        let within = NMP_TARGETS.within_sigma(&far, 0.1);
        assert!(!within.iter().any(|&b| b));

        // 1σ away: should fail at 0.5σ, pass at 2σ
        let one_sigma_high = [
            NMP_TARGETS.rho0.value + NMP_TARGETS.rho0.sigma,
            NMP_TARGETS.e_a.value + NMP_TARGETS.e_a.sigma,
            NMP_TARGETS.k_inf.value + NMP_TARGETS.k_inf.sigma,
            NMP_TARGETS.m_eff.value + NMP_TARGETS.m_eff.sigma,
            NMP_TARGETS.j.value + NMP_TARGETS.j.sigma,
        ];
        let within_05 = NMP_TARGETS.within_sigma(&one_sigma_high, 0.5);
        let within_2 = NMP_TARGETS.within_sigma(&one_sigma_high, 2.0);
        assert!(!within_05.iter().all(|&b| b));
        assert!(within_2.iter().all(|&b| b));
    }

    #[test]
    fn print_nmp_analysis_no_panic() {
        let nmp = NuclearMatterProps {
            rho0_fm3: 0.1595,
            e_a_mev: -15.97,
            k_inf_mev: 230.0,
            m_eff_ratio: 0.69,
            j_mev: 32.0,
        };
        // Cannot easily capture stdout; just verify it doesn't panic
        print_nmp_analysis(&nmp);
    }

    #[test]
    fn ame2020_doi_non_empty() {
        assert!(!AME2020_DOI.is_empty());
        assert!(AME2020_DOI.contains("10."));
    }

    #[test]
    fn param_names_length_matches_sly4_params() {
        assert_eq!(PARAM_NAMES.len(), SLY4_PARAMS.len());
    }

    #[test]
    fn baseline_provenance_records_non_empty_fields() {
        for p in [
            &L1_PYTHON_CHI2,
            &L1_PYTHON_CANDIDATES,
            &L2_PYTHON_CHI2,
            &L2_PYTHON_CANDIDATES,
            &L2_PYTHON_TOTAL_CHI2,
            &HOTQCD_EOS_PROVENANCE,
            &SCREENED_COULOMB_PROVENANCE,
            &DALIGAULT_CALIBRATION_PROVENANCE,
            &DALIGAULT_FIT_PROVENANCE,
            &TRANSPORT_MD_BASELINE_PROVENANCE,
            &HFB_TEST_NUCLEI_PROVENANCE,
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

    #[test]
    fn hotqcd_doi_is_valid() {
        assert!(HOTQCD_DOI.starts_with("10."));
        assert!(!HOTQCD_DOI.is_empty());
    }

    #[test]
    fn pure_gauge_refs_non_empty() {
        assert!(!PURE_GAUGE_REFS.is_empty());
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn provenance_records_have_content() {
        assert!(!L1_PYTHON_CHI2.script.is_empty());
        assert!(!L1_PYTHON_CHI2.commit.is_empty());
        assert!(!L1_PYTHON_CHI2.command.is_empty());
        assert!(L1_PYTHON_CHI2.value > 0.0);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn sly4_params_have_correct_length() {
        assert_eq!(SLY4_PARAMS.len(), 10);
        assert_eq!(UNEDF0_PARAMS.len(), 10);
        assert_eq!(PARAM_NAMES.len(), 10);
    }

    #[test]
    fn nmp_chi2_sly4_is_small() {
        // SLy4 is within ~2σ of all NMP targets, so χ² should be < 25
        let sly4_nmp = [0.1595, -15.97, 230.0, 0.69, 32.0];
        let chi2 = nmp_chi2(&sly4_nmp);
        assert!(chi2 < 25.0, "SLy4 NMP χ² should be small, got {chi2}");
    }

    #[test]
    fn nmp_chi2_exact_match_is_zero() {
        let exact = NMP_TARGETS.values();
        let chi2 = nmp_chi2(&exact);
        assert!(
            chi2.abs() < 1e-15,
            "exact match χ² should be ~0, got {chi2}"
        );
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
