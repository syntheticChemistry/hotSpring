// SPDX-License-Identifier: AGPL-3.0-or-later

use super::types::BaselineProvenance;

/// Sarkas software version used for control runs
pub const SARKAS_VERSION: &str = "1.0.0";
/// Sarkas pinned commit
pub const SARKAS_COMMIT: &str = "fd908c41";
/// Publication reference for DSF study
pub const SARKAS_PAPER: &str = "Choi, Dharuman, Murillo, Phys. Rev. E 100, 013206 (2019)";

/// Daligault (2012) analytical fit reference.
///
/// Test values in `md/transport.rs::d_star_matches_all_12_sarkas_calibration_points`
/// are computed from the Daligault (2012) practical model, NOT from MD simulation.
/// They are analytical values that both Rust and Python must reproduce.
///
/// Script distinction: `calibrate_daligault_fit.py` fits `C_w`(κ) from Sarkas MD;
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

/// Calibration script for Daligault weak-coupling correction `C_w`(κ).
///
/// `calibrate_daligault_fit.py` runs a grid search over (κ, Γ) points from
/// Sarkas MD and fits `C_w`(κ) = exp(1.435 + 0.715κ + 0.401κ²).
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

/// TTM local model equilibrium temperature — Argon (Te₀=15 000 K, Ti₀=300 K).
///
/// From `run_local_model.py` SMT transport model. `CONTROL_EXPERIMENT_STATUS` §3.
pub const TTM_ARGON_EQUILIBRIUM_K: BaselineProvenance = BaselineProvenance {
    label: "TTM Argon equilibrium T (Te₀=15 000 K, Ti₀=300 K)",
    script: "ttm/scripts/run_local_model.py",
    commit: "03c0403 (hotSpring control)",
    date: "2026-02-26",
    command: "python run_local_model.py --species argon,xenon,helium --model SMT",
    environment: "conda activate ttm",
    value: 8100.0,
    unit: "K",
};

/// TTM local model equilibrium temperature — Xenon (Te₀=20 000 K, Ti₀=300 K).
///
/// From `run_local_model.py` SMT transport model. `CONTROL_EXPERIMENT_STATUS` §3.
pub const TTM_XENON_EQUILIBRIUM_K: BaselineProvenance = BaselineProvenance {
    label: "TTM Xenon equilibrium T (Te₀=20 000 K, Ti₀=300 K)",
    script: "ttm/scripts/run_local_model.py",
    commit: "03c0403 (hotSpring control)",
    date: "2026-02-26",
    command: "python run_local_model.py --species argon,xenon,helium --model SMT",
    environment: "conda activate ttm",
    value: 14_085.0,
    unit: "K",
};

/// TTM local model equilibrium temperature — Helium (Te₀=30 000 K, Ti₀=300 K).
///
/// From `run_local_model.py` SMT transport model. `CONTROL_EXPERIMENT_STATUS` §3.
pub const TTM_HELIUM_EQUILIBRIUM_K: BaselineProvenance = BaselineProvenance {
    label: "TTM Helium equilibrium T (Te₀=30 000 K, Ti₀=300 K)",
    script: "ttm/scripts/run_local_model.py",
    commit: "03c0403 (hotSpring control)",
    date: "2026-02-26",
    command: "python run_local_model.py --species argon,xenon,helium --model SMT",
    environment: "conda activate ttm",
    value: 10_700.0,
    unit: "K",
};
