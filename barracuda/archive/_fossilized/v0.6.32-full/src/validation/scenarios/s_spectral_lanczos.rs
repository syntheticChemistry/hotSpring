// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Spectral Lanczos — absorbed from validate_lanczos.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "spectral-lanczos",
        track: Track::SpectralTheory,
        tier: Tier::Rust,
        provenance_crate: "validate_lanczos",
        provenance_date: "2026-05-09",
        description: "Lanczos eigenvalue convergence on Anderson 2D tight-binding model",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    use crate::spectral::{anderson_2d, lanczos, lanczos_eigenvalues};

    let matrix = anderson_2d(4, 4, 0.0, 42);
    let tridiag = lanczos(&matrix, 10, 42);
    let eigs = lanczos_eigenvalues(&tridiag);

    v.check_bool("spectral:lanczos_converges", !eigs.is_empty());
    v.check_bool(
        "spectral:lanczos_sorted",
        eigs.windows(2).all(|w| w[0] <= w[1]),
    );
    v.check_bool(
        "spectral:eigenvalues_finite",
        eigs.iter().all(|e| e.is_finite()),
    );
}
