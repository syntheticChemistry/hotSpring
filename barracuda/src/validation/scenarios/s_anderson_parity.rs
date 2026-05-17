// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Anderson Cross-Tier Parity — Python vs Rust
//!
//! Validates that the Rust spectral implementations produce results
//! matching the Python control baselines within documented tolerances.
//! Three-layer proof pattern: Tier 1 (Python) confirms science,
//! Tier 2 (Rust) confirms implementation parity.
//!
//! Python baseline: control/spectral_theory/scripts/spectral_control.py
//! Rust reference:  validate_spectral + validate_anderson_3d

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "anderson-cross-tier-parity",
        track: Track::SpectralTheory,
        tier: Tier::Rust,
        provenance_crate: "spectral_control.py + validate_spectral",
        provenance_date: "2026-05-17",
        description: "Anderson module Python vs Rust cross-tier parity (lithoSpore pattern)",
    },
    run,
};

const LN2: f64 = core::f64::consts::LN_2;
const POISSON_R: f64 = 2.0 * LN2 - 1.0;

pub fn run(v: &mut ValidationHarness) {
    check_anderson_1d_spectrum(v);
    check_herman_lyapunov(v);
    check_level_statistics(v);
    check_anderson_3d_bandwidth(v);
    check_dimensional_hierarchy(v);
}

fn check_anderson_1d_spectrum(v: &mut ValidationHarness) {
    use crate::spectral::{anderson_hamiltonian, find_all_eigenvalues};

    let n = 1000;
    let w = 4.0;
    let (d, e) = anderson_hamiltonian(n, w, 42);
    let evals = find_all_eigenvalues(&d, &e);

    v.check_bool("parity:anderson1d_count", evals.len() == n);

    let bound = 2.0 + w / 2.0;
    let all_bounded = evals.iter().all(|&ev| ev.abs() <= bound + 0.01);
    v.check_bool("parity:anderson1d_gershgorin", all_bounded);

    v.check_bool("parity:anderson1d_sorted", evals.windows(2).all(|w| w[0] <= w[1]));

    // Python: e_min ≈ -3.53, e_max ≈ 3.61 (both within [-4,4])
    let e_min = evals[0];
    let e_max = evals[evals.len() - 1];
    v.check_bool("parity:anderson1d_emin_range", e_min > -bound && e_min < 0.0);
    v.check_bool("parity:anderson1d_emax_range", e_max > 0.0 && e_max < bound);
}

fn check_herman_lyapunov(v: &mut ValidationHarness) {
    use crate::spectral::lyapunov_averaged;

    let n_sites = 100_000;
    let n_realizations = 5;

    // Python baseline: λ=2 → γ=0.6932, theory=ln(2)=0.6931
    // Rust lyapunov_averaged uses the same transfer-matrix algorithm
    let test_cases = [
        (1.5, (1.5_f64).ln()),
        (2.0, LN2),
        (3.0, (3.0_f64).ln()),
    ];

    for &(lambda, theory) in &test_cases {
        let gamma = lyapunov_averaged(n_sites, lambda, 0.0, n_realizations, 42);
        let rel_err = (gamma - theory).abs() / theory;
        let tag = format!("parity:herman_lambda{}", (lambda * 10.0) as u32);
        v.check_bool(&tag, rel_err < 1e-2);
    }
}

fn check_level_statistics(v: &mut ValidationHarness) {
    use crate::spectral::{anderson_hamiltonian, find_all_eigenvalues, level_spacing_ratio};

    let n = 1000;
    let w = 8.0;
    let n_realizations: u64 = 10;

    let mut r_sum = 0.0;
    for seed in 0..n_realizations {
        let (d, e) = anderson_hamiltonian(n, w, seed);
        let evals = find_all_eigenvalues(&d, &e);
        r_sum += level_spacing_ratio(&evals);
    }
    let r_mean = r_sum / n_realizations as f64;

    // Python baseline: r_mean ≈ 0.3889, Poisson = 0.3863
    let dev = (r_mean - POISSON_R).abs();
    v.check_bool("parity:level_stats_poisson", dev < 0.05);
    v.check_bool("parity:level_stats_localized", r_mean < 0.45);
}

fn check_anderson_3d_bandwidth(v: &mut ValidationHarness) {
    use crate::spectral::{clean_3d_lattice, lanczos, lanczos_eigenvalues};

    let l: usize = 8;
    let n = l * l * l;
    let mat = clean_3d_lattice(l);
    let tri = lanczos(&mat, n, 42);
    let evals = lanczos_eigenvalues(&tri);
    let bw = evals[evals.len() - 1] - evals[0];

    // Python baseline: 11.276311449430906
    let python_bw = 11.276311449430906;
    let rel_err = (bw - python_bw).abs() / python_bw;
    v.check_bool("parity:anderson3d_bw_python_match", rel_err < 1e-6);
    v.check_bool("parity:anderson3d_bw_nonzero", bw > 10.0);
}

fn check_dimensional_hierarchy(v: &mut ValidationHarness) {
    use crate::spectral::{
        anderson_3d, anderson_hamiltonian, find_all_eigenvalues, lanczos, lanczos_eigenvalues,
    };

    let w = 2.0;

    // 1D: N=500
    let (d, e) = anderson_hamiltonian(500, w, 42);
    let evals_1d = find_all_eigenvalues(&d, &e);
    let bw_1d = evals_1d[evals_1d.len() - 1] - evals_1d[0];

    // 3D: L=8, N=512
    let l: usize = 8;
    let mat_3d = anderson_3d(l, l, l, w, 42);
    let tri_3d = lanczos(&mat_3d, l * l * l, 42);
    let evals_3d = lanczos_eigenvalues(&tri_3d);
    let bw_3d = evals_3d[evals_3d.len() - 1] - evals_3d[0];

    // Python: bw_1d ≈ 5.30, bw_3d ≈ 11.45
    v.check_bool("parity:dim_hierarchy_1d_lt_3d", bw_1d < bw_3d);
    v.check_bool("parity:dim_hierarchy_1d_positive", bw_1d > 4.0);
    v.check_bool("parity:dim_hierarchy_3d_positive", bw_3d > 10.0);
}
