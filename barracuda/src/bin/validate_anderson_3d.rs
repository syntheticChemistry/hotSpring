// SPDX-License-Identifier: AGPL-3.0-only

//! 3D Anderson Model Validation — Kachkovskiy Extension (Tier 3)
//!
//! Validates the **only** dimensionality (d ≥ 3) where a genuine Anderson
//! metal-insulator transition exists:
//!
//! **Scaling theory (Abrahams et al. 1979)**:
//! - d = 1, 2: ALL states localized for any disorder (no transition)
//! - d ≥ 3: a critical disorder W_c exists above which all states localize
//!
//! **3D Anderson model on cubic lattice**:
//! - Coordination z = 6 (nearest neighbors), clean bandwidth = 12 ([-6, 6])
//! - Critical disorder W_c ≈ 16.5 (Slevin & Ohtsuki 1999)
//! - **Mobility edge**: at W < W_c, extended states near band center coexist
//!   with localized states at band edges — a coexistence that is IMPOSSIBLE
//!   in 1D or 2D
//!
//! This validates the complete dimensional progression:
//! 1D (always localized) → 2D (always localized, ξ exponentially large) →
//! 3D (genuine phase transition with mobility edge)
//!
//! # Provenance
//!
//! Anderson (1958) Phys. Rev. 109, 1492
//! Abrahams, Anderson, Licciardello, Ramakrishnan (1979) Phys. Rev. Lett. 42, 673
//! Slevin & Ohtsuki (1999) Phys. Rev. Lett. 82, 382
//! Oganesyan & Huse (2007) Phys. Rev. B 75, 155111

use hotspring_barracuda::spectral;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  3D Anderson Model — Kachkovskiy Tier 3                    ║");
    println!("║  Mobility edge: the phase transition impossible in d<3     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("anderson_3d");

    check_3d_nnz(&mut harness);
    check_clean_3d_bandwidth(&mut harness);
    check_3d_spectrum_bounds(&mut harness);
    check_3d_goe_statistics(&mut harness);
    check_3d_poisson_statistics(&mut harness);
    check_3d_statistics_transition(&mut harness);
    check_mobility_edge(&mut harness);
    check_dimensional_bandwidth_hierarchy(&mut harness);
    check_dimensional_statistics_hierarchy(&mut harness);
    check_3d_spectrum_symmetry(&mut harness);

    println!();
    harness.finish();
}

/// [1] CSR construction: verify nnz = N + 2 × (3 bond types).
fn check_3d_nnz(harness: &mut ValidationHarness) {
    println!("[1] 3D Anderson — CSR Construction");

    let l = 8;
    let n = l * l * l;
    let mat = spectral::anderson_3d(l, l, l, 1.0, 42);

    // nnz = N + 2*[(l-1)*l*l + l*(l-1)*l + l*l*(l-1)]
    let bonds_per_axis = (l - 1) * l * l;
    let expected_nnz = n + 2 * 3 * bonds_per_axis;

    println!("  L={l}, N={n}");
    println!("  nnz = {} (expected {expected_nnz})", mat.nnz());
    println!("  nnz/N = {:.2} (theory: ≤7 for interior sites)", mat.nnz() as f64 / n as f64);

    harness.check_bool("nnz matches theory", mat.nnz() == expected_nnz);
    println!();
}

/// [2] Clean 3D bandwidth: approaches 12 as L→∞.
fn check_clean_3d_bandwidth(harness: &mut ValidationHarness) {
    println!("[2] Clean 3D Lattice — Bandwidth");

    let l = 10;
    let n = l * l * l;
    let mat = spectral::clean_3d_lattice(l);
    let result = spectral::lanczos(&mat, n, 42);
    let evals = spectral::lanczos_eigenvalues(&result);

    let e_min = evals[0];
    let e_max = *evals.last().unwrap();
    let bw = e_max - e_min;

    // Exact for open BCs: BW = 12·cos(π/(L+1))
    let exact_bw = 12.0 * (std::f64::consts::PI / (l as f64 + 1.0)).cos();

    println!("  L={l}, N={n}");
    println!("  E_min = {e_min:.4}, E_max = {e_max:.4}");
    println!("  Bandwidth = {bw:.4} (exact OBC: {exact_bw:.4}, L→∞: 12.0)");

    harness.check_upper("bandwidth matches open-BC theory", (bw - exact_bw).abs(), 0.1);
    println!();
}

/// [3] 3D Anderson spectrum bounded by [-6 - W/2, 6 + W/2].
fn check_3d_spectrum_bounds(harness: &mut ValidationHarness) {
    println!("[3] 3D Anderson — Spectrum Bounds");

    let l = 8;
    let w = 8.0;
    let mat = spectral::anderson_3d(l, l, l, w, 42);
    let result = spectral::lanczos(&mat, l * l * l, 42);
    let evals = spectral::lanczos_eigenvalues(&result);

    let bound = 6.0 + w / 2.0;
    let e_min = evals[0];
    let e_max = *evals.last().unwrap();
    let in_bounds = e_min >= -(bound + 0.1) && e_max <= bound + 0.1;

    println!("  L={l}, W={w}, N={}", l * l * l);
    println!("  Spectrum: [{e_min:.4}, {e_max:.4}]");
    println!("  Bound: [-{bound:.1}, {bound:.1}]");

    harness.check_bool("3D spectrum within Gershgorin bounds", in_bounds);
    println!();
}

/// [4] 3D weak disorder: GOE level statistics (extended states, ξ >> L).
fn check_3d_goe_statistics(harness: &mut ValidationHarness) {
    println!("[4] 3D Weak Disorder — GOE Level Statistics");
    println!("    Theory: ⟨r⟩ ≈ 0.531 for W << W_c ≈ 16.5 (metallic regime)\n");

    let l = 8;
    let w = 4.0;
    let n_real = 8;
    let goe_r = 0.531;

    let mut r_sum = 0.0;
    for seed in 0..n_real {
        let mat = spectral::anderson_3d(l, l, l, w, seed * 137 + 42);
        let result = spectral::lanczos(&mat, l * l * l, seed * 37 + 1);
        let evals = spectral::lanczos_eigenvalues(&result);
        let mid = evals.len() / 4;
        let end = 3 * evals.len() / 4;
        let bulk = &evals[mid..end];
        r_sum += spectral::level_spacing_ratio(bulk);
    }
    let r_mean = r_sum / n_real as f64;

    println!("  L={l}, W={w}, realizations={n_real}");
    println!("  ⟨r⟩ = {r_mean:.4} (GOE = {goe_r:.4})");

    harness.check_bool("⟨r⟩ > 0.48 (GOE-like for 3D metallic regime)", r_mean > 0.48);
    println!();
}

/// [5] 3D strong disorder: Poisson level statistics (all states localized).
fn check_3d_poisson_statistics(harness: &mut ValidationHarness) {
    println!("[5] 3D Strong Disorder — Poisson Level Statistics");
    println!(
        "    Theory: ⟨r⟩ ≈ {:.4} for W >> W_c ≈ 16.5 (insulating regime)\n",
        spectral::POISSON_R
    );

    let l = 8;
    let w = 30.0;
    let n_real = 8;

    let mut r_sum = 0.0;
    for seed in 0..n_real {
        let mat = spectral::anderson_3d(l, l, l, w, seed * 137 + 42);
        let result = spectral::lanczos(&mat, l * l * l, seed * 37 + 1);
        let evals = spectral::lanczos_eigenvalues(&result);
        let mid = evals.len() / 4;
        let end = 3 * evals.len() / 4;
        let bulk = &evals[mid..end];
        r_sum += spectral::level_spacing_ratio(bulk);
    }
    let r_mean = r_sum / n_real as f64;

    println!("  L={l}, W={w}, realizations={n_real}");
    println!("  ⟨r⟩ = {r_mean:.4} (Poisson = {:.4})", spectral::POISSON_R);

    let deviation = (r_mean - spectral::POISSON_R).abs();
    harness.check_upper("⟨r⟩ within 0.04 of Poisson", deviation, 0.04);
    println!();
}

/// [6] 3D statistics transition: GOE → Poisson with increasing disorder.
fn check_3d_statistics_transition(harness: &mut ValidationHarness) {
    println!("[6] 3D Statistics Transition — GOE → Poisson");
    println!("    W_c ≈ 16.5 for band center (Slevin & Ohtsuki 1999)\n");

    let l = 8;
    let w_values = [2.0, 6.0, 12.0, 20.0, 35.0];
    let n_real = 5;

    let mut r_values = Vec::new();

    for &w in &w_values {
        let mut r_sum = 0.0;
        for seed in 0..n_real {
            let mat = spectral::anderson_3d(l, l, l, w, seed * 137 + 42);
            let result = spectral::lanczos(&mat, l * l * l, seed * 37 + 1);
            let evals = spectral::lanczos_eigenvalues(&result);
            let mid = evals.len() / 4;
            let end = 3 * evals.len() / 4;
            let bulk = &evals[mid..end];
            r_sum += spectral::level_spacing_ratio(bulk);
        }
        let r_mean = r_sum / n_real as f64;
        r_values.push(r_mean);
        println!("  W={w:>5.1}: ⟨r⟩ = {r_mean:.4}");
    }

    let transition = r_values.first().unwrap() > r_values.last().unwrap();
    let delta = r_values.first().unwrap() - r_values.last().unwrap();
    println!("  Δ⟨r⟩ = {delta:.4} (weak→strong)");

    harness.check_bool(
        "⟨r⟩ decreases from weak to strong disorder (3D transition)",
        transition && delta > 0.05,
    );
    println!();
}

/// [7] Mobility edge: at moderate disorder (W < W_c), band center has
/// extended states (GOE) while band edges have localized states (Poisson).
///
/// This is the defining feature of d ≥ 3 that does NOT exist in 1D or 2D.
fn check_mobility_edge(harness: &mut ValidationHarness) {
    println!("[7] 3D Mobility Edge — The d≥3 Signature");
    println!("    Band center: extended (GOE); Band edges: localized (Poisson)");
    println!("    This coexistence is IMPOSSIBLE in 1D or 2D\n");

    let l = 8;
    let w = 12.0; // Below W_c ≈ 16.5 — near the transition
    let n_real = 10;

    let mut center_r_sum = 0.0;
    let mut edge_r_sum = 0.0;

    for seed in 0..n_real {
        let mat = spectral::anderson_3d(l, l, l, w, seed * 137 + 42);
        let result = spectral::lanczos(&mat, l * l * l, seed * 37 + 1);
        let evals = spectral::lanczos_eigenvalues(&result);

        let n_evals = evals.len();
        let quarter = n_evals / 4;

        // Band center: middle 50% of eigenvalues
        let center = &evals[quarter..3 * quarter];
        center_r_sum += spectral::level_spacing_ratio(center);

        // Band edges: lowest 20% + highest 20%
        let edge_low = &evals[..n_evals / 5];
        let edge_high = &evals[4 * n_evals / 5..];
        let r_low = spectral::level_spacing_ratio(edge_low);
        let r_high = spectral::level_spacing_ratio(edge_high);
        edge_r_sum += (r_low + r_high) / 2.0;
    }

    let center_r = center_r_sum / n_real as f64;
    let edge_r = edge_r_sum / n_real as f64;

    println!("  L={l}, W={w} (below W_c ≈ 16.5), realizations={n_real}");
    println!("  Band center ⟨r⟩ = {center_r:.4} (expect GOE ≈ 0.53)");
    println!("  Band edges  ⟨r⟩ = {edge_r:.4}  (expect closer to Poisson ≈ 0.39)");
    println!("  Δ⟨r⟩(center-edge) = {:.4}", center_r - edge_r);

    // The center should have higher r than the edges (more extended)
    harness.check_bool(
        "band center ⟨r⟩ > band edge ⟨r⟩ (mobility edge signature)",
        center_r > edge_r,
    );
    println!();
}

/// [8] Dimensional bandwidth hierarchy: 3D > 2D > 1D.
fn check_dimensional_bandwidth_hierarchy(harness: &mut ValidationHarness) {
    println!("[8] Dimensional Bandwidth Hierarchy — 1D < 2D < 3D");

    let w = 2.0;

    // 1D: N=500
    let n_1d = 500;
    let (d, e) = spectral::anderson_hamiltonian(n_1d, w, 42);
    let evals_1d = spectral::find_all_eigenvalues(&d, &e);
    let bw_1d = evals_1d.last().unwrap() - evals_1d.first().unwrap();

    // 2D: ~22×22 ≈ 484
    let l_2d = 22;
    let mat_2d = spectral::anderson_2d(l_2d, l_2d, w, 42);
    let result_2d = spectral::lanczos(&mat_2d, l_2d * l_2d, 42);
    let evals_2d = spectral::lanczos_eigenvalues(&result_2d);
    let bw_2d = evals_2d.last().unwrap() - evals_2d.first().unwrap();

    // 3D: 8×8×8 = 512
    let l_3d = 8;
    let mat_3d = spectral::anderson_3d(l_3d, l_3d, l_3d, w, 42);
    let result_3d = spectral::lanczos(&mat_3d, l_3d * l_3d * l_3d, 42);
    let evals_3d = spectral::lanczos_eigenvalues(&result_3d);
    let bw_3d = evals_3d.last().unwrap() - evals_3d.first().unwrap();

    println!("  W={w}");
    println!("  1D (N={n_1d}):    bandwidth = {bw_1d:.4}  (clean: 4)");
    println!("  2D ({l_2d}×{l_2d}={n_2d}): bandwidth = {bw_2d:.4}  (clean: 8)", n_2d = l_2d * l_2d);
    println!("  3D ({l_3d}³={n_3d}):  bandwidth = {bw_3d:.4}  (clean: 12)", n_3d = l_3d * l_3d * l_3d);

    harness.check_bool(
        "3D bandwidth > 2D bandwidth > 1D bandwidth",
        bw_3d > bw_2d && bw_2d > bw_1d,
    );
    println!();
}

/// [9] Dimensional statistics hierarchy: 3D has strongest level repulsion
/// at same disorder (most extended states).
fn check_dimensional_statistics_hierarchy(harness: &mut ValidationHarness) {
    println!("[9] Dimensional Level Statistics — 3D Most Extended");
    println!("    At weak disorder, higher d → more extended → higher ⟨r⟩\n");

    let w = 4.0;
    let n_real = 5;

    // 1D: Poisson (always localized)
    let mut r_1d_sum = 0.0;
    for seed in 0..n_real {
        let (d, e) = spectral::anderson_hamiltonian(500, w, seed * 100 + 42);
        let evals = spectral::find_all_eigenvalues(&d, &e);
        let mid = evals.len() / 4;
        let end = 3 * evals.len() / 4;
        r_1d_sum += spectral::level_spacing_ratio(&evals[mid..end]);
    }
    let r_1d = r_1d_sum / n_real as f64;

    // 2D: intermediate
    let l_2d = 16;
    let mut r_2d_sum = 0.0;
    for seed in 0..n_real {
        let mat = spectral::anderson_2d(l_2d, l_2d, w, seed * 137 + 42);
        let result = spectral::lanczos(&mat, l_2d * l_2d, seed * 37 + 1);
        let evals = spectral::lanczos_eigenvalues(&result);
        let mid = evals.len() / 4;
        let end = 3 * evals.len() / 4;
        r_2d_sum += spectral::level_spacing_ratio(&evals[mid..end]);
    }
    let r_2d = r_2d_sum / n_real as f64;

    // 3D: most extended (GOE)
    let l_3d = 8;
    let mut r_3d_sum = 0.0;
    for seed in 0..n_real {
        let mat = spectral::anderson_3d(l_3d, l_3d, l_3d, w, seed * 137 + 42);
        let result = spectral::lanczos(&mat, l_3d * l_3d * l_3d, seed * 37 + 1);
        let evals = spectral::lanczos_eigenvalues(&result);
        let mid = evals.len() / 4;
        let end = 3 * evals.len() / 4;
        r_3d_sum += spectral::level_spacing_ratio(&evals[mid..end]);
    }
    let r_3d = r_3d_sum / n_real as f64;

    println!("  W={w}");
    println!("  1D ⟨r⟩ = {r_1d:.4}  (Poisson ≈ 0.386, always localized)");
    println!("  2D ⟨r⟩ = {r_2d:.4}  (intermediate)");
    println!("  3D ⟨r⟩ = {r_3d:.4}  (GOE ≈ 0.531, metallic)");

    // 3D should be most extended (highest r), 1D most localized (lowest r)
    harness.check_bool(
        "⟨r⟩(3D) > ⟨r⟩(1D): higher d → more extended at same W",
        r_3d > r_1d,
    );
    println!();
}

/// [10] 3D spectrum symmetric about E = 0 (particle-hole symmetry).
fn check_3d_spectrum_symmetry(harness: &mut ValidationHarness) {
    println!("[10] 3D Anderson — Spectrum Symmetry");
    println!("    Bipartite lattice → spectrum symmetric about E = 0\n");

    let l = 8;
    let mat = spectral::clean_3d_lattice(l);
    let result = spectral::lanczos(&mat, l * l * l, 42);
    let evals = spectral::lanczos_eigenvalues(&result);

    let e_min = evals[0];
    let e_max = *evals.last().unwrap();
    let asymmetry = (e_min + e_max).abs();

    println!("  L={l} (clean, no disorder)");
    println!("  E_min = {e_min:.6}, E_max = {e_max:.6}");
    println!("  |E_min + E_max| = {asymmetry:.2e} (should be ≈ 0)");

    harness.check_upper("spectrum symmetric about E=0", asymmetry, 1e-8);
    println!();
}
